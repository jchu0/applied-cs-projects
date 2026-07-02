# Distributed Key-Value Store with Raft Consensus

A strongly consistent distributed key-value store built from scratch on the Raft
consensus algorithm in Rust. It provides replicated read/write operations across a
cluster of nodes with automatic leader election, a write-ahead log, snapshot-based log
compaction, and joint-consensus membership changes.

## Features

- **Full Raft state machine** — follower, candidate, and leader states with randomized
  election timeouts, term tracking, and the no-op-on-election rule (`RaftNode` / `node.rs`).
- **Log replication** — per-peer `next_index` / `match_index` tracking, conflict-index
  fast backtracking, and quorum-based commit advancement (`handle_append_entries`).
- **Pre-vote extension** — a disruption-avoiding pre-election round so a partitioned node
  cannot force term inflation (`PreVoteState`, `start_pre_vote`).
- **Check-quorum** — a leader steps down when it can no longer reach a majority of peers
  (`check_quorum`).
- **Write-ahead log** — length-prefixed, CRC32-checksummed records with crash recovery
  and segment files (`WriteAheadLog` / `storage.rs`).
- **Snapshots and compaction** — `InstallSnapshot` RPC plus FSM snapshot/restore and log
  truncation (`SnapshotStore`, `KeyValueFSM`).
- **Joint-consensus membership** — two-phase `C_old,new` then `C_new` configuration
  changes (`propose_membership_change`, `ClusterConfig::quorum_size`).
- **Pluggable transport** — one `Transport` trait with an in-process `MemoryNetwork` and a
  `tonic`-based gRPC implementation (`transport.rs`, `grpc.rs`).
- **Metrics** — counters, gauges, and a bucketed histogram for elections, replication, and
  latency (`RaftMetrics`, `Histogram`).
- **Test harness** — `TestCluster` over `MemoryNetwork` for partition and election tests.

## Architecture

```mermaid
flowchart TD
    Client(KVClient) --> Leader(Leader Node)
    Leader --> Log(Write-Ahead Log)
    Leader --> FSM(KeyValue FSM)
    Leader --> Snap(Snapshot Store)
    Leader -->|AppendEntries / RequestVote| F1(Follower Node)
    Leader -->|AppendEntries / RequestVote| F2(Follower Node)
    F1 --> Transport(Transport trait)
    F2 --> Transport
    Transport --> Mem(MemoryNetwork)
    Transport --> Grpc(GrpcTransport)
```

| Component | Module | Responsibility |
|-----------|--------|----------------|
| Raft node | `node.rs` | Elections, log replication, commit/apply, snapshots, membership |
| Storage | `storage.rs` | `WriteAheadLog`, `KeyValueFSM`, `SnapshotStore`, `MemoryStorage` |
| Transport | `transport.rs` | `Transport` trait, in-process `MemoryNetwork`, `PeerTracker` |
| gRPC transport | `grpc.rs` | `tonic` client/server (`GrpcTransport`, `GrpcServerBuilder`) |
| RPC types | `rpc.rs` | `AppendEntries`, `RequestVote`, `InstallSnapshot`, client messages |
| Cluster | `cluster.rs` | `RaftClusterNode` event loop and `TestCluster` harness |
| Client | `client.rs` | `KVClient` end-to-end command path with leader-hint redirect and retry/backoff |
| Config | `config.rs` | `RaftConfig` builder and `ClusterConfig` quorum logic |
| Metrics | `metrics.rs` | `RaftMetrics`, `Histogram`, `HealthCheck` |
| Errors | `error.rs` | Unified `Error` / `Result` |

## Quick Start

### Prerequisites

- Rust stable, edition 2021 (`cargo`).
- `protoc`, the Protocol Buffers compiler — `build.rs` runs `tonic-build` against
  `proto/raft.proto` to generate the gRPC stubs at build time.

### Installation

```bash
cd 11-distributed-kv-raft
cargo build
```

### Running

This crate is a library, not a standalone binary. Drive it from tests, benchmarks, or
your own code:

```bash
cargo test          # run the full suite
cargo bench         # Criterion microbenchmarks (optional)
```

## Usage

A node is created from a `RaftConfig`, then driven through Raft RPCs. The example below
exercises the in-memory FSM and a single-node leader directly via the real public API:

```rust
use distributed_kv_raft::{Command, RaftConfig, RaftNode, RequestVoteResponse};

// Build a single-node configuration (no peers => quorum is 1).
let config = RaftConfig::builder().id(0).build();
let mut node = RaftNode::new(config);
node.start().unwrap();

// With no peers, a candidate immediately wins the election.
node.transition_to_candidate();
node.handle_request_vote_response(0, RequestVoteResponse {
    term: node.current_term,
    vote_granted: true,
});
assert!(node.is_leader());

// Propose a write; it is appended to the log on the leader.
let index = node.propose(Command::Put {
    key: b"name".to_vec(),
    value: b"raft".to_vec(),
}).unwrap();
println!("appended at log index {index}");
```

For a multi-node cluster, build one `RaftConfig` per node, wire them together over a
`MemoryNetwork`, and run them through `TestCluster`. Once a leader is elected, a `KVClient`
attached to the network's client transport drives the real end-to-end command path —
propose to the leader, replicate to a quorum, commit, apply, and return the applied result:

```rust,no_run
use std::sync::Arc;
use std::time::Duration;
use distributed_kv_raft::{KVClient, NodeAddress, TestCluster};
use distributed_kv_raft::config::RaftConfigBuilder;
use distributed_kv_raft::transport::MemoryNetwork;

# async fn run() {
let ids = [0u64, 1, 2];
let configs = ids.iter().map(|&id| {
    let mut b = RaftConfigBuilder::default().id(id);
    for &p in &ids { if p != id { b = b.peer(p, format!("127.0.0.1:{}", 5000 + p)); } }
    b.build()
}).collect();

// Wire the nodes together over an in-process MemoryNetwork and start them.
let mut network = MemoryNetwork::new(&ids);
let client_transport = network.client_transport(); // reaches every node
let mut cluster = TestCluster::new(configs, &mut network);
cluster.start().await;
cluster.wait_for_leader(Duration::from_secs(3)).await.unwrap();

// The client follows NotLeader hints automatically; sending to a follower is fine.
let addrs = ids.iter().map(|&id| NodeAddress { id, addr: format!("127.0.0.1:{}", 5000 + id) }).collect();
let mut client = KVClient::new(addrs).with_transport(client_transport as Arc<_>);

client.put(b"name".to_vec(), b"raft".to_vec()).await.unwrap();
assert_eq!(client.get(b"name").await.unwrap(), Some(b"raft".to_vec()));
cluster.stop().await;
# }
```

The same `KVClient` path would run over `GrpcTransport` once that transport's client leg is
wired (see below); the request/redirect logic is transport-agnostic. See `cluster.rs` and
`tests/e2e_client_tests.rs` for the full working flow.

## What's Real vs Simulated

- **Real:** The Raft state machine (elections, log replication, commit/apply, conflict
  backtracking), pre-vote, check-quorum, the WAL with CRC checks and crash recovery, FSM
  snapshot/restore, `InstallSnapshot` handling, and joint-consensus membership are fully
  implemented. The `MemoryNetwork` transport is complete and backs the multi-node tests,
  including partition injection. **The end-to-end client path is real over
  `MemoryNetwork`:** `KVClient::put`/`get`/`delete` deliver a `ClientRequest` to a node's
  event loop, which proposes it through Raft, replicates to a quorum, commits, applies to
  the `KeyValueFSM`, and returns the applied result. A request sent to a follower is
  answered with `NotLeader { leader_hint }`, and the client automatically retries against
  the hinted leader (`Transport::send_client_request`, `RaftClusterNode::propose`,
  `tests/e2e_client_tests.rs`). Reads are served from the leader's applied state after a
  leadership check (read-after-commit).
- **Not yet wired:** The `GrpcTransport` client/server is structurally complete and uses
  real `tonic` channels for Raft RPCs, but its client-request leg is not wired — the
  default `Transport::send_client_request` returns a network error there, so the
  end-to-end path above runs only over `MemoryNetwork` today. gRPC TLS is a configuration
  flag (`GrpcConfig::enable_tls`) only; encryption is not wired up. WAL `truncate_suffix`
  and `compact` update in-memory state and the index but do not yet rewrite or delete
  on-disk segment files (stale segments remain until the next full rewrite).

## Testing

```bash
cargo test
```

The suite is 435 test functions across 9 files in `tests/` (plus inline `#[cfg(test)]`
modules). It covers Raft election and replication (`raft_tests.rs`), RPC handling
(`rpc_tests.rs`), WAL/FSM/snapshot storage (`storage_comprehensive_tests.rs`), the
transport layer and partitions (`transport_tests.rs`), client and config behavior
(`client_config_tests.rs`), metrics (`metrics_tests.rs`), edge cases (`edge_case_tests.rs`),
broad integration scenarios (`comprehensive_tests.rs`), and the end-to-end client command
path — put/get through Raft and follower redirect — over `MemoryNetwork`
(`e2e_client_tests.rs`). No external services are required; multi-node tests run over the
in-process `MemoryNetwork`.

## Project Structure

```
11-distributed-kv-raft/
  src/
    node.rs        # RaftNode: elections, replication, snapshots, membership
    storage.rs     # WriteAheadLog, KeyValueFSM, SnapshotStore, MemoryStorage
    transport.rs   # Transport trait, MemoryNetwork, PeerTracker
    grpc.rs        # tonic-based GrpcTransport, server builder, KV client
    rpc.rs         # Raft RPC message types
    cluster.rs     # RaftClusterNode event loop, TestCluster harness
    client.rs      # KVClient with leader discovery and retry
    config.rs      # RaftConfig builder, ClusterConfig quorum logic
    metrics.rs     # RaftMetrics, Histogram, HealthCheck
    error.rs       # Error / Result
    lib.rs         # public exports
  proto/raft.proto # gRPC service and message definitions
  tests/           # integration tests (9 files, incl. e2e_client_tests.rs)
  benches/         # Criterion microbenchmarks
  docs/BLUEPRINT.md   # Full architecture and Raft protocol design
```

## License

MIT — see [LICENSE](../LICENSE)
