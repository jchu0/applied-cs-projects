# Distributed Key-Value Store with Raft Consensus

A strongly consistent distributed key-value store built on the Raft consensus algorithm, providing linearizable read/write operations across a replicated cluster with automatic leader election, log compaction via snapshots, and dynamic cluster membership changes.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §01 `rust/*` (Rust systems programming), §02 `04-streaming` (replication intuition)

---

## Layout

```
src/
  node.rs        — RaftNode state machine: leader election, log replication, pre-vote
  storage.rs     — WriteAheadLog, KeyValueFSM, MemoryStorage, SnapshotStore
  transport.rs   — Transport trait, MemoryNetwork (in-process), PeerTracker
  grpc.rs        — gRPC transport (GrpcTransport, GrpcServerBuilder, GrpcKvClient)
  rpc.rs         — Raft RPC message types (AppendEntries, RequestVote, InstallSnapshot)
  cluster.rs     — RaftClusterNode and TestCluster harness for multi-node tests
  client.rs      — KVClient, NodeAddress, RequestTracker
  config.rs      — RaftConfig / ClusterConfig with builder
  metrics.rs     — RaftMetrics, ClusterMetrics, HealthCheck, Histogram
  error.rs       — unified Error / Result types
proto/           — .proto definitions for gRPC transport
tests/           — 838 integration tests across 8 files (raft, storage, transport, RPC, chaos, edge cases)
benches/         — Criterion benchmarks (raft_benchmarks)
BLUEPRINT.md     — full architecture and Raft protocol design doc
```

## What's real vs simulated

The Raft state machine, WAL, FSM, snapshot install, and pre-vote extension are fully implemented in Rust. The in-process `MemoryNetwork` transport is used for all multi-node tests and functions correctly. The gRPC transport layer (`GrpcTransport`) is structurally complete; TLS is marked as a placeholder and is not implemented. No core paths use `unimplemented!` or random/mock outputs.

## Build and test

```bash
cd 06-real-world-projects/11-distributed-kv-raft
cargo build
cargo test
cargo bench   # optional — Criterion benchmarks
```

Requires Rust stable (edition 2021) and the `protoc` protobuf compiler for the gRPC build step (`build.rs` invokes `tonic-build`).

## Key capabilities

- Leader election with randomized timeouts and pre-vote phase
- Log replication with back-pressure and per-peer `nextIndex` / `matchIndex` tracking
- Log compaction via `InstallSnapshot` RPC
- Joint-consensus membership changes (`ConfigurationChange`)
- In-process `MemoryNetwork` and gRPC (`tonic`) transports — same `Transport` trait
- Prometheus-style metrics (counters, gauges, histograms) via `RaftMetrics`
- `TestCluster` harness enabling multi-node chaos and partition tests
