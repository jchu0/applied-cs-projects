# High-Performance Cache (redis-lite)

A Redis-compatible in-memory caching server written from scratch in Rust. It implements
the RESP serialization protocol, a custom incrementally-rehashing hash table, typed value
objects, approximate LRU/LFU eviction, AOF and RDB persistence, and master-replica
replication primitives — exposed through a command dispatch layer that speaks the Redis
wire protocol.

## Features

- **RESP protocol** — full RESP2 parser and serializer for simple strings, errors,
  integers, bulk strings, and (nested) arrays (`RespValue` / `RespParser` in `resp/`),
  plus a RESP3 value layer (`Resp3Value`).
- **Custom hash table** — `Dict<K, V>` with two-table incremental rehashing (10 buckets
  moved per operation) and random-key sampling for eviction (`storage/dict.rs`).
- **Typed objects** — `RedisObject` covering strings, lists, sets, hashes, and sorted
  sets, with integer-encoded strings (`StringObject::Int`) and a score-sorted `ZSetObject`.
- **Command surface** — strings, lists, sets, hashes, sorted sets, key/TTL, and server
  commands via `CommandExecutor`, extended with Lua scripting and streams via
  `ExtendedExecutor`.
- **Eviction** — `EvictionManager` implements approximate LRU/LFU and random/TTL policies
  using random sampling (default sample size 5), driven by `EvictionPolicy`. The live
  server honors `--maxmemory` / `--maxmemory-policy`: before each write it recomputes the
  database's approximate memory usage and either rejects the write (`noeviction`, replying
  `OOM`) or frees keys per policy.
- **Persistence** — append-only file with `Always` / `EverySecond` / `No` fsync policies
  (`AOF` / `FsyncPolicy`) and RDB snapshotting (`RDB`) with a real CRC-64 trailing
  checksum that is verified on load. When `--appendonly` is set the live server appends
  every mutating command to the AOF and, on startup, replays the AOF through the real
  command-execution path. Otherwise it loads any existing RDB snapshot on startup.
- **Multi-database** — `--databases N` allocates N independent keyspaces; each connection
  tracks its own selected database via `SELECT <n>` (and `SWAPDB` swaps two), so keys are
  isolated per database.
- **Replication** — `ReplicationManager` with a ring-buffer replication backlog, replica
  registry, offset tracking, partial-resync checks, and replica-to-master promotion (the
  module exists but is not wired into the running server; see "What's Real vs Simulated").
- **TTL and expiration** — per-key expiry with lazy deletion on access plus an active
  expiration cycle the server runs each event-loop tick, and `TTL`/`PTTL`/`PERSIST`
  support (`storage/database.rs`).
- **Streams** — `Stream`, `StreamId`, `StreamEntry`, and `ConsumerGroup` backing the
  `XADD`/`XREAD`/`XRANGE`/`XGROUP` command family.
- **Transactions** — `MULTI`/`EXEC`/`DISCARD`/`WATCH`/`UNWATCH` queuing with optimistic
  locking (`TransactionContext`).
- **TLS** — `TlsAcceptor` / `TlsConfig` built on `rustls` for encrypted connections.

## Architecture

```mermaid
flowchart TD
    Client(Redis client / redis-cli) --> Server(Server event loop with mio)
    Server --> Conn(Connection read and write buffers)
    Conn --> Parser(RESP parser and serializer)
    Parser --> Exec(CommandExecutor per selected DB)
    Exec --> DB(Database)
    DB --> Dict(Dict with incremental rehashing)
    DB --> Obj(RedisObject typed values)
    Server --> Evict(EvictionManager per DB, on write path)
    Server --> Persist(AOF append and startup RDB or AOF load)
    Server --> Expire(Active expiration each tick)
```

| Component | Module | Responsibility |
|-----------|--------|----------------|
| Server | `server::Server` | mio-based event loop, accept loop, connection lifecycle |
| Connection | `server::Connection` | Per-client read/write buffering |
| TLS | `server::tls` | rustls-based TLS acceptor and stream |
| RESP | `resp::RespParser` / `RespValue` | Protocol parsing and serialization |
| Dispatch | `commands::CommandExecutor` | Maps command names to handlers |
| Extended dispatch | `commands::ExtendedExecutor` | Adds scripting and stream commands |
| Storage | `storage::Database` | Key-value store, expiration, type ops |
| Hash table | `storage::Dict` | Incrementally-rehashing dictionary |
| Objects | `storage::RedisObject` | Typed string/list/set/hash/zset values |
| Eviction | `eviction::EvictionManager` | Approximate LRU/LFU/random/TTL eviction |
| Persistence | `persistence::AOF` / `RDB` | Append-only log and snapshots |
| Replication | `replication::ReplicationManager` | Backlog, replica tracking, promotion |
| Streams | `storage::Stream` | Stream entries and consumer groups |
| Transactions | `transactions::TransactionContext` | MULTI/EXEC queuing and WATCH |

## Quick Start

### Prerequisites

- Rust 1.75+ (stable) with `cargo`
- No external services are needed to build, run, or test.

### Installation

The crate manifest lives in `src/`, so build from there:

```bash
cd 03-high-performance-cache/src
cargo build
```

### Running

```bash
cargo build --release
./target/release/redis-lite --port 6379
```

Available CLI flags include `--host`, `--port`, `--maxmemory`, `--maxmemory-policy`,
`--databases`, `--appendonly`, `--appendfilename`, `--dbfilename`, `--dir`, and
`--loglevel` (see `bin/main.rs`). Connect with any Redis client:

```bash
redis-cli -p 6379 SET foo bar
redis-cli -p 6379 GET foo
```

## Usage

The command layer is a pure function over `RespValue` arguments and a mutable `Database`,
which makes it easy to drive directly from Rust:

```rust
use redis_lite::commands::CommandExecutor;
use redis_lite::resp::RespValue;
use redis_lite::storage::Database;

let mut db = Database::new();

// SET foo bar
let resp = CommandExecutor::execute(
    "SET",
    &[RespValue::bulk_string("foo"), RespValue::bulk_string("bar")],
    &mut db,
);
assert_eq!(resp, RespValue::ok());

// GET foo
let resp = CommandExecutor::execute(
    "GET",
    &[RespValue::bulk_string("foo")],
    &mut db,
);
assert_eq!(resp, RespValue::BulkString(Some(b"bar".to_vec())));

// INCR counter
let resp = CommandExecutor::execute(
    "INCR",
    &[RespValue::bulk_string("counter")],
    &mut db,
);
assert_eq!(resp, RespValue::Integer(1));
```

Parsing and serializing the wire protocol directly:

```rust
use redis_lite::resp::{RespParser, RespValue};

let mut parser = RespParser::new();
parser.feed(b"*2\r\n$3\r\nGET\r\n$3\r\nfoo\r\n");
let value = parser.parse().unwrap().unwrap();

let bytes = RespValue::ok().serialize();
assert_eq!(bytes, b"+OK\r\n");
```

## What's Real vs Simulated

**Real and wired into the live server:** RESP2 parsing/serialization; the `Dict` hash
table with incremental rehashing; the `Database` key-value store with lazy + active
expiration; string/list/set/hash/sorted-set/key/TTL command handlers; per-connection
multi-database `SELECT`/`SWAPDB`; `--maxmemory` eviction enforcement on the write path
(`noeviction` → `OOM`, plus `allkeys`/`volatile` LRU/LFU/random/TTL sampling); AOF append
with all three fsync policies and startup replay through the real command path; and RDB
snapshot save/load with a verified CRC-64 checksum. These are exercised by both the unit
tests and the `tests/live_server.rs` integration tests that drive a real server over TCP.

**Real but not wired into the running server** (library types exist and are unit-tested,
but the event loop does not invoke them):
- **Replication** — `ReplicationManager`, backlog, offset tracking, and promotion.
- **Streams** — stream storage and consumer groups (reachable only via `ExtendedExecutor`,
  which the server does not construct).
- **Transactions / MULTI-EXEC** — queuing and WATCH logic exist but are not driven per
  connection by the server.
- **TLS** — the `rustls`-backed acceptor is implemented but not used by the plaintext
  event loop.
- **ThreadedIO** — the threaded I/O helper is standalone and unused by the server.

**Simulated / partial:**
- **Lua scripting** — `ScriptEngine` runs a small custom interpreter for a Lua-like
  subset (`scripting/mod.rs` notes that full Lua needs `mlua`/`rlua`); it is not a
  complete Lua VM, and is not wired into the server.
- **Pub/Sub commands** — a full `PubSub` registry type exists, but the `PUBLISH`/`PUBSUB`
  command handlers are not wired to it and return placeholder values (`PUBLISH` returns 0)
  because cross-connection client state is not propagated at the command layer.
- **Cluster mode** — `ClusterState` and slot mapping exist, but the executor invokes
  `CLUSTER` with no cluster state, so multi-node sharding and redirection are not active.
- **RDB scope** — snapshots serialize only database 0 and do not persist per-key TTLs, so
  a save/load round-trip preserves values but not expirations for keys outside DB 0.
- **RDB CRC** — the trailing checksum is a standard CRC-64 (`crc` crate, `CRC_64_XZ`); it
  detects corruption and round-trips within this implementation but is not byte-compatible
  with the CRC-64-Jones variant upstream Redis uses.

## Testing

```bash
cd 03-high-performance-cache/src
cargo test
```

The suite contains 340 unit tests across the storage, RESP, command, eviction,
persistence, replication, and supporting modules, plus 5 integration tests in
`tests/live_server.rs` that start a real `Server` over TCP and verify eviction (`OOM`
and `allkeys-lru`), AOF write-then-restart replay, `SELECT` database isolation, and
active expiration. No external services are required. Run with `cargo test -- --nocapture`
to see test output.

## Project Structure

```
03-high-performance-cache/
  README.md                  # This file
  docs/BLUEPRINT.md          # Full architecture and design
  src/
    Cargo.toml               # Crate manifest (build from here)
    lib.rs                   # Crate root and public re-exports
    bin/main.rs              # CLI entry point (clap)
    resp/                    # RESP2/RESP3 parser, value, serializer
    storage/                 # Database, Dict, RedisObject, streams
    commands/                # Command dispatch (strings, lists, sets, ...)
    eviction/                # LRU/LFU/random/TTL eviction manager
    persistence/             # AOF and RDB
    replication/             # Master/replica state and backlog
    pubsub/                  # Channel subscription registry
    transactions/            # MULTI/EXEC/DISCARD/WATCH
    scripting/               # Lua-subset script engine
    cluster/                 # Cluster slot mapping and node state
    server/                  # Event loop, connection, threaded IO, TLS
    config.rs                # Server configuration
    tests/live_server.rs     # Integration tests driving a live server over TCP
```

## License

MIT — see [LICENSE](../LICENSE)
