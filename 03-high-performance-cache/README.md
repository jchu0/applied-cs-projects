# High-Performance Cache (redis-lite)

A Redis-compatible in-memory caching server implemented in Rust, covering the full stack from TCP I/O and RESP protocol parsing through eviction policies, persistence, pub/sub, Lua scripting, and master-replica replication.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §01 `rust/00-fundamentals`, `rust/05-async-rust`; §07 `benchmarks/databases`

---

## What's real vs simulated

The core data path (storage, RESP parsing, eviction, AOF/RDB persistence, replication backlog) is fully implemented. Two areas are stubs:

- **Cluster mode** — `CLUSTER INFO` and `CLUSTER KEYSLOT` return placeholder values; true multi-node sharding is not wired.
- **Pub/Sub subscriber count** — `PUBLISH` returns 0 in standalone mode because the subscriber registry is not propagated across connections.

Everything else — string/list/set/sorted-set/hash/stream commands, LRU/LFU eviction, TTL management, TLS, Lua scripting hooks, and master/replica sync — is real.

---

## Layout

```
src/
  bin/main.rs          Entry point (CLI via clap)
  lib.rs               Crate root
  server/              TCP listener, connection lifecycle
  resp/                RESP2/RESP3 parser and encoder
  commands/            Command dispatch (strings, lists, sets, hashes, sorted sets,
                       streams, pub/sub, transactions, scripting, cluster)
  storage/             In-memory database, dict, typed objects, streams
  eviction/            LRU / LFU policy manager
  persistence/         AOF (append-only file) and RDB snapshot
  replication/         Master broadcaster, replica sync, replication backlog
  pubsub/              Channel subscription state
  transactions/        MULTI/EXEC/DISCARD/WATCH
  scripting/           Lua scripting hooks
  cluster/             Cluster command stubs
  config.rs            TOML configuration

BLUEPRINT.md           Full design document (read before modifying)
PROGRESS.md            Implementation status notes
```

---

## Build & Run

```bash
cd 06-real-world-projects/03-high-performance-cache
cargo build
cargo test
```

Release build (LTO enabled):

```bash
cargo build --release
./target/release/redis-lite --port 6379
```

Connect with any Redis client or `redis-cli`:

```bash
redis-cli -p 6379 SET foo bar
redis-cli -p 6379 GET foo
```

Run tests with output:

```bash
cargo test -- --nocapture
```

The test suite contains 338 unit tests across all modules.
