# Message Queue

A high-performance, persistent message queue implemented in Rust, featuring topic-based pub/sub with partitioning, consumer groups with rebalancing, and at-least-once / exactly-once delivery semantics backed by a Write-Ahead Log.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §01 `rust/05-async-rust/rust-async.md` (Tokio async I/O) · §01 `rust/03-concurrency/rust-concurrency.md` · §02 `04-streaming` (Kafka, exactly-once, consumer groups). Pairs with [Project 12 (distributed log)](../12-distributed-log-system/) and [Project 08 (streaming platform)](../08-streaming-platform/).

---

## What's real vs simulated

The core in-process queue (storage, segmented WAL, partitioning, consumer groups, compression) is fully implemented. What the PROJECTS_STATUS tracker notes as absent: **no networked server/client** — `mq-server` binary exists but the broker runs in-process only, so multi-node clustering, replication, and KRaft-style coordination are not implemented.

---

## Layout

```
51-message-queue/
├── src/
│   ├── lib.rs              # Crate root and public API
│   ├── broker.rs           # Broker — routes producers/consumers to topics
│   ├── topic.rs / partition.rs  # Topic and partition management
│   ├── segment.rs          # Segment files (append-only log slices)
│   ├── storage.rs          # Persistent storage layer
│   ├── index.rs            # Offset → file-position index
│   ├── offset.rs           # Offset tracking
│   ├── producer.rs         # Batched producer API
│   ├── consumer.rs         # Consumer API with ack semantics
│   ├── consumer_group.rs   # Group membership and partition rebalancing
│   ├── transaction.rs      # Exactly-once transaction support
│   ├── compression.rs      # LZ4 / Snappy / Gzip compression codecs
│   ├── message.rs          # Message and MessageId types
│   ├── config.rs           # Broker / topic configuration
│   ├── error.rs            # Typed error hierarchy
│   └── bin/server.rs       # Binary entry-point (in-process only)
├── tests/
│   └── integration_tests.rs   # 75 integration tests
├── benches/
│   └── throughput.rs       # Criterion throughput benchmarks
├── BLUEPRINT.md            # Full design doc and architecture diagrams
├── PROGRESS.md             # Implementation status notes
└── Cargo.toml
```

---

## Build & Run

```bash
cd 06-real-world-projects/51-message-queue
cargo build
cargo test
cargo build --release        # LTO + codegen-units=1 optimised build
cargo bench                  # throughput benchmarks (release profile)
```

Run the server binary (in-process mode):

```bash
cargo run --bin mq-server
```
