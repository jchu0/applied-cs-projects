# Distributed Log System (Kafka-lite)

A high-throughput distributed commit log in Rust, inspired by Apache Kafka: durable append-only storage with topic partitioning, ISR-based replication, consumer groups, and exactly-once producer semantics.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §02 Kafka streaming & exactly-once semantics, §02 real-time analytics. Compare to [Project 51 (Tokio message queue)](../51-message-queue/) and [Project 11 (Raft KV)](../11-distributed-kv-raft/). Used by [Project 08 (streaming platform)](../08-streaming-platform/) and [Project 36 (streaming analytics)](../36-distributed-streaming-analytics/).

---

## What's real vs simulated

Core mechanics are genuinely implemented: segmented write-ahead log with memory-mapped index files (via `memmap2`), CRC32 integrity checks, LZ4 compression, ISR replication protocol, consumer-group offset tracking, and idempotent producer sequence numbers. Two minor stubs remain: the log compaction scan in `cleaner.rs` returns an empty candidate list (placeholder comment, line 210), and the replication catch-up path in `replication.rs` uses a simplified placeholder approach (line 714). Neither affects the main produce/consume flow.

---

## Layout

```
src/
  log.rs           — segment files, index, mmap I/O
  broker.rs        — partition leader, request dispatch
  replication.rs   — ISR management, follower fetch loop
  controller.rs    — cluster metadata, leader election
  producer.rs      — batching, idempotent sequence tracking
  consumer.rs      — offset management, fetch protocol
  group.rs         — consumer group coordinator
  protocol.rs      — wire format (RecordBatch, varint, CRC)
  transport.rs     — Tokio TCP + tonic/gRPC transport
  idempotent.rs    — exactly-once producer state
  cleaner.rs       — log compaction (partially stubbed)
  lib.rs           — public API surface

tests/
  log_tests.rs              — segment, index, compaction (838 lines)
  broker_tests.rs           — partition, replication, failover (658 lines)
  producer_consumer_tests.rs — end-to-end produce/consume (734 lines)
  protocol_tests.rs         — wire encoding/decoding (735 lines)
  integration_tests.rs      — multi-broker cluster scenarios (768 lines)

benches/kafka_benchmarks.rs — Criterion throughput benchmarks
proto/                      — Protobuf definitions (tonic/gRPC)
BLUEPRINT.md                — full architecture, data structures, design rationale
PROGRESS.md                 — implementation status notes
```

225 test functions across 5 test files (~3 700 lines of tests).

---

## Build & Test

```bash
cd 06-real-world-projects/12-distributed-log-system
cargo build
cargo test
cargo test -- --nocapture          # show tracing output
cargo bench                        # throughput benchmarks
cargo clippy
```

Requires Rust 1.75+ (edition 2021). No external services needed — tests spin up in-process brokers via `tempfile`.
