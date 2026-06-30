# Time-Series Database

A high-performance time-series database (TSDB) written in Rust, featuring columnar storage, multiple compression schemes, a write-ahead log, and a query engine with aggregation and predicate pushdown.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §01 Rust async/concurrency · §02 Data warehousing · §02 Real-time analytics. See [../CONCEPT_TO_PROJECT_MAP.md](../CONCEPT_TO_PROJECT_MAP.md) for the full tutorial–project bridge.

## What's real vs simulated

The storage engine, compression codecs, WAL, and query executor are all genuine implementations — no random/placeholder outputs. One caveat: the BLUEPRINT describes "distributed sharding," but the implementation provides local-only in-process sharding. There is no network layer or true multi-node distribution.

## Layout

```
src/
  database.rs         — top-level Database API (open, insert, query, flush)
  types.rs            — DataPoint, Metric, SeriesKey, QueryResult types
  error.rs            — unified TsdbError type
  storage/
    engine.rs         — StorageEngine coordinating memtable + SSTables
    memtable.rs       — in-memory write buffer with sorted series map
    shard.rs          — time-range sharding logic
    sstable.rs        — SSTable read/write (block-based on-disk format)
  compression/
    delta.rs          — delta encoding for timestamps
    gorilla.rs        — Gorilla XOR compression for f64 values
    rle.rs            — run-length encoding
    varint.rs         — variable-length integer encoding
    dictionary.rs     — dictionary encoding for string tags
    block.rs          — compressed block container
  query/
    executor.rs       — query planning and execution
    aggregation.rs    — sum, avg, min, max, count, percentile aggregators
    functions.rs      — downsampling / rollup functions
    label_matcher.rs  — tag/label filter matching
    predicate.rs      — predicate pushdown interface
  wal/                — write-ahead log for durability
  retention/          — retention policy enforcement and compaction
tests/
  integration_tests.rs  — 161 integration tests across the full stack
benches/              — Criterion benchmarks (ingestion, query, compression)
BLUEPRINT.md          — architecture design and implementation plan
docs/                 — additional design notes
```

## Build & Run

```bash
cd 06-real-world-projects/52-time-series-database
cargo build
cargo test
cargo test -- --nocapture    # show println! output
cargo bench                  # run Criterion benchmarks
```

Requires Rust stable (edition 2021). No external services needed — all storage is local disk or in-memory.

## Key stats

- ~378 tests (217 unit + 161 integration)
- ~9,600 LOC
- Compression codecs: delta, Gorilla XOR, RLE, varint, dictionary
- Query features: time-range scan, aggregation, group-by, downsampling, predicate pushdown
