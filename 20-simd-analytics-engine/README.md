# SIMD Analytics Engine

A CPU-optimized columnar analytics engine in Rust that explores SIMD vectorization, cache-conscious algorithms, NUMA-aware memory management, and vectorized query execution across scan, filter, project, and aggregate operators.

> **Status:** reference implementation / teaching scaffold built to a strong blueprint — not production-grade. See [../PROJECTS_STATUS.md](../PROJECTS_STATUS.md) and the [2026-06 audit](../../docs/AUDIT_2026-06_public-readiness.md).

> **Concepts covered:** §01 `rust/04-unsafe-rust/`, §07 `benchmarks/languages/`

## What's real vs simulated

The columnar storage, query planner, filter/aggregate/hash pipelines, and multi-core scheduling via Rayon are fully implemented. However:

- **No SIMD intrinsics** — all "SIMD" paths rely on LLVM auto-vectorization rather than explicit `std::arch::x86_64::_mm256_*` calls. The blueprint's AVX2/AVX-512 code paths are not yet written.
- **NUMA is simulated** — `numa.rs` falls back to a single-node topology on macOS and any system without `libnuma`. Node-pinned allocation and cross-node bandwidth are modelled in software, not enforced by the OS.
- **Prefetch and non-temporal stores are annotated but not emitted** — `optimize.rs` documents where `_mm256_stream_pd` and prefetch intrinsics would go; the actual calls are ordinary scalar or iterator code.

## Layout

```
src/
  simd.rs        — vectorized kernels (LLVM autovec; no hand-written intrinsics)
  column.rs      — aligned columnar buffer primitives
  filter.rs      — predicate evaluation over columns
  aggregate.rs   — SUM / MIN / MAX / AVG aggregation
  hash.rs        — hash aggregation and probe tables
  planner.rs     — query plan builder and optimizer rules
  scheduler.rs   — multi-core work scheduling (Rayon)
  optimize.rs    — cache-blocking, prefetch hints (simulated), streaming writes
  numa.rs        — NUMA topology detection and simulated node-local allocation
  metrics.rs     — performance counters and timing

tests/
  test_hash.rs      — 48 hash pipeline tests
  test_metrics.rs   — 28 metrics and counter tests

benches/
  simd_benchmarks   — Criterion benchmarks for hot paths

BLUEPRINT.md      — full architecture and design decisions
PROGRESS.md       — implementation status
```

## Build & test

```bash
cd 06-real-world-projects/20-simd-analytics-engine
cargo build
cargo test
```

To enable the `avx2` feature flag (scaffolded; intrinsics not yet wired):

```bash
cargo build --features avx2
```

Run benchmarks:

```bash
cargo bench
```

The test suite covers ~145 unit tests across source modules and the two integration test files (~85 % of blueprint scope complete).
