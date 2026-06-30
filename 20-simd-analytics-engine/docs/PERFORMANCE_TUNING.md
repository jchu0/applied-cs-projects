# Performance Tuning Guide

This guide covers optimization techniques and best practices for achieving maximum performance with the SIMD Analytics Engine.

## Table of Contents

1. [SIMD Vectorization](#simd-vectorization)
2. [Cache Optimization](#cache-optimization)
3. [Parallel Scaling](#parallel-scaling)
4. [NUMA Optimization](#numa-optimization)
5. [Memory Management](#memory-management)
6. [Query Planning](#query-planning)
7. [Profiling and Measurement](#profiling-and-measurement)
8. [Benchmarking Methodology](#benchmarking-methodology)

---

## SIMD Vectorization

### Understanding SIMD in This Engine

The engine uses AVX2-style vectorization (256-bit registers) processing 8 elements per operation:
- 8 x f32 (32-bit floats)
- 4 x f64 (64-bit floats)
- 8 x i32 (32-bit integers)
- 4 x i64 (64-bit integers)

### Vectorization Patterns

**Chunk Processing**: All SIMD operations process data in chunks of `VECTOR_WIDTH` (8 elements):

```rust
use simd_analytics_engine::{SimdOps, VECTOR_WIDTH};

// Process in vectorizable chunks
let data: Vec<f32> = vec![1.0; 1000];
let chunks = data.chunks_exact(VECTOR_WIDTH);
let remainder = chunks.remainder();

// Main loop - auto-vectorizes
for chunk in chunks {
    // Process 8 elements at once
}
// Handle remainder
```

**Verifying Vectorization**: Use `cargo rustc --release -- --emit=asm` to check generated assembly:

```bash
# Look for SIMD instructions
cargo rustc --release -- --emit=asm 2>/dev/null
grep -E "(vaddps|vmulps|vfmadd|vmovaps)" target/release/deps/*.s
```

Expected AVX2 instructions:
- `vaddps/vaddpd` - Packed addition
- `vmulps/vmulpd` - Packed multiplication
- `vfmadd*` - Fused multiply-add
- `vmovaps/vmovapd` - Aligned load/store
- `vcmpps/vcmppd` - Packed comparison

### Optimization Tips

1. **Alignment**: Always use `AlignedVec` for SIMD data:
   ```rust
   use simd_analytics_engine::AlignedVec;
   let aligned = AlignedVec::<f32>::with_capacity(1000);
   ```

2. **Avoid Branching**: Use `BranchFreeOps` for conditional operations:
   ```rust
   use simd_analytics_engine::BranchFreeOps;
   // Branch-free min/max avoids pipeline stalls
   let result = BranchFreeOps::select_f32(condition, a, b);
   ```

3. **Batch Operations**: Process data in large batches to amortize overhead:
   ```rust
   use simd_analytics_engine::SimdOps;
   // Single call processes entire array
   let sum = SimdOps::sum_f32(&data);
   ```

---

## Cache Optimization

### Cache Hierarchy Awareness

The engine is designed around typical cache sizes:
- L1: 32-64 KB per core (4-8 clock cycles)
- L2: 256-512 KB per core (12-20 clock cycles)
- L3: 8-32 MB shared (40-60 clock cycles)
- RAM: (100+ clock cycles)

### Cache Blocking

Use `CacheBlockedOps` to process data in cache-friendly blocks:

```rust
use simd_analytics_engine::{CacheBlockConfig, CacheBlockedOps};

let config = CacheBlockConfig {
    l1_block_size: 4096,    // ~16KB for f32
    l2_block_size: 32768,   // ~128KB for f32
    l3_block_size: 262144,  // ~1MB for f32
};

// Process with L1-resident blocking
let sum = CacheBlockedOps::blocked_sum_f32(&data, config.l1_block_size);
```

### Prefetching

Use `StreamingOps` for sequential access patterns:

```rust
use simd_analytics_engine::StreamingOps;

// Prefetch-optimized sequential processing
let sum = StreamingOps::streaming_sum_f32(&data);
```

Prefetch distance is set to 16 cache lines (`PREFETCH_DISTANCE * CACHE_LINE_SIZE = 1024 bytes`).

### Data Layout Guidelines

1. **Column-oriented**: Store data in columns for analytics workloads:
   ```rust
   use simd_analytics_engine::Column;
   // Each column is contiguous in memory
   let ages = Column::from_i32(vec![25, 30, 35, 40]);
   let salaries = Column::from_f64(vec![50000.0, 60000.0, 70000.0, 80000.0]);
   ```

2. **Hot/Cold Separation**: Keep frequently accessed columns together

3. **Struct of Arrays**: Prefer columnar layout over row-oriented:
   ```rust
   // Good: Columnar
   struct Table {
       col_a: Vec<f32>,
       col_b: Vec<f32>,
   }

   // Avoid: Row-oriented (poor cache utilization)
   struct Row { a: f32, b: f32 }
   struct Table { rows: Vec<Row> }
   ```

---

## Parallel Scaling

### Thread Pool Configuration

```rust
use simd_analytics_engine::ParallelExecutor;

let executor = ParallelExecutor::new();

// Configure thread count (defaults to num_cpus)
let executor = ParallelExecutor::with_threads(8);
```

### Parallel Aggregation Pattern

The engine uses a local-then-merge strategy for parallel aggregation:

```rust
use simd_analytics_engine::{ParallelExecutor, AggregateOp};

let executor = ParallelExecutor::new();

// Each thread computes local sum, then merge
let sum = executor.parallel_sum_f32(&data);
```

### Scaling Expectations

| Threads | Expected Scaling | Notes |
|---------|-----------------|-------|
| 1 | 1.0x | Baseline |
| 2 | 1.8-1.9x | Near-linear |
| 4 | 3.5-3.8x | Good scaling |
| 8 | 6.5-7.5x | Memory bandwidth may limit |
| 16+ | 8-12x | Diminishing returns |

### Scaling Limiters

1. **Memory Bandwidth**: Analytics workloads are often memory-bound
   - DDR4: ~50-70 GB/s per channel
   - SIMD can saturate memory bandwidth with few threads

2. **Work Granularity**: Ensure chunks are large enough:
   ```rust
   // Good: Large chunks (thousands of elements)
   let chunk_size = data.len() / num_threads;

   // Bad: Small chunks create scheduling overhead
   ```

3. **False Sharing**: Use padded accumulators (handled internally)

---

## NUMA Optimization

### NUMA Topology Detection

```rust
use simd_analytics_engine::NumaTopology;

let topology = NumaTopology::detect();
println!("NUMA nodes: {}", topology.num_nodes());
println!("Total CPUs: {}", topology.total_cpus());

for node in topology.nodes() {
    println!("Node {}: {} CPUs, {} MB memory",
             node.id, node.cpus.len(), node.memory_mb);
}
```

### NUMA-Local Allocation

```rust
use simd_analytics_engine::{NumaAllocator, NumaTopology};

let topology = NumaTopology::detect();
let allocator = NumaAllocator::new(topology);

// Allocate on specific NUMA node
let data: Vec<f32> = allocator.allocate_local(1_000_000, 0);

// Interleaved allocation across nodes
let interleaved: Vec<f32> = allocator.allocate_interleaved(1_000_000);
```

### NUMA-Partitioned Data

```rust
use simd_analytics_engine::NumaPartitionedVec;

// Data automatically partitioned across NUMA nodes
let partitioned = NumaPartitionedVec::new(data, &topology);

// Each partition is local to its NUMA node
for (node_id, partition) in partitioned.partitions() {
    // Process partition on its local node
}
```

### NUMA-Aware Execution

```rust
use simd_analytics_engine::NumaExecutor;

let executor = NumaExecutor::new(topology);

// Threads pinned to process local data
let sum = executor.parallel_sum_numa(&partitioned);
```

### NUMA Best Practices

1. **First-Touch Policy**: Initialize data on the thread that will use it
2. **Minimize Cross-Node Access**: Keep related data on same node
3. **Balance Load**: Distribute data evenly across nodes
4. **Pin Threads**: Use `AffinitySettings` to prevent migration

---

## Memory Management

### Aligned Allocation

All data should be 64-byte aligned (cache line size):

```rust
use simd_analytics_engine::AlignedVec;

// Automatically aligned to CACHE_LINE_SIZE (64 bytes)
let mut data = AlignedVec::<f32>::with_capacity(1000);
data.extend_from_slice(&[1.0; 1000]);

assert_eq!(data.as_ptr() as usize % 64, 0);
```

### Memory Pool Pattern

For repeated allocations, consider pre-allocating:

```rust
// Reuse buffers for multiple queries
let mut buffer = AlignedVec::<f32>::with_capacity(MAX_BATCH_SIZE);

for batch in batches {
    buffer.clear();
    buffer.extend_from_slice(&batch);
    process(&buffer);
}
```

### Avoiding Allocations in Hot Paths

```rust
// Bad: Allocates per iteration
for _ in 0..iterations {
    let temp = vec![0.0f32; 1000];  // Allocation!
    process(&temp);
}

// Good: Pre-allocate
let mut temp = vec![0.0f32; 1000];
for _ in 0..iterations {
    temp.fill(0.0);  // Reuse
    process(&temp);
}
```

---

## Query Planning

### Cost Model Configuration

```rust
use simd_analytics_engine::{CostModel, HardwareParams, QueryPlanner};

let hw_params = HardwareParams {
    l1_cache_size: 32 * 1024,
    l2_cache_size: 256 * 1024,
    l3_cache_size: 8 * 1024 * 1024,
    memory_bandwidth_gb_s: 50.0,
    num_cores: 8,
};

let cost_model = CostModel::new(hw_params);
let planner = QueryPlanner::new(cost_model);
```

### Column Statistics

Provide statistics for better planning:

```rust
use simd_analytics_engine::ColumnStats;

let stats = ColumnStats {
    row_count: 1_000_000,
    distinct_count: 10_000,
    null_count: 0,
    min_value: 0.0,
    max_value: 100.0,
    avg_size_bytes: 4,
};

planner.set_column_stats("price", stats);
```

### Plan Analysis

```rust
let plan = planner.plan_query(&query_ops);

// Examine plan
println!("Estimated cost: {}", plan.estimated_cost());
println!("Estimated rows: {}", plan.estimated_rows());

// Get cost breakdown
let estimate = cost_model.estimate(&plan);
println!("CPU cycles: {}", estimate.cpu_cycles);
println!("Memory accesses: {}", estimate.memory_accesses);
println!("Cache misses: {}", estimate.estimated_cache_misses);
```

---

## Profiling and Measurement

### Built-in Metrics

```rust
use simd_analytics_engine::{QueryMetrics, Timer};

let mut metrics = QueryMetrics::new();

// Time operations
metrics.start_phase("filter");
let filtered = filter(&data);
metrics.end_phase("filter");

metrics.start_phase("aggregate");
let sum = aggregate(&filtered);
metrics.end_phase("aggregate");

// Report
println!("{}", metrics.format_dashboard());
```

### Linux perf Integration

```bash
# Basic CPU counters
perf stat -e cycles,instructions,cache-misses,cache-references \
    cargo bench --bench simd_bench

# Detailed cache analysis
perf stat -e L1-dcache-loads,L1-dcache-load-misses,\
LLC-loads,LLC-load-misses \
    cargo bench --bench simd_bench

# Branch analysis
perf stat -e branches,branch-misses \
    cargo bench --bench simd_bench
```

### Key Metrics to Watch

| Metric | Good Value | Indicates |
|--------|------------|-----------|
| IPC (Instructions/Cycle) | > 2.0 | Good parallelism |
| L1 Cache Hit Rate | > 95% | Good locality |
| L3 Cache Hit Rate | > 80% | Reasonable working set |
| Branch Miss Rate | < 1% | Predictable code |
| Memory Bandwidth | Near peak | Efficient streaming |

### Interpreting Results

**Memory-Bound**: High IPC but low bandwidth utilization
- Solution: Improve prefetching, reduce working set

**Compute-Bound**: Low IPC, high cache hit rate
- Solution: Improve vectorization, reduce dependencies

**Cache-Thrashing**: Low cache hit rates
- Solution: Use cache blocking, improve data layout

---

## Benchmarking Methodology

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench -- simd_sum

# With detailed output
cargo bench -- --verbose
```

### Benchmark Best Practices

1. **Warm Up**: Let CPU reach steady state
   ```rust
   // Warm up run (discarded)
   for _ in 0..10 {
       let _ = operation(&data);
   }

   // Measured runs
   let start = Instant::now();
   for _ in 0..iterations {
       let _ = operation(&data);
   }
   ```

2. **Prevent Optimization**: Use `black_box` to prevent dead code elimination
   ```rust
   use std::hint::black_box;

   let result = black_box(operation(black_box(&data)));
   ```

3. **Multiple Iterations**: Use statistical analysis
   ```rust
   use simd_analytics_engine::Benchmark;

   let bench = Benchmark::new("sum_f32", || {
       SimdOps::sum_f32(&data)
   });

   let result = bench.run(100);  // 100 iterations
   println!("Mean: {} ns", result.mean_ns);
   println!("Std Dev: {} ns", result.std_dev_ns);
   println!("Throughput: {} GB/s", result.throughput_gb_s);
   ```

4. **Control Environment**:
   ```bash
   # Disable turbo boost for consistent results
   echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

   # Set CPU governor to performance
   sudo cpupower frequency-set -g performance

   # Isolate CPUs for benchmarking
   taskset -c 0-7 cargo bench
   ```

### Throughput Calculation

```rust
let bytes_processed = data.len() * std::mem::size_of::<f32>();
let seconds = duration.as_secs_f64();
let throughput_gb_s = (bytes_processed as f64 / 1e9) / seconds;

println!("Throughput: {:.2} GB/s", throughput_gb_s);
```

### Expected Performance Targets

| Operation | Target Throughput | Notes |
|-----------|------------------|-------|
| SIMD Sum (f32) | 40-60 GB/s | Memory bandwidth limited |
| SIMD Filter | 30-50 GB/s | Depends on selectivity |
| Hash Aggregate | 15-25 GB/s | Hash table overhead |
| Parallel Sum (8 cores) | 100-200 GB/s | Near memory bandwidth peak |

---

## Troubleshooting Performance Issues

### Issue: Poor SIMD Utilization

**Symptoms**: Low throughput, scalar instructions in assembly

**Solutions**:
1. Ensure data is aligned (`AlignedVec`)
2. Process in `VECTOR_WIDTH` chunks
3. Avoid complex conditionals in hot loops
4. Check compiler flags: `-C target-cpu=native`

### Issue: Cache Thrashing

**Symptoms**: High L3 miss rate, low throughput

**Solutions**:
1. Use cache blocking (`CacheBlockedOps`)
2. Reduce working set size
3. Improve data locality (columnar layout)
4. Prefetch data (`StreamingOps`)

### Issue: Poor Parallel Scaling

**Symptoms**: Adding threads doesn't improve performance

**Solutions**:
1. Check for contention (shared mutable state)
2. Increase work granularity
3. Use NUMA-aware allocation
4. Pin threads to cores

### Issue: Memory Bandwidth Saturation

**Symptoms**: All cores at 100% but throughput plateaus

**Solutions**:
1. Accept this is the limit (good problem to have!)
2. Reduce memory traffic (compress data)
3. Improve cache hit rate
4. Use streaming stores for write-only data

---

## Summary Checklist

Before deploying to production, verify:

- [ ] Data is 64-byte aligned (`AlignedVec`)
- [ ] Operations use SIMD paths (`SimdOps`)
- [ ] Cache blocking configured for data size
- [ ] Thread count matches available cores
- [ ] NUMA topology detected (if applicable)
- [ ] Query planner configured with statistics
- [ ] Benchmarks run and targets met
- [ ] perf analysis shows expected behavior
