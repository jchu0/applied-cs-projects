# High-Performance Columnar Analytics Engine (SIMD + Vectorized)

> **Concepts covered:** §01 software-engineering — `rust/04-unsafe-rust`; §07 infrastructure — `benchmarks/languages`

## Project Overview

A CPU-optimized columnar analytics engine that leverages modern CPU features for maximum performance. This project explores SIMD vectorization (AVX2/AVX-512/NEON), cache-conscious algorithms, NUMA-aware memory management, and hardware performance counters. The goal is to achieve optimal CPU utilization for analytical queries through deep understanding of CPU microarchitecture.

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     Query Execution Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐     │
│  │SQL/DataFrame│──►│Query Planner │──►│SIMD Code Generator│     │
│  │   Query     │   │& Optimizer   │   │                   │     │
│  └─────────────┘   └──────────────┘   └─────────┬─────────┘     │
│                                                  │               │
│  ┌──────────────────────────────────────────────▼──────────────┐│
│  │              Vectorized Execution Engine                     ││
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐         ││
│  │  │  Scan   │  │ Filter  │  │ Project │  │Aggregate│         ││
│  │  │(SIMD)   │  │(SIMD)   │  │(SIMD)   │  │(SIMD)   │         ││
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘         ││
│  │       │            │            │            │               ││
│  │       └────────────┴────────────┴────────────┘               ││
│  │                         │                                    ││
│  │  ┌──────────────────────▼───────────────────────────────┐   ││
│  │  │            Multi-Core Parallel Execution              │   ││
│  │  │   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐             │   ││
│  │  │   │Core 0│  │Core 1│  │Core 2│  │Core N│             │   ││
│  │  │   │NUMA 0│  │NUMA 0│  │NUMA 1│  │NUMA 1│             │   ││
│  │  │   └──────┘  └──────┘  └──────┘  └──────┘             │   ││
│  │  └──────────────────────────────────────────────────────┘   ││
│  └──────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │              Columnar Storage Engine                          ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           ││
│  │  │ Column A    │  │ Column B    │  │ Column C    │           ││
│  │  │ [aligned]   │  │ [aligned]   │  │ [aligned]   │           ││
│  │  │ [prefetch]  │  │ [prefetch]  │  │ [prefetch]  │           ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘           ││
│  └──────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### CPU Architecture Awareness

```
┌───────────────────────────────────────────────────────────┐
│                   Modern CPU Architecture                  │
├───────────────────────────────────────────────────────────┤
│                                                            │
│  Core 0                        Core 1                      │
│  ┌─────────────────┐          ┌─────────────────┐         │
│  │ L1D: 32KB, 4cy  │          │ L1D: 32KB, 4cy  │         │
│  │ L1I: 32KB       │          │ L1I: 32KB       │         │
│  └────────┬────────┘          └────────┬────────┘         │
│  ┌────────┴────────┐          ┌────────┴────────┐         │
│  │ L2: 256KB, 12cy │          │ L2: 256KB, 12cy │         │
│  └────────┬────────┘          └────────┬────────┘         │
│           └──────────┬─────────────────┘                   │
│           ┌──────────┴──────────┐                          │
│           │ L3: 30MB, 40cy      │                          │
│           └──────────┬──────────┘                          │
│                      │                                     │
│  ┌───────────────────┴───────────────────┐                │
│  │  Memory Controller  (DDR4/DDR5)       │                │
│  │  Bandwidth: 50-100 GB/s               │                │
│  │  Latency: ~100ns                      │                │
│  └───────────────────────────────────────┘                │
│                                                            │
│  SIMD Units:                                               │
│  - SSE: 128-bit (4 x float)                               │
│  - AVX2: 256-bit (8 x float)                              │
│  - AVX-512: 512-bit (16 x float)                          │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

## SIMD Vectorization

### Vector Types and Operations

```cpp
#include <immintrin.h>

// AVX2 (256-bit = 8 floats or 4 doubles)
using Vec8f = __m256;
using Vec4d = __m256d;
using Vec8i = __m256i;

// AVX-512 (512-bit = 16 floats or 8 doubles)
using Vec16f = __m512;
using Vec8d = __m512d;
using Vec16i = __m512i;

// Load/Store operations
Vec8f load_aligned(const float* ptr) {
    return _mm256_load_ps(ptr);
}

Vec8f load_unaligned(const float* ptr) {
    return _mm256_loadu_ps(ptr);
}

void store_aligned(float* ptr, Vec8f v) {
    _mm256_store_ps(ptr, v);
}

// Arithmetic
Vec8f add(Vec8f a, Vec8f b) { return _mm256_add_ps(a, b); }
Vec8f mul(Vec8f a, Vec8f b) { return _mm256_mul_ps(a, b); }
Vec8f fma(Vec8f a, Vec8f b, Vec8f c) { return _mm256_fmadd_ps(a, b, c); }
```

### Vectorized Filter

```cpp
// Filter: col > threshold
// Returns bitmask of matching elements

__m256i simd_filter_gt_f32(
    const float* col,
    float threshold,
    int64_t count
) {
    __m256 thresh_vec = _mm256_set1_ps(threshold);
    int64_t i = 0;

    // Process 8 elements at a time
    for (; i + 8 <= count; i += 8) {
        __m256 data = _mm256_loadu_ps(col + i);
        __m256 cmp = _mm256_cmp_ps(data, thresh_vec, _CMP_GT_OQ);
        int mask = _mm256_movemask_ps(cmp);

        // Store or process mask
        // Each bit indicates if element passes filter
    }

    // Handle remainder
    for (; i < count; i++) {
        // Scalar fallback
    }
}

// AVX-512 version with native masking
__mmask16 simd_filter_gt_f32_avx512(
    const float* col,
    float threshold,
    int64_t count,
    uint16_t* result_masks
) {
    __m512 thresh_vec = _mm512_set1_ps(threshold);

    for (int64_t i = 0; i + 16 <= count; i += 16) {
        __m512 data = _mm512_loadu_ps(col + i);
        __mmask16 mask = _mm512_cmp_ps_mask(data, thresh_vec, _CMP_GT_OQ);
        result_masks[i / 16] = mask;
    }
}
```

### Vectorized Aggregation

```cpp
// SUM aggregation with SIMD
double simd_sum_f64(const double* col, int64_t count) {
    __m256d sum_vec = _mm256_setzero_pd();

    int64_t i = 0;
    // Process 4 doubles at a time
    for (; i + 4 <= count; i += 4) {
        __m256d data = _mm256_loadu_pd(col + i);
        sum_vec = _mm256_add_pd(sum_vec, data);
    }

    // Horizontal sum
    __m128d sum_hi = _mm256_extractf128_pd(sum_vec, 1);
    __m128d sum_lo = _mm256_castpd256_pd128(sum_vec);
    __m128d sum128 = _mm_add_pd(sum_lo, sum_hi);
    sum128 = _mm_hadd_pd(sum128, sum128);

    double result = _mm_cvtsd_f64(sum128);

    // Handle remainder
    for (; i < count; i++) {
        result += col[i];
    }

    return result;
}

// Conditional SUM with predicate mask
double simd_sum_filtered(
    const double* col,
    const uint64_t* mask,
    int64_t count
) {
    __m256d sum_vec = _mm256_setzero_pd();

    for (int64_t i = 0; i + 4 <= count; i += 4) {
        // Load mask bits
        uint8_t m = (mask[i / 64] >> (i % 64)) & 0xF;

        // Masked load (AVX2: use blend)
        __m256d data = _mm256_loadu_pd(col + i);
        __m256d mask_vec = _mm256_castsi256_pd(
            _mm256_set_epi64x(
                (m & 8) ? -1 : 0,
                (m & 4) ? -1 : 0,
                (m & 2) ? -1 : 0,
                (m & 1) ? -1 : 0
            )
        );
        data = _mm256_and_pd(data, mask_vec);
        sum_vec = _mm256_add_pd(sum_vec, data);
    }

    // Horizontal sum...
}
```

### Vectorized Hash Aggregation

```cpp
// Hash group keys and aggregate
struct HashAggregator {
    struct Entry {
        uint64_t key_hash;
        double sum;
        int64_t count;
    };

    std::vector<Entry> table;
    size_t mask;  // power of 2 - 1

    void aggregate(const int64_t* keys, const double* values, int64_t count) {
        // Vectorized hashing
        for (int64_t i = 0; i + 4 <= count; i += 4) {
            __m256i key_vec = _mm256_loadu_si256((__m256i*)(keys + i));
            __m256i hash_vec = hash_keys(key_vec);

            // Store hashes and process
            alignas(32) int64_t hashes[4];
            _mm256_store_si256((__m256i*)hashes, hash_vec);

            for (int j = 0; j < 4; j++) {
                size_t idx = hashes[j] & mask;
                // Linear probing...
                table[idx].sum += values[i + j];
                table[idx].count++;
            }
        }
    }

private:
    __m256i hash_keys(__m256i keys) {
        // MurmurHash3 finalizer
        keys = _mm256_xor_si256(keys, _mm256_srli_epi64(keys, 33));
        keys = _mm256_mullo_epi64(keys, _mm256_set1_epi64x(0xff51afd7ed558ccd));
        keys = _mm256_xor_si256(keys, _mm256_srli_epi64(keys, 33));
        return keys;
    }
};
```

## Cache Optimization

### Prefetching

```cpp
// Software prefetching for sequential scan
void scan_with_prefetch(const float* data, float* output, int64_t count) {
    constexpr int PREFETCH_DISTANCE = 16;  // Cache lines ahead

    for (int64_t i = 0; i < count; i += 8) {
        // Prefetch future data
        _mm_prefetch(&data[i + PREFETCH_DISTANCE * 8], _MM_HINT_T0);

        // Process current data
        __m256 vec = _mm256_load_ps(&data[i]);
        __m256 result = process(vec);
        _mm256_store_ps(&output[i], result);
    }
}

// Prefetch into L2 for later use
_mm_prefetch(ptr, _MM_HINT_T1);

// Non-temporal hint (bypass cache)
_mm_prefetch(ptr, _MM_HINT_NTA);
```

### Cache-Blocking

```cpp
// Process data in cache-friendly blocks
void blocked_aggregate(
    const double* col_a,
    const double* col_b,
    double* results,
    int64_t count
) {
    constexpr int64_t BLOCK_SIZE = 4096;  // Fits in L1

    for (int64_t block = 0; block < count; block += BLOCK_SIZE) {
        int64_t block_end = std::min(block + BLOCK_SIZE, count);

        // Process block that fits in cache
        for (int64_t i = block; i < block_end; i += 4) {
            __m256d a = _mm256_load_pd(&col_a[i]);
            __m256d b = _mm256_load_pd(&col_b[i]);
            __m256d r = _mm256_mul_pd(a, b);
            _mm256_store_pd(&results[i], r);
        }
    }
}
```

### Memory Alignment

```cpp
// Aligned allocation
template<typename T>
T* aligned_alloc(size_t count, size_t alignment = 64) {
    void* ptr = std::aligned_alloc(alignment, count * sizeof(T));
    return static_cast<T*>(ptr);
}

// Check alignment
bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) & (alignment - 1)) == 0;
}

// Columnar storage with aligned columns
struct Column {
    void* data;
    size_t count;
    DataType type;

    Column(DataType type, size_t count)
        : type(type), count(count) {
        size_t bytes = count * type_size(type);
        data = std::aligned_alloc(64, bytes);  // Cache line aligned
    }
};
```

## NUMA-Aware Execution

### NUMA Topology Detection

```cpp
#include <numa.h>

struct NUMATopology {
    int num_nodes;
    std::vector<std::vector<int>> node_cpus;
    std::vector<size_t> node_memory;

    NUMATopology() {
        if (numa_available() < 0) {
            // NUMA not available, single node
            num_nodes = 1;
            return;
        }

        num_nodes = numa_num_configured_nodes();
        node_cpus.resize(num_nodes);

        for (int node = 0; node < num_nodes; node++) {
            struct bitmask* cpus = numa_allocate_cpumask();
            numa_node_to_cpus(node, cpus);

            for (int cpu = 0; cpu < numa_num_configured_cpus(); cpu++) {
                if (numa_bitmask_isbitset(cpus, cpu)) {
                    node_cpus[node].push_back(cpu);
                }
            }

            numa_free_cpumask(cpus);
        }
    }
};
```

### NUMA-Aware Memory Allocation

```cpp
// Allocate on specific NUMA node
void* numa_alloc_on_node(size_t size, int node) {
    return numa_alloc_onnode(size, node);
}

// Interleave across nodes for bandwidth
void* numa_alloc_interleaved(size_t size) {
    return numa_alloc_interleaved(size);
}

// Partition data across NUMA nodes
struct NUMAPartitionedColumn {
    std::vector<void*> partitions;
    std::vector<size_t> partition_sizes;
    int num_nodes;

    NUMAPartitionedColumn(size_t total_size, DataType type) {
        num_nodes = numa_num_configured_nodes();
        size_t per_node = (total_size + num_nodes - 1) / num_nodes;

        for (int node = 0; node < num_nodes; node++) {
            size_t size = std::min(per_node, total_size - node * per_node);
            partitions.push_back(numa_alloc_onnode(size, node));
            partition_sizes.push_back(size);
        }
    }
};
```

### NUMA-Aware Scheduling

```cpp
class NUMAScheduler {
    NUMATopology topology;
    std::vector<std::thread> workers;

public:
    void execute_partitioned(
        NUMAPartitionedColumn& data,
        std::function<void(void*, size_t)> task
    ) {
        std::vector<std::future<void>> futures;

        for (int node = 0; node < topology.num_nodes; node++) {
            // Pin thread to NUMA node
            auto future = std::async(std::launch::async, [&, node]() {
                // Set CPU affinity to this NUMA node
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                for (int cpu : topology.node_cpus[node]) {
                    CPU_SET(cpu, &cpuset);
                }
                pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);

                // Process local partition
                task(data.partitions[node], data.partition_sizes[node]);
            });

            futures.push_back(std::move(future));
        }

        // Wait for all nodes
        for (auto& f : futures) {
            f.wait();
        }
    }
};
```

## Performance Monitoring

### Hardware Performance Counters

```cpp
#include <linux/perf_event.h>
#include <sys/ioctl.h>

class PerfCounter {
    int fd;
    uint64_t count;

public:
    enum Type {
        CYCLES,
        INSTRUCTIONS,
        CACHE_MISSES,
        BRANCH_MISSES,
        L1D_MISS,
        LLC_MISS,
    };

    PerfCounter(Type type) {
        struct perf_event_attr pe = {};
        pe.type = PERF_TYPE_HARDWARE;
        pe.size = sizeof(pe);
        pe.disabled = 1;
        pe.exclude_kernel = 1;

        switch (type) {
            case CYCLES:
                pe.config = PERF_COUNT_HW_CPU_CYCLES;
                break;
            case INSTRUCTIONS:
                pe.config = PERF_COUNT_HW_INSTRUCTIONS;
                break;
            case CACHE_MISSES:
                pe.config = PERF_COUNT_HW_CACHE_MISSES;
                break;
            // ...
        }

        fd = perf_event_open(&pe, 0, -1, -1, 0);
    }

    void start() {
        ioctl(fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    }

    uint64_t stop() {
        ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
        read(fd, &count, sizeof(count));
        return count;
    }
};

// Usage
PerfCounter cycles(PerfCounter::CYCLES);
PerfCounter instructions(PerfCounter::INSTRUCTIONS);

cycles.start();
instructions.start();

// ... computation ...

uint64_t c = cycles.stop();
uint64_t i = instructions.stop();
double ipc = (double)i / c;
```

### Performance Dashboard

```cpp
struct QueryMetrics {
    // Timing
    double total_time_ms;
    double scan_time_ms;
    double filter_time_ms;
    double aggregate_time_ms;

    // Throughput
    double rows_per_second;
    double gb_per_second;

    // CPU metrics
    double ipc;
    double cpu_utilization;

    // Cache metrics
    double l1_hit_rate;
    double l2_hit_rate;
    double l3_hit_rate;

    // Memory
    double memory_bandwidth_gb;
    int numa_local_percent;
};

void print_dashboard(const QueryMetrics& m) {
    printf("╔═══════════════════════════════════════════╗\n");
    printf("║     Query Performance Dashboard          ║\n");
    printf("╠═══════════════════════════════════════════╣\n");
    printf("║ Total Time:      %8.2f ms              ║\n", m.total_time_ms);
    printf("║ Throughput:      %8.2f M rows/s       ║\n", m.rows_per_second / 1e6);
    printf("║ Bandwidth:       %8.2f GB/s           ║\n", m.gb_per_second);
    printf("║                                          ║\n");
    printf("║ IPC:             %8.2f                 ║\n", m.ipc);
    printf("║ CPU Utilization: %7.1f%%                ║\n", m.cpu_utilization * 100);
    printf("║                                          ║\n");
    printf("║ L1 Hit Rate:     %7.1f%%                ║\n", m.l1_hit_rate * 100);
    printf("║ L2 Hit Rate:     %7.1f%%                ║\n", m.l2_hit_rate * 100);
    printf("║ L3 Hit Rate:     %7.1f%%                ║\n", m.l3_hit_rate * 100);
    printf("║                                          ║\n");
    printf("║ NUMA Local:      %7.1f%%                ║\n", m.numa_local_percent);
    printf("╚═══════════════════════════════════════════╝\n");
}
```

## Enterprise Features

### Cost Model

```cpp
struct CostEstimate {
    double cpu_cycles;
    double memory_accesses;
    double cache_misses;
    double total_cost;
};

class CostModel {
    // Hardware parameters
    double cycles_per_simd_op = 1.0;
    double cycles_per_branch = 5.0;
    double l1_latency = 4.0;
    double l2_latency = 12.0;
    double l3_latency = 40.0;
    double mem_latency = 100.0;

public:
    CostEstimate estimate_scan(size_t rows, size_t columns) {
        CostEstimate cost;
        size_t bytes = rows * columns * sizeof(double);

        // Estimate cache behavior
        double l3_size = 30 * 1024 * 1024;
        if (bytes < l3_size) {
            cost.cache_misses = bytes / 64;  // Cache line misses
            cost.memory_accesses = 0;
        } else {
            cost.memory_accesses = bytes / 64;
        }

        cost.cpu_cycles = rows * cycles_per_simd_op / 8;  // SIMD width
        cost.total_cost = cost.cpu_cycles +
                         cost.cache_misses * l3_latency +
                         cost.memory_accesses * mem_latency;
        return cost;
    }

    CostEstimate estimate_filter(size_t rows, double selectivity) {
        // Filter cost depends on branching
        // Vectorized filter has predictable cost
    }

    CostEstimate estimate_hash_aggregate(size_t rows, size_t groups) {
        // Hash table access pattern
    }
};
```

### Profiling Integration

```bash
# perf profiling
perf record -e cycles,instructions,cache-misses,L1-dcache-load-misses ./analytics_engine
perf report

# VTune-style analysis
perf record -e cpu-cycles,cache-references,cache-misses,branch-instructions,branch-misses ./analytics_engine
perf stat ./analytics_engine

# Memory bandwidth
perf stat -e uncore_imc/cas_count_read/,uncore_imc/cas_count_write/ ./analytics_engine
```

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] Columnar storage with aligned allocations
- [ ] Basic SIMD utilities (load, store, arithmetic)
- [ ] Microbenchmark framework
- [ ] Performance counter integration

### Phase 2: SIMD Operators (Weeks 3-5)
- [ ] Vectorized scan
- [ ] Vectorized filter (comparison predicates)
- [ ] Vectorized projection (arithmetic expressions)
- [ ] Vectorized aggregation (SUM, COUNT, AVG, MIN, MAX)

### Phase 3: Multi-Core Execution (Weeks 6-7)
- [ ] Thread pool implementation
- [ ] Data partitioning
- [ ] Parallel scan and filter
- [ ] Parallel aggregation with local + merge

### Phase 4: Hash Operations (Weeks 8-9)
- [ ] Vectorized hashing
- [ ] SIMD hash table probing
- [ ] Hash aggregation
- [ ] Hash join (probe phase)

### Phase 5: NUMA Awareness (Weeks 10-11)
- [ ] NUMA topology detection
- [ ] NUMA-local allocation
- [ ] Partitioned execution
- [ ] NUMA-aware scheduling

### Phase 6: Advanced Optimizations (Weeks 12-13)
- [ ] Cache blocking
- [ ] Software prefetching tuning
- [ ] Branch-free algorithms
- [ ] Memory bandwidth optimization

### Phase 7: Enterprise Features (Weeks 14-15)
- [ ] Performance dashboard
- [ ] Cost model implementation
- [ ] Query profiling
- [ ] Autotuning framework

### Phase 8: Polish (Week 16)
- [ ] Comprehensive benchmarks
- [ ] Documentation
- [ ] Performance tuning guide
- [ ] Demo queries

## Testing Strategy

### Microbenchmarks

```cpp
void benchmark_simd_sum() {
    std::vector<double> data(10'000'000);
    // Initialize with random data

    auto start = std::chrono::high_resolution_clock::now();

    double result = simd_sum_f64(data.data(), data.size());

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    double gb = data.size() * sizeof(double) / 1e9;
    double bandwidth = gb / (ms / 1000);

    printf("SIMD Sum: %.2f ms, %.2f GB/s\n", ms, bandwidth);
}
```

### Correctness Tests

```cpp
TEST(SIMDFilter, Correctness) {
    std::vector<float> data = {1, 2, 3, 4, 5, 6, 7, 8};
    auto result = simd_filter_gt_f32(data.data(), 4.0f, data.size());

    // Should match: 5, 6, 7, 8 (indices 4, 5, 6, 7)
    EXPECT_EQ(popcount(result), 4);
}

TEST(Aggregation, SumCorrectness) {
    std::vector<double> data(1000);
    double expected = 0;
    for (int i = 0; i < 1000; i++) {
        data[i] = i;
        expected += i;
    }

    double result = simd_sum_f64(data.data(), data.size());
    EXPECT_DOUBLE_EQ(result, expected);
}
```

### Scalability Tests

```cpp
void test_numa_scalability() {
    std::vector<size_t> sizes = {1'000'000, 10'000'000, 100'000'000};

    for (size_t size : sizes) {
        for (int threads = 1; threads <= max_threads; threads *= 2) {
            auto result = benchmark_with_threads(size, threads);
            printf("Size: %zu, Threads: %d, Time: %.2f ms, Speedup: %.2fx\n",
                   size, threads, result.time_ms, result.speedup);
        }
    }
}
```

## Stretch Goals

### Vectorized UDF Wrappers

```cpp
// Allow user-defined functions to benefit from SIMD
template<typename F>
void simd_apply(F&& f, const double* input, double* output, size_t count) {
    // Compile-time check if F can be vectorized

    for (size_t i = 0; i + 4 <= count; i += 4) {
        __m256d in = _mm256_loadu_pd(&input[i]);

        // Attempt to vectorize f
        // Fall back to scalar if not possible

        _mm256_storeu_pd(&output[i], result);
    }
}
```

### JIT Expression Compilation

```cpp
// Generate specialized SIMD code for expressions
class ExpressionCompiler {
    void compile(const Expr& expr) {
        // Generate AVX2/AVX-512 code for expression tree
        // Handle different architectures
    }
};
```

### Adaptive Execution

```cpp
// Choose best implementation based on data characteristics
class AdaptiveExecutor {
    void execute(Query& query) {
        // Analyze data distribution
        // Choose vectorized vs scalar
        // Adjust parallelism
        // Select optimal algorithms
    }
};
```

## Technology Stack

- **Language**: C++17/20
- **SIMD**: Intel Intrinsics, Highway library
- **Parallelism**: OpenMP or custom thread pool
- **NUMA**: libnuma
- **Profiling**: perf, PAPI
- **Testing**: Google Test, Google Benchmark

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)
- [What Every Programmer Should Know About Memory](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf)
- [Performance Analysis Guide](https://www.intel.com/content/www/us/en/develop/documentation/vtune-cookbook/)
- [SIMD Everywhere (Highway)](https://github.com/google/highway)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
