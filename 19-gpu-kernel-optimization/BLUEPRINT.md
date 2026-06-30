# GPU Numerical Kernel Optimization (cuBLAS-lite GEMM)

## Project Overview

A deep dive into GPU kernel optimization through implementing progressively optimized versions of General Matrix Multiplication (GEMM). This project explores GPU architecture fundamentals including memory hierarchy, thread organization, and optimization techniques like tiling, register blocking, and double-buffering. The goal is to achieve near-cuBLAS performance while understanding the underlying principles.

> **Concepts covered:** [§03 CUDA basics](../../03-machine-learning-engineering/06-cuda-optimization/cuda-basics/cuda-basics.md) · [§03 Custom CUDA kernels](../../03-machine-learning-engineering/06-cuda-optimization/custom-kernels/cuda-custom-kernels.md) · [§03 Triton programming](../../03-machine-learning-engineering/06-cuda-optimization/triton/triton-programming.md). Pairs with [Project 39 (GPU memory manager)](../39-gpu-memory-manager/) and [Project 48 (multi-GPU kernel scheduler)](../48-multi-gpu-kernel-scheduler/) for the surrounding infrastructure. Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture

### Optimization Progression

```
┌─────────────────────────────────────────────────────────────┐
│                    Optimization Stages                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: Naive GEMM                                        │
│  - One thread per output element                            │
│  - Global memory access only                                │
│  - ~10 GFLOPS                                               │
│                                                              │
│  ───────────────────▼──────────────────────                 │
│                                                              │
│  Stage 2: Tiled GEMM (Shared Memory)                        │
│  - Block-level tiling                                       │
│  - Shared memory caching                                    │
│  - ~100 GFLOPS                                              │
│                                                              │
│  ───────────────────▼──────────────────────                 │
│                                                              │
│  Stage 3: Register Tiling                                   │
│  - Thread-level tiling                                      │
│  - Register accumulation                                    │
│  - ~300 GFLOPS                                              │
│                                                              │
│  ───────────────────▼──────────────────────                 │
│                                                              │
│  Stage 4: Double Buffering                                  │
│  - Overlap compute and memory                               │
│  - Software pipelining                                      │
│  - ~500 GFLOPS                                              │
│                                                              │
│  ───────────────────▼──────────────────────                 │
│                                                              │
│  Stage 5: Advanced Optimizations                            │
│  - Vectorized loads (LDG.128)                               │
│  - Bank conflict resolution                                 │
│  - ~800 GFLOPS                                              │
│                                                              │
│  ───────────────────▼──────────────────────                 │
│                                                              │
│  Stage 6: Tensor Cores (Stretch)                            │
│  - WMMA API                                                 │
│  - Mixed precision                                          │
│  - ~4000+ GFLOPS                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### GPU Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Architecture                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │
│  │    SM 0     │ │    SM 1     │ │    SM N     │        │
│  │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │        │
│  │ │  Warp   │ │ │ │  Warp   │ │ │ │  Warp   │ │        │
│  │ │Scheduler│ │ │ │Scheduler│ │ │ │Scheduler│ │        │
│  │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │        │
│  │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │        │
│  │ │ 64KB L1 │ │ │ │ 64KB L1 │ │ │ │ 64KB L1 │ │        │
│  │ │/Shared  │ │ │ │/Shared  │ │ │ │/Shared  │ │        │
│  │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │        │
│  │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │        │
│  │ │Register │ │ │ │Register │ │ │ │Register │ │        │
│  │ │  File   │ │ │ │  File   │ │ │ │  File   │ │        │
│  │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │        │
│  └─────────────┘ └─────────────┘ └─────────────┘        │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │              L2 Cache (6MB)                  │       │
│  └──────────────────────────────────────────────┘       │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │          Global Memory (HBM2 80GB)           │       │
│  │              Bandwidth: 2TB/s                 │       │
│  └──────────────────────────────────────────────┘       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Memory Hierarchy

### Access Latencies and Bandwidth

| Memory Type | Size | Latency | Bandwidth | Scope |
|-------------|------|---------|-----------|-------|
| Registers | 256KB/SM | 1 cycle | ~20 TB/s | Thread |
| Shared Memory | 48-164KB/SM | ~20 cycles | ~10 TB/s | Block |
| L1 Cache | 48-128KB/SM | ~30 cycles | ~4 TB/s | SM |
| L2 Cache | 6MB | ~200 cycles | ~2 TB/s | Device |
| Global Memory | 80GB | ~400 cycles | ~2 TB/s | Device |

### Memory Access Patterns

```cuda
// Bad: Strided access (not coalesced)
// Each thread accesses different cache line
float val = A[threadIdx.x * N + col];

// Good: Coalesced access
// Adjacent threads access adjacent memory
float val = A[row * N + threadIdx.x];

// Optimal: Vectorized coalesced access
float4 vals = *reinterpret_cast<float4*>(&A[row * N + threadIdx.x * 4]);
```

## GEMM Kernel Implementations

### Stage 1: Naive GEMM

```cuda
// C = A * B, where A is MxK, B is KxN, C is MxN
__global__ void naive_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch: <<<dim3((N+15)/16, (M+15)/16), dim3(16, 16)>>>
// Issues:
// - No data reuse
// - K loads of A and B per output element
// - Poor arithmetic intensity
```

### Stage 2: Tiled GEMM with Shared Memory

```cuda
#define TILE_SIZE 32

__global__ void tiled_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Iterate over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative load into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### Stage 3: Register Tiling

```cuda
#define BM 128      // Block tile M
#define BN 128      // Block tile N
#define BK 8        // Block tile K
#define TM 8        // Thread tile M
#define TN 8        // Thread tile N

__global__ void register_tiled_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // Thread position within block
    int tx = threadIdx.x % (BN / TN);
    int ty = threadIdx.x / (BN / TN);

    // Register accumulation
    float Cregs[TM][TN] = {0.0f};
    float Aregs[TM];
    float Bregs[TN];

    // Global positions
    int a_row_start = blockIdx.y * BM;
    int b_col_start = blockIdx.x * BN;

    for (int bk = 0; bk < K; bk += BK) {
        // Load A and B tiles into shared memory
        // (collaborative loading by all threads)
        load_tile_A(A, As, a_row_start, bk, M, K);
        load_tile_B(B, Bs, bk, b_col_start, K, N);
        __syncthreads();

        // Compute on registers
        for (int k = 0; k < BK; k++) {
            // Load from shared to registers
            for (int m = 0; m < TM; m++) {
                Aregs[m] = As[ty * TM + m][k];
            }
            for (int n = 0; n < TN; n++) {
                Bregs[n] = Bs[k][tx * TN + n];
            }
            // Outer product
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    Cregs[m][n] += Aregs[m] * Bregs[n];
                }
            }
        }
        __syncthreads();
    }

    // Write results to global memory
    store_results(C, Cregs, a_row_start + ty * TM, b_col_start + tx * TN, M, N);
}
```

### Stage 4: Double Buffering

```cuda
__global__ void double_buffered_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[2][BM][BK];  // Double buffer
    __shared__ float Bs[2][BK][BN];

    int buffer_idx = 0;

    // Prefetch first tile
    load_tile_A(A, As[0], ...);
    load_tile_B(B, Bs[0], ...);

    float Cregs[TM][TN] = {0.0f};

    for (int bk = 0; bk < K; bk += BK) {
        __syncthreads();

        // Prefetch next tile while computing current
        if (bk + BK < K) {
            load_tile_A(A, As[1 - buffer_idx], ...);
            load_tile_B(B, Bs[1 - buffer_idx], ...);
        }

        // Compute on current buffer
        compute_tile(As[buffer_idx], Bs[buffer_idx], Cregs);

        buffer_idx = 1 - buffer_idx;
    }

    store_results(C, Cregs, ...);
}
```

### Stage 5: Advanced Optimizations

```cuda
// Vectorized loads with float4
__device__ void load_tile_vectorized(
    const float* A, float As[BM][BK+4],  // Padding to avoid bank conflicts
    int row_start, int k_start, int M, int K
) {
    int tid = threadIdx.x;
    int loads_per_thread = (BM * BK) / (blockDim.x * 4);

    for (int i = 0; i < loads_per_thread; i++) {
        int idx = (tid * 4 + i * blockDim.x * 4);
        int local_row = idx / BK;
        int local_col = idx % BK;

        int global_row = row_start + local_row;
        int global_col = k_start + local_col;

        if (global_row < M && global_col + 3 < K) {
            float4 val = *reinterpret_cast<const float4*>(
                &A[global_row * K + global_col]);
            As[local_row][local_col] = val.x;
            As[local_row][local_col + 1] = val.y;
            As[local_row][local_col + 2] = val.z;
            As[local_row][local_col + 3] = val.w;
        }
    }
}

// Bank conflict-free shared memory layout
// Use padding or swizzling
__shared__ float As[BM][BK + 4];  // +4 padding
```

## Data Layout Considerations

### Row-Major vs Column-Major

```
Row-Major (C style):        Column-Major (Fortran):
A[i][j] = A[i*N + j]        A[i][j] = A[j*M + i]

Memory layout:              Memory layout:
[a00 a01 a02]               [a00 a10 a20]
[a10 a11 a12]               [a01 a11 a21]
[a20 a21 a22]               [a02 a12 a22]
```

### Coalescing Impact

```cuda
// Row-major A (good for reading rows)
float val = A[row * K + k];  // Threads read same row, different k

// Column-major A (good for reading columns)
float val = A[k * M + row];  // Adjacent threads read adjacent memory

// For optimal GEMM:
// - A stored column-major or transposed
// - B stored row-major
// - Or use transpose kernels
```

## Performance Analysis

### Arithmetic Intensity

```
GEMM: C = A * B
Operations: 2*M*N*K (multiply-add)
Memory: M*K + K*N + M*N (read A, B, write C)

Arithmetic Intensity = 2*M*N*K / (M*K + K*N + M*N)

For M=N=K: AI = 2*N^3 / 3*N^2 = 2N/3

Large N → compute bound
Small N → memory bound

To be compute bound on A100 (19.5 TFLOPS, 2 TB/s):
AI > 19.5 / 2 = 9.75 FLOP/byte
Need N > 3 * 9.75 / 2 = 14.6 → N > ~15
```

### Occupancy Calculation

```
Given:
- Kernel uses 64 registers per thread
- Kernel uses 48KB shared memory per block
- Block size: 256 threads

Limits:
- Registers: 65536 per SM / 64 per thread = 1024 threads max
- Shared: 164KB per SM / 48KB per block = 3 blocks max
- Threads per SM: 2048 max

Occupancy: min(
    256 * 3 / 2048,  // from shared memory
    1024 / 2048,      // from registers
    1.0               // from threads
) = min(0.375, 0.5, 1.0) = 37.5%
```

### Performance Metrics

```cpp
// CUDA timing
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);

// GFLOPS calculation
double gflops = (2.0 * M * N * K) / (ms * 1e6);

// Effective bandwidth
double bytes = (M * K + K * N + M * N) * sizeof(float);
double bandwidth = bytes / (ms * 1e6);  // GB/s
```

## Enterprise Features

### Kernel Autotuner

```cpp
struct KernelConfig {
    int BM, BN, BK;
    int TM, TN;
    int warps_per_block;
};

class Autotuner {
public:
    KernelConfig tune(int M, int N, int K) {
        std::vector<KernelConfig> configs = generate_configs();
        KernelConfig best;
        float best_time = INFINITY;

        for (const auto& config : configs) {
            if (!is_valid(config, M, N, K)) continue;

            float time = benchmark(config, M, N, K);
            if (time < best_time) {
                best_time = time;
                best = config;
            }
        }

        return best;
    }

private:
    std::vector<KernelConfig> generate_configs() {
        std::vector<KernelConfig> configs;
        for (int BM : {64, 128, 256}) {
            for (int BN : {64, 128, 256}) {
                for (int BK : {4, 8, 16}) {
                    for (int TM : {4, 8}) {
                        for (int TN : {4, 8}) {
                            configs.push_back({BM, BN, BK, TM, TN, ...});
                        }
                    }
                }
            }
        }
        return configs;
    }

    bool is_valid(const KernelConfig& config, int M, int N, int K) {
        // Check shared memory usage
        int smem = (config.BM * config.BK + config.BK * config.BN) * sizeof(float);
        if (smem > MAX_SHARED_MEMORY) return false;

        // Check register usage
        int regs_per_thread = config.TM * config.TN + config.TM + config.TN;
        if (regs_per_thread > MAX_REGISTERS) return false;

        return true;
    }
};
```

### Performance Dashboard

```cpp
struct KernelMetrics {
    float gflops;
    float efficiency;         // vs theoretical peak
    float occupancy;
    float memory_throughput;
    int l2_cache_hit_rate;
    int shared_mem_bank_conflicts;
};

void collect_metrics(KernelMetrics& metrics) {
    // Use CUPTI or Nsight Compute
    // Collect hardware counters
}

void print_dashboard(const KernelMetrics& metrics) {
    printf("╔════════════════════════════════════╗\n");
    printf("║      GEMM Kernel Performance       ║\n");
    printf("╠════════════════════════════════════╣\n");
    printf("║ GFLOPS:           %8.2f         ║\n", metrics.gflops);
    printf("║ Efficiency:       %7.1f%%         ║\n", metrics.efficiency * 100);
    printf("║ Occupancy:        %7.1f%%         ║\n", metrics.occupancy * 100);
    printf("║ Mem Throughput:   %6.1f GB/s      ║\n", metrics.memory_throughput);
    printf("║ L2 Hit Rate:      %7.1f%%         ║\n", metrics.l2_cache_hit_rate);
    printf("╚════════════════════════════════════╝\n");
}
```

## Performance Considerations

### Optimization Checklist

1. **Memory Coalescing**
   - Adjacent threads access adjacent memory
   - Align memory accesses to 128 bytes
   - Use vectorized loads (float4)

2. **Shared Memory**
   - Avoid bank conflicts (32 banks)
   - Use padding or swizzling
   - Balance with register usage

3. **Register Usage**
   - Maximize data reuse in registers
   - Balance occupancy vs registers
   - Use register tiling

4. **Occupancy**
   - Aim for 50%+ for memory-bound
   - Can be lower for compute-bound
   - Use occupancy calculator

5. **Instruction-Level Parallelism**
   - Unroll loops
   - Double buffering
   - Independent operations

### Common Pitfalls

```cuda
// Pitfall 1: Strided global access
float val = A[threadIdx.x * stride];  // Bad!

// Pitfall 2: Shared memory bank conflicts
As[threadIdx.x][col];  // All threads hit same bank if col is same

// Pitfall 3: Thread divergence
if (threadIdx.x % 2 == 0) { ... }  // Warp divergence

// Pitfall 4: Excessive synchronization
for (int k = 0; k < K; k++) {
    __syncthreads();  // Unnecessary in loop
}

// Pitfall 5: Low occupancy from register spilling
// Too many local variables
```

## Implementation Phases

### Phase 1: Setup & Naive (Weeks 1-2)
- [ ] CUDA development environment
- [ ] Matrix generation and validation
- [ ] Naive GEMM implementation
- [ ] Basic timing infrastructure
- [ ] Compare with cuBLAS baseline

### Phase 2: Tiled GEMM (Weeks 3-4)
- [ ] Shared memory tiling
- [ ] Optimal tile size selection
- [ ] Boundary handling
- [ ] Performance comparison

### Phase 3: Register Tiling (Weeks 5-6)
- [ ] Thread-level tiling design
- [ ] Register blocking implementation
- [ ] Memory access pattern optimization
- [ ] Performance analysis

### Phase 4: Double Buffering (Weeks 7-8)
- [ ] Double buffer implementation
- [ ] Memory/compute overlap
- [ ] Pipeline analysis
- [ ] Synchronization tuning

### Phase 5: Advanced Optimizations (Weeks 9-10)
- [ ] Vectorized loads
- [ ] Bank conflict elimination
- [ ] Instruction scheduling
- [ ] Launch bound optimization

### Phase 6: Autotuner (Weeks 11-12)
- [ ] Configuration space definition
- [ ] Benchmarking infrastructure
- [ ] Search algorithm
- [ ] Caching tuned results

### Phase 7: Stretch Goals (Weeks 13-14)
- [ ] Tensor Core WMMA
- [ ] Batched GEMM
- [ ] Mixed precision (FP16)
- [ ] Fused operations

## Testing Strategy

### Correctness Tests

```cpp
void test_correctness() {
    // Test against cuBLAS
    float *A, *B, *C_custom, *C_cublas;
    // Allocate and initialize...

    custom_gemm(A, B, C_custom, M, N, K);
    cublas_gemm(A, B, C_cublas, M, N, K);

    float max_error = 0;
    for (int i = 0; i < M * N; i++) {
        float error = fabs(C_custom[i] - C_cublas[i]);
        float relative = error / fabs(C_cublas[i]);
        max_error = max(max_error, relative);
    }

    assert(max_error < 1e-5);  // FP32 tolerance
}
```

### Performance Tests

```cpp
void benchmark_all() {
    std::vector<int> sizes = {512, 1024, 2048, 4096, 8192};

    for (int N : sizes) {
        auto [time_naive, gflops_naive] = benchmark(naive_gemm, N);
        auto [time_tiled, gflops_tiled] = benchmark(tiled_gemm, N);
        auto [time_opt, gflops_opt] = benchmark(optimized_gemm, N);
        auto [time_cublas, gflops_cublas] = benchmark(cublas_gemm, N);

        printf("%d x %d:\n", N, N);
        printf("  Naive:     %7.2f GFLOPS\n", gflops_naive);
        printf("  Tiled:     %7.2f GFLOPS\n", gflops_tiled);
        printf("  Optimized: %7.2f GFLOPS\n", gflops_opt);
        printf("  cuBLAS:    %7.2f GFLOPS\n", gflops_cublas);
        printf("  Efficiency: %.1f%%\n", 100.0 * gflops_opt / gflops_cublas);
    }
}
```

## Stretch Goals

### Tensor Cores (WMMA)

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void tensor_core_gemm(
    half* A, half* B, float* C, int M, int N, int K
) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + row * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + col, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    wmma::store_matrix_sync(C + row * N + col, c_frag, N, wmma::mem_row_major);
}
```

### Batched GEMM

```cuda
// C[i] = A[i] * B[i] for batch of matrices
__global__ void batched_gemm(
    float** A, float** B, float** C,
    int M, int N, int K, int batch_size
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) return;

    // Standard GEMM on A[batch_idx], B[batch_idx], C[batch_idx]
}
```

### Fused Operations

```cuda
// C = alpha * A * B + beta * C (standard BLAS)
// C = ReLU(A * B + bias)        (neural network)
```

## Technology Stack

- **Language**: CUDA C++
- **Compiler**: nvcc with -arch=sm_80 (or appropriate)
- **Profiling**: Nsight Compute, nvprof
- **Testing**: Catch2 or GoogleTest
- **Baseline**: cuBLAS

## References

- [CUTLASS](https://github.com/NVIDIA/cutlass) - NVIDIA's template library
- [How to Optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [GPU Gems](https://developer.nvidia.com/gpugems/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
