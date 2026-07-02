# GPU GEMM Optimization - API Documentation

This crate (`gpu_gemm_optimization`) is organized as a set of **free functions**
and plain data structs, not an object-oriented kernel API. GEMM kernels are
functions that take `&Matrix` inputs and write into a `&mut Matrix` output.

## Table of Contents
1. [Error Handling](#error-handling)
2. [Matrix Module](#matrix-module)
3. [GEMM Module](#gemm-module)
4. [Vectorize Module](#vectorize-module)
5. [Memory Module](#memory-module)
6. [Metrics Module](#metrics-module)
7. [Autotuner Module](#autotuner-module)
8. [Complete Examples](#complete-examples)

## Error Handling

The entire crate uses a single error enum and result alias, re-exported at the
crate root:

```rust
pub enum Error {
    DimensionMismatch(String),
    InvalidConfig(String),
}

pub type Result<T> = std::result::Result<T, Error>;
```

`DimensionMismatch` is returned when matrix / batch shapes are incompatible;
`InvalidConfig` is returned by `GemmConfig::validate` when block/thread tiles do
not divide evenly. There are no per-module error types.

```rust
use gpu_gemm_optimization::{naive_gemm, Error, Matrix};

let a = Matrix::random(4, 8);
let b = Matrix::random(3, 4); // wrong: b.rows != a.cols
let mut c = Matrix::zeros(4, 3);

match naive_gemm(&a, &b, &mut c) {
    Ok(()) => println!("ok"),
    Err(Error::DimensionMismatch(msg)) => eprintln!("shape error: {msg}"),
    Err(Error::InvalidConfig(msg)) => eprintln!("config error: {msg}"),
}
```

## Matrix Module

### `Matrix`
Dense, row-major matrix. Fields are public; there is no `Matrix::new` — use a
factory method.

```rust
pub struct Matrix {
    pub data: Vec<f32>, // row-major
    pub rows: usize,
    pub cols: usize,
}
```

### Creation Methods

| Method | Description |
| --- | --- |
| `Matrix::zeros(rows, cols) -> Matrix` | All elements `0.0`. |
| `Matrix::ones(rows, cols) -> Matrix` | All elements `1.0`. |
| `Matrix::random(rows, cols) -> Matrix` | Random values in `[-1, 1]`. |
| `Matrix::identity(size) -> Matrix` | `size × size` identity matrix. |

```rust
use gpu_gemm_optimization::Matrix;

let z = Matrix::zeros(10, 10);
let i = Matrix::identity(4);
let r = Matrix::random(50, 50);
```

### Access Methods

```rust
pub fn get(&self, row: usize, col: usize) -> f32;
pub fn set(&mut self, row: usize, col: usize, value: f32);
pub fn get_mut(&mut self, row: usize, col: usize) -> &mut f32;
pub fn row_ptr(&self, row: usize) -> &[f32];
pub fn row_ptr_mut(&mut self, row: usize) -> &mut [f32];
```

```rust
let mut mat = Matrix::zeros(3, 3);
mat.set(1, 1, 5.0);
assert_eq!(mat.get(1, 1), 5.0);

let row = Matrix::ones(4, 8).row_ptr(2).to_vec();
assert_eq!(row.len(), 8);
```

### Operations & Utilities

```rust
pub fn transpose(&self) -> Matrix;
pub fn frobenius_norm(&self) -> f32;
pub fn approx_eq(&self, other: &Matrix, tolerance: f32) -> bool;
pub fn max_diff(&self, other: &Matrix) -> f32;
pub fn max_relative_error(&self, other: &Matrix) -> f32;
```

```rust
let m = Matrix::random(2, 3);
let t = m.transpose();
assert_eq!((t.rows, t.cols), (3, 2));

let norm = Matrix::ones(2, 2).frobenius_norm(); // sqrt(4) = 2.0
```

### Layout Helpers

```rust
pub enum Layout {
    RowMajor,
    ColumnMajor,
}

pub fn convert_layout(data: &[f32], rows: usize, cols: usize, from: Layout, to: Layout) -> Vec<f32>;
pub fn pack_matrix_a(/* ... */) -> Vec<f32>;
pub fn pack_matrix_b(/* ... */) -> Vec<f32>;
```

`Layout` is a two-variant enum (`RowMajor`, `ColumnMajor`). `convert_layout`
re-lays out a flat buffer; `pack_matrix_a` / `pack_matrix_b` pack operand tiles
for blocked kernels.

## GEMM Module

### Configuration

```rust
pub struct GemmConfig {
    pub block_m: usize,   // block tile size, M
    pub block_n: usize,   // block tile size, N
    pub block_k: usize,   // block tile size, K
    pub thread_m: usize,  // per-thread tile, M
    pub thread_n: usize,  // per-thread tile, N
}
```

`GemmConfig::default()` yields `block_m=64, block_n=64, block_k=8, thread_m=8,
thread_n=8`. `config.validate() -> Result<()>` fails with `Error::InvalidConfig`
if `block_m % thread_m != 0` or `block_n % thread_n != 0`.

`GemmKernel` is a **trait** (`compute(&self, a, b, c) -> Result<()>` and
`name(&self) -> &'static str`), not a struct — implement it to wrap a kernel
behind a common interface. The kernels below are plain free functions.

### Core GEMM Kernels

All compute `C = A × B` and return `Result<()>`, validating that
`a.cols == b.rows`.

```rust
pub fn naive_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<()>;
pub fn tiled_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix, tile_size: usize) -> Result<()>;
pub fn register_tiled_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix, config: &GemmConfig) -> Result<()>;
pub fn parallel_tiled_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix, tile_size: usize) -> Result<()>;
pub fn double_buffered_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix, tile_size: usize) -> Result<()>;
```

```rust
use gpu_gemm_optimization::{Matrix, tiled_gemm};

let a = Matrix::random(128, 256);
let b = Matrix::random(256, 64);
let mut c = Matrix::zeros(128, 64);
tiled_gemm(&a, &b, &mut c, 32)?;
```

### Scaled GEMM

```rust
// C = alpha * (A * B) + beta * C
pub fn scaled_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix, alpha: f32, beta: f32) -> Result<()>;
```

### Batched GEMM

```rust
// C[i] = A[i] * B[i]
pub fn batched_gemm(a_batch: &[Matrix], b_batch: &[Matrix], c_batch: &mut [Matrix]) -> Result<()>;

// C[i] = alpha * (A[i] * B[i]) + beta * C[i]
pub fn batched_scaled_gemm(
    a_batch: &[Matrix], b_batch: &[Matrix], c_batch: &mut [Matrix],
    alpha: f32, beta: f32,
) -> Result<()>;

// Contiguous 3D-tensor layout addressed by per-operand strides
pub fn strided_batched_gemm(
    a_data: &[f32], b_data: &[f32], c_data: &mut [f32],
    batch_size: usize, m: usize, n: usize, k: usize,
    stride_a: usize, stride_b: usize, stride_c: usize,
) -> Result<()>;
```

### Fused Activation GEMM

```rust
pub enum Activation {
    None,
    ReLU,
    LeakyReLU(f32),
    GeLU,
    Sigmoid,
    Tanh,
    SiLU,
}

impl Activation {
    pub fn apply(&self, x: f32) -> f32;
}

// C = activation(A * B)
pub fn gemm_activation(a: &Matrix, b: &Matrix, c: &mut Matrix, activation: Activation) -> Result<()>;

// C = activation(A * B + bias)   (bias.len() == b.cols)
pub fn gemm_bias_activation(
    a: &Matrix, b: &Matrix, c: &mut Matrix, bias: &[f32], activation: Activation,
) -> Result<()>;

// C = alpha * activation(A * B + bias) + beta * C   (bias optional)
pub fn gemm_fused(
    a: &Matrix, b: &Matrix, c: &mut Matrix,
    alpha: f32, beta: f32, bias: Option<&[f32]>, activation: Activation,
) -> Result<()>;
```

```rust
use gpu_gemm_optimization::{Matrix, Activation, gemm_activation};

let a = Matrix::random(32, 64);
let b = Matrix::random(64, 16);
let mut c = Matrix::zeros(32, 16);
gemm_activation(&a, &b, &mut c, Activation::ReLU)?;
```

### Tensor Core / WMMA Simulation

```rust
pub struct WmmaConfig { pub m: usize, pub n: usize, pub k: usize }

impl WmmaConfig {
    pub fn m16n16k16() -> Self; // also: m8n32k16(), m32n8k16(), Default = 16x16x16
}

pub struct WmmaFragment { /* rows, cols, data */ }

impl WmmaFragment {
    pub fn new(rows: usize, cols: usize) -> Self;
    pub fn load_matrix_sync(&mut self, matrix: &Matrix, row_offset: usize, col_offset: usize);
    pub fn store_matrix_sync(&self, matrix: &mut Matrix, row_offset: usize, col_offset: usize);
    pub fn fill(&mut self, value: f32);
    pub fn get(&self, row: usize, col: usize) -> f32;
    pub fn set(&mut self, row: usize, col: usize, value: f32);
}

// D = A * B + C over WMMA fragments
pub fn wmma_mma_sync(
    a_frag: &WmmaFragment, b_frag: &WmmaFragment,
    c_frag: &WmmaFragment, d_frag: &mut WmmaFragment,
) -> Result<()>;

// Full GEMM tiled into WMMA fragments
pub fn wmma_gemm(a: &Matrix, b: &Matrix, c: &mut Matrix, config: WmmaConfig) -> Result<()>;
```

### Roofline Analysis

```rust
pub enum PerformanceBound { ComputeBound, MemoryBound, Balanced }

pub struct RooflineModel {
    pub peak_gflops: f64,
    pub peak_bandwidth_gbs: f64,
}

impl RooflineModel {
    pub fn new(peak_gflops: f64, peak_bandwidth_gbs: f64) -> Self; // Default = 10000, 900
    pub fn ridge_point(&self) -> f64;
    pub fn theoretical_peak(&self, arithmetic_intensity: f64) -> f64;
    pub fn classify_bound(&self, arithmetic_intensity: f64) -> PerformanceBound;
    pub fn gemm_arithmetic_intensity(m: usize, n: usize, k: usize) -> f64;
    pub fn tiled_gemm_arithmetic_intensity(m, n, k, tile_m, tile_n, tile_k: usize) -> f64;
    pub fn analyze_gemm(&self, m: usize, n: usize, k: usize, achieved_gflops: f64) -> RooflineAnalysis;
}

pub struct RooflineAnalysis {
    pub arithmetic_intensity: f64,
    pub theoretical_peak_gflops: f64,
    pub achieved_gflops: f64,
    pub efficiency_percent: f64,
    pub bound: PerformanceBound,
    pub ridge_point: f64,
}

impl RooflineAnalysis {
    pub fn recommendations(&self) -> Vec<&'static str>;
}
```

```rust
use gpu_gemm_optimization::RooflineModel;

let model = RooflineModel::default();
let analysis = model.analyze_gemm(1024, 1024, 1024, 350.0);
println!("{:?}: {:.1}% of peak", analysis.bound, analysis.efficiency_percent);
for rec in analysis.recommendations() {
    println!("- {rec}");
}
```

## Vectorize Module

```rust
pub struct Float4 { pub x: f32, pub y: f32, pub z: f32, pub w: f32 }
impl Float4 {
    pub fn new(data: [f32; 4]) -> Self;
    pub fn zero() -> Self;
    pub fn to_array(self) -> [f32; 4];
    pub fn add(self, other: Self) -> Self;
    pub fn scale(self, s: f32) -> Self;
    pub fn fma(self, a: f32, b: Self) -> Self;
}

pub struct Float8 { pub data: [f32; 8] }
impl Float8 {
    pub fn new(data: [f32; 8]) -> Self;
    pub fn zero() -> Self;
    pub fn add(self, other: Self) -> Self;
    pub fn scale(self, s: f32) -> Self;
}

pub trait VectorizedOps {
    fn load_float4(data: &[f32], offset: usize) -> Float4;
    fn load_float8(data: &[f32], offset: usize) -> Float8;
    fn store_float4(data: &mut [f32], offset: usize, value: Float4);
    fn store_float8(data: &mut [f32], offset: usize, value: Float8);
    fn is_aligned(offset: usize, alignment: usize) -> bool;
}

pub struct SimdVectorOps; // implements VectorizedOps

pub struct CoalescingAnalysis { /* ... */ }
impl CoalescingAnalysis {
    pub fn analyze(/* ... */) -> Self;
}
```

`vectorized_gemm` and `vectorized_gemm_transposed_b` are also provided as module
functions (`gpu_gemm_optimization::vectorize::*`) but are not re-exported at the
crate root.

## Memory Module

Shared-memory bank-conflict and occupancy analysis for GPU kernels.

```rust
pub const NUM_BANKS: usize = 32;
pub const BANK_WIDTH: usize = 4;
pub const MAX_SHARED_MEM: usize = 49152;

pub struct BankConflictAnalysis { /* ... */ }
impl BankConflictAnalysis {
    pub fn analyze(addresses: &[usize]) -> Self;
    pub fn is_conflict_free(&self) -> bool;
}

pub struct SharedMemoryConfig { /* width, height, element_size, padding */ }
impl SharedMemoryConfig {
    pub fn new(width: usize, height: usize, element_size: usize) -> Self;
    pub fn with_auto_padding(width: usize, height: usize, element_size: usize) -> Self;
    pub fn stride(&self) -> usize;
    pub fn size_bytes(&self) -> usize;
    pub fn index(&self, row: usize, col: usize) -> usize;
    pub fn fits_in_shared_memory(&self) -> bool;
}

pub struct OccupancyCalculator {
    pub max_threads_per_sm: usize,
    pub max_blocks_per_sm: usize,
    pub registers_per_sm: usize,
    pub shared_mem_per_sm: usize,
}
impl OccupancyCalculator {
    pub fn ampere() -> Self;  // also volta(), turing()
    pub fn calculate(&self, reqs: &KernelRequirements) -> OccupancyResult;
    pub fn find_optimal_block_size(/* ... */);
}

pub struct KernelRequirements {
    pub threads_per_block: usize,
    pub registers_per_thread: usize,
    pub shared_mem_per_block: usize,
}

pub struct OccupancyResult {
    pub blocks_per_sm: usize,
    pub warps_per_sm: usize,
    pub max_warps_per_sm: usize,
    pub occupancy: f32,
    pub limiting_factor: LimitingFactor, // module-local enum
}

pub struct MemoryAccessPattern { /* ... */ }
impl MemoryAccessPattern {
    pub fn analyze(addresses: &[usize]) -> Self;
}
```

```rust
use gpu_gemm_optimization::{OccupancyCalculator, KernelRequirements};

let calc = OccupancyCalculator::ampere();
let reqs = KernelRequirements {
    threads_per_block: 256,
    registers_per_thread: 32,
    shared_mem_per_block: 8192,
};
let result = calc.calculate(&reqs);
println!("occupancy: {:.1}%", result.occupancy);
```

## Metrics Module

```rust
pub struct GemmMetrics {
    pub execution_time: std::time::Duration,
    pub dimensions: (usize, usize, usize),
    pub gflops: f64,
    pub bandwidth_gbs: f64,
    pub arithmetic_intensity: f64,
    pub efficiency: f64,
}

impl GemmMetrics {
    pub fn calculate(m: usize, n: usize, k: usize, execution_time: Duration, peak_gflops: f64) -> Self;
    pub fn format(&self) -> String;
}
```

`GemmMetrics` is the only metrics type re-exported at the crate root. The
`metrics` module additionally provides `Benchmark`, `BenchmarkResults`,
`PerformanceModel`, `KernelComparison`, and `MemoryProfile`
(`gpu_gemm_optimization::metrics::*`):

```rust
pub struct Benchmark {
    pub warmup_iters: usize,
    pub measure_iters: usize,
    pub peak_gflops: f64,
}

impl Benchmark {
    pub fn new(warmup_iters: usize, measure_iters: usize, peak_gflops: f64) -> Self;

    // F: Fn(&Matrix, &Matrix, &mut Matrix) -> Result<()>
    pub fn run<F>(&self, a: &Matrix, b: &Matrix, kernel: F) -> Result<GemmMetrics>;
    pub fn run_detailed<F>(&self, a: &Matrix, b: &Matrix, kernel: F) -> Result<BenchmarkResults>;
}
```

```rust
use gpu_gemm_optimization::Matrix;
use gpu_gemm_optimization::metrics::Benchmark;
use gpu_gemm_optimization::naive_gemm;

let a = Matrix::random(128, 128);
let b = Matrix::random(128, 128);

let bench = Benchmark::new(3, 10, 100.0);
let metrics = bench.run(&a, &b, |a, b, c| naive_gemm(a, b, c))?;
println!("{}", metrics.format());
```

## Autotuner Module

The autotuner searches a `ParameterSpace` of `GemmConfig`s using a kernel
closure. Types live in `gpu_gemm_optimization::autotuner::*`; only `Autotuner`
is re-exported at the crate root.

```rust
pub struct AutotuneConfig {
    pub strategy: SearchStrategy,
    pub max_trials: usize,
    pub time_budget: Option<std::time::Duration>,
    pub early_stop: usize,
    pub benchmark_iters: usize,
}
// Default: GridSearch, max_trials=100, time_budget=None, early_stop=20, benchmark_iters=5

pub enum SearchStrategy { GridSearch, Random, SimulatedAnnealing, Genetic }

pub struct ParameterSpace {
    pub block_m: Vec<usize>,
    pub block_n: Vec<usize>,
    pub block_k: Vec<usize>,
    pub thread_m: Vec<usize>,
    pub thread_n: Vec<usize>,
}
impl ParameterSpace {
    pub fn small() -> Self;   // also large(), Default
    pub fn generate_configs(&self) -> Vec<GemmConfig>;
    pub fn num_configs(&self) -> usize;
}

pub struct AutotuneResult {
    pub best_config: GemmConfig,
    pub best_metrics: GemmMetrics,
    pub num_trials: usize,
    pub total_time: std::time::Duration,
    pub history: Vec<(GemmConfig, GemmMetrics)>,
}
impl AutotuneResult {
    pub fn format(&self) -> String;
}

pub struct Autotuner {
    pub config: AutotuneConfig,
    pub param_space: ParameterSpace,
    pub benchmark: Benchmark,
}

impl Autotuner {
    pub fn new(config: AutotuneConfig, param_space: ParameterSpace) -> Self;

    // F: Fn(&Matrix, &Matrix, &mut Matrix, &GemmConfig) -> Result<()>
    pub fn tune<F>(&self, a: &Matrix, b: &Matrix, kernel_fn: F) -> Result<AutotuneResult>;
}
```

The autotuner module also exposes `TuningCache` (an M/N/K → `GemmConfig` cache
with `get_or_tune`) and `HeuristicSelector` (rule-based config selection).

## Complete Examples

### Example 1: Basic GEMM

```rust
use gpu_gemm_optimization::{Matrix, naive_gemm};

fn main() -> gpu_gemm_optimization::Result<()> {
    let (m, n, k) = (256, 256, 256);
    let a = Matrix::random(m, k);
    let b = Matrix::random(k, n);
    let mut c = Matrix::zeros(m, n);

    naive_gemm(&a, &b, &mut c)?;

    println!("GEMM completed, result norm: {}", c.frobenius_norm());
    Ok(())
}
```

### Example 2: Autotuning

```rust
use gpu_gemm_optimization::{Matrix, register_tiled_gemm};
use gpu_gemm_optimization::autotuner::{Autotuner, AutotuneConfig, ParameterSpace};

fn main() -> gpu_gemm_optimization::Result<()> {
    let a = Matrix::random(512, 512);
    let b = Matrix::random(512, 512);

    let tuner = Autotuner::new(AutotuneConfig::default(), ParameterSpace::large());
    let result = tuner.tune(&a, &b, |a, b, c, config| register_tiled_gemm(a, b, c, config))?;

    println!("{}", result.format());
    let best = result.best_config;
    println!(
        "best: block {}x{}x{}, thread {}x{}",
        best.block_m, best.block_n, best.block_k, best.thread_m, best.thread_n
    );
    Ok(())
}
```

### Example 3: Benchmark + Roofline

```rust
use gpu_gemm_optimization::{Matrix, tiled_gemm, RooflineModel};
use gpu_gemm_optimization::metrics::Benchmark;

fn main() -> gpu_gemm_optimization::Result<()> {
    let (m, n, k) = (512, 512, 512);
    let a = Matrix::random(m, k);
    let b = Matrix::random(k, n);

    let bench = Benchmark::new(3, 20, 100.0);
    let metrics = bench.run(&a, &b, |a, b, c| tiled_gemm(a, b, c, 32))?;
    println!("{}", metrics.format());

    let analysis = RooflineModel::default().analyze_gemm(m, n, k, metrics.gflops);
    println!("{:?} at {:.1}% of peak", analysis.bound, analysis.efficiency_percent);
    Ok(())
}
```
