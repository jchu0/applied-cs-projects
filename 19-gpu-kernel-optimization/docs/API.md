# GPU GEMM Optimization - API Documentation

## Table of Contents
1. [Matrix Module](#matrix-module)
2. [GEMM Module](#gemm-module)
3. [Autotuner Module](#autotuner-module)
4. [Metrics Module](#metrics-module)
5. [Complete Examples](#complete-examples)

## Matrix Module

### Core Types

#### `Matrix`
Main matrix data structure for dense matrices.

```rust
pub struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}
```

### Creation Methods

#### `Matrix::new(rows: usize, cols: usize) -> Self`
Creates a new matrix with uninitialized values.

```rust
use gpu_gemm_optimization::matrix::Matrix;

let mat = Matrix::new(100, 200);
assert_eq!(mat.rows(), 100);
assert_eq!(mat.cols(), 200);
```

#### `Matrix::zeros(rows: usize, cols: usize) -> Self`
Creates a matrix filled with zeros.

```rust
let mat = Matrix::zeros(10, 10);
// All elements are 0.0
```

#### `Matrix::ones(rows: usize, cols: usize) -> Self`
Creates a matrix filled with ones.

```rust
let mat = Matrix::ones(5, 5);
// All elements are 1.0
```

#### `Matrix::identity(size: usize) -> Self`
Creates an identity matrix.

```rust
let mat = Matrix::identity(4);
// 4x4 identity matrix
```

#### `Matrix::random(rows: usize, cols: usize) -> Self`
Creates a matrix with random values in [-1, 1].

```rust
let mat = Matrix::random(50, 50);
// Random values between -1.0 and 1.0
```

#### `Matrix::from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Self`
Creates a matrix from existing data.

```rust
let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let mat = Matrix::from_vec(2, 3, data);
// [[1.0, 2.0, 3.0],
//  [4.0, 5.0, 6.0]]
```

### Access Methods

#### `get(&self, row: usize, col: usize) -> f32`
Returns the value at the specified position.

```rust
let mat = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
assert_eq!(mat.get(0, 1), 2.0);
```

#### `set(&mut self, row: usize, col: usize, value: f32)`
Sets the value at the specified position.

```rust
let mut mat = Matrix::zeros(3, 3);
mat.set(1, 1, 5.0);
assert_eq!(mat.get(1, 1), 5.0);
```

### Operations

#### `transpose(&self) -> Matrix`
Returns the transpose of the matrix.

```rust
let mat = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
let transposed = mat.transpose();
assert_eq!(transposed.rows(), 3);
assert_eq!(transposed.cols(), 2);
```

#### `add(&self, other: &Matrix) -> Result<Matrix, MatrixError>`
Element-wise addition.

```rust
let a = Matrix::ones(3, 3);
let b = Matrix::ones(3, 3);
let c = a.add(&b)?;
// All elements are 2.0
```

#### `multiply_scalar(&self, scalar: f32) -> Matrix`
Scalar multiplication.

```rust
let mat = Matrix::ones(2, 2);
let scaled = mat.multiply_scalar(3.0);
// All elements are 3.0
```

#### `submatrix(&self, row_start: usize, col_start: usize, rows: usize, cols: usize) -> Result<Matrix, MatrixError>`
Extracts a submatrix.

```rust
let mat = Matrix::random(10, 10);
let sub = mat.submatrix(2, 3, 4, 5)?;
assert_eq!(sub.rows(), 4);
assert_eq!(sub.cols(), 5);
```

### Utility Methods

#### `frobenius_norm(&self) -> f32`
Computes the Frobenius norm.

```rust
let mat = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
let norm = mat.frobenius_norm();
// sqrt(1 + 4 + 9 + 16) = sqrt(30)
```

#### `approx_equal(&self, other: &Matrix, tolerance: f32) -> bool`
Checks if two matrices are approximately equal.

```rust
let a = Matrix::from_vec(1, 2, vec![1.0, 2.0]);
let b = Matrix::from_vec(1, 2, vec![1.0001, 2.0001]);
assert!(a.approx_equal(&b, 1e-3));
```

## GEMM Module

### Core Types

#### `GemmKernel`
Main GEMM computation kernel.

```rust
pub struct GemmKernel {
    config: GemmConfig,
    // Internal state
}
```

#### `GemmParams`
Parameters for GEMM operation: C = α×A×B + β×C

```rust
pub struct GemmParams {
    pub m: usize,              // Rows of A and C
    pub n: usize,              // Columns of B and C
    pub k: usize,              // Columns of A, rows of B
    pub alpha: f32,            // Scaling factor for A×B
    pub beta: f32,             // Scaling factor for C
    pub trans_a: TransposeMode, // Transpose mode for A
    pub trans_b: TransposeMode, // Transpose mode for B
}
```

#### `GemmConfig`
Configuration for kernel optimization.

```rust
pub struct GemmConfig {
    pub tile_config: TileConfig,
    pub prefetch_distance: usize,
    pub vector_width: usize,
    pub use_fma: bool,
    pub memory_layout: MemoryLayout,
}
```

### Main Operations

#### `GemmKernel::new(config: GemmConfig) -> Self`
Creates a new GEMM kernel with the specified configuration.

```rust
use gpu_gemm_optimization::gemm::{GemmKernel, GemmConfig};

let config = GemmConfig::default();
let kernel = GemmKernel::new(config);
```

#### `execute(&self, a: &Matrix, b: &Matrix, c: &mut Matrix, params: &GemmParams) -> Result<(), GemmError>`
Executes the GEMM operation.

```rust
use gpu_gemm_optimization::gemm::{GemmKernel, GemmParams};
use gpu_gemm_optimization::matrix::{Matrix, TransposeMode};

let a = Matrix::random(100, 200);
let b = Matrix::random(200, 150);
let mut c = Matrix::zeros(100, 150);

let params = GemmParams {
    m: 100,
    n: 150,
    k: 200,
    alpha: 1.0,
    beta: 0.0,
    trans_a: TransposeMode::NoTranspose,
    trans_b: TransposeMode::NoTranspose,
};

let kernel = GemmKernel::new(GemmConfig::default());
kernel.execute(&a, &b, &mut c, &params)?;
```

### Configuration Options

#### `TileConfig`
Tiling parameters for cache optimization.

```rust
pub struct TileConfig {
    pub tile_m: usize,  // M dimension tile size
    pub tile_n: usize,  // N dimension tile size
    pub tile_k: usize,  // K dimension tile size
}

// Example: Custom tiling
let config = GemmConfig {
    tile_config: TileConfig {
        tile_m: 64,
        tile_n: 64,
        tile_k: 16,
    },
    ..GemmConfig::default()
};
```

#### `MemoryLayout`
Memory access pattern optimization.

```rust
pub enum MemoryLayout {
    RowMajor,      // Row-major storage
    ColumnMajor,   // Column-major storage
    Packed,        // Packed format for kernels
}
```

## Autotuner Module

### Core Types

#### `Autotuner`
Automatic performance tuning system.

```rust
pub struct Autotuner {
    config: AutotuneConfig,
    // Internal state
}
```

#### `AutotuneConfig`
Configuration for autotuning process.

```rust
pub struct AutotuneConfig {
    pub search_space: SearchSpace,
    pub max_iterations: usize,
    pub convergence_threshold: f32,
    pub timeout_seconds: u64,
    pub parallel_evaluations: usize,
    pub optimization_objective: OptimizationObjective,
    pub search_strategy: SearchStrategy,
}
```

### Main Operations

#### `Autotuner::new(config: AutotuneConfig) -> Result<Self, AutotunerError>`
Creates a new autotuner.

```rust
use gpu_gemm_optimization::autotuner::{
    Autotuner, AutotuneConfig, SearchSpace, OptimizationObjective, SearchStrategy
};

let search_space = SearchSpace {
    tile_m_options: vec![32, 64, 128],
    tile_n_options: vec![32, 64, 128],
    tile_k_options: vec![8, 16, 32],
    prefetch_options: vec![1, 2, 4],
    vector_width_options: vec![4, 8, 16],
    memory_layouts: vec![MemoryLayout::RowMajor, MemoryLayout::ColumnMajor],
};

let config = AutotuneConfig {
    search_space,
    max_iterations: 100,
    convergence_threshold: 0.01,
    timeout_seconds: 300,
    parallel_evaluations: 4,
    optimization_objective: OptimizationObjective::Throughput,
    search_strategy: SearchStrategy::BayesianOptimization,
};

let mut tuner = Autotuner::new(config)?;
```

#### `tune(&mut self, a: &Matrix, b: &Matrix, c: &Matrix) -> Result<GemmConfig, AutotunerError>`
Finds the optimal configuration for given matrices.

```rust
let a = Matrix::random(512, 512);
let b = Matrix::random(512, 512);
let c = Matrix::zeros(512, 512);

let best_config = tuner.tune(&a, &b, &c)?;
println!("Best tile size: {}x{}x{}",
         best_config.tile_config.tile_m,
         best_config.tile_config.tile_n,
         best_config.tile_config.tile_k);
```

### Search Strategies

```rust
pub enum SearchStrategy {
    GridSearch,           // Exhaustive search
    RandomSearch,         // Random sampling
    BayesianOptimization, // Gaussian process based
    GeneticAlgorithm,     // Evolutionary optimization
}
```

### Optimization Objectives

```rust
pub enum OptimizationObjective {
    Throughput,        // Maximize GFLOPS
    Latency,          // Minimize execution time
    EnergyEfficiency, // Maximize GFLOPS/Watt
    MemoryBandwidth,  // Maximize bandwidth utilization
}
```

## Metrics Module

### Core Types

#### `PerformanceMetrics`
Complete performance measurement structure.

```rust
pub struct PerformanceMetrics {
    pub throughput_gflops: f32,
    pub latency_ms: f32,
    pub memory_bandwidth_gb: f32,
    pub cache_hit_rate: f32,
    pub register_pressure: u32,
    pub occupancy: f32,
    pub energy_joules: Option<f32>,
    pub power_watts: Option<f32>,
}
```

#### `MetricsCollector`
Runtime metrics collection system.

```rust
pub struct MetricsCollector {
    // Internal storage for metrics
}
```

### Collection Operations

#### `MetricsCollector::new() -> Self`
Creates a new metrics collector.

```rust
use gpu_gemm_optimization::metrics::MetricsCollector;

let mut collector = MetricsCollector::new();
```

#### `record(&mut self, metric_type: MetricType, value: f64)`
Records a metric value.

```rust
use gpu_gemm_optimization::metrics::MetricType;

collector.record(MetricType::Throughput, 150.5);
collector.record(MetricType::Latency, 12.3);
```

#### `aggregate(&self, metric_type: MetricType, method: AggregationMethod) -> f64`
Aggregates collected metrics.

```rust
use gpu_gemm_optimization::metrics::AggregationMethod;

let avg_throughput = collector.aggregate(
    MetricType::Throughput,
    AggregationMethod::Average
);

let p95_latency = collector.aggregate(
    MetricType::Latency,
    AggregationMethod::Percentile95
);
```

### Analysis Operations

#### `PerformanceAnalyzer::compare(&self, a: &PerformanceMetrics, b: &PerformanceMetrics) -> ComparisonResult`
Compares two performance measurements.

```rust
use gpu_gemm_optimization::metrics::PerformanceAnalyzer;

let analyzer = PerformanceAnalyzer::new();
let comparison = analyzer.compare(&metrics1, &metrics2);

println!("Throughput improvement: {:.2}%",
         comparison.throughput_improvement * 100.0);
```

#### `identify_bottlenecks(&self, metrics: &PerformanceMetrics) -> Vec<String>`
Identifies performance bottlenecks.

```rust
let bottlenecks = analyzer.identify_bottlenecks(&metrics);
for bottleneck in bottlenecks {
    println!("Bottleneck detected: {}", bottleneck);
}
```

### Benchmarking

#### `Benchmark::run<F>(&mut self, f: F) -> Result<BenchmarkResult, BenchmarkError>`
Runs a benchmark with warmup and statistics.

```rust
use gpu_gemm_optimization::metrics::{Benchmark, BenchmarkConfig};
use std::time::Duration;

let config = BenchmarkConfig {
    warmup_iterations: 10,
    measurement_iterations: 100,
    timeout_per_iteration: Duration::from_secs(10),
};

let mut benchmark = Benchmark::new(config);

let result = benchmark.run(|| {
    // Code to benchmark
    kernel.execute(&a, &b, &mut c, &params).unwrap();
})?;

println!("Mean time: {:?}", result.mean_time);
println!("Std deviation: {:?}", result.std_deviation);
```

## Complete Examples

### Example 1: Basic GEMM Operation

```rust
use gpu_gemm_optimization::{
    matrix::{Matrix, TransposeMode},
    gemm::{GemmKernel, GemmParams, GemmConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create matrices
    let m = 256;
    let n = 256;
    let k = 256;

    let a = Matrix::random(m, k);
    let b = Matrix::random(k, n);
    let mut c = Matrix::zeros(m, n);

    // Setup GEMM parameters
    let params = GemmParams {
        m,
        n,
        k,
        alpha: 1.0,
        beta: 0.0,
        trans_a: TransposeMode::NoTranspose,
        trans_b: TransposeMode::NoTranspose,
    };

    // Create and execute kernel
    let kernel = GemmKernel::new(GemmConfig::default());
    kernel.execute(&a, &b, &mut c, &params)?;

    println!("GEMM completed successfully");
    println!("Result norm: {}", c.frobenius_norm());

    Ok(())
}
```

### Example 2: Autotuning for Performance

```rust
use gpu_gemm_optimization::{
    matrix::Matrix,
    gemm::{GemmKernel, GemmParams, MemoryLayout},
    autotuner::{
        Autotuner, AutotuneConfig, SearchSpace,
        OptimizationObjective, SearchStrategy
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Problem size
    let size = 1024;

    // Create test matrices
    let a = Matrix::random(size, size);
    let b = Matrix::random(size, size);
    let c = Matrix::zeros(size, size);

    // Define search space
    let search_space = SearchSpace {
        tile_m_options: vec![32, 64, 128, 256],
        tile_n_options: vec![32, 64, 128, 256],
        tile_k_options: vec![8, 16, 32, 64],
        prefetch_options: vec![1, 2, 4, 8],
        vector_width_options: vec![4, 8, 16],
        memory_layouts: vec![
            MemoryLayout::RowMajor,
            MemoryLayout::ColumnMajor,
            MemoryLayout::Packed,
        ],
    };

    // Configure autotuner
    let config = AutotuneConfig {
        search_space,
        max_iterations: 50,
        convergence_threshold: 0.01,
        timeout_seconds: 120,
        parallel_evaluations: 4,
        optimization_objective: OptimizationObjective::Throughput,
        search_strategy: SearchStrategy::BayesianOptimization,
    };

    // Run autotuning
    let mut tuner = Autotuner::new(config)?;
    let best_config = tuner.tune(&a, &b, &c)?;

    // Use optimized configuration
    let kernel = GemmKernel::new(best_config);
    let mut c_result = Matrix::zeros(size, size);

    let params = GemmParams {
        m: size,
        n: size,
        k: size,
        alpha: 1.0,
        beta: 0.0,
        trans_a: TransposeMode::NoTranspose,
        trans_b: TransposeMode::NoTranspose,
    };

    kernel.execute(&a, &b, &mut c_result, &params)?;

    println!("Autotuning completed");
    println!("Best configuration found:");
    println!("  Tile: {}x{}x{}",
             best_config.tile_config.tile_m,
             best_config.tile_config.tile_n,
             best_config.tile_config.tile_k);

    Ok(())
}
```

### Example 3: Performance Analysis

```rust
use gpu_gemm_optimization::{
    matrix::Matrix,
    gemm::{GemmKernel, GemmParams, GemmConfig},
    metrics::{
        MetricsCollector, PerformanceAnalyzer,
        MetricType, AggregationMethod, Benchmark, BenchmarkConfig
    },
};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let size = 512;
    let a = Matrix::random(size, size);
    let b = Matrix::random(size, size);

    // Create metrics collector
    let mut collector = MetricsCollector::new();

    // Benchmark different configurations
    let configs = vec![
        GemmConfig::default(),
        GemmConfig {
            tile_config: TileConfig { tile_m: 64, tile_n: 64, tile_k: 16 },
            ..GemmConfig::default()
        },
        GemmConfig {
            tile_config: TileConfig { tile_m: 128, tile_n: 128, tile_k: 32 },
            ..GemmConfig::default()
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        let kernel = GemmKernel::new(config.clone());
        let mut c = Matrix::zeros(size, size);

        let params = GemmParams {
            m: size,
            n: size,
            k: size,
            alpha: 1.0,
            beta: 0.0,
            trans_a: TransposeMode::NoTranspose,
            trans_b: TransposeMode::NoTranspose,
        };

        // Run benchmark
        let bench_config = BenchmarkConfig {
            warmup_iterations: 5,
            measurement_iterations: 20,
            timeout_per_iteration: Duration::from_secs(5),
        };

        let mut benchmark = Benchmark::new(bench_config);
        let result = benchmark.run(|| {
            kernel.execute(&a, &b, &mut c, &params).unwrap();
        })?;

        // Calculate and record metrics
        let gflops = (2.0 * size as f64 * size as f64 * size as f64) /
                     (result.mean_time.as_secs_f64() * 1e9);

        collector.record(MetricType::Throughput, gflops);
        collector.record(MetricType::Latency, result.mean_time.as_millis() as f64);

        println!("Config {}: {:.2} GFLOPS, {:.2} ms",
                 i, gflops, result.mean_time.as_millis());
    }

    // Analyze results
    let avg_throughput = collector.aggregate(
        MetricType::Throughput,
        AggregationMethod::Average
    );
    let max_throughput = collector.aggregate(
        MetricType::Throughput,
        AggregationMethod::Max
    );

    println!("\nPerformance Summary:");
    println!("  Average: {:.2} GFLOPS", avg_throughput);
    println!("  Maximum: {:.2} GFLOPS", max_throughput);

    Ok(())
}
```

## Error Handling

All operations that can fail return `Result<T, E>` types with specific error enums:

- `MatrixError`: Matrix operation errors (dimension mismatch, out of bounds)
- `GemmError`: GEMM execution errors (invalid parameters, numerical issues)
- `AutotunerError`: Autotuning failures (timeout, convergence failure)
- `BenchmarkError`: Benchmarking errors (timeout, measurement failure)

Example error handling:

```rust
match kernel.execute(&a, &b, &mut c, &params) {
    Ok(()) => println!("Success"),
    Err(GemmError::DimensionMismatch { expected, actual }) => {
        eprintln!("Dimension mismatch: expected {}, got {}", expected, actual);
    },
    Err(e) => eprintln!("GEMM failed: {}", e),
}
```