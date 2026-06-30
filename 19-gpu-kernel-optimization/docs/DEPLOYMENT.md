# GPU GEMM Optimization - Deployment Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Building from Source](#building-from-source)
4. [Configuration](#configuration)
5. [Deployment Scenarios](#deployment-scenarios)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: x86_64 or ARM64 processor with SIMD support
- **RAM**: 8 GB minimum, 16 GB recommended
- **Storage**: 500 MB for binaries and libraries
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+), macOS 11+, Windows 10+

#### Recommended Requirements
- **CPU**: Multi-core processor (8+ cores) with AVX2/AVX-512
- **RAM**: 32 GB or more for large matrix operations
- **Storage**: SSD with 10+ GB free space for datasets
- **GPU**: NVIDIA GPU with CUDA 11.0+ (for future GPU backend)

### Software Dependencies

#### Required
- Rust 1.70.0 or later
- Cargo (comes with Rust)
- C compiler (gcc/clang for build dependencies)

#### Optional
- BLAS/LAPACK libraries for validation
- Python 3.8+ for benchmarking scripts
- Docker for containerized deployment

### Install Rust

```bash
# Install Rust via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Add to PATH
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

## Installation

### Option 1: From Crates.io (When Published)

```bash
cargo install gpu-gemm-optimization
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/gpu-gemm-optimization.git
cd gpu-gemm-optimization

# Build and install
cargo install --path .
```

### Option 3: As a Library Dependency

Add to your `Cargo.toml`:

```toml
[dependencies]
gpu-gemm-optimization = { git = "https://github.com/yourusername/gpu-gemm-optimization.git" }
# Or when published:
# gpu-gemm-optimization = "0.1.0"
```

## Building from Source

### Development Build

```bash
# Clone repository
git clone https://github.com/yourusername/gpu-gemm-optimization.git
cd gpu-gemm-optimization

# Build in debug mode
cargo build

# Run tests
cargo test

# Run benchmarks
cargo bench
```

### Release Build

```bash
# Build with optimizations
cargo build --release

# Run release tests
cargo test --release

# Install locally
cargo install --path . --force
```

### Build Features

```toml
# Cargo.toml features
[features]
default = ["parallel", "metrics"]
parallel = ["rayon"]  # Enable parallel execution
metrics = []          # Enable performance metrics
cuda = []            # Future: CUDA backend
opencl = []          # Future: OpenCL backend
```

Build with specific features:

```bash
# Build with all features
cargo build --release --all-features

# Build without parallel support
cargo build --release --no-default-features

# Build with CUDA support (future)
cargo build --release --features cuda
```

## Configuration

### Environment Variables

```bash
# Thread pool configuration
export RAYON_NUM_THREADS=8

# Memory limits
export GEMM_MAX_MEMORY_GB=16

# Cache configuration
export GEMM_L1_CACHE_KB=32
export GEMM_L2_CACHE_MB=1

# Logging
export RUST_LOG=info
export RUST_BACKTRACE=1

# Performance settings
export GEMM_DEFAULT_TILE_M=64
export GEMM_DEFAULT_TILE_N=64
export GEMM_DEFAULT_TILE_K=16
```

### Configuration Files

Create `gemm_config.toml`:

```toml
[default]
tile_m = 64
tile_n = 64
tile_k = 16
prefetch_distance = 2
vector_width = 8
use_fma = true
memory_layout = "RowMajor"

[autotuner]
max_iterations = 100
convergence_threshold = 0.01
timeout_seconds = 300
parallel_evaluations = 4
search_strategy = "BayesianOptimization"

[performance]
enable_metrics = true
benchmark_warmup = 10
benchmark_iterations = 100

[system]
thread_pool_size = 0  # 0 = auto-detect
memory_limit_gb = 0   # 0 = no limit
```

Load configuration in code:

```rust
use gpu_gemm_optimization::config::load_config;

let config = load_config("gemm_config.toml")?;
```

## Deployment Scenarios

### 1. Standalone Application

```bash
#!/bin/bash
# deploy.sh

# Set environment
export RUST_LOG=info
export RAYON_NUM_THREADS=$(nproc)

# Run the application
./gpu-gemm-optimization \
    --matrix-size 1024 \
    --tile-size 64 \
    --iterations 100 \
    --output results.json
```

### 2. Library Integration

```rust
// main.rs
use gpu_gemm_optimization::{
    matrix::Matrix,
    gemm::{GemmKernel, GemmParams, GemmConfig},
    autotuner::{Autotuner, AutotuneConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize matrices
    let a = Matrix::random(1024, 1024);
    let b = Matrix::random(1024, 1024);
    let mut c = Matrix::zeros(1024, 1024);

    // Auto-tune for best performance
    let mut tuner = Autotuner::new(AutotuneConfig::default())?;
    let config = tuner.tune(&a, &b, &c)?;

    // Execute with optimal configuration
    let kernel = GemmKernel::new(config);
    kernel.execute(&a, &b, &mut c, &GemmParams::default())?;

    Ok(())
}
```

### 3. Docker Deployment

```dockerfile
# Dockerfile
FROM rust:1.70 as builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/gpu-gemm-optimization /usr/local/bin/

ENV RUST_LOG=info
ENV RAYON_NUM_THREADS=8

ENTRYPOINT ["gpu-gemm-optimization"]
```

Build and run:

```bash
# Build Docker image
docker build -t gpu-gemm-optimization .

# Run container
docker run --rm \
    -v $(pwd)/data:/data \
    -e RAYON_NUM_THREADS=$(nproc) \
    gpu-gemm-optimization \
    --input /data/matrices.bin \
    --output /data/results.json
```

### 4. Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gemm-optimization
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gemm-optimization
  template:
    metadata:
      labels:
        app: gemm-optimization
    spec:
      containers:
      - name: gemm
        image: gpu-gemm-optimization:latest
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "16Gi"
            cpu: "8"
        env:
        - name: RAYON_NUM_THREADS
          value: "8"
        - name: RUST_LOG
          value: "info"
        volumeMounts:
        - name: config
          mountPath: /etc/gemm
      volumes:
      - name: config
        configMap:
          name: gemm-config
```

### 5. Cloud Deployment (AWS)

```bash
# Deploy on AWS EC2
#!/bin/bash

# Launch instance
aws ec2 run-instances \
    --image-id ami-0c55b159cbfafe1f0 \
    --instance-type c5.4xlarge \
    --key-name my-key \
    --security-group-ids sg-12345678 \
    --user-data file://setup.sh

# setup.sh
#!/bin/bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
git clone https://github.com/yourusername/gpu-gemm-optimization.git
cd gpu-gemm-optimization
cargo build --release
./target/release/gpu-gemm-optimization --autotune
```

## Performance Tuning

### 1. CPU Optimization

```bash
# Set CPU affinity
taskset -c 0-7 ./gpu-gemm-optimization

# Set CPU governor
sudo cpupower frequency-set -g performance

# Disable CPU frequency scaling
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

### 2. Memory Optimization

```bash
# Increase memory limits
ulimit -v unlimited
ulimit -m unlimited

# Configure huge pages
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
export MALLOC_ARENA_MAX=2

# NUMA optimization
numactl --cpunodebind=0 --membind=0 ./gpu-gemm-optimization
```

### 3. Autotuning Strategy

```rust
// Custom autotuning for specific hardware
let config = AutotuneConfig {
    search_space: SearchSpace {
        tile_m_options: vec![32, 64, 128, 256],
        tile_n_options: vec![32, 64, 128, 256],
        tile_k_options: vec![8, 16, 32],
        prefetch_options: vec![1, 2, 4, 8],
        vector_width_options: vec![8, 16, 32],
        memory_layouts: vec![MemoryLayout::RowMajor, MemoryLayout::Packed],
    },
    max_iterations: 200,
    convergence_threshold: 0.005,
    timeout_seconds: 600,
    parallel_evaluations: num_cpus::get(),
    optimization_objective: OptimizationObjective::Throughput,
    search_strategy: SearchStrategy::BayesianOptimization,
};
```

### 4. Profile-Guided Optimization

```bash
# Collect profile data
cargo build --release
CARGO_PROFILE_RELEASE_LTO=true cargo build --release

# Run with profiling
perf record -g ./target/release/gpu-gemm-optimization
perf report

# Valgrind for cache analysis
valgrind --tool=cachegrind ./target/release/gpu-gemm-optimization
```

## Monitoring

### 1. Performance Metrics

```rust
// Enable metrics collection
use gpu_gemm_optimization::metrics::{MetricsCollector, MetricType};

let mut collector = MetricsCollector::new();
collector.start_monitoring();

// Perform operations...

let metrics = collector.get_summary();
println!("Average throughput: {} GFLOPS", metrics.avg_throughput);
println!("P95 latency: {} ms", metrics.p95_latency);
```

### 2. Prometheus Integration

```rust
// prometheus_exporter.rs
use prometheus::{Encoder, TextEncoder, Counter, Gauge, Histogram};

lazy_static! {
    static ref GEMM_OPS: Counter = register_counter!(
        "gemm_operations_total",
        "Total number of GEMM operations"
    ).unwrap();

    static ref GEMM_THROUGHPUT: Gauge = register_gauge!(
        "gemm_throughput_gflops",
        "Current GEMM throughput in GFLOPS"
    ).unwrap();

    static ref GEMM_LATENCY: Histogram = register_histogram!(
        "gemm_latency_ms",
        "GEMM operation latency in milliseconds"
    ).unwrap();
}
```

### 3. Logging

```rust
// Configure logging
use env_logger;
use log::{info, debug, error};

fn main() {
    env_logger::init();

    info!("Starting GEMM optimization");
    debug!("Configuration: {:?}", config);

    match kernel.execute(&a, &b, &mut c, &params) {
        Ok(_) => info!("GEMM completed successfully"),
        Err(e) => error!("GEMM failed: {}", e),
    }
}
```

### 4. Health Checks

```rust
// health_check.rs
use actix_web::{web, App, HttpResponse, HttpServer};

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime": get_uptime(),
    }))
}

async fn metrics() -> HttpResponse {
    let metrics = collect_metrics();
    HttpResponse::Ok().json(metrics)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/health", web::get().to(health))
            .route("/metrics", web::get().to(metrics))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory

**Symptom**: Process killed or allocation failure

**Solution**:
```bash
# Increase system memory limits
sudo sysctl -w vm.overcommit_memory=1
sudo sysctl -w vm.max_map_count=262144

# Reduce matrix sizes or tile sizes
export GEMM_MAX_TILE_SIZE=32
```

#### 2. Poor Performance

**Symptom**: Low GFLOPS, high latency

**Solution**:
```bash
# Check CPU frequency
cat /proc/cpuinfo | grep MHz

# Disable power saving
sudo systemctl stop thermald
sudo systemctl disable thermald

# Run autotuning
./gpu-gemm-optimization --autotune --save-config optimal.toml
```

#### 3. Numerical Instability

**Symptom**: NaN or Inf in results

**Solution**:
```rust
// Enable numerical checks
let config = GemmConfig {
    enable_nan_check: true,
    use_stable_accumulation: true,
    ..Default::default()
};
```

#### 4. Thread Contention

**Symptom**: Poor scaling with threads

**Solution**:
```bash
# Reduce thread count
export RAYON_NUM_THREADS=4

# Set thread affinity
export OMP_PROC_BIND=true
export OMP_PLACES=cores
```

### Debug Mode

```bash
# Enable debug assertions
cargo build
RUST_BACKTRACE=full ./target/debug/gpu-gemm-optimization

# Enable sanitizers
RUSTFLAGS="-Z sanitizer=address" cargo build
RUSTFLAGS="-Z sanitizer=thread" cargo build
```

### Performance Analysis

```bash
# CPU profiling
perf stat -d ./gpu-gemm-optimization

# Cache analysis
perf stat -e cache-references,cache-misses ./gpu-gemm-optimization

# Memory bandwidth
perf stat -e uncore_imc/data_reads/,uncore_imc/data_writes/ ./gpu-gemm-optimization
```

### Validation

```bash
# Run validation suite
cargo test --release -- --test-threads=1

# Compare with reference BLAS
./scripts/validate_against_blas.sh

# Numerical accuracy tests
cargo test --features strict_numerical_tests
```

## Support

### Documentation
- API Documentation: `cargo doc --open`
- Examples: See `examples/` directory
- Benchmarks: Run `cargo bench`

### Reporting Issues
1. Check existing issues on GitHub
2. Provide system information (OS, CPU, RAM)
3. Include minimal reproducible example
4. Attach relevant logs and configuration

### Community
- GitHub Discussions for questions
- Issue tracker for bugs
- Pull requests welcome

## License

This project is licensed under the MIT License. See LICENSE file for details.