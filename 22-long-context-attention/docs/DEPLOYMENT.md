# Deployment Guide

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Building from Source](#building-from-source)
- [Configuration](#configuration)
- [Deployment Scenarios](#deployment-scenarios)
- [Performance Tuning](#performance-tuning)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)

## System Requirements

### Minimum Requirements

- **CPU**: x86-64 or ARM64 processor with AVX2 support
- **Memory**: 8GB RAM (16GB recommended)
- **Storage**: 2GB free disk space
- **OS**: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+
- **Rust**: 1.70.0 or higher

### Recommended Requirements

- **CPU**: Modern multi-core processor (8+ cores)
- **Memory**: 32GB+ RAM for large models
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for acceleration)
- **Storage**: SSD with 10GB+ free space

### GPU Support (Optional)

- **NVIDIA**: CUDA 11.8+ and cuDNN 8.6+
- **AMD**: ROCm 5.4+
- **Apple**: macOS 12+ with Metal support
- **Intel**: OneAPI 2023.1+

## Installation

### Using Cargo

```bash
# Add to Cargo.toml
[dependencies]
long-context-attention = "0.1.0"

# With specific features
[dependencies]
long-context-attention = { version = "0.1.0", features = ["cuda", "triton"] }
```

### Available Features

```toml
[features]
default = ["cpu"]
cpu = []              # CPU-only implementation
cuda = []            # NVIDIA GPU support
rocm = []            # AMD GPU support
metal = []           # Apple Silicon support
triton = []          # Triton kernel support
distributed = []     # Multi-GPU/node support
all = ["cpu", "cuda", "rocm", "metal", "triton", "distributed"]
```

### Pre-built Binaries

Download pre-built binaries from the releases page:

```bash
# Linux
wget https://github.com/yourorg/long-context-attention/releases/download/v0.1.0/long-context-attention-linux-x64.tar.gz
tar -xzf long-context-attention-linux-x64.tar.gz

# macOS
wget https://github.com/yourorg/long-context-attention/releases/download/v0.1.0/long-context-attention-macos-arm64.tar.gz
tar -xzf long-context-attention-macos-arm64.tar.gz
```

## Building from Source

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install build dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config

# Install build dependencies (macOS)
brew install cmake pkg-config

# Install build dependencies (Windows)
# Install Visual Studio Build Tools
```

### Build Steps

```bash
# Clone repository
git clone https://github.com/yourorg/long-context-attention.git
cd long-context-attention

# Build with default features
cargo build --release

# Build with CUDA support
cargo build --release --features cuda

# Build with all optimizations
cargo build --release --features all

# Run tests
cargo test --release

# Build documentation
cargo doc --open
```

### Cross-Compilation

```bash
# Install cross-compilation tools
cargo install cross

# Build for different targets
cross build --release --target aarch64-unknown-linux-gnu
cross build --release --target x86_64-pc-windows-gnu
```

## Configuration

### Environment Variables

```bash
# Memory management
export LCA_MAX_MEMORY=16G           # Maximum memory usage
export LCA_CACHE_DIR=/tmp/lca      # Cache directory
export LCA_PREFETCH=true           # Enable prefetching

# Performance tuning
export LCA_NUM_THREADS=8           # Number of CPU threads
export LCA_BATCH_SIZE=4           # Default batch size
export LCA_BLOCK_SIZE=64          # Default block size

# Hardware selection
export LCA_DEVICE=cuda:0          # Device selection (cpu, cuda:0, rocm:0)
export LCA_MIXED_PRECISION=true   # Enable mixed precision

# Logging
export RUST_LOG=info              # Log level (trace, debug, info, warn, error)
export LCA_LOG_FILE=/var/log/lca.log  # Log file path
```

### Configuration File

Create a `config.toml` file:

```toml
[attention]
default_type = "auto"
max_seq_len = 8192
enable_caching = true

[memory]
max_memory_gb = 16
cache_size_gb = 4
compression = "quantize8bit"
eviction_policy = "lru"

[performance]
num_threads = 8
batch_size = 4
prefetch = true
profile = true

[hardware]
device = "cuda:0"
mixed_precision = true
kernel_fusion = true

[monitoring]
enable_metrics = true
metrics_port = 9090
health_check_port = 8080
```

### Runtime Configuration

```rust
use long_context_attention::config::RuntimeConfig;

let config = RuntimeConfig::builder()
    .max_memory(16 * 1024 * 1024 * 1024)  // 16GB
    .device("cuda:0")
    .num_threads(8)
    .enable_profiling(true)
    .build();

// Apply configuration
attention.set_runtime_config(config);
```

## Deployment Scenarios

### 1. Single-Node CPU Deployment

```bash
# Dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --features cpu

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y libgomp1
COPY --from=builder /app/target/release/long-context-attention /usr/local/bin/
ENV LCA_DEVICE=cpu
ENV LCA_NUM_THREADS=8
CMD ["long-context-attention"]
```

### 2. Single-Node GPU Deployment

```bash
# Dockerfile for CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y wget
COPY --from=builder /app/target/release/long-context-attention /usr/local/bin/
ENV LCA_DEVICE=cuda:0
ENV LCA_MIXED_PRECISION=true
CMD ["long-context-attention"]
```

### 3. Multi-GPU Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  lca-gpu0:
    image: long-context-attention:cuda
    environment:
      - LCA_DEVICE=cuda:0
      - LCA_DISTRIBUTED=true
      - LCA_RANK=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']

  lca-gpu1:
    image: long-context-attention:cuda
    environment:
      - LCA_DEVICE=cuda:1
      - LCA_DISTRIBUTED=true
      - LCA_RANK=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
```

### 4. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: long-context-attention
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lca
  template:
    metadata:
      labels:
        app: lca
    spec:
      containers:
      - name: lca
        image: long-context-attention:latest
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: LCA_DEVICE
          value: "cuda:0"
        - name: LCA_MAX_MEMORY
          value: "30G"
---
apiVersion: v1
kind: Service
metadata:
  name: lca-service
spec:
  selector:
    app: lca
  ports:
  - port: 8080
    targetPort: 8080
```

### 5. Serverless Deployment (AWS Lambda)

```python
# handler.py
import json
import subprocess

def lambda_handler(event, context):
    input_data = event['body']

    # Run attention computation
    result = subprocess.run(
        ['./long-context-attention', '--input', input_data],
        capture_output=True,
        text=True
    )

    return {
        'statusCode': 200,
        'body': json.dumps(result.stdout)
    }
```

## Performance Tuning

### Memory Optimization

```bash
# Reduce memory usage for long sequences
export LCA_COMPRESSION=quantize8bit
export LCA_EVICTION_POLICY=sliding_window
export LCA_CACHE_SIZE=2G

# Use memory-efficient attention
export LCA_DEFAULT_ATTENTION=flash
export LCA_FLASH_BLOCK_SIZE=64
export LCA_BACKWARD_MODE=recompute
```

### Throughput Optimization

```bash
# Maximize throughput
export LCA_BATCH_SIZE=8
export LCA_PREFETCH=true
export LCA_KERNEL_FUSION=true
export LCA_MIXED_PRECISION=true

# Use faster attention for short sequences
export LCA_AUTO_SELECT=true
export LCA_OPTIMIZE_FOR=throughput
```

### Latency Optimization

```bash
# Minimize latency
export LCA_BATCH_SIZE=1
export LCA_WARM_START=true
export LCA_PRECOMPILE_KERNELS=true

# Use optimized kernels
export LCA_USE_TRITON=true
export LCA_KERNEL_CACHE=/tmp/kernel_cache
```

### Profile-Guided Optimization

```rust
use long_context_attention::profiler::Profiler;

let mut profiler = Profiler::new();
profiler.start();

// Run workload
for _ in 0..100 {
    attention.forward(&q, &k, &v);
}

let report = profiler.stop();
report.save("profile.json");

// Apply optimizations based on profile
attention.apply_profile_optimizations("profile.json");
```

## Monitoring

### Metrics Collection

```rust
use long_context_attention::metrics::MetricsCollector;

let collector = MetricsCollector::new()
    .with_prometheus_exporter(9090)
    .with_statsd("localhost:8125");

attention.set_metrics_collector(collector);
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'lca'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Long Context Attention Metrics",
    "panels": [
      {
        "title": "Throughput",
        "targets": [
          {
            "expr": "rate(lca_tokens_processed_total[5m])"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "lca_memory_usage_bytes"
          }
        ]
      },
      {
        "title": "Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.99, lca_latency_seconds_bucket)"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

```rust
use long_context_attention::health::HealthChecker;

let health_checker = HealthChecker::new()
    .with_http_endpoint(8080)
    .with_checks(vec![
        "memory_available",
        "gpu_available",
        "cache_hit_rate",
    ]);

attention.set_health_checker(health_checker);
```

## Troubleshooting

### Common Issues

#### Out of Memory

```bash
# Symptoms: OOM errors, process killed
# Solutions:
export LCA_MAX_MEMORY=8G  # Reduce memory limit
export LCA_COMPRESSION=quantize4bit  # More aggressive compression
export LCA_SPARSITY_RATIO=0.95  # Increase sparsity

# Use sliding window for very long sequences
export LCA_DEFAULT_ATTENTION=sliding_window
export LCA_WINDOW_SIZE=512
```

#### Low Performance

```bash
# Symptoms: Slow inference, high latency
# Solutions:
export LCA_USE_TRITON=true  # Enable optimized kernels
export LCA_KERNEL_FUSION=true  # Fuse operations
export RUST_LOG=trace  # Check for bottlenecks

# Profile to identify issues
export LCA_PROFILE=true
export LCA_PROFILE_OUTPUT=/tmp/profile.json
```

#### CUDA Errors

```bash
# Symptoms: CUDA out of memory, kernel launch failures
# Solutions:
nvidia-smi  # Check GPU status
export CUDA_VISIBLE_DEVICES=0  # Select specific GPU
export LCA_CUDA_MEMORY_FRACTION=0.8  # Limit GPU memory usage

# Clear GPU cache
export LCA_CLEAR_CACHE_INTERVAL=100  # Clear cache every N iterations
```

### Debug Mode

```bash
# Enable comprehensive debugging
export RUST_LOG=debug
export LCA_DEBUG=true
export LCA_TRACE_KERNELS=true
export LCA_SAVE_INTERMEDIATES=/tmp/debug

# Run with debugging
./long-context-attention --debug --verbose
```

### Logging

```rust
use log::{info, warn, error};
use env_logger;

fn main() {
    env_logger::init();

    info!("Starting Long Context Attention");

    match attention.forward(&q, &k, &v) {
        Ok(output) => info!("Forward pass successful"),
        Err(e) => error!("Forward pass failed: {}", e),
    }
}
```

## Security Considerations

### Input Validation

```rust
use long_context_attention::security::InputValidator;

let validator = InputValidator::new()
    .max_seq_len(8192)
    .max_batch_size(16)
    .validate_shapes(true);

// Validate inputs before processing
validator.validate(&q, &k, &v)?;
```

### Resource Limits

```yaml
# systemd service file
[Service]
Type=simple
ExecStart=/usr/local/bin/long-context-attention
MemoryMax=32G
CPUQuota=800%
TasksMax=100
PrivateTmp=yes
NoNewPrivileges=yes
```

### Network Security

```nginx
# Nginx reverse proxy configuration
server {
    listen 443 ssl http2;
    server_name api.example.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location /attention {
        proxy_pass http://localhost:8080;
        proxy_set_header X-Real-IP $remote_addr;

        # Rate limiting
        limit_req zone=api burst=10;

        # Request size limit
        client_max_body_size 100M;
    }
}
```

### Access Control

```rust
use long_context_attention::auth::TokenValidator;

let validator = TokenValidator::new("secret_key");

// Validate API token
if !validator.validate(token) {
    return Err("Unauthorized");
}
```

## Production Checklist

- [ ] System requirements verified
- [ ] Dependencies installed and versions locked
- [ ] Configuration files created and validated
- [ ] Environment variables set correctly
- [ ] Resource limits configured
- [ ] Monitoring and alerting setup
- [ ] Health checks configured
- [ ] Logging configured with appropriate levels
- [ ] Backup and recovery procedures in place
- [ ] Security measures implemented
- [ ] Performance benchmarks run
- [ ] Load testing completed
- [ ] Documentation updated
- [ ] Rollback procedure defined
- [ ] Support contacts established