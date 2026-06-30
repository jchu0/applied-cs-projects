# Graph Neural Network Runtime - Deployment Guide

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Production Deployment](#production-deployment)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4-core x86_64 processor
- **RAM**: 8GB DDR4
- **Storage**: 20GB SSD
- **GPU**: NVIDIA GPU with 4GB VRAM (optional)

#### Recommended Requirements
- **CPU**: 8-core x86_64 processor
- **RAM**: 32GB DDR4
- **Storage**: 100GB NVMe SSD
- **GPU**: NVIDIA GPU with 16GB+ VRAM (V100, A100, or RTX 3090+)

#### Memory Requirements by Graph Size

| Graph Size | Nodes | Edges | Full Batch RAM | Sampling RAM | GPU VRAM |
|------------|-------|-------|----------------|--------------|----------|
| Small | 10K | 100K | 2GB | 512MB | 1GB |
| Medium | 100K | 1M | 8GB | 1GB | 4GB |
| Large | 1M | 10M | 32GB | 2GB | 8GB |
| Very Large | 10M | 100M | 128GB+ | 4GB | 16GB |
| Massive | 100M+ | 1B+ | N/A | 8GB | 32GB |

### Software Requirements

#### Operating System
- Ubuntu 20.04 LTS or later
- CentOS 7/8 or RHEL 7/8
- macOS 11.0 or later
- Windows 10/11 with WSL2

#### Python Environment
- Python 3.8 or later
- pip 21.0 or later
- virtualenv or conda

#### Dependencies
```bash
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
networkx>=2.6.0

# GPU support (optional)
cupy>=10.0.0  # For CUDA operations
torch>=1.10.0  # For PyTorch backend

# Performance
numba>=0.55.0
cython>=0.29.0

# Monitoring
prometheus-client>=0.12.0
psutil>=5.8.0
```

## Installation

### Method 1: From Source

```bash
# Clone repository
git clone https://github.com/your-org/gnn-runtime.git
cd gnn-runtime

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .

# Verify installation
python -c "import gnnruntime; print(gnnruntime.__version__)"
```

### Method 2: Using pip

```bash
# Install from PyPI
pip install gnn-runtime

# Install with GPU support
pip install gnn-runtime[gpu]

# Install with all extras
pip install gnn-runtime[all]
```

### Method 3: Using Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV GNN_RUNTIME_CACHE=/data/cache
ENV OMP_NUM_THREADS=4

EXPOSE 8000

CMD ["python", "-m", "gnnruntime.server"]
```

Build and run:
```bash
# Build image
docker build -t gnn-runtime:latest .

# Run container
docker run -d \
    --name gnn-server \
    -p 8000:8000 \
    -v /path/to/data:/data \
    --gpus all \
    gnn-runtime:latest
```

### Method 4: Using Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gnn-runtime
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gnn-runtime
  template:
    metadata:
      labels:
        app: gnn-runtime
    spec:
      containers:
      - name: gnn-runtime
        image: gnn-runtime:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: GNN_BATCH_SIZE
          value: "1024"
        - name: GNN_NUM_WORKERS
          value: "4"
        volumeMounts:
        - name: data-volume
          mountPath: /data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: gnn-data-pvc
```

## Quick Start

### 1. Load and Process a Graph

```bash
# Using CLI
gnn-runtime process \
    --input-graph data/graph.npz \
    --model gcn \
    --layers 2 \
    --hidden-dim 128 \
    --output embeddings.npy
```

```python
# Using Python API
from gnnruntime import GCNConv, Graph
import numpy as np

# Load graph
edge_index = np.load("edges.npy")
features = np.load("features.npy")

graph = Graph(x=features, edge_index=edge_index)

# Create model
gcn1 = GCNConv(features.shape[1], 128)
gcn2 = GCNConv(128, 64)

# Process
h = gcn1(graph.x, graph.edge_index)
h = np.maximum(h, 0)  # ReLU
out = gcn2(h, graph.edge_index)

np.save("embeddings.npy", out)
```

### 2. Start Inference Server

```bash
# Start server
gnn-runtime serve \
    --model-path models/trained_gcn.pkl \
    --host 0.0.0.0 \
    --port 8000 \
    --batch-size 1024 \
    --num-workers 4

# Test server
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d @graph_data.json
```

### 3. Batch Processing

```python
from gnnruntime import BatchProcessor

processor = BatchProcessor(
    model_path="models/trained_gnn.pkl",
    batch_size=32,
    num_workers=4,
    device="cuda"
)

# Process dataset
results = processor.process_dataset(
    input_dir="data/graphs/",
    output_dir="results/",
    file_pattern="*.npz"
)

print(f"Processed {results['total']} graphs")
print(f"Throughput: {results['graphs_per_second']:.2f} graphs/sec")
```

## Production Deployment

### Architecture

```
                    Load Balancer
                         |
           +-------------+-------------+
           |             |             |
      Worker 1      Worker 2      Worker 3
           |             |             |
     GNN Runtime   GNN Runtime   GNN Runtime
           |             |             |
    GPU/CPU Pool   GPU/CPU Pool  GPU/CPU Pool
           |             |             |
      Data Cache    Data Cache    Data Cache
```

### High Availability Setup

```yaml
# docker-compose.yaml
version: '3.8'

services:
  nginx:
    image: nginx:latest
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - gnn1
      - gnn2
      - gnn3

  gnn1:
    image: gnn-runtime:latest
    environment:
      - INSTANCE_ID=1
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/models
    volumes:
      - models:/models
      - cache1:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  gnn2:
    image: gnn-runtime:latest
    environment:
      - INSTANCE_ID=2
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/models
    volumes:
      - models:/models
      - cache2:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  gnn3:
    image: gnn-runtime:latest
    environment:
      - INSTANCE_ID=3
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/models
    volumes:
      - models:/models
      - cache3:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./dashboards:/var/lib/grafana/dashboards

volumes:
  models:
  cache1:
  cache2:
  cache3:
  redis_data:
  prometheus_data:
  grafana_data:
```

### Configuration Management

```python
# config.py
import os

class ProductionConfig:
    # Model settings
    MODEL_PATH = os.environ.get("MODEL_PATH", "/models/gnn.pkl")
    MODEL_TYPE = os.environ.get("MODEL_TYPE", "gcn")
    NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "3"))
    HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "256"))

    # Processing settings
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1024"))
    NUM_NEIGHBORS = [int(x) for x in os.environ.get("NUM_NEIGHBORS", "25,10").split(",")]
    SAMPLING_METHOD = os.environ.get("SAMPLING_METHOD", "neighbor")

    # Performance settings
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
    USE_GPU = os.environ.get("USE_GPU", "true").lower() == "true"
    GPU_MEMORY_FRACTION = float(os.environ.get("GPU_MEMORY_FRACTION", "0.8"))

    # Cache settings
    CACHE_DIR = os.environ.get("CACHE_DIR", "/cache")
    CACHE_SIZE_GB = int(os.environ.get("CACHE_SIZE_GB", "10"))
    ENABLE_CACHE = os.environ.get("ENABLE_CACHE", "true").lower() == "true"

    # Monitoring
    ENABLE_METRICS = os.environ.get("ENABLE_METRICS", "true").lower() == "true"
    METRICS_PORT = int(os.environ.get("METRICS_PORT", "9091"))

    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FILE = os.environ.get("LOG_FILE", "/logs/gnn_runtime.log")
```

### Load Balancing

```nginx
# nginx.conf
upstream gnn_backend {
    least_conn;
    server gnn1:8000 max_fails=3 fail_timeout=30s;
    server gnn2:8000 max_fails=3 fail_timeout=30s;
    server gnn3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.example.com;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location /predict {
        proxy_pass http://gnn_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # Timeouts
        proxy_connect_timeout 30s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }

    location /health {
        proxy_pass http://gnn_backend/health;
        access_log off;
    }

    location /metrics {
        proxy_pass http://gnn_backend/metrics;
        allow 10.0.0.0/8;
        deny all;
    }
}
```

## Performance Tuning

### Graph Processing Optimization

```python
# optimization.py
import os
import numpy as np

class GraphOptimizer:
    @staticmethod
    def reorder_nodes_by_degree(graph):
        """Reorder nodes by degree for better cache locality."""
        degrees = graph.degree()
        perm = np.argsort(degrees)[::-1]

        # Reorder features
        graph.x = graph.x[perm]

        # Reorder edges
        mapping = {old: new for new, old in enumerate(perm)}
        new_edges = []
        for src, dst in graph.edge_index.T:
            new_edges.append([mapping[src], mapping[dst]])
        graph.edge_index = np.array(new_edges).T

        return graph

    @staticmethod
    def coarsen_graph(graph, ratio=0.5):
        """Coarsen graph for multi-level processing."""
        # Implement graph coarsening
        pass

    @staticmethod
    def partition_graph(graph, num_parts):
        """Partition graph for distributed processing."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        # Convert to CSR
        adj = csr_matrix((np.ones(graph.num_edges),
                         (graph.edge_index[0], graph.edge_index[1])),
                        shape=(graph.num_nodes, graph.num_nodes))

        # Find components
        n_components, labels = connected_components(adj, directed=False)

        return labels
```

### GPU Optimization

```python
# gpu_config.py
import os

# CUDA settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use specific GPUs
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Memory optimization
def optimize_gpu_memory():
    """Optimize GPU memory usage."""
    import torch

    # Set memory fraction
    torch.cuda.set_per_process_memory_fraction(0.8)

    # Enable memory caching
    torch.cuda.empty_cache()

    # Set allocator settings
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Kernel optimization
def enable_gpu_optimizations():
    """Enable GPU-specific optimizations."""
    import torch

    # Enable TF32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cuDNN autotuner
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
```

### CPU Optimization

```python
# cpu_config.py
import os

# Threading settings
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# NUMA settings
os.environ["OMP_PROC_BIND"] = "true"
os.environ["OMP_PLACES"] = "cores"

# Vectorization
os.environ["NPY_DISABLE_OPTIMIZATION"] = "0"

def optimize_cpu():
    """Optimize CPU performance."""
    import numpy as np

    # Set BLAS threads
    np.__config__.show()

    # Enable AVX512 if available
    if "avx512" in np.__config__.get_info("blas_info").get("libraries", []):
        os.environ["MKL_ENABLE_AVX512"] = "1"
```

### Sampling Optimization

```python
# sampling_optimization.py

class SamplingOptimizer:
    @staticmethod
    def adaptive_sampling(graph, target_nodes, budget):
        """Adaptively sample based on node importance."""
        degrees = graph.degree()

        # High-degree nodes get fewer samples
        num_samples = []
        for node in target_nodes:
            if degrees[node] > 100:
                num_samples.append(10)
            elif degrees[node] > 50:
                num_samples.append(25)
            else:
                num_samples.append(50)

        return num_samples

    @staticmethod
    def cache_samples(sampler, num_batches=100):
        """Pre-compute and cache samples."""
        cached_samples = []

        for _ in range(num_batches):
            sample = sampler.sample()
            cached_samples.append(sample)

        return cached_samples
```

## Monitoring

### Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, Summary
import time

# Define metrics
graph_processed = Counter('gnn_graphs_processed_total', 'Total graphs processed')
processing_time = Histogram('gnn_processing_duration_seconds', 'Processing time')
active_requests = Gauge('gnn_active_requests', 'Active requests')
memory_usage = Gauge('gnn_memory_usage_bytes', 'Memory usage')
gpu_utilization = Gauge('gnn_gpu_utilization_percent', 'GPU utilization')
cache_hit_rate = Gauge('gnn_cache_hit_rate', 'Cache hit rate')

class MetricsCollector:
    @staticmethod
    def track_processing(func):
        def wrapper(*args, **kwargs):
            active_requests.inc()

            with processing_time.time():
                result = func(*args, **kwargs)

            graph_processed.inc()
            active_requests.dec()

            return result
        return wrapper

    @staticmethod
    def update_system_metrics():
        """Update system metrics."""
        import psutil
        import GPUtil

        # CPU and memory
        memory_usage.set(psutil.Process().memory_info().rss)

        # GPU metrics
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_utilization.set(gpus[0].load * 100)
        except:
            pass
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "GNN Runtime Monitoring",
    "panels": [
      {
        "title": "Throughput",
        "targets": [
          {
            "expr": "rate(gnn_graphs_processed_total[5m])"
          }
        ]
      },
      {
        "title": "Processing Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, gnn_processing_duration_seconds)"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "targets": [
          {
            "expr": "gnn_memory_usage_bytes / 1024 / 1024 / 1024"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "targets": [
          {
            "expr": "gnn_gpu_utilization_percent"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "gnn_cache_hit_rate"
          }
        ]
      }
    ]
  }
}
```

### Health Checks

```python
# health.py
from typing import Dict
import psutil

class HealthChecker:
    @staticmethod
    def check_health() -> Dict:
        """Comprehensive health check."""
        health = {
            "status": "healthy",
            "checks": {}
        }

        # Check model loaded
        try:
            from gnnruntime import get_loaded_model
            model = get_loaded_model()
            health["checks"]["model"] = {
                "status": "ok" if model else "error",
                "loaded": model is not None
            }
        except Exception as e:
            health["checks"]["model"] = {"status": "error", "error": str(e)}

        # Check memory
        mem = psutil.virtual_memory()
        health["checks"]["memory"] = {
            "status": "ok" if mem.percent < 90 else "warning",
            "used_percent": mem.percent,
            "available_gb": mem.available / (1024**3)
        }

        # Check GPU
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                health["checks"]["gpu"] = {
                    "status": "ok",
                    "count": len(gpus),
                    "memory_free_gb": gpus[0].memoryFree / 1024
                }
        except:
            health["checks"]["gpu"] = {"status": "unavailable"}

        # Overall status
        if any(check.get("status") == "error" for check in health["checks"].values()):
            health["status"] = "unhealthy"
        elif any(check.get("status") == "warning" for check in health["checks"].values()):
            health["status"] = "degraded"

        return health
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Errors

**Symptoms:**
```
MemoryError: Unable to allocate array with shape...
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# Reduce batch size
config.batch_size = 512  # From 1024

# Use neighbor sampling
sampler = NeighborSampler(
    num_neighbors=[10, 5],  # Reduce from [25, 10]
    batch_size=256
)

# Enable gradient checkpointing
config.gradient_checkpointing = True

# Clear cache periodically
import gc
gc.collect()
```

#### 2. Slow Performance

**Symptoms:**
- Low throughput (<100 graphs/sec)
- High latency (>100ms per graph)

**Solutions:**
```python
# Enable graph reordering
graph = optimize_graph_ordering(graph)

# Use sparse operations
config.use_sparse_ops = True

# Increase batch size
config.batch_size = 2048

# Enable GPU
config.device = "cuda"

# Profile to find bottlenecks
from gnnruntime.profiler import profile_execution
profile_execution(model, graph)
```

#### 3. Sampling Issues

**Symptoms:**
- Biased samples
- Poor convergence
- High variance

**Solutions:**
```python
# Use importance sampling
sampler = ImportanceSampler(
    edge_index=graph.edge_index,
    importance_scores=compute_importance()
)

# Increase sample coverage
sampler = GraphSAINTSampler(
    sample_coverage=5  # Increase from 3
)

# Use variance reduction
sampler = AdaptiveSampler(
    method="variance_reduction"
)
```

#### 4. Installation Issues

**Problem:** Missing CUDA libraries
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Problem:** Dependency conflicts
```bash
# Create clean environment
conda create -n gnn python=3.9
conda activate gnn

# Install specific versions
pip install numpy==1.23.5
pip install scipy==1.10.1
```

### Debug Mode

```python
# Enable debug logging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable debug mode
from gnnruntime import enable_debug_mode
enable_debug_mode()

# Trace execution
from gnnruntime.debug import trace_execution
with trace_execution():
    output = model(graph)
```

### Performance Profiling

```python
# profiling.py
import cProfile
import pstats
from line_profiler import LineProfiler

def profile_gnn_execution():
    """Profile GNN execution."""
    profiler = cProfile.Profile()

    profiler.enable()
    # Run GNN processing
    result = process_graph(graph)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)

# Line-by-line profiling
lp = LineProfiler()
lp.add_function(gcn_forward)

lp.enable()
output = gcn_forward(x, edge_index)
lp.disable()

lp.print_stats()
```

## Support

### Resources

- Documentation: https://docs.gnnruntime.ai
- GitHub: https://github.com/your-org/gnn-runtime
- Discord: https://discord.gg/gnnruntime
- Email: support@gnnruntime.ai

### FAQ

**Q: Which GNN architecture should I use?**
A: Start with GCN for simplicity, use GAT for heterogeneous importance, GraphSAGE for inductive learning.

**Q: How to handle very large graphs?**
A: Use sampling methods (NeighborSampler, GraphSAINT, ClusterGCN) or distributed processing.

**Q: Can I use custom GNN layers?**
A: Yes, inherit from MessagePassing base class and implement message, aggregate, and update methods.

**Q: How to improve training speed?**
A: Use GPU acceleration, optimize batch sizes, enable sparse operations, and consider sampling methods.

**Q: Is distributed training supported?**
A: Yes, through data parallelism and graph partitioning strategies.