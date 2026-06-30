# Project 30: Large-Scale Parameter Server - Setup Guide

## Overview
Distributed parameter server for large-scale model training with gradient aggregation, sharding, and multiple consistency models.

## Prerequisites
- Python 3.9+
- 32GB+ RAM recommended
- Multi-GPU setup (optional but recommended)
- MPI implementation (OpenMPI or MPICH)
- Network with high bandwidth between nodes

## Installation

### 1. System Dependencies

**MPI Installation**
```bash
# Ubuntu/Debian
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev

# macOS
brew install open-mpi

# Verify installation
mpirun --version
```

**GPU Libraries (for GPU training)**
```bash
# CUDA toolkit (NVIDIA)
# Download from: https://developer.nvidia.com/cuda-downloads

# Verify CUDA
nvcc --version
nvidia-smi
```

**gRPC Tools**
```bash
# Ubuntu/Debian
sudo apt-get install -y protobuf-compiler

# macOS
brew install protobuf
```

### 2. Python Environment
```bash
python -m venv venv
source venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Development dependencies
pip install -r requirements-dev.txt
```

### 3. Build gRPC Stubs
```bash
# Generate Python code from proto files
python -m grpc_tools.protoc \
    -I=./proto \
    --python_out=./src/paramserver/communication \
    --grpc_python_out=./src/paramserver/communication \
    ./proto/parameter_server.proto
```

### 4. Optional: Install Horovod
```bash
# Horovod for distributed training
HOROVOD_WITH_PYTORCH=1 pip install horovod[pytorch]

# Verify installation
horovodrun --check-build
```

## Configuration

### 1. Environment Variables
```bash
# .env file
# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=5000
NUM_SHARDS=4
LEARNING_RATE=0.01

# Consistency Model
# Options: bsp, asp, ssp, bounded
CONSISTENCY_MODEL=bsp

# Communication
COMMUNICATION_BACKEND=grpc
MESSAGE_COMPRESSION=zstd

# Storage
PARAMETER_STORE_TYPE=rocksdb
STORAGE_PATH=./param_store
CHECKPOINT_DIR=./checkpoints

# Redis (for coordination)
REDIS_HOST=localhost
REDIS_PORT=6379

# Monitoring
ENABLE_METRICS=true
PROMETHEUS_PORT=9091
```

### 2. Consistency Models
- **BSP (Bulk Synchronous Parallel)**: Strict synchronization at barriers
- **ASP (Asynchronous Parallel)**: No synchronization, fastest but may diverge
- **SSP (Stale Synchronous Parallel)**: Bounded staleness
- **Bounded**: Custom bounded delay

## Usage

### Single-Node Setup

**1. Start Parameter Server**
```python
from paramserver import create_server

# Create server
server = create_server(
    num_shards=4,
    consistency_model="bsp",
    learning_rate=0.01
)

# Initialize parameters
await server.initialize_parameter(
    name="model.layer1.weight",
    shape=(1000, 512),
    initializer="xavier"
)

await server.initialize_parameter(
    name="model.layer1.bias",
    shape=(512,),
    initializer="zeros"
)

# Start server
await server.start()
```

**2. Worker Training Loop**
```python
import torch
from paramserver import ParameterServerClient

# Connect to server
client = ParameterServerClient(
    server_host="localhost",
    server_port=5000,
    worker_id="worker-1"
)

# Training loop
for iteration in range(num_iterations):
    # Pull parameters
    params = await client.pull(["model.layer1.weight", "model.layer1.bias"])

    # Compute gradients (your training code)
    loss = compute_loss(params, batch)
    gradients = compute_gradients(loss)

    # Push gradients
    await client.push(gradients, iteration)

    # Barrier (for BSP)
    if consistency_model == "bsp":
        await client.barrier(iteration)
```

### Multi-Node Distributed Setup

**1. Start Server on Node 0**
```bash
# On server node
python -m paramserver.server \
    --host 10.0.0.1 \
    --port 5000 \
    --num-shards 8 \
    --consistency bsp
```

**2. Start Workers on Multiple Nodes**
```bash
# On worker node 1
python train.py \
    --worker-id worker-1 \
    --server-host 10.0.0.1 \
    --server-port 5000

# On worker node 2
python train.py \
    --worker-id worker-2 \
    --server-host 10.0.0.1 \
    --server-port 5000

# On worker node 3
python train.py \
    --worker-id worker-3 \
    --server-host 10.0.0.1 \
    --server-port 5000
```

### Using MPI

```bash
# Launch with mpirun
mpirun -n 4 \
    --hostfile hosts.txt \
    python train_mpi.py \
    --server-host 10.0.0.1
```

`train_mpi.py`:
```python
from mpi4py import MPI
from paramserver import ParameterServerClient

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Each rank is a worker
client = ParameterServerClient(
    server_host="10.0.0.1",
    worker_id=f"worker-{rank}"
)

# Training loop
for iteration in range(num_iterations):
    params = await client.pull(param_names)

    # Compute on local data shard
    gradients = compute_gradients(params, local_data)

    # Push to server
    await client.push(gradients, iteration)

    # Barrier
    await client.barrier(iteration)
```

### Advanced Features

**1. Gradient Compression**
```python
from paramserver import GradientCompressor

compressor = GradientCompressor(
    method="topk",
    compression_ratio=0.01  # Keep top 1% gradients
)

compressed_grads = compressor.compress(gradients)
await client.push(compressed_grads, iteration)
```

**2. Adaptive Learning Rate**
```python
from paramserver import LRScheduler

scheduler = LRScheduler(
    initial_lr=0.01,
    schedule="exponential",
    decay_rate=0.96,
    decay_steps=1000
)

# Server-side learning rate adjustment
lr = scheduler.get_lr(iteration)
```

**3. Checkpointing**
```python
# Save checkpoint
await server.save_checkpoint(
    path="./checkpoints/iter-1000",
    iteration=1000
)

# Load checkpoint
await server.load_checkpoint(
    path="./checkpoints/iter-1000"
)
```

**4. Parameter Sharding**
```python
from paramserver import PartitionStrategy

# Custom sharding strategy
strategy = PartitionStrategy(
    num_shards=8,
    method="hash"  # or "range", "custom"
)

server = create_server(
    num_shards=8,
    partition_strategy=strategy
)
```

## Consistency Models Comparison

### BSP (Best for Convergence)
```python
# Strict synchronization
server = create_server(consistency_model="bsp")

# All workers must complete before proceeding
await client.barrier(iteration)
```

### ASP (Best for Speed)
```python
# No synchronization - fastest but may diverge
server = create_server(consistency_model="asp")

# No barrier needed
await client.push(gradients, iteration)
```

### SSP (Balanced)
```python
# Bounded staleness
from paramserver import SSPConfig

config = SSPConfig(staleness_threshold=5)
server = create_server(
    consistency_model="ssp",
    ssp_config=config
)
```

## Monitoring & Debugging

### Server Statistics
```python
stats = server.get_stats()

print(f"Num workers: {stats['num_workers']}")
print(f"Pull count: {stats['pull_count']}")
print(f"Push count: {stats['push_count']}")
print(f"Bytes transferred: {stats['bytes_transferred_mb']} MB")
print(f"Global iteration: {stats['global_iteration']}")
```

### Prometheus Metrics
```bash
# Metrics endpoint
curl http://localhost:9091/metrics

# Key metrics:
# - paramserver_pull_operations_total
# - paramserver_push_operations_total
# - paramserver_gradient_staleness
# - paramserver_worker_count
# - paramserver_bytes_transferred_total
```

### Profiling
```python
from paramserver import enable_profiling

# Enable profiling
enable_profiling(output_dir="./profiles")

# After training, analyze with:
# python -m torch.utils.bottleneck train.py
```

## Testing

```bash
# Run all tests
pytest

# Test parameter server
pytest tests/test_server.py

# Test with MPI (requires mpirun)
mpirun -n 4 pytest tests/test_mpi.py

# Benchmark
pytest tests/test_benchmark.py --benchmark-only
```

## Performance Tuning

### 1. Network Optimization
```python
server = create_server(
    num_shards=16,  # More shards for parallel processing
    compression="zstd",  # Enable compression
    batch_size=100  # Batch multiple parameter updates
)
```

### 2. Memory Management
```python
# Use memory-mapped files for large models
from paramserver import MemoryMappedStore

store = MemoryMappedStore(
    path="./mmap",
    size_gb=100
)
```

### 3. GPU Optimization
```python
# Pin memory for faster GPU transfers
import torch

params = await client.pull(param_names)
params_gpu = {
    k: torch.from_numpy(v).cuda(non_blocking=True)
    for k, v in params.items()
}
```

## Common Issues

### Issue: MPI initialization failed
**Solution**: Ensure MPI is properly installed
```bash
mpirun --version
# Reinstall if needed
pip uninstall mpi4py
pip install mpi4py --no-cache-dir
```

### Issue: Out of memory on server
**Solution**: Increase sharding or use disk-based storage
```python
server = create_server(
    num_shards=32,  # Increase shards
    use_disk_storage=True
)
```

### Issue: Slow gradient aggregation
**Solution**: Enable compression or increase batch size
```python
config = AggregationConfig(
    use_compression=True,
    batch_size=200
)
```

## Project Structure
```
30-large-scale-parameter-server/
├── src/paramserver/
│   ├── server.py           # Main server
│   ├── storage/           # Parameter storage
│   ├── coordination/      # Synchronization
│   ├── aggregation/       # Gradient aggregation
│   ├── communication/     # Network messaging
│   └── ...
├── proto/                 # Protocol buffers
├── tests/
├── requirements.txt
└── SETUP.md
```

## Example: Distributed Training

```python
# Full example: train_distributed.py
import torch
import torch.nn as nn
from paramserver import create_server, ParameterServerClient

# Define model
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
)

# Start server (on main node)
if rank == 0:
    server = create_server(num_shards=4, consistency_model="bsp")
    for name, param in model.named_parameters():
        await server.initialize_parameter(
            name=name,
            shape=param.shape,
            initializer="xavier"
        )
    await server.start()

# Workers
client = ParameterServerClient(
    server_host="10.0.0.1",
    worker_id=f"worker-{rank}"
)

# Training
for epoch in range(num_epochs):
    for batch in dataloader:
        # Pull parameters
        params = await client.pull(list(model.state_dict().keys()))

        # Load into model
        model.load_state_dict(params)

        # Forward & backward
        loss = criterion(model(batch), labels)
        loss.backward()

        # Extract gradients
        grads = [p.grad for p in model.parameters()]

        # Push gradients
        await client.push(grads, epoch)

        # Synchronize
        await client.barrier(epoch)
```

## Next Steps
1. Set up distributed environment
2. Initialize parameter server
3. Configure workers
4. Run distributed training
5. Monitor and optimize

## Resources
- [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Horovod Documentation](https://horovod.readthedocs.io/)
- [MPI for Python](https://mpi4py.readthedocs.io/)
- [gRPC Python](https://grpc.io/docs/languages/python/)
