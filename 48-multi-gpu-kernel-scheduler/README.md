# Multi-GPU Kernel Scheduler

Advanced kernel scheduler for multi-GPU systems with critical path analysis and CUDA graph optimization.

## Features

- **Critical Path Analysis**: Optimize execution order
- **Dependency Tracking**: Automatic kernel dependencies
- **CUDA Graphs**: Capture and replay kernel sequences
- **Multi-GPU**: Distribute across multiple GPUs
- **Memory Planning**: Optimize memory allocation

## Installation

```bash
pip install -e .

# With CUDA support
pip install -e ".[cuda]"
```

## Quick Start

```python
from kernelsched import KernelScheduler, Kernel

# Create scheduler
scheduler = KernelScheduler(num_gpus=4)

# Define kernels with dependencies
k1 = Kernel("matmul_1", duration_ms=10)
k2 = Kernel("matmul_2", duration_ms=10)
k3 = Kernel("reduce", duration_ms=5, depends_on=[k1, k2])

# Add to scheduler
scheduler.add_kernel(k1)
scheduler.add_kernel(k2)
scheduler.add_kernel(k3)

# Optimize and execute
schedule = scheduler.optimize()
scheduler.execute(schedule)
```

## Critical Path Optimization

```python
from kernelsched import CriticalPathAnalyzer

analyzer = CriticalPathAnalyzer()
analyzer.add_kernels(kernels)

# Find critical path
critical_path = analyzer.compute_critical_path()
print(f"Critical path length: {critical_path.total_time}ms")

# Get optimized schedule
schedule = analyzer.get_optimized_schedule()
```

## CUDA Graph Capture

```python
from kernelsched import CUDAGraphCapture

# Capture kernel sequence
with CUDAGraphCapture() as capture:
    result = model(input_batch)

# Replay captured graph (faster execution)
graph = capture.get_graph()
for batch in batches:
    graph.replay(batch)
```

## Multi-GPU Distribution

```python
scheduler = KernelScheduler(
    num_gpus=4,
    strategy="load_balance"  # or "memory_balance", "latency_optimize"
)

# Kernels automatically distributed
for kernel in kernels:
    scheduler.add_kernel(kernel)

scheduler.execute()
```

## Testing

```bash
pytest tests/ -v  # 209 tests
```
