# Distributed Tensor Algebra

JAX-like tensor library with automatic differentiation, JIT compilation, and distributed sharding.

## Features

- **Lazy Evaluation**: Build computation graphs before execution
- **Autodiff**: Automatic gradient computation
- **JIT Compilation**: Trace-based optimization
- **Distributed Sharding**: Tensor partitioning across devices
- **NumPy Compatible**: Familiar array operations

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from tensorlib import array, grad, jit

# Create tensors
x = array([[1.0, 2.0], [3.0, 4.0]])
w = array([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)

# Define computation
def loss_fn(w, x):
    return (w @ x).sum()

# Compute gradients
grad_fn = grad(loss_fn)
dw = grad_fn(w, x)

# JIT compile for speed
fast_loss = jit(loss_fn)
result = fast_loss(w, x)
```

## Distributed Training

```python
from tensorlib import pmap, shard_tensor

# Shard data across devices
sharded_x = shard_tensor(x, axis=0, num_shards=4)

# Parallel map across shards
@pmap
def train_step(params, batch):
    loss, grads = value_and_grad(loss_fn)(params, batch)
    return params - 0.01 * grads

# Execute in parallel
new_params = train_step(params, sharded_x)
```

## API Reference

### Core Functions

| Function | Description |
|----------|-------------|
| `array(data)` | Create a tensor |
| `grad(fn)` | Get gradient function |
| `jit(fn)` | JIT compile function |
| `pmap(fn)` | Parallel map |
| `shard_tensor(t, axis, n)` | Shard tensor |

## Testing

```bash
pytest tests/ -v  # 175 tests
```
