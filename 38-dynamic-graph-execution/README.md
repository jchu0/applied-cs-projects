# Dynamic Graph Execution

PyTorch-like dynamic computation graph framework with eager execution and automatic differentiation.

> **Concepts covered:** [§03 PyTorch deep learning](../../03-machine-learning-engineering/02-deep-learning/pytorch/pytorch-deep-learning.md) (eager autograd, dynamic graphs) · [§03 Custom layers](../../03-machine-learning-engineering/02-deep-learning/custom-layers/custom-layers.md). Pairs with [Project 35 (autograd-only sibling)](../35-differentiable-programming/), [Project 37 (JIT-tracing runtime)](../37-dynamic-graph-runtime/), [Project 40 (distributed autograd)](../40-distributed-autograd/), [Project 31 (ML compiler — fusion targets)](../31-ml-compiler/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Features

- **Eager Execution**: Immediate operation execution
- **Dynamic Graphs**: Build-by-run computation graphs
- **Autograd**: Automatic gradient computation
- **Graph Optimization**: JIT tracing and fusion

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from dynagraph import Tensor

# Create tensors with gradient tracking
x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
w = Tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)

# Dynamic computation
y = x @ w
z = y.sum()

# Backward pass builds graph dynamically
z.backward()

print(x.grad)  # Gradients computed
print(w.grad)
```

## Control Flow

```python
from dynagraph import Tensor

def dynamic_rnn(x, hidden, weights):
    outputs = []
    for t in range(x.shape[0]):
        # Control flow works naturally
        if t > 0:
            hidden = hidden + x[t] @ weights
        else:
            hidden = x[t] @ weights

        hidden = hidden.tanh()
        outputs.append(hidden)

    return outputs

# Gradients flow through control flow
loss = sum(o.sum() for o in outputs)
loss.backward()
```

## Graph Optimization

```python
from dynagraph import jit_trace

# Trace for optimization
@jit_trace
def optimized_forward(x, w):
    return (x @ w).relu().sum()

# First call traces, subsequent calls use optimized graph
result = optimized_forward(x, w)
```

## Testing

```bash
pytest tests/ -v  # 127 tests
```
