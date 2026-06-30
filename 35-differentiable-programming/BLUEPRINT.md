# Differentiable Programming Runtime (Autograd-lite) - Technical Blueprint

## Executive Summary

This project implements a production-grade automatic differentiation engine for differentiable programming. It features a dynamic computational graph, reverse-mode autodiff with gradient tape, custom operation support, gradient checkpointing for memory efficiency, and thread-safe execution for production environments.

> **Concepts covered:** [§03 PyTorch deep learning](../../03-machine-learning-engineering/02-deep-learning/pytorch/pytorch-deep-learning.md) (the autograd system this project rebuilds from scratch) · [§03 Custom layers](../../03-machine-learning-engineering/02-deep-learning/custom-layers/custom-layers.md) (custom backward passes). Pairs with [Project 38 (dynamic graph execution)](../38-dynamic-graph-execution/), [Project 37 (TorchDynamo-style runtime)](../37-dynamic-graph-runtime/), [Project 40 (distributed autograd)](../40-distributed-autograd/), [Project 32 (tensor algebra)](../32-distributed-tensor-algebra/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  Differentiable Programming Runtime Architecture             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         User API Layer                                │   │
│  │   Tensor    requires_grad    backward()    no_grad    autograd.grad  │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Tensor and Operations                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Tensor     │  │  Function    │  │  Operation   │               │   │
│  │  │  (data,grad, │  │  (forward,   │  │  Registry    │               │   │
│  │  │   grad_fn)   │  │   backward)  │  │              │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Backward Graph Engine                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Graph      │  │  Topological │  │  Gradient    │               │   │
│  │  │   Builder    │──│   Sort       │──│   Engine     │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
│                                     │                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                   Memory & Performance Optimization                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │  Gradient    │  │   Mixed      │  │   Thread     │               │   │
│  │  │ Checkpointing│  │  Precision   │  │   Safety     │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Tensor Class

```python
import numpy as np
from typing import Optional, List, Tuple, Callable, Set, Any
from dataclasses import dataclass, field
import threading
from contextlib import contextmanager

class Tensor:
    """
    Core tensor class with automatic differentiation support.
    Similar to PyTorch's Tensor with requires_grad.
    """

    def __init__(self,
                 data: np.ndarray,
                 requires_grad: bool = False,
                 dtype: np.dtype = None):
        if isinstance(data, Tensor):
            data = data.data

        self.data = np.array(data, dtype=dtype or np.float32)
        self.requires_grad = requires_grad

        # Gradient storage
        self.grad: Optional[np.ndarray] = None

        # Backward graph node
        self.grad_fn: Optional['Function'] = None

        # For leaf tensors tracking
        self._is_leaf = True

        # Hooks
        self._backward_hooks: List[Callable] = []
        self._forward_hooks: List[Callable] = []

        # Version for detecting in-place modifications
        self._version = 0

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def is_leaf(self) -> bool:
        """Leaf tensors are created directly (not from operations)."""
        return self._is_leaf

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        if self.requires_grad:
            raise RuntimeError(
                "Can't call numpy() on Tensor that requires grad. "
                "Use tensor.detach().numpy() instead."
            )
        return self.data.copy()

    def detach(self) -> 'Tensor':
        """Return a new tensor detached from computation graph."""
        t = Tensor(self.data.copy(), requires_grad=False)
        return t

    def clone(self) -> 'Tensor':
        """Return a copy that shares the graph."""
        return CloneFunction.apply(self)

    def retain_grad(self):
        """Retain gradient for non-leaf tensor."""
        if not self.is_leaf:
            # Register hook to save gradient
            def hook(grad):
                self.grad = grad.copy()
            self.register_hook(hook)

    def register_hook(self, hook: Callable):
        """Register a backward hook."""
        self._backward_hooks.append(hook)

    def backward(self, gradient: Optional['Tensor'] = None):
        """
        Compute gradients via backpropagation.

        Args:
            gradient: Gradient of loss w.r.t. this tensor.
                      Default is ones for scalar tensors.
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot backward on tensor that doesn't require grad")

        if gradient is None:
            if self.data.size == 1:
                gradient = Tensor(np.ones_like(self.data))
            else:
                raise RuntimeError(
                    "grad must be specified for non-scalar tensors"
                )

        # Run the backward engine
        engine = BackwardEngine()
        engine.run(self, gradient)

    def zero_grad(self):
        """Zero out the gradient."""
        self.grad = None

    # Arithmetic operations
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __mul__(self, other):
        return mul(self, other)

    def __rmul__(self, other):
        return mul(other, self)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return sub(self, other)

    def __rsub__(self, other):
        return sub(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    # Shape operations
    @property
    def T(self):
        return transpose(self)

    def transpose(self, *dims):
        return transpose(self, dims if dims else None)

    def reshape(self, *shape):
        return reshape(self, shape)

    def view(self, *shape):
        return reshape(self, shape)

    def sum(self, dim=None, keepdim=False):
        return sum(self, dim, keepdim)

    def mean(self, dim=None, keepdim=False):
        return mean(self, dim, keepdim)

    def squeeze(self, dim=None):
        return squeeze(self, dim)

    def unsqueeze(self, dim):
        return unsqueeze(self, dim)

    # Activation functions
    def relu(self):
        return relu(self)

    def sigmoid(self):
        return sigmoid(self)

    def tanh(self):
        return tanh(self)

    def exp(self):
        return exp(self)

    def log(self):
        return log(self)

    def __repr__(self):
        grad_info = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        grad_fn_info = f", grad_fn={self.grad_fn.__class__.__name__}" if self.grad_fn else ""
        return f"Tensor({self.data}{grad_info}{grad_fn_info})"

# Utility functions for creating tensors
def tensor(data, requires_grad=False, dtype=None):
    """Create a tensor from data."""
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)

def zeros(*shape, requires_grad=False, dtype=np.float32):
    """Create tensor of zeros."""
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad=requires_grad)

def ones(*shape, requires_grad=False, dtype=np.float32):
    """Create tensor of ones."""
    return Tensor(np.ones(shape, dtype=dtype), requires_grad=requires_grad)

def randn(*shape, requires_grad=False, dtype=np.float32):
    """Create tensor with random normal values."""
    return Tensor(np.random.randn(*shape).astype(dtype), requires_grad=requires_grad)

def rand(*shape, requires_grad=False, dtype=np.float32):
    """Create tensor with random uniform values."""
    return Tensor(np.random.rand(*shape).astype(dtype), requires_grad=requires_grad)

def arange(start, end=None, step=1, requires_grad=False, dtype=np.float32):
    """Create tensor with range of values."""
    if end is None:
        start, end = 0, start
    return Tensor(np.arange(start, end, step, dtype=dtype), requires_grad=requires_grad)
```

### 2. Function and Backward Graph

```python
class Function:
    """
    Base class for differentiable functions.
    Each function defines forward and backward passes.
    """

    def __init__(self):
        # Saved tensors for backward
        self.saved_tensors: List[np.ndarray] = []

        # Input grad_fns for graph traversal
        self.next_functions: List[Tuple[Optional['Function'], int]] = []

        # Metadata
        self.needs_input_grad: List[bool] = []

    @classmethod
    def apply(cls, *inputs):
        """Apply the function to inputs."""
        func = cls()

        # Convert inputs to tensors
        tensor_inputs = []
        for inp in inputs:
            if isinstance(inp, Tensor):
                tensor_inputs.append(inp)
            else:
                tensor_inputs.append(Tensor(inp))

        # Determine if we need gradients
        requires_grad = any(t.requires_grad for t in tensor_inputs)

        # Track which inputs need gradients
        func.needs_input_grad = [t.requires_grad for t in tensor_inputs]

        # Forward pass
        output_data = func.forward(*[t.data for t in tensor_inputs])

        # Create output tensor
        output = Tensor(output_data, requires_grad=requires_grad)

        if requires_grad:
            # Mark as non-leaf
            output._is_leaf = False

            # Set grad_fn
            output.grad_fn = func

            # Record graph connections
            for i, inp in enumerate(tensor_inputs):
                if inp.requires_grad:
                    func.next_functions.append((inp.grad_fn, i))
                else:
                    func.next_functions.append((None, i))

        return output

    def forward(self, *inputs) -> np.ndarray:
        """Forward computation. Override in subclass."""
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> Tuple[Optional[np.ndarray], ...]:
        """Backward computation. Override in subclass."""
        raise NotImplementedError

    def save_for_backward(self, *tensors):
        """Save tensors for use in backward pass."""
        self.saved_tensors = list(tensors)

# ============= Core Operations =============

class AddFunction(Function):
    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class Add(Function):
    def forward(self, a, b):
        # Handle broadcasting
        self.save_for_backward(a.shape, b.shape)
        return a + b

    def backward(self, grad_output):
        a_shape, b_shape = self.saved_tensors

        grad_a = grad_output
        grad_b = grad_output

        # Reduce gradients if broadcasting occurred
        grad_a = _reduce_broadcast_gradient(grad_a, a_shape)
        grad_b = _reduce_broadcast_gradient(grad_b, b_shape)

        return grad_a, grad_b

class Mul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a * b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_output * b
        grad_b = grad_output * a

        # Reduce for broadcasting
        grad_a = _reduce_broadcast_gradient(grad_a, a.shape)
        grad_b = _reduce_broadcast_gradient(grad_b, b.shape)

        return grad_a, grad_b

class Sub(Function):
    def forward(self, a, b):
        self.save_for_backward(a.shape, b.shape)
        return a - b

    def backward(self, grad_output):
        a_shape, b_shape = self.saved_tensors
        grad_a = _reduce_broadcast_gradient(grad_output, a_shape)
        grad_b = _reduce_broadcast_gradient(-grad_output, b_shape)
        return grad_a, grad_b

class Div(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return a / b

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_output / b
        grad_b = -grad_output * a / (b * b)

        grad_a = _reduce_broadcast_gradient(grad_a, a.shape)
        grad_b = _reduce_broadcast_gradient(grad_b, b.shape)

        return grad_a, grad_b

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, grad_output):
        return (-grad_output,)

class Pow(Function):
    def forward(self, x, p):
        self.save_for_backward(x, p)
        return np.power(x, p)

    def backward(self, grad_output):
        x, p = self.saved_tensors
        grad_x = grad_output * p * np.power(x, p - 1)
        # Gradient w.r.t. power (if needed)
        grad_p = grad_output * np.power(x, p) * np.log(x + 1e-8)
        return grad_x, grad_p

class MatMul(Function):
    def forward(self, a, b):
        self.save_for_backward(a, b)
        return np.matmul(a, b)

    def backward(self, grad_output):
        a, b = self.saved_tensors

        # d(a @ b)/da = grad @ b.T
        # d(a @ b)/db = a.T @ grad
        if a.ndim == 1 and b.ndim == 1:
            # Vector-vector: outer product for grad
            grad_a = grad_output * b
            grad_b = grad_output * a
        elif a.ndim == 2 and b.ndim == 2:
            grad_a = np.matmul(grad_output, b.T)
            grad_b = np.matmul(a.T, grad_output)
        else:
            # Batched matmul
            grad_a = np.matmul(grad_output, np.swapaxes(b, -1, -2))
            grad_b = np.matmul(np.swapaxes(a, -1, -2), grad_output)

        return grad_a, grad_b

# ============= Activation Functions =============

class ReLU(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.maximum(x, 0)

    def backward(self, grad_output):
        x, = self.saved_tensors
        return (grad_output * (x > 0).astype(grad_output.dtype),)

class Sigmoid(Function):
    def forward(self, x):
        out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        self.save_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.saved_tensors
        return (grad_output * out * (1 - out),)

class Tanh(Function):
    def forward(self, x):
        out = np.tanh(x)
        self.save_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.saved_tensors
        return (grad_output * (1 - out * out),)

class Exp(Function):
    def forward(self, x):
        out = np.exp(x)
        self.save_for_backward(out)
        return out

    def backward(self, grad_output):
        out, = self.saved_tensors
        return (grad_output * out,)

class Log(Function):
    def forward(self, x):
        self.save_for_backward(x)
        return np.log(x + 1e-8)

    def backward(self, grad_output):
        x, = self.saved_tensors
        return (grad_output / (x + 1e-8),)

# ============= Reduction Operations =============

class Sum(Function):
    def forward(self, x, axis=None, keepdims=False):
        self.save_for_backward(x.shape, axis, keepdims)
        return np.sum(x, axis=axis, keepdims=keepdims)

    def backward(self, grad_output):
        input_shape, axis, keepdims = self.saved_tensors

        # Expand grad to match input shape
        if axis is not None and not keepdims:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                grad_output = np.expand_dims(grad_output, ax)

        # Broadcast to input shape
        return (np.broadcast_to(grad_output, input_shape).copy(),)

class Mean(Function):
    def forward(self, x, axis=None, keepdims=False):
        self.save_for_backward(x.shape, axis, keepdims)
        return np.mean(x, axis=axis, keepdims=keepdims)

    def backward(self, grad_output):
        input_shape, axis, keepdims = self.saved_tensors

        # Compute number of elements averaged
        if axis is None:
            n = np.prod(input_shape)
        else:
            if isinstance(axis, int):
                axis = (axis,)
            n = np.prod([input_shape[ax] for ax in axis])

        # Expand and scale
        if axis is not None and not keepdims:
            for ax in sorted(axis):
                grad_output = np.expand_dims(grad_output, ax)

        grad = np.broadcast_to(grad_output, input_shape).copy() / n
        return (grad,)

# ============= Shape Operations =============

class Reshape(Function):
    def forward(self, x, shape):
        self.save_for_backward(x.shape)
        return x.reshape(shape)

    def backward(self, grad_output):
        input_shape, = self.saved_tensors
        return (grad_output.reshape(input_shape),)

class Transpose(Function):
    def forward(self, x, axes=None):
        if axes is None:
            axes = tuple(reversed(range(x.ndim)))
        self.save_for_backward(axes)
        return np.transpose(x, axes)

    def backward(self, grad_output):
        axes, = self.saved_tensors
        # Invert the permutation
        inv_axes = np.argsort(axes)
        return (np.transpose(grad_output, inv_axes),)

class Squeeze(Function):
    def forward(self, x, dim=None):
        self.save_for_backward(x.shape)
        return np.squeeze(x, axis=dim)

    def backward(self, grad_output):
        input_shape, = self.saved_tensors
        return (grad_output.reshape(input_shape),)

class Unsqueeze(Function):
    def forward(self, x, dim):
        self.save_for_backward(dim)
        return np.expand_dims(x, axis=dim)

    def backward(self, grad_output):
        dim, = self.saved_tensors
        return (np.squeeze(grad_output, axis=dim),)

class Clone(Function):
    def forward(self, x):
        return x.copy()

    def backward(self, grad_output):
        return (grad_output,)

# ============= Helper Functions =============

def _reduce_broadcast_gradient(grad: np.ndarray, target_shape: Tuple) -> np.ndarray:
    """Reduce gradient to match target shape (undo broadcasting)."""
    if grad.shape == target_shape:
        return grad

    # Sum over dimensions that were broadcast
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

# ============= Public API =============

def add(a, b):
    return Add.apply(a, b if isinstance(b, Tensor) else Tensor(b))

def mul(a, b):
    return Mul.apply(a if isinstance(a, Tensor) else Tensor(a),
                     b if isinstance(b, Tensor) else Tensor(b))

def sub(a, b):
    return Sub.apply(a if isinstance(a, Tensor) else Tensor(a),
                     b if isinstance(b, Tensor) else Tensor(b))

def div(a, b):
    return Div.apply(a if isinstance(a, Tensor) else Tensor(a),
                     b if isinstance(b, Tensor) else Tensor(b))

def neg(x):
    return Neg.apply(x)

def pow(x, p):
    return Pow.apply(x, p)

def matmul(a, b):
    return MatMul.apply(a, b)

def relu(x):
    return ReLU.apply(x)

def sigmoid(x):
    return Sigmoid.apply(x)

def tanh(x):
    return Tanh.apply(x)

def exp(x):
    return Exp.apply(x)

def log(x):
    return Log.apply(x)

def sum(x, dim=None, keepdim=False):
    return Sum.apply(x, dim, keepdim)

def mean(x, dim=None, keepdim=False):
    return Mean.apply(x, dim, keepdim)

def reshape(x, shape):
    return Reshape.apply(x, shape)

def transpose(x, dims=None):
    return Transpose.apply(x, dims)

def squeeze(x, dim=None):
    return Squeeze.apply(x, dim)

def unsqueeze(x, dim):
    return Unsqueeze.apply(x, dim)

CloneFunction = Clone
```

### 3. Backward Engine

```python
from collections import defaultdict
import heapq

class BackwardEngine:
    """
    Engine for executing backward pass.
    Implements topological sort and gradient accumulation.
    """

    def __init__(self):
        self.grad_map: Dict[int, np.ndarray] = {}  # tensor id -> gradient
        self._lock = threading.Lock()

    def run(self, root: Tensor, grad_output: Tensor):
        """
        Execute backward pass from root tensor.

        Args:
            root: Tensor to differentiate
            grad_output: Gradient of loss w.r.t. root
        """
        if root.grad_fn is None:
            # Leaf tensor - just accumulate gradient
            if root.is_leaf and root.requires_grad:
                if root.grad is None:
                    root.grad = grad_output.data.copy()
                else:
                    root.grad += grad_output.data
            return

        # Build topological order
        topo_order = self._build_topo_order(root)

        # Initialize gradient at root
        self.grad_map[id(root)] = grad_output.data.copy()

        # Process in reverse topological order
        for node in topo_order:
            if node.grad_fn is None:
                continue

            # Get accumulated gradient for this node
            grad = self.grad_map.get(id(node))
            if grad is None:
                continue

            # Execute backward hooks
            for hook in node._backward_hooks:
                new_grad = hook(grad)
                if new_grad is not None:
                    grad = new_grad

            # Compute input gradients
            input_grads = node.grad_fn.backward(grad)

            # Distribute gradients to inputs
            for i, (next_fn, idx) in enumerate(node.grad_fn.next_functions):
                if i >= len(input_grads) or input_grads[i] is None:
                    continue

                input_grad = input_grads[i]

                # Find the input tensor
                # This requires tracking inputs during forward
                # For now, we'll use the function's saved info

                # Accumulate gradient
                # In practice, need to track tensor IDs through the graph

        # Accumulate gradients to leaf tensors
        self._accumulate_leaf_grads(root, grad_output.data)

    def _build_topo_order(self, root: Tensor) -> List[Tensor]:
        """Build topological order via DFS."""
        visited = set()
        order = []

        def dfs(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))

            if tensor.grad_fn is not None:
                for next_fn, _ in tensor.grad_fn.next_functions:
                    if next_fn is not None:
                        # Need to traverse to input tensors
                        # This is simplified - full impl needs tensor tracking
                        pass

            order.append(tensor)

        dfs(root)
        return order

    def _accumulate_leaf_grads(self, root: Tensor, grad_output: np.ndarray):
        """
        Accumulate gradients to leaf tensors.
        Uses explicit graph traversal.
        """
        # Map from grad_fn to its output gradient
        fn_grads: Dict[int, np.ndarray] = {}
        fn_grads[id(root.grad_fn)] = grad_output

        # BFS traversal
        queue = [(root.grad_fn, grad_output)]
        visited = set()

        while queue:
            fn, grad = queue.pop(0)

            if fn is None or id(fn) in visited:
                continue
            visited.add(id(fn))

            # Compute backward
            input_grads = fn.backward(grad)

            # Process inputs
            for i, (next_fn, _) in enumerate(fn.next_functions):
                if i >= len(input_grads):
                    continue

                input_grad = input_grads[i]
                if input_grad is None:
                    continue

                if next_fn is None:
                    # This is a leaf tensor - find it and accumulate
                    # In full implementation, track tensor references
                    pass
                else:
                    # Accumulate and continue
                    if id(next_fn) in fn_grads:
                        fn_grads[id(next_fn)] = fn_grads[id(next_fn)] + input_grad
                    else:
                        fn_grads[id(next_fn)] = input_grad

                    queue.append((next_fn, fn_grads[id(next_fn)]))

class GradTracker:
    """
    Simplified gradient tracker that maintains tensor references.
    """

    def __init__(self):
        self.tensor_map: Dict[int, Tensor] = {}  # fn_id -> output tensor
        self.input_map: Dict[int, List[Tensor]] = {}  # fn_id -> input tensors

    def register_output(self, fn: Function, output: Tensor, inputs: List[Tensor]):
        """Register function output and inputs."""
        self.tensor_map[id(fn)] = output
        self.input_map[id(fn)] = inputs

    def backward(self, root: Tensor, grad_output: np.ndarray):
        """Execute backward pass with proper tensor tracking."""
        # Initialize
        fn_grads = {id(root.grad_fn): grad_output}

        # Topological sort via DFS
        order = []
        visited = set()

        def dfs(fn):
            if fn is None or id(fn) in visited:
                return
            visited.add(id(fn))

            for next_fn, _ in fn.next_functions:
                dfs(next_fn)

            order.append(fn)

        dfs(root.grad_fn)
        order.reverse()

        # Execute backward
        for fn in order:
            grad = fn_grads.get(id(fn))
            if grad is None:
                continue

            # Get input tensors
            inputs = self.input_map.get(id(fn), [])

            # Compute gradients
            input_grads = fn.backward(grad)

            # Distribute to inputs
            for i, (input_tensor, input_grad) in enumerate(zip(inputs, input_grads)):
                if input_grad is None:
                    continue

                if input_tensor.grad_fn is None:
                    # Leaf tensor
                    if input_tensor.requires_grad:
                        if input_tensor.grad is None:
                            input_tensor.grad = input_grad.copy()
                        else:
                            input_tensor.grad += input_grad
                else:
                    # Non-leaf - accumulate for its grad_fn
                    fn_id = id(input_tensor.grad_fn)
                    if fn_id in fn_grads:
                        fn_grads[fn_id] = fn_grads[fn_id] + input_grad
                    else:
                        fn_grads[fn_id] = input_grad
```

### 4. Context Managers and Utilities

```python
# Thread-local storage for gradient context
_grad_enabled = threading.local()
_grad_enabled.value = True

class no_grad:
    """Context manager to disable gradient computation."""

    def __enter__(self):
        self.prev = getattr(_grad_enabled, 'value', True)
        _grad_enabled.value = False
        return self

    def __exit__(self, *args):
        _grad_enabled.value = self.prev

class enable_grad:
    """Context manager to enable gradient computation."""

    def __enter__(self):
        self.prev = getattr(_grad_enabled, 'value', True)
        _grad_enabled.value = True
        return self

    def __exit__(self, *args):
        _grad_enabled.value = self.prev

def is_grad_enabled() -> bool:
    """Check if gradient computation is enabled."""
    return getattr(_grad_enabled, 'value', True)

def set_grad_enabled(mode: bool):
    """Set gradient computation mode."""
    _grad_enabled.value = mode

@contextmanager
def inference_mode():
    """Context manager for inference (no gradients, no graph)."""
    with no_grad():
        yield
```

### 5. Autograd Functions

```python
class autograd:
    """Autograd utilities."""

    @staticmethod
    def grad(outputs: List[Tensor],
             inputs: List[Tensor],
             grad_outputs: Optional[List[Tensor]] = None,
             retain_graph: bool = False,
             create_graph: bool = False) -> List[Optional[Tensor]]:
        """
        Compute gradients of outputs w.r.t. inputs.

        Args:
            outputs: Tensors to differentiate
            inputs: Tensors w.r.t. which to differentiate
            grad_outputs: Gradients w.r.t. outputs
            retain_graph: Keep computation graph for further backward
            create_graph: Create graph for higher-order derivatives

        Returns:
            List of gradients for each input
        """
        if grad_outputs is None:
            grad_outputs = [Tensor(np.ones_like(out.data)) for out in outputs]

        # Clear existing gradients
        for inp in inputs:
            inp.grad = None

        # Run backward for each output
        for out, grad_out in zip(outputs, grad_outputs):
            out.backward(grad_out)

        # Collect gradients
        grads = []
        for inp in inputs:
            if inp.grad is not None:
                grads.append(Tensor(inp.grad.copy()))
            else:
                grads.append(None)

        return grads

    @staticmethod
    def backward(tensors: List[Tensor],
                 grad_tensors: Optional[List[Tensor]] = None,
                 retain_graph: bool = False):
        """
        Compute gradients for multiple tensors.

        Args:
            tensors: Tensors to differentiate
            grad_tensors: Gradients w.r.t. each tensor
        """
        if grad_tensors is None:
            grad_tensors = [None] * len(tensors)

        for tensor, grad in zip(tensors, grad_tensors):
            tensor.backward(grad)

class Function:
    """Enhanced Function with create_graph support."""

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Setup for custom backward. Override in subclass."""
        pass

    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Forward pass. Override in subclass."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass. Override in subclass."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply function."""
        # Implementation similar to before
        pass
```

### 6. Gradient Checkpointing

```python
class CheckpointFunction(Function):
    """
    Checkpoint function for memory-efficient training.
    Recomputes forward pass during backward instead of storing activations.
    """

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        if preserve_rng_state:
            ctx.rng_state = np.random.get_state()

        # Run forward without gradient tracking
        with no_grad():
            outputs = run_function(*args)

        # Save input tensors (not activations)
        ctx.save_for_backward(*[a.data if isinstance(a, Tensor) else a for a in args])

        return outputs.data if isinstance(outputs, Tensor) else outputs

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors

        # Restore RNG state
        if ctx.preserve_rng_state:
            np.random.set_state(ctx.rng_state)

        # Recreate tensors with gradient tracking
        input_tensors = [
            Tensor(inp, requires_grad=True) if isinstance(inp, np.ndarray) else inp
            for inp in inputs
        ]

        # Recompute forward
        with enable_grad():
            outputs = ctx.run_function(*input_tensors)

        # Backward through recomputed graph
        if isinstance(outputs, Tensor):
            outputs.backward(Tensor(grad_output))

        # Return gradients
        return tuple(
            t.grad if isinstance(t, Tensor) and t.grad is not None else None
            for t in input_tensors
        )

def checkpoint(function: Callable, *args, preserve_rng_state: bool = True) -> Tensor:
    """
    Checkpoint a function for memory efficiency.

    Args:
        function: Function to checkpoint
        *args: Arguments to function
        preserve_rng_state: Preserve random state for reproducibility

    Returns:
        Output of function
    """
    return CheckpointFunction.apply(function, preserve_rng_state, *args)

def checkpoint_sequential(functions: List[Callable], *args) -> Tensor:
    """Checkpoint a sequence of functions."""
    def sequential_fn(*inputs):
        x = inputs[0] if len(inputs) == 1 else inputs
        for fn in functions:
            x = fn(x)
        return x

    return checkpoint(sequential_fn, *args)
```

### 7. Mixed Precision Support

```python
class GradScaler:
    """
    Gradient scaler for mixed precision training.
    Scales loss to prevent underflow in float16 gradients.
    """

    def __init__(self,
                 init_scale: float = 65536.0,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval

        self._growth_tracker = 0
        self._found_inf = False

    def scale(self, loss: Tensor) -> Tensor:
        """Scale loss for backward pass."""
        return loss * self._scale

    def unscale_(self, optimizer):
        """Unscale gradients in optimizer."""
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad = param.grad / self._scale

                    # Check for inf/nan
                    if np.any(np.isinf(param.grad)) or np.any(np.isnan(param.grad)):
                        self._found_inf = True

    def step(self, optimizer):
        """Perform optimizer step if gradients are valid."""
        if not self._found_inf:
            optimizer.step()

    def update(self):
        """Update scale factor."""
        if self._found_inf:
            # Backoff
            self._scale *= self._backoff_factor
            self._found_inf = False
            self._growth_tracker = 0
        else:
            # Potentially grow
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0

    @property
    def scale_value(self) -> float:
        return self._scale

@contextmanager
def autocast(dtype=np.float16):
    """
    Context manager for automatic mixed precision.
    Converts operations to lower precision.
    """
    # In a full implementation, this would intercept tensor creation
    # and operations to use the specified dtype
    yield
```

### 8. Hooks and Custom Operations

```python
class Hook:
    """Base class for hooks."""

    def __init__(self, tensor: Tensor):
        self.tensor = tensor
        self.id = id(self)

    def remove(self):
        """Remove this hook."""
        pass

def register_hook(tensor: Tensor, hook: Callable) -> Hook:
    """
    Register a backward hook on a tensor.

    The hook is called with the gradient during backward.
    It can modify and return a new gradient, or return None.
    """
    tensor._backward_hooks.append(hook)
    return Hook(tensor)

class ModuleHook:
    """Hook for module forward/backward."""

    def __init__(self):
        self.forward_hooks: List[Callable] = []
        self.backward_hooks: List[Callable] = []

    def register_forward_hook(self, hook: Callable):
        self.forward_hooks.append(hook)

    def register_backward_hook(self, hook: Callable):
        self.backward_hooks.append(hook)

def custom_bwd(fn: Callable):
    """Decorator for custom backward function."""
    def wrapper(ctx, *grad_outputs):
        return fn(ctx, *grad_outputs)
    return wrapper

def custom_fwd(fn: Callable):
    """Decorator for custom forward function."""
    def wrapper(ctx, *args, **kwargs):
        return fn(ctx, *args, **kwargs)
    return wrapper

# Example custom function
class CustomFunction(Function):
    """Example of a custom differentiable function."""

    @staticmethod
    @custom_fwd
    def forward(ctx, x, y, alpha):
        ctx.save_for_backward(x, y)
        ctx.alpha = alpha
        return x * alpha + y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        alpha = ctx.alpha
        return grad_output * alpha, grad_output, None
```

### 9. Thread Safety

```python
import threading

class ThreadSafeGradAccumulator:
    """Thread-safe gradient accumulator for data parallel training."""

    def __init__(self):
        self._lock = threading.Lock()
        self._grads: Dict[int, np.ndarray] = {}

    def accumulate(self, param_id: int, grad: np.ndarray):
        """Accumulate gradient for a parameter."""
        with self._lock:
            if param_id in self._grads:
                self._grads[param_id] += grad
            else:
                self._grads[param_id] = grad.copy()

    def get_grad(self, param_id: int) -> Optional[np.ndarray]:
        """Get accumulated gradient."""
        with self._lock:
            return self._grads.get(param_id)

    def clear(self):
        """Clear all gradients."""
        with self._lock:
            self._grads.clear()

class ThreadLocalEngine:
    """Thread-local backward engine."""

    _local = threading.local()

    @classmethod
    def get_engine(cls) -> BackwardEngine:
        if not hasattr(cls._local, 'engine'):
            cls._local.engine = BackwardEngine()
        return cls._local.engine
```

## Implementation Phases

### Phase 1: Core Tensor (Weeks 1-3)
- [ ] Tensor class with data and requires_grad
- [ ] Basic arithmetic operations
- [ ] Shape operations
- [ ] NumPy interoperability
- [ ] Unit tests

### Phase 2: Function and Graph (Weeks 4-6)
- [ ] Function base class
- [ ] Forward/backward interface
- [ ] save_for_backward
- [ ] All core operations
- [ ] Graph construction

### Phase 3: Backward Engine (Weeks 7-9)
- [ ] Topological sort
- [ ] Gradient accumulation
- [ ] Leaf tensor handling
- [ ] backward() method
- [ ] autograd.grad()

### Phase 4: Activation and Loss (Weeks 10-11)
- [ ] Activation functions
- [ ] Loss functions (MSE, CrossEntropy)
- [ ] Broadcasting gradients
- [ ] Reduction operations
- [ ] Higher-order ops

### Phase 5: Memory Optimization (Weeks 12-14)
- [ ] Gradient checkpointing
- [ ] Memory profiling
- [ ] In-place operations
- [ ] Gradient accumulation
- [ ] Hooks system

### Phase 6: Enterprise Features (Weeks 15-17)
- [ ] Mixed precision (GradScaler)
- [ ] Thread safety
- [ ] Custom functions
- [ ] no_grad/enable_grad
- [ ] Performance optimization

### Phase 7: Advanced (Weeks 18-20)
- [ ] VJP/JVP (stretch)
- [ ] Forward-mode autodiff (stretch)
- [ ] Higher-order derivatives
- [ ] Documentation
- [ ] Benchmarks

## Testing Strategy

### Unit Tests
```python
import pytest
import numpy as np

class TestTensorOps:
    def test_add_backward(self):
        """Test addition gradient."""
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a + b
        c.sum().backward()

        assert np.allclose(a.grad, [1, 1, 1])
        assert np.allclose(b.grad, [1, 1, 1])

    def test_mul_backward(self):
        """Test multiplication gradient."""
        a = Tensor([1, 2, 3], requires_grad=True)
        b = Tensor([4, 5, 6], requires_grad=True)
        c = a * b
        c.sum().backward()

        assert np.allclose(a.grad, [4, 5, 6])
        assert np.allclose(b.grad, [1, 2, 3])

    def test_matmul_backward(self):
        """Test matrix multiplication gradient."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 5), requires_grad=True)
        c = a @ b
        c.sum().backward()

        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4, 5)

    def test_broadcast_backward(self):
        """Test gradient reduction for broadcasting."""
        a = Tensor([[1, 2, 3]], requires_grad=True)  # (1, 3)
        b = Tensor([[1], [2]], requires_grad=True)    # (2, 1)
        c = a + b  # (2, 3)
        c.sum().backward()

        assert a.grad.shape == (1, 3)
        assert b.grad.shape == (2, 1)

class TestBackward:
    def test_chain_rule(self):
        """Test chain rule through multiple operations."""
        x = Tensor([2.0], requires_grad=True)
        y = x * x * x  # y = x^3
        y.backward()

        # dy/dx = 3x^2 = 12
        assert np.allclose(x.grad, [12.0])

    def test_multiple_paths(self):
        """Test gradient accumulation through multiple paths."""
        x = Tensor([2.0], requires_grad=True)
        y = x + x  # Two paths
        y.backward()

        assert np.allclose(x.grad, [2.0])

class TestCheckpoint:
    def test_checkpoint_correctness(self):
        """Test that checkpointing gives correct gradients."""
        def fn(x):
            return x.relu().sum()

        x = Tensor(np.random.randn(10), requires_grad=True)

        # Without checkpoint
        y1 = fn(x)
        y1.backward()
        grad1 = x.grad.copy()

        x.grad = None

        # With checkpoint
        y2 = checkpoint(fn, x)
        y2.backward()
        grad2 = x.grad

        assert np.allclose(grad1, grad2)
```

### Numerical Gradient Check
```python
def numerical_gradient(fn, inputs, eps=1e-5):
    """Compute numerical gradient for testing."""
    grads = []

    for i, inp in enumerate(inputs):
        grad = np.zeros_like(inp.data)

        for idx in np.ndindex(inp.shape):
            # f(x + eps)
            inp.data[idx] += eps
            y_plus = fn(*inputs)
            if isinstance(y_plus, Tensor):
                y_plus = y_plus.data.sum()

            # f(x - eps)
            inp.data[idx] -= 2 * eps
            y_minus = fn(*inputs)
            if isinstance(y_minus, Tensor):
                y_minus = y_minus.data.sum()

            # Restore
            inp.data[idx] += eps

            grad[idx] = (y_plus - y_minus) / (2 * eps)

        grads.append(grad)

    return grads

def check_gradient(fn, inputs, atol=1e-4, rtol=1e-3):
    """Check analytical gradient against numerical."""
    # Compute analytical gradient
    output = fn(*inputs)
    output.sum().backward()
    analytical = [inp.grad for inp in inputs]

    # Compute numerical gradient
    numerical = numerical_gradient(fn, inputs)

    # Compare
    for ana, num in zip(analytical, numerical):
        if ana is not None:
            assert np.allclose(ana, num, atol=atol, rtol=rtol), \
                f"Gradient mismatch: {ana} vs {num}"
```

## Performance Targets

| Operation | Size | Target Time | Notes |
|-----------|------|-------------|-------|
| Forward matmul | 1024x1024 | 1 ms | Using NumPy |
| Backward matmul | 1024x1024 | 2 ms | |
| Full MLP forward+backward | 1M params | 50 ms | |
| Gradient checkpointing | - | 2x time, 0.5x memory | |

## Dependencies

- Python 3.8+
- NumPy

## References

- PyTorch Autograd
- JAX Autodiff
- TensorFlow GradientTape
- Automatic Differentiation in Machine Learning: a Survey
