"""Differentiable operations with backward functions."""

import numpy as np
from typing import Tuple, Optional, Union, List

from ..core.tensor import Tensor


def _ensure_tensor(x: Union[Tensor, float, np.ndarray]) -> Tensor:
    """Convert to Tensor if needed."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


# Arithmetic operations

def add(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise addition."""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)

    result = Tensor(
        a.data + b.data,
        requires_grad=a.requires_grad or b.requires_grad
    )

    if result.requires_grad:
        def grad_fn(g):
            # Handle broadcasting
            ga = g
            gb = g

            # Sum over broadcasted dimensions
            while ga.ndim > a.ndim:
                ga = ga.sum(axis=0)
            while gb.ndim > b.ndim:
                gb = gb.sum(axis=0)

            for i, (sa, sb) in enumerate(zip(a.shape, b.shape)):
                if sa == 1 and sb != 1:
                    ga = ga.sum(axis=i, keepdims=True)
                if sb == 1 and sa != 1:
                    gb = gb.sum(axis=i, keepdims=True)

            return ga, gb

        result._set_grad_fn(grad_fn, [a, b])

    return result


def sub(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise subtraction."""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)

    result = Tensor(
        a.data - b.data,
        requires_grad=a.requires_grad or b.requires_grad
    )

    if result.requires_grad:
        def grad_fn(g):
            ga = g
            gb = -g

            while ga.ndim > a.ndim:
                ga = ga.sum(axis=0)
            while gb.ndim > b.ndim:
                gb = gb.sum(axis=0)

            return ga, gb

        result._set_grad_fn(grad_fn, [a, b])

    return result


def mul(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise multiplication."""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)

    result = Tensor(
        a.data * b.data,
        requires_grad=a.requires_grad or b.requires_grad
    )

    if result.requires_grad:
        def grad_fn(g):
            ga = g * b.data
            gb = g * a.data

            while ga.ndim > a.ndim:
                ga = ga.sum(axis=0)
            while gb.ndim > b.ndim:
                gb = gb.sum(axis=0)

            return ga, gb

        result._set_grad_fn(grad_fn, [a, b])

    return result


def div(a: Union[Tensor, float], b: Union[Tensor, float]) -> Tensor:
    """Element-wise division."""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)

    result = Tensor(
        a.data / b.data,
        requires_grad=a.requires_grad or b.requires_grad
    )

    if result.requires_grad:
        def grad_fn(g):
            ga = g / b.data
            gb = -g * a.data / (b.data ** 2)

            while ga.ndim > a.ndim:
                ga = ga.sum(axis=0)
            while gb.ndim > b.ndim:
                gb = gb.sum(axis=0)

            return ga, gb

        result._set_grad_fn(grad_fn, [a, b])

    return result


def neg(a: Tensor) -> Tensor:
    """Negation."""
    result = Tensor(-a.data, requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            return -g
        result._set_grad_fn(grad_fn, [a])

    return result


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication."""
    result = Tensor(
        a.data @ b.data,
        requires_grad=a.requires_grad or b.requires_grad
    )

    if result.requires_grad:
        def grad_fn(g):
            if a.ndim == 1 and b.ndim == 1:
                ga = g * b.data
                gb = g * a.data
            elif a.ndim == 1:
                ga = g @ b.data.T
                gb = np.outer(a.data, g)
            elif b.ndim == 1:
                ga = np.outer(g, b.data)
                gb = a.data.T @ g
            else:
                ga = g @ np.swapaxes(b.data, -2, -1)
                gb = np.swapaxes(a.data, -2, -1) @ g

            return ga, gb

        result._set_grad_fn(grad_fn, [a, b])

    return result


# Math functions

def exp(a: Tensor) -> Tensor:
    """Exponential."""
    result_data = np.exp(a.data)
    result = Tensor(result_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            return g * result_data
        result._set_grad_fn(grad_fn, [a])

    return result


def log(a: Tensor) -> Tensor:
    """Natural logarithm."""
    result = Tensor(np.log(a.data), requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            return g / a.data
        result._set_grad_fn(grad_fn, [a])

    return result


def sqrt(a: Tensor) -> Tensor:
    """Square root."""
    result_data = np.sqrt(a.data)
    result = Tensor(result_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            return g / (2 * result_data)
        result._set_grad_fn(grad_fn, [a])

    return result


def sin(a: Tensor) -> Tensor:
    """Sine."""
    result = Tensor(np.sin(a.data), requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            return g * np.cos(a.data)
        result._set_grad_fn(grad_fn, [a])

    return result


def cos(a: Tensor) -> Tensor:
    """Cosine."""
    result = Tensor(np.cos(a.data), requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            return -g * np.sin(a.data)
        result._set_grad_fn(grad_fn, [a])

    return result


def tanh(a: Tensor) -> Tensor:
    """Hyperbolic tangent."""
    result_data = np.tanh(a.data)
    result = Tensor(result_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            return g * (1 - result_data ** 2)
        result._set_grad_fn(grad_fn, [a])

    return result


def sigmoid(a: Tensor) -> Tensor:
    """Sigmoid activation."""
    result_data = 1 / (1 + np.exp(-a.data))
    result = Tensor(result_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            return g * result_data * (1 - result_data)
        result._set_grad_fn(grad_fn, [a])

    return result


def relu(a: Tensor) -> Tensor:
    """ReLU activation."""
    result = Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            return g * (a.data > 0)
        result._set_grad_fn(grad_fn, [a])

    return result


def softmax(a: Tensor, axis: int = -1) -> Tensor:
    """Softmax activation."""
    # Numerical stability
    shifted = a.data - np.max(a.data, axis=axis, keepdims=True)
    exp_data = np.exp(shifted)
    result_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)

    result = Tensor(result_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            # Jacobian-vector product
            s = result_data
            return s * (g - np.sum(g * s, axis=axis, keepdims=True))
        result._set_grad_fn(grad_fn, [a])

    return result


# Reduction operations

def sum(a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Sum reduction."""
    result = Tensor(
        np.sum(a.data, axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad
    )

    if a.requires_grad:
        def grad_fn(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            return np.broadcast_to(g, a.shape).copy()
        result._set_grad_fn(grad_fn, [a])

    return result


def mean(a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Mean reduction."""
    result = Tensor(
        np.mean(a.data, axis=axis, keepdims=keepdims),
        requires_grad=a.requires_grad
    )

    if a.requires_grad:
        if axis is None:
            n = a.size
        else:
            n = a.shape[axis]

        def grad_fn(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis=axis)
            return np.broadcast_to(g / n, a.shape).copy()
        result._set_grad_fn(grad_fn, [a])

    return result


def max(a: Tensor, axis: Optional[int] = None, keepdims: bool = False) -> Tensor:
    """Max reduction."""
    result_data = np.max(a.data, axis=axis, keepdims=keepdims)
    result = Tensor(result_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            if axis is not None and not keepdims:
                expanded_result = np.expand_dims(result_data, axis=axis)
                expanded_g = np.expand_dims(g, axis=axis)
            else:
                expanded_result = result_data
                expanded_g = g

            mask = (a.data == expanded_result).astype(np.float32)
            mask = mask / mask.sum(axis=axis, keepdims=True)
            return expanded_g * mask
        result._set_grad_fn(grad_fn, [a])

    return result


# Shape operations

def reshape(a: Tensor, shape: Tuple[int, ...]) -> Tensor:
    """Reshape tensor."""
    result = Tensor(
        a.data.reshape(shape),
        requires_grad=a.requires_grad
    )

    if a.requires_grad:
        original_shape = a.shape

        def grad_fn(g):
            return g.reshape(original_shape)
        result._set_grad_fn(grad_fn, [a])

    return result


def transpose(a: Tensor, axes: Optional[Tuple[int, ...]] = None) -> Tensor:
    """Transpose tensor."""
    if axes is None:
        result_data = a.data.T
    else:
        result_data = np.transpose(a.data, axes)

    result = Tensor(result_data, requires_grad=a.requires_grad)

    if a.requires_grad:
        def grad_fn(g):
            if axes is None:
                return g.T
            # Invert permutation
            inv_axes = [0] * len(axes)
            for i, ax in enumerate(axes):
                inv_axes[ax] = i
            return np.transpose(g, inv_axes)
        result._set_grad_fn(grad_fn, [a])

    return result


def concat(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Concatenate tensors."""
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)

    result = Tensor(data, requires_grad=requires_grad)

    if requires_grad:
        sizes = [t.shape[axis] for t in tensors]

        def grad_fn(g):
            grads = []
            start = 0
            for size in sizes:
                idx = [slice(None)] * g.ndim
                idx[axis] = slice(start, start + size)
                grads.append(g[tuple(idx)])
                start += size
            return grads
        result._set_grad_fn(grad_fn, tensors)

    return result
