"""Tensor with automatic differentiation support."""

import numpy as np
from typing import Tuple, List, Optional, Callable, Union, Set
from contextlib import contextmanager

# Global gradient computation state
_grad_enabled = True


@contextmanager
def no_grad():
    """Context manager to disable gradient computation."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


@contextmanager
def enable_grad():
    """Context manager to enable gradient computation."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = True
    try:
        yield
    finally:
        _grad_enabled = prev


class Tensor:
    """
    Tensor with automatic differentiation.

    Supports:
    - Forward computation
    - Backward gradient computation
    - Gradient accumulation
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, float],
        requires_grad: bool = False,
        dtype: np.dtype = np.float32
    ):
        if isinstance(data, np.ndarray):
            self.data = data.astype(dtype)
        elif isinstance(data, (list, tuple)):
            self.data = np.array(data, dtype=dtype)
        else:
            self.data = np.array([data], dtype=dtype)

        self.requires_grad = requires_grad and _grad_enabled
        self.grad: Optional[np.ndarray] = None

        # Computational graph
        self._grad_fn: Optional[Callable] = None
        self._inputs: List['Tensor'] = []
        self._is_leaf = True

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    def numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.copy()

    def item(self) -> float:
        """Get scalar value."""
        return self.data.item()

    def detach(self) -> 'Tensor':
        """Detach from computational graph."""
        return Tensor(self.data.copy(), requires_grad=False)

    def zero_grad(self):
        """Zero the gradients."""
        self.grad = None

    def backward(self, grad: Optional[np.ndarray] = None):
        """
        Compute gradients through backpropagation.

        Args:
            grad: Upstream gradient. If None, uses ones.
        """
        if not self.requires_grad:
            return

        if grad is None:
            grad = np.ones_like(self.data)

        # Accumulate gradient
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad

        # Propagate through graph
        if self._grad_fn is not None:
            input_grads = self._grad_fn(grad)
            if not isinstance(input_grads, (list, tuple)):
                input_grads = [input_grads]

            for inp, g in zip(self._inputs, input_grads):
                if inp.requires_grad and g is not None:
                    inp.backward(g)

    def _set_grad_fn(self, grad_fn: Callable, inputs: List['Tensor']):
        """Set gradient function and inputs."""
        self._grad_fn = grad_fn
        self._inputs = inputs
        self._is_leaf = False

    # Factory methods
    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        return Tensor(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        return Tensor(np.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)

    @staticmethod
    def rand(shape: Tuple[int, ...], requires_grad: bool = False) -> 'Tensor':
        return Tensor(np.random.rand(*shape), requires_grad=requires_grad)

    @staticmethod
    def eye(n: int, requires_grad: bool = False) -> 'Tensor':
        return Tensor(np.eye(n), requires_grad=requires_grad)

    @staticmethod
    def arange(start: int, stop: int = None, step: int = 1, requires_grad: bool = False) -> 'Tensor':
        if stop is None:
            return Tensor(np.arange(start), requires_grad=requires_grad)
        return Tensor(np.arange(start, stop, step), requires_grad=requires_grad)

    # Arithmetic operations
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        from ..ops import add
        return add(self, other)

    def __radd__(self, other: Union['Tensor', float]) -> 'Tensor':
        from ..ops import add
        return add(other, self)

    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        from ..ops import sub
        return sub(self, other)

    def __rsub__(self, other: Union['Tensor', float]) -> 'Tensor':
        from ..ops import sub
        return sub(other, self)

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        from ..ops import mul
        return mul(self, other)

    def __rmul__(self, other: Union['Tensor', float]) -> 'Tensor':
        from ..ops import mul
        return mul(other, self)

    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        from ..ops import div
        return div(self, other)

    def __rtruediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        from ..ops import div
        return div(other, self)

    def __neg__(self) -> 'Tensor':
        from ..ops import neg
        return neg(self)

    def __pow__(self, power: float) -> 'Tensor':
        result = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad
        )

        if self.requires_grad:
            def grad_fn(g):
                return power * (self.data ** (power - 1)) * g
            result._set_grad_fn(grad_fn, [self])

        return result

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        from ..ops import matmul
        return matmul(self, other)

    def __getitem__(self, idx) -> 'Tensor':
        result = Tensor(self.data[idx], requires_grad=self.requires_grad)

        if self.requires_grad:
            def grad_fn(g):
                full_grad = np.zeros_like(self.data)
                full_grad[idx] = g
                return full_grad
            result._set_grad_fn(grad_fn, [self])

        return result

    # Reduction operations
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        from ..ops import sum as tensor_sum
        return tensor_sum(self, axis, keepdims)

    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        from ..ops import mean
        return mean(self, axis, keepdims)

    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> 'Tensor':
        from ..ops import max as tensor_max
        return tensor_max(self, axis, keepdims)

    # Shape operations
    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        from ..ops import reshape
        return reshape(self, shape)

    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'Tensor':
        from ..ops import transpose
        return transpose(self, axes)

    @property
    def T(self) -> 'Tensor':
        return self.transpose()

    def flatten(self) -> 'Tensor':
        return self.reshape((-1,))

    def squeeze(self, axis: Optional[int] = None) -> 'Tensor':
        result = Tensor(
            np.squeeze(self.data, axis=axis),
            requires_grad=self.requires_grad
        )

        if self.requires_grad:
            original_shape = self.shape

            def grad_fn(g):
                return g.reshape(original_shape)
            result._set_grad_fn(grad_fn, [self])

        return result

    def unsqueeze(self, axis: int) -> 'Tensor':
        result = Tensor(
            np.expand_dims(self.data, axis=axis),
            requires_grad=self.requires_grad
        )

        if self.requires_grad:
            def grad_fn(g):
                return np.squeeze(g, axis=axis)
            result._set_grad_fn(grad_fn, [self])

        return result

    def __repr__(self) -> str:
        grad_info = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data}{grad_info})"


def grad(func: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """
    Create a function that computes gradients.

    Args:
        func: Function to differentiate
        argnums: Argument indices to differentiate

    Returns:
        Function that returns gradients
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def grad_func(*args, **kwargs):
        # Create tensors with requires_grad
        new_args = list(args)
        for i in argnums:
            if isinstance(args[i], Tensor):
                new_args[i] = Tensor(args[i].data, requires_grad=True)
            else:
                new_args[i] = Tensor(args[i], requires_grad=True)

        # Forward pass
        result = func(*new_args, **kwargs)

        # Backward pass
        if isinstance(result, Tensor):
            result.backward()

        # Collect gradients
        grads = []
        for i in argnums:
            grads.append(new_args[i].grad)

        if len(grads) == 1:
            return grads[0]
        return tuple(grads)

    return grad_func


def value_and_grad(func: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """
    Create a function that returns both value and gradients.

    Args:
        func: Function to differentiate
        argnums: Argument indices to differentiate

    Returns:
        Function that returns (value, gradients)
    """
    grad_func = grad(func, argnums)

    def value_and_grad_func(*args, **kwargs):
        # Forward pass
        result = func(*args, **kwargs)

        # Gradient computation
        grads = grad_func(*args, **kwargs)

        if isinstance(result, Tensor):
            return result.numpy(), grads
        return result, grads

    return value_and_grad_func
