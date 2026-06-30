"""Core tensor implementation with dynamic graph support."""

import numpy as np
from typing import Any, List, Optional, Tuple, Union, Callable
from contextlib import contextmanager
import threading

# Thread-local gradient state
_grad_state = threading.local()


def _get_grad_enabled() -> bool:
    """Get current gradient enabled state."""
    if not hasattr(_grad_state, 'enabled'):
        _grad_state.enabled = True
    return _grad_state.enabled


def _set_grad_enabled(enabled: bool):
    """Set gradient enabled state."""
    _grad_state.enabled = enabled


def is_grad_enabled() -> bool:
    """Check if gradients are enabled."""
    return _get_grad_enabled()


def set_grad_enabled(mode: bool) -> bool:
    """Set gradient mode and return previous state."""
    prev = _get_grad_enabled()
    _set_grad_enabled(mode)
    return prev


@contextmanager
def no_grad():
    """Context manager to disable gradient computation."""
    prev = set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(prev)


@contextmanager
def enable_grad():
    """Context manager to enable gradient computation."""
    prev = set_grad_enabled(True)
    try:
        yield
    finally:
        set_grad_enabled(prev)


class Tensor:
    """
    Multi-dimensional array with automatic differentiation.

    Features:
    - NumPy-backed storage
    - Dynamic graph construction
    - Automatic gradient computation
    - Broadcasting support
    """

    def __init__(
        self,
        data: Union[np.ndarray, list, float, int],
        requires_grad: bool = False,
        dtype: np.dtype = None
    ):
        if isinstance(data, Tensor):
            data = data.data
        if isinstance(data, (list, float, int)):
            data = np.array(data)

        self.data = data.astype(dtype) if dtype else data.astype(np.float32)
        self.requires_grad = requires_grad and is_grad_enabled()
        self.grad: Optional[np.ndarray] = None

        # Graph construction
        self._grad_fn: Optional['GradFunction'] = None
        self._is_leaf = True
        self._version = 0
        self._saved_tensors: List['Tensor'] = []

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def grad_fn(self) -> Optional['GradFunction']:
        return self._grad_fn

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    def detach(self) -> 'Tensor':
        """Return a tensor detached from computation graph."""
        result = Tensor(self.data.copy(), requires_grad=False)
        return result

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        if self.requires_grad:
            raise RuntimeError("Cannot convert tensor with grad to numpy. Use detach().")
        return self.data

    def item(self) -> float:
        """Get scalar value."""
        return self.data.item()

    def zero_grad(self):
        """Zero the gradient."""
        self.grad = None

    def backward(self, grad_output: Optional[np.ndarray] = None):
        """Compute gradients via backpropagation."""
        if not self.requires_grad:
            return

        if grad_output is None:
            if self.size != 1:
                raise RuntimeError("grad_output required for non-scalar tensors")
            grad_output = np.ones_like(self.data)

        # Accumulate gradient
        if self.grad is None:
            self.grad = grad_output.copy()
        else:
            self.grad = self.grad + grad_output

        # Propagate to inputs
        if self._grad_fn is not None:
            self._grad_fn.backward(grad_output)

    def _set_grad_fn(self, grad_fn: 'GradFunction'):
        """Set gradient function."""
        self._grad_fn = grad_fn
        self._is_leaf = False

    def retain_grad(self):
        """Retain gradient for non-leaf tensor."""
        self._retain_grad = True

    def clone(self) -> 'Tensor':
        """Create a copy of the tensor."""
        result = Tensor(self.data.copy(), requires_grad=self.requires_grad)
        if self.requires_grad:
            grad_fn = CloneBackward([self])
            result._set_grad_fn(grad_fn)
        return result

    def contiguous(self) -> 'Tensor':
        """Return contiguous tensor."""
        if self.data.flags['C_CONTIGUOUS']:
            return self
        result = Tensor(np.ascontiguousarray(self.data), requires_grad=self.requires_grad)
        if self.requires_grad:
            result._set_grad_fn(CloneBackward([self]))
        return result

    def view(self, *shape) -> 'Tensor':
        """Reshape tensor."""
        result = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)
        if self.requires_grad:
            grad_fn = ViewBackward([self], self.shape)
            result._set_grad_fn(grad_fn)
        return result

    def reshape(self, *shape) -> 'Tensor':
        """Reshape tensor."""
        return self.view(*shape)

    def transpose(self, dim0: int, dim1: int) -> 'Tensor':
        """Transpose dimensions."""
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        result = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad)
        if self.requires_grad:
            grad_fn = TransposeBackward([self], dim0, dim1)
            result._set_grad_fn(grad_fn)
        return result

    @property
    def T(self) -> 'Tensor':
        """Transpose."""
        result = Tensor(self.data.T, requires_grad=self.requires_grad)
        if self.requires_grad:
            grad_fn = TransposeTBackward([self])
            result._set_grad_fn(grad_fn)
        return result

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        """Sum elements."""
        result = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad
        )
        if self.requires_grad:
            grad_fn = SumBackward([self], axis, keepdims, self.shape)
            result._set_grad_fn(grad_fn)
        return result

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        """Mean of elements."""
        result = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad
        )
        if self.requires_grad:
            grad_fn = MeanBackward([self], axis, keepdims, self.shape)
            result._set_grad_fn(grad_fn)
        return result

    def max(self, axis=None, keepdims=False) -> 'Tensor':
        """Maximum element."""
        result = Tensor(
            self.data.max(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad
        )
        if self.requires_grad:
            grad_fn = MaxBackward([self], axis, keepdims)
            result._set_grad_fn(grad_fn)
        return result

    def min(self, axis=None, keepdims=False) -> 'Tensor':
        """Minimum element."""
        result = Tensor(
            self.data.min(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad
        )
        if self.requires_grad:
            grad_fn = MinBackward([self], axis, keepdims)
            result._set_grad_fn(grad_fn)
        return result

    # Arithmetic operations
    def __add__(self, other) -> 'Tensor':
        return add(self, other)

    def __radd__(self, other) -> 'Tensor':
        return add(other, self)

    def __sub__(self, other) -> 'Tensor':
        return sub(self, other)

    def __rsub__(self, other) -> 'Tensor':
        return sub(other, self)

    def __mul__(self, other) -> 'Tensor':
        return mul(self, other)

    def __rmul__(self, other) -> 'Tensor':
        return mul(other, self)

    def __truediv__(self, other) -> 'Tensor':
        return div(self, other)

    def __rtruediv__(self, other) -> 'Tensor':
        return div(other, self)

    def __neg__(self) -> 'Tensor':
        return neg(self)

    def __pow__(self, other) -> 'Tensor':
        return pow(self, other)

    def __matmul__(self, other) -> 'Tensor':
        return matmul(self, other)

    def __getitem__(self, key) -> 'Tensor':
        result = Tensor(self.data[key], requires_grad=self.requires_grad)
        if self.requires_grad:
            grad_fn = GetItemBackward([self], key, self.shape)
            result._set_grad_fn(grad_fn)
        return result

    def __repr__(self) -> str:
        grad_info = ", requires_grad=True" if self.requires_grad else ""
        grad_fn_info = f", grad_fn={type(self._grad_fn).__name__}" if self._grad_fn else ""
        return f"Tensor({self.data}{grad_info}{grad_fn_info})"

    def __len__(self) -> int:
        return len(self.data)


class Parameter(Tensor):
    """A tensor that's automatically considered as requiring gradient."""

    def __init__(self, data: Union[np.ndarray, list, 'Tensor'], requires_grad: bool = True):
        if isinstance(data, Tensor):
            data = data.data
        super().__init__(data, requires_grad=requires_grad)


# Gradient functions
class GradFunction:
    """Base class for gradient functions."""

    def __init__(self, inputs: List[Tensor]):
        self.inputs = inputs
        self.saved_tensors: List[Tensor] = []

    def save_for_backward(self, *tensors):
        """Save tensors for backward pass."""
        self.saved_tensors = list(tensors)

    def backward(self, grad_output: np.ndarray):
        """Compute and propagate gradients."""
        raise NotImplementedError


class AddBackward(GradFunction):
    def backward(self, grad_output):
        for inp in self.inputs:
            if isinstance(inp, Tensor) and inp.requires_grad:
                grad = grad_output
                # Handle broadcasting
                if inp.shape != grad.shape:
                    grad = _unbroadcast(grad, inp.shape)
                inp.backward(grad)


class SubBackward(GradFunction):
    def backward(self, grad_output):
        if isinstance(self.inputs[0], Tensor) and self.inputs[0].requires_grad:
            grad = _unbroadcast(grad_output, self.inputs[0].shape)
            self.inputs[0].backward(grad)
        if isinstance(self.inputs[1], Tensor) and self.inputs[1].requires_grad:
            grad = _unbroadcast(-grad_output, self.inputs[1].shape)
            self.inputs[1].backward(grad)


class MulBackward(GradFunction):
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if isinstance(self.inputs[0], Tensor) and self.inputs[0].requires_grad:
            grad = grad_output * b
            grad = _unbroadcast(grad, self.inputs[0].shape)
            self.inputs[0].backward(grad)
        if isinstance(self.inputs[1], Tensor) and self.inputs[1].requires_grad:
            grad = grad_output * a
            grad = _unbroadcast(grad, self.inputs[1].shape)
            self.inputs[1].backward(grad)


class DivBackward(GradFunction):
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if isinstance(self.inputs[0], Tensor) and self.inputs[0].requires_grad:
            grad = grad_output / b
            grad = _unbroadcast(grad, self.inputs[0].shape)
            self.inputs[0].backward(grad)
        if isinstance(self.inputs[1], Tensor) and self.inputs[1].requires_grad:
            grad = -grad_output * a / (b ** 2)
            grad = _unbroadcast(grad, self.inputs[1].shape)
            self.inputs[1].backward(grad)


class NegBackward(GradFunction):
    def backward(self, grad_output):
        self.inputs[0].backward(-grad_output)


class PowBackward(GradFunction):
    def __init__(self, inputs, exp):
        super().__init__(inputs)
        self.exp = exp

    def backward(self, grad_output):
        x = self.saved_tensors[0]
        grad = self.exp * (x ** (self.exp - 1)) * grad_output
        self.inputs[0].backward(grad)


class MatMulBackward(GradFunction):
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if isinstance(self.inputs[0], Tensor) and self.inputs[0].requires_grad:
            grad = grad_output @ np.swapaxes(b, -2, -1)
            self.inputs[0].backward(grad)
        if isinstance(self.inputs[1], Tensor) and self.inputs[1].requires_grad:
            grad = np.swapaxes(a, -2, -1) @ grad_output
            self.inputs[1].backward(grad)


class SumBackward(GradFunction):
    def __init__(self, inputs, axis, keepdims, original_shape):
        super().__init__(inputs)
        self.axis = axis
        self.keepdims = keepdims
        self.original_shape = original_shape

    def backward(self, grad_output):
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)
        grad = np.broadcast_to(grad_output, self.original_shape)
        self.inputs[0].backward(grad)


class MeanBackward(GradFunction):
    def __init__(self, inputs, axis, keepdims, original_shape):
        super().__init__(inputs)
        self.axis = axis
        self.keepdims = keepdims
        self.original_shape = original_shape

    def backward(self, grad_output):
        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)

        if self.axis is None:
            n = np.prod(self.original_shape)
        else:
            n = self.original_shape[self.axis]

        grad = np.broadcast_to(grad_output / n, self.original_shape)
        self.inputs[0].backward(grad)


class MaxBackward(GradFunction):
    def __init__(self, inputs, axis, keepdims):
        super().__init__(inputs)
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, grad_output):
        x = self.inputs[0].data
        max_val = x.max(axis=self.axis, keepdims=True)
        mask = (x == max_val).astype(float)
        mask = mask / mask.sum(axis=self.axis, keepdims=True)

        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)

        grad = mask * grad_output
        self.inputs[0].backward(grad)


class MinBackward(GradFunction):
    def __init__(self, inputs, axis, keepdims):
        super().__init__(inputs)
        self.axis = axis
        self.keepdims = keepdims

    def backward(self, grad_output):
        x = self.inputs[0].data
        min_val = x.min(axis=self.axis, keepdims=True)
        mask = (x == min_val).astype(float)
        mask = mask / mask.sum(axis=self.axis, keepdims=True)

        if not self.keepdims and self.axis is not None:
            grad_output = np.expand_dims(grad_output, self.axis)

        grad = mask * grad_output
        self.inputs[0].backward(grad)


class CloneBackward(GradFunction):
    def backward(self, grad_output):
        self.inputs[0].backward(grad_output.copy())


class ViewBackward(GradFunction):
    def __init__(self, inputs, original_shape):
        super().__init__(inputs)
        self.original_shape = original_shape

    def backward(self, grad_output):
        grad = grad_output.reshape(self.original_shape)
        self.inputs[0].backward(grad)


class TransposeBackward(GradFunction):
    def __init__(self, inputs, dim0, dim1):
        super().__init__(inputs)
        self.dim0 = dim0
        self.dim1 = dim1

    def backward(self, grad_output):
        axes = list(range(grad_output.ndim))
        axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
        grad = grad_output.transpose(axes)
        self.inputs[0].backward(grad)


class TransposeTBackward(GradFunction):
    def backward(self, grad_output):
        self.inputs[0].backward(grad_output.T)


class GetItemBackward(GradFunction):
    def __init__(self, inputs, key, original_shape):
        super().__init__(inputs)
        self.key = key
        self.original_shape = original_shape

    def backward(self, grad_output):
        grad = np.zeros(self.original_shape)
        grad[self.key] = grad_output
        self.inputs[0].backward(grad)


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Unbroadcast gradient to match original shape."""
    if grad.shape == shape:
        return grad

    # Sum over broadcasted dimensions
    ndim_diff = grad.ndim - len(shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    # Sum over dimensions that were broadcast
    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# Tensor operations
def _ensure_tensor(x) -> Tensor:
    """Convert to tensor if needed."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


def add(a, b) -> Tensor:
    """Element-wise addition."""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)

    result = Tensor(
        a.data + b.data,
        requires_grad=(a.requires_grad or b.requires_grad) and is_grad_enabled()
    )

    if result.requires_grad:
        grad_fn = AddBackward([a, b])
        result._set_grad_fn(grad_fn)

    return result


def sub(a, b) -> Tensor:
    """Element-wise subtraction."""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)

    result = Tensor(
        a.data - b.data,
        requires_grad=(a.requires_grad or b.requires_grad) and is_grad_enabled()
    )

    if result.requires_grad:
        grad_fn = SubBackward([a, b])
        result._set_grad_fn(grad_fn)

    return result


def mul(a, b) -> Tensor:
    """Element-wise multiplication."""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)

    result = Tensor(
        a.data * b.data,
        requires_grad=(a.requires_grad or b.requires_grad) and is_grad_enabled()
    )

    if result.requires_grad:
        grad_fn = MulBackward([a, b])
        grad_fn.save_for_backward(a.data, b.data)
        result._set_grad_fn(grad_fn)

    return result


def div(a, b) -> Tensor:
    """Element-wise division."""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)

    result = Tensor(
        a.data / b.data,
        requires_grad=(a.requires_grad or b.requires_grad) and is_grad_enabled()
    )

    if result.requires_grad:
        grad_fn = DivBackward([a, b])
        grad_fn.save_for_backward(a.data, b.data)
        result._set_grad_fn(grad_fn)

    return result


def neg(a) -> Tensor:
    """Negation."""
    a = _ensure_tensor(a)

    result = Tensor(-a.data, requires_grad=a.requires_grad and is_grad_enabled())

    if result.requires_grad:
        grad_fn = NegBackward([a])
        result._set_grad_fn(grad_fn)

    return result


def pow(a, exp) -> Tensor:
    """Power operation."""
    a = _ensure_tensor(a)

    result = Tensor(
        a.data ** exp,
        requires_grad=a.requires_grad and is_grad_enabled()
    )

    if result.requires_grad:
        grad_fn = PowBackward([a], exp)
        grad_fn.save_for_backward(a.data)
        result._set_grad_fn(grad_fn)

    return result


def matmul(a, b) -> Tensor:
    """Matrix multiplication."""
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)

    result = Tensor(
        a.data @ b.data,
        requires_grad=(a.requires_grad or b.requires_grad) and is_grad_enabled()
    )

    if result.requires_grad:
        grad_fn = MatMulBackward([a, b])
        grad_fn.save_for_backward(a.data, b.data)
        result._set_grad_fn(grad_fn)

    return result
