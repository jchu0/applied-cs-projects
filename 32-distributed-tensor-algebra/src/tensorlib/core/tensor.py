"""Core lazy tensor implementation."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple, Union
import numpy as np


class DeviceType(Enum):
    """Device types for computation."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"


@dataclass
class Device:
    """Represents a compute device."""
    device_type: DeviceType
    device_id: int
    memory_bytes: int = 16 * 1024**3

    def __str__(self):
        return f"{self.device_type.value}:{self.device_id}"

    def __hash__(self):
        return hash((self.device_type, self.device_id))


@dataclass
class ShapeSpec:
    """Shape specification with optional dynamic dimensions."""
    dims: Tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def numel(self) -> int:
        result = 1
        for d in self.dims:
            if d < 0:
                return -1
            result *= d
        return result

    def __getitem__(self, idx):
        return self.dims[idx]

    def __len__(self):
        return len(self.dims)


@dataclass
class TensorSpec:
    """Type specification for tensors."""
    shape: ShapeSpec
    dtype: np.dtype
    device: Optional[Device] = None
    sharding: Optional[Any] = None


class Tracer:
    """Abstract tracer for symbolic execution."""
    _counter = 0

    def __init__(self, spec: TensorSpec, name: str = None):
        self.spec = spec
        self.name = name or f"tracer_{Tracer._counter}"
        Tracer._counter += 1
        self._aval = spec

    @property
    def shape(self):
        return self.spec.shape.dims

    @property
    def dtype(self):
        return self.spec.dtype

    def __repr__(self):
        return f"Tracer({self.name}, shape={self.shape}, dtype={self.dtype})"


class LazyTensor:
    """Lazy tensor that defers computation until explicitly evaluated."""

    def __init__(
        self,
        value: Optional[np.ndarray] = None,
        tracer: Optional[Tracer] = None,
        producer: Optional[Any] = None
    ):
        self._value = value
        self._tracer = tracer
        self._producer = producer
        self._materialized = value is not None

        if value is not None:
            self._shape = value.shape
            self._dtype = value.dtype
        elif tracer is not None:
            self._shape = tracer.shape
            self._dtype = tracer.dtype
        else:
            raise ValueError("Must provide either value or tracer")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return int(np.prod(self._shape))

    @property
    def is_materialized(self) -> bool:
        return self._materialized

    def materialize(self) -> np.ndarray:
        """Force evaluation and return concrete value."""
        if self._materialized:
            return self._value

        if self._producer:
            self._value = self._producer.evaluate()
            self._materialized = True
            return self._value

        raise RuntimeError("Cannot materialize tensor without value or producer")

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return self.materialize()

    # Arithmetic operations
    def __add__(self, other):
        from .primitives import add
        return add(self, other)

    def __radd__(self, other):
        from .primitives import add
        return add(other, self)

    def __mul__(self, other):
        from .primitives import mul
        return mul(self, other)

    def __rmul__(self, other):
        from .primitives import mul
        return mul(other, self)

    def __matmul__(self, other):
        from .primitives import matmul
        return matmul(self, other)

    def __neg__(self):
        from .primitives import neg
        return neg(self)

    def __sub__(self, other):
        from .primitives import sub
        return sub(self, other)

    def __rsub__(self, other):
        from .primitives import sub
        return sub(other, self)

    def __truediv__(self, other):
        from .primitives import div
        return div(self, other)

    def __rtruediv__(self, other):
        from .primitives import div
        return div(other, self)

    def __pow__(self, other):
        from .primitives import power
        return power(self, other)

    # Reductions
    def sum(self, axis=None, keepdims=False):
        from .primitives import reduce_sum
        return reduce_sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        from .primitives import reduce_mean
        return reduce_mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        from .primitives import reduce_max
        return reduce_max(self, axis, keepdims)

    def min(self, axis=None, keepdims=False):
        from .primitives import reduce_min
        return reduce_min(self, axis, keepdims)

    # Shape operations
    def reshape(self, *shape):
        from .primitives import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return reshape(self, shape)

    def transpose(self, *axes):
        from .primitives import transpose
        return transpose(self, axes if axes else None)

    @property
    def T(self):
        return self.transpose()

    def flatten(self):
        return self.reshape(-1)

    def squeeze(self, axis=None):
        from .primitives import squeeze
        return squeeze(self, axis)

    def __repr__(self):
        if self._materialized:
            return f"LazyTensor({self._value})"
        return f"LazyTensor(shape={self._shape}, dtype={self._dtype}, lazy=True)"


# Factory functions
def array(data, dtype=None) -> LazyTensor:
    """Create a LazyTensor from array-like data."""
    if isinstance(data, LazyTensor):
        return data
    arr = np.array(data, dtype=dtype)
    return LazyTensor(value=arr)


def zeros(shape, dtype=np.float32) -> LazyTensor:
    """Create tensor of zeros."""
    return LazyTensor(value=np.zeros(shape, dtype=dtype))


def ones(shape, dtype=np.float32) -> LazyTensor:
    """Create tensor of ones."""
    return LazyTensor(value=np.ones(shape, dtype=dtype))


def randn(*shape, dtype=np.float32) -> LazyTensor:
    """Create tensor with random normal values."""
    return LazyTensor(value=np.random.randn(*shape).astype(dtype))


def rand(*shape, dtype=np.float32) -> LazyTensor:
    """Create tensor with random uniform values."""
    return LazyTensor(value=np.random.rand(*shape).astype(dtype))


def arange(start, stop=None, step=1, dtype=np.float32) -> LazyTensor:
    """Create tensor with evenly spaced values."""
    if stop is None:
        stop = start
        start = 0
    return LazyTensor(value=np.arange(start, stop, step, dtype=dtype))


def linspace(start, stop, num, dtype=np.float32) -> LazyTensor:
    """Create tensor with linearly spaced values."""
    return LazyTensor(value=np.linspace(start, stop, num, dtype=dtype))


def eye(n, m=None, dtype=np.float32) -> LazyTensor:
    """Create identity matrix."""
    return LazyTensor(value=np.eye(n, m, dtype=dtype))


def full(shape, fill_value, dtype=np.float32) -> LazyTensor:
    """Create tensor filled with value."""
    return LazyTensor(value=np.full(shape, fill_value, dtype=dtype))
