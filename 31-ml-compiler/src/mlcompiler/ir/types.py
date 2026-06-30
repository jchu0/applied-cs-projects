"""Core types for ML compiler IR."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import numpy as np


class DType(Enum):
    """Data types supported by the IR."""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    BOOL = "bool"

    @property
    def numpy_dtype(self):
        """Get corresponding numpy dtype."""
        mapping = {
            DType.FLOAT16: np.float16,
            DType.FLOAT32: np.float32,
            DType.FLOAT64: np.float64,
            DType.INT8: np.int8,
            DType.INT16: np.int16,
            DType.INT32: np.int32,
            DType.INT64: np.int64,
            DType.UINT8: np.uint8,
            DType.BOOL: np.bool_,
        }
        return mapping[self]

    @property
    def size_bytes(self) -> int:
        """Get size in bytes."""
        sizes = {
            DType.FLOAT16: 2,
            DType.FLOAT32: 4,
            DType.FLOAT64: 8,
            DType.INT8: 1,
            DType.INT16: 2,
            DType.INT32: 4,
            DType.INT64: 8,
            DType.UINT8: 1,
            DType.BOOL: 1,
        }
        return sizes[self]


@dataclass
class TensorType:
    """Tensor type with shape and dtype."""
    shape: tuple[int, ...]
    dtype: DType = DType.FLOAT32

    @property
    def num_elements(self) -> int:
        """Get total number of elements."""
        result = 1
        for dim in self.shape:
            if dim > 0:
                result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        """Get total size in bytes."""
        return self.num_elements * self.dtype.size_bytes

    @property
    def rank(self) -> int:
        """Get tensor rank."""
        return len(self.shape)

    def __str__(self) -> str:
        shape_str = "x".join(str(d) for d in self.shape)
        return f"tensor<{shape_str}x{self.dtype.value}>"


@dataclass
class Value:
    """SSA value in the IR."""
    id: str
    type: TensorType
    name: str = ""
    defining_op: Any = None
    uses: list = field(default_factory=list)

    def __str__(self) -> str:
        if self.name:
            return f"%{self.name}"
        return f"%{self.id}"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Value):
            return self.id == other.id
        return False


@dataclass
class Constant:
    """Constant value."""
    value: np.ndarray
    type: TensorType

    @classmethod
    def from_scalar(cls, value: float, dtype: DType = DType.FLOAT32):
        """Create constant from scalar."""
        arr = np.array(value, dtype=dtype.numpy_dtype)
        tensor_type = TensorType(shape=(), dtype=dtype)
        return cls(value=arr, type=tensor_type)

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Create constant from numpy array."""
        dtype_map = {
            np.float16: DType.FLOAT16,
            np.float32: DType.FLOAT32,
            np.float64: DType.FLOAT64,
            np.int8: DType.INT8,
            np.int16: DType.INT16,
            np.int32: DType.INT32,
            np.int64: DType.INT64,
            np.uint8: DType.UINT8,
            np.bool_: DType.BOOL,
        }
        dtype = dtype_map.get(arr.dtype.type, DType.FLOAT32)
        tensor_type = TensorType(shape=arr.shape, dtype=dtype)
        return cls(value=arr, type=tensor_type)


@dataclass
class Attribute:
    """Operation attribute."""
    name: str
    value: Any

    def __str__(self) -> str:
        return f"{self.name}={self.value}"


class Layout(Enum):
    """Memory layout for tensors."""
    ROW_MAJOR = "row_major"  # C-style
    COLUMN_MAJOR = "column_major"  # Fortran-style
    BLOCKED = "blocked"


@dataclass
class MemorySpace:
    """Memory space specification."""
    name: str
    id: int
    bandwidth_gb_s: float = 0.0
    size_bytes: int = 0


# Predefined memory spaces
GLOBAL_MEMORY = MemorySpace("global", 0, bandwidth_gb_s=900.0)
SHARED_MEMORY = MemorySpace("shared", 1, bandwidth_gb_s=12000.0)
REGISTER = MemorySpace("register", 2, bandwidth_gb_s=float("inf"))
