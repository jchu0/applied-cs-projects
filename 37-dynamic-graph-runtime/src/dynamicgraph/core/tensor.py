"""Symbolic tensor representation for dynamic graph execution."""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid


@dataclass
class TensorMetadata:
    """Metadata for symbolic tensors."""
    dtype: Optional[np.dtype] = None
    shape: Optional[Tuple[int, ...]] = None
    device: str = "cpu"
    requires_grad: bool = False
    is_parameter: bool = False
    is_buffer: bool = False
    name: Optional[str] = None


class SymbolicTensor:
    """Represents a tensor symbolically during graph construction."""

    def __init__(
        self,
        node_id: Optional[str] = None,
        metadata: Optional[TensorMetadata] = None,
        concrete_value: Optional[np.ndarray] = None,
        source: Optional[str] = None
    ):
        self.node_id = node_id or f"tensor_{uuid.uuid4().hex[:8]}"
        self.metadata = metadata or TensorMetadata()
        self.concrete_value = concrete_value
        self.source = source  # e.g., "arg0", "getattr.weight"
        self._grad: Optional['SymbolicTensor'] = None
        self._version = 0

    @property
    def dtype(self) -> Optional[np.dtype]:
        """Get tensor dtype."""
        if self.metadata.dtype:
            return self.metadata.dtype
        if self.concrete_value is not None:
            return self.concrete_value.dtype
        return None

    @property
    def shape(self) -> Optional[Tuple[int, ...]]:
        """Get tensor shape."""
        if self.metadata.shape:
            return self.metadata.shape
        if self.concrete_value is not None:
            return self.concrete_value.shape
        return None

    @property
    def device(self) -> str:
        """Get tensor device."""
        return self.metadata.device

    @property
    def requires_grad(self) -> bool:
        """Check if tensor requires gradient."""
        return self.metadata.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set gradient requirement."""
        self.metadata.requires_grad = value

    @property
    def grad(self) -> Optional['SymbolicTensor']:
        """Get gradient tensor."""
        return self._grad

    @grad.setter
    def grad(self, value: Optional['SymbolicTensor']) -> None:
        """Set gradient tensor."""
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require gradient")
        self._grad = value

    def detach(self) -> 'SymbolicTensor':
        """Create a detached copy."""
        return SymbolicTensor(
            node_id=f"{self.node_id}_detached",
            metadata=TensorMetadata(
                dtype=self.metadata.dtype,
                shape=self.metadata.shape,
                device=self.metadata.device,
                requires_grad=False,
                is_parameter=False,
                is_buffer=self.metadata.is_buffer,
                name=self.metadata.name
            ),
            concrete_value=self.concrete_value.copy() if self.concrete_value is not None else None,
            source=f"{self.source}.detach()" if self.source else None
        )

    def clone(self) -> 'SymbolicTensor':
        """Create a cloned copy."""
        return SymbolicTensor(
            node_id=f"{self.node_id}_clone",
            metadata=TensorMetadata(
                dtype=self.metadata.dtype,
                shape=self.metadata.shape,
                device=self.metadata.device,
                requires_grad=self.metadata.requires_grad,
                is_parameter=self.metadata.is_parameter,
                is_buffer=self.metadata.is_buffer,
                name=self.metadata.name
            ),
            concrete_value=self.concrete_value.copy() if self.concrete_value is not None else None,
            source=f"{self.source}.clone()" if self.source else None
        )

    def to(self, device: str) -> 'SymbolicTensor':
        """Move tensor to device."""
        result = self.clone()
        result.metadata.device = device
        return result

    def is_concrete(self) -> bool:
        """Check if tensor has concrete value."""
        return self.concrete_value is not None

    def is_symbolic(self) -> bool:
        """Check if tensor is purely symbolic."""
        return self.concrete_value is None

    def numel(self) -> Optional[int]:
        """Get number of elements."""
        if self.shape:
            return int(np.prod(self.shape))
        return None

    def ndim(self) -> Optional[int]:
        """Get number of dimensions."""
        if self.shape:
            return len(self.shape)
        return None

    def size(self, dim: Optional[int] = None) -> Union[Tuple[int, ...], int, None]:
        """Get size along dimension."""
        if self.shape is None:
            return None
        if dim is None:
            return self.shape
        if -len(self.shape) <= dim < len(self.shape):
            return self.shape[dim]
        raise IndexError(f"Dimension {dim} out of range for {len(self.shape)}-D tensor")

    def __repr__(self) -> str:
        parts = [f"SymbolicTensor(id={self.node_id}"]
        if self.shape:
            parts.append(f"shape={self.shape}")
        if self.dtype:
            parts.append(f"dtype={self.dtype}")
        if self.metadata.device != "cpu":
            parts.append(f"device={self.metadata.device}")
        if self.requires_grad:
            parts.append("requires_grad=True")
        if self.is_concrete():
            parts.append("concrete=True")
        return ", ".join(parts) + ")"

    # Operator overloads for graph construction
    def __add__(self, other: Union['SymbolicTensor', float]) -> 'SymbolicTensor':
        """Addition operator."""
        return self._create_binop_result("add", other)

    def __radd__(self, other: Union['SymbolicTensor', float]) -> 'SymbolicTensor':
        """Right addition."""
        return self._create_binop_result("add", other, reverse=True)

    def __sub__(self, other: Union['SymbolicTensor', float]) -> 'SymbolicTensor':
        """Subtraction operator."""
        return self._create_binop_result("sub", other)

    def __rsub__(self, other: Union['SymbolicTensor', float]) -> 'SymbolicTensor':
        """Right subtraction."""
        return self._create_binop_result("sub", other, reverse=True)

    def __mul__(self, other: Union['SymbolicTensor', float]) -> 'SymbolicTensor':
        """Multiplication operator."""
        return self._create_binop_result("mul", other)

    def __rmul__(self, other: Union['SymbolicTensor', float]) -> 'SymbolicTensor':
        """Right multiplication."""
        return self._create_binop_result("mul", other, reverse=True)

    def __truediv__(self, other: Union['SymbolicTensor', float]) -> 'SymbolicTensor':
        """Division operator."""
        return self._create_binop_result("div", other)

    def __rtruediv__(self, other: Union['SymbolicTensor', float]) -> 'SymbolicTensor':
        """Right division."""
        return self._create_binop_result("div", other, reverse=True)

    def __matmul__(self, other: 'SymbolicTensor') -> 'SymbolicTensor':
        """Matrix multiplication."""
        # Matrix multiplication has special shape rules
        if not isinstance(other, SymbolicTensor):
            raise TypeError("Matrix multiplication requires two tensors")

        # Calculate output shape for matrix multiplication
        s1, s2 = self.shape, other.shape
        if len(s1) < 1 or len(s2) < 1:
            raise ValueError("Matrix multiplication requires at least 1D tensors")

        # For 2D tensors: (m, n) @ (n, p) -> (m, p)
        if len(s1) >= 2 and len(s2) >= 2:
            if s1[-1] != s2[-2]:
                raise ValueError(f"Matrix dimensions don't match: {s1} @ {s2}")
            result_shape = s1[:-1] + (s2[-1],)
        elif len(s1) == 1 and len(s2) == 2:
            # (n,) @ (n, p) -> (p,)
            if s1[-1] != s2[-2]:
                raise ValueError(f"Matrix dimensions don't match: {s1} @ {s2}")
            result_shape = (s2[-1],)
        elif len(s1) == 2 and len(s2) == 1:
            # (m, n) @ (n,) -> (m,)
            if s1[-1] != s2[-1]:
                raise ValueError(f"Matrix dimensions don't match: {s1} @ {s2}")
            result_shape = (s1[-2],)
        else:
            result_shape = ()

        dtype = self._promote_dtypes(self.dtype, other.dtype)
        requires_grad = self.requires_grad or other.requires_grad

        node_id = f"matmul_{uuid.uuid4().hex[:8]}"
        return SymbolicTensor(
            node_id=node_id,
            metadata=TensorMetadata(
                dtype=dtype,
                shape=result_shape,
                device=self.device,
                requires_grad=requires_grad
            ),
            source=f"{self.source}.matmul({other})" if self.source else None
        )

    def __neg__(self) -> 'SymbolicTensor':
        """Negation operator."""
        return self._create_unop_result("neg")

    def _create_binop_result(
        self,
        op: str,
        other: Union['SymbolicTensor', float],
        reverse: bool = False
    ) -> 'SymbolicTensor':
        """Create result tensor for binary operation."""
        # Determine output shape
        if isinstance(other, SymbolicTensor):
            shape = self._broadcast_shapes(self.shape, other.shape)
            dtype = self._promote_dtypes(self.dtype, other.dtype)
            requires_grad = self.requires_grad or other.requires_grad
        else:
            shape = self.shape
            dtype = self.dtype
            requires_grad = self.requires_grad

        node_id = f"{op}_{uuid.uuid4().hex[:8]}"
        return SymbolicTensor(
            node_id=node_id,
            metadata=TensorMetadata(
                dtype=dtype,
                shape=shape,
                device=self.device,
                requires_grad=requires_grad
            ),
            source=f"{self.source}.{op}({other})" if self.source else None
        )

    def _create_unop_result(self, op: str) -> 'SymbolicTensor':
        """Create result tensor for unary operation."""
        node_id = f"{op}_{uuid.uuid4().hex[:8]}"
        return SymbolicTensor(
            node_id=node_id,
            metadata=TensorMetadata(
                dtype=self.dtype,
                shape=self.shape,
                device=self.device,
                requires_grad=self.requires_grad
            ),
            source=f"{self.source}.{op}()" if self.source else None
        )

    @staticmethod
    def _broadcast_shapes(
        shape1: Optional[Tuple[int, ...]],
        shape2: Optional[Tuple[int, ...]]
    ) -> Optional[Tuple[int, ...]]:
        """Compute broadcast shape."""
        if shape1 is None or shape2 is None:
            return None

        # NumPy-style broadcasting
        s1, s2 = list(shape1), list(shape2)

        # Pad shorter shape with 1s
        while len(s1) < len(s2):
            s1.insert(0, 1)
        while len(s2) < len(s1):
            s2.insert(0, 1)

        result = []
        for d1, d2 in zip(s1, s2):
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")

        return tuple(result)

    @staticmethod
    def _promote_dtypes(
        dtype1: Optional[np.dtype],
        dtype2: Optional[np.dtype]
    ) -> Optional[np.dtype]:
        """Promote dtypes according to NumPy rules."""
        if dtype1 is None or dtype2 is None:
            return dtype1 or dtype2

        # Simple promotion rules
        return np.promote_types(dtype1, dtype2)


class TensorFactory:
    """Factory for creating symbolic tensors."""

    @staticmethod
    def zeros(
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        device: str = "cpu",
        requires_grad: bool = False
    ) -> SymbolicTensor:
        """Create a zero tensor."""
        return SymbolicTensor(
            metadata=TensorMetadata(
                dtype=dtype,
                shape=shape,
                device=device,
                requires_grad=requires_grad
            ),
            concrete_value=np.zeros(shape, dtype=dtype),
            source="zeros"
        )

    @staticmethod
    def ones(
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        device: str = "cpu",
        requires_grad: bool = False
    ) -> SymbolicTensor:
        """Create a ones tensor."""
        return SymbolicTensor(
            metadata=TensorMetadata(
                dtype=dtype,
                shape=shape,
                device=device,
                requires_grad=requires_grad
            ),
            concrete_value=np.ones(shape, dtype=dtype),
            source="ones"
        )

    @staticmethod
    def randn(
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        device: str = "cpu",
        requires_grad: bool = False
    ) -> SymbolicTensor:
        """Create a random normal tensor."""
        return SymbolicTensor(
            metadata=TensorMetadata(
                dtype=dtype,
                shape=shape,
                device=device,
                requires_grad=requires_grad
            ),
            concrete_value=np.random.randn(*shape).astype(dtype),
            source="randn"
        )

    @staticmethod
    def from_numpy(
        array: np.ndarray,
        device: str = "cpu",
        requires_grad: bool = False
    ) -> SymbolicTensor:
        """Create tensor from NumPy array."""
        return SymbolicTensor(
            metadata=TensorMetadata(
                dtype=array.dtype,
                shape=array.shape,
                device=device,
                requires_grad=requires_grad
            ),
            concrete_value=array.copy(),
            source="from_numpy"
        )