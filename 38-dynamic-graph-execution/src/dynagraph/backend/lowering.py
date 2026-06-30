"""Backend lowering framework.

Provides the core infrastructure for lowering high-level computation
graphs to backend-specific representations.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import numpy as np


class DataType(Enum):
    """Supported data types."""

    FLOAT32 = auto()
    FLOAT64 = auto()
    FLOAT16 = auto()
    BFLOAT16 = auto()
    INT8 = auto()
    INT16 = auto()
    INT32 = auto()
    INT64 = auto()
    UINT8 = auto()
    BOOL = auto()

    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> 'DataType':
        """Convert numpy dtype to DataType."""
        mapping = {
            np.float32: cls.FLOAT32,
            np.float64: cls.FLOAT64,
            np.float16: cls.FLOAT16,
            np.int8: cls.INT8,
            np.int16: cls.INT16,
            np.int32: cls.INT32,
            np.int64: cls.INT64,
            np.uint8: cls.UINT8,
            np.bool_: cls.BOOL,
        }
        return mapping.get(dtype.type, cls.FLOAT32)

    def to_numpy(self) -> np.dtype:
        """Convert to numpy dtype."""
        mapping = {
            DataType.FLOAT32: np.float32,
            DataType.FLOAT64: np.float64,
            DataType.FLOAT16: np.float16,
            DataType.INT8: np.int8,
            DataType.INT16: np.int16,
            DataType.INT32: np.int32,
            DataType.INT64: np.int64,
            DataType.UINT8: np.uint8,
            DataType.BOOL: np.bool_,
        }
        return np.dtype(mapping.get(self, np.float32))


class MemoryLayout(Enum):
    """Memory layout options."""

    ROW_MAJOR = auto()     # C-contiguous (default)
    COLUMN_MAJOR = auto()  # Fortran-contiguous
    BLOCKED = auto()       # Blocked layout for cache efficiency
    CHANNELS_LAST = auto() # NHWC format for images


@dataclass
class TensorSpec:
    """Specification for a tensor in the lowered graph."""

    name: str
    shape: Tuple[int, ...]
    dtype: DataType = DataType.FLOAT32
    layout: MemoryLayout = MemoryLayout.ROW_MAJOR
    device: str = "cpu"
    is_constant: bool = False
    constant_value: Optional[np.ndarray] = None

    @property
    def size(self) -> int:
        """Total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def nbytes(self) -> int:
        """Size in bytes."""
        dtype_sizes = {
            DataType.FLOAT32: 4,
            DataType.FLOAT64: 8,
            DataType.FLOAT16: 2,
            DataType.BFLOAT16: 2,
            DataType.INT8: 1,
            DataType.INT16: 2,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.UINT8: 1,
            DataType.BOOL: 1,
        }
        return self.size * dtype_sizes.get(self.dtype, 4)


@dataclass
class OpMapping:
    """Mapping of an operation to backend representation."""

    op_type: str
    backend_op: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"OpMapping({self.op_type} -> {self.backend_op})"


@dataclass
class LoweredNode:
    """A node in the lowered graph."""

    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"LoweredNode({self.name}: {self.op_type})"


@dataclass
class LoweredGraph:
    """A lowered computation graph."""

    name: str = "graph"
    inputs: List[TensorSpec] = field(default_factory=list)
    outputs: List[TensorSpec] = field(default_factory=list)
    nodes: List[LoweredNode] = field(default_factory=list)
    constants: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_input(self, spec: TensorSpec) -> None:
        """Add an input tensor specification."""
        self.inputs.append(spec)

    def add_output(self, spec: TensorSpec) -> None:
        """Add an output tensor specification."""
        self.outputs.append(spec)

    def add_node(self, node: LoweredNode) -> None:
        """Add a computation node."""
        self.nodes.append(node)

    def add_constant(self, name: str, value: np.ndarray) -> None:
        """Add a constant value."""
        self.constants[name] = value

    def get_node(self, name: str) -> Optional[LoweredNode]:
        """Get a node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def topological_sort(self) -> List[LoweredNode]:
        """Return nodes in topological order."""
        # Build dependency graph
        dependencies: Dict[str, set] = {node.name: set() for node in self.nodes}
        node_outputs: Dict[str, str] = {}

        for node in self.nodes:
            for output in node.outputs:
                node_outputs[output] = node.name

        for node in self.nodes:
            for inp in node.inputs:
                if inp in node_outputs:
                    dependencies[node.name].add(node_outputs[inp])

        # Topological sort
        sorted_nodes = []
        remaining = set(node.name for node in self.nodes)

        while remaining:
            ready = [name for name in remaining if not (dependencies[name] & remaining)]
            if not ready:
                raise RuntimeError("Cycle detected in graph")
            for name in ready:
                remaining.remove(name)
                sorted_nodes.append(self.get_node(name))

        return sorted_nodes

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'inputs': [
                {'name': s.name, 'shape': s.shape, 'dtype': s.dtype.name}
                for s in self.inputs
            ],
            'outputs': [
                {'name': s.name, 'shape': s.shape, 'dtype': s.dtype.name}
                for s in self.outputs
            ],
            'nodes': [
                {
                    'name': n.name,
                    'op_type': n.op_type,
                    'inputs': n.inputs,
                    'outputs': n.outputs,
                    'attributes': n.attributes,
                }
                for n in self.nodes
            ],
            'metadata': self.metadata,
        }


@dataclass
class LoweringContext:
    """Context for lowering operations."""

    graph: LoweredGraph = field(default_factory=LoweredGraph)
    tensor_map: Dict[str, TensorSpec] = field(default_factory=dict)
    op_counter: int = 0
    target_dtype: Optional[DataType] = None
    target_layout: Optional[MemoryLayout] = None
    optimization_level: int = 2

    def new_tensor_name(self, prefix: str = "tensor") -> str:
        """Generate a unique tensor name."""
        self.op_counter += 1
        return f"{prefix}_{self.op_counter}"

    def new_node_name(self, prefix: str = "node") -> str:
        """Generate a unique node name."""
        self.op_counter += 1
        return f"{prefix}_{self.op_counter}"

    def register_tensor(self, spec: TensorSpec) -> None:
        """Register a tensor specification."""
        self.tensor_map[spec.name] = spec

    def get_tensor(self, name: str) -> Optional[TensorSpec]:
        """Get tensor specification by name."""
        return self.tensor_map.get(name)


class LoweringPass(ABC):
    """Base class for lowering passes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Pass name."""
        pass

    @abstractmethod
    def run(self, graph: LoweredGraph, ctx: LoweringContext) -> LoweredGraph:
        """Run the lowering pass."""
        pass


class BackendLowering(ABC):
    """Base class for backend lowering implementations."""

    def __init__(self):
        self.passes: List[LoweringPass] = []
        self.op_registry: Dict[str, Callable] = {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass

    @abstractmethod
    def supported_ops(self) -> List[str]:
        """List of supported operations."""
        pass

    @abstractmethod
    def supported_dtypes(self) -> List[DataType]:
        """List of supported data types."""
        pass

    def register_op(self, op_type: str, lowering_fn: Callable) -> None:
        """Register an operation lowering function."""
        self.op_registry[op_type] = lowering_fn

    def add_pass(self, lowering_pass: LoweringPass) -> None:
        """Add a lowering pass."""
        self.passes.append(lowering_pass)

    def lower(self, graph: Any) -> LoweredGraph:
        """
        Lower a high-level graph to this backend.

        Args:
            graph: High-level computation graph

        Returns:
            Lowered graph for this backend
        """
        ctx = LoweringContext()

        # Convert high-level graph to lowered representation
        lowered = self._initial_lowering(graph, ctx)

        # Apply lowering passes
        for pass_ in self.passes:
            lowered = pass_.run(lowered, ctx)

        return lowered

    @abstractmethod
    def _initial_lowering(self, graph: Any, ctx: LoweringContext) -> LoweredGraph:
        """Perform initial lowering from high-level graph."""
        pass

    @abstractmethod
    def compile(self, graph: LoweredGraph) -> Any:
        """Compile lowered graph to executable form."""
        pass

    @abstractmethod
    def execute(self, compiled: Any, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute compiled graph."""
        pass

    def validate(self, graph: LoweredGraph) -> List[str]:
        """
        Validate lowered graph for this backend.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        supported_ops = set(self.supported_ops())
        supported_dtypes = set(self.supported_dtypes())

        for node in graph.nodes:
            if node.op_type not in supported_ops:
                errors.append(f"Unsupported operation: {node.op_type}")

        for tensor in graph.inputs + graph.outputs:
            if tensor.dtype not in supported_dtypes:
                errors.append(f"Unsupported dtype {tensor.dtype.name} for tensor {tensor.name}")

        return errors
