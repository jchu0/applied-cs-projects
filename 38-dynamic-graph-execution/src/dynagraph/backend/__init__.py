"""Backend lowering for graph execution.

Provides infrastructure for lowering computation graphs to various
backend targets including ONNX, native code, and hardware accelerators.
"""

from .lowering import (
    BackendLowering,
    LoweringPass,
    LoweringContext,
    LoweredGraph,
    LoweredNode,
    OpMapping,
    TensorSpec,
    DataType,
    MemoryLayout,
)
from .native import NativeBackend
from .onnx_backend import ONNXBackend
from .passes import (
    DtypeCastPass,
    MemoryLayoutPass,
    OpFusionPass,
    ConstantPropagationPass,
    DeadNodeEliminationPass,
)

__all__ = [
    # Core lowering
    "BackendLowering",
    "LoweringPass",
    "LoweringContext",
    "LoweredGraph",
    "LoweredNode",
    "OpMapping",
    "TensorSpec",
    "DataType",
    "MemoryLayout",
    # Backends
    "NativeBackend",
    "ONNXBackend",
    # Passes
    "DtypeCastPass",
    "MemoryLayoutPass",
    "OpFusionPass",
    "ConstantPropagationPass",
    "DeadNodeEliminationPass",
]
