"""Dynamic Graph Execution Runtime - PyTorch-like dynamic graph execution."""

__version__ = "0.1.0"

# Core graph structures
from .core.graph import Graph, Node, Edge, OpType, NodeMetadata
from .core.tensor import SymbolicTensor, TensorMetadata, TensorFactory
from .core.context import (
    CompilationContext,
    ExecutionContext,
    GlobalContext,
    get_current_context,
    set_current_context,
    no_grad,
    enable_grad,
)

# Tracing
from .tracer.bytecode_tracer import BytecodeTracer, TracingMode, TraceFrame
from .tracer.frame_guard import FrameGuard, GuardFailure, GuardCondition

# Optimization
from .optimizer.graph_optimizer import GraphOptimizer, optimize_graph
from .optimizer.passes import (
    OptimizationPass,
    ConstantFolding,
    DeadCodeElimination,
    CommonSubexpressionElimination,
    AlgebraicSimplification,
    OperatorFusion,
)

# Code generation and backends
from .codegen.backend import (
    Backend,
    BackendRegistry,
    EagerBackend,
    CompiledFunction,
)
from .codegen.compiler import (
    DynamicCompiler,
    CompilationCache,
    Guard,
    ShapeGuard,
    DtypeGuard,
    CacheEntry,
    compile,
    optimize,
    trace,
    explain,
)

__all__ = [
    # Version
    "__version__",

    # Core graph
    "Graph",
    "Node",
    "Edge",
    "OpType",
    "NodeMetadata",

    # Tensor
    "SymbolicTensor",
    "TensorMetadata",
    "TensorFactory",

    # Context
    "CompilationContext",
    "ExecutionContext",
    "GlobalContext",
    "get_current_context",
    "set_current_context",
    "no_grad",
    "enable_grad",

    # Tracing
    "BytecodeTracer",
    "TracingMode",
    "TraceFrame",
    "FrameGuard",
    "GuardFailure",
    "GuardCondition",

    # Optimization
    "GraphOptimizer",
    "optimize_graph",
    "OptimizationPass",
    "ConstantFolding",
    "DeadCodeElimination",
    "CommonSubexpressionElimination",
    "AlgebraicSimplification",
    "OperatorFusion",

    # Backends
    "Backend",
    "BackendRegistry",
    "EagerBackend",
    "CompiledFunction",

    # Compiler
    "DynamicCompiler",
    "CompilationCache",
    "Guard",
    "ShapeGuard",
    "DtypeGuard",
    "CacheEntry",
    "compile",
    "optimize",
    "trace",
    "explain",
]