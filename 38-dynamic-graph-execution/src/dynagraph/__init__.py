"""DynaGraph - Dynamic computation graph execution engine."""

from .core import (
    Tensor,
    Parameter,
    no_grad,
    enable_grad,
    set_grad_enabled,
    is_grad_enabled,
)
from .graph import (
    Node,
    Operation,
    Graph,
    GraphContext,
)
from .autograd import (
    Function,
    backward,
    grad,
    GradientTape,
    jit_trace,
    trace_graph,
)
from .executor import (
    Executor,
    EagerExecutor,
    LazyExecutor,
    GraphOptimizer,
    FusionPass,
    DeadCodePass,
)
from .backend import (
    BackendLowering,
    NativeBackend,
    ONNXBackend,
    LoweredGraph,
    TensorSpec,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Tensor",
    "Parameter",
    "no_grad",
    "enable_grad",
    "set_grad_enabled",
    "is_grad_enabled",
    # Graph
    "Node",
    "Operation",
    "Graph",
    "GraphContext",
    # Autograd
    "Function",
    "backward",
    "grad",
    "GradientTape",
    "jit_trace",
    "trace_graph",
    # Executor
    "Executor",
    "EagerExecutor",
    "LazyExecutor",
    "GraphOptimizer",
    "FusionPass",
    "DeadCodePass",
    # Backend
    "BackendLowering",
    "NativeBackend",
    "ONNXBackend",
    "LoweredGraph",
    "TensorSpec",
]
