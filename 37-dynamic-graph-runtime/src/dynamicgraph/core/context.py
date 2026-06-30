"""Execution and compilation contexts for dynamic graph."""

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from .graph import Graph
from .tensor import SymbolicTensor


@dataclass
class CompilationContext:
    """Context for graph compilation."""
    graph: Optional[Graph] = None
    optimization_level: int = 1  # 0=none, 1=basic, 2=aggressive
    backend: str = "eager"
    device: str = "cpu"
    enable_profiling: bool = False
    enable_debugging: bool = False
    cache_compiled: bool = True
    max_graph_size: int = 10000  # Maximum nodes before graph break
    min_graph_size: int = 10  # Minimum nodes for compilation

    # Compilation statistics
    num_compilations: int = 0
    num_cache_hits: int = 0
    num_graph_breaks: int = 0
    compilation_time_ms: float = 0.0

    def should_compile(self, graph: Graph) -> bool:
        """Determine if graph should be compiled."""
        num_nodes = len(graph.nodes)
        return (
            self.optimization_level > 0
            and self.min_graph_size <= num_nodes <= self.max_graph_size
            and not graph.has_cycle()
        )


@dataclass
class ExecutionContext:
    """Context for graph execution."""
    compilation_context: CompilationContext = field(default_factory=CompilationContext)
    tensor_cache: Dict[str, SymbolicTensor] = field(default_factory=dict)
    parameter_values: Dict[str, Any] = field(default_factory=dict)
    buffer_values: Dict[str, Any] = field(default_factory=dict)
    gradients: Dict[str, SymbolicTensor] = field(default_factory=dict)

    # Execution state
    current_graph: Optional[Graph] = None
    execution_stack: List[Graph] = field(default_factory=list)
    is_training: bool = True
    is_grad_enabled: bool = True

    # Thread-local storage
    _thread_local: threading.local = field(default_factory=threading.local)

    def push_graph(self, graph: Graph) -> None:
        """Push graph onto execution stack."""
        self.execution_stack.append(self.current_graph)
        self.current_graph = graph

    def pop_graph(self) -> Optional[Graph]:
        """Pop graph from execution stack."""
        if self.execution_stack:
            prev_graph = self.current_graph
            self.current_graph = self.execution_stack.pop()
            return prev_graph
        return None

    def get_tensor(self, node_id: str) -> Optional[SymbolicTensor]:
        """Get tensor from cache."""
        return self.tensor_cache.get(node_id)

    def set_tensor(self, node_id: str, tensor: SymbolicTensor) -> None:
        """Store tensor in cache."""
        self.tensor_cache[node_id] = tensor

    def clear_cache(self) -> None:
        """Clear tensor cache."""
        self.tensor_cache.clear()

    def set_parameter(self, name: str, value: Any) -> None:
        """Set parameter value."""
        self.parameter_values[name] = value

    def get_parameter(self, name: str) -> Optional[Any]:
        """Get parameter value."""
        return self.parameter_values.get(name)

    def set_buffer(self, name: str, value: Any) -> None:
        """Set buffer value."""
        self.buffer_values[name] = value

    def get_buffer(self, name: str) -> Optional[Any]:
        """Get buffer value."""
        return self.buffer_values.get(name)

    def compute_gradients(self, loss: SymbolicTensor) -> Dict[str, SymbolicTensor]:
        """Compute gradients via backward pass."""
        if not self.is_grad_enabled:
            return {}

        # Placeholder for actual gradient computation
        # This would implement automatic differentiation
        gradients = {}

        # For now, return empty gradients
        return gradients

    def zero_gradients(self) -> None:
        """Zero all gradients."""
        self.gradients.clear()


class GlobalContext:
    """Global context manager for execution."""

    _instance: Optional['GlobalContext'] = None
    _lock = threading.Lock()

    def __init__(self):
        self.contexts: Dict[int, ExecutionContext] = {}
        self.default_context = ExecutionContext()

    @classmethod
    def get_instance(cls) -> 'GlobalContext':
        """Get singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_context(self, thread_id: Optional[int] = None) -> ExecutionContext:
        """Get context for thread."""
        if thread_id is None:
            thread_id = threading.get_ident()

        if thread_id not in self.contexts:
            self.contexts[thread_id] = ExecutionContext()

        return self.contexts[thread_id]

    def set_context(self, context: ExecutionContext,
                    thread_id: Optional[int] = None) -> None:
        """Set context for thread."""
        if thread_id is None:
            thread_id = threading.get_ident()
        self.contexts[thread_id] = context

    def clear_context(self, thread_id: Optional[int] = None) -> None:
        """Clear context for thread."""
        if thread_id is None:
            thread_id = threading.get_ident()
        if thread_id in self.contexts:
            del self.contexts[thread_id]


# Convenience functions
def get_current_context() -> ExecutionContext:
    """Get current execution context."""
    return GlobalContext.get_instance().get_context()


def set_current_context(context: ExecutionContext) -> None:
    """Set current execution context."""
    GlobalContext.get_instance().set_context(context)


class no_grad:
    """Context manager to disable gradient computation."""

    def __init__(self):
        self.prev_state: Optional[bool] = None

    def __enter__(self):
        context = get_current_context()
        self.prev_state = context.is_grad_enabled
        context.is_grad_enabled = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        context = get_current_context()
        context.is_grad_enabled = self.prev_state


class enable_grad:
    """Context manager to enable gradient computation."""

    def __init__(self):
        self.prev_state: Optional[bool] = None

    def __enter__(self):
        context = get_current_context()
        self.prev_state = context.is_grad_enabled
        context.is_grad_enabled = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        context = get_current_context()
        context.is_grad_enabled = self.prev_state