"""Dynamic compiler with caching and guards."""

import hashlib
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np

from ..core.graph import Graph
from ..optimizer.graph_optimizer import GraphOptimizer
from ..tracer.bytecode_tracer import BytecodeTracer
from .backend import Backend, BackendRegistry, CompiledFunction


class Guard:
    """Condition that must be true for cached code to be valid."""

    def __init__(self, name: str, check_fn: Callable[..., bool], description: str = ""):
        self.name = name
        self.check_fn = check_fn
        self.description = description

    def check(self, *args, **kwargs) -> bool:
        """Check if guard condition holds."""
        try:
            return self.check_fn(*args, **kwargs)
        except Exception:
            return False


class ShapeGuard(Guard):
    """Guard on tensor shape."""

    def __init__(self, arg_index: int, expected_shape: Tuple[int, ...]):
        self.arg_index = arg_index
        self.expected_shape = expected_shape
        super().__init__(
            name=f"shape_guard_{arg_index}",
            check_fn=self._check,
            description=f"arg[{arg_index}].shape == {expected_shape}"
        )

    def _check(self, *args, **kwargs) -> bool:
        if self.arg_index >= len(args):
            return False
        arg = args[self.arg_index]
        if hasattr(arg, 'shape'):
            return tuple(arg.shape) == self.expected_shape
        return False


class DtypeGuard(Guard):
    """Guard on tensor dtype."""

    def __init__(self, arg_index: int, expected_dtype: str):
        self.arg_index = arg_index
        self.expected_dtype = expected_dtype
        super().__init__(
            name=f"dtype_guard_{arg_index}",
            check_fn=self._check,
            description=f"arg[{arg_index}].dtype == {expected_dtype}"
        )

    def _check(self, *args, **kwargs) -> bool:
        if self.arg_index >= len(args):
            return False
        arg = args[self.arg_index]
        if hasattr(arg, 'dtype'):
            return str(arg.dtype) == self.expected_dtype
        return False


@dataclass
class CacheEntry:
    """Cached compilation result with guards."""
    compiled_fn: CompiledFunction
    guards: List[Guard]
    graph: Graph
    hit_count: int = 0
    compile_time_ms: float = 0.0
    created_at: float = field(default_factory=time.time)

    def check_guards(self, *args, **kwargs) -> bool:
        """Check if all guards pass."""
        return all(guard.check(*args, **kwargs) for guard in self.guards)


class CompilationCache:
    """
    Cache for compiled functions keyed by source code and guards.

    Implements LRU eviction and statistics tracking.
    """

    def __init__(self, max_entries: int = 1000):
        self.cache: Dict[str, List[CacheEntry]] = {}
        self.max_entries = max_entries
        self.total_entries = 0
        self.hits = 0
        self.misses = 0
        self.compile_time_total_ms = 0.0

    def lookup(self, func: Callable, *args, **kwargs) -> Optional[CompiledFunction]:
        """Look up cached compilation for function and inputs."""
        key = self._compute_key(func)

        if key not in self.cache:
            self.misses += 1
            return None

        # Check guards for each entry
        for entry in self.cache[key]:
            if entry.check_guards(*args, **kwargs):
                self.hits += 1
                entry.hit_count += 1
                return entry.compiled_fn

        self.misses += 1
        return None

    def insert(
        self,
        func: Callable,
        compiled_fn: CompiledFunction,
        graph: Graph,
        guards: List[Guard],
        compile_time_ms: float,
    ):
        """Insert compiled function into cache."""
        key = self._compute_key(func)

        entry = CacheEntry(
            compiled_fn=compiled_fn,
            guards=guards,
            graph=graph,
            compile_time_ms=compile_time_ms,
        )

        if key not in self.cache:
            self.cache[key] = []

        self.cache[key].append(entry)
        self.total_entries += 1
        self.compile_time_total_ms += compile_time_ms

        # Evict if necessary
        if self.total_entries > self.max_entries:
            self._evict()

    def _compute_key(self, func: Callable) -> str:
        """Compute cache key from function."""
        if hasattr(func, '__code__'):
            code = func.__code__
            return f"{code.co_filename}:{code.co_firstlineno}:{code.co_name}"
        return str(id(func))

    def _evict(self):
        """Evict least recently used entries."""
        # Find entry with lowest hit count and oldest creation time
        min_score = float('inf')
        min_key = None
        min_idx = 0

        for key, entries in self.cache.items():
            for idx, entry in enumerate(entries):
                # Score based on hit count and age
                age = time.time() - entry.created_at
                score = entry.hit_count - (age / 3600)  # Decay over time

                if score < min_score:
                    min_score = score
                    min_key = key
                    min_idx = idx

        if min_key:
            del self.cache[min_key][min_idx]
            if not self.cache[min_key]:
                del self.cache[min_key]
            self.total_entries -= 1

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_requests = self.hits + self.misses
        return {
            'total_entries': self.total_entries,
            'num_functions': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total_requests if total_requests > 0 else 0,
            'total_compile_time_ms': self.compile_time_total_ms,
            'avg_compile_time_ms': (
                self.compile_time_total_ms / self.total_entries
                if self.total_entries > 0 else 0
            ),
        }

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.total_entries = 0
        self.hits = 0
        self.misses = 0
        self.compile_time_total_ms = 0.0


class DynamicCompiler:
    """
    Main entry point for dynamic compilation.

    Provides a torch.compile()-like API for JIT compilation of Python
    functions with automatic caching and guard generation.
    """

    def __init__(
        self,
        backend: str = "eager_numpy",
        optimization_level: int = 1,
        cache_enabled: bool = True,
        fallback_to_eager: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize the dynamic compiler.

        Args:
            backend: Name of backend to use
            optimization_level: 0=none, 1=basic, 2=aggressive
            cache_enabled: Enable compilation cache
            fallback_to_eager: Fall back to eager on compile failure
            verbose: Print compilation info
        """
        self.backend_name = backend
        self.optimization_level = optimization_level
        self.cache_enabled = cache_enabled
        self.fallback_to_eager = fallback_to_eager
        self.verbose = verbose

        # Components
        self.optimizer = GraphOptimizer(optimization_level=optimization_level)
        self.cache = CompilationCache()
        self.tracer = BytecodeTracer()

        # Get backend
        self.backend = BackendRegistry.get(backend)
        if self.backend is None:
            available = BackendRegistry.list_available()
            if available:
                self.backend = BackendRegistry.get(available[0])
            else:
                raise RuntimeError("No backends available")

    def __call__(self, func: Callable) -> Callable:
        """Decorator to compile a function."""
        @wraps(func)
        def compiled_func(*args, **kwargs):
            # Check cache first
            if self.cache_enabled:
                cached = self.cache.lookup(func, *args, **kwargs)
                if cached:
                    if self.verbose:
                        print(f"[compile] Cache hit for {func.__name__}")
                    return cached(*args, **kwargs)

            # Trace and compile
            try:
                start_time = time.time()

                # Trace function to build graph
                graph = self.tracer.trace_function(func, *args, **kwargs)

                # Optimize graph
                optimized, opt_stats = self.optimizer.optimize(graph)

                # Compile to backend
                compiled = self.backend.compile(optimized)

                compile_time_ms = (time.time() - start_time) * 1000

                if self.verbose:
                    print(f"[compile] Compiled {func.__name__} in {compile_time_ms:.2f}ms")
                    print(f"[compile] Graph: {len(optimized.nodes)} nodes")

                # Generate guards
                guards = self._create_guards(*args, **kwargs)

                # Cache result
                if self.cache_enabled:
                    self.cache.insert(
                        func, compiled, optimized, guards, compile_time_ms
                    )

                return compiled(*args, **kwargs)

            except Exception as e:
                if self.fallback_to_eager:
                    if self.verbose:
                        print(f"[compile] Fallback to eager: {e}")
                    return func(*args, **kwargs)
                raise

        # Attach metadata
        compiled_func._compiled = True
        compiled_func._compiler = self
        compiled_func._original = func

        return compiled_func

    def _create_guards(self, *args, **kwargs) -> List[Guard]:
        """Create guards from function arguments."""
        guards = []

        for i, arg in enumerate(args):
            if hasattr(arg, 'shape'):
                guards.append(ShapeGuard(i, tuple(arg.shape)))
            if hasattr(arg, 'dtype'):
                guards.append(DtypeGuard(i, str(arg.dtype)))

        return guards

    def get_stats(self) -> Dict[str, Any]:
        """Get compilation statistics."""
        return {
            'cache': self.cache.get_stats(),
            'optimizer': self.optimizer.get_pass_stats(),
            'backend': self.backend.name(),
        }

    def reset_stats(self):
        """Reset all statistics."""
        self.cache.clear()
        self.optimizer.reset_stats()


# Global default compiler
_default_compiler: Optional[DynamicCompiler] = None


def get_default_compiler() -> DynamicCompiler:
    """Get or create the default compiler."""
    global _default_compiler
    if _default_compiler is None:
        _default_compiler = DynamicCompiler()
    return _default_compiler


def compile(
    func: Optional[Callable] = None,
    *,
    backend: str = "eager_numpy",
    optimization_level: int = 1,
    **kwargs
) -> Callable:
    """
    Decorator to compile a function for optimized execution.

    Similar to torch.compile() API.

    Example:
        @compile(backend="eager_numpy")
        def forward(x, y):
            return x + y * 2

    Args:
        func: Function to compile (or None for decorator with args)
        backend: Backend name
        optimization_level: 0=none, 1=basic, 2=aggressive
        **kwargs: Additional compiler options

    Returns:
        Compiled function or decorator
    """
    compiler = DynamicCompiler(
        backend=backend,
        optimization_level=optimization_level,
        **kwargs
    )

    if func is not None:
        return compiler(func)
    return compiler


def optimize(func: Callable) -> Callable:
    """Alias for compile with default settings."""
    return compile(func)


def trace(func: Callable, *args, **kwargs) -> Graph:
    """
    Trace a function and return its graph representation.

    Useful for debugging and visualization.

    Args:
        func: Function to trace
        *args: Example arguments
        **kwargs: Example keyword arguments

    Returns:
        Computation graph
    """
    tracer = BytecodeTracer()
    return tracer.trace_function(func, *args, **kwargs)


def explain(func: Callable, *args, **kwargs) -> str:
    """
    Explain how a function would be compiled.

    Returns a string describing the graph and optimizations.
    """
    tracer = BytecodeTracer()
    optimizer = GraphOptimizer(optimization_level=2, verbose=False)

    # Trace
    graph = tracer.trace_function(func, *args, **kwargs)

    # Optimize
    optimized, stats = optimizer.optimize(graph)

    # Build explanation
    lines = [
        f"Function: {func.__name__}",
        f"",
        f"Original graph:",
        f"  Nodes: {len(graph.nodes)}",
        f"  Edges: {len(graph.edges)}",
        f"",
        f"Optimized graph:",
        f"  Nodes: {len(optimized.nodes)}",
        f"  Edges: {len(optimized.edges)}",
        f"",
        f"Optimization passes: {stats.total_passes}",
        f"Changes made: {stats.total_changes}",
        f"Nodes removed: {stats.nodes_removed}",
        f"Time: {stats.optimization_time_ms:.2f}ms",
        f"",
        f"Pass results:"
    ]

    for pass_name, results in stats.pass_results.items():
        total_changes = sum(1 for r in results if r.changed)
        lines.append(f"  {pass_name}: {total_changes} changes")

    return "\n".join(lines)
