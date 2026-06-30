"""JIT compilation for tensor functions."""

import hashlib
from typing import Any, Callable, Dict, Tuple, Union
from functools import wraps

from ..core.tensor import LazyTensor, Tracer, TensorSpec, ShapeSpec


class JittedFunction:
    """Compiled function with caching."""

    def __init__(self, fun: Callable, static_argnums: Tuple[int, ...] = ()):
        self.fun = fun
        self.static_argnums = static_argnums
        self._cache: Dict[str, Callable] = {}
        self._call_count = 0

    def __call__(self, *args, **kwargs):
        cache_key = self._make_cache_key(args, kwargs)

        if cache_key not in self._cache:
            compiled = self._compile(args, kwargs)
            self._cache[cache_key] = compiled

        self._call_count += 1
        return self._cache[cache_key](*args, **kwargs)

    def _make_cache_key(self, args, kwargs) -> str:
        """Create cache key from argument signatures."""
        key_parts = []

        for i, arg in enumerate(args):
            if i in self.static_argnums:
                key_parts.append(f"static_{i}_{arg}")
            elif isinstance(arg, LazyTensor):
                key_parts.append(f"tensor_{arg.shape}_{arg.dtype}")
            else:
                key_parts.append(f"other_{type(arg).__name__}")

        for k, v in sorted(kwargs.items()):
            if isinstance(v, LazyTensor):
                key_parts.append(f"{k}_{v.shape}_{v.dtype}")
            else:
                key_parts.append(f"{k}_{v}")

        key_str = "_".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _compile(self, args, kwargs) -> Callable:
        """Trace and compile the function."""
        tracers = []
        for i, arg in enumerate(args):
            if i in self.static_argnums:
                tracers.append(arg)
            elif isinstance(arg, LazyTensor):
                spec = TensorSpec(ShapeSpec(arg.shape), arg.dtype)
                tracers.append(Tracer(spec, f"arg_{i}"))
            else:
                tracers.append(arg)

        # For now, return original function
        # Full implementation would build and optimize IR
        return self.fun

    @property
    def cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self._call_count - len(self._cache),
            "misses": len(self._cache),
            "size": len(self._cache),
        }

    def clear_cache(self):
        """Clear compilation cache."""
        self._cache.clear()


def jit(
    fun: Callable = None,
    static_argnums: Union[int, Tuple[int, ...]] = ()
) -> Callable:
    """JIT compile a function.

    Args:
        fun: Function to compile
        static_argnums: Arguments treated as compile-time constants

    Returns:
        JIT-compiled function
    """
    if isinstance(static_argnums, int):
        static_argnums = (static_argnums,)

    def decorator(f):
        return JittedFunction(f, static_argnums)

    if fun is None:
        return decorator
    return decorator(fun)


class TracingContext:
    """Context for tracing computation graphs."""

    _stack = []

    def __init__(self):
        self.ops = []
        self.inputs = []
        self.outputs = []

    def __enter__(self):
        TracingContext._stack.append(self)
        return self

    def __exit__(self, *args):
        TracingContext._stack.pop()

    @classmethod
    def is_tracing(cls) -> bool:
        return len(cls._stack) > 0

    @classmethod
    def get_current(cls):
        return cls._stack[-1] if cls._stack else None


def trace(fun: Callable, *example_args):
    """Trace a function to build computation graph.

    Args:
        fun: Function to trace
        *example_args: Example inputs for tracing

    Returns:
        Traced graph representation
    """
    with TracingContext() as ctx:
        # Create tracers
        tracers = []
        for i, arg in enumerate(example_args):
            if isinstance(arg, LazyTensor):
                spec = TensorSpec(ShapeSpec(arg.shape), arg.dtype)
                tracer = Tracer(spec, f"input_{i}")
                tracers.append(tracer)
                ctx.inputs.append(tracer)
            else:
                tracers.append(arg)

        # Execute with tracers
        result = fun(*tracers)

        if isinstance(result, LazyTensor):
            ctx.outputs.append(result)
        elif isinstance(result, tuple):
            ctx.outputs.extend(result)

    return ctx
