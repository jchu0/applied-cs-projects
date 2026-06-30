"""Automatic differentiation implementation."""

import numpy as np
from typing import Any, Callable, List, Dict, Set, Optional, Tuple, Union
from abc import ABC, abstractmethod
from collections import defaultdict
import threading

from ..core.tensor import Tensor, is_grad_enabled


class Function:
    """
    Base class for differentiable functions.

    Defines forward and backward passes for custom operations.
    """

    @staticmethod
    def forward(ctx: 'FunctionContext', *args, **kwargs) -> Any:
        """Forward pass computation."""
        raise NotImplementedError

    @staticmethod
    def backward(ctx: 'FunctionContext', *grad_outputs) -> Tuple:
        """Backward pass computation."""
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        """Apply the function."""
        ctx = FunctionContext()

        # Run forward
        outputs = cls.forward(ctx, *args, **kwargs)

        # Setup backward
        if is_grad_enabled():
            tensors = [a for a in args if isinstance(a, Tensor) and a.requires_grad]
            if tensors:
                # Create gradient function
                grad_fn = FunctionGradient(cls, ctx, tensors)

                if isinstance(outputs, tuple):
                    for out in outputs:
                        if isinstance(out, Tensor):
                            out._set_grad_fn(grad_fn)
                elif isinstance(outputs, Tensor):
                    outputs._set_grad_fn(grad_fn)

        return outputs


class FunctionContext:
    """Context for saving tensors during forward pass."""

    def __init__(self):
        self.saved_tensors: List[np.ndarray] = []
        self.saved_values: Dict[str, Any] = {}
        self.needs_input_grad: List[bool] = []

    def save_for_backward(self, *tensors):
        """Save tensors for backward pass."""
        for t in tensors:
            if isinstance(t, Tensor):
                self.saved_tensors.append(t.data)
            else:
                self.saved_tensors.append(t)

    def save_value(self, name: str, value: Any):
        """Save arbitrary value."""
        self.saved_values[name] = value

    def get_value(self, name: str) -> Any:
        """Get saved value."""
        return self.saved_values.get(name)


class FunctionGradient:
    """Gradient function for custom operations."""

    def __init__(self, func_cls: type, ctx: FunctionContext, inputs: List[Tensor]):
        self.func_cls = func_cls
        self.ctx = ctx
        self.inputs = inputs

    def backward(self, grad_output: np.ndarray):
        """Compute and propagate gradients."""
        grad_inputs = self.func_cls.backward(self.ctx, grad_output)

        if not isinstance(grad_inputs, tuple):
            grad_inputs = (grad_inputs,)

        for inp, grad in zip(self.inputs, grad_inputs):
            if grad is not None and inp.requires_grad:
                inp.backward(grad)


def backward(
    tensors: Union[Tensor, List[Tensor]],
    grad_tensors: Union[np.ndarray, List[np.ndarray]] = None,
    retain_graph: bool = False,
    create_graph: bool = False
):
    """
    Compute gradients of tensors.

    Args:
        tensors: Tensors to compute gradients for
        grad_tensors: Gradient of the outputs
        retain_graph: Keep computation graph for multiple backward passes
        create_graph: Create graph for higher-order derivatives
    """
    if isinstance(tensors, Tensor):
        tensors = [tensors]

    if grad_tensors is None:
        grad_tensors = [np.ones_like(t.data) for t in tensors]
    elif isinstance(grad_tensors, np.ndarray):
        grad_tensors = [grad_tensors]

    for tensor, grad in zip(tensors, grad_tensors):
        tensor.backward(grad)


def grad(
    outputs: Union[Tensor, List[Tensor]],
    inputs: Union[Tensor, List[Tensor]],
    grad_outputs: Union[np.ndarray, List[np.ndarray]] = None,
    retain_graph: bool = False,
    create_graph: bool = False,
    allow_unused: bool = False
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Compute gradients of outputs with respect to inputs.

    Args:
        outputs: Differentiated tensors
        inputs: Tensors to compute gradients for
        grad_outputs: Gradients w.r.t. outputs
        retain_graph: Keep graph after backward
        create_graph: Create graph for higher-order derivatives
        allow_unused: Don't error on unused inputs

    Returns:
        Gradients of outputs w.r.t. inputs
    """
    if isinstance(outputs, Tensor):
        outputs = [outputs]
    if isinstance(inputs, Tensor):
        inputs = [inputs]
        single_input = True
    else:
        single_input = False

    # Zero gradients
    for inp in inputs:
        inp.zero_grad()

    # Backward pass
    backward(outputs, grad_outputs, retain_graph, create_graph)

    # Collect gradients
    grads = []
    for inp in inputs:
        if inp.grad is None:
            if allow_unused:
                grads.append(np.zeros_like(inp.data))
            else:
                raise RuntimeError(f"Input tensor has no gradient (unused)")
        else:
            grads.append(inp.grad)

    return grads[0] if single_input else grads


class GradientTape:
    """
    Context manager for recording operations for automatic differentiation.

    Similar to TensorFlow's GradientTape.
    """

    def __init__(self, persistent: bool = False, watch_accessed_variables: bool = True):
        self.persistent = persistent
        self.watch_accessed_variables = watch_accessed_variables
        self._tape: List[Tuple[Tensor, Any]] = []
        self._watched: Set[int] = set()
        self._active = False

    def __enter__(self):
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._active = False
        if not self.persistent:
            self._tape.clear()
        return False

    def watch(self, tensor: Tensor):
        """Watch a tensor for gradient computation."""
        self._watched.add(id(tensor))
        tensor.requires_grad = True

    def gradient(
        self,
        target: Tensor,
        sources: Union[Tensor, List[Tensor]],
        output_gradients: np.ndarray = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Compute gradients of target with respect to sources.

        Args:
            target: Tensor to differentiate
            sources: Tensors to compute gradients for
            output_gradients: Gradient of output

        Returns:
            Gradients of target w.r.t. sources
        """
        return grad(target, sources, output_gradients, retain_graph=self.persistent)

    def jacobian(
        self,
        target: Tensor,
        sources: Tensor
    ) -> np.ndarray:
        """
        Compute Jacobian matrix.

        Args:
            target: Output tensor
            sources: Input tensor

        Returns:
            Jacobian matrix
        """
        return jacobian(target, sources)

    def batch_jacobian(
        self,
        target: Tensor,
        sources: Tensor
    ) -> np.ndarray:
        """
        Compute batch Jacobian.

        Args:
            target: Output tensor (batch, ...)
            sources: Input tensor (batch, ...)

        Returns:
            Batch Jacobian matrix
        """
        # Simplified: compute per-sample Jacobian
        batch_size = target.shape[0]
        jacs = []

        for i in range(batch_size):
            jac = jacobian(target[i], sources[i])
            jacs.append(jac)

        return np.stack(jacs)


def jacobian(output: Tensor, input: Tensor) -> np.ndarray:
    """
    Compute the Jacobian matrix dy/dx via reverse-mode autodiff.

    Each scalar component of ``output`` is seeded with a one-hot gradient and a
    backward pass is run, reading the gradient accumulated on ``input`` — so each
    backward pass produces one row of the Jacobian. This reuses the existing
    computation graph; no forward recomputation or finite differences are needed.

    Args:
        output: Output tensor (any shape), connected to ``input`` through the graph.
        input: Input tensor (any shape) with ``requires_grad=True``.

    Returns:
        Jacobian of shape ``(*output.shape, *input.shape)``.
    """
    n_out = output.data.size
    input_size = input.data.size

    jac = np.zeros((n_out, input_size))
    for i in range(n_out):
        # One-hot seed selects the i-th output component; backward then
        # propagates it through the graph to `input`.
        grad_output = np.zeros_like(output.data)
        grad_output.flat[i] = 1.0

        # backward() accumulates, so clear the input grad to read a clean row.
        input.zero_grad()
        output.backward(grad_output)

        if input.grad is not None:
            jac[i, :] = np.asarray(input.grad).reshape(-1)

    return jac.reshape(*output.shape, *input.shape)


def hessian(
    output: Tensor,
    input: Tensor,
    func: Optional[Callable[[Tensor], Tensor]] = None,
) -> np.ndarray:
    """
    Compute the Hessian d²y/dx² of a scalar output.

    This eager autograd produces ndarray gradients (it has no higher-order /
    double backprop), so the second derivative cannot be read back from a static
    graph. Supply ``func`` — a callable that recomputes the scalar output from a
    fresh input tensor — and the Hessian is formed by central finite differences
    of the reverse-mode gradient:
    ``H[:, j] ≈ (∇f(x + eps·e_j) − ∇f(x − eps·e_j)) / (2·eps)``.

    Args:
        output: The scalar output (used only to validate shape).
        input: The point at which to evaluate the Hessian.
        func: Callable mapping an input ``Tensor`` to the scalar output ``Tensor``.
            Required — without it the second derivative is not recoverable, and
            this raises rather than silently returning zeros.

    Returns:
        Symmetric Hessian of shape ``(*input.shape, *input.shape)``.
    """
    if output.size != 1:
        raise ValueError("Hessian requires scalar output")
    if func is None:
        raise NotImplementedError(
            "hessian() needs a `func` callable that recomputes the scalar output "
            "from the input: this eager autograd has no higher-order backprop, so "
            "the second derivative cannot be read from a static graph. "
            "Example: hessian(y, x, func=lambda t: (t * t).sum())."
        )

    original = input.data.copy()
    input_size = original.size
    # Step size for the central difference. Tensors are stored in float32, so a
    # too-small step is swamped by round-off (the perturbation gets quantized);
    # ~1e-2 balances truncation vs. round-off well for float32 (≈1e-4 accuracy).
    eps = 1e-2

    def grad_at(flat_x: np.ndarray) -> np.ndarray:
        """Reverse-mode gradient of `func` evaluated at the given (flat) point."""
        xt = Tensor(flat_x.reshape(original.shape), requires_grad=True)
        y = func(xt)
        if y.size != 1:
            raise ValueError("func must return a scalar output")
        xt.zero_grad()
        y.backward(np.ones_like(y.data))
        return np.asarray(xt.grad).reshape(-1)

    hess = np.zeros((input_size, input_size))
    base = original.reshape(-1)
    for j in range(input_size):
        x_plus = base.copy()
        x_plus[j] += eps
        x_minus = base.copy()
        x_minus[j] -= eps
        hess[:, j] = (grad_at(x_plus) - grad_at(x_minus)) / (2.0 * eps)

    # The true Hessian is symmetric; symmetrize to cancel finite-difference asymmetry.
    hess = 0.5 * (hess + hess.T)
    return hess.reshape(*input.shape, *input.shape)


# Built-in differentiable functions
class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        if isinstance(a, Tensor):
            a = a.data
        if isinstance(b, Tensor):
            b = b.data
        return Tensor(a + b, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output
        grad_b = grad_output

        # Handle broadcasting
        if hasattr(a, 'shape') and a.shape != grad_a.shape:
            grad_a = _unbroadcast(grad_a, a.shape)
        if hasattr(b, 'shape') and b.shape != grad_b.shape:
            grad_b = _unbroadcast(grad_b, b.shape)

        return grad_a, grad_b


class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        if isinstance(a, Tensor):
            a_data = a.data
        else:
            a_data = a
        if isinstance(b, Tensor):
            b_data = b.data
        else:
            b_data = b
        return Tensor(a_data * b_data, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output * b
        grad_b = grad_output * a

        if hasattr(a, 'shape') and a.shape != grad_a.shape:
            grad_a = _unbroadcast(grad_a, a.shape)
        if hasattr(b, 'shape') and b.shape != grad_b.shape:
            grad_b = _unbroadcast(grad_b, b.shape)

        return grad_a, grad_b


class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        data = x.data if isinstance(x, Tensor) else x
        return Tensor(np.maximum(0, data), requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * (x > 0)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        data = x.data if isinstance(x, Tensor) else x
        result = 1 / (1 + np.exp(-data))
        ctx.save_value('output', result)
        return Tensor(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.get_value('output')
        return grad_output * output * (1 - output)


class Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        data = x.data if isinstance(x, Tensor) else x
        result = np.tanh(data)
        ctx.save_value('output', result)
        return Tensor(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.get_value('output')
        return grad_output * (1 - output ** 2)


class Softmax(Function):
    @staticmethod
    def forward(ctx, x, axis=-1):
        data = x.data if isinstance(x, Tensor) else x
        shifted = data - np.max(data, axis=axis, keepdims=True)
        exp_x = np.exp(shifted)
        result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        ctx.save_value('output', result)
        ctx.save_value('axis', axis)
        return Tensor(result, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.get_value('output')
        axis = ctx.get_value('axis')
        return output * (grad_output - (grad_output * output).sum(axis=axis, keepdims=True))


class MatMul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        a_data = a.data if isinstance(a, Tensor) else a
        b_data = b.data if isinstance(b, Tensor) else b
        return Tensor(a_data @ b_data, requires_grad=True)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_output @ np.swapaxes(b, -2, -1)
        grad_b = np.swapaxes(a, -2, -1) @ grad_output
        return grad_a, grad_b


def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Unbroadcast gradient to match original shape."""
    if grad.shape == shape:
        return grad

    ndim_diff = grad.ndim - len(shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    for i, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# ============================================================================
# JIT Tracing
# ============================================================================

class TracedFunction:
    """A function that has been traced for JIT optimization."""

    def __init__(self, func: Callable, traced_graph, compiled_fn: Optional[Callable] = None):
        self.func = func
        self.traced_graph = traced_graph
        self.compiled_fn = compiled_fn
        self._cache: Dict[Tuple, Any] = {}

    def __call__(self, *args, **kwargs):
        """Execute the traced function."""
        # Create cache key from input shapes and dtypes
        cache_key = self._make_cache_key(args)

        if cache_key in self._cache:
            # Use cached compiled function
            return self._cache[cache_key](*args, **kwargs)

        # Compile for this input configuration if not cached
        if self.compiled_fn is not None:
            self._cache[cache_key] = self.compiled_fn
            return self.compiled_fn(*args, **kwargs)

        # Fall back to original function
        return self.func(*args, **kwargs)

    def _make_cache_key(self, args) -> Tuple:
        """Create cache key from input shapes and types."""
        key_parts = []
        for arg in args:
            if isinstance(arg, Tensor):
                key_parts.append(('tensor', arg.shape, arg.data.dtype))
            elif isinstance(arg, np.ndarray):
                key_parts.append(('array', arg.shape, arg.dtype))
            else:
                key_parts.append(('value', type(arg).__name__))
        return tuple(key_parts)


class JITTracer:
    """Tracer for JIT compilation of functions."""

    def __init__(self):
        self._traced_ops: List[Dict[str, Any]] = []
        self._input_tensors: List[Tensor] = []
        self._output_tensors: List[Tensor] = []
        self._active = False

    def trace(self, func: Callable, *sample_inputs) -> TracedFunction:
        """
        Trace a function with sample inputs.

        Args:
            func: Function to trace
            *sample_inputs: Sample inputs for tracing

        Returns:
            TracedFunction with compiled execution
        """
        self._traced_ops.clear()
        self._input_tensors.clear()
        self._active = True

        try:
            # Create traced input tensors
            traced_inputs = []
            for inp in sample_inputs:
                if isinstance(inp, Tensor):
                    traced = Tensor(inp.data.copy(), requires_grad=inp.requires_grad)
                    traced._trace_id = len(self._input_tensors)
                    self._input_tensors.append(traced)
                    traced_inputs.append(traced)
                else:
                    traced_inputs.append(inp)

            # Execute function to capture operations
            outputs = func(*traced_inputs)

            # Record outputs
            if isinstance(outputs, Tensor):
                self._output_tensors = [outputs]
            elif isinstance(outputs, (list, tuple)):
                self._output_tensors = [o for o in outputs if isinstance(o, Tensor)]
            else:
                self._output_tensors = []

        finally:
            self._active = False

        # Build traced graph
        traced_graph = {
            'inputs': len(self._input_tensors),
            'outputs': len(self._output_tensors),
            'ops': self._traced_ops.copy(),
        }

        # Compile the traced function
        compiled_fn = self._compile(traced_graph, func)

        return TracedFunction(func, traced_graph, compiled_fn)

    def _compile(self, graph: Dict, original_func: Callable) -> Callable:
        """Compile traced graph to optimized function."""
        # For now, return a wrapper that applies optimizations
        def optimized_fn(*args, **kwargs):
            # Apply graph-level optimizations
            return original_func(*args, **kwargs)

        return optimized_fn

    def is_tracing(self) -> bool:
        """Check if currently tracing."""
        return self._active


# Global JIT tracer
_jit_tracer = JITTracer()


def jit_trace(func: Callable = None, *, sample_inputs: Tuple = None):
    """
    Decorator for JIT tracing a function.

    Can be used as:
        @jit_trace
        def my_func(x):
            return x * 2

    Or with sample inputs for immediate tracing:
        @jit_trace(sample_inputs=(Tensor([1, 2, 3]),))
        def my_func(x):
            return x * 2

    Args:
        func: Function to trace
        sample_inputs: Sample inputs for tracing

    Returns:
        TracedFunction or decorator
    """
    def decorator(fn: Callable) -> TracedFunction:
        if sample_inputs is not None:
            # Trace immediately with sample inputs
            return _jit_tracer.trace(fn, *sample_inputs)
        else:
            # Lazy tracing - trace on first call
            return LazyTracedFunction(fn)

    if func is not None:
        # Called without arguments: @jit_trace
        return decorator(func)
    else:
        # Called with arguments: @jit_trace(sample_inputs=...)
        return decorator


class LazyTracedFunction:
    """A function that will be traced on first call."""

    def __init__(self, func: Callable):
        self.func = func
        self._traced: Optional[TracedFunction] = None

    def __call__(self, *args, **kwargs):
        if self._traced is None:
            # Trace on first call
            self._traced = _jit_tracer.trace(self.func, *args)

        return self._traced(*args, **kwargs)


def trace_graph(func: Callable, *args) -> Dict:
    """
    Trace a function and return its computation graph.

    Args:
        func: Function to trace
        *args: Arguments to trace with

    Returns:
        Dictionary representing the traced graph
    """
    traced = _jit_tracer.trace(func, *args)
    return traced.traced_graph


def is_tracing() -> bool:
    """Check if currently JIT tracing."""
    return _jit_tracer.is_tracing()
