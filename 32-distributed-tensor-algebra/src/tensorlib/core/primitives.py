"""Primitive operations with autodiff rules."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union
import numpy as np

from .tensor import LazyTensor, Tracer, TensorSpec, ShapeSpec, array


class Primitive(ABC):
    """Base class for primitive operations."""
    name: str = "primitive"

    def __init__(self):
        self.inputs: List[LazyTensor] = []
        self.outputs: List[LazyTensor] = []
        self.params: Dict[str, Any] = {}

    @abstractmethod
    def impl(self, *args, **kwargs) -> np.ndarray:
        """Concrete implementation."""
        pass

    @abstractmethod
    def abstract_eval(self, *avals: TensorSpec, **kwargs) -> TensorSpec:
        """Compute output type from input types."""
        pass

    def vjp(self, primals, tangents, g):
        """Vector-Jacobian product for reverse-mode autodiff."""
        raise NotImplementedError(f"VJP not implemented for {self.name}")

    def jvp(self, primals, tangents):
        """Jacobian-vector product for forward-mode autodiff."""
        raise NotImplementedError(f"JVP not implemented for {self.name}")

    def evaluate(self) -> np.ndarray:
        """Evaluate with stored inputs."""
        concrete_inputs = [inp.materialize() for inp in self.inputs]
        return self.impl(*concrete_inputs, **self.params)


# Registry
_primitive_registry: Dict[str, type] = {}


def register_primitive(prim_class):
    """Register a primitive."""
    _primitive_registry[prim_class.name] = prim_class
    return prim_class


def _ensure_tensor(x) -> LazyTensor:
    """Ensure x is a LazyTensor."""
    if isinstance(x, LazyTensor):
        return x
    return array(x)


def _create_primitive_op(prim_class, *inputs, **params) -> LazyTensor:
    """Create a lazy operation from a primitive."""
    prim = prim_class()
    prim.inputs = list(inputs)
    prim.params = params

    input_specs = [
        TensorSpec(ShapeSpec(inp.shape), inp.dtype)
        for inp in inputs
    ]
    output_spec = prim.abstract_eval(*input_specs, **params)

    tracer = Tracer(output_spec)
    output = LazyTensor(tracer=tracer, producer=prim)
    prim.outputs = [output]

    # Record on gradient tape
    from ..autodiff.tape import AutoDiffContext
    tape = AutoDiffContext.get_current_tape()
    if tape:
        tape.record(prim, list(inputs), [output])

    return output


# Arithmetic Primitives

@register_primitive
class AddPrimitive(Primitive):
    name = "add"

    def impl(self, x, y):
        return np.add(x, y)

    def abstract_eval(self, x_spec: TensorSpec, y_spec: TensorSpec) -> TensorSpec:
        out_shape = np.broadcast_shapes(x_spec.shape.dims, y_spec.shape.dims)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g):
        x, y = primals
        gx = _reduce_broadcast(g, x.shape)
        gy = _reduce_broadcast(g, y.shape)
        return gx, gy


@register_primitive
class SubPrimitive(Primitive):
    name = "sub"

    def impl(self, x, y):
        return np.subtract(x, y)

    def abstract_eval(self, x_spec: TensorSpec, y_spec: TensorSpec) -> TensorSpec:
        out_shape = np.broadcast_shapes(x_spec.shape.dims, y_spec.shape.dims)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g):
        x, y = primals
        gx = _reduce_broadcast(g, x.shape)
        gy = _reduce_broadcast(neg(g), y.shape)
        return gx, gy


@register_primitive
class MulPrimitive(Primitive):
    name = "mul"

    def impl(self, x, y):
        return np.multiply(x, y)

    def abstract_eval(self, x_spec: TensorSpec, y_spec: TensorSpec) -> TensorSpec:
        out_shape = np.broadcast_shapes(x_spec.shape.dims, y_spec.shape.dims)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g):
        x, y = primals
        gx = _reduce_broadcast(g * y, x.shape)
        gy = _reduce_broadcast(g * x, y.shape)
        return gx, gy


@register_primitive
class DivPrimitive(Primitive):
    name = "div"

    def impl(self, x, y):
        return np.divide(x, y)

    def abstract_eval(self, x_spec: TensorSpec, y_spec: TensorSpec) -> TensorSpec:
        out_shape = np.broadcast_shapes(x_spec.shape.dims, y_spec.shape.dims)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g):
        x, y = primals
        gx = _reduce_broadcast(g / y, x.shape)
        gy = _reduce_broadcast(neg(g * x / (y * y)), y.shape)
        return gx, gy


@register_primitive
class NegPrimitive(Primitive):
    name = "neg"

    def impl(self, x):
        return np.negative(x)

    def abstract_eval(self, x_spec: TensorSpec) -> TensorSpec:
        return x_spec

    def vjp(self, primals, tangents, g):
        return (neg(g),)


@register_primitive
class PowerPrimitive(Primitive):
    name = "power"

    def impl(self, x, y):
        return np.power(x, y)

    def abstract_eval(self, x_spec: TensorSpec, y_spec: TensorSpec) -> TensorSpec:
        out_shape = np.broadcast_shapes(x_spec.shape.dims, y_spec.shape.dims)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g):
        x, y = primals
        gx = g * y * power(x, y - array(1.0))
        gy = g * power(x, y) * log(x)
        return _reduce_broadcast(gx, x.shape), _reduce_broadcast(gy, y.shape)


@register_primitive
class MatMulPrimitive(Primitive):
    name = "matmul"

    def impl(self, x, y):
        return np.matmul(x, y)

    def abstract_eval(self, x_spec: TensorSpec, y_spec: TensorSpec) -> TensorSpec:
        x_shape = x_spec.shape.dims
        y_shape = y_spec.shape.dims

        if len(x_shape) == 2 and len(y_shape) == 2:
            out_shape = (x_shape[0], y_shape[1])
        else:
            batch_shape = np.broadcast_shapes(x_shape[:-2], y_shape[:-2])
            out_shape = batch_shape + (x_shape[-2], y_shape[-1])

        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g):
        x, y = primals
        gx = matmul(g, transpose(y))
        gy = matmul(transpose(x), g)
        return gx, gy


# Transcendental Primitives

@register_primitive
class ExpPrimitive(Primitive):
    name = "exp"

    def impl(self, x):
        return np.exp(x)

    def abstract_eval(self, x_spec: TensorSpec) -> TensorSpec:
        return x_spec

    def vjp(self, primals, tangents, g):
        x, = primals
        return (g * exp(x),)


@register_primitive
class LogPrimitive(Primitive):
    name = "log"

    def impl(self, x):
        return np.log(x)

    def abstract_eval(self, x_spec: TensorSpec) -> TensorSpec:
        return x_spec

    def vjp(self, primals, tangents, g):
        x, = primals
        return (g / x,)


@register_primitive
class SqrtPrimitive(Primitive):
    name = "sqrt"

    def impl(self, x):
        return np.sqrt(x)

    def abstract_eval(self, x_spec: TensorSpec) -> TensorSpec:
        return x_spec

    def vjp(self, primals, tangents, g):
        x, = primals
        return (g / (array(2.0) * sqrt(x)),)


@register_primitive
class SinPrimitive(Primitive):
    name = "sin"

    def impl(self, x):
        return np.sin(x)

    def abstract_eval(self, x_spec: TensorSpec) -> TensorSpec:
        return x_spec

    def vjp(self, primals, tangents, g):
        x, = primals
        return (g * cos(x),)


@register_primitive
class CosPrimitive(Primitive):
    name = "cos"

    def impl(self, x):
        return np.cos(x)

    def abstract_eval(self, x_spec: TensorSpec) -> TensorSpec:
        return x_spec

    def vjp(self, primals, tangents, g):
        x, = primals
        return (neg(g * sin(x)),)


@register_primitive
class TanhPrimitive(Primitive):
    name = "tanh"

    def impl(self, x):
        return np.tanh(x)

    def abstract_eval(self, x_spec: TensorSpec) -> TensorSpec:
        return x_spec

    def vjp(self, primals, tangents, g):
        x, = primals
        y = tanh(x)
        return (g * (array(1.0) - y * y),)


# Reduction Primitives

@register_primitive
class ReduceSumPrimitive(Primitive):
    name = "reduce_sum"

    def impl(self, x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)

    def abstract_eval(self, x_spec: TensorSpec, axis=None, keepdims=False) -> TensorSpec:
        shape = list(x_spec.shape.dims)
        if axis is None:
            out_shape = (1,) if keepdims else ()
        else:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis, reverse=True):
                if keepdims:
                    shape[ax] = 1
                else:
                    shape.pop(ax)
            out_shape = tuple(shape)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g, axis=None, keepdims=False):
        x, = primals
        if not keepdims and axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                g = expand_dims(g, ax)
        return (broadcast_to(g, x.shape),)


@register_primitive
class ReduceMeanPrimitive(Primitive):
    name = "reduce_mean"

    def impl(self, x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    def abstract_eval(self, x_spec: TensorSpec, axis=None, keepdims=False) -> TensorSpec:
        shape = list(x_spec.shape.dims)
        if axis is None:
            out_shape = (1,) if keepdims else ()
        else:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis, reverse=True):
                if keepdims:
                    shape[ax] = 1
                else:
                    shape.pop(ax)
            out_shape = tuple(shape)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g, axis=None, keepdims=False):
        x, = primals
        n = np.prod([x.shape[i] for i in (range(len(x.shape)) if axis is None else ([axis] if isinstance(axis, int) else axis))])
        if not keepdims and axis is not None:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                g = expand_dims(g, ax)
        return (broadcast_to(g / array(float(n)), x.shape),)


@register_primitive
class ReduceMaxPrimitive(Primitive):
    name = "reduce_max"

    def impl(self, x, axis=None, keepdims=False):
        return np.max(x, axis=axis, keepdims=keepdims)

    def abstract_eval(self, x_spec: TensorSpec, axis=None, keepdims=False) -> TensorSpec:
        shape = list(x_spec.shape.dims)
        if axis is None:
            out_shape = (1,) if keepdims else ()
        else:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis, reverse=True):
                if keepdims:
                    shape[ax] = 1
                else:
                    shape.pop(ax)
            out_shape = tuple(shape)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)


@register_primitive
class ReduceMinPrimitive(Primitive):
    name = "reduce_min"

    def impl(self, x, axis=None, keepdims=False):
        return np.min(x, axis=axis, keepdims=keepdims)

    def abstract_eval(self, x_spec: TensorSpec, axis=None, keepdims=False) -> TensorSpec:
        shape = list(x_spec.shape.dims)
        if axis is None:
            out_shape = (1,) if keepdims else ()
        else:
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis, reverse=True):
                if keepdims:
                    shape[ax] = 1
                else:
                    shape.pop(ax)
            out_shape = tuple(shape)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)


# Shape Primitives

@register_primitive
class ReshapePrimitive(Primitive):
    name = "reshape"

    def impl(self, x, shape):
        return np.reshape(x, shape)

    def abstract_eval(self, x_spec: TensorSpec, shape) -> TensorSpec:
        return TensorSpec(ShapeSpec(tuple(shape)), x_spec.dtype)

    def vjp(self, primals, tangents, g, shape):
        x, = primals
        return (reshape(g, x.shape),)


@register_primitive
class TransposePrimitive(Primitive):
    name = "transpose"

    def impl(self, x, axes=None):
        return np.transpose(x, axes)

    def abstract_eval(self, x_spec: TensorSpec, axes=None) -> TensorSpec:
        if axes is None:
            axes = tuple(reversed(range(len(x_spec.shape.dims))))
        new_shape = tuple(x_spec.shape.dims[i] for i in axes)
        return TensorSpec(ShapeSpec(new_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g, axes=None):
        if axes is None:
            axes = tuple(reversed(range(len(g.shape))))
        inv_axes = tuple(np.argsort(axes))
        return (transpose(g, inv_axes),)


@register_primitive
class BroadcastToPrimitive(Primitive):
    name = "broadcast_to"

    def impl(self, x, shape):
        return np.broadcast_to(x, shape)

    def abstract_eval(self, x_spec: TensorSpec, shape) -> TensorSpec:
        return TensorSpec(ShapeSpec(tuple(shape)), x_spec.dtype)

    def vjp(self, primals, tangents, g, shape):
        x, = primals
        return (_reduce_broadcast(g, x.shape),)


@register_primitive
class ExpandDimsPrimitive(Primitive):
    name = "expand_dims"

    def impl(self, x, axis):
        return np.expand_dims(x, axis)

    def abstract_eval(self, x_spec: TensorSpec, axis) -> TensorSpec:
        shape = list(x_spec.shape.dims)
        shape.insert(axis, 1)
        return TensorSpec(ShapeSpec(tuple(shape)), x_spec.dtype)

    def vjp(self, primals, tangents, g, axis):
        return (squeeze(g, axis),)


@register_primitive
class SqueezePrimitive(Primitive):
    name = "squeeze"

    def impl(self, x, axis=None):
        return np.squeeze(x, axis)

    def abstract_eval(self, x_spec: TensorSpec, axis=None) -> TensorSpec:
        shape = list(x_spec.shape.dims)
        if axis is None:
            shape = [d for d in shape if d != 1]
        else:
            if shape[axis] == 1:
                del shape[axis]
        return TensorSpec(ShapeSpec(tuple(shape)), x_spec.dtype)


# Utility functions

def _reduce_broadcast(g, target_shape):
    """Reduce gradient to match target shape."""
    g_shape = g.shape
    target_rank = len(target_shape)
    g_rank = len(g_shape)

    if g_rank > target_rank:
        g = reduce_sum(g, axis=tuple(range(g_rank - target_rank)))

    reduce_axes = []
    for i, (g_dim, t_dim) in enumerate(zip(g.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            reduce_axes.append(i)

    if reduce_axes:
        g = reduce_sum(g, axis=tuple(reduce_axes), keepdims=True)

    return g


# Public API

def add(x, y):
    return _create_primitive_op(AddPrimitive, _ensure_tensor(x), _ensure_tensor(y))

def sub(x, y):
    return _create_primitive_op(SubPrimitive, _ensure_tensor(x), _ensure_tensor(y))

def mul(x, y):
    return _create_primitive_op(MulPrimitive, _ensure_tensor(x), _ensure_tensor(y))

def div(x, y):
    return _create_primitive_op(DivPrimitive, _ensure_tensor(x), _ensure_tensor(y))

def neg(x):
    return _create_primitive_op(NegPrimitive, _ensure_tensor(x))

def power(x, y):
    return _create_primitive_op(PowerPrimitive, _ensure_tensor(x), _ensure_tensor(y))

def matmul(x, y):
    return _create_primitive_op(MatMulPrimitive, x, y)

def exp(x):
    return _create_primitive_op(ExpPrimitive, _ensure_tensor(x))

def log(x):
    return _create_primitive_op(LogPrimitive, _ensure_tensor(x))

def sqrt(x):
    return _create_primitive_op(SqrtPrimitive, _ensure_tensor(x))

def sin(x):
    return _create_primitive_op(SinPrimitive, _ensure_tensor(x))

def cos(x):
    return _create_primitive_op(CosPrimitive, _ensure_tensor(x))

def tanh(x):
    return _create_primitive_op(TanhPrimitive, _ensure_tensor(x))

def reduce_sum(x, axis=None, keepdims=False):
    return _create_primitive_op(ReduceSumPrimitive, x, axis=axis, keepdims=keepdims)

def reduce_mean(x, axis=None, keepdims=False):
    return _create_primitive_op(ReduceMeanPrimitive, x, axis=axis, keepdims=keepdims)

def reduce_max(x, axis=None, keepdims=False):
    return _create_primitive_op(ReduceMaxPrimitive, x, axis=axis, keepdims=keepdims)

def reduce_min(x, axis=None, keepdims=False):
    return _create_primitive_op(ReduceMinPrimitive, x, axis=axis, keepdims=keepdims)

def reshape(x, shape):
    return _create_primitive_op(ReshapePrimitive, x, shape=shape)

def transpose(x, axes=None):
    return _create_primitive_op(TransposePrimitive, x, axes=axes)

def broadcast_to(x, shape):
    return _create_primitive_op(BroadcastToPrimitive, x, shape=shape)

def expand_dims(x, axis):
    return _create_primitive_op(ExpandDimsPrimitive, x, axis=axis)

def squeeze(x, axis=None):
    return _create_primitive_op(SqueezePrimitive, x, axis=axis)


# Neural network operations

def relu(x):
    """ReLU activation."""
    return mul(x, array((x.materialize() > 0).astype(x.dtype)))

def sigmoid(x):
    """Sigmoid activation."""
    return div(array(1.0), add(array(1.0), exp(neg(x))))

def softmax(x, axis=-1):
    """Softmax activation."""
    x_max = reduce_max(x, axis=axis, keepdims=True)
    e_x = exp(sub(x, x_max))
    return div(e_x, reduce_sum(e_x, axis=axis, keepdims=True))
