# Distributed Tensor Algebra Engine (JAX-lite) - Technical Blueprint

> **Concepts covered:** §03 ml-engineering — `05-distributed-training`

## Executive Summary

This project implements a JAX-inspired distributed tensor algebra engine featuring lazy computation graphs, automatic differentiation, device mesh mapping, and parallel execution primitives. The system enables efficient execution of tensor computations across multiple devices with automatic sharding and gradient computation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Distributed Tensor Algebra Architecture                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         User API Layer                                │   │
│  │   jit()    grad()    vmap()    pmap()    psum()    mesh()           │   │
│  └─────────────────────────────────┬────────────────────────────────────┘   │
│                                    │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Lazy Execution Engine                            │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                 │   │
│  │  │ Trace Mode  │   │ Graph Build │   │  Lowering   │                 │   │
│  │  │ (Symbolic)  │──▶│  (DAG)      │──▶│  (to XLA)   │                 │   │
│  │  └─────────────┘   └─────────────┘   └─────────────┘                 │   │
│  └─────────────────────────────────┬────────────────────────────────────┘   │
│                                    │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     Auto-Differentiation Engine                       │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                 │   │
│  │  │  Forward    │   │  Gradient   │   │  VJP/JVP    │                 │   │
│  │  │  Trace      │──▶│  Tape       │──▶│  Rules      │                 │   │
│  │  └─────────────┘   └─────────────┘   └─────────────┘                 │   │
│  └─────────────────────────────────┬────────────────────────────────────┘   │
│                                    │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Sharding & Mesh Engine                           │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                 │   │
│  │  │   Device    │   │  Partition  │   │    SPMD     │                 │   │
│  │  │   Mesh      │──▶│  Strategy   │──▶│   Lowering  │                 │   │
│  │  └─────────────┘   └─────────────┘   └─────────────┘                 │   │
│  └─────────────────────────────────┬────────────────────────────────────┘   │
│                                    │                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Parallel Execution Runtime                         │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                 │   │
│  │  │   Task      │   │   Device    │   │  Collective │                 │   │
│  │  │  Scheduler  │──▶│   Executor  │──▶│    Ops      │                 │   │
│  │  └─────────────┘   └─────────────┘   └─────────────┘                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Lazy Tensor and Tracing System

```python
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

class DeviceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"

@dataclass
class Device:
    """Represents a compute device."""
    device_type: DeviceType
    device_id: int
    memory_bytes: int = 16 * 1024**3  # 16GB default

    def __str__(self):
        return f"{self.device_type.value}:{self.device_id}"

@dataclass
class ShapeSpec:
    """Shape specification with optional dynamic dimensions."""
    dims: Tuple[int, ...]

    @property
    def rank(self) -> int:
        return len(self.dims)

    @property
    def numel(self) -> int:
        result = 1
        for d in self.dims:
            if d < 0:
                return -1
            result *= d
        return result

    def __getitem__(self, idx):
        return self.dims[idx]

    def __len__(self):
        return len(self.dims)

@dataclass
class TensorSpec:
    """Type specification for tensors."""
    shape: ShapeSpec
    dtype: np.dtype
    device: Optional[Device] = None
    sharding: Optional['ShardingSpec'] = None

class Tracer:
    """
    Abstract tracer for symbolic execution.
    Used for tracing computation graphs during JIT/grad/vmap.
    """
    _counter = 0

    def __init__(self, spec: TensorSpec, name: str = None):
        self.spec = spec
        self.name = name or f"tracer_{Tracer._counter}"
        Tracer._counter += 1
        self._aval = spec  # Abstract value

    @property
    def shape(self):
        return self.spec.shape.dims

    @property
    def dtype(self):
        return self.spec.dtype

    def __repr__(self):
        return f"Tracer({self.name}, shape={self.shape}, dtype={self.dtype})"

class LazyTensor:
    """
    Lazy tensor that defers computation until explicitly evaluated.
    Tracks computation graph for JIT compilation and autodiff.
    """

    def __init__(self,
                 value: Optional[np.ndarray] = None,
                 tracer: Optional[Tracer] = None,
                 producer: Optional['Primitive'] = None):
        self._value = value
        self._tracer = tracer
        self._producer = producer
        self._materialized = value is not None

        if value is not None:
            self._shape = value.shape
            self._dtype = value.dtype
        elif tracer is not None:
            self._shape = tracer.shape
            self._dtype = tracer.dtype
        else:
            raise ValueError("Must provide either value or tracer")

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def is_materialized(self) -> bool:
        return self._materialized

    def materialize(self) -> np.ndarray:
        """Force evaluation and return concrete value."""
        if self._materialized:
            return self._value

        # Trace back through producers and evaluate
        if self._producer:
            self._value = self._producer.evaluate()
            self._materialized = True
            return self._value

        raise RuntimeError("Cannot materialize tensor without value or producer")

    def numpy(self) -> np.ndarray:
        """Convert to NumPy array."""
        return self.materialize()

    # Arithmetic operations (create lazy operations)
    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __matmul__(self, other):
        return matmul(self, other)

    def __neg__(self):
        return neg(self)

    def __sub__(self, other):
        return add(self, neg(other))

    def __truediv__(self, other):
        return div(self, other)

    def sum(self, axis=None, keepdims=False):
        return reduce_sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return reduce_mean(self, axis, keepdims)

    def reshape(self, *shape):
        return reshape(self, shape)

    def transpose(self, *axes):
        return transpose(self, axes if axes else None)

    @property
    def T(self):
        return self.transpose()

def array(data, dtype=None) -> LazyTensor:
    """Create a LazyTensor from array-like data."""
    if isinstance(data, LazyTensor):
        return data
    arr = np.array(data, dtype=dtype)
    return LazyTensor(value=arr)

def zeros(shape, dtype=np.float32) -> LazyTensor:
    return LazyTensor(value=np.zeros(shape, dtype=dtype))

def ones(shape, dtype=np.float32) -> LazyTensor:
    return LazyTensor(value=np.ones(shape, dtype=dtype))

def randn(*shape, dtype=np.float32) -> LazyTensor:
    return LazyTensor(value=np.random.randn(*shape).astype(dtype))
```

### 2. Primitive Operations and Graph Building

```python
from typing import Sequence
import weakref

class Primitive(ABC):
    """Base class for primitive operations."""

    name: str = "primitive"

    def __init__(self):
        self.inputs: List[LazyTensor] = []
        self.outputs: List[LazyTensor] = []
        self.params: Dict[str, Any] = {}

    @abstractmethod
    def impl(self, *args, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """Concrete implementation."""
        pass

    @abstractmethod
    def abstract_eval(self, *avals: TensorSpec, **kwargs) -> TensorSpec:
        """Compute output type from input types."""
        pass

    def vjp(self, primals, tangents, output_tangent):
        """Vector-Jacobian product for reverse-mode autodiff."""
        raise NotImplementedError(f"VJP not implemented for {self.name}")

    def jvp(self, primals, tangents):
        """Jacobian-vector product for forward-mode autodiff."""
        raise NotImplementedError(f"JVP not implemented for {self.name}")

    def batch(self, batched_args, batch_dims):
        """Batching rule for vmap."""
        raise NotImplementedError(f"Batching not implemented for {self.name}")

    def evaluate(self) -> np.ndarray:
        """Evaluate this primitive with its stored inputs."""
        concrete_inputs = [inp.materialize() for inp in self.inputs]
        return self.impl(*concrete_inputs, **self.params)

# Registry of primitives
_primitive_registry: Dict[str, Primitive] = {}

def register_primitive(prim_class):
    """Decorator to register a primitive."""
    _primitive_registry[prim_class.name] = prim_class
    return prim_class

@register_primitive
class AddPrimitive(Primitive):
    name = "add"

    def impl(self, x, y):
        return np.add(x, y)

    def abstract_eval(self, x_spec: TensorSpec, y_spec: TensorSpec) -> TensorSpec:
        # Broadcasting
        out_shape = np.broadcast_shapes(x_spec.shape.dims, y_spec.shape.dims)
        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g):
        x, y = primals
        # Gradient of add is 1 for both inputs, with broadcasting reduction
        gx = _reduce_broadcast(g, x.shape)
        gy = _reduce_broadcast(g, y.shape)
        return gx, gy

    def jvp(self, primals, tangents):
        x, y = primals
        dx, dy = tangents
        return x + y, dx + dy

    def batch(self, batched_args, batch_dims):
        return add(*batched_args), 0

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

    def jvp(self, primals, tangents):
        x, y = primals
        dx, dy = tangents
        return x * y, x * dy + dx * y

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
            # Batched matmul
            batch_shape = np.broadcast_shapes(x_shape[:-2], y_shape[:-2])
            out_shape = batch_shape + (x_shape[-2], y_shape[-1])

        return TensorSpec(ShapeSpec(out_shape), x_spec.dtype)

    def vjp(self, primals, tangents, g):
        x, y = primals
        # d(x @ y)/dx = g @ y.T
        # d(x @ y)/dy = x.T @ g
        gx = matmul(g, transpose(y))
        gy = matmul(transpose(x), g)
        return gx, gy

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
        # Broadcast gradient back to input shape
        if not keepdims and axis is not None:
            # Add back reduced dimensions
            if isinstance(axis, int):
                axis = (axis,)
            for ax in sorted(axis):
                g = expand_dims(g, ax)
        return (broadcast_to(g, x.shape),)

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

# Utility functions
def _reduce_broadcast(g, target_shape):
    """Reduce gradient to match target shape (undo broadcasting)."""
    # Find axes that were broadcast
    g_shape = g.shape
    target_rank = len(target_shape)
    g_rank = len(g_shape)

    if g_rank > target_rank:
        # Sum over leading dimensions
        g = reduce_sum(g, axis=tuple(range(g_rank - target_rank)))

    # Sum over dimensions that were broadcast (size 1 in target)
    reduce_axes = []
    for i, (g_dim, t_dim) in enumerate(zip(g.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            reduce_axes.append(i)

    if reduce_axes:
        g = reduce_sum(g, axis=tuple(reduce_axes), keepdims=True)

    return g

# Operation creation functions
def _create_primitive_op(prim_class, *inputs, **params) -> LazyTensor:
    """Create a lazy operation from a primitive."""
    prim = prim_class()
    prim.inputs = list(inputs)
    prim.params = params

    # Compute output spec
    input_specs = [
        TensorSpec(ShapeSpec(inp.shape), inp.dtype)
        for inp in inputs
    ]
    output_spec = prim.abstract_eval(*input_specs, **params)

    # Create tracer for lazy execution
    tracer = Tracer(output_spec)
    output = LazyTensor(tracer=tracer, producer=prim)
    prim.outputs = [output]

    return output

# Public API
def add(x, y): return _create_primitive_op(AddPrimitive, x, _ensure_tensor(y))
def mul(x, y): return _create_primitive_op(MulPrimitive, x, _ensure_tensor(y))
def matmul(x, y): return _create_primitive_op(MatMulPrimitive, x, y)
def neg(x): return mul(x, array(-1.0))
def div(x, y): return mul(x, _create_primitive_op(ReciprocalPrimitive, _ensure_tensor(y)))
def exp(x): return _create_primitive_op(ExpPrimitive, x)
def log(x): return _create_primitive_op(LogPrimitive, x)
def reduce_sum(x, axis=None, keepdims=False):
    return _create_primitive_op(ReduceSumPrimitive, x, axis=axis, keepdims=keepdims)
def reduce_mean(x, axis=None, keepdims=False):
    s = reduce_sum(x, axis, keepdims)
    n = np.prod([x.shape[i] for i in (range(len(x.shape)) if axis is None else ([axis] if isinstance(axis, int) else axis))])
    return s / array(float(n))
def reshape(x, shape): return _create_primitive_op(ReshapePrimitive, x, shape=shape)
def transpose(x, axes=None): return _create_primitive_op(TransposePrimitive, x, axes=axes)
def broadcast_to(x, shape): return _create_primitive_op(BroadcastPrimitive, x, shape=shape)
def expand_dims(x, axis): return _create_primitive_op(ExpandDimsPrimitive, x, axis=axis)

def _ensure_tensor(x):
    if isinstance(x, LazyTensor):
        return x
    return array(x)
```

### 3. Auto-Differentiation Engine

```python
from collections import defaultdict
import heapq

class GradientTape:
    """
    Records operations for reverse-mode automatic differentiation.
    Similar to TensorFlow's GradientTape or PyTorch's autograd.
    """

    def __init__(self):
        self.trace: List[Tuple[Primitive, List[LazyTensor], List[LazyTensor]]] = []
        self._watching: Set[int] = set()

    def watch(self, tensor: LazyTensor):
        """Mark a tensor to be watched for gradients."""
        self._watching.add(id(tensor))

    def record(self, prim: Primitive, inputs: List[LazyTensor], outputs: List[LazyTensor]):
        """Record an operation on the tape."""
        self.trace.append((prim, inputs, outputs))

    def gradient(self, target: LazyTensor, sources: List[LazyTensor]) -> List[LazyTensor]:
        """
        Compute gradients of target with respect to sources.
        Uses reverse-mode autodiff (backpropagation).
        """
        # Initialize output gradient
        grad_map: Dict[int, LazyTensor] = {id(target): ones(target.shape, target.dtype)}

        # Build topological order (reverse)
        tensor_to_op: Dict[int, Tuple[Primitive, List[LazyTensor]]] = {}
        for prim, inputs, outputs in self.trace:
            for out in outputs:
                tensor_to_op[id(out)] = (prim, inputs)

        # Reverse topological traversal
        for prim, inputs, outputs in reversed(self.trace):
            # Get output gradients
            out_grads = [grad_map.get(id(out)) for out in outputs]

            if all(g is None for g in out_grads):
                continue

            # Compute input gradients via VJP
            if len(outputs) == 1 and out_grads[0] is not None:
                input_grads = prim.vjp(inputs, None, out_grads[0], **prim.params)

                # Accumulate gradients
                for inp, grad in zip(inputs, input_grads):
                    if grad is not None:
                        if id(inp) in grad_map:
                            grad_map[id(inp)] = grad_map[id(inp)] + grad
                        else:
                            grad_map[id(inp)] = grad

        # Extract gradients for sources
        return [grad_map.get(id(src)) for src in sources]

class AutoDiffContext:
    """Context manager for automatic differentiation."""

    _tape_stack: List[GradientTape] = []

    @classmethod
    def get_current_tape(cls) -> Optional[GradientTape]:
        return cls._tape_stack[-1] if cls._tape_stack else None

    def __init__(self):
        self.tape = GradientTape()

    def __enter__(self):
        AutoDiffContext._tape_stack.append(self.tape)
        return self.tape

    def __exit__(self, *args):
        AutoDiffContext._tape_stack.pop()

def grad(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """
    Create a function that computes gradients.

    Args:
        fun: Function to differentiate
        argnums: Which arguments to differentiate with respect to

    Returns:
        Function that computes gradients
    """
    if isinstance(argnums, int):
        argnums = (argnums,)

    def grad_fun(*args, **kwargs):
        with AutoDiffContext() as tape:
            # Watch specified arguments
            for i in argnums:
                tape.watch(args[i])

            # Forward pass
            result = fun(*args, **kwargs)

            # Backward pass
            sources = [args[i] for i in argnums]
            grads = tape.gradient(result, sources)

            if len(grads) == 1:
                return grads[0]
            return grads

    return grad_fun

def value_and_grad(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """Return both function value and gradients."""
    if isinstance(argnums, int):
        argnums = (argnums,)

    def value_and_grad_fun(*args, **kwargs):
        with AutoDiffContext() as tape:
            for i in argnums:
                tape.watch(args[i])

            result = fun(*args, **kwargs)
            sources = [args[i] for i in argnums]
            grads = tape.gradient(result, sources)

            if len(grads) == 1:
                return result, grads[0]
            return result, grads

    return value_and_grad_fun

def vjp(fun: Callable, *primals):
    """
    Compute vector-Jacobian product.

    Returns:
        (output, vjp_fun) where vjp_fun takes output tangent and returns input tangents
    """
    with AutoDiffContext() as tape:
        for p in primals:
            tape.watch(p)

        output = fun(*primals)

    def vjp_fun(g):
        return tape.gradient(output, list(primals))

    return output, vjp_fun

def jvp(fun: Callable, primals: Tuple, tangents: Tuple):
    """
    Compute Jacobian-vector product (forward-mode autodiff).

    Returns:
        (output, output_tangent)
    """
    # This requires forward-mode AD implementation
    # For now, we can use the identity: JVP = (VJP)^T
    # But proper implementation would trace forward
    raise NotImplementedError("JVP requires forward-mode tracing")
```

### 4. Sharding and Device Mesh

```python
from typing import NamedTuple
import itertools

@dataclass
class Mesh:
    """
    Device mesh for distributed computation.
    Maps logical axes to physical devices.
    """
    devices: np.ndarray  # N-dimensional array of Device objects
    axis_names: Tuple[str, ...]

    def __post_init__(self):
        if len(self.axis_names) != self.devices.ndim:
            raise ValueError("Axis names must match device array dimensions")

    @property
    def shape(self) -> Dict[str, int]:
        return {name: self.devices.shape[i] for i, name in enumerate(self.axis_names)}

    @property
    def size(self) -> int:
        return self.devices.size

    def __getitem__(self, key):
        if isinstance(key, str):
            idx = self.axis_names.index(key)
            return self.devices.shape[idx]
        return self.devices[key]

def create_device_mesh(shape: Tuple[int, ...],
                       axis_names: Tuple[str, ...],
                       device_type: DeviceType = DeviceType.GPU) -> Mesh:
    """Create a device mesh with given shape."""
    total_devices = np.prod(shape)
    devices = np.array([
        Device(device_type, i) for i in range(total_devices)
    ]).reshape(shape)
    return Mesh(devices, axis_names)

class PartitionSpec(NamedTuple):
    """
    Specification for how to partition a tensor across mesh axes.
    None means replicated along that tensor dimension.
    """
    partitions: Tuple[Optional[str], ...]

    @staticmethod
    def create(*args) -> 'PartitionSpec':
        return PartitionSpec(args)

# Aliases
P = PartitionSpec.create

@dataclass
class ShardingSpec:
    """Full sharding specification for a tensor."""
    mesh: Mesh
    partition_spec: PartitionSpec

    def get_shard_shape(self, global_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute local shard shape from global shape."""
        shard_shape = list(global_shape)

        for i, axis_name in enumerate(self.partition_spec.partitions):
            if axis_name is not None:
                mesh_dim = self.mesh[axis_name]
                if shard_shape[i] % mesh_dim != 0:
                    raise ValueError(
                        f"Dimension {i} (size {shard_shape[i]}) not divisible "
                        f"by mesh axis {axis_name} (size {mesh_dim})"
                    )
                shard_shape[i] //= mesh_dim

        return tuple(shard_shape)

    def get_device_for_index(self, shard_index: Tuple[int, ...]) -> Device:
        """Get device that holds a particular shard."""
        mesh_coords = []
        for i, axis_name in enumerate(self.partition_spec.partitions):
            if axis_name is not None:
                mesh_axis = self.mesh.axis_names.index(axis_name)
                mesh_coords.append(shard_index[i])

        return self.mesh.devices[tuple(mesh_coords)]

class ShardedTensor:
    """
    Tensor distributed across devices according to sharding spec.
    """

    def __init__(self,
                 global_shape: Tuple[int, ...],
                 dtype: np.dtype,
                 sharding: ShardingSpec):
        self.global_shape = global_shape
        self.dtype = dtype
        self.sharding = sharding
        self.local_shape = sharding.get_shard_shape(global_shape)

        # Local shards stored per device
        self._shards: Dict[Device, np.ndarray] = {}

    def set_shard(self, device: Device, data: np.ndarray):
        """Set local shard on a device."""
        if data.shape != self.local_shape:
            raise ValueError(f"Shard shape {data.shape} doesn't match expected {self.local_shape}")
        self._shards[device] = data

    def get_shard(self, device: Device) -> np.ndarray:
        """Get local shard from a device."""
        return self._shards[device]

    def to_global(self) -> np.ndarray:
        """Gather all shards into global tensor."""
        # This is expensive - only for debugging
        result = np.zeros(self.global_shape, dtype=self.dtype)
        # ... gather logic
        return result

def shard_tensor(tensor: LazyTensor, sharding: ShardingSpec) -> ShardedTensor:
    """Shard a tensor according to specification."""
    data = tensor.materialize()
    sharded = ShardedTensor(data.shape, data.dtype, sharding)

    # Split data across devices
    mesh = sharding.mesh
    partition_spec = sharding.partition_spec

    # Compute split indices for each dimension
    for device in mesh.devices.flat:
        # Determine which slice this device gets
        slices = []
        device_idx = np.where(mesh.devices == device)

        for i, axis_name in enumerate(partition_spec.partitions):
            if axis_name is None:
                slices.append(slice(None))
            else:
                mesh_axis = mesh.axis_names.index(axis_name)
                mesh_idx = device_idx[mesh_axis][0]
                chunk_size = data.shape[i] // mesh[axis_name]
                start = mesh_idx * chunk_size
                end = start + chunk_size
                slices.append(slice(start, end))

        sharded.set_shard(device, data[tuple(slices)].copy())

    return sharded
```

### 5. Parallel Primitives

```python
class CollectiveOp(Enum):
    """Collective communication operations."""
    ALL_REDUCE = "all_reduce"
    ALL_GATHER = "all_gather"
    REDUCE_SCATTER = "reduce_scatter"
    ALL_TO_ALL = "all_to_all"
    BROADCAST = "broadcast"

@dataclass
class CollectiveDescriptor:
    """Description of a collective operation."""
    op_type: CollectiveOp
    mesh_axes: Tuple[str, ...]  # Which mesh axes participate
    reduction_op: str = "sum"  # For reductions: sum, mean, max, min

def psum(x: LazyTensor, axis_name: str) -> LazyTensor:
    """
    Sum tensor across devices along named axis.
    Used inside pmap for cross-device reduction.
    """
    return _create_collective_op(CollectiveOp.ALL_REDUCE, x, (axis_name,), "sum")

def pmean(x: LazyTensor, axis_name: str) -> LazyTensor:
    """Mean across devices along named axis."""
    return _create_collective_op(CollectiveOp.ALL_REDUCE, x, (axis_name,), "mean")

def pmax(x: LazyTensor, axis_name: str) -> LazyTensor:
    """Max across devices along named axis."""
    return _create_collective_op(CollectiveOp.ALL_REDUCE, x, (axis_name,), "max")

def all_gather(x: LazyTensor, axis_name: str, axis: int = 0) -> LazyTensor:
    """Gather tensor from all devices along axis."""
    return _create_collective_op(CollectiveOp.ALL_GATHER, x, (axis_name,))

def _create_collective_op(op_type: CollectiveOp, x: LazyTensor,
                          mesh_axes: Tuple[str, ...],
                          reduction_op: str = "sum") -> LazyTensor:
    """Create a collective operation."""
    # Implementation depends on runtime
    prim = CollectivePrimitive()
    prim.params = {
        'op_type': op_type,
        'mesh_axes': mesh_axes,
        'reduction_op': reduction_op
    }
    return _create_primitive_op(type(prim), x, **prim.params)

def pmap(fun: Callable,
         axis_name: str = 'batch',
         in_axes: Union[int, Tuple] = 0,
         out_axes: Union[int, Tuple] = 0,
         devices: List[Device] = None) -> Callable:
    """
    Parallel map: execute function in parallel across devices.

    Args:
        fun: Function to parallelize
        axis_name: Name for the parallel axis (used in collectives)
        in_axes: Which input axes to split across devices
        out_axes: Which output axes are split across devices
        devices: Devices to run on (default: all available)

    Returns:
        Parallelized function
    """
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    def pmapped_fun(*args):
        # Split inputs across devices
        n_devices = len(devices) if devices else 8  # Default
        split_args = []

        for arg, in_axis in zip(args, in_axes):
            if in_axis is None:
                # Replicate
                split_args.append([arg] * n_devices)
            else:
                # Split along axis
                splits = np.array_split(arg.materialize(), n_devices, axis=in_axis)
                split_args.append([array(s) for s in splits])

        # Execute in parallel
        results = []
        for device_idx in range(n_devices):
            device_args = [split[device_idx] for split in split_args]

            # Set axis name context for psum/pmean
            with _axis_name_context(axis_name, device_idx, n_devices):
                result = fun(*device_args)

            results.append(result)

        # Concatenate outputs
        if isinstance(out_axes, int):
            return array(np.concatenate([r.materialize() for r in results], axis=out_axes))
        else:
            # Multiple outputs
            return tuple(
                array(np.concatenate([r[i].materialize() for r in results], axis=out_axes[i]))
                for i in range(len(out_axes))
            )

    return pmapped_fun

def vmap(fun: Callable,
         in_axes: Union[int, Tuple] = 0,
         out_axes: Union[int, Tuple] = 0) -> Callable:
    """
    Vectorizing map: automatically batch a function.

    Args:
        fun: Function to vectorize
        in_axes: Which axes of inputs are the batch dimension
        out_axes: Which axes of outputs are the batch dimension

    Returns:
        Vectorized function
    """
    if isinstance(in_axes, int):
        in_axes = (in_axes,)

    def vmapped_fun(*args):
        # Get batch size
        batch_size = None
        for arg, in_axis in zip(args, in_axes):
            if in_axis is not None:
                batch_size = arg.shape[in_axis]
                break

        if batch_size is None:
            return fun(*args)

        # Transform each primitive to batched version
        # This requires tracing and transforming the computation
        # For now, simple loop implementation
        results = []
        for i in range(batch_size):
            batch_args = []
            for arg, in_axis in zip(args, in_axes):
                if in_axis is None:
                    batch_args.append(arg)
                else:
                    # Select slice along batch axis
                    slices = [slice(None)] * len(arg.shape)
                    slices[in_axis] = i
                    batch_args.append(array(arg.materialize()[tuple(slices)]))

            result = fun(*batch_args)
            results.append(result)

        # Stack results
        return array(np.stack([r.materialize() for r in results], axis=out_axes))

    return vmapped_fun

# Context for axis names in pmap
_axis_name_stack: List[Tuple[str, int, int]] = []

class _axis_name_context:
    def __init__(self, name, idx, size):
        self.name = name
        self.idx = idx
        self.size = size

    def __enter__(self):
        _axis_name_stack.append((self.name, self.idx, self.size))

    def __exit__(self, *args):
        _axis_name_stack.pop()
```

### 6. JIT Compilation

```python
import hashlib
from functools import wraps

class JittedFunction:
    """Compiled function with caching."""

    def __init__(self, fun: Callable, static_argnums: Tuple[int, ...] = ()):
        self.fun = fun
        self.static_argnums = static_argnums
        self._cache: Dict[str, Callable] = {}

    def __call__(self, *args, **kwargs):
        # Create cache key from argument shapes/dtypes
        cache_key = self._make_cache_key(args, kwargs)

        if cache_key not in self._cache:
            # Trace and compile
            compiled = self._compile(args, kwargs)
            self._cache[cache_key] = compiled

        return self._cache[cache_key](*args, **kwargs)

    def _make_cache_key(self, args, kwargs) -> str:
        """Create cache key from argument signatures."""
        key_parts = []

        for i, arg in enumerate(args):
            if i in self.static_argnums:
                # Include actual value for static args
                key_parts.append(f"static_{i}_{arg}")
            elif isinstance(arg, LazyTensor):
                key_parts.append(f"tensor_{arg.shape}_{arg.dtype}")
            else:
                key_parts.append(f"other_{type(arg)}")

        for k, v in sorted(kwargs.items()):
            if isinstance(v, LazyTensor):
                key_parts.append(f"{k}_{v.shape}_{v.dtype}")
            else:
                key_parts.append(f"{k}_{v}")

        key_str = "_".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _compile(self, args, kwargs) -> Callable:
        """Trace and compile the function."""
        # Create tracers for tensor arguments
        tracers = []
        for i, arg in enumerate(args):
            if i in self.static_argnums:
                tracers.append(arg)
            elif isinstance(arg, LazyTensor):
                spec = TensorSpec(ShapeSpec(arg.shape), arg.dtype)
                tracers.append(Tracer(spec, f"arg_{i}"))
            else:
                tracers.append(arg)

        # Trace the function
        # This would build an IR graph
        # For now, return the original function
        return self.fun

def jit(fun: Callable = None,
        static_argnums: Union[int, Tuple[int, ...]] = ()) -> Callable:
    """
    JIT compile a function.

    Args:
        fun: Function to compile
        static_argnums: Arguments that should be treated as compile-time constants

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
```

### 7. Enterprise Features

```python
class DeviceFailover:
    """Handle device failures gracefully."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.healthy_devices: Set[Device] = set(mesh.devices.flat)
        self.failed_devices: Set[Device] = set()

    def mark_failed(self, device: Device):
        """Mark a device as failed."""
        self.healthy_devices.discard(device)
        self.failed_devices.add(device)

    def get_replacement(self, failed: Device) -> Optional[Device]:
        """Find replacement device for failed one."""
        # Simple strategy: find any healthy device
        for device in self.healthy_devices:
            if device not in self.failed_devices:
                return device
        return None

    def redistribute_shards(self, tensor: ShardedTensor) -> ShardedTensor:
        """Redistribute shards after device failure."""
        new_shards = {}

        for device, shard in tensor._shards.items():
            if device in self.failed_devices:
                replacement = self.get_replacement(device)
                if replacement:
                    new_shards[replacement] = shard
            else:
                new_shards[device] = shard

        tensor._shards = new_shards
        return tensor

class GraphCache:
    """Cache compiled computation graphs."""

    def __init__(self, cache_dir: str = ".jax_cache"):
        self.cache_dir = cache_dir
        self.memory_cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self.memory_cache:
            return self.memory_cache[key]
        # Could also check disk cache
        return None

    def put(self, key: str, value: Any):
        self.memory_cache[key] = value

class SPMDPartitioner:
    """
    Single Program Multiple Data partitioner.
    Automatically partitions computation for distributed execution.
    """

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def partition_function(self, fun: Callable,
                          in_shardings: List[ShardingSpec],
                          out_shardings: List[ShardingSpec]) -> Callable:
        """
        Partition a function for SPMD execution.

        Each device runs the same program but on different data shards.
        Collective operations are inserted where needed.
        """
        def partitioned_fun(*args):
            # Convert inputs to sharded tensors
            sharded_inputs = []
            for arg, sharding in zip(args, in_shardings):
                if sharding:
                    sharded_inputs.append(shard_tensor(arg, sharding))
                else:
                    sharded_inputs.append(arg)

            # Execute on each device
            results = {}
            for device in self.mesh.devices.flat:
                local_args = []
                for inp in sharded_inputs:
                    if isinstance(inp, ShardedTensor):
                        local_args.append(array(inp.get_shard(device)))
                    else:
                        local_args.append(inp)

                results[device] = fun(*local_args)

            # Gather outputs
            # ... implementation depends on out_shardings
            return results

        return partitioned_fun
```

## Implementation Phases

### Phase 1: Core Tensor Operations (Weeks 1-3)
- [ ] Implement LazyTensor class
- [ ] Basic operations (add, mul, matmul)
- [ ] Shape inference and broadcasting
- [ ] Eager evaluation mode
- [ ] NumPy interoperability

### Phase 2: Auto-Differentiation (Weeks 4-6)
- [ ] Gradient tape implementation
- [ ] VJP rules for all primitives
- [ ] grad() function
- [ ] value_and_grad() function
- [ ] Higher-order derivatives

### Phase 3: JIT Compilation (Weeks 7-9)
- [ ] Tracing infrastructure
- [ ] Graph building from traces
- [ ] Compilation cache
- [ ] Static argument handling
- [ ] Graph optimization passes

### Phase 4: Vectorization (Weeks 10-11)
- [ ] vmap() implementation
- [ ] Batching rules for primitives
- [ ] Nested vmap support
- [ ] Integration with autodiff

### Phase 5: Parallel Primitives (Weeks 12-14)
- [ ] pmap() basic implementation
- [ ] Collective operations (psum, pmean)
- [ ] Device placement
- [ ] Cross-device gradients
- [ ] Nested pmap

### Phase 6: Sharding Engine (Weeks 15-17)
- [ ] Device mesh abstraction
- [ ] Sharding specifications
- [ ] Tensor partitioning
- [ ] SPMD lowering
- [ ] Automatic sharding propagation

### Phase 7: Enterprise Features (Weeks 18-20)
- [ ] Device failover
- [ ] Graph caching
- [ ] Performance profiling
- [ ] PyTorch interop (stretch)
- [ ] Multi-framework support

## Testing Strategy

### Unit Tests
```python
import pytest
import numpy as np
from numpy.testing import assert_allclose

class TestBasicOps:
    def test_add_broadcast(self):
        """Test broadcasting in addition."""
        a = array([[1, 2, 3]])  # (1, 3)
        b = array([[1], [2]])   # (2, 1)
        c = a + b

        expected = np.array([[2, 3, 4], [3, 4, 5]])
        assert_allclose(c.numpy(), expected)

    def test_matmul_shapes(self):
        """Test matrix multiplication shape inference."""
        a = randn(32, 64)
        b = randn(64, 128)
        c = a @ b

        assert c.shape == (32, 128)

    def test_reduction_keepdims(self):
        """Test reduction with keepdims."""
        a = randn(4, 5, 6)
        b = a.sum(axis=1, keepdims=True)

        assert b.shape == (4, 1, 6)

class TestAutoDiff:
    def test_grad_simple(self):
        """Test gradient of simple function."""
        def f(x):
            return (x ** 2).sum()

        x = array([1.0, 2.0, 3.0])
        grad_f = grad(f)
        g = grad_f(x)

        # d/dx sum(x^2) = 2x
        assert_allclose(g.numpy(), [2.0, 4.0, 6.0])

    def test_grad_matmul(self):
        """Test gradient through matrix multiplication."""
        def f(x, w):
            return (x @ w).sum()

        x = randn(3, 4)
        w = randn(4, 5)

        grad_f = grad(f, argnums=(0, 1))
        gx, gw = grad_f(x, w)

        # Check shapes
        assert gx.shape == x.shape
        assert gw.shape == w.shape

    def test_value_and_grad(self):
        """Test value_and_grad returns both."""
        def f(x):
            return (x ** 2).sum()

        x = array([1.0, 2.0])
        val, g = value_and_grad(f)(x)

        assert_allclose(val.numpy(), 5.0)
        assert_allclose(g.numpy(), [2.0, 4.0])

class TestParallel:
    def test_pmap_simple(self):
        """Test basic pmap execution."""
        def f(x):
            return x * 2

        x = randn(8, 4)  # 8 samples
        pmapped_f = pmap(f, devices=[Device(DeviceType.CPU, i) for i in range(8)])
        y = pmapped_f(x)

        assert_allclose(y.numpy(), x.numpy() * 2)

    def test_vmap_batch(self):
        """Test vmap batching."""
        def f(x):
            return x.sum()

        x = randn(10, 5)  # 10 batches of size 5
        vmapped_f = vmap(f)
        y = vmapped_f(x)

        assert y.shape == (10,)
```

### Integration Tests
```python
class TestEndToEnd:
    def test_mlp_training_step(self):
        """Test complete MLP forward/backward pass."""
        # Model parameters
        w1 = randn(784, 256)
        w2 = randn(256, 10)

        def mlp(x, w1, w2):
            h = x @ w1
            h = relu(h)
            return h @ w2

        def loss_fn(x, y, w1, w2):
            logits = mlp(x, w1, w2)
            return cross_entropy(logits, y).mean()

        # Single training step
        x = randn(32, 784)
        y = randint(0, 10, (32,))

        loss, (gw1, gw2) = value_and_grad(loss_fn, argnums=(2, 3))(x, y, w1, w2)

        assert loss.shape == ()
        assert gw1.shape == w1.shape
        assert gw2.shape == w2.shape

    def test_distributed_training(self):
        """Test training with pmap."""
        mesh = create_device_mesh((4, 2), ('data', 'model'))

        # ... distributed training loop
```

## Performance Targets

| Operation | Configuration | Target | vs NumPy |
|-----------|--------------|--------|----------|
| grad(sum(x^2)) | 1M elements | 2 ms | - |
| vmap(matmul) | 64 x 512x512 | 50 ms | 1.5x |
| pmap(forward) | 8 GPUs, batch 256 | 10 ms | - |
| JIT compilation | MLP | < 100 ms | - |

## Dependencies

- Python 3.8+
- NumPy
- Optional: CUDA (for GPU execution)
- Optional: MPI (for multi-node)

## References

- JAX: Autograd and XLA
- Dex: Typed Functional Array Language
- PyTorch distributed
- Mesh TensorFlow
