# ML Compiler (XLA-lite / TVM-lite) - Technical Blueprint

## Executive Summary

This project implements a production-grade ML compiler that transforms high-level tensor operations into optimized machine code for multiple backends (CPU, GPU, Triton, LLVM). The system features a typed SSA-based intermediate representation, comprehensive graph optimization passes, intelligent memory planning, and extensible code generation backends.

> **Concepts covered:** [§03 Triton programming](../../03-machine-learning-engineering/06-cuda-optimization/triton/triton-programming.md) (one of the lowering backends) · [§03 Custom CUDA kernels (fusion)](../../03-machine-learning-engineering/06-cuda-optimization/custom-kernels/cuda-custom-kernels.md) · [§03 Model optimization](../../03-machine-learning-engineering/04-production-ml/model-optimization/model-optimization.md). Pairs with [Project 18 (compiler/interpreter — IR/SSA techniques)](../18-compiler-interpreter/), [Project 37 (TorchDynamo-style frontend)](../37-dynamic-graph-runtime/), [Project 38 (dynamic graph execution)](../38-dynamic-graph-execution/), [Project 48 (kernel-graph scheduler — execution backend)](../48-multi-gpu-kernel-scheduler/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ML Compiler Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                 │
│  │   Frontend   │────▶│  IR Builder  │────▶│   High IR    │                 │
│  │  (PyTorch/   │     │  (Type Inf,  │     │  (Typed SSA, │                 │
│  │   NumPy)     │     │   Shapes)    │     │   Shapes)    │                 │
│  └──────────────┘     └──────────────┘     └──────┬───────┘                 │
│                                                    │                         │
│                                                    ▼                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Graph Optimization Pipeline                       │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │    │
│  │  │ Constant │─▶│   DCE    │─▶│ Operator │─▶│  Layout  │─▶│ Common │ │    │
│  │  │ Folding  │  │          │  │  Fusion  │  │  Optim   │  │SubElim │ │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  └────────┘ │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                    │                         │
│                                                    ▼                         │
│                              ┌──────────────────────────┐                    │
│                              │    Memory Planner        │                    │
│                              │  (Lifetime, Reuse,       │                    │
│                              │   Arena Allocation)      │                    │
│                              └────────────┬─────────────┘                    │
│                                           │                                  │
│                                           ▼                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Code Generation Backends                        │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │    │
│  │  │   CPU    │  │   CUDA   │  │  Triton  │  │   LLVM   │            │    │
│  │  │(AVX/SSE) │  │ (cuBLAS) │  │ (PTX)    │  │  (JIT)   │            │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Intermediate Representation (IR)

#### IR Design Principles

The IR uses Static Single Assignment (SSA) form with typed tensors and explicit shape information:

```python
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import hashlib

class DType(Enum):
    """Supported data types with bit widths."""
    FLOAT32 = ("f32", 32)
    FLOAT16 = ("f16", 16)
    BFLOAT16 = ("bf16", 16)
    INT32 = ("i32", 32)
    INT64 = ("i64", 64)
    BOOL = ("bool", 1)

    @property
    def bits(self) -> int:
        return self.value[1]

    @property
    def bytes(self) -> int:
        return (self.bits + 7) // 8

@dataclass(frozen=True)
class TensorType:
    """Fully typed tensor with shape and memory layout."""
    dtype: DType
    shape: Tuple[int, ...]  # -1 for dynamic dimensions
    layout: str = "NCHW"  # Memory layout
    device: str = "cpu"

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        """Number of elements (returns -1 if dynamic)."""
        result = 1
        for dim in self.shape:
            if dim < 0:
                return -1
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        n = self.numel
        return n * self.dtype.bytes if n > 0 else -1

    def is_compatible(self, other: 'TensorType') -> bool:
        """Check if types are compatible for operations."""
        if self.dtype != other.dtype:
            return False
        if len(self.shape) != len(other.shape):
            return False
        for s1, s2 in zip(self.shape, other.shape):
            if s1 != s2 and s1 != -1 and s2 != -1:
                return False
        return True

@dataclass
class Value:
    """SSA value representing a tensor or scalar."""
    id: int
    name: str
    type: TensorType
    defining_op: Optional['Operation'] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Value) and self.id == other.id

class OpCode(Enum):
    """All supported operations in the IR."""
    # Elementwise
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    NEG = "neg"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    TANH = "tanh"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"

    # Reduction
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"

    # Linear Algebra
    MATMUL = "matmul"
    CONV2D = "conv2d"
    BATCH_MATMUL = "batch_matmul"

    # Shape
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    BROADCAST = "broadcast"
    CONCAT = "concat"
    SLICE = "slice"

    # Memory
    LOAD = "load"
    STORE = "store"
    CONSTANT = "constant"

    # Control Flow
    CALL = "call"
    RETURN = "return"

@dataclass
class Operation:
    """Single operation in the IR graph."""
    opcode: OpCode
    inputs: List[Value]
    outputs: List[Value]
    attributes: Dict[str, Any]

    # Metadata for optimization
    fusion_group: Optional[int] = None
    schedule_order: int = 0

    def __post_init__(self):
        # Link outputs back to this operation
        for output in self.outputs:
            output.defining_op = self

    @property
    def is_elementwise(self) -> bool:
        return self.opcode in {
            OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV,
            OpCode.NEG, OpCode.EXP, OpCode.LOG, OpCode.SQRT,
            OpCode.TANH, OpCode.RELU, OpCode.GELU, OpCode.SILU
        }

    @property
    def is_reduction(self) -> bool:
        return self.opcode in {OpCode.SUM, OpCode.MEAN, OpCode.MAX, OpCode.MIN}

    def get_memory_access_pattern(self) -> str:
        """Returns memory access pattern for scheduling."""
        if self.is_elementwise:
            return "streaming"
        elif self.opcode in {OpCode.MATMUL, OpCode.BATCH_MATMUL}:
            return "tiled"
        elif self.opcode == OpCode.CONV2D:
            return "im2col"
        else:
            return "random"

class IRModule:
    """Top-level IR module containing functions."""

    def __init__(self, name: str = "main"):
        self.name = name
        self.functions: Dict[str, 'Function'] = {}
        self.global_constants: Dict[str, Any] = {}
        self._value_counter = 0

    def create_value(self, name: str, tensor_type: TensorType) -> Value:
        """Create a new SSA value."""
        value = Value(
            id=self._value_counter,
            name=f"%{name}_{self._value_counter}",
            type=tensor_type
        )
        self._value_counter += 1
        return value

    def add_function(self, func: 'Function'):
        self.functions[func.name] = func

    def compute_hash(self) -> str:
        """Compute content hash for caching."""
        content = str(self.functions) + str(self.global_constants)
        return hashlib.sha256(content.encode()).hexdigest()

class Function:
    """IR function with operations in SSA form."""

    def __init__(self, name: str, input_types: List[TensorType], output_types: List[TensorType]):
        self.name = name
        self.input_types = input_types
        self.output_types = output_types
        self.operations: List[Operation] = []
        self.inputs: List[Value] = []
        self.outputs: List[Value] = []

    def add_operation(self, op: Operation):
        self.operations.append(op)

    def verify(self) -> List[str]:
        """Verify IR correctness and return list of errors."""
        errors = []
        defined_values = set(v.id for v in self.inputs)

        for op in self.operations:
            # Check all inputs are defined
            for inp in op.inputs:
                if inp.id not in defined_values:
                    errors.append(f"Use before def: {inp.name} in {op.opcode}")

            # Add outputs to defined set
            for out in op.outputs:
                if out.id in defined_values:
                    errors.append(f"Redefinition of {out.name}")
                defined_values.add(out.id)

        return errors

    def to_string(self) -> str:
        """Pretty print the function."""
        lines = [f"func @{self.name}("]

        # Input arguments
        args = [f"  {v.name}: {v.type.dtype.value[0]}{list(v.type.shape)}"
                for v in self.inputs]
        lines.append(",\n".join(args))
        lines.append(") {")

        # Operations
        for op in self.operations:
            outputs = ", ".join(v.name for v in op.outputs)
            inputs = ", ".join(v.name for v in op.inputs)
            attrs = ", ".join(f"{k}={v}" for k, v in op.attributes.items())
            line = f"  {outputs} = {op.opcode.value}({inputs})"
            if attrs:
                line += f" {{{attrs}}}"
            lines.append(line)

        # Return
        returns = ", ".join(v.name for v in self.outputs)
        lines.append(f"  return {returns}")
        lines.append("}")

        return "\n".join(lines)
```

#### IR Builder

```python
class IRBuilder:
    """High-level builder for constructing IR graphs."""

    def __init__(self, module: IRModule):
        self.module = module
        self.current_function: Optional[Function] = None

    def begin_function(self, name: str, input_types: List[TensorType],
                       output_types: List[TensorType]) -> List[Value]:
        """Start building a new function."""
        func = Function(name, input_types, output_types)

        # Create input values
        for i, t in enumerate(input_types):
            value = self.module.create_value(f"arg{i}", t)
            func.inputs.append(value)

        self.current_function = func
        return func.inputs

    def end_function(self, outputs: List[Value]):
        """Finish building current function."""
        self.current_function.outputs = outputs
        self.module.add_function(self.current_function)
        self.current_function = None

    def _infer_output_type(self, opcode: OpCode, inputs: List[Value],
                           attributes: Dict) -> TensorType:
        """Type inference for operations."""
        if not inputs:
            raise ValueError(f"No inputs for {opcode}")

        input_type = inputs[0].type

        if opcode in {OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV}:
            # Binary elementwise - broadcast shapes
            if len(inputs) == 2:
                shape = self._broadcast_shapes(
                    inputs[0].type.shape, inputs[1].type.shape
                )
                return TensorType(input_type.dtype, shape,
                                  input_type.layout, input_type.device)
            return input_type

        elif opcode == OpCode.MATMUL:
            # Matrix multiplication
            a_shape = inputs[0].type.shape
            b_shape = inputs[1].type.shape
            if len(a_shape) == 2 and len(b_shape) == 2:
                out_shape = (a_shape[0], b_shape[1])
            else:
                # Batched matmul
                batch = a_shape[:-2]
                out_shape = batch + (a_shape[-2], b_shape[-1])
            return TensorType(input_type.dtype, out_shape,
                              input_type.layout, input_type.device)

        elif opcode in {OpCode.SUM, OpCode.MEAN, OpCode.MAX, OpCode.MIN}:
            # Reduction
            axis = attributes.get('axis', None)
            keepdim = attributes.get('keepdim', False)
            shape = list(input_type.shape)

            if axis is None:
                shape = (1,) if keepdim else ()
            else:
                if isinstance(axis, int):
                    axis = [axis]
                for ax in sorted(axis, reverse=True):
                    if keepdim:
                        shape[ax] = 1
                    else:
                        shape.pop(ax)

            return TensorType(input_type.dtype, tuple(shape),
                              input_type.layout, input_type.device)

        elif opcode == OpCode.RESHAPE:
            new_shape = tuple(attributes['shape'])
            return TensorType(input_type.dtype, new_shape,
                              input_type.layout, input_type.device)

        elif opcode == OpCode.TRANSPOSE:
            perm = attributes.get('perm', tuple(reversed(range(len(input_type.shape)))))
            new_shape = tuple(input_type.shape[i] for i in perm)
            return TensorType(input_type.dtype, new_shape,
                              input_type.layout, input_type.device)

        else:
            # Default: same as input
            return input_type

    def _broadcast_shapes(self, shape1: Tuple[int, ...],
                          shape2: Tuple[int, ...]) -> Tuple[int, ...]:
        """Compute broadcast output shape."""
        result = []
        max_len = max(len(shape1), len(shape2))

        # Pad shorter shape with 1s
        s1 = (1,) * (max_len - len(shape1)) + shape1
        s2 = (1,) * (max_len - len(shape2)) + shape2

        for d1, d2 in zip(s1, s2):
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                raise ValueError(f"Cannot broadcast {shape1} and {shape2}")

        return tuple(result)

    def emit(self, opcode: OpCode, inputs: List[Value],
             attributes: Dict[str, Any] = None) -> Value:
        """Emit a single-output operation."""
        attributes = attributes or {}
        output_type = self._infer_output_type(opcode, inputs, attributes)
        output = self.module.create_value(opcode.value, output_type)

        op = Operation(opcode, inputs, [output], attributes)
        self.current_function.add_operation(op)

        return output

    # Convenience methods
    def add(self, a: Value, b: Value) -> Value:
        return self.emit(OpCode.ADD, [a, b])

    def matmul(self, a: Value, b: Value) -> Value:
        return self.emit(OpCode.MATMUL, [a, b])

    def relu(self, x: Value) -> Value:
        return self.emit(OpCode.RELU, [x])

    def sum(self, x: Value, axis: int = None, keepdim: bool = False) -> Value:
        return self.emit(OpCode.SUM, [x], {'axis': axis, 'keepdim': keepdim})

    def reshape(self, x: Value, shape: Tuple[int, ...]) -> Value:
        return self.emit(OpCode.RESHAPE, [x], {'shape': shape})

    def transpose(self, x: Value, perm: Tuple[int, ...] = None) -> Value:
        return self.emit(OpCode.TRANSPOSE, [x], {'perm': perm})
```

### 2. Graph Optimization Pipeline

```python
from abc import ABC, abstractmethod
from typing import Set
import copy

class OptimizationPass(ABC):
    """Base class for optimization passes."""

    @abstractmethod
    def run(self, func: Function) -> bool:
        """Run the pass. Returns True if graph was modified."""
        pass

class ConstantFoldingPass(OptimizationPass):
    """Evaluate operations with constant inputs at compile time."""

    def run(self, func: Function) -> bool:
        import numpy as np

        modified = False
        constant_values: Dict[int, np.ndarray] = {}
        new_operations = []

        for op in func.operations:
            # Check if all inputs are constants
            if op.opcode == OpCode.CONSTANT:
                constant_values[op.outputs[0].id] = op.attributes['value']
                new_operations.append(op)
                continue

            all_const = all(inp.id in constant_values for inp in op.inputs)

            if all_const and op.is_elementwise:
                # Evaluate at compile time
                inputs = [constant_values[inp.id] for inp in op.inputs]
                result = self._evaluate_elementwise(op.opcode, inputs)

                # Replace with constant
                const_op = Operation(
                    OpCode.CONSTANT,
                    [],
                    op.outputs,
                    {'value': result}
                )
                new_operations.append(const_op)
                constant_values[op.outputs[0].id] = result
                modified = True
            else:
                new_operations.append(op)

        func.operations = new_operations
        return modified

    def _evaluate_elementwise(self, opcode: OpCode, inputs: List) -> Any:
        import numpy as np

        if opcode == OpCode.ADD:
            return inputs[0] + inputs[1]
        elif opcode == OpCode.MUL:
            return inputs[0] * inputs[1]
        elif opcode == OpCode.RELU:
            return np.maximum(inputs[0], 0)
        # ... other ops
        raise NotImplementedError(f"Constant folding for {opcode}")

class DeadCodeEliminationPass(OptimizationPass):
    """Remove operations whose outputs are never used."""

    def run(self, func: Function) -> bool:
        # Build use counts
        use_count: Dict[int, int] = {}
        for op in func.operations:
            for inp in op.inputs:
                use_count[inp.id] = use_count.get(inp.id, 0) + 1

        # Mark outputs as used
        for out in func.outputs:
            use_count[out.id] = use_count.get(out.id, 0) + 1

        # Remove dead operations (reverse order for correctness)
        new_operations = []
        removed = 0

        for op in reversed(func.operations):
            is_dead = all(use_count.get(out.id, 0) == 0 for out in op.outputs)

            if is_dead:
                removed += 1
                # Decrement use counts of inputs
                for inp in op.inputs:
                    use_count[inp.id] = use_count.get(inp.id, 1) - 1
            else:
                new_operations.append(op)

        func.operations = list(reversed(new_operations))
        return removed > 0

class OperatorFusionPass(OptimizationPass):
    """Fuse compatible operations to reduce memory traffic."""

    def __init__(self):
        self.fusion_id = 0

    def run(self, func: Function) -> bool:
        modified = False

        # Build output-to-operation map
        output_to_op: Dict[int, Operation] = {}
        for op in func.operations:
            for out in op.outputs:
                output_to_op[out.id] = op

        # Find fusion opportunities
        for op in func.operations:
            if op.opcode == OpCode.MATMUL:
                # MatMul + Add (bias) + Activation fusion
                fused = self._fuse_matmul_bias_activation(op, output_to_op, func)
                if fused:
                    modified = True

            elif op.is_elementwise:
                # Chain of elementwise ops
                fused = self._fuse_elementwise_chain(op, output_to_op, func)
                if fused:
                    modified = True

        return modified

    def _fuse_matmul_bias_activation(self, matmul_op: Operation,
                                      output_to_op: Dict, func: Function) -> bool:
        """Fuse MatMul + Bias + Activation into single kernel."""
        matmul_out = matmul_op.outputs[0]

        # Find uses of matmul output
        users = [op for op in func.operations
                 if any(inp.id == matmul_out.id for inp in op.inputs)]

        if len(users) != 1:
            return False

        bias_op = users[0]
        if bias_op.opcode != OpCode.ADD:
            return False

        # Check for activation after bias
        bias_out = bias_op.outputs[0]
        activation_users = [op for op in func.operations
                           if any(inp.id == bias_out.id for inp in op.inputs)]

        if len(activation_users) == 1 and activation_users[0].opcode in {OpCode.RELU, OpCode.GELU}:
            activation_op = activation_users[0]

            # Create fused operation
            self.fusion_id += 1
            matmul_op.fusion_group = self.fusion_id
            bias_op.fusion_group = self.fusion_id
            activation_op.fusion_group = self.fusion_id

            # Mark as fused MatMul-Bias-Act
            matmul_op.attributes['fused_bias'] = True
            matmul_op.attributes['fused_activation'] = activation_op.opcode.value

            return True

        return False

    def _fuse_elementwise_chain(self, start_op: Operation,
                                 output_to_op: Dict, func: Function) -> bool:
        """Fuse chains of elementwise operations."""
        if start_op.fusion_group is not None:
            return False

        chain = [start_op]
        current = start_op

        # Follow the chain
        while True:
            out_id = current.outputs[0].id
            users = [op for op in func.operations
                     if any(inp.id == out_id for inp in op.inputs)]

            if len(users) != 1 or not users[0].is_elementwise:
                break

            if users[0].fusion_group is not None:
                break

            chain.append(users[0])
            current = users[0]

        if len(chain) >= 2:
            self.fusion_id += 1
            for op in chain:
                op.fusion_group = self.fusion_id
            return True

        return False

class LayoutOptimizationPass(OptimizationPass):
    """Optimize memory layout for target architecture."""

    def __init__(self, target: str = "cpu"):
        self.target = target

    def run(self, func: Function) -> bool:
        modified = False

        for op in func.operations:
            if op.opcode == OpCode.CONV2D:
                # Choose optimal layout for convolution
                if self.target == "cuda":
                    # NHWC is often better for GPU
                    preferred = "NHWC"
                else:
                    # NCHW for CPU with vectorization
                    preferred = "NCHW"

                for out in op.outputs:
                    if out.type.layout != preferred:
                        # Insert layout transformation
                        out.type = TensorType(
                            out.type.dtype,
                            out.type.shape,
                            preferred,
                            out.type.device
                        )
                        modified = True

        return modified

class CommonSubexpressionEliminationPass(OptimizationPass):
    """Eliminate redundant computations."""

    def run(self, func: Function) -> bool:
        modified = False
        expr_to_value: Dict[str, Value] = {}
        replacements: Dict[int, Value] = {}

        for op in func.operations:
            # Create expression key
            expr_key = self._make_expr_key(op, replacements)

            if expr_key in expr_to_value:
                # Found common subexpression
                existing = expr_to_value[expr_key]
                for out in op.outputs:
                    replacements[out.id] = existing
                modified = True
            else:
                # New expression
                for out in op.outputs:
                    expr_to_value[expr_key] = out

        # Apply replacements
        if replacements:
            self._apply_replacements(func, replacements)

        return modified

    def _make_expr_key(self, op: Operation, replacements: Dict[int, Value]) -> str:
        """Create unique key for expression."""
        inputs = []
        for inp in op.inputs:
            actual_id = replacements.get(inp.id, inp).id if inp.id in replacements else inp.id
            inputs.append(str(actual_id))

        return f"{op.opcode.value}({','.join(inputs)}){op.attributes}"

    def _apply_replacements(self, func: Function, replacements: Dict[int, Value]):
        """Replace all uses of redundant values."""
        for op in func.operations:
            op.inputs = [replacements.get(inp.id, inp) for inp in op.inputs]

class OptimizationPipeline:
    """Execute sequence of optimization passes."""

    def __init__(self, target: str = "cpu"):
        self.passes = [
            ConstantFoldingPass(),
            DeadCodeEliminationPass(),
            CommonSubexpressionEliminationPass(),
            OperatorFusionPass(),
            LayoutOptimizationPass(target),
        ]

    def run(self, module: IRModule, max_iterations: int = 10) -> Dict[str, int]:
        """Run all passes until fixpoint."""
        stats = {pass_.__class__.__name__: 0 for pass_ in self.passes}

        for func in module.functions.values():
            for iteration in range(max_iterations):
                changed = False

                for pass_ in self.passes:
                    if pass_.run(func):
                        stats[pass_.__class__.__name__] += 1
                        changed = True

                if not changed:
                    break

        return stats
```

### 3. Memory Planner

```python
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class MemoryBlock:
    """Allocated memory block."""
    offset: int
    size: int
    value_id: int

    @property
    def end(self) -> int:
        return self.offset + self.size

@dataclass
class Lifetime:
    """Lifetime interval for a value."""
    value_id: int
    start: int  # First use (definition)
    end: int    # Last use
    size: int   # Size in bytes

class MemoryPlanner:
    """Plan memory allocation with reuse optimization."""

    def __init__(self, alignment: int = 64):
        self.alignment = alignment

    def plan(self, func: Function) -> Tuple[int, Dict[int, int]]:
        """
        Plan memory for function.
        Returns (total_size, value_id -> offset mapping).
        """
        # Compute lifetimes
        lifetimes = self._compute_lifetimes(func)

        # Sort by size descending (first-fit decreasing)
        lifetimes.sort(key=lambda x: x.size, reverse=True)

        # Allocate using interval-based reuse
        allocations: Dict[int, int] = {}
        free_list: List[MemoryBlock] = []
        peak_usage = 0

        # Process in execution order
        lifetimes_by_start = sorted(lifetimes, key=lambda x: x.start)

        for lt in lifetimes_by_start:
            # Free blocks that are no longer needed
            newly_free = []
            for value_id, offset in list(allocations.items()):
                other_lt = next(l for l in lifetimes if l.value_id == value_id)
                if other_lt.end < lt.start:
                    # This block is free
                    free_list.append(MemoryBlock(offset, other_lt.size, value_id))

            # Try to reuse freed memory
            offset = self._find_reuse(free_list, lt.size)

            if offset is None:
                # Allocate new memory
                offset = self._align(peak_usage)
                peak_usage = offset + lt.size

            allocations[lt.value_id] = offset

        return peak_usage, allocations

    def _compute_lifetimes(self, func: Function) -> List[Lifetime]:
        """Compute lifetime intervals for all values."""
        lifetimes: Dict[int, Lifetime] = {}

        # Input lifetimes
        for i, inp in enumerate(func.inputs):
            lifetimes[inp.id] = Lifetime(
                inp.id, 0, len(func.operations), inp.type.size_bytes
            )

        # Operation outputs and uses
        for i, op in enumerate(func.operations):
            # Define outputs
            for out in op.outputs:
                lifetimes[out.id] = Lifetime(
                    out.id, i, i, out.type.size_bytes
                )

            # Extend lifetime for inputs
            for inp in op.inputs:
                if inp.id in lifetimes:
                    lifetimes[inp.id].end = max(lifetimes[inp.id].end, i)

        # Output values live until the end
        for out in func.outputs:
            if out.id in lifetimes:
                lifetimes[out.id].end = len(func.operations)

        return list(lifetimes.values())

    def _find_reuse(self, free_list: List[MemoryBlock], size: int) -> Optional[int]:
        """Find suitable block in free list (best-fit)."""
        best = None
        best_waste = float('inf')

        for block in free_list:
            if block.size >= size:
                waste = block.size - size
                if waste < best_waste:
                    best = block
                    best_waste = waste

        if best:
            free_list.remove(best)
            # If there's remaining space, return it to free list
            if best_waste > 0:
                free_list.append(MemoryBlock(
                    best.offset + size, best_waste, -1
                ))
            return best.offset

        return None

    def _align(self, offset: int) -> int:
        """Align offset to boundary."""
        return (offset + self.alignment - 1) // self.alignment * self.alignment

class ArenaAllocator:
    """Simple arena allocator for runtime."""

    def __init__(self, size: int):
        self.buffer = bytearray(size)
        self.size = size

    def get_pointer(self, offset: int) -> memoryview:
        """Get pointer to offset in arena."""
        return memoryview(self.buffer)[offset:]

    def reset(self):
        """Reset arena for reuse."""
        pass  # No-op for simple arena
```

### 4. Code Generation Backends

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import textwrap

class CodegenBackend(ABC):
    """Base class for code generation backends."""

    @abstractmethod
    def generate(self, func: Function, memory_plan: Dict[int, int]) -> str:
        """Generate code for function."""
        pass

class CPUCodegen(CodegenBackend):
    """Generate optimized CPU code with SIMD."""

    def generate(self, func: Function, memory_plan: Dict[int, int]) -> str:
        code_lines = []

        # Function signature
        code_lines.append(self._generate_signature(func))
        code_lines.append("{")

        # Generate code for each operation
        for op in func.operations:
            if op.fusion_group is not None:
                # Skip if part of fusion (handled by group leader)
                if not self._is_fusion_leader(op, func):
                    continue
                code_lines.append(self._generate_fused_op(op, func, memory_plan))
            else:
                code_lines.append(self._generate_op(op, memory_plan))

        code_lines.append("}")
        return "\n".join(code_lines)

    def _generate_signature(self, func: Function) -> str:
        args = []
        for i, inp in enumerate(func.inputs):
            dtype = self._dtype_to_c(inp.type.dtype)
            args.append(f"{dtype}* __restrict__ arg{i}")

        for i, out in enumerate(func.outputs):
            dtype = self._dtype_to_c(out.type.dtype)
            args.append(f"{dtype}* __restrict__ out{i}")

        return f"void {func.name}({', '.join(args)})"

    def _generate_op(self, op: Operation, memory_plan: Dict[int, int]) -> str:
        if op.opcode == OpCode.ADD:
            return self._generate_elementwise_binary(op, "+", memory_plan)
        elif op.opcode == OpCode.MUL:
            return self._generate_elementwise_binary(op, "*", memory_plan)
        elif op.opcode == OpCode.MATMUL:
            return self._generate_matmul(op, memory_plan)
        elif op.opcode == OpCode.RELU:
            return self._generate_relu(op, memory_plan)
        else:
            return f"  // TODO: {op.opcode.value}"

    def _generate_elementwise_binary(self, op: Operation, operator: str,
                                      memory_plan: Dict[int, int]) -> str:
        a = op.inputs[0]
        b = op.inputs[1]
        out = op.outputs[0]

        numel = out.type.numel

        return textwrap.dedent(f'''
            // {op.opcode.value}
            #pragma omp parallel for simd
            for (int i = 0; i < {numel}; i++) {{
                out[{memory_plan[out.id]}//sizeof(float) + i] =
                    in[{memory_plan[a.id]}//sizeof(float) + i] {operator}
                    in[{memory_plan[b.id]}//sizeof(float) + i];
            }}
        ''')

    def _generate_matmul(self, op: Operation, memory_plan: Dict[int, int]) -> str:
        a = op.inputs[0]
        b = op.inputs[1]
        out = op.outputs[0]

        M, K = a.type.shape
        _, N = b.type.shape

        # Use blocked algorithm for cache efficiency
        return textwrap.dedent(f'''
            // matmul [{M}x{K}] x [{K}x{N}]
            const int BLOCK = 64;
            memset(&arena[{memory_plan[out.id]}], 0, {M*N}*sizeof(float));

            #pragma omp parallel for collapse(2)
            for (int ii = 0; ii < {M}; ii += BLOCK) {{
                for (int jj = 0; jj < {N}; jj += BLOCK) {{
                    for (int kk = 0; kk < {K}; kk += BLOCK) {{
                        for (int i = ii; i < min(ii+BLOCK, {M}); i++) {{
                            for (int k = kk; k < min(kk+BLOCK, {K}); k++) {{
                                float a_ik = arena[{memory_plan[a.id]}/sizeof(float) + i*{K}+k];
                                #pragma omp simd
                                for (int j = jj; j < min(jj+BLOCK, {N}); j++) {{
                                    arena[{memory_plan[out.id]}/sizeof(float) + i*{N}+j] +=
                                        a_ik * arena[{memory_plan[b.id]}/sizeof(float) + k*{N}+j];
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        ''')

    def _generate_relu(self, op: Operation, memory_plan: Dict[int, int]) -> str:
        x = op.inputs[0]
        out = op.outputs[0]
        numel = out.type.numel

        return textwrap.dedent(f'''
            // relu
            #pragma omp parallel for simd
            for (int i = 0; i < {numel}; i++) {{
                float val = arena[{memory_plan[x.id]}/sizeof(float) + i];
                arena[{memory_plan[out.id]}/sizeof(float) + i] = val > 0 ? val : 0;
            }}
        ''')

    def _dtype_to_c(self, dtype: DType) -> str:
        mapping = {
            DType.FLOAT32: "float",
            DType.FLOAT16: "__fp16",
            DType.INT32: "int32_t",
            DType.INT64: "int64_t",
        }
        return mapping.get(dtype, "float")

    def _is_fusion_leader(self, op: Operation, func: Function) -> bool:
        """Check if op is the first in its fusion group."""
        for other in func.operations:
            if other.fusion_group == op.fusion_group:
                return other == op
        return True

    def _generate_fused_op(self, leader: Operation, func: Function,
                           memory_plan: Dict[int, int]) -> str:
        # Collect all ops in fusion group
        group = [op for op in func.operations if op.fusion_group == leader.fusion_group]

        # Generate fused kernel
        return f"  // Fused group {leader.fusion_group}: {[op.opcode.value for op in group]}"

class CUDACodegen(CodegenBackend):
    """Generate CUDA kernel code."""

    def generate(self, func: Function, memory_plan: Dict[int, int]) -> str:
        kernels = []

        for op in func.operations:
            if op.opcode == OpCode.MATMUL:
                kernels.append(self._generate_matmul_kernel(op, memory_plan))
            elif op.is_elementwise:
                kernels.append(self._generate_elementwise_kernel(op, memory_plan))

        # Host function to launch kernels
        host_code = self._generate_host_launcher(func, memory_plan)

        return "\n\n".join(kernels + [host_code])

    def _generate_matmul_kernel(self, op: Operation, memory_plan: Dict[int, int]) -> str:
        a = op.inputs[0]
        b = op.inputs[1]
        out = op.outputs[0]

        M, K = a.type.shape
        _, N = b.type.shape

        return textwrap.dedent(f'''
            __global__ void matmul_kernel_{op.outputs[0].id}(
                float* __restrict__ A,
                float* __restrict__ B,
                float* __restrict__ C,
                int M, int N, int K) {{

                // Tile dimensions
                const int TILE_M = 32;
                const int TILE_N = 32;
                const int TILE_K = 32;

                __shared__ float As[TILE_M][TILE_K];
                __shared__ float Bs[TILE_K][TILE_N];

                int bx = blockIdx.x, by = blockIdx.y;
                int tx = threadIdx.x, ty = threadIdx.y;

                int row = by * TILE_M + ty;
                int col = bx * TILE_N + tx;

                float sum = 0.0f;

                for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {{
                    // Load tiles into shared memory
                    if (row < M && t * TILE_K + tx < K)
                        As[ty][tx] = A[row * K + t * TILE_K + tx];
                    else
                        As[ty][tx] = 0.0f;

                    if (t * TILE_K + ty < K && col < N)
                        Bs[ty][tx] = B[(t * TILE_K + ty) * N + col];
                    else
                        Bs[ty][tx] = 0.0f;

                    __syncthreads();

                    // Compute partial dot product
                    #pragma unroll
                    for (int k = 0; k < TILE_K; k++) {{
                        sum += As[ty][k] * Bs[k][tx];
                    }}

                    __syncthreads();
                }}

                if (row < M && col < N) {{
                    C[row * N + col] = sum;
                }}
            }}
        ''')

    def _generate_elementwise_kernel(self, op: Operation, memory_plan: Dict[int, int]) -> str:
        out = op.outputs[0]
        numel = out.type.numel

        if op.opcode == OpCode.ADD:
            op_code = "a[i] + b[i]"
        elif op.opcode == OpCode.MUL:
            op_code = "a[i] * b[i]"
        elif op.opcode == OpCode.RELU:
            op_code = "fmaxf(a[i], 0.0f)"
        else:
            op_code = "a[i]"

        return textwrap.dedent(f'''
            __global__ void elementwise_{op.opcode.value}_{out.id}(
                float* __restrict__ a,
                float* __restrict__ b,
                float* __restrict__ out,
                int n) {{

                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {{
                    out[i] = {op_code};
                }}
            }}
        ''')

    def _generate_host_launcher(self, func: Function, memory_plan: Dict[int, int]) -> str:
        return textwrap.dedent(f'''
            void {func.name}(float* arena, cudaStream_t stream) {{
                // Launch kernels
                // TODO: Generate kernel launch code
            }}
        ''')

class TritonCodegen(CodegenBackend):
    """Generate Triton (Python DSL for GPU) code."""

    def generate(self, func: Function, memory_plan: Dict[int, int]) -> str:
        code = ["import triton", "import triton.language as tl", ""]

        for op in func.operations:
            if op.opcode == OpCode.MATMUL:
                code.append(self._generate_matmul(op))

        return "\n".join(code)

    def _generate_matmul(self, op: Operation) -> str:
        return textwrap.dedent('''
            @triton.jit
            def matmul_kernel(
                a_ptr, b_ptr, c_ptr,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            ):
                pid_m = tl.program_id(0)
                pid_n = tl.program_id(1)

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                for k in range(0, K, BLOCK_K):
                    a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k)
                    b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k)
                    acc += tl.dot(a, b)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk

                c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
                tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        ''')
```

### 5. Compilation Cache and Enterprise Features

```python
import hashlib
import pickle
import os
from pathlib import Path
from typing import Optional, Dict, Any
import time

class CompilationCache:
    """Cache compiled kernels to avoid recompilation."""

    def __init__(self, cache_dir: str = ".mlc_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[Any]:
        """Get cached compilation result."""
        # Check memory cache first
        if key in self.memory_cache:
            return self.memory_cache[key]

        # Check disk cache
        cache_path = self.cache_dir / f"{key}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                result = pickle.load(f)
                self.memory_cache[key] = result
                return result

        return None

    def put(self, key: str, value: Any):
        """Store compilation result."""
        self.memory_cache[key] = value

        cache_path = self.cache_dir / f"{key}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(value, f)

    def make_key(self, module: IRModule, target: str, options: Dict) -> str:
        """Create cache key from compilation inputs."""
        content = f"{module.compute_hash()}-{target}-{sorted(options.items())}"
        return hashlib.sha256(content.encode()).hexdigest()

class KernelSelector:
    """A/B testing for kernel selection."""

    def __init__(self):
        self.kernel_stats: Dict[str, Dict[str, float]] = {}

    def select_kernel(self, op_key: str, variants: List[str]) -> str:
        """Select best kernel variant based on profiling data."""
        if op_key not in self.kernel_stats:
            # No data - use round-robin for exploration
            return variants[hash(op_key) % len(variants)]

        stats = self.kernel_stats[op_key]

        # Thompson sampling for exploration-exploitation
        import random

        best_variant = None
        best_sample = float('inf')

        for variant in variants:
            if variant in stats:
                # Sample from posterior
                mean = stats[variant]
                sample = random.gauss(mean, mean * 0.1)
            else:
                sample = 0  # Optimistic for unexplored

            if sample < best_sample:
                best_sample = sample
                best_variant = variant

        return best_variant or variants[0]

    def record_timing(self, op_key: str, variant: str, time_ms: float):
        """Record kernel execution time."""
        if op_key not in self.kernel_stats:
            self.kernel_stats[op_key] = {}

        # Exponential moving average
        alpha = 0.1
        old = self.kernel_stats[op_key].get(variant, time_ms)
        self.kernel_stats[op_key][variant] = alpha * time_ms + (1 - alpha) * old

class MLCompiler:
    """Main compiler interface."""

    def __init__(self, target: str = "cpu", cache_enabled: bool = True):
        self.target = target
        self.cache = CompilationCache() if cache_enabled else None
        self.kernel_selector = KernelSelector()

        # Select backend
        if target == "cpu":
            self.backend = CPUCodegen()
        elif target == "cuda":
            self.backend = CUDACodegen()
        elif target == "triton":
            self.backend = TritonCodegen()
        else:
            raise ValueError(f"Unknown target: {target}")

    def compile(self, module: IRModule, options: Dict = None) -> str:
        """Compile IR module to target code."""
        options = options or {}

        # Check cache
        if self.cache:
            cache_key = self.cache.make_key(module, self.target, options)
            cached = self.cache.get(cache_key)
            if cached:
                return cached

        # Run optimization pipeline
        optimizer = OptimizationPipeline(self.target)
        stats = optimizer.run(module)

        # Plan memory
        planner = MemoryPlanner()

        generated_code = []
        for func in module.functions.values():
            # Verify IR
            errors = func.verify()
            if errors:
                raise ValueError(f"IR verification failed: {errors}")

            # Memory planning
            total_mem, mem_plan = planner.plan(func)

            # Code generation
            code = self.backend.generate(func, mem_plan)
            generated_code.append(code)

        result = "\n\n".join(generated_code)

        # Store in cache
        if self.cache:
            self.cache.put(cache_key, result)

        return result

    def compile_function(self,
                         input_types: List[TensorType],
                         output_types: List[TensorType],
                         build_fn) -> str:
        """Compile a function defined via builder API."""
        module = IRModule()
        builder = IRBuilder(module)

        inputs = builder.begin_function("main", input_types, output_types)
        outputs = build_fn(builder, inputs)
        builder.end_function([outputs] if not isinstance(outputs, list) else outputs)

        return self.compile(module)
```

## Implementation Phases

### Phase 1: Core IR and Basic Operations (Weeks 1-3)
- [ ] Implement SSA-based IR data structures
- [ ] Type inference system with shape propagation
- [ ] Basic elementwise operations
- [ ] IR verification and pretty printing
- [ ] Unit tests for IR construction

### Phase 2: Graph Optimization (Weeks 4-6)
- [ ] Constant folding pass
- [ ] Dead code elimination
- [ ] Common subexpression elimination
- [ ] Basic operator fusion (elementwise chains)
- [ ] Optimization pass infrastructure

### Phase 3: Memory Planning (Weeks 7-8)
- [ ] Lifetime analysis
- [ ] Memory reuse algorithm
- [ ] Arena allocator
- [ ] Memory alignment handling
- [ ] Memory usage profiling

### Phase 4: CPU Code Generation (Weeks 9-11)
- [ ] Basic C code generation
- [ ] OpenMP parallelization
- [ ] SIMD vectorization hints
- [ ] Blocked algorithms for cache
- [ ] JIT compilation with TinyCC/LLVM

### Phase 5: GPU Code Generation (Weeks 12-14)
- [ ] CUDA kernel templates
- [ ] Shared memory optimization
- [ ] Triton backend
- [ ] Kernel fusion code generation
- [ ] Multi-stream execution

### Phase 6: Enterprise Features (Weeks 15-17)
- [ ] Compilation cache
- [ ] A/B kernel selection
- [ ] Auto-tuning infrastructure
- [ ] Multi-target compilation
- [ ] Performance profiling integration

### Phase 7: Advanced Optimizations (Weeks 18-20)
- [ ] Auto-differentiation (stretch)
- [ ] Operator decomposition
- [ ] Layout optimization
- [ ] Memory bandwidth optimization
- [ ] Advanced fusion patterns

## Testing Strategy

### Unit Tests
```python
import pytest

class TestIRConstruction:
    def test_type_inference_matmul(self):
        """Test shape inference for matrix multiplication."""
        module = IRModule()
        builder = IRBuilder(module)

        a_type = TensorType(DType.FLOAT32, (128, 256))
        b_type = TensorType(DType.FLOAT32, (256, 512))

        inputs = builder.begin_function("test", [a_type, b_type], [])
        output = builder.matmul(inputs[0], inputs[1])

        assert output.type.shape == (128, 512)
        assert output.type.dtype == DType.FLOAT32

    def test_broadcast_shapes(self):
        """Test broadcasting logic."""
        builder = IRBuilder(IRModule())

        assert builder._broadcast_shapes((3, 1), (1, 4)) == (3, 4)
        assert builder._broadcast_shapes((2, 3, 4), (4,)) == (2, 3, 4)

        with pytest.raises(ValueError):
            builder._broadcast_shapes((3, 4), (5, 4))

class TestOptimization:
    def test_constant_folding(self):
        """Test constant folding eliminates compile-time computations."""
        # Build graph with constants
        module = IRModule()
        # ... setup

        optimizer = ConstantFoldingPass()
        modified = optimizer.run(module.functions['main'])

        assert modified
        # Check that constant ops were folded

    def test_dead_code_elimination(self):
        """Test DCE removes unused operations."""
        # ... test implementation

    def test_operator_fusion(self):
        """Test matmul+bias+relu fusion."""
        # ... test implementation

class TestMemoryPlanning:
    def test_memory_reuse(self):
        """Test that non-overlapping lifetimes share memory."""
        # ... test implementation

    def test_alignment(self):
        """Test proper memory alignment."""
        planner = MemoryPlanner(alignment=64)

        assert planner._align(0) == 0
        assert planner._align(1) == 64
        assert planner._align(64) == 64
        assert planner._align(65) == 128
```

### Integration Tests
```python
class TestEndToEnd:
    def test_mlp_compilation(self):
        """Test compilation of MLP forward pass."""
        compiler = MLCompiler(target="cpu")

        def build_mlp(builder, inputs):
            x = inputs[0]
            w1, w2 = inputs[1], inputs[2]
            b1, b2 = inputs[3], inputs[4]

            # Layer 1
            h = builder.matmul(x, w1)
            h = builder.add(h, b1)
            h = builder.relu(h)

            # Layer 2
            out = builder.matmul(h, w2)
            out = builder.add(out, b2)

            return out

        input_types = [
            TensorType(DType.FLOAT32, (32, 784)),   # x
            TensorType(DType.FLOAT32, (784, 256)),  # w1
            TensorType(DType.FLOAT32, (256, 10)),   # w2
            TensorType(DType.FLOAT32, (256,)),      # b1
            TensorType(DType.FLOAT32, (10,)),       # b2
        ]
        output_types = [TensorType(DType.FLOAT32, (32, 10))]

        code = compiler.compile_function(input_types, output_types, build_mlp)

        assert "matmul" in code or "TILE" in code
        assert "relu" in code or "fmax" in code
```

### Performance Benchmarks
```python
class TestPerformance:
    def test_matmul_performance(self):
        """Benchmark matrix multiplication against baseline."""
        import numpy as np
        import time

        # Compile optimized kernel
        # ... compilation

        # Benchmark
        a = np.random.randn(1024, 1024).astype(np.float32)
        b = np.random.randn(1024, 1024).astype(np.float32)

        # Warmup
        for _ in range(5):
            np.matmul(a, b)

        # Measure NumPy baseline
        start = time.perf_counter()
        for _ in range(100):
            np.matmul(a, b)
        numpy_time = (time.perf_counter() - start) / 100

        # Measure compiled kernel
        # ... measure compiled version

        # Should be within 2x of NumPy (which uses MKL/OpenBLAS)
        # assert compiled_time < numpy_time * 2
```

## Performance Targets

| Operation | Size | Target Throughput | vs NumPy |
|-----------|------|-------------------|----------|
| MatMul | 1024x1024 | 100 GFLOPS | 0.5x |
| Conv2D | 64x64x256 | 50 GFLOPS | 0.5x |
| Elementwise | 10M elements | 50 GB/s | 1.0x |
| Fused MatMul+ReLU | 1024x1024 | 120 GFLOPS | - |

## Dependencies

- Python 3.8+
- NumPy (reference implementations)
- LLVM/Clang (JIT compilation, optional)
- CUDA Toolkit (GPU backend, optional)
- Triton (GPU backend, optional)
- pytest (testing)

## References

- XLA: Optimizing Compiler for Machine Learning
- TVM: An Automated End-to-End Optimizing Compiler
- Halide: A Language for Fast, Portable Computation
- MLIR: Multi-Level Intermediate Representation
- Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations
