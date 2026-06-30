# Dynamic Graph Execution Runtime - Technical Blueprint

## Executive Summary

A Python-native dynamic graph execution runtime inspired by TorchDynamo and Torch FX, designed for capturing Python code into optimizable intermediate representations (IR), performing graph-level transformations, and lowering to efficient backends. This system enables just-in-time compilation of dynamic Python ML code while maintaining Python semantics and debugging capabilities.

> **Concepts covered:** [§03 PyTorch deep learning (TorchDynamo / FX context)](../../03-machine-learning-engineering/02-deep-learning/pytorch/pytorch-deep-learning.md) · [§03 Custom layers](../../03-machine-learning-engineering/02-deep-learning/custom-layers/custom-layers.md). Pairs with [Project 38 (dynamic graph execution — eager-mode sibling)](../38-dynamic-graph-execution/), [Project 31 (ML compiler — lowering targets)](../31-ml-compiler/), [Project 35 (autograd from scratch)](../35-differentiable-programming/), [Project 18 (compiler/interpreter for the IR side)](../18-compiler-interpreter/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## System Architecture

### High-Level Architecture

```
+------------------+     +-------------------+     +------------------+
|   Python Code    |     |  Bytecode Tracer  |     |   Frame Guard    |
|   (decorated)    | --> | (monkey-patch)    | --> |   (recompile)    |
+------------------+     +-------------------+     +------------------+
                                   |
                                   v
+------------------+     +-------------------+     +------------------+
|   IR Builder     |     |    FX Graph       |     | Graph Optimizer  |
| (capture ops)    | --> | (nodes/edges)     | --> | (fuse/fold)      |
+------------------+     +-------------------+     +------------------+
                                                           |
                              +----------------------------+
                              v
+----------------------------------------------------------+
|                    Backend Lowering                       |
|  +-------------+  +-------------+  +------------------+  |
|  | PyTorch     |  | TensorRT    |  | Custom Kernels   |  |
|  | Eager       |  | Triton      |  | MLIR             |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
                              |
                              v
                    +-------------------+
                    |   Code Cache      |
                    | (compiled funcs)  |
                    +-------------------+
```

### Core Design Principles

1. **Python Semantics Preservation**: Exact match with eager execution behavior
2. **Incremental Capture**: Fall back to eager for unsupported operations
3. **Graph Breaks Minimization**: Maximize captured graph size
4. **Compile-Once-Run-Many**: Cache compiled functions by guard conditions
5. **Debuggability**: Maintain source-level debugging and profiling

## Component Design

### 1. Bytecode Tracer

```python
import dis
import types
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Set
from enum import Enum

class TracingMode(Enum):
    NONE = "none"
    SYMBOLIC = "symbolic"
    CONCRETE = "concrete"

@dataclass
class SymbolicValue:
    """Represents a value during symbolic tracing"""
    node_id: str
    dtype: Optional[str] = None
    shape: Optional[tuple] = None
    device: Optional[str] = None
    source: Optional[str] = None  # e.g., "input.0", "getattr.weight"

@dataclass
class TraceFrame:
    """Represents a frame during tracing"""
    code: types.CodeType
    locals: Dict[str, Any]
    globals: Dict[str, Any]
    instructions: List[dis.Instruction]
    ip: int = 0  # instruction pointer
    stack: List[Any] = field(default_factory=list)
    block_stack: List[tuple] = field(default_factory=list)

class BytecodeTracer:
    """
    Traces Python bytecode by interpreting it with symbolic values.
    Captures operations into an IR graph.
    """

    def __init__(self, graph_builder: 'GraphBuilder'):
        self.graph_builder = graph_builder
        self.frames: List[TraceFrame] = []
        self.mode = TracingMode.NONE
        self.graph_break_reasons: List[str] = []

        # Bytecode handlers
        self.handlers = {
            'LOAD_FAST': self._handle_load_fast,
            'STORE_FAST': self._handle_store_fast,
            'LOAD_GLOBAL': self._handle_load_global,
            'LOAD_ATTR': self._handle_load_attr,
            'STORE_ATTR': self._handle_store_attr,
            'CALL_FUNCTION': self._handle_call_function,
            'CALL_METHOD': self._handle_call_method,
            'BINARY_ADD': self._handle_binary_op('add'),
            'BINARY_MULTIPLY': self._handle_binary_op('mul'),
            'BINARY_SUBTRACT': self._handle_binary_op('sub'),
            'BINARY_TRUE_DIVIDE': self._handle_binary_op('div'),
            'BINARY_MATMUL': self._handle_binary_op('matmul'),
            'UNARY_NEGATIVE': self._handle_unary_op('neg'),
            'COMPARE_OP': self._handle_compare_op,
            'POP_JUMP_IF_FALSE': self._handle_conditional_jump,
            'POP_JUMP_IF_TRUE': self._handle_conditional_jump,
            'JUMP_FORWARD': self._handle_jump,
            'JUMP_ABSOLUTE': self._handle_jump,
            'FOR_ITER': self._handle_for_iter,
            'GET_ITER': self._handle_get_iter,
            'BUILD_TUPLE': self._handle_build_tuple,
            'BUILD_LIST': self._handle_build_list,
            'UNPACK_SEQUENCE': self._handle_unpack_sequence,
            'RETURN_VALUE': self._handle_return_value,
        }

    def trace(self, func: Callable, args: tuple, kwargs: dict) -> 'Graph':
        """Trace a function call and return captured graph"""
        self.mode = TracingMode.SYMBOLIC
        self.graph_builder.reset()

        # Create input nodes for arguments
        symbolic_args = []
        for i, arg in enumerate(args):
            if self._is_tensor(arg):
                sym = self.graph_builder.add_input(
                    name=f"arg_{i}",
                    shape=arg.shape,
                    dtype=str(arg.dtype),
                    device=str(arg.device)
                )
                symbolic_args.append(sym)
            else:
                symbolic_args.append(arg)

        symbolic_kwargs = {}
        for k, v in kwargs.items():
            if self._is_tensor(v):
                sym = self.graph_builder.add_input(
                    name=k,
                    shape=v.shape,
                    dtype=str(v.dtype),
                    device=str(v.device)
                )
                symbolic_kwargs[k] = sym
            else:
                symbolic_kwargs[k] = v

        # Create initial frame
        code = func.__code__
        frame = TraceFrame(
            code=code,
            locals=self._bind_arguments(code, symbolic_args, symbolic_kwargs),
            globals=func.__globals__,
            instructions=list(dis.get_instructions(code))
        )
        self.frames.append(frame)

        # Execute bytecode symbolically
        try:
            result = self._execute_frame(frame)

            # Mark output
            if isinstance(result, SymbolicValue):
                self.graph_builder.add_output(result)
            elif isinstance(result, (tuple, list)):
                for r in result:
                    if isinstance(r, SymbolicValue):
                        self.graph_builder.add_output(r)

            return self.graph_builder.finalize()

        except GraphBreakException as e:
            self.graph_break_reasons.append(str(e))
            raise

        finally:
            self.mode = TracingMode.NONE
            self.frames.pop()

    def _execute_frame(self, frame: TraceFrame) -> Any:
        """Execute bytecode instructions in a frame"""
        while frame.ip < len(frame.instructions):
            instr = frame.instructions[frame.ip]
            handler = self.handlers.get(instr.opname)

            if handler is None:
                raise GraphBreakException(f"Unsupported bytecode: {instr.opname}")

            # Execute instruction
            result = handler(frame, instr)

            if result is not None:
                return result

            frame.ip += 1

        raise RuntimeError("Frame completed without RETURN_VALUE")

    def _handle_load_fast(self, frame: TraceFrame, instr: dis.Instruction):
        """Load local variable onto stack"""
        value = frame.locals.get(instr.argval)
        frame.stack.append(value)

    def _handle_store_fast(self, frame: TraceFrame, instr: dis.Instruction):
        """Store TOS into local variable"""
        value = frame.stack.pop()
        frame.locals[instr.argval] = value

    def _handle_load_attr(self, frame: TraceFrame, instr: dis.Instruction):
        """Load attribute from TOS"""
        obj = frame.stack.pop()
        attr_name = instr.argval

        if isinstance(obj, SymbolicValue):
            # Symbolic getattr
            result = self.graph_builder.add_node(
                op='getattr',
                args=[obj],
                kwargs={'name': attr_name}
            )
            frame.stack.append(result)
        else:
            # Concrete getattr
            frame.stack.append(getattr(obj, attr_name))

    def _handle_call_function(self, frame: TraceFrame, instr: dis.Instruction):
        """Call function with positional arguments"""
        nargs = instr.argval
        args = [frame.stack.pop() for _ in range(nargs)][::-1]
        func = frame.stack.pop()

        result = self._trace_call(func, args, {})
        frame.stack.append(result)

    def _handle_call_method(self, frame: TraceFrame, instr: dis.Instruction):
        """Call method with arguments"""
        nargs = instr.argval
        args = [frame.stack.pop() for _ in range(nargs)][::-1]
        method = frame.stack.pop()
        obj = frame.stack.pop()

        if obj is not None:
            args = [obj] + args

        result = self._trace_call(method, args, {})
        frame.stack.append(result)

    def _trace_call(self, func: Callable, args: list, kwargs: dict) -> Any:
        """Trace a function call, either capturing or breaking"""

        # Check if all args are symbolic
        has_symbolic = any(isinstance(a, SymbolicValue) for a in args)
        has_symbolic = has_symbolic or any(isinstance(v, SymbolicValue) for v in kwargs.values())

        if not has_symbolic:
            # Pure Python call with concrete values
            return func(*args, **kwargs)

        # Check if function is traceable
        if self._is_traceable_op(func):
            # Map to graph operation
            return self._trace_op(func, args, kwargs)

        elif self._is_traceable_function(func):
            # Inline function tracing
            return self._inline_trace(func, args, kwargs)

        else:
            # Graph break
            raise GraphBreakException(f"Cannot trace function: {func}")

    def _trace_op(self, func: Callable, args: list, kwargs: dict) -> SymbolicValue:
        """Add operation node to graph"""
        op_name = self._get_op_name(func)

        # Infer output shape
        output_shape = self._infer_shape(op_name, args, kwargs)

        return self.graph_builder.add_node(
            op=op_name,
            args=args,
            kwargs=kwargs,
            shape=output_shape
        )

    def _handle_binary_op(self, op_name: str):
        """Create handler for binary operations"""
        def handler(frame: TraceFrame, instr: dis.Instruction):
            right = frame.stack.pop()
            left = frame.stack.pop()

            if isinstance(left, SymbolicValue) or isinstance(right, SymbolicValue):
                result = self.graph_builder.add_node(
                    op=op_name,
                    args=[left, right],
                    kwargs={}
                )
            else:
                # Concrete operation
                ops = {
                    'add': lambda a, b: a + b,
                    'mul': lambda a, b: a * b,
                    'sub': lambda a, b: a - b,
                    'div': lambda a, b: a / b,
                    'matmul': lambda a, b: a @ b,
                }
                result = ops[op_name](left, right)

            frame.stack.append(result)

        return handler

    def _handle_conditional_jump(self, frame: TraceFrame, instr: dis.Instruction):
        """Handle conditional jump - may cause graph break"""
        condition = frame.stack.pop()

        if isinstance(condition, SymbolicValue):
            # Data-dependent control flow - graph break
            raise GraphBreakException(
                f"Data-dependent control flow at {instr.offset}"
            )

        # Concrete condition
        should_jump = bool(condition)
        if instr.opname == 'POP_JUMP_IF_FALSE':
            should_jump = not should_jump

        if should_jump:
            # Find instruction at target
            for i, inst in enumerate(frame.instructions):
                if inst.offset == instr.argval:
                    frame.ip = i - 1  # -1 because we'll increment
                    return

    def _handle_return_value(self, frame: TraceFrame, instr: dis.Instruction) -> Any:
        """Handle function return"""
        return frame.stack.pop()


class GraphBreakException(Exception):
    """Raised when tracing cannot continue"""
    pass
```

### 2. IR Builder and Graph Representation

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
import uuid

@dataclass
class Node:
    """A node in the computation graph"""
    id: str
    op: str
    args: List[Any]  # Can be Node references or concrete values
    kwargs: Dict[str, Any]
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    device: Optional[str] = None
    users: Set[str] = field(default_factory=set)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        args_str = ', '.join(
            a.id if isinstance(a, Node) else repr(a)
            for a in self.args
        )
        return f"{self.id} = {self.op}({args_str})"

@dataclass
class Graph:
    """Computation graph with nodes and edges"""
    nodes: Dict[str, Node] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    name: str = "graph"

    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def topological_sort(self) -> List[Node]:
        """Return nodes in topological order"""
        visited = set()
        result = []

        def visit(node_id: str):
            if node_id in visited:
                return
            visited.add(node_id)

            node = self.nodes[node_id]
            for arg in node.args:
                if isinstance(arg, Node):
                    visit(arg.id)

            result.append(node)

        for output_id in self.outputs:
            visit(output_id)

        return result

    def print_tabular(self):
        """Print graph in tabular format like FX"""
        print(f"Graph: {self.name}")
        print("-" * 80)
        print(f"{'Node':<20} {'Op':<15} {'Args':<30} {'Shape':<15}")
        print("-" * 80)

        for node in self.topological_sort():
            args_str = ', '.join(
                a.id if isinstance(a, Node) else repr(a)[:20]
                for a in node.args
            )
            shape_str = str(node.shape) if node.shape else ''
            print(f"{node.id:<20} {node.op:<15} {args_str:<30} {shape_str:<15}")

    def to_python(self) -> str:
        """Generate Python code from graph"""
        lines = []
        lines.append(f"def {self.name}({', '.join(self.inputs)}):")

        for node in self.topological_sort():
            if node.op == 'input':
                continue

            args_str = ', '.join(
                a.id if isinstance(a, Node) else repr(a)
                for a in node.args
            )

            if node.kwargs:
                kwargs_str = ', '.join(f"{k}={repr(v)}" for k, v in node.kwargs.items())
                args_str = f"{args_str}, {kwargs_str}" if args_str else kwargs_str

            if node.op in ('add', 'mul', 'sub', 'div', 'matmul'):
                op_symbols = {'add': '+', 'mul': '*', 'sub': '-', 'div': '/', 'matmul': '@'}
                left = node.args[0].id if isinstance(node.args[0], Node) else repr(node.args[0])
                right = node.args[1].id if isinstance(node.args[1], Node) else repr(node.args[1])
                lines.append(f"    {node.id} = {left} {op_symbols[node.op]} {right}")
            else:
                lines.append(f"    {node.id} = {node.op}({args_str})")

        if len(self.outputs) == 1:
            lines.append(f"    return {self.outputs[0]}")
        else:
            lines.append(f"    return ({', '.join(self.outputs)})")

        return '\n'.join(lines)


class GraphBuilder:
    """Builds computation graph during tracing"""

    def __init__(self):
        self.graph = Graph()
        self.node_counter = 0
        self.name_map: Dict[str, str] = {}  # user name -> node id

    def reset(self):
        self.graph = Graph()
        self.node_counter = 0
        self.name_map = {}

    def _next_id(self, prefix: str = "n") -> str:
        self.node_counter += 1
        return f"{prefix}_{self.node_counter}"

    def add_input(self, name: str, shape: tuple = None,
                  dtype: str = None, device: str = None) -> SymbolicValue:
        """Add input placeholder node"""
        node_id = self._next_id("input")

        node = Node(
            id=node_id,
            op='input',
            args=[],
            kwargs={'name': name},
            shape=shape,
            dtype=dtype,
            device=device
        )

        self.graph.nodes[node_id] = node
        self.graph.inputs.append(node_id)
        self.name_map[name] = node_id

        return SymbolicValue(
            node_id=node_id,
            shape=shape,
            dtype=dtype,
            device=device,
            source=f"input.{name}"
        )

    def add_node(self, op: str, args: list, kwargs: dict,
                 shape: tuple = None) -> SymbolicValue:
        """Add operation node to graph"""
        node_id = self._next_id(op)

        # Convert SymbolicValues to Node references
        processed_args = []
        for arg in args:
            if isinstance(arg, SymbolicValue):
                node = self.graph.nodes[arg.node_id]
                processed_args.append(node)
                node.users.add(node_id)
            else:
                processed_args.append(arg)

        # Infer shape if not provided
        if shape is None:
            shape = self._infer_shape(op, processed_args, kwargs)

        # Infer dtype
        dtype = self._infer_dtype(op, processed_args, kwargs)

        node = Node(
            id=node_id,
            op=op,
            args=processed_args,
            kwargs=kwargs,
            shape=shape,
            dtype=dtype
        )

        self.graph.nodes[node_id] = node

        return SymbolicValue(
            node_id=node_id,
            shape=shape,
            dtype=dtype,
            source=f"{op}.{self.node_counter}"
        )

    def add_output(self, value: SymbolicValue):
        """Mark a value as graph output"""
        self.graph.outputs.append(value.node_id)

    def finalize(self) -> Graph:
        """Finalize and return the graph"""
        # Remove dead nodes
        self._eliminate_dead_code()
        return self.graph

    def _infer_shape(self, op: str, args: list, kwargs: dict) -> Optional[tuple]:
        """Infer output shape from operation and inputs"""
        # Get input shapes
        shapes = []
        for arg in args:
            if isinstance(arg, Node) and arg.shape:
                shapes.append(arg.shape)

        if not shapes:
            return None

        # Shape inference rules
        if op in ('add', 'mul', 'sub', 'div'):
            return self._broadcast_shapes(shapes)
        elif op == 'matmul':
            if len(shapes) == 2:
                return shapes[0][:-1] + shapes[1][-1:]
        elif op == 'reshape':
            return kwargs.get('shape')
        elif op == 'transpose':
            if shapes:
                return tuple(reversed(shapes[0]))
        elif op == 'sum':
            dim = kwargs.get('dim')
            if dim is not None and shapes:
                shape = list(shapes[0])
                if kwargs.get('keepdim', False):
                    shape[dim] = 1
                else:
                    del shape[dim]
                return tuple(shape)
        elif op == 'linear':
            # Linear layer: [batch, in] -> [batch, out]
            weight_shape = kwargs.get('weight_shape')
            if shapes and weight_shape:
                return shapes[0][:-1] + (weight_shape[0],)

        return shapes[0] if shapes else None

    def _eliminate_dead_code(self):
        """Remove nodes not reachable from outputs"""
        live = set()

        def mark_live(node_id: str):
            if node_id in live:
                return
            live.add(node_id)
            node = self.graph.nodes[node_id]
            for arg in node.args:
                if isinstance(arg, Node):
                    mark_live(arg.id)

        for output_id in self.graph.outputs:
            mark_live(output_id)

        dead = set(self.graph.nodes.keys()) - live
        for node_id in dead:
            del self.graph.nodes[node_id]
```

### 3. Graph Optimizer

```python
class GraphOptimizer:
    """
    Optimizes computation graphs through pattern matching and transformation.
    """

    def __init__(self):
        self.passes = [
            self._constant_folding,
            self._operator_fusion,
            self._common_subexpression_elimination,
            self._algebraic_simplification,
            self._dead_code_elimination,
            self._layout_optimization,
        ]

    def optimize(self, graph: Graph) -> Graph:
        """Apply optimization passes until convergence"""
        changed = True
        max_iterations = 10

        for iteration in range(max_iterations):
            if not changed:
                break
            changed = False

            for pass_fn in self.passes:
                new_graph, did_change = pass_fn(graph)
                if did_change:
                    graph = new_graph
                    changed = True

        return graph

    def _constant_folding(self, graph: Graph) -> Tuple[Graph, bool]:
        """Evaluate operations on constant inputs at compile time"""
        changed = False
        new_nodes = {}

        for node_id, node in graph.nodes.items():
            # Check if all inputs are constants
            all_constant = all(
                not isinstance(arg, Node) or arg.op == 'constant'
                for arg in node.args
            )

            if all_constant and node.op not in ('input', 'constant'):
                # Evaluate at compile time
                try:
                    args = [
                        arg.meta['value'] if isinstance(arg, Node) else arg
                        for arg in node.args
                    ]
                    result = self._evaluate_op(node.op, args, node.kwargs)

                    # Replace with constant node
                    new_node = Node(
                        id=node_id,
                        op='constant',
                        args=[],
                        kwargs={'value': result},
                        shape=result.shape if hasattr(result, 'shape') else None,
                        dtype=str(result.dtype) if hasattr(result, 'dtype') else None,
                        meta={'value': result}
                    )
                    new_nodes[node_id] = new_node
                    changed = True
                    continue

            new_nodes[node_id] = node

        if changed:
            graph.nodes = new_nodes

        return graph, changed

    def _operator_fusion(self, graph: Graph) -> Tuple[Graph, bool]:
        """Fuse compatible operations into single kernels"""
        changed = False
        fusions = []

        # Pattern: MatMul + BiasAdd
        for node_id, node in graph.nodes.items():
            if node.op == 'add':
                left = node.args[0] if isinstance(node.args[0], Node) else None
                if left and left.op == 'matmul' and len(left.users) == 1:
                    fusions.append({
                        'type': 'linear',
                        'matmul': left.id,
                        'add': node_id,
                        'output': node_id
                    })

        # Pattern: Conv + BatchNorm + ReLU
        for node_id, node in graph.nodes.items():
            if node.op == 'relu':
                bn = node.args[0] if isinstance(node.args[0], Node) else None
                if bn and bn.op == 'batch_norm' and len(bn.users) == 1:
                    conv = bn.args[0] if isinstance(bn.args[0], Node) else None
                    if conv and conv.op == 'conv2d' and len(conv.users) == 1:
                        fusions.append({
                            'type': 'conv_bn_relu',
                            'conv': conv.id,
                            'bn': bn.id,
                            'relu': node_id,
                            'output': node_id
                        })

        # Apply fusions
        for fusion in fusions:
            if fusion['type'] == 'linear':
                matmul_node = graph.nodes[fusion['matmul']]
                add_node = graph.nodes[fusion['add']]

                # Create fused node
                fused = Node(
                    id=fusion['output'],
                    op='linear_bias',
                    args=[matmul_node.args[0], matmul_node.args[1], add_node.args[1]],
                    kwargs={},
                    shape=add_node.shape,
                    dtype=add_node.dtype
                )

                # Update graph
                del graph.nodes[fusion['matmul']]
                graph.nodes[fusion['output']] = fused
                changed = True

            elif fusion['type'] == 'conv_bn_relu':
                conv_node = graph.nodes[fusion['conv']]
                bn_node = graph.nodes[fusion['bn']]

                # Create fused node
                fused = Node(
                    id=fusion['output'],
                    op='conv_bn_relu',
                    args=conv_node.args,
                    kwargs={
                        **conv_node.kwargs,
                        'bn_weight': bn_node.kwargs.get('weight'),
                        'bn_bias': bn_node.kwargs.get('bias'),
                        'bn_mean': bn_node.kwargs.get('running_mean'),
                        'bn_var': bn_node.kwargs.get('running_var'),
                    },
                    shape=bn_node.shape,
                    dtype=bn_node.dtype
                )

                del graph.nodes[fusion['conv']]
                del graph.nodes[fusion['bn']]
                graph.nodes[fusion['output']] = fused
                changed = True

        return graph, changed

    def _common_subexpression_elimination(self, graph: Graph) -> Tuple[Graph, bool]:
        """Eliminate redundant computations"""
        changed = False
        expr_to_node: Dict[str, str] = {}

        for node in graph.topological_sort():
            if node.op in ('input', 'constant'):
                continue

            # Create expression signature
            args_sig = tuple(
                a.id if isinstance(a, Node) else repr(a)
                for a in node.args
            )
            kwargs_sig = tuple(sorted(node.kwargs.items()))
            signature = (node.op, args_sig, kwargs_sig)

            if signature in expr_to_node:
                # Found duplicate - replace uses
                original_id = expr_to_node[signature]

                for other_node in graph.nodes.values():
                    new_args = []
                    for arg in other_node.args:
                        if isinstance(arg, Node) and arg.id == node.id:
                            new_args.append(graph.nodes[original_id])
                        else:
                            new_args.append(arg)
                    other_node.args = new_args

                # Update outputs
                graph.outputs = [
                    original_id if o == node.id else o
                    for o in graph.outputs
                ]

                del graph.nodes[node.id]
                changed = True
            else:
                expr_to_node[signature] = node.id

        return graph, changed

    def _algebraic_simplification(self, graph: Graph) -> Tuple[Graph, bool]:
        """Apply algebraic identities to simplify graph"""
        changed = False

        for node_id, node in list(graph.nodes.items()):
            # x + 0 = x
            if node.op == 'add':
                for i, arg in enumerate(node.args):
                    if not isinstance(arg, Node) and arg == 0:
                        other = node.args[1 - i]
                        self._replace_node(graph, node_id, other)
                        changed = True
                        break

            # x * 1 = x
            elif node.op == 'mul':
                for i, arg in enumerate(node.args):
                    if not isinstance(arg, Node) and arg == 1:
                        other = node.args[1 - i]
                        self._replace_node(graph, node_id, other)
                        changed = True
                        break

            # x * 0 = 0
            elif node.op == 'mul':
                for arg in node.args:
                    if not isinstance(arg, Node) and arg == 0:
                        # Replace with zero constant
                        zero_node = Node(
                            id=node_id,
                            op='constant',
                            args=[],
                            kwargs={'value': 0},
                            shape=node.shape
                        )
                        graph.nodes[node_id] = zero_node
                        changed = True
                        break

            # relu(relu(x)) = relu(x)
            elif node.op == 'relu':
                arg = node.args[0]
                if isinstance(arg, Node) and arg.op == 'relu':
                    self._replace_node(graph, node_id, arg)
                    changed = True

        return graph, changed

    def _replace_node(self, graph: Graph, old_id: str, new_node):
        """Replace all uses of old_id with new_node"""
        if isinstance(new_node, Node):
            new_id = new_node.id
        else:
            # Create constant node
            new_id = f"const_{len(graph.nodes)}"
            graph.nodes[new_id] = Node(
                id=new_id,
                op='constant',
                args=[],
                kwargs={'value': new_node}
            )

        # Update all users
        for node in graph.nodes.values():
            node.args = [
                graph.nodes[new_id] if (isinstance(a, Node) and a.id == old_id) else a
                for a in node.args
            ]

        # Update outputs
        graph.outputs = [new_id if o == old_id else o for o in graph.outputs]

        # Remove old node
        if old_id in graph.nodes:
            del graph.nodes[old_id]
```

### 4. Backend Lowering

```python
from abc import ABC, abstractmethod
from typing import Callable

class Backend(ABC):
    """Base class for execution backends"""

    @abstractmethod
    def compile(self, graph: Graph) -> Callable:
        """Compile graph to executable function"""
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class EagerBackend(Backend):
    """Execute graph using PyTorch eager mode"""

    def name(self) -> str:
        return "eager"

    def compile(self, graph: Graph) -> Callable:
        """Generate Python function that executes operations eagerly"""
        import torch

        # Map op names to PyTorch functions
        op_map = {
            'add': torch.add,
            'mul': torch.mul,
            'sub': torch.sub,
            'div': torch.div,
            'matmul': torch.matmul,
            'relu': torch.relu,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'softmax': lambda x, dim=-1: torch.softmax(x, dim=dim),
            'linear_bias': lambda x, w, b: torch.addmm(b, x, w.t()),
            'conv2d': torch.conv2d,
            'batch_norm': torch.batch_norm,
        }

        def execute(*args):
            # Map input names to values
            values = {}
            for i, input_id in enumerate(graph.inputs):
                values[input_id] = args[i]

            # Execute in topological order
            for node in graph.topological_sort():
                if node.op == 'input':
                    continue
                elif node.op == 'constant':
                    values[node.id] = node.kwargs['value']
                else:
                    # Gather arguments
                    node_args = [
                        values[a.id] if isinstance(a, Node) else a
                        for a in node.args
                    ]

                    # Execute operation
                    op_fn = op_map.get(node.op)
                    if op_fn:
                        values[node.id] = op_fn(*node_args, **node.kwargs)
                    else:
                        raise RuntimeError(f"Unknown op: {node.op}")

            # Gather outputs
            outputs = [values[o] for o in graph.outputs]
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        return execute


class TritonBackend(Backend):
    """Compile graph to Triton kernels"""

    def name(self) -> str:
        return "triton"

    def compile(self, graph: Graph) -> Callable:
        """Generate Triton kernel code"""
        import triton
        import triton.language as tl

        # Group fusible operations
        fusion_groups = self._find_fusion_groups(graph)

        kernels = []
        for group in fusion_groups:
            kernel_code = self._generate_kernel(group)
            kernel = self._compile_kernel(kernel_code)
            kernels.append(kernel)

        def execute(*args):
            values = {}
            for i, input_id in enumerate(graph.inputs):
                values[input_id] = args[i]

            for kernel, group in zip(kernels, fusion_groups):
                # Gather inputs for this kernel
                kernel_inputs = self._gather_kernel_inputs(group, values)

                # Allocate output
                output = self._allocate_output(group, kernel_inputs)

                # Launch kernel
                grid = self._compute_grid(group, kernel_inputs)
                kernel[grid](*kernel_inputs, output)

                # Store output
                values[group.output_id] = output

            outputs = [values[o] for o in graph.outputs]
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        return execute

    def _generate_kernel(self, fusion_group) -> str:
        """Generate Triton kernel for fused operations"""
        lines = []
        lines.append("@triton.jit")
        lines.append("def fused_kernel(")

        # Add parameters
        for i, inp in enumerate(fusion_group.inputs):
            lines.append(f"    input_{i}_ptr,")
        lines.append("    output_ptr,")
        lines.append("    N: tl.constexpr,")
        lines.append("):")

        # Compute program ID and offsets
        lines.append("    pid = tl.program_id(0)")
        lines.append("    BLOCK_SIZE: tl.constexpr = 1024")
        lines.append("    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)")
        lines.append("    mask = offsets < N")

        # Load inputs
        for i, inp in enumerate(fusion_group.inputs):
            lines.append(f"    x{i} = tl.load(input_{i}_ptr + offsets, mask=mask)")

        # Generate operations
        result_var = "x0"
        for op in fusion_group.operations:
            if op.op == 'add':
                lines.append(f"    {op.id} = {result_var} + x1")
            elif op.op == 'mul':
                lines.append(f"    {op.id} = {result_var} * x1")
            elif op.op == 'relu':
                lines.append(f"    {op.id} = tl.maximum({result_var}, 0)")
            result_var = op.id

        # Store output
        lines.append(f"    tl.store(output_ptr + offsets, {result_var}, mask=mask)")

        return '\n'.join(lines)


class AOTInductorBackend(Backend):
    """Compile graph to ahead-of-time compiled code via Inductor"""

    def name(self) -> str:
        return "aot_inductor"

    def compile(self, graph: Graph) -> Callable:
        """Lower to Inductor IR and compile"""
        # Convert graph to Inductor's IR
        inductor_graph = self._to_inductor_ir(graph)

        # Schedule operations
        scheduled = self._schedule(inductor_graph)

        # Generate C++/CUDA code
        code = self._codegen(scheduled)

        # Compile to shared library
        lib_path = self._compile_to_so(code)

        # Load and return
        return self._load_compiled(lib_path)
```

### 5. Compilation Cache and Guards

```python
import hashlib
from typing import Callable, Dict, Tuple, Any

class Guard:
    """Condition that must be true for cached code to be valid"""

    def __init__(self, check_fn: Callable[[], bool], description: str):
        self.check_fn = check_fn
        self.description = description

    def check(self) -> bool:
        return self.check_fn()


class ShapeGuard(Guard):
    """Guard on tensor shape"""

    def __init__(self, tensor_name: str, expected_shape: tuple):
        self.tensor_name = tensor_name
        self.expected_shape = expected_shape
        super().__init__(
            lambda: True,  # Actual check done in CacheEntry
            f"{tensor_name}.shape == {expected_shape}"
        )


class DtypeGuard(Guard):
    """Guard on tensor dtype"""

    def __init__(self, tensor_name: str, expected_dtype: str):
        self.tensor_name = tensor_name
        self.expected_dtype = expected_dtype
        super().__init__(
            lambda: True,
            f"{tensor_name}.dtype == {expected_dtype}"
        )


@dataclass
class CacheEntry:
    """Cached compilation result with guards"""
    compiled_fn: Callable
    guards: List[Guard]
    graph: Graph
    hit_count: int = 0
    compile_time_ms: float = 0

    def check_guards(self, args: tuple, kwargs: dict) -> bool:
        """Check if all guards pass for given inputs"""
        # Extract tensor info from args
        for i, arg in enumerate(args):
            if hasattr(arg, 'shape'):
                for guard in self.guards:
                    if isinstance(guard, ShapeGuard):
                        if guard.tensor_name == f"arg_{i}":
                            if arg.shape != guard.expected_shape:
                                return False
                    elif isinstance(guard, DtypeGuard):
                        if guard.tensor_name == f"arg_{i}":
                            if str(arg.dtype) != guard.expected_dtype:
                                return False
        return True


class CompilationCache:
    """
    Caches compiled functions keyed by source code and guards.
    Implements cache eviction and statistics.
    """

    def __init__(self, max_entries: int = 1000):
        self.cache: Dict[str, List[CacheEntry]] = {}
        self.max_entries = max_entries
        self.total_entries = 0
        self.hits = 0
        self.misses = 0

    def lookup(self, func: Callable, args: tuple, kwargs: dict) -> Optional[Callable]:
        """Look up cached compilation for function and inputs"""
        key = self._compute_key(func)

        if key not in self.cache:
            self.misses += 1
            return None

        # Check guards for each entry
        for entry in self.cache[key]:
            if entry.check_guards(args, kwargs):
                self.hits += 1
                entry.hit_count += 1
                return entry.compiled_fn

        self.misses += 1
        return None

    def insert(self, func: Callable, args: tuple, kwargs: dict,
               compiled_fn: Callable, graph: Graph, compile_time: float):
        """Insert compiled function into cache"""
        key = self._compute_key(func)

        # Create guards from args
        guards = []
        for i, arg in enumerate(args):
            if hasattr(arg, 'shape'):
                guards.append(ShapeGuard(f"arg_{i}", tuple(arg.shape)))
            if hasattr(arg, 'dtype'):
                guards.append(DtypeGuard(f"arg_{i}", str(arg.dtype)))

        entry = CacheEntry(
            compiled_fn=compiled_fn,
            guards=guards,
            graph=graph,
            compile_time_ms=compile_time
        )

        if key not in self.cache:
            self.cache[key] = []

        self.cache[key].append(entry)
        self.total_entries += 1

        # Evict if necessary
        if self.total_entries > self.max_entries:
            self._evict()

    def _compute_key(self, func: Callable) -> str:
        """Compute cache key from function"""
        # Use code object identity
        code = func.__code__
        return f"{code.co_filename}:{code.co_firstlineno}:{code.co_name}"

    def _evict(self):
        """Evict least recently used entries"""
        # Find entry with lowest hit count
        min_hits = float('inf')
        min_key = None
        min_idx = 0

        for key, entries in self.cache.items():
            for idx, entry in enumerate(entries):
                if entry.hit_count < min_hits:
                    min_hits = entry.hit_count
                    min_key = key
                    min_idx = idx

        if min_key:
            del self.cache[min_key][min_idx]
            if not self.cache[min_key]:
                del self.cache[min_key]
            self.total_entries -= 1

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics"""
        return {
            'total_entries': self.total_entries,
            'num_functions': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if self.hits + self.misses > 0 else 0,
        }
```

### 6. Decorator API

```python
from functools import wraps
import time

class DynamicCompiler:
    """
    Main entry point for dynamic compilation.
    Similar to torch.compile() API.
    """

    def __init__(self, backend: str = "eager", mode: str = "default"):
        self.backend_name = backend
        self.mode = mode

        # Initialize components
        self.graph_builder = GraphBuilder()
        self.tracer = BytecodeTracer(self.graph_builder)
        self.optimizer = GraphOptimizer()
        self.cache = CompilationCache()

        # Initialize backend
        if backend == "eager":
            self.backend = EagerBackend()
        elif backend == "triton":
            self.backend = TritonBackend()
        elif backend == "aot_inductor":
            self.backend = AOTInductorBackend()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def __call__(self, func: Callable) -> Callable:
        """Decorator to compile a function"""
        @wraps(func)
        def compiled_func(*args, **kwargs):
            # Check cache
            cached = self.cache.lookup(func, args, kwargs)
            if cached:
                return cached(*args, **kwargs)

            # Trace function
            try:
                start = time.time()

                graph = self.tracer.trace(func, args, kwargs)

                # Optimize graph
                optimized = self.optimizer.optimize(graph)

                # Compile to backend
                compiled = self.backend.compile(optimized)

                compile_time = (time.time() - start) * 1000

                # Cache result
                self.cache.insert(
                    func, args, kwargs,
                    compiled, optimized, compile_time
                )

                return compiled(*args, **kwargs)

            except GraphBreakException as e:
                # Fall back to eager execution
                print(f"Graph break: {e}. Falling back to eager.")
                return func(*args, **kwargs)

        # Attach metadata
        compiled_func._compiled = True
        compiled_func._compiler = self

        return compiled_func


def compile(func=None, *, backend="eager", mode="default"):
    """
    Decorator to compile a function for optimized execution.

    Example:
        @compile(backend="triton")
        def forward(x, y):
            return x + y * 2
    """
    compiler = DynamicCompiler(backend=backend, mode=mode)

    if func is not None:
        return compiler(func)
    return compiler


# Convenience alias
optimize = compile
```

## Enterprise Features

### Partial Graph Capture

```python
class PartialTracer(BytecodeTracer):
    """
    Traces partial graphs when full tracing fails.
    Splits at graph breaks and captures multiple subgraphs.
    """

    def trace_with_breaks(self, func: Callable, args: tuple, kwargs: dict) -> List[Graph]:
        """Trace function, splitting at graph breaks"""
        graphs = []
        current_graph = None

        try:
            graph = self.trace(func, args, kwargs)
            graphs.append(graph)
        except GraphBreakException:
            # Capture partial graph
            current_graph = self.graph_builder.finalize()
            if len(current_graph.nodes) > 1:
                graphs.append(current_graph)

            # Continue from break point
            remaining = self._get_remaining_code()
            if remaining:
                sub_graphs = self.trace_with_breaks(
                    remaining, self._get_current_values(), {}
                )
                graphs.extend(sub_graphs)

        return graphs


class FallbackHandler:
    """Handles fallback to eager execution for unsupported operations"""

    def __init__(self):
        self.fallback_ops = set()
        self.fallback_counts: Dict[str, int] = {}

    def record_fallback(self, op_name: str, reason: str):
        """Record that an operation fell back to eager"""
        self.fallback_ops.add(op_name)
        self.fallback_counts[op_name] = self.fallback_counts.get(op_name, 0) + 1

    def get_report(self) -> str:
        """Generate fallback report"""
        lines = ["Fallback Report", "=" * 40]

        for op, count in sorted(self.fallback_counts.items(), key=lambda x: -x[1]):
            lines.append(f"{op}: {count} fallbacks")

        return '\n'.join(lines)
```

### Model Caching

```python
import pickle
import os

class ModelCache:
    """
    Persistent cache for compiled models.
    Saves compiled graphs to disk for faster loading.
    """

    def __init__(self, cache_dir: str = ".dynamo_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save_model(self, model_id: str, graphs: List[Graph],
                   metadata: Dict[str, Any]):
        """Save compiled model to disk"""
        path = os.path.join(self.cache_dir, f"{model_id}.pkl")

        data = {
            'graphs': [self._serialize_graph(g) for g in graphs],
            'metadata': metadata,
            'version': '1.0',
        }

        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_model(self, model_id: str) -> Optional[Tuple[List[Graph], Dict]]:
        """Load compiled model from disk"""
        path = os.path.join(self.cache_dir, f"{model_id}.pkl")

        if not os.path.exists(path):
            return None

        with open(path, 'rb') as f:
            data = pickle.load(f)

        graphs = [self._deserialize_graph(g) for g in data['graphs']]
        return graphs, data['metadata']

    def _serialize_graph(self, graph: Graph) -> dict:
        """Serialize graph to dictionary"""
        return {
            'name': graph.name,
            'nodes': {
                nid: {
                    'op': n.op,
                    'args': [(a.id if isinstance(a, Node) else a) for a in n.args],
                    'kwargs': n.kwargs,
                    'shape': n.shape,
                    'dtype': n.dtype,
                }
                for nid, n in graph.nodes.items()
            },
            'inputs': graph.inputs,
            'outputs': graph.outputs,
        }
```

## Development Phases

### Phase 1: Core Tracing (Weeks 1-3)
- Bytecode instruction handlers for basic operations
- Symbolic value tracking
- Simple expression graph building
- Basic binary operations (add, mul, sub, div)
- Input/output handling

### Phase 2: IR and Graph (Weeks 4-5)
- Complete node representation
- Shape inference
- Type inference
- Graph serialization
- Pretty printing and visualization

### Phase 3: Graph Optimization (Weeks 6-7)
- Constant folding
- Common subexpression elimination
- Algebraic simplification
- Dead code elimination
- Basic operator fusion

### Phase 4: Backend Integration (Weeks 8-9)
- Eager PyTorch backend
- Triton backend for pointwise ops
- Code generation infrastructure
- Kernel caching

### Phase 5: Caching and Guards (Weeks 10-11)
- Compilation cache
- Shape guards
- Dtype guards
- Cache eviction
- Statistics

### Phase 6: Enterprise Features (Week 12+)
- Partial graph capture
- Fallback handling
- Model caching
- AOTAutograd integration
- Advanced fusion patterns

## Testing Strategy

### Unit Tests
- Bytecode handler correctness
- Shape inference rules
- Optimization passes
- Code generation

### Integration Tests
- End-to-end tracing of simple functions
- Multi-operation graphs
- Nested function calls
- Control flow handling

### Correctness Tests
- Numerical accuracy vs eager
- Gradient correctness
- Edge cases (empty tensors, scalars)

### Performance Tests
- Compilation time benchmarks
- Runtime speedup measurement
- Memory usage comparison
- Cache hit rate

## Performance Targets

| Metric | Target |
|--------|--------|
| Compilation time (100 ops) | < 100ms |
| Cache lookup | < 0.1ms |
| Eager backend overhead | < 5% |
| Triton speedup (pointwise) | > 2x |
| Graph coverage | > 90% of PyTorch ops |

## Dependencies

- **Python 3.9+**: Required for bytecode stability
- **PyTorch**: Eager backend
- **Triton**: GPU kernel generation
- **networkx**: Graph algorithms (optional)

## References

- TorchDynamo: An Increment to Compiler Design
- Torch FX: Practical Program Capture and Transformation
- JAX: Composable Transformations of Python+NumPy
- TensorFlow XLA: Optimizing Compiler
