"""Bytecode tracer for capturing Python execution into graphs.

This walks a function's CPython bytecode (via :mod:`dis`) and symbolically
executes the straight-line tensor-math subset, recording each operation as a
graph node — the same idea as TorchDynamo, scoped to the opcodes that matter
for tracing numeric functions. When it meets an unsupported construct (a call,
branch, loop, or jump) it records a *graph break* and stops, returning the
partial graph traced so far (again, exactly how a real tracer degrades).

Supported: arg loading (incl. the 3.13/3.14 fused LOAD_FAST_BORROW variants),
constants, local stores, binary ops (+, -, *, /, @, and others as CUSTOM),
unary negation, simple stack ops, and return. Everything else graph-breaks.
"""

import dis
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

from ..core.graph import Graph, Node, OpType, NodeMetadata
from ..core.tensor import SymbolicTensor, TensorMetadata


class TracingMode(Enum):
    """Tracing modes."""
    NONE = "none"
    SYMBOLIC = "symbolic"
    CONCRETE = "concrete"


class _GraphBreak(Exception):
    """Internal signal: an unsupported construct ended the traceable region."""


# Maps BINARY_OP operator symbols (instr.argrepr, with any in-place '=' stripped)
# to graph op types. Operators not listed are still recorded, as CUSTOM nodes.
_BINOP_OPS = {
    "+": OpType.ADD,
    "-": OpType.SUB,
    "*": OpType.MUL,
    "/": OpType.DIV,
    "@": OpType.MATMUL,
}


@dataclass
class TraceFrame:
    """Represents a frame during tracing."""
    code: types.CodeType
    locals: Dict[str, Any]
    globals: Dict[str, Any]
    instructions: List[dis.Instruction]
    ip: int = 0  # instruction pointer
    stack: List[Any] = field(default_factory=list)
    block_stack: List[tuple] = field(default_factory=list)


class BytecodeTracer:
    """Traces Python bytecode and captures operations into a graph."""

    def __init__(self, graph: Optional[Graph] = None):
        self.graph = graph or Graph()
        self.mode = TracingMode.NONE
        self.frames: List[TraceFrame] = []
        self.symbolic_values: Dict[str, SymbolicTensor] = {}
        self.graph_break_reasons: List[str] = []

    def trace_function(self, func: Callable, *args, **kwargs) -> Graph:
        """Trace a function call and return the captured graph."""
        self.reset()
        self.mode = TracingMode.SYMBOLIC
        self.graph = Graph(name=getattr(func, "__name__", "traced"))

        # Bind positional args to symbolic input tensors and parameter names.
        code = func.__code__
        argnames = code.co_varnames[: code.co_argcount]
        local_vars: Dict[str, Any] = {}
        for i, arg in enumerate(args):
            sym = self._make_symbolic(arg, f"arg_{i}")
            node = Node(
                id=sym.node_id,
                op_type=OpType.INPUT,
                name=sym.source,
                metadata=self._node_metadata(sym),
            )
            self.graph.add_node(node)
            if i < len(argnames):
                local_vars[argnames[i]] = sym

        # Symbolically execute the bytecode; a graph break ends tracing cleanly.
        try:
            self._interpret(func, local_vars)
        except _GraphBreak as brk:
            self.record_graph_break(str(brk))

        self.mode = TracingMode.NONE
        return self.graph

    # ------------------------------------------------------------------ interp

    def _interpret(self, func: Callable, local_vars: Dict[str, Any]) -> None:
        """Walk the function's instructions, dispatching one handler per opcode."""
        instructions = list(dis.get_instructions(func))
        frame = TraceFrame(
            code=func.__code__,
            locals=local_vars,
            globals=getattr(func, "__globals__", {}),
            instructions=instructions,
        )
        self.frames.append(frame)
        stack = frame.stack

        for instr in instructions:
            op = instr.opname

            # --- no-ops / frame bookkeeping ---------------------------------
            if op in ("RESUME", "RESUME_CHECK", "NOP", "PRECALL", "NOT_TAKEN", "MAKE_CELL",
                      "COPY_FREE_VARS"):
                continue
            if op == "PUSH_NULL":
                stack.append(None)
                continue

            # --- loads ------------------------------------------------------
            if op in ("LOAD_FAST", "LOAD_FAST_BORROW", "LOAD_FAST_CHECK", "LOAD_DEREF"):
                stack.append(local_vars.get(instr.argval))
            elif op in ("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW"):
                a, b = instr.argval
                stack.append(local_vars.get(a))
                stack.append(local_vars.get(b))
            elif op == "LOAD_CONST":
                stack.append(instr.argval)

            # --- stores -----------------------------------------------------
            elif op == "STORE_FAST":
                local_vars[instr.argval] = stack.pop()
            elif op == "STORE_FAST_LOAD_FAST":
                store_name, load_name = instr.argval
                local_vars[store_name] = stack.pop()
                stack.append(local_vars.get(load_name))
            elif op == "STORE_FAST_STORE_FAST":
                n1, n2 = instr.argval
                local_vars[n1] = stack.pop()
                local_vars[n2] = stack.pop()

            # --- stack manipulation ----------------------------------------
            elif op == "POP_TOP":
                if stack:
                    stack.pop()
            elif op == "COPY":
                stack.append(stack[-instr.arg])
            elif op == "SWAP":
                stack[-1], stack[-instr.arg] = stack[-instr.arg], stack[-1]

            # --- arithmetic -------------------------------------------------
            elif op == "BINARY_OP":
                rhs = stack.pop()
                lhs = stack.pop()
                stack.append(self._binary(instr.argrepr.rstrip("="), lhs, rhs))
            elif op == "UNARY_NEGATIVE":
                stack.append(self._record_op(OpType.CUSTOM, [stack.pop()], name="neg",
                                             attributes={"op": "neg"}))

            # --- return -----------------------------------------------------
            elif op == "RETURN_VALUE":
                self._record_output(stack.pop() if stack else None)
                return
            elif op == "RETURN_CONST":
                self._record_output(instr.argval)
                return

            # --- anything else ends the traceable region -------------------
            else:
                raise _GraphBreak(f"Unsupported opcode: {op}")

    def _binary(self, symbol: str, lhs: Any, rhs: Any) -> SymbolicTensor:
        """Record a binary operation node and return its symbolic result."""
        op_type = _BINOP_OPS.get(symbol, OpType.CUSTOM)
        attributes = {} if symbol in _BINOP_OPS else {"op": symbol}
        return self._record_op(op_type, [lhs, rhs], attributes=attributes)

    # ------------------------------------------------------------- graph build

    def _record_op(
        self,
        op_type: OpType,
        operands: List[Any],
        name: Optional[str] = None,
        attributes: Optional[dict] = None,
    ) -> SymbolicTensor:
        """Create a node for an op, wiring edges from each operand's node."""
        result = SymbolicTensor(source=(name or op_type.name.lower()))
        result.metadata.shape = self._infer_shape(op_type, operands)
        node = Node(
            id=result.node_id,
            op_type=op_type,
            name=name or op_type.name.lower(),
            attributes=attributes or {},
            metadata=self._node_metadata(result),
        )
        self.graph.add_node(node)
        for index, operand in enumerate(operands):
            src_id = self._operand_node_id(operand)
            if src_id is not None and src_id in self.graph.nodes:
                self.graph.add_edge(src_id, result.node_id, index=index)
        return result

    def _record_output(self, value: Any) -> None:
        """Create an OUTPUT node fed by the returned value's node."""
        out = Node(op_type=OpType.OUTPUT, name="output")
        self.graph.add_node(out)
        src_id = self._operand_node_id(value)
        if src_id is not None and src_id in self.graph.nodes:
            self.graph.add_edge(src_id, out.id)

    def _operand_node_id(self, value: Any) -> Optional[str]:
        """Return the graph node id for an operand, materializing constants."""
        if isinstance(value, SymbolicTensor):
            return value.node_id
        if value is None:
            return None
        # A literal/constant consumed by an op — record it as a CONSTANT node.
        const = Node(op_type=OpType.CONSTANT, name=repr(value), attributes={"value": value})
        self.graph.add_node(const)
        return const.id

    @staticmethod
    def _infer_shape(op_type: OpType, operands: List[Any]) -> Optional[tuple]:
        """Best-effort output-shape inference for elementwise ops."""
        shapes = [o.shape for o in operands if isinstance(o, SymbolicTensor) and o.shape]
        if not shapes:
            return None
        if op_type == OpType.MATMUL or len(shapes) < 2:
            return shapes[0] if len(shapes) == 1 else None
        try:
            return SymbolicTensor._broadcast_shapes(shapes[0], shapes[1])
        except ValueError:
            return None

    @staticmethod
    def _node_metadata(sym: SymbolicTensor) -> NodeMetadata:
        """Build graph NodeMetadata from a symbolic tensor's tensor metadata."""
        return NodeMetadata(
            dtype=str(sym.dtype) if sym.dtype is not None else None,
            shape=sym.shape,
            device=sym.device,
            requires_grad=sym.requires_grad,
        )

    def _make_symbolic(self, value: Any, source: str) -> SymbolicTensor:
        """Convert a concrete value to symbolic."""
        import numpy as np

        if isinstance(value, np.ndarray):
            tensor = SymbolicTensor(source=source, concrete_value=value.copy())
        elif isinstance(value, (int, float)):
            tensor = SymbolicTensor(source=source, concrete_value=np.array(value))
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            tensor = SymbolicTensor(
                source=source,
                metadata=TensorMetadata(shape=value.shape, dtype=value.dtype),
            )
        else:
            tensor = SymbolicTensor(source=source)

        self.symbolic_values[source] = tensor
        return tensor

    def should_trace(self, frame: types.FrameType) -> bool:
        """Determine if a frame should be traced."""
        code = frame.f_code
        filename = code.co_filename

        if filename.startswith('<'):
            return False
        if '/site-packages/' in filename:
            return False
        if '/lib/python' in filename:
            return False

        return True

    def record_graph_break(self, reason: str) -> None:
        """Record reason for graph break."""
        self.graph_break_reasons.append(reason)

    def reset(self) -> None:
        """Reset tracer state."""
        self.graph = Graph()
        self.frames.clear()
        self.symbolic_values.clear()
        self.graph_break_reasons.clear()
        self.mode = TracingMode.NONE
