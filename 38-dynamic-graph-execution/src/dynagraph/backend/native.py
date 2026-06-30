"""Native Python/NumPy backend.

Generates optimized Python code for graph execution using NumPy.
"""

from __future__ import annotations
from typing import Any, Dict, List, Callable
import numpy as np
import textwrap

from .lowering import (
    BackendLowering,
    LoweringContext,
    LoweredGraph,
    LoweredNode,
    TensorSpec,
    DataType,
    MemoryLayout,
)


class NativeBackend(BackendLowering):
    """Native Python/NumPy backend for graph execution."""

    def __init__(self):
        super().__init__()
        self._register_default_ops()

    @property
    def name(self) -> str:
        return "native"

    def supported_ops(self) -> List[str]:
        return list(self.op_registry.keys())

    def supported_dtypes(self) -> List[DataType]:
        return [
            DataType.FLOAT32,
            DataType.FLOAT64,
            DataType.FLOAT16,
            DataType.INT8,
            DataType.INT16,
            DataType.INT32,
            DataType.INT64,
            DataType.UINT8,
            DataType.BOOL,
        ]

    def _register_default_ops(self) -> None:
        """Register default operation lowerings."""
        # Elementwise operations
        self.register_op('add', self._lower_add)
        self.register_op('sub', self._lower_sub)
        self.register_op('mul', self._lower_mul)
        self.register_op('div', self._lower_div)
        self.register_op('neg', self._lower_neg)
        self.register_op('pow', self._lower_pow)
        self.register_op('sqrt', self._lower_sqrt)
        self.register_op('exp', self._lower_exp)
        self.register_op('log', self._lower_log)
        self.register_op('abs', self._lower_abs)

        # Activation functions
        self.register_op('relu', self._lower_relu)
        self.register_op('sigmoid', self._lower_sigmoid)
        self.register_op('tanh', self._lower_tanh)
        self.register_op('softmax', self._lower_softmax)
        self.register_op('gelu', self._lower_gelu)

        # Reduction operations
        self.register_op('sum', self._lower_sum)
        self.register_op('mean', self._lower_mean)
        self.register_op('max', self._lower_max)
        self.register_op('min', self._lower_min)

        # Matrix operations
        self.register_op('matmul', self._lower_matmul)
        self.register_op('transpose', self._lower_transpose)
        self.register_op('reshape', self._lower_reshape)

        # Comparison operations
        self.register_op('equal', self._lower_equal)
        self.register_op('greater', self._lower_greater)
        self.register_op('less', self._lower_less)

    def _initial_lowering(self, graph: Any, ctx: LoweringContext) -> LoweredGraph:
        """Perform initial lowering from high-level graph."""
        lowered = LoweredGraph(name="native_graph")

        # Handle different graph input types
        if isinstance(graph, dict):
            # Traced graph from jit_trace
            return self._lower_traced_graph(graph, ctx, lowered)
        elif hasattr(graph, 'nodes'):
            # Graph object with nodes
            return self._lower_graph_object(graph, ctx, lowered)
        else:
            raise ValueError(f"Unsupported graph type: {type(graph)}")

    def _lower_traced_graph(
        self, graph: Dict, ctx: LoweringContext, lowered: LoweredGraph
    ) -> LoweredGraph:
        """Lower a traced graph dictionary."""
        # Add inputs
        for i in range(graph.get('inputs', 0)):
            spec = TensorSpec(
                name=f"input_{i}",
                shape=(-1,),  # Dynamic shape
                dtype=DataType.FLOAT32,
            )
            lowered.add_input(spec)
            ctx.register_tensor(spec)

        # Add outputs
        for i in range(graph.get('outputs', 0)):
            spec = TensorSpec(
                name=f"output_{i}",
                shape=(-1,),
                dtype=DataType.FLOAT32,
            )
            lowered.add_output(spec)
            ctx.register_tensor(spec)

        # Lower operations
        for op in graph.get('ops', []):
            node = self._lower_op(op, ctx)
            if node:
                lowered.add_node(node)

        return lowered

    def _lower_graph_object(
        self, graph: Any, ctx: LoweringContext, lowered: LoweredGraph
    ) -> LoweredGraph:
        """Lower a graph object with nodes."""
        # Process graph nodes
        for node in graph.nodes:
            lowered_node = self._lower_node(node, ctx)
            if lowered_node:
                lowered.add_node(lowered_node)

        return lowered

    def _lower_op(self, op: Dict, ctx: LoweringContext) -> LoweredNode:
        """Lower a single operation."""
        op_type = op.get('type', 'unknown')
        inputs = op.get('inputs', [])
        outputs = op.get('outputs', [ctx.new_tensor_name()])
        attrs = op.get('attributes', {})

        return LoweredNode(
            name=ctx.new_node_name(op_type),
            op_type=op_type,
            inputs=inputs,
            outputs=outputs,
            attributes=attrs,
        )

    def _lower_node(self, node: Any, ctx: LoweringContext) -> LoweredNode:
        """Lower a graph node."""
        op_type = getattr(node, 'op_type', 'unknown')
        inputs = getattr(node, 'inputs', [])
        outputs = getattr(node, 'outputs', [ctx.new_tensor_name()])

        return LoweredNode(
            name=ctx.new_node_name(op_type),
            op_type=op_type,
            inputs=[str(i) for i in inputs],
            outputs=[str(o) for o in outputs],
        )

    def compile(self, graph: LoweredGraph) -> 'CompiledNativeGraph':
        """Compile lowered graph to executable form."""
        return CompiledNativeGraph(graph, self)

    def execute(
        self, compiled: 'CompiledNativeGraph', inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute compiled graph."""
        return compiled.run(inputs)

    # Operation lowering implementations
    def _lower_add(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = {node.inputs[0]} + {node.inputs[1]}"

    def _lower_sub(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = {node.inputs[0]} - {node.inputs[1]}"

    def _lower_mul(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = {node.inputs[0]} * {node.inputs[1]}"

    def _lower_div(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = {node.inputs[0]} / {node.inputs[1]}"

    def _lower_neg(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = -{node.inputs[0]}"

    def _lower_pow(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = np.power({node.inputs[0]}, {node.inputs[1]})"

    def _lower_sqrt(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = np.sqrt({node.inputs[0]})"

    def _lower_exp(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = np.exp({node.inputs[0]})"

    def _lower_log(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = np.log({node.inputs[0]})"

    def _lower_abs(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = np.abs({node.inputs[0]})"

    def _lower_relu(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = np.maximum(0, {node.inputs[0]})"

    def _lower_sigmoid(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = 1 / (1 + np.exp(-{node.inputs[0]}))"

    def _lower_tanh(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = np.tanh({node.inputs[0]})"

    def _lower_softmax(self, node: LoweredNode) -> str:
        axis = node.attributes.get('axis', -1)
        return f"""
_exp = np.exp({node.inputs[0]} - np.max({node.inputs[0]}, axis={axis}, keepdims=True))
{node.outputs[0]} = _exp / np.sum(_exp, axis={axis}, keepdims=True)
""".strip()

    def _lower_gelu(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = 0.5 * {node.inputs[0]} * (1 + np.tanh(np.sqrt(2/np.pi) * ({node.inputs[0]} + 0.044715 * np.power({node.inputs[0]}, 3))))"

    def _lower_sum(self, node: LoweredNode) -> str:
        axis = node.attributes.get('axis', None)
        keepdims = node.attributes.get('keepdims', False)
        return f"{node.outputs[0]} = np.sum({node.inputs[0]}, axis={axis}, keepdims={keepdims})"

    def _lower_mean(self, node: LoweredNode) -> str:
        axis = node.attributes.get('axis', None)
        keepdims = node.attributes.get('keepdims', False)
        return f"{node.outputs[0]} = np.mean({node.inputs[0]}, axis={axis}, keepdims={keepdims})"

    def _lower_max(self, node: LoweredNode) -> str:
        axis = node.attributes.get('axis', None)
        keepdims = node.attributes.get('keepdims', False)
        return f"{node.outputs[0]} = np.max({node.inputs[0]}, axis={axis}, keepdims={keepdims})"

    def _lower_min(self, node: LoweredNode) -> str:
        axis = node.attributes.get('axis', None)
        keepdims = node.attributes.get('keepdims', False)
        return f"{node.outputs[0]} = np.min({node.inputs[0]}, axis={axis}, keepdims={keepdims})"

    def _lower_matmul(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = np.matmul({node.inputs[0]}, {node.inputs[1]})"

    def _lower_transpose(self, node: LoweredNode) -> str:
        perm = node.attributes.get('perm', None)
        if perm:
            return f"{node.outputs[0]} = np.transpose({node.inputs[0]}, {perm})"
        return f"{node.outputs[0]} = np.transpose({node.inputs[0]})"

    def _lower_reshape(self, node: LoweredNode) -> str:
        shape = node.attributes.get('shape', [-1])
        return f"{node.outputs[0]} = np.reshape({node.inputs[0]}, {shape})"

    def _lower_equal(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = ({node.inputs[0]} == {node.inputs[1]})"

    def _lower_greater(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = ({node.inputs[0]} > {node.inputs[1]})"

    def _lower_less(self, node: LoweredNode) -> str:
        return f"{node.outputs[0]} = ({node.inputs[0]} < {node.inputs[1]})"

    def generate_code(self, node: LoweredNode) -> str:
        """Generate Python code for a node."""
        op_type = node.op_type
        if op_type in self.op_registry:
            lowering_fn = self.op_registry[op_type]
            return lowering_fn(node)
        else:
            # Generic fallback
            inputs_str = ", ".join(node.inputs)
            return f"# Unknown op: {op_type}({inputs_str})"


class CompiledNativeGraph:
    """Compiled native graph ready for execution."""

    def __init__(self, graph: LoweredGraph, backend: NativeBackend):
        self.graph = graph
        self.backend = backend
        self._code = self._generate_code()
        self._compiled = None
        self._compile()

    def _generate_code(self) -> str:
        """Generate Python code for the graph."""
        lines = [
            "import numpy as np",
            "",
            "def execute(inputs):",
            "    # Unpack inputs",
        ]

        # Unpack inputs
        for i, inp in enumerate(self.graph.inputs):
            lines.append(f"    {inp.name} = inputs['{inp.name}']")

        lines.append("")
        lines.append("    # Execute operations")

        # Generate code for each node in topological order
        for node in self.graph.topological_sort():
            code = self.backend.generate_code(node)
            for line in code.split('\n'):
                lines.append(f"    {line}")

        lines.append("")
        lines.append("    # Pack outputs")
        lines.append("    return {")
        for out in self.graph.outputs:
            lines.append(f"        '{out.name}': {out.name},")
        lines.append("    }")

        return "\n".join(lines)

    def _compile(self) -> None:
        """Compile the generated code."""
        namespace = {'np': np}
        exec(self._code, namespace)
        self._compiled = namespace['execute']

    def run(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute the compiled graph."""
        if self._compiled is None:
            raise RuntimeError("Graph not compiled")
        return self._compiled(inputs)

    def get_code(self) -> str:
        """Get the generated code."""
        return self._code

    def benchmark(self, inputs: Dict[str, np.ndarray], iterations: int = 100) -> Dict[str, float]:
        """Benchmark graph execution."""
        import time

        # Warmup
        for _ in range(10):
            self.run(inputs)

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.run(inputs)
            end = time.perf_counter()
            times.append(end - start)

        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
            'min_ms': np.min(times) * 1000,
            'max_ms': np.max(times) * 1000,
            'iterations': iterations,
        }
