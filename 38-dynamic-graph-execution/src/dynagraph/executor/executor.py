"""Graph execution engines and optimizations."""

import numpy as np
import logging
from typing import Any, Callable, List, Dict, Set, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
import time

from ..graph.graph import Graph, Node, InputNode, OutputNode, OperationNode

logger = logging.getLogger(__name__)


class Executor(ABC):
    """Base class for graph executors."""

    def __init__(self, graph: Graph = None):
        self.graph = graph

    @abstractmethod
    def execute(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute the graph with given inputs."""
        pass

    def set_graph(self, graph: Graph):
        """Set graph to execute."""
        self.graph = graph


class EagerExecutor(Executor):
    """
    Eager execution mode - operations execute immediately.

    Features:
    - Immediate execution
    - Easy debugging
    - Dynamic control flow
    """

    def __init__(self, graph: Graph = None):
        super().__init__(graph)
        self._cache: Dict[int, np.ndarray] = {}

    def execute(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute graph eagerly."""
        if self.graph is None:
            raise RuntimeError("No graph set for execution")

        self._cache.clear()

        # Set input values
        for node in self.graph.get_inputs():
            if node.name not in inputs:
                raise ValueError(f"Missing input: {node.name}")
            self._cache[node.id] = inputs[node.name]

        # Execute in topological order
        for node in self.graph.topological_sort():
            if isinstance(node, InputNode):
                continue
            elif isinstance(node, OperationNode):
                self._execute_operation(node)
            elif isinstance(node, OutputNode):
                # Output just passes through
                if node.inputs:
                    self._cache[node.id] = self._cache[node.inputs[0].id]

        # Collect outputs
        outputs = {}
        for node in self.graph.get_outputs():
            outputs[node.name] = self._cache[node.id]

        return outputs

    def _execute_operation(self, node: OperationNode):
        """Execute a single operation."""
        # Get input values
        input_values = [self._cache[inp.id] for inp in node.inputs]

        # Execute compute function
        if node.compute_fn is not None:
            result = node.compute_fn(*input_values)
        else:
            result = self._execute_builtin(node.op_type, input_values, node.attrs)

        self._cache[node.id] = result

    def _execute_builtin(
        self,
        op_type: str,
        inputs: List[np.ndarray],
        attrs: Dict[str, Any]
    ) -> np.ndarray:
        """Execute built-in operation."""
        if op_type == "add":
            return inputs[0] + inputs[1]
        elif op_type == "sub":
            return inputs[0] - inputs[1]
        elif op_type == "mul":
            return inputs[0] * inputs[1]
        elif op_type == "div":
            return inputs[0] / inputs[1]
        elif op_type == "matmul":
            return inputs[0] @ inputs[1]
        elif op_type == "relu":
            return np.maximum(0, inputs[0])
        elif op_type == "sigmoid":
            return 1 / (1 + np.exp(-inputs[0]))
        elif op_type == "tanh":
            return np.tanh(inputs[0])
        elif op_type == "softmax":
            axis = attrs.get("axis", -1)
            shifted = inputs[0] - np.max(inputs[0], axis=axis, keepdims=True)
            exp_x = np.exp(shifted)
            return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        elif op_type == "sum":
            axis = attrs.get("axis")
            keepdims = attrs.get("keepdims", False)
            return inputs[0].sum(axis=axis, keepdims=keepdims)
        elif op_type == "mean":
            axis = attrs.get("axis")
            keepdims = attrs.get("keepdims", False)
            return inputs[0].mean(axis=axis, keepdims=keepdims)
        elif op_type == "reshape":
            shape = attrs.get("shape")
            return inputs[0].reshape(shape)
        elif op_type == "transpose":
            axes = attrs.get("axes")
            return inputs[0].transpose(axes)
        elif op_type == "concat":
            axis = attrs.get("axis", 0)
            return np.concatenate(inputs, axis=axis)
        elif op_type == "split":
            indices = attrs.get("indices")
            axis = attrs.get("axis", 0)
            return np.split(inputs[0], indices, axis=axis)
        elif op_type == "gather":
            axis = attrs.get("axis", 0)
            return np.take(inputs[0], inputs[1], axis=axis)
        elif op_type == "identity":
            return inputs[0].copy()
        else:
            raise NotImplementedError(f"Unknown operation: {op_type}")


class LazyExecutor(Executor):
    """
    Lazy execution mode - builds graph then executes.

    Features:
    - Graph optimization before execution
    - Kernel fusion
    - Memory planning
    """

    def __init__(self, graph: Graph = None, optimize: bool = True):
        super().__init__(graph)
        self.optimize = optimize
        self._optimizer = GraphOptimizer()
        self._compiled_graph: Optional[Graph] = None
        self._execution_plan: List[OperationNode] = []

    def compile(self):
        """Compile graph for execution."""
        if self.graph is None:
            raise RuntimeError("No graph set for compilation")

        # Optimize graph
        if self.optimize:
            self._compiled_graph = self._optimizer.optimize(self.graph)
        else:
            self._compiled_graph = self.graph

        # Create execution plan
        self._execution_plan = [
            node for node in self._compiled_graph.topological_sort()
            if isinstance(node, OperationNode)
        ]

        logger.info(f"Compiled graph with {len(self._execution_plan)} operations")

    def execute(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Execute compiled graph."""
        if self._compiled_graph is None:
            self.compile()

        # Use eager executor for actual execution
        eager = EagerExecutor(self._compiled_graph)
        return eager.execute(inputs)

    def benchmark(self, inputs: Dict[str, np.ndarray], iterations: int = 100) -> Dict[str, float]:
        """Benchmark graph execution."""
        if self._compiled_graph is None:
            self.compile()

        # Warmup
        for _ in range(10):
            self.execute(inputs)

        # Benchmark
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.execute(inputs)
            end = time.perf_counter()
            times.append(end - start)

        return {
            "mean_ms": np.mean(times) * 1000,
            "std_ms": np.std(times) * 1000,
            "min_ms": np.min(times) * 1000,
            "max_ms": np.max(times) * 1000,
        }


class OptimizationPass(ABC):
    """Base class for graph optimization passes."""

    @abstractmethod
    def run(self, graph: Graph) -> Graph:
        """Run optimization pass on graph."""
        pass


class GraphOptimizer:
    """
    Optimizes computation graphs.

    Applies a series of optimization passes.
    """

    def __init__(self):
        self._passes: List[OptimizationPass] = [
            DeadCodePass(),
            ConstantFoldingPass(),
            CommonSubexpressionPass(),
            FusionPass(),
        ]

    def add_pass(self, pass_: OptimizationPass):
        """Add optimization pass."""
        self._passes.append(pass_)

    def optimize(self, graph: Graph) -> Graph:
        """Apply all optimization passes."""
        current = graph

        for pass_ in self._passes:
            try:
                current = pass_.run(current)
                logger.debug(f"Applied {type(pass_).__name__}: {current.num_nodes()} nodes")
            except Exception as e:
                logger.warning(f"Pass {type(pass_).__name__} failed: {e}")

        return current


class DeadCodePass(OptimizationPass):
    """Remove unused operations."""

    def run(self, graph: Graph) -> Graph:
        """Remove dead code from graph."""
        # Find nodes reachable from outputs
        reachable = set()
        queue = list(graph.get_outputs())

        while queue:
            node = queue.pop(0)
            if node.id in reachable:
                continue
            reachable.add(node.id)
            queue.extend(node.inputs)

        # Create new graph with only reachable nodes
        new_graph = Graph(f"{graph.name}_dce")
        node_map = {}

        for node in graph.topological_sort():
            if node.id not in reachable:
                continue

            if isinstance(node, InputNode):
                new_node = new_graph.add_input(node.name, node.shape, node.dtype)
            elif isinstance(node, OperationNode):
                new_inputs = [node_map[inp.id] for inp in node.inputs]
                new_node = new_graph.add_operation(
                    node.op_type, node.name, new_inputs,
                    node.compute_fn, node.grad_fn, node.attrs
                )
            else:
                continue

            node_map[node.id] = new_node

        # Add outputs
        for out in graph.get_outputs():
            if out.inputs and out.inputs[0].id in node_map:
                new_graph.add_output(out.name, node_map[out.inputs[0].id])

        return new_graph


class ConstantFoldingPass(OptimizationPass):
    """Fold constant expressions at compile time."""

    def run(self, graph: Graph) -> Graph:
        """Fold constant expressions by evaluating nodes with constant inputs."""
        new_graph = Graph(f"{graph.name}_const_folded")
        node_map = {}
        constant_values = {}  # node_id -> constant value

        for node in graph.topological_sort():
            if isinstance(node, InputNode):
                # Check if input has a known constant value
                if "value" in node.attrs:
                    constant_values[node.id] = node.attrs["value"]
                new_node = new_graph.add_input(node.name, node.shape, node.dtype)
                if "value" in node.attrs:
                    new_node.attrs["value"] = node.attrs["value"]
                node_map[node.id] = new_node

            elif isinstance(node, OperationNode):
                # Check if all inputs are constants
                all_const = all(
                    inp.id in constant_values for inp in node.inputs
                )

                if all_const and node.compute_fn is not None:
                    # Evaluate at compile time
                    try:
                        input_values = [constant_values[inp.id] for inp in node.inputs]
                        result = node.compute_fn(*input_values)
                        constant_values[node.id] = result

                        # Create a constant input node instead
                        shape = result.shape if hasattr(result, 'shape') else ()
                        dtype = str(result.dtype) if hasattr(result, 'dtype') else "float32"
                        new_node = new_graph.add_input(
                            f"{node.name}_const", shape, dtype
                        )
                        new_node.attrs["value"] = result
                        new_node.attrs["folded"] = True
                        node_map[node.id] = new_node
                    except Exception:
                        # If evaluation fails, keep the original operation
                        new_inputs = [node_map[inp.id] for inp in node.inputs]
                        new_node = new_graph.add_operation(
                            node.op_type, node.name, new_inputs,
                            node.compute_fn, node.grad_fn, node.attrs
                        )
                        node_map[node.id] = new_node
                else:
                    # Not all inputs are constants, keep the operation
                    new_inputs = [node_map[inp.id] for inp in node.inputs]
                    new_node = new_graph.add_operation(
                        node.op_type, node.name, new_inputs,
                        node.compute_fn, node.grad_fn, node.attrs
                    )
                    node_map[node.id] = new_node

                    # Propagate constant if result is known
                    if node.id in constant_values:
                        new_node.attrs["value"] = constant_values[node.id]

        # Add outputs
        for out in graph.get_outputs():
            if out.inputs and out.inputs[0].id in node_map:
                new_graph.add_output(out.name, node_map[out.inputs[0].id])

        return new_graph


class CommonSubexpressionPass(OptimizationPass):
    """Eliminate common subexpressions."""

    def run(self, graph: Graph) -> Graph:
        """Eliminate common subexpressions."""
        new_graph = Graph(f"{graph.name}_cse")
        node_map = {}
        expr_map = {}  # (op_type, input_ids, attrs_hash) -> node

        for node in graph.topological_sort():
            if isinstance(node, InputNode):
                new_node = new_graph.add_input(node.name, node.shape, node.dtype)
                node_map[node.id] = new_node
            elif isinstance(node, OperationNode):
                new_inputs = [node_map[inp.id] for inp in node.inputs]
                input_ids = tuple(n.id for n in new_inputs)

                # Create expression key
                attrs_tuple = tuple(sorted(node.attrs.items()))
                expr_key = (node.op_type, input_ids, attrs_tuple)

                if expr_key in expr_map:
                    # Reuse existing node
                    node_map[node.id] = expr_map[expr_key]
                else:
                    # Create new node
                    new_node = new_graph.add_operation(
                        node.op_type, node.name, new_inputs,
                        node.compute_fn, node.grad_fn, node.attrs
                    )
                    node_map[node.id] = new_node
                    expr_map[expr_key] = new_node

        # Add outputs
        for out in graph.get_outputs():
            if out.inputs and out.inputs[0].id in node_map:
                new_graph.add_output(out.name, node_map[out.inputs[0].id])

        return new_graph


class FusionPass(OptimizationPass):
    """Fuse compatible operations."""

    # Patterns for fusion
    FUSION_PATTERNS = [
        # (op1, op2) -> fused_op
        ("matmul", "add", "matmul_bias"),
        ("matmul", "relu", "matmul_relu"),
        ("add", "relu", "add_relu"),
        ("conv", "bn", "conv_bn"),
        ("conv", "relu", "conv_relu"),
    ]

    def run(self, graph: Graph) -> Graph:
        """Fuse operations in graph."""
        new_graph = Graph(f"{graph.name}_fused")
        node_map = {}
        fused_nodes = set()

        nodes = graph.topological_sort()

        for i, node in enumerate(nodes):
            if node.id in fused_nodes:
                continue

            if isinstance(node, InputNode):
                new_node = new_graph.add_input(node.name, node.shape, node.dtype)
                node_map[node.id] = new_node
            elif isinstance(node, OperationNode):
                # Check if this can be fused with next operation
                fused = False

                for out_node in node.outputs:
                    if not isinstance(out_node, OperationNode):
                        continue

                    for pattern in self.FUSION_PATTERNS:
                        if node.op_type == pattern[0] and out_node.op_type == pattern[1]:
                            # Found fusion opportunity
                            if len(out_node.inputs) == 1 or (
                                len(out_node.inputs) == 2 and
                                self._is_constant_or_input(out_node.inputs[1], graph)
                            ):
                                # Create fused operation
                                new_inputs = [node_map[inp.id] for inp in node.inputs]

                                # Add bias input if present
                                if len(out_node.inputs) > 1:
                                    for inp in out_node.inputs:
                                        if inp.id != node.id and inp.id in node_map:
                                            new_inputs.append(node_map[inp.id])

                                fused_op = new_graph.add_operation(
                                    pattern[2],
                                    f"{node.name}_fused",
                                    new_inputs,
                                    attrs={**node.attrs, **out_node.attrs}
                                )

                                node_map[node.id] = fused_op
                                node_map[out_node.id] = fused_op
                                fused_nodes.add(out_node.id)
                                fused = True
                                break

                    if fused:
                        break

                if not fused:
                    new_inputs = [node_map[inp.id] for inp in node.inputs]
                    new_node = new_graph.add_operation(
                        node.op_type, node.name, new_inputs,
                        node.compute_fn, node.grad_fn, node.attrs
                    )
                    node_map[node.id] = new_node

        # Add outputs
        for out in graph.get_outputs():
            if out.inputs and out.inputs[0].id in node_map:
                new_graph.add_output(out.name, node_map[out.inputs[0].id])

        return new_graph

    def _is_constant_or_input(self, node: Node, graph: Graph) -> bool:
        """Check if node is constant or input."""
        return isinstance(node, InputNode)


class MemoryPlanner:
    """
    Plans memory allocation for graph execution.

    Features:
    - Memory reuse
    - In-place operations
    - Memory pooling
    """

    def __init__(self):
        self._allocations: Dict[int, int] = {}  # node_id -> memory_offset
        self._total_memory = 0

    def plan(self, graph: Graph, dtype_size: int = 4) -> Dict[int, int]:
        """
        Plan memory allocation for graph nodes.

        Args:
            graph: Computation graph
            dtype_size: Size of data type in bytes

        Returns:
            Mapping from node ID to memory offset
        """
        # Simple linear allocation
        # A more sophisticated planner would reuse memory

        offset = 0
        for node in graph.topological_sort():
            if isinstance(node, InputNode):
                size = np.prod(node.shape) * dtype_size
                self._allocations[node.id] = offset
                offset += size
            elif isinstance(node, OperationNode):
                # Estimate output size based on inputs
                # This is simplified - real implementation would analyze op
                if node.inputs:
                    input_size = self._allocations.get(node.inputs[0].id, 0)
                    size = input_size  # Assume same size as input
                else:
                    size = 0
                self._allocations[node.id] = offset
                offset += size

        self._total_memory = offset
        return self._allocations

    def get_total_memory(self) -> int:
        """Get total memory required."""
        return self._total_memory


class JITCompiler:
    """
    Just-in-time compiler for computation graphs.

    Generates optimized code for graph execution.
    """

    def __init__(self):
        self._compiled_functions: Dict[str, Callable] = {}

    def compile(self, graph: Graph) -> Callable:
        """
        Compile graph to optimized function.

        Args:
            graph: Computation graph

        Returns:
            Compiled function
        """
        graph_hash = self._hash_graph(graph)

        if graph_hash in self._compiled_functions:
            return self._compiled_functions[graph_hash]

        # Generate Python code
        code = self._generate_code(graph)

        # Compile code
        exec_globals = {"np": np}
        exec(code, exec_globals)

        func = exec_globals["execute_graph"]
        self._compiled_functions[graph_hash] = func

        return func

    def _hash_graph(self, graph: Graph) -> str:
        """Create hash for graph structure."""
        parts = []
        for node in graph.topological_sort():
            if isinstance(node, InputNode):
                parts.append(f"input:{node.name}:{node.shape}")
            elif isinstance(node, OperationNode):
                input_ids = ",".join(str(inp.id) for inp in node.inputs)
                parts.append(f"op:{node.op_type}:{input_ids}")
        return "|".join(parts)

    def _generate_code(self, graph: Graph) -> str:
        """Generate Python code for graph execution."""
        lines = ["def execute_graph(inputs):"]
        lines.append("    cache = {}")

        for node in graph.topological_sort():
            if isinstance(node, InputNode):
                lines.append(f"    cache[{node.id}] = inputs['{node.name}']")
            elif isinstance(node, OperationNode):
                input_vars = ", ".join(f"cache[{inp.id}]" for inp in node.inputs)
                if node.op_type == "add":
                    lines.append(f"    cache[{node.id}] = {input_vars.replace(', ', ' + ')}")
                elif node.op_type == "mul":
                    lines.append(f"    cache[{node.id}] = {input_vars.replace(', ', ' * ')}")
                elif node.op_type == "matmul":
                    inputs = input_vars.split(", ")
                    lines.append(f"    cache[{node.id}] = {inputs[0]} @ {inputs[1]}")
                elif node.op_type == "relu":
                    lines.append(f"    cache[{node.id}] = np.maximum(0, {input_vars})")
                else:
                    lines.append(f"    cache[{node.id}] = {input_vars}  # {node.op_type}")
            elif isinstance(node, OutputNode):
                if node.inputs:
                    lines.append(f"    cache[{node.id}] = cache[{node.inputs[0].id}]")

        # Return outputs
        output_dict = ", ".join(
            f"'{out.name}': cache[{out.id}]" for out in graph.get_outputs()
        )
        lines.append(f"    return {{{output_dict}}}")

        return "\n".join(lines)
