"""Individual optimization passes for graph transformations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np

from ..core.graph import Graph, Node, Edge, OpType, NodeMetadata


@dataclass
class PassResult:
    """Result of an optimization pass."""
    changed: bool
    nodes_removed: int = 0
    nodes_added: int = 0
    nodes_modified: int = 0
    message: str = ""


class OptimizationPass(ABC):
    """Base class for optimization passes."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.stats = {
            "runs": 0,
            "changes": 0,
            "nodes_removed": 0,
            "nodes_added": 0,
        }

    @abstractmethod
    def run(self, graph: Graph) -> PassResult:
        """Run the optimization pass on the graph."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"


class ConstantFolding(OptimizationPass):
    """
    Evaluate operations on constant inputs at compile time.

    For example:
        x = 2 + 3  ->  x = 5
        y = x * 2 where x=5  ->  y = 10
    """

    def __init__(self):
        super().__init__("constant_folding")
        self._constant_cache: Dict[str, Any] = {}

    def run(self, graph: Graph) -> PassResult:
        self.stats["runs"] += 1
        changed = False
        nodes_to_fold = []

        # Find nodes where all inputs are constants
        for node_id, node in list(graph.nodes.items()):
            if node.op_type in (OpType.INPUT, OpType.OUTPUT, OpType.PARAMETER, OpType.BUFFER):
                continue

            if node.op_type == OpType.CONSTANT:
                # Already a constant
                continue

            # Check if all inputs are constants or have known values
            all_inputs_constant = True
            input_values = []

            for input_id in node.inputs:
                if input_id in self._constant_cache:
                    input_values.append(self._constant_cache[input_id])
                elif input_id in graph.nodes:
                    input_node = graph.nodes[input_id]
                    if input_node.op_type == OpType.CONSTANT:
                        value = input_node.attributes.get("value")
                        if value is not None:
                            input_values.append(value)
                            self._constant_cache[input_id] = value
                        else:
                            all_inputs_constant = False
                            break
                    else:
                        all_inputs_constant = False
                        break
                else:
                    all_inputs_constant = False
                    break

            if all_inputs_constant and len(input_values) == len(node.inputs):
                # Try to evaluate
                result = self._evaluate_op(node, input_values)
                if result is not None:
                    nodes_to_fold.append((node_id, result))

        # Apply folding
        for node_id, result in nodes_to_fold:
            node = graph.nodes[node_id]

            # Convert to constant node
            node.op_type = OpType.CONSTANT
            node.attributes["value"] = result
            node.inputs = []

            # Cache the result
            self._constant_cache[node_id] = result
            changed = True

        if changed:
            self.stats["changes"] += 1
            self.stats["nodes_removed"] += len(nodes_to_fold)

        return PassResult(
            changed=changed,
            nodes_modified=len(nodes_to_fold),
            message=f"Folded {len(nodes_to_fold)} constant expressions"
        )

    def _evaluate_op(self, node: Node, inputs: List[Any]) -> Optional[Any]:
        """Evaluate operation with constant inputs."""
        try:
            op = node.op_type

            if op == OpType.ADD and len(inputs) == 2:
                return np.add(inputs[0], inputs[1])
            elif op == OpType.SUB and len(inputs) == 2:
                return np.subtract(inputs[0], inputs[1])
            elif op == OpType.MUL and len(inputs) == 2:
                return np.multiply(inputs[0], inputs[1])
            elif op == OpType.DIV and len(inputs) == 2:
                return np.divide(inputs[0], inputs[1])
            elif op == OpType.MATMUL and len(inputs) == 2:
                return np.matmul(inputs[0], inputs[1])
            elif op == OpType.RELU and len(inputs) == 1:
                return np.maximum(inputs[0], 0)
            elif op == OpType.SIGMOID and len(inputs) == 1:
                return 1 / (1 + np.exp(-inputs[0]))

            return None
        except Exception:
            return None


class DeadCodeElimination(OptimizationPass):
    """
    Remove nodes not reachable from outputs.

    Identifies and removes computations whose results are never used.
    """

    def __init__(self):
        super().__init__("dead_code_elimination")

    def run(self, graph: Graph) -> PassResult:
        self.stats["runs"] += 1

        # Find all live nodes by backward traversal from outputs
        live_nodes = set()

        def mark_live(node_id: str):
            if node_id in live_nodes:
                return
            if node_id not in graph.nodes:
                return

            live_nodes.add(node_id)
            node = graph.nodes[node_id]

            for input_id in node.inputs:
                mark_live(input_id)

        # Start from output nodes
        for output_id in graph.output_nodes:
            mark_live(output_id)

        # Also keep all OUTPUT op_type nodes
        for node_id, node in graph.nodes.items():
            if node.op_type == OpType.OUTPUT:
                mark_live(node_id)

        # Find dead nodes
        all_nodes = set(graph.nodes.keys())
        dead_nodes = all_nodes - live_nodes

        # Remove dead nodes
        for node_id in dead_nodes:
            graph.remove_node(node_id)

        changed = len(dead_nodes) > 0

        if changed:
            self.stats["changes"] += 1
            self.stats["nodes_removed"] += len(dead_nodes)

        return PassResult(
            changed=changed,
            nodes_removed=len(dead_nodes),
            message=f"Removed {len(dead_nodes)} dead nodes"
        )


class CommonSubexpressionElimination(OptimizationPass):
    """
    Eliminate redundant computations.

    If two nodes compute the same operation with the same inputs,
    one can be replaced with a reference to the other.
    """

    def __init__(self):
        super().__init__("cse")

    def run(self, graph: Graph) -> PassResult:
        self.stats["runs"] += 1

        # Build expression signatures
        expr_to_node: Dict[tuple, str] = {}
        replacements: Dict[str, str] = {}  # node_to_replace -> replacement_node

        for node_id, node in graph.nodes.items():
            if node.op_type in (OpType.INPUT, OpType.OUTPUT, OpType.CONSTANT,
                               OpType.PARAMETER, OpType.BUFFER):
                continue

            # Create signature: (op_type, sorted_inputs, frozen_attributes)
            signature = self._compute_signature(node)

            if signature in expr_to_node:
                # Found duplicate - mark for replacement
                replacements[node_id] = expr_to_node[signature]
            else:
                expr_to_node[signature] = node_id

        # Apply replacements
        for old_id, new_id in replacements.items():
            self._replace_uses(graph, old_id, new_id)
            graph.remove_node(old_id)

        changed = len(replacements) > 0

        if changed:
            self.stats["changes"] += 1
            self.stats["nodes_removed"] += len(replacements)

        return PassResult(
            changed=changed,
            nodes_removed=len(replacements),
            message=f"Eliminated {len(replacements)} common subexpressions"
        )

    def _compute_signature(self, node: Node) -> tuple:
        """Compute a hashable signature for the expression."""
        inputs_tuple = tuple(sorted(node.inputs))

        # Convert attributes to hashable form
        attrs = []
        for k, v in sorted(node.attributes.items()):
            if isinstance(v, (list, np.ndarray)):
                v = tuple(v.flatten().tolist()) if hasattr(v, 'flatten') else tuple(v)
            attrs.append((k, v))

        return (node.op_type, inputs_tuple, tuple(attrs))

    def _replace_uses(self, graph: Graph, old_id: str, new_id: str):
        """Replace all uses of old_id with new_id."""
        for node_id, node in graph.nodes.items():
            if node_id == old_id:
                continue

            # Replace in inputs
            node.inputs = [new_id if inp == old_id else inp for inp in node.inputs]

        # Update output references
        graph.output_nodes = [new_id if o == old_id else o for o in graph.output_nodes]


class AlgebraicSimplification(OptimizationPass):
    """
    Apply algebraic identities to simplify expressions.

    Examples:
        x + 0 = x
        x * 1 = x
        x * 0 = 0
        x - x = 0
        relu(relu(x)) = relu(x)
    """

    def __init__(self):
        super().__init__("algebraic_simplification")

    def run(self, graph: Graph) -> PassResult:
        self.stats["runs"] += 1
        simplified = 0

        for node_id, node in list(graph.nodes.items()):
            if self._try_simplify(graph, node_id, node):
                simplified += 1

        changed = simplified > 0

        if changed:
            self.stats["changes"] += 1

        return PassResult(
            changed=changed,
            nodes_modified=simplified,
            message=f"Simplified {simplified} expressions"
        )

    def _try_simplify(self, graph: Graph, node_id: str, node: Node) -> bool:
        """Try to simplify a single node."""

        # x + 0 = x
        if node.op_type == OpType.ADD:
            for i, input_id in enumerate(node.inputs):
                if self._is_zero(graph, input_id):
                    other_input = node.inputs[1 - i]
                    self._replace_with(graph, node_id, other_input)
                    return True

        # x * 1 = x
        if node.op_type == OpType.MUL:
            for i, input_id in enumerate(node.inputs):
                if self._is_one(graph, input_id):
                    other_input = node.inputs[1 - i]
                    self._replace_with(graph, node_id, other_input)
                    return True

        # x * 0 = 0
        if node.op_type == OpType.MUL:
            for input_id in node.inputs:
                if self._is_zero(graph, input_id):
                    # Replace with zero constant
                    node.op_type = OpType.CONSTANT
                    node.attributes["value"] = 0.0
                    node.inputs = []
                    return True

        # x - x = 0
        if node.op_type == OpType.SUB:
            if len(node.inputs) == 2 and node.inputs[0] == node.inputs[1]:
                node.op_type = OpType.CONSTANT
                node.attributes["value"] = 0.0
                node.inputs = []
                return True

        # x / x = 1 (where x != 0)
        if node.op_type == OpType.DIV:
            if len(node.inputs) == 2 and node.inputs[0] == node.inputs[1]:
                node.op_type = OpType.CONSTANT
                node.attributes["value"] = 1.0
                node.inputs = []
                return True

        # relu(relu(x)) = relu(x)
        if node.op_type == OpType.RELU:
            if len(node.inputs) == 1:
                input_id = node.inputs[0]
                if input_id in graph.nodes:
                    input_node = graph.nodes[input_id]
                    if input_node.op_type == OpType.RELU:
                        self._replace_with(graph, node_id, input_id)
                        return True

        # sigmoid(sigmoid(x)) is not equal to sigmoid(x), skip

        return False

    def _is_zero(self, graph: Graph, node_id: str) -> bool:
        """Check if node represents zero."""
        if node_id not in graph.nodes:
            return False
        node = graph.nodes[node_id]
        if node.op_type == OpType.CONSTANT:
            value = node.attributes.get("value")
            if value is not None:
                try:
                    return np.all(np.array(value) == 0)
                except Exception:
                    return False
        return False

    def _is_one(self, graph: Graph, node_id: str) -> bool:
        """Check if node represents one."""
        if node_id not in graph.nodes:
            return False
        node = graph.nodes[node_id]
        if node.op_type == OpType.CONSTANT:
            value = node.attributes.get("value")
            if value is not None:
                try:
                    return np.all(np.array(value) == 1)
                except Exception:
                    return False
        return False

    def _replace_with(self, graph: Graph, old_id: str, new_id: str):
        """Replace all uses of old_id with new_id and remove old node."""
        for node in graph.nodes.values():
            node.inputs = [new_id if inp == old_id else inp for inp in node.inputs]
        graph.output_nodes = [new_id if o == old_id else o for o in graph.output_nodes]


class OperatorFusion(OptimizationPass):
    """
    Fuse compatible operations into single kernels.

    Common fusion patterns:
        - MatMul + BiasAdd -> Linear
        - Conv2D + BatchNorm + ReLU -> FusedConvBnRelu
        - Pointwise operations (add, mul, relu chains)
    """

    def __init__(self):
        super().__init__("operator_fusion")
        self._fusion_patterns = [
            self._fuse_matmul_bias,
            self._fuse_conv_bn_relu,
            self._fuse_pointwise_chain,
        ]

    def run(self, graph: Graph) -> PassResult:
        self.stats["runs"] += 1
        total_fused = 0

        # Try each fusion pattern
        for pattern_fn in self._fusion_patterns:
            while True:
                fused = pattern_fn(graph)
                if fused == 0:
                    break
                total_fused += fused

        changed = total_fused > 0

        if changed:
            self.stats["changes"] += 1

        return PassResult(
            changed=changed,
            nodes_modified=total_fused,
            message=f"Fused {total_fused} operator patterns"
        )

    def _fuse_matmul_bias(self, graph: Graph) -> int:
        """Fuse MatMul + Add into Linear."""
        fused = 0

        for node_id, node in list(graph.nodes.items()):
            if node.op_type != OpType.ADD:
                continue

            # Check if one input is matmul with single user
            for i, input_id in enumerate(node.inputs):
                if input_id not in graph.nodes:
                    continue

                input_node = graph.nodes[input_id]
                if input_node.op_type != OpType.MATMUL:
                    continue

                # Check if matmul has only one user (this add node)
                users = [n for n in graph.nodes.values() if input_id in n.inputs]
                if len(users) != 1:
                    continue

                # Get the bias (the other input to add) and the matmul operands
                # in operand (edge-index) order before we mutate the graph.
                bias_id = node.inputs[1 - i]
                matmul_operands = self._ordered_operands(graph, input_id)

                # Create fused linear node
                node.op_type = OpType.LINEAR
                node.attributes["has_bias"] = True
                node.attributes["fused_from"] = ["matmul", "add"]

                # Update inputs: matmul inputs + bias, keeping edges/adjacency
                # consistent. Remove the matmul first (drops its edges).
                new_inputs = list(matmul_operands) + [bias_id]
                graph.remove_node(input_id)
                self._rewire_inputs(graph, node, new_inputs)
                fused += 1
                break

        return fused

    def _fuse_conv_bn_relu(self, graph: Graph) -> int:
        """Fuse Conv2D + BatchNorm + ReLU."""
        fused = 0

        for node_id, node in list(graph.nodes.items()):
            if node.op_type != OpType.RELU:
                continue

            if not node.inputs:
                continue

            bn_id = node.inputs[0]
            if bn_id not in graph.nodes:
                continue

            bn_node = graph.nodes[bn_id]
            if bn_node.op_type != OpType.BATCHNORM:
                continue

            # Check single user
            bn_users = [n for n in graph.nodes.values() if bn_id in n.inputs]
            if len(bn_users) != 1:
                continue

            if not bn_node.inputs:
                continue

            conv_id = bn_node.inputs[0]
            if conv_id not in graph.nodes:
                continue

            conv_node = graph.nodes[conv_id]
            if conv_node.op_type != OpType.CONV2D:
                continue

            # Check single user
            conv_users = [n for n in graph.nodes.values() if conv_id in n.inputs]
            if len(conv_users) != 1:
                continue

            # Create fused node (reuse relu node)
            node.op_type = OpType.CUSTOM
            node.name = "fused_conv_bn_relu"
            node.attributes["fused_from"] = ["conv2d", "batchnorm", "relu"]
            node.attributes.update(conv_node.attributes)
            node.attributes.update({
                f"bn_{k}": v for k, v in bn_node.attributes.items()
            })

            # Update inputs to the conv operands (edge-index order), keeping the
            # graph's edges/adjacency consistent.
            conv_operands = self._ordered_operands(graph, conv_id)

            # Remove intermediate nodes, then rewire the fused node's inputs.
            graph.remove_node(bn_id)
            graph.remove_node(conv_id)
            self._rewire_inputs(graph, node, conv_operands)
            fused += 1

        return fused

    # Pointwise (elementwise) ops that are safe to fuse into a single chain.
    _POINTWISE_OPS = frozenset({
        OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV,
        OpType.RELU, OpType.SIGMOID,
    })

    def _fuse_pointwise_chain(self, graph: Graph) -> int:
        """Fuse a producer→consumer pair of pointwise ops into one node.

        Identifies a pointwise producer whose *only* user is another pointwise
        op, and collapses the pair into a single ``CUSTOM`` ``fused_pointwise``
        node that records the chain of ops in execution order. Running this to
        the fixed point in :meth:`run` grows the chain one op at a time, so a
        sequence ``add → mul → relu`` collapses into one fused node.

        The fused node keeps the consumer's remaining inputs after the
        producer's inputs, so the recorded ``op_chain`` and the flattened input
        list stay consistent for left-to-right evaluation. Fusion requires the
        producer to feed the consumer's *first* input slot and to have exactly
        one user, which keeps the producer-before-consumer evaluation order
        semantically exact (the backend evaluates the chain left-to-right over
        the flattened inputs).
        """
        # Fuse at most one pair per scan, then let run()'s loop re-scan. This
        # keeps the graph consistent between fusions and avoids reasoning about
        # a mutated node table mid-iteration.
        for node_id in list(graph.nodes.keys()):
            if node_id not in graph.nodes:
                continue
            node = graph.nodes[node_id]
            if node.op_type not in self._POINTWISE_OPS:
                continue

            # Operand order is taken from edge indices, not the raw inputs list,
            # which other passes mutate without preserving order.
            consumer_operands = self._ordered_operands(graph, node_id)
            if not consumer_operands:
                continue

            # The producer must feed the consumer's first operand slot so that
            # left-to-right chain evaluation stays exact.
            producer_id = consumer_operands[0]
            if producer_id not in graph.nodes:
                continue
            producer = graph.nodes[producer_id]
            if producer.op_type not in self._POINTWISE_OPS:
                continue
            # Producer must be used only by this consumer.
            users = {e.target for e in graph.edges if e.source == producer_id}
            if users != {node_id}:
                continue

            producer_operands = self._ordered_operands(graph, producer_id)

            # Build the fused op chain: producer's chain first, then consumer op.
            producer_chain = producer.attributes.get(
                "op_chain", [producer.op_type.name.lower()]
            )
            consumer_chain = node.attributes.get(
                "op_chain", [node.op_type.name.lower()]
            )
            op_chain = list(producer_chain) + list(consumer_chain)

            # Flatten operands: producer's operands replace slot 0, then the
            # consumer's remaining operands follow.
            new_inputs = list(producer_operands) + list(consumer_operands[1:])

            # Skip when the flattened chain would consume the same source in more
            # than one slot (e.g. ``(x - y) / x``). The graph's adjacency model
            # de-duplicates a node's successors, so a fused node with a repeated
            # input cannot be scheduled consistently; leaving the pair unfused
            # keeps the graph valid and still executes correctly.
            if len(set(new_inputs)) != len(new_inputs):
                continue

            node.op_type = OpType.CUSTOM
            node.name = "fused_pointwise"
            node.attributes["op_chain"] = op_chain
            node.attributes["fused_from"] = list(op_chain)

            # Remove the producer (drops its edges) then rewire the fused node's
            # inputs so the graph's edge list and adjacency stay consistent.
            graph.remove_node(producer_id)
            self._rewire_inputs(graph, node, new_inputs)
            return 1

        return 0

    @staticmethod
    def _ordered_operands(graph: Graph, node_id: str) -> List[str]:
        """Return a node's input sources in operand (edge-index) order.

        The graph tolerates the same source feeding several operand slots, so
        this reads the inbound edges and orders them by ``index`` rather than
        trusting the (mutation-prone) ``node.inputs`` list order.
        """
        inbound = [e for e in graph.edges if e.target == node_id]
        inbound.sort(key=lambda e: e.index)
        return [e.source for e in inbound]

    @staticmethod
    def _rewire_inputs(graph: Graph, node: Node, new_inputs: List[str]) -> None:
        """Replace ``node``'s inputs, keeping edges and adjacency consistent.

        Directly assigning ``node.inputs`` would leave the graph's ``edges``
        list and the source nodes' ``outputs`` adjacency stale, which breaks the
        topological sort. This drops the node's existing inbound edges and adds
        fresh ones for ``new_inputs`` (only for sources still in the graph).
        """
        node_id = node.id
        # Drop existing inbound edges and adjacency for this node.
        graph.edges = [e for e in graph.edges if e.target != node_id]
        for src in list(node.inputs):
            if src in graph.nodes:
                graph.nodes[src].remove_output(node_id)

        # Set inputs directly (may contain the same source more than once, e.g.
        # a fused ``x + x``) and add a matching edge + outputs entry per source.
        node.inputs = [src for src in new_inputs if src in graph.nodes]
        for index, src in enumerate(node.inputs):
            graph.edges.append(Edge(source=src, target=node_id, index=index))
            graph.nodes[src].add_output(node_id)
        graph._topological_order = None


class LayoutOptimization(OptimizationPass):
    """
    Optimize memory layout for better performance.

    Tracks tensor memory formats and inserts layout conversions
    where beneficial.
    """

    def __init__(self):
        super().__init__("layout_optimization")

    def run(self, graph: Graph) -> PassResult:
        self.stats["runs"] += 1

        # Analyze layouts
        layouts_optimized = 0

        for node_id, node in graph.nodes.items():
            # For conv2d, prefer NHWC or channels_last on some backends
            if node.op_type == OpType.CONV2D:
                if "memory_format" not in node.attributes:
                    node.attributes["memory_format"] = "channels_first"
                    layouts_optimized += 1

            # For matmul, ensure contiguous memory
            if node.op_type == OpType.MATMUL:
                if "ensure_contiguous" not in node.attributes:
                    node.attributes["ensure_contiguous"] = True
                    layouts_optimized += 1

        changed = layouts_optimized > 0

        if changed:
            self.stats["changes"] += 1

        return PassResult(
            changed=changed,
            nodes_modified=layouts_optimized,
            message=f"Optimized {layouts_optimized} layouts"
        )


class ShapeInference(OptimizationPass):
    """
    Infer and propagate shapes through the graph.

    This enables other optimizations that depend on shape information.
    """

    def __init__(self):
        super().__init__("shape_inference")

    def run(self, graph: Graph) -> PassResult:
        self.stats["runs"] += 1
        shapes_inferred = 0

        # Topological order for forward propagation
        try:
            order = graph.topological_sort()
        except ValueError:
            # Graph has cycles, can't infer shapes
            return PassResult(changed=False, message="Graph has cycles")

        for node_id in order:
            node = graph.nodes[node_id]

            if node.metadata.shape is not None:
                continue  # Already has shape

            inferred = self._infer_shape(graph, node)
            if inferred is not None:
                node.metadata.shape = inferred
                shapes_inferred += 1

        changed = shapes_inferred > 0

        if changed:
            self.stats["changes"] += 1

        return PassResult(
            changed=changed,
            nodes_modified=shapes_inferred,
            message=f"Inferred {shapes_inferred} shapes"
        )

    def _infer_shape(self, graph: Graph, node: Node) -> Optional[Tuple[int, ...]]:
        """Infer output shape for a node."""
        input_shapes = []
        for input_id in node.inputs:
            if input_id in graph.nodes:
                input_node = graph.nodes[input_id]
                if input_node.metadata.shape:
                    input_shapes.append(input_node.metadata.shape)

        if not input_shapes:
            return None

        op = node.op_type

        # Binary elementwise ops - broadcast
        if op in (OpType.ADD, OpType.SUB, OpType.MUL, OpType.DIV):
            if len(input_shapes) == 2:
                return self._broadcast_shapes(input_shapes[0], input_shapes[1])
            elif len(input_shapes) == 1:
                return input_shapes[0]

        # Unary ops - preserve shape
        if op in (OpType.RELU, OpType.SIGMOID, OpType.SOFTMAX):
            return input_shapes[0] if input_shapes else None

        # MatMul
        if op == OpType.MATMUL and len(input_shapes) == 2:
            s1, s2 = input_shapes
            if len(s1) >= 2 and len(s2) >= 2:
                return s1[:-1] + (s2[-1],)

        # Reshape
        if op == OpType.RESHAPE:
            target_shape = node.attributes.get("shape")
            if target_shape:
                return tuple(target_shape)

        # Transpose
        if op == OpType.TRANSPOSE and input_shapes:
            return tuple(reversed(input_shapes[0]))

        # Reductions
        if op in (OpType.SUM, OpType.MEAN, OpType.MAX, OpType.MIN):
            if input_shapes:
                dim = node.attributes.get("dim")
                keepdim = node.attributes.get("keepdim", False)
                shape = list(input_shapes[0])
                if dim is not None:
                    if keepdim:
                        shape[dim] = 1
                    else:
                        del shape[dim]
                    return tuple(shape)

        return None

    def _broadcast_shapes(
        self,
        shape1: Tuple[int, ...],
        shape2: Tuple[int, ...]
    ) -> Optional[Tuple[int, ...]]:
        """Compute broadcast shape."""
        s1, s2 = list(shape1), list(shape2)

        while len(s1) < len(s2):
            s1.insert(0, 1)
        while len(s2) < len(s1):
            s2.insert(0, 1)

        result = []
        for d1, d2 in zip(s1, s2):
            if d1 == d2:
                result.append(d1)
            elif d1 == 1:
                result.append(d2)
            elif d2 == 1:
                result.append(d1)
            else:
                return None  # Incompatible shapes

        return tuple(result)
