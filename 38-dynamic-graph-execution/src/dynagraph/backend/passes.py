"""Lowering passes for graph optimization.

Provides passes for dtype conversion, memory layout optimization,
operator fusion, and other graph transformations.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple
import numpy as np

from .lowering import (
    LoweringPass,
    LoweringContext,
    LoweredGraph,
    LoweredNode,
    TensorSpec,
    DataType,
    MemoryLayout,
)


class DtypeCastPass(LoweringPass):
    """Insert dtype cast operations for type consistency.

    Ensures all operations have consistent data types and inserts
    explicit casts where needed for mixed-precision computation.
    """

    def __init__(self, target_dtype: Optional[DataType] = None):
        self.target_dtype = target_dtype

    @property
    def name(self) -> str:
        return "DtypeCastPass"

    def run(self, graph: LoweredGraph, ctx: LoweringContext) -> LoweredGraph:
        """Run the dtype cast pass."""
        if self.target_dtype is None and ctx.target_dtype is None:
            return graph

        target = self.target_dtype or ctx.target_dtype

        new_nodes = []
        cast_map: Dict[str, str] = {}  # original_tensor -> cast_tensor

        for node in graph.nodes:
            # Check if inputs need casting
            new_inputs = []
            for inp in node.inputs:
                if inp in ctx.tensor_map:
                    tensor = ctx.tensor_map[inp]
                    if tensor.dtype != target and not tensor.is_constant:
                        # Need to cast
                        if inp not in cast_map:
                            cast_name = ctx.new_tensor_name("cast")
                            cast_node = LoweredNode(
                                name=ctx.new_node_name("cast"),
                                op_type="cast",
                                inputs=[inp],
                                outputs=[cast_name],
                                attributes={'to_dtype': target.name},
                            )
                            new_nodes.append(cast_node)
                            cast_map[inp] = cast_name

                            # Register cast tensor
                            cast_spec = TensorSpec(
                                name=cast_name,
                                shape=tensor.shape,
                                dtype=target,
                            )
                            ctx.register_tensor(cast_spec)

                        new_inputs.append(cast_map[inp])
                    else:
                        new_inputs.append(inp)
                else:
                    new_inputs.append(inp)

            # Create node with potentially updated inputs
            new_node = LoweredNode(
                name=node.name,
                op_type=node.op_type,
                inputs=new_inputs,
                outputs=node.outputs,
                attributes=node.attributes,
            )
            new_nodes.append(new_node)

        # Build new graph
        result = LoweredGraph(
            name=graph.name,
            inputs=graph.inputs,
            outputs=graph.outputs,
            constants=graph.constants,
            metadata=graph.metadata,
        )
        for node in new_nodes:
            result.add_node(node)

        return result


class MemoryLayoutPass(LoweringPass):
    """Optimize memory layout for operations.

    Inserts layout transformations to optimize memory access patterns
    for different operation types.
    """

    def __init__(self, prefer_channels_last: bool = False):
        self.prefer_channels_last = prefer_channels_last

    @property
    def name(self) -> str:
        return "MemoryLayoutPass"

    def run(self, graph: LoweredGraph, ctx: LoweringContext) -> LoweredGraph:
        """Run the memory layout optimization pass."""
        new_nodes = []
        layout_map: Dict[str, MemoryLayout] = {}

        for node in graph.nodes:
            # Determine optimal layout for this operation
            optimal_layout = self._get_optimal_layout(node)

            # Check if inputs need layout transformation
            new_inputs = []
            for inp in node.inputs:
                current_layout = layout_map.get(inp, MemoryLayout.ROW_MAJOR)

                if current_layout != optimal_layout and inp in ctx.tensor_map:
                    # Insert transpose/layout transform
                    transform_name = ctx.new_tensor_name("layout")
                    transform_node = LoweredNode(
                        name=ctx.new_node_name("transpose"),
                        op_type="transpose",
                        inputs=[inp],
                        outputs=[transform_name],
                        attributes={
                            'from_layout': current_layout.name,
                            'to_layout': optimal_layout.name,
                        },
                    )
                    new_nodes.append(transform_node)
                    new_inputs.append(transform_name)
                    layout_map[transform_name] = optimal_layout
                else:
                    new_inputs.append(inp)

            # Add node with potentially updated inputs
            new_node = LoweredNode(
                name=node.name,
                op_type=node.op_type,
                inputs=new_inputs,
                outputs=node.outputs,
                attributes=node.attributes,
            )
            new_nodes.append(new_node)

            # Track output layout
            for out in node.outputs:
                layout_map[out] = optimal_layout

        # Build new graph
        result = LoweredGraph(
            name=graph.name,
            inputs=graph.inputs,
            outputs=graph.outputs,
            constants=graph.constants,
            metadata=graph.metadata,
        )
        for node in new_nodes:
            result.add_node(node)

        return result

    def _get_optimal_layout(self, node: LoweredNode) -> MemoryLayout:
        """Determine optimal memory layout for an operation."""
        if node.op_type in ('conv2d', 'batch_norm', 'max_pool', 'avg_pool'):
            if self.prefer_channels_last:
                return MemoryLayout.CHANNELS_LAST
            return MemoryLayout.ROW_MAJOR

        if node.op_type == 'matmul':
            # Matrix ops prefer row-major
            return MemoryLayout.ROW_MAJOR

        return MemoryLayout.ROW_MAJOR


class OpFusionPass(LoweringPass):
    """Fuse compatible operations.

    Identifies and fuses patterns like:
    - matmul + add (gemm)
    - matmul + relu
    - conv + batch_norm + relu
    - add + relu
    """

    @property
    def name(self) -> str:
        return "OpFusionPass"

    def run(self, graph: LoweredGraph, ctx: LoweringContext) -> LoweredGraph:
        """Run the operator fusion pass."""
        # Build output -> consumer map
        consumers: Dict[str, List[LoweredNode]] = {}
        for node in graph.nodes:
            for inp in node.inputs:
                if inp not in consumers:
                    consumers[inp] = []
                consumers[inp].append(node)

        # Track which nodes to remove
        removed: Set[str] = set()
        fused_nodes: List[LoweredNode] = []

        for node in graph.nodes:
            if node.name in removed:
                continue

            # Try to fuse this node with its consumers
            fused = self._try_fuse(node, consumers, removed, ctx)
            fused_nodes.append(fused)

        # Build new graph
        result = LoweredGraph(
            name=graph.name,
            inputs=graph.inputs,
            outputs=graph.outputs,
            constants=graph.constants,
            metadata=graph.metadata,
        )
        for node in fused_nodes:
            result.add_node(node)

        return result

    def _try_fuse(
        self,
        node: LoweredNode,
        consumers: Dict[str, List[LoweredNode]],
        removed: Set[str],
        ctx: LoweringContext,
    ) -> LoweredNode:
        """Try to fuse a node with its consumers."""
        # Matmul + Add -> GEMM
        if node.op_type == 'matmul' and len(node.outputs) == 1:
            output = node.outputs[0]
            if output in consumers and len(consumers[output]) == 1:
                consumer = consumers[output][0]
                if consumer.op_type == 'add' and consumer.name not in removed:
                    # Fuse into GEMM
                    removed.add(consumer.name)
                    return LoweredNode(
                        name=node.name,
                        op_type='gemm',
                        inputs=node.inputs + [c for c in consumer.inputs if c != output],
                        outputs=consumer.outputs,
                        attributes={'alpha': 1.0, 'beta': 1.0},
                    )

        # Matmul + ReLU -> MatMul with ReLU activation
        if node.op_type == 'matmul' and len(node.outputs) == 1:
            output = node.outputs[0]
            if output in consumers and len(consumers[output]) == 1:
                consumer = consumers[output][0]
                if consumer.op_type == 'relu' and consumer.name not in removed:
                    removed.add(consumer.name)
                    return LoweredNode(
                        name=node.name,
                        op_type='matmul_relu',
                        inputs=node.inputs,
                        outputs=consumer.outputs,
                        attributes=node.attributes,
                    )

        # Add + ReLU -> FusedAddReLU
        if node.op_type == 'add' and len(node.outputs) == 1:
            output = node.outputs[0]
            if output in consumers and len(consumers[output]) == 1:
                consumer = consumers[output][0]
                if consumer.op_type == 'relu' and consumer.name not in removed:
                    removed.add(consumer.name)
                    return LoweredNode(
                        name=node.name,
                        op_type='add_relu',
                        inputs=node.inputs,
                        outputs=consumer.outputs,
                        attributes=node.attributes,
                    )

        # Conv + BatchNorm + ReLU fusion
        if node.op_type == 'conv2d' and len(node.outputs) == 1:
            output = node.outputs[0]
            if output in consumers and len(consumers[output]) == 1:
                bn_node = consumers[output][0]
                if bn_node.op_type == 'batch_norm' and bn_node.name not in removed:
                    bn_output = bn_node.outputs[0]
                    if bn_output in consumers and len(consumers[bn_output]) == 1:
                        relu_node = consumers[bn_output][0]
                        if relu_node.op_type == 'relu' and relu_node.name not in removed:
                            removed.add(bn_node.name)
                            removed.add(relu_node.name)
                            return LoweredNode(
                                name=node.name,
                                op_type='conv_bn_relu',
                                inputs=node.inputs + bn_node.inputs[1:],  # Include BN params
                                outputs=relu_node.outputs,
                                attributes={**node.attributes, **bn_node.attributes},
                            )

        return node


class ConstantPropagationPass(LoweringPass):
    """Propagate and fold constant values.

    Evaluates operations with constant inputs at compile time.
    """

    @property
    def name(self) -> str:
        return "ConstantPropagationPass"

    def run(self, graph: LoweredGraph, ctx: LoweringContext) -> LoweredGraph:
        """Run constant propagation pass."""
        # Track constant values
        constants: Dict[str, np.ndarray] = dict(graph.constants)

        # Track which nodes produce constants
        constant_outputs: Set[str] = set(constants.keys())

        new_nodes = []

        for node in graph.nodes:
            # Check if all inputs are constants
            all_const = all(inp in constant_outputs for inp in node.inputs)

            if all_const and self._can_evaluate(node.op_type):
                # Evaluate at compile time
                input_values = [constants[inp] for inp in node.inputs]
                try:
                    result = self._evaluate(node.op_type, input_values, node.attributes)
                    # Store result as constant
                    for i, out in enumerate(node.outputs):
                        if isinstance(result, (list, tuple)):
                            constants[out] = result[i]
                        else:
                            constants[out] = result
                        constant_outputs.add(out)
                    # Don't add node - it's been folded
                    continue
                except Exception:
                    # Fall back to keeping the node
                    pass

            new_nodes.append(node)

        # Build new graph
        result = LoweredGraph(
            name=graph.name,
            inputs=graph.inputs,
            outputs=graph.outputs,
            constants=constants,
            metadata=graph.metadata,
        )
        for node in new_nodes:
            result.add_node(node)

        return result

    def _can_evaluate(self, op_type: str) -> bool:
        """Check if operation can be evaluated at compile time."""
        evaluatable = {
            'add', 'sub', 'mul', 'div', 'neg', 'pow', 'sqrt', 'exp', 'log',
            'abs', 'reshape', 'transpose', 'concat', 'squeeze', 'unsqueeze',
        }
        return op_type in evaluatable

    def _evaluate(
        self, op_type: str, inputs: List[np.ndarray], attrs: Dict
    ) -> np.ndarray:
        """Evaluate an operation with constant inputs."""
        if op_type == 'add':
            return inputs[0] + inputs[1]
        elif op_type == 'sub':
            return inputs[0] - inputs[1]
        elif op_type == 'mul':
            return inputs[0] * inputs[1]
        elif op_type == 'div':
            return inputs[0] / inputs[1]
        elif op_type == 'neg':
            return -inputs[0]
        elif op_type == 'pow':
            return np.power(inputs[0], inputs[1])
        elif op_type == 'sqrt':
            return np.sqrt(inputs[0])
        elif op_type == 'exp':
            return np.exp(inputs[0])
        elif op_type == 'log':
            return np.log(inputs[0])
        elif op_type == 'abs':
            return np.abs(inputs[0])
        elif op_type == 'reshape':
            shape = attrs.get('shape', [-1])
            return np.reshape(inputs[0], shape)
        elif op_type == 'transpose':
            perm = attrs.get('perm', None)
            return np.transpose(inputs[0], perm)
        elif op_type == 'concat':
            axis = attrs.get('axis', 0)
            return np.concatenate(inputs, axis=axis)
        elif op_type == 'squeeze':
            axis = attrs.get('axis', None)
            return np.squeeze(inputs[0], axis=axis)
        elif op_type == 'unsqueeze':
            axis = attrs.get('axis', 0)
            return np.expand_dims(inputs[0], axis=axis)
        else:
            raise ValueError(f"Cannot evaluate {op_type}")


class DeadNodeEliminationPass(LoweringPass):
    """Remove unused nodes from the graph.

    Identifies and removes nodes whose outputs are not used by
    any other node or graph output.
    """

    @property
    def name(self) -> str:
        return "DeadNodeEliminationPass"

    def run(self, graph: LoweredGraph, ctx: LoweringContext) -> LoweredGraph:
        """Run dead node elimination pass."""
        # Find all used tensors (starting from outputs)
        used: Set[str] = set()

        # Add output tensors
        for out in graph.outputs:
            used.add(out.name)

        # Propagate backwards to find all used tensors
        changed = True
        while changed:
            changed = False
            for node in graph.nodes:
                # If any output is used, all inputs are used
                if any(out in used for out in node.outputs):
                    for inp in node.inputs:
                        if inp not in used:
                            used.add(inp)
                            changed = True

        # Keep only nodes with used outputs
        new_nodes = []
        for node in graph.nodes:
            if any(out in used for out in node.outputs):
                new_nodes.append(node)

        # Build new graph
        result = LoweredGraph(
            name=graph.name,
            inputs=graph.inputs,
            outputs=graph.outputs,
            constants={k: v for k, v in graph.constants.items() if k in used},
            metadata=graph.metadata,
        )
        for node in new_nodes:
            result.add_node(node)

        return result
