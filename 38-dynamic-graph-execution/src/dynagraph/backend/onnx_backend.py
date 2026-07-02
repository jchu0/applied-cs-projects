"""ONNX backend for graph export.

Exports computation graphs to ONNX format for interoperability
with other frameworks and deployment targets.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .lowering import (
    BackendLowering,
    LoweringContext,
    LoweredGraph,
    LoweredNode,
    TensorSpec,
    DataType,
)


# ONNX data type mapping
DTYPE_TO_ONNX = {
    DataType.FLOAT32: 1,   # TensorProto.FLOAT
    DataType.FLOAT64: 11,  # TensorProto.DOUBLE
    DataType.FLOAT16: 10,  # TensorProto.FLOAT16
    DataType.INT8: 3,      # TensorProto.INT8
    DataType.INT16: 5,     # TensorProto.INT16
    DataType.INT32: 6,     # TensorProto.INT32
    DataType.INT64: 7,     # TensorProto.INT64
    DataType.UINT8: 2,     # TensorProto.UINT8
    DataType.BOOL: 9,      # TensorProto.BOOL
}


class ONNXNode:
    """ONNX node representation."""

    def __init__(
        self,
        op_type: str,
        inputs: List[str],
        outputs: List[str],
        name: str = "",
        domain: str = "",
        attributes: Optional[Dict[str, Any]] = None
    ):
        self.op_type = op_type
        self.inputs = inputs
        self.outputs = outputs
        self.name = name
        self.domain = domain
        self.attributes = attributes or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'op_type': self.op_type,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'name': self.name,
            'domain': self.domain,
            'attributes': self.attributes,
        }


class ONNXGraph:
    """ONNX graph representation."""

    def __init__(self, name: str = "main"):
        self.name = name
        self.nodes: List[ONNXNode] = []
        self.inputs: List[Dict] = []
        self.outputs: List[Dict] = []
        self.initializers: Dict[str, np.ndarray] = {}
        self.opset_version: int = 13

    def add_input(
        self,
        name: str,
        dtype: DataType,
        shape: Tuple[int, ...]
    ) -> None:
        """Add an input to the graph."""
        self.inputs.append({
            'name': name,
            'dtype': DTYPE_TO_ONNX.get(dtype, 1),
            'shape': shape,
        })

    def add_output(
        self,
        name: str,
        dtype: DataType,
        shape: Tuple[int, ...]
    ) -> None:
        """Add an output to the graph."""
        self.outputs.append({
            'name': name,
            'dtype': DTYPE_TO_ONNX.get(dtype, 1),
            'shape': shape,
        })

    def add_node(self, node: ONNXNode) -> None:
        """Add a node to the graph."""
        self.nodes.append(node)

    def add_initializer(self, name: str, value: np.ndarray) -> None:
        """Add a constant initializer."""
        self.initializers[name] = value

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'name': self.name,
            'opset_version': self.opset_version,
            'inputs': self.inputs,
            'outputs': self.outputs,
            'nodes': [n.to_dict() for n in self.nodes],
            'initializers': list(self.initializers.keys()),
        }


class ONNXBackend(BackendLowering):
    """ONNX export backend."""

    def __init__(self):
        super().__init__()
        self._op_mapping: Dict[str, str] = {}
        self._register_default_mappings()

    @property
    def name(self) -> str:
        return "onnx"

    def supported_ops(self) -> List[str]:
        return list(self._op_mapping.keys())

    def supported_dtypes(self) -> List[DataType]:
        return list(DTYPE_TO_ONNX.keys())

    def _register_default_mappings(self) -> None:
        """Register default operation mappings to ONNX."""
        # Elementwise operations
        self._op_mapping['add'] = 'Add'
        self._op_mapping['sub'] = 'Sub'
        self._op_mapping['mul'] = 'Mul'
        self._op_mapping['div'] = 'Div'
        self._op_mapping['neg'] = 'Neg'
        self._op_mapping['pow'] = 'Pow'
        self._op_mapping['sqrt'] = 'Sqrt'
        self._op_mapping['exp'] = 'Exp'
        self._op_mapping['log'] = 'Log'
        self._op_mapping['abs'] = 'Abs'

        # Activation functions
        self._op_mapping['relu'] = 'Relu'
        self._op_mapping['sigmoid'] = 'Sigmoid'
        self._op_mapping['tanh'] = 'Tanh'
        self._op_mapping['softmax'] = 'Softmax'
        self._op_mapping['gelu'] = 'Gelu'  # ONNX 20+

        # Reduction operations
        self._op_mapping['sum'] = 'ReduceSum'
        self._op_mapping['mean'] = 'ReduceMean'
        self._op_mapping['max'] = 'ReduceMax'
        self._op_mapping['min'] = 'ReduceMin'

        # Matrix operations
        self._op_mapping['matmul'] = 'MatMul'
        self._op_mapping['transpose'] = 'Transpose'
        self._op_mapping['reshape'] = 'Reshape'

        # Comparison operations
        self._op_mapping['equal'] = 'Equal'
        self._op_mapping['greater'] = 'Greater'
        self._op_mapping['less'] = 'Less'

        # Other operations
        self._op_mapping['concat'] = 'Concat'
        self._op_mapping['split'] = 'Split'
        self._op_mapping['gather'] = 'Gather'
        self._op_mapping['unsqueeze'] = 'Unsqueeze'
        self._op_mapping['squeeze'] = 'Squeeze'

    def _initial_lowering(self, graph: Any, ctx: LoweringContext) -> LoweredGraph:
        """Perform initial lowering from high-level graph."""
        lowered = LoweredGraph(name="onnx_graph")

        # Handle different graph input types
        if isinstance(graph, dict):
            # Traced graph from jit_trace
            for i in range(graph.get('inputs', 0)):
                spec = TensorSpec(
                    name=f"input_{i}",
                    shape=(-1,),
                    dtype=DataType.FLOAT32,
                )
                lowered.add_input(spec)

            for i in range(graph.get('outputs', 0)):
                spec = TensorSpec(
                    name=f"output_{i}",
                    shape=(-1,),
                    dtype=DataType.FLOAT32,
                )
                lowered.add_output(spec)

            for op in graph.get('ops', []):
                node = LoweredNode(
                    name=ctx.new_node_name(op.get('type', 'unknown')),
                    op_type=op.get('type', 'unknown'),
                    inputs=op.get('inputs', []),
                    outputs=op.get('outputs', [ctx.new_tensor_name()]),
                    attributes=op.get('attributes', {}),
                )
                lowered.add_node(node)

        elif hasattr(graph, 'nodes'):
            for node in graph.nodes:
                op_type = getattr(node, 'op_type', 'unknown')
                inputs = getattr(node, 'inputs', [])
                outputs = getattr(node, 'outputs', [ctx.new_tensor_name()])

                lowered_node = LoweredNode(
                    name=ctx.new_node_name(op_type),
                    op_type=op_type,
                    inputs=[str(i) for i in inputs],
                    outputs=[str(o) for o in outputs],
                )
                lowered.add_node(lowered_node)

        return lowered

    def compile(self, graph: LoweredGraph) -> ONNXGraph:
        """Compile lowered graph to ONNX format."""
        onnx_graph = ONNXGraph(name=graph.name)

        # Add inputs
        for inp in graph.inputs:
            onnx_graph.add_input(inp.name, inp.dtype, inp.shape)

        # Add outputs
        for out in graph.outputs:
            onnx_graph.add_output(out.name, out.dtype, out.shape)

        # Add constants
        for name, value in graph.constants.items():
            onnx_graph.add_initializer(name, value)

        # Convert nodes
        for node in graph.topological_sort():
            onnx_node = self._convert_node(node)
            if onnx_node:
                onnx_graph.add_node(onnx_node)

        return onnx_graph

    def _convert_node(self, node: LoweredNode) -> Optional[ONNXNode]:
        """Convert a lowered node to ONNX node."""
        onnx_op = self._op_mapping.get(node.op_type)
        if onnx_op is None:
            # Try direct mapping
            onnx_op = node.op_type.capitalize()

        # Convert attributes to ONNX format
        onnx_attrs = self._convert_attributes(node.op_type, node.attributes)

        return ONNXNode(
            op_type=onnx_op,
            inputs=node.inputs,
            outputs=node.outputs,
            name=node.name,
            attributes=onnx_attrs,
        )

    def _convert_attributes(self, op_type: str, attrs: Dict) -> Dict:
        """Convert attributes to ONNX format."""
        onnx_attrs = {}

        # Handle specific attribute conversions
        if op_type in ('sum', 'mean', 'max', 'min'):
            if 'axis' in attrs:
                axis = attrs['axis']
                if axis is not None:
                    onnx_attrs['axes'] = [axis] if isinstance(axis, int) else list(axis)
            if 'keepdims' in attrs:
                onnx_attrs['keepdims'] = int(attrs['keepdims'])

        elif op_type == 'transpose':
            if 'perm' in attrs:
                onnx_attrs['perm'] = list(attrs['perm'])

        elif op_type == 'softmax':
            if 'axis' in attrs:
                onnx_attrs['axis'] = attrs['axis']

        elif op_type == 'reshape':
            # Reshape shape is passed as second input in ONNX
            pass

        else:
            # Pass through other attributes
            onnx_attrs.update(attrs)

        return onnx_attrs

    def execute(
        self, compiled: ONNXGraph, inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Execute an ONNX graph in-process.

        Not implemented: this backend only exports ONNX (see ``export_onnx``
        and ``to_onnx_dict``); it does not run an in-process ONNX inference
        session. This raises ``NotImplementedError`` unconditionally regardless
        of whether ``onnxruntime`` is installed.
        """
        raise NotImplementedError(
            "ONNXBackend.execute() is not implemented: this backend only "
            "exports ONNX graphs. Use export_onnx() to save a model and run it "
            "with onnxruntime (or another ONNX runtime) separately."
        )

    def export_onnx(self, graph: LoweredGraph, filepath: str) -> None:
        """
        Export graph to ONNX file.

        Requires onnx package to be installed.
        """
        try:
            import onnx
            from onnx import helper, TensorProto
        except ImportError:
            raise RuntimeError("onnx package is required for ONNX export")

        # Compile to ONNX representation
        onnx_graph = self.compile(graph)

        # Build ONNX proto
        inputs = []
        for inp in onnx_graph.inputs:
            inputs.append(helper.make_tensor_value_info(
                inp['name'],
                inp['dtype'],
                inp['shape'],
            ))

        outputs = []
        for out in onnx_graph.outputs:
            outputs.append(helper.make_tensor_value_info(
                out['name'],
                out['dtype'],
                out['shape'],
            ))

        nodes = []
        for node in onnx_graph.nodes:
            nodes.append(helper.make_node(
                node.op_type,
                node.inputs,
                node.outputs,
                name=node.name,
                **node.attributes,
            ))

        initializers = []
        for name, value in onnx_graph.initializers.items():
            initializers.append(onnx.numpy_helper.from_array(value, name))

        graph_def = helper.make_graph(
            nodes,
            onnx_graph.name,
            inputs,
            outputs,
            initializers,
        )

        model_def = helper.make_model(
            graph_def,
            opset_imports=[helper.make_opsetid("", onnx_graph.opset_version)],
        )

        onnx.checker.check_model(model_def)
        onnx.save(model_def, filepath)

    def to_onnx_dict(self, graph: LoweredGraph) -> Dict:
        """
        Convert graph to ONNX-compatible dictionary.

        Can be used without the onnx package for inspection.
        """
        onnx_graph = self.compile(graph)
        return onnx_graph.to_dict()
