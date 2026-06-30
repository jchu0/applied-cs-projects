"""Tests for backend lowering functionality."""

import pytest
import numpy as np

from dynagraph.backend import (
    BackendLowering,
    NativeBackend,
    ONNXBackend,
    LoweredGraph,
    TensorSpec,
    LoweringContext,
    LoweredNode,
    OpMapping,
)
from dynagraph.backend.lowering import DataType, MemoryLayout
from dynagraph.backend.passes import (
    DtypeCastPass,
    MemoryLayoutPass,
    OpFusionPass,
    ConstantPropagationPass,
    DeadNodeEliminationPass,
)


class TestDataType:
    """Tests for DataType enum."""

    def test_from_numpy(self):
        """Test conversion from numpy dtype."""
        assert DataType.from_numpy(np.dtype(np.float32)) == DataType.FLOAT32
        assert DataType.from_numpy(np.dtype(np.float64)) == DataType.FLOAT64
        assert DataType.from_numpy(np.dtype(np.int32)) == DataType.INT32
        assert DataType.from_numpy(np.dtype(np.int64)) == DataType.INT64

    def test_to_numpy(self):
        """Test conversion to numpy dtype."""
        assert DataType.FLOAT32.to_numpy() == np.dtype(np.float32)
        assert DataType.FLOAT64.to_numpy() == np.dtype(np.float64)
        assert DataType.INT32.to_numpy() == np.dtype(np.int32)


class TestTensorSpec:
    """Tests for TensorSpec."""

    def test_tensor_spec_creation(self):
        """Test creating a tensor spec."""
        spec = TensorSpec(
            name="input_0",
            shape=(2, 3, 4),
            dtype=DataType.FLOAT32,
        )

        assert spec.name == "input_0"
        assert spec.shape == (2, 3, 4)
        assert spec.dtype == DataType.FLOAT32

    def test_tensor_spec_size(self):
        """Test tensor spec size calculation."""
        spec = TensorSpec(name="t", shape=(2, 3, 4))
        assert spec.size == 24

    def test_tensor_spec_nbytes(self):
        """Test tensor spec byte size calculation."""
        spec = TensorSpec(name="t", shape=(10,), dtype=DataType.FLOAT32)
        assert spec.nbytes == 40  # 10 * 4 bytes

        spec = TensorSpec(name="t", shape=(10,), dtype=DataType.FLOAT64)
        assert spec.nbytes == 80  # 10 * 8 bytes


class TestLoweredGraph:
    """Tests for LoweredGraph."""

    def test_graph_creation(self):
        """Test creating a lowered graph."""
        graph = LoweredGraph(name="test_graph")
        assert graph.name == "test_graph"
        assert len(graph.nodes) == 0
        assert len(graph.inputs) == 0
        assert len(graph.outputs) == 0

    def test_add_input_output(self):
        """Test adding inputs and outputs."""
        graph = LoweredGraph()

        input_spec = TensorSpec(name="input", shape=(10,))
        output_spec = TensorSpec(name="output", shape=(10,))

        graph.add_input(input_spec)
        graph.add_output(output_spec)

        assert len(graph.inputs) == 1
        assert len(graph.outputs) == 1
        assert graph.inputs[0].name == "input"

    def test_add_node(self):
        """Test adding nodes."""
        graph = LoweredGraph()

        node = LoweredNode(
            name="add_0",
            op_type="add",
            inputs=["a", "b"],
            outputs=["c"],
        )
        graph.add_node(node)

        assert len(graph.nodes) == 1
        assert graph.nodes[0].op_type == "add"

    def test_get_node(self):
        """Test getting nodes by name."""
        graph = LoweredGraph()
        node = LoweredNode(name="test_node", op_type="relu", inputs=["x"], outputs=["y"])
        graph.add_node(node)

        found = graph.get_node("test_node")
        assert found is not None
        assert found.op_type == "relu"

        not_found = graph.get_node("nonexistent")
        assert not_found is None

    def test_topological_sort(self):
        """Test topological sorting of nodes."""
        graph = LoweredGraph()

        # Add nodes in non-topological order
        graph.add_node(LoweredNode(name="c", op_type="relu", inputs=["b_out"], outputs=["c_out"]))
        graph.add_node(LoweredNode(name="a", op_type="input", inputs=[], outputs=["a_out"]))
        graph.add_node(LoweredNode(name="b", op_type="add", inputs=["a_out"], outputs=["b_out"]))

        sorted_nodes = graph.topological_sort()

        # a should come before b, b should come before c
        names = [n.name for n in sorted_nodes]
        assert names.index("a") < names.index("b")
        assert names.index("b") < names.index("c")

    def test_to_dict(self):
        """Test conversion to dictionary."""
        graph = LoweredGraph(name="test")
        graph.add_input(TensorSpec(name="x", shape=(10,), dtype=DataType.FLOAT32))
        graph.add_node(LoweredNode(name="n1", op_type="relu", inputs=["x"], outputs=["y"]))
        graph.add_output(TensorSpec(name="y", shape=(10,), dtype=DataType.FLOAT32))

        d = graph.to_dict()
        assert d['name'] == "test"
        assert len(d['inputs']) == 1
        assert len(d['nodes']) == 1
        assert len(d['outputs']) == 1


class TestLoweringContext:
    """Tests for LoweringContext."""

    def test_context_creation(self):
        """Test creating a lowering context."""
        ctx = LoweringContext()
        assert ctx.op_counter == 0
        assert len(ctx.tensor_map) == 0

    def test_new_names(self):
        """Test generating unique names."""
        ctx = LoweringContext()

        name1 = ctx.new_tensor_name("tensor")
        name2 = ctx.new_tensor_name("tensor")

        assert name1 != name2
        assert name1.startswith("tensor_")
        assert name2.startswith("tensor_")

    def test_register_tensor(self):
        """Test registering tensors."""
        ctx = LoweringContext()
        spec = TensorSpec(name="test", shape=(5,))
        ctx.register_tensor(spec)

        found = ctx.get_tensor("test")
        assert found is spec


class TestNativeBackend:
    """Tests for NativeBackend."""

    def test_backend_creation(self):
        """Test creating native backend."""
        backend = NativeBackend()
        assert backend.name == "native"

    def test_supported_ops(self):
        """Test listing supported operations."""
        backend = NativeBackend()
        ops = backend.supported_ops()

        assert "add" in ops
        assert "mul" in ops
        assert "matmul" in ops
        assert "relu" in ops

    def test_supported_dtypes(self):
        """Test listing supported data types."""
        backend = NativeBackend()
        dtypes = backend.supported_dtypes()

        assert DataType.FLOAT32 in dtypes
        assert DataType.FLOAT64 in dtypes
        assert DataType.INT32 in dtypes

    def test_lower_simple_graph(self):
        """Test lowering a simple graph."""
        backend = NativeBackend()

        graph = {'inputs': 1, 'outputs': 1, 'ops': []}
        lowered = backend.lower(graph)

        assert isinstance(lowered, LoweredGraph)

    def test_compile_and_execute(self):
        """Test compiling and executing a graph."""
        backend = NativeBackend()

        # Create a simple graph
        lowered = LoweredGraph(name="test")
        lowered.add_input(TensorSpec(name="x", shape=(3,), dtype=DataType.FLOAT32))
        lowered.add_node(LoweredNode(
            name="add_1",
            op_type="add",
            inputs=["x", "x"],
            outputs=["y"],
        ))
        lowered.add_output(TensorSpec(name="y", shape=(3,), dtype=DataType.FLOAT32))

        # Compile
        compiled = backend.compile(lowered)
        assert compiled is not None

        # Execute
        inputs = {"x": np.array([1.0, 2.0, 3.0])}
        outputs = backend.execute(compiled, inputs)

        np.testing.assert_array_almost_equal(
            outputs["y"],
            np.array([2.0, 4.0, 6.0])
        )

    def test_generate_code(self):
        """Test code generation for nodes."""
        backend = NativeBackend()

        node = LoweredNode(
            name="add_0",
            op_type="add",
            inputs=["a", "b"],
            outputs=["c"],
        )
        code = backend.generate_code(node)
        assert "c = a + b" in code


class TestONNXBackend:
    """Tests for ONNXBackend."""

    def test_backend_creation(self):
        """Test creating ONNX backend."""
        backend = ONNXBackend()
        assert backend.name == "onnx"

    def test_supported_ops(self):
        """Test listing supported operations."""
        backend = ONNXBackend()
        ops = backend.supported_ops()

        assert "add" in ops
        assert "matmul" in ops
        assert "relu" in ops

    def test_to_onnx_dict(self):
        """Test converting to ONNX dictionary."""
        backend = ONNXBackend()

        lowered = LoweredGraph(name="test")
        lowered.add_input(TensorSpec(name="x", shape=(10,), dtype=DataType.FLOAT32))
        lowered.add_node(LoweredNode(
            name="relu_0",
            op_type="relu",
            inputs=["x"],
            outputs=["y"],
        ))
        lowered.add_output(TensorSpec(name="y", shape=(10,), dtype=DataType.FLOAT32))

        onnx_dict = backend.to_onnx_dict(lowered)

        assert 'name' in onnx_dict
        assert 'nodes' in onnx_dict
        assert len(onnx_dict['nodes']) == 1


class TestLoweringPasses:
    """Tests for lowering passes."""

    def test_constant_propagation(self):
        """Test constant propagation pass."""
        ctx = LoweringContext()
        pass_ = ConstantPropagationPass()

        # Create graph with constant operation
        graph = LoweredGraph()
        graph.add_constant("a", np.array([1.0, 2.0]))
        graph.add_constant("b", np.array([3.0, 4.0]))
        graph.add_node(LoweredNode(
            name="add",
            op_type="add",
            inputs=["a", "b"],
            outputs=["c"],
        ))
        graph.add_output(TensorSpec(name="c", shape=(2,)))

        result = pass_.run(graph, ctx)

        # Should fold the constant add
        assert "c" in result.constants

    def test_dead_node_elimination(self):
        """Test dead node elimination pass."""
        ctx = LoweringContext()
        pass_ = DeadNodeEliminationPass()

        # Create graph with unused node
        graph = LoweredGraph()
        graph.add_node(LoweredNode(name="used", op_type="relu", inputs=["x"], outputs=["y"]))
        graph.add_node(LoweredNode(name="unused", op_type="relu", inputs=["x"], outputs=["z"]))
        graph.add_output(TensorSpec(name="y", shape=(10,)))

        result = pass_.run(graph, ctx)

        # Should remove unused node
        assert len(result.nodes) == 1
        assert result.nodes[0].name == "used"

    def test_op_fusion_add_relu(self):
        """Test add+relu fusion."""
        ctx = LoweringContext()
        pass_ = OpFusionPass()

        # Create graph with add followed by relu
        graph = LoweredGraph()
        graph.add_node(LoweredNode(name="add", op_type="add", inputs=["a", "b"], outputs=["c"]))
        graph.add_node(LoweredNode(name="relu", op_type="relu", inputs=["c"], outputs=["d"]))
        graph.add_output(TensorSpec(name="d", shape=(10,)))

        result = pass_.run(graph, ctx)

        # Should fuse into add_relu
        fused_ops = [n.op_type for n in result.nodes]
        assert "add_relu" in fused_ops or ("add" in fused_ops and "relu" not in fused_ops)


class TestOpMapping:
    """Tests for OpMapping."""

    def test_op_mapping_creation(self):
        """Test creating an op mapping."""
        mapping = OpMapping(
            op_type="add",
            backend_op="Add",
            inputs=["a", "b"],
            outputs=["c"],
            attributes={"alpha": 1.0},
        )

        assert mapping.op_type == "add"
        assert mapping.backend_op == "Add"
        assert len(mapping.inputs) == 2

    def test_op_mapping_repr(self):
        """Test op mapping string representation."""
        mapping = OpMapping(op_type="mul", backend_op="Mul", inputs=[], outputs=[])
        assert "mul" in repr(mapping)
        assert "Mul" in repr(mapping)
