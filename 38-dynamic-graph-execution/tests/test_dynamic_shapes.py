"""Tests for dynamic shape handling in DynaGraph."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynagraph import Tensor, Parameter, Graph, GraphContext
from dynagraph.executor.executor import EagerExecutor, LazyExecutor
from dynagraph.graph.graph import trace_graph, InputNode, OperationNode


class TestDynamicTensorShapes:
    """Tests for dynamic tensor shape handling."""

    def test_tensor_shape_property(self):
        """Test tensor shape property."""
        x = Tensor([[1, 2, 3], [4, 5, 6]])
        assert x.shape == (2, 3)

    def test_tensor_ndim_property(self):
        """Test tensor ndim property."""
        x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert x.ndim == 3

    def test_tensor_size_property(self):
        """Test tensor size property."""
        x = Tensor([[1, 2], [3, 4], [5, 6]])
        assert x.size == 6

    def test_reshape_changes_shape(self):
        """Test reshape changes tensor shape."""
        x = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
        y = x.reshape(2, 3)

        assert x.shape == (6,)
        assert y.shape == (2, 3)

    def test_view_changes_shape(self):
        """Test view changes tensor shape."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        y = x.view(6)

        assert x.shape == (2, 3)
        assert y.shape == (6,)

    def test_transpose_changes_shape(self):
        """Test transpose changes tensor shape."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        y = x.transpose(0, 1)

        assert x.shape == (2, 3)
        assert y.shape == (3, 2)

    def test_T_property_transposes(self):
        """Test .T property transposes tensor."""
        x = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
        y = x.T

        assert x.shape == (3, 2)
        assert y.shape == (2, 3)


class TestDynamicOperations:
    """Tests for operations with dynamic shapes."""

    def test_matmul_shape_inference(self):
        """Test matrix multiplication shape inference."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 5), requires_grad=True)
        c = a @ b

        assert c.shape == (3, 5)

    def test_batched_matmul_shape(self):
        """Test batched matrix multiplication shape."""
        a = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        b = Tensor(np.random.randn(2, 4, 5), requires_grad=True)
        c = a @ b

        assert c.shape == (2, 3, 5)

    def test_sum_reduces_shape(self):
        """Test sum reduces tensor shape."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

        # Full sum
        y1 = x.sum()
        assert y1.shape == ()

        # Sum along axis
        y2 = x.sum(axis=0)
        assert y2.shape == (3,)

        y3 = x.sum(axis=1)
        assert y3.shape == (2,)

    def test_sum_keepdims(self):
        """Test sum with keepdims preserves dimensions."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

        y = x.sum(axis=0, keepdims=True)
        assert y.shape == (1, 3)

        z = x.sum(axis=1, keepdims=True)
        assert z.shape == (2, 1)

    def test_mean_reduces_shape(self):
        """Test mean reduces tensor shape."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

        y = x.mean()
        assert y.shape == ()

        y_axis = x.mean(axis=1)
        assert y_axis.shape == (2,)

    def test_max_reduces_shape(self):
        """Test max reduces tensor shape."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)

        y = x.max()
        assert y.shape == ()

        y_axis = x.max(axis=0)
        assert y_axis.shape == (3,)

    def test_getitem_changes_shape(self):
        """Test indexing changes shape."""
        x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], requires_grad=True)

        # Single element
        y1 = x[0, 0]
        assert y1.shape == ()

        # Row selection
        y2 = x[0]
        assert y2.shape == (3,)

        # Column selection
        y3 = x[:, 0]
        assert y3.shape == (3,)

        # Slice
        y4 = x[0:2, 1:3]
        assert y4.shape == (2, 2)


class TestBroadcasting:
    """Tests for broadcasting with dynamic shapes."""

    def test_scalar_broadcast(self):
        """Test broadcasting with scalar."""
        x = Tensor([[1, 2], [3, 4]], requires_grad=True)
        y = x + 1

        assert y.shape == (2, 2)

    def test_vector_to_matrix_broadcast(self):
        """Test broadcasting vector to matrix."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        y = x + b

        assert y.shape == (2, 3)

    def test_column_vector_broadcast(self):
        """Test broadcasting column vector."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        b = Tensor([[1], [2]], requires_grad=True)
        y = x + b

        assert y.shape == (2, 3)

    def test_broadcast_gradient_unbroadcast(self):
        """Test gradients are correctly unbroadcast."""
        x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
        b = Tensor([1, 2, 3], requires_grad=True)
        y = (x * b).sum()

        y.backward()

        assert b.grad.shape == b.shape


class TestDynamicGraphConstruction:
    """Tests for dynamic graph construction with varying shapes."""

    def test_graph_input_shapes(self):
        """Test graph captures input shapes."""
        graph = Graph("shape_test")
        x = graph.add_input("x", (4, 8))
        y = graph.add_input("y", (8, 16))

        assert x.shape == (4, 8)
        assert y.shape == (8, 16)

    def test_graph_with_different_batch_sizes(self):
        """Test graph can handle different batch sizes."""
        # Create graph with one batch size
        graph1 = Graph("batch1")
        x1 = graph1.add_input("x", (2, 4))
        y1 = graph1.add_input("y", (4, 3))
        mm1 = graph1.add_operation("matmul", "mm", [x1, y1])
        graph1.add_output("output", mm1)

        # Create graph with different batch size
        graph2 = Graph("batch2")
        x2 = graph2.add_input("x", (8, 4))
        y2 = graph2.add_input("y", (4, 3))
        mm2 = graph2.add_operation("matmul", "mm", [x2, y2])
        graph2.add_output("output", mm2)

        # Both should work
        assert graph1.num_nodes() == graph2.num_nodes()

    def test_trace_graph_captures_shapes(self):
        """Test trace_graph captures dynamic shapes."""
        def model(x, w):
            return x @ w

        x = Tensor(np.random.randn(3, 4))
        w = Tensor(np.random.randn(4, 5))

        graph, output = trace_graph(model, x, w)

        inputs = graph.get_inputs()
        # The trace_graph implementation may create additional input nodes
        # for the output tensor. At minimum we should have the 2 function args.
        assert len(inputs) >= 2
        # Check that arg inputs are captured
        arg_inputs = [inp for inp in inputs if inp.name.startswith("arg_")]
        assert len(arg_inputs) == 2


class TestExecutorDynamicShapes:
    """Tests for executor with dynamic shapes."""

    def test_eager_executor_various_shapes(self):
        """Test eager executor with various input shapes."""
        shapes = [(2, 3), (4, 5), (1, 10), (10, 1)]

        for shape in shapes:
            graph = Graph(f"test_{shape}")
            x = graph.add_input("x", shape)
            y = graph.add_input("y", shape)
            add = graph.add_operation("add", "add", [x, y])
            graph.add_output("output", add)

            executor = EagerExecutor(graph)
            x_data = np.random.randn(*shape).astype(np.float32)
            y_data = np.random.randn(*shape).astype(np.float32)

            result = executor.execute({"x": x_data, "y": y_data})

            expected = x_data + y_data
            np.testing.assert_allclose(result["output"], expected, rtol=1e-5)

    def test_executor_reshape_operation(self):
        """Test executor with reshape operation."""
        graph = Graph("reshape_test")
        x = graph.add_input("x", (2, 6))
        reshape = graph.add_operation("reshape", "reshape", [x], attrs={"shape": (3, 4)})
        graph.add_output("output", reshape)

        executor = EagerExecutor(graph)
        x_data = np.arange(12).reshape(2, 6).astype(np.float32)

        result = executor.execute({"x": x_data})

        assert result["output"].shape == (3, 4)

    def test_executor_transpose_operation(self):
        """Test executor with transpose operation."""
        graph = Graph("transpose_test")
        x = graph.add_input("x", (2, 3, 4))
        transpose = graph.add_operation("transpose", "transpose", [x], attrs={"axes": (2, 0, 1)})
        graph.add_output("output", transpose)

        executor = EagerExecutor(graph)
        x_data = np.random.randn(2, 3, 4).astype(np.float32)

        result = executor.execute({"x": x_data})

        assert result["output"].shape == (4, 2, 3)

    def test_executor_concat_operation(self):
        """Test executor with concat operation."""
        graph = Graph("concat_test")
        x = graph.add_input("x", (2, 3))
        y = graph.add_input("y", (2, 3))
        concat = graph.add_operation("concat", "concat", [x, y], attrs={"axis": 0})
        graph.add_output("output", concat)

        executor = EagerExecutor(graph)
        x_data = np.ones((2, 3)).astype(np.float32)
        y_data = np.zeros((2, 3)).astype(np.float32)

        result = executor.execute({"x": x_data, "y": y_data})

        assert result["output"].shape == (4, 3)


class TestDynamicGraphOperations:
    """Tests for dynamic graph operations that change shapes."""

    def test_subgraph_extraction_preserves_shapes(self):
        """Test subgraph extraction preserves shape information."""
        graph = Graph("main")
        x = graph.add_input("x", (4, 8))
        w = graph.add_input("w", (8, 16))
        mm = graph.add_operation("matmul", "mm", [x, w])
        relu = graph.add_operation("relu", "relu", [mm])
        graph.add_output("output", relu)

        subgraph = graph.subgraph([relu])

        # Check input shapes are preserved
        for inp in subgraph.get_inputs():
            assert inp.shape is not None

    def test_topological_sort_with_shape_ops(self):
        """Test topological sort handles shape-changing operations."""
        graph = Graph("topo_test")
        x = graph.add_input("x", (12,))
        reshape = graph.add_operation("reshape", "reshape", [x], attrs={"shape": (3, 4)})
        transpose = graph.add_operation("transpose", "transpose", [reshape], attrs={"axes": (1, 0)})
        graph.add_output("output", transpose)

        sorted_nodes = graph.topological_sort()

        # Input should come first, then reshape, then transpose
        node_names = [n.name for n in sorted_nodes if hasattr(n, 'name')]
        assert node_names.index("x") < node_names.index("reshape")
        assert node_names.index("reshape") < node_names.index("transpose")


class TestDynamicShapeGradients:
    """Tests for gradients with dynamic shapes."""

    def test_reshape_gradient_shape(self):
        """Test gradient shape matches original tensor shape."""
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        y = x.reshape(6)
        z = y.sum()

        z.backward()

        assert x.grad.shape == x.shape

    def test_transpose_gradient_shape(self):
        """Test transposed gradient has correct shape."""
        x = Tensor(np.random.randn(2, 3), requires_grad=True)
        y = x.T
        z = y.sum()

        z.backward()

        assert x.grad.shape == x.shape

    def test_matmul_gradient_shapes(self):
        """Test matmul gradients have correct shapes."""
        a = Tensor(np.random.randn(3, 4), requires_grad=True)
        b = Tensor(np.random.randn(4, 5), requires_grad=True)
        c = a @ b
        loss = c.sum()

        loss.backward()

        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_reduction_gradient_expansion(self):
        """Test gradient is correctly expanded after reduction."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = x.sum(axis=1)  # (3,)
        z = y.sum()

        z.backward()

        # Gradient should be expanded back to original shape
        assert x.grad.shape == x.shape

    def test_broadcast_gradient_reduction(self):
        """Test gradient is correctly reduced after broadcast."""
        x = Tensor(np.random.randn(1, 4), requires_grad=True)
        y = Tensor(np.random.randn(3, 4), requires_grad=True)
        z = (x + y).sum()  # x is broadcast

        z.backward()

        # x gradient should be reduced to original shape
        assert x.grad.shape == x.shape
        assert y.grad.shape == y.shape


class TestVaryingInputSizes:
    """Tests for handling varying input sizes at runtime."""

    def test_parameter_shape(self):
        """Test Parameter maintains its shape."""
        w = Parameter(np.random.randn(10, 5))
        assert w.shape == (10, 5)

    def test_contiguous_preserves_shape(self):
        """Test contiguous operation preserves shape."""
        x = Tensor(np.random.randn(3, 4), requires_grad=True)
        y = x.contiguous()

        assert y.shape == x.shape

    def test_clone_preserves_shape(self):
        """Test clone operation preserves shape."""
        x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
        y = x.clone()

        assert y.shape == x.shape

    def test_detach_preserves_shape(self):
        """Test detach operation preserves shape."""
        x = Tensor(np.random.randn(5, 6), requires_grad=True)
        y = x.detach()

        assert y.shape == x.shape


class TestShapeErrors:
    """Tests for shape-related error handling."""

    def test_matmul_incompatible_shapes(self):
        """Test matmul raises error for incompatible shapes."""
        a = Tensor(np.random.randn(3, 4))
        b = Tensor(np.random.randn(5, 6))  # Incompatible

        with pytest.raises(Exception):  # Could be ValueError or similar
            c = a @ b

    def test_add_incompatible_shapes(self):
        """Test add with non-broadcastable shapes."""
        a = Tensor(np.random.randn(3, 4))
        b = Tensor(np.random.randn(5, 6))  # Cannot broadcast

        with pytest.raises(Exception):
            c = a + b

    def test_reshape_invalid_size(self):
        """Test reshape with wrong total size."""
        x = Tensor(np.random.randn(2, 3))  # 6 elements

        with pytest.raises(Exception):
            y = x.reshape(2, 4)  # 8 elements - wrong
