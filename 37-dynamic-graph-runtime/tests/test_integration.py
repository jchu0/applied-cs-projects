"""Integration tests for dynamic graph execution runtime."""

import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynamicgraph.core import (
    Graph, Node, OpType, SymbolicTensor, NodeMetadata,
    ExecutionContext, CompilationContext, no_grad, enable_grad, get_current_context
)
from dynamicgraph.tracer import BytecodeTracer, TracingMode


class TestEndToEndGraphConstruction(unittest.TestCase):
    """Test end-to-end graph construction scenarios."""

    def test_simple_linear_model(self):
        """Test building a simple linear model graph."""
        graph = Graph(name="linear_model")

        # Input node
        input_node = Node(
            op_type=OpType.INPUT,
            name="input",
            metadata=NodeMetadata(shape=(32, 10), dtype="float32")
        )
        input_id = graph.add_node(input_node)

        # Weight parameter
        weight_node = Node(
            op_type=OpType.PARAMETER,
            name="weight",
            metadata=NodeMetadata(
                shape=(10, 5),
                dtype="float32",
                is_parameter=True,
                requires_grad=True
            )
        )
        weight_id = graph.add_node(weight_node)

        # Bias parameter
        bias_node = Node(
            op_type=OpType.PARAMETER,
            name="bias",
            metadata=NodeMetadata(
                shape=(5,),
                dtype="float32",
                is_parameter=True,
                requires_grad=True
            )
        )
        bias_id = graph.add_node(bias_node)

        # MatMul operation
        matmul_node = Node(
            op_type=OpType.MATMUL,
            name="matmul"
        )
        matmul_id = graph.add_node(matmul_node)
        graph.add_edge(input_id, matmul_id, index=0)
        graph.add_edge(weight_id, matmul_id, index=1)

        # Add bias
        add_node = Node(
            op_type=OpType.ADD,
            name="add_bias"
        )
        add_id = graph.add_node(add_node)
        graph.add_edge(matmul_id, add_id, index=0)
        graph.add_edge(bias_id, add_id, index=1)

        # ReLU activation
        relu_node = Node(
            op_type=OpType.RELU,
            name="relu"
        )
        relu_id = graph.add_node(relu_node)
        graph.add_edge(add_id, relu_id)

        # Output node
        output_node = Node(
            op_type=OpType.OUTPUT,
            name="output"
        )
        output_id = graph.add_node(output_node)
        graph.add_edge(relu_id, output_id)

        # Validate graph
        issues = graph.validate()
        self.assertEqual(len(issues), 0)

        # Check topological order
        topo_order = graph.topological_sort()
        self.assertIn(input_id, topo_order[:3])  # Inputs/params first
        self.assertIn(weight_id, topo_order[:3])
        self.assertIn(bias_id, topo_order[:3])
        self.assertEqual(topo_order[-1], output_id)  # Output last

        # Verify no cycles
        self.assertFalse(graph.has_cycle())

    def test_conv_batch_norm_graph(self):
        """Test building a Conv2D -> BatchNorm -> ReLU graph."""
        graph = Graph(name="conv_bn_relu")

        # Input: NCHW format
        input_node = Node(
            op_type=OpType.INPUT,
            name="input",
            metadata=NodeMetadata(shape=(1, 3, 224, 224), dtype="float32")
        )
        input_id = graph.add_node(input_node)

        # Conv2D parameters
        conv_weight = Node(
            op_type=OpType.PARAMETER,
            name="conv.weight",
            metadata=NodeMetadata(
                shape=(64, 3, 7, 7),
                dtype="float32",
                is_parameter=True
            )
        )
        conv_weight_id = graph.add_node(conv_weight)

        conv_bias = Node(
            op_type=OpType.PARAMETER,
            name="conv.bias",
            metadata=NodeMetadata(
                shape=(64,),
                dtype="float32",
                is_parameter=True
            )
        )
        conv_bias_id = graph.add_node(conv_bias)

        # Conv2D operation
        conv_node = Node(
            op_type=OpType.CONV2D,
            name="conv2d",
            attributes={
                "stride": (2, 2),
                "padding": (3, 3),
                "dilation": (1, 1),
                "groups": 1
            }
        )
        conv_id = graph.add_node(conv_node)
        graph.add_edge(input_id, conv_id, index=0)
        graph.add_edge(conv_weight_id, conv_id, index=1)
        graph.add_edge(conv_bias_id, conv_id, index=2)

        # BatchNorm parameters
        bn_weight = Node(
            op_type=OpType.PARAMETER,
            name="bn.weight",
            metadata=NodeMetadata(shape=(64,), is_parameter=True)
        )
        bn_weight_id = graph.add_node(bn_weight)

        bn_bias = Node(
            op_type=OpType.PARAMETER,
            name="bn.bias",
            metadata=NodeMetadata(shape=(64,), is_parameter=True)
        )
        bn_bias_id = graph.add_node(bn_bias)

        bn_mean = Node(
            op_type=OpType.BUFFER,
            name="bn.running_mean",
            metadata=NodeMetadata(shape=(64,), is_buffer=True)
        )
        bn_mean_id = graph.add_node(bn_mean)

        bn_var = Node(
            op_type=OpType.BUFFER,
            name="bn.running_var",
            metadata=NodeMetadata(shape=(64,), is_buffer=True)
        )
        bn_var_id = graph.add_node(bn_var)

        # BatchNorm operation
        bn_node = Node(
            op_type=OpType.BATCHNORM,
            name="batch_norm",
            attributes={
                "eps": 1e-5,
                "momentum": 0.1,
                "training": True
            }
        )
        bn_id = graph.add_node(bn_node)
        graph.add_edge(conv_id, bn_id, index=0)
        graph.add_edge(bn_weight_id, bn_id, index=1)
        graph.add_edge(bn_bias_id, bn_id, index=2)
        graph.add_edge(bn_mean_id, bn_id, index=3)
        graph.add_edge(bn_var_id, bn_id, index=4)

        # ReLU
        relu_node = Node(op_type=OpType.RELU, name="relu")
        relu_id = graph.add_node(relu_node)
        graph.add_edge(bn_id, relu_id)

        # Output
        output_node = Node(op_type=OpType.OUTPUT, name="output")
        output_id = graph.add_node(output_node)
        graph.add_edge(relu_id, output_id)

        # Validate
        self.assertEqual(len(graph.nodes), 11)
        self.assertFalse(graph.has_cycle())
        self.assertEqual(len(graph.validate()), 0)


class TestExecutionContext(unittest.TestCase):
    """Test execution context management."""

    def test_context_creation(self):
        """Test creating execution context."""
        context = ExecutionContext()
        self.assertIsNotNone(context.compilation_context)
        self.assertEqual(len(context.tensor_cache), 0)
        self.assertTrue(context.is_training)
        self.assertTrue(context.is_grad_enabled)

    def test_tensor_cache_operations(self):
        """Test tensor cache operations."""
        context = ExecutionContext()

        # Create and cache tensors
        tensor1 = SymbolicTensor(node_id="t1")
        tensor2 = SymbolicTensor(node_id="t2")

        context.set_tensor("t1", tensor1)
        context.set_tensor("t2", tensor2)

        # Retrieve tensors
        self.assertEqual(context.get_tensor("t1"), tensor1)
        self.assertEqual(context.get_tensor("t2"), tensor2)
        self.assertIsNone(context.get_tensor("t3"))

        # Clear cache
        context.clear_cache()
        self.assertEqual(len(context.tensor_cache), 0)

    def test_parameter_management(self):
        """Test parameter and buffer management."""
        context = ExecutionContext()

        # Set parameters
        weight = np.random.randn(10, 5).astype(np.float32)
        bias = np.zeros(5, dtype=np.float32)

        context.set_parameter("weight", weight)
        context.set_parameter("bias", bias)

        # Get parameters
        np.testing.assert_array_equal(context.get_parameter("weight"), weight)
        np.testing.assert_array_equal(context.get_parameter("bias"), bias)

        # Set buffers
        running_mean = np.zeros(5, dtype=np.float32)
        context.set_buffer("running_mean", running_mean)

        np.testing.assert_array_equal(
            context.get_buffer("running_mean"),
            running_mean
        )

    def test_graph_stack_operations(self):
        """Test graph stack push/pop."""
        context = ExecutionContext()

        graph1 = Graph(name="graph1")
        graph2 = Graph(name="graph2")
        graph3 = Graph(name="graph3")

        # Push graphs
        context.current_graph = graph1
        context.push_graph(graph2)
        self.assertEqual(context.current_graph, graph2)

        context.push_graph(graph3)
        self.assertEqual(context.current_graph, graph3)

        # Pop graphs
        popped = context.pop_graph()
        self.assertEqual(popped, graph3)
        self.assertEqual(context.current_graph, graph2)

        popped = context.pop_graph()
        self.assertEqual(popped, graph2)
        self.assertEqual(context.current_graph, graph1)

    def test_gradient_context_managers(self):
        """Test gradient enable/disable context managers."""
        # Use the global context since no_grad/enable_grad work on the global context
        context = get_current_context()

        # Initially enabled
        self.assertTrue(context.is_grad_enabled)

        # Disable gradients
        with no_grad():
            self.assertFalse(get_current_context().is_grad_enabled)

            # Nested enable
            with enable_grad():
                self.assertTrue(get_current_context().is_grad_enabled)

            # Back to disabled
            self.assertFalse(get_current_context().is_grad_enabled)

        # Back to enabled
        self.assertTrue(get_current_context().is_grad_enabled)


class TestCompilationContext(unittest.TestCase):
    """Test compilation context."""

    def test_compilation_settings(self):
        """Test compilation context settings."""
        context = CompilationContext(
            optimization_level=2,
            backend="tensorrt",
            device="cuda:0",
            enable_profiling=True
        )

        self.assertEqual(context.optimization_level, 2)
        self.assertEqual(context.backend, "tensorrt")
        self.assertEqual(context.device, "cuda:0")
        self.assertTrue(context.enable_profiling)

    def test_should_compile_logic(self):
        """Test compilation decision logic."""
        context = CompilationContext(
            optimization_level=1,
            min_graph_size=10,
            max_graph_size=1000
        )

        # Small graph - should not compile
        small_graph = Graph()
        for i in range(5):
            small_graph.add_node(Node())

        self.assertFalse(context.should_compile(small_graph))

        # Medium graph - should compile
        medium_graph = Graph()
        nodes = []
        for i in range(50):
            node = Node()
            node_id = medium_graph.add_node(node)
            nodes.append(node_id)

        # Add edges to make it a chain
        for i in range(len(nodes) - 1):
            medium_graph.add_edge(nodes[i], nodes[i + 1])

        self.assertTrue(context.should_compile(medium_graph))

        # Large graph - should not compile
        large_graph = Graph()
        for i in range(2000):
            large_graph.add_node(Node())

        self.assertFalse(context.should_compile(large_graph))

        # Graph with cycle - should not compile
        cycle_graph = Graph()
        n1 = cycle_graph.add_node(Node())
        n2 = cycle_graph.add_node(Node())
        n3 = cycle_graph.add_node(Node())
        cycle_graph.add_edge(n1, n2)
        cycle_graph.add_edge(n2, n3)
        cycle_graph.add_edge(n3, n1)  # Create cycle

        self.assertFalse(context.should_compile(cycle_graph))

    def test_compilation_statistics(self):
        """Test compilation statistics tracking."""
        context = CompilationContext()

        self.assertEqual(context.num_compilations, 0)
        self.assertEqual(context.num_cache_hits, 0)
        self.assertEqual(context.num_graph_breaks, 0)

        # Simulate compilation events
        context.num_compilations += 1
        context.num_cache_hits += 5
        context.num_graph_breaks += 2
        context.compilation_time_ms += 123.45

        self.assertEqual(context.num_compilations, 1)
        self.assertEqual(context.num_cache_hits, 5)
        self.assertEqual(context.num_graph_breaks, 2)
        self.assertAlmostEqual(context.compilation_time_ms, 123.45)


class TestBytecodeTracing(unittest.TestCase):
    """Test bytecode tracing functionality."""

    def test_tracer_initialization(self):
        """Test tracer initialization."""
        tracer = BytecodeTracer()

        self.assertIsNotNone(tracer.graph)
        self.assertEqual(tracer.mode, TracingMode.NONE)
        self.assertEqual(len(tracer.frames), 0)
        self.assertEqual(len(tracer.symbolic_values), 0)

    def test_simple_function_tracing(self):
        """Test tracing a simple function."""
        tracer = BytecodeTracer()

        def simple_add(x, y):
            return x + y

        # Trace the function
        x = np.array([1, 2, 3], dtype=np.float32)
        y = np.array([4, 5, 6], dtype=np.float32)

        graph = tracer.trace_function(simple_add, x, y)

        # Check that input nodes were created
        self.assertEqual(len(graph.input_nodes), 2)
        self.assertGreater(len(graph.nodes), 0)

        # Check that symbolic values were created
        self.assertIn("arg_0", tracer.symbolic_values)
        self.assertIn("arg_1", tracer.symbolic_values)

    def test_graph_break_recording(self):
        """Test recording graph break reasons."""
        tracer = BytecodeTracer()

        tracer.record_graph_break("Unsupported operation: print")
        tracer.record_graph_break("Side effect detected")

        self.assertEqual(len(tracer.graph_break_reasons), 2)
        self.assertIn("Unsupported operation: print", tracer.graph_break_reasons)

    def test_tracer_reset(self):
        """Test resetting tracer state."""
        tracer = BytecodeTracer()

        # Add some state
        tracer.symbolic_values["test"] = SymbolicTensor()
        tracer.record_graph_break("test break")
        tracer.graph.add_node(Node())

        # Reset
        tracer.reset()

        self.assertEqual(len(tracer.graph.nodes), 0)
        self.assertEqual(len(tracer.symbolic_values), 0)
        self.assertEqual(len(tracer.graph_break_reasons), 0)
        self.assertEqual(tracer.mode, TracingMode.NONE)


if __name__ == "__main__":
    unittest.main()