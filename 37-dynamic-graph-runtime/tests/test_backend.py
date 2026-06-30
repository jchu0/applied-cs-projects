"""Tests for backends and compilation."""

import builtins
import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynamicgraph.core.graph import Graph, Node, Edge, OpType, NodeMetadata
from dynamicgraph.codegen.backend import (
    Backend,
    BackendRegistry,
    EagerBackend,
    CompiledFunction,
)
from dynamicgraph.codegen.compiler import (
    DynamicCompiler,
    CompilationCache,
    Guard,
    ShapeGuard,
    DtypeGuard,
    compile,
    trace,
)


class TestEagerBackend(unittest.TestCase):
    """Tests for EagerBackend."""

    def setUp(self):
        self.backend = EagerBackend()

    def test_backend_available(self):
        """Test that eager backend is always available."""
        self.assertTrue(self.backend.is_available())

    def test_backend_name(self):
        """Test backend name."""
        self.assertEqual(self.backend.name(), "eager_numpy")

    def test_compile_simple_add(self):
        """Test compiling a simple addition graph."""
        graph = Graph(name="add_graph")

        # Create: output = x + y
        x = Node(op_type=OpType.INPUT, name="x")
        y = Node(op_type=OpType.INPUT, name="y")
        graph.add_node(x)
        graph.add_node(y)

        add = Node(op_type=OpType.ADD, name="add")
        add.inputs = [x.id, y.id]
        graph.add_node(add)
        graph.add_edge(x.id, add.id)
        graph.add_edge(y.id, add.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [add.id]
        graph.add_node(out)
        graph.add_edge(add.id, out.id)

        # Compile
        compiled = self.backend.compile(graph)

        self.assertIsInstance(compiled, CompiledFunction)
        self.assertEqual(compiled.backend_name, "eager_numpy")

        # Execute
        result = compiled(np.array([1, 2, 3]), np.array([4, 5, 6]))
        expected = np.array([5, 7, 9])
        np.testing.assert_array_equal(result, expected)

    def test_compile_matmul(self):
        """Test compiling matrix multiplication."""
        graph = Graph(name="matmul_graph")

        x = Node(op_type=OpType.INPUT, name="x")
        y = Node(op_type=OpType.INPUT, name="y")
        graph.add_node(x)
        graph.add_node(y)

        matmul = Node(op_type=OpType.MATMUL, name="matmul")
        matmul.inputs = [x.id, y.id]
        graph.add_node(matmul)
        graph.add_edge(x.id, matmul.id)
        graph.add_edge(y.id, matmul.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [matmul.id]
        graph.add_node(out)
        graph.add_edge(matmul.id, out.id)

        compiled = self.backend.compile(graph)

        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)
        result = compiled(A, B)

        expected = np.matmul(A, B)
        np.testing.assert_array_almost_equal(result, expected)

    def test_compile_relu(self):
        """Test compiling ReLU activation."""
        graph = Graph(name="relu_graph")

        x = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(x)

        relu = Node(op_type=OpType.RELU, name="relu")
        relu.inputs = [x.id]
        graph.add_node(relu)
        graph.add_edge(x.id, relu.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [relu.id]
        graph.add_node(out)
        graph.add_edge(relu.id, out.id)

        compiled = self.backend.compile(graph)

        x_data = np.array([-1, 0, 1, 2])
        result = compiled(x_data)
        expected = np.array([0, 0, 1, 2])
        np.testing.assert_array_equal(result, expected)

    def test_compile_chain(self):
        """Test compiling a chain of operations."""
        graph = Graph(name="chain_graph")

        # x -> relu -> sigmoid -> output
        x = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(x)

        relu = Node(op_type=OpType.RELU, name="relu")
        relu.inputs = [x.id]
        graph.add_node(relu)
        graph.add_edge(x.id, relu.id)

        sigmoid = Node(op_type=OpType.SIGMOID, name="sigmoid")
        sigmoid.inputs = [relu.id]
        graph.add_node(sigmoid)
        graph.add_edge(relu.id, sigmoid.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [sigmoid.id]
        graph.add_node(out)
        graph.add_edge(sigmoid.id, out.id)

        compiled = self.backend.compile(graph)

        x_data = np.array([-1, 0, 1, 2], dtype=np.float32)
        result = compiled(x_data)

        # relu then sigmoid
        expected = 1 / (1 + np.exp(-np.maximum(x_data, 0)))
        np.testing.assert_array_almost_equal(result, expected)

    def test_compile_with_constant(self):
        """Test compiling graph with constant."""
        graph = Graph(name="const_graph")

        x = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(x)

        const = Node(op_type=OpType.CONSTANT, name="const")
        const.attributes["value"] = np.array([2.0])
        graph.add_node(const)

        mul = Node(op_type=OpType.MUL, name="mul")
        mul.inputs = [x.id, const.id]
        graph.add_node(mul)
        graph.add_edge(x.id, mul.id)
        graph.add_edge(const.id, mul.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [mul.id]
        graph.add_node(out)
        graph.add_edge(mul.id, out.id)

        compiled = self.backend.compile(graph)

        x_data = np.array([1, 2, 3])
        result = compiled(x_data)
        expected = np.array([2, 4, 6])
        np.testing.assert_array_equal(result, expected)


class TestBackendRegistry(unittest.TestCase):
    """Tests for BackendRegistry."""

    def test_get_eager_backend(self):
        """Test getting eager backend."""
        backend = BackendRegistry.get("eager_numpy")
        self.assertIsNotNone(backend)
        self.assertIsInstance(backend, EagerBackend)

    def test_list_backends(self):
        """Test listing backends."""
        backends = BackendRegistry.list_backends()
        self.assertIn("eager_numpy", backends)

    def test_list_available(self):
        """Test listing available backends."""
        available = BackendRegistry.list_available()
        self.assertIn("eager_numpy", available)

    def test_get_default(self):
        """Test getting default backend."""
        default = BackendRegistry.get_default()
        self.assertIsNotNone(default)


class TestCompilationCache(unittest.TestCase):
    """Tests for CompilationCache."""

    def setUp(self):
        self.cache = CompilationCache(max_entries=10)

    def test_cache_miss(self):
        """Test cache miss."""
        def dummy_fn():
            pass

        result = self.cache.lookup(dummy_fn, np.array([1, 2, 3]))
        self.assertIsNone(result)
        self.assertEqual(self.cache.misses, 1)

    def test_cache_hit(self):
        """Test cache hit."""
        def dummy_fn():
            pass

        # Create a compiled function
        graph = Graph()
        compiled = CompiledFunction(
            execute_fn=lambda x: x * 2,
            graph=graph,
            backend_name="test",
            input_names=["x"],
            output_names=["y"],
        )

        # Insert
        guards = [ShapeGuard(0, (3,))]
        self.cache.insert(dummy_fn, compiled, graph, guards, 10.0)

        # Lookup
        result = self.cache.lookup(dummy_fn, np.array([1, 2, 3]))
        self.assertIsNotNone(result)
        self.assertEqual(self.cache.hits, 1)

    def test_cache_guard_failure(self):
        """Test cache miss due to guard failure."""
        def dummy_fn():
            pass

        graph = Graph()
        compiled = CompiledFunction(
            execute_fn=lambda x: x * 2,
            graph=graph,
            backend_name="test",
            input_names=["x"],
            output_names=["y"],
        )

        # Insert with shape guard (3,)
        guards = [ShapeGuard(0, (3,))]
        self.cache.insert(dummy_fn, compiled, graph, guards, 10.0)

        # Lookup with different shape - should miss
        result = self.cache.lookup(dummy_fn, np.array([1, 2, 3, 4]))
        self.assertIsNone(result)

    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        self.assertIn('total_entries', stats)
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        self.assertIn('hit_rate', stats)

    def test_cache_eviction(self):
        """Test cache eviction when full."""
        small_cache = CompilationCache(max_entries=2)

        graph = Graph()

        for i in range(5):
            def fn():
                pass
            fn.__name__ = f"fn_{i}"
            fn.__code__ = builtins.compile(f"x = {i}", f"<fn_{i}>", "exec")

            compiled = CompiledFunction(
                execute_fn=lambda x: x,
                graph=graph,
                backend_name="test",
                input_names=["x"],
                output_names=["y"],
            )
            small_cache.insert(fn, compiled, graph, [], 1.0)

        # Should have evicted to stay at max
        self.assertLessEqual(small_cache.total_entries, 2)


class TestGuards(unittest.TestCase):
    """Tests for guards."""

    def test_shape_guard_pass(self):
        """Test shape guard passing."""
        guard = ShapeGuard(0, (10, 20))
        self.assertTrue(guard.check(np.zeros((10, 20))))

    def test_shape_guard_fail(self):
        """Test shape guard failing."""
        guard = ShapeGuard(0, (10, 20))
        self.assertFalse(guard.check(np.zeros((10, 30))))

    def test_dtype_guard_pass(self):
        """Test dtype guard passing."""
        guard = DtypeGuard(0, "float32")
        self.assertTrue(guard.check(np.zeros((10,), dtype=np.float32)))

    def test_dtype_guard_fail(self):
        """Test dtype guard failing."""
        guard = DtypeGuard(0, "float32")
        self.assertFalse(guard.check(np.zeros((10,), dtype=np.float64)))


class TestDynamicCompiler(unittest.TestCase):
    """Tests for DynamicCompiler."""

    def test_compiler_creation(self):
        """Test creating a compiler."""
        compiler = DynamicCompiler(backend="eager_numpy")
        self.assertIsNotNone(compiler)

    def test_compile_decorator(self):
        """Test using compiler as decorator."""
        compiler = DynamicCompiler(backend="eager_numpy", fallback_to_eager=True)

        @compiler
        def add_fn(x, y):
            return x + y

        self.assertTrue(hasattr(add_fn, '_compiled'))
        self.assertTrue(add_fn._compiled)

    def test_get_stats(self):
        """Test getting compiler stats."""
        compiler = DynamicCompiler(backend="eager_numpy")
        stats = compiler.get_stats()

        self.assertIn('cache', stats)
        self.assertIn('optimizer', stats)
        self.assertIn('backend', stats)


class TestCompileFunction(unittest.TestCase):
    """Tests for compile() convenience function."""

    def test_compile_with_backend(self):
        """Test compile with specified backend."""
        @compile(backend="eager_numpy")
        def my_fn(x):
            return x

        self.assertTrue(hasattr(my_fn, '_compiled'))

    def test_compile_with_optimization_level(self):
        """Test compile with optimization level."""
        @compile(optimization_level=2)
        def my_fn(x):
            return x

        self.assertTrue(hasattr(my_fn, '_compiled'))


if __name__ == "__main__":
    unittest.main()
