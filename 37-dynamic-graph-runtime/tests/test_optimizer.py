"""Tests for graph optimization passes."""

import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynamicgraph.core.graph import Graph, Node, Edge, OpType, NodeMetadata
from dynamicgraph.optimizer.graph_optimizer import GraphOptimizer, optimize_graph
from dynamicgraph.optimizer.passes import (
    ConstantFolding,
    DeadCodeElimination,
    CommonSubexpressionElimination,
    AlgebraicSimplification,
    OperatorFusion,
    ShapeInference,
)
from dynamicgraph.codegen.backend import EagerBackend


class TestConstantFolding(unittest.TestCase):
    """Tests for constant folding pass."""

    def test_fold_add_constants(self):
        """Test folding addition of constants."""
        graph = Graph()

        # Create: c1 + c2 where c1=2, c2=3
        c1 = Node(op_type=OpType.CONSTANT, name="c1")
        c1.attributes["value"] = np.array(2.0)
        graph.add_node(c1)

        c2 = Node(op_type=OpType.CONSTANT, name="c2")
        c2.attributes["value"] = np.array(3.0)
        graph.add_node(c2)

        add_node = Node(op_type=OpType.ADD, name="add")
        add_node.inputs = [c1.id, c2.id]
        graph.add_node(add_node)
        graph.add_edge(c1.id, add_node.id)
        graph.add_edge(c2.id, add_node.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [add_node.id]
        graph.add_node(out)
        graph.add_edge(add_node.id, out.id)

        # Run constant folding
        pass_ = ConstantFolding()
        result = pass_.run(graph)

        self.assertTrue(result.changed)
        # The add node should now be a constant
        self.assertEqual(graph.nodes[add_node.id].op_type, OpType.CONSTANT)
        self.assertAlmostEqual(graph.nodes[add_node.id].attributes["value"], 5.0)

    def test_fold_mul_constants(self):
        """Test folding multiplication of constants."""
        graph = Graph()

        c1 = Node(op_type=OpType.CONSTANT, name="c1")
        c1.attributes["value"] = np.array(4.0)
        graph.add_node(c1)

        c2 = Node(op_type=OpType.CONSTANT, name="c2")
        c2.attributes["value"] = np.array(5.0)
        graph.add_node(c2)

        mul_node = Node(op_type=OpType.MUL, name="mul")
        mul_node.inputs = [c1.id, c2.id]
        graph.add_node(mul_node)
        graph.add_edge(c1.id, mul_node.id)
        graph.add_edge(c2.id, mul_node.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [mul_node.id]
        graph.add_node(out)
        graph.add_edge(mul_node.id, out.id)

        pass_ = ConstantFolding()
        result = pass_.run(graph)

        self.assertTrue(result.changed)
        self.assertEqual(graph.nodes[mul_node.id].op_type, OpType.CONSTANT)
        self.assertAlmostEqual(graph.nodes[mul_node.id].attributes["value"], 20.0)


class TestDeadCodeElimination(unittest.TestCase):
    """Tests for dead code elimination pass."""

    def test_remove_unused_node(self):
        """Test removal of unused computation."""
        graph = Graph()

        # Create input
        inp = Node(op_type=OpType.INPUT, name="input")
        graph.add_node(inp)

        # Create a used computation
        used = Node(op_type=OpType.RELU, name="used")
        used.inputs = [inp.id]
        graph.add_node(used)
        graph.add_edge(inp.id, used.id)

        # Create an unused computation
        unused = Node(op_type=OpType.SIGMOID, name="unused")
        unused.inputs = [inp.id]
        graph.add_node(unused)
        graph.add_edge(inp.id, unused.id)

        # Output only uses the relu
        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [used.id]
        graph.add_node(out)
        graph.add_edge(used.id, out.id)

        self.assertEqual(len(graph.nodes), 4)

        pass_ = DeadCodeElimination()
        result = pass_.run(graph)

        self.assertTrue(result.changed)
        self.assertEqual(result.nodes_removed, 1)
        self.assertNotIn(unused.id, graph.nodes)


class TestCommonSubexpressionElimination(unittest.TestCase):
    """Tests for CSE pass."""

    def test_eliminate_duplicate_add(self):
        """Test elimination of duplicate additions."""
        graph = Graph()

        # Create: x + y (computed twice)
        x = Node(op_type=OpType.INPUT, name="x")
        y = Node(op_type=OpType.INPUT, name="y")
        graph.add_node(x)
        graph.add_node(y)

        add1 = Node(op_type=OpType.ADD, name="add1")
        add1.inputs = [x.id, y.id]
        graph.add_node(add1)
        graph.add_edge(x.id, add1.id)
        graph.add_edge(y.id, add1.id)

        add2 = Node(op_type=OpType.ADD, name="add2")
        add2.inputs = [x.id, y.id]  # Same inputs
        graph.add_node(add2)
        graph.add_edge(x.id, add2.id)
        graph.add_edge(y.id, add2.id)

        # Output uses both
        out = Node(op_type=OpType.MUL, name="mul")
        out.inputs = [add1.id, add2.id]
        graph.add_node(out)
        graph.add_edge(add1.id, out.id)
        graph.add_edge(add2.id, out.id)

        out_node = Node(op_type=OpType.OUTPUT, name="output")
        out_node.inputs = [out.id]
        graph.add_node(out_node)
        graph.add_edge(out.id, out_node.id)

        pass_ = CommonSubexpressionElimination()
        result = pass_.run(graph)

        self.assertTrue(result.changed)
        self.assertEqual(result.nodes_removed, 1)


class TestAlgebraicSimplification(unittest.TestCase):
    """Tests for algebraic simplification pass."""

    def test_add_zero(self):
        """Test x + 0 = x simplification."""
        graph = Graph()

        x = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(x)

        zero = Node(op_type=OpType.CONSTANT, name="zero")
        zero.attributes["value"] = 0.0
        graph.add_node(zero)

        add = Node(op_type=OpType.ADD, name="add")
        add.inputs = [x.id, zero.id]
        graph.add_node(add)
        graph.add_edge(x.id, add.id)
        graph.add_edge(zero.id, add.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [add.id]
        graph.add_node(out)
        graph.add_edge(add.id, out.id)

        pass_ = AlgebraicSimplification()
        result = pass_.run(graph)

        self.assertTrue(result.changed)

    def test_mul_one(self):
        """Test x * 1 = x simplification."""
        graph = Graph()

        x = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(x)

        one = Node(op_type=OpType.CONSTANT, name="one")
        one.attributes["value"] = 1.0
        graph.add_node(one)

        mul = Node(op_type=OpType.MUL, name="mul")
        mul.inputs = [x.id, one.id]
        graph.add_node(mul)
        graph.add_edge(x.id, mul.id)
        graph.add_edge(one.id, mul.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [mul.id]
        graph.add_node(out)
        graph.add_edge(mul.id, out.id)

        pass_ = AlgebraicSimplification()
        result = pass_.run(graph)

        self.assertTrue(result.changed)

    def test_mul_zero(self):
        """Test x * 0 = 0 simplification."""
        graph = Graph()

        x = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(x)

        zero = Node(op_type=OpType.CONSTANT, name="zero")
        zero.attributes["value"] = 0.0
        graph.add_node(zero)

        mul = Node(op_type=OpType.MUL, name="mul")
        mul.inputs = [x.id, zero.id]
        graph.add_node(mul)
        graph.add_edge(x.id, mul.id)
        graph.add_edge(zero.id, mul.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [mul.id]
        graph.add_node(out)
        graph.add_edge(mul.id, out.id)

        pass_ = AlgebraicSimplification()
        result = pass_.run(graph)

        self.assertTrue(result.changed)
        self.assertEqual(graph.nodes[mul.id].op_type, OpType.CONSTANT)
        self.assertEqual(graph.nodes[mul.id].attributes["value"], 0.0)


class TestOperatorFusion(unittest.TestCase):
    """Tests for the operator fusion pass.

    Each pattern must leave the graph valid (acyclic, consistent adjacency) and
    numerically equivalent to the unfused graph.
    """

    def _pointwise_chain_graph(self):
        """Build relu((a + b) * c) as an add -> mul -> relu chain."""
        g = Graph(name="pw")
        a = g.add_node(Node(op_type=OpType.INPUT, name="a"))
        b = g.add_node(Node(op_type=OpType.INPUT, name="b"))
        c = g.add_node(Node(op_type=OpType.INPUT, name="c"))
        add = g.add_node(Node(op_type=OpType.ADD, name="add"))
        mul = g.add_node(Node(op_type=OpType.MUL, name="mul"))
        relu = g.add_node(Node(op_type=OpType.RELU, name="relu"))
        out = g.add_node(Node(op_type=OpType.OUTPUT, name="out"))
        g.add_edge(a, add, 0)
        g.add_edge(b, add, 1)
        g.add_edge(add, mul, 0)
        g.add_edge(c, mul, 1)
        g.add_edge(mul, relu, 0)
        g.add_edge(relu, out)
        return g

    def test_pointwise_chain_fuses(self):
        """A pointwise pair is fused into a fused_pointwise CUSTOM node."""
        g = self._pointwise_chain_graph()
        result = OperatorFusion().run(g)

        self.assertTrue(result.changed)
        custom = [n for n in g.nodes.values() if n.op_type == OpType.CUSTOM]
        self.assertEqual(len(custom), 1)
        self.assertEqual(custom[0].name, "fused_pointwise")
        self.assertEqual(custom[0].attributes["op_chain"], ["add", "mul"])

    def test_pointwise_chain_graph_stays_valid(self):
        """Fusion must not introduce cycles or dangling adjacency."""
        g = self._pointwise_chain_graph()
        OperatorFusion().run(g)
        self.assertFalse(g.has_cycle())
        self.assertEqual(g.validate(), [])

    def test_pointwise_chain_execution_matches(self):
        """The fused graph computes the same result as the unfused one."""
        backend = EagerBackend()
        a = np.array([1.0, -5.0, 2.0])
        b = np.array([2.0, 1.0, 3.0])
        c = np.array([1.0, 1.0, -1.0])

        reference = backend.compile(self._pointwise_chain_graph())(a, b, c)

        g = self._pointwise_chain_graph()
        OperatorFusion().run(g)
        fused_result = backend.compile(g)(a, b, c)

        np.testing.assert_allclose(fused_result, reference)
        np.testing.assert_allclose(fused_result, np.maximum((a + b) * c, 0))

    def test_pointwise_skips_repeated_operand(self):
        """A chain that would reuse the same source is left unfused (still valid)."""
        # (x - y) / x reuses x, which cannot be represented as a single fused node.
        g = Graph(name="shared")
        x = g.add_node(Node(op_type=OpType.INPUT, name="x"))
        y = g.add_node(Node(op_type=OpType.INPUT, name="y"))
        sub = g.add_node(Node(op_type=OpType.SUB, name="sub"))
        div = g.add_node(Node(op_type=OpType.DIV, name="div"))
        out = g.add_node(Node(op_type=OpType.OUTPUT, name="out"))
        g.add_edge(x, sub, 0)
        g.add_edge(y, sub, 1)
        g.add_edge(sub, div, 0)
        g.add_edge(x, div, 1)
        g.add_edge(div, out)

        OperatorFusion().run(g)

        # No fused_pointwise node, graph still valid and executes correctly.
        self.assertFalse(any(n.name == "fused_pointwise" for n in g.nodes.values()))
        self.assertFalse(g.has_cycle())
        self.assertEqual(g.validate(), [])
        xv, yv = np.array([10.0, 20.0]), np.array([3.0, 4.0])
        np.testing.assert_allclose(EagerBackend().compile(g)(xv, yv), (xv - yv) / xv)

    def test_matmul_bias_fuses_to_linear(self):
        """matmul + add fuses into a single LINEAR node and stays valid."""
        g = Graph(name="mm")
        x = g.add_node(Node(op_type=OpType.INPUT, name="x"))
        w = g.add_node(Node(op_type=OpType.INPUT, name="w"))
        bias = g.add_node(Node(op_type=OpType.INPUT, name="bias"))
        mm = g.add_node(Node(op_type=OpType.MATMUL, name="mm"))
        add = g.add_node(Node(op_type=OpType.ADD, name="add"))
        out = g.add_node(Node(op_type=OpType.OUTPUT, name="out"))
        g.add_edge(x, mm, 0)
        g.add_edge(w, mm, 1)
        g.add_edge(mm, add, 0)
        g.add_edge(bias, add, 1)
        g.add_edge(add, out)

        OperatorFusion().run(g)

        linear = [n for n in g.nodes.values() if n.op_type == OpType.LINEAR]
        self.assertEqual(len(linear), 1)
        self.assertFalse(g.has_cycle())
        self.assertEqual(g.validate(), [])

        # LINEAR computes x @ w.T + bias.
        xv = np.random.rand(2, 3)
        wv = np.random.rand(4, 3)
        bv = np.random.rand(4)
        result = EagerBackend().compile(g)(xv, wv, bv)
        np.testing.assert_allclose(result, xv @ wv.T + bv)

    def test_conv_bn_relu_fuses(self):
        """conv2d + batchnorm + relu fuses to one CUSTOM node, valid + equivalent."""
        def build():
            g = Graph(name="cbr")
            inp = g.add_node(Node(op_type=OpType.INPUT, name="inp"))
            conv = g.add_node(Node(
                op_type=OpType.CONV2D, name="conv",
                attributes={"weight": np.random.RandomState(0).rand(2, 1, 2, 2)},
            ))
            bn = g.add_node(Node(
                op_type=OpType.BATCHNORM, name="bn",
                attributes={"weight": np.ones(2), "bias": np.zeros(2)},
            ))
            relu = g.add_node(Node(op_type=OpType.RELU, name="relu"))
            out = g.add_node(Node(op_type=OpType.OUTPUT, name="out"))
            g.add_edge(inp, conv, 0)
            g.add_edge(conv, bn, 0)
            g.add_edge(bn, relu, 0)
            g.add_edge(relu, out)
            return g

        backend = EagerBackend()
        x = np.random.RandomState(1).rand(4, 1, 4, 4)
        reference = backend.compile(build())(x)

        g = build()
        OperatorFusion().run(g)
        custom = [n for n in g.nodes.values() if n.op_type == OpType.CUSTOM]
        self.assertEqual(len(custom), 1)
        self.assertEqual(custom[0].attributes["fused_from"],
                         ["conv2d", "batchnorm", "relu"])
        self.assertFalse(g.has_cycle())
        self.assertEqual(g.validate(), [])
        np.testing.assert_allclose(backend.compile(g)(x), reference)


class TestShapeInference(unittest.TestCase):
    """Tests for shape inference pass."""

    def test_infer_add_shape(self):
        """Test shape inference for addition."""
        graph = Graph()

        x = Node(op_type=OpType.INPUT, name="x")
        x.metadata.shape = (10, 20)
        graph.add_node(x)

        y = Node(op_type=OpType.INPUT, name="y")
        y.metadata.shape = (10, 20)
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

        pass_ = ShapeInference()
        result = pass_.run(graph)

        self.assertTrue(result.changed)
        self.assertEqual(graph.nodes[add.id].metadata.shape, (10, 20))

    def test_infer_broadcast_shape(self):
        """Test shape inference with broadcasting."""
        graph = Graph()

        x = Node(op_type=OpType.INPUT, name="x")
        x.metadata.shape = (10, 1)
        graph.add_node(x)

        y = Node(op_type=OpType.INPUT, name="y")
        y.metadata.shape = (1, 20)
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

        pass_ = ShapeInference()
        result = pass_.run(graph)

        self.assertTrue(result.changed)
        self.assertEqual(graph.nodes[add.id].metadata.shape, (10, 20))


class TestGraphOptimizer(unittest.TestCase):
    """Tests for the GraphOptimizer."""

    def test_optimizer_level_0(self):
        """Test optimization level 0 (no optimization)."""
        graph = Graph()
        inp = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(inp)

        optimizer = GraphOptimizer(optimization_level=0)
        optimized, stats = optimizer.optimize(graph)

        self.assertEqual(stats.total_passes, 0)

    def test_optimizer_level_1(self):
        """Test optimization level 1 (basic optimization)."""
        graph = Graph()

        # Create simple graph
        inp = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(inp)

        relu = Node(op_type=OpType.RELU, name="relu")
        relu.inputs = [inp.id]
        graph.add_node(relu)
        graph.add_edge(inp.id, relu.id)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [relu.id]
        graph.add_node(out)
        graph.add_edge(relu.id, out.id)

        optimizer = GraphOptimizer(optimization_level=1)
        optimized, stats = optimizer.optimize(graph)

        self.assertGreater(stats.total_passes, 0)

    def test_level2_preserves_execution_semantics(self):
        """A level-2 optimize (with fusion) must not change numeric results.

        This guards the input-ordering invariant: clone() preserves the declared
        input order and the eager backend binds positional args by that order,
        not by topological order.
        """
        backend = EagerBackend()

        def build():
            g = Graph(name="e2e")
            a = g.add_node(Node(op_type=OpType.INPUT, name="a"))
            b = g.add_node(Node(op_type=OpType.INPUT, name="b"))
            c = g.add_node(Node(op_type=OpType.INPUT, name="c"))
            add = g.add_node(Node(op_type=OpType.ADD, name="add"))
            mul = g.add_node(Node(op_type=OpType.MUL, name="mul"))
            relu = g.add_node(Node(op_type=OpType.RELU, name="relu"))
            out = g.add_node(Node(op_type=OpType.OUTPUT, name="out"))
            g.add_edge(a, add, 0)
            g.add_edge(b, add, 1)
            g.add_edge(add, mul, 0)
            g.add_edge(c, mul, 1)
            g.add_edge(mul, relu, 0)
            g.add_edge(relu, out)
            return g

        # Distinct magnitudes so any operand mix-up changes the result.
        a = np.array([100.0])
        b = np.array([10.0])
        c = np.array([1.0])
        reference = np.maximum((a + b) * c, 0)

        # Run repeatedly: node-table iteration order varies with node ids.
        for _ in range(25):
            optimized, _ = GraphOptimizer(optimization_level=2).optimize(build())
            self.assertFalse(optimized.has_cycle())
            self.assertEqual(optimized.validate(), [])
            result = backend.compile(optimized)(a, b, c)
            np.testing.assert_allclose(result, reference)

    def test_optimizer_convergence(self):
        """Test that optimizer converges."""
        graph = Graph()

        inp = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(inp)

        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [inp.id]
        graph.add_node(out)
        graph.add_edge(inp.id, out.id)

        optimizer = GraphOptimizer(optimization_level=2, max_iterations=100)
        optimized, stats = optimizer.optimize(graph)

        # Should converge quickly on this simple graph
        self.assertLess(stats.total_passes, 50)


class TestOptimizeGraph(unittest.TestCase):
    """Tests for optimize_graph convenience function."""

    def test_optimize_graph_default(self):
        """Test optimize_graph with default settings."""
        graph = Graph()
        inp = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(inp)
        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [inp.id]
        graph.add_node(out)
        graph.add_edge(inp.id, out.id)

        optimized = optimize_graph(graph)
        self.assertIsInstance(optimized, Graph)

    def test_optimize_graph_inference_mode(self):
        """Test optimize_graph in inference mode."""
        graph = Graph()
        inp = Node(op_type=OpType.INPUT, name="x")
        graph.add_node(inp)
        out = Node(op_type=OpType.OUTPUT, name="output")
        out.inputs = [inp.id]
        graph.add_node(out)
        graph.add_edge(inp.id, out.id)

        optimized = optimize_graph(graph, mode="inference")
        self.assertIsInstance(optimized, Graph)


if __name__ == "__main__":
    unittest.main()
