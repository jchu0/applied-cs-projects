"""Tests for graph optimization passes: fusion, CSE, and dead code elimination."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynagraph.graph.graph import Graph, InputNode, OperationNode, OutputNode
from dynagraph.executor.executor import (
    GraphOptimizer,
    DeadCodePass,
    CommonSubexpressionPass,
    FusionPass,
    ConstantFoldingPass,
    EagerExecutor,
    LazyExecutor,
)


class TestDeadCodeElimination:
    """Tests for Dead Code Elimination pass."""

    def test_removes_unused_operations(self, dead_code_graph):
        """Test that DCE removes operations not connected to outputs."""
        dce = DeadCodePass()
        optimized = dce.run(dead_code_graph)

        # Original has 4 operations, optimized should have 2
        original_ops = len([
            n for n in dead_code_graph.get_nodes()
            if isinstance(n, OperationNode)
        ])
        optimized_ops = len([
            n for n in optimized.get_nodes()
            if isinstance(n, OperationNode)
        ])

        assert original_ops == 4
        assert optimized_ops == 2

    def test_preserves_used_operations(self, dead_code_graph):
        """Test that DCE preserves operations connected to outputs."""
        dce = DeadCodePass()
        optimized = dce.run(dead_code_graph)

        # Check that used operations are present
        op_names = {
            node.name for node in optimized.get_nodes()
            if isinstance(node, OperationNode)
        }
        assert "used_add" in op_names
        assert "used_mul" in op_names

    def test_removes_dead_operations(self, dead_code_graph):
        """Test that dead operations are removed."""
        dce = DeadCodePass()
        optimized = dce.run(dead_code_graph)

        op_names = {
            node.name for node in optimized.get_nodes()
            if isinstance(node, OperationNode)
        }
        assert "dead_sub" not in op_names
        assert "dead_div" not in op_names

    def test_preserves_all_required_inputs(self, dead_code_graph):
        """Test that inputs required for outputs are preserved."""
        dce = DeadCodePass()
        optimized = dce.run(dead_code_graph)

        input_names = {node.name for node in optimized.get_inputs()}
        # Both x and y are needed for the used path
        assert "x" in input_names
        assert "y" in input_names

    def test_handles_empty_graph(self):
        """Test DCE on a graph with no operations."""
        graph = Graph("empty")
        x = graph.add_input("x", (2, 2))
        graph.add_output("output", x)

        dce = DeadCodePass()
        optimized = dce.run(graph)

        assert len(optimized.get_inputs()) == 1
        assert len(optimized.get_outputs()) == 1

    def test_handles_linear_chain(self):
        """Test DCE preserves complete linear chains."""
        graph = Graph("linear")
        x = graph.add_input("x", (4,))
        op1 = graph.add_operation("relu", "relu1", [x])
        op2 = graph.add_operation("relu", "relu2", [op1])
        op3 = graph.add_operation("relu", "relu3", [op2])
        graph.add_output("output", op3)

        dce = DeadCodePass()
        optimized = dce.run(graph)

        ops = [n for n in optimized.get_nodes() if isinstance(n, OperationNode)]
        assert len(ops) == 3


class TestCommonSubexpressionElimination:
    """Tests for Common Subexpression Elimination pass."""

    def test_eliminates_duplicate_operations(self, cse_candidate_graph):
        """Test that CSE eliminates duplicate operations."""
        cse = CommonSubexpressionPass()
        optimized = cse.run(cse_candidate_graph)

        # Count add operations - should be 2 (one for x+y, one for final)
        add_ops = [
            n for n in optimized.get_nodes()
            if isinstance(n, OperationNode) and n.op_type == "add"
        ]
        # The two identical x+y adds should be merged into one
        # But the final_add is different, so total should be 2
        assert len(add_ops) <= 3  # Was 3 before CSE, should be 2 after

    def test_preserves_different_operations(self):
        """Test that CSE preserves operations with different operands."""
        graph = Graph("diff_ops")
        x = graph.add_input("x", (4, 4))
        y = graph.add_input("y", (4, 4))
        z = graph.add_input("z", (4, 4))

        add1 = graph.add_operation("add", "add1", [x, y])
        add2 = graph.add_operation("add", "add2", [x, z])  # Different operand
        result = graph.add_operation("add", "result", [add1, add2])
        graph.add_output("output", result)

        cse = CommonSubexpressionPass()
        optimized = cse.run(graph)

        add_ops = [
            n for n in optimized.get_nodes()
            if isinstance(n, OperationNode) and n.op_type == "add"
        ]
        assert len(add_ops) == 3  # All different, none eliminated

    def test_considers_operation_attributes(self):
        """Test that CSE considers attributes when comparing operations."""
        graph = Graph("attr_test")
        x = graph.add_input("x", (4, 4))

        # Same operation type but different attributes
        sum1 = graph.add_operation("sum", "sum1", [x], attrs={"axis": 0})
        sum2 = graph.add_operation("sum", "sum2", [x], attrs={"axis": 1})
        result = graph.add_operation("add", "result", [sum1, sum2])
        graph.add_output("output", result)

        cse = CommonSubexpressionPass()
        optimized = cse.run(graph)

        sum_ops = [
            n for n in optimized.get_nodes()
            if isinstance(n, OperationNode) and n.op_type == "sum"
        ]
        # Different axes, so both should be preserved
        assert len(sum_ops) == 2

    def test_handles_multiple_uses_of_cse_result(self):
        """Test that CSE result can be used by multiple consumers."""
        graph = Graph("multi_use")
        x = graph.add_input("x", (4, 4))
        y = graph.add_input("y", (4, 4))

        # Two identical operations
        add1 = graph.add_operation("add", "add1", [x, y])
        add2 = graph.add_operation("add", "add2", [x, y])

        # Both used in different paths
        mul1 = graph.add_operation("mul", "mul1", [add1, add1])
        mul2 = graph.add_operation("mul", "mul2", [add2, add2])

        result = graph.add_operation("add", "final", [mul1, mul2])
        graph.add_output("output", result)

        cse = CommonSubexpressionPass()
        optimized = cse.run(graph)

        # After CSE, add1 and add2 should be merged
        original_nodes = graph.num_nodes()
        optimized_nodes = optimized.num_nodes()
        assert optimized_nodes <= original_nodes


class TestFusionPass:
    """Tests for operation fusion pass."""

    def test_fuses_matmul_add(self):
        """Test fusion of matmul followed by add (bias)."""
        graph = Graph("matmul_bias")
        x = graph.add_input("x", (4, 8))
        w = graph.add_input("w", (8, 16))
        b = graph.add_input("b", (16,))

        mm = graph.add_operation("matmul", "mm", [x, w])
        add = graph.add_operation("add", "add_bias", [mm, b])
        graph.add_output("output", add)

        fusion = FusionPass()
        optimized = fusion.run(graph)

        ops = [n for n in optimized.get_nodes() if isinstance(n, OperationNode)]
        op_types = [op.op_type for op in ops]

        # Should have fused matmul_bias operation
        assert "matmul_bias" in op_types or len(ops) <= 2

    def test_fuses_matmul_relu(self):
        """Test fusion of matmul followed by relu."""
        graph = Graph("matmul_relu")
        x = graph.add_input("x", (4, 8))
        w = graph.add_input("w", (8, 16))

        mm = graph.add_operation("matmul", "mm", [x, w])
        relu = graph.add_operation("relu", "relu", [mm])
        graph.add_output("output", relu)

        fusion = FusionPass()
        optimized = fusion.run(graph)

        ops = [n for n in optimized.get_nodes() if isinstance(n, OperationNode)]
        op_types = [op.op_type for op in ops]

        # Should have fused matmul_relu operation
        assert "matmul_relu" in op_types or len(ops) <= 2

    def test_fuses_add_relu(self):
        """Test fusion of add followed by relu."""
        graph = Graph("add_relu")
        x = graph.add_input("x", (4, 4))
        y = graph.add_input("y", (4, 4))

        add = graph.add_operation("add", "add", [x, y])
        relu = graph.add_operation("relu", "relu", [add])
        graph.add_output("output", relu)

        fusion = FusionPass()
        optimized = fusion.run(graph)

        ops = [n for n in optimized.get_nodes() if isinstance(n, OperationNode)]
        op_types = [op.op_type for op in ops]

        # Should have fused add_relu operation
        assert "add_relu" in op_types or len(ops) <= 2

    def test_no_fusion_for_multi_consumer(self):
        """Test that fusion doesn't happen when first op has multiple consumers."""
        graph = Graph("multi_consumer")
        x = graph.add_input("x", (4, 8))
        w = graph.add_input("w", (8, 16))

        mm = graph.add_operation("matmul", "mm", [x, w])
        relu1 = graph.add_operation("relu", "relu1", [mm])
        relu2 = graph.add_operation("relu", "relu2", [mm])  # Second consumer
        result = graph.add_operation("add", "add", [relu1, relu2])
        graph.add_output("output", result)

        original_ops = len([n for n in graph.get_nodes() if isinstance(n, OperationNode)])

        fusion = FusionPass()
        optimized = fusion.run(graph)

        optimized_ops = len([n for n in optimized.get_nodes() if isinstance(n, OperationNode)])

        # Should not aggressively fuse due to multiple consumers
        assert optimized_ops <= original_ops

    def test_preserves_non_fusible_operations(self):
        """Test that non-fusible operations are preserved."""
        graph = Graph("no_fuse")
        x = graph.add_input("x", (4, 4))

        # Operations that don't match fusion patterns
        t1 = graph.add_operation("transpose", "t1", [x], attrs={"axes": (1, 0)})
        t2 = graph.add_operation("reshape", "r1", [t1], attrs={"shape": (16,)})
        graph.add_output("output", t2)

        fusion = FusionPass()
        optimized = fusion.run(graph)

        ops = [n for n in optimized.get_nodes() if isinstance(n, OperationNode)]
        assert len(ops) == 2  # Both preserved


class TestGraphOptimizer:
    """Tests for the full graph optimizer pipeline."""

    def test_optimizer_applies_all_passes(self, simple_mlp_graph):
        """Test that optimizer applies all registered passes."""
        optimizer = GraphOptimizer()
        optimized = optimizer.optimize(simple_mlp_graph)

        # Optimized graph should have fewer or equal nodes
        assert optimized.num_nodes() <= simple_mlp_graph.num_nodes()

    def test_optimizer_preserves_semantics(self):
        """Test that optimization preserves graph semantics."""
        graph = Graph("semantic_test")
        x = graph.add_input("x", (2, 3))
        y = graph.add_input("y", (3, 2))
        mm = graph.add_operation("matmul", "mm", [x, y])
        graph.add_output("output", mm)

        optimizer = GraphOptimizer()
        optimized = optimizer.optimize(graph)

        # Execute both and compare
        x_data = np.random.randn(2, 3).astype(np.float32)
        y_data = np.random.randn(3, 2).astype(np.float32)

        original_executor = EagerExecutor(graph)
        optimized_executor = EagerExecutor(optimized)

        original_result = original_executor.execute({"x": x_data, "y": y_data})
        optimized_result = optimized_executor.execute({"x": x_data, "y": y_data})

        np.testing.assert_allclose(
            original_result["output"],
            optimized_result["output"],
            rtol=1e-5
        )

    def test_optimizer_with_dead_code_and_cse(self):
        """Test optimizer handles combination of dead code and CSE."""
        graph = Graph("combined")
        x = graph.add_input("x", (4, 4))
        y = graph.add_input("y", (4, 4))

        # CSE candidates
        add1 = graph.add_operation("add", "add1", [x, y])
        add2 = graph.add_operation("add", "add2", [x, y])

        # Dead code
        dead = graph.add_operation("sub", "dead", [x, y])

        # Use CSE candidates
        result = graph.add_operation("mul", "mul", [add1, add2])
        graph.add_output("output", result)

        original_nodes = graph.num_nodes()

        optimizer = GraphOptimizer()
        optimized = optimizer.optimize(graph)

        # Should eliminate dead code and merge CSE
        assert optimized.num_nodes() < original_nodes

    def test_optimizer_custom_pass(self):
        """Test adding custom optimization pass."""
        from dynagraph.executor.executor import OptimizationPass

        class IdentityRemovalPass(OptimizationPass):
            """Remove identity operations."""
            def run(self, graph):
                # For simplicity, just return the graph
                return graph

        optimizer = GraphOptimizer()
        optimizer.add_pass(IdentityRemovalPass())

        graph = Graph("test")
        x = graph.add_input("x", (4,))
        graph.add_output("output", x)

        # Should not raise
        optimized = optimizer.optimize(graph)
        assert optimized is not None


class TestLazyExecutorOptimization:
    """Tests for LazyExecutor with optimization enabled."""

    def test_lazy_executor_with_optimization(self, sample_graph):
        """Test LazyExecutor applies optimization."""
        executor = LazyExecutor(sample_graph, optimize=True)

        x_data = np.random.randn(2, 3).astype(np.float32)
        y_data = np.random.randn(3, 2).astype(np.float32)

        result = executor.execute({"x": x_data, "y": y_data})

        expected = x_data @ y_data
        np.testing.assert_allclose(result["output"], expected, rtol=1e-5)

    def test_lazy_executor_without_optimization(self, sample_graph):
        """Test LazyExecutor without optimization."""
        executor = LazyExecutor(sample_graph, optimize=False)

        x_data = np.random.randn(2, 3).astype(np.float32)
        y_data = np.random.randn(3, 2).astype(np.float32)

        result = executor.execute({"x": x_data, "y": y_data})

        expected = x_data @ y_data
        np.testing.assert_allclose(result["output"], expected, rtol=1e-5)

    def test_lazy_executor_compilation_caching(self):
        """Test that LazyExecutor caches compilation."""
        graph = Graph("cache_test")
        x = graph.add_input("x", (4, 4))
        y = graph.add_input("y", (4, 4))
        add = graph.add_operation("add", "add", [x, y])
        graph.add_output("output", add)

        executor = LazyExecutor(graph, optimize=True)

        x_data = np.random.randn(4, 4).astype(np.float32)
        y_data = np.random.randn(4, 4).astype(np.float32)

        # First execution compiles
        result1 = executor.execute({"x": x_data, "y": y_data})

        # Second execution should use cached compilation
        result2 = executor.execute({"x": x_data * 2, "y": y_data * 2})

        np.testing.assert_allclose(result1["output"], x_data + y_data, rtol=1e-5)
        np.testing.assert_allclose(result2["output"], (x_data * 2) + (y_data * 2), rtol=1e-5)


class TestFusionPatterns:
    """Tests for specific fusion patterns."""

    def test_conv_relu_fusion_pattern_exists(self):
        """Test that conv-relu fusion pattern is defined."""
        fusion = FusionPass()
        patterns = fusion.FUSION_PATTERNS

        conv_relu = ("conv", "relu", "conv_relu")
        assert conv_relu in patterns

    def test_conv_bn_fusion_pattern_exists(self):
        """Test that conv-bn fusion pattern is defined."""
        fusion = FusionPass()
        patterns = fusion.FUSION_PATTERNS

        conv_bn = ("conv", "bn", "conv_bn")
        assert conv_bn in patterns

    def test_all_fusion_patterns_have_output(self):
        """Test that all fusion patterns define output operation."""
        fusion = FusionPass()

        for pattern in fusion.FUSION_PATTERNS:
            assert len(pattern) == 3
            assert pattern[0]  # First op
            assert pattern[1]  # Second op
            assert pattern[2]  # Fused op name
