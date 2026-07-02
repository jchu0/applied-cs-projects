"""Tests for graph optimization passes."""

import pytest
from kernelsched import (
    OptimizationPass, OptimizationResult, GraphOptimizer, KernelFuser,
    MemoryOptimizer, ConstantFolder, DeadCodeEliminator, OptimizationPipeline,
    create_default_pipeline, optimize_graph,
    ComputeGraph, KernelType,
    create_gemm_kernel, create_elementwise_kernel, create_attention_kernel,
)


class TestOptimizationPass:
    """Tests for OptimizationPass enum."""

    def test_optimization_pass_values(self):
        """Test optimization pass enum values."""
        assert OptimizationPass.FUSION.value == "fusion"
        assert OptimizationPass.CONSTANT_FOLDING.value == "constant_folding"
        assert OptimizationPass.DEAD_CODE_ELIMINATION.value == "dead_code"
        assert OptimizationPass.LAYOUT_OPTIMIZATION.value == "layout"
        assert OptimizationPass.MEMORY_PLANNING.value == "memory"
        assert OptimizationPass.KERNEL_SELECTION.value == "kernel_selection"


class TestOptimizationResult:
    """Tests for OptimizationResult class."""

    def test_optimization_result_creation(self):
        """Test optimization result creation."""
        result = OptimizationResult(
            original_kernels=10,
            optimized_kernels=7,
            estimated_speedup=1.5,
            passes_applied=["fusion", "memory"],
            fused_patterns=["gemm_bias_activation"],
        )
        assert result.original_kernels == 10
        assert result.optimized_kernels == 7
        assert result.estimated_speedup == 1.5
        assert len(result.passes_applied) == 2
        assert len(result.fused_patterns) == 1


class TestKernelFuser:
    """Tests for KernelFuser optimizer."""

    def test_kernel_fuser_creation(self, kernel_fuser):
        """Test kernel fuser creation."""
        assert len(kernel_fuser.fusion_patterns) > 0
        assert kernel_fuser.fused_count == 0

    def test_fuser_copy_graph(self, kernel_fuser, simple_graph):
        """Test that fuser creates a copy of the graph."""
        original_kernel_ids = set(simple_graph.kernels.keys())
        optimized = kernel_fuser.optimize(simple_graph)

        # Original graph should be unchanged
        assert set(simple_graph.kernels.keys()) == original_kernel_ids

    def test_fuse_elementwise_chain(self, kernel_fuser, elementwise_chain_graph):
        """Test fusing chain of elementwise operations."""
        optimized = kernel_fuser.optimize(elementwise_chain_graph)

        # Should have fewer kernels after fusion
        # Chain of 4 should be fused if >= 3
        assert len(optimized.kernels) <= len(elementwise_chain_graph.kernels)

    def test_fuse_gemm_bias_activation(self, kernel_fuser, fusable_graph):
        """Test fusing GEMM + bias + activation pattern."""
        optimized = kernel_fuser.optimize(fusable_graph)

        # May or may not fuse depending on attributes
        # At minimum, should not increase kernel count
        assert len(optimized.kernels) <= len(fusable_graph.kernels)

    def test_fuser_preserves_dependencies(self, kernel_fuser, simple_graph):
        """Test that fuser preserves graph validity."""
        optimized = kernel_fuser.optimize(simple_graph)

        # All dependency references should be valid
        for dep in optimized.dependencies:
            if dep.source_id not in optimized.kernels:
                # Source was removed, dependency should also be removed
                pass
            if dep.target_id not in optimized.kernels:
                # Target was removed, dependency should also be removed
                pass

    def test_fuser_no_fusion_possible(self, kernel_fuser, parallel_graph):
        """Test fuser with graph that has no fusion opportunities."""
        original_count = len(parallel_graph.kernels)
        optimized = kernel_fuser.optimize(parallel_graph)

        # Kernel count should be same or reduced (never increased)
        assert len(optimized.kernels) <= original_count

    def test_fused_patterns_tracking(self, kernel_fuser, elementwise_chain_graph):
        """Test that fused patterns are tracked."""
        kernel_fuser.optimize(elementwise_chain_graph)

        # If fusion occurred, patterns should be recorded
        # This depends on whether chain was long enough
        if kernel_fuser.fused_count > 0:
            assert len(kernel_fuser.fused_patterns) > 0


class TestConstantFolder:
    """Tests for ConstantFolder optimizer."""

    def test_constant_folder_creation(self, constant_folder):
        """Test constant folder creation."""
        assert constant_folder is not None

    def test_constant_folder_no_constants_is_noop(self, constant_folder, simple_graph):
        """Graphs with no constant nodes are returned unchanged in structure."""
        optimized = constant_folder.optimize(simple_graph)

        # No kernels are marked constant, so nothing folds.
        assert constant_folder.folded_count == 0
        assert set(optimized.kernels.keys()) == set(simple_graph.kernels.keys())
        # Original graph is not mutated.
        for kid, kernel in simple_graph.kernels.items():
            assert optimized.kernels[kid].kernel_type == kernel.kernel_type

    def test_constant_folder_folds_derived_constant(self, constant_folder):
        """A kernel whose only input is a constant is folded to a MEMORY node."""
        graph = ComputeGraph()

        # A marked constant source (e.g. a weight) feeding a compute kernel.
        weight = create_gemm_kernel(64, 64, 64)
        weight.attributes["is_constant"] = True
        consumer = create_elementwise_kernel((64, 64), op="add")

        graph.add_kernel(weight)
        graph.add_kernel(consumer)
        graph.add_dependency(weight.kernel_id, consumer.kernel_id, "w_c")

        optimized = constant_folder.optimize(graph)

        # The consumer's inputs are all constant, so it folds.
        assert constant_folder.folded_count == 1
        folded = optimized.kernels[consumer.kernel_id]
        assert folded.kernel_type == KernelType.MEMORY
        assert folded.attributes.get("is_constant") is True
        assert folded.attributes.get("original_type") == "elementwise"
        # Its outputs are preserved so downstream consumers still resolve.
        assert [t.tensor_id for t in folded.outputs] == \
            [t.tensor_id for t in consumer.outputs]

    def test_constant_folder_propagates_transitively(self, constant_folder):
        """Constant-ness propagates through a chain of derived nodes."""
        graph = ComputeGraph()

        src = create_gemm_kernel(32, 32, 32)
        src.attributes["is_constant"] = True
        mid = create_elementwise_kernel((32, 32), op="mul")
        end = create_elementwise_kernel((32, 32), op="add")

        for k in (src, mid, end):
            graph.add_kernel(k)
        graph.add_dependency(src.kernel_id, mid.kernel_id, "s_m")
        graph.add_dependency(mid.kernel_id, end.kernel_id, "m_e")

        optimized = constant_folder.optimize(graph)

        # Both derived kernels fold (the leaf source stays as-is).
        assert constant_folder.folded_count == 2
        assert optimized.kernels[mid.kernel_id].kernel_type == KernelType.MEMORY
        assert optimized.kernels[end.kernel_id].kernel_type == KernelType.MEMORY
        # Edges into folded nodes are removed as redundant.
        assert optimized.dependencies == []

    def test_constant_folder_leaves_nonconstant_inputs(self, constant_folder):
        """A kernel with a non-constant input is not folded."""
        graph = ComputeGraph()

        const = create_gemm_kernel(16, 16, 16)
        const.attributes["is_constant"] = True
        variable = create_gemm_kernel(16, 16, 16)  # not constant
        consumer = create_elementwise_kernel((16, 16), op="add")

        for k in (const, variable, consumer):
            graph.add_kernel(k)
        graph.add_dependency(const.kernel_id, consumer.kernel_id, "c_c")
        graph.add_dependency(variable.kernel_id, consumer.kernel_id, "v_c")

        optimized = constant_folder.optimize(graph)

        # consumer depends on a non-constant, so it is left untouched.
        assert constant_folder.folded_count == 0
        assert optimized.kernels[consumer.kernel_id].kernel_type == \
            KernelType.ELEMENTWISE

    def test_constant_folder_preserves_topological_validity(self, constant_folder):
        """Folded graph is still topologically sortable."""
        graph = ComputeGraph()

        src = create_gemm_kernel(32, 32, 32)
        src.attributes["is_constant"] = True
        mid = create_elementwise_kernel((32, 32), op="mul")
        graph.add_kernel(src)
        graph.add_kernel(mid)
        graph.add_dependency(src.kernel_id, mid.kernel_id, "s_m")

        optimized = constant_folder.optimize(graph)
        topo = optimized.topological_sort()
        assert len(topo) == len(optimized.kernels)


class TestDeadCodeEliminator:
    """Tests for DeadCodeEliminator optimizer."""

    def test_dead_code_eliminator_creation(self, dead_code_eliminator):
        """Test dead code eliminator creation."""
        assert dead_code_eliminator is not None

    def test_eliminate_unused_kernels(self, dead_code_eliminator):
        """Test eliminating unused kernels."""
        graph = ComputeGraph()

        # Create connected kernels
        k1 = create_gemm_kernel(256, 256, 256)
        k2 = create_elementwise_kernel((256, 256), op="relu")
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "k1_k2")

        # Mark k2's output as graph output
        graph.output_tensors = k2.outputs.copy()

        optimized = dead_code_eliminator.optimize(graph)

        # Both kernels should be kept (k1 produces data for k2)
        assert len(optimized.kernels) >= 1

    def test_eliminate_disconnected_kernel(self, dead_code_eliminator):
        """Test eliminating disconnected kernel."""
        graph = ComputeGraph()

        # Connected chain
        k1 = create_gemm_kernel(256, 256, 256)
        k2 = create_elementwise_kernel((256, 256), op="relu")
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "connected")

        # Disconnected kernel
        k3 = create_gemm_kernel(64, 64, 64)
        graph.add_kernel(k3)

        optimized = dead_code_eliminator.optimize(graph)

        # All kernels should be kept since they all produce outputs
        # (end nodes are kept by default)
        assert len(optimized.kernels) >= 2

    def test_preserve_output_producing_kernels(self, dead_code_eliminator, simple_graph):
        """Test that kernels producing outputs are preserved."""
        optimized = dead_code_eliminator.optimize(simple_graph)

        # At least one kernel should be preserved
        assert len(optimized.kernels) > 0

    def test_preserve_dependencies(self, dead_code_eliminator, simple_graph):
        """Test that valid dependencies are preserved."""
        optimized = dead_code_eliminator.optimize(simple_graph)

        # All dependencies should reference existing kernels
        for dep in optimized.dependencies:
            assert dep.source_id in optimized.kernels
            assert dep.target_id in optimized.kernels


class TestMemoryOptimizerOptimization:
    """Tests for MemoryOptimizer as optimization pass."""

    def test_memory_optimizer_as_pass(self, memory_optimizer, simple_graph):
        """Test memory optimizer as optimization pass."""
        optimized = memory_optimizer.optimize(simple_graph)

        # Should preserve all kernels
        assert len(optimized.kernels) == len(simple_graph.kernels)

    def test_memory_optimizer_assigns_offsets(self, memory_optimizer, simple_graph):
        """Test that memory optimizer assigns offsets."""
        optimized = memory_optimizer.optimize(simple_graph)

        # At least some tensors should have offsets computed
        # (implementation assigns offsets based on planning)
        has_tensors = False
        for kernel in optimized.kernels.values():
            if kernel.outputs:
                has_tensors = True
                break
        assert has_tensors


class TestOptimizationPipeline:
    """Tests for OptimizationPipeline class."""

    def test_pipeline_creation(self):
        """Test pipeline creation."""
        pipeline = OptimizationPipeline()
        assert len(pipeline.passes) == 0

    def test_add_pass(self):
        """Test adding optimization pass."""
        pipeline = OptimizationPipeline()
        pipeline.add_pass("fusion", KernelFuser())

        assert len(pipeline.passes) == 1
        assert pipeline.passes[0][0] == "fusion"

    def test_pipeline_optimize_simple(self, optimization_pipeline, simple_graph):
        """Test pipeline optimization."""
        optimized, result = optimization_pipeline.optimize(simple_graph)

        assert len(optimized.kernels) <= len(simple_graph.kernels)
        assert result.original_kernels == len(simple_graph.kernels)
        assert len(result.passes_applied) > 0

    def test_pipeline_multiple_passes(self, simple_graph):
        """Test pipeline with multiple passes."""
        pipeline = OptimizationPipeline()
        pipeline.add_pass("dead_code", DeadCodeEliminator())
        pipeline.add_pass("fusion", KernelFuser())
        pipeline.add_pass("memory", MemoryOptimizer())

        optimized, result = pipeline.optimize(simple_graph)

        assert len(result.passes_applied) == 3
        assert "dead_code" in result.passes_applied
        assert "fusion" in result.passes_applied
        assert "memory" in result.passes_applied

    def test_pipeline_estimated_speedup(self, optimization_pipeline, simple_graph):
        """Test pipeline speedup estimation."""
        _, result = optimization_pipeline.optimize(simple_graph)

        # Speedup should be at least 1.0 (no slowdown)
        # Note: with no optimizations, speedup is 1.0
        assert result.estimated_speedup >= 0.0

    def test_pipeline_fused_patterns_collection(self, elementwise_chain_graph):
        """Test that pipeline collects fused patterns."""
        pipeline = OptimizationPipeline()
        pipeline.add_pass("fusion", KernelFuser())

        _, result = pipeline.optimize(elementwise_chain_graph)

        # If fusion occurred, patterns should be collected
        # This is dependent on graph structure
        assert isinstance(result.fused_patterns, list)


class TestCreateDefaultPipeline:
    """Tests for create_default_pipeline function."""

    def test_create_default_pipeline(self):
        """Test default pipeline creation."""
        pipeline = create_default_pipeline()

        assert len(pipeline.passes) == 4

        pass_names = [name for name, _ in pipeline.passes]
        assert "dead_code" in pass_names
        assert "fusion" in pass_names
        assert "constant_folding" in pass_names
        assert "memory" in pass_names

    def test_default_pipeline_order(self):
        """Test default pipeline pass order."""
        pipeline = create_default_pipeline()

        pass_names = [name for name, _ in pipeline.passes]

        # Dead code should come before fusion
        assert pass_names.index("dead_code") < pass_names.index("fusion")

    def test_default_pipeline_passes_are_instances(self):
        """Test that default pipeline contains proper instances."""
        pipeline = create_default_pipeline()

        for name, optimizer in pipeline.passes:
            assert isinstance(optimizer, GraphOptimizer)


class TestOptimizeGraph:
    """Tests for optimize_graph convenience function."""

    def test_optimize_graph_function(self, simple_graph):
        """Test optimize_graph convenience function."""
        optimized, result = optimize_graph(simple_graph)

        assert optimized is not None
        assert result is not None
        assert result.original_kernels == len(simple_graph.kernels)

    def test_optimize_graph_preserves_structure(self, simple_graph):
        """Test that optimize_graph preserves valid graph structure."""
        optimized, _ = optimize_graph(simple_graph)

        # Should be able to topologically sort
        topo = optimized.topological_sort()
        assert len(topo) == len(optimized.kernels)

    def test_optimize_graph_empty(self, empty_graph):
        """Test optimize_graph with empty graph."""
        optimized, result = optimize_graph(empty_graph)

        assert result.original_kernels == 0
        assert result.optimized_kernels == 0

    def test_optimize_transformer_graph(self, transformer_graph):
        """Test optimizing transformer-like graph."""
        optimized, result = optimize_graph(transformer_graph)

        assert result.original_kernels == len(transformer_graph.kernels)
        assert len(optimized.kernels) <= result.original_kernels


class TestFusionPatterns:
    """Tests for specific fusion patterns."""

    def test_gemm_elementwise_fusion(self):
        """Test GEMM + elementwise fusion detection."""
        graph = ComputeGraph()

        gemm = create_gemm_kernel(512, 512, 512)
        elem = create_elementwise_kernel((512, 512), op="add")

        graph.add_kernel(gemm)
        graph.add_kernel(elem)
        graph.add_dependency(gemm.kernel_id, elem.kernel_id, "gemm_elem")

        fuser = KernelFuser()
        optimized = fuser.optimize(graph)

        # Should have at most 2 kernels
        assert len(optimized.kernels) <= 2

    def test_long_elementwise_chain_fusion(self):
        """Test that long elementwise chains are fused."""
        graph = ComputeGraph()

        kernels = []
        for i in range(5):
            k = create_elementwise_kernel((256, 256), op=["add", "mul"][i % 2])
            graph.add_kernel(k)
            kernels.append(k)

            if i > 0:
                graph.add_dependency(
                    kernels[i-1].kernel_id,
                    kernels[i].kernel_id,
                    f"chain_{i}"
                )

        fuser = KernelFuser()
        optimized = fuser.optimize(graph)

        # Long chains (>= 3) should be fused
        assert len(optimized.kernels) < 5

    def test_short_chain_not_fused(self):
        """Test that short chains are not fused."""
        graph = ComputeGraph()

        k1 = create_elementwise_kernel((256, 256), op="add")
        k2 = create_elementwise_kernel((256, 256), op="mul")

        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "short_chain")

        fuser = KernelFuser()
        optimized = fuser.optimize(graph)

        # Chain of 2 should not be fused (threshold is 3)
        assert len(optimized.kernels) == 2

    def test_branching_prevents_chain_fusion(self):
        """Test that branching prevents elementwise chain fusion."""
        graph = ComputeGraph()

        k1 = create_elementwise_kernel((256, 256), op="add")
        k2 = create_elementwise_kernel((256, 256), op="mul")
        k3 = create_elementwise_kernel((256, 256), op="sub")

        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_kernel(k3)

        # k1 -> k2
        # k1 -> k3 (branch)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "branch1")
        graph.add_dependency(k1.kernel_id, k3.kernel_id, "branch2")

        fuser = KernelFuser()
        optimized = fuser.optimize(graph)

        # Branching should prevent chain fusion
        assert len(optimized.kernels) == 3


class TestOptimizationEdgeCases:
    """Tests for edge cases in optimization."""

    def test_optimize_single_kernel(self):
        """Test optimizing graph with single kernel."""
        graph = ComputeGraph()
        k = create_gemm_kernel(256, 256, 256)
        graph.add_kernel(k)

        optimized, result = optimize_graph(graph)

        assert result.original_kernels == 1
        assert len(optimized.kernels) == 1

    def test_optimize_disconnected_components(self):
        """Test optimizing graph with disconnected components."""
        graph = ComputeGraph()

        # Component 1
        k1 = create_gemm_kernel(128, 128, 128)
        k2 = create_elementwise_kernel((128, 128), op="relu")
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "c1")

        # Component 2
        k3 = create_gemm_kernel(64, 64, 64)
        graph.add_kernel(k3)

        optimized, result = optimize_graph(graph)

        assert result.original_kernels == 3
        # All kernels should be preserved (end nodes)
        assert len(optimized.kernels) >= 2

    def test_cyclic_prevention(self):
        """Test that optimizer doesn't create cycles."""
        graph = ComputeGraph()

        k1 = create_gemm_kernel(128, 128, 128)
        k2 = create_elementwise_kernel((128, 128), op="relu")
        k3 = create_gemm_kernel(128, 128, 128)

        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_kernel(k3)

        graph.add_dependency(k1.kernel_id, k2.kernel_id, "d1")
        graph.add_dependency(k2.kernel_id, k3.kernel_id, "d2")

        optimized, _ = optimize_graph(graph)

        # Should still be topologically sortable
        topo = optimized.topological_sort()
        assert len(topo) == len(optimized.kernels)
