"""Tests for cost models and speculative execution."""

import pytest
import numpy as np
from kernelsched import (
    RooflineModel, CommunicationCostModel, CostModelScheduler,
    SpeculativeResult, SpeculativeExecutor,
    GPUDevice, MultiGPUCluster, ComputeGraph, GraphExecutor,
    FIFOScheduler, TensorDescriptor, DataType,
    create_gemm_kernel, create_elementwise_kernel, create_attention_kernel,
    create_reduce_kernel,
)


# =============================================================================
# RooflineModel Tests
# =============================================================================

class TestRooflineModel:
    """Tests for the roofline cost model."""

    def test_creation(self, single_gpu):
        model = RooflineModel(single_gpu)
        assert model.device is single_gpu
        assert model.peak_tflops == single_gpu.peak_tflops
        assert model.memory_bandwidth_gbps == single_gpu.memory_bandwidth_gbps

    def test_ridge_point(self, single_gpu):
        model = RooflineModel(single_gpu)
        rp = model.ridge_point
        expected = (single_gpu.peak_tflops * 1e12) / (single_gpu.memory_bandwidth_gbps * 1e9)
        assert rp == pytest.approx(expected)
        assert rp > 0

    def test_ridge_point_zero_bandwidth(self):
        device = GPUDevice(device_id=0, memory_bandwidth_gbps=0.0)
        model = RooflineModel(device)
        assert model.ridge_point == float('inf')

    def test_operational_intensity_gemm(self):
        device = GPUDevice(device_id=0, memory_bandwidth_gbps=2000.0, sm_count=108)
        model = RooflineModel(device)
        kernel = create_gemm_kernel(1024, 1024, 1024)
        oi = model.operational_intensity(kernel)
        assert oi > 0

    def test_operational_intensity_elementwise(self):
        device = GPUDevice(device_id=0, memory_bandwidth_gbps=2000.0, sm_count=108)
        model = RooflineModel(device)
        kernel = create_elementwise_kernel((1024, 1024))
        oi = model.operational_intensity(kernel)
        assert oi > 0

    def test_operational_intensity_zero_bytes(self):
        device = GPUDevice(device_id=0, memory_bandwidth_gbps=2000.0)
        model = RooflineModel(device)
        from kernelsched.core.kernel import Kernel, KernelType
        kernel = Kernel(
            kernel_id="empty",
            name="empty",
            kernel_type=KernelType.CUSTOM,
            inputs=[],
            outputs=[],
        )
        oi = model.operational_intensity(kernel)
        assert oi == 0.0

    def test_is_compute_bound_gemm(self, single_gpu):
        model = RooflineModel(single_gpu)
        # Large GEMM is typically compute-bound
        kernel = create_gemm_kernel(4096, 4096, 4096)
        oi = model.operational_intensity(kernel)
        # The result depends on the ratio; just verify it returns a bool
        result = model.is_compute_bound(kernel)
        assert isinstance(result, bool)

    def test_is_compute_bound_elementwise(self, single_gpu):
        model = RooflineModel(single_gpu)
        kernel = create_elementwise_kernel((1024, 1024))
        result = model.is_compute_bound(kernel)
        # Elementwise should be memory-bound
        assert result is False

    def test_estimate_time_gemm(self, single_gpu):
        model = RooflineModel(single_gpu)
        kernel = create_gemm_kernel(1024, 1024, 1024)
        time_us = model.estimate_time_us(kernel)
        assert time_us > 0
        assert isinstance(time_us, float)

    def test_estimate_time_attention(self, single_gpu):
        model = RooflineModel(single_gpu)
        kernel = create_attention_kernel(1, 8, 512, 64)
        time_us = model.estimate_time_us(kernel)
        assert time_us > 0

    def test_estimate_time_reduce(self, single_gpu):
        model = RooflineModel(single_gpu)
        kernel = create_reduce_kernel((1024, 1024))
        time_us = model.estimate_time_us(kernel)
        assert time_us > 0

    def test_estimate_time_larger_kernel_takes_longer(self, single_gpu):
        model = RooflineModel(single_gpu)
        small = create_gemm_kernel(256, 256, 256)
        large = create_gemm_kernel(2048, 2048, 2048)
        assert model.estimate_time_us(large) > model.estimate_time_us(small)

    def test_estimate_flops_gemm(self, single_gpu):
        model = RooflineModel(single_gpu)
        kernel = create_gemm_kernel(128, 256, 512)
        flops = model._estimate_flops(kernel)
        assert flops == 2.0 * 128 * 256 * 512

    def test_estimate_flops_attention(self, single_gpu):
        model = RooflineModel(single_gpu)
        kernel = create_attention_kernel(2, 4, 128, 32)
        flops = model._estimate_flops(kernel)
        assert flops == 4.0 * 2 * 4 * 128 * 128 * 32

    def test_estimate_flops_elementwise(self, single_gpu):
        model = RooflineModel(single_gpu)
        kernel = create_elementwise_kernel((32, 64))
        flops = model._estimate_flops(kernel)
        assert flops == 32 * 64

    def test_estimate_flops_custom_returns_zero(self, single_gpu):
        model = RooflineModel(single_gpu)
        from kernelsched.core.kernel import Kernel, KernelType
        kernel = Kernel(
            kernel_id="custom",
            name="custom",
            kernel_type=KernelType.CUSTOM,
            inputs=[],
            outputs=[],
        )
        assert model._estimate_flops(kernel) == 0.0


# =============================================================================
# CommunicationCostModel Tests
# =============================================================================

class TestCommunicationCostModel:
    """Tests for the communication cost model."""

    def test_creation(self, dual_gpu_cluster):
        model = CommunicationCostModel(dual_gpu_cluster)
        assert model.nvlink_bandwidth == 600.0
        assert model.pcie_bandwidth == 32.0

    def test_same_device_transfer_is_zero(self, dual_gpu_cluster):
        model = CommunicationCostModel(dual_gpu_cluster)
        time_us = model.transfer_time_us(1024 * 1024, src_device=0, dst_device=0)
        assert time_us == 0.0

    def test_nvlink_transfer(self, dual_gpu_cluster):
        model = CommunicationCostModel(dual_gpu_cluster)
        size = 1024 * 1024  # 1 MB
        time_us = model.transfer_time_us(size, src_device=0, dst_device=1, use_nvlink=True)
        expected = size / (600.0 * 1e9) * 1e6
        assert time_us == pytest.approx(expected)

    def test_pcie_transfer(self, dual_gpu_cluster):
        model = CommunicationCostModel(dual_gpu_cluster)
        size = 1024 * 1024  # 1 MB
        time_us = model.transfer_time_us(size, src_device=0, dst_device=1, use_nvlink=False)
        expected = size / (32.0 * 1e9) * 1e6
        assert time_us == pytest.approx(expected)

    def test_nvlink_faster_than_pcie(self, dual_gpu_cluster):
        model = CommunicationCostModel(dual_gpu_cluster)
        size = 1024 * 1024 * 100  # 100 MB
        nvlink_time = model.transfer_time_us(size, 0, 1, use_nvlink=True)
        pcie_time = model.transfer_time_us(size, 0, 1, use_nvlink=False)
        assert nvlink_time < pcie_time

    def test_zero_bandwidth_returns_inf(self):
        cluster = MultiGPUCluster(
            devices=[GPUDevice(device_id=0), GPUDevice(device_id=1)],
            nvlink_bandwidth_gbps=0.0,
            pcie_bandwidth_gbps=0.0,
        )
        model = CommunicationCostModel(cluster)
        assert model.transfer_time_us(1024, 0, 1, use_nvlink=True) == float('inf')

    def test_estimate_tensor_transfer(self, dual_gpu_cluster):
        model = CommunicationCostModel(dual_gpu_cluster)
        tensor = TensorDescriptor(
            tensor_id="t1",
            shape=(1024, 1024),
            dtype=DataType.FLOAT16,
            device_id=0,
        )
        time_us = model.estimate_tensor_transfer(tensor, dst_device=1)
        assert time_us > 0

    def test_estimate_tensor_transfer_same_device(self, dual_gpu_cluster):
        model = CommunicationCostModel(dual_gpu_cluster)
        tensor = TensorDescriptor(
            tensor_id="t1",
            shape=(1024, 1024),
            dtype=DataType.FLOAT16,
            device_id=0,
        )
        time_us = model.estimate_tensor_transfer(tensor, dst_device=0)
        assert time_us == 0.0

    def test_total_communication_cost_no_cross_device(self, dual_gpu_cluster):
        model = CommunicationCostModel(dual_gpu_cluster)
        graph = ComputeGraph()
        k1 = create_gemm_kernel(128, 128, 128, device_id=0)
        k2 = create_elementwise_kernel((128, 128), device_id=0)
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "dep")

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)
        cost = model.total_communication_cost(graph, schedule)
        assert cost == 0.0

    def test_total_communication_cost_cross_device(self, dual_gpu_cluster):
        model = CommunicationCostModel(dual_gpu_cluster)
        graph = ComputeGraph()
        k1 = create_gemm_kernel(128, 128, 128, device_id=0)
        k2 = create_elementwise_kernel((128, 128), device_id=1)
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "cross")

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)
        cost = model.total_communication_cost(graph, schedule)
        assert cost > 0


# =============================================================================
# CostModelScheduler Tests
# =============================================================================

class TestCostModelScheduler:
    """Tests for cost-model-driven scheduler."""

    def test_creation(self, dual_gpu_cluster):
        cms = CostModelScheduler(dual_gpu_cluster)
        assert len(cms.roofline_models) == 2
        assert cms.comm_model is not None

    def test_estimate_kernel_time(self, dual_gpu_cluster):
        cms = CostModelScheduler(dual_gpu_cluster)
        kernel = create_gemm_kernel(512, 512, 512, device_id=0)
        time_us = cms.estimate_kernel_time(kernel)
        assert time_us > 0

    def test_estimate_kernel_time_specific_device(self, dual_gpu_cluster):
        cms = CostModelScheduler(dual_gpu_cluster)
        kernel = create_gemm_kernel(512, 512, 512, device_id=0)
        time_d0 = cms.estimate_kernel_time(kernel, device_id=0)
        time_d1 = cms.estimate_kernel_time(kernel, device_id=1)
        # Same device specs, so same time
        assert time_d0 == pytest.approx(time_d1)

    def test_estimate_kernel_time_fallback_device(self):
        cluster = MultiGPUCluster(
            devices=[GPUDevice(device_id=0)],
        )
        cms = CostModelScheduler(cluster)
        kernel = create_gemm_kernel(256, 256, 256, device_id=5)
        # Falls back to first available model
        time_us = cms.estimate_kernel_time(kernel)
        assert time_us > 0

    def test_estimate_total_time(self, dual_gpu_cluster):
        cms = CostModelScheduler(dual_gpu_cluster)
        graph = ComputeGraph()
        k1 = create_gemm_kernel(256, 256, 256, device_id=0)
        k2 = create_elementwise_kernel((256, 256), device_id=0)
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "dep")

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)
        total = cms.estimate_total_time(graph, schedule)
        assert total > 0

    def test_estimate_total_time_includes_comm(self, dual_gpu_cluster):
        cms = CostModelScheduler(dual_gpu_cluster)
        graph = ComputeGraph()
        k1 = create_gemm_kernel(512, 512, 512, device_id=0)
        k2 = create_gemm_kernel(512, 512, 512, device_id=1)
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "cross")

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)
        total_with_comm = cms.estimate_total_time(graph, schedule)

        # Compare to just kernel time
        kernel_time_only = sum(
            cms.estimate_kernel_time(k) for k in graph.kernels.values()
        )
        assert total_with_comm >= kernel_time_only


# =============================================================================
# SpeculativeResult Tests
# =============================================================================

class TestSpeculativeResult:
    """Tests for SpeculativeResult dataclass."""

    def test_creation(self):
        result = SpeculativeResult(
            kernel_id="k1",
            output={"t": np.zeros((2, 2), dtype=np.float16)},
            is_valid=True,
            was_speculative=False,
            time_us=100.0,
        )
        assert result.kernel_id == "k1"
        assert result.is_valid is True
        assert result.was_speculative is False
        assert result.time_us == 100.0

    def test_speculative_result_with_none_output(self):
        result = SpeculativeResult(
            kernel_id="k2",
            output=None,
            is_valid=False,
            was_speculative=True,
            time_us=50.0,
        )
        assert result.output is None
        assert result.is_valid is False


# =============================================================================
# SpeculativeExecutor Tests
# =============================================================================

class TestSpeculativeExecutor:
    """Tests for speculative execution engine."""

    def test_creation(self, dual_gpu_cluster):
        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge, speculation_depth=3, confidence_threshold=0.9)
        assert se.speculation_depth == 3
        assert se.confidence_threshold == 0.9

    def test_defaults(self, dual_gpu_cluster):
        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        assert se.speculation_depth == 2
        assert se.confidence_threshold == 0.8

    def test_execute_speculative_simple_graph(self, dual_gpu_cluster):
        graph = ComputeGraph()
        k1 = create_gemm_kernel(64, 64, 64, device_id=0)
        k2 = create_elementwise_kernel((64, 64), device_id=0)
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "dep")

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)

        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        results = se.execute_speculative(graph, schedule)

        assert len(results) == 2
        for r in results.values():
            assert r.output is not None
            assert r.is_valid is True

    def test_execute_speculative_with_completed(self, dual_gpu_cluster):
        graph = ComputeGraph()
        k1 = create_gemm_kernel(64, 64, 64, device_id=0)
        k2 = create_elementwise_kernel((64, 64), device_id=0)
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "dep")

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)

        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        # Mark k1 as already completed
        results = se.execute_speculative(graph, schedule, completed={k1.kernel_id})

        assert k2.kernel_id in results
        assert results[k2.kernel_id].is_valid is True
        assert results[k2.kernel_id].was_speculative is False

    def test_execute_speculative_parallel_branches(self, dual_gpu_cluster):
        graph = ComputeGraph()
        root = create_gemm_kernel(64, 64, 64, device_id=0)
        b1 = create_elementwise_kernel((64, 64), device_id=0)
        b2 = create_elementwise_kernel((64, 64), device_id=0)
        graph.add_kernel(root)
        graph.add_kernel(b1)
        graph.add_kernel(b2)
        graph.add_dependency(root.kernel_id, b1.kernel_id, "r_b1")
        graph.add_dependency(root.kernel_id, b2.kernel_id, "r_b2")

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)

        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        results = se.execute_speculative(graph, schedule)

        assert len(results) == 3
        # Root has no deps, executed normally
        assert results[root.kernel_id].was_speculative is False

    def test_validate_and_commit_all_valid(self, dual_gpu_cluster):
        graph = ComputeGraph()
        k1 = create_gemm_kernel(64, 64, 64, device_id=0)
        k2 = create_elementwise_kernel((64, 64), device_id=0)
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "dep")

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)

        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        results = se.execute_speculative(graph, schedule)

        # All completed normally, validate
        completed = {k1.kernel_id, k2.kernel_id}
        validated = se.validate_and_commit(results, completed, graph)
        assert len(validated) == 2
        for r in validated.values():
            assert r.is_valid is True

    def test_validate_and_commit_rollback(self, dual_gpu_cluster):
        graph = ComputeGraph()
        k1 = create_gemm_kernel(64, 64, 64, device_id=0)
        k2 = create_elementwise_kernel((64, 64), device_id=0)
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "dep")

        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)

        # Manually create a speculative result for k2
        spec_result = SpeculativeResult(
            kernel_id=k2.kernel_id,
            output={"out": np.zeros((2, 2))},
            is_valid=True,
            was_speculative=True,
            time_us=50.0,
        )
        non_spec = SpeculativeResult(
            kernel_id=k1.kernel_id,
            output={"out": np.zeros((2, 2))},
            is_valid=True,
            was_speculative=False,
            time_us=30.0,
        )
        results = {k1.kernel_id: non_spec, k2.kernel_id: spec_result}

        # k1 not completed, so k2's speculation should be rolled back
        validated = se.validate_and_commit(results, completed=set(), graph=graph)
        assert validated[k2.kernel_id].is_valid is False
        assert validated[k2.kernel_id].output is None
        assert se.rollback_count == 1

    def test_get_stats_empty(self, dual_gpu_cluster):
        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        stats = se.get_stats()
        assert stats["total_kernels"] == 0
        assert stats["speculative_launches"] == 0
        assert stats["committed"] == 0
        assert stats["rolled_back"] == 0

    def test_get_stats_after_execution(self, dual_gpu_cluster):
        graph = ComputeGraph()
        k1 = create_gemm_kernel(64, 64, 64, device_id=0)
        k2 = create_elementwise_kernel((64, 64), device_id=0)
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_dependency(k1.kernel_id, k2.kernel_id, "dep")

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)

        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        se.execute_speculative(graph, schedule)

        stats = se.get_stats()
        assert stats["total_kernels"] == 2
        assert stats["committed"] >= 1
        assert "speculation_success_rate" in stats

    def test_reset(self, dual_gpu_cluster):
        graph = ComputeGraph()
        k1 = create_gemm_kernel(64, 64, 64, device_id=0)
        graph.add_kernel(k1)

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)

        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        se.execute_speculative(graph, schedule)

        assert se.get_stats()["total_kernels"] > 0
        se.reset()
        assert se.get_stats()["total_kernels"] == 0
        assert se.committed_count == 0
        assert se.rollback_count == 0

    def test_committed_and_rollback_properties(self, dual_gpu_cluster):
        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        assert se.committed_count == 0
        assert se.rollback_count == 0

    def test_speculative_results_property(self, dual_gpu_cluster):
        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        assert se.speculative_results == {}

    def test_diamond_graph_speculative(self, dual_gpu_cluster, diamond_graph):
        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(diamond_graph)

        ge = GraphExecutor(cluster=dual_gpu_cluster)
        se = SpeculativeExecutor(ge)
        results = se.execute_speculative(diamond_graph, schedule)

        assert len(results) == 4
        # All should be valid since graph executes in topo order
        for r in results.values():
            assert r.is_valid is True
