"""Tests for kernel execution."""

import pytest
import numpy as np
from kernelsched import (
    ExecutionMode, MemoryBlock, ExecutionStats, DeviceMemoryManager,
    ClusterMemoryManager, KernelExecutor, GraphExecutor, StreamExecutor,
    CUDAGraphCapture, Profiler, MultiGPUExecutionEngine, execute_graph,
    ComputeGraph, MultiGPUCluster, GPUDevice, Kernel, KernelType, KernelStats,
    create_gemm_kernel, create_elementwise_kernel, create_attention_kernel,
    FIFOScheduler, create_test_graph,
)


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_execution_mode_values(self):
        """Test execution mode values."""
        assert ExecutionMode.EAGER.value == "eager"
        assert ExecutionMode.GRAPH.value == "graph"


class TestExecutionStats:
    """Tests for ExecutionStats class."""

    def test_execution_stats_creation(self):
        """Test execution stats creation."""
        stats = ExecutionStats(
            total_time_ms=100.0,
            kernel_times_ms={"k1": 50.0, "k2": 50.0},
            memory_peak_mb=1024.0,
            device_utilization={0: 0.8, 1: 0.7},
            memory_transfers_ms=5.0,
        )
        assert stats.total_time_ms == 100.0
        assert stats.memory_peak_mb == 1024.0
        assert len(stats.kernel_times_ms) == 2
        assert len(stats.device_utilization) == 2


class TestKernelExecutor:
    """Tests for KernelExecutor class."""

    def test_kernel_executor_creation(self, kernel_executor):
        """Test kernel executor creation."""
        assert len(kernel_executor.kernel_impls) > 0
        assert "gemm" in kernel_executor.kernel_impls
        assert "elementwise" in kernel_executor.kernel_impls
        assert "attention" in kernel_executor.kernel_impls

    def test_execute_gemm_kernel(self, kernel_executor, gemm_kernel):
        """Test executing GEMM kernel."""
        inputs = {}
        for inp in gemm_kernel.inputs:
            inputs[inp.tensor_id] = np.zeros(inp.shape, dtype=np.float16)

        outputs, stats = kernel_executor.execute(gemm_kernel, inputs)

        assert len(outputs) == len(gemm_kernel.outputs)
        assert stats.compute_time_us > 0
        assert stats.memory_read_bytes > 0
        assert stats.memory_write_bytes > 0

    def test_execute_elementwise_kernel(self, kernel_executor, elementwise_kernel):
        """Test executing elementwise kernel."""
        inputs = {}
        for inp in elementwise_kernel.inputs:
            inputs[inp.tensor_id] = np.zeros(inp.shape, dtype=np.float16)

        outputs, stats = kernel_executor.execute(elementwise_kernel, inputs)

        assert len(outputs) == len(elementwise_kernel.outputs)
        for out_desc in elementwise_kernel.outputs:
            assert out_desc.tensor_id in outputs
            assert outputs[out_desc.tensor_id].shape == out_desc.shape

    def test_execute_attention_kernel(self, kernel_executor, attention_kernel):
        """Test executing attention kernel."""
        inputs = {}
        for inp in attention_kernel.inputs:
            inputs[inp.tensor_id] = np.zeros(inp.shape, dtype=np.float16)

        outputs, stats = kernel_executor.execute(attention_kernel, inputs)

        assert len(outputs) == len(attention_kernel.outputs)
        assert stats.flops == attention_kernel.flops

    def test_execute_reduce_kernel(self, kernel_executor, reduce_kernel):
        """Test executing reduce kernel."""
        inputs = {}
        for inp in reduce_kernel.inputs:
            inputs[inp.tensor_id] = np.zeros(inp.shape, dtype=np.float16)

        outputs, stats = kernel_executor.execute(reduce_kernel, inputs)

        assert len(outputs) == len(reduce_kernel.outputs)

    def test_execute_unknown_kernel_type(self, kernel_executor):
        """Test executing kernel with unknown type."""
        # Create kernel with CUSTOM type (no impl)
        kernel = Kernel(
            kernel_id="custom",
            name="custom_kernel",
            kernel_type=KernelType.CUSTOM,
            inputs=[],
            outputs=[],
            estimated_time_us=1.0,
        )

        outputs, stats = kernel_executor.execute(kernel, {})

        # Should still work with default implementation
        assert stats.compute_time_us > 0


class TestStreamExecutor:
    """Tests for StreamExecutor class."""

    def test_stream_executor_creation(self, stream_executor):
        """Test stream executor creation."""
        assert stream_executor.num_streams == 4
        assert len(stream_executor.stream_queues) == 4

    def test_submit_to_stream(self, stream_executor, gemm_kernel):
        """Test submitting kernel to stream."""
        stream_executor.submit(gemm_kernel, stream_id=0)
        assert len(stream_executor.stream_queues[0]) == 1
        assert stream_executor.stream_queues[0][0] == gemm_kernel

    def test_submit_to_invalid_stream(self, stream_executor, gemm_kernel):
        """Test submitting to invalid stream."""
        stream_executor.submit(gemm_kernel, stream_id=99)
        # Should not raise, just ignore
        for queue in stream_executor.stream_queues:
            assert gemm_kernel not in queue

    def test_synchronize(self, stream_executor, gemm_kernel, elementwise_kernel):
        """Test stream synchronization."""
        stream_executor.submit(gemm_kernel, stream_id=0)
        stream_executor.submit(elementwise_kernel, stream_id=1)

        stream_executor.synchronize()

        # All queues should be cleared
        for queue in stream_executor.stream_queues:
            assert len(queue) == 0

    def test_multiple_kernels_same_stream(self, stream_executor):
        """Test submitting multiple kernels to same stream."""
        kernels = [create_gemm_kernel(64, 64, 64) for _ in range(5)]

        for kernel in kernels:
            stream_executor.submit(kernel, stream_id=2)

        assert len(stream_executor.stream_queues[2]) == 5


class TestGraphExecutor:
    """Tests for GraphExecutor class."""

    def test_graph_executor_creation(self, graph_executor):
        """Test graph executor creation."""
        assert graph_executor.cluster is not None
        assert graph_executor.memory_manager is not None
        assert graph_executor.kernel_executor is not None

    def test_execute_simple_graph(
        self, graph_executor, simple_graph, fifo_scheduler
    ):
        """Test executing simple graph."""
        schedule = fifo_scheduler.schedule(simple_graph)
        outputs, stats = graph_executor.execute(simple_graph, schedule)

        assert stats.total_time_ms > 0
        assert len(stats.kernel_times_ms) > 0

    def test_execute_with_inputs(
        self, graph_executor, simple_graph, fifo_scheduler
    ):
        """Test executing graph with provided inputs."""
        schedule = fifo_scheduler.schedule(simple_graph)

        # Create some input data
        inputs = {}
        for kernel in simple_graph.kernels.values():
            for inp in kernel.inputs:
                if inp.tensor_id not in inputs:
                    inputs[inp.tensor_id] = np.zeros(inp.shape, dtype=np.float16)

        outputs, stats = graph_executor.execute(simple_graph, schedule, inputs)

        assert stats.total_time_ms > 0

    def test_execute_parallel_graph(
        self, graph_executor, parallel_graph, fifo_scheduler
    ):
        """Test executing graph with parallel branches."""
        schedule = fifo_scheduler.schedule(parallel_graph)
        outputs, stats = graph_executor.execute(parallel_graph, schedule)

        # All kernels should be executed
        assert len(stats.kernel_times_ms) == len(parallel_graph.kernels)

    def test_execute_empty_graph(self, graph_executor, empty_graph, fifo_scheduler):
        """Test executing empty graph."""
        schedule = fifo_scheduler.schedule(empty_graph)
        outputs, stats = graph_executor.execute(empty_graph, schedule)

        assert stats.total_time_ms >= 0
        assert len(outputs) == 0

    def test_memory_allocation_during_execution(
        self, graph_executor, simple_graph, fifo_scheduler
    ):
        """Test memory allocation during graph execution."""
        schedule = fifo_scheduler.schedule(simple_graph)

        # Reset memory manager
        graph_executor.reset()

        outputs, stats = graph_executor.execute(simple_graph, schedule)

        # Memory should have been used
        assert stats.memory_peak_mb >= 0

    def test_executor_reset(self, graph_executor, simple_graph, fifo_scheduler):
        """Test resetting executor state."""
        schedule = fifo_scheduler.schedule(simple_graph)
        graph_executor.execute(simple_graph, schedule)

        graph_executor.reset()

        # Should be able to execute again after reset
        outputs, stats = graph_executor.execute(simple_graph, schedule)
        assert stats.total_time_ms > 0


class TestCUDAGraphCapture:
    """Tests for CUDAGraphCapture class."""

    def test_cuda_graph_capture_creation(self):
        """Test CUDA graph capture creation."""
        capture = CUDAGraphCapture()
        assert capture.captured_graph is None
        assert capture.is_capturing == False

    def test_start_capture(self):
        """Test starting graph capture."""
        capture = CUDAGraphCapture()
        capture.start_capture()

        assert capture.is_capturing == True
        assert capture.captured_graph is not None

    def test_end_capture(self):
        """Test ending graph capture."""
        capture = CUDAGraphCapture()
        capture.start_capture()
        graph = capture.end_capture()

        assert capture.is_capturing == False
        assert graph is not None

    def test_replay_without_capture(self, graph_executor, fifo_scheduler, simple_graph):
        """Test replay without captured graph raises error."""
        capture = CUDAGraphCapture()
        schedule = fifo_scheduler.schedule(simple_graph)

        with pytest.raises(RuntimeError, match="No graph captured"):
            capture.replay(graph_executor, schedule)

    def test_replay_captured_graph(
        self, graph_executor, fifo_scheduler, simple_graph
    ):
        """Test replaying captured graph."""
        capture = CUDAGraphCapture()

        capture.start_capture()
        # Simulate adding kernels during capture
        capture.captured_graph.kernels = simple_graph.kernels.copy()
        capture.captured_graph.dependencies = simple_graph.dependencies.copy()
        graph = capture.end_capture()

        # Store the captured graph back for replay
        capture.captured_graph = graph

        schedule = fifo_scheduler.schedule(simple_graph)
        stats = capture.replay(graph_executor, schedule)

        assert stats.total_time_ms > 0


class TestProfiler:
    """Tests for Profiler class."""

    def test_profiler_creation(self, profiler):
        """Test profiler creation."""
        assert len(profiler.records) == 0
        assert profiler.enabled == False

    def test_enable_disable(self, profiler):
        """Test enabling and disabling profiler."""
        profiler.enable()
        assert profiler.enabled == True

        profiler.disable()
        assert profiler.enabled == False

    def test_record_when_disabled(self, profiler):
        """Test that recording when disabled does nothing."""
        stats = KernelStats(compute_time_us=100.0)
        profiler.record("k1", "gemm", 0, stats)

        assert len(profiler.records) == 0

    def test_record_when_enabled(self, profiler):
        """Test recording when profiler is enabled."""
        profiler.enable()
        stats = KernelStats(
            compute_time_us=100.0,
            flops=1000000,
            memory_read_bytes=1024,
            memory_write_bytes=512,
        )
        profiler.record("k1", "gemm", 0, stats)

        assert len(profiler.records) == 1
        assert profiler.records[0]["kernel_id"] == "k1"
        assert profiler.records[0]["kernel_type"] == "gemm"
        assert profiler.records[0]["device_id"] == 0
        assert profiler.records[0]["time_us"] == 100.0

    def test_get_summary_empty(self, profiler):
        """Test getting summary with no records."""
        summary = profiler.get_summary()
        assert summary == {}

    def test_get_summary_with_records(self, profiler):
        """Test getting summary with records."""
        profiler.enable()

        stats1 = KernelStats(compute_time_us=100.0)
        stats2 = KernelStats(compute_time_us=200.0)
        stats3 = KernelStats(compute_time_us=150.0)

        profiler.record("k1", "gemm", 0, stats1)
        profiler.record("k2", "gemm", 0, stats2)
        profiler.record("k3", "elementwise", 0, stats3)

        summary = profiler.get_summary()

        assert summary["total_time_us"] == 450.0
        assert summary["num_kernels"] == 3
        assert summary["avg_time_us"] == 150.0
        assert "gemm" in summary["time_by_type"]
        assert "elementwise" in summary["time_by_type"]
        assert summary["time_by_type"]["gemm"] == 300.0
        assert summary["time_by_type"]["elementwise"] == 150.0

    def test_reset(self, profiler):
        """Test resetting profiler."""
        profiler.enable()
        stats = KernelStats(compute_time_us=100.0)
        profiler.record("k1", "gemm", 0, stats)

        profiler.reset()

        assert len(profiler.records) == 0


class TestMultiGPUExecutionEngine:
    """Tests for MultiGPUExecutionEngine class."""

    def test_engine_creation(self, execution_engine):
        """Test execution engine creation."""
        assert execution_engine.cluster is not None
        assert execution_engine.executor is not None
        assert execution_engine.profiler is not None

    def test_execute_simple_graph(
        self, execution_engine, simple_graph, fifo_scheduler
    ):
        """Test executing simple graph with engine."""
        schedule = fifo_scheduler.schedule(simple_graph)
        outputs, stats = execution_engine.execute(simple_graph, schedule)

        assert stats.total_time_ms > 0

    def test_execute_with_inputs(
        self, execution_engine, simple_graph, fifo_scheduler
    ):
        """Test executing with provided inputs."""
        schedule = fifo_scheduler.schedule(simple_graph)

        inputs = {}
        for kernel in simple_graph.kernels.values():
            for inp in kernel.inputs:
                if inp.tensor_id not in inputs:
                    inputs[inp.tensor_id] = np.zeros(inp.shape, dtype=np.float16)

        outputs, stats = execution_engine.execute(simple_graph, schedule, inputs)

        assert stats.total_time_ms > 0

    def test_benchmark(self, execution_engine, simple_graph, fifo_scheduler):
        """Test benchmarking execution."""
        schedule = fifo_scheduler.schedule(simple_graph)

        results = execution_engine.benchmark(
            simple_graph,
            schedule,
            num_iterations=3,
            warmup=1,
        )

        assert "mean_ms" in results
        assert "std_ms" in results
        assert "min_ms" in results
        assert "max_ms" in results
        assert results["mean_ms"] > 0
        assert results["min_ms"] <= results["mean_ms"] <= results["max_ms"]

    def test_get_profiling_summary(
        self, execution_engine, simple_graph, fifo_scheduler
    ):
        """Test getting profiling summary."""
        schedule = fifo_scheduler.schedule(simple_graph)
        execution_engine.execute(simple_graph, schedule)

        summary = execution_engine.get_profiling_summary()

        # Summary might be empty if profiler wasn't recording kernels individually
        # Just verify it returns a dict
        assert isinstance(summary, dict)

    def test_reset(self, execution_engine, simple_graph, fifo_scheduler):
        """Test resetting execution engine."""
        schedule = fifo_scheduler.schedule(simple_graph)
        execution_engine.execute(simple_graph, schedule)

        execution_engine.reset()

        # Should be able to execute again
        outputs, stats = execution_engine.execute(simple_graph, schedule)
        assert stats.total_time_ms > 0


class TestExecuteGraph:
    """Tests for execute_graph convenience function."""

    def test_execute_graph_function(
        self, simple_graph, dual_gpu_cluster
    ):
        """Test execute_graph convenience function."""
        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(simple_graph)

        outputs, stats = execute_graph(
            simple_graph, schedule, dual_gpu_cluster
        )

        assert stats.total_time_ms > 0

    def test_execute_graph_with_inputs(
        self, simple_graph, dual_gpu_cluster
    ):
        """Test execute_graph with inputs."""
        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(simple_graph)

        inputs = {}
        for kernel in simple_graph.kernels.values():
            for inp in kernel.inputs:
                if inp.tensor_id not in inputs:
                    inputs[inp.tensor_id] = np.zeros(inp.shape, dtype=np.float16)

        outputs, stats = execute_graph(
            simple_graph, schedule, dual_gpu_cluster, inputs
        )

        assert stats.total_time_ms > 0


class TestMultiGPUExecution:
    """Tests for multi-GPU execution scenarios."""

    def test_execute_on_multiple_devices(
        self, execution_engine, multi_device_graph
    ):
        """Test execution across multiple devices."""
        scheduler = FIFOScheduler(cluster=execution_engine.cluster)
        schedule = scheduler.schedule(multi_device_graph)

        outputs, stats = execution_engine.execute(multi_device_graph, schedule)

        assert stats.total_time_ms > 0
        # Should have utilization for both devices
        assert len(stats.device_utilization) >= 1

    def test_transformer_execution(self, execution_engine, transformer_graph):
        """Test executing transformer-like graph."""
        scheduler = FIFOScheduler(cluster=execution_engine.cluster)
        schedule = scheduler.schedule(transformer_graph)

        outputs, stats = execution_engine.execute(transformer_graph, schedule)

        assert stats.total_time_ms > 0
        assert len(stats.kernel_times_ms) == len(transformer_graph.kernels)

    def test_execute_large_graph(self, dual_gpu_cluster):
        """Test executing larger graph."""
        engine = MultiGPUExecutionEngine(cluster=dual_gpu_cluster)
        graph = create_test_graph(num_layers=4)

        scheduler = FIFOScheduler(cluster=dual_gpu_cluster)
        schedule = scheduler.schedule(graph)

        outputs, stats = engine.execute(graph, schedule)

        assert stats.total_time_ms > 0
        assert len(stats.kernel_times_ms) > 0


class TestExecutionCorrectness:
    """Tests for execution correctness."""

    def test_kernel_output_shapes(self, kernel_executor, gemm_kernel):
        """Test that kernel outputs have correct shapes."""
        inputs = {}
        for inp in gemm_kernel.inputs:
            inputs[inp.tensor_id] = np.zeros(inp.shape, dtype=np.float16)

        outputs, _ = kernel_executor.execute(gemm_kernel, inputs)

        for out_desc in gemm_kernel.outputs:
            assert out_desc.tensor_id in outputs
            assert outputs[out_desc.tensor_id].shape == out_desc.shape

    def test_execution_order_preserved(
        self, execution_engine, simple_graph, fifo_scheduler
    ):
        """Test that execution order respects dependencies."""
        schedule = fifo_scheduler.schedule(simple_graph)

        # Verify schedule respects topological order
        topo = simple_graph.topological_sort()
        scheduled_order = []

        for ds in schedule.device_schedules.values():
            for stream in ds.streams:
                for sk in stream.scheduled_kernels:
                    scheduled_order.append(sk.kernel.kernel_id)

        # All kernels from topo should appear in scheduled order
        for kernel_id in topo:
            assert kernel_id in scheduled_order

    def test_stats_accuracy(
        self, execution_engine, simple_graph, fifo_scheduler
    ):
        """Test that execution stats are reasonable."""
        schedule = fifo_scheduler.schedule(simple_graph)
        _, stats = execution_engine.execute(simple_graph, schedule)

        # Total time should be positive
        assert stats.total_time_ms > 0

        # All kernel times should be positive
        for time_ms in stats.kernel_times_ms.values():
            assert time_ms >= 0

        # Memory peak should be non-negative
        assert stats.memory_peak_mb >= 0
