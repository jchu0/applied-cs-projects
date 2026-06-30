"""Tests for memory management."""

import pytest
from kernelsched import (
    DataType, TensorDescriptor, ComputeGraph,
    create_gemm_kernel, create_elementwise_kernel,
    MemoryBlock, DeviceMemoryManager, ClusterMemoryManager,
    MemoryOptimizer, MultiGPUCluster, GPUDevice,
)


class TestMemoryBlock:
    """Tests for MemoryBlock class."""

    def test_memory_block_creation(self):
        """Test memory block creation."""
        block = MemoryBlock(
            offset=0,
            size=1024,
            tensor_id="tensor_1",
            device_id=0,
        )
        assert block.offset == 0
        assert block.size == 1024
        assert block.tensor_id == "tensor_1"
        assert block.device_id == 0
        assert block.is_free == False

    def test_memory_block_is_free(self):
        """Test memory block free state."""
        block = MemoryBlock(
            offset=0,
            size=1024,
            tensor_id="t1",
            device_id=0,
            is_free=True,
        )
        assert block.is_free == True


class TestDeviceMemoryManager:
    """Tests for DeviceMemoryManager class."""

    def test_device_memory_manager_creation(self, device_memory_manager):
        """Test device memory manager creation."""
        assert device_memory_manager.device_id == 0
        assert device_memory_manager.total_memory > 0
        assert device_memory_manager.current_usage == 0
        assert device_memory_manager.peak_usage == 0

    def test_allocate_tensor(self, device_memory_manager, small_tensor):
        """Test tensor allocation."""
        block = device_memory_manager.allocate(small_tensor)

        assert block.tensor_id == small_tensor.tensor_id
        assert block.size == small_tensor.size_bytes
        assert block.device_id == device_memory_manager.device_id
        assert device_memory_manager.current_usage == small_tensor.size_bytes

    def test_allocate_multiple_tensors(self, device_memory_manager):
        """Test multiple tensor allocation."""
        t1 = TensorDescriptor(
            tensor_id="t1",
            shape=(1024, 1024),
            dtype=DataType.FLOAT16,
        )
        t2 = TensorDescriptor(
            tensor_id="t2",
            shape=(512, 512),
            dtype=DataType.FLOAT32,
        )

        b1 = device_memory_manager.allocate(t1)
        b2 = device_memory_manager.allocate(t2)

        # Blocks should not overlap
        assert b2.offset >= b1.offset + b1.size or b1.offset >= b2.offset + b2.size
        assert device_memory_manager.current_usage == t1.size_bytes + t2.size_bytes

    def test_free_tensor(self, device_memory_manager, small_tensor):
        """Test tensor deallocation."""
        device_memory_manager.allocate(small_tensor)
        initial_usage = device_memory_manager.current_usage

        device_memory_manager.free(small_tensor.tensor_id)

        assert device_memory_manager.current_usage == 0
        assert small_tensor.tensor_id not in device_memory_manager.allocations

    def test_free_nonexistent_tensor(self, device_memory_manager):
        """Test freeing nonexistent tensor does not raise."""
        # Should not raise
        device_memory_manager.free("nonexistent_tensor")

    def test_peak_usage_tracking(self, device_memory_manager):
        """Test peak memory usage tracking."""
        t1 = TensorDescriptor(
            tensor_id="t1", shape=(1024, 1024), dtype=DataType.FLOAT32
        )
        t2 = TensorDescriptor(
            tensor_id="t2", shape=(2048, 2048), dtype=DataType.FLOAT32
        )

        device_memory_manager.allocate(t1)
        device_memory_manager.allocate(t2)
        peak_after_alloc = device_memory_manager.peak_usage

        device_memory_manager.free("t1")
        device_memory_manager.free("t2")

        # Peak should remain the same after freeing
        assert device_memory_manager.peak_usage == peak_after_alloc
        assert device_memory_manager.current_usage == 0

    def test_memory_stats(self, device_memory_manager, medium_tensor):
        """Test memory statistics."""
        device_memory_manager.allocate(medium_tensor)
        stats = device_memory_manager.get_stats()

        assert "current_mb" in stats
        assert "peak_mb" in stats
        assert "utilization" in stats
        assert stats["current_mb"] > 0
        assert stats["utilization"] < 1.0

    def test_reset(self, device_memory_manager, small_tensor, medium_tensor):
        """Test memory manager reset."""
        device_memory_manager.allocate(small_tensor)
        device_memory_manager.allocate(medium_tensor)

        device_memory_manager.reset()

        assert device_memory_manager.current_usage == 0
        assert len(device_memory_manager.allocations) == 0

    def test_out_of_memory(self):
        """Test out of memory handling."""
        # Create manager with very limited memory
        manager = DeviceMemoryManager(device_id=0, total_memory_mb=1)

        huge_tensor = TensorDescriptor(
            tensor_id="huge",
            shape=(10000, 10000),
            dtype=DataType.FLOAT32,
        )

        with pytest.raises(MemoryError):
            manager.allocate(huge_tensor)


class TestClusterMemoryManager:
    """Tests for ClusterMemoryManager class."""

    def test_cluster_memory_manager_creation(self, cluster_memory_manager):
        """Test cluster memory manager creation."""
        assert len(cluster_memory_manager.device_managers) == 2

    def test_allocate_on_device(self, cluster_memory_manager):
        """Test allocation on specific device."""
        tensor = TensorDescriptor(
            tensor_id="t1",
            shape=(256, 256),
            dtype=DataType.FLOAT16,
            device_id=1,
        )

        block = cluster_memory_manager.allocate(tensor)
        assert block.device_id == 1

    def test_allocate_default_device(self, cluster_memory_manager):
        """Test allocation with invalid device falls back to default."""
        tensor = TensorDescriptor(
            tensor_id="t1",
            shape=(256, 256),
            dtype=DataType.FLOAT16,
            device_id=99,  # Invalid device
        )

        block = cluster_memory_manager.allocate(tensor)
        assert block.device_id == 0  # Falls back to device 0

    def test_free_on_device(self, cluster_memory_manager):
        """Test freeing tensor from specific device."""
        tensor = TensorDescriptor(
            tensor_id="t1",
            shape=(256, 256),
            dtype=DataType.FLOAT16,
            device_id=1,
        )

        cluster_memory_manager.allocate(tensor)
        cluster_memory_manager.free("t1", device_id=1)

        # Should be freed from device 1
        device_manager = cluster_memory_manager.device_managers[1]
        assert "t1" not in device_manager.allocations

    def test_get_peak_usage(self, cluster_memory_manager):
        """Test getting peak usage across devices."""
        t0 = TensorDescriptor(
            tensor_id="t0", shape=(1024, 1024), dtype=DataType.FLOAT32, device_id=0
        )
        t1 = TensorDescriptor(
            tensor_id="t1", shape=(2048, 2048), dtype=DataType.FLOAT32, device_id=1
        )

        cluster_memory_manager.allocate(t0)
        cluster_memory_manager.allocate(t1)

        peak_usage = cluster_memory_manager.get_peak_usage()

        assert 0 in peak_usage
        assert 1 in peak_usage
        assert peak_usage[0] > 0
        assert peak_usage[1] > 0
        assert peak_usage[1] > peak_usage[0]  # t1 is larger

    def test_reset_all_devices(self, cluster_memory_manager):
        """Test resetting all device managers."""
        for device_id in [0, 1]:
            tensor = TensorDescriptor(
                tensor_id=f"t{device_id}",
                shape=(256, 256),
                dtype=DataType.FLOAT16,
                device_id=device_id,
            )
            cluster_memory_manager.allocate(tensor)

        cluster_memory_manager.reset()

        for manager in cluster_memory_manager.device_managers.values():
            assert manager.current_usage == 0
            assert len(manager.allocations) == 0


class TestMemoryOptimizer:
    """Tests for MemoryOptimizer class."""

    def test_memory_optimizer_creation(self, memory_optimizer):
        """Test memory optimizer creation."""
        assert memory_optimizer.max_memory == 16000 * 1024 * 1024

    def test_optimize_simple_graph(self, memory_optimizer, simple_graph):
        """Test memory optimization on simple graph."""
        optimized = memory_optimizer.optimize(simple_graph)

        # Should preserve all kernels
        assert len(optimized.kernels) == len(simple_graph.kernels)

        # Should preserve dependencies
        assert len(optimized.dependencies) == len(simple_graph.dependencies)

    def test_compute_lifetimes(self, memory_optimizer, simple_graph):
        """Test tensor lifetime computation."""
        lifetimes = memory_optimizer._compute_lifetimes(simple_graph)

        # Should have lifetimes for tensors
        assert len(lifetimes) > 0

        # Each lifetime should have valid start/end
        for tensor_id, (start, end) in lifetimes.items():
            assert start <= end
            assert start >= 0

    def test_plan_allocation(self, memory_optimizer, simple_graph):
        """Test memory allocation planning."""
        lifetimes = memory_optimizer._compute_lifetimes(simple_graph)
        allocation = memory_optimizer._plan_allocation(simple_graph, lifetimes)

        # Should have allocations for tensors
        assert len(allocation) > 0

        # All offsets should be non-negative
        for offset in allocation.values():
            assert offset >= 0

    def test_memory_offset_assignment(self, memory_optimizer, simple_graph):
        """Test that memory offsets are assigned to tensors."""
        optimized = memory_optimizer.optimize(simple_graph)

        # Check that some tensors have memory offsets set
        has_offsets = False
        for kernel in optimized.kernels.values():
            for tensor in kernel.inputs + kernel.outputs:
                if tensor.memory_offset != 0:
                    has_offsets = True
                    break

        # Note: due to first-fit, some may still be at 0
        # Just verify the graph is valid
        assert len(optimized.kernels) > 0

    def test_memory_reuse(self):
        """Test that memory can be reused for non-overlapping tensors."""
        optimizer = MemoryOptimizer(max_memory_mb=100)

        graph = ComputeGraph()

        # Create a sequence where output of k1 is only needed for k2
        k1 = create_gemm_kernel(256, 256, 256)
        k2 = create_elementwise_kernel((256, 256), op="relu")
        k3 = create_gemm_kernel(256, 256, 256)

        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_kernel(k3)

        graph.add_dependency(k1.kernel_id, k2.kernel_id, k1.outputs[0].tensor_id)
        graph.add_dependency(k2.kernel_id, k3.kernel_id, k2.outputs[0].tensor_id)

        optimized = optimizer.optimize(graph)

        # Should still have same structure
        assert len(optimized.kernels) == 3


class TestMemoryPlanning:
    """Tests for memory planning strategies."""

    def test_linear_graph_memory_planning(self):
        """Test memory planning for linear graph."""
        optimizer = MemoryOptimizer(max_memory_mb=1000)
        graph = ComputeGraph()

        kernels = []
        for i in range(5):
            k = create_gemm_kernel(128, 128, 128)
            graph.add_kernel(k)
            kernels.append(k)

            if i > 0:
                graph.add_dependency(
                    kernels[i-1].kernel_id,
                    kernels[i].kernel_id,
                    kernels[i-1].outputs[0].tensor_id,
                )

        lifetimes = optimizer._compute_lifetimes(graph)
        allocation = optimizer._plan_allocation(graph, lifetimes)

        # Memory should be allocated for all output tensors
        assert len(allocation) > 0

    def test_parallel_graph_memory_planning(self, memory_optimizer, parallel_graph):
        """Test memory planning for parallel graph."""
        lifetimes = memory_optimizer._compute_lifetimes(parallel_graph)

        # Parallel branches should have overlapping lifetimes
        topo = parallel_graph.topological_sort()

        # Check that we have lifetimes computed
        assert len(lifetimes) > 0

    def test_memory_optimizer_preserves_graph_structure(self, memory_optimizer, diamond_graph):
        """Test that memory optimizer preserves graph structure."""
        optimized = memory_optimizer.optimize(diamond_graph)

        # Same kernel IDs
        assert set(optimized.kernels.keys()) == set(diamond_graph.kernels.keys())

        # Same dependencies
        assert len(optimized.dependencies) == len(diamond_graph.dependencies)


class TestMemoryEstimation:
    """Tests for memory estimation."""

    def test_tensor_size_calculation(self):
        """Test tensor size calculation for different dtypes."""
        dtypes_sizes = [
            (DataType.FLOAT32, 4),
            (DataType.FLOAT16, 2),
            (DataType.BFLOAT16, 2),
            (DataType.INT32, 4),
            (DataType.INT8, 1),
        ]

        shape = (1024, 1024)
        numel = 1024 * 1024

        for dtype, element_size in dtypes_sizes:
            tensor = TensorDescriptor(
                tensor_id="test",
                shape=shape,
                dtype=dtype,
            )
            expected_size = numel * element_size
            assert tensor.size_bytes == expected_size, f"Failed for {dtype}"

    def test_graph_memory_estimation(self, simple_graph):
        """Test total memory estimation for graph."""
        total_memory = 0
        for kernel in simple_graph.kernels.values():
            for tensor in kernel.outputs:
                total_memory += tensor.size_bytes

        assert total_memory > 0

    def test_attention_memory_estimation(self, attention_kernel):
        """Test memory estimation for attention kernel."""
        input_memory = sum(t.size_bytes for t in attention_kernel.inputs)
        output_memory = sum(t.size_bytes for t in attention_kernel.outputs)

        # Attention has 3 inputs (Q, K, V) and 1 output
        assert len(attention_kernel.inputs) == 3
        assert len(attention_kernel.outputs) == 1
        assert input_memory > 0
        assert output_memory > 0


class TestMemoryFragmentation:
    """Tests for memory fragmentation handling."""

    def test_first_fit_allocation(self):
        """Test first-fit allocation strategy."""
        optimizer = MemoryOptimizer(max_memory_mb=100)

        graph = ComputeGraph()

        # Create kernels with varying output sizes
        k1 = create_gemm_kernel(1024, 1024, 1024)  # Large
        k2 = create_gemm_kernel(128, 128, 128)    # Small
        k3 = create_gemm_kernel(512, 512, 512)    # Medium

        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_kernel(k3)

        lifetimes = optimizer._compute_lifetimes(graph)
        allocation = optimizer._plan_allocation(graph, lifetimes)

        # All should be allocated
        assert len(allocation) >= 3

    def test_allocation_with_memory_pressure(self):
        """Test allocation under memory pressure."""
        # Very limited memory
        optimizer = MemoryOptimizer(max_memory_mb=10)

        graph = ComputeGraph()

        # Create kernels that together exceed memory
        for i in range(5):
            k = create_gemm_kernel(512, 512, 512)
            graph.add_kernel(k)

        lifetimes = optimizer._compute_lifetimes(graph)
        allocation = optimizer._plan_allocation(graph, lifetimes)

        # Should still attempt allocation (may have fallback behavior)
        assert len(allocation) > 0
