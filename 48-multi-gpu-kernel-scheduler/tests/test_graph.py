"""Tests for compute graph operations."""

import pytest
from kernelsched import (
    DataType, KernelType, DeviceType, TensorDescriptor, KernelConfig,
    KernelStats, Kernel, KernelDependency, ComputeGraph, GPUDevice,
    MultiGPUCluster, create_gemm_kernel, create_attention_kernel,
    create_elementwise_kernel, create_reduce_kernel, create_test_graph,
)


class TestDataType:
    """Tests for DataType enum."""

    def test_data_type_values(self):
        """Test all data type values."""
        assert DataType.FLOAT32.value == "float32"
        assert DataType.FLOAT16.value == "float16"
        assert DataType.BFLOAT16.value == "bfloat16"
        assert DataType.INT32.value == "int32"
        assert DataType.INT8.value == "int8"


class TestKernelType:
    """Tests for KernelType enum."""

    def test_kernel_type_values(self):
        """Test all kernel type values."""
        assert KernelType.GEMM.value == "gemm"
        assert KernelType.CONV.value == "conv"
        assert KernelType.ELEMENTWISE.value == "elementwise"
        assert KernelType.REDUCE.value == "reduce"
        assert KernelType.SOFTMAX.value == "softmax"
        assert KernelType.ATTENTION.value == "attention"
        assert KernelType.LAYERNORM.value == "layernorm"
        assert KernelType.TRANSPOSE.value == "transpose"
        assert KernelType.MEMORY.value == "memory"
        assert KernelType.CUSTOM.value == "custom"


class TestTensorDescriptor:
    """Tests for TensorDescriptor class."""

    def test_tensor_descriptor_creation(self, small_tensor):
        """Test tensor descriptor creation."""
        assert small_tensor.tensor_id == "small_tensor"
        assert small_tensor.shape == (16, 16)
        assert small_tensor.dtype == DataType.FLOAT16
        assert small_tensor.device_id == 0

    def test_tensor_size_bytes_float16(self, small_tensor):
        """Test size calculation for FLOAT16 tensor."""
        # 16 * 16 * 2 bytes = 512 bytes
        assert small_tensor.size_bytes == 512

    def test_tensor_size_bytes_float32(self, large_tensor):
        """Test size calculation for FLOAT32 tensor."""
        # 4096 * 4096 * 4 bytes = 67,108,864 bytes
        assert large_tensor.size_bytes == 67_108_864

    def test_tensor_numel(self, small_tensor, medium_tensor, large_tensor):
        """Test number of elements calculation."""
        assert small_tensor.numel == 256
        assert medium_tensor.numel == 1_048_576
        assert large_tensor.numel == 16_777_216

    def test_tensor_stride_default(self, small_tensor):
        """Test default stride is None."""
        assert small_tensor.stride is None

    def test_tensor_memory_offset_default(self, small_tensor):
        """Test default memory offset."""
        assert small_tensor.memory_offset == 0

    def test_tensor_with_custom_stride(self):
        """Test tensor with custom stride."""
        tensor = TensorDescriptor(
            tensor_id="strided",
            shape=(32, 32),
            dtype=DataType.FLOAT32,
            stride=(64, 1),
        )
        assert tensor.stride == (64, 1)


class TestKernelConfig:
    """Tests for KernelConfig class."""

    def test_kernel_config_defaults(self):
        """Test default kernel configuration."""
        config = KernelConfig()
        assert config.block_size == (256, 1, 1)
        assert config.grid_size == (1, 1, 1)
        assert config.shared_memory_bytes == 0
        assert config.stream_id == 0

    def test_kernel_config_custom(self):
        """Test custom kernel configuration."""
        config = KernelConfig(
            block_size=(128, 2, 1),
            grid_size=(32, 32, 1),
            shared_memory_bytes=48000,
            stream_id=3,
        )
        assert config.block_size == (128, 2, 1)
        assert config.grid_size == (32, 32, 1)
        assert config.shared_memory_bytes == 48000
        assert config.stream_id == 3


class TestKernelStats:
    """Tests for KernelStats class."""

    def test_kernel_stats_defaults(self):
        """Test default kernel statistics."""
        stats = KernelStats()
        assert stats.compute_time_us == 0.0
        assert stats.memory_bandwidth_gbps == 0.0
        assert stats.flops == 0
        assert stats.memory_read_bytes == 0
        assert stats.memory_write_bytes == 0

    def test_kernel_stats_custom(self):
        """Test custom kernel statistics."""
        stats = KernelStats(
            compute_time_us=100.5,
            memory_bandwidth_gbps=1800.0,
            flops=1_000_000_000,
            memory_read_bytes=1024 * 1024,
            memory_write_bytes=512 * 1024,
        )
        assert stats.compute_time_us == 100.5
        assert stats.flops == 1_000_000_000


class TestKernel:
    """Tests for Kernel class."""

    def test_gemm_kernel_creation(self, gemm_kernel):
        """Test GEMM kernel creation."""
        assert gemm_kernel.kernel_type == KernelType.GEMM
        assert len(gemm_kernel.inputs) == 2
        assert len(gemm_kernel.outputs) == 1
        assert gemm_kernel.estimated_time_us > 0

    def test_attention_kernel_creation(self, attention_kernel):
        """Test attention kernel creation."""
        assert attention_kernel.kernel_type == KernelType.ATTENTION
        assert len(attention_kernel.inputs) == 3  # Q, K, V
        assert len(attention_kernel.outputs) == 1

    def test_elementwise_kernel_creation(self, elementwise_kernel):
        """Test elementwise kernel creation."""
        assert elementwise_kernel.kernel_type == KernelType.ELEMENTWISE
        assert len(elementwise_kernel.inputs) == 2
        assert len(elementwise_kernel.outputs) == 1

    def test_reduce_kernel_creation(self, reduce_kernel):
        """Test reduce kernel creation."""
        assert reduce_kernel.kernel_type == KernelType.REDUCE
        assert len(reduce_kernel.inputs) == 1
        assert len(reduce_kernel.outputs) == 1

    def test_gemm_flops(self, gemm_kernel):
        """Test GEMM FLOPs calculation."""
        # 2 * M * K * N for 1024x1024x1024
        expected_flops = 2 * 1024 * 1024 * 1024
        assert gemm_kernel.flops == expected_flops

    def test_kernel_memory_bytes(self, gemm_kernel):
        """Test kernel memory bytes calculation."""
        memory = gemm_kernel.memory_bytes
        # 2 inputs + 1 output, each 1024x1024 FP16
        assert memory > 0

    def test_kernel_attributes(self, gemm_kernel):
        """Test kernel attributes."""
        assert "m" in gemm_kernel.attributes
        assert "k" in gemm_kernel.attributes
        assert "n" in gemm_kernel.attributes

    def test_create_gemm_with_device(self):
        """Test creating GEMM kernel on specific device."""
        kernel = create_gemm_kernel(256, 256, 256, device_id=2)
        assert kernel.device_id == 2

    def test_create_attention_with_device(self):
        """Test creating attention kernel on specific device."""
        kernel = create_attention_kernel(
            batch=2, heads=16, seq_len=1024, head_dim=64, device_id=3
        )
        assert kernel.device_id == 3
        assert kernel.attributes["batch"] == 2
        assert kernel.attributes["heads"] == 16

    def test_reduce_output_shape(self):
        """Test reduce kernel output shape."""
        kernel = create_reduce_kernel(shape=(32, 64, 128), axis=-1)
        assert kernel.outputs[0].shape == (32, 64)

    def test_reduce_axis_0(self):
        """Test reduce kernel with axis 0."""
        kernel = create_reduce_kernel(shape=(32, 64, 128), axis=0)
        assert kernel.outputs[0].shape == (64, 128)


class TestKernelDependency:
    """Tests for KernelDependency class."""

    def test_dependency_creation(self):
        """Test dependency creation."""
        dep = KernelDependency(
            source_id="kernel_1",
            target_id="kernel_2",
            tensor_id="tensor_1",
        )
        assert dep.source_id == "kernel_1"
        assert dep.target_id == "kernel_2"
        assert dep.tensor_id == "tensor_1"
        assert dep.dependency_type == "data"

    def test_dependency_with_type(self):
        """Test dependency with explicit type."""
        dep = KernelDependency(
            source_id="k1",
            target_id="k2",
            tensor_id="t1",
            dependency_type="control",
        )
        assert dep.dependency_type == "control"


class TestComputeGraph:
    """Tests for ComputeGraph class."""

    def test_empty_graph(self, empty_graph):
        """Test empty graph creation."""
        assert len(empty_graph.kernels) == 0
        assert len(empty_graph.dependencies) == 0

    def test_add_kernel(self, empty_graph, gemm_kernel):
        """Test adding kernel to graph."""
        empty_graph.add_kernel(gemm_kernel)
        assert gemm_kernel.kernel_id in empty_graph.kernels
        assert empty_graph.kernels[gemm_kernel.kernel_id] == gemm_kernel

    def test_add_dependency(self, empty_graph, gemm_kernel, elementwise_kernel):
        """Test adding dependency to graph."""
        empty_graph.add_kernel(gemm_kernel)
        empty_graph.add_kernel(elementwise_kernel)
        empty_graph.add_dependency(
            gemm_kernel.kernel_id,
            elementwise_kernel.kernel_id,
            "output_tensor",
        )
        assert len(empty_graph.dependencies) == 1

    def test_get_dependencies(self, simple_graph):
        """Test getting dependencies of a kernel."""
        topo = simple_graph.topological_sort()

        # First kernel should have no dependencies
        deps = simple_graph.get_dependencies(topo[0])
        assert len(deps) == 0

        # Later kernels should have dependencies
        deps = simple_graph.get_dependencies(topo[1])
        assert len(deps) == 1

    def test_get_dependents(self, simple_graph):
        """Test getting dependents of a kernel."""
        topo = simple_graph.topological_sort()

        # First kernel should have dependents
        deps = simple_graph.get_dependents(topo[0])
        assert len(deps) == 1

        # Last kernel should have no dependents
        deps = simple_graph.get_dependents(topo[-1])
        assert len(deps) == 0

    def test_topological_sort(self, simple_graph):
        """Test topological sort."""
        topo = simple_graph.topological_sort()
        assert len(topo) == 3

        # Dependencies should be satisfied
        kernel_idx = {k: i for i, k in enumerate(topo)}
        for dep in simple_graph.dependencies:
            assert kernel_idx[dep.source_id] < kernel_idx[dep.target_id]

    def test_topological_sort_caching(self, simple_graph):
        """Test that topological sort is cached."""
        topo1 = simple_graph.topological_sort()
        topo2 = simple_graph.topological_sort()
        assert topo1 == topo2

    def test_topological_sort_cache_invalidation(self, simple_graph):
        """Test that adding kernel invalidates cache."""
        _ = simple_graph.topological_sort()
        new_kernel = create_gemm_kernel(128, 128, 128)
        simple_graph.add_kernel(new_kernel)
        # Cache should be invalidated
        assert simple_graph._topo_order is None

    def test_critical_path(self, simple_graph):
        """Test critical path calculation."""
        path, duration = simple_graph.get_critical_path()
        assert len(path) > 0
        assert duration > 0

    def test_critical_path_diamond(self, diamond_graph):
        """Test critical path in diamond graph."""
        path, duration = diamond_graph.get_critical_path()

        # Should include source and sink
        assert len(path) >= 2
        assert duration > 0

    def test_parallelizable_groups(self, parallel_graph):
        """Test parallelizable groups detection."""
        groups = parallel_graph.get_parallelizable_groups()

        # Should have multiple levels
        assert len(groups) >= 2

        # First level should have root kernel only
        assert len(groups[0]) == 1

        # Second level should have parallel branches
        assert len(groups[1]) == 2

    def test_parallelizable_groups_linear(self, simple_graph):
        """Test parallelizable groups in linear graph."""
        groups = simple_graph.get_parallelizable_groups()

        # Each kernel in its own level for linear graph
        assert len(groups) == 3
        for group in groups:
            assert len(group) == 1


class TestGPUDevice:
    """Tests for GPUDevice class."""

    def test_gpu_device_creation(self, single_gpu):
        """Test GPU device creation."""
        assert single_gpu.device_id == 0
        assert single_gpu.total_memory_gb == 80.0
        assert single_gpu.sm_count == 108

    def test_gpu_device_defaults(self):
        """Test GPU device default values."""
        gpu = GPUDevice(device_id=0)
        assert gpu.name == "GPU"
        assert gpu.compute_capability == (8, 0)
        assert gpu.warp_size == 32

    def test_gpu_peak_tflops(self, single_gpu):
        """Test peak TFLOPS estimation."""
        tflops = single_gpu.peak_tflops
        assert tflops > 0


class TestMultiGPUCluster:
    """Tests for MultiGPUCluster class."""

    def test_cluster_creation(self, dual_gpu_cluster):
        """Test cluster creation."""
        assert dual_gpu_cluster.num_devices == 2
        assert dual_gpu_cluster.nvlink_bandwidth_gbps == 600.0

    def test_cluster_total_memory(self, dual_gpu_cluster):
        """Test cluster total memory calculation."""
        assert dual_gpu_cluster.total_memory_gb == 160.0

    def test_cluster_get_device(self, dual_gpu_cluster):
        """Test getting device by ID."""
        device = dual_gpu_cluster.get_device(0)
        assert device.device_id == 0
        assert device.name == "GPU0"

    def test_cluster_get_device_invalid(self, dual_gpu_cluster):
        """Test getting invalid device raises error."""
        with pytest.raises(ValueError, match="Device 99 not found"):
            dual_gpu_cluster.get_device(99)

    def test_quad_cluster(self, quad_gpu_cluster):
        """Test 4-GPU cluster."""
        assert quad_gpu_cluster.num_devices == 4
        assert quad_gpu_cluster.total_memory_gb == 320.0


class TestCreateTestGraph:
    """Tests for test graph creation utility."""

    def test_create_test_graph(self, transformer_graph):
        """Test test graph creation."""
        assert len(transformer_graph.kernels) > 0
        assert len(transformer_graph.dependencies) > 0

    def test_create_test_graph_layers(self):
        """Test test graph with different layer counts."""
        graph1 = create_test_graph(num_layers=1)
        graph2 = create_test_graph(num_layers=4)

        assert len(graph2.kernels) > len(graph1.kernels)

    def test_test_graph_structure(self, transformer_graph):
        """Test test graph has valid structure."""
        # Should be topologically sortable
        topo = transformer_graph.topological_sort()
        assert len(topo) == len(transformer_graph.kernels)

        # Dependencies should reference valid kernels
        for dep in transformer_graph.dependencies:
            assert dep.source_id in transformer_graph.kernels
            assert dep.target_id in transformer_graph.kernels


class TestGraphOperations:
    """Tests for graph manipulation operations."""

    def test_graph_copy_independence(self, simple_graph):
        """Test that modifying copied graph doesn't affect original."""
        original_count = len(simple_graph.kernels)

        # Create a shallow copy of kernels
        copied_kernels = dict(simple_graph.kernels)

        # Add to copy
        new_kernel = create_gemm_kernel(64, 64, 64)
        copied_kernels[new_kernel.kernel_id] = new_kernel

        # Original should be unchanged
        assert len(simple_graph.kernels) == original_count

    def test_multi_path_dependencies(self):
        """Test graph with multiple dependency paths."""
        graph = ComputeGraph()

        a = create_gemm_kernel(256, 256, 256)
        b1 = create_elementwise_kernel((256, 256), op="relu")
        b2 = create_elementwise_kernel((256, 256), op="sigmoid")
        c = create_elementwise_kernel((256, 256), op="add")
        d = create_reduce_kernel((256, 256), axis=-1)

        for k in [a, b1, b2, c, d]:
            graph.add_kernel(k)

        # A -> B1 -> C -> D
        # A -> B2 -> C
        graph.add_dependency(a.kernel_id, b1.kernel_id, "a_b1")
        graph.add_dependency(a.kernel_id, b2.kernel_id, "a_b2")
        graph.add_dependency(b1.kernel_id, c.kernel_id, "b1_c")
        graph.add_dependency(b2.kernel_id, c.kernel_id, "b2_c")
        graph.add_dependency(c.kernel_id, d.kernel_id, "c_d")

        topo = graph.topological_sort()
        kernel_idx = {k: i for i, k in enumerate(topo)}

        # Verify order constraints
        assert kernel_idx[a.kernel_id] < kernel_idx[b1.kernel_id]
        assert kernel_idx[a.kernel_id] < kernel_idx[b2.kernel_id]
        assert kernel_idx[b1.kernel_id] < kernel_idx[c.kernel_id]
        assert kernel_idx[b2.kernel_id] < kernel_idx[c.kernel_id]
        assert kernel_idx[c.kernel_id] < kernel_idx[d.kernel_id]

    def test_disconnected_components(self):
        """Test graph with disconnected components."""
        graph = ComputeGraph()

        # Component 1
        a = create_gemm_kernel(128, 128, 128)
        b = create_elementwise_kernel((128, 128), op="relu")
        graph.add_kernel(a)
        graph.add_kernel(b)
        graph.add_dependency(a.kernel_id, b.kernel_id, "a_b")

        # Component 2 (disconnected)
        c = create_gemm_kernel(64, 64, 64)
        d = create_elementwise_kernel((64, 64), op="sigmoid")
        graph.add_kernel(c)
        graph.add_kernel(d)
        graph.add_dependency(c.kernel_id, d.kernel_id, "c_d")

        topo = graph.topological_sort()
        assert len(topo) == 4

        groups = graph.get_parallelizable_groups()
        # First level should have two independent roots
        assert len(groups[0]) == 2
