"""Pytest fixtures for Multi-GPU Kernel Scheduler tests."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernelsched import (
    # Core
    DataType, KernelType, DeviceType, TensorDescriptor, KernelConfig,
    KernelStats, Kernel, KernelDependency, ComputeGraph, GPUDevice,
    MultiGPUCluster, create_gemm_kernel, create_attention_kernel,
    create_elementwise_kernel, create_reduce_kernel, create_test_graph,
    PipelineStage, PipelineConfig, PipelinePartitioner,
    # Scheduler
    SchedulingPolicy, ScheduledKernel, Stream, DeviceSchedule, Schedule,
    KernelScheduler, FIFOScheduler, CriticalPathScheduler, LoadBalanceScheduler,
    StreamScheduler, MemoryScheduler, create_scheduler,
    MicrobatchSchedule, PipelineSchedule, PipelineScheduler,
    # Optimizer
    OptimizationPass, OptimizationResult, GraphOptimizer, KernelFuser,
    MemoryOptimizer, ConstantFolder, DeadCodeEliminator, OptimizationPipeline,
    create_default_pipeline, optimize_graph,
    # Executor
    ExecutionMode, MemoryBlock, ExecutionStats, DeviceMemoryManager,
    ClusterMemoryManager, KernelExecutor, GraphExecutor, StreamExecutor,
    CUDAGraphCapture, Profiler, MultiGPUExecutionEngine, execute_graph,
)


# =============================================================================
# Device Fixtures
# =============================================================================

@pytest.fixture
def single_gpu() -> GPUDevice:
    """Create a single GPU device."""
    return GPUDevice(
        device_id=0,
        name="Test GPU",
        compute_capability=(8, 0),
        total_memory_gb=80.0,
        memory_bandwidth_gbps=2000.0,
        sm_count=108,
    )


@pytest.fixture
def dual_gpu_cluster() -> MultiGPUCluster:
    """Create a 2-GPU cluster."""
    devices = [
        GPUDevice(device_id=0, name="GPU0", total_memory_gb=80.0),
        GPUDevice(device_id=1, name="GPU1", total_memory_gb=80.0),
    ]
    return MultiGPUCluster(
        devices=devices,
        nvlink_bandwidth_gbps=600.0,
        pcie_bandwidth_gbps=32.0,
    )


@pytest.fixture
def quad_gpu_cluster() -> MultiGPUCluster:
    """Create a 4-GPU cluster."""
    devices = [
        GPUDevice(device_id=i, name=f"GPU{i}", total_memory_gb=80.0)
        for i in range(4)
    ]
    return MultiGPUCluster(
        devices=devices,
        nvlink_bandwidth_gbps=600.0,
        pcie_bandwidth_gbps=32.0,
    )


# =============================================================================
# Tensor Fixtures
# =============================================================================

@pytest.fixture
def small_tensor() -> TensorDescriptor:
    """Create a small tensor descriptor."""
    return TensorDescriptor(
        tensor_id="small_tensor",
        shape=(16, 16),
        dtype=DataType.FLOAT16,
        device_id=0,
    )


@pytest.fixture
def medium_tensor() -> TensorDescriptor:
    """Create a medium tensor descriptor."""
    return TensorDescriptor(
        tensor_id="medium_tensor",
        shape=(1024, 1024),
        dtype=DataType.FLOAT16,
        device_id=0,
    )


@pytest.fixture
def large_tensor() -> TensorDescriptor:
    """Create a large tensor descriptor."""
    return TensorDescriptor(
        tensor_id="large_tensor",
        shape=(4096, 4096),
        dtype=DataType.FLOAT32,
        device_id=0,
    )


# =============================================================================
# Kernel Fixtures
# =============================================================================

@pytest.fixture
def gemm_kernel() -> Kernel:
    """Create a GEMM kernel."""
    return create_gemm_kernel(m=1024, k=1024, n=1024)


@pytest.fixture
def attention_kernel() -> Kernel:
    """Create an attention kernel."""
    return create_attention_kernel(batch=1, heads=8, seq_len=512, head_dim=64)


@pytest.fixture
def elementwise_kernel() -> Kernel:
    """Create an elementwise kernel."""
    return create_elementwise_kernel(shape=(1024, 1024), op="add")


@pytest.fixture
def reduce_kernel() -> Kernel:
    """Create a reduce kernel."""
    return create_reduce_kernel(shape=(1024, 1024), axis=-1)


# =============================================================================
# Graph Fixtures
# =============================================================================

@pytest.fixture
def empty_graph() -> ComputeGraph:
    """Create an empty compute graph."""
    return ComputeGraph()


@pytest.fixture
def simple_graph() -> ComputeGraph:
    """Create a simple linear graph with 3 kernels."""
    graph = ComputeGraph()

    k1 = create_gemm_kernel(512, 512, 512, device_id=0)
    k2 = create_elementwise_kernel((512, 512), op="relu", device_id=0)
    k3 = create_gemm_kernel(512, 512, 512, device_id=0)

    graph.add_kernel(k1)
    graph.add_kernel(k2)
    graph.add_kernel(k3)

    graph.add_dependency(k1.kernel_id, k2.kernel_id, "dep_1_2")
    graph.add_dependency(k2.kernel_id, k3.kernel_id, "dep_2_3")

    return graph


@pytest.fixture
def parallel_graph() -> ComputeGraph:
    """Create a graph with parallel branches."""
    graph = ComputeGraph()

    # Root kernel
    root = create_gemm_kernel(512, 512, 512, device_id=0)
    graph.add_kernel(root)

    # Two parallel branches
    branch1 = create_gemm_kernel(512, 512, 512, device_id=0)
    branch2 = create_gemm_kernel(512, 512, 512, device_id=0)
    graph.add_kernel(branch1)
    graph.add_kernel(branch2)

    graph.add_dependency(root.kernel_id, branch1.kernel_id, "root_b1")
    graph.add_dependency(root.kernel_id, branch2.kernel_id, "root_b2")

    # Merge kernel
    merge = create_elementwise_kernel((512, 512), op="add", device_id=0)
    graph.add_kernel(merge)

    graph.add_dependency(branch1.kernel_id, merge.kernel_id, "b1_merge")
    graph.add_dependency(branch2.kernel_id, merge.kernel_id, "b2_merge")

    return graph


@pytest.fixture
def diamond_graph() -> ComputeGraph:
    """Create a diamond-shaped graph (A -> B,C -> D)."""
    graph = ComputeGraph()

    a = create_gemm_kernel(256, 256, 256, device_id=0)
    b = create_elementwise_kernel((256, 256), op="relu", device_id=0)
    c = create_elementwise_kernel((256, 256), op="sigmoid", device_id=0)
    d = create_elementwise_kernel((256, 256), op="add", device_id=0)

    graph.add_kernel(a)
    graph.add_kernel(b)
    graph.add_kernel(c)
    graph.add_kernel(d)

    graph.add_dependency(a.kernel_id, b.kernel_id, "a_b")
    graph.add_dependency(a.kernel_id, c.kernel_id, "a_c")
    graph.add_dependency(b.kernel_id, d.kernel_id, "b_d")
    graph.add_dependency(c.kernel_id, d.kernel_id, "c_d")

    return graph


@pytest.fixture
def transformer_graph() -> ComputeGraph:
    """Create a small transformer-like graph."""
    return create_test_graph(num_layers=2)


@pytest.fixture
def multi_device_graph(dual_gpu_cluster) -> ComputeGraph:
    """Create a graph distributed across multiple devices."""
    graph = ComputeGraph()

    # Kernels on device 0
    k0_1 = create_gemm_kernel(512, 512, 512, device_id=0)
    k0_2 = create_elementwise_kernel((512, 512), op="relu", device_id=0)

    # Kernels on device 1
    k1_1 = create_gemm_kernel(512, 512, 512, device_id=1)
    k1_2 = create_elementwise_kernel((512, 512), op="relu", device_id=1)

    graph.add_kernel(k0_1)
    graph.add_kernel(k0_2)
    graph.add_kernel(k1_1)
    graph.add_kernel(k1_2)

    # Dependencies with cross-device communication
    graph.add_dependency(k0_1.kernel_id, k0_2.kernel_id, "local_0")
    graph.add_dependency(k1_1.kernel_id, k1_2.kernel_id, "local_1")
    graph.add_dependency(k0_2.kernel_id, k1_2.kernel_id, "cross_device")

    return graph


# =============================================================================
# Scheduler Fixtures
# =============================================================================

@pytest.fixture
def fifo_scheduler(dual_gpu_cluster) -> FIFOScheduler:
    """Create a FIFO scheduler."""
    return FIFOScheduler(cluster=dual_gpu_cluster, num_streams_per_device=4)


@pytest.fixture
def critical_path_scheduler(dual_gpu_cluster) -> CriticalPathScheduler:
    """Create a critical path scheduler."""
    return CriticalPathScheduler(cluster=dual_gpu_cluster, num_streams_per_device=4)


@pytest.fixture
def load_balance_scheduler(dual_gpu_cluster) -> LoadBalanceScheduler:
    """Create a load balance scheduler."""
    return LoadBalanceScheduler(cluster=dual_gpu_cluster, num_streams_per_device=4)


@pytest.fixture
def stream_scheduler() -> StreamScheduler:
    """Create a stream scheduler."""
    return StreamScheduler(num_streams=4)


@pytest.fixture
def memory_scheduler(dual_gpu_cluster) -> MemoryScheduler:
    """Create a memory scheduler."""
    return MemoryScheduler(cluster=dual_gpu_cluster)


# =============================================================================
# Optimizer Fixtures
# =============================================================================

@pytest.fixture
def kernel_fuser() -> KernelFuser:
    """Create a kernel fuser optimizer."""
    return KernelFuser()


@pytest.fixture
def memory_optimizer() -> MemoryOptimizer:
    """Create a memory optimizer."""
    return MemoryOptimizer(max_memory_mb=16000)


@pytest.fixture
def constant_folder() -> ConstantFolder:
    """Create a constant folder optimizer."""
    return ConstantFolder()


@pytest.fixture
def dead_code_eliminator() -> DeadCodeEliminator:
    """Create a dead code eliminator optimizer."""
    return DeadCodeEliminator()


@pytest.fixture
def optimization_pipeline() -> OptimizationPipeline:
    """Create a default optimization pipeline."""
    return create_default_pipeline()


# =============================================================================
# Executor Fixtures
# =============================================================================

@pytest.fixture
def device_memory_manager() -> DeviceMemoryManager:
    """Create a device memory manager."""
    return DeviceMemoryManager(device_id=0, total_memory_mb=80000)


@pytest.fixture
def cluster_memory_manager(dual_gpu_cluster) -> ClusterMemoryManager:
    """Create a cluster memory manager."""
    return ClusterMemoryManager(cluster=dual_gpu_cluster)


@pytest.fixture
def kernel_executor() -> KernelExecutor:
    """Create a kernel executor."""
    return KernelExecutor()


@pytest.fixture
def graph_executor(dual_gpu_cluster) -> GraphExecutor:
    """Create a graph executor."""
    return GraphExecutor(cluster=dual_gpu_cluster)


@pytest.fixture
def stream_executor() -> StreamExecutor:
    """Create a stream executor."""
    return StreamExecutor(num_streams=4)


@pytest.fixture
def profiler() -> Profiler:
    """Create a profiler."""
    return Profiler()


@pytest.fixture
def execution_engine(dual_gpu_cluster) -> MultiGPUExecutionEngine:
    """Create a multi-GPU execution engine."""
    return MultiGPUExecutionEngine(cluster=dual_gpu_cluster)


# =============================================================================
# Helper Fixtures
# =============================================================================

@pytest.fixture
def scheduled_simple_graph(simple_graph, fifo_scheduler) -> tuple[ComputeGraph, Schedule]:
    """Create a scheduled simple graph."""
    schedule = fifo_scheduler.schedule(simple_graph)
    return simple_graph, schedule


@pytest.fixture
def fusable_graph() -> ComputeGraph:
    """Create a graph with fusable kernel patterns."""
    graph = ComputeGraph()

    # GEMM -> Bias Add -> Activation pattern
    gemm = create_gemm_kernel(512, 512, 512)
    bias = create_elementwise_kernel((512, 512), op="add")
    bias.attributes["op"] = "add"  # Bias add
    relu = create_elementwise_kernel((512, 512), op="relu")

    graph.add_kernel(gemm)
    graph.add_kernel(bias)
    graph.add_kernel(relu)

    graph.add_dependency(gemm.kernel_id, bias.kernel_id, "gemm_bias")
    graph.add_dependency(bias.kernel_id, relu.kernel_id, "bias_relu")

    return graph


@pytest.fixture
def elementwise_chain_graph() -> ComputeGraph:
    """Create a graph with chain of elementwise operations."""
    graph = ComputeGraph()

    # Chain of 4 elementwise ops
    ops = ["add", "mul", "sub", "add"]
    kernels = []

    for i, op in enumerate(ops):
        kernel = create_elementwise_kernel((256, 256), op=op)
        graph.add_kernel(kernel)
        kernels.append(kernel)

        if i > 0:
            graph.add_dependency(
                kernels[i-1].kernel_id,
                kernels[i].kernel_id,
                f"dep_{i-1}_{i}"
            )

    return graph
