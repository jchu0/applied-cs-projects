"""Core kernel and graph definitions for multi-GPU scheduling."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import uuid
import numpy as np


class DataType(Enum):
    """Tensor data types."""
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT32 = "int32"
    INT8 = "int8"


class KernelType(Enum):
    """Types of compute kernels."""
    GEMM = "gemm"  # Matrix multiplication
    CONV = "conv"  # Convolution
    ELEMENTWISE = "elementwise"
    REDUCE = "reduce"
    SOFTMAX = "softmax"
    ATTENTION = "attention"
    LAYERNORM = "layernorm"
    TRANSPOSE = "transpose"
    MEMORY = "memory"  # Memory operations
    CUSTOM = "custom"


class DeviceType(Enum):
    """Device types."""
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class TensorDescriptor:
    """Description of a tensor."""
    tensor_id: str
    shape: tuple[int, ...]
    dtype: DataType
    device_id: int = 0
    stride: tuple[int, ...] | None = None
    memory_offset: int = 0

    @property
    def size_bytes(self) -> int:
        """Get size in bytes."""
        dtype_sizes = {
            DataType.FLOAT32: 4,
            DataType.FLOAT16: 2,
            DataType.BFLOAT16: 2,
            DataType.INT32: 4,
            DataType.INT8: 1,
        }
        return int(np.prod(self.shape)) * dtype_sizes.get(self.dtype, 4)

    @property
    def numel(self) -> int:
        """Get number of elements."""
        return int(np.prod(self.shape))


@dataclass
class KernelConfig:
    """Kernel launch configuration."""
    block_size: tuple[int, int, int] = (256, 1, 1)
    grid_size: tuple[int, int, int] = (1, 1, 1)
    shared_memory_bytes: int = 0
    stream_id: int = 0


@dataclass
class KernelStats:
    """Statistics for a kernel execution."""
    compute_time_us: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    flops: int = 0
    memory_read_bytes: int = 0
    memory_write_bytes: int = 0


@dataclass
class Kernel:
    """Represents a GPU kernel."""
    kernel_id: str
    name: str
    kernel_type: KernelType
    inputs: list[TensorDescriptor]
    outputs: list[TensorDescriptor]
    config: KernelConfig = field(default_factory=KernelConfig)
    attributes: dict[str, Any] = field(default_factory=dict)
    device_id: int = 0
    estimated_time_us: float = 0.0
    stats: KernelStats | None = None

    @property
    def flops(self) -> int:
        """Estimate FLOPs for kernel."""
        if self.kernel_type == KernelType.GEMM:
            # M x K @ K x N -> M x N with 2*M*K*N FLOPs
            if len(self.inputs) >= 2:
                m = self.inputs[0].shape[0] if len(self.inputs[0].shape) > 0 else 1
                k = self.inputs[0].shape[1] if len(self.inputs[0].shape) > 1 else 1
                n = self.inputs[1].shape[1] if len(self.inputs[1].shape) > 1 else 1
                return 2 * m * k * n
        return 0

    @property
    def memory_bytes(self) -> int:
        """Get total memory accessed."""
        total = sum(inp.size_bytes for inp in self.inputs)
        total += sum(out.size_bytes for out in self.outputs)
        return total


@dataclass
class KernelDependency:
    """Dependency between kernels."""
    source_id: str
    target_id: str
    tensor_id: str  # Tensor flowing from source to target
    dependency_type: str = "data"  # data, control


class ComputeGraph:
    """Computational graph of kernels."""

    def __init__(self):
        self.kernels: dict[str, Kernel] = {}
        self.dependencies: list[KernelDependency] = []
        self.input_tensors: list[TensorDescriptor] = []
        self.output_tensors: list[TensorDescriptor] = []
        self._topo_order: list[str] | None = None

    def add_kernel(self, kernel: Kernel) -> None:
        """Add kernel to graph."""
        self.kernels[kernel.kernel_id] = kernel
        self._topo_order = None

    def add_dependency(
        self,
        source_id: str,
        target_id: str,
        tensor_id: str
    ) -> None:
        """Add dependency between kernels."""
        self.dependencies.append(KernelDependency(
            source_id=source_id,
            target_id=target_id,
            tensor_id=tensor_id
        ))
        self._topo_order = None

    def get_dependencies(self, kernel_id: str) -> list[str]:
        """Get kernel IDs that this kernel depends on."""
        return [d.source_id for d in self.dependencies if d.target_id == kernel_id]

    def get_dependents(self, kernel_id: str) -> list[str]:
        """Get kernel IDs that depend on this kernel."""
        return [d.target_id for d in self.dependencies if d.source_id == kernel_id]

    def topological_sort(self) -> list[str]:
        """Get kernels in topological order."""
        if self._topo_order:
            return self._topo_order

        in_degree: dict[str, int] = {k: 0 for k in self.kernels}
        for dep in self.dependencies:
            if dep.target_id in in_degree:
                in_degree[dep.target_id] += 1

        queue = [k for k, d in in_degree.items() if d == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for dep in self.dependencies:
                if dep.source_id == current:
                    in_degree[dep.target_id] -= 1
                    if in_degree[dep.target_id] == 0:
                        queue.append(dep.target_id)

        self._topo_order = result
        return result

    def get_critical_path(self) -> tuple[list[str], float]:
        """Get critical path and its duration."""
        topo_order = self.topological_sort()

        # Dynamic programming for longest path
        dist: dict[str, float] = {k: 0 for k in self.kernels}
        pred: dict[str, str | None] = {k: None for k in self.kernels}

        for kernel_id in topo_order:
            kernel = self.kernels[kernel_id]
            for dep_id in self.get_dependents(kernel_id):
                new_dist = dist[kernel_id] + kernel.estimated_time_us
                if new_dist > dist[dep_id]:
                    dist[dep_id] = new_dist
                    pred[dep_id] = kernel_id

        # Find endpoint with maximum distance
        end_node = max(dist, key=lambda k: dist[k] + self.kernels[k].estimated_time_us)
        total_time = dist[end_node] + self.kernels[end_node].estimated_time_us

        # Reconstruct path
        path = [end_node]
        while pred[path[-1]]:
            path.append(pred[path[-1]])
        path.reverse()

        return path, total_time

    def get_parallelizable_groups(self) -> list[list[str]]:
        """Get groups of kernels that can run in parallel."""
        topo_order = self.topological_sort()
        levels: list[list[str]] = []
        kernel_level: dict[str, int] = {}

        for kernel_id in topo_order:
            deps = self.get_dependencies(kernel_id)
            if not deps:
                level = 0
            else:
                level = max(kernel_level[d] for d in deps) + 1

            kernel_level[kernel_id] = level

            while len(levels) <= level:
                levels.append([])
            levels[level].append(kernel_id)

        return levels


@dataclass
class GPUDevice:
    """GPU device representation."""
    device_id: int
    name: str = "GPU"
    compute_capability: tuple[int, int] = (8, 0)
    total_memory_gb: float = 80.0
    memory_bandwidth_gbps: float = 2000.0
    sm_count: int = 108
    max_threads_per_sm: int = 1536
    warp_size: int = 32

    @property
    def peak_tflops(self) -> float:
        """Estimate peak TFLOPS."""
        # Simplified estimate
        return self.sm_count * 128 * 1.5 / 1000  # ~20 TFLOPS for A100


@dataclass
class MultiGPUCluster:
    """Cluster of multiple GPUs."""
    devices: list[GPUDevice]
    nvlink_bandwidth_gbps: float = 600.0  # NVLink bandwidth
    pcie_bandwidth_gbps: float = 32.0  # PCIe bandwidth

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    @property
    def total_memory_gb(self) -> float:
        return sum(d.total_memory_gb for d in self.devices)

    def get_device(self, device_id: int) -> GPUDevice:
        """Get device by ID."""
        for d in self.devices:
            if d.device_id == device_id:
                return d
        raise ValueError(f"Device {device_id} not found")


def create_gemm_kernel(
    m: int, k: int, n: int,
    dtype: DataType = DataType.FLOAT16,
    device_id: int = 0
) -> Kernel:
    """Create a GEMM kernel."""
    kernel_id = str(uuid.uuid4())[:8]

    input_a = TensorDescriptor(
        tensor_id=f"{kernel_id}_a",
        shape=(m, k),
        dtype=dtype,
        device_id=device_id
    )
    input_b = TensorDescriptor(
        tensor_id=f"{kernel_id}_b",
        shape=(k, n),
        dtype=dtype,
        device_id=device_id
    )
    output_c = TensorDescriptor(
        tensor_id=f"{kernel_id}_c",
        shape=(m, n),
        dtype=dtype,
        device_id=device_id
    )

    # Estimate time based on FLOPs
    flops = 2 * m * k * n
    tflops = 100  # Assume 100 TFLOPS effective
    estimated_time_us = flops / (tflops * 1e6)

    return Kernel(
        kernel_id=kernel_id,
        name=f"gemm_{m}x{k}x{n}",
        kernel_type=KernelType.GEMM,
        inputs=[input_a, input_b],
        outputs=[output_c],
        device_id=device_id,
        estimated_time_us=estimated_time_us,
        attributes={"m": m, "k": k, "n": n}
    )


def create_attention_kernel(
    batch: int, heads: int, seq_len: int, head_dim: int,
    dtype: DataType = DataType.FLOAT16,
    device_id: int = 0
) -> Kernel:
    """Create an attention kernel."""
    kernel_id = str(uuid.uuid4())[:8]

    q = TensorDescriptor(
        tensor_id=f"{kernel_id}_q",
        shape=(batch, heads, seq_len, head_dim),
        dtype=dtype,
        device_id=device_id
    )
    k = TensorDescriptor(
        tensor_id=f"{kernel_id}_k",
        shape=(batch, heads, seq_len, head_dim),
        dtype=dtype,
        device_id=device_id
    )
    v = TensorDescriptor(
        tensor_id=f"{kernel_id}_v",
        shape=(batch, heads, seq_len, head_dim),
        dtype=dtype,
        device_id=device_id
    )
    output = TensorDescriptor(
        tensor_id=f"{kernel_id}_out",
        shape=(batch, heads, seq_len, head_dim),
        dtype=dtype,
        device_id=device_id
    )

    # Attention FLOPs: 2*B*H*S*S*D + 2*B*H*S*S*D
    flops = 4 * batch * heads * seq_len * seq_len * head_dim
    tflops = 50  # Lower due to memory bound
    estimated_time_us = flops / (tflops * 1e6)

    return Kernel(
        kernel_id=kernel_id,
        name=f"attention_{batch}x{heads}x{seq_len}",
        kernel_type=KernelType.ATTENTION,
        inputs=[q, k, v],
        outputs=[output],
        device_id=device_id,
        estimated_time_us=estimated_time_us,
        attributes={"batch": batch, "heads": heads, "seq_len": seq_len, "head_dim": head_dim}
    )


def create_elementwise_kernel(
    shape: tuple[int, ...],
    op: str = "add",
    dtype: DataType = DataType.FLOAT16,
    device_id: int = 0
) -> Kernel:
    """Create an elementwise kernel."""
    kernel_id = str(uuid.uuid4())[:8]

    input_a = TensorDescriptor(
        tensor_id=f"{kernel_id}_a",
        shape=shape,
        dtype=dtype,
        device_id=device_id
    )
    input_b = TensorDescriptor(
        tensor_id=f"{kernel_id}_b",
        shape=shape,
        dtype=dtype,
        device_id=device_id
    )
    output = TensorDescriptor(
        tensor_id=f"{kernel_id}_out",
        shape=shape,
        dtype=dtype,
        device_id=device_id
    )

    numel = int(np.prod(shape))
    # Memory bound: estimate based on bandwidth
    memory_bytes = 3 * numel * 2  # 2 inputs + 1 output, FP16
    bandwidth_gbps = 2000
    estimated_time_us = memory_bytes / (bandwidth_gbps * 1e3)

    return Kernel(
        kernel_id=kernel_id,
        name=f"elementwise_{op}",
        kernel_type=KernelType.ELEMENTWISE,
        inputs=[input_a, input_b],
        outputs=[output],
        device_id=device_id,
        estimated_time_us=estimated_time_us,
        attributes={"op": op}
    )


def create_reduce_kernel(
    shape: tuple[int, ...],
    axis: int = -1,
    dtype: DataType = DataType.FLOAT16,
    device_id: int = 0
) -> Kernel:
    """Create a reduce kernel."""
    kernel_id = str(uuid.uuid4())[:8]

    input_tensor = TensorDescriptor(
        tensor_id=f"{kernel_id}_in",
        shape=shape,
        dtype=dtype,
        device_id=device_id
    )

    output_shape = list(shape)
    if axis < 0:
        axis = len(shape) + axis
    output_shape.pop(axis)
    output_shape = tuple(output_shape) if output_shape else (1,)

    output = TensorDescriptor(
        tensor_id=f"{kernel_id}_out",
        shape=output_shape,
        dtype=dtype,
        device_id=device_id
    )

    numel = int(np.prod(shape))
    estimated_time_us = numel / (1000 * 1e6)  # Very rough

    return Kernel(
        kernel_id=kernel_id,
        name="reduce_sum",
        kernel_type=KernelType.REDUCE,
        inputs=[input_tensor],
        outputs=[output],
        device_id=device_id,
        estimated_time_us=estimated_time_us,
        attributes={"axis": axis}
    )


@dataclass
class PipelineStage:
    """Represents a pipeline stage for pipeline parallelism."""
    stage_id: int
    device_id: int
    kernel_ids: list[str] = field(default_factory=list)
    input_tensors: list[TensorDescriptor] = field(default_factory=list)
    output_tensors: list[TensorDescriptor] = field(default_factory=list)

    @property
    def num_kernels(self) -> int:
        """Get number of kernels in this stage."""
        return len(self.kernel_ids)


@dataclass
class PipelineConfig:
    """Configuration for pipeline parallelism."""
    num_stages: int
    num_microbatches: int = 4
    interleave_stages: bool = False  # 1F1B vs interleaved
    recompute_activations: bool = False  # Gradient checkpointing


class PipelinePartitioner:
    """Partition a compute graph into pipeline stages."""

    def __init__(self, num_stages: int, num_devices: int):
        self.num_stages = num_stages
        self.num_devices = num_devices

    def partition(
        self,
        graph: ComputeGraph,
        strategy: str = "balanced"
    ) -> list[PipelineStage]:
        """
        Partition graph into pipeline stages.

        Args:
            graph: The compute graph to partition
            strategy: Partitioning strategy:
                - "balanced": Equal compute per stage
                - "memory": Balance memory per stage
                - "layer": Split by layer boundaries

        Returns:
            List of PipelineStage objects
        """
        topo_order = graph.topological_sort()

        if strategy == "balanced":
            return self._partition_balanced(graph, topo_order)
        elif strategy == "memory":
            return self._partition_memory(graph, topo_order)
        elif strategy == "layer":
            return self._partition_layer(graph, topo_order)
        else:
            return self._partition_balanced(graph, topo_order)

    def _partition_balanced(
        self,
        graph: ComputeGraph,
        topo_order: list[str]
    ) -> list[PipelineStage]:
        """Partition with balanced compute across stages."""
        # Calculate total compute
        total_compute = sum(
            graph.kernels[kid].estimated_time_us for kid in topo_order
        )
        compute_per_stage = total_compute / self.num_stages

        stages = []
        current_stage_kernels = []
        current_compute = 0.0
        stage_id = 0

        for kernel_id in topo_order:
            kernel = graph.kernels[kernel_id]
            current_stage_kernels.append(kernel_id)
            current_compute += kernel.estimated_time_us

            # Check if we should start a new stage
            if current_compute >= compute_per_stage and stage_id < self.num_stages - 1:
                stage = PipelineStage(
                    stage_id=stage_id,
                    device_id=stage_id % self.num_devices,
                    kernel_ids=current_stage_kernels.copy()
                )
                stages.append(stage)
                current_stage_kernels = []
                current_compute = 0.0
                stage_id += 1

        # Add remaining kernels to last stage
        if current_stage_kernels:
            stage = PipelineStage(
                stage_id=stage_id,
                device_id=stage_id % self.num_devices,
                kernel_ids=current_stage_kernels
            )
            stages.append(stage)

        # Compute input/output tensors for each stage
        self._compute_stage_boundaries(graph, stages)

        return stages

    def _partition_memory(
        self,
        graph: ComputeGraph,
        topo_order: list[str]
    ) -> list[PipelineStage]:
        """Partition with balanced memory across stages."""
        # Calculate total memory
        total_memory = sum(
            graph.kernels[kid].memory_bytes for kid in topo_order
        )
        memory_per_stage = total_memory / self.num_stages

        stages = []
        current_stage_kernels = []
        current_memory = 0
        stage_id = 0

        for kernel_id in topo_order:
            kernel = graph.kernels[kernel_id]
            current_stage_kernels.append(kernel_id)
            current_memory += kernel.memory_bytes

            if current_memory >= memory_per_stage and stage_id < self.num_stages - 1:
                stage = PipelineStage(
                    stage_id=stage_id,
                    device_id=stage_id % self.num_devices,
                    kernel_ids=current_stage_kernels.copy()
                )
                stages.append(stage)
                current_stage_kernels = []
                current_memory = 0
                stage_id += 1

        if current_stage_kernels:
            stage = PipelineStage(
                stage_id=stage_id,
                device_id=stage_id % self.num_devices,
                kernel_ids=current_stage_kernels
            )
            stages.append(stage)

        self._compute_stage_boundaries(graph, stages)
        return stages

    def _partition_layer(
        self,
        graph: ComputeGraph,
        topo_order: list[str]
    ) -> list[PipelineStage]:
        """Partition by natural layer boundaries (attention -> FFN)."""
        # Group by layer based on kernel names
        layers: list[list[str]] = [[]]

        for kernel_id in topo_order:
            kernel = graph.kernels[kernel_id]
            layers[-1].append(kernel_id)

            # Start new layer after attention
            if kernel.kernel_type == KernelType.ATTENTION and len(layers) < self.num_stages:
                layers.append([])

        # Remove empty layers
        layers = [layer for layer in layers if layer]

        # Merge layers to match stage count
        while len(layers) > self.num_stages:
            # Merge two smallest adjacent layers
            min_idx = 0
            min_size = float('inf')
            for i in range(len(layers) - 1):
                size = len(layers[i]) + len(layers[i + 1])
                if size < min_size:
                    min_size = size
                    min_idx = i
            layers[min_idx].extend(layers[min_idx + 1])
            layers.pop(min_idx + 1)

        stages = []
        for stage_id, layer_kernels in enumerate(layers):
            stage = PipelineStage(
                stage_id=stage_id,
                device_id=stage_id % self.num_devices,
                kernel_ids=layer_kernels
            )
            stages.append(stage)

        self._compute_stage_boundaries(graph, stages)
        return stages

    def _compute_stage_boundaries(
        self,
        graph: ComputeGraph,
        stages: list[PipelineStage]
    ) -> None:
        """Compute input/output tensors for each stage."""
        # Build stage membership
        kernel_to_stage: dict[str, int] = {}
        for stage in stages:
            for kid in stage.kernel_ids:
                kernel_to_stage[kid] = stage.stage_id

        # Compute boundaries
        for stage in stages:
            stage_kernel_set = set(stage.kernel_ids)
            input_tensors = []
            output_tensors = []

            for kernel_id in stage.kernel_ids:
                kernel = graph.kernels[kernel_id]

                # Check dependencies - if source is in different stage, it's an input
                for dep_id in graph.get_dependencies(kernel_id):
                    if kernel_to_stage.get(dep_id, -1) != stage.stage_id:
                        source_kernel = graph.kernels[dep_id]
                        for out in source_kernel.outputs:
                            if out not in input_tensors:
                                input_tensors.append(out)

                # Check dependents - if target is in different stage, it's an output
                for dep_id in graph.get_dependents(kernel_id):
                    if kernel_to_stage.get(dep_id, -1) != stage.stage_id:
                        for out in kernel.outputs:
                            if out not in output_tensors:
                                output_tensors.append(out)

            stage.input_tensors = input_tensors
            stage.output_tensors = output_tensors


def create_test_graph(num_layers: int = 4) -> ComputeGraph:
    """Create a test compute graph (transformer-like)."""
    graph = ComputeGraph()

    batch, seq, hidden = 1, 512, 2048
    heads, head_dim = 16, 128

    prev_output = None

    for i in range(num_layers):
        # Attention
        attn = create_attention_kernel(batch, heads, seq, head_dim)
        graph.add_kernel(attn)

        if prev_output:
            graph.add_dependency(prev_output, attn.kernel_id, f"dep_{i}_attn")

        # FFN: GEMM + GELU + GEMM
        ffn1 = create_gemm_kernel(seq, hidden, hidden * 4)
        graph.add_kernel(ffn1)
        graph.add_dependency(attn.kernel_id, ffn1.kernel_id, f"attn_ffn_{i}")

        act = create_elementwise_kernel((batch, seq, hidden * 4), "gelu")
        graph.add_kernel(act)
        graph.add_dependency(ffn1.kernel_id, act.kernel_id, f"ffn1_act_{i}")

        ffn2 = create_gemm_kernel(seq, hidden * 4, hidden)
        graph.add_kernel(ffn2)
        graph.add_dependency(act.kernel_id, ffn2.kernel_id, f"act_ffn2_{i}")

        prev_output = ffn2.kernel_id

    return graph
