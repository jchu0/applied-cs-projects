"""Multi-GPU kernel execution engine."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
import time
import numpy as np
from collections import defaultdict

from ..core.kernel import (
    Kernel, ComputeGraph, MultiGPUCluster, GPUDevice,
    TensorDescriptor, KernelStats, DataType
)
from ..scheduler.scheduler import Schedule, ScheduledKernel


class ExecutionMode(Enum):
    """Execution modes."""
    EAGER = "eager"  # Execute immediately
    GRAPH = "graph"  # Graph capture and replay


@dataclass
class MemoryBlock:
    """Memory allocation block."""
    offset: int
    size: int
    tensor_id: str
    device_id: int
    is_free: bool = False


@dataclass
class ExecutionStats:
    """Statistics from execution."""
    total_time_ms: float
    kernel_times_ms: dict[str, float]
    memory_peak_mb: float
    device_utilization: dict[int, float]
    memory_transfers_ms: float


class DeviceMemoryManager:
    """Manage memory on a single device."""

    def __init__(self, device_id: int, total_memory_mb: float = 80000):
        self.device_id = device_id
        self.total_memory = int(total_memory_mb * 1024 * 1024)
        self.allocations: dict[str, MemoryBlock] = {}
        self.peak_usage = 0
        self.current_usage = 0

    def allocate(self, tensor: TensorDescriptor) -> MemoryBlock:
        """Allocate memory for tensor."""
        size = tensor.size_bytes

        if self.current_usage + size > self.total_memory:
            raise MemoryError(f"Out of memory on device {self.device_id}")

        block = MemoryBlock(
            offset=self.current_usage,
            size=size,
            tensor_id=tensor.tensor_id,
            device_id=self.device_id
        )

        self.allocations[tensor.tensor_id] = block
        self.current_usage += size
        self.peak_usage = max(self.peak_usage, self.current_usage)

        return block

    def free(self, tensor_id: str) -> None:
        """Free memory for tensor."""
        if tensor_id in self.allocations:
            block = self.allocations[tensor_id]
            block.is_free = True
            self.current_usage -= block.size
            del self.allocations[tensor_id]

    def get_stats(self) -> dict[str, float]:
        """Get memory statistics."""
        return {
            "current_mb": self.current_usage / (1024 * 1024),
            "peak_mb": self.peak_usage / (1024 * 1024),
            "utilization": self.current_usage / self.total_memory
        }

    def reset(self) -> None:
        """Reset memory manager."""
        self.allocations.clear()
        self.current_usage = 0


class ClusterMemoryManager:
    """Manage memory across cluster."""

    def __init__(self, cluster: MultiGPUCluster):
        self.cluster = cluster
        self.device_managers: dict[int, DeviceMemoryManager] = {}

        for device in cluster.devices:
            self.device_managers[device.device_id] = DeviceMemoryManager(
                device.device_id,
                device.total_memory_gb * 1024
            )

    def allocate(self, tensor: TensorDescriptor) -> MemoryBlock:
        """Allocate on specific device."""
        manager = self.device_managers.get(tensor.device_id)
        if not manager:
            manager = self.device_managers[0]
        return manager.allocate(tensor)

    def free(self, tensor_id: str, device_id: int) -> None:
        """Free tensor from device."""
        if device_id in self.device_managers:
            self.device_managers[device_id].free(tensor_id)

    def get_peak_usage(self) -> dict[int, float]:
        """Get peak memory usage per device."""
        return {
            d: m.peak_usage / (1024 * 1024)
            for d, m in self.device_managers.items()
        }

    def reset(self) -> None:
        """Reset all device managers."""
        for manager in self.device_managers.values():
            manager.reset()


class KernelExecutor:
    """Execute individual kernels (simulated)."""

    def __init__(self):
        self.kernel_impls: dict[str, Callable] = {}
        self._setup_default_impls()

    def _setup_default_impls(self) -> None:
        """Set up default kernel implementations."""
        # These are simulation stubs
        self.kernel_impls["gemm"] = self._execute_gemm
        self.kernel_impls["elementwise"] = self._execute_elementwise
        self.kernel_impls["attention"] = self._execute_attention
        self.kernel_impls["reduce"] = self._execute_reduce
        self.kernel_impls["softmax"] = self._execute_softmax

    def execute(
        self,
        kernel: Kernel,
        inputs: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], KernelStats]:
        """Execute kernel."""
        start_time = time.perf_counter()

        # Get implementation
        impl = self.kernel_impls.get(kernel.kernel_type.value)
        if impl:
            outputs = impl(kernel, inputs)
        else:
            # Default: just create output shapes
            outputs = {}
            for out_desc in kernel.outputs:
                outputs[out_desc.tensor_id] = np.zeros(out_desc.shape, dtype=np.float16)

        # Simulate compute time
        time.sleep(kernel.estimated_time_us / 1e6)

        elapsed = time.perf_counter() - start_time

        stats = KernelStats(
            compute_time_us=elapsed * 1e6,
            memory_read_bytes=sum(inp.size_bytes for inp in kernel.inputs),
            memory_write_bytes=sum(out.size_bytes for out in kernel.outputs),
            flops=kernel.flops
        )

        return outputs, stats

    def _execute_gemm(
        self,
        kernel: Kernel,
        inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Simulated GEMM execution."""
        outputs = {}
        for out_desc in kernel.outputs:
            outputs[out_desc.tensor_id] = np.zeros(out_desc.shape, dtype=np.float16)
        return outputs

    def _execute_elementwise(
        self,
        kernel: Kernel,
        inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Simulated elementwise execution."""
        outputs = {}
        for out_desc in kernel.outputs:
            outputs[out_desc.tensor_id] = np.zeros(out_desc.shape, dtype=np.float16)
        return outputs

    def _execute_attention(
        self,
        kernel: Kernel,
        inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Simulated attention execution."""
        outputs = {}
        for out_desc in kernel.outputs:
            outputs[out_desc.tensor_id] = np.zeros(out_desc.shape, dtype=np.float16)
        return outputs

    def _execute_reduce(
        self,
        kernel: Kernel,
        inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Simulated reduce execution."""
        outputs = {}
        for out_desc in kernel.outputs:
            outputs[out_desc.tensor_id] = np.zeros(out_desc.shape, dtype=np.float16)
        return outputs

    def _execute_softmax(
        self,
        kernel: Kernel,
        inputs: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Simulated softmax execution."""
        outputs = {}
        for out_desc in kernel.outputs:
            outputs[out_desc.tensor_id] = np.zeros(out_desc.shape, dtype=np.float16)
        return outputs


class GraphExecutor:
    """Execute scheduled compute graph."""

    def __init__(self, cluster: MultiGPUCluster):
        self.cluster = cluster
        self.memory_manager = ClusterMemoryManager(cluster)
        self.kernel_executor = KernelExecutor()
        self.tensor_data: dict[str, np.ndarray] = {}

    def execute(
        self,
        graph: ComputeGraph,
        schedule: Schedule,
        inputs: dict[str, np.ndarray] | None = None
    ) -> tuple[dict[str, np.ndarray], ExecutionStats]:
        """Execute scheduled graph."""
        start_time = time.perf_counter()
        kernel_times: dict[str, float] = {}

        # Initialize input tensors
        self.tensor_data = inputs or {}

        # Allocate memory for all tensors
        self._allocate_memory(graph)

        # Execute in schedule order
        for device_id, device_sched in schedule.device_schedules.items():
            for stream in device_sched.streams:
                for scheduled in stream.scheduled_kernels:
                    # Gather inputs
                    kernel_inputs = {}
                    for inp in scheduled.kernel.inputs:
                        if inp.tensor_id in self.tensor_data:
                            kernel_inputs[inp.tensor_id] = self.tensor_data[inp.tensor_id]

                    # Execute kernel
                    outputs, stats = self.kernel_executor.execute(
                        scheduled.kernel,
                        kernel_inputs
                    )

                    # Store outputs
                    self.tensor_data.update(outputs)
                    kernel_times[scheduled.kernel.kernel_id] = stats.compute_time_us / 1000

        # Gather final outputs
        outputs = {}
        for tensor in graph.output_tensors:
            if tensor.tensor_id in self.tensor_data:
                outputs[tensor.tensor_id] = self.tensor_data[tensor.tensor_id]

        total_time = (time.perf_counter() - start_time) * 1000

        # Calculate stats
        peak_memory = max(self.memory_manager.get_peak_usage().values())
        device_util = {
            d: ds.get_utilization()
            for d, ds in schedule.device_schedules.items()
        }

        stats = ExecutionStats(
            total_time_ms=total_time,
            kernel_times_ms=kernel_times,
            memory_peak_mb=peak_memory,
            device_utilization=device_util,
            memory_transfers_ms=0.0
        )

        return outputs, stats

    def _allocate_memory(self, graph: ComputeGraph) -> None:
        """Allocate memory for graph tensors."""
        for kernel in graph.kernels.values():
            for tensor in kernel.outputs:
                try:
                    self.memory_manager.allocate(tensor)
                except MemoryError:
                    pass  # Handle out of memory

    def reset(self) -> None:
        """Reset executor state."""
        self.memory_manager.reset()
        self.tensor_data.clear()


class StreamExecutor:
    """Execute with stream parallelism (simulated)."""

    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams
        self.stream_queues: list[list[Kernel]] = [[] for _ in range(num_streams)]

    def submit(self, kernel: Kernel, stream_id: int) -> None:
        """Submit kernel to stream."""
        if 0 <= stream_id < self.num_streams:
            self.stream_queues[stream_id].append(kernel)

    def synchronize(self) -> None:
        """Synchronize all streams."""
        # In simulation, just clear queues
        for queue in self.stream_queues:
            queue.clear()


class CUDAGraphCapture:
    """Capture and replay CUDA graphs (simulated)."""

    def __init__(self):
        self.captured_graph: ComputeGraph | None = None
        self.is_capturing = False

    def start_capture(self) -> None:
        """Start graph capture."""
        self.is_capturing = True
        self.captured_graph = ComputeGraph()

    def end_capture(self) -> ComputeGraph:
        """End capture and return graph."""
        self.is_capturing = False
        graph = self.captured_graph
        self.captured_graph = None
        return graph

    def replay(
        self,
        executor: GraphExecutor,
        schedule: Schedule
    ) -> ExecutionStats:
        """Replay captured graph."""
        if not self.captured_graph:
            raise RuntimeError("No graph captured")

        _, stats = executor.execute(self.captured_graph, schedule)
        return stats


class Profiler:
    """Profile kernel execution."""

    def __init__(self):
        self.records: list[dict] = []
        self.enabled = False

    def enable(self) -> None:
        """Enable profiling."""
        self.enabled = True

    def disable(self) -> None:
        """Disable profiling."""
        self.enabled = False

    def record(
        self,
        kernel_id: str,
        kernel_type: str,
        device_id: int,
        stats: KernelStats
    ) -> None:
        """Record kernel execution."""
        if not self.enabled:
            return

        self.records.append({
            "kernel_id": kernel_id,
            "kernel_type": kernel_type,
            "device_id": device_id,
            "time_us": stats.compute_time_us,
            "flops": stats.flops,
            "memory_read": stats.memory_read_bytes,
            "memory_write": stats.memory_write_bytes
        })

    def get_summary(self) -> dict[str, Any]:
        """Get profiling summary."""
        if not self.records:
            return {}

        total_time = sum(r["time_us"] for r in self.records)
        by_type: dict[str, float] = defaultdict(float)

        for r in self.records:
            by_type[r["kernel_type"]] += r["time_us"]

        return {
            "total_time_us": total_time,
            "num_kernels": len(self.records),
            "time_by_type": dict(by_type),
            "avg_time_us": total_time / len(self.records)
        }

    def reset(self) -> None:
        """Reset profiler."""
        self.records.clear()


class MultiGPUExecutionEngine:
    """Main execution engine for multi-GPU workloads."""

    def __init__(self, cluster: MultiGPUCluster):
        self.cluster = cluster
        self.executor = GraphExecutor(cluster)
        self.profiler = Profiler()

    def execute(
        self,
        graph: ComputeGraph,
        schedule: Schedule,
        inputs: dict[str, np.ndarray] | None = None
    ) -> tuple[dict[str, np.ndarray], ExecutionStats]:
        """Execute graph with schedule."""
        self.profiler.enable()

        try:
            outputs, stats = self.executor.execute(graph, schedule, inputs)
            return outputs, stats
        finally:
            self.profiler.disable()

    def benchmark(
        self,
        graph: ComputeGraph,
        schedule: Schedule,
        num_iterations: int = 10,
        warmup: int = 3
    ) -> dict[str, float]:
        """Benchmark execution."""
        # Warmup
        for _ in range(warmup):
            self.executor.execute(graph, schedule)
            self.executor.reset()

        # Benchmark
        times = []
        for _ in range(num_iterations):
            _, stats = self.executor.execute(graph, schedule)
            times.append(stats.total_time_ms)
            self.executor.reset()

        return {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times)
        }

    def get_profiling_summary(self) -> dict[str, Any]:
        """Get profiling summary."""
        return self.profiler.get_summary()

    def reset(self) -> None:
        """Reset engine state."""
        self.executor.reset()
        self.profiler.reset()


def execute_graph(
    graph: ComputeGraph,
    schedule: Schedule,
    cluster: MultiGPUCluster,
    inputs: dict[str, np.ndarray] | None = None
) -> tuple[dict[str, np.ndarray], ExecutionStats]:
    """Convenience function to execute graph."""
    engine = MultiGPUExecutionEngine(cluster)
    return engine.execute(graph, schedule, inputs)


# =============================================================================
# Cost Models
# =============================================================================


class RooflineModel:
    """Roofline cost model for kernel execution time estimation.

    Classifies kernels as compute-bound or memory-bound and
    estimates execution time based on device characteristics.
    """

    def __init__(self, device: GPUDevice):
        self.device = device
        self.peak_tflops = device.peak_tflops
        self.memory_bandwidth_gbps = device.memory_bandwidth_gbps

    @property
    def ridge_point(self) -> float:
        """Operational intensity at the ridge point (FLOPs/byte)."""
        if self.memory_bandwidth_gbps == 0:
            return float('inf')
        return (self.peak_tflops * 1e12) / (self.memory_bandwidth_gbps * 1e9)

    def is_compute_bound(self, kernel: Kernel) -> bool:
        """Check if kernel is compute-bound."""
        oi = self.operational_intensity(kernel)
        return oi >= self.ridge_point

    def operational_intensity(self, kernel: Kernel) -> float:
        """Compute operational intensity (FLOPs per byte)."""
        total_bytes = sum(t.size_bytes for t in kernel.inputs) + \
                      sum(t.size_bytes for t in kernel.outputs)
        if total_bytes == 0:
            return 0.0

        flops = self._estimate_flops(kernel)
        return flops / total_bytes

    def estimate_time_us(self, kernel: Kernel) -> float:
        """Estimate kernel execution time in microseconds."""
        flops = self._estimate_flops(kernel)
        total_bytes = sum(t.size_bytes for t in kernel.inputs) + \
                      sum(t.size_bytes for t in kernel.outputs)

        # Compute time (seconds)
        compute_time = flops / (self.peak_tflops * 1e12) if self.peak_tflops > 0 else 0

        # Memory time (seconds)
        mem_time = total_bytes / (self.memory_bandwidth_gbps * 1e9) if self.memory_bandwidth_gbps > 0 else 0

        # Roofline: actual time is max of compute and memory time
        return max(compute_time, mem_time) * 1e6

    def _estimate_flops(self, kernel: Kernel) -> float:
        """Estimate FLOPs for a kernel."""
        attrs = kernel.attributes

        if kernel.kernel_type.value == "gemm":
            m = attrs.get("m", 1)
            k = attrs.get("k", 1)
            n = attrs.get("n", 1)
            return 2.0 * m * k * n

        elif kernel.kernel_type.value == "attention":
            b = attrs.get("batch", 1)
            h = attrs.get("heads", 1)
            s = attrs.get("seq_len", 1)
            d = attrs.get("head_dim", 1)
            return 4.0 * b * h * s * s * d

        elif kernel.kernel_type.value == "elementwise":
            numel = 1
            if kernel.inputs:
                for dim in kernel.inputs[0].shape:
                    numel *= dim
            return float(numel)

        elif kernel.kernel_type.value == "reduce":
            numel = 1
            if kernel.inputs:
                for dim in kernel.inputs[0].shape:
                    numel *= dim
            return float(numel)

        return 0.0


class CommunicationCostModel:
    """Model for inter-device data transfer costs."""

    def __init__(self, cluster: MultiGPUCluster):
        self.cluster = cluster
        self.nvlink_bandwidth = cluster.nvlink_bandwidth_gbps
        self.pcie_bandwidth = cluster.pcie_bandwidth_gbps

    def transfer_time_us(
        self,
        size_bytes: int,
        src_device: int,
        dst_device: int,
        use_nvlink: bool = True
    ) -> float:
        """Estimate transfer time in microseconds."""
        if src_device == dst_device:
            return 0.0

        bandwidth = self.nvlink_bandwidth if use_nvlink else self.pcie_bandwidth
        if bandwidth == 0:
            return float('inf')

        # Transfer time (seconds) = bytes / (bandwidth in bytes/s)
        transfer_s = size_bytes / (bandwidth * 1e9)
        return transfer_s * 1e6

    def estimate_tensor_transfer(
        self,
        tensor: TensorDescriptor,
        dst_device: int,
        use_nvlink: bool = True
    ) -> float:
        """Estimate transfer time for a tensor."""
        return self.transfer_time_us(
            tensor.size_bytes,
            tensor.device_id,
            dst_device,
            use_nvlink
        )

    def total_communication_cost(
        self,
        graph: ComputeGraph,
        schedule: Schedule
    ) -> float:
        """Estimate total communication cost for a scheduled graph."""
        total_us = 0.0

        for dep in graph.dependencies:
            src_kernel = graph.kernels.get(dep.source_id)
            dst_kernel = graph.kernels.get(dep.target_id)
            if not src_kernel or not dst_kernel:
                continue

            if src_kernel.device_id != dst_kernel.device_id:
                # Cross-device dependency -- need transfer
                for tensor in src_kernel.outputs:
                    total_us += self.transfer_time_us(
                        tensor.size_bytes,
                        src_kernel.device_id,
                        dst_kernel.device_id
                    )

        return total_us


class CostModelScheduler:
    """Cost-model-driven kernel scheduler.

    Uses RooflineModel and CommunicationCostModel to make
    informed placement and scheduling decisions.
    """

    def __init__(self, cluster: MultiGPUCluster):
        self.cluster = cluster
        self.roofline_models = {
            d.device_id: RooflineModel(d) for d in cluster.devices
        }
        self.comm_model = CommunicationCostModel(cluster)

    def estimate_kernel_time(self, kernel: Kernel, device_id: int | None = None) -> float:
        """Estimate kernel time on a specific device."""
        did = device_id if device_id is not None else kernel.device_id
        model = self.roofline_models.get(did)
        if model is None:
            model = next(iter(self.roofline_models.values()))
        return model.estimate_time_us(kernel)

    def estimate_total_time(self, graph: ComputeGraph, schedule: Schedule) -> float:
        """Estimate total execution time including communication."""
        kernel_time = sum(
            self.estimate_kernel_time(k)
            for k in graph.kernels.values()
        )
        comm_time = self.comm_model.total_communication_cost(graph, schedule)
        return kernel_time + comm_time


# =============================================================================
# Speculative Execution
# =============================================================================


@dataclass
class SpeculativeResult:
    """Result of a speculative kernel execution."""
    kernel_id: str
    output: dict[str, np.ndarray] | None
    is_valid: bool
    was_speculative: bool
    time_us: float


class SpeculativeExecutor:
    """Execute kernels speculatively before dependencies confirm.

    Pre-launches kernels when their dependencies are likely
    (but not yet confirmed) complete. Provides rollback on
    invalid results.
    """

    def __init__(
        self,
        executor: GraphExecutor,
        speculation_depth: int = 2,
        confidence_threshold: float = 0.8
    ):
        self.executor = executor
        self.speculation_depth = speculation_depth
        self.confidence_threshold = confidence_threshold
        self._speculative_results: dict[str, SpeculativeResult] = {}
        self._committed: set[str] = set()
        self._rolled_back: set[str] = set()

    @property
    def speculative_results(self) -> dict[str, SpeculativeResult]:
        return self._speculative_results

    @property
    def committed_count(self) -> int:
        return len(self._committed)

    @property
    def rollback_count(self) -> int:
        return len(self._rolled_back)

    def execute_speculative(
        self,
        graph: ComputeGraph,
        schedule: Schedule,
        completed: set[str] | None = None
    ) -> dict[str, SpeculativeResult]:
        """Execute graph with speculative launches.

        Args:
            graph: Compute graph
            schedule: Execution schedule
            completed: Already-completed kernel IDs

        Returns:
            Dict mapping kernel_id to SpeculativeResult
        """
        completed = completed or set()
        results: dict[str, SpeculativeResult] = {}
        topo_order = graph.topological_sort()

        for kernel_id in topo_order:
            kernel = graph.kernels[kernel_id]
            deps = graph.get_dependencies(kernel_id)

            # Check if all deps are truly completed
            all_deps_done = all(d in completed for d in deps)
            # Check if deps are likely completed (within speculation depth)
            deps_likely = all(
                d in completed or d in results
                for d in deps
            )

            if all_deps_done:
                # Normal execution
                start = time.time()
                output = self._execute_kernel(kernel)
                elapsed = (time.time() - start) * 1e6

                result = SpeculativeResult(
                    kernel_id=kernel_id,
                    output=output,
                    is_valid=True,
                    was_speculative=False,
                    time_us=elapsed
                )
                results[kernel_id] = result
                completed.add(kernel_id)
                self._committed.add(kernel_id)

            elif deps_likely and len(deps) <= self.speculation_depth:
                # Speculative execution
                start = time.time()
                output = self._execute_kernel(kernel)
                elapsed = (time.time() - start) * 1e6

                result = SpeculativeResult(
                    kernel_id=kernel_id,
                    output=output,
                    is_valid=True,
                    was_speculative=True,
                    time_us=elapsed
                )
                results[kernel_id] = result

        self._speculative_results = results
        return results

    def validate_and_commit(
        self,
        results: dict[str, SpeculativeResult],
        completed: set[str],
        graph: ComputeGraph
    ) -> dict[str, SpeculativeResult]:
        """Validate speculative results and commit or rollback.

        Args:
            results: Speculative results to validate
            completed: Set of confirmed-complete kernel IDs
            graph: The compute graph

        Returns:
            Validated results (invalid ones marked and rolled back)
        """
        validated = {}

        for kernel_id, result in results.items():
            if not result.was_speculative:
                validated[kernel_id] = result
                continue

            deps = graph.get_dependencies(kernel_id)
            all_deps_valid = all(d in completed for d in deps)

            if all_deps_valid:
                # Speculation was correct
                validated[kernel_id] = result
                self._committed.add(kernel_id)
            else:
                # Rollback
                rolled_back = SpeculativeResult(
                    kernel_id=kernel_id,
                    output=None,
                    is_valid=False,
                    was_speculative=True,
                    time_us=result.time_us
                )
                validated[kernel_id] = rolled_back
                self._rolled_back.add(kernel_id)

        return validated

    def _execute_kernel(self, kernel: Kernel) -> dict[str, np.ndarray]:
        """Execute a single kernel."""
        outputs = {}
        for tensor in kernel.outputs:
            outputs[tensor.tensor_id] = np.zeros(tensor.shape, dtype=np.float16)
        return outputs

    def get_stats(self) -> dict[str, Any]:
        """Get speculative execution statistics."""
        total = len(self._speculative_results)
        speculative = sum(
            1 for r in self._speculative_results.values() if r.was_speculative
        )
        return {
            "total_kernels": total,
            "speculative_launches": speculative,
            "committed": self.committed_count,
            "rolled_back": self.rollback_count,
            "speculation_success_rate": (
                self.committed_count / max(speculative, 1)
            )
        }

    def reset(self) -> None:
        """Reset speculative executor state."""
        self._speculative_results.clear()
        self._committed.clear()
        self._rolled_back.clear()
