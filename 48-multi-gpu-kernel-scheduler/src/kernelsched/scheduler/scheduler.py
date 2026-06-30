"""Kernel scheduling for multi-GPU execution."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import heapq
from collections import defaultdict

from ..core.kernel import (
    Kernel, ComputeGraph, MultiGPUCluster, GPUDevice,
    KernelDependency, TensorDescriptor, PipelineStage,
    PipelineConfig, PipelinePartitioner
)


class SchedulingPolicy(Enum):
    """Scheduling policies."""
    FIFO = "fifo"
    PRIORITY = "priority"
    SHORTEST_JOB = "shortest"
    CRITICAL_PATH = "critical"
    LOAD_BALANCE = "balance"


@dataclass
class ScheduledKernel:
    """Kernel with scheduling information."""
    kernel: Kernel
    device_id: int
    stream_id: int
    start_time_us: float
    end_time_us: float
    priority: int = 0


@dataclass
class Stream:
    """CUDA stream representation."""
    stream_id: int
    device_id: int
    scheduled_kernels: list[ScheduledKernel] = field(default_factory=list)
    current_time_us: float = 0.0

    def schedule(self, kernel: Kernel, start_time: float) -> ScheduledKernel:
        """Schedule kernel on this stream."""
        actual_start = max(start_time, self.current_time_us)
        end_time = actual_start + kernel.estimated_time_us

        scheduled = ScheduledKernel(
            kernel=kernel,
            device_id=self.device_id,
            stream_id=self.stream_id,
            start_time_us=actual_start,
            end_time_us=end_time
        )

        self.scheduled_kernels.append(scheduled)
        self.current_time_us = end_time

        return scheduled

    def get_utilization(self, total_time: float) -> float:
        """Get stream utilization."""
        if total_time <= 0:
            return 0.0
        busy_time = sum(
            sk.end_time_us - sk.start_time_us
            for sk in self.scheduled_kernels
        )
        return busy_time / total_time


@dataclass
class DeviceSchedule:
    """Schedule for a single device."""
    device_id: int
    streams: list[Stream]
    total_time_us: float = 0.0

    def get_free_stream(self) -> Stream:
        """Get stream with earliest finish time."""
        return min(self.streams, key=lambda s: s.current_time_us)

    def get_utilization(self) -> float:
        """Get device utilization."""
        if self.total_time_us <= 0:
            return 0.0
        return sum(s.get_utilization(self.total_time_us) for s in self.streams) / len(self.streams)


@dataclass
class Schedule:
    """Complete schedule for multi-GPU execution."""
    device_schedules: dict[int, DeviceSchedule]
    total_time_us: float = 0.0
    memory_transfers: list[dict] = field(default_factory=list)

    def get_schedule_summary(self) -> dict[str, Any]:
        """Get summary of schedule."""
        total_kernels = sum(
            len(s.scheduled_kernels)
            for ds in self.device_schedules.values()
            for s in ds.streams
        )

        return {
            "total_time_us": self.total_time_us,
            "total_kernels": total_kernels,
            "num_devices": len(self.device_schedules),
            "device_utilization": {
                d: ds.get_utilization()
                for d, ds in self.device_schedules.items()
            },
            "num_transfers": len(self.memory_transfers)
        }


class KernelScheduler:
    """Base kernel scheduler."""

    def __init__(
        self,
        cluster: MultiGPUCluster,
        num_streams_per_device: int = 4
    ):
        self.cluster = cluster
        self.num_streams = num_streams_per_device

    def schedule(self, graph: ComputeGraph) -> Schedule:
        """Schedule compute graph on cluster."""
        raise NotImplementedError


class FIFOScheduler(KernelScheduler):
    """Simple FIFO scheduler."""

    def schedule(self, graph: ComputeGraph) -> Schedule:
        """Schedule in topological order."""
        # Initialize device schedules
        device_schedules = {}
        for device in self.cluster.devices:
            streams = [
                Stream(stream_id=i, device_id=device.device_id)
                for i in range(self.num_streams)
            ]
            device_schedules[device.device_id] = DeviceSchedule(
                device_id=device.device_id,
                streams=streams
            )

        # Track completion times for dependencies
        completion_times: dict[str, float] = {}

        # Schedule in topological order
        topo_order = graph.topological_sort()

        for kernel_id in topo_order:
            kernel = graph.kernels[kernel_id]

            # Find earliest start time based on dependencies
            earliest_start = 0.0
            for dep_id in graph.get_dependencies(kernel_id):
                earliest_start = max(earliest_start, completion_times.get(dep_id, 0))

            # Get device schedule
            device_id = kernel.device_id
            if device_id not in device_schedules:
                device_id = 0  # Default to device 0

            device_sched = device_schedules[device_id]
            stream = device_sched.get_free_stream()

            # Schedule kernel
            scheduled = stream.schedule(kernel, earliest_start)
            completion_times[kernel_id] = scheduled.end_time_us

        # Calculate total time
        total_time = max(
            ds.streams[-1].current_time_us if ds.streams else 0
            for ds in device_schedules.values()
        )

        for ds in device_schedules.values():
            ds.total_time_us = total_time

        return Schedule(
            device_schedules=device_schedules,
            total_time_us=total_time
        )


class CriticalPathScheduler(KernelScheduler):
    """Schedule based on critical path analysis."""

    def schedule(self, graph: ComputeGraph) -> Schedule:
        """Schedule prioritizing critical path."""
        # Initialize
        device_schedules = {}
        for device in self.cluster.devices:
            streams = [
                Stream(stream_id=i, device_id=device.device_id)
                for i in range(self.num_streams)
            ]
            device_schedules[device.device_id] = DeviceSchedule(
                device_id=device.device_id,
                streams=streams
            )

        # Calculate priorities (distance to end)
        priorities = self._calculate_priorities(graph)

        # Ready queue (priority, kernel_id)
        ready_queue: list[tuple[float, str]] = []
        completion_times: dict[str, float] = {}
        remaining_deps: dict[str, int] = {}

        # Initialize
        for kernel_id in graph.kernels:
            deps = graph.get_dependencies(kernel_id)
            remaining_deps[kernel_id] = len(deps)
            if len(deps) == 0:
                heapq.heappush(ready_queue, (-priorities[kernel_id], kernel_id))

        # Schedule
        while ready_queue:
            _, kernel_id = heapq.heappop(ready_queue)
            kernel = graph.kernels[kernel_id]

            # Find earliest start time
            earliest_start = 0.0
            for dep_id in graph.get_dependencies(kernel_id):
                earliest_start = max(earliest_start, completion_times.get(dep_id, 0))

            # Select device and stream
            device_id = kernel.device_id
            if device_id not in device_schedules:
                device_id = 0

            device_sched = device_schedules[device_id]
            stream = device_sched.get_free_stream()

            # Schedule
            scheduled = stream.schedule(kernel, earliest_start)
            completion_times[kernel_id] = scheduled.end_time_us

            # Update dependents
            for dep_id in graph.get_dependents(kernel_id):
                remaining_deps[dep_id] -= 1
                if remaining_deps[dep_id] == 0:
                    heapq.heappush(ready_queue, (-priorities[dep_id], dep_id))

        # Calculate total time
        total_time = max(
            max((s.current_time_us for s in ds.streams), default=0)
            for ds in device_schedules.values()
        )

        for ds in device_schedules.values():
            ds.total_time_us = total_time

        return Schedule(
            device_schedules=device_schedules,
            total_time_us=total_time
        )

    def _calculate_priorities(self, graph: ComputeGraph) -> dict[str, float]:
        """Calculate priority as distance to end (sum of times)."""
        topo_order = graph.topological_sort()
        priorities: dict[str, float] = {}

        # Backward pass
        for kernel_id in reversed(topo_order):
            kernel = graph.kernels[kernel_id]
            dependents = graph.get_dependents(kernel_id)

            if not dependents:
                priorities[kernel_id] = kernel.estimated_time_us
            else:
                max_child = max(priorities[d] for d in dependents)
                priorities[kernel_id] = kernel.estimated_time_us + max_child

        return priorities


class LoadBalanceScheduler(KernelScheduler):
    """Load-balanced multi-GPU scheduler."""

    def schedule(self, graph: ComputeGraph) -> Schedule:
        """Schedule with load balancing across devices."""
        # Initialize
        device_schedules = {}
        device_loads: dict[int, float] = {}

        for device in self.cluster.devices:
            streams = [
                Stream(stream_id=i, device_id=device.device_id)
                for i in range(self.num_streams)
            ]
            device_schedules[device.device_id] = DeviceSchedule(
                device_id=device.device_id,
                streams=streams
            )
            device_loads[device.device_id] = 0.0

        completion_times: dict[str, float] = {}
        kernel_device: dict[str, int] = {}

        # Schedule in topological order with load balancing
        topo_order = graph.topological_sort()

        for kernel_id in topo_order:
            kernel = graph.kernels[kernel_id]

            # Find earliest start time
            earliest_start = 0.0
            for dep_id in graph.get_dependencies(kernel_id):
                earliest_start = max(earliest_start, completion_times.get(dep_id, 0))

            # Select device with minimum load
            # Prefer keeping data on same device
            deps = graph.get_dependencies(kernel_id)
            if deps:
                # Get device of dependencies
                dep_devices = [kernel_device.get(d, 0) for d in deps]
                # Count devices
                device_counts = defaultdict(int)
                for d in dep_devices:
                    device_counts[d] += 1
                # Prefer device with most dependencies
                preferred_device = max(device_counts, key=device_counts.get)
            else:
                # No dependencies, use least loaded device
                preferred_device = min(device_loads, key=device_loads.get)

            # Check if load difference justifies transfer
            min_load_device = min(device_loads, key=device_loads.get)
            if device_loads[preferred_device] > device_loads[min_load_device] * 1.2:
                device_id = min_load_device
            else:
                device_id = preferred_device

            device_sched = device_schedules[device_id]
            stream = device_sched.get_free_stream()

            # Schedule
            scheduled = stream.schedule(kernel, earliest_start)
            completion_times[kernel_id] = scheduled.end_time_us
            kernel_device[kernel_id] = device_id
            device_loads[device_id] += kernel.estimated_time_us

        # Calculate total time
        total_time = max(
            max((s.current_time_us for s in ds.streams), default=0)
            for ds in device_schedules.values()
        )

        for ds in device_schedules.values():
            ds.total_time_us = total_time

        return Schedule(
            device_schedules=device_schedules,
            total_time_us=total_time
        )


class StreamScheduler:
    """Schedule kernels to streams with overlap."""

    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams

    def assign_streams(
        self,
        graph: ComputeGraph,
        device_id: int = 0
    ) -> dict[str, int]:
        """Assign kernels to streams for overlap."""
        stream_assignments: dict[str, int] = {}
        stream_finish_times = [0.0] * self.num_streams

        topo_order = graph.topological_sort()
        completion_times: dict[str, float] = {}

        for kernel_id in topo_order:
            kernel = graph.kernels[kernel_id]

            # Find earliest start
            earliest_start = 0.0
            for dep_id in graph.get_dependencies(kernel_id):
                earliest_start = max(earliest_start, completion_times.get(dep_id, 0))

            # Find stream that can start earliest
            best_stream = 0
            best_start = float('inf')

            for i in range(self.num_streams):
                start = max(earliest_start, stream_finish_times[i])
                if start < best_start:
                    best_start = start
                    best_stream = i

            # Assign to stream
            stream_assignments[kernel_id] = best_stream
            stream_finish_times[best_stream] = best_start + kernel.estimated_time_us
            completion_times[kernel_id] = stream_finish_times[best_stream]

        return stream_assignments


class MemoryScheduler:
    """Schedule memory transfers between devices."""

    def __init__(self, cluster: MultiGPUCluster):
        self.cluster = cluster

    def schedule_transfers(
        self,
        graph: ComputeGraph,
        kernel_placement: dict[str, int]
    ) -> list[dict]:
        """Schedule necessary memory transfers."""
        transfers = []

        for dep in graph.dependencies:
            source_device = kernel_placement.get(dep.source_id, 0)
            target_device = kernel_placement.get(dep.target_id, 0)

            if source_device != target_device:
                # Need transfer
                source_kernel = graph.kernels[dep.source_id]

                # Find tensor size
                tensor_size = 0
                for out in source_kernel.outputs:
                    if out.tensor_id == dep.tensor_id or dep.tensor_id.startswith(out.tensor_id):
                        tensor_size = out.size_bytes
                        break

                if tensor_size == 0:
                    tensor_size = 1024 * 1024  # Default 1MB

                # Calculate transfer time
                bandwidth = self.cluster.nvlink_bandwidth_gbps * 1e9 / 8  # bytes/s
                transfer_time_us = (tensor_size / bandwidth) * 1e6

                transfers.append({
                    "source_device": source_device,
                    "target_device": target_device,
                    "tensor_id": dep.tensor_id,
                    "size_bytes": tensor_size,
                    "transfer_time_us": transfer_time_us
                })

        return transfers


@dataclass
class MicrobatchSchedule:
    """Schedule for a single microbatch."""
    microbatch_id: int
    stage_times: dict[int, tuple[float, float]]  # stage_id -> (start, end)


@dataclass
class PipelineSchedule:
    """Complete pipeline parallel schedule."""
    stages: list[PipelineStage]
    microbatch_schedules: list[MicrobatchSchedule]
    total_time_us: float
    pipeline_bubble_us: float  # Time lost to pipeline bubbles
    num_microbatches: int

    @property
    def bubble_ratio(self) -> float:
        """Get pipeline bubble ratio (0 = no bubble, 1 = all bubble)."""
        if self.total_time_us <= 0:
            return 0.0
        return self.pipeline_bubble_us / self.total_time_us

    @property
    def efficiency(self) -> float:
        """Get pipeline efficiency (1 - bubble_ratio)."""
        return 1.0 - self.bubble_ratio

    def get_stage_utilization(self, stage_id: int) -> float:
        """Get utilization for a specific stage."""
        busy_time = sum(
            mb.stage_times[stage_id][1] - mb.stage_times[stage_id][0]
            for mb in self.microbatch_schedules
            if stage_id in mb.stage_times
        )
        return busy_time / self.total_time_us if self.total_time_us > 0 else 0.0


class PipelineScheduler:
    """
    Pipeline parallel scheduler for multi-GPU training.

    Implements GPipe-style pipeline parallelism with microbatching
    to minimize pipeline bubbles.
    """

    def __init__(
        self,
        cluster: MultiGPUCluster,
        config: PipelineConfig
    ):
        self.cluster = cluster
        self.config = config
        self.partitioner = PipelinePartitioner(
            config.num_stages,
            cluster.num_devices
        )

    def schedule(
        self,
        graph: ComputeGraph,
        strategy: str = "balanced"
    ) -> PipelineSchedule:
        """
        Create a pipeline schedule for the compute graph.

        Args:
            graph: Compute graph to schedule
            strategy: Partitioning strategy (balanced, memory, layer)

        Returns:
            PipelineSchedule with microbatch timings
        """
        # Partition graph into stages
        stages = self.partitioner.partition(graph, strategy)

        # Calculate stage execution times
        stage_times = self._compute_stage_times(graph, stages)

        # Schedule microbatches
        if self.config.interleave_stages:
            microbatch_schedules = self._schedule_1f1b(stages, stage_times)
        else:
            microbatch_schedules = self._schedule_gpipe(stages, stage_times)

        # Calculate total time and bubbles
        total_time = self._compute_total_time(microbatch_schedules)
        ideal_time = self._compute_ideal_time(stage_times)
        bubble_time = total_time - ideal_time

        return PipelineSchedule(
            stages=stages,
            microbatch_schedules=microbatch_schedules,
            total_time_us=total_time,
            pipeline_bubble_us=max(0, bubble_time),
            num_microbatches=self.config.num_microbatches
        )

    def _compute_stage_times(
        self,
        graph: ComputeGraph,
        stages: list[PipelineStage]
    ) -> dict[int, float]:
        """Compute execution time for each stage."""
        stage_times = {}
        for stage in stages:
            time = sum(
                graph.kernels[kid].estimated_time_us
                for kid in stage.kernel_ids
            )
            stage_times[stage.stage_id] = time
        return stage_times

    def _schedule_gpipe(
        self,
        stages: list[PipelineStage],
        stage_times: dict[int, float]
    ) -> list[MicrobatchSchedule]:
        """
        GPipe scheduling: all forward passes then all backward passes.

        Timeline (2 stages, 4 microbatches):
        Stage 0: F0 F1 F2 F3 ---- B3 B2 B1 B0
        Stage 1: -- F0 F1 F2 F3 B3 B2 B1 B0 --
        """
        schedules = []
        num_stages = len(stages)
        num_mb = self.config.num_microbatches

        # Track when each stage is free
        stage_finish = {s.stage_id: 0.0 for s in stages}

        # Forward passes
        for mb in range(num_mb):
            mb_schedule = MicrobatchSchedule(
                microbatch_id=mb,
                stage_times={}
            )

            for stage in stages:
                # Can start when previous stage finishes this microbatch
                # and this stage finishes previous microbatch
                if stage.stage_id > 0:
                    prev_stage = stages[stage.stage_id - 1]
                    # Wait for previous stage to complete this microbatch
                    prev_mb_end = 0.0
                    for s in schedules:
                        if s.microbatch_id == mb and (prev_stage.stage_id) in s.stage_times:
                            prev_mb_end = s.stage_times[prev_stage.stage_id][1]
                            break
                    start_time = max(stage_finish[stage.stage_id], prev_mb_end)
                else:
                    start_time = stage_finish[stage.stage_id]

                end_time = start_time + stage_times[stage.stage_id]
                mb_schedule.stage_times[stage.stage_id] = (start_time, end_time)
                stage_finish[stage.stage_id] = end_time

            schedules.append(mb_schedule)

        return schedules

    def _schedule_1f1b(
        self,
        stages: list[PipelineStage],
        stage_times: dict[int, float]
    ) -> list[MicrobatchSchedule]:
        """
        1F1B (One Forward One Backward) interleaved scheduling.

        This reduces memory usage by interleaving forward and backward
        passes, allowing gradients to be computed as soon as possible.
        """
        schedules = []
        num_stages = len(stages)
        num_mb = self.config.num_microbatches

        # Track when each stage is free
        stage_finish = {s.stage_id: 0.0 for s in stages}

        # Warmup phase: fill pipeline with forward passes
        warmup_mb = min(num_stages - 1, num_mb)

        for mb in range(warmup_mb):
            mb_schedule = MicrobatchSchedule(
                microbatch_id=mb,
                stage_times={}
            )

            for stage in stages[:mb + 1]:
                if stage.stage_id > 0:
                    prev_stage = stages[stage.stage_id - 1]
                    prev_mb_end = 0.0
                    for s in schedules:
                        if s.microbatch_id == mb and prev_stage.stage_id in s.stage_times:
                            prev_mb_end = s.stage_times[prev_stage.stage_id][1]
                            break
                    start_time = max(stage_finish[stage.stage_id], prev_mb_end)
                else:
                    start_time = stage_finish[stage.stage_id]

                end_time = start_time + stage_times[stage.stage_id]
                mb_schedule.stage_times[stage.stage_id] = (start_time, end_time)
                stage_finish[stage.stage_id] = end_time

            schedules.append(mb_schedule)

        # Steady state: interleaved forward and backward
        for mb in range(warmup_mb, num_mb):
            mb_schedule = MicrobatchSchedule(
                microbatch_id=mb,
                stage_times={}
            )

            for stage in stages:
                if stage.stage_id > 0:
                    prev_stage = stages[stage.stage_id - 1]
                    prev_mb_end = 0.0
                    for s in schedules:
                        if s.microbatch_id == mb and prev_stage.stage_id in s.stage_times:
                            prev_mb_end = s.stage_times[prev_stage.stage_id][1]
                            break
                    start_time = max(stage_finish[stage.stage_id], prev_mb_end)
                else:
                    start_time = stage_finish[stage.stage_id]

                end_time = start_time + stage_times[stage.stage_id]
                mb_schedule.stage_times[stage.stage_id] = (start_time, end_time)
                stage_finish[stage.stage_id] = end_time

            schedules.append(mb_schedule)

        return schedules

    def _compute_total_time(
        self,
        schedules: list[MicrobatchSchedule]
    ) -> float:
        """Compute total execution time."""
        max_time = 0.0
        for mb in schedules:
            for start, end in mb.stage_times.values():
                max_time = max(max_time, end)
        return max_time

    def _compute_ideal_time(
        self,
        stage_times: dict[int, float]
    ) -> float:
        """Compute ideal time with no bubbles."""
        max_stage_time = max(stage_times.values()) if stage_times else 0.0
        return max_stage_time * self.config.num_microbatches


def create_scheduler(
    policy: SchedulingPolicy,
    cluster: MultiGPUCluster,
    **kwargs
) -> KernelScheduler:
    """Factory function to create scheduler."""
    if policy == SchedulingPolicy.FIFO:
        return FIFOScheduler(cluster, **kwargs)
    elif policy == SchedulingPolicy.CRITICAL_PATH:
        return CriticalPathScheduler(cluster, **kwargs)
    elif policy == SchedulingPolicy.LOAD_BALANCE:
        return LoadBalanceScheduler(cluster, **kwargs)
    else:
        return FIFOScheduler(cluster, **kwargs)
