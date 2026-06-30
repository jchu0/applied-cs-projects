"""Distributed Data Parallel implementation."""

import numpy as np
import threading
import queue
import time
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum, auto
from concurrent.futures import Future, ThreadPoolExecutor

from ..core.context import (
    DistributedTensor,
    ProcessGroup,
    DistributedContext,
)

logger = logging.getLogger(__name__)


class AllReduceStrategy(Enum):
    """Strategies for all-reduce."""
    RING = auto()
    TREE = auto()
    RECURSIVE_HALVING = auto()
    BUCKET = auto()


@dataclass
class CommunicationStats:
    """Statistics for communication operations."""
    total_comm_time_ms: float = 0.0
    total_overlap_time_ms: float = 0.0
    total_compute_time_ms: float = 0.0
    num_operations: int = 0
    total_bytes_transferred: int = 0

    # Per-bucket stats
    bucket_times_ms: List[float] = field(default_factory=list)
    bucket_sizes_bytes: List[int] = field(default_factory=list)

    @property
    def overlap_ratio(self) -> float:
        """Ratio of overlapped communication to total communication."""
        if self.total_comm_time_ms == 0:
            return 0.0
        return self.total_overlap_time_ms / self.total_comm_time_ms

    @property
    def bandwidth_gbps(self) -> float:
        """Effective bandwidth in Gbps."""
        if self.total_comm_time_ms == 0:
            return 0.0
        bytes_per_second = self.total_bytes_transferred / (self.total_comm_time_ms / 1000.0)
        return bytes_per_second * 8 / 1e9

    @property
    def avg_bucket_time_ms(self) -> float:
        """Average time per bucket."""
        if not self.bucket_times_ms:
            return 0.0
        return sum(self.bucket_times_ms) / len(self.bucket_times_ms)

    def reset(self):
        """Reset all statistics."""
        self.total_comm_time_ms = 0.0
        self.total_overlap_time_ms = 0.0
        self.total_compute_time_ms = 0.0
        self.num_operations = 0
        self.total_bytes_transferred = 0
        self.bucket_times_ms.clear()
        self.bucket_sizes_bytes.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_comm_time_ms": self.total_comm_time_ms,
            "total_overlap_time_ms": self.total_overlap_time_ms,
            "total_compute_time_ms": self.total_compute_time_ms,
            "overlap_ratio": self.overlap_ratio,
            "bandwidth_gbps": self.bandwidth_gbps,
            "num_operations": self.num_operations,
            "total_bytes_transferred": self.total_bytes_transferred,
            "avg_bucket_time_ms": self.avg_bucket_time_ms,
        }


@dataclass
class WorkItem:
    """A unit of work for the communication scheduler."""
    bucket: 'GradientBucket'
    allreduce_fn: Callable
    callback: Optional[Callable] = None
    start_time: float = 0.0
    priority: int = 0


class CommunicationScheduler:
    """
    Schedules communication to overlap with computation.

    Features:
    - Background thread pool for async operations
    - Callback system for completion notification
    - Statistics collection for performance analysis
    - Priority-based scheduling
    """

    def __init__(self, num_threads: int = 2, enable_stats: bool = True):
        self.num_threads = num_threads
        self.enable_stats = enable_stats

        # Work queue with priority
        self._work_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._pending_futures: List[Future] = []

        # Thread pool for async execution
        self._executor: Optional[ThreadPoolExecutor] = None
        self._running = False

        # Statistics
        self._stats = CommunicationStats()
        self._stats_lock = threading.Lock()

        # Callbacks
        self._global_callbacks: List[Callable] = []
        self._completion_callbacks: Dict[int, Callable] = {}

        # Timing for overlap calculation
        self._backward_start_time: Optional[float] = None
        self._backward_end_time: Optional[float] = None

    def start(self):
        """Start the communication scheduler."""
        if self._running:
            return

        self._running = True
        self._executor = ThreadPoolExecutor(
            max_workers=self.num_threads,
            thread_name_prefix="comm_scheduler"
        )
        logger.debug(f"CommunicationScheduler started with {self.num_threads} threads")

    def stop(self):
        """Stop the communication scheduler."""
        if not self._running:
            return

        self._running = False

        # Wait for pending work
        self.wait_all()

        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.debug("CommunicationScheduler stopped")

    def schedule_allreduce(
        self,
        bucket: 'GradientBucket',
        allreduce_fn: Callable,
        callback: Optional[Callable] = None,
        priority: int = 0
    ) -> Future:
        """
        Schedule an AllReduce operation for a bucket.

        Args:
            bucket: The gradient bucket to reduce
            allreduce_fn: Function to perform the AllReduce
            callback: Optional callback when complete
            priority: Lower values = higher priority

        Returns:
            Future for the operation
        """
        if not self._running:
            self.start()

        work_item = WorkItem(
            bucket=bucket,
            allreduce_fn=allreduce_fn,
            callback=callback,
            start_time=time.perf_counter(),
            priority=priority
        )

        # Submit to thread pool
        future = self._executor.submit(self._execute_work, work_item)
        self._pending_futures.append(future)

        return future

    def _execute_work(self, work_item: WorkItem):
        """Execute a work item (runs in thread pool)."""
        bucket = work_item.bucket

        # Record queue wait time (overlap time)
        queue_wait_time = time.perf_counter() - work_item.start_time

        # Execute AllReduce
        comm_start = time.perf_counter()
        try:
            if bucket.flat_tensor is not None:
                result = work_item.allreduce_fn(bucket.flat_tensor)
                bucket.flat_tensor = result
            else:
                # Flatten, reduce, unflatten
                flat_grads = []
                for param in bucket.params:
                    if hasattr(param, 'grad') and param.grad is not None:
                        flat_grads.append(param.grad.flatten())
                    else:
                        flat_grads.append(np.zeros(param.data.size))

                if flat_grads:
                    flat_tensor = np.concatenate(flat_grads)
                    result = work_item.allreduce_fn(flat_tensor)

                    # Unflatten
                    offset = 0
                    for param in bucket.params:
                        size = param.data.size
                        if hasattr(param, 'grad') and param.grad is not None:
                            param.grad = result[offset:offset + size].reshape(param.data.shape)
                        offset += size

        except Exception as e:
            logger.error(f"AllReduce failed for bucket {bucket.index}: {e}")
            raise

        comm_end = time.perf_counter()
        comm_time = (comm_end - comm_start) * 1000  # ms

        # Update statistics
        if self.enable_stats:
            with self._stats_lock:
                self._stats.total_comm_time_ms += comm_time
                self._stats.total_overlap_time_ms += queue_wait_time * 1000
                self._stats.num_operations += 1
                self._stats.total_bytes_transferred += bucket.size_bytes
                self._stats.bucket_times_ms.append(comm_time)
                self._stats.bucket_sizes_bytes.append(bucket.size_bytes)

        # Invoke callbacks
        if work_item.callback:
            try:
                work_item.callback(bucket)
            except Exception as e:
                logger.error(f"Callback failed for bucket {bucket.index}: {e}")

        # Global callbacks
        for cb in self._global_callbacks:
            try:
                cb(bucket, comm_time)
            except Exception as e:
                logger.error(f"Global callback failed: {e}")

        return bucket

    def wait_all(self, timeout: Optional[float] = None):
        """Wait for all scheduled operations to complete."""
        for future in self._pending_futures:
            try:
                future.result(timeout=timeout)
            except Exception as e:
                logger.error(f"Operation failed: {e}")

        self._pending_futures.clear()

    def mark_backward_start(self):
        """Mark the start of backward pass for overlap calculation."""
        self._backward_start_time = time.perf_counter()

    def mark_backward_end(self):
        """Mark the end of backward pass."""
        self._backward_end_time = time.perf_counter()
        if self._backward_start_time:
            compute_time = (self._backward_end_time - self._backward_start_time) * 1000
            with self._stats_lock:
                self._stats.total_compute_time_ms += compute_time

    def add_callback(self, callback: Callable):
        """Add a global callback for all completions."""
        self._global_callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """Remove a global callback."""
        if callback in self._global_callbacks:
            self._global_callbacks.remove(callback)

    def get_stats(self) -> CommunicationStats:
        """Get current communication statistics."""
        with self._stats_lock:
            return CommunicationStats(
                total_comm_time_ms=self._stats.total_comm_time_ms,
                total_overlap_time_ms=self._stats.total_overlap_time_ms,
                total_compute_time_ms=self._stats.total_compute_time_ms,
                num_operations=self._stats.num_operations,
                total_bytes_transferred=self._stats.total_bytes_transferred,
                bucket_times_ms=self._stats.bucket_times_ms.copy(),
                bucket_sizes_bytes=self._stats.bucket_sizes_bytes.copy(),
            )

    def reset_stats(self):
        """Reset statistics."""
        with self._stats_lock:
            self._stats.reset()

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


class PipelineScheduler:
    """
    Pipeline scheduler for overlapping communication with computation.

    Implements a pipeline where:
    1. Buckets are processed in order (last to first for backward overlap)
    2. Communication is pipelined with the next bucket's preparation
    3. Bandwidth utilization is tracked
    """

    def __init__(
        self,
        comm_scheduler: CommunicationScheduler,
        num_pipeline_stages: int = 2
    ):
        self.comm_scheduler = comm_scheduler
        self.num_pipeline_stages = num_pipeline_stages

        # Pipeline state
        self._active_stages: List[Optional['GradientBucket']] = [None] * num_pipeline_stages
        self._stage_futures: List[Optional[Future]] = [None] * num_pipeline_stages

        # Bandwidth tracking
        self._bandwidth_samples: List[float] = []
        self._target_bandwidth_gbps: float = 10.0  # Default 10 Gbps

    def schedule_bucket(
        self,
        bucket: 'GradientBucket',
        allreduce_fn: Callable,
        callback: Optional[Callable] = None
    ) -> Future:
        """
        Schedule a bucket with pipelining.

        Automatically manages pipeline stages for optimal overlap.
        """
        # Find available stage
        stage_idx = self._find_available_stage()

        # If no stage available, wait for one
        if stage_idx is None:
            self._wait_for_stage()
            stage_idx = self._find_available_stage()

        # Record stage
        self._active_stages[stage_idx] = bucket

        # Schedule with priority based on bucket index
        future = self.comm_scheduler.schedule_allreduce(
            bucket,
            allreduce_fn,
            callback=lambda b: self._stage_complete(stage_idx, b),
            priority=bucket.index
        )

        self._stage_futures[stage_idx] = future
        return future

    def _find_available_stage(self) -> Optional[int]:
        """Find an available pipeline stage."""
        for i, stage in enumerate(self._active_stages):
            if stage is None:
                return i
        return None

    def _wait_for_stage(self):
        """Wait for any pipeline stage to complete."""
        for i, future in enumerate(self._stage_futures):
            if future is not None:
                try:
                    future.result(timeout=0.001)
                    return
                except:
                    pass

        # Wait for first future
        if self._stage_futures[0] is not None:
            self._stage_futures[0].result()

    def _stage_complete(self, stage_idx: int, bucket: 'GradientBucket'):
        """Called when a pipeline stage completes."""
        self._active_stages[stage_idx] = None
        self._stage_futures[stage_idx] = None

        # Track bandwidth
        if bucket.size_bytes > 0:
            stats = self.comm_scheduler.get_stats()
            if stats.bucket_times_ms:
                last_time = stats.bucket_times_ms[-1]
                if last_time > 0:
                    bandwidth = (bucket.size_bytes * 8) / (last_time / 1000) / 1e9
                    self._bandwidth_samples.append(bandwidth)

    def flush(self):
        """Wait for all pipeline stages to complete."""
        for future in self._stage_futures:
            if future is not None:
                future.result()

        self._active_stages = [None] * self.num_pipeline_stages
        self._stage_futures = [None] * self.num_pipeline_stages

    def get_bandwidth_utilization(self) -> float:
        """Get average bandwidth utilization as a ratio of target."""
        if not self._bandwidth_samples:
            return 0.0
        avg_bandwidth = sum(self._bandwidth_samples) / len(self._bandwidth_samples)
        return min(1.0, avg_bandwidth / self._target_bandwidth_gbps)

    def set_target_bandwidth(self, bandwidth_gbps: float):
        """Set target bandwidth for utilization calculation."""
        self._target_bandwidth_gbps = bandwidth_gbps


@dataclass
class GradientBucket:
    """Bucket for gradient reduction."""
    index: int
    params: List[Any] = field(default_factory=list)
    grads: List[np.ndarray] = field(default_factory=list)
    size_bytes: int = 0
    pending: int = 0
    future: Any = None
    flat_tensor: Optional[np.ndarray] = None
    is_last: bool = False

    def flatten(self) -> np.ndarray:
        """Flatten all gradients into a single contiguous tensor."""
        flat_grads = []
        for param in self.params:
            if hasattr(param, 'grad') and param.grad is not None:
                flat_grads.append(param.grad.flatten())
            elif hasattr(param, 'data'):
                flat_grads.append(np.zeros(param.data.size))

        if flat_grads:
            self.flat_tensor = np.concatenate(flat_grads)
        else:
            self.flat_tensor = np.array([])

        return self.flat_tensor

    def unflatten(self):
        """Copy reduced gradients back to original tensors."""
        if self.flat_tensor is None:
            return

        offset = 0
        for param in self.params:
            if hasattr(param, 'data'):
                size = param.data.size
                if hasattr(param, 'grad') and param.grad is not None:
                    param.grad = self.flat_tensor[offset:offset + size].reshape(param.data.shape)
                offset += size


class Reducer:
    """
    Gradient reducer for distributed training.

    Features:
    - Bucketed all-reduce
    - Gradient compression
    - Overlap with backward
    - Async communication via CommunicationScheduler
    """

    def __init__(
        self,
        parameters: List[Any],
        process_group: ProcessGroup = None,
        bucket_cap_mb: float = 25.0,
        find_unused: bool = False,
        use_async: bool = False,
        comm_scheduler: Optional[CommunicationScheduler] = None
    ):
        self.parameters = list(parameters)
        self.process_group = process_group
        self.bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)
        self.find_unused = find_unused
        self.use_async = use_async

        self._lock = threading.Lock()
        self._buckets: List[GradientBucket] = []
        self._param_to_bucket: Dict[int, int] = {}

        # Communication scheduler for async operations
        self._comm_scheduler = comm_scheduler
        self._owns_scheduler = False
        if use_async and comm_scheduler is None:
            self._comm_scheduler = CommunicationScheduler(num_threads=2)
            self._owns_scheduler = True

        # Completion tracking
        self._completion_event = threading.Event()
        self._buckets_reduced = 0

        # Callbacks
        self._bucket_callbacks: List[Callable] = []

        self._rebuild_buckets()

    def _rebuild_buckets(self):
        """Build gradient buckets."""
        self._buckets = []
        current_bucket = GradientBucket(0)

        # Reverse order for overlap with backward
        for i, param in enumerate(reversed(self.parameters)):
            param_idx = len(self.parameters) - 1 - i
            param_size = param.data.nbytes if hasattr(param, 'data') else 0

            if current_bucket.size_bytes + param_size > self.bucket_cap_bytes and current_bucket.params:
                self._buckets.append(current_bucket)
                current_bucket = GradientBucket(len(self._buckets))

            current_bucket.params.append(param)
            current_bucket.size_bytes += param_size
            self._param_to_bucket[param_idx] = current_bucket.index

        if current_bucket.params:
            current_bucket.is_last = True
            self._buckets.append(current_bucket)

    def prepare_for_backward(self):
        """Prepare reducer for backward pass."""
        with self._lock:
            self._completion_event.clear()
            self._buckets_reduced = 0

            for bucket in self._buckets:
                bucket.grads.clear()
                bucket.pending = len(bucket.params)
                bucket.flat_tensor = None
                bucket.future = None

        if self._comm_scheduler:
            self._comm_scheduler.mark_backward_start()

    def mark_grad_ready(self, param_idx: int):
        """Mark a gradient as ready for reduction."""
        bucket_idx = self._param_to_bucket.get(param_idx)
        if bucket_idx is None:
            return

        bucket_to_reduce = None
        with self._lock:
            bucket = self._buckets[bucket_idx]
            bucket.pending -= 1

            if bucket.pending == 0:
                bucket_to_reduce = bucket

        # Reduce outside lock to avoid deadlock with callbacks
        if bucket_to_reduce is not None:
            if self.use_async and self._comm_scheduler:
                self._async_reduce_bucket(bucket_to_reduce)
            else:
                self._reduce_bucket(bucket_to_reduce)

    def _reduce_bucket(self, bucket: GradientBucket):
        """Reduce a bucket of gradients (synchronous)."""
        if not bucket.params:
            return

        # Flatten gradients
        flat_grads = []
        for param in bucket.params:
            if hasattr(param, 'grad') and param.grad is not None:
                flat_grads.append(param.grad.flatten())
            else:
                flat_grads.append(np.zeros(param.data.size))

        flat_tensor = np.concatenate(flat_grads)

        # Simulated all-reduce (average)
        if self.process_group:
            world_size = self.process_group.size()
            flat_tensor /= world_size

        # Unflatten back to gradients
        offset = 0
        for param in bucket.params:
            size = param.data.size
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad = flat_tensor[offset:offset + size].reshape(param.data.shape)
            offset += size

        # Callbacks
        self._on_bucket_complete(bucket)

    def _async_reduce_bucket(self, bucket: GradientBucket):
        """Reduce a bucket of gradients (asynchronous)."""
        if not bucket.params:
            self._on_bucket_complete(bucket)
            return

        # Flatten first
        bucket.flatten()

        def allreduce_fn(tensor):
            if self.process_group:
                world_size = self.process_group.size()
                return tensor / world_size
            return tensor

        def on_complete(b):
            b.unflatten()
            self._on_bucket_complete(b)

        bucket.future = self._comm_scheduler.schedule_allreduce(
            bucket,
            allreduce_fn,
            callback=on_complete,
            priority=bucket.index
        )

    def _on_bucket_complete(self, bucket: GradientBucket):
        """Called when a bucket reduction completes."""
        with self._lock:
            self._buckets_reduced += 1

            # Invoke callbacks
            for cb in self._bucket_callbacks:
                try:
                    cb(bucket)
                except Exception as e:
                    logger.error(f"Bucket callback failed: {e}")

            # Check if all done
            if self._buckets_reduced >= len(self._buckets):
                self._completion_event.set()

    def add_bucket_callback(self, callback: Callable):
        """Add callback for bucket completion."""
        self._bucket_callbacks.append(callback)

    def remove_bucket_callback(self, callback: Callable):
        """Remove bucket callback."""
        if callback in self._bucket_callbacks:
            self._bucket_callbacks.remove(callback)

    def finalize(self, timeout: Optional[float] = None):
        """Finalize reduction after backward."""
        if self._comm_scheduler:
            self._comm_scheduler.mark_backward_end()

        if self.use_async:
            # Wait for all async operations
            self._completion_event.wait(timeout=timeout)
            if self._comm_scheduler:
                self._comm_scheduler.wait_all(timeout=timeout)

    def get_stats(self) -> Optional[CommunicationStats]:
        """Get communication statistics."""
        if self._comm_scheduler:
            return self._comm_scheduler.get_stats()
        return None

    def reset_stats(self):
        """Reset communication statistics."""
        if self._comm_scheduler:
            self._comm_scheduler.reset_stats()

    def shutdown(self):
        """Shutdown the reducer."""
        if self._owns_scheduler and self._comm_scheduler:
            self._comm_scheduler.stop()
            self._comm_scheduler = None


class GradReducer:
    """
    High-performance gradient reducer with overlap.

    Features:
    - Overlap communication with computation
    - Gradient bucketing
    - Support for gradient compression
    """

    def __init__(
        self,
        parameters: Iterator[Any],
        process_group: ProcessGroup = None,
        strategy: AllReduceStrategy = AllReduceStrategy.BUCKET
    ):
        self.parameters = list(parameters)
        self.process_group = process_group
        self.strategy = strategy

        self._reducer = Reducer(
            self.parameters,
            process_group,
            bucket_cap_mb=25.0
        )

    def reduce_gradients(self):
        """Reduce all gradients."""
        self._reducer.prepare_for_backward()

        for i, param in enumerate(self.parameters):
            if hasattr(param, 'grad') and param.grad is not None:
                self._reducer.mark_grad_ready(i)

        self._reducer.finalize()

    def compress_gradients(self, ratio: float = 0.1):
        """Apply gradient compression."""
        for param in self.parameters:
            if hasattr(param, 'grad') and param.grad is not None:
                # Top-k sparsification
                grad = param.grad.flatten()
                k = int(len(grad) * ratio)
                if k > 0:
                    indices = np.argsort(np.abs(grad))[-k:]
                    sparse_grad = np.zeros_like(grad)
                    sparse_grad[indices] = grad[indices]
                    param.grad = sparse_grad.reshape(param.data.shape)


class DistributedDataParallel:
    """
    Distributed data parallel wrapper.

    Wraps a module for data parallel training across multiple GPUs/nodes.

    Features:
    - Automatic gradient synchronization
    - Overlap with backward pass
    - Gradient bucketing
    - Find unused parameters
    """

    def __init__(
        self,
        module: Any,
        device_ids: List[int] = None,
        output_device: int = None,
        process_group: ProcessGroup = None,
        bucket_cap_mb: float = 25.0,
        find_unused_parameters: bool = False,
        gradient_as_bucket_view: bool = False
    ):
        self.module = module
        self.device_ids = device_ids or [0]
        self.output_device = output_device or self.device_ids[0]
        self.process_group = process_group
        self.bucket_cap_mb = bucket_cap_mb
        self.find_unused_parameters = find_unused_parameters
        self.gradient_as_bucket_view = gradient_as_bucket_view

        # Get parameters
        if hasattr(module, 'parameters'):
            self._parameters = list(module.parameters())
        else:
            self._parameters = []

        # Create reducer
        self._reducer = Reducer(
            self._parameters,
            process_group,
            bucket_cap_mb,
            find_unused_parameters
        )

        # Synchronize initial parameters
        self._sync_params()

        # Hook for backward
        self._register_hooks()

    def _sync_params(self):
        """Synchronize parameters across workers."""
        # Simulated - in real impl would broadcast from rank 0
        pass

    def _register_hooks(self):
        """Register backward hooks for gradient reduction."""
        for i, param in enumerate(self._parameters):
            if hasattr(param, 'register_hook'):
                param.register_hook(lambda grad, idx=i: self._grad_hook(idx, grad))

    def _grad_hook(self, param_idx: int, grad: np.ndarray):
        """Hook called when gradient is ready."""
        self._reducer.mark_grad_ready(param_idx)
        return grad

    def forward(self, *inputs, **kwargs):
        """Forward pass."""
        self._reducer.prepare_for_backward()

        # Run module forward
        if hasattr(self.module, '__call__'):
            output = self.module(*inputs, **kwargs)
        elif hasattr(self.module, 'forward'):
            output = self.module.forward(*inputs, **kwargs)
        else:
            raise RuntimeError("Module has no forward method")

        return output

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def parameters(self) -> Iterator[Any]:
        """Get module parameters."""
        return iter(self._parameters)

    def no_sync(self):
        """Context manager to disable gradient sync."""
        return _NoSyncContext(self)

    def join(self):
        """Join any pending operations."""
        pass

    def state_dict(self) -> Dict:
        """Get state dict."""
        if hasattr(self.module, 'state_dict'):
            return self.module.state_dict()
        return {}

    def load_state_dict(self, state_dict: Dict):
        """Load state dict."""
        if hasattr(self.module, 'load_state_dict'):
            self.module.load_state_dict(state_dict)


class _NoSyncContext:
    """Context manager for disabling gradient sync."""

    def __init__(self, ddp: DistributedDataParallel):
        self.ddp = ddp
        self._old_reducer = None

    def __enter__(self):
        # Disable reducer
        self._old_reducer = self.ddp._reducer
        self.ddp._reducer = _DummyReducer()
        return self

    def __exit__(self, *args):
        self.ddp._reducer = self._old_reducer


class _DummyReducer:
    """Dummy reducer that does nothing."""

    def prepare_for_backward(self):
        pass

    def mark_grad_ready(self, idx):
        pass

    def finalize(self):
        pass


class FullyShardedDataParallel:
    """
    Fully Sharded Data Parallel (FSDP).

    Shards model parameters, gradients, and optimizer states
    across data parallel workers.
    """

    def __init__(
        self,
        module: Any,
        process_group: ProcessGroup = None,
        sharding_strategy: str = "FULL_SHARD",
        cpu_offload: bool = False,
        backward_prefetch: str = "BACKWARD_PRE"
    ):
        self.module = module
        self.process_group = process_group
        self.sharding_strategy = sharding_strategy
        self.cpu_offload = cpu_offload
        self.backward_prefetch = backward_prefetch

        # Get parameters
        if hasattr(module, 'parameters'):
            self._parameters = list(module.parameters())
        else:
            self._parameters = []

        # Shard parameters
        self._shard_parameters()

    def _shard_parameters(self):
        """Shard parameters across workers."""
        if not self.process_group:
            return

        world_size = self.process_group.size()
        rank = 0  # Would get from distributed context

        for param in self._parameters:
            if hasattr(param, 'data'):
                # Shard along first dimension
                shard_size = param.data.size // world_size
                start = rank * shard_size
                end = start + shard_size
                param._full_data = param.data
                param.data = param.data.flatten()[start:end]

    def _all_gather_params(self):
        """All-gather parameters before forward."""
        for param in self._parameters:
            if hasattr(param, '_full_data'):
                param.data = param._full_data

    def forward(self, *inputs, **kwargs):
        """Forward pass with all-gather."""
        self._all_gather_params()

        if hasattr(self.module, '__call__'):
            return self.module(*inputs, **kwargs)
        return self.module.forward(*inputs, **kwargs)

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def parameters(self) -> Iterator[Any]:
        return iter(self._parameters)


# =============================================================================
# Phase 5: Fault Tolerance
# =============================================================================

class WorkerState(Enum):
    """State of a distributed worker."""
    INITIALIZING = auto()
    RUNNING = auto()
    SUSPENDED = auto()
    FAILED = auto()
    TERMINATED = auto()


@dataclass
class Heartbeat:
    """Heartbeat message from a worker."""
    rank: int
    timestamp: float
    state: WorkerState
    iteration: int = 0
    memory_used_mb: float = 0.0

    def is_stale(self, timeout_seconds: float) -> bool:
        """Check if heartbeat is stale."""
        return (time.perf_counter() - self.timestamp) > timeout_seconds


@dataclass
class FailureEvent:
    """Event representing a worker failure."""
    failed_rank: int
    detected_at: float
    last_heartbeat: Optional[Heartbeat]
    reason: str
    recoverable: bool = True


class HealthMonitor:
    """
    Monitors health of distributed workers via heartbeats.

    Features:
    - Periodic heartbeat collection
    - Timeout-based failure detection
    - Failure notification callbacks
    """

    def __init__(
        self,
        process_group: ProcessGroup,
        heartbeat_interval: float = 5.0,
        timeout_seconds: float = 30.0,
        max_missed_heartbeats: int = 3
    ):
        self.process_group = process_group
        self.heartbeat_interval = heartbeat_interval
        self.timeout_seconds = timeout_seconds
        self.max_missed_heartbeats = max_missed_heartbeats

        self.rank = 0  # Would get from distributed context
        self.world_size = process_group.size() if process_group else 1

        # State tracking
        self._heartbeats: Dict[int, Heartbeat] = {}
        self._missed_counts: Dict[int, int] = {r: 0 for r in range(self.world_size)}
        self._worker_states: Dict[int, WorkerState] = {
            r: WorkerState.INITIALIZING for r in range(self.world_size)
        }
        self._failed_workers: List[int] = []

        # Threading
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

        # Callbacks
        self._failure_callbacks: List[Callable[[FailureEvent], None]] = []
        self._recovery_callbacks: List[Callable[[int], None]] = []

        # Current iteration for heartbeat
        self._current_iteration = 0

    def start(self):
        """Start health monitoring."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="health_monitor"
        )
        self._monitor_thread.start()
        logger.debug("HealthMonitor started")

    def stop(self):
        """Stop health monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=self.heartbeat_interval * 2)
            self._monitor_thread = None
        logger.debug("HealthMonitor stopped")

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self._send_heartbeat()
                self._check_heartbeats()
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            time.sleep(self.heartbeat_interval)

    def _send_heartbeat(self):
        """Send heartbeat to all workers."""
        heartbeat = Heartbeat(
            rank=self.rank,
            timestamp=time.perf_counter(),
            state=self._worker_states.get(self.rank, WorkerState.RUNNING),
            iteration=self._current_iteration
        )

        # In real implementation, would broadcast via process group
        # For simulation, just record locally
        with self._lock:
            self._heartbeats[self.rank] = heartbeat

    def receive_heartbeat(self, heartbeat: Heartbeat):
        """Receive heartbeat from another worker."""
        with self._lock:
            self._heartbeats[heartbeat.rank] = heartbeat
            self._missed_counts[heartbeat.rank] = 0

            # Update worker state
            if heartbeat.rank in self._failed_workers:
                # Worker recovered
                self._failed_workers.remove(heartbeat.rank)
                self._worker_states[heartbeat.rank] = heartbeat.state
                self._notify_recovery(heartbeat.rank)

    def _check_heartbeats(self):
        """Check for stale heartbeats."""
        current_time = time.perf_counter()

        with self._lock:
            for rank in range(self.world_size):
                if rank == self.rank:
                    continue

                heartbeat = self._heartbeats.get(rank)

                if heartbeat is None or heartbeat.is_stale(self.timeout_seconds):
                    self._missed_counts[rank] += 1

                    if self._missed_counts[rank] >= self.max_missed_heartbeats:
                        if rank not in self._failed_workers:
                            self._handle_failure(rank, heartbeat)

    def _handle_failure(self, rank: int, last_heartbeat: Optional[Heartbeat]):
        """Handle a detected worker failure."""
        self._failed_workers.append(rank)
        self._worker_states[rank] = WorkerState.FAILED

        event = FailureEvent(
            failed_rank=rank,
            detected_at=time.perf_counter(),
            last_heartbeat=last_heartbeat,
            reason=f"Missed {self.max_missed_heartbeats} heartbeats",
            recoverable=True
        )

        logger.warning(f"Worker {rank} failed: {event.reason}")

        # Notify callbacks (outside lock to avoid deadlock)
        callbacks = self._failure_callbacks.copy()

        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Failure callback error: {e}")

    def _notify_recovery(self, rank: int):
        """Notify that a worker has recovered."""
        callbacks = self._recovery_callbacks.copy()

        for callback in callbacks:
            try:
                callback(rank)
            except Exception as e:
                logger.error(f"Recovery callback error: {e}")

    def add_failure_callback(self, callback: Callable[[FailureEvent], None]):
        """Add callback for failure events."""
        self._failure_callbacks.append(callback)

    def add_recovery_callback(self, callback: Callable[[int], None]):
        """Add callback for recovery events."""
        self._recovery_callbacks.append(callback)

    def mark_iteration(self, iteration: int):
        """Mark current training iteration."""
        self._current_iteration = iteration

    def get_healthy_workers(self) -> List[int]:
        """Get list of healthy worker ranks."""
        with self._lock:
            return [r for r in range(self.world_size) if r not in self._failed_workers]

    def get_failed_workers(self) -> List[int]:
        """Get list of failed worker ranks."""
        with self._lock:
            return self._failed_workers.copy()

    def is_healthy(self, rank: int) -> bool:
        """Check if a specific worker is healthy."""
        with self._lock:
            return rank not in self._failed_workers

    @property
    def all_healthy(self) -> bool:
        """Check if all workers are healthy."""
        with self._lock:
            return len(self._failed_workers) == 0


class Checkpointer:
    """
    Manages training checkpoints for fault tolerance.

    Features:
    - Periodic checkpoint saving
    - Automatic cleanup of old checkpoints
    - Distributed checkpoint coordination
    - Async checkpoint writing
    """

    def __init__(
        self,
        save_dir: str,
        process_group: Optional[ProcessGroup] = None,
        save_interval: int = 1000,
        keep_last_n: int = 3,
        async_save: bool = True
    ):
        self.save_dir = save_dir
        self.process_group = process_group
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.async_save = async_save

        self.rank = 0  # Would get from distributed context
        self.is_main = (self.rank == 0)

        # State
        self._last_save_iteration = 0
        self._pending_saves: List[Future] = []
        self._executor: Optional[ThreadPoolExecutor] = None

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        if async_save:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpointer")

    def should_save(self, iteration: int) -> bool:
        """Check if checkpoint should be saved at this iteration."""
        return iteration > 0 and iteration % self.save_interval == 0

    def save(
        self,
        state: Dict[str, Any],
        iteration: int,
        force: bool = False
    ) -> Optional[str]:
        """
        Save checkpoint.

        Args:
            state: State dict to save
            iteration: Current training iteration
            force: Force save even if not at interval

        Returns:
            Path to saved checkpoint, or None if not saved
        """
        if not force and not self.should_save(iteration):
            return None

        # Only main process saves
        if not self.is_main:
            self._barrier()
            return None

        filename = f"checkpoint_{iteration:08d}.pt"
        filepath = os.path.join(self.save_dir, filename)

        # Add metadata
        state = state.copy()
        state['_checkpoint_meta'] = {
            'iteration': iteration,
            'timestamp': time.time(),
            'rank': self.rank
        }

        if self.async_save and self._executor:
            future = self._executor.submit(self._save_sync, state, filepath)
            self._pending_saves.append(future)
        else:
            self._save_sync(state, filepath)

        self._last_save_iteration = iteration

        # Update latest pointer
        self._update_latest(filename)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        # Barrier for distributed
        self._barrier()

        logger.info(f"Saved checkpoint: {filepath}")
        return filepath

    def _save_sync(self, state: Dict[str, Any], filepath: str):
        """Synchronous save operation."""
        # Write to temp file first
        temp_path = filepath + ".tmp"
        with open(temp_path, 'wb') as f:
            pickle.dump(state, f)

        # Atomic rename
        os.replace(temp_path, filepath)

    def _update_latest(self, filename: str):
        """Update pointer to latest checkpoint."""
        latest_path = os.path.join(self.save_dir, "latest.txt")
        with open(latest_path, 'w') as f:
            f.write(filename)

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted([
            f for f in os.listdir(self.save_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ])

        # Keep last N
        to_delete = checkpoints[:-self.keep_last_n] if len(checkpoints) > self.keep_last_n else []

        for old_ckpt in to_delete:
            try:
                os.remove(os.path.join(self.save_dir, old_ckpt))
                logger.debug(f"Removed old checkpoint: {old_ckpt}")
            except OSError as e:
                logger.warning(f"Failed to remove checkpoint {old_ckpt}: {e}")

    def _barrier(self):
        """Distributed barrier for synchronization."""
        if self.process_group:
            self.process_group.barrier()

    def load(self, filepath: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint.

        Args:
            filepath: Path to checkpoint, or None to load latest

        Returns:
            State dict, or None if no checkpoint found
        """
        if filepath is None:
            filepath = self._get_latest_checkpoint()
            if filepath is None:
                return None

        if not os.path.exists(filepath):
            return None

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        logger.info(f"Loaded checkpoint: {filepath}")
        return state

    def _get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        latest_path = os.path.join(self.save_dir, "latest.txt")
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                filename = f.read().strip()
            return os.path.join(self.save_dir, filename)

        # Fallback: find most recent checkpoint file
        checkpoints = sorted([
            f for f in os.listdir(self.save_dir)
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ])

        if checkpoints:
            return os.path.join(self.save_dir, checkpoints[-1])

        return None

    def wait_pending(self):
        """Wait for any pending async saves."""
        for future in self._pending_saves:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Async save failed: {e}")
        self._pending_saves.clear()

    def shutdown(self):
        """Shutdown checkpointer."""
        self.wait_pending()
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None


class RecoveryManager:
    """
    Manages recovery from failures during distributed training.

    Features:
    - Automatic restart from checkpoint
    - Parameter re-broadcast after recovery
    - State restoration
    - Elastic training support
    """

    def __init__(
        self,
        checkpointer: Checkpointer,
        health_monitor: Optional[HealthMonitor] = None,
        process_group: Optional[ProcessGroup] = None,
        auto_recover: bool = True
    ):
        self.checkpointer = checkpointer
        self.health_monitor = health_monitor
        self.process_group = process_group
        self.auto_recover = auto_recover

        self.rank = 0
        self.world_size = process_group.size() if process_group else 1

        # State
        self._is_recovering = False
        self._recovery_count = 0
        self._last_recovery_iteration = 0

        # Register failure callback if health monitor provided
        if health_monitor and auto_recover:
            health_monitor.add_failure_callback(self._on_failure)

    def _on_failure(self, event: FailureEvent):
        """Handle failure event."""
        if event.recoverable and self.auto_recover:
            logger.info(f"Initiating recovery for worker {event.failed_rank}")
            # In a real system, this would trigger recovery
            # For simulation, just log

    def save_state(
        self,
        model: Any,
        optimizer: Any = None,
        iteration: int = 0,
        extra_state: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Save training state for recovery.

        Args:
            model: Model to save
            optimizer: Optimizer to save (optional)
            iteration: Current iteration
            extra_state: Additional state to save

        Returns:
            Path to checkpoint, or None
        """
        state = {}

        # Model state
        if hasattr(model, 'state_dict'):
            state['model'] = model.state_dict()
        elif hasattr(model, 'module') and hasattr(model.module, 'state_dict'):
            state['model'] = model.module.state_dict()

        # Optimizer state
        if optimizer is not None and hasattr(optimizer, 'state_dict'):
            state['optimizer'] = optimizer.state_dict()

        # Training state
        state['iteration'] = iteration
        state['world_size'] = self.world_size
        state['recovery_count'] = self._recovery_count

        # Extra state
        if extra_state:
            state['extra'] = extra_state

        return self.checkpointer.save(state, iteration)

    def restore_state(
        self,
        model: Any,
        optimizer: Any = None,
        filepath: Optional[str] = None
    ) -> Optional[int]:
        """
        Restore training state from checkpoint.

        Args:
            model: Model to restore
            optimizer: Optimizer to restore (optional)
            filepath: Path to checkpoint (None for latest)

        Returns:
            Iteration number to resume from, or None if no checkpoint
        """
        state = self.checkpointer.load(filepath)
        if state is None:
            return None

        self._is_recovering = True

        # Restore model
        if 'model' in state:
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(state['model'])
            elif hasattr(model, 'module') and hasattr(model.module, 'load_state_dict'):
                model.module.load_state_dict(state['model'])

        # Restore optimizer
        if optimizer is not None and 'optimizer' in state:
            if hasattr(optimizer, 'load_state_dict'):
                optimizer.load_state_dict(state['optimizer'])

        # Get iteration
        iteration = state.get('iteration', 0)
        self._last_recovery_iteration = iteration
        self._recovery_count += 1

        # Broadcast parameters to ensure consistency
        self._broadcast_parameters(model)

        self._is_recovering = False

        logger.info(f"Restored from iteration {iteration}")
        return iteration

    def _broadcast_parameters(self, model: Any):
        """Broadcast parameters from rank 0 to all workers."""
        if not self.process_group or self.world_size <= 1:
            return

        # Get parameters
        if hasattr(model, 'parameters'):
            params = list(model.parameters())
        elif hasattr(model, 'module') and hasattr(model.module, 'parameters'):
            params = list(model.module.parameters())
        else:
            return

        # In real implementation, would broadcast via NCCL/Gloo
        # For simulation, just log
        logger.debug(f"Broadcasting {len(params)} parameters from rank 0")

    @property
    def is_recovering(self) -> bool:
        """Check if currently recovering."""
        return self._is_recovering

    @property
    def recovery_count(self) -> int:
        """Get number of recoveries performed."""
        return self._recovery_count


class FaultTolerantTrainer:
    """
    High-level wrapper for fault-tolerant distributed training.

    Combines DDP, health monitoring, checkpointing, and recovery
    into a single easy-to-use interface.
    """

    def __init__(
        self,
        model: Any,
        process_group: Optional[ProcessGroup] = None,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_interval: int = 1000,
        heartbeat_interval: float = 5.0,
        failure_timeout: float = 30.0,
        auto_recover: bool = True
    ):
        self.model = model
        self.process_group = process_group

        # Wrap model in DDP
        self.ddp = DistributedDataParallel(
            model,
            process_group=process_group
        )

        # Health monitoring
        self.health_monitor = HealthMonitor(
            process_group=process_group,
            heartbeat_interval=heartbeat_interval,
            timeout_seconds=failure_timeout
        ) if process_group else None

        # Checkpointing
        self.checkpointer = Checkpointer(
            save_dir=checkpoint_dir,
            process_group=process_group,
            save_interval=checkpoint_interval
        )

        # Recovery
        self.recovery_manager = RecoveryManager(
            checkpointer=self.checkpointer,
            health_monitor=self.health_monitor,
            process_group=process_group,
            auto_recover=auto_recover
        )

        # State
        self._iteration = 0
        self._optimizer = None
        self._is_training = False

    def set_optimizer(self, optimizer: Any):
        """Set optimizer for checkpointing."""
        self._optimizer = optimizer

    def start(self):
        """Start fault-tolerant training."""
        if self.health_monitor:
            self.health_monitor.start()
        self._is_training = True

    def stop(self):
        """Stop fault-tolerant training."""
        self._is_training = False
        if self.health_monitor:
            self.health_monitor.stop()
        self.checkpointer.shutdown()

    def forward(self, *inputs, **kwargs):
        """Forward pass through DDP model."""
        return self.ddp(*inputs, **kwargs)

    def step(self, iteration: Optional[int] = None):
        """
        Complete a training step.

        Call after backward pass and optimizer step.
        """
        if iteration is not None:
            self._iteration = iteration
        else:
            self._iteration += 1

        # Update health monitor
        if self.health_monitor:
            self.health_monitor.mark_iteration(self._iteration)

        # Periodic checkpoint
        self.recovery_manager.save_state(
            self.ddp,
            self._optimizer,
            self._iteration
        )

    def resume(self) -> int:
        """
        Resume training from latest checkpoint.

        Returns:
            Iteration to resume from
        """
        iteration = self.recovery_manager.restore_state(
            self.ddp,
            self._optimizer
        )

        if iteration is not None:
            self._iteration = iteration
            return iteration

        return 0

    @property
    def is_healthy(self) -> bool:
        """Check if all workers are healthy."""
        if self.health_monitor:
            return self.health_monitor.all_healthy
        return True

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


# =============================================================================
# Phase 6: Enterprise Features
# =============================================================================

# -----------------------------------------------------------------------------
# Communication Backends
# -----------------------------------------------------------------------------

class CommunicationBackend(Enum):
    """Available communication backends."""
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"
    SIMULATED = "simulated"


@dataclass
class BackendConfig:
    """Configuration for a communication backend."""
    backend_type: CommunicationBackend
    is_available: bool = True
    supports_gpu: bool = False
    supports_cpu: bool = True
    max_message_size: int = 2**30  # 1GB default
    requires_init: bool = True


class BackendRegistry:
    """Registry for available communication backends."""

    _backends: Dict[str, BackendConfig] = {
        "nccl": BackendConfig(
            backend_type=CommunicationBackend.NCCL,
            is_available=False,  # Requires CUDA
            supports_gpu=True,
            supports_cpu=False
        ),
        "gloo": BackendConfig(
            backend_type=CommunicationBackend.GLOO,
            is_available=True,
            supports_gpu=True,
            supports_cpu=True
        ),
        "mpi": BackendConfig(
            backend_type=CommunicationBackend.MPI,
            is_available=False,  # Requires MPI
            supports_gpu=False,
            supports_cpu=True
        ),
        "simulated": BackendConfig(
            backend_type=CommunicationBackend.SIMULATED,
            is_available=True,
            supports_gpu=False,
            supports_cpu=True,
            requires_init=False
        ),
    }

    @classmethod
    def get_backend(cls, name: str) -> Optional[BackendConfig]:
        """Get backend configuration by name."""
        return cls._backends.get(name.lower())

    @classmethod
    def list_available(cls) -> List[str]:
        """List available backends."""
        return [name for name, cfg in cls._backends.items() if cfg.is_available]

    @classmethod
    def register_backend(cls, name: str, config: BackendConfig):
        """Register a new backend."""
        cls._backends[name.lower()] = config


class AbstractBackend:
    """Abstract base class for communication backends."""

    def __init__(self, rank: int, world_size: int, config: Optional[BackendConfig] = None):
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self._initialized = False
        self._stream_sync_enabled = True

    def initialize(self) -> bool:
        """Initialize the backend."""
        self._initialized = True
        return True

    def shutdown(self):
        """Shutdown the backend."""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def all_reduce(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        """AllReduce operation."""
        raise NotImplementedError

    def all_gather(self, tensor: np.ndarray) -> List[np.ndarray]:
        """AllGather operation."""
        raise NotImplementedError

    def broadcast(self, tensor: np.ndarray, src: int = 0) -> np.ndarray:
        """Broadcast operation."""
        raise NotImplementedError

    def reduce_scatter(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        """ReduceScatter operation."""
        raise NotImplementedError

    def barrier(self):
        """Synchronization barrier."""
        pass

    def sync_stream(self):
        """Synchronize CUDA streams (for GPU backends)."""
        pass


class NCCLBackend(AbstractBackend):
    """
    NCCL backend for high-performance GPU communication.

    This is a simulation - real implementation would use torch.cuda.nccl
    or the nccl-py bindings.
    """

    def __init__(self, rank: int, world_size: int):
        config = BackendRegistry.get_backend("nccl")
        super().__init__(rank, world_size, config)

        # NCCL-specific state
        self._comm_handle = None
        self._streams: Dict[str, Any] = {}

    def initialize(self) -> bool:
        """Initialize NCCL communicator."""
        # In real implementation:
        # from torch.cuda import nccl
        # self._comm_handle = nccl.init(self.world_size, self.rank)

        logger.info(f"NCCLBackend initialized: rank={self.rank}, world_size={self.world_size}")
        self._initialized = True
        return True

    def all_reduce(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        """AllReduce using NCCL."""
        if not self._initialized:
            raise RuntimeError("NCCL backend not initialized")

        # Simulation: In real implementation would use:
        # nccl.all_reduce(tensor, op=nccl.sum, comm=self._comm_handle)

        # For simulation, just return the tensor (as if only one worker)
        logger.debug(f"NCCL AllReduce: shape={tensor.shape}, op={op}")
        return tensor

    def all_gather(self, tensor: np.ndarray) -> List[np.ndarray]:
        """AllGather using NCCL."""
        if not self._initialized:
            raise RuntimeError("NCCL backend not initialized")

        # Simulation: return copies
        logger.debug(f"NCCL AllGather: shape={tensor.shape}")
        return [tensor.copy() for _ in range(self.world_size)]

    def broadcast(self, tensor: np.ndarray, src: int = 0) -> np.ndarray:
        """Broadcast using NCCL."""
        if not self._initialized:
            raise RuntimeError("NCCL backend not initialized")

        logger.debug(f"NCCL Broadcast: shape={tensor.shape}, src={src}")
        return tensor.copy()

    def reduce_scatter(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        """ReduceScatter using NCCL."""
        if not self._initialized:
            raise RuntimeError("NCCL backend not initialized")

        # Simulation: return a chunk
        chunk_size = len(tensor.flatten()) // self.world_size
        start = self.rank * chunk_size
        end = start + chunk_size

        logger.debug(f"NCCL ReduceScatter: shape={tensor.shape}, op={op}")
        return tensor.flatten()[start:end]

    def sync_stream(self):
        """Synchronize CUDA streams."""
        # In real implementation:
        # torch.cuda.synchronize()
        logger.debug("NCCL stream sync")


class GlooBackend(AbstractBackend):
    """
    Gloo backend for CPU and GPU communication.

    Gloo is a collective communications library that supports both
    CPU and GPU operations.
    """

    def __init__(self, rank: int, world_size: int):
        config = BackendRegistry.get_backend("gloo")
        super().__init__(rank, world_size, config)

        self._store = None  # TCPStore for coordination
        self._context = None

    def initialize(self) -> bool:
        """Initialize Gloo backend."""
        # In real implementation:
        # import pygloo
        # self._store = pygloo.rendezvous.TCPStore(...)
        # self._context = pygloo.rendezvous.Context(self.rank, self.world_size)

        logger.info(f"GlooBackend initialized: rank={self.rank}, world_size={self.world_size}")
        self._initialized = True
        return True

    def all_reduce(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        """AllReduce using Gloo."""
        if not self._initialized:
            raise RuntimeError("Gloo backend not initialized")

        logger.debug(f"Gloo AllReduce: shape={tensor.shape}, op={op}")
        return tensor

    def all_gather(self, tensor: np.ndarray) -> List[np.ndarray]:
        """AllGather using Gloo."""
        if not self._initialized:
            raise RuntimeError("Gloo backend not initialized")

        logger.debug(f"Gloo AllGather: shape={tensor.shape}")
        return [tensor.copy() for _ in range(self.world_size)]

    def broadcast(self, tensor: np.ndarray, src: int = 0) -> np.ndarray:
        """Broadcast using Gloo."""
        if not self._initialized:
            raise RuntimeError("Gloo backend not initialized")

        logger.debug(f"Gloo Broadcast: shape={tensor.shape}, src={src}")
        return tensor.copy()

    def barrier(self):
        """Gloo barrier synchronization."""
        logger.debug("Gloo barrier")


class MPIBackend(AbstractBackend):
    """
    MPI backend for distributed communication.

    Uses MPI (Message Passing Interface) for communication.
    """

    def __init__(self, rank: int, world_size: int):
        config = BackendRegistry.get_backend("mpi")
        super().__init__(rank, world_size, config)

        self._comm = None  # MPI communicator

    def initialize(self) -> bool:
        """Initialize MPI backend."""
        # In real implementation:
        # from mpi4py import MPI
        # self._comm = MPI.COMM_WORLD

        logger.info(f"MPIBackend initialized: rank={self.rank}, world_size={self.world_size}")
        self._initialized = True
        return True

    def all_reduce(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        """AllReduce using MPI."""
        if not self._initialized:
            raise RuntimeError("MPI backend not initialized")

        logger.debug(f"MPI AllReduce: shape={tensor.shape}, op={op}")
        return tensor

    def all_gather(self, tensor: np.ndarray) -> List[np.ndarray]:
        """AllGather using MPI."""
        if not self._initialized:
            raise RuntimeError("MPI backend not initialized")

        logger.debug(f"MPI AllGather: shape={tensor.shape}")
        return [tensor.copy() for _ in range(self.world_size)]


class SimulatedBackend(AbstractBackend):
    """Simulated backend for testing without real distributed setup."""

    def __init__(self, rank: int = 0, world_size: int = 1):
        config = BackendRegistry.get_backend("simulated")
        super().__init__(rank, world_size, config)
        self._initialized = True

    def all_reduce(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        return tensor

    def all_gather(self, tensor: np.ndarray) -> List[np.ndarray]:
        return [tensor.copy() for _ in range(self.world_size)]

    def broadcast(self, tensor: np.ndarray, src: int = 0) -> np.ndarray:
        return tensor.copy()

    def reduce_scatter(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        chunk_size = max(1, len(tensor.flatten()) // self.world_size)
        start = self.rank * chunk_size
        end = min(start + chunk_size, len(tensor.flatten()))
        return tensor.flatten()[start:end]


def create_backend(backend_name: str, rank: int, world_size: int) -> AbstractBackend:
    """Factory function to create a communication backend."""
    backend_name = backend_name.lower()

    if backend_name == "nccl":
        return NCCLBackend(rank, world_size)
    elif backend_name == "gloo":
        return GlooBackend(rank, world_size)
    elif backend_name == "mpi":
        return MPIBackend(rank, world_size)
    elif backend_name == "simulated":
        return SimulatedBackend(rank, world_size)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")


# -----------------------------------------------------------------------------
# Gradient Compression
# -----------------------------------------------------------------------------

@dataclass
class CompressionStats:
    """Statistics for gradient compression."""
    original_size_bytes: int = 0
    compressed_size_bytes: int = 0
    num_compressions: int = 0
    total_compression_time_ms: float = 0.0
    total_decompression_time_ms: float = 0.0
    compression_errors: int = 0

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed to original size."""
        if self.original_size_bytes == 0:
            return 1.0
        return self.compressed_size_bytes / self.original_size_bytes

    @property
    def space_savings(self) -> float:
        """Percentage of space saved."""
        return 1.0 - self.compression_ratio

    @property
    def avg_compression_time_ms(self) -> float:
        """Average compression time."""
        if self.num_compressions == 0:
            return 0.0
        return self.total_compression_time_ms / self.num_compressions

    def reset(self):
        """Reset all statistics."""
        self.original_size_bytes = 0
        self.compressed_size_bytes = 0
        self.num_compressions = 0
        self.total_compression_time_ms = 0.0
        self.total_decompression_time_ms = 0.0
        self.compression_errors = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_size_bytes": self.original_size_bytes,
            "compressed_size_bytes": self.compressed_size_bytes,
            "compression_ratio": self.compression_ratio,
            "space_savings": self.space_savings,
            "num_compressions": self.num_compressions,
            "avg_compression_time_ms": self.avg_compression_time_ms,
        }


class GradientCompressor:
    """Base class for gradient compression algorithms."""

    def __init__(self, enable_stats: bool = True):
        self.enable_stats = enable_stats
        self._stats = CompressionStats()

    @property
    def stats(self) -> CompressionStats:
        return self._stats

    def compress(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compress a gradient tensor.

        Returns:
            Tuple of (compressed_tensor, metadata)
        """
        raise NotImplementedError

    def decompress(self, compressed: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress a gradient tensor."""
        raise NotImplementedError


class TopKCompressor(GradientCompressor):
    """
    Top-K sparsification compressor.

    Only communicates the K largest gradient elements,
    dramatically reducing communication volume.
    """

    def __init__(self, compression_ratio: float = 0.01, enable_stats: bool = True):
        super().__init__(enable_stats)
        self.compression_ratio = compression_ratio

    def compress(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compress by keeping top-K elements."""
        start_time = time.perf_counter()

        original_shape = tensor.shape
        flat = tensor.flatten()
        original_size = flat.nbytes

        # Calculate K
        k = max(1, int(len(flat) * self.compression_ratio))

        # Find top-k indices by absolute value
        abs_flat = np.abs(flat)
        indices = np.argpartition(abs_flat, -k)[-k:]
        values = flat[indices]

        # Pack into sparse representation
        compressed = np.stack([indices.astype(np.float64), values])

        metadata = {
            "original_shape": original_shape,
            "original_numel": len(flat),
            "k": k,
            "dtype": str(tensor.dtype),
        }

        # Update stats
        if self.enable_stats:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.original_size_bytes += original_size
            self._stats.compressed_size_bytes += compressed.nbytes
            self._stats.num_compressions += 1
            self._stats.total_compression_time_ms += elapsed

        return compressed, metadata

    def decompress(self, compressed: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress sparse representation."""
        start_time = time.perf_counter()

        indices = compressed[0].astype(np.int64)
        values = compressed[1]

        # Reconstruct dense tensor
        result = np.zeros(metadata["original_numel"])
        result[indices] = values

        if self.enable_stats:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.total_decompression_time_ms += elapsed

        return result.reshape(metadata["original_shape"])


class QuantizedCompressor(GradientCompressor):
    """
    Quantization-based compressor.

    Reduces precision to save bandwidth while maintaining
    reasonable accuracy.
    """

    def __init__(self, bits: int = 8, enable_stats: bool = True):
        super().__init__(enable_stats)
        self.bits = bits
        self.max_val = 2 ** (bits - 1) - 1

    def compress(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compress using quantization."""
        start_time = time.perf_counter()

        original_size = tensor.nbytes

        # Compute scale
        abs_max = np.abs(tensor).max()
        scale = abs_max / self.max_val if abs_max > 0 else 1.0

        # Quantize
        quantized = np.round(tensor / scale).astype(np.int8)

        metadata = {
            "scale": float(scale),
            "original_dtype": str(tensor.dtype),
            "original_shape": tensor.shape,
        }

        # Update stats
        if self.enable_stats:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.original_size_bytes += original_size
            self._stats.compressed_size_bytes += quantized.nbytes
            self._stats.num_compressions += 1
            self._stats.total_compression_time_ms += elapsed

        return quantized, metadata

    def decompress(self, compressed: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress quantized tensor."""
        start_time = time.perf_counter()

        scale = metadata["scale"]
        result = compressed.astype(np.float64) * scale

        if self.enable_stats:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.total_decompression_time_ms += elapsed

        return result.reshape(metadata["original_shape"])


class ErrorFeedbackCompressor(GradientCompressor):
    """
    Compression with error feedback.

    Stores the compression error and adds it to the next
    gradient, ensuring convergence despite lossy compression.
    """

    def __init__(self, base_compressor: GradientCompressor, enable_stats: bool = True):
        super().__init__(enable_stats)
        self.base_compressor = base_compressor
        self._error: Optional[np.ndarray] = None

    def compress(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compress with error feedback."""
        # Add previous error
        if self._error is not None and self._error.shape == tensor.shape:
            tensor = tensor + self._error

        # Compress using base compressor
        compressed, metadata = self.base_compressor.compress(tensor)

        # Compute and store error
        decompressed = self.base_compressor.decompress(compressed, metadata)
        self._error = tensor - decompressed

        # Propagate stats
        self._stats = self.base_compressor.stats

        return compressed, metadata

    def decompress(self, compressed: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress using base compressor."""
        return self.base_compressor.decompress(compressed, metadata)

    def reset_error(self):
        """Reset accumulated error."""
        self._error = None


class PowerSGDCompressor(GradientCompressor):
    """
    PowerSGD: Low-rank gradient compression.

    Uses power iteration to find a low-rank approximation
    of the gradient matrix.
    """

    def __init__(self, rank: int = 4, num_iters: int = 1, enable_stats: bool = True):
        super().__init__(enable_stats)
        self.matrix_rank = rank
        self.num_iters = num_iters
        self._q: Optional[np.ndarray] = None  # Right singular vectors

    def compress(self, tensor: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compress using low-rank approximation."""
        start_time = time.perf_counter()

        original_shape = tensor.shape
        original_size = tensor.nbytes

        # Reshape to matrix (flatten all but last dimension if needed)
        if len(original_shape) == 1:
            # 1D tensor: make it a column vector
            matrix = tensor.reshape(-1, 1)
        elif len(original_shape) == 2:
            matrix = tensor
        else:
            # Reshape to 2D
            matrix = tensor.reshape(original_shape[0], -1)

        m, n = matrix.shape
        r = min(self.matrix_rank, m, n)

        # Initialize Q if needed
        if self._q is None or self._q.shape != (n, r):
            self._q = np.random.randn(n, r)
            self._q, _ = np.linalg.qr(self._q)

        # Power iteration
        for _ in range(self.num_iters):
            p = matrix @ self._q
            self._q = matrix.T @ p
            self._q, _ = np.linalg.qr(self._q)

        # Compute P = M @ Q
        p = matrix @ self._q

        # Compressed representation is (P, Q)
        compressed = np.concatenate([p.flatten(), self._q.flatten()])

        metadata = {
            "original_shape": original_shape,
            "matrix_shape": (m, n),
            "rank": r,
            "p_shape": p.shape,
            "q_shape": self._q.shape,
        }

        # Update stats
        if self.enable_stats:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.original_size_bytes += original_size
            self._stats.compressed_size_bytes += compressed.nbytes
            self._stats.num_compressions += 1
            self._stats.total_compression_time_ms += elapsed

        return compressed, metadata

    def decompress(self, compressed: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress low-rank approximation."""
        start_time = time.perf_counter()

        p_size = np.prod(metadata["p_shape"])
        p = compressed[:p_size].reshape(metadata["p_shape"])
        q = compressed[p_size:].reshape(metadata["q_shape"])

        # Reconstruct: M ≈ P @ Q.T
        matrix = p @ q.T

        if self.enable_stats:
            elapsed = (time.perf_counter() - start_time) * 1000
            self._stats.total_decompression_time_ms += elapsed

        return matrix.reshape(metadata["original_shape"])


# -----------------------------------------------------------------------------
# Adaptive Communication
# -----------------------------------------------------------------------------

@dataclass
class NetworkMetrics:
    """Network performance metrics."""
    bandwidth_gbps: float = 10.0  # Default 10 Gbps
    latency_us: float = 1.0  # Default 1 microsecond
    jitter_us: float = 0.1
    packet_loss_rate: float = 0.0

    # Historical measurements
    bandwidth_history: List[float] = field(default_factory=list)
    latency_history: List[float] = field(default_factory=list)

    def update_bandwidth(self, value: float, window_size: int = 10):
        """Update bandwidth estimate."""
        self.bandwidth_history.append(value)
        if len(self.bandwidth_history) > window_size:
            self.bandwidth_history.pop(0)
        self.bandwidth_gbps = sum(self.bandwidth_history) / len(self.bandwidth_history)

    def update_latency(self, value: float, window_size: int = 10):
        """Update latency estimate."""
        self.latency_history.append(value)
        if len(self.latency_history) > window_size:
            self.latency_history.pop(0)
        self.latency_us = sum(self.latency_history) / len(self.latency_history)


class AdaptiveCommunicator:
    """
    Adapts communication strategy based on network conditions.

    Features:
    - Automatic bandwidth and latency estimation
    - Algorithm selection based on message size
    - Dynamic switching between strategies
    """

    def __init__(
        self,
        process_group: Optional[ProcessGroup] = None,
        initial_bandwidth_gbps: float = 10.0,
        initial_latency_us: float = 1.0
    ):
        self.process_group = process_group
        self.world_size = process_group.size() if process_group else 1

        # Network metrics
        self.metrics = NetworkMetrics(
            bandwidth_gbps=initial_bandwidth_gbps,
            latency_us=initial_latency_us
        )

        # Strategy selection thresholds (bytes)
        # For small messages, tree is better (lower latency)
        # For large messages, ring is better (higher bandwidth)
        self._small_message_threshold = 1024 * 1024  # 1MB

        # Statistics
        self._strategy_usage: Dict[str, int] = {
            "ring": 0,
            "tree": 0,
            "recursive_halving": 0,
        }

    def estimate_bandwidth(self, tensor_size_bytes: int, elapsed_time_s: float):
        """Update bandwidth estimate from a transfer."""
        if elapsed_time_s > 0:
            bandwidth_gbps = (tensor_size_bytes * 8) / elapsed_time_s / 1e9
            self.metrics.update_bandwidth(bandwidth_gbps)

    def estimate_latency(self, elapsed_time_us: float):
        """Update latency estimate."""
        self.metrics.update_latency(elapsed_time_us)

    def select_strategy(self, message_size_bytes: int) -> AllReduceStrategy:
        """Select optimal AllReduce strategy based on message size and network."""
        n = self.world_size

        # Estimate time for each algorithm
        ring_time = self._estimate_ring_time(message_size_bytes)
        tree_time = self._estimate_tree_time(message_size_bytes)
        rhd_time = self._estimate_rhd_time(message_size_bytes)

        # Choose fastest
        times = {
            AllReduceStrategy.RING: ring_time,
            AllReduceStrategy.TREE: tree_time,
            AllReduceStrategy.RECURSIVE_HALVING: rhd_time,
        }

        selected = min(times, key=times.get)

        # Update usage stats
        self._strategy_usage[selected.name.lower()] = \
            self._strategy_usage.get(selected.name.lower(), 0) + 1

        return selected

    def _estimate_ring_time(self, size_bytes: int) -> float:
        """Estimate ring AllReduce time in microseconds."""
        # Time = 2 * (n-1) * (latency + size / bandwidth / n)
        n = self.world_size
        bandwidth_bytes_per_us = self.metrics.bandwidth_gbps * 1e9 / 8 / 1e6

        return 2 * (n - 1) * (
            self.metrics.latency_us +
            size_bytes / bandwidth_bytes_per_us / n
        )

    def _estimate_tree_time(self, size_bytes: int) -> float:
        """Estimate tree AllReduce time in microseconds."""
        # Time = 2 * log(n) * (latency + size / bandwidth)
        import math
        n = max(self.world_size, 2)
        bandwidth_bytes_per_us = self.metrics.bandwidth_gbps * 1e9 / 8 / 1e6

        return 2 * math.log2(n) * (
            self.metrics.latency_us +
            size_bytes / bandwidth_bytes_per_us
        )

    def _estimate_rhd_time(self, size_bytes: int) -> float:
        """Estimate recursive halving-doubling time in microseconds."""
        import math
        n = max(self.world_size, 2)
        bandwidth_bytes_per_us = self.metrics.bandwidth_gbps * 1e9 / 8 / 1e6

        # Only optimal for power-of-2 world sizes
        if n & (n - 1) != 0:
            return float('inf')

        return math.log2(n) * (
            self.metrics.latency_us +
            size_bytes / bandwidth_bytes_per_us
        )

    def all_reduce(self, tensor: np.ndarray, op: str = "sum") -> np.ndarray:
        """Perform AllReduce with automatic strategy selection."""
        size_bytes = tensor.nbytes
        strategy = self.select_strategy(size_bytes)

        start_time = time.perf_counter()

        # In real implementation, would dispatch to appropriate backend
        # For simulation, just return the tensor
        result = tensor.copy()

        elapsed_s = time.perf_counter() - start_time
        self.estimate_bandwidth(size_bytes, elapsed_s)

        logger.debug(f"AdaptiveCommunicator: strategy={strategy.name}, "
                    f"size={size_bytes}B, time={elapsed_s*1000:.2f}ms")

        return result

    def get_strategy_stats(self) -> Dict[str, int]:
        """Get strategy usage statistics."""
        return self._strategy_usage.copy()


# -----------------------------------------------------------------------------
# Elastic Training
# -----------------------------------------------------------------------------

class MembershipChange(Enum):
    """Type of membership change in elastic training."""
    JOIN = auto()
    LEAVE = auto()
    FAILURE = auto()


@dataclass
class MembershipEvent:
    """Event representing a membership change."""
    change_type: MembershipChange
    rank: int
    timestamp: float
    new_world_size: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class ElasticTrainer:
    """
    Elastic training support for dynamic worker management.

    Features:
    - Dynamic worker join/leave
    - Automatic rebalancing
    - Checkpoint-based recovery
    - Epoch restart handling
    """

    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 64,
        checkpoint_dir: str = "/tmp/elastic_checkpoints",
        rebalance_strategy: str = "immediate"
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.checkpoint_dir = checkpoint_dir
        self.rebalance_strategy = rebalance_strategy

        # Current state
        self._current_workers: List[int] = []
        self._world_size = 0
        self._rank = 0

        # Membership tracking
        self._membership_history: List[MembershipEvent] = []
        self._pending_joins: List[int] = []
        self._pending_leaves: List[int] = []

        # Rebalancing state
        self._rebalance_in_progress = False
        self._last_rebalance_time: Optional[float] = None

        # Callbacks
        self._join_callbacks: List[Callable[[int], None]] = []
        self._leave_callbacks: List[Callable[[int], None]] = []
        self._rebalance_callbacks: List[Callable[[int], None]] = []

        # Threading
        self._lock = threading.Lock()

        os.makedirs(checkpoint_dir, exist_ok=True)

    def initialize(self, rank: int, world_size: int):
        """Initialize with current worker configuration."""
        self._rank = rank
        self._world_size = world_size
        self._current_workers = list(range(world_size))
        logger.info(f"ElasticTrainer initialized: rank={rank}, world_size={world_size}")

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def is_rebalancing(self) -> bool:
        return self._rebalance_in_progress

    def add_join_callback(self, callback: Callable[[int], None]):
        """Add callback for worker join events."""
        self._join_callbacks.append(callback)

    def add_leave_callback(self, callback: Callable[[int], None]):
        """Add callback for worker leave events."""
        self._leave_callbacks.append(callback)

    def add_rebalance_callback(self, callback: Callable[[int], None]):
        """Add callback for rebalancing events."""
        self._rebalance_callbacks.append(callback)

    def request_join(self, new_rank: int) -> bool:
        """Request a new worker to join."""
        with self._lock:
            if self._world_size >= self.max_workers:
                logger.warning(f"Cannot join: max workers ({self.max_workers}) reached")
                return False

            if new_rank in self._current_workers:
                logger.warning(f"Worker {new_rank} already in cluster")
                return False

            self._pending_joins.append(new_rank)
            logger.info(f"Worker {new_rank} requested to join")
            return True

    def request_leave(self, leaving_rank: int) -> bool:
        """Request a worker to leave."""
        with self._lock:
            if self._world_size <= self.min_workers:
                logger.warning(f"Cannot leave: min workers ({self.min_workers}) required")
                return False

            if leaving_rank not in self._current_workers:
                logger.warning(f"Worker {leaving_rank} not in cluster")
                return False

            self._pending_leaves.append(leaving_rank)
            logger.info(f"Worker {leaving_rank} requested to leave")
            return True

    def handle_failure(self, failed_rank: int):
        """Handle a worker failure."""
        with self._lock:
            if failed_rank in self._current_workers:
                event = MembershipEvent(
                    change_type=MembershipChange.FAILURE,
                    rank=failed_rank,
                    timestamp=time.perf_counter(),
                    new_world_size=self._world_size - 1
                )
                self._membership_history.append(event)
                self._current_workers.remove(failed_rank)
                self._world_size -= 1

                # Trigger rebalancing
                self._trigger_rebalance()

                logger.warning(f"Worker {failed_rank} failed, new world_size={self._world_size}")

    def commit_membership_changes(self) -> bool:
        """Apply pending membership changes."""
        with self._lock:
            if not self._pending_joins and not self._pending_leaves:
                return False

            # Process joins
            for new_rank in self._pending_joins:
                if self._world_size < self.max_workers:
                    self._current_workers.append(new_rank)
                    self._world_size += 1

                    event = MembershipEvent(
                        change_type=MembershipChange.JOIN,
                        rank=new_rank,
                        timestamp=time.perf_counter(),
                        new_world_size=self._world_size
                    )
                    self._membership_history.append(event)

                    for callback in self._join_callbacks:
                        callback(new_rank)

            # Process leaves
            for leaving_rank in self._pending_leaves:
                if leaving_rank in self._current_workers and self._world_size > self.min_workers:
                    self._current_workers.remove(leaving_rank)
                    self._world_size -= 1

                    event = MembershipEvent(
                        change_type=MembershipChange.LEAVE,
                        rank=leaving_rank,
                        timestamp=time.perf_counter(),
                        new_world_size=self._world_size
                    )
                    self._membership_history.append(event)

                    for callback in self._leave_callbacks:
                        callback(leaving_rank)

            # Clear pending
            self._pending_joins.clear()
            self._pending_leaves.clear()

            # Trigger rebalancing
            self._trigger_rebalance()

            return True

    def _trigger_rebalance(self):
        """Trigger data rebalancing after membership change."""
        if self.rebalance_strategy == "immediate":
            self._rebalance_data()
        elif self.rebalance_strategy == "deferred":
            # Would schedule rebalancing for later
            pass

    def _rebalance_data(self):
        """Rebalance data across workers."""
        self._rebalance_in_progress = True

        try:
            # Reassign ranks to be contiguous
            self._current_workers.sort()

            # In real implementation:
            # 1. Save checkpoint
            # 2. Redistribute data shards
            # 3. Broadcast new assignments
            # 4. Resume training

            for callback in self._rebalance_callbacks:
                callback(self._world_size)

            self._last_rebalance_time = time.perf_counter()
            logger.info(f"Rebalancing complete: {len(self._current_workers)} workers")

        finally:
            self._rebalance_in_progress = False

    def get_data_shard(self, total_samples: int) -> Tuple[int, int]:
        """Get the data shard for this worker."""
        samples_per_worker = total_samples // self._world_size
        start = self._rank * samples_per_worker

        # Last worker gets remaining samples
        if self._rank == self._world_size - 1:
            end = total_samples
        else:
            end = start + samples_per_worker

        return start, end

    def should_restart_epoch(self) -> bool:
        """Check if epoch should be restarted due to membership change."""
        # Restart if rebalancing happened recently
        if self._last_rebalance_time is not None:
            elapsed = time.perf_counter() - self._last_rebalance_time
            if elapsed < 1.0:  # Within 1 second
                return True
        return False

    def get_membership_history(self) -> List[MembershipEvent]:
        """Get membership change history."""
        return self._membership_history.copy()


# -----------------------------------------------------------------------------
# Enhanced FSDP
# -----------------------------------------------------------------------------

class ShardingStrategy(Enum):
    """FSDP sharding strategies."""
    FULL_SHARD = auto()  # Shard params, grads, and optimizer state
    SHARD_GRAD_OP = auto()  # Shard grads and optimizer state only
    NO_SHARD = auto()  # DDP-like behavior
    HYBRID_SHARD = auto()  # Shard within node, replicate across nodes


@dataclass
class FSDPConfig:
    """Configuration for FSDP."""
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    cpu_offload: bool = False
    mixed_precision: bool = False
    backward_prefetch: str = "BACKWARD_PRE"
    forward_prefetch: bool = False
    limit_all_gathers: bool = False
    use_orig_params: bool = False


class CPUOffloadPolicy:
    """Policy for offloading parameters/gradients to CPU."""

    def __init__(self, offload_params: bool = True, offload_grads: bool = True):
        self.offload_params = offload_params
        self.offload_grads = offload_grads

        # Pinned memory for faster CPU-GPU transfers
        self._pinned_buffers: Dict[int, np.ndarray] = {}

    def offload_to_cpu(self, tensor: np.ndarray, tensor_id: int) -> np.ndarray:
        """Offload tensor to CPU (pinned memory)."""
        # In real implementation, would use torch.cuda.pin_memory
        cpu_tensor = tensor.copy()
        self._pinned_buffers[tensor_id] = cpu_tensor
        return cpu_tensor

    def prefetch_to_gpu(self, tensor_id: int) -> Optional[np.ndarray]:
        """Prefetch tensor from CPU to GPU."""
        return self._pinned_buffers.get(tensor_id)

    def clear_buffer(self, tensor_id: int):
        """Clear a pinned buffer."""
        self._pinned_buffers.pop(tensor_id, None)

    def clear_all(self):
        """Clear all pinned buffers."""
        self._pinned_buffers.clear()


class MixedPrecisionPolicy:
    """Policy for mixed precision training in FSDP."""

    def __init__(
        self,
        param_dtype: str = "float32",
        reduce_dtype: str = "float16",
        buffer_dtype: str = "float32"
    ):
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.buffer_dtype = buffer_dtype

        # Mapping to numpy dtypes
        self._dtype_map = {
            "float32": np.float32,
            "float16": np.float16,
            "bfloat16": np.float32,  # numpy doesn't have bfloat16
            "float64": np.float64,
        }

    def cast_for_compute(self, tensor: np.ndarray) -> np.ndarray:
        """Cast tensor for computation."""
        target_dtype = self._dtype_map.get(self.param_dtype, np.float32)
        return tensor.astype(target_dtype)

    def cast_for_reduce(self, tensor: np.ndarray) -> np.ndarray:
        """Cast tensor for reduction operations."""
        target_dtype = self._dtype_map.get(self.reduce_dtype, np.float16)
        return tensor.astype(target_dtype)

    def cast_for_storage(self, tensor: np.ndarray) -> np.ndarray:
        """Cast tensor for storage."""
        target_dtype = self._dtype_map.get(self.buffer_dtype, np.float32)
        return tensor.astype(target_dtype)


class EnhancedFSDP:
    """
    Enhanced Fully Sharded Data Parallel implementation.

    Features:
    - Multiple sharding strategies
    - CPU offloading
    - Mixed precision support
    - Prefetching for overlap
    """

    def __init__(
        self,
        module: Any,
        process_group: Optional[ProcessGroup] = None,
        config: Optional[FSDPConfig] = None
    ):
        self.module = module
        self.process_group = process_group
        self.config = config or FSDPConfig()

        # Get rank and world size
        self.rank = 0
        self.world_size = process_group.size() if process_group else 1

        # Policies
        self._cpu_offload = CPUOffloadPolicy() if self.config.cpu_offload else None
        self._mixed_precision = MixedPrecisionPolicy() if self.config.mixed_precision else None

        # Parameter management
        self._flat_params: Optional[np.ndarray] = None
        self._param_shapes: List[Tuple] = []
        self._param_numels: List[int] = []
        self._shard_size: int = 0
        self._local_shard: Optional[np.ndarray] = None

        # Gather state
        self._gathered_params: Optional[np.ndarray] = None
        self._is_gathered = False

        # Initialize
        self._init_params()

    def _init_params(self):
        """Initialize parameter sharding."""
        if hasattr(self.module, 'parameters'):
            params = list(self.module.parameters())
        else:
            params = []

        if not params:
            return

        # Flatten all parameters
        flat_parts = []
        for p in params:
            if hasattr(p, 'data'):
                data = p.data if isinstance(p.data, np.ndarray) else np.array(p.data)
                flat_parts.append(data.flatten())
                self._param_shapes.append(p.data.shape if hasattr(p.data, 'shape') else ())
                self._param_numels.append(data.size)

        if flat_parts:
            self._flat_params = np.concatenate(flat_parts)

            # Calculate shard size
            total_numel = len(self._flat_params)
            self._shard_size = (total_numel + self.world_size - 1) // self.world_size

            # Get local shard
            start = self.rank * self._shard_size
            end = min(start + self._shard_size, total_numel)
            self._local_shard = self._flat_params[start:end].copy()

            # Optionally offload to CPU
            if self._cpu_offload:
                self._local_shard = self._cpu_offload.offload_to_cpu(
                    self._local_shard, tensor_id=0
                )

            logger.debug(f"FSDP initialized: total_numel={total_numel}, "
                        f"shard_size={self._shard_size}, local_shard={len(self._local_shard)}")

    def _all_gather_params(self):
        """All-gather parameters before forward pass."""
        if self._is_gathered:
            return

        if self._local_shard is None:
            return

        # Prefetch from CPU if offloaded
        if self._cpu_offload:
            shard = self._cpu_offload.prefetch_to_gpu(tensor_id=0)
            if shard is not None:
                self._local_shard = shard

        # All-gather simulation
        # In real implementation: dist.all_gather(...)
        shards = [self._local_shard.copy() for _ in range(self.world_size)]
        self._gathered_params = np.concatenate(shards)

        # Trim to original size
        if self._flat_params is not None:
            self._gathered_params = self._gathered_params[:len(self._flat_params)]

        self._is_gathered = True
        logger.debug("FSDP: Parameters gathered")

    def _reduce_scatter_grads(self, grads: np.ndarray):
        """Reduce-scatter gradients after backward pass."""
        # Apply mixed precision if enabled
        if self._mixed_precision:
            grads = self._mixed_precision.cast_for_reduce(grads)

        # Reduce-scatter simulation
        # In real implementation: dist.reduce_scatter(...)
        chunk_size = self._shard_size
        start = self.rank * chunk_size
        end = min(start + chunk_size, len(grads))

        local_grad = grads[start:end].copy()

        # Cast back for storage
        if self._mixed_precision:
            local_grad = self._mixed_precision.cast_for_storage(local_grad)

        logger.debug("FSDP: Gradients reduce-scattered")
        return local_grad

    def _free_gathered_params(self):
        """Free gathered parameters to save memory."""
        self._gathered_params = None
        self._is_gathered = False
        logger.debug("FSDP: Gathered parameters freed")

    def forward(self, *inputs, **kwargs):
        """Forward pass with all-gather."""
        # Gather all parameters
        self._all_gather_params()

        # Apply mixed precision if enabled
        if self._mixed_precision and self._gathered_params is not None:
            self._gathered_params = self._mixed_precision.cast_for_compute(
                self._gathered_params
            )

        # Unflatten and update module parameters
        # (In real implementation, would update module's parameter views)

        # Call module forward
        if hasattr(self.module, '__call__'):
            result = self.module(*inputs, **kwargs)
        else:
            result = self.module.forward(*inputs, **kwargs)

        # Register backward hook to reduce-scatter gradients
        # (In real implementation, would use torch hooks)

        # Free gathered params if using FULL_SHARD
        if self.config.sharding_strategy == ShardingStrategy.FULL_SHARD:
            self._free_gathered_params()

        return result

    def __call__(self, *inputs, **kwargs):
        return self.forward(*inputs, **kwargs)

    def parameters(self) -> Iterator[Any]:
        """Iterate over parameters."""
        if hasattr(self.module, 'parameters'):
            return self.module.parameters()
        return iter([])

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict with full (unsharded) parameters."""
        self._all_gather_params()

        state = {
            "flat_params": self._gathered_params,
            "param_shapes": self._param_shapes,
            "param_numels": self._param_numels,
            "config": {
                "sharding_strategy": self.config.sharding_strategy.name,
                "cpu_offload": self.config.cpu_offload,
                "mixed_precision": self.config.mixed_precision,
            }
        }

        return state

    def load_state_dict(self, state: Dict[str, Any]):
        """Load state dict and re-shard."""
        if "flat_params" in state:
            self._flat_params = state["flat_params"]
            self._param_shapes = state.get("param_shapes", [])
            self._param_numels = state.get("param_numels", [])

            # Re-calculate shard
            total_numel = len(self._flat_params)
            self._shard_size = (total_numel + self.world_size - 1) // self.world_size

            start = self.rank * self._shard_size
            end = min(start + self._shard_size, total_numel)
            self._local_shard = self._flat_params[start:end].copy()

            self._is_gathered = False
            logger.debug("FSDP: State dict loaded and re-sharded")
