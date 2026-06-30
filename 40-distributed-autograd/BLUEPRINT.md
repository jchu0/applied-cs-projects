# Distributed Autograd Engine - Technical Blueprint

## Executive Summary

A distributed training framework inspired by Horovod, designed to enable efficient data-parallel training across multiple GPUs and nodes. This system implements AllReduce algorithms, gradient bucketization, communication/computation overlap, and fault tolerance to achieve near-linear scaling on multi-GPU clusters while maintaining ease of use.

> **Concepts covered:** [§03 Data parallelism](../../03-machine-learning-engineering/05-distributed-training/data-parallelism/data-parallelism.md) (DDP / FSDP / AllReduce — this project *implements* the patterns the tutorial describes) · [§03 Model parallelism](../../03-machine-learning-engineering/05-distributed-training/model-parallelism/model-parallelism.md) (pipeline parallelism). For the parameter-server alternative, see [Project 30](../30-parameter-server/); for orchestration, [Project 04](../04-ml-training-orchestrator/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## System Architecture

### High-Level Architecture

```
+----------------------------------------------------------+
|                    User Model/Optimizer                   |
|  DistributedDataParallel(model)                          |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                   Gradient Hook Layer                     |
|  (Captures gradients as they become ready)               |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                   Bucketing Engine                        |
|  +-------------+  +-------------+  +------------------+  |
|  | Bucket      |  | Fusion      |  | Ready Queue      |  |
|  | Builder     |  | Logic       |  | (FIFO)           |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                   Communication Scheduler                 |
|  (Overlaps AllReduce with backward pass)                 |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    AllReduce Engine                       |
|  +-------------+  +-------------+  +------------------+  |
|  | Ring        |  | Tree        |  | Parameter       |  |
|  | AllReduce   |  | AllReduce   |  | Server          |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    Communication Backend                  |
|  +-------------+  +-------------+  +------------------+  |
|  | NCCL        |  | Gloo        |  | MPI              |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
```

### Core Design Principles

1. **Communication/Computation Overlap**: Start AllReduce while backward is still running
2. **Gradient Bucketization**: Fuse small tensors to reduce communication overhead
3. **Backend Abstraction**: Support multiple communication libraries
4. **Elastic Training**: Handle worker failures gracefully
5. **Minimal User Code Changes**: Wrap model once, train normally

## Component Design

### 1. Gradient Bucketing

```python
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
import threading
import torch
import torch.distributed as dist

@dataclass
class GradientBucket:
    """A bucket containing multiple gradient tensors for fused communication"""
    bucket_id: int
    tensors: List[torch.Tensor] = field(default_factory=list)
    params: List[torch.nn.Parameter] = field(default_factory=list)
    numel: int = 0
    flat_tensor: Optional[torch.Tensor] = None
    future: Optional[torch.futures.Future] = None
    ready_count: int = 0
    is_last: bool = False

    def add_gradient(self, param: torch.nn.Parameter):
        """Add a parameter's gradient to this bucket"""
        grad = param.grad
        self.tensors.append(grad)
        self.params.append(param)
        self.numel += grad.numel()
        self.ready_count += 1

    def is_ready(self) -> bool:
        """Check if all gradients in bucket are ready"""
        return self.ready_count == len(self.params)

    def flatten(self):
        """Flatten all gradients into a single contiguous tensor"""
        if self.flat_tensor is None:
            self.flat_tensor = torch.empty(
                self.numel,
                dtype=self.tensors[0].dtype,
                device=self.tensors[0].device
            )

        # Copy gradients into flat tensor
        offset = 0
        for grad in self.tensors:
            numel = grad.numel()
            self.flat_tensor[offset:offset + numel].copy_(grad.view(-1))
            offset += numel

    def unflatten(self):
        """Copy reduced gradients back to original tensors"""
        offset = 0
        for grad in self.tensors:
            numel = grad.numel()
            grad.copy_(self.flat_tensor[offset:offset + numel].view_as(grad))
            offset += numel


class BucketBuilder:
    """
    Builds gradient buckets from model parameters.
    Groups parameters by type and size for efficient communication.
    """

    def __init__(self, parameters: List[torch.nn.Parameter],
                 bucket_size_mb: float = 25.0):
        self.parameters = list(parameters)
        self.bucket_size_bytes = int(bucket_size_mb * 1024 * 1024)
        self.buckets: List[GradientBucket] = []
        self.param_to_bucket: Dict[torch.nn.Parameter, int] = {}

        self._build_buckets()

    def _build_buckets(self):
        """Group parameters into buckets"""
        # Group by dtype and device
        param_groups: Dict[tuple, List[torch.nn.Parameter]] = {}
        for param in self.parameters:
            if param.requires_grad:
                key = (param.dtype, param.device)
                if key not in param_groups:
                    param_groups[key] = []
                param_groups[key].append(param)

        bucket_id = 0
        for (dtype, device), params in param_groups.items():
            # Sort by parameter size (largest first for better overlap)
            params = sorted(params, key=lambda p: -p.numel())

            current_bucket = GradientBucket(bucket_id=bucket_id)
            current_size = 0

            for param in params:
                param_size = param.numel() * param.element_size()

                # Start new bucket if this one is full
                if current_size + param_size > self.bucket_size_bytes and current_size > 0:
                    self.buckets.append(current_bucket)
                    bucket_id += 1
                    current_bucket = GradientBucket(bucket_id=bucket_id)
                    current_size = 0

                current_bucket.params.append(param)
                current_bucket.tensors.append(None)  # Placeholder
                current_bucket.numel += param.numel()
                current_size += param_size

                self.param_to_bucket[param] = bucket_id

            if current_bucket.params:
                self.buckets.append(current_bucket)
                bucket_id += 1

        # Mark last bucket
        if self.buckets:
            self.buckets[-1].is_last = True

    def get_bucket(self, param: torch.nn.Parameter) -> GradientBucket:
        """Get the bucket containing a parameter"""
        bucket_id = self.param_to_bucket[param]
        return self.buckets[bucket_id]


class GradientReducer:
    """
    Manages gradient reduction with bucketing and overlap.
    """

    def __init__(self, module: torch.nn.Module,
                 process_group: dist.ProcessGroup,
                 bucket_size_mb: float = 25.0):
        self.module = module
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)

        # Build buckets from parameters
        params = [p for p in module.parameters() if p.requires_grad]
        self.bucket_builder = BucketBuilder(params, bucket_size_mb)
        self.buckets = self.bucket_builder.buckets

        # Register gradient hooks
        self._register_hooks()

        # State for current backward pass
        self.pending_buckets: List[GradientBucket] = []
        self.lock = threading.Lock()
        self.backward_complete = threading.Event()

    def _register_hooks(self):
        """Register hooks on parameters to track gradient readiness"""
        for param in self.bucket_builder.parameters:
            if param.requires_grad:
                # Use accumulate_grad hook to detect when gradient is ready
                param.register_hook(self._make_grad_hook(param))

    def _make_grad_hook(self, param: torch.nn.Parameter):
        """Create a hook function for a parameter"""
        def hook(grad):
            bucket = self.bucket_builder.get_bucket(param)

            with self.lock:
                # Update bucket with gradient
                bucket_idx = bucket.params.index(param)
                bucket.tensors[bucket_idx] = grad
                bucket.ready_count += 1

                # Check if bucket is ready
                if bucket.ready_count == len(bucket.params):
                    self._launch_bucket_allreduce(bucket)

                    # Check if all done
                    if bucket.is_last:
                        self.backward_complete.set()

            return grad

        return hook

    def _launch_bucket_allreduce(self, bucket: GradientBucket):
        """Launch async AllReduce for a ready bucket"""
        # Flatten gradients
        bucket.flatten()

        # Launch async AllReduce
        bucket.future = dist.all_reduce(
            bucket.flat_tensor,
            group=self.process_group,
            async_op=True
        )

        self.pending_buckets.append(bucket)

    def wait_for_completion(self):
        """Wait for all gradient reductions to complete"""
        # Wait for backward pass to signal completion
        self.backward_complete.wait()

        # Wait for all AllReduce operations
        for bucket in self.pending_buckets:
            bucket.future.wait()

            # Average gradients
            bucket.flat_tensor.div_(self.world_size)

            # Copy back to original tensors
            bucket.unflatten()

        # Reset state for next iteration
        self._reset()

    def _reset(self):
        """Reset state for next backward pass"""
        self.pending_buckets = []
        self.backward_complete.clear()

        for bucket in self.buckets:
            bucket.ready_count = 0
            bucket.future = None
```

### 2. AllReduce Algorithms

```python
import torch
import torch.distributed as dist
from typing import List
import math

class AllReduceAlgorithm:
    """Base class for AllReduce algorithms"""

    def __init__(self, process_group: dist.ProcessGroup):
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class RingAllReduce(AllReduceAlgorithm):
    """
    Ring AllReduce algorithm.
    Optimal bandwidth utilization with 2*(n-1)/n efficiency.
    """

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform ring AllReduce on tensor"""
        # Split tensor into world_size chunks
        chunks = tensor.chunk(self.world_size)
        chunk_sizes = [c.numel() for c in chunks]

        # Allocate send/recv buffers
        send_buf = torch.zeros_like(chunks[0])
        recv_buf = torch.zeros_like(chunks[0])

        # Ring reduce-scatter phase
        # Each iteration, rank i sends chunk (i-k) to rank (i+1)
        # and receives chunk (i-k-1) from rank (i-1)
        for k in range(self.world_size - 1):
            send_chunk_idx = (self.rank - k) % self.world_size
            recv_chunk_idx = (self.rank - k - 1) % self.world_size

            send_rank = (self.rank + 1) % self.world_size
            recv_rank = (self.rank - 1) % self.world_size

            # Copy to send buffer
            send_buf.copy_(chunks[send_chunk_idx])

            # Async send/recv
            send_req = dist.isend(send_buf, send_rank, group=self.process_group)
            dist.recv(recv_buf, recv_rank, group=self.process_group)
            send_req.wait()

            # Accumulate received data
            chunks[recv_chunk_idx].add_(recv_buf)

        # Ring all-gather phase
        # Each iteration, rank i sends its complete chunk to rank (i+1)
        for k in range(self.world_size - 1):
            send_chunk_idx = (self.rank - k + 1) % self.world_size
            recv_chunk_idx = (self.rank - k) % self.world_size

            send_rank = (self.rank + 1) % self.world_size
            recv_rank = (self.rank - 1) % self.world_size

            send_buf.copy_(chunks[send_chunk_idx])

            send_req = dist.isend(send_buf, send_rank, group=self.process_group)
            dist.recv(recv_buf, recv_rank, group=self.process_group)
            send_req.wait()

            chunks[recv_chunk_idx].copy_(recv_buf)

        # Reconstruct tensor
        return torch.cat([c.view(-1) for c in chunks])


class TreeAllReduce(AllReduceAlgorithm):
    """
    Tree AllReduce algorithm.
    Lower latency for small messages (O(log n) vs O(n)).
    """

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform tree AllReduce using reduce + broadcast"""
        # Reduce to root
        self._tree_reduce(tensor, root=0)

        # Broadcast from root
        self._tree_broadcast(tensor, root=0)

        return tensor

    def _tree_reduce(self, tensor: torch.Tensor, root: int):
        """Reduce to root using binary tree"""
        recv_buf = torch.zeros_like(tensor)
        mask = 1

        while mask < self.world_size:
            if self.rank & mask:
                # Send to parent
                parent = self.rank - mask
                if parent >= 0:
                    dist.send(tensor, parent, group=self.process_group)
                break
            else:
                # Receive from child
                child = self.rank + mask
                if child < self.world_size:
                    dist.recv(recv_buf, child, group=self.process_group)
                    tensor.add_(recv_buf)
            mask <<= 1

    def _tree_broadcast(self, tensor: torch.Tensor, root: int):
        """Broadcast from root using binary tree"""
        mask = 1 << (self.world_size - 1).bit_length()

        while mask > 0:
            if self.rank < mask:
                child = self.rank + mask
                if child < self.world_size:
                    dist.send(tensor, child, group=self.process_group)
            else:
                parent = self.rank - mask
                if parent >= 0:
                    dist.recv(tensor, parent, group=self.process_group)
            mask >>= 1


class RecursiveHalvingDoubling(AllReduceAlgorithm):
    """
    Recursive halving-doubling AllReduce.
    Optimal for power-of-2 process counts.
    """

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Perform recursive halving-doubling AllReduce"""
        n = self.world_size

        if n & (n - 1) != 0:
            raise ValueError("World size must be power of 2")

        # Recursive halving (reduce-scatter)
        distance = 1
        while distance < n:
            partner = self.rank ^ distance

            # Exchange data
            send_buf = tensor.clone()
            recv_buf = torch.zeros_like(tensor)

            dist.sendrecv(send_buf, partner, recv_buf, partner, group=self.process_group)

            # Reduce appropriate half
            if self.rank < partner:
                # Keep first half, reduce with received second half
                mid = tensor.numel() // 2
                tensor[:mid].add_(recv_buf[:mid])
            else:
                # Keep second half
                mid = tensor.numel() // 2
                tensor[mid:].add_(recv_buf[mid:])

            distance <<= 1

        # Recursive doubling (all-gather)
        distance = n // 2
        while distance > 0:
            partner = self.rank ^ distance

            send_buf = tensor.clone()
            recv_buf = torch.zeros_like(tensor)

            dist.sendrecv(send_buf, partner, recv_buf, partner, group=self.process_group)

            # Combine halves
            if self.rank < partner:
                mid = tensor.numel() // 2
                tensor[mid:].copy_(recv_buf[mid:])
            else:
                mid = tensor.numel() // 2
                tensor[:mid].copy_(recv_buf[:mid])

            distance >>= 1

        return tensor
```

### 3. Communication Scheduler

```python
import queue
import threading
from typing import List, Optional
import time

class CommunicationScheduler:
    """
    Schedules communication to overlap with computation.
    Uses background threads to manage async operations.
    """

    def __init__(self, num_threads: int = 2):
        self.num_threads = num_threads
        self.work_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.threads: List[threading.Thread] = []
        self.running = False

        # Statistics
        self.total_comm_time = 0
        self.total_overlap_time = 0

    def start(self):
        """Start worker threads"""
        self.running = True
        for i in range(self.num_threads):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self):
        """Stop worker threads"""
        self.running = False
        for _ in self.threads:
            self.work_queue.put(None)  # Sentinel to stop threads
        for t in self.threads:
            t.join()

    def schedule_allreduce(self, bucket: GradientBucket,
                          allreduce_fn: callable,
                          callback: Optional[callable] = None):
        """Schedule an AllReduce operation"""
        work_item = {
            'bucket': bucket,
            'allreduce_fn': allreduce_fn,
            'callback': callback,
            'start_time': time.time()
        }
        self.work_queue.put(work_item)

    def _worker(self):
        """Worker thread that executes AllReduce operations"""
        while self.running:
            try:
                work_item = self.work_queue.get(timeout=0.1)
                if work_item is None:
                    break

                bucket = work_item['bucket']
                allreduce_fn = work_item['allreduce_fn']
                callback = work_item['callback']

                # Execute AllReduce
                comm_start = time.time()
                result = allreduce_fn(bucket.flat_tensor)
                comm_end = time.time()

                # Update statistics
                self.total_comm_time += comm_end - comm_start
                self.total_overlap_time += comm_start - work_item['start_time']

                # Copy result back
                bucket.flat_tensor.copy_(result)

                # Invoke callback if provided
                if callback:
                    callback(bucket)

                self.work_queue.task_done()

            except queue.Empty:
                continue

    def wait_all(self):
        """Wait for all scheduled operations to complete"""
        self.work_queue.join()

    def get_stats(self) -> dict:
        """Get communication statistics"""
        return {
            'total_comm_time_ms': self.total_comm_time * 1000,
            'total_overlap_time_ms': self.total_overlap_time * 1000,
            'overlap_ratio': self.total_overlap_time / max(self.total_comm_time, 1e-6)
        }
```

### 4. Distributed Data Parallel Wrapper

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, List

class DistributedDataParallel(nn.Module):
    """
    Wrapper for distributed data parallel training.
    Similar to PyTorch DDP but simplified for learning.
    """

    def __init__(self,
                 module: nn.Module,
                 device_ids: Optional[List[int]] = None,
                 process_group: Optional[dist.ProcessGroup] = None,
                 bucket_size_mb: float = 25.0,
                 find_unused_parameters: bool = False):
        super().__init__()

        self.module = module
        self.device_ids = device_ids
        self.process_group = process_group or dist.distributed_c10d._get_default_group()
        self.find_unused_parameters = find_unused_parameters

        # Move to device if specified
        if device_ids:
            self.device = torch.device(f'cuda:{device_ids[0]}')
            self.module.to(self.device)
        else:
            self.device = next(module.parameters()).device

        # Broadcast initial parameters
        self._broadcast_parameters()

        # Create gradient reducer
        self.reducer = GradientReducer(
            module=self.module,
            process_group=self.process_group,
            bucket_size_mb=bucket_size_mb
        )

        # State
        self.require_backward_grad_sync = True

    def _broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all other ranks"""
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0, group=self.process_group)

        for buffer in self.module.buffers():
            dist.broadcast(buffer, src=0, group=self.process_group)

    def forward(self, *inputs, **kwargs):
        """Forward pass through the wrapped module"""
        # Move inputs to device if needed
        inputs = tuple(
            x.to(self.device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        )

        # Forward through module
        return self.module(*inputs, **kwargs)

    def backward(self, loss: torch.Tensor):
        """
        Backward pass with gradient synchronization.
        Call this instead of loss.backward().
        """
        # Compute gradients
        loss.backward()

        # Wait for AllReduce to complete
        if self.require_backward_grad_sync:
            self.reducer.wait_for_completion()

    def no_sync(self):
        """Context manager to disable gradient sync (for gradient accumulation)"""
        return _NoSync(self)


class _NoSync:
    """Context manager for disabling gradient sync"""

    def __init__(self, ddp: DistributedDataParallel):
        self.ddp = ddp

    def __enter__(self):
        self.old_value = self.ddp.require_backward_grad_sync
        self.ddp.require_backward_grad_sync = False

    def __exit__(self, *args):
        self.ddp.require_backward_grad_sync = self.old_value
```

### 5. Failure Handling and Checkpointing

```python
import os
import signal
import torch
import torch.distributed as dist
from typing import Dict, Any, Optional
import threading
import time

class FailureHandler:
    """
    Handles worker failures during distributed training.
    Implements timeouts, health checks, and recovery.
    """

    def __init__(self, process_group: dist.ProcessGroup,
                 timeout_seconds: float = 300,
                 health_check_interval: float = 30):
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)
        self.world_size = dist.get_world_size(process_group)
        self.timeout_seconds = timeout_seconds
        self.health_check_interval = health_check_interval

        # State
        self.last_heartbeat: Dict[int, float] = {
            r: time.time() for r in range(self.world_size)
        }
        self.is_healthy = True
        self.failed_ranks: List[int] = []

        # Start health check thread
        self.check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.check_thread.start()

    def _health_check_loop(self):
        """Background thread for health checks"""
        while self.is_healthy:
            try:
                self._perform_health_check()
            except Exception as e:
                print(f"Health check failed: {e}")
            time.sleep(self.health_check_interval)

    def _perform_health_check(self):
        """Perform all-to-all health check"""
        # Send heartbeat to all other ranks
        heartbeat = torch.tensor([self.rank, time.time()], dtype=torch.float64)
        heartbeats = [torch.zeros(2) for _ in range(self.world_size)]

        dist.all_gather(heartbeats, heartbeat, group=self.process_group)

        # Update last heartbeat times
        current_time = time.time()
        for hb in heartbeats:
            rank = int(hb[0].item())
            timestamp = hb[1].item()
            self.last_heartbeat[rank] = timestamp

            # Check for timeout
            if current_time - timestamp > self.timeout_seconds:
                if rank not in self.failed_ranks:
                    self.failed_ranks.append(rank)
                    self._handle_failure(rank)

    def _handle_failure(self, failed_rank: int):
        """Handle a detected worker failure"""
        print(f"Rank {failed_rank} failed!")

        # Options:
        # 1. Abort training
        # 2. Continue without failed rank (elastic training)
        # 3. Wait for recovery

        # For now, signal abort
        self.is_healthy = False


class Checkpointer:
    """
    Saves and loads training checkpoints for fault tolerance.
    """

    def __init__(self, save_dir: str, process_group: dist.ProcessGroup):
        self.save_dir = save_dir
        self.process_group = process_group
        self.rank = dist.get_rank(process_group)

        # Only rank 0 saves by default
        self.is_saver = (self.rank == 0)

        os.makedirs(save_dir, exist_ok=True)

    def save(self, state: Dict[str, Any], filename: str):
        """Save checkpoint (rank 0 only)"""
        if not self.is_saver:
            return

        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)

        # Also save "latest" pointer
        latest_path = os.path.join(self.save_dir, 'latest.txt')
        with open(latest_path, 'w') as f:
            f.write(filename)

    def load(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """Load checkpoint on all ranks"""
        if filename is None:
            # Load latest
            latest_path = os.path.join(self.save_dir, 'latest.txt')
            if os.path.exists(latest_path):
                with open(latest_path, 'r') as f:
                    filename = f.read().strip()
            else:
                return {}

        filepath = os.path.join(self.save_dir, filename)
        if not os.path.exists(filepath):
            return {}

        state = torch.load(filepath, map_location='cpu')

        # Broadcast to ensure consistency
        dist.barrier(group=self.process_group)

        return state

    def save_periodic(self, state: Dict[str, Any], step: int, interval: int = 1000):
        """Save checkpoint periodically"""
        if step % interval == 0:
            filename = f'checkpoint_{step:08d}.pt'
            self.save(state, filename)

            # Clean up old checkpoints (keep last 3)
            self._cleanup_old_checkpoints(keep=3)

    def _cleanup_old_checkpoints(self, keep: int):
        """Remove old checkpoints"""
        if not self.is_saver:
            return

        checkpoints = sorted([
            f for f in os.listdir(self.save_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ])

        for old_ckpt in checkpoints[:-keep]:
            os.remove(os.path.join(self.save_dir, old_ckpt))
```

### 6. Gradient Compression

```python
import torch
from typing import Tuple

class GradientCompressor:
    """Base class for gradient compression algorithms"""

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError

    def decompress(self, compressed: torch.Tensor, metadata: dict) -> torch.Tensor:
        raise NotImplementedError


class TopKCompressor(GradientCompressor):
    """
    Top-K sparsification: only communicate K largest elements.
    """

    def __init__(self, compression_ratio: float = 0.01):
        self.k_ratio = compression_ratio

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compress tensor by keeping top-K elements"""
        original_shape = tensor.shape
        flat = tensor.view(-1)

        k = max(1, int(flat.numel() * self.k_ratio))

        # Find top-k values and indices
        values, indices = torch.topk(flat.abs(), k)
        signs = flat[indices].sign()
        values = values * signs

        # Pack into sparse representation
        compressed = torch.stack([indices.float(), values])

        metadata = {
            'original_shape': original_shape,
            'original_numel': flat.numel(),
            'k': k
        }

        return compressed, metadata

    def decompress(self, compressed: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Decompress sparse representation"""
        indices = compressed[0].long()
        values = compressed[1]

        # Reconstruct dense tensor
        result = torch.zeros(metadata['original_numel'], device=compressed.device)
        result[indices] = values

        return result.view(metadata['original_shape'])


class QuantizedCompressor(GradientCompressor):
    """
    Quantization-based compression: reduce bits per element.
    """

    def __init__(self, bits: int = 8):
        self.bits = bits
        self.max_val = 2 ** (bits - 1) - 1

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compress tensor using quantization"""
        # Compute scale
        abs_max = tensor.abs().max()
        scale = abs_max / self.max_val if abs_max > 0 else 1.0

        # Quantize
        quantized = (tensor / scale).round().to(torch.int8)

        metadata = {
            'scale': scale,
            'original_dtype': tensor.dtype
        }

        return quantized, metadata

    def decompress(self, compressed: torch.Tensor, metadata: dict) -> torch.Tensor:
        """Decompress quantized tensor"""
        return compressed.to(metadata['original_dtype']) * metadata['scale']


class ErrorFeedbackCompressor(GradientCompressor):
    """
    Compression with error feedback to maintain convergence.
    Stores compression error for next iteration.
    """

    def __init__(self, base_compressor: GradientCompressor):
        self.base_compressor = base_compressor
        self.error: Optional[torch.Tensor] = None

    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Compress with error feedback"""
        # Add previous error
        if self.error is not None:
            tensor = tensor + self.error

        # Compress
        compressed, metadata = self.base_compressor.compress(tensor)

        # Compute and store error
        decompressed = self.base_compressor.decompress(compressed, metadata)
        self.error = tensor - decompressed

        return compressed, metadata

    def decompress(self, compressed: torch.Tensor, metadata: dict) -> torch.Tensor:
        return self.base_compressor.decompress(compressed, metadata)
```

## Enterprise Features

### NCCL Backend Integration

```python
class NCCLBackend:
    """
    High-performance NCCL backend for GPU communication.
    """

    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

        # Initialize NCCL
        import torch.cuda.nccl as nccl
        self.nccl = nccl

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """AllReduce using NCCL"""
        if not tensor.is_cuda:
            raise ValueError("NCCL requires CUDA tensors")

        # Use PyTorch's NCCL bindings
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def all_gather(self, tensors: List[torch.Tensor],
                   tensor: torch.Tensor) -> List[torch.Tensor]:
        """AllGather using NCCL"""
        dist.all_gather(tensors, tensor)
        return tensors
```

### Adaptive Communication

```python
class AdaptiveCommunicator:
    """
    Adapts communication strategy based on network conditions.
    """

    def __init__(self, process_group: dist.ProcessGroup):
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)

        # Create algorithm instances
        self.ring = RingAllReduce(process_group)
        self.tree = TreeAllReduce(process_group)

        # Bandwidth/latency estimates
        self.bandwidth_mbps = 10000  # 10 Gbps default
        self.latency_us = 1  # 1 us default

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Choose optimal algorithm based on message size"""
        size_bytes = tensor.numel() * tensor.element_size()

        # Estimate time for each algorithm
        ring_time = self._estimate_ring_time(size_bytes)
        tree_time = self._estimate_tree_time(size_bytes)

        if ring_time < tree_time:
            return self.ring.all_reduce(tensor)
        else:
            return self.tree.all_reduce(tensor)

    def _estimate_ring_time(self, size_bytes: int) -> float:
        """Estimate ring AllReduce time"""
        # Time = 2 * (n-1) * (latency + size / bandwidth / n)
        n = self.world_size
        return 2 * (n - 1) * (
            self.latency_us +
            size_bytes / self.bandwidth_mbps / n * 8
        )

    def _estimate_tree_time(self, size_bytes: int) -> float:
        """Estimate tree AllReduce time"""
        # Time = 2 * log(n) * (latency + size / bandwidth)
        import math
        n = self.world_size
        return 2 * math.log2(n) * (
            self.latency_us +
            size_bytes / self.bandwidth_mbps * 8
        )
```

## Development Phases

### Phase 1: Basic Communication (Weeks 1-3)
- Process group initialization
- Basic AllReduce implementations (ring, tree)
- Point-to-point operations
- Barrier synchronization

### Phase 2: Gradient Bucketization (Weeks 4-5)
- Bucket builder
- Gradient hooks
- Flatten/unflatten operations
- Bucket management

### Phase 3: DDP Wrapper (Weeks 6-7)
- Parameter broadcast
- Forward pass handling
- Backward hook integration
- Gradient sync API

### Phase 4: Communication/Computation Overlap (Weeks 8-9)
- Async AllReduce
- Communication scheduler
- Ready queue management
- Overlap statistics

### Phase 5: Fault Tolerance (Weeks 10-11)
- Health checking
- Timeout handling
- Checkpointing
- Recovery logic

### Phase 6: Enterprise Features (Week 12+)
- NCCL integration
- Gradient compression
- Adaptive communication
- Elastic training

## Testing Strategy

### Unit Tests
- AllReduce correctness
- Bucket building
- Compression algorithms
- Checkpoint save/load

### Integration Tests
- Multi-GPU training
- Gradient synchronization
- DDP wrapper functionality
- Fault injection

### Performance Tests
- AllReduce bandwidth
- Overlap efficiency
- Scaling efficiency
- Compression ratio

### Stress Tests
- Large model training
- Network failures
- Worker crashes
- Long-running training

## Performance Targets

| Metric | Target |
|--------|--------|
| Scaling efficiency (8 GPUs) | > 90% |
| Communication overlap | > 70% |
| AllReduce bandwidth | > 80% of peak |
| Checkpoint overhead | < 5% |
| Recovery time | < 60 seconds |

## Dependencies

- **PyTorch**: Tensor operations, autograd
- **NCCL**: GPU communication (optional)
- **Gloo**: CPU communication
- **MPI**: Alternative backend (optional)

## References

- Horovod: Fast and Easy Distributed Deep Learning
- PyTorch DistributedDataParallel
- "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
- "Deep Gradient Compression"
- "PowerSGD: Practical Low-Rank Gradient Compression"
