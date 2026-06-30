"""Core distributed context and tensor types."""

import numpy as np
import threading
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger(__name__)


class Backend(Enum):
    """Communication backends."""
    GLOO = auto()
    NCCL = auto()
    MPI = auto()


@dataclass
class WorkerInfo:
    """Information about a worker."""
    rank: int
    world_size: int
    local_rank: int
    hostname: str = "localhost"
    port: int = 29500


@dataclass
class ProcessGroup:
    """Group of processes for collective operations."""
    ranks: List[int]
    backend: Backend = Backend.GLOO
    name: str = "default"

    def __post_init__(self):
        self._lock = threading.Lock()
        self._barriers: Dict[str, int] = {}

    def size(self) -> int:
        return len(self.ranks)

    def barrier(self, name: str = "default"):
        """Synchronization barrier."""
        with self._lock:
            if name not in self._barriers:
                self._barriers[name] = 0
            self._barriers[name] += 1

            while self._barriers[name] < len(self.ranks):
                time.sleep(0.001)

            # Reset for reuse
            if self._barriers[name] >= len(self.ranks):
                self._barriers[name] = 0


@dataclass
class DeviceMesh:
    """N-dimensional mesh of devices for parallelism."""
    shape: Tuple[int, ...]
    device_ids: List[int]
    mesh_dim_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        expected_size = 1
        for dim in self.shape:
            expected_size *= dim
        if len(self.device_ids) != expected_size:
            raise ValueError(f"Device count {len(self.device_ids)} != mesh size {expected_size}")

    def get_device(self, *indices) -> int:
        """Get device ID by mesh coordinates."""
        if len(indices) != len(self.shape):
            raise ValueError(f"Expected {len(self.shape)} indices")

        flat_idx = 0
        multiplier = 1
        for i in reversed(range(len(self.shape))):
            flat_idx += indices[i] * multiplier
            multiplier *= self.shape[i]

        return self.device_ids[flat_idx]

    def get_submesh(self, dim: int, index: int) -> 'DeviceMesh':
        """Get submesh at given dimension and index."""
        new_shape = list(self.shape)
        del new_shape[dim]

        # Get device IDs for submesh
        new_ids = []
        step = 1
        for i in range(dim + 1, len(self.shape)):
            step *= self.shape[i]

        for i, dev_id in enumerate(self.device_ids):
            coord = (i // step) % self.shape[dim]
            if coord == index:
                new_ids.append(dev_id)

        return DeviceMesh(tuple(new_shape), new_ids)


class DistributedTensor:
    """
    Tensor distributed across multiple devices.

    Supports:
    - Sharding strategies
    - Gradient synchronization
    - Collective operations
    """

    def __init__(
        self,
        data: np.ndarray,
        device: int = 0,
        requires_grad: bool = False,
        process_group: ProcessGroup = None
    ):
        self.data = data
        self.device = device
        self.requires_grad = requires_grad
        self.process_group = process_group

        self.grad: Optional[np.ndarray] = None
        self._grad_fn = None
        self._is_leaf = True

        # Sharding info
        self._sharding_spec: Optional[Dict] = None
        self._local_shard = data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def local_shape(self) -> Tuple[int, ...]:
        return self._local_shard.shape

    def shard(self, dim: int, num_shards: int) -> List['DistributedTensor']:
        """Shard tensor along dimension."""
        shards = np.array_split(self.data, num_shards, axis=dim)
        result = []

        for i, shard_data in enumerate(shards):
            shard = DistributedTensor(
                shard_data,
                device=i % 8,  # Distribute across devices
                requires_grad=self.requires_grad,
                process_group=self.process_group
            )
            shard._sharding_spec = {"dim": dim, "index": i, "total": num_shards}
            result.append(shard)

        return result

    def all_gather(self) -> 'DistributedTensor':
        """Gather all shards."""
        # Simulated - in real implementation would use NCCL
        return self

    def reduce_scatter(self, op: str = "sum") -> 'DistributedTensor':
        """Reduce and scatter result."""
        return self

    def backward(self, grad: np.ndarray = None):
        """Backward pass with gradient synchronization."""
        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad

        if self._grad_fn is not None:
            self._grad_fn(grad)


@dataclass
class GradBucket:
    """Bucket for gradient accumulation and reduction."""
    index: int
    tensors: List[DistributedTensor] = field(default_factory=list)
    gradients: List[np.ndarray] = field(default_factory=list)
    size: int = 0
    ready: bool = False

    def add_gradient(self, tensor: DistributedTensor, grad: np.ndarray):
        """Add gradient to bucket."""
        self.tensors.append(tensor)
        self.gradients.append(grad)
        self.size += grad.nbytes

    def flatten(self) -> np.ndarray:
        """Flatten all gradients into single buffer."""
        return np.concatenate([g.flatten() for g in self.gradients])

    def unflatten(self, flat_grad: np.ndarray):
        """Unflatten buffer back to gradients."""
        offset = 0
        for i, grad in enumerate(self.gradients):
            size = grad.size
            self.gradients[i] = flat_grad[offset:offset + size].reshape(grad.shape)
            offset += size

    def clear(self):
        """Clear bucket."""
        self.tensors.clear()
        self.gradients.clear()
        self.size = 0
        self.ready = False


class DistributedContext:
    """
    Context for distributed autograd computation.

    Manages:
    - Process groups
    - Gradient accumulation
    - Communication
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, worker_info: WorkerInfo = None):
        self.worker_info = worker_info or WorkerInfo(0, 1, 0)
        self._process_groups: Dict[str, ProcessGroup] = {}
        self._grad_buckets: List[GradBucket] = []
        self._autograd_contexts: Dict[int, 'AutogradContext'] = {}
        self._next_context_id = 0

        # Default process group
        self._process_groups["default"] = ProcessGroup(
            list(range(self.worker_info.world_size))
        )

    @classmethod
    def get_instance(cls) -> 'DistributedContext':
        """Get singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def init(
        cls,
        rank: int,
        world_size: int,
        backend: Backend = Backend.GLOO
    ) -> 'DistributedContext':
        """Initialize distributed context."""
        with cls._lock:
            worker_info = WorkerInfo(
                rank=rank,
                world_size=world_size,
                local_rank=rank
            )
            cls._instance = cls(worker_info)
            return cls._instance

    def get_rank(self) -> int:
        return self.worker_info.rank

    def get_world_size(self) -> int:
        return self.worker_info.world_size

    def new_group(self, ranks: List[int], name: str = None) -> ProcessGroup:
        """Create new process group."""
        name = name or f"group_{len(self._process_groups)}"
        group = ProcessGroup(ranks)
        self._process_groups[name] = group
        return group

    def get_group(self, name: str = "default") -> ProcessGroup:
        """Get process group by name."""
        return self._process_groups.get(name)

    def new_autograd_context(self) -> int:
        """Create new autograd context for RPC."""
        context_id = self._next_context_id
        self._next_context_id += 1
        self._autograd_contexts[context_id] = AutogradContext(context_id)
        return context_id

    def get_autograd_context(self, context_id: int) -> 'AutogradContext':
        """Get autograd context by ID."""
        return self._autograd_contexts.get(context_id)

    def barrier(self, group: str = "default"):
        """Global barrier."""
        self._process_groups[group].barrier()


class AutogradContext:
    """Context for tracking distributed autograd operations."""

    def __init__(self, context_id: int):
        self.context_id = context_id
        self.send_functions: Dict[int, Callable] = {}
        self.recv_functions: Dict[int, Callable] = {}
        self._grad_to_send: Dict[int, np.ndarray] = {}

    def add_send_function(self, seq_id: int, func: Callable):
        """Add send function for backward."""
        self.send_functions[seq_id] = func

    def add_recv_function(self, seq_id: int, func: Callable):
        """Add receive function for backward."""
        self.recv_functions[seq_id] = func

    def accumulate_grad(self, seq_id: int, grad: np.ndarray):
        """Accumulate gradient to send."""
        if seq_id in self._grad_to_send:
            self._grad_to_send[seq_id] += grad
        else:
            self._grad_to_send[seq_id] = grad.copy()


# Collective operations
def all_reduce(
    tensor: DistributedTensor,
    op: str = "sum",
    group: ProcessGroup = None
) -> DistributedTensor:
    """All-reduce operation across group."""
    # Simulated - returns input
    return tensor


def broadcast(
    tensor: DistributedTensor,
    src: int = 0,
    group: ProcessGroup = None
) -> DistributedTensor:
    """Broadcast from source rank."""
    return tensor


def all_gather(
    tensors: List[DistributedTensor],
    tensor: DistributedTensor,
    group: ProcessGroup = None
):
    """All-gather operation."""
    pass


def reduce_scatter(
    output: DistributedTensor,
    input_list: List[DistributedTensor],
    op: str = "sum",
    group: ProcessGroup = None
):
    """Reduce-scatter operation."""
    pass
