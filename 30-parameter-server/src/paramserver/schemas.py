"""Core data schemas for the parameter server."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import time


class WorkerStatus(Enum):
    """Status of a worker node."""
    IDLE = "idle"
    TRAINING = "training"
    PULLING = "pulling"
    PUSHING = "pushing"
    FAILED = "failed"


class ServerStatus(Enum):
    """Status of a parameter server."""
    READY = "ready"
    BUSY = "busy"
    CHECKPOINTING = "checkpointing"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class ShardInfo:
    """Information about a parameter shard.

    Attributes:
        shard_id: Unique identifier for this shard.
        server_address: Network address of the server hosting this shard.
        param_ranges: List of (start, end) index tuples for parameters in this shard.
        param_names: List of parameter names assigned to this shard.
        total_params: Total number of parameters in this shard.
        memory_bytes: Estimated memory usage in bytes.
    """
    shard_id: int
    server_address: str
    param_ranges: List[Tuple[int, int]] = field(default_factory=list)
    param_names: List[str] = field(default_factory=list)
    total_params: int = 0
    memory_bytes: int = 0

    def __post_init__(self):
        """Calculate memory if not set."""
        if self.memory_bytes == 0 and self.total_params > 0:
            # Assume float32 (4 bytes per param)
            self.memory_bytes = self.total_params * 4


@dataclass
class GradientUpdate:
    """Gradient update from a worker.

    Attributes:
        worker_id: ID of the worker sending the gradient.
        param_name: Name of the parameter being updated.
        gradient: The gradient values.
        clock: Worker's logical clock at time of update.
        timestamp: Wall-clock time of update.
        compressed: Whether the gradient is compressed.
        compression_metadata: Metadata for decompression if compressed.
    """
    worker_id: int
    param_name: str
    gradient: np.ndarray
    clock: int
    timestamp: float = field(default_factory=time.time)
    compressed: bool = False
    compression_metadata: Optional[Dict[str, Any]] = None


@dataclass
class WorkerInfo:
    """Information about a training worker.

    Attributes:
        worker_id: Unique identifier for this worker.
        address: Network address of the worker.
        status: Current status of the worker.
        clock: Current logical clock value.
        last_heartbeat: Time of last heartbeat.
        assigned_data_range: Range of training data assigned to this worker.
        total_updates: Total number of gradient updates pushed.
        total_steps: Total training steps completed.
    """
    worker_id: int
    address: str = ""
    status: WorkerStatus = WorkerStatus.IDLE
    clock: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    assigned_data_range: Tuple[int, int] = (0, 0)
    total_updates: int = 0
    total_steps: int = 0

    def update_heartbeat(self):
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = time.time()

    def is_stale(self, timeout: float = 30.0) -> bool:
        """Check if worker hasn't sent a heartbeat within timeout."""
        return time.time() - self.last_heartbeat > timeout


@dataclass
class ParameterMetadata:
    """Metadata for a stored parameter.

    Attributes:
        name: Parameter name.
        shape: Shape of the parameter tensor.
        dtype: Data type of the parameter.
        version: Current version number (incremented on each update).
        last_update_worker: ID of worker that last updated this parameter.
        last_update_time: Time of last update.
        total_updates: Total number of updates applied.
    """
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float32))
    version: int = 0
    last_update_worker: Optional[int] = None
    last_update_time: float = field(default_factory=time.time)
    total_updates: int = 0

    @property
    def size(self) -> int:
        """Return total number of elements."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    @property
    def memory_bytes(self) -> int:
        """Return memory usage in bytes."""
        return self.size * self.dtype.itemsize


@dataclass
class PullRequest:
    """Request to pull parameters from server.

    Attributes:
        worker_id: ID of requesting worker.
        param_names: List of parameter names to pull.
        include_versions: Whether to include version numbers in response.
    """
    worker_id: int
    param_names: List[str]
    include_versions: bool = True


@dataclass
class PullResponse:
    """Response to a pull request.

    Attributes:
        params: Dict mapping parameter names to values.
        versions: Dict mapping parameter names to versions (if requested).
        server_id: ID of the server that responded.
    """
    params: Dict[str, np.ndarray]
    versions: Dict[str, int] = field(default_factory=dict)
    server_id: int = 0


@dataclass
class PushRequest:
    """Request to push gradients to server.

    Attributes:
        worker_id: ID of worker sending gradients.
        gradients: Dict mapping parameter names to gradient values.
        clock: Worker's logical clock.
        compressed: Whether gradients are compressed.
    """
    worker_id: int
    gradients: Dict[str, np.ndarray]
    clock: int
    compressed: bool = False


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint.

    Attributes:
        checkpoint_id: Unique identifier for this checkpoint.
        epoch: Training epoch at checkpoint time.
        global_step: Global training step at checkpoint time.
        worker_clocks: Dict of worker IDs to their clock values.
        param_versions: Dict of parameter names to versions.
        timestamp: Time when checkpoint was created.
        path: File path where checkpoint is stored.
    """
    checkpoint_id: str
    epoch: int
    global_step: int
    worker_clocks: Dict[int, int] = field(default_factory=dict)
    param_versions: Dict[str, int] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    path: str = ""
