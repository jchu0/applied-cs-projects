"""Core schemas for parameter server."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import numpy as np
import time
import uuid


def generate_id() -> str:
    """Generate unique identifier."""
    return str(uuid.uuid4())[:8]


class ConsistencyModel(Enum):
    """Parameter consistency models."""
    BSP = "bsp"           # Bulk Synchronous Parallel
    ASP = "asp"           # Asynchronous Parallel
    SSP = "ssp"           # Stale Synchronous Parallel
    BOUNDED = "bounded"   # Bounded staleness


class AggregationType(Enum):
    """Gradient aggregation types."""
    SUM = "sum"
    MEAN = "mean"
    WEIGHTED = "weighted"


class NodeStatus(Enum):
    """Server node status."""
    ACTIVE = "active"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class Parameter:
    """Model parameter."""
    name: str
    shape: tuple
    dtype: str = "float32"
    data: np.ndarray = None
    version: int = 0
    shard_id: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class Gradient:
    """Parameter gradient."""
    name: str
    data: np.ndarray
    worker_id: str
    iteration: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class ShardInfo:
    """Parameter shard information."""
    shard_id: int
    parameter_names: list[str]
    node_id: str
    size_bytes: int
    num_parameters: int


@dataclass
class ServerNode:
    """Parameter server node."""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.ACTIVE
    shards: list[int] = field(default_factory=list)
    load: float = 0.0
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class WorkerNode:
    """Worker node information."""
    worker_id: str
    host: str
    port: int
    status: str = "active"
    current_iteration: int = 0
    last_heartbeat: float = field(default_factory=time.time)


@dataclass
class PullRequest:
    """Request to pull parameters."""
    request_id: str
    worker_id: str
    parameter_names: list[str]
    min_version: int = 0


@dataclass
class PushRequest:
    """Request to push gradients."""
    request_id: str
    worker_id: str
    gradients: list[Gradient]
    iteration: int


@dataclass
class SyncBarrier:
    """Synchronization barrier."""
    barrier_id: str
    iteration: int
    expected_workers: int
    arrived_workers: set = field(default_factory=set)
    created_at: float = field(default_factory=time.time)

    @property
    def is_complete(self) -> bool:
        return len(self.arrived_workers) >= self.expected_workers


@dataclass
class AggregationConfig:
    """Configuration for gradient aggregation."""
    aggregation_type: AggregationType = AggregationType.MEAN
    clip_norm: float = None
    momentum: float = 0.0
    weight_decay: float = 0.0


@dataclass
class PartitionStrategy:
    """Strategy for partitioning parameters."""
    num_shards: int
    strategy_type: str = "hash"  # hash, range, round_robin
    max_shard_size_bytes: int = 1024 * 1024 * 1024  # 1GB
