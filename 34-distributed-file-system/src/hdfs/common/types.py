"""Core types for HDFS."""

import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum


# Type aliases
BlockID = str
NodeID = str


def generate_block_id() -> BlockID:
    """Generate unique block ID."""
    return f"blk_{uuid.uuid4().hex[:16]}"


def generate_node_id() -> NodeID:
    """Generate unique node ID."""
    return f"node_{uuid.uuid4().hex[:12]}"


class ReplicationPolicy(Enum):
    """Block replication policies."""
    DEFAULT = "default"       # Standard 3x replication
    ERASURE_CODING = "ec"     # Erasure coding
    SINGLE = "single"         # No replication


@dataclass
class Block:
    """A data block in HDFS."""
    block_id: BlockID
    size: int = 0
    generation_stamp: int = 0

    def __post_init__(self):
        if self.generation_stamp == 0:
            self.generation_stamp = int(time.time() * 1000)


@dataclass
class BlockLocation:
    """Location of a block replica."""
    block_id: BlockID
    node_id: NodeID
    host: str
    port: int
    rack: str = "/default-rack"


@dataclass
class DataNodeInfo:
    """Information about a DataNode."""
    node_id: NodeID
    host: str
    port: int
    rack: str = "/default-rack"

    # Capacity info
    capacity: int = 0           # Total capacity in bytes
    used: int = 0               # Used space in bytes
    remaining: int = 0          # Remaining space in bytes

    # State
    last_heartbeat: float = 0.0
    blocks: Set[BlockID] = field(default_factory=set)

    @property
    def is_alive(self) -> bool:
        """Check if node is alive (heartbeat within 30s)."""
        return time.time() - self.last_heartbeat < 30.0

    @property
    def usage_percent(self) -> float:
        """Get storage usage percentage."""
        if self.capacity == 0:
            return 0.0
        return (self.used / self.capacity) * 100


@dataclass
class FileInfo:
    """Metadata for a file."""
    path: str
    size: int = 0
    block_size: int = 128 * 1024 * 1024  # 128MB default
    replication: int = 3
    blocks: List[BlockID] = field(default_factory=list)

    # Timestamps
    modification_time: float = field(default_factory=time.time)
    access_time: float = field(default_factory=time.time)

    # Permissions
    owner: str = "hdfs"
    group: str = "supergroup"
    permission: int = 0o644

    @property
    def num_blocks(self) -> int:
        """Number of blocks in file."""
        return len(self.blocks)


@dataclass
class DirectoryInfo:
    """Metadata for a directory."""
    path: str
    children: Set[str] = field(default_factory=set)  # Names only

    # Timestamps
    modification_time: float = field(default_factory=time.time)

    # Permissions
    owner: str = "hdfs"
    group: str = "supergroup"
    permission: int = 0o755


class NamespaceOperation(Enum):
    """Operations on the namespace."""
    CREATE = "create"
    DELETE = "delete"
    RENAME = "rename"
    MKDIR = "mkdir"
    SET_REPLICATION = "set_replication"
    SET_PERMISSION = "set_permission"


@dataclass
class BlockReport:
    """Block report from DataNode."""
    node_id: NodeID
    blocks: List[BlockID]
    timestamp: float = field(default_factory=time.time)


@dataclass
class HeartbeatResponse:
    """Response to DataNode heartbeat."""
    commands: List[Dict] = field(default_factory=list)
    # Commands can include:
    # {"type": "replicate", "block_id": ..., "targets": [...]}
    # {"type": "delete", "block_ids": [...]}
    # {"type": "invalidate", "block_ids": [...]}
