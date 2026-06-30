# Distributed File System (Ceph-lite / HDFS-lite) - Technical Blueprint

## Executive Summary

This project implements a production-grade distributed file system featuring distributed metadata management, consistent reads/writes with GFS-like primary/secondary replication, CRUSH-based placement, self-healing re-replication, and scalable chunk storage. The system supports ML workloads with high throughput and fault tolerance.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Distributed File System Architecture                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                          Client Library                               │   │
│  │   open()    read()    write()    close()    mkdir()    ls()         │   │
│  └──────────────────────────────────┬───────────────────────────────────┘   │
│                                     │                                        │
│     ┌───────────────────────────────┼───────────────────────────────┐       │
│     │                               │                               │       │
│     ▼                               ▼                               ▼       │
│  ┌──────────────┐           ┌──────────────┐           ┌──────────────┐    │
│  │   Metadata   │           │    Primary   │           │   Secondary  │    │
│  │   Server     │           │    Chunk     │           │    Chunk     │    │
│  │   (MDS)      │           │   Server     │           │   Server     │    │
│  │              │           │              │           │              │    │
│  │ • Namespace  │           │ • Write Path │           │ • Replicate  │    │
│  │ • Inode Map  │           │ • Checksums  │           │ • Heartbeat  │    │
│  │ • Block Map  │           │ • Primary    │           │ • Recovery   │    │
│  │ • Leases     │           │   Lease      │           │              │    │
│  └──────┬───────┘           └──────┬───────┘           └──────┬───────┘    │
│         │                          │                          │             │
│         └──────────────────────────┼──────────────────────────┘             │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Coordination Layer                              │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │  │    CRUSH     │  │   Placement  │  │  Consistency │              │    │
│  │  │    Map       │  │    Groups    │  │   Protocol   │              │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Structures

```python
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum
import struct
import threading

class FileType(Enum):
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"

@dataclass
class Inode:
    """Inode representing a file or directory."""
    inode_id: int
    parent_id: int
    name: str
    file_type: FileType

    # Metadata
    size: int = 0
    mode: int = 0o644
    uid: int = 0
    gid: int = 0
    ctime: float = field(default_factory=time.time)  # Creation time
    mtime: float = field(default_factory=time.time)  # Modification time
    atime: float = field(default_factory=time.time)  # Access time

    # For files: block mapping
    blocks: List[int] = field(default_factory=list)  # List of block IDs

    # For directories: children
    children: Dict[str, int] = field(default_factory=dict)  # name -> inode_id

    # Replication
    replication_factor: int = 3

    def to_bytes(self) -> bytes:
        """Serialize inode to bytes."""
        # Simplified serialization
        data = {
            'inode_id': self.inode_id,
            'parent_id': self.parent_id,
            'name': self.name,
            'file_type': self.file_type.value,
            'size': self.size,
            'mode': self.mode,
            'uid': self.uid,
            'gid': self.gid,
            'ctime': self.ctime,
            'mtime': self.mtime,
            'atime': self.atime,
            'blocks': self.blocks,
            'children': self.children,
            'replication_factor': self.replication_factor
        }
        import json
        return json.dumps(data).encode()

    @staticmethod
    def from_bytes(data: bytes) -> 'Inode':
        """Deserialize inode from bytes."""
        import json
        d = json.loads(data.decode())
        inode = Inode(
            inode_id=d['inode_id'],
            parent_id=d['parent_id'],
            name=d['name'],
            file_type=FileType(d['file_type'])
        )
        inode.size = d['size']
        inode.mode = d['mode']
        inode.uid = d['uid']
        inode.gid = d['gid']
        inode.ctime = d['ctime']
        inode.mtime = d['mtime']
        inode.atime = d['atime']
        inode.blocks = d['blocks']
        inode.children = d['children']
        inode.replication_factor = d['replication_factor']
        return inode

@dataclass
class Block:
    """Data block stored on chunk servers."""
    block_id: int
    size: int  # Actual data size in this block
    checksum: str  # MD5 or CRC32

    # Locations (chunk server IDs)
    locations: List[str] = field(default_factory=list)

    # Version for consistency
    version: int = 0

    # Generation stamp for detecting stale replicas
    generation_stamp: int = 0

@dataclass
class Chunk:
    """Physical storage unit on chunk server."""
    chunk_id: str
    block_id: int
    data: bytes
    checksum: str
    version: int
    generation_stamp: int

    @staticmethod
    def compute_checksum(data: bytes) -> str:
        """Compute checksum of data."""
        return hashlib.md5(data).hexdigest()

    def verify_checksum(self) -> bool:
        """Verify data integrity."""
        return self.compute_checksum(self.data) == self.checksum

@dataclass
class ChunkServerInfo:
    """Information about a chunk server."""
    server_id: str
    host: str
    port: int

    # Capacity
    total_space: int
    used_space: int
    available_space: int

    # Health
    last_heartbeat: float
    is_healthy: bool = True

    # Stored blocks
    blocks: Set[int] = field(default_factory=set)

    @property
    def usage_percent(self) -> float:
        return self.used_space / self.total_space * 100

@dataclass
class Lease:
    """Lease for coordinating writes."""
    block_id: int
    primary_server: str
    expiration: float
    holder: str  # Client ID

    def is_valid(self) -> bool:
        return time.time() < self.expiration

@dataclass
class PlacementGroup:
    """Group of blocks managed together for consistency."""
    pg_id: int
    blocks: Set[int]
    primary_server: str
    secondary_servers: List[str]
```

### 2. Metadata Server (MDS)

```python
import threading
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class MetadataServer:
    """
    Manages filesystem metadata: namespace, inodes, block mappings.
    Uses in-memory structures with WAL for persistence.
    """

    def __init__(self, config: dict):
        self.config = config

        # Inode management
        self._inodes: Dict[int, Inode] = {}
        self._next_inode_id = 1
        self._inode_lock = threading.RLock()

        # Block management
        self._blocks: Dict[int, Block] = {}
        self._next_block_id = 1
        self._block_lock = threading.RLock()

        # Chunk server tracking
        self._chunk_servers: Dict[str, ChunkServerInfo] = {}
        self._server_lock = threading.RLock()

        # Leases
        self._leases: Dict[int, Lease] = {}
        self._lease_lock = threading.RLock()

        # Write-ahead log
        self._wal = WriteAheadLog(config.get('wal_path', 'metadata.wal'))

        # Initialize root directory
        self._init_root()

        # CRUSH map for placement
        self.crush_map = CRUSHMap()

        # Block size
        self.block_size = config.get('block_size', 64 * 1024 * 1024)  # 64MB

    def _init_root(self):
        """Initialize root directory."""
        root = Inode(
            inode_id=0,
            parent_id=-1,
            name="/",
            file_type=FileType.DIRECTORY
        )
        root.mode = 0o755
        self._inodes[0] = root

    # Namespace operations

    def create_file(self, path: str, mode: int = 0o644,
                    replication: int = 3) -> int:
        """
        Create a new file.

        Returns:
            Inode ID of created file
        """
        parent_path, name = self._split_path(path)
        parent_inode = self._resolve_path(parent_path)

        if parent_inode is None:
            raise FileNotFoundError(f"Parent directory not found: {parent_path}")

        if parent_inode.file_type != FileType.DIRECTORY:
            raise NotADirectoryError(f"Not a directory: {parent_path}")

        if name in parent_inode.children:
            raise FileExistsError(f"File already exists: {path}")

        with self._inode_lock:
            inode_id = self._next_inode_id
            self._next_inode_id += 1

            inode = Inode(
                inode_id=inode_id,
                parent_id=parent_inode.inode_id,
                name=name,
                file_type=FileType.FILE
            )
            inode.mode = mode
            inode.replication_factor = replication

            # Log operation
            self._wal.log_operation('CREATE_FILE', {
                'inode_id': inode_id,
                'parent_id': parent_inode.inode_id,
                'name': name,
                'mode': mode
            })

            self._inodes[inode_id] = inode
            parent_inode.children[name] = inode_id

        return inode_id

    def create_directory(self, path: str, mode: int = 0o755) -> int:
        """Create a new directory."""
        parent_path, name = self._split_path(path)
        parent_inode = self._resolve_path(parent_path)

        if parent_inode is None:
            raise FileNotFoundError(f"Parent directory not found: {parent_path}")

        if name in parent_inode.children:
            raise FileExistsError(f"Directory already exists: {path}")

        with self._inode_lock:
            inode_id = self._next_inode_id
            self._next_inode_id += 1

            inode = Inode(
                inode_id=inode_id,
                parent_id=parent_inode.inode_id,
                name=name,
                file_type=FileType.DIRECTORY
            )
            inode.mode = mode

            self._wal.log_operation('CREATE_DIR', {
                'inode_id': inode_id,
                'parent_id': parent_inode.inode_id,
                'name': name
            })

            self._inodes[inode_id] = inode
            parent_inode.children[name] = inode_id

        return inode_id

    def delete_file(self, path: str):
        """Delete a file."""
        inode = self._resolve_path(path)
        if inode is None:
            raise FileNotFoundError(f"File not found: {path}")

        if inode.file_type == FileType.DIRECTORY:
            if inode.children:
                raise OSError("Directory not empty")

        parent = self._inodes[inode.parent_id]

        with self._inode_lock:
            # Remove from parent
            del parent.children[inode.name]

            # Schedule blocks for deletion
            for block_id in inode.blocks:
                self._schedule_block_deletion(block_id)

            # Remove inode
            del self._inodes[inode.inode_id]

            self._wal.log_operation('DELETE', {
                'inode_id': inode.inode_id,
                'parent_id': parent.inode_id
            })

    def list_directory(self, path: str) -> List[Tuple[str, int]]:
        """List directory contents."""
        inode = self._resolve_path(path)
        if inode is None:
            raise FileNotFoundError(f"Directory not found: {path}")

        if inode.file_type != FileType.DIRECTORY:
            raise NotADirectoryError(f"Not a directory: {path}")

        return [(name, child_id) for name, child_id in inode.children.items()]

    def get_file_info(self, path: str) -> dict:
        """Get file metadata."""
        inode = self._resolve_path(path)
        if inode is None:
            raise FileNotFoundError(f"File not found: {path}")

        return {
            'inode_id': inode.inode_id,
            'name': inode.name,
            'type': inode.file_type.value,
            'size': inode.size,
            'mode': inode.mode,
            'mtime': inode.mtime,
            'blocks': len(inode.blocks),
            'replication': inode.replication_factor
        }

    # Block operations

    def allocate_block(self, inode_id: int) -> Tuple[int, List[str]]:
        """
        Allocate a new block for a file.

        Returns:
            (block_id, [chunk_server_ids])
        """
        inode = self._inodes.get(inode_id)
        if inode is None:
            raise FileNotFoundError(f"Inode not found: {inode_id}")

        with self._block_lock:
            block_id = self._next_block_id
            self._next_block_id += 1

            # Choose chunk servers using CRUSH
            servers = self.crush_map.select_servers(
                block_id,
                inode.replication_factor
            )

            block = Block(
                block_id=block_id,
                size=0,
                checksum="",
                locations=servers,
                version=0,
                generation_stamp=int(time.time() * 1000)
            )

            self._blocks[block_id] = block
            inode.blocks.append(block_id)
            inode.mtime = time.time()

            self._wal.log_operation('ALLOCATE_BLOCK', {
                'block_id': block_id,
                'inode_id': inode_id,
                'servers': servers
            })

        # Create lease for primary
        if servers:
            self._create_lease(block_id, servers[0])

        return block_id, servers

    def get_block_locations(self, block_id: int) -> List[str]:
        """Get chunk servers storing a block."""
        block = self._blocks.get(block_id)
        if block is None:
            raise KeyError(f"Block not found: {block_id}")

        # Filter to healthy servers only
        healthy = [
            loc for loc in block.locations
            if self._is_server_healthy(loc)
        ]

        return healthy

    def report_block_received(self, block_id: int, server_id: str,
                              size: int, checksum: str):
        """Report that a chunk server has received a block."""
        with self._block_lock:
            block = self._blocks.get(block_id)
            if block is None:
                return

            if server_id not in block.locations:
                block.locations.append(server_id)

            block.size = size
            block.checksum = checksum

        # Update chunk server info
        with self._server_lock:
            if server_id in self._chunk_servers:
                self._chunk_servers[server_id].blocks.add(block_id)

    # Chunk server management

    def register_chunk_server(self, server_id: str, host: str, port: int,
                              capacity: int) -> dict:
        """Register a new chunk server."""
        with self._server_lock:
            info = ChunkServerInfo(
                server_id=server_id,
                host=host,
                port=port,
                total_space=capacity,
                used_space=0,
                available_space=capacity,
                last_heartbeat=time.time()
            )
            self._chunk_servers[server_id] = info

            # Add to CRUSH map
            self.crush_map.add_server(server_id, capacity)

            logger.info(f"Registered chunk server: {server_id} at {host}:{port}")

        return {'status': 'ok', 'server_id': server_id}

    def process_heartbeat(self, server_id: str, blocks: List[int],
                          used_space: int) -> dict:
        """Process heartbeat from chunk server."""
        with self._server_lock:
            server = self._chunk_servers.get(server_id)
            if server is None:
                return {'status': 'error', 'message': 'Unknown server'}

            server.last_heartbeat = time.time()
            server.used_space = used_space
            server.available_space = server.total_space - used_space
            server.blocks = set(blocks)
            server.is_healthy = True

        # Check for missing replicas
        commands = self._check_replication(server_id, blocks)

        return {
            'status': 'ok',
            'commands': commands
        }

    def _is_server_healthy(self, server_id: str) -> bool:
        """Check if chunk server is healthy."""
        server = self._chunk_servers.get(server_id)
        if server is None:
            return False

        # Consider unhealthy if no heartbeat in 30 seconds
        return time.time() - server.last_heartbeat < 30

    # Lease management

    def _create_lease(self, block_id: int, primary_server: str):
        """Create write lease for a block."""
        with self._lease_lock:
            lease = Lease(
                block_id=block_id,
                primary_server=primary_server,
                expiration=time.time() + 60,  # 60 second lease
                holder=""
            )
            self._leases[block_id] = lease

    def get_lease(self, block_id: int, client_id: str) -> Optional[Lease]:
        """Get or create lease for writing."""
        with self._lease_lock:
            lease = self._leases.get(block_id)

            if lease is None or not lease.is_valid():
                # Create new lease
                block = self._blocks.get(block_id)
                if block and block.locations:
                    lease = Lease(
                        block_id=block_id,
                        primary_server=block.locations[0],
                        expiration=time.time() + 60,
                        holder=client_id
                    )
                    self._leases[block_id] = lease

            return lease

    def renew_lease(self, block_id: int, client_id: str) -> bool:
        """Renew an existing lease."""
        with self._lease_lock:
            lease = self._leases.get(block_id)
            if lease and lease.holder == client_id:
                lease.expiration = time.time() + 60
                return True
        return False

    # Replication and self-healing

    def _check_replication(self, server_id: str, blocks: List[int]) -> List[dict]:
        """Check block replication and generate commands."""
        commands = []

        for block_id in blocks:
            block = self._blocks.get(block_id)
            if block is None:
                # Block should be deleted
                commands.append({
                    'command': 'DELETE_BLOCK',
                    'block_id': block_id
                })
                continue

            # Check if under-replicated
            healthy_replicas = len([
                loc for loc in block.locations
                if self._is_server_healthy(loc)
            ])

            # Get expected replication from inode
            inode_id = self._find_block_inode(block_id)
            if inode_id is not None:
                inode = self._inodes.get(inode_id)
                expected = inode.replication_factor if inode else 3
            else:
                expected = 3

            if healthy_replicas < expected:
                # Need more replicas
                target = self._select_replication_target(block_id)
                if target:
                    commands.append({
                        'command': 'REPLICATE',
                        'block_id': block_id,
                        'target': target
                    })

        return commands

    def _schedule_block_deletion(self, block_id: int):
        """Schedule a block for deletion on all servers."""
        block = self._blocks.get(block_id)
        if block:
            # Servers will delete on next heartbeat
            del self._blocks[block_id]

    def _find_block_inode(self, block_id: int) -> Optional[int]:
        """Find which inode owns a block."""
        for inode_id, inode in self._inodes.items():
            if block_id in inode.blocks:
                return inode_id
        return None

    def _select_replication_target(self, block_id: int) -> Optional[str]:
        """Select a server for additional replica."""
        block = self._blocks.get(block_id)
        if block is None:
            return None

        current = set(block.locations)

        # Find servers not already storing this block
        candidates = [
            (sid, info) for sid, info in self._chunk_servers.items()
            if sid not in current and info.is_healthy
        ]

        if not candidates:
            return None

        # Select server with most free space
        candidates.sort(key=lambda x: x[1].available_space, reverse=True)
        return candidates[0][0]

    # Path resolution

    def _resolve_path(self, path: str) -> Optional[Inode]:
        """Resolve path to inode."""
        if path == "/":
            return self._inodes[0]

        parts = [p for p in path.split('/') if p]
        current = self._inodes[0]

        for part in parts:
            if current.file_type != FileType.DIRECTORY:
                return None

            if part not in current.children:
                return None

            current = self._inodes[current.children[part]]

        return current

    def _split_path(self, path: str) -> Tuple[str, str]:
        """Split path into parent and name."""
        if path == "/":
            return "/", ""

        parts = path.rstrip('/').rsplit('/', 1)
        if len(parts) == 1:
            return "/", parts[0]

        parent = parts[0] if parts[0] else "/"
        return parent, parts[1]

class WriteAheadLog:
    """Write-ahead log for metadata durability."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._file = open(path, 'ab')

    def log_operation(self, op_type: str, data: dict):
        """Log an operation."""
        import json
        entry = {
            'timestamp': time.time(),
            'op_type': op_type,
            'data': data
        }
        line = json.dumps(entry) + '\n'

        with self._lock:
            self._file.write(line.encode())
            self._file.flush()

    def replay(self) -> List[dict]:
        """Replay log entries."""
        import json
        entries = []

        with open(self.path, 'rb') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line.decode()))

        return entries

    def checkpoint(self):
        """Create checkpoint and truncate log."""
        # In production, write full metadata snapshot
        pass
```

### 3. CRUSH Algorithm

```python
import hashlib

class CRUSHMap:
    """
    CRUSH (Controlled Replication Under Scalable Hashing) algorithm.
    Pseudo-random placement with failure domain awareness.
    """

    def __init__(self):
        self.servers: Dict[str, int] = {}  # server_id -> weight
        self.total_weight = 0

    def add_server(self, server_id: str, weight: int):
        """Add a server to the map."""
        if server_id in self.servers:
            self.total_weight -= self.servers[server_id]

        self.servers[server_id] = weight
        self.total_weight += weight

    def remove_server(self, server_id: str):
        """Remove a server from the map."""
        if server_id in self.servers:
            self.total_weight -= self.servers[server_id]
            del self.servers[server_id]

    def select_servers(self, object_id: int, count: int) -> List[str]:
        """
        Select servers for placing an object.

        Args:
            object_id: Object (block) ID
            count: Number of replicas needed

        Returns:
            List of server IDs
        """
        if not self.servers:
            return []

        selected = []
        server_list = list(self.servers.keys())

        for replica in range(min(count, len(server_list))):
            # CRUSH hash
            hash_input = f"{object_id}:{replica}".encode()
            hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)

            # Straw2 algorithm
            best_server = None
            best_draw = -1

            for server_id in server_list:
                if server_id in selected:
                    continue

                weight = self.servers[server_id]
                if weight == 0:
                    continue

                # Compute draw
                server_hash = f"{hash_value}:{server_id}".encode()
                draw = int(hashlib.sha256(server_hash).hexdigest(), 16)

                # Weight the draw
                weighted_draw = draw * weight

                if weighted_draw > best_draw:
                    best_draw = weighted_draw
                    best_server = server_id

            if best_server:
                selected.append(best_server)

        return selected

    def get_movement_factor(self, old_weight: int, new_weight: int) -> float:
        """
        Calculate expected data movement when weight changes.

        Returns fraction of data that needs to move.
        """
        if self.total_weight == 0:
            return 1.0

        # Approximate movement based on weight change
        delta = abs(new_weight - old_weight)
        return delta / (self.total_weight + delta)
```

### 4. Chunk Server

```python
import os
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

class ChunkServer:
    """
    Stores actual data chunks.
    Handles read/write requests and replication.
    """

    def __init__(self, server_id: str, data_dir: str, config: dict):
        self.server_id = server_id
        self.data_dir = data_dir
        self.config = config

        # MDS connection
        self.mds_host = config['mds_host']
        self.mds_port = config['mds_port']

        # Storage
        self.block_size = config.get('block_size', 64 * 1024 * 1024)
        self._chunks: Dict[int, str] = {}  # block_id -> file_path
        self._chunk_lock = threading.RLock()

        # Capacity
        self.total_space = config.get('total_space', 1024**4)  # 1TB default
        self.used_space = 0

        # Network
        self.host = config.get('host', 'localhost')
        self.port = config.get('port', 50010)

        # Background tasks
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._running = False

        # Initialize storage
        os.makedirs(data_dir, exist_ok=True)
        self._scan_existing_chunks()

    def start(self):
        """Start the chunk server."""
        self._running = True

        # Register with MDS
        self._register_with_mds()

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self._heartbeat_thread.start()

        # Start server socket
        self._server_thread = threading.Thread(target=self._server_loop)
        self._server_thread.start()

        logger.info(f"Chunk server {self.server_id} started on {self.host}:{self.port}")

    def stop(self):
        """Stop the chunk server."""
        self._running = False

    # Data operations

    def write_chunk(self, block_id: int, data: bytes, offset: int,
                    secondaries: List[str]) -> dict:
        """
        Write data to a chunk.

        GFS-style: primary writes then forwards to secondaries.
        """
        # Compute checksum
        checksum = hashlib.md5(data).hexdigest()

        # Write locally
        chunk_path = self._get_chunk_path(block_id)
        with self._chunk_lock:
            # Create or update chunk file
            mode = 'r+b' if os.path.exists(chunk_path) else 'wb'
            with open(chunk_path, mode) as f:
                f.seek(offset)
                f.write(data)

            self._chunks[block_id] = chunk_path

        # Forward to secondaries
        success_count = 1  # Self
        for secondary in secondaries:
            try:
                self._forward_to_secondary(secondary, block_id, data, offset)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to replicate to {secondary}: {e}")

        # Report to MDS
        file_size = os.path.getsize(chunk_path)
        self._report_block_to_mds(block_id, file_size, checksum)

        return {
            'status': 'ok',
            'block_id': block_id,
            'checksum': checksum,
            'replicated_to': success_count
        }

    def read_chunk(self, block_id: int, offset: int, length: int) -> bytes:
        """Read data from a chunk."""
        chunk_path = self._chunks.get(block_id)
        if chunk_path is None or not os.path.exists(chunk_path):
            raise FileNotFoundError(f"Chunk not found: {block_id}")

        with open(chunk_path, 'rb') as f:
            f.seek(offset)
            data = f.read(length)

        return data

    def delete_chunk(self, block_id: int):
        """Delete a chunk."""
        with self._chunk_lock:
            chunk_path = self._chunks.get(block_id)
            if chunk_path and os.path.exists(chunk_path):
                os.remove(chunk_path)
                del self._chunks[block_id]
                logger.info(f"Deleted chunk {block_id}")

    def get_chunk_checksum(self, block_id: int) -> str:
        """Compute checksum of stored chunk."""
        chunk_path = self._chunks.get(block_id)
        if chunk_path is None:
            raise FileNotFoundError(f"Chunk not found: {block_id}")

        with open(chunk_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    # Replication

    def replicate_chunk(self, block_id: int, target_server: str):
        """Replicate a chunk to another server."""
        chunk_path = self._chunks.get(block_id)
        if chunk_path is None:
            raise FileNotFoundError(f"Chunk not found: {block_id}")

        with open(chunk_path, 'rb') as f:
            data = f.read()

        # Send to target
        self._forward_to_secondary(target_server, block_id, data, 0)
        logger.info(f"Replicated block {block_id} to {target_server}")

    def _forward_to_secondary(self, server_id: str, block_id: int,
                              data: bytes, offset: int):
        """Forward write to secondary chunk server."""
        # In production, use proper RPC
        # Simplified: direct socket connection
        pass

    # MDS communication

    def _register_with_mds(self):
        """Register this server with metadata server."""
        # RPC call to MDS
        pass

    def _heartbeat_loop(self):
        """Send periodic heartbeats to MDS."""
        while self._running:
            try:
                # Collect block list
                blocks = list(self._chunks.keys())

                # Calculate used space
                self.used_space = sum(
                    os.path.getsize(path)
                    for path in self._chunks.values()
                    if os.path.exists(path)
                )

                # Send heartbeat
                response = self._send_heartbeat(blocks)

                # Process commands from MDS
                for cmd in response.get('commands', []):
                    self._process_command(cmd)

            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")

            time.sleep(3)  # Heartbeat interval

    def _send_heartbeat(self, blocks: List[int]) -> dict:
        """Send heartbeat to MDS."""
        # RPC call
        return {'status': 'ok', 'commands': []}

    def _report_block_to_mds(self, block_id: int, size: int, checksum: str):
        """Report block received to MDS."""
        # RPC call
        pass

    def _process_command(self, cmd: dict):
        """Process command from MDS."""
        if cmd['command'] == 'DELETE_BLOCK':
            self.delete_chunk(cmd['block_id'])

        elif cmd['command'] == 'REPLICATE':
            self._executor.submit(
                self.replicate_chunk,
                cmd['block_id'],
                cmd['target']
            )

    # Storage management

    def _get_chunk_path(self, block_id: int) -> str:
        """Get file path for a chunk."""
        # Use subdirectories to avoid too many files in one dir
        subdir = str(block_id % 256)
        dir_path = os.path.join(self.data_dir, subdir)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"blk_{block_id}")

    def _scan_existing_chunks(self):
        """Scan data directory for existing chunks."""
        for subdir in os.listdir(self.data_dir):
            subdir_path = os.path.join(self.data_dir, subdir)
            if not os.path.isdir(subdir_path):
                continue

            for filename in os.listdir(subdir_path):
                if filename.startswith('blk_'):
                    block_id = int(filename[4:])
                    self._chunks[block_id] = os.path.join(subdir_path, filename)

        logger.info(f"Found {len(self._chunks)} existing chunks")

    # Server loop

    def _server_loop(self):
        """Main server loop for handling client requests."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(100)

        while self._running:
            try:
                client_socket, address = server_socket.accept()
                self._executor.submit(self._handle_client, client_socket)
            except Exception as e:
                if self._running:
                    logger.error(f"Server error: {e}")

        server_socket.close()

    def _handle_client(self, client_socket: socket.socket):
        """Handle a client connection."""
        try:
            # Read request
            request = self._receive_message(client_socket)

            # Process based on operation
            if request['op'] == 'READ':
                data = self.read_chunk(
                    request['block_id'],
                    request['offset'],
                    request['length']
                )
                self._send_response(client_socket, {'status': 'ok', 'data': data})

            elif request['op'] == 'WRITE':
                result = self.write_chunk(
                    request['block_id'],
                    request['data'],
                    request['offset'],
                    request.get('secondaries', [])
                )
                self._send_response(client_socket, result)

            elif request['op'] == 'CHECKSUM':
                checksum = self.get_chunk_checksum(request['block_id'])
                self._send_response(client_socket, {'status': 'ok', 'checksum': checksum})

        except Exception as e:
            self._send_response(client_socket, {'status': 'error', 'message': str(e)})

        finally:
            client_socket.close()

    def _receive_message(self, sock: socket.socket) -> dict:
        """Receive a message from socket."""
        # Simplified - use proper framing in production
        import json
        data = sock.recv(1024 * 1024)
        return json.loads(data.decode())

    def _send_response(self, sock: socket.socket, response: dict):
        """Send response to socket."""
        import json
        data = json.dumps(response).encode()
        sock.sendall(data)
```

### 5. Client Library

```python
class DFSClient:
    """
    Client library for distributed file system.
    Provides file-like interface.
    """

    def __init__(self, config: dict):
        self.config = config
        self.mds_host = config['mds_host']
        self.mds_port = config['mds_port']

        self.client_id = str(uuid.uuid4())
        self.block_size = config.get('block_size', 64 * 1024 * 1024)

        # Caching
        self._block_cache: Dict[int, bytes] = {}
        self._location_cache: Dict[int, List[str]] = {}

        # Connection pool
        self._connections: Dict[str, socket.socket] = {}

    # File operations

    def open(self, path: str, mode: str = 'r') -> 'DFSFile':
        """Open a file."""
        return DFSFile(self, path, mode)

    def create(self, path: str, replication: int = 3) -> int:
        """Create a new file."""
        response = self._mds_rpc('CREATE_FILE', {
            'path': path,
            'replication': replication
        })
        return response['inode_id']

    def delete(self, path: str):
        """Delete a file."""
        self._mds_rpc('DELETE', {'path': path})

    def mkdir(self, path: str):
        """Create directory."""
        self._mds_rpc('MKDIR', {'path': path})

    def ls(self, path: str) -> List[dict]:
        """List directory contents."""
        response = self._mds_rpc('LIST', {'path': path})
        return response['entries']

    def stat(self, path: str) -> dict:
        """Get file info."""
        response = self._mds_rpc('STAT', {'path': path})
        return response['info']

    # Block operations

    def read_block(self, block_id: int, offset: int, length: int) -> bytes:
        """Read data from a block."""
        # Check cache
        if block_id in self._block_cache:
            return self._block_cache[block_id][offset:offset + length]

        # Get locations
        locations = self._get_block_locations(block_id)
        if not locations:
            raise IOError(f"No locations for block {block_id}")

        # Try each location
        for server_id in locations:
            try:
                data = self._read_from_server(server_id, block_id, offset, length)
                return data
            except Exception as e:
                logger.warning(f"Failed to read from {server_id}: {e}")

        raise IOError(f"Failed to read block {block_id} from any server")

    def write_block(self, inode_id: int, data: bytes) -> int:
        """Write data to a new block."""
        # Allocate block
        response = self._mds_rpc('ALLOCATE_BLOCK', {'inode_id': inode_id})
        block_id = response['block_id']
        servers = response['servers']

        if not servers:
            raise IOError("No servers available for write")

        # Get lease
        lease = self._get_lease(block_id)

        # Write to primary (primary forwards to secondaries)
        primary = lease['primary_server']
        secondaries = [s for s in servers if s != primary]

        result = self._write_to_server(
            primary, block_id, data, 0, secondaries
        )

        return block_id

    def append_to_block(self, block_id: int, data: bytes) -> int:
        """Append data to existing block."""
        # Get lease for writing
        lease = self._get_lease(block_id)

        # Get current size
        locations = self._get_block_locations(block_id)
        primary = lease['primary_server']
        secondaries = [s for s in locations if s != primary]

        # Get current offset
        block_info = self._mds_rpc('GET_BLOCK_INFO', {'block_id': block_id})
        offset = block_info['size']

        # Write at offset
        result = self._write_to_server(
            primary, block_id, data, offset, secondaries
        )

        return offset

    # Helper methods

    def _get_block_locations(self, block_id: int) -> List[str]:
        """Get chunk servers storing a block."""
        if block_id in self._location_cache:
            return self._location_cache[block_id]

        response = self._mds_rpc('GET_BLOCK_LOCATIONS', {'block_id': block_id})
        locations = response['locations']
        self._location_cache[block_id] = locations
        return locations

    def _get_lease(self, block_id: int) -> dict:
        """Get write lease for a block."""
        response = self._mds_rpc('GET_LEASE', {
            'block_id': block_id,
            'client_id': self.client_id
        })
        return response['lease']

    def _read_from_server(self, server_id: str, block_id: int,
                          offset: int, length: int) -> bytes:
        """Read data from a chunk server."""
        sock = self._get_connection(server_id)
        request = {
            'op': 'READ',
            'block_id': block_id,
            'offset': offset,
            'length': length
        }

        self._send_message(sock, request)
        response = self._receive_message(sock)

        if response['status'] != 'ok':
            raise IOError(response.get('message', 'Read failed'))

        return response['data']

    def _write_to_server(self, server_id: str, block_id: int,
                         data: bytes, offset: int,
                         secondaries: List[str]) -> dict:
        """Write data to a chunk server."""
        sock = self._get_connection(server_id)
        request = {
            'op': 'WRITE',
            'block_id': block_id,
            'data': data.decode('latin-1'),  # Encode bytes as string for JSON
            'offset': offset,
            'secondaries': secondaries
        }

        self._send_message(sock, request)
        response = self._receive_message(sock)

        if response['status'] != 'ok':
            raise IOError(response.get('message', 'Write failed'))

        return response

    def _mds_rpc(self, method: str, params: dict) -> dict:
        """Make RPC call to metadata server."""
        # Simplified RPC
        import json
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.mds_host, self.mds_port))

        request = {'method': method, 'params': params}
        sock.sendall(json.dumps(request).encode())

        response = json.loads(sock.recv(1024 * 1024).decode())
        sock.close()

        if response.get('status') == 'error':
            raise IOError(response.get('message', 'RPC failed'))

        return response

    def _get_connection(self, server_id: str) -> socket.socket:
        """Get or create connection to chunk server."""
        # In production, use connection pooling
        # Simplified: return new connection
        return socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def _send_message(self, sock: socket.socket, message: dict):
        """Send message to socket."""
        import json
        sock.sendall(json.dumps(message).encode())

    def _receive_message(self, sock: socket.socket) -> dict:
        """Receive message from socket."""
        import json
        return json.loads(sock.recv(1024 * 1024).decode())

class DFSFile:
    """File handle for distributed file system."""

    def __init__(self, client: DFSClient, path: str, mode: str):
        self.client = client
        self.path = path
        self.mode = mode
        self._position = 0

        # Get file info
        info = client.stat(path)
        self.inode_id = info['inode_id']
        self.size = info['size']
        self.blocks = info.get('block_ids', [])

    def read(self, size: int = -1) -> bytes:
        """Read from file."""
        if 'r' not in self.mode:
            raise IOError("File not open for reading")

        if size < 0:
            size = self.size - self._position

        data = b''
        remaining = size

        while remaining > 0 and self._position < self.size:
            # Find which block
            block_idx = self._position // self.client.block_size
            block_offset = self._position % self.client.block_size

            if block_idx >= len(self.blocks):
                break

            block_id = self.blocks[block_idx]

            # Read from block
            read_size = min(
                remaining,
                self.client.block_size - block_offset,
                self.size - self._position
            )

            block_data = self.client.read_block(block_id, block_offset, read_size)
            data += block_data

            self._position += len(block_data)
            remaining -= len(block_data)

        return data

    def write(self, data: bytes) -> int:
        """Write to file."""
        if 'w' not in self.mode and 'a' not in self.mode:
            raise IOError("File not open for writing")

        written = 0

        while written < len(data):
            # Calculate how much fits in current block
            block_offset = self._position % self.client.block_size
            space_in_block = self.client.block_size - block_offset

            chunk = data[written:written + space_in_block]

            if block_offset == 0:
                # Need new block
                block_id = self.client.write_block(self.inode_id, chunk)
                self.blocks.append(block_id)
            else:
                # Append to existing block
                block_id = self.blocks[-1]
                self.client.append_to_block(block_id, chunk)

            written += len(chunk)
            self._position += len(chunk)
            self.size = max(self.size, self._position)

        return written

    def seek(self, position: int):
        """Seek to position."""
        self._position = position

    def tell(self) -> int:
        """Get current position."""
        return self._position

    def close(self):
        """Close the file."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

## Implementation Phases

### Phase 1: Core Data Structures (Weeks 1-2)
- [ ] Inode and Block classes
- [ ] Serialization/deserialization
- [ ] ChunkServerInfo and Lease
- [ ] File/directory types
- [ ] Basic tests

### Phase 2: Metadata Server (Weeks 3-5)
- [ ] Namespace operations (create, delete, list)
- [ ] Path resolution
- [ ] Block allocation
- [ ] Write-ahead log
- [ ] Lease management

### Phase 3: Chunk Server (Weeks 6-8)
- [ ] Chunk storage and retrieval
- [ ] Checksum verification
- [ ] Heartbeat mechanism
- [ ] Replication forwarding
- [ ] Command processing

### Phase 4: Client Library (Weeks 9-11)
- [ ] File operations API
- [ ] Block read/write
- [ ] Connection management
- [ ] Location caching
- [ ] DFSFile implementation

### Phase 5: Consistency (Weeks 12-14)
- [ ] CRUSH placement algorithm
- [ ] Primary/secondary replication
- [ ] Lease-based writes
- [ ] Read consistency
- [ ] Generation stamps

### Phase 6: Self-Healing (Weeks 15-17)
- [ ] Under-replication detection
- [ ] Re-replication scheduling
- [ ] Block deletion
- [ ] Server failure handling
- [ ] Checksums and verification

### Phase 7: Enterprise Features (Weeks 18-20)
- [ ] Snapshots (stretch)
- [ ] Tiered storage (stretch)
- [ ] FUSE adapter (stretch)
- [ ] Performance optimization
- [ ] Monitoring and metrics

## Testing Strategy

### Unit Tests
```python
class TestInode:
    def test_serialization(self):
        """Test inode serialization round-trip."""
        inode = Inode(
            inode_id=1,
            parent_id=0,
            name="test.txt",
            file_type=FileType.FILE
        )
        inode.size = 1000
        inode.blocks = [1, 2, 3]

        data = inode.to_bytes()
        restored = Inode.from_bytes(data)

        assert restored.inode_id == inode.inode_id
        assert restored.blocks == inode.blocks

class TestCRUSH:
    def test_placement_stability(self):
        """Test that placement is stable."""
        crush = CRUSHMap()
        for i in range(10):
            crush.add_server(f"server_{i}", 100)

        # Same input should give same output
        servers1 = crush.select_servers(12345, 3)
        servers2 = crush.select_servers(12345, 3)

        assert servers1 == servers2

    def test_rebalancing(self):
        """Test minimal data movement on server add."""
        crush = CRUSHMap()
        for i in range(5):
            crush.add_server(f"server_{i}", 100)

        # Place some blocks
        placements_before = {
            i: crush.select_servers(i, 3)
            for i in range(100)
        }

        # Add server
        crush.add_server("server_5", 100)

        # Check movement
        moved = 0
        for i in range(100):
            new_placement = crush.select_servers(i, 3)
            if set(new_placement) != set(placements_before[i]):
                moved += 1

        # Should move approximately 1/6 of data
        assert moved < 30  # Allow some variance

class TestMetadataServer:
    def test_create_file(self):
        """Test file creation."""
        mds = MetadataServer({'wal_path': '/tmp/test.wal'})
        inode_id = mds.create_file("/test.txt")

        assert inode_id > 0

        info = mds.get_file_info("/test.txt")
        assert info['name'] == 'test.txt'
        assert info['type'] == 'file'

    def test_nested_directory(self):
        """Test nested directory creation."""
        mds = MetadataServer({'wal_path': '/tmp/test.wal'})

        mds.create_directory("/a")
        mds.create_directory("/a/b")
        mds.create_file("/a/b/file.txt")

        entries = mds.list_directory("/a/b")
        assert len(entries) == 1
        assert entries[0][0] == 'file.txt'
```

### Integration Tests
```python
class TestDistributedFS:
    def test_write_read_cycle(self):
        """Test writing and reading data."""
        # Setup cluster
        mds = MetadataServer(config)
        servers = [ChunkServer(f"cs{i}", f"/tmp/cs{i}", config)
                   for i in range(3)]

        for server in servers:
            server.start()

        client = DFSClient(config)

        # Write file
        with client.open("/test.txt", "w") as f:
            f.write(b"Hello, distributed world!")

        # Read back
        with client.open("/test.txt", "r") as f:
            data = f.read()

        assert data == b"Hello, distributed world!"

    def test_replication(self):
        """Test that data is replicated."""
        # Write data with replication=3
        client.create("/replicated.txt", replication=3)

        with client.open("/replicated.txt", "w") as f:
            f.write(b"replicate me")

        # Verify on multiple servers
        # ...
```

## Performance Targets

| Operation | Target Latency | Throughput |
|-----------|---------------|------------|
| File create | < 10 ms | - |
| File open | < 5 ms | - |
| Sequential read | - | 100 MB/s |
| Sequential write | - | 50 MB/s |
| Directory list | < 20 ms | - |

## Dependencies

- Python 3.8+
- (Optional) FUSE for filesystem mount

## References

- GFS Paper: The Google File System
- HDFS Architecture Guide
- Ceph: A Scalable, High-Performance Distributed File System
- CRUSH: Controlled, Scalable, Decentralized Placement of Replicated Data
