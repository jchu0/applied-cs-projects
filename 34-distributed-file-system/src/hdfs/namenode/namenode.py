"""NameNode implementation for HDFS."""

import asyncio
import time
import logging
import json
import os
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

from ..common.types import (
    BlockID, NodeID, Block, BlockLocation, FileInfo, DirectoryInfo,
    DataNodeInfo, ReplicationPolicy, BlockReport, HeartbeatResponse,
    generate_block_id
)
from ..common.protocol import (
    Message, MessageType, HDFSError, FileNotFoundError,
    FileExistsError, DirectoryNotEmptyError, NoDataNodeError
)

logger = logging.getLogger(__name__)


class NameNode:
    """
    NameNode manages the filesystem namespace and block mapping.

    Responsibilities:
    - Maintain filesystem tree (directories and files)
    - Track block-to-DataNode mapping
    - Handle DataNode registration and heartbeats
    - Manage block replication
    """

    def __init__(
        self,
        default_replication: int = 3,
        default_block_size: int = 128 * 1024 * 1024,
        heartbeat_interval: float = 3.0,
        checkpoint_interval: float = 3600.0
    ):
        self.default_replication = default_replication
        self.default_block_size = default_block_size
        self.heartbeat_interval = heartbeat_interval
        self.checkpoint_interval = checkpoint_interval

        # Namespace
        self._files: Dict[str, FileInfo] = {}
        self._directories: Dict[str, DirectoryInfo] = {"/": DirectoryInfo("/")}

        # Block mapping
        self._blocks: Dict[BlockID, Block] = {}
        self._block_to_nodes: Dict[BlockID, Set[NodeID]] = defaultdict(set)

        # DataNode tracking
        self._datanodes: Dict[NodeID, DataNodeInfo] = {}

        # Pending operations
        # Each pending replication is (block_id, source_node_id, target_node_id):
        # the target (which lacks the block) is told to pull it from the source.
        self._pending_replications: List[Tuple[BlockID, NodeID, NodeID]] = []
        self._pending_deletions: Dict[NodeID, List[BlockID]] = defaultdict(list)

        # In-flight replications, so we don't re-schedule the same target for a
        # block before its BLOCK_RECEIVED lands. Maps block_id -> set of targets.
        self._replications_in_flight: Dict[BlockID, Set[NodeID]] = defaultdict(set)

        # Checkpointing (set by NameNodeServer / start_background_tasks)
        self._checkpoint_path: Optional[str] = None

        # Safe mode
        self._safe_mode = True
        self._safe_mode_threshold = 0.999  # 99.9% of blocks must be reported

    # File operations

    def create_file(
        self,
        path: str,
        replication: Optional[int] = None,
        block_size: Optional[int] = None,
        overwrite: bool = False
    ) -> FileInfo:
        """Create a new file."""
        # Check if exists
        if path in self._files:
            if not overwrite:
                raise FileExistsError(f"File already exists: {path}")
            self.delete_file(path)

        # Check parent directory
        parent_path = self._get_parent_path(path)
        if parent_path not in self._directories:
            raise FileNotFoundError(f"Parent directory not found: {parent_path}")

        # Create file
        file_info = FileInfo(
            path=path,
            replication=replication or self.default_replication,
            block_size=block_size or self.default_block_size
        )

        self._files[path] = file_info

        # Add to parent directory
        filename = self._get_filename(path)
        self._directories[parent_path].children.add(filename)
        self._directories[parent_path].modification_time = time.time()

        logger.info(f"Created file: {path}")
        return file_info

    def add_block(self, path: str) -> Tuple[Block, List[BlockLocation]]:
        """Add a new block to a file being written."""
        # Check safe mode - block allocation is not allowed in safe mode
        if self._safe_mode:
            raise HDFSError("NameNode is in safe mode. Cannot allocate blocks.")

        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")

        # Select DataNodes for this block
        targets = self._select_datanodes_for_block(
            self._files[path].replication
        )

        if not targets:
            raise NoDataNodeError("No DataNodes available for block allocation")

        # Create block
        block = Block(block_id=generate_block_id())
        self._blocks[block.block_id] = block
        self._files[path].blocks.append(block.block_id)

        # Create locations
        locations = []
        for node_id in targets:
            node = self._datanodes[node_id]
            locations.append(BlockLocation(
                block_id=block.block_id,
                node_id=node_id,
                host=node.host,
                port=node.port,
                rack=node.rack
            ))

        logger.info(f"Added block {block.block_id} to {path}, targets: {targets}")
        return block, locations

    def complete_file(self, path: str, size: int) -> FileInfo:
        """Complete file creation."""
        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")

        # Check quota with actual file size
        self._check_quota_for_write(path, size)

        file_info = self._files[path]
        file_info.size = size
        file_info.modification_time = time.time()

        logger.info(f"Completed file: {path}, size: {size}")
        return file_info

    def get_block_locations(self, path_or_block_id: str):
        """Get block locations for a file path or a specific block ID.

        If path is a block ID, returns List[BlockLocation] for that block.
        If path is a file path, returns List[List[BlockLocation]] grouped by block.
        """
        # Check if it's a block ID (starts with blk_)
        if path_or_block_id.startswith("blk_"):
            return self.get_block_locations_by_id(path_or_block_id)

        # It's a file path - return grouped by block
        return self.get_block_locations_for_file(path_or_block_id)

    def get_block_locations_for_file(self, path: str) -> List[List[BlockLocation]]:
        """Get block locations grouped by block for a file."""
        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")

        file_info = self._files[path]
        all_locations = []

        for block_id in file_info.blocks:
            locations = self.get_block_locations_by_id(block_id)
            all_locations.append(locations)

        return all_locations

    def delete_file(self, path: str) -> bool:
        """Delete a file."""
        if path not in self._files:
            raise FileNotFoundError(f"File not found: {path}")

        file_info = self._files[path]

        # Mark blocks for deletion
        for block_id in file_info.blocks:
            for node_id in self._block_to_nodes.get(block_id, set()):
                self._pending_deletions[node_id].append(block_id)
            del self._blocks[block_id]
            if block_id in self._block_to_nodes:
                del self._block_to_nodes[block_id]

        # Remove from parent
        parent_path = self._get_parent_path(path)
        filename = self._get_filename(path)
        if parent_path in self._directories:
            self._directories[parent_path].children.discard(filename)

        del self._files[path]
        logger.info(f"Deleted file: {path}")
        return True

    def rename(self, src: str, dst: str) -> bool:
        """Rename file or directory."""
        # Check if destination exists
        if dst in self._files or dst in self._directories:
            raise FileExistsError(f"Destination already exists: {dst}")

        if src in self._files:
            file_info = self._files.pop(src)
            file_info.path = dst
            self._files[dst] = file_info
        elif src in self._directories:
            dir_info = self._directories.pop(src)
            dir_info.path = dst
            self._directories[dst] = dir_info
            # Update children paths
            self._update_subtree_paths(src, dst)
        else:
            raise FileNotFoundError(f"Path not found: {src}")

        # Update parent directories
        src_parent = self._get_parent_path(src)
        dst_parent = self._get_parent_path(dst)
        src_name = self._get_filename(src)
        dst_name = self._get_filename(dst)

        if src_parent in self._directories:
            self._directories[src_parent].children.discard(src_name)
        if dst_parent in self._directories:
            self._directories[dst_parent].children.add(dst_name)

        logger.info(f"Renamed {src} to {dst}")
        return True

    # Directory operations

    def mkdir(self, path: str, create_parents: bool = False) -> DirectoryInfo:
        """Create a directory."""
        if path in self._directories:
            raise FileExistsError(f"Directory already exists: {path}")

        parent_path = self._get_parent_path(path)

        if parent_path not in self._directories:
            if create_parents:
                self.mkdir(parent_path, create_parents=True)
            else:
                raise FileNotFoundError(f"Parent directory not found: {parent_path}")

        dir_info = DirectoryInfo(path=path)
        self._directories[path] = dir_info

        # Add to parent
        dirname = self._get_filename(path)
        self._directories[parent_path].children.add(dirname)

        logger.info(f"Created directory: {path}")
        return dir_info

    def create_directory(self, path: str, create_parents: bool = False) -> DirectoryInfo:
        """Create a directory (alias for mkdir)."""
        return self.mkdir(path, create_parents)

    def delete_directory(self, path: str, recursive: bool = False) -> bool:
        """Delete a directory."""
        if path not in self._directories:
            raise FileNotFoundError(f"Directory not found: {path}")

        if path == "/":
            raise ValueError("Cannot delete root directory")

        dir_info = self._directories[path]

        if dir_info.children and not recursive:
            raise DirectoryNotEmptyError(f"Directory not empty: {path}")

        if recursive:
            # Delete all children first
            for child in list(dir_info.children):
                child_path = f"{path}/{child}" if path != "/" else f"/{child}"
                if child_path in self._directories:
                    self.delete_directory(child_path, recursive=True)
                elif child_path in self._files:
                    self.delete_file(child_path)

        # Remove from parent
        parent_path = self._get_parent_path(path)
        if parent_path in self._directories:
            dirname = self._get_filename(path)
            self._directories[parent_path].children.discard(dirname)

        del self._directories[path]
        logger.info(f"Deleted directory: {path}")
        return True

    def list_directory(self, path: str) -> List[str]:
        """List contents of a directory (returns list of names)."""
        if path not in self._directories:
            raise FileNotFoundError(f"Directory not found: {path}")

        dir_info = self._directories[path]
        return sorted(dir_info.children)

    def list_directory_detailed(self, path: str) -> List[Dict]:
        """List contents of a directory with detailed info."""
        if path not in self._directories:
            raise FileNotFoundError(f"Directory not found: {path}")

        result = []
        dir_info = self._directories[path]

        for name in sorted(dir_info.children):
            child_path = f"{path.rstrip('/')}/{name}"

            if child_path in self._files:
                file_info = self._files[child_path]
                result.append({
                    "name": name,
                    "type": "file",
                    "size": file_info.size,
                    "replication": file_info.replication,
                    "modification_time": file_info.modification_time
                })
            elif child_path in self._directories:
                dir_child = self._directories[child_path]
                result.append({
                    "name": name,
                    "type": "directory",
                    "modification_time": dir_child.modification_time
                })

        return result

    def get_file_info(self, path: str, raise_if_missing: bool = True) -> Optional[FileInfo]:
        """Get file info. Returns FileInfo object or raises FileNotFoundError."""
        if path in self._files:
            return self._files[path]
        elif path in self._directories:
            # Return DirectoryInfo but tests may expect FileInfo
            return self._directories[path]

        if raise_if_missing:
            raise FileNotFoundError(f"Path not found: {path}")
        return None

    def get_file_info_dict(self, path: str) -> Optional[Dict]:
        """Get file or directory info as dict (for protocol responses)."""
        if path in self._files:
            f = self._files[path]
            return {
                "path": f.path,
                "type": "file",
                "size": f.size,
                "block_size": f.block_size,
                "replication": f.replication,
                "num_blocks": f.num_blocks,
                "blocks": f.blocks,
                "modification_time": f.modification_time,
                "owner": f.owner,
                "group": f.group,
                "permission": f.permission
            }
        elif path in self._directories:
            d = self._directories[path]
            return {
                "path": d.path,
                "type": "directory",
                "modification_time": d.modification_time,
                "owner": d.owner,
                "group": d.group,
                "permission": d.permission
            }
        return None

    # DataNode management

    def register_datanode(
        self,
        node_id: NodeID,
        host: str,
        port: int,
        capacity: int = 100 * 1024 * 1024 * 1024,  # 100GB default
        used: int = 0,
        remaining: Optional[int] = None,
        rack: str = "/default-rack"
    ) -> bool:
        """Register a new DataNode."""
        if remaining is None:
            remaining = capacity - used
        self._datanodes[node_id] = DataNodeInfo(
            node_id=node_id,
            host=host,
            port=port,
            rack=rack,
            capacity=capacity,
            used=used,
            remaining=remaining,
            last_heartbeat=time.time()
        )
        logger.info(f"Registered DataNode: {node_id} at {host}:{port}")
        return True

    def heartbeat(self, node_id: NodeID, used: int, remaining: int) -> HeartbeatResponse:
        """Process DataNode heartbeat."""
        if node_id not in self._datanodes:
            return HeartbeatResponse(commands=[{"type": "re-register"}])

        node = self._datanodes[node_id]
        node.last_heartbeat = time.time()
        node.used = used
        node.remaining = remaining

        # Check for dead nodes on each heartbeat
        self.check_and_remove_dead_nodes()

        # Collect commands for this node
        commands = []

        # Pending deletions
        if node_id in self._pending_deletions:
            block_ids = self._pending_deletions.pop(node_id)
            if block_ids:
                commands.append({
                    "type": "delete",
                    "block_ids": block_ids
                })

        # Pending replications (pull model): if this node is the target for a
        # scheduled replication, tell it to copy the block from the source.
        for entry in list(self._pending_replications):
            block_id, source_id, target_id = entry
            if target_id != node_id:
                continue
            source = self._datanodes.get(source_id)
            if source is None or not source.is_alive:
                # Source vanished; drop this task, the monitor will reschedule.
                self._pending_replications.remove(entry)
                self._replications_in_flight[block_id].discard(target_id)
                continue
            commands.append({
                "type": "replicate",
                "block_id": block_id,
                "source": {
                    "node_id": source_id,
                    "host": source.host,
                    "port": source.port,
                },
            })
            self._pending_replications.remove(entry)

        return HeartbeatResponse(commands=commands)

    def block_report(self, report: BlockReport) -> None:
        """Process block report from DataNode."""
        node_id = report.node_id

        if node_id not in self._datanodes:
            logger.warning(f"Block report from unknown node: {node_id}")
            return

        # Clear old mapping for this node
        for block_id in list(self._block_to_nodes.keys()):
            self._block_to_nodes[block_id].discard(node_id)

        # Update with new blocks
        self._datanodes[node_id].blocks = set(report.blocks)

        for block_id in report.blocks:
            if block_id in self._blocks:
                self._block_to_nodes[block_id].add(node_id)

        logger.info(f"Processed block report from {node_id}: {len(report.blocks)} blocks")

        # Check safe mode
        self._check_safe_mode()

    def block_received(self, node_id: NodeID, block_id: BlockID, size: int) -> None:
        """DataNode reports receiving a block."""
        if block_id in self._blocks:
            self._blocks[block_id].size = size
            self._block_to_nodes[block_id].add(node_id)
            if node_id in self._datanodes:
                self._datanodes[node_id].blocks.add(block_id)

        # A replica landed here, so any in-flight re-replication to this node
        # for this block is done.
        self._replications_in_flight[block_id].discard(node_id)

        logger.info(f"Block {block_id} received at {node_id}, size: {size}")

    # Test-compatible aliases and additional methods

    def handle_heartbeat(
        self,
        node_id: NodeID,
        used: int = 0,
        remaining: int = 0,
        capacity: int = None
    ) -> HeartbeatResponse:
        """Handle heartbeat from DataNode (extended version with capacity update)."""
        if node_id not in self._datanodes:
            return HeartbeatResponse(commands=[{"type": "re-register"}])

        node = self._datanodes[node_id]
        node.last_heartbeat = time.time()
        node.used = used
        node.remaining = remaining
        if capacity is not None:
            node.capacity = capacity

        # Check for dead nodes on each heartbeat
        self.check_and_remove_dead_nodes()

        # Collect commands for this node
        commands = []

        # Pending deletions
        if node_id in self._pending_deletions:
            block_ids = self._pending_deletions.pop(node_id)
            if block_ids:
                commands.append({
                    "type": "delete",
                    "block_ids": block_ids
                })

        return HeartbeatResponse(commands=commands)

    def handle_block_report(self, node_id: NodeID, block_ids: List[BlockID]) -> None:
        """Handle block report from DataNode (simplified interface)."""
        if node_id not in self._datanodes:
            logger.warning(f"Block report from unknown node: {node_id}")
            return

        # Clear old mapping for this node
        for block_id in list(self._block_to_nodes.keys()):
            self._block_to_nodes[block_id].discard(node_id)

        # Update with new blocks - add to mapping even if block not known
        # (block might exist on DataNode from previous run)
        self._datanodes[node_id].blocks = set(block_ids)

        for block_id in block_ids:
            self._block_to_nodes[block_id].add(node_id)

        logger.info(f"Processed block report from {node_id}: {len(block_ids)} blocks")
        self._check_safe_mode()

    def allocate_blocks(self, path: str, num_blocks: int, block_size: int) -> List[Block]:
        """Allocate multiple blocks (creates file if needed for test compatibility)."""
        # Create file if it doesn't exist (for test compatibility)
        if path not in self._files:
            parent = self._get_parent_path(path)
            if parent not in self._directories and parent != "/":
                self.mkdir(parent, create_parents=True)
            self.create_file(path)

        result = []
        for _ in range(num_blocks):
            block, locations = self.add_block(path)
            block.size = block_size
            result.append(block)

        return result

    def get_block_locations_by_id(self, block_id: BlockID) -> List[BlockLocation]:
        """Get locations for a specific block ID."""
        locations = []
        for node_id in self._block_to_nodes.get(block_id, set()):
            if node_id in self._datanodes:
                node = self._datanodes[node_id]
                if node.is_alive:
                    locations.append(BlockLocation(
                        block_id=block_id,
                        node_id=node_id,
                        host=node.host,
                        port=node.port,
                        rack=node.rack
                    ))
        return locations

    def _check_under_replicated_blocks(self) -> List[BlockID]:
        """Check for under-replicated blocks."""
        under_replicated = []

        for path, file_info in self._files.items():
            target_replication = file_info.replication

            for block_id in file_info.blocks:
                current_replicas = len(self._block_to_nodes.get(block_id, set()))
                if current_replicas < target_replication:
                    under_replicated.append(block_id)

        return under_replicated

    def _check_safe_mode_exit(self) -> bool:
        """Check if safe mode can be exited and exit if possible."""
        if not self._safe_mode:
            return True

        self._check_safe_mode()
        return not self._safe_mode

    # Internal methods

    def _select_datanodes_for_block(self, replication: int) -> List[NodeID]:
        """Select DataNodes for a new block with load balancing."""
        available = [
            (node_id, node)
            for node_id, node in self._datanodes.items()
            if node.is_alive and node.remaining > self.default_block_size
        ]

        if len(available) < replication:
            # Use all available if not enough
            return [node_id for node_id, _ in available]

        # Count blocks per DataNode for load balancing
        block_counts = {}
        for node_id, _ in available:
            block_counts[node_id] = sum(
                1 for nodes in self._block_to_nodes.values()
                if node_id in nodes
            )

        # Calculate load score: lower is better (fewer blocks + more space)
        # Normalize remaining space to [0, 1] and block count inversely
        max_remaining = max(node.remaining for _, node in available) or 1
        max_blocks = max(block_counts.values()) + 1  # +1 to avoid div by zero

        def load_score(node_id: str, node) -> float:
            # Weight: 60% block count, 40% remaining space
            block_factor = block_counts.get(node_id, 0) / max_blocks
            space_factor = 1 - (node.remaining / max_remaining)
            return 0.6 * block_factor + 0.4 * space_factor

        # Sort by load score (lowest first = best candidates)
        available.sort(key=lambda x: load_score(x[0], x[1]))

        # Rack-aware selection with load balancing
        selected = []
        racks_used = set()

        # First pass: select from different racks
        for node_id, node in available:
            if len(selected) >= replication:
                break
            if node.rack not in racks_used:
                selected.append(node_id)
                racks_used.add(node.rack)

        # Second pass: fill remaining slots if needed
        for node_id, node in available:
            if len(selected) >= replication:
                break
            if node_id not in selected:
                selected.append(node_id)

        return selected

    def _check_safe_mode(self) -> None:
        """Check if we can exit safe mode."""
        if not self._safe_mode:
            return

        total_blocks = len(self._blocks)
        if total_blocks == 0:
            self._safe_mode = False
            return

        reported_blocks = sum(
            1 for block_id in self._blocks
            if len(self._block_to_nodes.get(block_id, set())) > 0
        )

        ratio = reported_blocks / total_blocks
        if ratio >= self._safe_mode_threshold:
            self._safe_mode = False
            logger.info(f"Exiting safe mode: {ratio*100:.1f}% blocks reported")

    def _get_parent_path(self, path: str) -> str:
        """Get parent directory path."""
        if path == "/":
            return "/"
        parts = path.rstrip("/").rsplit("/", 1)
        return parts[0] if parts[0] else "/"

    def _get_filename(self, path: str) -> str:
        """Get filename from path."""
        return path.rstrip("/").rsplit("/", 1)[-1]

    def _update_subtree_paths(self, old_prefix: str, new_prefix: str) -> None:
        """Update paths in subtree after rename."""
        # Update files
        for path in list(self._files.keys()):
            if path.startswith(old_prefix + "/"):
                new_path = new_prefix + path[len(old_prefix):]
                file_info = self._files.pop(path)
                file_info.path = new_path
                self._files[new_path] = file_info

        # Update directories
        for path in list(self._directories.keys()):
            if path.startswith(old_prefix + "/"):
                new_path = new_prefix + path[len(old_prefix):]
                dir_info = self._directories.pop(path)
                dir_info.path = new_path
                self._directories[new_path] = dir_info

    # Statistics

    def get_statistics(self) -> Dict:
        """Get filesystem statistics."""
        total_capacity = sum(dn.capacity for dn in self._datanodes.values())
        total_used = sum(dn.used for dn in self._datanodes.values())
        total_remaining = sum(dn.remaining for dn in self._datanodes.values())

        return {
            'total_files': len(self._files),
            'total_directories': len(self._directories),
            'total_blocks': len(self._blocks),
            'total_datanodes': len(self._datanodes),
            'total_capacity': total_capacity,
            'total_used': total_used,
            'total_remaining': total_remaining,
            'safe_mode': self._safe_mode
        }

    # Quota management

    def set_quota(
        self,
        path: str,
        namespace_quota: Optional[int] = None,
        space_quota: Optional[int] = None
    ) -> None:
        """Set quota on a directory."""
        if path not in self._directories:
            raise FileNotFoundError(f"Directory not found: {path}")

        dir_info = self._directories[path]

        if namespace_quota is not None:
            dir_info.namespace_quota = namespace_quota
        if space_quota is not None:
            dir_info.space_quota = space_quota

        logger.info(f"Set quota on {path}: namespace={namespace_quota}, space={space_quota}")

    def get_quota(self, path: str) -> Dict:
        """Get quota information for a directory."""
        if path not in self._directories:
            raise FileNotFoundError(f"Directory not found: {path}")

        dir_info = self._directories[path]
        return {
            "namespace_quota": getattr(dir_info, 'namespace_quota', None),
            "space_quota": getattr(dir_info, 'space_quota', None),
            "namespace_count": len(dir_info.children),
            "space_consumed": self._calculate_space_consumed(path)
        }

    def _calculate_space_consumed(self, path: str) -> int:
        """Calculate total space consumed under a directory."""
        total = 0
        for file_path, file_info in self._files.items():
            if file_path.startswith(path):
                total += file_info.size
        return total

    def _check_quota_for_write(self, path: str, size: int) -> None:
        """Check if write would exceed quota. Raises HDFSError if quota exceeded."""
        # Walk up the path to find parent directories with quotas
        current_path = path
        while current_path != "/":
            parent_path = self._get_parent_path(current_path)
            if parent_path in self._directories:
                dir_info = self._directories[parent_path]
                space_quota = getattr(dir_info, 'space_quota', None)
                if space_quota is not None:
                    space_consumed = self._calculate_space_consumed(parent_path)
                    if space_consumed + size > space_quota:
                        raise HDFSError(
                            f"Quota exceeded: {parent_path} has space quota {space_quota}, "
                            f"consumed {space_consumed}, requested {size}"
                        )
            current_path = parent_path

    def check_and_remove_dead_nodes(self, timeout: float = 30.0) -> List[NodeID]:
        """Check for dead nodes and remove them. Returns list of removed node IDs."""
        current_time = time.time()
        dead_nodes = []

        for node_id, node in list(self._datanodes.items()):
            if current_time - node.last_heartbeat > timeout:
                dead_nodes.append(node_id)
                logger.warning(f"DataNode {node_id} is dead (last heartbeat: {node.last_heartbeat})")

        # Remove dead nodes
        for node_id in dead_nodes:
            self._remove_datanode(node_id)

        return dead_nodes

    def _remove_datanode(self, node_id: NodeID) -> None:
        """Remove a DataNode and schedule re-replication of its blocks."""
        if node_id not in self._datanodes:
            return

        node = self._datanodes[node_id]

        # Remove node from block mappings
        for block_id in list(self._block_to_nodes.keys()):
            self._block_to_nodes[block_id].discard(node_id)

        # Delete node
        del self._datanodes[node_id]
        logger.info(f"Removed DataNode: {node_id}")

        # Schedule re-replication for any now under-replicated blocks.
        self.scan_and_schedule_replication()

    def _replication_target_for_block(self, block_id: BlockID) -> int:
        """Return the desired replication factor for a block."""
        for file_info in self._files.values():
            if block_id in file_info.blocks:
                return file_info.replication
        return self.default_replication

    def scan_and_schedule_replication(self) -> int:
        """Scan for under-replicated blocks and schedule real re-replication.

        For each block below its replication factor, picks a live source
        DataNode that already holds a healthy replica and a live target that
        lacks it, then enqueues a pending replication delivered to the target
        on its next heartbeat (pull model). Returns the number of new tasks
        scheduled. Idempotent: never schedules the same (block, target) twice
        while a copy is in flight or already queued.
        """
        scheduled = 0

        for block_id in self._check_under_replicated_blocks():
            target_replication = self._replication_target_for_block(block_id)

            # Live nodes that hold a healthy replica -> possible sources.
            holders = {
                nid for nid in self._block_to_nodes.get(block_id, set())
                if nid in self._datanodes and self._datanodes[nid].is_alive
            }
            if not holders:
                # No live source: the block is lost, nothing we can copy.
                logger.warning(
                    f"Block {block_id} has no live replica; cannot re-replicate"
                )
                continue

            # Nodes already queued/in-flight as targets for this block.
            queued_targets = {
                t for (b, _s, t) in self._pending_replications if b == block_id
            } | self._replications_in_flight.get(block_id, set())

            # Effective replica count includes copies already on the way.
            effective = len(holders) + len(queued_targets)
            if effective >= target_replication:
                continue

            # Candidate targets: live, lacking the block, not already targeted.
            candidates = [
                nid for nid, node in self._datanodes.items()
                if node.is_alive
                and nid not in holders
                and nid not in queued_targets
            ]

            source = sorted(holders)[0]
            need = target_replication - effective
            for target in sorted(candidates)[:need]:
                self._pending_replications.append((block_id, source, target))
                self._replications_in_flight[block_id].add(target)
                scheduled += 1
                logger.info(
                    f"Scheduled re-replication of {block_id} from {source} "
                    f"to {target}"
                )

        return scheduled

    # Checkpointing

    def save_checkpoint(self, path: str) -> None:
        """Save namespace to checkpoint file."""
        checkpoint = {
            "files": {
                p: {
                    "path": f.path,
                    "size": f.size,
                    "block_size": f.block_size,
                    "replication": f.replication,
                    "blocks": f.blocks,
                    "modification_time": f.modification_time,
                    "owner": f.owner,
                    "group": f.group,
                    "permission": f.permission
                }
                for p, f in self._files.items()
            },
            "directories": {
                p: {
                    "path": d.path,
                    "children": list(d.children),
                    "modification_time": d.modification_time,
                    "owner": d.owner,
                    "group": d.group,
                    "permission": d.permission
                }
                for p, d in self._directories.items()
            },
            "blocks": {
                bid: {"block_id": b.block_id, "size": b.size, "generation_stamp": b.generation_stamp}
                for bid, b in self._blocks.items()
            },
            "datanodes": {
                nid: {
                    "node_id": n.node_id,
                    "host": n.host,
                    "port": n.port,
                    "rack": n.rack,
                    "capacity": n.capacity,
                    "used": n.used,
                    "remaining": n.remaining
                }
                for nid, n in self._datanodes.items()
            },
            "block_to_nodes": {
                bid: list(nodes) for bid, nodes in self._block_to_nodes.items()
            }
        }

        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load namespace from checkpoint file."""
        with open(path, 'r') as f:
            checkpoint = json.load(f)

        # Load files
        self._files = {}
        for p, data in checkpoint.get("files", {}).items():
            self._files[p] = FileInfo(
                path=data["path"],
                size=data["size"],
                block_size=data["block_size"],
                replication=data["replication"],
                blocks=data["blocks"],
                modification_time=data["modification_time"],
                owner=data["owner"],
                group=data["group"],
                permission=data["permission"]
            )

        # Load directories
        self._directories = {}
        for p, data in checkpoint.get("directories", {}).items():
            self._directories[p] = DirectoryInfo(
                path=data["path"],
                children=set(data["children"]),
                modification_time=data["modification_time"],
                owner=data["owner"],
                group=data["group"],
                permission=data["permission"]
            )

        # Load blocks
        self._blocks = {}
        for bid, data in checkpoint.get("blocks", {}).items():
            self._blocks[bid] = Block(
                block_id=data["block_id"],
                size=data["size"],
                generation_stamp=data["generation_stamp"]
            )

        # Load datanodes
        self._datanodes = {}
        for nid, data in checkpoint.get("datanodes", {}).items():
            self._datanodes[nid] = DataNodeInfo(
                node_id=data["node_id"],
                host=data["host"],
                port=data["port"],
                rack=data.get("rack", "/default-rack"),
                capacity=data["capacity"],
                used=data["used"],
                remaining=data["remaining"],
                last_heartbeat=time.time()
            )

        # Load block-to-nodes mapping (defaultdict so callers can index freely)
        self._block_to_nodes = defaultdict(set)
        for bid, nodes in checkpoint.get("block_to_nodes", {}).items():
            self._block_to_nodes[bid] = set(nodes)

        logger.info(f"Loaded checkpoint from {path}")

    # Durable-namespace helpers (checkpoint_interval made live)

    def configure_checkpointing(self, checkpoint_path: str, load: bool = True) -> None:
        """Point the NameNode at a checkpoint file and optionally load it.

        Loading recovers the namespace (files, directories, blocks and block
        locations) from a prior run so a restarted NameNode comes back with its
        state. A full edit-log/WAL is intentionally out of scope: only the
        periodic checkpoint is persisted, so writes since the last checkpoint
        are not durable.
        """
        self._checkpoint_path = checkpoint_path
        if load and os.path.exists(checkpoint_path):
            try:
                self.load_checkpoint(checkpoint_path)
            except (OSError, ValueError, KeyError) as e:
                logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")

    def _write_checkpoint_atomic(self, path: str) -> None:
        """Save the checkpoint atomically (write temp then rename)."""
        tmp_path = f"{path}.tmp"
        self.save_checkpoint(tmp_path)
        os.replace(tmp_path, path)

    async def checkpoint_loop(self) -> None:
        """Background task: persist the namespace every checkpoint_interval.

        This is what makes ``checkpoint_interval`` live. Runs off the asyncio
        event loop and writes atomically so a crash mid-write cannot corrupt an
        existing checkpoint.
        """
        while getattr(self, "_bg_running", False):
            await asyncio.sleep(self.checkpoint_interval)
            if not getattr(self, "_bg_running", False):
                break
            if not self._checkpoint_path:
                continue
            try:
                await asyncio.to_thread(
                    self._write_checkpoint_atomic, self._checkpoint_path
                )
                logger.info(
                    f"Periodic checkpoint written to {self._checkpoint_path}"
                )
            except (OSError, ValueError) as e:
                logger.error(f"Periodic checkpoint failed: {e}")

    async def replication_monitor_loop(self, interval: Optional[float] = None) -> None:
        """Background task: periodically self-heal under-replicated blocks.

        Drives ``scan_and_schedule_replication`` so that missed heartbeats /
        dead DataNodes result in real byte-copying re-replication (the actual
        copy is issued to the target DataNode on its heartbeat).
        """
        poll = interval if interval is not None else self.heartbeat_interval
        while getattr(self, "_bg_running", False):
            await asyncio.sleep(poll)
            if not getattr(self, "_bg_running", False):
                break
            try:
                self.check_and_remove_dead_nodes()
                self.scan_and_schedule_replication()
            except Exception:
                # Never let the self-healing loop die on an unexpected error.
                logger.exception("Replication monitor iteration failed")

    def start_background_tasks(self) -> List[asyncio.Task]:
        """Start the checkpoint and replication-monitor background tasks."""
        self._bg_running = True
        return [
            asyncio.create_task(self.checkpoint_loop()),
            asyncio.create_task(self.replication_monitor_loop()),
        ]

    def stop_background_tasks(self) -> None:
        """Signal background tasks to stop at their next wakeup."""
        self._bg_running = False


class NameNodeServer:
    """Async server for NameNode."""

    def __init__(
        self,
        namenode: NameNode,
        host: str = "0.0.0.0",
        port: int = 9000,
        checkpoint_path: Optional[str] = None
    ):
        self.namenode = namenode
        self.host = host
        self.port = port
        self.checkpoint_path = checkpoint_path
        self._server = None
        self._bg_tasks: List[asyncio.Task] = []

    async def start(self):
        """Start the NameNode server and its background tasks."""
        # Recover namespace from a prior checkpoint before serving.
        if self.checkpoint_path:
            self.namenode.configure_checkpointing(self.checkpoint_path, load=True)

        self._server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port
        )

        # Periodic checkpointing (makes checkpoint_interval live) and
        # self-healing re-replication.
        self._bg_tasks = self.namenode.start_background_tasks()

        logger.info(f"NameNode server started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the server and background tasks; write a final checkpoint."""
        self.namenode.stop_background_tasks()
        for task in self._bg_tasks:
            task.cancel()
        for task in self._bg_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._bg_tasks = []

        # Best-effort final checkpoint so a clean shutdown is durable.
        if self.checkpoint_path:
            try:
                self.namenode._write_checkpoint_atomic(self.checkpoint_path)
            except (OSError, ValueError) as e:
                logger.error(f"Final checkpoint failed: {e}")

        if self._server:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle client connection."""
        try:
            while True:
                # Read message length
                length_data = await reader.read(4)
                if not length_data:
                    break

                length = int.from_bytes(length_data, 'big')
                data = await reader.read(length)

                # Process message
                from ..common.protocol import deserialize_message, serialize_message
                message = deserialize_message(data)
                response = await self._process_message(message)

                # Send response
                response_data = serialize_message(response)
                writer.write(len(response_data).to_bytes(4, 'big'))
                writer.write(response_data)
                await writer.drain()

        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_message(self, message: Message) -> Message:
        """Process incoming message."""
        try:
            if message.msg_type == MessageType.CREATE_FILE:
                result = self.namenode.create_file(**message.payload)
                return Message(MessageType.SUCCESS, {"path": result.path})

            elif message.msg_type == MessageType.ADD_BLOCK:
                block, locations = self.namenode.add_block(message.payload["path"])
                return Message(MessageType.SUCCESS, {
                    "block_id": block.block_id,
                    "locations": [
                        {"node_id": loc.node_id, "host": loc.host, "port": loc.port}
                        for loc in locations
                    ]
                })

            elif message.msg_type == MessageType.COMPLETE_FILE:
                result = self.namenode.complete_file(**message.payload)
                return Message(MessageType.SUCCESS, {"size": result.size})

            elif message.msg_type == MessageType.GET_BLOCK_LOCATIONS:
                locations = self.namenode.get_block_locations_for_file(message.payload["path"])
                return Message(MessageType.SUCCESS, {
                    "locations": [
                        [{"block_id": loc.block_id, "host": loc.host, "port": loc.port}
                         for loc in block_locs]
                        for block_locs in locations
                    ]
                })

            elif message.msg_type == MessageType.LIST_DIR:
                result = self.namenode.list_directory(message.payload["path"])
                return Message(MessageType.SUCCESS, {"entries": result})

            elif message.msg_type == MessageType.GET_FILE_INFO:
                result = self.namenode.get_file_info_dict(message.payload["path"])
                if result is None:
                    return Message(MessageType.ERROR, {"error": "Path not found"})
                return Message(MessageType.SUCCESS, {"info": result})

            elif message.msg_type == MessageType.MKDIR:
                self.namenode.mkdir(**message.payload)
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.DELETE_FILE:
                self.namenode.delete_file(message.payload["path"])
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.REGISTER_DATANODE:
                self.namenode.register_datanode(**message.payload)
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.HEARTBEAT:
                response = self.namenode.heartbeat(**message.payload)
                return Message(MessageType.SUCCESS, {"commands": response.commands})

            elif message.msg_type == MessageType.BLOCK_REPORT:
                report = BlockReport(**message.payload)
                self.namenode.block_report(report)
                return Message(MessageType.SUCCESS, {})

            elif message.msg_type == MessageType.BLOCK_RECEIVED:
                self.namenode.block_received(**message.payload)
                return Message(MessageType.SUCCESS, {})

            else:
                return Message(MessageType.ERROR, {"error": f"Unknown message type: {message.msg_type}"})

        except HDFSError as e:
            return Message(MessageType.ERROR, {"error": str(e)})
        except Exception as e:
            logger.exception("Error processing message")
            return Message(MessageType.ERROR, {"error": str(e)})
