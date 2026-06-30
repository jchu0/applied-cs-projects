"""Delta Lake transaction log implementation."""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


@dataclass
class Action:
    """Base class for Delta Lake actions."""
    pass


@dataclass
class AddFile(Action):
    """Action to add a file to the table."""

    path: str
    partition_values: Dict[str, str]
    size: int
    modification_time: int
    data_change: bool
    stats: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class RemoveFile(Action):
    """Action to remove a file from the table."""

    path: str
    deletion_timestamp: int
    data_change: bool
    extended_file_metadata: bool = False
    partition_values: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metadata(Action):
    """Action to set table metadata."""

    id: str
    name: str
    description: str
    schema_string: str
    partition_columns: List[str]
    configuration: Dict[str, str] = field(default_factory=dict)
    format_provider: str = "parquet"


@dataclass
class Protocol(Action):
    """Action to set table protocol."""

    min_reader_version: int
    min_writer_version: int
    reader_features: List[str] = field(default_factory=list)
    writer_features: List[str] = field(default_factory=list)


@dataclass
class CommitInfo(Action):
    """Action to record commit metadata."""

    timestamp: int
    user_id: str = ""
    operation: str = ""
    operation_parameters: Dict[str, str] = field(default_factory=dict)
    job_name: str = ""
    notebook_id: str = ""
    is_blind_append: bool = True


@dataclass
class SetTransaction(Action):
    """Action to set an application transaction identifier."""

    app_id: str
    version: int
    last_updated: Optional[int] = None


@dataclass
class TableState:
    """Current state of a Delta table."""

    files: List[AddFile] = field(default_factory=list)
    metadata: Optional[Metadata] = None
    protocol: Optional[Protocol] = None

    def apply_actions(self, actions: List[Action]) -> "TableState":
        """Apply a list of actions to get new state."""
        new_state = TableState(
            files=self.files.copy(),
            metadata=self.metadata,
            protocol=self.protocol,
        )

        for action in actions:
            if isinstance(action, AddFile):
                new_state.files.append(action)
            elif isinstance(action, RemoveFile):
                new_state.files = [
                    f for f in new_state.files if f.path != action.path
                ]
            elif isinstance(action, Metadata):
                new_state.metadata = action
            elif isinstance(action, Protocol):
                new_state.protocol = action

        return new_state


@dataclass
class Snapshot:
    """Snapshot of table state at a specific version."""

    version: int
    state: TableState


class DeltaLog:
    """
    Delta Lake transaction log manager.

    The transaction log (_delta_log/) provides ACID guarantees.
    Each transaction creates a new JSON file (00000000000000000000.json).
    """

    def __init__(self, table_path: str):
        self.table_path = str(table_path)
        self.log_path = Path(table_path) / "_delta_log"
        self._snapshot_cache: Optional[Snapshot] = None
        self._current_version: int = -1
        self._snapshot: List[Action] = []
        self._metadata: Optional[Metadata] = None
        self._protocol: Optional[Protocol] = None

    @property
    def current_version(self) -> int:
        """Get the current version of the table."""
        return self._current_version

    @property
    def snapshot(self) -> List[Action]:
        """Get the current snapshot of actions."""
        return self._snapshot

    def get_snapshot(self, version: Optional[int] = None) -> Snapshot:
        """
        Read the current state of the table.

        Uses checkpoints (parquet) for efficiency with log replay.

        Args:
            version: Optional specific version to read

        Returns:
            Snapshot of table state
        """
        # Find latest checkpoint
        checkpoint_version = self._find_latest_checkpoint()

        if checkpoint_version is not None:
            # Read checkpoint
            state = self._read_checkpoint(checkpoint_version)
            start_version = checkpoint_version + 1
        else:
            state = TableState()
            start_version = 0

        # Replay logs since checkpoint
        target = version if version is not None else self._get_latest_version()
        for v in range(start_version, target + 1):
            actions = self._read_log_file(v)
            state = state.apply_actions(actions)

        return Snapshot(version=target, state=state)

    def commit(self, actions: List[Action], max_retries: int = 3) -> int:
        """
        Atomically commit a set of actions.

        Uses optimistic concurrency control.

        Args:
            actions: List of actions to commit
            max_retries: Maximum number of retry attempts

        Returns:
            Version number of the commit

        Raises:
            Exception: If commit fails after max retries
        """
        # Retry loop for concurrent commits
        for attempt in range(max_retries):
            try:
                version = self._get_latest_version() + 1
                log_file = self.log_path / f"{version:020d}.json"

                # Atomic write (put-if-absent semantics)
                self._atomic_write(log_file, actions)

                # Update internal state
                self._current_version = version
                self._snapshot.extend(actions)

                # Update metadata/protocol if present
                for action in actions:
                    if isinstance(action, Metadata):
                        self._metadata = action
                    elif isinstance(action, Protocol):
                        self._protocol = action

                return version

            except FileExistsError:
                # Conflict detected, retry with new version
                continue

        raise Exception("Max retries exceeded for commit")

    def read_version(self, version: int) -> List[Dict[str, Any]]:
        """
        Read actions from a specific version.

        Args:
            version: The version number to read

        Returns:
            List of action dictionaries
        """
        log_file = self.log_path / f"{version:020d}.json"
        if not log_file.exists():
            return []

        actions = []
        with open(log_file, "r") as f:
            for line in f:
                actions.append(json.loads(line))
        return actions

    def create_checkpoint(self) -> None:
        """
        Create a checkpoint at the current version.

        Checkpoints are parquet files that contain the full table state,
        allowing faster reads by avoiding log replay.
        """
        if not HAS_PYARROW:
            # Create a simple JSON checkpoint as fallback
            self._create_json_checkpoint()
            return

        checkpoint_file = self.log_path / f"{self._current_version:020d}.checkpoint.parquet"

        # Build checkpoint data from active files
        active_files = self.get_active_files()

        # Create checkpoint table
        data = {
            "path": [f.path for f in active_files],
            "partition_values": [json.dumps(f.partition_values) for f in active_files],
            "size": [f.size for f in active_files],
            "modification_time": [f.modification_time for f in active_files],
            "data_change": [f.data_change for f in active_files],
            "stats": [f.stats or "" for f in active_files],
        }

        table = pa.table(data)
        pq.write_table(table, checkpoint_file)

    def _create_json_checkpoint(self) -> None:
        """Create a JSON checkpoint as fallback when pyarrow is unavailable."""
        checkpoint_file = self.log_path / f"{self._current_version:020d}.checkpoint.json"
        active_files = self.get_active_files()

        data = []
        for f in active_files:
            data.append({
                "add": {
                    "path": f.path,
                    "partitionValues": f.partition_values,
                    "size": f.size,
                    "modificationTime": f.modification_time,
                    "dataChange": f.data_change,
                    "stats": f.stats,
                }
            })

        # Also save metadata and protocol
        if self._metadata:
            data.append(self._action_to_json(self._metadata))
        if self._protocol:
            data.append(self._action_to_json(self._protocol))

        with open(checkpoint_file, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

        # Create .parquet placeholder for checkpoint detection
        parquet_placeholder = self.log_path / f"{self._current_version:020d}.checkpoint.parquet"
        parquet_placeholder.touch()

    def time_travel(self, version: int) -> List[Action]:
        """
        Get the table state at a specific version.

        Args:
            version: The version to travel to

        Returns:
            List of actions representing the state at that version
        """
        state = TableState()

        for v in range(version + 1):
            actions = self._read_log_file(v)
            state = state.apply_actions(actions)

        return state.files

    def get_active_files(self) -> List[AddFile]:
        """
        Get list of currently active files in the table.

        Returns:
            List of AddFile actions for active files
        """
        # Build current state from snapshot
        active: Dict[str, AddFile] = {}
        removed: Set[str] = set()

        for action in self._snapshot:
            if isinstance(action, AddFile):
                if action.path not in removed:
                    active[action.path] = action
            elif isinstance(action, RemoveFile):
                removed.add(action.path)
                if action.path in active:
                    del active[action.path]

        return list(active.values())

    def get_table_properties(self) -> Dict[str, Any]:
        """
        Get table properties from metadata.

        Returns:
            Dictionary of table properties
        """
        if self._metadata is None:
            # Try to find metadata in snapshot
            for action in self._snapshot:
                if isinstance(action, Metadata):
                    self._metadata = action
                    break

        if self._metadata is None:
            return {}

        return {
            "id": self._metadata.id,
            "name": self._metadata.name,
            "description": self._metadata.description,
            "schema_string": self._metadata.schema_string,
            "partition_columns": self._metadata.partition_columns,
            "configuration": self._metadata.configuration,
            "format_provider": self._metadata.format_provider,
        }

    def vacuum(self, retention_hours: int = 168) -> List[str]:
        """
        Remove old deleted files that are outside retention period.

        Args:
            retention_hours: Hours to retain deleted files (default 7 days)

        Returns:
            List of removed file paths
        """
        retention_ms = retention_hours * 3600 * 1000
        current_time_ms = int(time.time() * 1000)
        cutoff_time = current_time_ms - retention_ms

        # Find files to remove
        removed_paths = []
        for action in self._snapshot:
            if isinstance(action, RemoveFile):
                if action.deletion_timestamp < cutoff_time or retention_hours == 0:
                    removed_paths.append(action.path)

        return removed_paths

    def get_partitions(self) -> Dict[str, List[AddFile]]:
        """
        Get files grouped by partition.

        Returns:
            Dictionary mapping partition key to list of files
        """
        partitions: Dict[str, List[AddFile]] = {}

        for f in self.get_active_files():
            partition_key = json.dumps(f.partition_values, sort_keys=True)
            if partition_key not in partitions:
                partitions[partition_key] = []
            partitions[partition_key].append(f)

        return partitions

    def optimize_z_order(self, columns: List[str]) -> List[AddFile]:
        """
        Simulate Z-order optimization for specified columns.

        Args:
            columns: Columns to optimize for

        Returns:
            List of optimized files (simulated)
        """
        # This is a simulation - real Z-order would rewrite files
        active_files = self.get_active_files()

        # Return files that would be optimized (all active files)
        return active_files

    def collect_stats(self) -> Dict[str, Any]:
        """
        Collect statistics about the table.

        Returns:
            Dictionary of statistics
        """
        active_files = self.get_active_files()
        partitions = self.get_partitions()

        total_size = sum(f.size for f in active_files)
        total_records = 0

        for f in active_files:
            if f.stats:
                try:
                    stats = json.loads(f.stats)
                    total_records += stats.get("numRecords", 0)
                except (json.JSONDecodeError, TypeError):
                    pass

        return {
            "total_files": len(active_files),
            "total_size": total_size,
            "total_records": total_records,
            "partitions": list(partitions.keys()),
            "version": self._current_version,
        }

    def _get_latest_version(self) -> int:
        """Get the latest version number in the log."""
        if not self.log_path.exists():
            return -1

        versions = []
        for f in self.log_path.glob("*.json"):
            try:
                version = int(f.stem)
                versions.append(version)
            except ValueError:
                continue

        return max(versions) if versions else -1

    def _find_latest_checkpoint(self) -> Optional[int]:
        """Find the latest checkpoint version."""
        if not self.log_path.exists():
            return None

        checkpoints = []
        for f in self.log_path.glob("*.checkpoint.parquet"):
            try:
                version = int(f.stem.split(".")[0])
                checkpoints.append(version)
            except (ValueError, IndexError):
                continue

        return max(checkpoints) if checkpoints else None

    def _read_checkpoint(self, version: int) -> TableState:
        """
        Read table state from a checkpoint file.

        Args:
            version: The checkpoint version to read

        Returns:
            TableState reconstructed from checkpoint
        """
        state = TableState()

        # Try parquet checkpoint first
        checkpoint_file = self.log_path / f"{version:020d}.checkpoint.parquet"
        json_checkpoint = self.log_path / f"{version:020d}.checkpoint.json"

        if HAS_PYARROW and checkpoint_file.exists():
            try:
                table = pq.read_table(checkpoint_file)
                df = table.to_pandas()

                for _, row in df.iterrows():
                    partition_values = json.loads(row["partition_values"]) if row["partition_values"] else {}
                    add_file = AddFile(
                        path=row["path"],
                        partition_values=partition_values,
                        size=int(row["size"]),
                        modification_time=int(row["modification_time"]),
                        data_change=bool(row["data_change"]),
                        stats=row["stats"] if row["stats"] else None,
                    )
                    state.files.append(add_file)

                return state
            except Exception:
                pass  # Fall through to JSON checkpoint

        # Try JSON checkpoint as fallback
        if json_checkpoint.exists():
            try:
                with open(json_checkpoint, "r") as f:
                    for line in f:
                        action_data = json.loads(line)
                        action = self._parse_action(action_data)
                        if action:
                            if isinstance(action, AddFile):
                                state.files.append(action)
                            elif isinstance(action, Metadata):
                                state.metadata = action
                            elif isinstance(action, Protocol):
                                state.protocol = action
                return state
            except Exception:
                pass

        return state

    def _read_log_file(self, version: int) -> List[Action]:
        """Read actions from a log file."""
        log_file = self.log_path / f"{version:020d}.json"

        if not log_file.exists():
            return []

        actions = []
        with open(log_file, "r") as f:
            for line in f:
                action_data = json.loads(line)
                action = self._parse_action(action_data)
                if action:
                    actions.append(action)

        return actions

    def _parse_action(self, data: Dict[str, Any]) -> Optional[Action]:
        """Parse an action from JSON data."""
        if "add" in data:
            add = data["add"]
            return AddFile(
                path=add["path"],
                partition_values=add.get("partitionValues", {}),
                size=add["size"],
                modification_time=add["modificationTime"],
                data_change=add.get("dataChange", True),
                stats=add.get("stats"),
                tags=add.get("tags"),
            )
        elif "remove" in data:
            remove = data["remove"]
            return RemoveFile(
                path=remove["path"],
                deletion_timestamp=remove.get("deletionTimestamp", 0),
                data_change=remove.get("dataChange", True),
                partition_values=remove.get("partitionValues", {}),
            )
        elif "metaData" in data:
            meta = data["metaData"]
            return Metadata(
                id=meta["id"],
                name=meta.get("name", ""),
                description=meta.get("description", ""),
                schema_string=meta["schemaString"],
                partition_columns=meta.get("partitionColumns", []),
                configuration=meta.get("configuration", {}),
            )
        elif "protocol" in data:
            proto = data["protocol"]
            return Protocol(
                min_reader_version=proto["minReaderVersion"],
                min_writer_version=proto["minWriterVersion"],
            )
        return None

    def _atomic_write(self, path: Path, actions: List[Action]) -> None:
        """Atomically write actions to a log file."""
        # Ensure log directory exists
        self.log_path.mkdir(parents=True, exist_ok=True)

        # Use exclusive creation to ensure atomicity
        with open(path, "x") as f:
            for action in actions:
                action_json = self._action_to_json(action)
                f.write(json.dumps(action_json) + "\n")

    def _action_to_json(self, action: Action) -> Dict[str, Any]:
        """Convert an action to JSON format."""
        if isinstance(action, AddFile):
            return {
                "add": {
                    "path": action.path,
                    "partitionValues": action.partition_values,
                    "size": action.size,
                    "modificationTime": action.modification_time,
                    "dataChange": action.data_change,
                    "stats": action.stats,
                    "tags": action.tags,
                }
            }
        elif isinstance(action, RemoveFile):
            return {
                "remove": {
                    "path": action.path,
                    "deletionTimestamp": action.deletion_timestamp,
                    "dataChange": action.data_change,
                    "partitionValues": action.partition_values,
                }
            }
        elif isinstance(action, Metadata):
            return {
                "metaData": {
                    "id": action.id,
                    "name": action.name,
                    "description": action.description,
                    "schemaString": action.schema_string,
                    "partitionColumns": action.partition_columns,
                    "configuration": action.configuration,
                }
            }
        elif isinstance(action, Protocol):
            return {
                "protocol": {
                    "minReaderVersion": action.min_reader_version,
                    "minWriterVersion": action.min_writer_version,
                }
            }
        return {}
