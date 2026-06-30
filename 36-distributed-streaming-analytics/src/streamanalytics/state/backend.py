"""State backends for fault-tolerant stream processing."""

import os
import json
import time
import pickle
import logging
import threading
from typing import Any, Callable, List, Iterator, TypeVar, Generic, Dict, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')
ACC = TypeVar('ACC')


@dataclass
class StateDescriptor(Generic[T]):
    """Descriptor for state registration."""
    name: str
    default_value: T = None


class KeyedState(ABC, Generic[K, T]):
    """Base class for keyed state."""

    def __init__(self, name: str):
        self.name = name
        self._current_key: Optional[K] = None

    def set_current_key(self, key: K):
        """Set the current key context."""
        self._current_key = key

    @abstractmethod
    def clear(self):
        """Clear state for current key."""
        pass


class ValueState(KeyedState[K, T]):
    """Single value state per key."""

    def __init__(self, name: str, default: T = None):
        super().__init__(name)
        self._default = default
        self._values: Dict[K, T] = {}

    def value(self) -> Optional[T]:
        """Get current value."""
        if self._current_key is None:
            return self._default
        return self._values.get(self._current_key, self._default)

    def update(self, value: T):
        """Update current value."""
        if self._current_key is not None:
            self._values[self._current_key] = value

    def clear(self):
        """Clear state for current key."""
        if self._current_key is not None and self._current_key in self._values:
            del self._values[self._current_key]

    def get_all(self) -> Dict[K, T]:
        """Get all key-value pairs."""
        return self._values.copy()


class ListState(KeyedState[K, T]):
    """List of values per key."""

    def __init__(self, name: str):
        super().__init__(name)
        self._lists: Dict[K, List[T]] = {}

    def get(self) -> List[T]:
        """Get current list."""
        if self._current_key is None:
            return []
        return self._lists.get(self._current_key, [])

    def add(self, value: T):
        """Add value to list."""
        if self._current_key is not None:
            if self._current_key not in self._lists:
                self._lists[self._current_key] = []
            self._lists[self._current_key].append(value)

    def add_all(self, values: List[T]):
        """Add multiple values."""
        for v in values:
            self.add(v)

    def update(self, values: List[T]):
        """Replace list with new values."""
        if self._current_key is not None:
            self._lists[self._current_key] = list(values)

    def clear(self):
        """Clear list for current key."""
        if self._current_key is not None and self._current_key in self._lists:
            del self._lists[self._current_key]

    def get_all(self) -> Dict[K, List[T]]:
        """Get all lists."""
        return {k: list(v) for k, v in self._lists.items()}


class MapState(KeyedState[K, Dict[str, V]], Generic[K, V]):
    """Map state per key."""

    def __init__(self, name: str):
        super().__init__(name)
        self._maps: Dict[K, Dict[str, V]] = {}

    def get(self, map_key: str) -> Optional[V]:
        """Get value by map key."""
        if self._current_key is None:
            return None
        return self._maps.get(self._current_key, {}).get(map_key)

    def put(self, map_key: str, value: V):
        """Put value for map key."""
        if self._current_key is not None:
            if self._current_key not in self._maps:
                self._maps[self._current_key] = {}
            self._maps[self._current_key][map_key] = value

    def put_all(self, mapping: Dict[str, V]):
        """Put multiple values."""
        for k, v in mapping.items():
            self.put(k, v)

    def remove(self, map_key: str):
        """Remove value by map key."""
        if self._current_key is not None and self._current_key in self._maps:
            self._maps[self._current_key].pop(map_key, None)

    def contains(self, map_key: str) -> bool:
        """Check if map contains key."""
        if self._current_key is None:
            return False
        return map_key in self._maps.get(self._current_key, {})

    def keys(self) -> List[str]:
        """Get all map keys."""
        if self._current_key is None:
            return []
        return list(self._maps.get(self._current_key, {}).keys())

    def values(self) -> List[V]:
        """Get all map values."""
        if self._current_key is None:
            return []
        return list(self._maps.get(self._current_key, {}).values())

    def entries(self) -> Dict[str, V]:
        """Get all entries."""
        if self._current_key is None:
            return {}
        return self._maps.get(self._current_key, {}).copy()

    def clear(self):
        """Clear map for current key."""
        if self._current_key is not None and self._current_key in self._maps:
            del self._maps[self._current_key]

    def get_all(self) -> Dict[K, Dict[str, V]]:
        """Get all maps."""
        return {k: dict(v) for k, v in self._maps.items()}


class ReducingState(KeyedState[K, T]):
    """Reducing state that applies reduce function."""

    def __init__(self, name: str, reduce_func: Callable[[T, T], T]):
        super().__init__(name)
        self.reduce_func = reduce_func
        self._values: Dict[K, T] = {}

    def get(self) -> Optional[T]:
        """Get current reduced value."""
        if self._current_key is None:
            return None
        return self._values.get(self._current_key)

    def add(self, value: T):
        """Add value and reduce."""
        if self._current_key is not None:
            if self._current_key in self._values:
                self._values[self._current_key] = self.reduce_func(
                    self._values[self._current_key], value
                )
            else:
                self._values[self._current_key] = value

    def clear(self):
        """Clear state for current key."""
        if self._current_key is not None and self._current_key in self._values:
            del self._values[self._current_key]


class AggregatingState(KeyedState[K, ACC], Generic[K, T, ACC, V]):
    """Aggregating state with custom accumulator."""

    def __init__(
        self,
        name: str,
        create_accumulator: Callable[[], ACC],
        add: Callable[[ACC, T], ACC],
        get_result: Callable[[ACC], V]
    ):
        super().__init__(name)
        self.create_accumulator = create_accumulator
        self.add_func = add
        self.get_result = get_result
        self._accumulators: Dict[K, ACC] = {}

    def get(self) -> Optional[V]:
        """Get current aggregated value."""
        if self._current_key is None or self._current_key not in self._accumulators:
            return None
        return self.get_result(self._accumulators[self._current_key])

    def add(self, value: T):
        """Add value to accumulator."""
        if self._current_key is not None:
            if self._current_key not in self._accumulators:
                self._accumulators[self._current_key] = self.create_accumulator()
            self._accumulators[self._current_key] = self.add_func(
                self._accumulators[self._current_key], value
            )

    def clear(self):
        """Clear state for current key."""
        if self._current_key is not None and self._current_key in self._accumulators:
            del self._accumulators[self._current_key]


@dataclass
class Checkpoint:
    """Checkpoint containing state snapshot."""
    checkpoint_id: int
    timestamp: float
    state_handles: Dict[str, bytes] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> bytes:
        """Serialize checkpoint."""
        return pickle.dumps({
            'checkpoint_id': self.checkpoint_id,
            'timestamp': self.timestamp,
            'state_handles': self.state_handles,
            'metadata': self.metadata
        })

    @staticmethod
    def deserialize(data: bytes) -> 'Checkpoint':
        """Deserialize checkpoint."""
        obj = pickle.loads(data)
        return Checkpoint(
            checkpoint_id=obj['checkpoint_id'],
            timestamp=obj['timestamp'],
            state_handles=obj['state_handles'],
            metadata=obj['metadata']
        )


class CheckpointStorage(ABC):
    """Storage for checkpoints."""

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> str:
        """Save checkpoint and return handle."""
        pass

    @abstractmethod
    def load(self, handle: str) -> Checkpoint:
        """Load checkpoint by handle."""
        pass

    @abstractmethod
    def delete(self, handle: str):
        """Delete checkpoint."""
        pass


class FileCheckpointStorage(CheckpointStorage):
    """File-based checkpoint storage."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: Checkpoint) -> str:
        """Save checkpoint to file."""
        filename = f"checkpoint_{checkpoint.checkpoint_id}.bin"
        path = self.base_path / filename
        with open(path, 'wb') as f:
            f.write(checkpoint.serialize())
        return str(path)

    def load(self, handle: str) -> Checkpoint:
        """Load checkpoint from file."""
        with open(handle, 'rb') as f:
            return Checkpoint.deserialize(f.read())

    def delete(self, handle: str):
        """Delete checkpoint file."""
        path = Path(handle)
        if path.exists():
            path.unlink()


class StateBackend(ABC):
    """Backend for managing state."""

    @abstractmethod
    def create_value_state(self, descriptor: StateDescriptor[T]) -> ValueState[Any, T]:
        """Create value state."""
        pass

    @abstractmethod
    def create_list_state(self, descriptor: StateDescriptor[List[T]]) -> ListState[Any, T]:
        """Create list state."""
        pass

    @abstractmethod
    def create_map_state(self, descriptor: StateDescriptor[Dict]) -> MapState[Any, Any]:
        """Create map state."""
        pass

    @abstractmethod
    def checkpoint(self, checkpoint_id: int) -> Checkpoint:
        """Create checkpoint of all state."""
        pass

    @abstractmethod
    def restore(self, checkpoint: Checkpoint):
        """Restore state from checkpoint."""
        pass


class MemoryStateBackend(StateBackend):
    """In-memory state backend."""

    def __init__(self):
        self._states: Dict[str, KeyedState] = {}
        self._lock = threading.Lock()

    def create_value_state(self, descriptor: StateDescriptor[T]) -> ValueState[Any, T]:
        """Create value state."""
        with self._lock:
            if descriptor.name not in self._states:
                self._states[descriptor.name] = ValueState(
                    descriptor.name, descriptor.default_value
                )
            return self._states[descriptor.name]

    def create_list_state(self, descriptor: StateDescriptor[List[T]]) -> ListState[Any, T]:
        """Create list state."""
        with self._lock:
            if descriptor.name not in self._states:
                self._states[descriptor.name] = ListState(descriptor.name)
            return self._states[descriptor.name]

    def create_map_state(self, descriptor: StateDescriptor[Dict]) -> MapState[Any, Any]:
        """Create map state."""
        with self._lock:
            if descriptor.name not in self._states:
                self._states[descriptor.name] = MapState(descriptor.name)
            return self._states[descriptor.name]

    def create_reducing_state(
        self,
        descriptor: StateDescriptor[T],
        reduce_func: Callable[[T, T], T]
    ) -> ReducingState[Any, T]:
        """Create reducing state."""
        with self._lock:
            if descriptor.name not in self._states:
                self._states[descriptor.name] = ReducingState(
                    descriptor.name, reduce_func
                )
            return self._states[descriptor.name]

    def checkpoint(self, checkpoint_id: int) -> Checkpoint:
        """Create checkpoint of all state."""
        with self._lock:
            state_handles = {}
            for name, state in self._states.items():
                if isinstance(state, ValueState):
                    state_handles[name] = pickle.dumps(state.get_all())
                elif isinstance(state, ListState):
                    state_handles[name] = pickle.dumps(state.get_all())
                elif isinstance(state, MapState):
                    state_handles[name] = pickle.dumps(state.get_all())
                elif isinstance(state, ReducingState):
                    state_handles[name] = pickle.dumps(state._values)

            return Checkpoint(
                checkpoint_id=checkpoint_id,
                timestamp=time.time(),
                state_handles=state_handles
            )

    def restore(self, checkpoint: Checkpoint):
        """Restore state from checkpoint."""
        with self._lock:
            for name, data in checkpoint.state_handles.items():
                if name in self._states:
                    state = self._states[name]
                    restored = pickle.loads(data)

                    if isinstance(state, ValueState):
                        state._values = restored
                    elif isinstance(state, ListState):
                        state._lists = restored
                    elif isinstance(state, MapState):
                        state._maps = restored
                    elif isinstance(state, ReducingState):
                        state._values = restored


class RocksDBStateBackend(StateBackend):
    """RocksDB-based state backend for large state.

    Note: This is a simplified implementation that stores
    state in files. A real implementation would use RocksDB.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._memory_backend = MemoryStateBackend()
        self._write_buffer: Dict[str, Any] = {}
        self._buffer_size = 0
        self._max_buffer_size = 1000

    def create_value_state(self, descriptor: StateDescriptor[T]) -> ValueState[Any, T]:
        """Create value state backed by RocksDB."""
        # Load existing state from disk
        state_file = self.db_path / f"{descriptor.name}.state"
        state = self._memory_backend.create_value_state(descriptor)

        if state_file.exists():
            with open(state_file, 'rb') as f:
                state._values = pickle.load(f)

        return state

    def create_list_state(self, descriptor: StateDescriptor[List[T]]) -> ListState[Any, T]:
        """Create list state backed by RocksDB."""
        state_file = self.db_path / f"{descriptor.name}.state"
        state = self._memory_backend.create_list_state(descriptor)

        if state_file.exists():
            with open(state_file, 'rb') as f:
                state._lists = pickle.load(f)

        return state

    def create_map_state(self, descriptor: StateDescriptor[Dict]) -> MapState[Any, Any]:
        """Create map state backed by RocksDB."""
        state_file = self.db_path / f"{descriptor.name}.state"
        state = self._memory_backend.create_map_state(descriptor)

        if state_file.exists():
            with open(state_file, 'rb') as f:
                state._maps = pickle.load(f)

        return state

    def flush(self):
        """Flush state to disk."""
        for name, state in self._memory_backend._states.items():
            state_file = self.db_path / f"{name}.state"

            if isinstance(state, ValueState):
                data = state._values
            elif isinstance(state, ListState):
                data = state._lists
            elif isinstance(state, MapState):
                data = state._maps
            else:
                continue

            with open(state_file, 'wb') as f:
                pickle.dump(data, f)

    def checkpoint(self, checkpoint_id: int) -> Checkpoint:
        """Create checkpoint."""
        self.flush()
        return self._memory_backend.checkpoint(checkpoint_id)

    def restore(self, checkpoint: Checkpoint):
        """Restore from checkpoint."""
        self._memory_backend.restore(checkpoint)
        self.flush()

    def compact(self):
        """Compact RocksDB (placeholder)."""
        logger.info("Compacting RocksDB state backend")
        self.flush()


class CheckpointCoordinator:
    """Coordinates checkpoints across operators."""

    def __init__(
        self,
        storage: CheckpointStorage,
        interval_ms: int = 60000,
        min_pause_ms: int = 0,
        timeout_ms: int = 600000
    ):
        self.storage = storage
        self.interval_ms = interval_ms
        self.min_pause_ms = min_pause_ms
        self.timeout_ms = timeout_ms
        self._checkpoint_id = 0
        self._last_checkpoint_time = 0.0
        self._pending_checkpoints: Dict[int, Checkpoint] = {}
        self._completed_checkpoints: List[str] = []
        self._max_retained = 3

    def trigger_checkpoint(self, state_backends: List[StateBackend]) -> Optional[Checkpoint]:
        """Trigger a new checkpoint."""
        current_time = time.time() * 1000
        if current_time - self._last_checkpoint_time < self.min_pause_ms:
            return None

        self._checkpoint_id += 1
        logger.info(f"Triggering checkpoint {self._checkpoint_id}")

        # Collect checkpoints from all backends
        state_handles = {}
        for i, backend in enumerate(state_backends):
            cp = backend.checkpoint(self._checkpoint_id)
            for name, handle in cp.state_handles.items():
                state_handles[f"{i}_{name}"] = handle

        checkpoint = Checkpoint(
            checkpoint_id=self._checkpoint_id,
            timestamp=time.time(),
            state_handles=state_handles
        )

        # Save checkpoint
        handle = self.storage.save(checkpoint)
        self._completed_checkpoints.append(handle)
        self._last_checkpoint_time = current_time

        # Cleanup old checkpoints
        while len(self._completed_checkpoints) > self._max_retained:
            old_handle = self._completed_checkpoints.pop(0)
            self.storage.delete(old_handle)

        logger.info(f"Completed checkpoint {self._checkpoint_id}")
        return checkpoint

    def restore_latest(self, state_backends: List[StateBackend]) -> bool:
        """Restore from latest checkpoint."""
        if not self._completed_checkpoints:
            return False

        latest_handle = self._completed_checkpoints[-1]
        checkpoint = self.storage.load(latest_handle)

        # Distribute state to backends
        for i, backend in enumerate(state_backends):
            backend_handles = {
                name.split('_', 1)[1]: handle
                for name, handle in checkpoint.state_handles.items()
                if name.startswith(f"{i}_")
            }

            backend_checkpoint = Checkpoint(
                checkpoint_id=checkpoint.checkpoint_id,
                timestamp=checkpoint.timestamp,
                state_handles=backend_handles
            )
            backend.restore(backend_checkpoint)

        logger.info(f"Restored from checkpoint {checkpoint.checkpoint_id}")
        return True
