"""State management for streaming applications."""

import json
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class StateBackend(ABC):
    """Abstract base class for state backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[bytes]:
        """Get state value."""
        pass

    @abstractmethod
    def put(self, key: str, value: bytes) -> None:
        """Put state value."""
        pass

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete state value."""
        pass

    @abstractmethod
    def keys(self) -> List[str]:
        """Get all keys."""
        pass


class InMemoryStateBackend(StateBackend):
    """In-memory state backend for development/testing."""

    def __init__(self):
        self._state: Dict[str, bytes] = {}

    def get(self, key: str) -> Optional[bytes]:
        return self._state.get(key)

    def put(self, key: str, value: bytes) -> None:
        self._state[key] = value

    def delete(self, key: str) -> None:
        self._state.pop(key, None)

    def keys(self) -> List[str]:
        return list(self._state.keys())

    def clear(self) -> None:
        """Clear all state."""
        self._state.clear()


class ValueState(Generic[V]):
    """State for a single value per key."""

    def __init__(
        self,
        name: str,
        backend: StateBackend,
        key: str,
        serializer: Callable[[V], bytes] = pickle.dumps,
        deserializer: Callable[[bytes], V] = pickle.loads,
    ):
        self._name = name
        self._backend = backend
        self._key = f"{name}:{key}"
        self._serializer = serializer
        self._deserializer = deserializer

    def value(self) -> Optional[V]:
        """Get current value."""
        data = self._backend.get(self._key)
        if data is None:
            return None
        return self._deserializer(data)

    def update(self, value: V) -> None:
        """Update value."""
        data = self._serializer(value)
        self._backend.put(self._key, data)

    def clear(self) -> None:
        """Clear value."""
        self._backend.delete(self._key)


class ListState(Generic[V]):
    """State for a list of values per key."""

    def __init__(
        self,
        name: str,
        backend: StateBackend,
        key: str,
        serializer: Callable[[List[V]], bytes] = pickle.dumps,
        deserializer: Callable[[bytes], List[V]] = pickle.loads,
    ):
        self._name = name
        self._backend = backend
        self._key = f"{name}:{key}"
        self._serializer = serializer
        self._deserializer = deserializer

    def get(self) -> List[V]:
        """Get current list."""
        data = self._backend.get(self._key)
        if data is None:
            return []
        return self._deserializer(data)

    def add(self, value: V) -> None:
        """Add value to list."""
        current = self.get()
        current.append(value)
        self._backend.put(self._key, self._serializer(current))

    def add_all(self, values: List[V]) -> None:
        """Add multiple values to list."""
        current = self.get()
        current.extend(values)
        self._backend.put(self._key, self._serializer(current))

    def update(self, values: List[V]) -> None:
        """Replace entire list."""
        self._backend.put(self._key, self._serializer(values))

    def clear(self) -> None:
        """Clear list."""
        self._backend.delete(self._key)


class MapState(Generic[K, V]):
    """State for a map of key-value pairs."""

    def __init__(
        self,
        name: str,
        backend: StateBackend,
        key: str,
        serializer: Callable[[Dict[K, V]], bytes] = pickle.dumps,
        deserializer: Callable[[bytes], Dict[K, V]] = pickle.loads,
    ):
        self._name = name
        self._backend = backend
        self._key = f"{name}:{key}"
        self._serializer = serializer
        self._deserializer = deserializer

    def get(self, key: K) -> Optional[V]:
        """Get value for key."""
        map_data = self._get_map()
        return map_data.get(key)

    def put(self, key: K, value: V) -> None:
        """Put key-value pair."""
        map_data = self._get_map()
        map_data[key] = value
        self._put_map(map_data)

    def remove(self, key: K) -> None:
        """Remove key."""
        map_data = self._get_map()
        map_data.pop(key, None)
        self._put_map(map_data)

    def contains(self, key: K) -> bool:
        """Check if key exists."""
        return key in self._get_map()

    def keys(self) -> List[K]:
        """Get all keys."""
        return list(self._get_map().keys())

    def values(self) -> List[V]:
        """Get all values."""
        return list(self._get_map().values())

    def items(self) -> List[tuple]:
        """Get all items."""
        return list(self._get_map().items())

    def clear(self) -> None:
        """Clear map."""
        self._backend.delete(self._key)

    def _get_map(self) -> Dict[K, V]:
        data = self._backend.get(self._key)
        if data is None:
            return {}
        return self._deserializer(data)

    def _put_map(self, map_data: Dict[K, V]) -> None:
        self._backend.put(self._key, self._serializer(map_data))


@dataclass
class Timer:
    """Timer for event-time or processing-time callbacks."""

    timestamp: int
    key: str
    timer_type: str = "event_time"  # event_time or processing_time


class TimerService:
    """Service for managing timers."""

    def __init__(self):
        self._event_time_timers: Dict[str, List[int]] = {}
        self._processing_time_timers: Dict[str, List[int]] = {}
        self._current_watermark = 0
        self._current_processing_time = 0

    def register_event_time_timer(self, key: str, timestamp: int) -> None:
        """Register an event-time timer."""
        if key not in self._event_time_timers:
            self._event_time_timers[key] = []
        self._event_time_timers[key].append(timestamp)
        self._event_time_timers[key].sort()

    def register_processing_time_timer(self, key: str, timestamp: int) -> None:
        """Register a processing-time timer."""
        if key not in self._processing_time_timers:
            self._processing_time_timers[key] = []
        self._processing_time_timers[key].append(timestamp)
        self._processing_time_timers[key].sort()

    def delete_event_time_timer(self, key: str, timestamp: int) -> None:
        """Delete an event-time timer."""
        if key in self._event_time_timers:
            try:
                self._event_time_timers[key].remove(timestamp)
            except ValueError:
                pass

    def delete_processing_time_timer(self, key: str, timestamp: int) -> None:
        """Delete a processing-time timer."""
        if key in self._processing_time_timers:
            try:
                self._processing_time_timers[key].remove(timestamp)
            except ValueError:
                pass

    def advance_watermark(self, watermark: int) -> List[Timer]:
        """Advance watermark and return fired timers."""
        self._current_watermark = watermark
        fired = []

        for key, timers in list(self._event_time_timers.items()):
            while timers and timers[0] <= watermark:
                timestamp = timers.pop(0)
                fired.append(Timer(timestamp, key, "event_time"))

        return fired

    def advance_processing_time(self, timestamp: int) -> List[Timer]:
        """Advance processing time and return fired timers."""
        self._current_processing_time = timestamp
        fired = []

        for key, timers in list(self._processing_time_timers.items()):
            while timers and timers[0] <= timestamp:
                ts = timers.pop(0)
                fired.append(Timer(ts, key, "processing_time"))

        return fired

    def current_watermark(self) -> int:
        """Get current watermark."""
        return self._current_watermark

    def current_processing_time(self) -> int:
        """Get current processing time."""
        return self._current_processing_time


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""

    checkpoint_id: int
    timestamp: int
    completed: bool = False


class CheckpointCoordinator:
    """Coordinate checkpoints for exactly-once processing."""

    def __init__(self, backend: StateBackend):
        self._backend = backend
        self._checkpoint_counter = 0

    def trigger_checkpoint(self) -> CheckpointMetadata:
        """Trigger a new checkpoint."""
        self._checkpoint_counter += 1

        metadata = CheckpointMetadata(
            checkpoint_id=self._checkpoint_counter,
            timestamp=int(time.time() * 1000),
        )

        # Store checkpoint metadata
        key = f"_checkpoint:{metadata.checkpoint_id}"
        data = json.dumps({
            "id": metadata.checkpoint_id,
            "timestamp": metadata.timestamp,
            "completed": False,
        }).encode()
        self._backend.put(key, data)

        return metadata

    def complete_checkpoint(self, checkpoint_id: int) -> None:
        """Mark checkpoint as completed."""
        key = f"_checkpoint:{checkpoint_id}"
        data = self._backend.get(key)

        if data:
            metadata = json.loads(data.decode())
            metadata["completed"] = True
            self._backend.put(key, json.dumps(metadata).encode())

    def get_latest_checkpoint(self) -> Optional[CheckpointMetadata]:
        """Get latest completed checkpoint."""
        for i in range(self._checkpoint_counter, 0, -1):
            key = f"_checkpoint:{i}"
            data = self._backend.get(key)

            if data:
                metadata = json.loads(data.decode())
                if metadata["completed"]:
                    return CheckpointMetadata(
                        checkpoint_id=metadata["id"],
                        timestamp=metadata["timestamp"],
                        completed=True,
                    )

        return None

    def restore_from_checkpoint(self, checkpoint_id: int) -> bool:
        """Restore state from a checkpoint."""
        key = f"_checkpoint:{checkpoint_id}"
        data = self._backend.get(key)

        if not data:
            return False

        metadata = json.loads(data.decode())
        if not metadata["completed"]:
            return False

        self._checkpoint_counter = checkpoint_id
        return True


class StateContext:
    """Context for accessing state in processing functions."""

    def __init__(self, backend: StateBackend, key: str):
        self._backend = backend
        self._key = key
        self._value_states: Dict[str, ValueState] = {}
        self._list_states: Dict[str, ListState] = {}
        self._map_states: Dict[str, MapState] = {}

    def get_value_state(self, name: str) -> ValueState:
        """Get or create a value state."""
        if name not in self._value_states:
            self._value_states[name] = ValueState(name, self._backend, self._key)
        return self._value_states[name]

    def get_list_state(self, name: str) -> ListState:
        """Get or create a list state."""
        if name not in self._list_states:
            self._list_states[name] = ListState(name, self._backend, self._key)
        return self._list_states[name]

    def get_map_state(self, name: str) -> MapState:
        """Get or create a map state."""
        if name not in self._map_states:
            self._map_states[name] = MapState(name, self._backend, self._key)
        return self._map_states[name]
