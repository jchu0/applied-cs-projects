"""Tests for state backends and state types."""

import pytest
import os
import tempfile
import shutil
from typing import Dict, List, Any

from streamanalytics import (
    StateBackend,
    MemoryStateBackend,
    RocksDBStateBackend,
    KeyedState,
    ValueState,
    ListState,
    MapState,
)
from streamanalytics.state import (
    StateDescriptor,
    ReducingState,
    AggregatingState,
)


class TestStateDescriptor:
    """Tests for StateDescriptor."""

    def test_descriptor_creation(self):
        """Test creating a state descriptor."""
        descriptor = StateDescriptor(name="my_state")
        assert descriptor.name == "my_state"
        assert descriptor.default_value is None

    def test_descriptor_with_default(self):
        """Test creating a state descriptor with default value."""
        descriptor = StateDescriptor(name="counter", default_value=0)
        assert descriptor.name == "counter"
        assert descriptor.default_value == 0

    def test_descriptor_with_complex_default(self):
        """Test descriptor with complex default value."""
        descriptor = StateDescriptor(
            name="complex_state",
            default_value={"count": 0, "sum": 0.0}
        )
        assert descriptor.default_value == {"count": 0, "sum": 0.0}


class TestValueState:
    """Tests for ValueState."""

    def test_value_state_creation(self):
        """Test creating a ValueState."""
        state = ValueState(name="test_value")
        assert state.name == "test_value"

    def test_value_state_with_default(self):
        """Test ValueState with default value."""
        state = ValueState(name="counter", default=100)
        state.set_current_key("key1")
        assert state.value() == 100  # Default value

    def test_value_state_update_and_get(self):
        """Test updating and getting ValueState."""
        state = ValueState[str, int](name="test")
        state.set_current_key("key1")

        state.update(42)
        assert state.value() == 42

    def test_value_state_multiple_keys(self):
        """Test ValueState with multiple keys."""
        state = ValueState[str, int](name="test")

        state.set_current_key("key1")
        state.update(100)

        state.set_current_key("key2")
        state.update(200)

        state.set_current_key("key1")
        assert state.value() == 100

        state.set_current_key("key2")
        assert state.value() == 200

    def test_value_state_clear(self):
        """Test clearing ValueState."""
        state = ValueState[str, int](name="test", default=0)
        state.set_current_key("key1")
        state.update(42)
        assert state.value() == 42

        state.clear()
        assert state.value() == 0  # Back to default

    def test_value_state_get_all(self):
        """Test getting all values from ValueState."""
        state = ValueState[str, int](name="test")

        state.set_current_key("a")
        state.update(1)
        state.set_current_key("b")
        state.update(2)
        state.set_current_key("c")
        state.update(3)

        all_values = state.get_all()
        assert all_values == {"a": 1, "b": 2, "c": 3}

    def test_value_state_without_key(self):
        """Test ValueState operations without setting key."""
        state = ValueState[str, int](name="test", default=99)
        # Without setting key, should return default
        assert state.value() == 99

    def test_value_state_update_without_key(self):
        """Test that update without key is a no-op."""
        state = ValueState[str, int](name="test")
        state.update(42)  # Should not raise, but also not store
        assert state.get_all() == {}


class TestListState:
    """Tests for ListState."""

    def test_list_state_creation(self):
        """Test creating a ListState."""
        state = ListState(name="test_list")
        assert state.name == "test_list"

    def test_list_state_add_and_get(self):
        """Test adding and getting from ListState."""
        state = ListState[str, int](name="test")
        state.set_current_key("key1")

        state.add(1)
        state.add(2)
        state.add(3)

        assert state.get() == [1, 2, 3]

    def test_list_state_add_all(self):
        """Test adding multiple values to ListState."""
        state = ListState[str, int](name="test")
        state.set_current_key("key1")

        state.add_all([1, 2, 3, 4, 5])
        assert state.get() == [1, 2, 3, 4, 5]

    def test_list_state_update(self):
        """Test updating (replacing) ListState."""
        state = ListState[str, int](name="test")
        state.set_current_key("key1")

        state.add_all([1, 2, 3])
        state.update([10, 20, 30])

        assert state.get() == [10, 20, 30]

    def test_list_state_multiple_keys(self):
        """Test ListState with multiple keys."""
        state = ListState[str, str](name="test")

        state.set_current_key("user1")
        state.add("event1")
        state.add("event2")

        state.set_current_key("user2")
        state.add("event3")

        state.set_current_key("user1")
        assert state.get() == ["event1", "event2"]

        state.set_current_key("user2")
        assert state.get() == ["event3"]

    def test_list_state_clear(self):
        """Test clearing ListState."""
        state = ListState[str, int](name="test")
        state.set_current_key("key1")
        state.add_all([1, 2, 3])

        state.clear()
        assert state.get() == []

    def test_list_state_get_all(self):
        """Test getting all lists from ListState."""
        state = ListState[str, int](name="test")

        state.set_current_key("a")
        state.add_all([1, 2])
        state.set_current_key("b")
        state.add_all([3, 4, 5])

        all_lists = state.get_all()
        assert all_lists == {"a": [1, 2], "b": [3, 4, 5]}

    def test_list_state_empty_without_key(self):
        """Test ListState operations without setting key."""
        state = ListState[str, int](name="test")
        assert state.get() == []


class TestMapState:
    """Tests for MapState."""

    def test_map_state_creation(self):
        """Test creating a MapState."""
        state = MapState(name="test_map")
        assert state.name == "test_map"

    def test_map_state_put_and_get(self):
        """Test putting and getting from MapState."""
        state = MapState[str, int](name="test")
        state.set_current_key("key1")

        state.put("count", 42)
        assert state.get("count") == 42

    def test_map_state_put_all(self):
        """Test putting multiple values to MapState."""
        state = MapState[str, Any](name="test")
        state.set_current_key("key1")

        state.put_all({"a": 1, "b": 2, "c": 3})
        assert state.get("a") == 1
        assert state.get("b") == 2
        assert state.get("c") == 3

    def test_map_state_contains(self):
        """Test checking if MapState contains a key."""
        state = MapState[str, int](name="test")
        state.set_current_key("key1")

        state.put("exists", 1)
        assert state.contains("exists") is True
        assert state.contains("not_exists") is False

    def test_map_state_remove(self):
        """Test removing from MapState."""
        state = MapState[str, int](name="test")
        state.set_current_key("key1")

        state.put("to_remove", 42)
        assert state.contains("to_remove") is True

        state.remove("to_remove")
        assert state.contains("to_remove") is False

    def test_map_state_keys(self):
        """Test getting keys from MapState."""
        state = MapState[str, int](name="test")
        state.set_current_key("key1")

        state.put_all({"a": 1, "b": 2, "c": 3})
        keys = state.keys()
        assert sorted(keys) == ["a", "b", "c"]

    def test_map_state_values(self):
        """Test getting values from MapState."""
        state = MapState[str, int](name="test")
        state.set_current_key("key1")

        state.put_all({"a": 1, "b": 2, "c": 3})
        values = state.values()
        assert sorted(values) == [1, 2, 3]

    def test_map_state_entries(self):
        """Test getting entries from MapState."""
        state = MapState[str, int](name="test")
        state.set_current_key("key1")

        state.put_all({"x": 10, "y": 20})
        entries = state.entries()
        assert entries == {"x": 10, "y": 20}

    def test_map_state_multiple_keys(self):
        """Test MapState with multiple partition keys."""
        state = MapState[str, int](name="test")

        state.set_current_key("user1")
        state.put("score", 100)

        state.set_current_key("user2")
        state.put("score", 200)

        state.set_current_key("user1")
        assert state.get("score") == 100

        state.set_current_key("user2")
        assert state.get("score") == 200

    def test_map_state_clear(self):
        """Test clearing MapState."""
        state = MapState[str, int](name="test")
        state.set_current_key("key1")

        state.put_all({"a": 1, "b": 2})
        state.clear()

        assert state.keys() == []
        assert state.contains("a") is False

    def test_map_state_get_all(self):
        """Test getting all maps from MapState."""
        state = MapState[str, int](name="test")

        state.set_current_key("k1")
        state.put_all({"a": 1})
        state.set_current_key("k2")
        state.put_all({"b": 2, "c": 3})

        all_maps = state.get_all()
        assert all_maps == {"k1": {"a": 1}, "k2": {"b": 2, "c": 3}}

    def test_map_state_get_nonexistent(self):
        """Test getting nonexistent key from MapState."""
        state = MapState[str, int](name="test")
        state.set_current_key("key1")
        assert state.get("nonexistent") is None


class TestReducingState:
    """Tests for ReducingState."""

    def test_reducing_state_creation(self):
        """Test creating a ReducingState."""
        state = ReducingState(name="sum", reduce_func=lambda a, b: a + b)
        assert state.name == "sum"

    def test_reducing_state_add_and_get(self):
        """Test adding and getting from ReducingState."""
        state = ReducingState[str, int](name="sum", reduce_func=lambda a, b: a + b)
        state.set_current_key("key1")

        state.add(10)
        assert state.get() == 10

        state.add(20)
        assert state.get() == 30

        state.add(30)
        assert state.get() == 60

    def test_reducing_state_multiple_keys(self):
        """Test ReducingState with multiple keys."""
        state = ReducingState[str, int](name="sum", reduce_func=lambda a, b: a + b)

        state.set_current_key("a")
        state.add(100)
        state.add(50)

        state.set_current_key("b")
        state.add(200)

        state.set_current_key("a")
        assert state.get() == 150

        state.set_current_key("b")
        assert state.get() == 200

    def test_reducing_state_max(self):
        """Test ReducingState with max function."""
        state = ReducingState[str, int](name="max", reduce_func=max)
        state.set_current_key("key1")

        state.add(5)
        state.add(10)
        state.add(3)
        state.add(8)

        assert state.get() == 10

    def test_reducing_state_clear(self):
        """Test clearing ReducingState."""
        state = ReducingState[str, int](name="sum", reduce_func=lambda a, b: a + b)
        state.set_current_key("key1")

        state.add(100)
        state.clear()

        assert state.get() is None


class TestAggregatingState:
    """Tests for AggregatingState."""

    def test_aggregating_state_creation(self):
        """Test creating an AggregatingState."""
        state = AggregatingState(
            name="avg",
            create_accumulator=lambda: {"sum": 0, "count": 0},
            add=lambda acc, v: {"sum": acc["sum"] + v, "count": acc["count"] + 1},
            get_result=lambda acc: acc["sum"] / acc["count"] if acc["count"] > 0 else 0
        )
        assert state.name == "avg"

    def test_aggregating_state_add_and_get(self):
        """Test adding and getting from AggregatingState."""
        state = AggregatingState[str, int, Dict, float](
            name="avg",
            create_accumulator=lambda: {"sum": 0, "count": 0},
            add=lambda acc, v: {"sum": acc["sum"] + v, "count": acc["count"] + 1},
            get_result=lambda acc: acc["sum"] / acc["count"] if acc["count"] > 0 else 0
        )
        state.set_current_key("key1")

        state.add(10)
        assert state.get() == 10.0

        state.add(20)
        assert state.get() == 15.0  # (10 + 20) / 2

        state.add(30)
        assert state.get() == 20.0  # (10 + 20 + 30) / 3

    def test_aggregating_state_multiple_keys(self):
        """Test AggregatingState with multiple keys."""
        state = AggregatingState[str, int, List, int](
            name="count",
            create_accumulator=lambda: [],
            add=lambda acc, v: acc + [v],
            get_result=lambda acc: len(acc)
        )

        state.set_current_key("a")
        state.add(1)
        state.add(2)

        state.set_current_key("b")
        state.add(3)

        state.set_current_key("a")
        assert state.get() == 2

        state.set_current_key("b")
        assert state.get() == 1

    def test_aggregating_state_clear(self):
        """Test clearing AggregatingState."""
        state = AggregatingState[str, int, int, int](
            name="sum",
            create_accumulator=lambda: 0,
            add=lambda acc, v: acc + v,
            get_result=lambda acc: acc
        )
        state.set_current_key("key1")

        state.add(100)
        state.clear()

        assert state.get() is None


class TestMemoryStateBackend:
    """Tests for MemoryStateBackend."""

    def test_backend_creation(self, memory_backend):
        """Test creating a MemoryStateBackend."""
        assert memory_backend is not None

    def test_create_value_state(self, memory_backend, value_state_descriptor):
        """Test creating a ValueState from backend."""
        state = memory_backend.create_value_state(value_state_descriptor)
        assert isinstance(state, ValueState)
        assert state.name == "test_value"

    def test_create_list_state(self, memory_backend, list_state_descriptor):
        """Test creating a ListState from backend."""
        state = memory_backend.create_list_state(list_state_descriptor)
        assert isinstance(state, ListState)
        assert state.name == "test_list"

    def test_create_map_state(self, memory_backend, map_state_descriptor):
        """Test creating a MapState from backend."""
        state = memory_backend.create_map_state(map_state_descriptor)
        assert isinstance(state, MapState)
        assert state.name == "test_map"

    def test_create_reducing_state(self, memory_backend):
        """Test creating a ReducingState from backend."""
        descriptor = StateDescriptor(name="sum_state")
        state = memory_backend.create_reducing_state(
            descriptor,
            reduce_func=lambda a, b: a + b
        )
        assert isinstance(state, ReducingState)

    def test_state_reuse(self, memory_backend):
        """Test that same descriptor returns same state instance."""
        descriptor = StateDescriptor(name="shared_state", default_value=0)

        state1 = memory_backend.create_value_state(descriptor)
        state2 = memory_backend.create_value_state(descriptor)

        assert state1 is state2

    def test_backend_thread_safety(self, memory_backend):
        """Test that backend uses locks for thread safety."""
        import threading

        descriptor = StateDescriptor(name="concurrent_state", default_value=0)
        results = []
        errors = []

        def create_state(i):
            try:
                state = memory_backend.create_value_state(descriptor)
                state.set_current_key(f"key_{i}")
                state.update(i)
                results.append((i, state.value()))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_state, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


class TestRocksDBStateBackend:
    """Tests for RocksDBStateBackend."""

    def test_backend_creation(self, rocksdb_backend):
        """Test creating a RocksDBStateBackend."""
        assert rocksdb_backend is not None
        assert rocksdb_backend.db_path.exists()

    def test_create_value_state(self, rocksdb_backend, value_state_descriptor):
        """Test creating a ValueState from RocksDB backend."""
        state = rocksdb_backend.create_value_state(value_state_descriptor)
        assert isinstance(state, ValueState)

    def test_create_list_state(self, rocksdb_backend, list_state_descriptor):
        """Test creating a ListState from RocksDB backend."""
        state = rocksdb_backend.create_list_state(list_state_descriptor)
        assert isinstance(state, ListState)

    def test_create_map_state(self, rocksdb_backend, map_state_descriptor):
        """Test creating a MapState from RocksDB backend."""
        state = rocksdb_backend.create_map_state(map_state_descriptor)
        assert isinstance(state, MapState)

    def test_state_persistence(self, temp_dir):
        """Test that state is persisted to disk."""
        backend = RocksDBStateBackend(db_path=temp_dir)
        descriptor = StateDescriptor(name="persistent", default_value=0)

        state = backend.create_value_state(descriptor)
        state.set_current_key("key1")
        state.update(42)

        # Flush to disk
        backend.flush()

        # Check that state file exists
        state_file = backend.db_path / "persistent.state"
        assert state_file.exists()

    def test_state_recovery(self, temp_dir):
        """Test that state can be recovered after restart."""
        descriptor = StateDescriptor(name="recoverable", default_value=0)

        # Create and populate state
        backend1 = RocksDBStateBackend(db_path=temp_dir)
        state1 = backend1.create_value_state(descriptor)
        state1.set_current_key("key1")
        state1.update(12345)
        backend1.flush()

        # Create new backend pointing to same directory
        backend2 = RocksDBStateBackend(db_path=temp_dir)
        state2 = backend2.create_value_state(descriptor)
        state2.set_current_key("key1")

        # Should recover the value
        assert state2.value() == 12345

    def test_flush_operation(self, rocksdb_backend):
        """Test explicit flush operation."""
        descriptor = StateDescriptor(name="flush_test")
        state = rocksdb_backend.create_value_state(descriptor)
        state.set_current_key("key")
        state.update("test_value")

        # Flush should not raise
        rocksdb_backend.flush()

    def test_compact_operation(self, rocksdb_backend):
        """Test compact operation."""
        # Compact should not raise
        rocksdb_backend.compact()


class TestStateIntegration:
    """Integration tests for state with stream processing."""

    def test_state_across_operations(self, memory_backend):
        """Test using state across multiple operations."""
        # Create multiple states
        value_desc = StateDescriptor(name="counter", default_value=0)
        list_desc = StateDescriptor(name="history")
        map_desc = StateDescriptor(name="details")

        counter = memory_backend.create_value_state(value_desc)
        history = memory_backend.create_list_state(list_desc)
        details = memory_backend.create_map_state(map_desc)

        # Simulate processing events for a key
        key = "user123"
        counter.set_current_key(key)
        history.set_current_key(key)
        details.set_current_key(key)

        # Update states
        counter.update(counter.value() + 1)
        history.add("login")
        details.put("last_action", "login")
        details.put("login_count", 1)

        counter.update(counter.value() + 1)
        history.add("purchase")
        details.put("last_action", "purchase")

        # Verify
        assert counter.value() == 2
        assert history.get() == ["login", "purchase"]
        assert details.get("last_action") == "purchase"
        assert details.get("login_count") == 1

    def test_state_isolation_between_keys(self, memory_backend):
        """Test that state is properly isolated between keys."""
        descriptor = StateDescriptor(name="isolated")
        state = memory_backend.create_value_state(descriptor)

        # Set values for different keys
        state.set_current_key("key_a")
        state.update("value_a")

        state.set_current_key("key_b")
        state.update("value_b")

        state.set_current_key("key_c")
        state.update("value_c")

        # Verify isolation
        state.set_current_key("key_a")
        assert state.value() == "value_a"

        state.set_current_key("key_b")
        assert state.value() == "value_b"

        state.set_current_key("key_c")
        assert state.value() == "value_c"

    def test_large_state(self, memory_backend):
        """Test handling large state."""
        descriptor = StateDescriptor(name="large_state")
        state = memory_backend.create_list_state(descriptor)
        state.set_current_key("key1")

        # Add many elements
        for i in range(10000):
            state.add(i)

        assert len(state.get()) == 10000
        assert state.get()[0] == 0
        assert state.get()[-1] == 9999

    def test_state_with_complex_values(self, memory_backend):
        """Test state with complex nested values."""
        descriptor = StateDescriptor(name="complex")
        state = memory_backend.create_value_state(descriptor)
        state.set_current_key("key1")

        complex_value = {
            "nested": {
                "deeply": {
                    "value": [1, 2, 3]
                }
            },
            "list": [{"a": 1}, {"b": 2}],
            "tuple": (1, 2, 3)
        }

        state.update(complex_value)
        retrieved = state.value()

        assert retrieved["nested"]["deeply"]["value"] == [1, 2, 3]
        assert len(retrieved["list"]) == 2
