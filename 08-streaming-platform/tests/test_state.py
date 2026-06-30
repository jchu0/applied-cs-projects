"""Tests for state management."""

import pytest
import pickle
import json
import time

pytest.importorskip("confluent_kafka")

from streaming.state import (
    StateBackend,
    InMemoryStateBackend,
    ValueState,
    ListState,
    MapState,
    Timer,
    TimerService,
    CheckpointMetadata,
    CheckpointCoordinator,
    StateContext,
)


# --- InMemoryStateBackend Tests ---


class TestInMemoryStateBackend:
    """Tests for InMemoryStateBackend class."""

    def test_put_and_get(self):
        """Test basic put and get operations."""
        backend = InMemoryStateBackend()

        backend.put("key1", b"value1")
        result = backend.get("key1")

        assert result == b"value1"

    def test_get_nonexistent_key(self):
        """Test getting a nonexistent key returns None."""
        backend = InMemoryStateBackend()

        result = backend.get("nonexistent")

        assert result is None

    def test_delete(self):
        """Test deleting a key."""
        backend = InMemoryStateBackend()
        backend.put("key1", b"value1")

        backend.delete("key1")
        result = backend.get("key1")

        assert result is None

    def test_delete_nonexistent_key(self):
        """Test deleting nonexistent key doesn't raise."""
        backend = InMemoryStateBackend()

        # Should not raise
        backend.delete("nonexistent")

    def test_keys(self):
        """Test listing all keys."""
        backend = InMemoryStateBackend()
        backend.put("key1", b"value1")
        backend.put("key2", b"value2")
        backend.put("key3", b"value3")

        keys = backend.keys()

        assert len(keys) == 3
        assert set(keys) == {"key1", "key2", "key3"}

    def test_keys_empty(self):
        """Test listing keys when empty."""
        backend = InMemoryStateBackend()

        keys = backend.keys()

        assert keys == []

    def test_clear(self):
        """Test clearing all state."""
        backend = InMemoryStateBackend()
        backend.put("key1", b"value1")
        backend.put("key2", b"value2")

        backend.clear()

        assert backend.keys() == []

    def test_overwrite_value(self):
        """Test overwriting an existing value."""
        backend = InMemoryStateBackend()
        backend.put("key1", b"value1")

        backend.put("key1", b"new_value")
        result = backend.get("key1")

        assert result == b"new_value"


# --- ValueState Tests ---


class TestValueState:
    """Tests for ValueState class."""

    @pytest.fixture
    def backend(self):
        """Create a fresh backend for each test."""
        return InMemoryStateBackend()

    def test_initial_value_is_none(self, backend):
        """Test initial value is None."""
        state = ValueState("counter", backend, "user1")

        result = state.value()

        assert result is None

    def test_update_and_retrieve(self, backend):
        """Test updating and retrieving value."""
        state = ValueState("counter", backend, "user1")

        state.update(42)
        result = state.value()

        assert result == 42

    def test_clear_value(self, backend):
        """Test clearing value."""
        state = ValueState("counter", backend, "user1")
        state.update(42)

        state.clear()
        result = state.value()

        assert result is None

    def test_different_keys_isolated(self, backend):
        """Test different keys have isolated state."""
        state1 = ValueState("counter", backend, "user1")
        state2 = ValueState("counter", backend, "user2")

        state1.update(100)
        state2.update(200)

        assert state1.value() == 100
        assert state2.value() == 200

    def test_different_names_isolated(self, backend):
        """Test different state names have isolated state."""
        state1 = ValueState("counter", backend, "user1")
        state2 = ValueState("total", backend, "user1")

        state1.update(100)
        state2.update(200)

        assert state1.value() == 100
        assert state2.value() == 200

    def test_complex_value(self, backend):
        """Test storing complex values."""
        state = ValueState("session", backend, "user1")

        value = {"user_id": "user1", "events": [1, 2, 3], "active": True}
        state.update(value)
        result = state.value()

        assert result == value

    def test_custom_serializer(self, backend):
        """Test custom serializer/deserializer."""
        def json_serialize(v):
            return json.dumps(v).encode()

        def json_deserialize(b):
            return json.loads(b.decode())

        state = ValueState(
            "config",
            backend,
            "app1",
            serializer=json_serialize,
            deserializer=json_deserialize,
        )

        state.update({"setting": "value"})
        result = state.value()

        assert result == {"setting": "value"}


# --- ListState Tests ---


class TestListState:
    """Tests for ListState class."""

    @pytest.fixture
    def backend(self):
        """Create a fresh backend for each test."""
        return InMemoryStateBackend()

    def test_initial_list_is_empty(self, backend):
        """Test initial list is empty."""
        state = ListState("events", backend, "user1")

        result = state.get()

        assert result == []

    def test_add_single_element(self, backend):
        """Test adding a single element."""
        state = ListState("events", backend, "user1")

        state.add("event1")
        result = state.get()

        assert result == ["event1"]

    def test_add_multiple_elements(self, backend):
        """Test adding multiple elements."""
        state = ListState("events", backend, "user1")

        state.add("event1")
        state.add("event2")
        state.add("event3")
        result = state.get()

        assert result == ["event1", "event2", "event3"]

    def test_add_all(self, backend):
        """Test adding all elements at once."""
        state = ListState("events", backend, "user1")
        state.add("event1")

        state.add_all(["event2", "event3", "event4"])
        result = state.get()

        assert result == ["event1", "event2", "event3", "event4"]

    def test_update_replaces_list(self, backend):
        """Test update replaces entire list."""
        state = ListState("events", backend, "user1")
        state.add("event1")
        state.add("event2")

        state.update(["new1", "new2"])
        result = state.get()

        assert result == ["new1", "new2"]

    def test_clear_list(self, backend):
        """Test clearing list."""
        state = ListState("events", backend, "user1")
        state.add("event1")
        state.add("event2")

        state.clear()
        result = state.get()

        assert result == []

    def test_preserves_order(self, backend):
        """Test list preserves insertion order."""
        state = ListState("events", backend, "user1")

        for i in range(10):
            state.add(i)

        result = state.get()
        assert result == list(range(10))


# --- MapState Tests ---


class TestMapState:
    """Tests for MapState class."""

    @pytest.fixture
    def backend(self):
        """Create a fresh backend for each test."""
        return InMemoryStateBackend()

    def test_get_nonexistent_key(self, backend):
        """Test getting nonexistent key returns None."""
        state = MapState("session", backend, "user1")

        result = state.get("nonexistent")

        assert result is None

    def test_put_and_get(self, backend):
        """Test putting and getting values."""
        state = MapState("session", backend, "user1")

        state.put("count", 42)
        result = state.get("count")

        assert result == 42

    def test_remove(self, backend):
        """Test removing a key."""
        state = MapState("session", backend, "user1")
        state.put("count", 42)

        state.remove("count")
        result = state.get("count")

        assert result is None

    def test_remove_nonexistent_key(self, backend):
        """Test removing nonexistent key doesn't raise."""
        state = MapState("session", backend, "user1")

        # Should not raise
        state.remove("nonexistent")

    def test_contains(self, backend):
        """Test contains check."""
        state = MapState("session", backend, "user1")
        state.put("count", 42)

        assert state.contains("count") is True
        assert state.contains("nonexistent") is False

    def test_keys(self, backend):
        """Test getting all keys."""
        state = MapState("session", backend, "user1")
        state.put("a", 1)
        state.put("b", 2)
        state.put("c", 3)

        keys = state.keys()

        assert set(keys) == {"a", "b", "c"}

    def test_values(self, backend):
        """Test getting all values."""
        state = MapState("session", backend, "user1")
        state.put("a", 1)
        state.put("b", 2)
        state.put("c", 3)

        values = state.values()

        assert set(values) == {1, 2, 3}

    def test_items(self, backend):
        """Test getting all items."""
        state = MapState("session", backend, "user1")
        state.put("a", 1)
        state.put("b", 2)

        items = state.items()

        assert set(items) == {("a", 1), ("b", 2)}

    def test_clear(self, backend):
        """Test clearing map."""
        state = MapState("session", backend, "user1")
        state.put("a", 1)
        state.put("b", 2)

        state.clear()

        assert state.keys() == []


# --- TimerService Tests ---


class TestTimerService:
    """Tests for TimerService class."""

    def test_register_event_time_timer(self):
        """Test registering event-time timer."""
        service = TimerService()

        service.register_event_time_timer("key1", 10000)

        assert "key1" in service._event_time_timers
        assert 10000 in service._event_time_timers["key1"]

    def test_register_processing_time_timer(self):
        """Test registering processing-time timer."""
        service = TimerService()

        service.register_processing_time_timer("key1", 10000)

        assert "key1" in service._processing_time_timers
        assert 10000 in service._processing_time_timers["key1"]

    def test_multiple_timers_same_key(self):
        """Test multiple timers for same key."""
        service = TimerService()

        service.register_event_time_timer("key1", 10000)
        service.register_event_time_timer("key1", 20000)
        service.register_event_time_timer("key1", 15000)

        timers = service._event_time_timers["key1"]
        assert len(timers) == 3
        assert timers == [10000, 15000, 20000]  # Should be sorted

    def test_delete_event_time_timer(self):
        """Test deleting event-time timer."""
        service = TimerService()
        service.register_event_time_timer("key1", 10000)
        service.register_event_time_timer("key1", 20000)

        service.delete_event_time_timer("key1", 10000)

        assert 10000 not in service._event_time_timers["key1"]
        assert 20000 in service._event_time_timers["key1"]

    def test_delete_nonexistent_timer(self):
        """Test deleting nonexistent timer doesn't raise."""
        service = TimerService()

        # Should not raise
        service.delete_event_time_timer("key1", 10000)

    def test_advance_watermark_fires_timers(self):
        """Test advancing watermark fires due timers."""
        service = TimerService()
        service.register_event_time_timer("key1", 10000)
        service.register_event_time_timer("key1", 20000)
        service.register_event_time_timer("key2", 15000)

        fired = service.advance_watermark(17000)

        assert len(fired) == 2
        # key1:10000 and key2:15000 should fire
        fired_timestamps = [(t.key, t.timestamp) for t in fired]
        assert ("key1", 10000) in fired_timestamps
        assert ("key2", 15000) in fired_timestamps
        # key1:20000 should not fire
        assert 20000 in service._event_time_timers.get("key1", [])

    def test_advance_processing_time_fires_timers(self):
        """Test advancing processing time fires due timers."""
        service = TimerService()
        service.register_processing_time_timer("key1", 10000)
        service.register_processing_time_timer("key1", 20000)

        fired = service.advance_processing_time(15000)

        assert len(fired) == 1
        assert fired[0].key == "key1"
        assert fired[0].timestamp == 10000
        assert fired[0].timer_type == "processing_time"

    def test_timers_removed_after_firing(self):
        """Test timers are removed after firing."""
        service = TimerService()
        service.register_event_time_timer("key1", 10000)

        service.advance_watermark(15000)

        assert 10000 not in service._event_time_timers.get("key1", [])

    def test_current_watermark(self):
        """Test current watermark tracking."""
        service = TimerService()

        assert service.current_watermark() == 0

        service.advance_watermark(10000)
        assert service.current_watermark() == 10000

    def test_current_processing_time(self):
        """Test current processing time tracking."""
        service = TimerService()

        assert service.current_processing_time() == 0

        service.advance_processing_time(10000)
        assert service.current_processing_time() == 10000


# --- CheckpointCoordinator Tests ---


class TestCheckpointCoordinator:
    """Tests for CheckpointCoordinator class."""

    @pytest.fixture
    def backend(self):
        """Create a fresh backend for each test."""
        return InMemoryStateBackend()

    def test_trigger_checkpoint(self, backend):
        """Test triggering a checkpoint."""
        coordinator = CheckpointCoordinator(backend)

        metadata = coordinator.trigger_checkpoint()

        assert metadata.checkpoint_id == 1
        assert metadata.timestamp > 0
        assert metadata.completed is False

    def test_multiple_checkpoints(self, backend):
        """Test triggering multiple checkpoints."""
        coordinator = CheckpointCoordinator(backend)

        meta1 = coordinator.trigger_checkpoint()
        meta2 = coordinator.trigger_checkpoint()
        meta3 = coordinator.trigger_checkpoint()

        assert meta1.checkpoint_id == 1
        assert meta2.checkpoint_id == 2
        assert meta3.checkpoint_id == 3

    def test_complete_checkpoint(self, backend):
        """Test completing a checkpoint."""
        coordinator = CheckpointCoordinator(backend)
        metadata = coordinator.trigger_checkpoint()

        coordinator.complete_checkpoint(metadata.checkpoint_id)

        latest = coordinator.get_latest_checkpoint()
        assert latest is not None
        assert latest.completed is True

    def test_get_latest_completed_checkpoint(self, backend):
        """Test getting latest completed checkpoint."""
        coordinator = CheckpointCoordinator(backend)

        # Create and complete first checkpoint
        meta1 = coordinator.trigger_checkpoint()
        coordinator.complete_checkpoint(meta1.checkpoint_id)

        # Create second checkpoint (not completed)
        coordinator.trigger_checkpoint()

        # Should return first completed checkpoint
        latest = coordinator.get_latest_checkpoint()
        assert latest.checkpoint_id == 1

    def test_get_latest_checkpoint_none(self, backend):
        """Test getting latest when none completed."""
        coordinator = CheckpointCoordinator(backend)
        coordinator.trigger_checkpoint()  # Not completed

        latest = coordinator.get_latest_checkpoint()

        assert latest is None

    def test_restore_from_checkpoint(self, backend):
        """Test restoring from checkpoint."""
        coordinator = CheckpointCoordinator(backend)
        meta = coordinator.trigger_checkpoint()
        coordinator.complete_checkpoint(meta.checkpoint_id)

        result = coordinator.restore_from_checkpoint(meta.checkpoint_id)

        assert result is True
        assert coordinator._checkpoint_counter == 1

    def test_restore_from_incomplete_checkpoint(self, backend):
        """Test restoring from incomplete checkpoint fails."""
        coordinator = CheckpointCoordinator(backend)
        meta = coordinator.trigger_checkpoint()  # Not completed

        result = coordinator.restore_from_checkpoint(meta.checkpoint_id)

        assert result is False

    def test_restore_from_nonexistent_checkpoint(self, backend):
        """Test restoring from nonexistent checkpoint fails."""
        coordinator = CheckpointCoordinator(backend)

        result = coordinator.restore_from_checkpoint(999)

        assert result is False


# --- StateContext Tests ---


class TestStateContext:
    """Tests for StateContext class."""

    @pytest.fixture
    def backend(self):
        """Create a fresh backend for each test."""
        return InMemoryStateBackend()

    def test_get_value_state(self, backend):
        """Test getting value state."""
        ctx = StateContext(backend, "user1")

        state = ctx.get_value_state("counter")

        assert isinstance(state, ValueState)

    def test_get_value_state_cached(self, backend):
        """Test getting same value state returns cached instance."""
        ctx = StateContext(backend, "user1")

        state1 = ctx.get_value_state("counter")
        state2 = ctx.get_value_state("counter")

        assert state1 is state2

    def test_get_list_state(self, backend):
        """Test getting list state."""
        ctx = StateContext(backend, "user1")

        state = ctx.get_list_state("events")

        assert isinstance(state, ListState)

    def test_get_list_state_cached(self, backend):
        """Test getting same list state returns cached instance."""
        ctx = StateContext(backend, "user1")

        state1 = ctx.get_list_state("events")
        state2 = ctx.get_list_state("events")

        assert state1 is state2

    def test_get_map_state(self, backend):
        """Test getting map state."""
        ctx = StateContext(backend, "user1")

        state = ctx.get_map_state("session")

        assert isinstance(state, MapState)

    def test_get_map_state_cached(self, backend):
        """Test getting same map state returns cached instance."""
        ctx = StateContext(backend, "user1")

        state1 = ctx.get_map_state("session")
        state2 = ctx.get_map_state("session")

        assert state1 is state2

    def test_different_contexts_isolated(self, backend):
        """Test different contexts have isolated state."""
        ctx1 = StateContext(backend, "user1")
        ctx2 = StateContext(backend, "user2")

        state1 = ctx1.get_value_state("counter")
        state2 = ctx2.get_value_state("counter")

        state1.update(100)
        state2.update(200)

        assert state1.value() == 100
        assert state2.value() == 200


# --- Timer Tests ---


class TestTimer:
    """Tests for Timer dataclass."""

    def test_timer_creation(self):
        """Test timer creation."""
        timer = Timer(timestamp=10000, key="user1")

        assert timer.timestamp == 10000
        assert timer.key == "user1"
        assert timer.timer_type == "event_time"

    def test_timer_with_type(self):
        """Test timer with explicit type."""
        timer = Timer(timestamp=10000, key="user1", timer_type="processing_time")

        assert timer.timer_type == "processing_time"


# --- Integration Tests ---


class TestStateIntegration:
    """Integration tests for state management."""

    def test_session_tracking(self):
        """Test using state for session tracking."""
        backend = InMemoryStateBackend()
        ctx = StateContext(backend, "user123")

        # Get state handles
        event_count = ctx.get_value_state("event_count")
        events = ctx.get_list_state("events")
        metadata = ctx.get_map_state("metadata")

        # Simulate session activity
        for i in range(5):
            current = event_count.value() or 0
            event_count.update(current + 1)
            events.add(f"event_{i}")

        metadata.put("last_activity", time.time())
        metadata.put("user_agent", "TestBrowser/1.0")

        # Verify state
        assert event_count.value() == 5
        assert len(events.get()) == 5
        assert metadata.contains("last_activity")
        assert metadata.get("user_agent") == "TestBrowser/1.0"

    def test_timer_based_session_expiry(self):
        """Test using timers for session expiry."""
        timer_service = TimerService()

        # Simulate session start
        session_start_time = 10000
        session_timeout = 30000

        # Register expiry timer
        timer_service.register_event_time_timer(
            "user1",
            session_start_time + session_timeout
        )

        # Advance time but not past expiry
        fired = timer_service.advance_watermark(session_start_time + 15000)
        assert len(fired) == 0

        # Advance past expiry
        fired = timer_service.advance_watermark(session_start_time + session_timeout + 1)
        assert len(fired) == 1
        assert fired[0].key == "user1"

    def test_checkpoint_with_state(self):
        """Test checkpointing with state."""
        backend = InMemoryStateBackend()
        coordinator = CheckpointCoordinator(backend)

        # Store some state
        value_state = ValueState("counter", backend, "key1")
        value_state.update(100)

        list_state = ListState("events", backend, "key1")
        list_state.add_all(["e1", "e2", "e3"])

        # Create checkpoint
        meta = coordinator.trigger_checkpoint()

        # Modify state
        value_state.update(200)
        list_state.add("e4")

        # Verify current state
        assert value_state.value() == 200
        assert len(list_state.get()) == 4

        # In real implementation, restore would reset state
        # Here we just verify checkpoint was created
        coordinator.complete_checkpoint(meta.checkpoint_id)
        latest = coordinator.get_latest_checkpoint()
        assert latest is not None
        assert latest.checkpoint_id == meta.checkpoint_id
