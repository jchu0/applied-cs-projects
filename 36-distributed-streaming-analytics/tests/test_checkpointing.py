"""Tests for checkpointing and recovery."""

import pytest
import os
import time
import tempfile
import shutil
import pickle
from pathlib import Path
from typing import Dict, List, Any

from streamanalytics import (
    StreamExecutionEnvironment,
    MemoryStateBackend,
    RocksDBStateBackend,
    Checkpoint,
)
from streamanalytics.state import (
    StateDescriptor,
    FileCheckpointStorage,
    CheckpointCoordinator,
)


class TestCheckpoint:
    """Tests for Checkpoint class."""

    def test_checkpoint_creation(self):
        """Test creating a checkpoint."""
        checkpoint = Checkpoint(
            checkpoint_id=1,
            timestamp=1000.0,
            state_handles={},
            metadata={}
        )
        assert checkpoint.checkpoint_id == 1
        assert checkpoint.timestamp == 1000.0

    def test_checkpoint_with_state_handles(self):
        """Test checkpoint with state handles."""
        state_data = pickle.dumps({"key": "value"})
        checkpoint = Checkpoint(
            checkpoint_id=2,
            timestamp=2000.0,
            state_handles={"state1": state_data},
            metadata={"operator": "map"}
        )
        assert "state1" in checkpoint.state_handles
        assert checkpoint.metadata["operator"] == "map"

    def test_checkpoint_serialize(self):
        """Test checkpoint serialization."""
        checkpoint = Checkpoint(
            checkpoint_id=3,
            timestamp=3000.0,
            state_handles={"test": pickle.dumps([1, 2, 3])},
            metadata={"version": "1.0"}
        )

        serialized = checkpoint.serialize()
        assert isinstance(serialized, bytes)
        assert len(serialized) > 0

    def test_checkpoint_deserialize(self):
        """Test checkpoint deserialization."""
        original = Checkpoint(
            checkpoint_id=4,
            timestamp=4000.0,
            state_handles={"data": pickle.dumps({"a": 1, "b": 2})},
            metadata={"source": "test"}
        )

        serialized = original.serialize()
        restored = Checkpoint.deserialize(serialized)

        assert restored.checkpoint_id == original.checkpoint_id
        assert restored.timestamp == original.timestamp
        assert restored.state_handles.keys() == original.state_handles.keys()
        assert restored.metadata == original.metadata

    def test_checkpoint_roundtrip(self):
        """Test complete checkpoint roundtrip."""
        # Create complex checkpoint
        state1 = pickle.dumps({"count": 100, "sum": 500.0})
        state2 = pickle.dumps(["event1", "event2", "event3"])

        original = Checkpoint(
            checkpoint_id=5,
            timestamp=time.time(),
            state_handles={
                "counter_state": state1,
                "event_log": state2
            },
            metadata={
                "parallelism": 4,
                "operator_ids": ["op1", "op2"]
            }
        )

        # Serialize and deserialize
        restored = Checkpoint.deserialize(original.serialize())

        # Verify state data
        restored_counter = pickle.loads(restored.state_handles["counter_state"])
        assert restored_counter["count"] == 100
        assert restored_counter["sum"] == 500.0

        restored_events = pickle.loads(restored.state_handles["event_log"])
        assert restored_events == ["event1", "event2", "event3"]


class TestFileCheckpointStorage:
    """Tests for FileCheckpointStorage."""

    def test_storage_creation(self, checkpoint_storage):
        """Test creating a FileCheckpointStorage."""
        assert checkpoint_storage is not None
        assert checkpoint_storage.base_path.exists()

    def test_save_checkpoint(self, checkpoint_storage):
        """Test saving a checkpoint."""
        checkpoint = Checkpoint(
            checkpoint_id=1,
            timestamp=time.time(),
            state_handles={"test": pickle.dumps("data")},
            metadata={}
        )

        handle = checkpoint_storage.save(checkpoint)
        assert handle is not None
        assert os.path.exists(handle)

    def test_load_checkpoint(self, checkpoint_storage):
        """Test loading a checkpoint."""
        original = Checkpoint(
            checkpoint_id=2,
            timestamp=1234567890.0,
            state_handles={"state": pickle.dumps({"value": 42})},
            metadata={"test": True}
        )

        handle = checkpoint_storage.save(original)
        loaded = checkpoint_storage.load(handle)

        assert loaded.checkpoint_id == original.checkpoint_id
        assert loaded.timestamp == original.timestamp
        assert loaded.metadata == original.metadata

    def test_delete_checkpoint(self, checkpoint_storage):
        """Test deleting a checkpoint."""
        checkpoint = Checkpoint(
            checkpoint_id=3,
            timestamp=time.time(),
            state_handles={},
            metadata={}
        )

        handle = checkpoint_storage.save(checkpoint)
        assert os.path.exists(handle)

        checkpoint_storage.delete(handle)
        assert not os.path.exists(handle)

    def test_delete_nonexistent_checkpoint(self, checkpoint_storage):
        """Test deleting a nonexistent checkpoint doesn't raise."""
        # Should not raise an error
        checkpoint_storage.delete("/nonexistent/path/checkpoint.bin")

    def test_multiple_checkpoints(self, checkpoint_storage):
        """Test saving multiple checkpoints."""
        handles = []
        for i in range(5):
            checkpoint = Checkpoint(
                checkpoint_id=i,
                timestamp=time.time(),
                state_handles={"id": pickle.dumps(i)},
                metadata={"index": i}
            )
            handle = checkpoint_storage.save(checkpoint)
            handles.append(handle)

        # Verify all were saved
        for i, handle in enumerate(handles):
            loaded = checkpoint_storage.load(handle)
            assert loaded.checkpoint_id == i
            assert pickle.loads(loaded.state_handles["id"]) == i


class TestCheckpointCoordinator:
    """Tests for CheckpointCoordinator."""

    def test_coordinator_creation(self, checkpoint_coordinator):
        """Test creating a CheckpointCoordinator."""
        assert checkpoint_coordinator is not None
        assert checkpoint_coordinator.interval_ms == 1000
        assert checkpoint_coordinator.min_pause_ms == 100
        assert checkpoint_coordinator.timeout_ms == 10000

    def test_trigger_checkpoint(self, checkpoint_coordinator, memory_backend):
        """Test triggering a checkpoint."""
        # Create some state
        descriptor = StateDescriptor(name="test_state")
        state = memory_backend.create_value_state(descriptor)
        state.set_current_key("key1")
        state.update(42)

        checkpoint = checkpoint_coordinator.trigger_checkpoint([memory_backend])
        assert checkpoint is not None
        assert checkpoint.checkpoint_id == 1

    def test_checkpoint_id_increments(self, checkpoint_coordinator, memory_backend):
        """Test that checkpoint ID increments."""
        # Need to wait between checkpoints due to min_pause_ms
        checkpoint1 = checkpoint_coordinator.trigger_checkpoint([memory_backend])

        # Wait for min_pause
        time.sleep(0.15)

        checkpoint2 = checkpoint_coordinator.trigger_checkpoint([memory_backend])

        assert checkpoint1.checkpoint_id == 1
        assert checkpoint2.checkpoint_id == 2

    def test_min_pause_enforcement(self, temp_dir):
        """Test that min_pause is enforced between checkpoints."""
        storage = FileCheckpointStorage(os.path.join(temp_dir, "checkpoints"))
        coordinator = CheckpointCoordinator(
            storage=storage,
            interval_ms=100,
            min_pause_ms=500,  # Half second minimum pause
            timeout_ms=5000
        )
        backend = MemoryStateBackend()

        # First checkpoint should succeed
        cp1 = coordinator.trigger_checkpoint([backend])
        assert cp1 is not None

        # Immediate second checkpoint should fail due to min_pause
        cp2 = coordinator.trigger_checkpoint([backend])
        assert cp2 is None

        # After waiting, should succeed
        time.sleep(0.6)
        cp3 = coordinator.trigger_checkpoint([backend])
        assert cp3 is not None

    def test_restore_latest(self, checkpoint_coordinator, memory_backend):
        """Test restoring from latest checkpoint."""
        # Create state and checkpoint
        descriptor = StateDescriptor(name="restorable")
        state = memory_backend.create_value_state(descriptor)
        state.set_current_key("key1")
        state.update(100)

        checkpoint_coordinator.trigger_checkpoint([memory_backend])

        # Modify state
        state.update(200)
        assert state.value() == 200

        # Restore from checkpoint
        success = checkpoint_coordinator.restore_latest([memory_backend])
        assert success is True

        # State should be restored
        state.set_current_key("key1")
        assert state.value() == 100

    def test_restore_with_no_checkpoints(self, checkpoint_coordinator, memory_backend):
        """Test restore when no checkpoints exist."""
        success = checkpoint_coordinator.restore_latest([memory_backend])
        assert success is False

    def test_checkpoint_cleanup(self, temp_dir):
        """Test that old checkpoints are cleaned up."""
        storage = FileCheckpointStorage(os.path.join(temp_dir, "checkpoints"))
        coordinator = CheckpointCoordinator(
            storage=storage,
            interval_ms=100,
            min_pause_ms=0,  # No pause for testing
            timeout_ms=5000
        )
        coordinator._max_retained = 2  # Only keep 2 checkpoints

        backend = MemoryStateBackend()

        # Create 5 checkpoints
        handles = []
        for i in range(5):
            cp = coordinator.trigger_checkpoint([backend])
            if cp:
                handles.append(cp.checkpoint_id)
            time.sleep(0.01)

        # Should only have 2 completed checkpoints retained
        assert len(coordinator._completed_checkpoints) <= 2


class TestMemoryBackendCheckpointing:
    """Tests for checkpointing with MemoryStateBackend."""

    def test_checkpoint_value_state(self, memory_backend):
        """Test checkpointing ValueState."""
        descriptor = StateDescriptor(name="value_cp", default_value=0)
        state = memory_backend.create_value_state(descriptor)

        state.set_current_key("k1")
        state.update(10)
        state.set_current_key("k2")
        state.update(20)

        checkpoint = memory_backend.checkpoint(1)

        assert "value_cp" in checkpoint.state_handles
        restored_data = pickle.loads(checkpoint.state_handles["value_cp"])
        assert restored_data == {"k1": 10, "k2": 20}

    def test_checkpoint_list_state(self, memory_backend):
        """Test checkpointing ListState."""
        descriptor = StateDescriptor(name="list_cp")
        state = memory_backend.create_list_state(descriptor)

        state.set_current_key("k1")
        state.add_all([1, 2, 3])
        state.set_current_key("k2")
        state.add_all([4, 5])

        checkpoint = memory_backend.checkpoint(1)

        assert "list_cp" in checkpoint.state_handles
        restored_data = pickle.loads(checkpoint.state_handles["list_cp"])
        assert restored_data == {"k1": [1, 2, 3], "k2": [4, 5]}

    def test_checkpoint_map_state(self, memory_backend):
        """Test checkpointing MapState."""
        descriptor = StateDescriptor(name="map_cp")
        state = memory_backend.create_map_state(descriptor)

        state.set_current_key("k1")
        state.put_all({"a": 1, "b": 2})
        state.set_current_key("k2")
        state.put_all({"c": 3})

        checkpoint = memory_backend.checkpoint(1)

        assert "map_cp" in checkpoint.state_handles
        restored_data = pickle.loads(checkpoint.state_handles["map_cp"])
        assert restored_data == {"k1": {"a": 1, "b": 2}, "k2": {"c": 3}}

    def test_checkpoint_reducing_state(self, memory_backend):
        """Test checkpointing ReducingState."""
        descriptor = StateDescriptor(name="reduce_cp")
        state = memory_backend.create_reducing_state(
            descriptor,
            reduce_func=lambda a, b: a + b
        )

        state.set_current_key("k1")
        state.add(10)
        state.add(20)
        state.set_current_key("k2")
        state.add(100)

        checkpoint = memory_backend.checkpoint(1)

        assert "reduce_cp" in checkpoint.state_handles
        restored_data = pickle.loads(checkpoint.state_handles["reduce_cp"])
        assert restored_data == {"k1": 30, "k2": 100}

    def test_restore_value_state(self, memory_backend):
        """Test restoring ValueState from checkpoint."""
        descriptor = StateDescriptor(name="restore_value")
        state = memory_backend.create_value_state(descriptor)

        state.set_current_key("k1")
        state.update(42)

        # Create checkpoint
        checkpoint = memory_backend.checkpoint(1)

        # Modify state
        state.update(99)
        assert state.value() == 99

        # Restore
        memory_backend.restore(checkpoint)

        state.set_current_key("k1")
        assert state.value() == 42

    def test_restore_multiple_states(self, memory_backend):
        """Test restoring multiple states from checkpoint."""
        value_desc = StateDescriptor(name="multi_value")
        list_desc = StateDescriptor(name="multi_list")

        value_state = memory_backend.create_value_state(value_desc)
        list_state = memory_backend.create_list_state(list_desc)

        value_state.set_current_key("k1")
        value_state.update("original_value")

        list_state.set_current_key("k1")
        list_state.add_all(["a", "b", "c"])

        # Checkpoint
        checkpoint = memory_backend.checkpoint(1)

        # Modify states
        value_state.update("modified_value")
        list_state.update(["x", "y", "z"])

        # Restore
        memory_backend.restore(checkpoint)

        value_state.set_current_key("k1")
        list_state.set_current_key("k1")

        assert value_state.value() == "original_value"
        assert list_state.get() == ["a", "b", "c"]


class TestRocksDBBackendCheckpointing:
    """Tests for checkpointing with RocksDBStateBackend."""

    def test_checkpoint_creates_snapshot(self, rocksdb_backend):
        """Test that checkpoint creates a snapshot."""
        descriptor = StateDescriptor(name="rocksdb_cp")
        state = rocksdb_backend.create_value_state(descriptor)

        state.set_current_key("k1")
        state.update(100)

        checkpoint = rocksdb_backend.checkpoint(1)
        assert checkpoint is not None
        assert checkpoint.checkpoint_id == 1

    def test_checkpoint_persists_to_disk(self, rocksdb_backend):
        """Test that checkpoint persists state to disk."""
        descriptor = StateDescriptor(name="persist_test")
        state = rocksdb_backend.create_value_state(descriptor)

        state.set_current_key("k1")
        state.update("persisted_value")

        rocksdb_backend.checkpoint(1)

        # Verify file exists
        state_file = rocksdb_backend.db_path / "persist_test.state"
        assert state_file.exists()

    def test_restore_from_checkpoint(self, rocksdb_backend):
        """Test restoring RocksDB backend from checkpoint."""
        descriptor = StateDescriptor(name="rocksdb_restore")
        state = rocksdb_backend.create_value_state(descriptor)

        state.set_current_key("k1")
        state.update(500)

        checkpoint = rocksdb_backend.checkpoint(1)

        # Modify
        state.update(999)

        # Restore
        rocksdb_backend.restore(checkpoint)

        state.set_current_key("k1")
        assert state.value() == 500


class TestCheckpointRecovery:
    """Integration tests for checkpoint-based recovery."""

    def test_full_recovery_flow(self, temp_dir):
        """Test complete checkpoint and recovery flow."""
        storage = FileCheckpointStorage(os.path.join(temp_dir, "checkpoints"))
        coordinator = CheckpointCoordinator(storage=storage, min_pause_ms=0)
        backend = MemoryStateBackend()

        # Create states
        counter_desc = StateDescriptor(name="counter")
        events_desc = StateDescriptor(name="events")

        counter = backend.create_value_state(counter_desc)
        events = backend.create_list_state(events_desc)

        # Simulate processing
        for i in range(5):
            counter.set_current_key(f"user_{i % 2}")
            current = counter.value() or 0
            counter.update(current + 1)

            events.set_current_key(f"user_{i % 2}")
            events.add(f"event_{i}")

        # Checkpoint
        coordinator.trigger_checkpoint([backend])

        # More processing
        counter.set_current_key("user_0")
        counter.update(100)  # This should be lost

        # Simulate failure and recovery
        coordinator.restore_latest([backend])

        # Verify state
        counter.set_current_key("user_0")
        assert counter.value() == 3  # Had 3 events for user_0

        counter.set_current_key("user_1")
        assert counter.value() == 2  # Had 2 events for user_1

    def test_incremental_checkpointing(self, temp_dir):
        """Test multiple incremental checkpoints."""
        storage = FileCheckpointStorage(os.path.join(temp_dir, "checkpoints"))
        coordinator = CheckpointCoordinator(storage=storage, min_pause_ms=0)
        backend = MemoryStateBackend()

        desc = StateDescriptor(name="incremental")
        state = backend.create_value_state(desc)
        state.set_current_key("key")

        # First checkpoint
        state.update(10)
        cp1 = coordinator.trigger_checkpoint([backend])

        # Second checkpoint
        state.update(20)
        cp2 = coordinator.trigger_checkpoint([backend])

        # Third checkpoint
        state.update(30)
        cp3 = coordinator.trigger_checkpoint([backend])

        # Verify we have 3 checkpoints
        assert cp3.checkpoint_id == 3

        # Restore should use latest
        state.update(9999)
        coordinator.restore_latest([backend])

        state.set_current_key("key")
        assert state.value() == 30

    def test_recovery_with_multiple_backends(self, temp_dir):
        """Test recovery with multiple state backends."""
        storage = FileCheckpointStorage(os.path.join(temp_dir, "checkpoints"))
        coordinator = CheckpointCoordinator(storage=storage, min_pause_ms=0)

        backend1 = MemoryStateBackend()
        backend2 = MemoryStateBackend()

        desc1 = StateDescriptor(name="backend1_state")
        desc2 = StateDescriptor(name="backend2_state")

        state1 = backend1.create_value_state(desc1)
        state2 = backend2.create_value_state(desc2)

        state1.set_current_key("k1")
        state1.update("value1")

        state2.set_current_key("k2")
        state2.update("value2")

        # Checkpoint both backends
        coordinator.trigger_checkpoint([backend1, backend2])

        # Modify both
        state1.update("modified1")
        state2.update("modified2")

        # Restore both
        coordinator.restore_latest([backend1, backend2])

        state1.set_current_key("k1")
        state2.set_current_key("k2")

        assert state1.value() == "value1"
        assert state2.value() == "value2"


class TestStreamExecutionEnvironmentCheckpointing:
    """Tests for checkpointing through StreamExecutionEnvironment."""

    def test_enable_checkpointing(self, env):
        """Test enabling checkpointing on environment."""
        env.enable_checkpointing(10000)
        assert env._checkpointing_enabled is True
        assert env._checkpoint_interval == 10000

    def test_set_state_backend(self, env, memory_backend):
        """Test setting state backend on environment."""
        result = env.set_state_backend(memory_backend)
        assert result is env  # Returns self for chaining
        assert env._state_backend is memory_backend

    def test_environment_with_checkpointing_config(self):
        """Test environment with full checkpointing configuration."""
        env = StreamExecutionEnvironment.get_execution_environment(parallelism=4)
        backend = MemoryStateBackend()

        env.enable_checkpointing(5000)
        env.set_state_backend(backend)

        assert env.parallelism == 4
        assert env._checkpointing_enabled is True
        assert env._checkpoint_interval == 5000
        assert env._state_backend is backend


class TestCheckpointEdgeCases:
    """Tests for edge cases in checkpointing."""

    def test_checkpoint_empty_state(self, memory_backend):
        """Test checkpointing with no state registered."""
        checkpoint = memory_backend.checkpoint(1)
        assert checkpoint.state_handles == {}

    def test_restore_empty_checkpoint(self, memory_backend):
        """Test restoring from empty checkpoint."""
        checkpoint = Checkpoint(
            checkpoint_id=1,
            timestamp=time.time(),
            state_handles={},
            metadata={}
        )
        # Should not raise
        memory_backend.restore(checkpoint)

    def test_checkpoint_with_none_values(self, memory_backend):
        """Test checkpointing state with None values."""
        desc = StateDescriptor(name="nullable")
        state = memory_backend.create_value_state(desc)

        state.set_current_key("k1")
        state.update(None)

        checkpoint = memory_backend.checkpoint(1)
        restored_data = pickle.loads(checkpoint.state_handles["nullable"])
        assert restored_data == {"k1": None}

    def test_checkpoint_with_empty_collections(self, memory_backend):
        """Test checkpointing empty collections."""
        list_desc = StateDescriptor(name="empty_list")
        map_desc = StateDescriptor(name="empty_map")

        list_state = memory_backend.create_list_state(list_desc)
        map_state = memory_backend.create_map_state(map_desc)

        list_state.set_current_key("k1")
        list_state.update([])  # Empty list

        map_state.set_current_key("k1")
        # Map with no entries

        checkpoint = memory_backend.checkpoint(1)

        list_data = pickle.loads(checkpoint.state_handles["empty_list"])
        assert list_data == {"k1": []}

    def test_restore_partial_state(self, memory_backend):
        """Test restoring when checkpoint has fewer states than current."""
        # Create checkpoint with one state
        desc1 = StateDescriptor(name="state1")
        state1 = memory_backend.create_value_state(desc1)
        state1.set_current_key("k1")
        state1.update("original")

        checkpoint = memory_backend.checkpoint(1)

        # Create additional state after checkpoint
        desc2 = StateDescriptor(name="state2")
        state2 = memory_backend.create_value_state(desc2)
        state2.set_current_key("k2")
        state2.update("new_state")

        # Restore should not affect state2
        memory_backend.restore(checkpoint)

        state1.set_current_key("k1")
        state2.set_current_key("k2")

        assert state1.value() == "original"
        assert state2.value() == "new_state"  # Unchanged

    def test_large_checkpoint(self, memory_backend):
        """Test checkpointing large state."""
        desc = StateDescriptor(name="large")
        state = memory_backend.create_list_state(desc)
        state.set_current_key("k1")

        # Add 10000 items
        large_list = list(range(10000))
        state.add_all(large_list)

        checkpoint = memory_backend.checkpoint(1)

        # Verify checkpoint is valid
        restored_data = pickle.loads(checkpoint.state_handles["large"])
        assert restored_data["k1"] == large_list
