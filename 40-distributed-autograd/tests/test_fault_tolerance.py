"""Comprehensive tests for Fault Tolerance features (Phase 5)."""

import pytest
import numpy as np
import threading
import time
import tempfile
import os
import shutil
from unittest.mock import Mock, MagicMock, patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from distautograd.distributed.ddp import (
    WorkerState,
    Heartbeat,
    FailureEvent,
    HealthMonitor,
    Checkpointer,
    RecoveryManager,
    FaultTolerantTrainer,
    DistributedDataParallel,
)
from distautograd.core.context import ProcessGroup, Backend


# =============================================================================
# Test Fixtures
# =============================================================================

class MockParameter:
    """Mock parameter for testing."""

    def __init__(self, shape, requires_grad=True):
        self.data = np.random.randn(*shape).astype(np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._hooks = []

    def register_hook(self, hook):
        self._hooks.append(hook)
        return len(self._hooks) - 1


class MockModule:
    """Mock module for testing."""

    def __init__(self, num_params=3):
        self._params = [
            MockParameter((100, 100)),
            MockParameter((50, 50)),
            MockParameter((20, 20)),
        ][:num_params]
        self._state = {"layer1": np.ones((10, 10)), "layer2": np.ones((5, 5))}

    def parameters(self):
        return iter(self._params)

    def __call__(self, x):
        return x

    def state_dict(self):
        return self._state.copy()

    def load_state_dict(self, state):
        self._state = state.copy()


class MockOptimizer:
    """Mock optimizer for testing."""

    def __init__(self):
        self._state = {"lr": 0.01, "momentum": 0.9}

    def state_dict(self):
        return self._state.copy()

    def load_state_dict(self, state):
        self._state = state.copy()


@pytest.fixture
def process_group():
    """Create a process group for testing."""
    return ProcessGroup(ranks=[0, 1, 2, 3], backend=Backend.GLOO, name="test_group")


@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp(prefix="checkpoint_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_module():
    """Create a mock module."""
    return MockModule()


# =============================================================================
# WorkerState Tests
# =============================================================================

class TestWorkerState:
    """Tests for WorkerState enum."""

    def test_worker_states_exist(self):
        """Test all worker states are defined."""
        assert WorkerState.INITIALIZING is not None
        assert WorkerState.RUNNING is not None
        assert WorkerState.SUSPENDED is not None
        assert WorkerState.FAILED is not None
        assert WorkerState.TERMINATED is not None

    def test_worker_states_unique(self):
        """Test worker states are unique."""
        states = [
            WorkerState.INITIALIZING,
            WorkerState.RUNNING,
            WorkerState.SUSPENDED,
            WorkerState.FAILED,
            WorkerState.TERMINATED,
        ]
        assert len(states) == len(set(states))


# =============================================================================
# Heartbeat Tests
# =============================================================================

class TestHeartbeat:
    """Tests for Heartbeat dataclass."""

    def test_heartbeat_creation(self):
        """Test creating a heartbeat."""
        hb = Heartbeat(
            rank=0,
            timestamp=time.perf_counter(),
            state=WorkerState.RUNNING,
            iteration=100
        )

        assert hb.rank == 0
        assert hb.state == WorkerState.RUNNING
        assert hb.iteration == 100

    def test_heartbeat_is_stale_false(self):
        """Test heartbeat is not stale when fresh."""
        hb = Heartbeat(
            rank=0,
            timestamp=time.perf_counter(),
            state=WorkerState.RUNNING
        )

        assert not hb.is_stale(30.0)

    def test_heartbeat_is_stale_true(self):
        """Test heartbeat is stale after timeout."""
        hb = Heartbeat(
            rank=0,
            timestamp=time.perf_counter() - 60.0,  # 60 seconds ago
            state=WorkerState.RUNNING
        )

        assert hb.is_stale(30.0)

    def test_heartbeat_default_values(self):
        """Test heartbeat default values."""
        hb = Heartbeat(
            rank=1,
            timestamp=time.perf_counter(),
            state=WorkerState.RUNNING
        )

        assert hb.iteration == 0
        assert hb.memory_used_mb == 0.0


# =============================================================================
# FailureEvent Tests
# =============================================================================

class TestFailureEvent:
    """Tests for FailureEvent dataclass."""

    def test_failure_event_creation(self):
        """Test creating a failure event."""
        event = FailureEvent(
            failed_rank=2,
            detected_at=time.perf_counter(),
            last_heartbeat=None,
            reason="Connection timeout"
        )

        assert event.failed_rank == 2
        assert event.reason == "Connection timeout"
        assert event.recoverable == True  # Default

    def test_failure_event_with_heartbeat(self):
        """Test failure event with last heartbeat."""
        hb = Heartbeat(
            rank=2,
            timestamp=time.perf_counter() - 100,
            state=WorkerState.RUNNING,
            iteration=50
        )

        event = FailureEvent(
            failed_rank=2,
            detected_at=time.perf_counter(),
            last_heartbeat=hb,
            reason="Missed heartbeats",
            recoverable=True
        )

        assert event.last_heartbeat == hb
        assert event.last_heartbeat.iteration == 50

    def test_failure_event_not_recoverable(self):
        """Test non-recoverable failure event."""
        event = FailureEvent(
            failed_rank=1,
            detected_at=time.perf_counter(),
            last_heartbeat=None,
            reason="Fatal error",
            recoverable=False
        )

        assert not event.recoverable


# =============================================================================
# HealthMonitor Tests
# =============================================================================

class TestHealthMonitor:
    """Tests for HealthMonitor class."""

    def test_health_monitor_creation(self, process_group):
        """Test creating a health monitor."""
        monitor = HealthMonitor(
            process_group=process_group,
            heartbeat_interval=1.0,
            timeout_seconds=5.0
        )

        assert monitor.heartbeat_interval == 1.0
        assert monitor.timeout_seconds == 5.0
        assert monitor.world_size == 4

    def test_health_monitor_start_stop(self, process_group):
        """Test starting and stopping health monitor."""
        monitor = HealthMonitor(
            process_group=process_group,
            heartbeat_interval=0.1
        )

        monitor.start()
        assert monitor._running

        monitor.stop()
        assert not monitor._running

    def test_all_healthy_initially(self, process_group):
        """Test all workers healthy initially."""
        monitor = HealthMonitor(process_group=process_group)

        # No workers have failed yet
        assert monitor.all_healthy
        assert len(monitor.get_failed_workers()) == 0

    def test_receive_heartbeat(self, process_group):
        """Test receiving a heartbeat."""
        monitor = HealthMonitor(process_group=process_group)

        hb = Heartbeat(
            rank=1,
            timestamp=time.perf_counter(),
            state=WorkerState.RUNNING,
            iteration=10
        )

        monitor.receive_heartbeat(hb)

        assert monitor._heartbeats.get(1) == hb

    def test_failure_callback(self, process_group):
        """Test failure callback is invoked."""
        monitor = HealthMonitor(
            process_group=process_group,
            max_missed_heartbeats=1
        )

        failure_events = []

        def on_failure(event):
            failure_events.append(event)

        monitor.add_failure_callback(on_failure)

        # Simulate failure by directly calling handler
        monitor._handle_failure(2, None)

        assert len(failure_events) == 1
        assert failure_events[0].failed_rank == 2

    def test_recovery_callback(self, process_group):
        """Test recovery callback is invoked."""
        monitor = HealthMonitor(process_group=process_group)

        recovered_ranks = []

        def on_recovery(rank):
            recovered_ranks.append(rank)

        monitor.add_recovery_callback(on_recovery)

        # Simulate failure then recovery
        monitor._handle_failure(2, None)
        assert 2 in monitor._failed_workers

        # Receive heartbeat from failed worker
        hb = Heartbeat(
            rank=2,
            timestamp=time.perf_counter(),
            state=WorkerState.RUNNING
        )
        monitor.receive_heartbeat(hb)

        assert 2 in recovered_ranks
        assert 2 not in monitor._failed_workers

    def test_get_healthy_workers(self, process_group):
        """Test getting list of healthy workers."""
        monitor = HealthMonitor(process_group=process_group)

        # All healthy initially
        healthy = monitor.get_healthy_workers()
        assert len(healthy) == 4
        assert 0 in healthy and 1 in healthy

        # Simulate failure
        monitor._handle_failure(2, None)

        healthy = monitor.get_healthy_workers()
        assert 2 not in healthy
        assert len(healthy) == 3

    def test_is_healthy(self, process_group):
        """Test checking if specific worker is healthy."""
        monitor = HealthMonitor(process_group=process_group)

        assert monitor.is_healthy(1)

        monitor._handle_failure(1, None)

        assert not monitor.is_healthy(1)

    def test_mark_iteration(self, process_group):
        """Test marking current iteration."""
        monitor = HealthMonitor(process_group=process_group)

        monitor.mark_iteration(100)

        assert monitor._current_iteration == 100


# =============================================================================
# Checkpointer Tests
# =============================================================================

class TestCheckpointer:
    """Tests for Checkpointer class."""

    def test_checkpointer_creation(self, temp_checkpoint_dir):
        """Test creating a checkpointer."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            save_interval=100,
            keep_last_n=5
        )

        assert checkpointer.save_interval == 100
        assert checkpointer.keep_last_n == 5
        assert os.path.exists(temp_checkpoint_dir)

        checkpointer.shutdown()

    def test_should_save(self, temp_checkpoint_dir):
        """Test checkpoint save interval."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            save_interval=100
        )

        assert not checkpointer.should_save(0)
        assert not checkpointer.should_save(50)
        assert checkpointer.should_save(100)
        assert checkpointer.should_save(200)
        assert not checkpointer.should_save(150)

        checkpointer.shutdown()

    def test_save_and_load(self, temp_checkpoint_dir):
        """Test saving and loading checkpoint."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            save_interval=100,
            async_save=False  # Sync for testing
        )

        state = {
            "model": {"weight": np.ones((10, 10))},
            "optimizer": {"lr": 0.01},
            "iteration": 100
        }

        # Force save
        filepath = checkpointer.save(state, iteration=100, force=True)

        assert filepath is not None
        assert os.path.exists(filepath)

        # Load
        loaded = checkpointer.load(filepath)

        assert loaded is not None
        assert loaded["iteration"] == 100
        np.testing.assert_array_equal(loaded["model"]["weight"], np.ones((10, 10)))

        checkpointer.shutdown()

    def test_load_latest(self, temp_checkpoint_dir):
        """Test loading latest checkpoint."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            async_save=False
        )

        # Save multiple checkpoints
        for i in [100, 200, 300]:
            checkpointer.save({"iteration": i}, iteration=i, force=True)

        # Load latest should get iteration 300
        loaded = checkpointer.load()

        assert loaded is not None
        assert loaded["iteration"] == 300

        checkpointer.shutdown()

    def test_cleanup_old_checkpoints(self, temp_checkpoint_dir):
        """Test cleanup of old checkpoints."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            keep_last_n=2,
            async_save=False
        )

        # Save 5 checkpoints
        for i in [100, 200, 300, 400, 500]:
            checkpointer.save({"iteration": i}, iteration=i, force=True)

        # Should only keep last 2
        checkpoints = [f for f in os.listdir(temp_checkpoint_dir)
                       if f.startswith("checkpoint_") and f.endswith(".pt")]

        assert len(checkpoints) == 2

        checkpointer.shutdown()

    def test_latest_pointer(self, temp_checkpoint_dir):
        """Test latest checkpoint pointer file."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            async_save=False
        )

        checkpointer.save({"iteration": 100}, iteration=100, force=True)

        latest_path = os.path.join(temp_checkpoint_dir, "latest.txt")
        assert os.path.exists(latest_path)

        with open(latest_path, 'r') as f:
            content = f.read().strip()
        assert "checkpoint_00000100.pt" == content

        checkpointer.shutdown()

    def test_load_nonexistent(self, temp_checkpoint_dir):
        """Test loading non-existent checkpoint."""
        checkpointer = Checkpointer(save_dir=temp_checkpoint_dir)

        loaded = checkpointer.load("/nonexistent/path.pt")
        assert loaded is None

        checkpointer.shutdown()

    def test_load_external_path_rejected(self, temp_checkpoint_dir):
        """An existing checkpoint outside save_dir is refused by default.

        Guards the pickle deserialization: load() must not unpickle files
        outside the trusted save_dir unless allow_external=True.
        """
        checkpointer = Checkpointer(save_dir=temp_checkpoint_dir, async_save=False)

        # Write a real pickle file in a *different* directory.
        outside_dir = tempfile.mkdtemp()
        try:
            outside_path = os.path.join(outside_dir, "evil.pt")
            import pickle
            with open(outside_path, "wb") as f:
                pickle.dump({"iteration": 7}, f)

            with pytest.raises(ValueError, match="outside the trusted save_dir"):
                checkpointer.load(outside_path)

            # Explicit opt-in bypasses the guard for trusted paths.
            loaded = checkpointer.load(outside_path, allow_external=True)
            assert loaded is not None
            assert loaded["iteration"] == 7
        finally:
            shutil.rmtree(outside_dir, ignore_errors=True)
            checkpointer.shutdown()

    def test_load_from_save_dir_allowed(self, temp_checkpoint_dir):
        """A checkpoint written into save_dir loads without allow_external."""
        checkpointer = Checkpointer(save_dir=temp_checkpoint_dir, async_save=False)

        filepath = checkpointer.save({"iteration": 42}, iteration=42, force=True)
        # Path lives inside save_dir, so the default guard permits it.
        loaded = checkpointer.load(filepath)
        assert loaded is not None
        assert loaded["iteration"] == 42

        checkpointer.shutdown()

    def test_async_save(self, temp_checkpoint_dir):
        """Test async checkpoint saving."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            async_save=True
        )

        filepath = checkpointer.save({"iteration": 100}, iteration=100, force=True)

        # Wait for async save
        checkpointer.wait_pending()

        assert filepath is not None
        assert os.path.exists(filepath)

        checkpointer.shutdown()


# =============================================================================
# RecoveryManager Tests
# =============================================================================

class TestRecoveryManager:
    """Tests for RecoveryManager class."""

    def test_recovery_manager_creation(self, temp_checkpoint_dir):
        """Test creating a recovery manager."""
        checkpointer = Checkpointer(save_dir=temp_checkpoint_dir)
        recovery = RecoveryManager(checkpointer=checkpointer)

        assert recovery.checkpointer == checkpointer
        assert recovery.recovery_count == 0
        assert not recovery.is_recovering

        checkpointer.shutdown()

    def test_save_state(self, temp_checkpoint_dir, mock_module):
        """Test saving training state."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            save_interval=100,
            async_save=False
        )
        recovery = RecoveryManager(checkpointer=checkpointer)

        optimizer = MockOptimizer()

        # Save at checkpoint interval
        filepath = recovery.save_state(
            mock_module,
            optimizer,
            iteration=100
        )

        assert filepath is not None

        # Load and verify
        loaded = checkpointer.load(filepath)
        assert "model" in loaded
        assert "optimizer" in loaded
        assert loaded["iteration"] == 100

        checkpointer.shutdown()

    def test_restore_state(self, temp_checkpoint_dir, mock_module):
        """Test restoring training state."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            save_interval=100,  # Match iteration
            async_save=False
        )
        recovery = RecoveryManager(checkpointer=checkpointer)

        optimizer = MockOptimizer()

        # Save state at interval
        recovery.save_state(mock_module, optimizer, iteration=100)
        checkpointer.wait_pending()

        # Modify model state
        mock_module._state = {"changed": True}

        # Restore
        iteration = recovery.restore_state(mock_module, optimizer)

        assert iteration == 100
        assert recovery.recovery_count == 1
        assert "layer1" in mock_module._state  # Original state restored

        checkpointer.shutdown()

    def test_restore_no_checkpoint(self, temp_checkpoint_dir, mock_module):
        """Test restore when no checkpoint exists."""
        checkpointer = Checkpointer(save_dir=temp_checkpoint_dir)
        recovery = RecoveryManager(checkpointer=checkpointer)

        # Should return None when no checkpoint
        result = recovery.restore_state(mock_module)
        assert result is None

        checkpointer.shutdown()

    def test_recovery_count_increments(self, temp_checkpoint_dir, mock_module):
        """Test recovery count increments on each restore."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            save_interval=100,
            async_save=False
        )
        recovery = RecoveryManager(checkpointer=checkpointer)

        # Save at checkpoint intervals
        for i in [100, 200, 300]:
            recovery.save_state(mock_module, None, iteration=i)
            checkpointer.wait_pending()

        # Restore 3 times
        for _ in range(3):
            recovery.restore_state(mock_module)

        assert recovery.recovery_count == 3

        checkpointer.shutdown()

    def test_extra_state(self, temp_checkpoint_dir, mock_module):
        """Test saving extra state."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            save_interval=100,
            async_save=False
        )
        recovery = RecoveryManager(checkpointer=checkpointer)

        extra = {"custom_value": 42, "data": [1, 2, 3]}

        filepath = recovery.save_state(
            mock_module,
            None,
            iteration=100,
            extra_state=extra
        )

        assert filepath is not None
        loaded = checkpointer.load(filepath)
        assert "extra" in loaded
        assert loaded["extra"]["custom_value"] == 42

        checkpointer.shutdown()


# =============================================================================
# FaultTolerantTrainer Tests
# =============================================================================

class TestFaultTolerantTrainer:
    """Tests for FaultTolerantTrainer class."""

    def test_trainer_creation(self, temp_checkpoint_dir, mock_module, process_group):
        """Test creating a fault-tolerant trainer."""
        trainer = FaultTolerantTrainer(
            model=mock_module,
            process_group=process_group,
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_interval=100
        )

        assert trainer.ddp is not None
        assert trainer.health_monitor is not None
        assert trainer.checkpointer is not None
        assert trainer.recovery_manager is not None

        trainer.stop()

    def test_trainer_without_process_group(self, temp_checkpoint_dir, mock_module):
        """Test trainer without process group (single worker)."""
        trainer = FaultTolerantTrainer(
            model=mock_module,
            process_group=None,
            checkpoint_dir=temp_checkpoint_dir
        )

        assert trainer.health_monitor is None
        assert trainer.is_healthy  # Always healthy without monitoring

        trainer.stop()

    def test_trainer_context_manager(self, temp_checkpoint_dir, mock_module):
        """Test trainer as context manager."""
        with FaultTolerantTrainer(
            model=mock_module,
            checkpoint_dir=temp_checkpoint_dir
        ) as trainer:
            assert trainer._is_training

        assert not trainer._is_training

    def test_trainer_forward(self, temp_checkpoint_dir, mock_module):
        """Test forward pass through trainer."""
        trainer = FaultTolerantTrainer(
            model=mock_module,
            checkpoint_dir=temp_checkpoint_dir
        )
        trainer.start()

        input_data = np.random.randn(32, 100).astype(np.float32)
        output = trainer.forward(input_data)

        np.testing.assert_array_equal(output, input_data)

        trainer.stop()

    def test_trainer_step(self, temp_checkpoint_dir, mock_module):
        """Test training step."""
        trainer = FaultTolerantTrainer(
            model=mock_module,
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_interval=10
        )
        trainer.start()

        for i in range(15):
            trainer.step(iteration=i)

        assert trainer._iteration == 14

        trainer.stop()

    def test_trainer_set_optimizer(self, temp_checkpoint_dir, mock_module):
        """Test setting optimizer."""
        trainer = FaultTolerantTrainer(
            model=mock_module,
            checkpoint_dir=temp_checkpoint_dir
        )

        optimizer = MockOptimizer()
        trainer.set_optimizer(optimizer)

        assert trainer._optimizer == optimizer

        trainer.stop()

    def test_trainer_resume(self, temp_checkpoint_dir, mock_module):
        """Test resuming training."""
        # First training session
        trainer1 = FaultTolerantTrainer(
            model=mock_module,
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_interval=10
        )
        trainer1.checkpointer.async_save = False
        trainer1.start()

        for i in range(25):
            trainer1.step(iteration=i)

        trainer1.stop()

        # Second training session - resume
        mock_module2 = MockModule()
        trainer2 = FaultTolerantTrainer(
            model=mock_module2,
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_interval=10
        )
        trainer2.checkpointer.async_save = False

        iteration = trainer2.resume()

        # Should resume from checkpoint at iteration 20 (last checkpoint at interval)
        assert iteration == 20

        trainer2.stop()

    def test_trainer_is_healthy(self, temp_checkpoint_dir, mock_module, process_group):
        """Test health status check."""
        trainer = FaultTolerantTrainer(
            model=mock_module,
            process_group=process_group,
            checkpoint_dir=temp_checkpoint_dir
        )
        trainer.start()

        assert trainer.is_healthy

        trainer.stop()


# =============================================================================
# Integration Tests
# =============================================================================

class TestPhase5Integration:
    """Integration tests for Phase 5 features."""

    def test_full_fault_tolerant_workflow(self, temp_checkpoint_dir, mock_module):
        """Test complete fault-tolerant training workflow (no process group for speed)."""
        # Setup without process group (avoids health monitor threading)
        trainer = FaultTolerantTrainer(
            model=mock_module,
            process_group=None,
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_interval=5
        )
        trainer.checkpointer.async_save = False

        optimizer = MockOptimizer()
        trainer.set_optimizer(optimizer)

        # Train for some iterations
        trainer.start()

        for i in range(12):
            output = trainer.forward(np.random.randn(8, 100).astype(np.float32))
            trainer.step(iteration=i)

        trainer.stop()

        # Verify checkpoints were saved
        checkpoints = [f for f in os.listdir(temp_checkpoint_dir)
                       if f.startswith("checkpoint_")]
        assert len(checkpoints) > 0

    def test_health_monitor_with_recovery(self, temp_checkpoint_dir, process_group):
        """Test health monitor integration with recovery."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            async_save=False
        )

        monitor = HealthMonitor(
            process_group=process_group,
            heartbeat_interval=0.1,
            timeout_seconds=1.0,
            max_missed_heartbeats=2
        )

        recovery = RecoveryManager(
            checkpointer=checkpointer,
            health_monitor=monitor,
            process_group=process_group,
            auto_recover=True
        )

        failure_detected = []

        def on_failure(event):
            failure_detected.append(event)

        monitor.add_failure_callback(on_failure)

        # Simulate failure (don't start monitor to avoid threading issues)
        monitor._handle_failure(1, None)

        assert len(failure_detected) == 1
        assert failure_detected[0].failed_rank == 1

        checkpointer.shutdown()

    def test_checkpoint_recovery_cycle(self, temp_checkpoint_dir, mock_module):
        """Test multiple checkpoint-recovery cycles."""
        checkpointer = Checkpointer(
            save_dir=temp_checkpoint_dir,
            save_interval=10,
            keep_last_n=3,
            async_save=False
        )
        recovery = RecoveryManager(checkpointer=checkpointer)

        # Multiple training cycles with recovery
        for cycle in range(3):
            # Save at checkpoints (10, 20, 110, 120, 210, 220)
            for i in [10, 20]:
                iteration = cycle * 100 + i
                recovery.save_state(mock_module, None, iteration=iteration)

            # Simulate failure and recovery
            if cycle < 2:
                mock_module._state = {"corrupted": True}
                recovery.restore_state(mock_module)

        # Should have recovered twice
        assert recovery.recovery_count == 2

        checkpointer.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
