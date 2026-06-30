"""Tests for checkpoint system."""

import pytest
import asyncio
import os
import tempfile
import numpy as np

from paramserver.fault_tolerance.checkpoint import Checkpoint, CheckpointManager


class TestCheckpoint:
    """Tests for Checkpoint dataclass."""

    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        params = {"w1": np.array([1.0, 2.0])}
        checkpoint = Checkpoint(
            checkpoint_id="test_001",
            epoch=5,
            global_step=1000,
            params=params,
        )

        assert checkpoint.checkpoint_id == "test_001"
        assert checkpoint.epoch == 5
        assert checkpoint.global_step == 1000
        assert checkpoint.timestamp > 0

    def test_checkpoint_with_optimizer_state(self):
        """Test checkpoint with optimizer state."""
        params = {"w1": np.array([1.0])}
        opt_state = {
            0: {"momentum": {"w1": np.array([0.1])}},
            1: {"momentum": {"w2": np.array([0.2])}},
        }

        checkpoint = Checkpoint(
            checkpoint_id="test",
            epoch=0,
            global_step=0,
            params=params,
            optimizer_state=opt_state,
        )

        assert len(checkpoint.optimizer_state) == 2

    def test_checkpoint_with_worker_clocks(self):
        """Test checkpoint with worker clocks."""
        checkpoint = Checkpoint(
            checkpoint_id="test",
            epoch=0,
            global_step=0,
            params={},
            worker_clocks={0: 100, 1: 98, 2: 102},
        )

        assert checkpoint.worker_clocks[0] == 100
        assert checkpoint.worker_clocks[1] == 98


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def manager(self, temp_dir):
        """Create checkpoint manager."""
        return CheckpointManager(
            storage_path=temp_dir,
            checkpoint_interval=100,
            max_checkpoints=3,
        )

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, manager):
        """Test saving a checkpoint."""
        params = {
            "w1": np.random.randn(10, 5).astype(np.float32),
            "w2": np.random.randn(5).astype(np.float32),
        }

        path = await manager.save_checkpoint(
            params=params,
            epoch=1,
            global_step=100,
        )

        assert os.path.exists(path)
        assert path.endswith(".pkl")

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, manager):
        """Test loading a checkpoint."""
        params = {
            "w1": np.array([1.0, 2.0, 3.0]),
        }

        path = await manager.save_checkpoint(
            params=params,
            epoch=1,
            global_step=100,
        )

        loaded = await manager.load_checkpoint(path)

        assert loaded is not None
        assert loaded.epoch == 1
        assert loaded.global_step == 100
        np.testing.assert_array_equal(loaded.params["w1"], params["w1"])

    @pytest.mark.asyncio
    async def test_load_latest_checkpoint(self, manager):
        """Test loading latest checkpoint."""
        # Save multiple checkpoints
        for step in [100, 200, 300]:
            await manager.save_checkpoint(
                params={"w": np.array([float(step)])},
                epoch=step // 100,
                global_step=step,
            )

        # Load latest
        loaded = await manager.load_checkpoint()

        assert loaded is not None
        assert loaded.global_step == 300

    @pytest.mark.asyncio
    async def test_checkpoint_rotation(self, manager):
        """Test that old checkpoints are rotated out."""
        # Save more than max_checkpoints
        for step in range(0, 500, 100):
            await manager.save_checkpoint(
                params={"w": np.array([1.0])},
                epoch=0,
                global_step=step,
            )

        # Should only have max_checkpoints
        checkpoints = manager.get_all_checkpoints()
        assert len(checkpoints) <= manager.max_checkpoints

    @pytest.mark.asyncio
    async def test_should_checkpoint(self, manager):
        """Test checkpoint interval logic."""
        assert manager.should_checkpoint(0) is False
        assert manager.should_checkpoint(50) is False
        assert manager.should_checkpoint(100) is True
        assert manager.should_checkpoint(200) is True
        assert manager.should_checkpoint(150) is False

    @pytest.mark.asyncio
    async def test_save_with_metadata(self, manager):
        """Test saving checkpoint with metadata."""
        path = await manager.save_checkpoint(
            params={"w": np.array([1.0])},
            epoch=1,
            global_step=100,
            metadata={"loss": 0.5, "accuracy": 0.9},
        )

        loaded = await manager.load_checkpoint(path)
        assert loaded.metadata["loss"] == 0.5
        assert loaded.metadata["accuracy"] == 0.9

    @pytest.mark.asyncio
    async def test_get_checkpoint_info(self, manager):
        """Test getting checkpoint info without loading."""
        path = await manager.save_checkpoint(
            params={"w": np.array([1.0])},
            epoch=2,
            global_step=200,
            metadata={"test": True},
        )

        info = manager.get_checkpoint_info(path)

        assert info is not None
        assert info["epoch"] == 2
        assert info["global_step"] == 200
        assert info["metadata"]["test"] is True

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, manager):
        """Test loading nonexistent checkpoint."""
        loaded = await manager.load_checkpoint("/nonexistent/path.pkl")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_cleanup(self, manager):
        """Test cleanup removes all checkpoints."""
        # Save some checkpoints
        for step in [100, 200]:
            await manager.save_checkpoint(
                params={"w": np.array([1.0])},
                epoch=0,
                global_step=step,
            )

        assert len(manager.get_all_checkpoints()) == 2

        await manager.cleanup()

        assert len(manager.get_all_checkpoints()) == 0

    @pytest.mark.asyncio
    async def test_save_with_optimizer_state(self, manager):
        """Test saving/loading optimizer state."""
        opt_state = {
            0: {"velocity": {"w": np.array([0.1, 0.2])}},
        }

        path = await manager.save_checkpoint(
            params={"w": np.array([1.0, 2.0])},
            epoch=1,
            global_step=100,
            optimizer_state=opt_state,
        )

        loaded = await manager.load_checkpoint(path)
        np.testing.assert_array_equal(
            loaded.optimizer_state[0]["velocity"]["w"],
            opt_state[0]["velocity"]["w"],
        )

    @pytest.mark.asyncio
    async def test_save_with_worker_clocks(self, manager):
        """Test saving/loading worker clocks."""
        clocks = {0: 100, 1: 98, 2: 102}

        path = await manager.save_checkpoint(
            params={"w": np.array([1.0])},
            epoch=1,
            global_step=100,
            worker_clocks=clocks,
        )

        loaded = await manager.load_checkpoint(path)
        assert loaded.worker_clocks == clocks


class TestCheckpointConcurrency:
    """Tests for concurrent checkpoint operations."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, temp_dir):
        """Test concurrent checkpoint saves."""
        manager = CheckpointManager(
            storage_path=temp_dir,
            max_checkpoints=10,
        )

        async def save_checkpoint(step):
            return await manager.save_checkpoint(
                params={"w": np.array([float(step)])},
                epoch=0,
                global_step=step,
            )

        tasks = [save_checkpoint(i * 100) for i in range(5)]
        paths = await asyncio.gather(*tasks)

        assert len(paths) == 5
        assert all(os.path.exists(p) for p in paths)
