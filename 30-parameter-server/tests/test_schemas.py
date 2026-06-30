"""Tests for schema definitions."""

import pytest
import numpy as np
import time

from paramserver.schemas import (
    ShardInfo,
    GradientUpdate,
    WorkerInfo,
    WorkerStatus,
    ServerStatus,
    ParameterMetadata,
    PullRequest,
    PullResponse,
    PushRequest,
    CheckpointMetadata,
)


class TestShardInfo:
    """Tests for ShardInfo dataclass."""

    def test_create_shard_info(self):
        """Test basic ShardInfo creation."""
        shard = ShardInfo(
            shard_id=0,
            server_address="server-0",
            param_names=["w1", "w2"],
            total_params=100,
        )
        assert shard.shard_id == 0
        assert shard.server_address == "server-0"
        assert len(shard.param_names) == 2
        assert shard.total_params == 100

    def test_memory_bytes_calculation(self):
        """Test automatic memory bytes calculation."""
        shard = ShardInfo(
            shard_id=0,
            server_address="server-0",
            total_params=1000,
        )
        # 1000 params * 4 bytes (float32) = 4000 bytes
        assert shard.memory_bytes == 4000

    def test_memory_bytes_explicit(self):
        """Test explicit memory bytes."""
        shard = ShardInfo(
            shard_id=0,
            server_address="server-0",
            total_params=1000,
            memory_bytes=8000,  # Explicit value
        )
        assert shard.memory_bytes == 8000

    def test_default_values(self):
        """Test default values."""
        shard = ShardInfo(shard_id=0, server_address="s0")
        assert shard.param_ranges == []
        assert shard.param_names == []
        assert shard.total_params == 0


class TestGradientUpdate:
    """Tests for GradientUpdate dataclass."""

    def test_create_gradient_update(self):
        """Test GradientUpdate creation."""
        grad = np.array([1.0, 2.0, 3.0])
        update = GradientUpdate(
            worker_id=1,
            param_name="layer1.weight",
            gradient=grad,
            clock=5,
        )
        assert update.worker_id == 1
        assert update.param_name == "layer1.weight"
        np.testing.assert_array_equal(update.gradient, grad)
        assert update.clock == 5
        assert update.timestamp > 0

    def test_compressed_update(self):
        """Test compressed gradient update."""
        update = GradientUpdate(
            worker_id=0,
            param_name="w",
            gradient=np.array([1.0]),
            clock=0,
            compressed=True,
            compression_metadata={"type": "topk", "k": 10},
        )
        assert update.compressed is True
        assert update.compression_metadata["type"] == "topk"


class TestWorkerInfo:
    """Tests for WorkerInfo dataclass."""

    def test_create_worker_info(self):
        """Test WorkerInfo creation."""
        info = WorkerInfo(worker_id=5, address="worker-5:8080")
        assert info.worker_id == 5
        assert info.address == "worker-5:8080"
        assert info.status == WorkerStatus.IDLE
        assert info.clock == 0

    def test_update_heartbeat(self):
        """Test heartbeat update."""
        info = WorkerInfo(worker_id=0)
        old_heartbeat = info.last_heartbeat
        time.sleep(0.01)
        info.update_heartbeat()
        assert info.last_heartbeat > old_heartbeat

    def test_is_stale(self):
        """Test staleness detection."""
        info = WorkerInfo(worker_id=0)
        info.last_heartbeat = time.time() - 60  # 60 seconds ago
        assert info.is_stale(timeout=30.0) is True
        assert info.is_stale(timeout=120.0) is False

    def test_status_transitions(self):
        """Test status changes."""
        info = WorkerInfo(worker_id=0)
        assert info.status == WorkerStatus.IDLE

        info.status = WorkerStatus.TRAINING
        assert info.status == WorkerStatus.TRAINING

        info.status = WorkerStatus.FAILED
        assert info.status == WorkerStatus.FAILED


class TestParameterMetadata:
    """Tests for ParameterMetadata dataclass."""

    def test_create_metadata(self):
        """Test metadata creation."""
        meta = ParameterMetadata(
            name="layer1.weight",
            shape=(100, 64),
            dtype=np.dtype(np.float32),
        )
        assert meta.name == "layer1.weight"
        assert meta.shape == (100, 64)
        assert meta.version == 0

    def test_size_calculation(self):
        """Test size property."""
        meta = ParameterMetadata(
            name="w",
            shape=(10, 20, 30),
        )
        assert meta.size == 10 * 20 * 30

    def test_memory_bytes(self):
        """Test memory bytes calculation."""
        meta = ParameterMetadata(
            name="w",
            shape=(100,),
            dtype=np.dtype(np.float32),
        )
        assert meta.memory_bytes == 100 * 4  # float32 is 4 bytes

        meta_f64 = ParameterMetadata(
            name="w",
            shape=(100,),
            dtype=np.dtype(np.float64),
        )
        assert meta_f64.memory_bytes == 100 * 8  # float64 is 8 bytes


class TestPullRequest:
    """Tests for PullRequest dataclass."""

    def test_create_pull_request(self):
        """Test PullRequest creation."""
        req = PullRequest(
            worker_id=0,
            param_names=["w1", "w2", "b"],
        )
        assert req.worker_id == 0
        assert len(req.param_names) == 3
        assert req.include_versions is True


class TestPullResponse:
    """Tests for PullResponse dataclass."""

    def test_create_pull_response(self):
        """Test PullResponse creation."""
        params = {
            "w1": np.array([1.0, 2.0]),
            "w2": np.array([3.0]),
        }
        resp = PullResponse(
            params=params,
            versions={"w1": 5, "w2": 3},
            server_id=0,
        )
        assert len(resp.params) == 2
        assert resp.versions["w1"] == 5


class TestPushRequest:
    """Tests for PushRequest dataclass."""

    def test_create_push_request(self):
        """Test PushRequest creation."""
        grads = {
            "w1": np.array([0.1, 0.2]),
        }
        req = PushRequest(
            worker_id=1,
            gradients=grads,
            clock=10,
        )
        assert req.worker_id == 1
        assert req.clock == 10
        assert req.compressed is False


class TestCheckpointMetadata:
    """Tests for CheckpointMetadata dataclass."""

    def test_create_checkpoint_metadata(self):
        """Test CheckpointMetadata creation."""
        meta = CheckpointMetadata(
            checkpoint_id="ckpt_001",
            epoch=5,
            global_step=1000,
            worker_clocks={0: 500, 1: 498},
        )
        assert meta.checkpoint_id == "ckpt_001"
        assert meta.epoch == 5
        assert meta.global_step == 1000
        assert meta.worker_clocks[0] == 500


class TestEnums:
    """Tests for enum classes."""

    def test_worker_status_values(self):
        """Test WorkerStatus enum values."""
        assert WorkerStatus.IDLE.value == "idle"
        assert WorkerStatus.TRAINING.value == "training"
        assert WorkerStatus.PULLING.value == "pulling"
        assert WorkerStatus.PUSHING.value == "pushing"
        assert WorkerStatus.FAILED.value == "failed"

    def test_server_status_values(self):
        """Test ServerStatus enum values."""
        assert ServerStatus.READY.value == "ready"
        assert ServerStatus.BUSY.value == "busy"
        assert ServerStatus.CHECKPOINTING.value == "checkpointing"
        assert ServerStatus.RECOVERING.value == "recovering"
        assert ServerStatus.FAILED.value == "failed"
