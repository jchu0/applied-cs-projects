"""Tests for core data models."""

import pytest
from datetime import datetime, timedelta

from ml_orchestrator.core.models import (
    Checkpoint,
    CheckpointConfig,
    CheckpointPolicy,
    CheckpointStatus,
    DistributedConfig,
    Experiment,
    JobConfig,
    JobPriority,
    JobStatus,
    MetricType,
    MetricValue,
    ResourceQuota,
    ResourceRequest,
    StorageBackend,
    TrainingJob,
    WorkerInfo,
    WorkerStatus,
)


class TestResourceRequest:
    """Tests for ResourceRequest model."""

    def test_default_values(self):
        """Test default resource values."""
        resources = ResourceRequest()
        assert resources.cpus == 1
        assert resources.memory_gb == 4.0
        assert resources.gpus == 0
        assert resources.storage_gb == 10.0

    def test_custom_values(self):
        """Test custom resource values."""
        resources = ResourceRequest(
            cpus=8, memory_gb=32.0, gpus=2, gpu_type="A100"
        )
        assert resources.cpus == 8
        assert resources.memory_gb == 32.0
        assert resources.gpus == 2
        assert resources.gpu_type == "A100"

    def test_fits_in_success(self):
        """Test fits_in returns True when request fits."""
        request = ResourceRequest(cpus=4, memory_gb=16.0, gpus=1)
        available = ResourceRequest(cpus=8, memory_gb=32.0, gpus=2)
        assert request.fits_in(available) is True

    def test_fits_in_failure_cpu(self):
        """Test fits_in returns False when CPUs don't fit."""
        request = ResourceRequest(cpus=16, memory_gb=16.0, gpus=1)
        available = ResourceRequest(cpus=8, memory_gb=32.0, gpus=2)
        assert request.fits_in(available) is False

    def test_fits_in_failure_gpu(self):
        """Test fits_in returns False when GPUs don't fit."""
        request = ResourceRequest(cpus=4, memory_gb=16.0, gpus=4)
        available = ResourceRequest(cpus=8, memory_gb=32.0, gpus=2)
        assert request.fits_in(available) is False

    def test_subtract(self):
        """Test resource subtraction."""
        total = ResourceRequest(cpus=16, memory_gb=64.0, gpus=4, storage_gb=100.0)
        used = ResourceRequest(cpus=4, memory_gb=16.0, gpus=1, storage_gb=20.0)
        remaining = total.subtract(used)
        assert remaining.cpus == 12
        assert remaining.memory_gb == 48.0
        assert remaining.gpus == 3
        assert remaining.storage_gb == 80.0

    def test_add(self):
        """Test resource addition."""
        r1 = ResourceRequest(cpus=8, memory_gb=32.0, gpus=2)
        r2 = ResourceRequest(cpus=4, memory_gb=16.0, gpus=1)
        total = r1.add(r2)
        assert total.cpus == 12
        assert total.memory_gb == 48.0
        assert total.gpus == 3


class TestJobConfig:
    """Tests for JobConfig model."""

    def test_minimal_config(self):
        """Test minimal job configuration."""
        config = JobConfig(script_path="/train.py")
        assert config.script_path == "/train.py"
        assert config.timeout_hours == 24.0  # Default timeout

    def test_distributed_config(self):
        """Test distributed training configuration."""
        config = JobConfig(
            script_path="/train.py",
            distributed=DistributedConfig(
                enabled=True, world_size=4, num_nodes=2
            ),
        )
        assert config.distributed.enabled is True
        assert config.distributed.world_size == 4

    def test_checkpoint_config(self):
        """Test checkpoint configuration."""
        config = JobConfig(
            script_path="/train.py",
            checkpoint=CheckpointConfig(
                enabled=True,
                policy=CheckpointPolicy.BEST_METRIC,
                metric_to_track="val_loss",
            ),
        )
        assert config.checkpoint.enabled is True
        assert config.checkpoint.policy == CheckpointPolicy.BEST_METRIC

    def test_validation_epochs_and_steps(self):
        """Test validation of epochs/steps mutual exclusivity."""
        with pytest.raises(ValueError):
            JobConfig(script_path="/train.py", epochs=10, max_steps=1000)


class TestDistributedConfig:
    """Tests for DistributedConfig model."""

    def test_default_disabled(self):
        """Test distributed is disabled by default."""
        config = DistributedConfig()
        assert config.enabled is False
        assert config.world_size == 1

    def test_elastic_config(self):
        """Test elastic training configuration."""
        config = DistributedConfig(
            enabled=True,
            world_size=4,
            elastic=True,
            min_nodes=2,
            max_nodes=8,
        )
        assert config.elastic is True
        assert config.min_nodes == 2
        assert config.max_nodes == 8

    def test_elastic_validation(self):
        """Test elastic config validation."""
        config = DistributedConfig(
            enabled=True, world_size=4, elastic=True
        )
        # min/max nodes should default properly
        assert config.min_nodes == 1
        assert config.max_nodes == 4


class TestTrainingJob:
    """Tests for TrainingJob model."""

    def test_creation(self):
        """Test job creation with defaults."""
        config = JobConfig(script_path="/train.py")
        job = TrainingJob(name="test", user_id="user-1", config=config)

        assert job.name == "test"
        assert job.user_id == "user-1"
        assert job.status == JobStatus.PENDING
        assert job.priority == JobPriority.NORMAL
        assert job.current_epoch == 0
        assert job.current_step == 0

    def test_is_terminal(self):
        """Test terminal state detection."""
        config = JobConfig(script_path="/train.py")

        # Not terminal
        job = TrainingJob(name="test", user_id="user-1", config=config)
        assert job.is_terminal is False

        job.status = JobStatus.RUNNING
        assert job.is_terminal is False

        # Terminal states
        job.status = JobStatus.COMPLETED
        assert job.is_terminal is True

        job.status = JobStatus.FAILED
        assert job.is_terminal is True

        job.status = JobStatus.CANCELLED
        assert job.is_terminal is True

    def test_is_active(self):
        """Test active state detection."""
        config = JobConfig(script_path="/train.py")
        job = TrainingJob(name="test", user_id="user-1", config=config)

        job.status = JobStatus.RUNNING
        assert job.is_active is True

        job.status = JobStatus.CHECKPOINTING
        assert job.is_active is True

        job.status = JobStatus.QUEUED
        assert job.is_active is False

    def test_is_schedulable(self):
        """Test schedulable state detection."""
        config = JobConfig(script_path="/train.py")
        job = TrainingJob(name="test", user_id="user-1", config=config)

        job.status = JobStatus.QUEUED
        assert job.is_schedulable is True

        job.status = JobStatus.PREEMPTED
        assert job.is_schedulable is True

        job.status = JobStatus.RUNNING
        assert job.is_schedulable is False

    def test_add_metric(self):
        """Test adding metrics to job."""
        config = JobConfig(script_path="/train.py")
        job = TrainingJob(name="test", user_id="user-1", config=config)

        metric = MetricValue(name="loss", value=0.5, step=100)
        job.add_metric(metric)

        assert len(job.metrics) == 1
        assert job.best_metrics["loss"] == 0.5

        # Add better metric
        metric2 = MetricValue(name="loss", value=0.3, step=200)
        job.add_metric(metric2)

        assert len(job.metrics) == 2
        assert job.best_metrics["loss"] == 0.3  # Lower is better for loss

    def test_get_latest_checkpoint(self):
        """Test getting latest checkpoint."""
        config = JobConfig(script_path="/train.py")
        job = TrainingJob(name="test", user_id="user-1", config=config)

        # No checkpoints
        assert job.get_latest_checkpoint() is None

        # Add checkpoints
        ckpt1 = Checkpoint(
            job_id=job.id,
            epoch=1,
            step=100,
            path="/ckpt1",
            size_bytes=1000,
            status=CheckpointStatus.COMPLETED,
        )
        ckpt2 = Checkpoint(
            job_id=job.id,
            epoch=2,
            step=200,
            path="/ckpt2",
            size_bytes=1000,
            status=CheckpointStatus.COMPLETED,
        )
        job.checkpoints = [ckpt1, ckpt2]

        latest = job.get_latest_checkpoint()
        assert latest.epoch == 2
        assert latest.step == 200

    def test_queue_time(self):
        """Test queue time calculation."""
        config = JobConfig(script_path="/train.py")
        job = TrainingJob(name="test", user_id="user-1", config=config)

        assert job.queue_time is None

        job.queued_at = datetime.utcnow() - timedelta(minutes=5)
        queue_time = job.queue_time
        assert queue_time is not None
        assert queue_time.total_seconds() >= 5 * 60


class TestWorkerInfo:
    """Tests for WorkerInfo model."""

    def test_creation(self):
        """Test worker creation."""
        resources = ResourceRequest(cpus=16, memory_gb=64.0, gpus=4)
        worker = WorkerInfo(
            hostname="worker-1",
            ip_address="192.168.1.100",
            resources=resources,
        )

        assert worker.hostname == "worker-1"
        assert worker.status == WorkerStatus.INITIALIZING
        assert worker.resources.cpus == 16

    def test_available_resources(self):
        """Test available resources calculation."""
        resources = ResourceRequest(cpus=16, memory_gb=64.0, gpus=4)
        worker = WorkerInfo(
            hostname="worker-1",
            ip_address="192.168.1.100",
            resources=resources,
        )
        worker.allocated_resources = ResourceRequest(
            cpus=4, memory_gb=16.0, gpus=1
        )

        available = worker.available_resources
        assert available.cpus == 12
        assert available.memory_gb == 48.0
        assert available.gpus == 3

    def test_is_healthy(self):
        """Test health check based on heartbeat."""
        resources = ResourceRequest(cpus=16, memory_gb=64.0, gpus=4)
        worker = WorkerInfo(
            hostname="worker-1",
            ip_address="192.168.1.100",
            resources=resources,
            status=WorkerStatus.READY,
        )

        # Recent heartbeat
        worker.last_heartbeat = datetime.utcnow()
        assert worker.is_healthy(timeout_seconds=60) is True

        # Old heartbeat
        worker.last_heartbeat = datetime.utcnow() - timedelta(minutes=5)
        assert worker.is_healthy(timeout_seconds=60) is False

        # Unhealthy status
        worker.status = WorkerStatus.UNHEALTHY
        worker.last_heartbeat = datetime.utcnow()
        assert worker.is_healthy(timeout_seconds=60) is False

    def test_can_run_job(self):
        """Test job compatibility check."""
        resources = ResourceRequest(cpus=16, memory_gb=64.0, gpus=4)
        worker = WorkerInfo(
            hostname="worker-1",
            ip_address="192.168.1.100",
            resources=resources,
            status=WorkerStatus.READY,
        )

        # Small job
        small_job = ResourceRequest(cpus=4, memory_gb=16.0, gpus=1)
        assert worker.can_run_job(small_job) is True

        # Job that doesn't fit
        large_job = ResourceRequest(cpus=32, memory_gb=128.0, gpus=8)
        assert worker.can_run_job(large_job) is False

        # Worker not ready
        worker.status = WorkerStatus.DRAINING
        assert worker.can_run_job(small_job) is False


class TestResourceQuota:
    """Tests for ResourceQuota model."""

    def test_can_allocate(self):
        """Test quota allocation check."""
        quota = ResourceQuota(
            entity_id="user-1",
            entity_type="user",
            max_cpus=100,
            max_memory_gb=500.0,
            max_gpus=8,
            max_concurrent_jobs=10,
        )

        # Within quota
        resources = ResourceRequest(cpus=10, memory_gb=50.0, gpus=2)
        assert quota.can_allocate(resources) is True

        # Exceeds CPU quota
        resources = ResourceRequest(cpus=150, memory_gb=50.0, gpus=2)
        assert quota.can_allocate(resources) is False

    def test_allocate_and_release(self):
        """Test quota allocation and release."""
        quota = ResourceQuota(
            entity_id="user-1",
            entity_type="user",
            max_cpus=100,
            max_memory_gb=500.0,
            max_gpus=8,
        )

        resources = ResourceRequest(cpus=10, memory_gb=50.0, gpus=2)

        quota.allocate(resources)
        assert quota.used_cpus == 10
        assert quota.used_memory_gb == 50.0
        assert quota.used_gpus == 2
        assert quota.active_jobs == 1

        quota.release(resources)
        assert quota.used_cpus == 0
        assert quota.used_memory_gb == 0.0
        assert quota.used_gpus == 0
        assert quota.active_jobs == 0


class TestCheckpoint:
    """Tests for Checkpoint model."""

    def test_is_valid(self):
        """Test checkpoint validity check."""
        ckpt = Checkpoint(
            job_id="job-1",
            epoch=1,
            step=100,
            path="/ckpt",
            size_bytes=1000,
            status=CheckpointStatus.COMPLETED,
        )
        assert ckpt.is_valid() is True

        ckpt.status = CheckpointStatus.PENDING
        assert ckpt.is_valid() is False

        ckpt.status = CheckpointStatus.COMPLETED
        ckpt.size_bytes = 0
        assert ckpt.is_valid() is False


class TestMetricValue:
    """Tests for MetricValue model."""

    def test_creation(self):
        """Test metric creation."""
        metric = MetricValue(
            name="accuracy",
            value=0.95,
            step=1000,
            epoch=5,
        )

        assert metric.name == "accuracy"
        assert metric.value == 0.95
        assert metric.step == 1000
        assert metric.epoch == 5
        assert metric.metric_type == MetricType.SCALAR


class TestExperiment:
    """Tests for Experiment model."""

    def test_creation(self):
        """Test experiment creation."""
        exp = Experiment(
            name="test-experiment",
            user_id="user-1",
            description="Test description",
        )

        assert exp.name == "test-experiment"
        assert exp.status == "active"
        assert exp.job_ids == []
