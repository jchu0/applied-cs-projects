"""Tests for JobManager."""

import pytest
from datetime import datetime, timedelta
from pydantic import ValidationError as PydanticValidationError

from ml_orchestrator.core.models import (
    JobConfig,
    JobPriority,
    JobStatus,
    ResourceRequest,
)
from ml_orchestrator.core.job_manager import JobManager, JobStore
from ml_orchestrator.core.exceptions import (
    JobNotFoundError,
    JobStateError,
    ValidationError,
)


class TestJobStore:
    """Tests for JobStore."""

    @pytest.mark.asyncio
    async def test_add_and_get(self, sample_job):
        """Test adding and retrieving jobs."""
        store = JobStore()
        await store.add(sample_job)

        retrieved = await store.get(sample_job.id)
        assert retrieved is not None
        assert retrieved.id == sample_job.id
        assert retrieved.name == sample_job.name

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        """Test getting non-existent job."""
        store = JobStore()
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, sample_job):
        """Test deleting a job."""
        store = JobStore()
        await store.add(sample_job)

        deleted = await store.delete(sample_job.id)
        assert deleted is not None
        assert deleted.id == sample_job.id

        # Should be gone
        result = await store.get(sample_job.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_status(self, make_job):
        """Test listing jobs by status."""
        store = JobStore()

        job1 = make_job(name="job-1", status=JobStatus.PENDING)
        job2 = make_job(name="job-2", status=JobStatus.RUNNING)
        job3 = make_job(name="job-3", status=JobStatus.PENDING)

        await store.add(job1)
        await store.add(job2)
        await store.add(job3)

        pending = await store.list_by_status(JobStatus.PENDING)
        assert len(pending) == 2

        running = await store.list_by_status(JobStatus.RUNNING)
        assert len(running) == 1

    @pytest.mark.asyncio
    async def test_list_by_user(self, make_job):
        """Test listing jobs by user."""
        store = JobStore()

        job1 = make_job(name="job-1", user_id="user-1")
        job2 = make_job(name="job-2", user_id="user-2")
        job3 = make_job(name="job-3", user_id="user-1")

        await store.add(job1)
        await store.add(job2)
        await store.add(job3)

        user1_jobs = await store.list_by_user("user-1")
        assert len(user1_jobs) == 2

        user2_jobs = await store.list_by_user("user-2")
        assert len(user2_jobs) == 1

    @pytest.mark.asyncio
    async def test_count(self, make_job):
        """Test counting jobs."""
        store = JobStore()

        for i in range(5):
            await store.add(make_job(name=f"job-{i}"))

        total = await store.count()
        assert total == 5


class TestJobManager:
    """Tests for JobManager."""

    @pytest.mark.asyncio
    async def test_submit_job(self, job_manager, sample_job_config):
        """Test job submission."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        assert job.id is not None
        assert job.name == "test-job"
        assert job.user_id == "user-123"
        assert job.status == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_submit_job_with_resources(self, job_manager, sample_job_config):
        """Test job submission with resources."""
        resources = ResourceRequest(cpus=8, memory_gb=32.0, gpus=2)

        job = await job_manager.submit_job(
            name="gpu-job",
            user_id="user-123",
            config=sample_job_config,
            resources=resources,
        )

        assert job.resources.cpus == 8
        assert job.resources.gpus == 2

    @pytest.mark.asyncio
    async def test_submit_job_validation_error(self, job_manager):
        """Test job submission with invalid config (validation happens at JobConfig creation)."""
        # Pydantic validates at object creation time
        with pytest.raises(PydanticValidationError):
            JobConfig(script_path="")  # Empty path raises validation error

    @pytest.mark.asyncio
    async def test_get_job(self, job_manager, sample_job_config):
        """Test getting a job."""
        submitted = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        retrieved = await job_manager.get_job(submitted.id)
        assert retrieved.id == submitted.id

    @pytest.mark.asyncio
    async def test_get_job_not_found(self, job_manager):
        """Test getting non-existent job."""
        with pytest.raises(JobNotFoundError):
            await job_manager.get_job("nonexistent-id")

    @pytest.mark.asyncio
    async def test_queue_job(self, job_manager, sample_job_config):
        """Test queuing a job."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        queued = await job_manager.queue_job(job.id)
        assert queued.status == JobStatus.QUEUED
        assert queued.queued_at is not None

    @pytest.mark.asyncio
    async def test_start_and_run_job(self, job_manager, sample_job_config):
        """Test starting and running a job."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        await job_manager.queue_job(job.id)
        await job_manager.schedule_job(job.id, ["worker-1"])
        await job_manager.start_job(job.id)
        running = await job_manager.run_job(job.id)

        assert running.status == JobStatus.RUNNING
        assert running.started_at is not None

    @pytest.mark.asyncio
    async def test_pause_and_resume(self, job_manager, sample_job_config):
        """Test pausing and resuming a job."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        # Get to running state
        await job_manager.queue_job(job.id)
        await job_manager.schedule_job(job.id, ["worker-1"])
        await job_manager.start_job(job.id)
        await job_manager.run_job(job.id)

        # Pause
        paused = await job_manager.pause_job(job.id)
        assert paused.status == JobStatus.PAUSED
        assert paused.paused_at is not None

        # Resume
        resumed = await job_manager.resume_job(job.id)
        assert resumed.status == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_complete_job(self, job_manager, sample_job_config):
        """Test completing a job."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        await job_manager.queue_job(job.id)
        await job_manager.schedule_job(job.id, ["worker-1"])
        await job_manager.start_job(job.id)
        await job_manager.run_job(job.id)

        completed = await job_manager.complete_job(job.id)
        assert completed.status == JobStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.progress_percent == 100.0

    @pytest.mark.asyncio
    async def test_fail_job_with_retry(self, job_manager, sample_job_config):
        """Test failing a job with retry."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        await job_manager.queue_job(job.id)
        await job_manager.schedule_job(job.id, ["worker-1"])
        await job_manager.start_job(job.id)
        await job_manager.run_job(job.id)

        # First failure - should retry
        failed = await job_manager.fail_job(job.id, "Test error", retry=True)
        assert failed.status == JobStatus.QUEUED  # Re-queued
        assert failed.retry_count == 1
        assert "retry_delay_seconds" in failed.metadata

    @pytest.mark.asyncio
    async def test_fail_job_no_retry(self, job_manager, sample_job_config):
        """Test failing a job without retry."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        await job_manager.queue_job(job.id)
        await job_manager.schedule_job(job.id, ["worker-1"])
        await job_manager.start_job(job.id)
        await job_manager.run_job(job.id)

        failed = await job_manager.fail_job(job.id, "Test error", retry=False)
        assert failed.status == JobStatus.FAILED
        assert failed.error_message == "Test error"

    @pytest.mark.asyncio
    async def test_cancel_job(self, job_manager, sample_job_config):
        """Test cancelling a job."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        cancelled = await job_manager.cancel_job(job.id, "User requested")
        assert cancelled.status == JobStatus.CANCELLED
        assert cancelled.metadata["cancellation_reason"] == "User requested"

    @pytest.mark.asyncio
    async def test_cancel_terminal_job_error(self, job_manager, sample_job_config):
        """Test cancelling already completed job fails."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        await job_manager.queue_job(job.id)
        await job_manager.schedule_job(job.id, ["worker-1"])
        await job_manager.start_job(job.id)
        await job_manager.run_job(job.id)
        await job_manager.complete_job(job.id)

        with pytest.raises(JobStateError):
            await job_manager.cancel_job(job.id)

    @pytest.mark.asyncio
    async def test_preempt_job(self, job_manager, sample_job_config):
        """Test preempting a job."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
            preemptible=True,
        )

        await job_manager.queue_job(job.id)
        await job_manager.schedule_job(job.id, ["worker-1"])
        await job_manager.start_job(job.id)
        await job_manager.run_job(job.id)

        preempted = await job_manager.preempt_job(job.id, "Higher priority job")
        assert preempted.status == JobStatus.PREEMPTED

    @pytest.mark.asyncio
    async def test_preempt_non_preemptible_error(self, job_manager, sample_job_config):
        """Test preempting non-preemptible job fails."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
            preemptible=False,
        )

        await job_manager.queue_job(job.id)
        await job_manager.schedule_job(job.id, ["worker-1"])
        await job_manager.start_job(job.id)
        await job_manager.run_job(job.id)

        with pytest.raises(JobStateError):
            await job_manager.preempt_job(job.id)

    @pytest.mark.asyncio
    async def test_update_progress(self, job_manager, sample_job_config):
        """Test updating job progress."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        await job_manager.queue_job(job.id)
        await job_manager.schedule_job(job.id, ["worker-1"])
        await job_manager.start_job(job.id)
        await job_manager.run_job(job.id)

        updated = await job_manager.update_progress(
            job.id, epoch=5, step=1000, progress_percent=50.0
        )

        assert updated.current_epoch == 5
        assert updated.current_step == 1000
        assert updated.progress_percent == 50.0

    @pytest.mark.asyncio
    async def test_log_metric(self, job_manager, sample_job_config):
        """Test logging metrics."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        updated = await job_manager.log_metric(
            job.id, "loss", 0.5, step=100, epoch=1
        )

        assert len(updated.metrics) == 1
        assert updated.metrics[0].name == "loss"
        assert updated.metrics[0].value == 0.5
        assert updated.best_metrics["loss"] == 0.5

    @pytest.mark.asyncio
    async def test_add_checkpoint(self, job_manager, sample_job_config):
        """Test adding checkpoint."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        ckpt = await job_manager.add_checkpoint(
            job.id,
            path="/checkpoints/epoch_1.pt",
            epoch=1,
            step=1000,
            metrics={"loss": 0.3},
            size_bytes=1000000,
        )

        assert ckpt.epoch == 1
        assert ckpt.step == 1000

        retrieved = await job_manager.get_job(job.id)
        assert len(retrieved.checkpoints) == 1
        assert retrieved.latest_checkpoint_id == ckpt.id

    @pytest.mark.asyncio
    async def test_list_jobs(self, job_manager, sample_job_config):
        """Test listing jobs."""
        # Create multiple jobs
        for i in range(5):
            await job_manager.submit_job(
                name=f"job-{i}",
                user_id="user-123",
                config=sample_job_config,
            )

        jobs = await job_manager.list_jobs(limit=10)
        assert len(jobs) == 5

        # Test with limit
        jobs = await job_manager.list_jobs(limit=3)
        assert len(jobs) == 3

    @pytest.mark.asyncio
    async def test_list_jobs_by_status(self, job_manager, sample_job_config):
        """Test listing jobs by status."""
        # Create jobs with different statuses
        job1 = await job_manager.submit_job(
            name="job-1", user_id="user-123", config=sample_job_config
        )
        job2 = await job_manager.submit_job(
            name="job-2", user_id="user-123", config=sample_job_config
        )
        job3 = await job_manager.submit_job(
            name="job-3", user_id="user-123", config=sample_job_config
        )

        await job_manager.queue_job(job2.id)
        await job_manager.queue_job(job3.id)

        pending = await job_manager.list_jobs(status=JobStatus.PENDING)
        assert len(pending) == 1

        queued = await job_manager.list_jobs(status=JobStatus.QUEUED)
        assert len(queued) == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, job_manager, sample_job_config):
        """Test getting statistics."""
        # Create some jobs
        for i in range(3):
            await job_manager.submit_job(
                name=f"job-{i}",
                user_id="user-123",
                config=sample_job_config,
            )

        stats = await job_manager.get_stats()
        assert stats["total_jobs"] == 3
        assert "jobs_by_status" in stats

    @pytest.mark.asyncio
    async def test_invalid_state_transition(self, job_manager, sample_job_config):
        """Test invalid state transition raises error."""
        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        # Can't run a pending job directly
        with pytest.raises(JobStateError):
            await job_manager.run_job(job.id)

    @pytest.mark.asyncio
    async def test_callbacks(self, job_manager, sample_job_config):
        """Test event callbacks."""
        events = []

        def callback(job, data):
            events.append(("event", job.id, data))

        job_manager.register_callback("job_submitted", callback)
        job_manager.register_callback("state_changed", callback)

        job = await job_manager.submit_job(
            name="test-job",
            user_id="user-123",
            config=sample_job_config,
        )

        # Should have received submit event
        assert len(events) == 1
        assert events[0][1] == job.id

        await job_manager.queue_job(job.id)

        # Should have received state change event
        assert len(events) == 2
