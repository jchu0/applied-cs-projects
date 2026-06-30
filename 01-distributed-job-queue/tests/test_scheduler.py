"""Tests for scheduler implementation."""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta

# Skip if required dependencies are not available
pytest.importorskip("croniter")
pytest.importorskip("structlog")

from jobqueue.scheduler import Scheduler, ScheduledJob, schedule_delayed
from jobqueue.broker import InMemoryBroker
from jobqueue.models import TaskPriority


@pytest.fixture
def broker():
    """Create a fresh broker for each test."""
    return InMemoryBroker()


@pytest.fixture
def scheduler(broker):
    """Create a scheduler for testing."""
    return Scheduler(broker, poll_interval=0.1)


class TestScheduledJob:
    """Tests for ScheduledJob model."""

    def test_cron_next_run(self):
        """Test calculating next run from cron expression."""
        job = ScheduledJob(
            name="test",
            task_name="test_task",
            cron="0 * * * *",  # Every hour
        )

        next_run = job.calculate_next_run()
        assert next_run > datetime.now(timezone.utc)
        assert next_run.minute == 0

    def test_interval_next_run(self):
        """Test calculating next run from interval."""
        job = ScheduledJob(
            name="test",
            task_name="test_task",
            interval_seconds=60,
        )

        now = datetime.now(timezone.utc)
        next_run = job.calculate_next_run(now)

        expected = now + timedelta(seconds=60)
        assert abs((next_run - expected).total_seconds()) < 1


class TestScheduler:
    """Tests for Scheduler."""

    def test_add_cron_job(self, scheduler):
        """Test adding a cron job."""
        job = scheduler.add_job(
            name="hourly",
            task_name="hourly_task",
            cron="0 * * * *",
        )

        assert job.name == "hourly"
        assert job.task_name == "hourly_task"
        assert job.next_run is not None
        assert scheduler.get_job("hourly") is not None

    def test_add_interval_job(self, scheduler):
        """Test adding an interval job."""
        job = scheduler.add_job(
            name="every_minute",
            task_name="minute_task",
            interval_seconds=60,
        )

        assert job.interval_seconds == 60
        assert job.next_run is not None

    def test_invalid_job_no_schedule(self, scheduler):
        """Test that job without schedule raises error."""
        with pytest.raises(ValueError):
            scheduler.add_job(
                name="invalid",
                task_name="task",
            )

    def test_invalid_job_both_schedules(self, scheduler):
        """Test that job with both cron and interval raises error."""
        with pytest.raises(ValueError):
            scheduler.add_job(
                name="invalid",
                task_name="task",
                cron="0 * * * *",
                interval_seconds=60,
            )

    def test_invalid_cron_expression(self, scheduler):
        """Test that invalid cron expression raises error."""
        with pytest.raises(ValueError):
            scheduler.add_job(
                name="invalid",
                task_name="task",
                cron="invalid",
            )

    def test_remove_job(self, scheduler):
        """Test removing a job."""
        scheduler.add_job(
            name="test",
            task_name="task",
            interval_seconds=60,
        )

        assert scheduler.remove_job("test") is True
        assert scheduler.get_job("test") is None

    def test_list_jobs(self, scheduler):
        """Test listing all jobs."""
        scheduler.add_job("job1", "task1", interval_seconds=60)
        scheduler.add_job("job2", "task2", interval_seconds=120)

        jobs = scheduler.list_jobs()
        assert len(jobs) == 2

    def test_enable_disable_job(self, scheduler):
        """Test enabling and disabling a job."""
        scheduler.add_job("test", "task", interval_seconds=60)

        scheduler.disable_job("test")
        job = scheduler.get_job("test")
        assert job.enabled is False

        scheduler.enable_job("test")
        job = scheduler.get_job("test")
        assert job.enabled is True

    async def test_job_execution(self, scheduler, broker):
        """Test that scheduler creates tasks for due jobs."""
        # Add job with very short interval
        scheduler.add_job(
            name="quick",
            task_name="quick_task",
            interval_seconds=1,
            payload={"key": "value"},
        )

        # Manually set next_run to past
        job = scheduler.get_job("quick")
        job.next_run = datetime.now(timezone.utc) - timedelta(seconds=1)

        # Run scheduler check
        await scheduler._check_jobs()

        # Verify task was created
        stats = await broker.get_queue_stats("default")
        assert stats.pending >= 1

        # Verify job was updated
        assert job.run_count == 1
        assert job.last_run is not None

    async def test_run_once(self, scheduler, broker):
        """Test manually triggering a job."""
        scheduler.add_job(
            name="manual",
            task_name="manual_task",
            interval_seconds=3600,  # 1 hour
        )

        task = await scheduler.run_once("manual")
        assert task is not None
        assert task.name == "manual_task"

        # Verify task was enqueued
        stats = await broker.get_queue_stats("default")
        assert stats.pending >= 1


class TestScheduleDelayed:
    """Tests for delayed task scheduling."""

    async def test_schedule_delayed(self, broker):
        """Test scheduling a delayed task."""
        task = await schedule_delayed(
            broker,
            task_name="delayed_task",
            delay_seconds=60,
            payload={"key": "value"},
        )

        assert task.name == "delayed_task"
        assert task.eta is not None
        assert task.eta > datetime.now(timezone.utc)

    async def test_delayed_task_not_immediately_dequeued(self, broker):
        """Test that delayed task is not immediately dequeued."""
        await schedule_delayed(
            broker,
            task_name="delayed",
            delay_seconds=3600,  # 1 hour
        )

        # Try to dequeue immediately
        task = await broker.dequeue(["default"])
        assert task is None  # Should not be available yet
