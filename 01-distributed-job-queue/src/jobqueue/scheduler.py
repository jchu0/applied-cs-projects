"""Scheduler for recurring and delayed tasks."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

import structlog
from croniter import croniter
from pydantic import BaseModel, Field

from jobqueue.broker import Broker
from jobqueue.models import Task, TaskPriority

logger = structlog.get_logger()


class ScheduledJob(BaseModel):
    """A scheduled job definition."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Unique job name")
    task_name: str = Field(..., description="Task type to create")
    queue: str = Field(default="default")
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)

    # Schedule configuration
    cron: str | None = Field(default=None, description="Cron expression")
    interval_seconds: int | None = Field(default=None, description="Run every N seconds")

    # State
    enabled: bool = True
    last_run: datetime | None = None
    next_run: datetime | None = None
    run_count: int = 0

    # Options
    max_retries: int = 3
    timeout_ms: int = 30000
    metadata: dict[str, Any] = Field(default_factory=dict)

    def calculate_next_run(self, from_time: datetime | None = None) -> datetime:
        """Calculate the next run time."""
        base_time = from_time or datetime.now(timezone.utc)

        if self.cron:
            cron = croniter(self.cron, base_time)
            return cron.get_next(datetime).replace(tzinfo=timezone.utc)
        elif self.interval_seconds:
            return base_time + timedelta(seconds=self.interval_seconds)
        else:
            raise ValueError("Job must have either cron or interval_seconds")


class Scheduler:
    """
    Scheduler for managing recurring and delayed tasks.

    Supports:
    - Cron expressions for complex schedules
    - Fixed interval scheduling
    - One-time delayed execution
    - Job management (add, remove, enable, disable)
    """

    def __init__(
        self,
        broker: Broker,
        poll_interval: float = 1.0,
    ):
        self.broker = broker
        self.poll_interval = poll_interval

        self._jobs: dict[str, ScheduledJob] = {}
        self._running = False
        self._lock = asyncio.Lock()

        logger.info("Scheduler initialized", poll_interval=poll_interval)

    def add_job(
        self,
        name: str,
        task_name: str,
        cron: str | None = None,
        interval_seconds: int | None = None,
        queue: str = "default",
        payload: dict[str, Any] | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs: Any,
    ) -> ScheduledJob:
        """
        Add a scheduled job.

        Args:
            name: Unique job name
            task_name: Task type to execute
            cron: Cron expression (e.g., "0 * * * *" for hourly)
            interval_seconds: Run every N seconds
            queue: Target queue
            payload: Task payload
            priority: Task priority

        Returns:
            The created ScheduledJob

        Examples:
            # Run every hour
            scheduler.add_job("hourly_report", "generate_report", cron="0 * * * *")

            # Run every 5 minutes
            scheduler.add_job("health_check", "check_health", interval_seconds=300)
        """
        if not cron and not interval_seconds:
            raise ValueError("Must specify either cron or interval_seconds")

        if cron and interval_seconds:
            raise ValueError("Cannot specify both cron and interval_seconds")

        # Validate cron expression
        if cron:
            try:
                croniter(cron)
            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid cron expression: {e}")

        job = ScheduledJob(
            name=name,
            task_name=task_name,
            cron=cron,
            interval_seconds=interval_seconds,
            queue=queue,
            payload=payload or {},
            priority=priority,
            **kwargs,
        )

        # Calculate initial next_run
        job.next_run = job.calculate_next_run()

        self._jobs[name] = job

        logger.info(
            "Job added",
            job_name=name,
            task_name=task_name,
            next_run=job.next_run.isoformat(),
        )

        return job

    def remove_job(self, name: str) -> bool:
        """Remove a scheduled job."""
        if name in self._jobs:
            del self._jobs[name]
            logger.info("Job removed", job_name=name)
            return True
        return False

    def get_job(self, name: str) -> ScheduledJob | None:
        """Get a job by name."""
        return self._jobs.get(name)

    def list_jobs(self) -> list[ScheduledJob]:
        """List all scheduled jobs."""
        return list(self._jobs.values())

    def enable_job(self, name: str) -> bool:
        """Enable a job."""
        job = self._jobs.get(name)
        if job:
            job.enabled = True
            job.next_run = job.calculate_next_run()
            logger.info("Job enabled", job_name=name)
            return True
        return False

    def disable_job(self, name: str) -> bool:
        """Disable a job."""
        job = self._jobs.get(name)
        if job:
            job.enabled = False
            logger.info("Job disabled", job_name=name)
            return True
        return False

    async def start(self) -> None:
        """Start the scheduler loop."""
        self._running = True
        logger.info("Scheduler started")

        while self._running:
            try:
                await self._check_jobs()
            except Exception as e:
                logger.error("Scheduler error", error=str(e))

            await asyncio.sleep(self.poll_interval)

        logger.info("Scheduler stopped")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

    async def _check_jobs(self) -> None:
        """Check and execute due jobs."""
        now = datetime.now(timezone.utc)

        async with self._lock:
            for job in self._jobs.values():
                if not job.enabled:
                    continue

                if job.next_run and job.next_run <= now:
                    await self._execute_job(job)

    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a scheduled job by creating a task."""
        try:
            # Create task
            task = Task(
                name=job.task_name,
                queue=job.queue,
                payload=job.payload,
                priority=job.priority,
                max_retries=job.max_retries,
                timeout_ms=job.timeout_ms,
                metadata={
                    **job.metadata,
                    "scheduled_job": job.name,
                    "schedule_time": job.next_run.isoformat() if job.next_run else None,
                },
            )

            await self.broker.enqueue(task)

            # Update job state
            job.last_run = datetime.now(timezone.utc)
            job.run_count += 1
            job.next_run = job.calculate_next_run(job.last_run)

            logger.info(
                "Scheduled task created",
                job_name=job.name,
                task_id=task.id,
                next_run=job.next_run.isoformat(),
            )

        except Exception as e:
            logger.error(
                "Failed to execute scheduled job",
                job_name=job.name,
                error=str(e),
            )

    async def run_once(self, name: str) -> Task | None:
        """Manually trigger a job to run immediately."""
        job = self._jobs.get(name)
        if not job:
            return None

        task = Task(
            name=job.task_name,
            queue=job.queue,
            payload=job.payload,
            priority=job.priority,
            max_retries=job.max_retries,
            timeout_ms=job.timeout_ms,
            metadata={
                **job.metadata,
                "scheduled_job": job.name,
                "manual_trigger": True,
            },
        )

        await self.broker.enqueue(task)
        logger.info("Manual job trigger", job_name=name, task_id=task.id)
        return task


async def schedule_delayed(
    broker: Broker,
    task_name: str,
    delay_seconds: float,
    payload: dict[str, Any] | None = None,
    queue: str = "default",
    **kwargs: Any,
) -> Task:
    """
    Schedule a one-time delayed task.

    Args:
        broker: The broker to use
        task_name: Task type to execute
        delay_seconds: Seconds to wait before execution
        payload: Task payload
        queue: Target queue

    Returns:
        The created Task
    """
    eta = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)

    task = Task(
        name=task_name,
        queue=queue,
        payload=payload or {},
        eta=eta,
        **kwargs,
    )

    await broker.enqueue(task)

    logger.info(
        "Delayed task scheduled",
        task_id=task.id,
        task_name=task_name,
        eta=eta.isoformat(),
    )

    return task
