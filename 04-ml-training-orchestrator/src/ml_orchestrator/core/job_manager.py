"""Job Manager - Core component for managing training job lifecycle."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
from collections import defaultdict
import structlog

from ml_orchestrator.core.models import (
    Checkpoint,
    CheckpointStatus,
    JobConfig,
    JobPriority,
    JobStatus,
    MetricValue,
    ResourceRequest,
    TrainingJob,
)
from ml_orchestrator.core.exceptions import (
    JobNotFoundError,
    JobStateError,
    ValidationError,
)


logger = structlog.get_logger(__name__)


# Valid state transitions
VALID_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.PENDING: {JobStatus.QUEUED, JobStatus.CANCELLED, JobStatus.FAILED},
    JobStatus.QUEUED: {
        JobStatus.SCHEDULED,
        JobStatus.CANCELLED,
        JobStatus.FAILED,
        JobStatus.TIMEOUT,
    },
    JobStatus.SCHEDULED: {
        JobStatus.STARTING,
        JobStatus.QUEUED,  # If resources become unavailable
        JobStatus.CANCELLED,
        JobStatus.FAILED,
    },
    JobStatus.STARTING: {
        JobStatus.RUNNING,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.TIMEOUT,
    },
    JobStatus.RUNNING: {
        JobStatus.PAUSED,
        JobStatus.CHECKPOINTING,
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.PREEMPTED,
        JobStatus.TIMEOUT,
        JobStatus.QUEUED,  # For retry on failure
    },
    JobStatus.PAUSED: {
        JobStatus.RUNNING,
        JobStatus.CANCELLED,
        JobStatus.FAILED,
    },
    JobStatus.CHECKPOINTING: {
        JobStatus.RUNNING,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
    },
    JobStatus.PREEMPTED: {
        JobStatus.QUEUED,  # Re-queue for later
        JobStatus.CANCELLED,
        JobStatus.FAILED,
    },
    # Terminal states - no transitions out
    JobStatus.COMPLETED: set(),
    JobStatus.FAILED: set(),
    JobStatus.CANCELLED: set(),
    JobStatus.TIMEOUT: set(),
}


class JobStore:
    """In-memory job store with indexing."""

    def __init__(self):
        self._jobs: dict[str, TrainingJob] = {}
        self._by_status: dict[JobStatus, set[str]] = defaultdict(set)
        self._by_user: dict[str, set[str]] = defaultdict(set)
        self._by_team: dict[str, set[str]] = defaultdict(set)
        self._by_experiment: dict[str, set[str]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def add(self, job: TrainingJob) -> None:
        """Add a job to the store."""
        async with self._lock:
            self._jobs[job.id] = job
            self._by_status[job.status].add(job.id)
            self._by_user[job.user_id].add(job.id)
            if job.team_id:
                self._by_team[job.team_id].add(job.id)
            if job.experiment_id:
                self._by_experiment[job.experiment_id].add(job.id)

    async def get(self, job_id: str) -> Optional[TrainingJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def update(self, job: TrainingJob, old_status: Optional[JobStatus] = None) -> None:
        """Update a job in the store."""
        async with self._lock:
            if old_status and old_status != job.status:
                self._by_status[old_status].discard(job.id)
                self._by_status[job.status].add(job.id)
            self._jobs[job.id] = job

    async def delete(self, job_id: str) -> Optional[TrainingJob]:
        """Delete a job from the store."""
        async with self._lock:
            job = self._jobs.pop(job_id, None)
            if job:
                self._by_status[job.status].discard(job_id)
                self._by_user[job.user_id].discard(job_id)
                if job.team_id:
                    self._by_team[job.team_id].discard(job_id)
                if job.experiment_id:
                    self._by_experiment[job.experiment_id].discard(job_id)
            return job

    async def list_by_status(self, status: JobStatus) -> list[TrainingJob]:
        """List jobs by status."""
        job_ids = self._by_status.get(status, set())
        return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    async def list_by_user(self, user_id: str) -> list[TrainingJob]:
        """List jobs by user."""
        job_ids = self._by_user.get(user_id, set())
        return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    async def list_by_team(self, team_id: str) -> list[TrainingJob]:
        """List jobs by team."""
        job_ids = self._by_team.get(team_id, set())
        return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    async def list_by_experiment(self, experiment_id: str) -> list[TrainingJob]:
        """List jobs by experiment."""
        job_ids = self._by_experiment.get(experiment_id, set())
        return [self._jobs[jid] for jid in job_ids if jid in self._jobs]

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[JobStatus] = None,
        user_id: Optional[str] = None,
    ) -> list[TrainingJob]:
        """List all jobs with filtering."""
        if status:
            jobs = await self.list_by_status(status)
        elif user_id:
            jobs = await self.list_by_user(user_id)
        else:
            jobs = list(self._jobs.values())

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[offset : offset + limit]

    async def count(self, status: Optional[JobStatus] = None) -> int:
        """Count jobs."""
        if status:
            return len(self._by_status.get(status, set()))
        return len(self._jobs)

    def get_stats(self) -> dict[str, int]:
        """Get job statistics by status."""
        return {status.value: len(ids) for status, ids in self._by_status.items()}


class JobManager:
    """
    Manages training job lifecycle including submission, state transitions,
    progress updates, and checkpointing coordination.
    """

    def __init__(
        self,
        store: Optional[JobStore] = None,
        max_retries: int = 3,
        heartbeat_timeout_seconds: int = 120,
    ):
        self._store = store or JobStore()
        self._max_retries = max_retries
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._job_heartbeats: dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    @property
    def store(self) -> JobStore:
        """Get the job store."""
        return self._store

    def register_callback(
        self, event: str, callback: Callable[[TrainingJob, dict[str, Any]], None]
    ) -> None:
        """Register a callback for job events."""
        self._callbacks[event].append(callback)

    async def _emit_event(
        self, event: str, job: TrainingJob, data: Optional[dict[str, Any]] = None
    ) -> None:
        """Emit an event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(job, data or {})
                else:
                    callback(job, data or {})
            except Exception as e:
                logger.error("callback_error", event=event, job_id=job.id, error=str(e))

    def _validate_job_config(self, config: JobConfig) -> None:
        """Validate job configuration."""
        if not config.script_path:
            raise ValidationError("script_path is required", "script_path")

        if config.epochs and config.max_steps:
            raise ValidationError(
                "Cannot specify both epochs and max_steps", "epochs/max_steps"
            )

        if config.distributed.enabled:
            if config.distributed.world_size < 2:
                raise ValidationError(
                    "world_size must be >= 2 for distributed training", "world_size"
                )

    def _validate_resources(self, resources: ResourceRequest) -> None:
        """Validate resource request."""
        if resources.gpus > 0 and resources.gpu_memory_gb:
            if resources.gpu_memory_gb < 4:
                raise ValidationError(
                    "gpu_memory_gb should be at least 4GB", "gpu_memory_gb"
                )

    async def submit_job(
        self,
        name: str,
        user_id: str,
        config: JobConfig,
        resources: Optional[ResourceRequest] = None,
        priority: JobPriority = JobPriority.NORMAL,
        team_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        preemptible: bool = True,
        tags: Optional[dict[str, str]] = None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> TrainingJob:
        """
        Submit a new training job.

        Args:
            name: Human-readable job name
            user_id: ID of the user submitting the job
            config: Job configuration
            resources: Resource requirements
            priority: Job priority level
            team_id: Optional team ID
            experiment_id: Optional experiment ID
            preemptible: Whether job can be preempted
            tags: Optional tags
            resume_from_checkpoint: Optional checkpoint ID to resume from

        Returns:
            The created TrainingJob

        Raises:
            ValidationError: If configuration is invalid
        """
        self._validate_job_config(config)

        if resources:
            self._validate_resources(resources)
        else:
            resources = ResourceRequest()

        job = TrainingJob(
            name=name,
            user_id=user_id,
            team_id=team_id,
            experiment_id=experiment_id,
            config=config,
            resources=resources,
            priority=priority,
            preemptible=preemptible,
            tags=tags or {},
            resume_from_checkpoint=resume_from_checkpoint,
            status=JobStatus.PENDING,
        )

        await self._store.add(job)

        logger.info(
            "job_submitted",
            job_id=job.id,
            name=name,
            user_id=user_id,
            priority=priority.value,
        )

        await self._emit_event("job_submitted", job)

        return job

    async def get_job(self, job_id: str) -> TrainingJob:
        """
        Get a job by ID.

        Raises:
            JobNotFoundError: If job doesn't exist
        """
        job = await self._store.get(job_id)
        if not job:
            raise JobNotFoundError(job_id)
        return job

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        experiment_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[TrainingJob]:
        """List jobs with optional filtering."""
        if experiment_id:
            return await self._store.list_by_experiment(experiment_id)
        if team_id:
            jobs = await self._store.list_by_team(team_id)
            if status:
                jobs = [j for j in jobs if j.status == status]
            return jobs[offset : offset + limit]
        return await self._store.list_all(
            limit=limit, offset=offset, status=status, user_id=user_id
        )

    async def _transition_state(
        self,
        job: TrainingJob,
        new_status: JobStatus,
        error_message: Optional[str] = None,
        error_traceback: Optional[str] = None,
    ) -> TrainingJob:
        """
        Transition job to a new state.

        Raises:
            JobStateError: If transition is not valid
        """
        if new_status not in VALID_TRANSITIONS.get(job.status, set()):
            raise JobStateError(
                job.id,
                job.status.value,
                new_status.value,
                f"Cannot transition from {job.status.value} to {new_status.value}",
            )

        old_status = job.status
        job.status = new_status

        # Update timestamps
        now = datetime.utcnow()
        if new_status == JobStatus.QUEUED and not job.queued_at:
            job.queued_at = now
        elif new_status in (JobStatus.STARTING, JobStatus.RUNNING) and not job.started_at:
            job.started_at = now
        elif new_status == JobStatus.PAUSED:
            job.paused_at = now
        elif new_status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.TIMEOUT,
        ):
            job.completed_at = now

        # Set error info
        if error_message:
            job.error_message = error_message
        if error_traceback:
            job.error_traceback = error_traceback

        await self._store.update(job, old_status)

        logger.info(
            "job_state_changed",
            job_id=job.id,
            old_status=old_status.value,
            new_status=new_status.value,
        )

        await self._emit_event(
            "state_changed",
            job,
            {"old_status": old_status.value, "new_status": new_status.value},
        )

        return job

    async def queue_job(self, job_id: str) -> TrainingJob:
        """Queue a pending job for scheduling."""
        job = await self.get_job(job_id)
        return await self._transition_state(job, JobStatus.QUEUED)

    async def schedule_job(self, job_id: str, worker_ids: list[str]) -> TrainingJob:
        """Mark a job as scheduled with assigned workers."""
        job = await self.get_job(job_id)
        job.assigned_workers = worker_ids
        job = await self._transition_state(job, JobStatus.SCHEDULED)
        return job

    async def start_job(self, job_id: str) -> TrainingJob:
        """Mark a job as starting."""
        job = await self.get_job(job_id)
        return await self._transition_state(job, JobStatus.STARTING)

    async def run_job(self, job_id: str) -> TrainingJob:
        """Mark a job as running."""
        job = await self.get_job(job_id)
        job = await self._transition_state(job, JobStatus.RUNNING)
        self._job_heartbeats[job_id] = datetime.utcnow()
        return job

    async def pause_job(self, job_id: str) -> TrainingJob:
        """Pause a running job."""
        job = await self.get_job(job_id)
        return await self._transition_state(job, JobStatus.PAUSED)

    async def resume_job(self, job_id: str) -> TrainingJob:
        """Resume a paused job."""
        job = await self.get_job(job_id)
        if job.status != JobStatus.PAUSED:
            raise JobStateError(
                job_id,
                job.status.value,
                JobStatus.RUNNING.value,
                "Can only resume paused jobs",
            )
        job = await self._transition_state(job, JobStatus.RUNNING)
        self._job_heartbeats[job_id] = datetime.utcnow()
        return job

    async def preempt_job(self, job_id: str, reason: str = "Higher priority job") -> TrainingJob:
        """Preempt a running job."""
        job = await self.get_job(job_id)
        if not job.preemptible:
            raise JobStateError(
                job_id,
                job.status.value,
                JobStatus.PREEMPTED.value,
                "Job is not preemptible",
            )
        job.metadata["preemption_reason"] = reason
        job = await self._transition_state(job, JobStatus.PREEMPTED)
        await self._emit_event("job_preempted", job, {"reason": reason})
        return job

    async def complete_job(self, job_id: str) -> TrainingJob:
        """Mark a job as completed."""
        job = await self.get_job(job_id)
        job = await self._transition_state(job, JobStatus.COMPLETED)
        job.progress_percent = 100.0
        await self._store.update(job)
        self._job_heartbeats.pop(job_id, None)
        await self._emit_event("job_completed", job)
        return job

    async def fail_job(
        self,
        job_id: str,
        error_message: str,
        error_traceback: Optional[str] = None,
        retry: bool = True,
    ) -> TrainingJob:
        """Mark a job as failed, optionally retry."""
        job = await self.get_job(job_id)

        # Check if we should retry
        if retry and job.retry_count < job.config.retry.max_retries:
            job.retry_count += 1
            job.error_message = error_message
            job.error_traceback = error_traceback

            # Calculate retry delay
            delay = job.config.retry.retry_delay_seconds
            if job.config.retry.exponential_backoff:
                delay = min(
                    delay * (2 ** (job.retry_count - 1)),
                    job.config.retry.max_delay_seconds,
                )
            job.metadata["retry_delay_seconds"] = delay
            job.metadata["next_retry_at"] = (
                datetime.utcnow() + timedelta(seconds=delay)
            ).isoformat()

            # Re-queue the job
            job = await self._transition_state(job, JobStatus.QUEUED)
            logger.info(
                "job_retry_scheduled",
                job_id=job_id,
                retry_count=job.retry_count,
                delay=delay,
            )
            await self._emit_event(
                "job_retry", job, {"retry_count": job.retry_count, "delay": delay}
            )
        else:
            job = await self._transition_state(
                job, JobStatus.FAILED, error_message, error_traceback
            )
            await self._emit_event("job_failed", job, {"error": error_message})

        self._job_heartbeats.pop(job_id, None)
        return job

    async def cancel_job(self, job_id: str, reason: str = "User cancelled") -> TrainingJob:
        """Cancel a job."""
        job = await self.get_job(job_id)

        if job.is_terminal:
            raise JobStateError(
                job_id,
                job.status.value,
                JobStatus.CANCELLED.value,
                "Cannot cancel a job in terminal state",
            )

        job.metadata["cancellation_reason"] = reason
        job = await self._transition_state(job, JobStatus.CANCELLED)
        self._job_heartbeats.pop(job_id, None)
        await self._emit_event("job_cancelled", job, {"reason": reason})
        return job

    async def timeout_job(self, job_id: str) -> TrainingJob:
        """Mark a job as timed out."""
        job = await self.get_job(job_id)
        job = await self._transition_state(job, JobStatus.TIMEOUT)
        self._job_heartbeats.pop(job_id, None)
        await self._emit_event("job_timeout", job)
        return job

    async def update_progress(
        self,
        job_id: str,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        progress_percent: Optional[float] = None,
    ) -> TrainingJob:
        """Update job progress."""
        job = await self.get_job(job_id)

        if epoch is not None:
            job.current_epoch = epoch
        if step is not None:
            job.current_step = step
        if progress_percent is not None:
            job.progress_percent = min(100.0, max(0.0, progress_percent))

        self._job_heartbeats[job_id] = datetime.utcnow()
        await self._store.update(job)

        logger.debug(
            "job_progress_updated",
            job_id=job_id,
            epoch=job.current_epoch,
            step=job.current_step,
            progress=job.progress_percent,
        )

        return job

    async def log_metric(
        self,
        job_id: str,
        name: str,
        value: float,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> TrainingJob:
        """Log a metric for a job."""
        job = await self.get_job(job_id)

        metric = MetricValue(
            name=name,
            value=value,
            step=step or job.current_step,
            epoch=epoch or job.current_epoch,
        )
        job.add_metric(metric)

        self._job_heartbeats[job_id] = datetime.utcnow()
        await self._store.update(job)

        logger.debug(
            "metric_logged",
            job_id=job_id,
            metric=name,
            value=value,
            step=step,
        )

        return job

    async def add_checkpoint(
        self,
        job_id: str,
        path: str,
        epoch: int,
        step: int,
        metrics: Optional[dict[str, float]] = None,
        size_bytes: int = 0,
    ) -> Checkpoint:
        """Add a checkpoint to a job."""
        job = await self.get_job(job_id)

        checkpoint = Checkpoint(
            job_id=job_id,
            epoch=epoch,
            step=step,
            path=path,
            size_bytes=size_bytes,
            metrics=metrics or {},
            status=CheckpointStatus.COMPLETED,
        )

        job.checkpoints.append(checkpoint)
        job.latest_checkpoint_id = checkpoint.id

        # Clean up old checkpoints if needed
        keep_n = job.config.checkpoint.keep_last_n
        if len(job.checkpoints) > keep_n:
            to_remove = job.checkpoints[:-keep_n]
            for ckpt in to_remove:
                ckpt.status = CheckpointStatus.DELETED
            job.checkpoints = job.checkpoints[-keep_n:]

        await self._store.update(job)

        logger.info(
            "checkpoint_added",
            job_id=job_id,
            checkpoint_id=checkpoint.id,
            epoch=epoch,
            step=step,
        )

        await self._emit_event("checkpoint_created", job, {"checkpoint": checkpoint.model_dump()})

        return checkpoint

    async def heartbeat(self, job_id: str) -> None:
        """Update job heartbeat."""
        self._job_heartbeats[job_id] = datetime.utcnow()

    async def check_timeouts(self) -> list[str]:
        """Check for jobs that have timed out. Returns list of timed out job IDs."""
        timed_out = []
        now = datetime.utcnow()

        # Check running jobs for heartbeat timeout
        running_jobs = await self._store.list_by_status(JobStatus.RUNNING)
        for job in running_jobs:
            last_heartbeat = self._job_heartbeats.get(job.id)
            if last_heartbeat:
                if (now - last_heartbeat).total_seconds() > self._heartbeat_timeout:
                    timed_out.append(job.id)
                    logger.warning(
                        "job_heartbeat_timeout",
                        job_id=job.id,
                        last_heartbeat=last_heartbeat.isoformat(),
                    )

            # Check job duration timeout
            if job.config.timeout_hours and job.started_at:
                max_duration = timedelta(hours=job.config.timeout_hours)
                if now - job.started_at > max_duration:
                    if job.id not in timed_out:
                        timed_out.append(job.id)
                    logger.warning(
                        "job_duration_timeout",
                        job_id=job.id,
                        duration_hours=(now - job.started_at).total_seconds() / 3600,
                    )

        return timed_out

    async def get_stats(self) -> dict[str, Any]:
        """Get job manager statistics."""
        stats = self._store.get_stats()
        return {
            "jobs_by_status": stats,
            "total_jobs": await self._store.count(),
            "active_heartbeats": len(self._job_heartbeats),
        }

    async def cleanup_terminal_jobs(self, max_age_days: int = 30) -> int:
        """Clean up old terminal jobs. Returns count of deleted jobs."""
        deleted = 0
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)

        for status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            jobs = await self._store.list_by_status(status)
            for job in jobs:
                if job.completed_at and job.completed_at < cutoff:
                    await self._store.delete(job.id)
                    deleted += 1

        logger.info("terminal_jobs_cleaned", deleted=deleted)
        return deleted
