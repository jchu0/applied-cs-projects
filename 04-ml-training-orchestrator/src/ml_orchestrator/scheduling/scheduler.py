"""Main scheduler for training jobs."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Optional
import structlog

from ml_orchestrator.core.models import (
    JobPriority,
    JobStatus,
    ResourceQuota,
    ResourceRequest,
    SchedulingPolicy as SchedulingPolicyEnum,
    TrainingJob,
    WorkerInfo,
)
from ml_orchestrator.core.job_manager import JobManager
from ml_orchestrator.core.exceptions import SchedulingError
from ml_orchestrator.scheduling.priority_queue import PriorityQueue
from ml_orchestrator.scheduling.policies import (
    BackfillPolicy,
    FairSharePolicy,
    FIFOPolicy,
    GangSchedulingPolicy,
    PreemptivePolicy,
    PriorityPolicy,
    SchedulingDecision,
    SchedulingPolicy,
)


logger = structlog.get_logger(__name__)


class Scheduler:
    """
    Main scheduler coordinating job queue, resource allocation, and policies.

    The scheduler runs as a background task, periodically checking for jobs
    to schedule and resources to allocate.
    """

    def __init__(
        self,
        job_manager: JobManager,
        policy: SchedulingPolicyEnum = SchedulingPolicyEnum.PRIORITY,
        scheduling_interval_seconds: float = 1.0,
        priority_refresh_interval_seconds: float = 60.0,
        max_concurrent_scheduling: int = 10,
    ):
        self._job_manager = job_manager
        self._queue = PriorityQueue()
        self._policy = self._create_policy(policy)
        self._policy_type = policy
        self._scheduling_interval = scheduling_interval_seconds
        self._priority_refresh_interval = priority_refresh_interval_seconds
        self._max_concurrent = max_concurrent_scheduling

        # Worker management
        self._workers: dict[str, WorkerInfo] = {}
        self._worker_lock = asyncio.Lock()

        # Quota management
        self._quotas: dict[str, ResourceQuota] = {}

        # State
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._priority_refresh_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Metrics
        self._metrics = {
            "jobs_scheduled": 0,
            "jobs_preempted": 0,
            "scheduling_decisions": 0,
            "scheduling_failures": 0,
            "last_scheduling_time_ms": 0,
        }

        # Register job manager callbacks
        self._job_manager.register_callback("job_submitted", self._on_job_submitted)
        self._job_manager.register_callback("job_completed", self._on_job_completed)
        self._job_manager.register_callback("job_failed", self._on_job_completed)
        self._job_manager.register_callback("job_cancelled", self._on_job_completed)

    def _create_policy(self, policy_type: SchedulingPolicyEnum) -> SchedulingPolicy:
        """Create scheduling policy instance."""
        policies = {
            SchedulingPolicyEnum.FIFO: FIFOPolicy,
            SchedulingPolicyEnum.FAIR_SHARE: FairSharePolicy,
            SchedulingPolicyEnum.PRIORITY: PriorityPolicy,
            SchedulingPolicyEnum.GANG: GangSchedulingPolicy,
            SchedulingPolicyEnum.BACKFILL: BackfillPolicy,
        }
        policy_class = policies.get(policy_type, PriorityPolicy)
        return policy_class()

    async def _on_job_submitted(self, job: TrainingJob, data: dict[str, Any]) -> None:
        """Handle job submission - queue the job."""
        await self._job_manager.queue_job(job.id)
        job = await self._job_manager.get_job(job.id)
        await self._queue.push(job)
        logger.info("job_queued_for_scheduling", job_id=job.id)

    async def _on_job_completed(self, job: TrainingJob, data: dict[str, Any]) -> None:
        """Handle job completion - release resources."""
        await self._queue.remove(job.id)
        await self._release_resources(job)

    async def start(self) -> None:
        """Start the scheduler background tasks."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduling_loop())
        self._priority_refresh_task = asyncio.create_task(self._priority_refresh_loop())
        logger.info("scheduler_started", policy=self._policy_type.value)

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        if self._priority_refresh_task:
            self._priority_refresh_task.cancel()
            try:
                await self._priority_refresh_task
            except asyncio.CancelledError:
                pass

        logger.info("scheduler_stopped")

    async def _scheduling_loop(self) -> None:
        """Main scheduling loop."""
        while self._running:
            try:
                start_time = datetime.utcnow()
                await self._schedule_pending_jobs()
                elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
                self._metrics["last_scheduling_time_ms"] = elapsed
            except Exception as e:
                logger.error("scheduling_loop_error", error=str(e))
                self._metrics["scheduling_failures"] += 1

            await asyncio.sleep(self._scheduling_interval)

    async def _priority_refresh_loop(self) -> None:
        """Periodically refresh priorities for aging."""
        while self._running:
            try:
                await self._queue.refresh_priorities()
            except Exception as e:
                logger.error("priority_refresh_error", error=str(e))

            await asyncio.sleep(self._priority_refresh_interval)

    async def _schedule_pending_jobs(self) -> None:
        """Attempt to schedule pending jobs."""
        queue_size = await self._queue.size()
        if queue_size == 0:
            return

        # Get available resources
        available = await self._get_available_resources()
        if not available:
            return

        # Get pending jobs from queue
        pending_job_ids = await self._queue.get_all_job_ids()
        pending_jobs = []
        for job_id in pending_job_ids[:self._max_concurrent]:
            try:
                job = await self._job_manager.get_job(job_id)
                if job.is_schedulable:
                    pending_jobs.append(job)
            except Exception:
                await self._queue.remove(job_id)

        if not pending_jobs:
            return

        # Get running jobs for preemption consideration
        running_jobs = await self._job_manager.list_jobs(status=JobStatus.RUNNING)

        # Make scheduling decision
        self._metrics["scheduling_decisions"] += 1

        if isinstance(self._policy, PreemptivePolicy):
            decision = await self._policy.select_job(
                pending_jobs, available, self._quotas, running_jobs
            )
        else:
            decision = await self._policy.select_job(
                pending_jobs, available, self._quotas
            )

        if decision:
            await self._execute_decision(decision)

    async def _execute_decision(self, decision: SchedulingDecision) -> None:
        """Execute a scheduling decision."""
        try:
            # Handle preemption first
            for preempt_id in decision.should_preempt:
                await self._job_manager.preempt_job(preempt_id, decision.reason)
                self._metrics["jobs_preempted"] += 1

            # Remove from queue
            await self._queue.remove(decision.job_id)

            # Schedule the job
            await self._job_manager.schedule_job(
                decision.job_id, decision.assigned_workers
            )

            # Allocate resources
            job = await self._job_manager.get_job(decision.job_id)
            await self._allocate_resources(job, decision.assigned_workers)

            # Start the job
            await self._job_manager.start_job(decision.job_id)
            await self._job_manager.run_job(decision.job_id)

            self._metrics["jobs_scheduled"] += 1
            logger.info(
                "job_scheduled",
                job_id=decision.job_id,
                workers=decision.assigned_workers,
                reason=decision.reason,
            )

        except Exception as e:
            logger.error(
                "scheduling_execution_failed",
                job_id=decision.job_id,
                error=str(e),
            )
            self._metrics["scheduling_failures"] += 1

    async def _get_available_resources(self) -> dict[str, ResourceRequest]:
        """Get available resources per worker."""
        async with self._worker_lock:
            available = {}
            for worker_id, worker in self._workers.items():
                if worker.is_healthy() and worker.status in (
                    WorkerInfo.WorkerStatus.READY if hasattr(WorkerInfo, 'WorkerStatus') else True,
                ):
                    available[worker_id] = worker.available_resources
            return available

    async def _allocate_resources(
        self, job: TrainingJob, worker_ids: list[str]
    ) -> None:
        """Allocate resources to a job."""
        async with self._worker_lock:
            for worker_id in worker_ids:
                if worker_id in self._workers:
                    worker = self._workers[worker_id]
                    worker.allocated_resources = worker.allocated_resources.add(
                        job.resources
                    )
                    worker.current_jobs.append(job.id)

        # Update quota usage
        user_quota = self._quotas.get(job.user_id)
        if user_quota:
            user_quota.allocate(job.resources)

        if job.team_id:
            team_quota = self._quotas.get(job.team_id)
            if team_quota:
                team_quota.allocate(job.resources)

    async def _release_resources(self, job: TrainingJob) -> None:
        """Release resources from a job."""
        async with self._worker_lock:
            for worker_id in job.assigned_workers:
                if worker_id in self._workers:
                    worker = self._workers[worker_id]
                    worker.allocated_resources = worker.resources.subtract(
                        worker.allocated_resources.subtract(job.resources)
                    )
                    if job.id in worker.current_jobs:
                        worker.current_jobs.remove(job.id)

        # Update quota usage
        user_quota = self._quotas.get(job.user_id)
        if user_quota:
            user_quota.release(job.resources)

        if job.team_id:
            team_quota = self._quotas.get(job.team_id)
            if team_quota:
                team_quota.release(job.resources)

    # Worker management

    async def register_worker(self, worker: WorkerInfo) -> None:
        """Register a new worker."""
        async with self._worker_lock:
            self._workers[worker.id] = worker
            logger.info(
                "worker_registered",
                worker_id=worker.id,
                hostname=worker.hostname,
                resources=worker.resources.model_dump(),
            )

    async def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        async with self._worker_lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info("worker_unregistered", worker_id=worker_id)

    async def update_worker(self, worker: WorkerInfo) -> None:
        """Update worker information."""
        async with self._worker_lock:
            if worker.id in self._workers:
                self._workers[worker.id] = worker

    async def worker_heartbeat(self, worker_id: str) -> bool:
        """Update worker heartbeat."""
        async with self._worker_lock:
            if worker_id in self._workers:
                self._workers[worker_id].last_heartbeat = datetime.utcnow()
                return True
            return False

    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker by ID."""
        async with self._worker_lock:
            return self._workers.get(worker_id)

    async def list_workers(self) -> list[WorkerInfo]:
        """List all workers."""
        async with self._worker_lock:
            return list(self._workers.values())

    async def get_healthy_workers(self) -> list[WorkerInfo]:
        """Get all healthy workers."""
        async with self._worker_lock:
            return [w for w in self._workers.values() if w.is_healthy()]

    # Quota management

    async def set_quota(self, quota: ResourceQuota) -> None:
        """Set a resource quota."""
        self._quotas[quota.entity_id] = quota
        logger.info(
            "quota_set",
            entity_id=quota.entity_id,
            entity_type=quota.entity_type,
        )

    async def get_quota(self, entity_id: str) -> Optional[ResourceQuota]:
        """Get quota for an entity."""
        return self._quotas.get(entity_id)

    async def remove_quota(self, entity_id: str) -> bool:
        """Remove a quota."""
        if entity_id in self._quotas:
            del self._quotas[entity_id]
            return True
        return False

    # Queue management

    async def queue_job(self, job: TrainingJob) -> None:
        """Manually add a job to the queue."""
        await self._queue.push(job)

    async def dequeue_job(self, job_id: str) -> bool:
        """Remove a job from the queue."""
        return await self._queue.remove(job_id)

    async def update_job_priority(
        self, job_id: str, priority: JobPriority
    ) -> bool:
        """Update a queued job's priority."""
        return await self._queue.update_priority(job_id, priority)

    async def get_queue_position(self, job_id: str) -> Optional[int]:
        """Get a job's position in the queue."""
        job_ids = await self._queue.get_all_job_ids()
        try:
            return job_ids.index(job_id)
        except ValueError:
            return None

    async def get_queue_size(self) -> int:
        """Get the current queue size."""
        return await self._queue.size()

    # Statistics

    async def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        queue_stats = await self._queue.get_stats()
        worker_count = len(self._workers)
        healthy_workers = len(await self.get_healthy_workers())

        return {
            "policy": self._policy_type.value,
            "running": self._running,
            "queue": queue_stats,
            "workers": {
                "total": worker_count,
                "healthy": healthy_workers,
            },
            "quotas": len(self._quotas),
            "metrics": self._metrics.copy(),
        }

    async def get_resource_utilization(self) -> dict[str, float]:
        """Get current resource utilization."""
        async with self._worker_lock:
            if not self._workers:
                return {"cpu": 0.0, "memory": 0.0, "gpu": 0.0}

            total_cpus = sum(w.resources.cpus for w in self._workers.values())
            used_cpus = sum(w.allocated_resources.cpus for w in self._workers.values())

            total_memory = sum(w.resources.memory_gb for w in self._workers.values())
            used_memory = sum(
                w.allocated_resources.memory_gb for w in self._workers.values()
            )

            total_gpus = sum(w.resources.gpus for w in self._workers.values())
            used_gpus = sum(w.allocated_resources.gpus for w in self._workers.values())

            return {
                "cpu": (used_cpus / total_cpus * 100) if total_cpus > 0 else 0.0,
                "memory": (used_memory / total_memory * 100) if total_memory > 0 else 0.0,
                "gpu": (used_gpus / total_gpus * 100) if total_gpus > 0 else 0.0,
            }
