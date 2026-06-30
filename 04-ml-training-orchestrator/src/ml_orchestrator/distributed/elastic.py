"""Elastic training support for dynamic worker scaling."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4
import structlog

from ml_orchestrator.core.models import DistributedConfig, TrainingJob


logger = structlog.get_logger(__name__)


class MembershipChangeType(str, Enum):
    """Type of membership change in elastic training."""

    WORKER_JOINED = "worker_joined"
    WORKER_LEFT = "worker_left"
    WORKER_FAILED = "worker_failed"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"


@dataclass
class MembershipChange:
    """Record of a membership change."""

    id: str = field(default_factory=lambda: str(uuid4()))
    job_id: str = ""
    change_type: MembershipChangeType = MembershipChangeType.WORKER_JOINED
    worker_id: Optional[str] = None
    old_world_size: int = 0
    new_world_size: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ElasticState:
    """State for elastic training job."""

    job_id: str
    min_workers: int
    max_workers: int
    current_workers: int = 0
    target_workers: int = 0
    last_membership_change: Optional[datetime] = None
    total_restarts: int = 0
    active_workers: set[str] = field(default_factory=set)
    pending_workers: set[str] = field(default_factory=set)
    failed_workers: set[str] = field(default_factory=set)


class ElasticTrainingManager:
    """
    Manages elastic training with dynamic worker scaling.

    Supports:
    - Adding workers to running training
    - Removing workers gracefully
    - Handling worker failures
    - Automatic rescaling based on policies
    """

    def __init__(
        self,
        min_restart_interval_seconds: int = 30,
        max_restarts: int = 10,
        grace_period_seconds: int = 300,
    ):
        self._jobs: dict[str, ElasticState] = {}
        self._changes: list[MembershipChange] = []
        self._min_restart_interval = min_restart_interval_seconds
        self._max_restarts = max_restarts
        self._grace_period = grace_period_seconds
        self._lock = asyncio.Lock()
        self._callbacks: dict[str, list[Callable]] = {}

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for elastic events."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    async def _emit_event(self, event: str, data: dict[str, Any]) -> None:
        """Emit event to callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error("callback_error", event=event, error=str(e))

    async def initialize_elastic_job(
        self,
        job: TrainingJob,
    ) -> ElasticState:
        """
        Initialize elastic training for a job.

        Args:
            job: Training job with elastic config

        Returns:
            ElasticState for the job
        """
        async with self._lock:
            config = job.config.distributed

            if not config.elastic:
                raise ValueError("Job does not have elastic training enabled")

            state = ElasticState(
                job_id=job.id,
                min_workers=config.min_nodes or 1,
                max_workers=config.max_nodes or config.world_size,
                target_workers=config.world_size,
            )

            self._jobs[job.id] = state

            logger.info(
                "elastic_job_initialized",
                job_id=job.id,
                min_workers=state.min_workers,
                max_workers=state.max_workers,
            )

            return state

    async def get_state(self, job_id: str) -> Optional[ElasticState]:
        """Get elastic state for a job."""
        async with self._lock:
            return self._jobs.get(job_id)

    async def worker_joined(
        self,
        job_id: str,
        worker_id: str,
    ) -> Optional[MembershipChange]:
        """
        Record a worker joining elastic training.

        Returns:
            MembershipChange record or None if at capacity
        """
        async with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return None

            if state.current_workers >= state.max_workers:
                logger.warning(
                    "elastic_at_capacity",
                    job_id=job_id,
                    current=state.current_workers,
                    max=state.max_workers,
                )
                return None

            old_size = state.current_workers
            state.active_workers.add(worker_id)
            state.pending_workers.discard(worker_id)
            state.current_workers = len(state.active_workers)

            change = MembershipChange(
                job_id=job_id,
                change_type=MembershipChangeType.WORKER_JOINED,
                worker_id=worker_id,
                old_world_size=old_size,
                new_world_size=state.current_workers,
            )
            self._changes.append(change)
            state.last_membership_change = change.timestamp

            logger.info(
                "elastic_worker_joined",
                job_id=job_id,
                worker_id=worker_id,
                world_size=state.current_workers,
            )

            await self._emit_event(
                "worker_joined",
                {"change": change, "state": state},
            )

            return change

    async def worker_left(
        self,
        job_id: str,
        worker_id: str,
        graceful: bool = True,
    ) -> Optional[MembershipChange]:
        """
        Record a worker leaving elastic training.

        Args:
            job_id: Job ID
            worker_id: Worker ID
            graceful: Whether this is a graceful departure

        Returns:
            MembershipChange record
        """
        async with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return None

            if worker_id not in state.active_workers:
                return None

            old_size = state.current_workers
            state.active_workers.discard(worker_id)
            state.current_workers = len(state.active_workers)

            change_type = (
                MembershipChangeType.WORKER_LEFT
                if graceful
                else MembershipChangeType.WORKER_FAILED
            )

            if not graceful:
                state.failed_workers.add(worker_id)

            change = MembershipChange(
                job_id=job_id,
                change_type=change_type,
                worker_id=worker_id,
                old_world_size=old_size,
                new_world_size=state.current_workers,
            )
            self._changes.append(change)
            state.last_membership_change = change.timestamp

            logger.info(
                "elastic_worker_left",
                job_id=job_id,
                worker_id=worker_id,
                graceful=graceful,
                world_size=state.current_workers,
            )

            # Check if we fell below minimum
            if state.current_workers < state.min_workers:
                logger.warning(
                    "elastic_below_minimum",
                    job_id=job_id,
                    current=state.current_workers,
                    min=state.min_workers,
                )

            await self._emit_event(
                "worker_left",
                {"change": change, "state": state, "graceful": graceful},
            )

            return change

    async def request_scale(
        self,
        job_id: str,
        target_workers: int,
    ) -> bool:
        """
        Request scaling to a target number of workers.

        Args:
            job_id: Job ID
            target_workers: Desired number of workers

        Returns:
            True if scaling request accepted
        """
        async with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return False

            # Clamp to valid range
            target = max(state.min_workers, min(state.max_workers, target_workers))

            if target == state.target_workers:
                return True

            old_target = state.target_workers
            state.target_workers = target

            change_type = (
                MembershipChangeType.SCALE_UP
                if target > old_target
                else MembershipChangeType.SCALE_DOWN
            )

            change = MembershipChange(
                job_id=job_id,
                change_type=change_type,
                old_world_size=old_target,
                new_world_size=target,
            )
            self._changes.append(change)

            logger.info(
                "elastic_scale_requested",
                job_id=job_id,
                old_target=old_target,
                new_target=target,
                direction="up" if target > old_target else "down",
            )

            await self._emit_event("scale_requested", {"change": change, "state": state})

            return True

    async def can_restart(self, job_id: str) -> bool:
        """
        Check if job can restart after failure.

        Returns:
            True if restart is allowed
        """
        async with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return False

            # Check restart limit
            if state.total_restarts >= self._max_restarts:
                logger.warning(
                    "elastic_restart_limit_reached",
                    job_id=job_id,
                    restarts=state.total_restarts,
                )
                return False

            # Check restart interval
            if state.last_membership_change:
                elapsed = (
                    datetime.utcnow() - state.last_membership_change
                ).total_seconds()
                if elapsed < self._min_restart_interval:
                    logger.debug(
                        "elastic_restart_too_soon",
                        job_id=job_id,
                        elapsed=elapsed,
                        min_interval=self._min_restart_interval,
                    )
                    return False

            return True

    async def record_restart(self, job_id: str) -> bool:
        """Record a training restart."""
        async with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return False

            state.total_restarts += 1
            state.last_membership_change = datetime.utcnow()

            logger.info(
                "elastic_restart_recorded",
                job_id=job_id,
                total_restarts=state.total_restarts,
            )

            return True

    async def get_scaling_decision(
        self,
        job_id: str,
    ) -> tuple[int, int]:
        """
        Get scaling decision (workers to add, workers to remove).

        Returns:
            Tuple of (workers_to_add, workers_to_remove)
        """
        async with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return 0, 0

            diff = state.target_workers - state.current_workers

            if diff > 0:
                return diff, 0  # Need to add workers
            elif diff < 0:
                return 0, -diff  # Need to remove workers
            else:
                return 0, 0  # At target

    async def is_healthy(self, job_id: str) -> bool:
        """Check if elastic job is healthy."""
        async with self._lock:
            state = self._jobs.get(job_id)
            if not state:
                return False

            # Healthy if at or above minimum workers
            return state.current_workers >= state.min_workers

    async def get_membership_changes(
        self,
        job_id: str,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> list[MembershipChange]:
        """Get membership changes for a job."""
        changes = [c for c in self._changes if c.job_id == job_id]
        if since:
            changes = [c for c in changes if c.timestamp > since]
        return changes[-limit:]

    async def cleanup_job(self, job_id: str) -> None:
        """Clean up elastic state for a job."""
        async with self._lock:
            self._jobs.pop(job_id, None)
            # Keep changes for history
            logger.info("elastic_job_cleaned_up", job_id=job_id)

    async def get_stats(self) -> dict[str, Any]:
        """Get elastic training statistics."""
        async with self._lock:
            total_workers = sum(s.current_workers for s in self._jobs.values())
            total_restarts = sum(s.total_restarts for s in self._jobs.values())

            return {
                "active_jobs": len(self._jobs),
                "total_workers": total_workers,
                "total_restarts": total_restarts,
                "total_changes": len(self._changes),
                "jobs": {
                    job_id: {
                        "current": state.current_workers,
                        "target": state.target_workers,
                        "min": state.min_workers,
                        "max": state.max_workers,
                        "restarts": state.total_restarts,
                    }
                    for job_id, state in self._jobs.items()
                },
            }
