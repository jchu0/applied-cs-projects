"""Checkpoint coordination for distributed training."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4
import structlog

from ml_orchestrator.core.models import Checkpoint, CheckpointStatus


logger = structlog.get_logger(__name__)


class CoordinatedCheckpointState(str, Enum):
    """State of a coordinated checkpoint."""

    PENDING = "pending"
    BARRIER_WAIT = "barrier_wait"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class WorkerCheckpointState:
    """State of a worker in coordinated checkpointing."""

    worker_id: str
    rank: int
    state: CoordinatedCheckpointState = CoordinatedCheckpointState.PENDING
    checkpoint_path: Optional[str] = None
    size_bytes: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class CoordinatedCheckpoint:
    """A checkpoint coordinated across multiple workers."""

    id: str = field(default_factory=lambda: str(uuid4()))
    job_id: str = ""
    epoch: int = 0
    step: int = 0
    world_size: int = 1
    state: CoordinatedCheckpointState = CoordinatedCheckpointState.PENDING
    workers: dict[str, WorkerCheckpointState] = field(default_factory=dict)
    initiated_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.state in (
            CoordinatedCheckpointState.COMPLETED,
            CoordinatedCheckpointState.FAILED,
            CoordinatedCheckpointState.TIMEOUT,
        )

    @property
    def all_workers_ready(self) -> bool:
        if len(self.workers) < self.world_size:
            return False
        return all(
            w.state != CoordinatedCheckpointState.PENDING
            for w in self.workers.values()
        )

    @property
    def all_workers_saved(self) -> bool:
        if len(self.workers) < self.world_size:
            return False
        return all(
            w.state == CoordinatedCheckpointState.COMPLETED
            for w in self.workers.values()
        )

    @property
    def is_expired(self) -> bool:
        if self.is_complete:
            return False
        elapsed = (datetime.utcnow() - self.initiated_at).total_seconds()
        return elapsed > self.timeout_seconds


class CheckpointCoordinator:
    """
    Coordinates checkpointing across distributed training workers.

    Implements a barrier-based protocol:
    1. Leader initiates checkpoint
    2. All workers pause training and acknowledge
    3. All workers save their local state
    4. Workers report completion
    5. Leader records global checkpoint metadata
    6. Training resumes
    """

    def __init__(
        self,
        default_timeout_seconds: int = 300,
    ):
        self._checkpoints: dict[str, CoordinatedCheckpoint] = {}
        self._active_checkpoints: dict[str, str] = {}  # job_id -> checkpoint_id
        self._timeout = default_timeout_seconds
        self._lock = asyncio.Lock()
        self._callbacks: dict[str, list[Callable]] = {}

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for coordination events."""
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

    async def initiate_checkpoint(
        self,
        job_id: str,
        epoch: int,
        step: int,
        world_size: int,
        timeout_seconds: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> CoordinatedCheckpoint:
        """
        Initiate a coordinated checkpoint.

        Should be called by the master/leader worker.

        Args:
            job_id: Job ID
            epoch: Current epoch
            step: Current step
            world_size: Number of workers
            timeout_seconds: Timeout for coordination
            metadata: Additional metadata

        Returns:
            CoordinatedCheckpoint object
        """
        async with self._lock:
            # Check for existing active checkpoint
            if job_id in self._active_checkpoints:
                existing = self._checkpoints.get(self._active_checkpoints[job_id])
                if existing and not existing.is_complete:
                    raise ValueError(
                        f"Checkpoint already in progress for job {job_id}"
                    )

            checkpoint = CoordinatedCheckpoint(
                job_id=job_id,
                epoch=epoch,
                step=step,
                world_size=world_size,
                timeout_seconds=timeout_seconds or self._timeout,
                metadata=metadata or {},
            )

            self._checkpoints[checkpoint.id] = checkpoint
            self._active_checkpoints[job_id] = checkpoint.id

            logger.info(
                "coordinated_checkpoint_initiated",
                checkpoint_id=checkpoint.id,
                job_id=job_id,
                epoch=epoch,
                step=step,
                world_size=world_size,
            )

            await self._emit_event(
                "checkpoint_initiated",
                {"checkpoint": checkpoint},
            )

            return checkpoint

    async def worker_acknowledge(
        self,
        checkpoint_id: str,
        worker_id: str,
        rank: int,
    ) -> bool:
        """
        Worker acknowledges checkpoint request and enters barrier.

        Args:
            checkpoint_id: Checkpoint ID
            worker_id: Worker ID
            rank: Worker rank

        Returns:
            True if acknowledged successfully
        """
        async with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return False

            if checkpoint.is_complete or checkpoint.is_expired:
                return False

            if worker_id in checkpoint.workers:
                return True  # Already acknowledged

            checkpoint.workers[worker_id] = WorkerCheckpointState(
                worker_id=worker_id,
                rank=rank,
                state=CoordinatedCheckpointState.BARRIER_WAIT,
                started_at=datetime.utcnow(),
            )

            # Check if all workers have acknowledged
            if len(checkpoint.workers) >= checkpoint.world_size:
                checkpoint.state = CoordinatedCheckpointState.SAVING
                logger.info(
                    "all_workers_acknowledged",
                    checkpoint_id=checkpoint_id,
                    workers=len(checkpoint.workers),
                )

            return True

    async def wait_for_barrier(
        self,
        checkpoint_id: str,
        worker_id: str,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Wait for all workers to reach checkpoint barrier.

        Args:
            checkpoint_id: Checkpoint ID
            worker_id: Worker ID
            timeout: Timeout in seconds

        Returns:
            True when all workers are ready, False on timeout
        """
        async with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return False
            if checkpoint.is_complete:
                return checkpoint.state == CoordinatedCheckpointState.COMPLETED

        timeout = timeout or checkpoint.timeout_seconds
        start = datetime.utcnow()

        while True:
            async with self._lock:
                checkpoint = self._checkpoints.get(checkpoint_id)
                if not checkpoint:
                    return False

                if checkpoint.is_expired:
                    checkpoint.state = CoordinatedCheckpointState.TIMEOUT
                    return False

                if checkpoint.state == CoordinatedCheckpointState.SAVING:
                    return True

            elapsed = (datetime.utcnow() - start).total_seconds()
            if elapsed > timeout:
                return False

            await asyncio.sleep(0.1)

    async def worker_saved(
        self,
        checkpoint_id: str,
        worker_id: str,
        checkpoint_path: str,
        size_bytes: int = 0,
    ) -> bool:
        """
        Worker reports checkpoint saved successfully.

        Args:
            checkpoint_id: Checkpoint ID
            worker_id: Worker ID
            checkpoint_path: Path where checkpoint was saved
            size_bytes: Size of checkpoint

        Returns:
            True if reported successfully
        """
        async with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return False

            worker = checkpoint.workers.get(worker_id)
            if not worker:
                return False

            worker.state = CoordinatedCheckpointState.COMPLETED
            worker.checkpoint_path = checkpoint_path
            worker.size_bytes = size_bytes
            worker.completed_at = datetime.utcnow()

            logger.debug(
                "worker_checkpoint_saved",
                checkpoint_id=checkpoint_id,
                worker_id=worker_id,
                path=checkpoint_path,
            )

            # Check if all workers have saved
            if checkpoint.all_workers_saved:
                checkpoint.state = CoordinatedCheckpointState.COMPLETED
                checkpoint.completed_at = datetime.utcnow()

                logger.info(
                    "coordinated_checkpoint_complete",
                    checkpoint_id=checkpoint_id,
                    job_id=checkpoint.job_id,
                    epoch=checkpoint.epoch,
                    step=checkpoint.step,
                    duration_ms=(
                        checkpoint.completed_at - checkpoint.initiated_at
                    ).total_seconds() * 1000,
                )

                await self._emit_event(
                    "checkpoint_complete",
                    {"checkpoint": checkpoint},
                )

            return True

    async def worker_failed(
        self,
        checkpoint_id: str,
        worker_id: str,
        error: str,
    ) -> None:
        """
        Worker reports checkpoint failure.

        Args:
            checkpoint_id: Checkpoint ID
            worker_id: Worker ID
            error: Error message
        """
        async with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return

            worker = checkpoint.workers.get(worker_id)
            if not worker:
                return

            worker.state = CoordinatedCheckpointState.FAILED
            worker.error = error
            worker.completed_at = datetime.utcnow()

            # Mark entire checkpoint as failed
            checkpoint.state = CoordinatedCheckpointState.FAILED
            checkpoint.completed_at = datetime.utcnow()

            logger.error(
                "coordinated_checkpoint_failed",
                checkpoint_id=checkpoint_id,
                worker_id=worker_id,
                error=error,
            )

            await self._emit_event(
                "checkpoint_failed",
                {"checkpoint": checkpoint, "worker_id": worker_id, "error": error},
            )

    async def wait_for_completion(
        self,
        checkpoint_id: str,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Wait for coordinated checkpoint to complete.

        Args:
            checkpoint_id: Checkpoint ID
            timeout: Timeout in seconds

        Returns:
            True if completed successfully
        """
        async with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return False
            if checkpoint.is_complete:
                return checkpoint.state == CoordinatedCheckpointState.COMPLETED

        timeout = timeout or checkpoint.timeout_seconds
        start = datetime.utcnow()

        while True:
            async with self._lock:
                checkpoint = self._checkpoints.get(checkpoint_id)
                if not checkpoint:
                    return False

                if checkpoint.is_complete:
                    return checkpoint.state == CoordinatedCheckpointState.COMPLETED

                if checkpoint.is_expired:
                    checkpoint.state = CoordinatedCheckpointState.TIMEOUT
                    return False

            elapsed = (datetime.utcnow() - start).total_seconds()
            if elapsed > timeout:
                return False

            await asyncio.sleep(0.1)

    async def get_checkpoint(
        self,
        checkpoint_id: str,
    ) -> Optional[CoordinatedCheckpoint]:
        """Get checkpoint by ID."""
        async with self._lock:
            return self._checkpoints.get(checkpoint_id)

    async def get_active_checkpoint(
        self,
        job_id: str,
    ) -> Optional[CoordinatedCheckpoint]:
        """Get active checkpoint for a job."""
        async with self._lock:
            checkpoint_id = self._active_checkpoints.get(job_id)
            if checkpoint_id:
                return self._checkpoints.get(checkpoint_id)
            return None

    async def get_worker_paths(
        self,
        checkpoint_id: str,
    ) -> dict[int, str]:
        """
        Get checkpoint paths for all workers.

        Returns:
            Dict mapping rank to checkpoint path
        """
        async with self._lock:
            checkpoint = self._checkpoints.get(checkpoint_id)
            if not checkpoint:
                return {}

            return {
                w.rank: w.checkpoint_path
                for w in checkpoint.workers.values()
                if w.checkpoint_path
            }

    async def cleanup_checkpoint(self, checkpoint_id: str) -> None:
        """Clean up checkpoint resources."""
        async with self._lock:
            checkpoint = self._checkpoints.pop(checkpoint_id, None)
            if checkpoint:
                if self._active_checkpoints.get(checkpoint.job_id) == checkpoint_id:
                    del self._active_checkpoints[checkpoint.job_id]

    async def cleanup_job(self, job_id: str) -> int:
        """Clean up all checkpoints for a job."""
        async with self._lock:
            self._active_checkpoints.pop(job_id, None)

            to_remove = [
                cid for cid, c in self._checkpoints.items()
                if c.job_id == job_id
            ]
            for cid in to_remove:
                del self._checkpoints[cid]

            return len(to_remove)

    async def check_timeouts(self) -> list[str]:
        """Check for timed out checkpoints."""
        timed_out = []

        async with self._lock:
            for checkpoint_id, checkpoint in self._checkpoints.items():
                if checkpoint.is_expired and not checkpoint.is_complete:
                    checkpoint.state = CoordinatedCheckpointState.TIMEOUT
                    checkpoint.completed_at = datetime.utcnow()
                    timed_out.append(checkpoint_id)

                    logger.warning(
                        "coordinated_checkpoint_timeout",
                        checkpoint_id=checkpoint_id,
                        job_id=checkpoint.job_id,
                    )

        return timed_out

    async def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics."""
        async with self._lock:
            total = len(self._checkpoints)
            active = len(self._active_checkpoints)
            completed = sum(
                1 for c in self._checkpoints.values()
                if c.state == CoordinatedCheckpointState.COMPLETED
            )
            failed = sum(
                1 for c in self._checkpoints.values()
                if c.state in (
                    CoordinatedCheckpointState.FAILED,
                    CoordinatedCheckpointState.TIMEOUT,
                )
            )

            return {
                "total_checkpoints": total,
                "active_checkpoints": active,
                "completed_checkpoints": completed,
                "failed_checkpoints": failed,
            }
