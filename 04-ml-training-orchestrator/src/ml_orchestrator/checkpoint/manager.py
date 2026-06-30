"""Checkpoint management for training jobs."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional
import structlog

from ml_orchestrator.core.models import (
    Checkpoint,
    CheckpointConfig,
    CheckpointPolicy,
    CheckpointStatus,
    StorageBackend,
    TrainingJob,
)
from ml_orchestrator.core.exceptions import (
    CheckpointError,
    CheckpointNotFoundError,
)
from ml_orchestrator.checkpoint.storage import (
    CheckpointMetadata,
    CheckpointStorage,
    LocalStorage,
    S3Storage,
)


logger = structlog.get_logger(__name__)


@dataclass
class CheckpointRequest:
    """Request to create a checkpoint."""

    job_id: str
    epoch: int
    step: int
    data: bytes
    metrics: dict[str, float]
    priority: int = 0  # Higher = more important
    is_best: bool = False
    metadata: dict[str, Any] = None


class CheckpointManager:
    """
    Manages checkpoint lifecycle including creation, storage, and recovery.

    Supports:
    - Multiple storage backends (local, S3)
    - Checkpoint policies (periodic, best model, on-failure)
    - Automatic cleanup of old checkpoints
    - Distributed checkpoint coordination
    """

    def __init__(
        self,
        default_storage: Optional[CheckpointStorage] = None,
        local_path: str = "/checkpoints",
    ):
        self._storages: dict[StorageBackend, CheckpointStorage] = {}

        # Initialize default local storage
        self._local_storage = LocalStorage(local_path)
        self._storages[StorageBackend.LOCAL] = self._local_storage

        self._default_storage = default_storage or self._local_storage

        # Track checkpoints by job
        self._job_checkpoints: dict[str, list[Checkpoint]] = {}
        self._best_checkpoints: dict[str, dict[str, Checkpoint]] = {}  # job_id -> {metric: ckpt}
        self._lock = asyncio.Lock()

        # Callbacks
        self._callbacks: dict[str, list[Callable]] = {}

        # Metrics
        self._metrics = {
            "checkpoints_created": 0,
            "checkpoints_deleted": 0,
            "total_bytes_saved": 0,
            "total_bytes_deleted": 0,
        }

    def register_storage(self, backend: StorageBackend, storage: CheckpointStorage) -> None:
        """Register a storage backend."""
        self._storages[backend] = storage

    def get_storage(self, backend: StorageBackend) -> Optional[CheckpointStorage]:
        """Get a storage backend."""
        return self._storages.get(backend)

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for checkpoint events."""
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

    async def create_checkpoint(
        self,
        job: TrainingJob,
        data: bytes,
        epoch: int,
        step: int,
        metrics: Optional[dict[str, float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Create and save a checkpoint.

        Args:
            job: Training job
            data: Checkpoint binary data
            epoch: Current epoch
            step: Current step
            metrics: Metrics at checkpoint time
            metadata: Additional metadata

        Returns:
            Created Checkpoint
        """
        config = job.config.checkpoint
        storage = self._storages.get(config.storage_backend, self._default_storage)

        # Create checkpoint object
        checkpoint = Checkpoint(
            job_id=job.id,
            epoch=epoch,
            step=step,
            path="",  # Will be set after save
            size_bytes=len(data),
            metrics=metrics or {},
            status=CheckpointStatus.IN_PROGRESS,
        )

        try:
            # Save to storage
            path = await storage.save(checkpoint, data, metadata)
            checkpoint.path = path
            checkpoint.status = CheckpointStatus.COMPLETED

            # Track checkpoint
            async with self._lock:
                if job.id not in self._job_checkpoints:
                    self._job_checkpoints[job.id] = []
                self._job_checkpoints[job.id].append(checkpoint)

                # Update best checkpoint if applicable
                if metrics and config.metric_to_track:
                    await self._update_best_checkpoint(job.id, checkpoint, config)

                # Cleanup old checkpoints
                await self._cleanup_checkpoints(job.id, config.keep_last_n, storage)

            self._metrics["checkpoints_created"] += 1
            self._metrics["total_bytes_saved"] += len(data)

            logger.info(
                "checkpoint_created",
                job_id=job.id,
                checkpoint_id=checkpoint.id,
                epoch=epoch,
                step=step,
                path=path,
                size_bytes=len(data),
            )

            await self._emit_event("checkpoint_created", {"checkpoint": checkpoint})

            return checkpoint

        except Exception as e:
            checkpoint.status = CheckpointStatus.FAILED
            logger.error(
                "checkpoint_creation_failed",
                job_id=job.id,
                error=str(e),
            )
            raise CheckpointError(f"Failed to create checkpoint: {e}")

    async def _update_best_checkpoint(
        self,
        job_id: str,
        checkpoint: Checkpoint,
        config: CheckpointConfig,
    ) -> None:
        """Update best checkpoint tracking."""
        metric = config.metric_to_track
        if not metric or metric not in checkpoint.metrics:
            return

        if job_id not in self._best_checkpoints:
            self._best_checkpoints[job_id] = {}

        current_best = self._best_checkpoints[job_id].get(metric)
        new_value = checkpoint.metrics[metric]

        should_update = False
        if current_best is None:
            should_update = True
        elif config.metric_mode == "min":
            should_update = new_value < current_best.metrics.get(metric, float("inf"))
        else:  # max
            should_update = new_value > current_best.metrics.get(metric, float("-inf"))

        if should_update:
            self._best_checkpoints[job_id][metric] = checkpoint
            logger.info(
                "best_checkpoint_updated",
                job_id=job_id,
                metric=metric,
                value=new_value,
                checkpoint_id=checkpoint.id,
            )

    async def _cleanup_checkpoints(
        self,
        job_id: str,
        keep_last_n: int,
        storage: CheckpointStorage,
    ) -> None:
        """Clean up old checkpoints, keeping the last N."""
        checkpoints = self._job_checkpoints.get(job_id, [])

        if len(checkpoints) <= keep_last_n:
            return

        # Sort by step and keep last N
        sorted_ckpts = sorted(checkpoints, key=lambda c: (c.epoch, c.step))
        to_delete = sorted_ckpts[:-keep_last_n]

        # Protect best checkpoints
        best_ids = set()
        for metric_ckpts in self._best_checkpoints.get(job_id, {}).values():
            best_ids.add(metric_ckpts.id)

        for ckpt in to_delete:
            if ckpt.id in best_ids:
                continue  # Don't delete best checkpoints

            try:
                if await storage.delete(ckpt.path):
                    ckpt.status = CheckpointStatus.DELETED
                    self._metrics["checkpoints_deleted"] += 1
                    self._metrics["total_bytes_deleted"] += ckpt.size_bytes
            except Exception as e:
                logger.warning(
                    "checkpoint_cleanup_failed",
                    checkpoint_id=ckpt.id,
                    error=str(e),
                )

        # Update checkpoint list
        self._job_checkpoints[job_id] = [
            c for c in checkpoints if c.status != CheckpointStatus.DELETED
        ]

    async def load_checkpoint(
        self,
        job_id: str,
        checkpoint_id: Optional[str] = None,
        storage_backend: StorageBackend = StorageBackend.LOCAL,
    ) -> tuple[Checkpoint, bytes]:
        """
        Load a checkpoint.

        Args:
            job_id: Job ID
            checkpoint_id: Specific checkpoint ID, or None for latest
            storage_backend: Storage backend to use

        Returns:
            Tuple of (Checkpoint, data bytes)
        """
        storage = self._storages.get(storage_backend, self._default_storage)

        async with self._lock:
            checkpoints = self._job_checkpoints.get(job_id, [])

        if not checkpoints:
            # Try to list from storage
            meta_list = await storage.list_checkpoints(job_id)
            if not meta_list:
                raise CheckpointNotFoundError(checkpoint_id or "latest")

            # Convert to checkpoints
            checkpoints = [
                Checkpoint(
                    id=m.checkpoint_id,
                    job_id=m.job_id,
                    epoch=m.epoch,
                    step=m.step,
                    path=m.path,
                    size_bytes=m.size_bytes,
                    metrics=m.metrics,
                    status=CheckpointStatus.COMPLETED,
                )
                for m in meta_list
            ]

        # Find the checkpoint
        if checkpoint_id:
            checkpoint = next(
                (c for c in checkpoints if c.id == checkpoint_id), None
            )
            if not checkpoint:
                raise CheckpointNotFoundError(checkpoint_id)
        else:
            # Get latest
            valid = [c for c in checkpoints if c.status == CheckpointStatus.COMPLETED]
            if not valid:
                raise CheckpointNotFoundError("latest")
            checkpoint = max(valid, key=lambda c: (c.epoch, c.step))

        # Load data
        data = await storage.load(checkpoint.path)

        logger.info(
            "checkpoint_loaded",
            job_id=job_id,
            checkpoint_id=checkpoint.id,
            epoch=checkpoint.epoch,
            step=checkpoint.step,
        )

        return checkpoint, data

    async def get_best_checkpoint(
        self,
        job_id: str,
        metric: str,
        storage_backend: StorageBackend = StorageBackend.LOCAL,
    ) -> tuple[Optional[Checkpoint], Optional[bytes]]:
        """
        Get the best checkpoint for a metric.

        Returns:
            Tuple of (Checkpoint, data) or (None, None) if not found
        """
        async with self._lock:
            best = self._best_checkpoints.get(job_id, {}).get(metric)
            if not best:
                return None, None

        storage = self._storages.get(storage_backend, self._default_storage)
        data = await storage.load(best.path)

        return best, data

    async def list_checkpoints(
        self,
        job_id: str,
        limit: int = 100,
        include_deleted: bool = False,
    ) -> list[Checkpoint]:
        """List checkpoints for a job."""
        async with self._lock:
            checkpoints = self._job_checkpoints.get(job_id, [])

            if not include_deleted:
                checkpoints = [
                    c for c in checkpoints if c.status != CheckpointStatus.DELETED
                ]

            # Sort by step descending
            checkpoints = sorted(
                checkpoints,
                key=lambda c: (c.epoch, c.step),
                reverse=True,
            )

            return checkpoints[:limit]

    async def delete_checkpoint(
        self,
        job_id: str,
        checkpoint_id: str,
        storage_backend: StorageBackend = StorageBackend.LOCAL,
    ) -> bool:
        """Delete a specific checkpoint."""
        storage = self._storages.get(storage_backend, self._default_storage)

        async with self._lock:
            checkpoints = self._job_checkpoints.get(job_id, [])
            checkpoint = next(
                (c for c in checkpoints if c.id == checkpoint_id), None
            )

            if not checkpoint:
                return False

            if await storage.delete(checkpoint.path):
                checkpoint.status = CheckpointStatus.DELETED
                self._metrics["checkpoints_deleted"] += 1
                self._metrics["total_bytes_deleted"] += checkpoint.size_bytes

                # Update job checkpoints
                self._job_checkpoints[job_id] = [
                    c for c in checkpoints if c.id != checkpoint_id
                ]

                # Remove from best checkpoints if applicable
                if job_id in self._best_checkpoints:
                    for metric in list(self._best_checkpoints[job_id].keys()):
                        if self._best_checkpoints[job_id][metric].id == checkpoint_id:
                            del self._best_checkpoints[job_id][metric]

                logger.info(
                    "checkpoint_deleted",
                    job_id=job_id,
                    checkpoint_id=checkpoint_id,
                )

                return True

            return False

    async def cleanup_job_checkpoints(
        self,
        job_id: str,
        storage_backend: StorageBackend = StorageBackend.LOCAL,
    ) -> int:
        """Delete all checkpoints for a job."""
        storage = self._storages.get(storage_backend, self._default_storage)

        async with self._lock:
            checkpoints = self._job_checkpoints.pop(job_id, [])
            self._best_checkpoints.pop(job_id, None)

        deleted = 0
        for checkpoint in checkpoints:
            try:
                if await storage.delete(checkpoint.path):
                    deleted += 1
                    self._metrics["checkpoints_deleted"] += 1
                    self._metrics["total_bytes_deleted"] += checkpoint.size_bytes
            except Exception as e:
                logger.warning(
                    "checkpoint_cleanup_failed",
                    checkpoint_id=checkpoint.id,
                    error=str(e),
                )

        logger.info(
            "job_checkpoints_cleaned",
            job_id=job_id,
            deleted=deleted,
        )

        return deleted

    async def should_checkpoint(
        self,
        job: TrainingJob,
        epoch: int,
        step: int,
        metrics: Optional[dict[str, float]] = None,
    ) -> bool:
        """
        Determine if a checkpoint should be created based on policy.

        Args:
            job: Training job
            epoch: Current epoch
            step: Current step
            metrics: Current metrics

        Returns:
            True if checkpoint should be created
        """
        config = job.config.checkpoint

        if not config.enabled:
            return False

        if config.policy == CheckpointPolicy.PERIODIC:
            # Check epoch interval
            if config.interval_epochs and epoch > 0:
                if epoch % config.interval_epochs == 0:
                    # Check if we already have a checkpoint for this epoch
                    async with self._lock:
                        checkpoints = self._job_checkpoints.get(job.id, [])
                        existing = [c for c in checkpoints if c.epoch == epoch]
                        if not existing:
                            return True

            # Check step interval
            if config.interval_steps and step > 0:
                if step % config.interval_steps == 0:
                    return True

        elif config.policy == CheckpointPolicy.BEST_METRIC:
            if not metrics or not config.metric_to_track:
                return False

            metric = config.metric_to_track
            if metric not in metrics:
                return False

            async with self._lock:
                best = self._best_checkpoints.get(job.id, {}).get(metric)

            if best is None:
                return True

            current_value = metrics[metric]
            best_value = best.metrics.get(metric)

            if best_value is None:
                return True

            if config.metric_mode == "min":
                return current_value < best_value
            else:
                return current_value > best_value

        elif config.policy == CheckpointPolicy.MANUAL:
            return False  # Only checkpoint on explicit request

        return False

    async def get_resume_checkpoint(
        self,
        job: TrainingJob,
    ) -> Optional[tuple[Checkpoint, bytes]]:
        """
        Get the checkpoint to resume training from.

        Returns:
            Tuple of (Checkpoint, data) or None if starting fresh
        """
        # Check if job specifies a checkpoint to resume from
        if job.resume_from_checkpoint:
            try:
                return await self.load_checkpoint(
                    job.id,
                    job.resume_from_checkpoint,
                    job.config.checkpoint.storage_backend,
                )
            except CheckpointNotFoundError:
                logger.warning(
                    "resume_checkpoint_not_found",
                    job_id=job.id,
                    checkpoint_id=job.resume_from_checkpoint,
                )
                return None

        # Otherwise try to get latest
        try:
            return await self.load_checkpoint(
                job.id,
                storage_backend=job.config.checkpoint.storage_backend,
            )
        except CheckpointNotFoundError:
            return None

    async def get_stats(self) -> dict[str, Any]:
        """Get checkpoint manager statistics."""
        async with self._lock:
            total_checkpoints = sum(
                len(ckpts) for ckpts in self._job_checkpoints.values()
            )
            jobs_with_checkpoints = len(self._job_checkpoints)

        return {
            "total_checkpoints": total_checkpoints,
            "jobs_with_checkpoints": jobs_with_checkpoints,
            "storage_backends": list(self._storages.keys()),
            **self._metrics,
        }
