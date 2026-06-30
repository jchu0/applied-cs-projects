"""Distributed training coordinator."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4
import structlog

from ml_orchestrator.core.models import DistributedConfig, TrainingJob


logger = structlog.get_logger(__name__)


class WorkerRole(str, Enum):
    """Role of a worker in distributed training."""

    MASTER = "master"
    WORKER = "worker"


class WorkerState(str, Enum):
    """State of a worker in a distributed training job."""

    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    BARRIER_WAITING = "barrier_waiting"
    CHECKPOINTING = "checkpointing"
    FAILED = "failed"
    FINISHED = "finished"


@dataclass
class DistributedWorker:
    """Representation of a worker in distributed training."""

    id: str = field(default_factory=lambda: str(uuid4()))
    job_id: str = ""
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    node_id: str = ""
    hostname: str = ""
    ip_address: str = ""
    port: int = 29500
    role: WorkerRole = WorkerRole.WORKER
    state: WorkerState = WorkerState.INITIALIZING
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    current_step: int = 0
    current_epoch: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_master(self) -> bool:
        return self.rank == 0 or self.role == WorkerRole.MASTER


@dataclass
class TrainingBarrier:
    """Synchronization barrier for distributed training."""

    id: str = field(default_factory=lambda: str(uuid4()))
    job_id: str = ""
    barrier_type: str = "checkpoint"  # checkpoint, epoch, custom
    expected_workers: int = 0
    arrived_workers: set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    timeout_seconds: int = 300
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return len(self.arrived_workers) >= self.expected_workers

    @property
    def is_expired(self) -> bool:
        if self.completed_at:
            return False
        age = (datetime.utcnow() - self.created_at).total_seconds()
        return age > self.timeout_seconds

    def arrive(self, worker_id: str) -> bool:
        """Mark worker as arrived at barrier."""
        if worker_id in self.arrived_workers:
            return False
        self.arrived_workers.add(worker_id)
        if self.is_complete:
            self.completed_at = datetime.utcnow()
        return True


class DistributedCoordinator:
    """
    Coordinates distributed training across multiple workers.

    Responsibilities:
    - Worker registration and rank assignment
    - Rendezvous coordination
    - Barrier synchronization
    - Failure detection and handling
    - Collective operation coordination
    """

    def __init__(
        self,
        heartbeat_timeout_seconds: int = 30,
        barrier_timeout_seconds: int = 300,
        rendezvous_timeout_seconds: int = 600,
    ):
        self._jobs: dict[str, list[DistributedWorker]] = {}  # job_id -> workers
        self._workers: dict[str, DistributedWorker] = {}  # worker_id -> worker
        self._barriers: dict[str, TrainingBarrier] = {}  # barrier_id -> barrier
        self._job_barriers: dict[str, list[str]] = {}  # job_id -> barrier_ids
        self._rendezvous_events: dict[str, asyncio.Event] = {}  # job_id -> event
        self._heartbeat_timeout = heartbeat_timeout_seconds
        self._barrier_timeout = barrier_timeout_seconds
        self._rendezvous_timeout = rendezvous_timeout_seconds
        self._lock = asyncio.Lock()
        self._callbacks: dict[str, list[Callable]] = {}

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for events."""
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

    async def initialize_job(
        self,
        job: TrainingJob,
    ) -> dict[str, Any]:
        """
        Initialize distributed training for a job.

        Returns:
            Rendezvous information for workers
        """
        async with self._lock:
            job_id = job.id
            config = job.config.distributed

            if job_id in self._jobs:
                raise ValueError(f"Job {job_id} already initialized")

            self._jobs[job_id] = []
            self._job_barriers[job_id] = []
            self._rendezvous_events[job_id] = asyncio.Event()

            # Determine master address
            master_addr = config.master_addr or "localhost"
            master_port = config.master_port

            logger.info(
                "distributed_job_initialized",
                job_id=job_id,
                world_size=config.world_size,
                master=f"{master_addr}:{master_port}",
            )

            return {
                "job_id": job_id,
                "world_size": config.world_size,
                "master_addr": master_addr,
                "master_port": master_port,
                "backend": config.backend,
                "rdzv_backend": config.rdzv_backend,
            }

    async def register_worker(
        self,
        job_id: str,
        node_id: str,
        hostname: str,
        ip_address: str,
        port: int = 29500,
        local_rank: int = 0,
    ) -> DistributedWorker:
        """
        Register a worker for distributed training.

        Assigns a global rank and adds to job's worker list.
        """
        async with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not initialized")

            workers = self._jobs[job_id]
            rank = len(workers)  # Assign next available rank

            worker = DistributedWorker(
                job_id=job_id,
                rank=rank,
                local_rank=local_rank,
                world_size=len(workers) + 1,  # Will be updated
                node_id=node_id,
                hostname=hostname,
                ip_address=ip_address,
                port=port,
                role=WorkerRole.MASTER if rank == 0 else WorkerRole.WORKER,
                state=WorkerState.INITIALIZING,
            )

            workers.append(worker)
            self._workers[worker.id] = worker

            # Update world size for all workers
            for w in workers:
                w.world_size = len(workers)

            logger.info(
                "distributed_worker_registered",
                job_id=job_id,
                worker_id=worker.id,
                rank=rank,
            )

            await self._emit_event(
                "worker_registered",
                {"job_id": job_id, "worker_id": worker.id, "rank": rank},
            )

            return worker

    async def unregister_worker(self, worker_id: str) -> bool:
        """Unregister a worker."""
        async with self._lock:
            worker = self._workers.pop(worker_id, None)
            if not worker:
                return False

            if worker.job_id in self._jobs:
                self._jobs[worker.job_id] = [
                    w for w in self._jobs[worker.job_id] if w.id != worker_id
                ]

            logger.info("distributed_worker_unregistered", worker_id=worker_id)
            return True

    async def wait_for_rendezvous(
        self,
        job_id: str,
        expected_workers: int,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Wait for all workers to rendezvous.

        Args:
            job_id: Job ID
            expected_workers: Number of workers to wait for
            timeout: Timeout in seconds

        Returns:
            True if all workers joined, False on timeout
        """
        timeout = timeout or self._rendezvous_timeout

        async with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")

        start = datetime.utcnow()
        while True:
            async with self._lock:
                workers = self._jobs.get(job_id, [])
                ready_workers = [
                    w for w in workers
                    if w.state in (WorkerState.READY, WorkerState.TRAINING)
                ]
                if len(ready_workers) >= expected_workers:
                    self._rendezvous_events[job_id].set()
                    logger.info(
                        "rendezvous_complete",
                        job_id=job_id,
                        workers=len(ready_workers),
                    )
                    return True

            if (datetime.utcnow() - start).total_seconds() > timeout:
                logger.warning("rendezvous_timeout", job_id=job_id)
                return False

            await asyncio.sleep(0.1)

    async def worker_ready(self, worker_id: str) -> bool:
        """Mark worker as ready after initialization."""
        async with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False

            worker.state = WorkerState.READY
            worker.last_heartbeat = datetime.utcnow()

            logger.debug("worker_ready", worker_id=worker_id, rank=worker.rank)
            return True

    async def worker_heartbeat(
        self,
        worker_id: str,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> bool:
        """Update worker heartbeat."""
        async with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return False

            worker.last_heartbeat = datetime.utcnow()
            if step is not None:
                worker.current_step = step
            if epoch is not None:
                worker.current_epoch = epoch

            return True

    async def create_barrier(
        self,
        job_id: str,
        barrier_type: str = "checkpoint",
        timeout_seconds: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> TrainingBarrier:
        """
        Create a synchronization barrier for a job.

        All workers must call arrive_barrier before any can proceed.
        """
        async with self._lock:
            if job_id not in self._jobs:
                raise ValueError(f"Job {job_id} not found")

            workers = self._jobs[job_id]
            barrier = TrainingBarrier(
                job_id=job_id,
                barrier_type=barrier_type,
                expected_workers=len(workers),
                timeout_seconds=timeout_seconds or self._barrier_timeout,
                metadata=metadata or {},
            )

            self._barriers[barrier.id] = barrier
            if job_id not in self._job_barriers:
                self._job_barriers[job_id] = []
            self._job_barriers[job_id].append(barrier.id)

            logger.info(
                "barrier_created",
                barrier_id=barrier.id,
                job_id=job_id,
                type=barrier_type,
                expected=len(workers),
            )

            return barrier

    async def arrive_barrier(
        self,
        barrier_id: str,
        worker_id: str,
    ) -> tuple[bool, bool]:
        """
        Worker arrives at a barrier.

        Args:
            barrier_id: Barrier ID
            worker_id: Worker ID

        Returns:
            Tuple of (arrived_successfully, barrier_complete)
        """
        async with self._lock:
            barrier = self._barriers.get(barrier_id)
            if not barrier:
                return False, False

            if barrier.is_expired:
                logger.warning(
                    "barrier_expired",
                    barrier_id=barrier_id,
                )
                return False, False

            arrived = barrier.arrive(worker_id)

            if barrier.is_complete:
                logger.info(
                    "barrier_complete",
                    barrier_id=barrier_id,
                    job_id=barrier.job_id,
                )
                await self._emit_event(
                    "barrier_complete",
                    {"barrier_id": barrier_id, "job_id": barrier.job_id},
                )

            return arrived, barrier.is_complete

    async def wait_barrier(
        self,
        barrier_id: str,
        worker_id: str,
        timeout: Optional[int] = None,
    ) -> bool:
        """
        Wait for barrier to complete.

        Args:
            barrier_id: Barrier ID
            worker_id: Worker ID (to track waiting workers)
            timeout: Timeout in seconds

        Returns:
            True if barrier completed, False on timeout/failure
        """
        async with self._lock:
            barrier = self._barriers.get(barrier_id)
            if not barrier:
                return False

        # Arrive at barrier first
        arrived, complete = await self.arrive_barrier(barrier_id, worker_id)
        if not arrived:
            return False
        if complete:
            return True

        # Wait for completion
        timeout = timeout or self._barrier_timeout
        start = datetime.utcnow()

        while True:
            async with self._lock:
                barrier = self._barriers.get(barrier_id)
                if not barrier:
                    return False
                if barrier.is_complete:
                    return True
                if barrier.is_expired:
                    return False

            if (datetime.utcnow() - start).total_seconds() > timeout:
                logger.warning("barrier_wait_timeout", barrier_id=barrier_id)
                return False

            await asyncio.sleep(0.05)

    async def worker_failed(
        self,
        worker_id: str,
        error_message: str,
    ) -> None:
        """Handle worker failure."""
        async with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return

            worker.state = WorkerState.FAILED

            logger.error(
                "distributed_worker_failed",
                worker_id=worker_id,
                rank=worker.rank,
                error=error_message,
            )

            await self._emit_event(
                "worker_failed",
                {
                    "worker_id": worker_id,
                    "job_id": worker.job_id,
                    "rank": worker.rank,
                    "error": error_message,
                },
            )

    async def worker_finished(self, worker_id: str) -> None:
        """Mark worker as finished."""
        async with self._lock:
            worker = self._workers.get(worker_id)
            if not worker:
                return

            worker.state = WorkerState.FINISHED

            logger.info(
                "distributed_worker_finished",
                worker_id=worker_id,
                rank=worker.rank,
            )

    async def get_worker(self, worker_id: str) -> Optional[DistributedWorker]:
        """Get worker by ID."""
        async with self._lock:
            return self._workers.get(worker_id)

    async def get_job_workers(self, job_id: str) -> list[DistributedWorker]:
        """Get all workers for a job."""
        async with self._lock:
            return list(self._jobs.get(job_id, []))

    async def get_master_info(self, job_id: str) -> Optional[dict[str, Any]]:
        """Get master worker information for a job."""
        async with self._lock:
            workers = self._jobs.get(job_id, [])
            for worker in workers:
                if worker.is_master:
                    return {
                        "worker_id": worker.id,
                        "hostname": worker.hostname,
                        "ip_address": worker.ip_address,
                        "port": worker.port,
                        "rank": worker.rank,
                    }
            return None

    async def check_unhealthy_workers(self) -> list[DistributedWorker]:
        """Find workers that have missed heartbeats."""
        async with self._lock:
            unhealthy = []
            now = datetime.utcnow()

            for worker in self._workers.values():
                if worker.state in (WorkerState.FAILED, WorkerState.FINISHED):
                    continue

                age = (now - worker.last_heartbeat).total_seconds()
                if age > self._heartbeat_timeout:
                    unhealthy.append(worker)

            return unhealthy

    async def cleanup_job(self, job_id: str) -> None:
        """Clean up distributed training resources for a job."""
        async with self._lock:
            # Remove workers
            workers = self._jobs.pop(job_id, [])
            for worker in workers:
                self._workers.pop(worker.id, None)

            # Remove barriers
            barrier_ids = self._job_barriers.pop(job_id, [])
            for bid in barrier_ids:
                self._barriers.pop(bid, None)

            # Remove rendezvous event
            self._rendezvous_events.pop(job_id, None)

            logger.info(
                "distributed_job_cleaned_up",
                job_id=job_id,
                workers_removed=len(workers),
                barriers_removed=len(barrier_ids),
            )

    async def get_stats(self) -> dict[str, Any]:
        """Get coordinator statistics."""
        async with self._lock:
            return {
                "active_jobs": len(self._jobs),
                "total_workers": len(self._workers),
                "active_barriers": sum(
                    1 for b in self._barriers.values()
                    if not b.is_complete and not b.is_expired
                ),
                "workers_by_state": {
                    state.value: sum(
                        1 for w in self._workers.values() if w.state == state
                    )
                    for state in WorkerState
                },
            }
