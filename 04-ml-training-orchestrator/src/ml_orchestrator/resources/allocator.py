"""Resource allocation strategies."""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
import structlog

from ml_orchestrator.core.models import (
    AllocationStrategy,
    ResourceRequest,
    TrainingJob,
    WorkerInfo,
)
from ml_orchestrator.core.exceptions import ResourceError, ResourceExhaustedError


logger = structlog.get_logger(__name__)


@dataclass
class AllocationResult:
    """Result of a resource allocation attempt."""

    success: bool
    job_id: str
    worker_id: Optional[str] = None
    allocated_resources: Optional[ResourceRequest] = None
    reason: str = ""
    allocation_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ResourceSnapshot:
    """Snapshot of resource state at a point in time."""

    timestamp: datetime
    total_cpus: int
    available_cpus: int
    total_memory_gb: float
    available_memory_gb: float
    total_gpus: int
    available_gpus: int
    worker_count: int
    allocated_jobs: int


class AllocationPolicy(ABC):
    """Base class for allocation policies."""

    @abstractmethod
    async def allocate(
        self,
        job: TrainingJob,
        workers: list[WorkerInfo],
    ) -> Optional[AllocationResult]:
        """
        Attempt to allocate resources for a job.

        Args:
            job: Job requesting resources
            workers: Available workers

        Returns:
            AllocationResult if successful, None if no suitable worker found
        """
        pass


class FirstFitPolicy(AllocationPolicy):
    """First-fit allocation: use first worker that fits."""

    async def allocate(
        self,
        job: TrainingJob,
        workers: list[WorkerInfo],
    ) -> Optional[AllocationResult]:
        for worker in workers:
            if worker.can_run_job(job.resources):
                return AllocationResult(
                    success=True,
                    job_id=job.id,
                    worker_id=worker.id,
                    allocated_resources=job.resources,
                    reason="First fit allocation",
                )
        return None


class BestFitPolicy(AllocationPolicy):
    """Best-fit allocation: use worker with least remaining capacity."""

    async def allocate(
        self,
        job: TrainingJob,
        workers: list[WorkerInfo],
    ) -> Optional[AllocationResult]:
        best_worker = None
        best_remaining = float("inf")

        for worker in workers:
            if worker.can_run_job(job.resources):
                available = worker.available_resources
                # Calculate "remaining" as sum of unused resources
                remaining = (
                    available.cpus
                    + available.memory_gb
                    + available.gpus * 10  # Weight GPUs more
                )
                if remaining < best_remaining:
                    best_remaining = remaining
                    best_worker = worker

        if best_worker:
            return AllocationResult(
                success=True,
                job_id=job.id,
                worker_id=best_worker.id,
                allocated_resources=job.resources,
                reason=f"Best fit: {best_remaining:.1f} remaining",
            )
        return None


class WorstFitPolicy(AllocationPolicy):
    """Worst-fit allocation: use worker with most remaining capacity."""

    async def allocate(
        self,
        job: TrainingJob,
        workers: list[WorkerInfo],
    ) -> Optional[AllocationResult]:
        best_worker = None
        best_remaining = -1.0

        for worker in workers:
            if worker.can_run_job(job.resources):
                available = worker.available_resources
                remaining = (
                    available.cpus
                    + available.memory_gb
                    + available.gpus * 10
                )
                if remaining > best_remaining:
                    best_remaining = remaining
                    best_worker = worker

        if best_worker:
            return AllocationResult(
                success=True,
                job_id=job.id,
                worker_id=best_worker.id,
                allocated_resources=job.resources,
                reason=f"Worst fit: {best_remaining:.1f} remaining",
            )
        return None


class BinPackingPolicy(AllocationPolicy):
    """
    Bin-packing allocation: minimize total workers used.

    Tries to consolidate jobs on fewer workers to maximize
    resource utilization and potentially save on idle nodes.
    """

    def __init__(self, utilization_target: float = 0.8):
        self._utilization_target = utilization_target

    def _calculate_utilization(self, worker: WorkerInfo) -> float:
        """Calculate current utilization of a worker."""
        allocated = worker.allocated_resources
        total = worker.resources

        cpu_util = allocated.cpus / total.cpus if total.cpus > 0 else 0
        mem_util = allocated.memory_gb / total.memory_gb if total.memory_gb > 0 else 0
        gpu_util = allocated.gpus / total.gpus if total.gpus > 0 else 0

        # Weighted average, GPUs weighted more
        weights = [1, 1, 3]
        utils = [cpu_util, mem_util, gpu_util]
        return sum(u * w for u, w in zip(utils, weights)) / sum(weights)

    async def allocate(
        self,
        job: TrainingJob,
        workers: list[WorkerInfo],
    ) -> Optional[AllocationResult]:
        # Sort workers by utilization (highest first)
        suitable = [w for w in workers if w.can_run_job(job.resources)]
        suitable.sort(key=self._calculate_utilization, reverse=True)

        for worker in suitable:
            util = self._calculate_utilization(worker)
            # Prefer workers close to target utilization
            return AllocationResult(
                success=True,
                job_id=job.id,
                worker_id=worker.id,
                allocated_resources=job.resources,
                reason=f"Bin packed: {util:.1%} utilization",
            )

        return None


class AffinityAwarePolicy(AllocationPolicy):
    """
    Allocation considering affinity and anti-affinity rules.

    Supports:
    - GPU type affinity
    - Node affinity (prefer certain nodes)
    - Anti-affinity (spread across nodes)
    """

    async def allocate(
        self,
        job: TrainingJob,
        workers: list[WorkerInfo],
    ) -> Optional[AllocationResult]:
        suitable = [w for w in workers if w.can_run_job(job.resources)]

        if not suitable:
            return None

        # Check for GPU type affinity
        if job.resources.gpu_type:
            suitable = [
                w for w in suitable
                if w.resources.gpu_type == job.resources.gpu_type
            ]
            if not suitable:
                return None

        # Check for node affinity labels
        affinity_labels = job.tags.get("node_affinity", "").split(",")
        if affinity_labels and affinity_labels[0]:
            preferred = []
            for worker in suitable:
                for label in affinity_labels:
                    if label in worker.labels.values():
                        preferred.append(worker)
                        break
            if preferred:
                suitable = preferred

        # Check for anti-affinity (spread jobs)
        anti_affinity = job.tags.get("anti_affinity") == "true"
        if anti_affinity:
            # Prefer workers with fewer jobs
            suitable.sort(key=lambda w: len(w.current_jobs))

        if suitable:
            return AllocationResult(
                success=True,
                job_id=job.id,
                worker_id=suitable[0].id,
                allocated_resources=job.resources,
                reason="Affinity-aware allocation",
            )

        return None


class ResourceAllocator:
    """
    Main resource allocator coordinating allocation policies and tracking.
    """

    def __init__(
        self,
        strategy: AllocationStrategy = AllocationStrategy.BEST_FIT,
    ):
        self._strategy = strategy
        self._policy = self._create_policy(strategy)
        self._workers: dict[str, WorkerInfo] = {}
        self._allocations: dict[str, AllocationResult] = {}  # job_id -> result
        self._history: list[AllocationResult] = []
        self._snapshots: list[ResourceSnapshot] = []
        self._lock = asyncio.Lock()

    def _create_policy(self, strategy: AllocationStrategy) -> AllocationPolicy:
        """Create allocation policy based on strategy."""
        policies = {
            AllocationStrategy.FIRST_FIT: FirstFitPolicy,
            AllocationStrategy.BEST_FIT: BestFitPolicy,
            AllocationStrategy.WORST_FIT: WorstFitPolicy,
            AllocationStrategy.BIN_PACKING: BinPackingPolicy,
        }
        policy_class = policies.get(strategy, BestFitPolicy)
        return policy_class()

    def set_strategy(self, strategy: AllocationStrategy) -> None:
        """Change the allocation strategy."""
        self._strategy = strategy
        self._policy = self._create_policy(strategy)

    async def register_worker(self, worker: WorkerInfo) -> None:
        """Register a worker."""
        async with self._lock:
            self._workers[worker.id] = worker
            logger.info(
                "worker_registered",
                worker_id=worker.id,
                cpus=worker.resources.cpus,
                memory_gb=worker.resources.memory_gb,
                gpus=worker.resources.gpus,
            )

    async def unregister_worker(self, worker_id: str) -> None:
        """Unregister a worker."""
        async with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                logger.info("worker_unregistered", worker_id=worker_id)

    async def update_worker(self, worker: WorkerInfo) -> None:
        """Update worker information."""
        async with self._lock:
            self._workers[worker.id] = worker

    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker by ID."""
        async with self._lock:
            return self._workers.get(worker_id)

    async def list_workers(self, healthy_only: bool = False) -> list[WorkerInfo]:
        """List all workers."""
        async with self._lock:
            workers = list(self._workers.values())
            if healthy_only:
                workers = [w for w in workers if w.is_healthy()]
            return workers

    async def allocate(self, job: TrainingJob) -> AllocationResult:
        """
        Allocate resources for a job.

        Args:
            job: Job requesting resources

        Returns:
            AllocationResult

        Raises:
            ResourceExhaustedError: If no worker can satisfy the request
        """
        async with self._lock:
            # Get healthy workers
            workers = [w for w in self._workers.values() if w.is_healthy()]

            if not workers:
                raise ResourceExhaustedError("workers", 1, 0)

            # Try allocation
            result = await self._policy.allocate(job, workers)

            if result and result.success:
                # Update worker allocation
                worker = self._workers.get(result.worker_id)
                if worker:
                    worker.allocated_resources = worker.allocated_resources.add(
                        job.resources
                    )
                    worker.current_jobs.append(job.id)

                # Track allocation
                self._allocations[job.id] = result
                self._history.append(result)

                logger.info(
                    "resources_allocated",
                    job_id=job.id,
                    worker_id=result.worker_id,
                    reason=result.reason,
                )

                return result

            # Allocation failed
            raise ResourceExhaustedError(
                "resources",
                1,
                0,
            )

    async def release(self, job_id: str) -> bool:
        """
        Release resources for a job.

        Returns:
            True if resources were released
        """
        async with self._lock:
            allocation = self._allocations.pop(job_id, None)
            if not allocation:
                return False

            # Update worker
            worker = self._workers.get(allocation.worker_id)
            if worker and allocation.allocated_resources:
                # Subtract released resources
                new_allocated = ResourceRequest(
                    cpus=max(0, worker.allocated_resources.cpus - allocation.allocated_resources.cpus),
                    memory_gb=max(0, worker.allocated_resources.memory_gb - allocation.allocated_resources.memory_gb),
                    gpus=max(0, worker.allocated_resources.gpus - allocation.allocated_resources.gpus),
                    storage_gb=max(0, worker.allocated_resources.storage_gb - allocation.allocated_resources.storage_gb),
                )
                worker.allocated_resources = new_allocated
                if job_id in worker.current_jobs:
                    worker.current_jobs.remove(job_id)

            logger.info("resources_released", job_id=job_id)
            return True

    async def get_allocation(self, job_id: str) -> Optional[AllocationResult]:
        """Get allocation for a job."""
        async with self._lock:
            return self._allocations.get(job_id)

    async def can_allocate(self, resources: ResourceRequest) -> bool:
        """Check if resources can be allocated."""
        async with self._lock:
            for worker in self._workers.values():
                if worker.is_healthy() and resources.fits_in(worker.available_resources):
                    return True
            return False

    async def get_available_resources(self) -> ResourceRequest:
        """Get total available resources across all workers."""
        async with self._lock:
            total = ResourceRequest(cpus=0, memory_gb=0, gpus=0, storage_gb=0)
            for worker in self._workers.values():
                if worker.is_healthy():
                    available = worker.available_resources
                    total = ResourceRequest(
                        cpus=total.cpus + available.cpus,
                        memory_gb=total.memory_gb + available.memory_gb,
                        gpus=total.gpus + available.gpus,
                        storage_gb=total.storage_gb + available.storage_gb,
                    )
            return total

    async def get_total_resources(self) -> ResourceRequest:
        """Get total resources across all workers."""
        async with self._lock:
            total = ResourceRequest(cpus=0, memory_gb=0, gpus=0, storage_gb=0)
            for worker in self._workers.values():
                total = total.add(worker.resources)
            return total

    async def take_snapshot(self) -> ResourceSnapshot:
        """Take a snapshot of current resource state."""
        async with self._lock:
            total = await self.get_total_resources()
            available = await self.get_available_resources()

            snapshot = ResourceSnapshot(
                timestamp=datetime.utcnow(),
                total_cpus=total.cpus,
                available_cpus=available.cpus,
                total_memory_gb=total.memory_gb,
                available_memory_gb=available.memory_gb,
                total_gpus=total.gpus,
                available_gpus=available.gpus,
                worker_count=len(self._workers),
                allocated_jobs=len(self._allocations),
            )

            self._snapshots.append(snapshot)
            # Keep last 1000 snapshots
            if len(self._snapshots) > 1000:
                self._snapshots = self._snapshots[-1000:]

            return snapshot

    async def get_utilization(self) -> dict[str, float]:
        """Get current resource utilization percentages."""
        # No outer self._lock here: get_total_resources/get_available_resources
        # each acquire it, and asyncio.Lock is not reentrant — wrapping them
        # would deadlock.
        total = await self.get_total_resources()
        available = await self.get_available_resources()

        def util(used: float, total: float) -> float:
            if total <= 0:
                return 0.0
            return ((total - used) / total) * 100

        # available_resources is what's left, so used = total - available
        return {
            "cpu_percent": util(available.cpus, total.cpus),
            "memory_percent": util(available.memory_gb, total.memory_gb),
            "gpu_percent": util(available.gpus, total.gpus),
        }

    async def get_stats(self) -> dict[str, Any]:
        """Get allocator statistics."""
        # Same reentrancy concern as get_utilization — call the locked helpers
        # without holding self._lock.
        total = await self.get_total_resources()
        available = await self.get_available_resources()
        utilization = await self.get_utilization()

        return {
            "strategy": self._strategy.value,
            "worker_count": len(self._workers),
            "healthy_workers": sum(1 for w in self._workers.values() if w.is_healthy()),
            "active_allocations": len(self._allocations),
            "total_allocations": len(self._history),
            "resources": {
                "total": total.model_dump(),
                "available": available.model_dump(),
            },
            "utilization": utilization,
        }

    async def get_fragmentation_score(self) -> float:
        """
        Calculate resource fragmentation score.

        Higher score means more fragmentation (resources spread across workers
        but not enough on any single worker for large jobs).

        Returns:
            Float between 0 (no fragmentation) and 1 (high fragmentation)
        """
        async with self._lock:
            if not self._workers:
                return 0.0

            # Calculate max single-worker capacity
            max_gpus = max(
                (w.available_resources.gpus for w in self._workers.values()),
                default=0,
            )
            max_cpus = max(
                (w.available_resources.cpus for w in self._workers.values()),
                default=0,
            )

            # Calculate total available
            total = await self.get_available_resources()

            # Fragmentation = 1 - (max_single / total)
            # If all resources on one node, fragmentation = 0
            # If spread evenly, fragmentation is high
            gpu_frag = 1 - (max_gpus / total.gpus) if total.gpus > 0 else 0
            cpu_frag = 1 - (max_cpus / total.cpus) if total.cpus > 0 else 0

            return (gpu_frag + cpu_frag) / 2
