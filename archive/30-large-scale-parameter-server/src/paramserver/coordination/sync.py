"""Synchronization primitives for parameter server."""

import asyncio
import time
from typing import Any
import logging

from ..schemas import SyncBarrier, ConsistencyModel, generate_id

logger = logging.getLogger(__name__)


class SyncManager:
    """Manages synchronization barriers for distributed training."""

    def __init__(self, consistency_model: ConsistencyModel = ConsistencyModel.BSP):
        """Initialize sync manager.

        Args:
            consistency_model: Consistency model to use
        """
        self.consistency_model = consistency_model
        self._barriers: dict[int, SyncBarrier] = {}
        self._worker_iterations: dict[str, int] = {}
        self._global_iteration = 0
        self._lock = asyncio.Lock()

    async def create_barrier(
        self,
        iteration: int,
        expected_workers: int
    ) -> SyncBarrier:
        """Create synchronization barrier.

        Args:
            iteration: Iteration number
            expected_workers: Number of workers expected

        Returns:
            Created barrier
        """
        async with self._lock:
            if iteration in self._barriers:
                return self._barriers[iteration]

            barrier = SyncBarrier(
                barrier_id=generate_id(),
                iteration=iteration,
                expected_workers=expected_workers
            )
            self._barriers[iteration] = barrier
            return barrier

    async def arrive_at_barrier(
        self,
        worker_id: str,
        iteration: int
    ) -> bool:
        """Worker arrives at barrier.

        Args:
            worker_id: Worker ID
            iteration: Iteration number

        Returns:
            True if barrier is complete
        """
        async with self._lock:
            if iteration not in self._barriers:
                return False

            barrier = self._barriers[iteration]
            barrier.arrived_workers.add(worker_id)
            self._worker_iterations[worker_id] = iteration

            if barrier.is_complete:
                self._global_iteration = max(self._global_iteration, iteration)
                return True

            return False

    async def wait_for_barrier(
        self,
        iteration: int,
        timeout: float = 30.0
    ) -> bool:
        """Wait for barrier to complete.

        Args:
            iteration: Iteration number
            timeout: Timeout in seconds

        Returns:
            True if barrier completed
        """
        start = time.time()
        while time.time() - start < timeout:
            async with self._lock:
                if iteration in self._barriers:
                    if self._barriers[iteration].is_complete:
                        return True
            await asyncio.sleep(0.1)
        return False

    async def get_barrier_status(self, iteration: int) -> dict[str, Any]:
        """Get barrier status.

        Args:
            iteration: Iteration number

        Returns:
            Barrier status
        """
        async with self._lock:
            if iteration not in self._barriers:
                return {"exists": False}

            barrier = self._barriers[iteration]
            return {
                "exists": True,
                "barrier_id": barrier.barrier_id,
                "iteration": barrier.iteration,
                "expected": barrier.expected_workers,
                "arrived": len(barrier.arrived_workers),
                "complete": barrier.is_complete
            }

    async def cleanup_old_barriers(self, keep_recent: int = 10):
        """Clean up old barriers.

        Args:
            keep_recent: Number of recent barriers to keep
        """
        async with self._lock:
            if len(self._barriers) <= keep_recent:
                return

            sorted_iters = sorted(self._barriers.keys())
            to_remove = sorted_iters[:-keep_recent]

            for iteration in to_remove:
                del self._barriers[iteration]

    def get_global_iteration(self) -> int:
        """Get global iteration number.

        Returns:
            Global iteration
        """
        return self._global_iteration

    def get_worker_iteration(self, worker_id: str) -> int:
        """Get worker's current iteration.

        Args:
            worker_id: Worker ID

        Returns:
            Worker's iteration
        """
        return self._worker_iterations.get(worker_id, 0)


class StalenessTracker:
    """Tracks parameter staleness for SSP/bounded staleness."""

    def __init__(self, max_staleness: int = 5):
        """Initialize staleness tracker.

        Args:
            max_staleness: Maximum allowed staleness
        """
        self.max_staleness = max_staleness
        self._parameter_versions: dict[str, int] = {}
        self._worker_versions: dict[str, dict[str, int]] = {}

    def update_parameter_version(self, name: str, version: int):
        """Update parameter version.

        Args:
            name: Parameter name
            version: New version
        """
        self._parameter_versions[name] = version

    def record_worker_pull(
        self,
        worker_id: str,
        parameter_name: str,
        version: int
    ):
        """Record when worker pulled parameter.

        Args:
            worker_id: Worker ID
            parameter_name: Parameter name
            version: Version pulled
        """
        if worker_id not in self._worker_versions:
            self._worker_versions[worker_id] = {}
        self._worker_versions[worker_id][parameter_name] = version

    def check_staleness(
        self,
        worker_id: str,
        parameter_name: str
    ) -> tuple[bool, int]:
        """Check if worker's view is too stale.

        Args:
            worker_id: Worker ID
            parameter_name: Parameter name

        Returns:
            Tuple of (is_stale, staleness)
        """
        current_version = self._parameter_versions.get(parameter_name, 0)
        worker_version = self._worker_versions.get(worker_id, {}).get(
            parameter_name, 0
        )

        staleness = current_version - worker_version
        is_stale = staleness > self.max_staleness

        return is_stale, staleness

    def should_block_push(
        self,
        worker_id: str,
        parameter_name: str
    ) -> bool:
        """Check if push should be blocked due to staleness.

        Args:
            worker_id: Worker ID
            parameter_name: Parameter name

        Returns:
            True if should block
        """
        is_stale, _ = self.check_staleness(worker_id, parameter_name)
        return is_stale

    def get_staleness_stats(self) -> dict[str, Any]:
        """Get staleness statistics.

        Returns:
            Statistics dictionary
        """
        total_staleness = 0
        count = 0

        for worker_id, params in self._worker_versions.items():
            for param_name, version in params.items():
                current = self._parameter_versions.get(param_name, 0)
                total_staleness += current - version
                count += 1

        return {
            "avg_staleness": total_staleness / max(count, 1),
            "max_allowed": self.max_staleness,
            "num_tracked": count
        }
