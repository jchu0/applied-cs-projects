"""Bulk Synchronous Parallel (BSP) consistency model."""

import asyncio
from typing import Dict, Optional, Set

from paramserver.consistency.base import ConsistencyModel


class BSPConsistency(ConsistencyModel):
    """Bulk Synchronous Parallel consistency model.

    In BSP mode, all workers must reach a synchronization barrier before
    any updates from that iteration are applied. This provides the strongest
    consistency guarantees but may lead to stragglers slowing down training.

    The model maintains barriers for each iteration (clock value). When all
    workers have arrived at a barrier, updates from that iteration can be
    applied.

    Attributes:
        num_workers: Total number of workers in the system.
    """

    def __init__(self, num_workers: int):
        """Initialize BSP consistency model.

        Args:
            num_workers: Number of workers that must sync at each barrier.

        Raises:
            ValueError: If num_workers is less than 1.
        """
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        self.num_workers = num_workers

        # Track which workers have arrived at each barrier
        # barriers[clock] = set of worker IDs that have arrived
        self._barriers: Dict[int, Set[int]] = {}

        # Lock for thread-safe barrier updates
        self._lock = asyncio.Lock()

        # Events for workers waiting at barriers
        self._barrier_events: Dict[int, asyncio.Event] = {}

    def can_apply(
        self,
        param_version: int,
        worker_clock: int,
    ) -> bool:
        """Check if update can be applied.

        In BSP, updates can only be applied when all workers have
        reached the same clock value.

        Args:
            param_version: Current parameter version (unused in BSP).
            worker_clock: Clock of the worker sending the update.

        Returns:
            True if all workers have reached this clock.
        """
        if worker_clock not in self._barriers:
            return False

        return len(self._barriers[worker_clock]) >= self.num_workers

    async def worker_arrived(self, worker_id: int, clock: int) -> bool:
        """Signal that a worker has arrived at a barrier.

        Args:
            worker_id: ID of the arriving worker.
            clock: Clock value (iteration number) of the barrier.

        Returns:
            True if all workers have now arrived at this barrier.
        """
        async with self._lock:
            if clock not in self._barriers:
                self._barriers[clock] = set()
            # Only create event if it doesn't exist (wait_for_barrier may have created it)
            if clock not in self._barrier_events:
                self._barrier_events[clock] = asyncio.Event()

            self._barriers[clock].add(worker_id)

            all_arrived = len(self._barriers[clock]) >= self.num_workers

            if all_arrived:
                # Signal waiting workers
                self._barrier_events[clock].set()

            return all_arrived

    async def wait_for_barrier(self, clock: int, timeout: Optional[float] = None) -> bool:
        """Wait for all workers to arrive at a barrier.

        Args:
            clock: Clock value to wait for.
            timeout: Maximum time to wait in seconds.

        Returns:
            True if barrier was reached, False if timed out.
        """
        async with self._lock:
            if clock not in self._barrier_events:
                self._barrier_events[clock] = asyncio.Event()

            event = self._barrier_events[clock]

        # Check if already complete
        if self.can_apply(0, clock):
            return True

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def get_barrier_status(self, clock: int) -> Dict[str, int]:
        """Get status of a specific barrier.

        Args:
            clock: Clock value to query.

        Returns:
            Dict with arrived count and total workers needed.
        """
        arrived = len(self._barriers.get(clock, set()))
        return {
            "clock": clock,
            "arrived": arrived,
            "total": self.num_workers,
            "complete": arrived >= self.num_workers,
        }

    def get_slowest_worker(self) -> Optional[int]:
        """Get the worker furthest behind (if any).

        Returns:
            Worker ID of the slowest worker, or None if all are synced.
        """
        if not self._barriers:
            return None

        # Find workers missing from the latest incomplete barrier
        for clock in sorted(self._barriers.keys(), reverse=True):
            arrived = self._barriers[clock]
            if len(arrived) < self.num_workers:
                # Find missing workers
                all_workers = set(range(self.num_workers))
                missing = all_workers - arrived
                if missing:
                    return min(missing)

        return None

    def clear_barrier(self, clock: int) -> None:
        """Clear a completed barrier to free memory.

        Args:
            clock: Clock value to clear.
        """
        self._barriers.pop(clock, None)
        self._barrier_events.pop(clock, None)

    def clear_old_barriers(self, current_clock: int) -> int:
        """Clear all barriers older than current clock.

        Args:
            current_clock: Current global clock value.

        Returns:
            Number of barriers cleared.
        """
        old_clocks = [c for c in self._barriers.keys() if c < current_clock]
        for clock in old_clocks:
            self.clear_barrier(clock)
        return len(old_clocks)

    def reset(self) -> None:
        """Reset all barrier state."""
        self._barriers.clear()
        self._barrier_events.clear()

    @property
    def pending_barriers(self) -> int:
        """Get count of incomplete barriers."""
        return sum(
            1 for arrived in self._barriers.values()
            if len(arrived) < self.num_workers
        )
