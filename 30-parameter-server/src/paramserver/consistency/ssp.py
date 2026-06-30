"""Stale Synchronous Parallel (SSP) consistency model."""

import asyncio
from typing import Dict, Optional

from paramserver.consistency.base import ConsistencyModel


class SSPConsistency(ConsistencyModel):
    """Stale Synchronous Parallel consistency model.

    SSP allows bounded staleness - workers can proceed even if they're
    reading slightly stale parameters, as long as the staleness is within
    a configurable threshold. This provides a balance between the strong
    consistency of BSP and the high parallelism of Hogwild.

    The staleness threshold determines how many iterations a worker can
    be ahead of the slowest worker. If worker A is at iteration 10 and
    the threshold is 3, worker A must wait if any worker is at iteration
    6 or below.

    Attributes:
        staleness_threshold: Maximum allowed difference between fastest
            and slowest workers.
    """

    def __init__(self, staleness_threshold: int = 3):
        """Initialize SSP consistency model.

        Args:
            staleness_threshold: Maximum staleness allowed.

        Raises:
            ValueError: If threshold is less than 0.
        """
        if staleness_threshold < 0:
            raise ValueError(f"staleness_threshold must be >= 0, got {staleness_threshold}")

        self.staleness_threshold = staleness_threshold

        # Track each worker's current clock
        self._worker_clocks: Dict[int, int] = {}

        # Lock for thread-safe updates
        self._lock = asyncio.Lock()

        # Events for workers waiting on staleness
        self._wait_events: Dict[int, asyncio.Event] = {}

    def can_apply(
        self,
        param_version: int,
        worker_clock: int,
    ) -> bool:
        """Check if update can be applied based on staleness.

        An update can be applied if the worker's clock is not too far
        ahead of the slowest worker.

        Args:
            param_version: Current parameter version (unused).
            worker_clock: Clock of the worker sending the update.

        Returns:
            True if staleness is within threshold.
        """
        if not self._worker_clocks:
            return True

        min_clock = min(self._worker_clocks.values())
        staleness = worker_clock - min_clock

        return staleness <= self.staleness_threshold

    async def update_worker_clock(self, worker_id: int, clock: int) -> None:
        """Update a worker's clock value.

        Args:
            worker_id: ID of the worker.
            clock: New clock value for the worker.
        """
        async with self._lock:
            old_clock = self._worker_clocks.get(worker_id, 0)
            self._worker_clocks[worker_id] = max(old_clock, clock)

            # Wake up any workers waiting on staleness
            self._notify_waiters()

    def _notify_waiters(self) -> None:
        """Notify workers waiting on staleness conditions."""
        for event in self._wait_events.values():
            event.set()
        self._wait_events.clear()

    async def wait_for_staleness(
        self,
        worker_id: int,
        worker_clock: int,
        timeout: Optional[float] = None,
    ) -> bool:
        """Wait until staleness is within threshold.

        Args:
            worker_id: ID of the waiting worker.
            worker_clock: Worker's target clock.
            timeout: Maximum wait time in seconds.

        Returns:
            True if staleness is now acceptable, False if timed out.
        """
        if self.can_apply(0, worker_clock):
            return True

        # Create wait event
        event = asyncio.Event()
        async with self._lock:
            self._wait_events[worker_id] = event

        try:
            while not self.can_apply(0, worker_clock):
                try:
                    await asyncio.wait_for(event.wait(), timeout=timeout)
                    event.clear()
                except asyncio.TimeoutError:
                    return False
            return True
        finally:
            async with self._lock:
                self._wait_events.pop(worker_id, None)

    def get_min_clock(self) -> int:
        """Get the minimum (slowest) worker clock.

        Returns:
            Minimum clock value, or 0 if no workers registered.
        """
        if not self._worker_clocks:
            return 0
        return min(self._worker_clocks.values())

    def get_max_clock(self) -> int:
        """Get the maximum (fastest) worker clock.

        Returns:
            Maximum clock value, or 0 if no workers registered.
        """
        if not self._worker_clocks:
            return 0
        return max(self._worker_clocks.values())

    def get_current_staleness(self) -> int:
        """Get the current staleness (max - min clock).

        Returns:
            Current staleness value.
        """
        if not self._worker_clocks:
            return 0
        return self.get_max_clock() - self.get_min_clock()

    def get_worker_clock(self, worker_id: int) -> int:
        """Get a specific worker's clock value.

        Args:
            worker_id: ID of the worker.

        Returns:
            Worker's clock value, or 0 if not registered.
        """
        return self._worker_clocks.get(worker_id, 0)

    def get_staleness_for_worker(self, worker_id: int) -> int:
        """Get how stale a specific worker is.

        Args:
            worker_id: ID of the worker.

        Returns:
            Staleness value (difference from max clock).
        """
        worker_clock = self.get_worker_clock(worker_id)
        return self.get_max_clock() - worker_clock

    def is_worker_blocked(self, worker_id: int, target_clock: int) -> bool:
        """Check if a worker would be blocked at a target clock.

        Args:
            worker_id: ID of the worker.
            target_clock: Clock the worker wants to reach.

        Returns:
            True if the worker would be blocked.
        """
        return not self.can_apply(0, target_clock)

    def get_stats(self) -> Dict:
        """Get SSP statistics.

        Returns:
            Dict with staleness statistics.
        """
        return {
            "threshold": self.staleness_threshold,
            "num_workers": len(self._worker_clocks),
            "min_clock": self.get_min_clock(),
            "max_clock": self.get_max_clock(),
            "current_staleness": self.get_current_staleness(),
            "worker_clocks": dict(self._worker_clocks),
        }

    def register_worker(self, worker_id: int, initial_clock: int = 0) -> None:
        """Register a new worker with initial clock.

        Args:
            worker_id: ID of the worker to register.
            initial_clock: Initial clock value.
        """
        self._worker_clocks[worker_id] = initial_clock

    def unregister_worker(self, worker_id: int) -> None:
        """Unregister a worker.

        Args:
            worker_id: ID of the worker to remove.
        """
        self._worker_clocks.pop(worker_id, None)
        self._notify_waiters()

    def reset(self) -> None:
        """Reset all worker clocks."""
        self._worker_clocks.clear()
        self._notify_waiters()
