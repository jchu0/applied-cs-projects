"""Adaptive staleness control for SSP.

Dynamically adjusts staleness thresholds based on
training progress and system conditions.
"""

from typing import Dict, List, Optional
import time


class StalenessController:
    """Controls and tracks staleness in distributed training.

    Monitors worker clocks and adjusts staleness thresholds
    to balance convergence and throughput.

    Attributes:
        max_staleness: Maximum allowed staleness.
        min_staleness: Minimum staleness threshold.
    """

    def __init__(
        self,
        max_staleness: int = 5,
        min_staleness: int = 0,
    ):
        """Initialize staleness controller.

        Args:
            max_staleness: Maximum staleness allowed.
            min_staleness: Minimum staleness threshold.
        """
        self.max_staleness = max_staleness
        self.min_staleness = min_staleness
        self._current_threshold = max_staleness

        # Track worker clocks
        self._worker_clocks: Dict[int, int] = {}

        # Track staleness history
        self._staleness_history: List[int] = []
        self._window_size = 100

    def register_worker(self, worker_id: int, initial_clock: int = 0) -> None:
        """Register a worker for staleness tracking.

        Args:
            worker_id: Worker identifier.
            initial_clock: Initial clock value.
        """
        self._worker_clocks[worker_id] = initial_clock

    def update_clock(self, worker_id: int, clock: int) -> None:
        """Update a worker's clock.

        Args:
            worker_id: Worker identifier.
            clock: New clock value.
        """
        self._worker_clocks[worker_id] = clock

    def get_staleness(self, worker_id: int) -> int:
        """Get staleness of a worker relative to fastest worker.

        Args:
            worker_id: Worker to check.

        Returns:
            Staleness (difference from fastest clock).
        """
        if worker_id not in self._worker_clocks:
            return 0

        if not self._worker_clocks:
            return 0

        fastest = max(self._worker_clocks.values())
        worker_clock = self._worker_clocks[worker_id]

        return fastest - worker_clock

    def get_slowest_worker(self) -> Optional[int]:
        """Get ID of slowest worker.

        Returns:
            Worker ID or None.
        """
        if not self._worker_clocks:
            return None

        return min(self._worker_clocks, key=self._worker_clocks.get)

    def get_fastest_worker(self) -> Optional[int]:
        """Get ID of fastest worker.

        Returns:
            Worker ID or None.
        """
        if not self._worker_clocks:
            return None

        return max(self._worker_clocks, key=self._worker_clocks.get)

    def is_too_stale(self, worker_id: int) -> bool:
        """Check if worker is too stale.

        Args:
            worker_id: Worker to check.

        Returns:
            True if worker exceeds staleness threshold.
        """
        staleness = self.get_staleness(worker_id)
        return staleness > self._current_threshold

    def can_proceed(self, worker_id: int) -> bool:
        """Check if worker can proceed without waiting.

        Args:
            worker_id: Worker to check.

        Returns:
            True if worker can proceed.
        """
        return not self.is_too_stale(worker_id)

    def get_threshold(self) -> int:
        """Get current staleness threshold.

        Returns:
            Current threshold value.
        """
        return self._current_threshold

    def set_threshold(self, threshold: int) -> None:
        """Set staleness threshold.

        Args:
            threshold: New threshold value.
        """
        self._current_threshold = max(
            self.min_staleness,
            min(self.max_staleness, threshold)
        )

    def get_stats(self) -> Dict[str, float]:
        """Get staleness statistics.

        Returns:
            Dictionary of statistics.
        """
        if not self._worker_clocks:
            return {
                "num_workers": 0,
                "current_threshold": self._current_threshold,
                "max_staleness": 0,
                "min_clock": 0,
                "max_clock": 0,
                "clock_spread": 0,
            }

        clocks = list(self._worker_clocks.values())
        min_clock = min(clocks)
        max_clock = max(clocks)

        return {
            "num_workers": len(self._worker_clocks),
            "current_threshold": self._current_threshold,
            "max_staleness": max_clock - min_clock,
            "min_clock": min_clock,
            "max_clock": max_clock,
            "clock_spread": max_clock - min_clock,
        }


class AdaptiveSSP(StalenessController):
    """Adaptive Stale Synchronous Parallel.

    Dynamically adjusts staleness threshold based on:
    - Loss convergence rate
    - Worker speed variance
    - System load

    Attributes:
        target_staleness: Target average staleness.
        adapt_rate: How quickly to adjust threshold.
    """

    def __init__(
        self,
        max_staleness: int = 10,
        min_staleness: int = 0,
        target_staleness: float = 2.0,
        adapt_rate: float = 0.1,
        check_interval: int = 100,
    ):
        """Initialize adaptive SSP controller.

        Args:
            max_staleness: Maximum allowed staleness.
            min_staleness: Minimum staleness.
            target_staleness: Target average staleness.
            adapt_rate: Adaptation rate (0-1).
            check_interval: Steps between threshold adjustments.
        """
        super().__init__(max_staleness, min_staleness)

        self.target_staleness = target_staleness
        self.adapt_rate = adapt_rate
        self.check_interval = check_interval

        self._step_count = 0
        self._loss_history: List[float] = []
        self._staleness_samples: List[float] = []
        self._last_adapt_time = time.time()

    def record_step(
        self,
        worker_id: int,
        loss: Optional[float] = None,
    ) -> None:
        """Record a training step.

        Args:
            worker_id: Worker that completed step.
            loss: Optional loss value.
        """
        self._step_count += 1

        # Record staleness
        staleness = self.get_staleness(worker_id)
        self._staleness_samples.append(staleness)
        if len(self._staleness_samples) > self._window_size:
            self._staleness_samples.pop(0)

        # Record loss if provided
        if loss is not None:
            self._loss_history.append(loss)
            if len(self._loss_history) > self._window_size:
                self._loss_history.pop(0)

        # Adapt threshold periodically
        if self._step_count % self.check_interval == 0:
            self._adapt_threshold()

    def _adapt_threshold(self) -> None:
        """Adapt staleness threshold based on observed metrics."""
        if not self._staleness_samples:
            return

        # Calculate average staleness
        avg_staleness = sum(self._staleness_samples) / len(self._staleness_samples)

        # Adjust threshold towards target
        if avg_staleness > self.target_staleness * 1.5:
            # Too stale, decrease threshold
            delta = -1
        elif avg_staleness < self.target_staleness * 0.5:
            # Too synchronized, can increase threshold
            delta = 1
        else:
            delta = 0

        if delta != 0:
            new_threshold = self._current_threshold + delta
            self.set_threshold(new_threshold)

        # Check loss convergence rate if available
        if len(self._loss_history) >= 10:
            recent_loss = sum(self._loss_history[-10:]) / 10
            older_loss = sum(self._loss_history[:10]) / 10

            if recent_loss > older_loss * 0.99:
                # Loss not improving, reduce staleness
                self.set_threshold(self._current_threshold - 1)

    def get_stats(self) -> Dict[str, float]:
        """Get adaptive SSP statistics."""
        base_stats = super().get_stats()

        avg_staleness = 0.0
        if self._staleness_samples:
            avg_staleness = sum(self._staleness_samples) / len(self._staleness_samples)

        base_stats.update({
            "target_staleness": self.target_staleness,
            "average_staleness": avg_staleness,
            "step_count": self._step_count,
            "adapt_rate": self.adapt_rate,
        })

        return base_stats
