"""Consistency model manager and factory."""

from enum import Enum
from typing import Any, Dict, Optional, Union

from paramserver.consistency.base import ConsistencyModel
from paramserver.consistency.hogwild import HogwildConsistency
from paramserver.consistency.bsp import BSPConsistency
from paramserver.consistency.ssp import SSPConsistency


class ConsistencyType(Enum):
    """Types of consistency models."""
    HOGWILD = "hogwild"
    BSP = "bsp"
    SSP = "ssp"


class ConsistencyManager:
    """Factory and manager for consistency models.

    Provides a unified interface for creating and configuring
    different consistency models.

    Attributes:
        consistency_type: Type of consistency model being used.
        model: The underlying consistency model instance.
    """

    def __init__(
        self,
        consistency_type: Union[str, ConsistencyType],
        num_workers: int = 1,
        **kwargs: Any,
    ):
        """Initialize consistency manager.

        Args:
            consistency_type: Type of consistency model to use.
                Can be a string ("hogwild", "bsp", "ssp") or ConsistencyType enum.
            num_workers: Number of workers (required for BSP).
            **kwargs: Additional arguments for the consistency model.
                - staleness_threshold: For SSP, maximum staleness allowed.

        Raises:
            ValueError: If consistency_type is unknown.
        """
        if isinstance(consistency_type, str):
            try:
                consistency_type = ConsistencyType(consistency_type.lower())
            except ValueError:
                raise ValueError(
                    f"Unknown consistency type: {consistency_type}. "
                    f"Valid types: {[t.value for t in ConsistencyType]}"
                )

        self.consistency_type = consistency_type
        self.num_workers = num_workers

        # Create the appropriate model
        if consistency_type == ConsistencyType.HOGWILD:
            self.model: ConsistencyModel = HogwildConsistency()
        elif consistency_type == ConsistencyType.BSP:
            self.model = BSPConsistency(num_workers=num_workers)
        elif consistency_type == ConsistencyType.SSP:
            staleness = kwargs.get("staleness_threshold", 3)
            self.model = SSPConsistency(staleness_threshold=staleness)
        else:
            raise ValueError(f"Unknown consistency type: {consistency_type}")

    def can_apply_gradient(
        self,
        param_version: int,
        worker_clock: int,
    ) -> bool:
        """Check if a gradient update can be applied.

        Args:
            param_version: Current parameter version.
            worker_clock: Worker's logical clock.

        Returns:
            True if the update can be applied.
        """
        return self.model.can_apply(param_version, worker_clock)

    async def on_worker_update(
        self,
        worker_id: int,
        clock: int,
    ) -> None:
        """Handle a worker completing an iteration.

        Updates the appropriate tracking structures based on
        the consistency model type.

        Args:
            worker_id: ID of the worker.
            clock: Worker's new clock value.
        """
        if isinstance(self.model, BSPConsistency):
            await self.model.worker_arrived(worker_id, clock)
        elif isinstance(self.model, SSPConsistency):
            await self.model.update_worker_clock(worker_id, clock)
        # Hogwild doesn't need tracking

    async def wait_if_needed(
        self,
        worker_id: int,
        target_clock: int,
        timeout: Optional[float] = None,
    ) -> bool:
        """Wait if consistency constraints require it.

        Args:
            worker_id: ID of the waiting worker.
            target_clock: Clock value the worker wants to reach.
            timeout: Maximum wait time in seconds.

        Returns:
            True if worker can proceed, False if timed out.
        """
        if isinstance(self.model, BSPConsistency):
            return await self.model.wait_for_barrier(target_clock, timeout)
        elif isinstance(self.model, SSPConsistency):
            return await self.model.wait_for_staleness(
                worker_id, target_clock, timeout
            )
        # Hogwild never waits
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for the consistency model.

        Returns:
            Dict with model-specific statistics.
        """
        base_stats = {
            "type": self.consistency_type.value,
            "num_workers": self.num_workers,
        }

        if isinstance(self.model, BSPConsistency):
            base_stats["pending_barriers"] = self.model.pending_barriers
            base_stats["slowest_worker"] = self.model.get_slowest_worker()
        elif isinstance(self.model, SSPConsistency):
            ssp_stats = self.model.get_stats()
            base_stats.update(ssp_stats)

        return base_stats

    def reset(self) -> None:
        """Reset the consistency model state."""
        if hasattr(self.model, "reset"):
            self.model.reset()

    @staticmethod
    def create_hogwild() -> "ConsistencyManager":
        """Create a Hogwild consistency manager.

        Returns:
            ConsistencyManager with Hogwild model.
        """
        return ConsistencyManager(ConsistencyType.HOGWILD)

    @staticmethod
    def create_bsp(num_workers: int) -> "ConsistencyManager":
        """Create a BSP consistency manager.

        Args:
            num_workers: Number of workers.

        Returns:
            ConsistencyManager with BSP model.
        """
        return ConsistencyManager(ConsistencyType.BSP, num_workers=num_workers)

    @staticmethod
    def create_ssp(
        num_workers: int = 1,
        staleness_threshold: int = 3,
    ) -> "ConsistencyManager":
        """Create an SSP consistency manager.

        Args:
            num_workers: Number of workers.
            staleness_threshold: Maximum staleness allowed.

        Returns:
            ConsistencyManager with SSP model.
        """
        return ConsistencyManager(
            ConsistencyType.SSP,
            num_workers=num_workers,
            staleness_threshold=staleness_threshold,
        )


def create_consistency_model(
    model_type: str,
    num_workers: int = 1,
    **kwargs: Any,
) -> ConsistencyModel:
    """Factory function to create a consistency model.

    Args:
        model_type: Type of model ("hogwild", "bsp", "ssp").
        num_workers: Number of workers (for BSP).
        **kwargs: Additional model-specific arguments.

    Returns:
        Configured ConsistencyModel instance.
    """
    model_type = model_type.lower()

    if model_type == "hogwild":
        return HogwildConsistency()
    elif model_type == "bsp":
        return BSPConsistency(num_workers=num_workers)
    elif model_type == "ssp":
        staleness = kwargs.get("staleness_threshold", 3)
        return SSPConsistency(staleness_threshold=staleness)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Valid types: hogwild, bsp, ssp"
        )
