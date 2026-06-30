"""Base class for consistency models."""

from abc import ABC, abstractmethod


class ConsistencyModel(ABC):
    """Abstract base class for consistency models.

    Consistency models determine when gradient updates can be applied
    to parameters based on worker clocks and parameter versions.

    Different models provide different trade-offs between consistency
    and performance:
    - Hogwild!: No synchronization, maximum parallelism
    - BSP: Full synchronization, strongest consistency
    - SSP: Bounded staleness, balance of both
    """

    @abstractmethod
    def can_apply(
        self,
        param_version: int,
        worker_clock: int,
    ) -> bool:
        """Check if a gradient update can be applied.

        Args:
            param_version: Current version of the parameter.
            worker_clock: Logical clock of the worker sending the update.

        Returns:
            True if the update can be applied, False if it should be buffered.
        """
        pass

    @property
    def name(self) -> str:
        """Return the name of this consistency model."""
        return self.__class__.__name__
