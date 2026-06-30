"""Hogwild! consistency model - no synchronization."""

from paramserver.consistency.base import ConsistencyModel


class HogwildConsistency(ConsistencyModel):
    """Hogwild! consistency model with no synchronization.

    In Hogwild! mode, gradient updates are applied immediately without
    any synchronization barriers. This provides maximum parallelism but
    allows workers to read stale parameters.

    This approach works well when:
    - The optimization problem is sparse
    - Gradient conflicts are rare
    - Speed is more important than exact consistency

    References:
        Recht, B., et al. "Hogwild!: A Lock-Free Approach to Parallelizing
        Stochastic Gradient Descent." NeurIPS 2011.
    """

    def can_apply(
        self,
        param_version: int,
        worker_clock: int,
    ) -> bool:
        """Check if update can be applied - always True for Hogwild!

        Args:
            param_version: Current version of the parameter (ignored).
            worker_clock: Worker's logical clock (ignored).

        Returns:
            Always True - updates are applied immediately.
        """
        return True
