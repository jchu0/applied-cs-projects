"""Gradient aggregation implementations."""

import numpy as np
import asyncio
from typing import Any
import logging

from ..schemas import Gradient, AggregationConfig, AggregationType

logger = logging.getLogger(__name__)


class GradientAggregator:
    """Aggregates gradients from multiple workers."""

    def __init__(self, config: AggregationConfig = None):
        """Initialize aggregator.

        Args:
            config: Aggregation configuration
        """
        self.config = config or AggregationConfig()
        self._pending: dict[str, list[Gradient]] = {}
        self._momentum_buffer: dict[str, np.ndarray] = {}
        self._lock = asyncio.Lock()

    async def add_gradient(self, gradient: Gradient):
        """Add gradient to pending aggregation.

        Args:
            gradient: Gradient to add
        """
        async with self._lock:
            name = gradient.name
            if name not in self._pending:
                self._pending[name] = []
            self._pending[name].append(gradient)

    async def aggregate(
        self,
        parameter_name: str,
        expected_count: int = None
    ) -> np.ndarray | None:
        """Aggregate pending gradients.

        Args:
            parameter_name: Parameter name
            expected_count: Expected number of gradients

        Returns:
            Aggregated gradient or None
        """
        async with self._lock:
            if parameter_name not in self._pending:
                return None

            gradients = self._pending[parameter_name]
            if expected_count and len(gradients) < expected_count:
                return None

            # Stack gradient data
            grad_data = [g.data for g in gradients]
            stacked = np.stack(grad_data)

            # Aggregate based on type
            if self.config.aggregation_type == AggregationType.SUM:
                aggregated = np.sum(stacked, axis=0)
            elif self.config.aggregation_type == AggregationType.MEAN:
                aggregated = np.mean(stacked, axis=0)
            elif self.config.aggregation_type == AggregationType.WEIGHTED:
                # Could implement weighted based on worker metadata
                aggregated = np.mean(stacked, axis=0)
            else:
                aggregated = np.mean(stacked, axis=0)

            # Apply gradient clipping
            if self.config.clip_norm:
                aggregated = self._clip_gradient(aggregated)

            # Apply momentum
            if self.config.momentum > 0:
                aggregated = self._apply_momentum(parameter_name, aggregated)

            # Clear pending
            self._pending[parameter_name] = []

            return aggregated

    def _clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Clip gradient by norm.

        Args:
            gradient: Gradient to clip

        Returns:
            Clipped gradient
        """
        norm = np.linalg.norm(gradient)
        if norm > self.config.clip_norm:
            gradient = gradient * (self.config.clip_norm / norm)
        return gradient

    def _apply_momentum(
        self,
        name: str,
        gradient: np.ndarray
    ) -> np.ndarray:
        """Apply momentum to gradient.

        Args:
            name: Parameter name
            gradient: Current gradient

        Returns:
            Gradient with momentum
        """
        if name not in self._momentum_buffer:
            self._momentum_buffer[name] = np.zeros_like(gradient)

        self._momentum_buffer[name] = (
            self.config.momentum * self._momentum_buffer[name] +
            gradient
        )

        return self._momentum_buffer[name]

    async def get_pending_count(self, parameter_name: str) -> int:
        """Get count of pending gradients.

        Args:
            parameter_name: Parameter name

        Returns:
            Count
        """
        async with self._lock:
            return len(self._pending.get(parameter_name, []))

    async def clear_pending(self, parameter_name: str = None):
        """Clear pending gradients.

        Args:
            parameter_name: Parameter name (None for all)
        """
        async with self._lock:
            if parameter_name:
                self._pending.pop(parameter_name, None)
            else:
                self._pending.clear()


class AsyncAggregator(GradientAggregator):
    """Asynchronous gradient aggregation with immediate updates."""

    def __init__(self, config: AggregationConfig = None):
        """Initialize async aggregator."""
        super().__init__(config)
        self._update_count: dict[str, int] = {}

    async def aggregate_immediate(
        self,
        gradient: Gradient
    ) -> np.ndarray:
        """Aggregate gradient immediately (ASP mode).

        Args:
            gradient: Gradient to aggregate

        Returns:
            Aggregated gradient (single gradient in this case)
        """
        name = gradient.name
        grad_data = gradient.data

        # Apply clipping
        if self.config.clip_norm:
            grad_data = self._clip_gradient(grad_data)

        # Apply momentum
        if self.config.momentum > 0:
            grad_data = self._apply_momentum(name, grad_data)

        # Track update count
        self._update_count[name] = self._update_count.get(name, 0) + 1

        return grad_data

    def get_update_count(self, parameter_name: str) -> int:
        """Get number of updates for parameter.

        Args:
            parameter_name: Parameter name

        Returns:
            Update count
        """
        return self._update_count.get(parameter_name, 0)


class SparsifiedAggregator(GradientAggregator):
    """Aggregator with gradient sparsification."""

    def __init__(
        self,
        config: AggregationConfig = None,
        top_k: float = 0.1
    ):
        """Initialize sparsified aggregator.

        Args:
            config: Aggregation config
            top_k: Fraction of elements to keep
        """
        super().__init__(config)
        self.top_k = top_k
        self._residuals: dict[str, np.ndarray] = {}

    async def aggregate_sparse(
        self,
        parameter_name: str,
        expected_count: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate with sparsification.

        Args:
            parameter_name: Parameter name
            expected_count: Expected gradient count

        Returns:
            Tuple of (sparse gradient, indices)
        """
        aggregated = await self.aggregate(parameter_name, expected_count)
        if aggregated is None:
            return None, None

        # Add residual from previous iteration
        if parameter_name in self._residuals:
            aggregated = aggregated + self._residuals[parameter_name]

        # Find top-k elements
        flat = aggregated.flatten()
        k = max(1, int(len(flat) * self.top_k))
        indices = np.argpartition(np.abs(flat), -k)[-k:]

        # Create sparse gradient
        sparse = np.zeros_like(flat)
        sparse[indices] = flat[indices]

        # Store residual
        self._residuals[parameter_name] = (
            aggregated.flatten() - sparse
        ).reshape(aggregated.shape)

        return sparse.reshape(aggregated.shape), indices
