"""Model selection and routing for SLM pipeline."""

from typing import Callable, Any
import logging

from ..schemas import (
    ModelInfo,
    SelectionConstraints,
    PerformanceStats,
)

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Tracks performance statistics for models."""

    def __init__(self):
        self._stats: dict[str, PerformanceStats] = {}

    def record(
        self,
        model_name: str,
        latency_ms: float,
        quality_score: float,
        error: bool = False
    ):
        """Record performance metrics.

        Args:
            model_name: Model identifier
            latency_ms: Request latency
            quality_score: Quality score
            error: Whether an error occurred
        """
        if model_name not in self._stats:
            self._stats[model_name] = PerformanceStats()

        stats = self._stats[model_name]
        n = stats.call_count

        # Update running averages
        stats.avg_latency = (stats.avg_latency * n + latency_ms) / (n + 1)
        stats.avg_quality = (stats.avg_quality * n + quality_score) / (n + 1)

        if error:
            stats.error_rate = (stats.error_rate * n + 1) / (n + 1)
        else:
            stats.error_rate = (stats.error_rate * n) / (n + 1)

        stats.call_count = n + 1

    def get_stats(self, model_name: str) -> PerformanceStats:
        """Get statistics for a model."""
        return self._stats.get(model_name, PerformanceStats())

    def reset(self, model_name: str = None):
        """Reset statistics."""
        if model_name:
            self._stats.pop(model_name, None)
        else:
            self._stats.clear()


class ModelSelector:
    """Intelligent model selection based on task and context."""

    def __init__(
        self,
        models: dict[str, ModelInfo],
        selection_strategy: str = "quality_latency_balance"
    ):
        """Initialize selector.

        Args:
            models: Available models
            selection_strategy: Selection strategy name
        """
        self.models = models
        self.strategy = selection_strategy
        self.performance_tracker = PerformanceTracker()

    async def select(
        self,
        task: str,
        context: dict[str, Any] = None,
        constraints: SelectionConstraints = None
    ) -> str:
        """Select best model for task given constraints.

        Args:
            task: Task type
            context: Task context
            constraints: Selection constraints

        Returns:
            Selected model name
        """
        context = context or {}
        candidates = self._filter_candidates(task, constraints)

        if not candidates:
            raise ValueError(f"No suitable models found for task: {task}")

        if self.strategy == "quality_latency_balance":
            return self._balance_selection(candidates, context)
        elif self.strategy == "lowest_latency":
            return self._latency_selection(candidates)
        elif self.strategy == "highest_quality":
            return self._quality_selection(candidates)
        else:
            return candidates[0]

    def _filter_candidates(
        self,
        task: str,
        constraints: SelectionConstraints = None
    ) -> list[str]:
        """Filter models based on task and constraints."""
        candidates = []

        for name, info in self.models.items():
            if info.task != task:
                continue

            if constraints:
                if constraints.max_latency_ms and info.avg_latency_ms > constraints.max_latency_ms:
                    continue
                if constraints.min_quality and info.quality_score < constraints.min_quality:
                    continue
                if constraints.max_cost and info.cost_per_1k > constraints.max_cost:
                    continue

            candidates.append(name)

        return candidates

    def _balance_selection(
        self,
        candidates: list[str],
        context: dict[str, Any]
    ) -> str:
        """Balance quality and latency."""
        scores = {}

        for model_name in candidates:
            info = self.models[model_name]
            perf = self.performance_tracker.get_stats(model_name)

            # Use historical stats if available, else model defaults
            avg_quality = perf.avg_quality if perf.call_count > 0 else info.quality_score
            avg_latency = perf.avg_latency if perf.call_count > 0 else info.avg_latency_ms

            # Normalize metrics (0-1)
            quality_score = avg_quality / 10
            latency_score = max(0, 1 - (avg_latency / 1000))

            # Weight based on context
            if context.get("latency_sensitive"):
                scores[model_name] = 0.3 * quality_score + 0.7 * latency_score
            else:
                scores[model_name] = 0.7 * quality_score + 0.3 * latency_score

        return max(scores, key=scores.get)

    def _latency_selection(self, candidates: list[str]) -> str:
        """Select lowest latency model."""
        return min(
            candidates,
            key=lambda m: self.models[m].avg_latency_ms
        )

    def _quality_selection(self, candidates: list[str]) -> str:
        """Select highest quality model."""
        return max(
            candidates,
            key=lambda m: self.models[m].quality_score
        )


class FallbackChain:
    """Manages model fallback chains."""

    def __init__(self, chains: dict[str, list[str]]):
        """Initialize fallback chain.

        Args:
            chains: Task to fallback list mapping
        """
        self.chains = chains

    async def execute_with_fallback(
        self,
        task: str,
        executor: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute with automatic fallback on failure.

        Args:
            task: Task type
            executor: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Execution result

        Raises:
            RuntimeError: If all fallbacks fail
        """
        chain = self.chains.get(task, [])

        for model_name in chain:
            try:
                return await executor(model_name, *args, **kwargs)
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}, trying fallback")
                continue

        raise RuntimeError(f"All models in fallback chain failed for {task}")
