"""Routing engine for model routing layer."""

from typing import Any
import logging

from ..schemas import InferenceRequest, WorkerInfo
from .cost import CostComputer

logger = logging.getLogger(__name__)


class NoWorkersAvailable(Exception):
    """No workers available for model."""
    pass


class RoutingEngine:
    """Routes requests to optimal workers."""

    def __init__(
        self,
        worker_registry,
        cost_computer: CostComputer = None,
        routing_strategy: str = "least_loaded",
        rl_router=None,
    ):
        """Initialize routing engine.

        Args:
            worker_registry: Worker registry
            cost_computer: Cost computer
            routing_strategy: Default routing strategy
            rl_router: Optional RLRouter for RL-based routing
        """
        self.registry = worker_registry
        self.cost_computer = cost_computer or CostComputer()
        self.strategy = routing_strategy
        self.rl_router = rl_router
        self._round_robin_idx: dict[str, int] = {}

    async def route(
        self,
        request: InferenceRequest
    ) -> WorkerInfo:
        """Route request to best worker.

        Args:
            request: Inference request

        Returns:
            Selected worker

        Raises:
            NoWorkersAvailable: If no workers for model
        """
        # Get available workers for model
        workers = await self.registry.get_workers(
            model=request.model,
            status="healthy"
        )

        if not workers:
            raise NoWorkersAvailable(f"No workers for model {request.model}")

        # Apply routing strategy
        if self.strategy == "least_loaded":
            return self._least_loaded(request, workers)
        elif self.strategy == "round_robin":
            return self._round_robin(request, workers)
        elif self.strategy == "token_based":
            return self._token_based(request, workers)
        elif self.strategy == "sla_based":
            return self._sla_based(request, workers)
        elif self.strategy == "weighted":
            return self._weighted_selection(request, workers)
        elif self.strategy == "rl":
            if self.rl_router is None:
                raise ValueError("RL strategy requires rl_router to be set")
            return self.rl_router.route(request, workers)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _least_loaded(
        self,
        request: InferenceRequest,
        workers: list[WorkerInfo]
    ) -> WorkerInfo:
        """Select worker with lowest current load.

        Args:
            request: Inference request
            workers: Available workers

        Returns:
            Selected worker
        """
        return min(workers, key=lambda w: w.current_load)

    def _round_robin(
        self,
        request: InferenceRequest,
        workers: list[WorkerInfo]
    ) -> WorkerInfo:
        """Round-robin selection.

        Args:
            request: Inference request
            workers: Available workers

        Returns:
            Selected worker
        """
        model = request.model
        if model not in self._round_robin_idx:
            self._round_robin_idx[model] = 0

        idx = self._round_robin_idx[model] % len(workers)
        self._round_robin_idx[model] = idx + 1

        return workers[idx]

    def _token_based(
        self,
        request: InferenceRequest,
        workers: list[WorkerInfo]
    ) -> WorkerInfo:
        """Select worker with most available token budget.

        Args:
            request: Inference request
            workers: Available workers

        Returns:
            Selected worker
        """
        def token_capacity(w):
            return w.token_budget - w.tokens_in_flight

        return max(workers, key=token_capacity)

    def _sla_based(
        self,
        request: InferenceRequest,
        workers: list[WorkerInfo]
    ) -> WorkerInfo:
        """Select worker most likely to meet SLA.

        Args:
            request: Inference request
            workers: Available workers

        Returns:
            Selected worker
        """
        if not request.sla_deadline_ms:
            return self._least_loaded(request, workers)

        # Predict latency for each worker
        predictions = []
        for worker in workers:
            latency = self.cost_computer.latency_predictor.predict(
                request, worker
            )
            predictions.append((worker, latency))

        # Filter workers that can meet SLA
        viable = [
            (w, lat) for w, lat in predictions
            if lat < request.sla_deadline_ms * 0.8  # 20% buffer
        ]

        if viable:
            # Among viable, pick least loaded
            return min(viable, key=lambda x: x[0].current_load)[0]
        else:
            # Fall back to fastest prediction
            return min(predictions, key=lambda x: x[1])[0]

    def _weighted_selection(
        self,
        request: InferenceRequest,
        workers: list[WorkerInfo]
    ) -> WorkerInfo:
        """Weighted routing based on multiple factors.

        Args:
            request: Inference request
            workers: Available workers

        Returns:
            Selected worker
        """
        weights = {
            "load": 0.4,
            "latency": 0.3,
            "cost": 0.2,
            "sla": 0.1
        }

        scores = []

        for worker in workers:
            cost = self.cost_computer.compute(request, worker)

            # Normalize metrics to 0-1 (lower is better)
            load_score = worker.current_load
            latency_score = cost.estimated_latency_ms / 10000  # Normalize to 10s
            cost_score = cost.dollar_cost / 0.1  # Normalize to $0.10

            # SLA score
            if request.sla_deadline_ms:
                sla_score = max(0, cost.estimated_latency_ms / request.sla_deadline_ms)
            else:
                sla_score = 0

            # Weighted sum
            total_score = (
                weights["load"] * load_score +
                weights["latency"] * latency_score +
                weights["cost"] * cost_score +
                weights["sla"] * sla_score
            )

            scores.append((worker, total_score))

        # Return worker with lowest score
        return min(scores, key=lambda x: x[1])[0]

    def set_strategy(self, strategy: str):
        """Set routing strategy.

        Args:
            strategy: Strategy name
        """
        self.strategy = strategy


class LocalityAwareStrategy:
    """Route based on data locality."""

    def __init__(self, locality_cache=None):
        """Initialize locality-aware strategy.

        Args:
            locality_cache: Cache tracking data locality
        """
        self.cache = locality_cache or {}

    def select(
        self,
        request: InferenceRequest,
        workers: list[WorkerInfo]
    ) -> WorkerInfo:
        """Select worker based on locality.

        Args:
            request: Inference request
            workers: Available workers

        Returns:
            Selected worker
        """
        # Check if any worker has cached context
        context_key = request.metadata.get("context_key")
        if context_key:
            for worker in workers:
                if self._has_context(worker.worker_id, context_key):
                    return worker

        # Fall back to least loaded
        return min(workers, key=lambda w: w.current_load)

    def _has_context(self, worker_id: str, context_key: str) -> bool:
        """Check if worker has cached context.

        Args:
            worker_id: Worker ID
            context_key: Context key

        Returns:
            True if cached
        """
        return context_key in self.cache.get(worker_id, set())

    def add_context(self, worker_id: str, context_key: str):
        """Record that worker has context.

        Args:
            worker_id: Worker ID
            context_key: Context key
        """
        if worker_id not in self.cache:
            self.cache[worker_id] = set()
        self.cache[worker_id].add(context_key)
