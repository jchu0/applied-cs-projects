"""Cost computation for scheduling decisions."""

from typing import Any

from ..schemas import InferenceRequest, WorkerInfo, RequestCost, ModelPricing


class LatencyPredictor:
    """Predicts request latency based on historical data."""

    def __init__(self, history_store=None):
        """Initialize predictor.

        Args:
            history_store: Historical latency storage
        """
        self.history = history_store or {}

    def predict(
        self,
        request: InferenceRequest,
        worker: WorkerInfo
    ) -> int:
        """Predict latency in milliseconds.

        Args:
            request: Inference request
            worker: Worker info

        Returns:
            Predicted latency in ms
        """
        # Get historical latencies for similar requests
        historical = self._get_similar(
            model=request.model,
            token_range=(
                request.estimated_tokens * 0.8,
                request.estimated_tokens * 1.2
            ),
            worker_type=worker.worker_type
        )

        if historical:
            # Use median of historical
            sorted_hist = sorted(historical)
            base_latency = sorted_hist[len(sorted_hist) // 2]
        else:
            # Fallback to formula: ~10ms per token
            base_latency = request.estimated_tokens * 10

        # Adjust for current load
        load_factor = 1 + worker.current_load * 0.5

        # Adjust for queue depth
        queue_factor = 1 + worker.queue_depth * 0.1

        return int(base_latency * load_factor * queue_factor)

    def _get_similar(
        self,
        model: str,
        token_range: tuple[float, float],
        worker_type: str
    ) -> list[float]:
        """Get historical latencies for similar requests.

        Args:
            model: Model name
            token_range: Token count range
            worker_type: Worker type

        Returns:
            List of historical latencies
        """
        key = f"{model}:{worker_type}"
        if key not in self.history:
            return []

        min_tokens, max_tokens = token_range
        return [
            lat for tokens, lat in self.history[key]
            if min_tokens <= tokens <= max_tokens
        ]

    def record(
        self,
        model: str,
        worker_type: str,
        tokens: int,
        latency_ms: float
    ):
        """Record observed latency.

        Args:
            model: Model name
            worker_type: Worker type
            tokens: Token count
            latency_ms: Observed latency
        """
        key = f"{model}:{worker_type}"
        if key not in self.history:
            self.history[key] = []

        self.history[key].append((tokens, latency_ms))

        # Keep last 1000 entries
        if len(self.history[key]) > 1000:
            self.history[key] = self.history[key][-1000:]


class CostComputer:
    """Computes cost metrics for scheduling decisions."""

    def __init__(
        self,
        pricing_config: dict[str, ModelPricing] = None,
        latency_predictor: LatencyPredictor = None
    ):
        """Initialize cost computer.

        Args:
            pricing_config: Model pricing configurations
            latency_predictor: Latency predictor
        """
        self.pricing = pricing_config or {}
        self.latency_predictor = latency_predictor or LatencyPredictor()

    def compute(
        self,
        request: InferenceRequest,
        worker: WorkerInfo
    ) -> RequestCost:
        """Compute request cost.

        Args:
            request: Inference request
            worker: Target worker

        Returns:
            Computed cost metrics
        """
        # Get pricing
        pricing = self.pricing.get(request.model, ModelPricing(
            model=request.model,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.03
        ))

        # Token cost
        tokens = request.estimated_tokens

        # Compute units (GPU-seconds)
        compute_units = self._estimate_compute_units(request, worker)

        # Latency prediction
        estimated_latency = self.latency_predictor.predict(request, worker)

        # Dollar cost
        input_tokens = tokens - request.max_tokens
        output_tokens = request.max_tokens

        dollar_cost = (
            input_tokens * pricing.input_cost_per_1k / 1000 +
            output_tokens * pricing.output_cost_per_1k / 1000 +
            compute_units * pricing.compute_cost_per_unit
        )

        return RequestCost(
            tokens=tokens,
            compute_units=compute_units,
            estimated_latency_ms=estimated_latency,
            dollar_cost=dollar_cost
        )

    def _estimate_compute_units(
        self,
        request: InferenceRequest,
        worker: WorkerInfo
    ) -> float:
        """Estimate GPU-seconds needed.

        Args:
            request: Inference request
            worker: Target worker

        Returns:
            Estimated GPU-seconds
        """
        # Base compute per token
        tokens = request.estimated_tokens
        base_compute = tokens * 0.001  # 1ms per token baseline

        # Adjust for worker performance
        worker_factor = worker.performance_factor

        return base_compute / worker_factor

    def set_pricing(self, model: str, pricing: ModelPricing):
        """Set pricing for a model.

        Args:
            model: Model name
            pricing: Pricing configuration
        """
        self.pricing[model] = pricing
