"""Main model router orchestrating all components."""

import asyncio
import time
import logging
from typing import Any

from .schemas import (
    InferenceRequest,
    InferenceResponse,
    WorkerInfo,
    GPUInfo,
    Tenant,
    Priority,
    CongestionThrottled,
    generate_id,
)
from .gateway import Gateway, TenantAuthenticator, RateLimiter, TokenEstimator
from .scheduler import QueueManager, CostComputer, RoutingEngine
from .registry import WorkerRegistry, CapacityTracker, HealthChecker
from .enterprise import QuotaManager, PreemptionManager, TrafficSplitter
from .optimization import (
    ExperienceBuffer,
    DQNPolicy,
    RLRouter,
    MetricsCollector,
    CongestionModel,
    CongestionPredictor,
)

logger = logging.getLogger(__name__)


class ModelRouter:
    """Main model routing layer orchestrator."""

    def __init__(
        self,
        routing_strategy: str = "least_loaded",
        max_queue_depth: int = 1000
    ):
        """Initialize model router.

        Args:
            routing_strategy: Routing strategy name
            max_queue_depth: Maximum queue depth
        """
        # Core components
        self.registry = WorkerRegistry()
        self.queue_manager = QueueManager()
        self.cost_computer = CostComputer()

        # Optimization components
        max_workers = 16
        state_dim = 4 + max_workers * 6
        self.experience_buffer = ExperienceBuffer()
        self.dqn_policy = DQNPolicy(state_dim=state_dim, action_dim=max_workers)
        self.rl_router = RLRouter(self.dqn_policy, self.experience_buffer, max_workers)
        self.metrics_collector = MetricsCollector()
        self.congestion_model = CongestionModel()
        self.congestion_predictor = CongestionPredictor(
            self.congestion_model, self.metrics_collector
        )

        self.routing_engine = RoutingEngine(
            self.registry,
            self.cost_computer,
            routing_strategy,
            rl_router=self.rl_router,
        )

        # Enterprise components
        self.quota_manager = QuotaManager()
        self.preemption_manager = PreemptionManager(self.queue_manager)
        self.traffic_splitter = TrafficSplitter()
        self.capacity_tracker = CapacityTracker(self.registry)
        self.health_checker = HealthChecker(self.registry)

        # Gateway
        self.rate_limiter = RateLimiter()
        self.token_estimator = TokenEstimator()
        self.authenticator = TenantAuthenticator()
        self.gateway = Gateway(
            rate_limiter=self.rate_limiter,
            authenticator=self.authenticator,
            token_estimator=self.token_estimator,
            scheduler=self
        )

        self.max_queue_depth = max_queue_depth
        self._metrics: list[dict] = []

    async def submit(self, request: InferenceRequest) -> InferenceResponse:
        """Submit request for processing.

        Args:
            request: Inference request

        Returns:
            Inference response

        Raises:
            CongestionThrottled: If congestion throttling activates for low-priority
        """
        start_time = time.time()

        # Congestion check — throttle LOW and BATCH priority only
        if request.priority.value >= Priority.LOW.value:
            if self.congestion_predictor.should_throttle(request.model):
                raise CongestionThrottled(
                    f"Model {request.model} is congested, throttling {request.priority.name} request"
                )

        # Record arrival for congestion tracking
        self.metrics_collector.record_arrival(request.model)

        # Check quotas
        await self.quota_manager.check_quota(request.tenant_id, request)

        # Add to queue
        await self.queue_manager.enqueue(request)

        # Check for preemption opportunity
        if request.priority.value <= Priority.HIGH.value:
            await self.preemption_manager.maybe_preempt(request, request.model)

        # Route to worker
        worker = await self.routing_engine.route(request)

        # Execute request (mock)
        response = await self._execute_on_worker(request, worker)

        # Record completion for congestion tracking
        self.metrics_collector.record_completion(request.model)

        # Feed reward to RL router
        latency = (time.time() - start_time) * 1000
        reward = RLRouter.compute_reward(latency, request.sla_deadline_ms)
        self.rl_router.update(request.request_id, reward)

        # Record usage
        await self.quota_manager.record_usage(
            request.tenant_id,
            response.tokens_used
        )

        # Update metrics
        self._record_metric(request, response, latency)

        return response

    async def _execute_on_worker(
        self,
        request: InferenceRequest,
        worker: WorkerInfo
    ) -> InferenceResponse:
        """Execute request on worker.

        Args:
            request: Inference request
            worker: Target worker

        Returns:
            Inference response
        """
        # Register as in-flight
        self.preemption_manager.register_in_flight(request)

        try:
            # Mock execution
            # In production, would call worker API
            await asyncio.sleep(0.01)  # Simulate processing

            # Dequeue
            await self.queue_manager.dequeue(request.model)

            response = InferenceResponse(
                request_id=request.request_id,
                text=f"[Mock response for {request.prompt[:50]}...]",
                tokens_used=request.estimated_tokens,
                latency_ms=int(request.estimated_tokens * 10),
                worker_id=worker.worker_id,
                model=request.model
            )

            return response

        finally:
            self.preemption_manager.complete_request(request.request_id)

    def _record_metric(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
        latency_ms: float
    ):
        """Record routing metrics.

        Args:
            request: Request
            response: Response
            latency_ms: Total latency
        """
        self._metrics.append({
            "request_id": request.request_id,
            "tenant_id": request.tenant_id,
            "model": request.model,
            "priority": request.priority.name,
            "tokens": response.tokens_used,
            "latency_ms": latency_ms,
            "worker_id": response.worker_id,
            "timestamp": time.time()
        })

        # Update latency predictor
        worker = None  # Would get from registry
        if worker:
            self.cost_computer.latency_predictor.record(
                request.model,
                worker.worker_type,
                response.tokens_used,
                latency_ms
            )

    async def register_worker(
        self,
        worker_id: str,
        host: str,
        port: int,
        models: list[str],
        token_budget: int = 100000
    ) -> str:
        """Register a worker.

        Args:
            worker_id: Worker ID
            host: Worker host
            port: Worker port
            models: Supported models
            token_budget: Token budget

        Returns:
            Worker ID
        """
        worker = WorkerInfo(
            worker_id=worker_id,
            host=host,
            port=port,
            models=models,
            gpu_info=GPUInfo(
                device_name="NVIDIA A100",
                memory_total_mb=40960,
                memory_used_mb=0,
                utilization_percent=0,
                temperature_celsius=50
            ),
            current_load=0,
            queue_depth=0,
            tokens_in_flight=0,
            token_budget=token_budget,
            status="healthy",
            last_heartbeat=time.time()
        )

        return await self.registry.register(worker)

    def register_tenant(
        self,
        tenant_id: str,
        name: str,
        api_key: str,
        monthly_budget: int = 10000000
    ):
        """Register a tenant.

        Args:
            tenant_id: Tenant ID
            name: Tenant name
            api_key: API key
            monthly_budget: Monthly token budget
        """
        tenant = Tenant(
            id=tenant_id,
            name=name,
            api_key=api_key
        )
        self.authenticator.register_tenant(tenant)

    async def handle_request(self, raw_request: dict) -> InferenceResponse:
        """Handle raw request through gateway.

        Args:
            raw_request: Raw request data

        Returns:
            Inference response
        """
        return await self.gateway.handle_request(raw_request)

    def set_routing_strategy(self, strategy: str):
        """Set routing strategy.

        Args:
            strategy: Strategy name
        """
        self.routing_engine.set_strategy(strategy)

    async def get_capacity(self, model: str = None) -> dict:
        """Get capacity information.

        Args:
            model: Optional model filter

        Returns:
            Capacity info
        """
        if model:
            return await self.capacity_tracker.get_capacity(model)
        return await self.capacity_tracker.get_all_capacity()

    async def get_queue_stats(self) -> dict:
        """Get queue statistics.

        Returns:
            Queue stats
        """
        return await self.queue_manager.get_all_stats()

    def get_metrics(self, limit: int = 100) -> list[dict]:
        """Get routing metrics.

        Args:
            limit: Maximum metrics to return

        Returns:
            List of metrics
        """
        return self._metrics[-limit:]


def create_router(
    routing_strategy: str = "least_loaded",
    max_queue_depth: int = 1000
) -> ModelRouter:
    """Create a configured model router.

    Args:
        routing_strategy: Routing strategy
        max_queue_depth: Max queue depth

    Returns:
        Configured router
    """
    return ModelRouter(
        routing_strategy=routing_strategy,
        max_queue_depth=max_queue_depth
    )


# Example usage
EXAMPLE_REQUEST = {
    "api_key": "test-key",
    "model": "gpt-4",
    "prompt": "What is the capital of France?",
    "max_tokens": 100,
    "temperature": 0.7
}
