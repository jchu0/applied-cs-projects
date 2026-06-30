"""Pytest fixtures for Model Routing Layer tests."""

import pytest
import asyncio
import time
from typing import Generator

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from modelrouter import (
    # Schemas
    Priority,
    InferenceRequest,
    InferenceResponse,
    GPUInfo,
    WorkerInfo,
    Tenant,
    TenantQuota,
    ModelPricing,
    CanaryConfig,
    RLExperience,
    CongestionMetrics,
    CongestionThrottled,
    generate_id,
    # Gateway
    Gateway,
    TenantAuthenticator,
    RateLimiter,
    TokenEstimator,
    # Scheduler
    QueueManager,
    CostComputer,
    LatencyPredictor,
    RoutingEngine,
    LocalityAwareStrategy,
    # Registry
    WorkerRegistry,
    CapacityTracker,
    HealthChecker,
    GPUMonitor,
    # Enterprise
    QuotaManager,
    PreemptionManager,
    TrafficSplitter,
    # Optimization
    ExperienceBuffer,
    DQNPolicy,
    RLRouter,
    MetricsCollector,
    CongestionModel,
    CongestionPredictor,
    # Router
    ModelRouter,
    create_router,
)


def make_gpu_info() -> GPUInfo:
    """Create a sample GPU info."""
    return GPUInfo(
        device_name="NVIDIA A100",
        memory_total_mb=40960,
        memory_used_mb=8192,
        utilization_percent=45.0,
        temperature_celsius=65
    )


def make_worker(worker_id: str = "worker-1", current_load: float = 0.3) -> WorkerInfo:
    """Create a sample worker."""
    return WorkerInfo(
        worker_id=worker_id,
        host="localhost",
        port=8080,
        models=["gpt-4", "gpt-3.5-turbo"],
        gpu_info=make_gpu_info(),
        current_load=current_load,
        queue_depth=5,
        tokens_in_flight=1000,
        token_budget=100000,
        status="healthy",
        last_heartbeat=time.time(),
        performance_factor=1.0,
        worker_type="standard"
    )


def make_workers() -> list[WorkerInfo]:
    """Create multiple sample workers with varying loads."""
    gpu_info = make_gpu_info()
    workers = []
    for i in range(3):
        workers.append(WorkerInfo(
            worker_id=f"worker-{i+1}",
            host="localhost",
            port=8080 + i,
            models=["gpt-4", "gpt-3.5-turbo"],
            gpu_info=gpu_info,
            current_load=0.2 + (i * 0.2),  # 0.2, 0.4, 0.6
            queue_depth=5 + i * 5,  # 5, 10, 15
            tokens_in_flight=1000 + i * 1000,  # 1000, 2000, 3000
            token_budget=100000,
            status="healthy",
            last_heartbeat=time.time(),
            performance_factor=1.0 - (i * 0.1),  # 1.0, 0.9, 0.8
            worker_type="standard"
        ))
    return workers


def make_tenant() -> Tenant:
    """Create a sample tenant."""
    return Tenant(
        id="tenant-1",
        name="Test Tenant",
        api_key="test-api-key-123",
        default_priority=Priority.NORMAL
    )


def make_tenant_quota() -> TenantQuota:
    """Create a sample tenant quota."""
    return TenantQuota(
        tenant_id="tenant-1",
        requests_per_second=100,
        tokens_per_minute=100000,
        monthly_token_budget=10000000,
        priority_access=True
    )


def make_request(
    priority: Priority = Priority.NORMAL,
    model: str = "gpt-4",
    tenant_id: str = "tenant-1",
    estimated_tokens: int = 125
) -> InferenceRequest:
    """Create a sample inference request."""
    return InferenceRequest(
        request_id=generate_id(),
        tenant_id=tenant_id,
        model=model,
        prompt="What is the capital of France?",
        max_tokens=100,
        temperature=0.7,
        priority=priority,
        sla_deadline_ms=10000,
        estimated_tokens=estimated_tokens
    )


def make_model_pricing() -> ModelPricing:
    """Create sample model pricing."""
    return ModelPricing(
        model="gpt-4",
        input_cost_per_1k=0.03,
        output_cost_per_1k=0.06,
        compute_cost_per_unit=0.001
    )


def make_raw_request() -> dict:
    """Create a raw API request."""
    return {
        "api_key": "test-api-key-123",
        "model": "gpt-4",
        "prompt": "What is the capital of France?",
        "max_tokens": 100,
        "temperature": 0.7
    }


# Fixtures that use the factory functions

@pytest.fixture
def sample_gpu_info() -> GPUInfo:
    """Create a sample GPU info."""
    return make_gpu_info()


@pytest.fixture
def sample_worker(sample_gpu_info) -> WorkerInfo:
    """Create a sample worker."""
    return make_worker()


@pytest.fixture
def sample_workers(sample_gpu_info) -> list[WorkerInfo]:
    """Create multiple sample workers with varying loads."""
    return make_workers()


@pytest.fixture
def sample_tenant() -> Tenant:
    """Create a sample tenant."""
    return make_tenant()


@pytest.fixture
def sample_tenant_quota() -> TenantQuota:
    """Create a sample tenant quota."""
    return make_tenant_quota()


@pytest.fixture
def sample_request() -> InferenceRequest:
    """Create a sample inference request."""
    return make_request()


@pytest.fixture
def high_priority_request() -> InferenceRequest:
    """Create a high priority inference request."""
    return InferenceRequest(
        request_id=generate_id(),
        tenant_id="tenant-1",
        model="gpt-4",
        prompt="URGENT: System alert analysis",
        max_tokens=50,
        temperature=0.5,
        priority=Priority.HIGH,
        sla_deadline_ms=3000,
        estimated_tokens=65
    )


@pytest.fixture
def critical_request() -> InferenceRequest:
    """Create a critical priority inference request."""
    return InferenceRequest(
        request_id=generate_id(),
        tenant_id="tenant-1",
        model="gpt-4",
        prompt="CRITICAL: Production incident",
        max_tokens=50,
        temperature=0.3,
        priority=Priority.CRITICAL,
        sla_deadline_ms=1000,
        estimated_tokens=65
    )


@pytest.fixture
def batch_request() -> InferenceRequest:
    """Create a batch priority inference request."""
    return InferenceRequest(
        request_id=generate_id(),
        tenant_id="tenant-1",
        model="gpt-4",
        prompt="Process these documents: " + "x" * 1000,
        max_tokens=500,
        temperature=0.7,
        priority=Priority.BATCH,
        sla_deadline_ms=300000,
        estimated_tokens=750
    )


@pytest.fixture
def sample_model_pricing() -> ModelPricing:
    """Create sample model pricing."""
    return make_model_pricing()


@pytest.fixture
def raw_request() -> dict:
    """Create a raw API request."""
    return make_raw_request()


# Component Fixtures

@pytest.fixture
def worker_registry() -> WorkerRegistry:
    """Create a worker registry."""
    return WorkerRegistry()


@pytest.fixture
def queue_manager() -> QueueManager:
    """Create a queue manager."""
    return QueueManager()


@pytest.fixture
def cost_computer(sample_model_pricing) -> CostComputer:
    """Create a cost computer with sample pricing."""
    cc = CostComputer()
    cc.set_pricing("gpt-4", sample_model_pricing)
    return cc


@pytest.fixture
def latency_predictor() -> LatencyPredictor:
    """Create a latency predictor."""
    return LatencyPredictor()


@pytest.fixture
def rate_limiter() -> RateLimiter:
    """Create a rate limiter."""
    return RateLimiter()


@pytest.fixture
def token_estimator() -> TokenEstimator:
    """Create a token estimator."""
    return TokenEstimator()


@pytest.fixture
def tenant_authenticator(sample_tenant) -> TenantAuthenticator:
    """Create a tenant authenticator with a sample tenant."""
    auth = TenantAuthenticator()
    auth.register_tenant(sample_tenant)
    return auth


@pytest.fixture
def quota_manager() -> QuotaManager:
    """Create a quota manager."""
    return QuotaManager()


@pytest.fixture
def preemption_manager(queue_manager) -> PreemptionManager:
    """Create a preemption manager."""
    return PreemptionManager(queue_manager)


@pytest.fixture
def traffic_splitter() -> TrafficSplitter:
    """Create a traffic splitter."""
    return TrafficSplitter()


@pytest.fixture
def health_checker(worker_registry) -> HealthChecker:
    """Create a health checker."""
    return HealthChecker(worker_registry)


@pytest.fixture
def capacity_tracker(worker_registry) -> CapacityTracker:
    """Create a capacity tracker."""
    return CapacityTracker(worker_registry)


@pytest.fixture
def locality_strategy() -> LocalityAwareStrategy:
    """Create a locality-aware routing strategy."""
    return LocalityAwareStrategy()


# Utility fixtures

@pytest.fixture
def many_requests() -> list[InferenceRequest]:
    """Create many requests with varying priorities."""
    requests = []
    priorities = [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW, Priority.BATCH]

    for i in range(20):
        priority = priorities[i % len(priorities)]
        requests.append(InferenceRequest(
            request_id=generate_id(),
            tenant_id=f"tenant-{i % 3}",
            model="gpt-4",
            prompt=f"Request {i}: " + "x" * (100 + i * 10),
            max_tokens=100 + i * 10,
            temperature=0.7,
            priority=priority,
            sla_deadline_ms=1000 * (priority.value + 1),
            estimated_tokens=125 + i * 10
        ))

    return requests


@pytest.fixture
def canary_config() -> CanaryConfig:
    """Create a sample canary configuration."""
    return CanaryConfig(
        id="canary-1",
        model="gpt-4",
        workers=["worker-canary-1", "worker-canary-2"],
        traffic_percentage=10.0,
        active=True
    )


# Optimization fixtures

def make_experience_buffer(capacity=1000):
    """Create an experience buffer."""
    return ExperienceBuffer(capacity=capacity)


def make_dqn_policy(max_workers=16):
    """Create a DQN policy."""
    state_dim = 4 + max_workers * 6
    return DQNPolicy(state_dim=state_dim, action_dim=max_workers)


def make_rl_router(max_workers=16):
    """Create an RL router."""
    policy = make_dqn_policy(max_workers)
    buffer = make_experience_buffer()
    return RLRouter(policy, buffer, max_workers)


def make_metrics_collector():
    """Create a metrics collector."""
    return MetricsCollector(window_minutes=30)


def make_congestion_predictor():
    """Create a congestion predictor."""
    model = CongestionModel()
    collector = make_metrics_collector()
    return CongestionPredictor(model, collector)


@pytest.fixture
def experience_buffer():
    """Create an experience buffer."""
    return make_experience_buffer()


@pytest.fixture
def dqn_policy():
    """Create a DQN policy."""
    return make_dqn_policy()


@pytest.fixture
def rl_router():
    """Create an RL router."""
    return make_rl_router()


@pytest.fixture
def metrics_collector():
    """Create a metrics collector."""
    return make_metrics_collector()


@pytest.fixture
def congestion_predictor():
    """Create a congestion predictor."""
    return make_congestion_predictor()


# Async helper for running async code in sync tests
def run_async(coro):
    """Helper to run async code in sync context."""
    return asyncio.get_event_loop().run_until_complete(coro)
