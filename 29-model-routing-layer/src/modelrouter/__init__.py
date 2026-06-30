"""Model Routing Layer - Capacity-aware scheduling for LLM inference."""

from .schemas import (
    Priority,
    WorkerStatus,
    InferenceRequest,
    InferenceResponse,
    GPUInfo,
    WorkerInfo,
    Tenant,
    TenantQuota,
    ModelPricing,
    RequestCost,
    CapacityInfo,
    QueuedRequest,
    UsageReport,
    CanaryConfig,
    RLExperience,
    CongestionMetrics,
    CongestionThrottled,
    generate_id,
)
from .gateway import (
    Gateway,
    TenantAuthenticator,
    AuthenticationError,
    RateLimiter,
    RateLimitExceeded,
    TokenEstimator,
    create_gateway,
)
from .scheduler import (
    QueueManager,
    CostComputer,
    LatencyPredictor,
    RoutingEngine,
    NoWorkersAvailable,
    LocalityAwareStrategy,
)
from .registry import (
    WorkerRegistry,
    CapacityTracker,
    HealthChecker,
    GPUMonitor,
)
from .enterprise import (
    QuotaManager,
    QuotaExceeded,
    PreemptionManager,
    TrafficSplitter,
)
from .optimization import (
    ExperienceBuffer,
    DQNPolicy,
    RLRouter,
    MetricsCollector,
    CongestionModel,
    CongestionPredictor,
)
from .router import ModelRouter, create_router, EXAMPLE_REQUEST
from .api import create_app

__version__ = "0.1.0"

__all__ = [
    # Schemas
    "Priority",
    "WorkerStatus",
    "InferenceRequest",
    "InferenceResponse",
    "GPUInfo",
    "WorkerInfo",
    "Tenant",
    "TenantQuota",
    "ModelPricing",
    "RequestCost",
    "CapacityInfo",
    "QueuedRequest",
    "UsageReport",
    "CanaryConfig",
    "RLExperience",
    "CongestionMetrics",
    "CongestionThrottled",
    "generate_id",
    # Gateway
    "Gateway",
    "TenantAuthenticator",
    "AuthenticationError",
    "RateLimiter",
    "RateLimitExceeded",
    "TokenEstimator",
    "create_gateway",
    # Scheduler
    "QueueManager",
    "CostComputer",
    "LatencyPredictor",
    "RoutingEngine",
    "NoWorkersAvailable",
    "LocalityAwareStrategy",
    # Registry
    "WorkerRegistry",
    "CapacityTracker",
    "HealthChecker",
    "GPUMonitor",
    # Enterprise
    "QuotaManager",
    "QuotaExceeded",
    "PreemptionManager",
    "TrafficSplitter",
    # Optimization
    "ExperienceBuffer",
    "DQNPolicy",
    "RLRouter",
    "MetricsCollector",
    "CongestionModel",
    "CongestionPredictor",
    # Router
    "ModelRouter",
    "create_router",
    "EXAMPLE_REQUEST",
    # API
    "create_app",
]
