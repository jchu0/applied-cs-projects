"""Core schemas for model routing layer."""

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
import time
import uuid


def generate_id() -> str:
    """Generate unique identifier."""
    return str(uuid.uuid4())[:8]


class Priority(Enum):
    """Request priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4


class WorkerStatus(Enum):
    """Worker health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class InferenceRequest:
    """Request for model inference."""
    request_id: str
    tenant_id: str
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    priority: Priority
    sla_deadline_ms: int = None
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    estimated_tokens: int = 0


@dataclass
class InferenceResponse:
    """Response from model inference."""
    request_id: str
    text: str
    tokens_used: int
    latency_ms: int
    worker_id: str
    model: str


@dataclass
class GPUInfo:
    """GPU metrics."""
    device_name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: float
    temperature_celsius: int


@dataclass
class WorkerInfo:
    """Worker node information."""
    worker_id: str
    host: str
    port: int
    models: list[str]
    gpu_info: GPUInfo
    current_load: float
    queue_depth: int
    tokens_in_flight: int
    token_budget: int
    status: str
    last_heartbeat: float
    performance_factor: float = 1.0
    worker_type: str = "standard"


@dataclass
class Tenant:
    """Tenant configuration."""
    id: str
    name: str
    api_key: str
    default_priority: Priority = Priority.NORMAL
    sla_overrides: dict = field(default_factory=dict)


@dataclass
class TenantQuota:
    """Tenant resource quota."""
    tenant_id: str
    requests_per_second: int
    tokens_per_minute: int
    monthly_token_budget: int
    priority_access: bool = False
    dedicated_workers: list[str] = field(default_factory=list)


@dataclass
class ModelPricing:
    """Model pricing configuration."""
    model: str
    input_cost_per_1k: float
    output_cost_per_1k: float
    compute_cost_per_unit: float = 0.0


@dataclass
class RequestCost:
    """Computed cost for request."""
    tokens: int
    compute_units: float
    estimated_latency_ms: int
    dollar_cost: float


@dataclass
class CapacityInfo:
    """Capacity information for model."""
    total_workers: int
    healthy_workers: int
    total_token_budget: int
    available_tokens: int
    utilization: float
    gpu_memory_total_mb: int
    gpu_memory_used_mb: int


@dataclass
class QueuedRequest:
    """Request in priority queue."""
    request: InferenceRequest
    enqueue_time: float
    priority_score: float

    def __lt__(self, other):
        return self.priority_score < other.priority_score


@dataclass
class UsageReport:
    """Tenant usage report."""
    tenant_id: str
    tokens_used: int
    tokens_remaining: int
    utilization: float
    period_start: float = field(default_factory=time.time)


@dataclass
class CanaryConfig:
    """Canary deployment configuration."""
    id: str
    model: str
    workers: list[str]
    traffic_percentage: float
    active: bool
    created_at: float = field(default_factory=time.time)


@dataclass
class RLExperience:
    """State/action/reward tuple for RL replay buffer."""
    request_id: str
    state: Any  # numpy array
    action: int
    reward: float = 0.0
    next_state: Any = None
    done: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class CongestionMetrics:
    """Timestamped congestion observation."""
    model: str
    timestamp: float
    queue_depth: int
    arrival_rate: float
    service_rate: float
    avg_latency_ms: float = 0.0


class CongestionThrottled(Exception):
    """Raised when congestion throttling activates."""
    pass
