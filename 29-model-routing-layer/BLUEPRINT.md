# Project 29: Model Routing Layer (Capacity-Aware Scheduling)

## Executive Summary

A sophisticated model routing and load balancing layer for LLM inference infrastructure. Implements capacity-aware scheduling, token-budget management, priority queuing, and intelligent traffic shaping. Supports multiple routing strategies (least-loaded, round-robin, token-based, SLA-based) with real-time GPU load tracking and tenant-based quotas.

> **Concepts covered:** [§04 LLM serving at scale](../../04-ai-engineering/04-llm-inference/serving-at-scale/llm-serving-at-scale.md) (the routing/gateway concerns) · [§04 LLM agents](../../04-ai-engineering/02-llm-applications/agents/llm-agents.md) (clients of the router) · [§05 Cost optimization](../../05-cross-cutting-concerns/cost-optimization/) (token budgets, tier routing). Pairs with [Project 44 (autoregressive inference engine — what's behind the route)](../44-autoregressive-inference/), [Project 23 (agentic runtime — caller)](../23-llm-agentic-runtime/), [Project 28 (workflow engine — caller)](../28-ai-workflow-engine/), [Project 46 (multi-tenant GPU scheduler)](../46-multi-tenant-gpu-scheduler/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Gateway Design](#gateway-design)
3. [Scheduler Implementation](#scheduler-implementation)
4. [Worker Registry](#worker-registry)
5. [Routing Strategies](#routing-strategies)
6. [Enterprise Features](#enterprise-features)
7. [Implementation Phases](#implementation-phases)
8. [Stretch Goals](#stretch-goals)

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Model Routing Layer                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                            Gateway Layer                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Request   │  │   Tenant    │  │  Priority   │  │   Token     │  │  │
│  │  │   Intake    │  │   Auth      │  │   Tagging   │  │   Counter   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                          Scheduler Layer                               │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Queue     │  │    Cost     │  │   Routing   │  │   SLA       │  │  │
│  │  │   Manager   │  │  Computer   │  │   Engine    │  │   Monitor   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        Worker Registry                                 │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │   Worker    │  │    GPU      │  │   Health    │  │   Capacity  │  │  │
│  │  │   Catalog   │  │   Monitor   │  │   Checker   │  │   Tracker   │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                       │                                      │
│                                       ▼                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                         Model Workers                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │  │
│  │  │  Worker 1   │  │  Worker 2   │  │  Worker 3   │  │  Worker N   │  │  │
│  │  │  GPT-4      │  │  Claude     │  │  Llama      │  │  Mixed      │  │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow

```
Client Request
      │
      ▼
┌─────────────────┐
│  Rate Limiter   │ ─── Check tenant quota
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Request Tagger  │ ─── Add tenant, priority, SLA
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Token Counter  │ ─── Estimate token cost
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Queue Manager  │ ─── Priority queue placement
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Routing Engine  │ ─── Select optimal worker
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Worker Pool    │ ─── Execute inference
└────────┬────────┘
         │
         ▼
    Response
```

---

## Gateway Design

### Request Intake

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum
import time

class Priority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BATCH = 4

@dataclass
class InferenceRequest:
    request_id: str
    tenant_id: str
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    priority: Priority
    sla_deadline_ms: Optional[int]
    metadata: Dict[str, Any]
    created_at: float = None
    estimated_tokens: int = 0

    def __post_init__(self):
        self.created_at = self.created_at or time.time()

class Gateway:
    """Main gateway for model routing layer."""

    def __init__(
        self,
        rate_limiter: RateLimiter,
        authenticator: TenantAuthenticator,
        token_estimator: TokenEstimator,
        scheduler: Scheduler
    ):
        self.rate_limiter = rate_limiter
        self.auth = authenticator
        self.token_estimator = token_estimator
        self.scheduler = scheduler

    async def handle_request(
        self,
        raw_request: Dict[str, Any]
    ) -> InferenceResponse:
        # Authenticate and get tenant config
        tenant = await self.auth.authenticate(raw_request)

        # Create typed request
        request = self._create_request(raw_request, tenant)

        # Check rate limits
        await self.rate_limiter.check(tenant.id, request)

        # Estimate token cost
        request.estimated_tokens = self.token_estimator.estimate(
            request.prompt,
            request.max_tokens
        )

        # Submit to scheduler
        result = await self.scheduler.submit(request)

        # Update metrics
        await self._update_metrics(request, result)

        return result

    def _create_request(
        self,
        raw: Dict[str, Any],
        tenant: Tenant
    ) -> InferenceRequest:
        # Determine priority
        priority = self._compute_priority(raw, tenant)

        # Compute SLA deadline
        sla_deadline = self._compute_sla(priority, tenant)

        return InferenceRequest(
            request_id=generate_id(),
            tenant_id=tenant.id,
            model=raw["model"],
            prompt=raw["prompt"],
            max_tokens=raw.get("max_tokens", 500),
            temperature=raw.get("temperature", 0.7),
            priority=priority,
            sla_deadline_ms=sla_deadline,
            metadata=raw.get("metadata", {})
        )

    def _compute_priority(
        self,
        raw: Dict[str, Any],
        tenant: Tenant
    ) -> Priority:
        # Check explicit priority
        if "priority" in raw:
            return Priority[raw["priority"].upper()]

        # Use tenant default
        return tenant.default_priority

    def _compute_sla(
        self,
        priority: Priority,
        tenant: Tenant
    ) -> int:
        """Compute SLA deadline in milliseconds."""
        sla_map = {
            Priority.CRITICAL: 1000,   # 1s
            Priority.HIGH: 3000,       # 3s
            Priority.NORMAL: 10000,    # 10s
            Priority.LOW: 30000,       # 30s
            Priority.BATCH: 300000     # 5min
        }
        return tenant.sla_overrides.get(priority, sla_map[priority])
```

### Rate Limiting

```python
class RateLimiter:
    """Token bucket rate limiter with per-tenant quotas."""

    def __init__(self, redis_client):
        self.redis = redis_client

    async def check(
        self,
        tenant_id: str,
        request: InferenceRequest
    ) -> bool:
        # Check requests per second
        rps_key = f"ratelimit:{tenant_id}:rps"
        rps_allowed = await self._check_bucket(
            rps_key,
            limit=self._get_rps_limit(tenant_id),
            window=1
        )

        if not rps_allowed:
            raise RateLimitExceeded("Requests per second limit exceeded")

        # Check tokens per minute
        tpm_key = f"ratelimit:{tenant_id}:tpm"
        tpm_allowed = await self._check_bucket(
            tpm_key,
            limit=self._get_tpm_limit(tenant_id),
            window=60,
            cost=request.estimated_tokens
        )

        if not tpm_allowed:
            raise RateLimitExceeded("Tokens per minute limit exceeded")

        return True

    async def _check_bucket(
        self,
        key: str,
        limit: int,
        window: int,
        cost: int = 1
    ) -> bool:
        """Check and consume from token bucket."""
        now = time.time()
        window_start = now - window

        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)

        # Check current count
        current = await self.redis.zcard(key)

        if current + cost > limit:
            return False

        # Add new entries
        pipe = self.redis.pipeline()
        for _ in range(cost):
            pipe.zadd(key, {f"{now}:{generate_id()}": now})
        pipe.expire(key, window * 2)
        await pipe.execute()

        return True
```

### Token Estimation

```python
class TokenEstimator:
    """Estimates token count for requests."""

    def __init__(self, tokenizers: Dict[str, Any]):
        self.tokenizers = tokenizers

    def estimate(
        self,
        prompt: str,
        max_tokens: int,
        model: str = "default"
    ) -> int:
        # Get tokenizer for model
        tokenizer = self.tokenizers.get(model, self.tokenizers["default"])

        # Count input tokens
        input_tokens = len(tokenizer.encode(prompt))

        # Total = input + output
        return input_tokens + max_tokens

    def estimate_cost(
        self,
        request: InferenceRequest,
        pricing: ModelPricing
    ) -> float:
        """Estimate cost in dollars."""
        input_tokens = len(self.tokenizers["default"].encode(request.prompt))
        output_tokens = request.max_tokens

        return (
            input_tokens * pricing.input_cost_per_1k / 1000 +
            output_tokens * pricing.output_cost_per_1k / 1000
        )
```

---

## Scheduler Implementation

### Queue Manager

```python
from dataclasses import dataclass
from typing import List, Dict
import heapq
import asyncio

@dataclass
class QueuedRequest:
    request: InferenceRequest
    enqueue_time: float
    priority_score: float

    def __lt__(self, other):
        return self.priority_score < other.priority_score

class QueueManager:
    """Manages priority queues per model."""

    def __init__(self):
        self.queues: Dict[str, List[QueuedRequest]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}

    async def enqueue(self, request: InferenceRequest) -> str:
        """Add request to appropriate queue."""
        model = request.model

        if model not in self.queues:
            self.queues[model] = []
            self.locks[model] = asyncio.Lock()

        # Compute priority score (lower = higher priority)
        score = self._compute_priority_score(request)

        queued = QueuedRequest(
            request=request,
            enqueue_time=time.time(),
            priority_score=score
        )

        async with self.locks[model]:
            heapq.heappush(self.queues[model], queued)

        return request.request_id

    async def dequeue(self, model: str) -> Optional[InferenceRequest]:
        """Get highest priority request for model."""
        if model not in self.queues:
            return None

        async with self.locks[model]:
            if not self.queues[model]:
                return None

            queued = heapq.heappop(self.queues[model])
            return queued.request

    async def peek(self, model: str) -> Optional[InferenceRequest]:
        """Peek at highest priority request without removing."""
        if model not in self.queues or not self.queues[model]:
            return None
        return self.queues[model][0].request

    def _compute_priority_score(self, request: InferenceRequest) -> float:
        """Compute priority score for queue ordering."""
        # Base priority (0-4)
        base = request.priority.value

        # Time factor (older requests get priority boost)
        age = time.time() - request.created_at
        age_bonus = -min(age / 60, 1.0)  # Up to -1 for 60s wait

        # SLA urgency
        if request.sla_deadline_ms:
            remaining = request.sla_deadline_ms - (time.time() - request.created_at) * 1000
            urgency = -max(0, 1 - remaining / request.sla_deadline_ms)
        else:
            urgency = 0

        return base + age_bonus + urgency

    async def get_queue_stats(self, model: str) -> Dict[str, Any]:
        """Get queue statistics."""
        if model not in self.queues:
            return {"depth": 0, "oldest_ms": 0}

        queue = self.queues[model]
        if not queue:
            return {"depth": 0, "oldest_ms": 0}

        oldest = min(q.enqueue_time for q in queue)

        return {
            "depth": len(queue),
            "oldest_ms": int((time.time() - oldest) * 1000),
            "by_priority": self._count_by_priority(queue)
        }
```

### Cost Computation

```python
@dataclass
class RequestCost:
    tokens: int
    compute_units: float
    estimated_latency_ms: int
    dollar_cost: float

class CostComputer:
    """Computes cost metrics for scheduling decisions."""

    def __init__(
        self,
        pricing_config: Dict[str, ModelPricing],
        latency_predictor: LatencyPredictor
    ):
        self.pricing = pricing_config
        self.latency_predictor = latency_predictor

    def compute(
        self,
        request: InferenceRequest,
        worker: WorkerInfo
    ) -> RequestCost:
        pricing = self.pricing[request.model]

        # Token cost
        tokens = request.estimated_tokens

        # Compute units (GPU-seconds)
        compute_units = self._estimate_compute_units(request, worker)

        # Latency prediction
        estimated_latency = self.latency_predictor.predict(
            request,
            worker
        )

        # Dollar cost
        dollar_cost = (
            tokens * pricing.cost_per_1k_tokens / 1000 +
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
        """Estimate GPU-seconds needed."""
        # Base compute per token
        tokens = request.estimated_tokens
        base_compute = tokens * 0.001  # 1ms per token baseline

        # Adjust for worker performance
        worker_factor = worker.performance_factor

        return base_compute / worker_factor

class LatencyPredictor:
    """Predicts request latency based on historical data."""

    def __init__(self, history_store):
        self.history = history_store

    def predict(
        self,
        request: InferenceRequest,
        worker: WorkerInfo
    ) -> int:
        """Predict latency in milliseconds."""
        # Get historical latencies for similar requests
        historical = self.history.get_similar(
            model=request.model,
            token_range=(request.estimated_tokens * 0.8, request.estimated_tokens * 1.2),
            worker_type=worker.worker_type
        )

        if historical:
            base_latency = np.percentile(historical, 50)
        else:
            # Fallback to formula
            base_latency = request.estimated_tokens * 10  # 10ms per token

        # Adjust for current load
        load_factor = 1 + worker.current_load * 0.5

        # Adjust for queue depth
        queue_factor = 1 + worker.queue_depth * 0.1

        return int(base_latency * load_factor * queue_factor)
```

### Routing Engine

```python
class RoutingEngine:
    """Routes requests to optimal workers."""

    def __init__(
        self,
        worker_registry: WorkerRegistry,
        cost_computer: CostComputer,
        routing_strategy: str = "least_loaded"
    ):
        self.registry = worker_registry
        self.cost_computer = cost_computer
        self.strategy = routing_strategy

    async def route(
        self,
        request: InferenceRequest
    ) -> WorkerInfo:
        """Route request to best worker."""
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
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _least_loaded(
        self,
        request: InferenceRequest,
        workers: List[WorkerInfo]
    ) -> WorkerInfo:
        """Select worker with lowest current load."""
        return min(workers, key=lambda w: w.current_load)

    def _round_robin(
        self,
        request: InferenceRequest,
        workers: List[WorkerInfo]
    ) -> WorkerInfo:
        """Round-robin selection."""
        # Get next worker index
        idx = self._get_next_index(request.model, len(workers))
        return workers[idx]

    def _token_based(
        self,
        request: InferenceRequest,
        workers: List[WorkerInfo]
    ) -> WorkerInfo:
        """Select worker with most available token budget."""
        def token_capacity(w):
            return w.token_budget - w.tokens_in_flight

        return max(workers, key=token_capacity)

    def _sla_based(
        self,
        request: InferenceRequest,
        workers: List[WorkerInfo]
    ) -> WorkerInfo:
        """Select worker most likely to meet SLA."""
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
            (w, l) for w, l in predictions
            if l < request.sla_deadline_ms * 0.8  # 20% buffer
        ]

        if viable:
            # Among viable, pick least loaded
            return min(viable, key=lambda x: x[0].current_load)[0]
        else:
            # Fall back to fastest prediction
            return min(predictions, key=lambda x: x[1])[0]
```

---

## Worker Registry

### Worker Management

```python
@dataclass
class WorkerInfo:
    worker_id: str
    host: str
    port: int
    models: List[str]
    gpu_info: GPUInfo
    current_load: float
    queue_depth: int
    tokens_in_flight: int
    token_budget: int
    status: str
    last_heartbeat: float
    performance_factor: float
    worker_type: str

@dataclass
class GPUInfo:
    device_name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_percent: float
    temperature_celsius: int

class WorkerRegistry:
    """Manages worker registration and health."""

    def __init__(self, storage, health_checker):
        self.storage = storage
        self.health_checker = health_checker

    async def register(self, worker: WorkerInfo) -> str:
        """Register a new worker."""
        await self.storage.set(
            f"worker:{worker.worker_id}",
            asdict(worker),
            ttl=60  # Requires heartbeat
        )

        # Add to model index
        for model in worker.models:
            await self.storage.sadd(f"model:{model}:workers", worker.worker_id)

        return worker.worker_id

    async def deregister(self, worker_id: str):
        """Remove worker from registry."""
        worker = await self.get_worker(worker_id)
        if worker:
            for model in worker.models:
                await self.storage.srem(f"model:{model}:workers", worker_id)
            await self.storage.delete(f"worker:{worker_id}")

    async def heartbeat(self, worker_id: str, status: WorkerStatus):
        """Update worker status."""
        await self.storage.set(
            f"worker:{worker_id}",
            {
                **status,
                "last_heartbeat": time.time()
            },
            ttl=60
        )

    async def get_workers(
        self,
        model: str = None,
        status: str = "healthy"
    ) -> List[WorkerInfo]:
        """Get workers, optionally filtered."""
        if model:
            worker_ids = await self.storage.smembers(f"model:{model}:workers")
        else:
            worker_ids = await self.storage.keys("worker:*")

        workers = []
        for worker_id in worker_ids:
            data = await self.storage.get(f"worker:{worker_id}")
            if data and (status is None or data["status"] == status):
                workers.append(WorkerInfo(**data))

        return workers

    async def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get specific worker."""
        data = await self.storage.get(f"worker:{worker_id}")
        return WorkerInfo(**data) if data else None
```

### GPU Monitoring

```python
class GPUMonitor:
    """Monitors GPU utilization and health."""

    def __init__(self, poll_interval: int = 5):
        self.poll_interval = poll_interval

    async def collect_metrics(self, worker_id: str) -> GPUInfo:
        """Collect GPU metrics using nvidia-smi."""
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        name = pynvml.nvmlDeviceGetName(handle)

        return GPUInfo(
            device_name=name,
            memory_total_mb=memory.total // (1024 * 1024),
            memory_used_mb=memory.used // (1024 * 1024),
            utilization_percent=utilization.gpu,
            temperature_celsius=temperature
        )

    async def start_monitoring(self, worker_id: str, callback):
        """Start continuous monitoring."""
        while True:
            try:
                metrics = await self.collect_metrics(worker_id)
                await callback(worker_id, metrics)
            except Exception as e:
                logger.error(f"GPU monitoring failed: {e}")

            await asyncio.sleep(self.poll_interval)

class CapacityTracker:
    """Tracks capacity across all workers."""

    def __init__(self, registry: WorkerRegistry):
        self.registry = registry

    async def get_capacity(self, model: str) -> CapacityInfo:
        """Get current capacity for a model."""
        workers = await self.registry.get_workers(model=model)

        total_token_budget = sum(w.token_budget for w in workers)
        tokens_in_flight = sum(w.tokens_in_flight for w in workers)
        available_tokens = total_token_budget - tokens_in_flight

        total_gpu_memory = sum(w.gpu_info.memory_total_mb for w in workers)
        used_gpu_memory = sum(w.gpu_info.memory_used_mb for w in workers)

        return CapacityInfo(
            total_workers=len(workers),
            healthy_workers=len([w for w in workers if w.status == "healthy"]),
            total_token_budget=total_token_budget,
            available_tokens=available_tokens,
            utilization=tokens_in_flight / total_token_budget if total_token_budget > 0 else 0,
            gpu_memory_total_mb=total_gpu_memory,
            gpu_memory_used_mb=used_gpu_memory
        )
```

### Health Checking

```python
class HealthChecker:
    """Performs health checks on workers."""

    def __init__(
        self,
        registry: WorkerRegistry,
        check_interval: int = 10,
        unhealthy_threshold: int = 3
    ):
        self.registry = registry
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.failure_counts: Dict[str, int] = {}

    async def start(self):
        """Start health checking loop."""
        while True:
            workers = await self.registry.get_workers(status=None)

            for worker in workers:
                is_healthy = await self._check_worker(worker)
                await self._update_status(worker.worker_id, is_healthy)

            await asyncio.sleep(self.check_interval)

    async def _check_worker(self, worker: WorkerInfo) -> bool:
        """Check if worker is healthy."""
        try:
            # Check heartbeat freshness
            if time.time() - worker.last_heartbeat > 30:
                return False

            # Ping worker
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{worker.host}:{worker.port}/health",
                    timeout=5
                ) as response:
                    return response.status == 200

        except Exception:
            return False

    async def _update_status(self, worker_id: str, is_healthy: bool):
        """Update worker health status."""
        if is_healthy:
            self.failure_counts[worker_id] = 0
            await self.registry.update_status(worker_id, "healthy")
        else:
            self.failure_counts[worker_id] = self.failure_counts.get(worker_id, 0) + 1

            if self.failure_counts[worker_id] >= self.unhealthy_threshold:
                await self.registry.update_status(worker_id, "unhealthy")
```

---

## Routing Strategies

### Strategy Comparison

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| Least-Loaded | General use | Simple, fair | May not optimize cost |
| Round-Robin | Homogeneous | Even distribution | Ignores load |
| Token-Based | Token budgets | Respects limits | Complex tracking |
| SLA-Based | Latency-critical | Meets deadlines | May waste capacity |

### Advanced Strategies

```python
class WeightedRoutingStrategy:
    """Weighted routing based on multiple factors."""

    def __init__(
        self,
        load_weight: float = 0.4,
        latency_weight: float = 0.3,
        cost_weight: float = 0.2,
        sla_weight: float = 0.1
    ):
        self.weights = {
            "load": load_weight,
            "latency": latency_weight,
            "cost": cost_weight,
            "sla": sla_weight
        }

    def select(
        self,
        request: InferenceRequest,
        workers: List[WorkerInfo],
        cost_computer: CostComputer
    ) -> WorkerInfo:
        scores = []

        for worker in workers:
            cost = cost_computer.compute(request, worker)

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
                self.weights["load"] * load_score +
                self.weights["latency"] * latency_score +
                self.weights["cost"] * cost_score +
                self.weights["sla"] * sla_score
            )

            scores.append((worker, total_score))

        # Return worker with lowest score
        return min(scores, key=lambda x: x[1])[0]

class LocalityAwareStrategy:
    """Route based on data locality."""

    def __init__(self, locality_cache):
        self.cache = locality_cache

    def select(
        self,
        request: InferenceRequest,
        workers: List[WorkerInfo]
    ) -> WorkerInfo:
        # Check if any worker has cached context
        for worker in workers:
            if self.cache.has_context(worker.worker_id, request.context_key):
                return worker

        # Fall back to least loaded
        return min(workers, key=lambda w: w.current_load)
```

---

## Enterprise Features

### Tenant Quotas

```python
@dataclass
class TenantQuota:
    tenant_id: str
    requests_per_second: int
    tokens_per_minute: int
    monthly_token_budget: int
    priority_access: bool
    dedicated_workers: List[str]

class QuotaManager:
    """Manages per-tenant quotas."""

    def __init__(self, storage):
        self.storage = storage

    async def check_quota(
        self,
        tenant_id: str,
        request: InferenceRequest
    ) -> bool:
        quota = await self._get_quota(tenant_id)
        usage = await self._get_usage(tenant_id)

        # Check monthly budget
        if usage.monthly_tokens + request.estimated_tokens > quota.monthly_token_budget:
            raise QuotaExceeded("Monthly token budget exceeded")

        return True

    async def record_usage(
        self,
        tenant_id: str,
        tokens_used: int
    ):
        """Record token usage for tenant."""
        await self.storage.incr(f"usage:{tenant_id}:tokens", tokens_used)

    async def get_usage_report(
        self,
        tenant_id: str
    ) -> UsageReport:
        """Get usage report for tenant."""
        usage = await self._get_usage(tenant_id)
        quota = await self._get_quota(tenant_id)

        return UsageReport(
            tenant_id=tenant_id,
            tokens_used=usage.monthly_tokens,
            tokens_remaining=quota.monthly_token_budget - usage.monthly_tokens,
            utilization=usage.monthly_tokens / quota.monthly_token_budget
        )
```

### Preemptive Cancellation

```python
class PreemptionManager:
    """Manages request preemption for priority handling."""

    def __init__(self, queue_manager: QueueManager):
        self.queue_manager = queue_manager

    async def maybe_preempt(
        self,
        high_priority_request: InferenceRequest,
        model: str
    ) -> bool:
        """Check if we should preempt lower priority work."""
        if high_priority_request.priority.value > Priority.HIGH.value:
            return False  # Only preempt for CRITICAL/HIGH

        # Get current queue
        queue_stats = await self.queue_manager.get_queue_stats(model)

        # Find preemption candidates
        candidates = await self._find_preemption_candidates(
            model,
            high_priority_request.priority
        )

        if not candidates:
            return False

        # Preempt lowest priority request
        victim = max(candidates, key=lambda r: r.priority.value)
        await self._preempt(victim)

        return True

    async def _preempt(self, request: InferenceRequest):
        """Cancel and requeue preempted request."""
        # Cancel execution
        await self._cancel_execution(request.request_id)

        # Requeue with penalty
        request.metadata["preempted"] = True
        await self.queue_manager.enqueue(request)
```

### Traffic Splitting for Canary

```python
class TrafficSplitter:
    """Splits traffic for canary deployments."""

    def __init__(self, config_store):
        self.config_store = config_store

    async def get_target(
        self,
        request: InferenceRequest
    ) -> str:
        """Determine which deployment to route to."""
        # Get canary config
        canary = await self.config_store.get_canary(request.model)

        if not canary or not canary.active:
            return "stable"

        # Consistent hashing for user
        hash_val = hash(f"{request.tenant_id}:{canary.id}") % 100

        if hash_val < canary.traffic_percentage:
            return "canary"
        else:
            return "stable"

    async def create_canary(
        self,
        model: str,
        canary_workers: List[str],
        traffic_percentage: float
    ) -> str:
        canary_id = generate_id()

        await self.config_store.set_canary(model, {
            "id": canary_id,
            "workers": canary_workers,
            "traffic_percentage": traffic_percentage,
            "active": True,
            "created_at": time.time()
        })

        return canary_id
```

---

## Implementation Phases

### Phase 1: Core Routing (Weeks 1-4)

**Deliverables:**
- [ ] Gateway request handling
- [ ] Basic queue manager
- [ ] Least-loaded routing
- [ ] Worker registry
- [ ] Health checking

### Phase 2: Advanced Scheduling (Weeks 5-8)

**Deliverables:**
- [ ] Token estimation
- [ ] Cost computation
- [ ] SLA-based routing
- [ ] Latency prediction
- [ ] Priority queuing

### Phase 3: GPU Monitoring (Weeks 9-11)

**Deliverables:**
- [ ] GPU metrics collection
- [ ] Capacity tracking
- [ ] Load-aware routing
- [ ] Real-time dashboards

### Phase 4: Enterprise Features (Weeks 12-15)

**Deliverables:**
- [ ] Tenant quotas
- [ ] Preemptive cancellation
- [ ] Traffic splitting
- [ ] Audit logging

### Phase 5: Optimization (Weeks 16-18)

**Deliverables:**
- [ ] Performance tuning
- [ ] Advanced strategies
- [ ] Load testing
- [ ] Documentation

---

## Stretch Goals

### Adaptive Routing with RL

```python
class RLRouter:
    """RL-based adaptive routing."""

    def __init__(self, policy_network, experience_buffer):
        self.policy = policy_network
        self.buffer = experience_buffer

    async def route(
        self,
        request: InferenceRequest,
        workers: List[WorkerInfo]
    ) -> WorkerInfo:
        # Extract state features
        state = self._extract_state(request, workers)

        # Get action from policy
        action = self.policy.select_action(state)

        # Map action to worker
        worker = workers[action % len(workers)]

        # Store for learning
        self.buffer.add(state, action, request.request_id)

        return worker

    async def update(self, request_id: str, reward: float):
        """Update policy with observed reward."""
        experience = self.buffer.get(request_id)
        if experience:
            self.policy.update(experience, reward)
```

### Predictive Congestion Avoidance

```python
class CongestionPredictor:
    """Predicts and avoids congestion."""

    def __init__(self, model):
        self.model = model

    async def predict_congestion(
        self,
        model: str,
        horizon_minutes: int = 5
    ) -> float:
        """Predict congestion probability."""
        # Get recent metrics
        metrics = await self._get_metrics(model, window_minutes=30)

        # Features: queue depth trend, arrival rate, service rate
        features = self._extract_features(metrics)

        # Predict
        return self.model.predict(features)

    async def should_throttle(self, model: str) -> bool:
        """Check if we should proactively throttle."""
        congestion_prob = await self.predict_congestion(model)
        return congestion_prob > 0.8
```

---

## File Structure

```
29-model-routing-layer/
├── src/
│   ├── __init__.py
│   ├── gateway/
│   │   ├── __init__.py
│   │   ├── gateway.py
│   │   ├── rate_limiter.py
│   │   └── token_estimator.py
│   ├── scheduler/
│   │   ├── __init__.py
│   │   ├── queue.py
│   │   ├── cost.py
│   │   └── router.py
│   ├── registry/
│   │   ├── __init__.py
│   │   ├── workers.py
│   │   ├── gpu_monitor.py
│   │   └── health.py
│   ├── enterprise/
│   │   ├── __init__.py
│   │   ├── quotas.py
│   │   ├── preemption.py
│   │   └── traffic_split.py
│   └── api/
│       └── main.py
├── config/
├── tests/
├── docs/
├── BLUEPRINT.md
├── PROGRESS.md
└── SESSION_CONTEXT.md
```

---

## Success Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Routing Latency | < 5ms | P95 |
| SLA Achievement | > 95% | Meet deadlines |
| Load Variance | < 10% | Across workers |
| Queue Wait Time | < 500ms | P95 for NORMAL |
| Availability | 99.99% | System uptime |

---

## References

- [Load Balancing Algorithms](https://nginx.org/en/docs/http/load_balancing.html)
- [Queueing Theory](https://en.wikipedia.org/wiki/Queueing_theory)
- [Token Bucket Algorithm](https://en.wikipedia.org/wiki/Token_bucket)
- [vLLM Scheduling](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)
