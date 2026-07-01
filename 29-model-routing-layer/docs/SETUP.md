# Project 29: Model Routing Layer - Setup Guide

## Overview
Intelligent model routing and capacity scheduling layer for multi-model LLM inference with load balancing, quotas, and preemption.

## Prerequisites
- Python 3.9+
- Redis (for queues and rate limiting)
- PostgreSQL (for quotas and state)
- 8GB+ RAM
- GPU workers (optional)

## Installation

### 1. System Dependencies

**Redis**
```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis
```

**PostgreSQL**
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# macOS
brew install postgresql
```

### 2. Python Environment
```bash
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Database Setup
```bash
# Create database
createdb model_router

# Initialize schema
python -m modelrouter.db init

# Or use environment variable
export DATABASE_URL="postgresql://user:pass@localhost/model_router"
```

### 4. GPU Monitoring Setup (Optional)
```bash
# For NVIDIA GPUs
pip install pynvml

# Verify GPU access
python -c "import pynvml; pynvml.nvmlInit(); print('GPU OK')"
```

## Configuration

### 1. Environment Variables
```bash
# .env file
# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Database
DATABASE_URL=postgresql://user:pass@localhost/model_router

# Routing
ROUTING_STRATEGY=least_loaded
MAX_QUEUE_DEPTH=1000

# Rate Limiting
RATE_LIMIT_REQUESTS_PER_MINUTE=100
RATE_LIMIT_TOKENS_PER_DAY=1000000

# Monitoring
PROMETHEUS_PORT=9090
ENABLE_METRICS=true
ENABLE_HEALTH_CHECKS=true
HEALTH_CHECK_INTERVAL=30

# LLM Providers (for fallback)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
COHERE_API_KEY=your_key
```

### 2. Routing Strategies
Available strategies:
- `least_loaded`: Route to worker with lowest current load
- `round_robin`: Round-robin distribution
- `latency_aware`: Route based on predicted latency
- `cost_aware`: Minimize cost per request
- `random`: Random selection

## Usage

### Basic Routing

```python
from modelrouter import create_router

# Create router
router = create_router(
    routing_strategy="least_loaded",
    max_queue_depth=1000
)

# Register workers
await router.register_worker(
    worker_id="worker-1",
    host="10.0.0.1",
    port=8000,
    models=["gpt-4", "gpt-3.5-turbo"],
    token_budget=100000
)

await router.register_worker(
    worker_id="worker-2",
    host="10.0.0.2",
    port=8000,
    models=["claude-3-opus", "claude-3-sonnet"],
    token_budget=150000
)

# Submit request
from modelrouter import InferenceRequest, Priority

request = InferenceRequest(
    request_id="req-123",
    tenant_id="tenant-a",
    model="gpt-4",
    prompt="What is machine learning?",
    max_tokens=500,
    priority=Priority.HIGH
)

response = await router.submit(request)
print(f"Response: {response.text}")
print(f"Worker: {response.worker_id}")
print(f"Latency: {response.latency_ms}ms")
```

### Tenant Management

```python
# Register tenant with quota
router.register_tenant(
    tenant_id="tenant-a",
    name="Acme Corp",
    api_key="acme-key-123",
    monthly_budget=10000000  # tokens
)

# Set rate limits
from modelrouter import RateLimiter

rate_limiter = router.rate_limiter
await rate_limiter.set_limit(
    tenant_id="tenant-a",
    requests_per_minute=100,
    tokens_per_day=1000000
)
```

### Priority & Preemption

```python
# High priority request can preempt lower priority
high_priority_req = InferenceRequest(
    tenant_id="tenant-a",
    model="gpt-4",
    prompt="Urgent query",
    priority=Priority.CRITICAL  # Will preempt NORMAL/LOW
)

# Submit and potentially preempt
response = await router.submit(high_priority_req)
```

### Quota Management

```python
from modelrouter import QuotaManager

quota_mgr = router.quota_manager

# Set monthly quota
await quota_mgr.set_quota(
    tenant_id="tenant-a",
    model="gpt-4",
    monthly_tokens=5000000
)

# Check usage
usage = await quota_mgr.get_usage(
    tenant_id="tenant-a",
    model="gpt-4"
)

print(f"Used: {usage.tokens_used} / {usage.quota}")
print(f"Remaining: {usage.remaining}")
```

### Traffic Splitting (A/B Testing)

```python
from modelrouter import TrafficSplitter

splitter = router.traffic_splitter

# Split 80% to gpt-4, 20% to gpt-4-turbo
await splitter.configure_split(
    tenant_id="tenant-a",
    model="gpt-4",
    variants=[
        {"model": "gpt-4", "weight": 0.8},
        {"model": "gpt-4-turbo", "weight": 0.2}
    ]
)
```

### Capacity Monitoring

```python
# Get capacity for specific model
capacity = await router.get_capacity(model="gpt-4")
print(f"Total workers: {capacity['total_workers']}")
print(f"Available capacity: {capacity['available_tokens']}")
print(f"Queue depth: {capacity['queue_depth']}")

# Get all capacities
all_capacity = await router.get_capacity()
```

### Queue Management

```python
# Get queue statistics
stats = await router.get_queue_stats()

for model, model_stats in stats.items():
    print(f"{model}:")
    print(f"  Queue depth: {model_stats['depth']}")
    print(f"  Avg wait time: {model_stats['avg_wait_ms']}ms")
    print(f"  Throughput: {model_stats['requests_per_sec']}/s")
```

## API Gateway

### Using HTTP Gateway
```python
# Handle raw HTTP request
raw_request = {
    "api_key": "tenant-key",
    "model": "gpt-4",
    "prompt": "What is ML?",
    "max_tokens": 100
}

response = await router.handle_request(raw_request)
```

### FastAPI Integration
```python
from fastapi import FastAPI
from modelrouter.gateway import create_gateway_app

app = create_gateway_app(router)

# Routes:
# POST /v1/completions
# POST /v1/chat/completions
# GET /v1/models
# GET /v1/capacity
```

## Monitoring & Metrics

### Prometheus Metrics
```bash
# Start Prometheus exporter
python -m modelrouter.metrics

# Metrics available at http://localhost:9090/metrics

# Key metrics:
# - modelrouter_requests_total
# - modelrouter_latency_seconds
# - modelrouter_queue_depth
# - modelrouter_worker_utilization
# - modelrouter_quota_usage
```

### Health Checks
```python
# Health checker runs automatically
from modelrouter import HealthChecker

checker = router.health_checker

# Manual health check
health = await checker.check_worker("worker-1")
print(f"Worker healthy: {health.is_healthy}")
print(f"Last seen: {health.last_heartbeat}")
```

### Request Tracing
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("inference_request"):
    response = await router.submit(request)
```

## Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/locustfile.py --host http://localhost:8000
```

Example `locustfile.py`:
```python
from locust import HttpUser, task, between

class RouterUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def submit_request(self):
        self.client.post("/v1/completions", json={
            "api_key": "test-key",
            "model": "gpt-4",
            "prompt": "Test prompt",
            "max_tokens": 100
        })
```

## Advanced Features

### Custom Routing Strategy
```python
from modelrouter.scheduler import BaseRoutingStrategy

class CustomStrategy(BaseRoutingStrategy):
    async def select_worker(
        self,
        request: InferenceRequest,
        available_workers: list[WorkerInfo]
    ) -> WorkerInfo:
        # Your custom logic
        return min(available_workers, key=lambda w: w.current_load)

# Register strategy
router.routing_engine.add_strategy("custom", CustomStrategy())
router.set_routing_strategy("custom")
```

### Cost Optimization
```python
from modelrouter.scheduler import CostComputer

# Configure cost per model
cost_computer = router.cost_computer
cost_computer.set_cost("gpt-4", cost_per_1k_tokens=0.03)
cost_computer.set_cost("gpt-3.5-turbo", cost_per_1k_tokens=0.002)

# Route to minimize cost
router.set_routing_strategy("cost_aware")
```

## Testing

```bash
# Run all tests
pytest

# Test routing logic
pytest tests/test_router.py

# Test with mock workers
pytest tests/test_integration.py

# Load testing
pytest tests/test_capacity.py --benchmark-only
```

## Common Issues

### Issue: Redis connection timeout
**Solution**: Increase connection pool size
```python
router = create_router(
    redis_pool_size=20  # Increase from default 10
)
```

### Issue: Queue overflow
**Solution**: Add more workers or increase queue depth
```python
router = create_router(max_queue_depth=5000)
```

### Issue: Uneven load distribution
**Solution**: Tune routing strategy or enable load-based routing
```python
router.set_routing_strategy("least_loaded")
```

## Project Structure
```
29-model-routing-layer/
├── src/modelrouter/
│   ├── router.py          # Main router
│   ├── gateway/          # API gateway
│   ├── scheduler/        # Routing & scheduling
│   ├── registry/         # Worker registry
│   ├── enterprise/       # Quotas & preemption
│   └── ...
├── tests/
├── requirements.txt
└── SETUP.md
```

## Deployment

### Docker Deployment
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "modelrouter.server"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-router
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-router
  template:
    metadata:
      labels:
        app: model-router
    spec:
      containers:
      - name: router
        image: model-router:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
```

## Next Steps
1. Register workers
2. Configure routing strategy
3. Set up tenant quotas
4. Enable monitoring
5. Load test and tune

## Resources
- [Redis Documentation](https://redis.io/docs/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Prometheus](https://prometheus.io/docs/)
- [Locust Load Testing](https://docs.locust.io/)
