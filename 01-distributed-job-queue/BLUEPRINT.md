# Project 1: Distributed Job Queue + Scheduler

> **Concepts covered:** §01 software-engineering — `python/03-concurrency`, `python/06-microservices`

## Staff-Level Design Document

**Complexity:** ⭐⭐⭐⭐ (Advanced)
**Timeline:** 6-8 weeks
**Languages:** Go, Rust, or Python

---

## What This Project Teaches

### Core Concepts
- **Distributed system fundamentals** - Coordination, consensus, failure handling
- **Queue semantics and message brokers** - FIFO guarantees, at-least-once/exactly-once delivery
- **Fault tolerance and retries** - Exponential backoff, dead-letter queues
- **Worker lifecycle & heartbeats** - Liveness detection, graceful shutdown
- **Backpressure and load balancing** - Rate limiting, queue depth monitoring
- **Designing idempotent operations** - Safe retries, deduplication
- **Scheduling + time-based triggers** - Cron expressions, delayed jobs

### Industry Relevance
This is how Celery, Sidekiq, Bull, and AWS SQS work internally. Understanding these patterns is essential for building reliable backend systems at scale.

---

## High-Level Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  API Server │────▶│   Broker    │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
              ┌─────▼─────┐            ┌───────▼───────┐          ┌───────▼───────┐
              │  Worker 1  │            │   Worker 2    │          │   Worker N    │
              └─────┬─────┘            └───────┬───────┘          └───────┬───────┘
                    │                          │                          │
                    └──────────────────────────┼──────────────────────────┘
                                               │
                                        ┌──────▼──────┐
                                        │Result Store │
                                        └─────────────┘
```

### Component Breakdown

#### 1. Broker Layer (Redis/Kafka-like)
**Responsibilities:**
- Maintain FIFO queues per topic
- Persist queued jobs to disk
- Support topics & routing keys
- Handle queue priorities

**Data Structures:**
```python
# Queue structure
queues = {
    "emails": [job1, job2, job3],      # Priority 1
    "reports": [job4, job5],            # Priority 2
    "default": [job6, job7, job8]       # Priority 3
}

# Job visibility timeout tracking
visibility_locks = {
    "job_id_1": {"worker": "w1", "expires": timestamp},
}
```

#### 2. API Server
**Endpoints:**
```
POST /tasks              - Enqueue new task
GET  /tasks/{id}         - Get task status
GET  /tasks/{id}/result  - Get task result
DELETE /tasks/{id}       - Cancel task
POST /tasks/{id}/retry   - Manual retry
GET  /queues             - List all queues
GET  /queues/{name}/stats - Queue statistics
POST /schedules          - Create scheduled task
```

#### 3. Workers
**Lifecycle:**
1. Register with broker
2. Poll for tasks (long-polling or pub/sub)
3. Acquire task (atomic operation)
4. Execute task
5. Send heartbeats during execution
6. Report result/failure
7. Acknowledge completion

#### 4. Scheduler
**Features:**
- Cron-like recurring task triggers
- Delayed job support (execute at specific time)
- Timezone-aware scheduling
- Missed job handling

#### 5. Result Store
**Storage Options:**
- Redis with TTL
- PostgreSQL for durability
- S3 for large results

---

## Core Internals

### Task Representation

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "queue": "emails",
  "name": "send_welcome_email",
  "payload": {
    "user_id": 12345,
    "template": "welcome_v2"
  },
  "priority": 1,
  "retries": 0,
  "max_retries": 3,
  "eta": "2024-01-15T10:30:00Z",
  "timeout_ms": 30000,
  "created_at": "2024-01-15T10:00:00Z",
  "idempotency_key": "welcome_12345",
  "metadata": {
    "trace_id": "abc123",
    "tenant_id": "tenant_1"
  }
}
```

### Worker Lifecycle State Machine

```
┌─────────┐    register    ┌─────────┐
│  INIT   │───────────────▶│  IDLE   │
└─────────┘                └────┬────┘
                                │ poll
                           ┌────▼────┐
                           │ WORKING │◀──┐
                           └────┬────┘   │
                      ┌─────────┼─────────┐
                      │         │         │
                 ┌────▼───┐ ┌───▼────┐ ┌──▼───┐
                 │SUCCESS │ │FAILURE │ │RETRY │
                 └────┬───┘ └───┬────┘ └──┬───┘
                      │         │         │
                      └─────────┴─────────┘
                                │
                           ┌────▼────┐
                           │  IDLE   │
                           └─────────┘
```

### Retry Semantics

```python
def calculate_backoff(attempt: int, base_delay: float = 1.0) -> float:
    """Exponential backoff with jitter"""
    delay = base_delay * (2 ** attempt)
    jitter = random.uniform(0, delay * 0.1)
    return min(delay + jitter, MAX_DELAY)

# Retry flow:
# Attempt 1: immediate
# Attempt 2: ~2 seconds
# Attempt 3: ~4 seconds
# Attempt 4: ~8 seconds
# After max_retries: move to dead-letter queue
```

### Dead-Letter Queue (DLQ) Handling

```python
class DeadLetterQueue:
    def __init__(self):
        self.dlq = []

    def move_to_dlq(self, task, error):
        dlq_entry = {
            "original_task": task,
            "error": str(error),
            "failed_at": datetime.utcnow(),
            "retry_count": task["retries"]
        }
        self.dlq.append(dlq_entry)
        self.notify_operators(dlq_entry)
```

### Heartbeat Protocol

```python
class WorkerHeartbeat:
    def __init__(self, worker_id, interval_ms=5000):
        self.worker_id = worker_id
        self.interval = interval_ms

    async def heartbeat_loop(self):
        while self.running:
            await self.broker.send_heartbeat(
                worker_id=self.worker_id,
                current_task=self.current_task_id,
                memory_usage=get_memory_usage(),
                cpu_usage=get_cpu_usage()
            )
            await asyncio.sleep(self.interval / 1000)

# Broker side: mark worker as dead if no heartbeat for 3 intervals
```

---

## Enterprise Features

### 1. Horizontal Worker Scaling
```yaml
# Kubernetes HPA config
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: job-workers
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: workers
  minReplicas: 2
  maxReplicas: 100
  metrics:
  - type: External
    external:
      metric:
        name: queue_depth
      target:
        type: AverageValue
        averageValue: "50"
```

### 2. Priority Queues
```python
class PriorityQueueBroker:
    def __init__(self):
        self.queues = {
            "critical": [],   # Always processed first
            "high": [],
            "normal": [],
            "low": []
        }

    def get_next_task(self):
        for priority in ["critical", "high", "normal", "low"]:
            if self.queues[priority]:
                return self.queues[priority].pop(0)
        return None
```

### 3. Circuit Breakers
```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, reset_timeout=60):
        self.failures = 0
        self.threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = "CLOSED"
        self.last_failure = None

    def call(self, func, *args):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.reset_timeout:
                self.state = "HALF-OPEN"
            else:
                raise CircuitOpenError()

        try:
            result = func(*args)
            if self.state == "HALF-OPEN":
                self.state = "CLOSED"
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = time.time()
            if self.failures >= self.threshold:
                self.state = "OPEN"
            raise
```

### 4. Task Deduplication
```python
class DeduplicationLayer:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 3600  # 1 hour

    def is_duplicate(self, idempotency_key: str) -> bool:
        key = f"dedup:{idempotency_key}"
        return self.redis.exists(key)

    def mark_processed(self, idempotency_key: str):
        key = f"dedup:{idempotency_key}"
        self.redis.setex(key, self.ttl, "1")
```

### 5. OpenTelemetry Tracing
```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

async def process_task(task):
    with tracer.start_as_current_span("process_task") as span:
        span.set_attribute("task.id", task["id"])
        span.set_attribute("task.queue", task["queue"])
        span.set_attribute("task.name", task["name"])

        try:
            result = await execute_task(task)
            span.set_status(StatusCode.OK)
            return result
        except Exception as e:
            span.set_status(StatusCode.ERROR, str(e))
            span.record_exception(e)
            raise
```

---

## Performance Considerations

### Throughput Optimization
- **Batching:** Group small tasks for batch processing
- **Connection pooling:** Reuse broker connections
- **Prefetching:** Workers prefetch N tasks to reduce latency

### Latency Optimization
- **Long polling:** Reduce polling overhead
- **Local queues:** Workers maintain local task buffers
- **Async I/O:** Non-blocking operations throughout

### Memory Management
- **Result TTL:** Auto-expire old results
- **Task payload limits:** Enforce max payload size
- **Queue depth limits:** Backpressure when queues grow

---

## Stretch Goals

### 1. Multi-Queue Routing
```python
class Router:
    def __init__(self):
        self.routes = {
            "email.*": "email-workers",
            "report.daily": "report-workers",
            "report.realtime": "fast-workers",
            "*": "default-workers"
        }

    def route(self, task_name: str) -> str:
        for pattern, queue in self.routes.items():
            if fnmatch(task_name, pattern):
                return queue
        return "default"
```

### 2. Distributed Scheduler with Leader Election
```python
class DistributedScheduler:
    def __init__(self, etcd_client):
        self.etcd = etcd_client
        self.is_leader = False

    async def run_election(self):
        lease = await self.etcd.lease(ttl=10)
        try:
            await self.etcd.put(
                "/scheduler/leader",
                self.node_id,
                lease=lease
            )
            self.is_leader = True
            await self.run_scheduler_loop()
        except KeyExistsError:
            self.is_leader = False
            await self.watch_leader()
```

### 3. Worker Autoscaling (KEDA-like)
```python
class WorkerAutoscaler:
    def __init__(self):
        self.min_workers = 1
        self.max_workers = 100
        self.target_queue_time = 5  # seconds

    def calculate_desired_replicas(self, metrics):
        queue_depth = metrics["queue_depth"]
        processing_rate = metrics["tasks_per_second"]

        if processing_rate == 0:
            return self.min_workers

        desired = queue_depth / (processing_rate * self.target_queue_time)
        return max(self.min_workers, min(self.max_workers, int(desired)))
```

---

## Testing Strategy

### Unit Tests
- Task serialization/deserialization
- Retry logic
- Priority queue ordering
- Circuit breaker state transitions

### Integration Tests
- End-to-end task processing
- Worker failure scenarios
- Broker failover
- Result store consistency

### Load Tests
- Throughput under high load
- Latency percentiles (p50, p99)
- Memory usage under load
- Queue depth behavior

### Chaos Tests
- Worker crashes mid-task
- Broker connection drops
- Network partitions
- Clock skew scenarios

---

## Monitoring & Alerting

### Key Metrics
```python
metrics = {
    # Queue health
    "queue_depth": Gauge,
    "queue_latency_p99": Histogram,
    "enqueue_rate": Counter,
    "dequeue_rate": Counter,

    # Worker health
    "active_workers": Gauge,
    "worker_utilization": Gauge,
    "task_duration": Histogram,

    # Failure tracking
    "task_failures": Counter,
    "dlq_depth": Gauge,
    "circuit_breaker_state": Gauge
}
```

### Alerts
- Queue depth > threshold for > 5 minutes
- DLQ depth increasing
- Worker count < minimum
- Task failure rate > 5%
- p99 latency > SLA

---

## Implementation Phases

### Phase 1: Core Queue (Week 1-2)
- [ ] Basic broker with FIFO queue
- [ ] Simple task structure
- [ ] Single worker processing
- [ ] Basic API endpoints

### Phase 2: Reliability (Week 3-4)
- [ ] Retry logic with backoff
- [ ] Dead-letter queue
- [ ] Worker heartbeats
- [ ] Task visibility timeout

### Phase 3: Scale (Week 5-6)
- [ ] Multiple workers
- [ ] Priority queues
- [ ] Task deduplication
- [ ] Result store with TTL

### Phase 4: Enterprise (Week 7-8)
- [ ] Scheduler with cron support
- [ ] Circuit breakers
- [ ] Tracing integration
- [ ] Metrics and monitoring

---

## References

- [Celery Architecture](https://docs.celeryproject.org/en/stable/internals/guide.html)
- [AWS SQS Developer Guide](https://docs.aws.amazon.com/sqs/)
- [Designing Data-Intensive Applications, Ch. 11](https://dataintensive.net/)
- [RabbitMQ Tutorials](https://www.rabbitmq.com/getstarted.html)
