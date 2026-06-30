# Distributed Job Queue

<p align="center">
  <strong>A high-performance, scalable distributed job queue system inspired by Celery</strong>
</p>

<p align="center">
  <a href="#features">Features</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#architecture">Architecture</a> |
  <a href="#usage">Usage</a> |
  <a href="#api-reference">API Reference</a> |
  <a href="#testing">Testing</a> |
  <a href="#documentation">Documentation</a>
</p>

---

## Overview

The Distributed Job Queue is a production-ready, asynchronous task processing system built with Python and Redis. It provides reliable background job processing with support for priority queues, automatic retries, circuit breakers, and comprehensive monitoring - everything you need to handle millions of tasks efficiently.

### Key Features

- **High Performance** - Process thousands of tasks per second with async/await support
- **Priority Queues** - Ensure critical tasks are processed first (CRITICAL, HIGH, NORMAL, LOW)
- **Automatic Retries** - Configurable retry logic with exponential backoff and jitter
- **Circuit Breakers** - Prevent cascading failures with intelligent circuit breaking
- **Persistent Storage** - Redis-backed for durability and fault tolerance
- **Real-time Monitoring** - Prometheus metrics and OpenTelemetry distributed tracing
- **Smart Routing** - Route tasks to specific queues and workers
- **Task Scheduling** - Cron and interval-based recurring tasks via croniter
- **Idempotency** - Prevent duplicate task execution with idempotency keys
- **Dead Letter Queue** - Automatic handling of permanently failed tasks
- **Worker Pools** - Dynamic scaling with configurable concurrency

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Redis 7.0 or higher
- Docker (optional, for containerized deployment)

### Installation

```bash
# Navigate to the project directory
cd projects/01-distributed-job-queue

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
pip install -e ".[dev]"
```

### Using Docker (Optional)

```bash
# Start all services with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale worker=5
```

### Running the System

#### 1. Start Redis

```bash
# Using Docker
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Or use your local Redis installation
redis-server
```

#### 2. Start the API Server

```bash
python -m jobqueue.api

# Or with uvicorn for development
uvicorn jobqueue.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at http://localhost:8000. View the interactive docs at http://localhost:8000/docs

#### 3. Start Workers

```bash
# Start a worker processing default queue
python -m jobqueue.worker

# Start a worker for specific queues with custom concurrency
python -m jobqueue.worker --queues priority,default --concurrency 10

# Start multiple workers (in separate terminals or use supervisor/systemd)
python -m jobqueue.worker --worker-id worker-1
python -m jobqueue.worker --worker-id worker-2
```

#### 4. Start the Scheduler (Optional)

```bash
# For recurring/scheduled tasks
python -m jobqueue.scheduler
```

## Architecture

The system follows a producer-consumer pattern with Redis as the message broker:

```
+---------------+     +------------------+     +-----------------+
|  Client Apps  |---->|   FastAPI API    |---->|  Redis Broker   |
+---------------+     +------------------+     +-----------------+
                                                       |
                              +------------------------+------------------------+
                              |                        |                        |
                              v                        v                        v
                       +-----------+            +-----------+            +-----------+
                       |  Worker 1 |            |  Worker 2 |            |  Worker N |
                       +-----------+            +-----------+            +-----------+
                              |                        |                        |
                              +------------------------+------------------------+
                                                       |
                                                       v
                                              +-----------------+
                                              | Metrics/Tracing |
                                              +-----------------+
```

### Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Task Models** | `src/jobqueue/models.py` | Pydantic models for Task, TaskResult, QueueStats |
| **In-Memory Broker** | `src/jobqueue/broker.py` | FIFO queue with priority support |
| **Redis Broker** | `src/jobqueue/redis_broker.py` | Persistent Redis-backed storage |
| **Worker** | `src/jobqueue/worker.py` | Task execution with error handling |
| **Worker Pool** | `src/jobqueue/pool.py` | Manages multiple concurrent workers |
| **Scheduler** | `src/jobqueue/scheduler.py` | Cron-based recurring task scheduling |
| **Circuit Breaker** | `src/jobqueue/circuit_breaker.py` | Fault tolerance pattern implementation |
| **API Server** | `src/jobqueue/api.py` | FastAPI REST endpoints |
| **Metrics** | `src/jobqueue/metrics.py` | Prometheus metrics collection |
| **Tracing** | `src/jobqueue/tracing.py` | OpenTelemetry integration |

### Task States

Tasks transition through the following states:

```
PENDING --> QUEUED --> RUNNING --> SUCCESS
                          |
                          +--> FAILURE --> RETRY --> RUNNING
                          |                    |
                          +--> CANCELLED       +--> FAILURE (max retries)
```

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Usage

### Basic Example

```python
import asyncio
from jobqueue import Client

async def main():
    # Initialize client
    client = Client(api_url="http://localhost:8000")

    # Submit a simple task
    task = await client.submit_task(
        name="send_email",
        payload={
            "to": "user@example.com",
            "subject": "Welcome!",
            "body": "Thanks for signing up!"
        }
    )

    print(f"Task submitted: {task.id}")

    # Wait for completion
    result = await client.wait_for_result(task.id, timeout=30)
    print(f"Task completed: {result}")

# Run the example
asyncio.run(main())
```

### Worker Implementation

```python
from datetime import datetime
from jobqueue.worker import Worker
from jobqueue.models import Task

# Create worker instance
worker = Worker(
    redis_url="redis://localhost:6379",
    queues=["default", "priority"],
    concurrency=10
)

# Register task handlers
@worker.task("send_email")
async def send_email_handler(task: Task):
    """Send email task handler."""
    to = task.payload["to"]
    subject = task.payload["subject"]

    # Your email sending logic here
    print(f"Sending email to {to}: {subject}")

    # Return result
    return {
        "status": "sent",
        "message_id": "msg-123456",
        "sent_at": datetime.now().isoformat()
    }

@worker.task("process_image")
async def process_image_handler(task: Task):
    """Process image task handler."""
    image_url = task.payload["image_url"]

    # Image processing logic
    # await download_image(image_url)
    # await resize_image()

    return {"status": "processed", "url": "processed_image_url"}

# Start the worker
if __name__ == "__main__":
    worker.start()
```

### Advanced Features

#### Priority Tasks

```python
from jobqueue.models import TaskPriority

# Submit critical priority task (processed first)
urgent_task = await client.submit_task(
    name="process_payment",
    payload={"order_id": "12345", "amount": 99.99},
    priority=TaskPriority.CRITICAL  # 0 = highest priority
)

# Submit low-priority background task
background_task = await client.submit_task(
    name="generate_report",
    payload={"report_type": "monthly"},
    priority=TaskPriority.LOW  # 3 = lowest priority
)

# Priority levels: CRITICAL (0), HIGH (1), NORMAL (2), LOW (3)
```

#### Retry Configuration

```python
# Task with custom retry settings
task = await client.submit_task(
    name="api_call",
    payload={"endpoint": "https://api.example.com/data"},
    max_retries=5,
    retry_delay=2.0,  # Initial delay in seconds
    retry_backoff=2.0  # Exponential backoff multiplier
)
```

#### Scheduled Tasks

```python
# Schedule a recurring task (cron expression)
schedule = await client.create_schedule(
    name="daily_backup",
    task_name="backup_database",
    cron="0 2 * * *",  # Run at 2 AM every day
    payload={"database": "production"}
)

# Schedule task with interval
schedule = await client.create_schedule(
    name="health_check",
    task_name="check_system_health",
    interval=300,  # Every 5 minutes
    payload={"services": ["api", "database", "cache"]}
)
```

#### Bulk Operations

```python
# Submit multiple tasks at once
tasks = await client.submit_bulk([
    {
        "name": "process_order",
        "payload": {"order_id": f"order-{i}"},
        "priority": 5
    }
    for i in range(100)
])

print(f"Submitted {len(tasks)} tasks")
```

#### Circuit Breaker Pattern

```python
# Worker with circuit breaker configuration
worker = Worker(
    redis_url="redis://localhost:6379",
    enable_circuit_breaker=True,
    circuit_failure_threshold=5,  # Open after 5 failures
    circuit_reset_timeout=60.0    # Try again after 60 seconds
)

@worker.task("external_api_call")
async def call_external_api(task: Task):
    """Protected by circuit breaker."""
    # If external API fails repeatedly, circuit opens
    # and subsequent tasks fail fast without calling API
    response = await external_api.call(task.payload)
    return response
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=jobqueue --cov-report=html

# Run specific test categories
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
pytest -m "not slow"        # Skip slow tests

# Run with verbose output
pytest -vv

# Run tests in parallel
pytest -n auto
```

### Test Coverage

The project maintains >80% test coverage. View the coverage report:

```bash
# Generate HTML coverage report
pytest --cov=jobqueue --cov-report=html

# Open in browser
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

### Integration Testing

```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/ --integration

# Cleanup
docker-compose -f docker-compose.test.yml down
```

## API Reference

### REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tasks` | POST | Submit a new task |
| `/tasks/{id}` | GET | Get task status |
| `/tasks/{id}/cancel` | POST | Cancel a task |
| `/tasks/{id}/retry` | POST | Retry a failed task |
| `/queues/stats` | GET | Get queue statistics |
| `/workers` | GET | List active workers |
| `/metrics` | GET | Prometheus metrics |

Full API documentation available at http://localhost:8000/docs when running the server.

### Python Client

```python
from jobqueue import Client

# Initialize client
client = Client(
    api_url="http://localhost:8000",
    api_key="your-api-key"
)

# Available methods
await client.submit_task(name, payload, **options)
await client.get_task(task_id)
await client.cancel_task(task_id)
await client.retry_task(task_id)
await client.list_tasks(queue=None, status=None)
await client.get_queue_stats()
await client.list_workers()
```

## Monitoring

### Prometheus Metrics

The system exports comprehensive metrics:

- `tasks_submitted_total` - Total tasks submitted
- `tasks_completed_total` - Total tasks completed
- `tasks_failed_total` - Total tasks failed
- `task_execution_duration_seconds` - Task execution time histogram
- `queue_depth` - Current queue depth per queue
- `worker_count` - Number of active workers

### Grafana Dashboards

Import the provided dashboards from `docs/grafana/`:
- `overview.json` - System overview
- `workers.json` - Worker performance
- `queues.json` - Queue metrics

### Health Checks

```bash
# Check system health
curl http://localhost:8000/health

# Response
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:00:00Z",
  "checks": {
    "redis": "connected",
    "workers": 5
  }
}
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=yourpassword
REDIS_SSL=false

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
API_KEY=your-secure-api-key

# Worker Configuration
WORKER_CONCURRENCY=10
WORKER_QUEUES=default,priority
WORKER_POLL_INTERVAL=1.0
WORKER_HEARTBEAT_INTERVAL=5.0

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
```

### Configuration File

Create `config.yaml`:

```yaml
redis:
  host: localhost
  port: 6379
  db: 0
  password: null
  ssl: false
  max_connections: 50

worker:
  concurrency: 10
  queues:
    - default
    - priority
  poll_interval: 1.0
  heartbeat_interval: 5.0
  circuit_breaker:
    enabled: true
    failure_threshold: 5
    reset_timeout: 60

api:
  host: 0.0.0.0
  port: 8000
  workers: 4
  cors_origins:
    - http://localhost:3000

monitoring:
  metrics_enabled: true
  tracing_enabled: true
  log_level: INFO
```

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [API Documentation](docs/API.md) - Complete API reference
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Contributing Guide](docs/CONTRIBUTING.md) - How to contribute

## Performance

The system is designed for high throughput:

- **Task Submission**: 10,000+ tasks/second
- **Task Processing**: 5,000+ tasks/second (with 10 workers)
- **Latency**: < 10ms for task submission
- **Scalability**: Horizontally scalable to 100+ workers

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting and type checking
black src/ tests/
ruff check src/ tests/
mypy src/

# Run tests
pytest

# Run tests with coverage
pytest --cov=jobqueue --cov-report=html
```

## Project Structure

```
01-distributed-job-queue/
├── src/jobqueue/          # Main source code
│   ├── __init__.py        # Package exports
│   ├── api.py             # FastAPI REST server
│   ├── broker.py          # In-memory broker
│   ├── redis_broker.py    # Redis broker
│   ├── worker.py          # Task worker
│   ├── pool.py            # Worker pool management
│   ├── scheduler.py       # Cron scheduling
│   ├── circuit_breaker.py # Fault tolerance
│   ├── models.py          # Pydantic models
│   ├── metrics.py         # Prometheus metrics
│   ├── tracing.py         # OpenTelemetry
│   └── config.py          # Configuration
├── tests/                 # Test suite
├── examples/              # Usage examples
├── docs/                  # Documentation
│   ├── API.md             # API reference
│   ├── ARCHITECTURE.md    # System design
│   ├── DEPLOYMENT.md      # Deployment guide
│   └── CONTRIBUTING.md    # Contribution guide
└── pyproject.toml         # Project configuration
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Inspired by [Celery](https://docs.celeryproject.org/)
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Powered by [Redis](https://redis.io/)
- Metrics via [Prometheus](https://prometheus.io/)
- Tracing via [OpenTelemetry](https://opentelemetry.io/)