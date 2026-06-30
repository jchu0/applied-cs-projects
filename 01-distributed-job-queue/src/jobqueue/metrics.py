"""Prometheus metrics for the job queue system."""

from prometheus_client import Counter, Gauge, Histogram, Info

# Service info
SERVICE_INFO = Info("jobqueue", "Job queue service information")

# Task metrics
TASKS_ENQUEUED = Counter(
    "jobqueue_tasks_enqueued_total",
    "Total number of tasks enqueued",
    ["queue", "priority", "task_name"]
)

TASKS_DEQUEUED = Counter(
    "jobqueue_tasks_dequeued_total",
    "Total number of tasks dequeued",
    ["queue"]
)

TASKS_COMPLETED = Counter(
    "jobqueue_tasks_completed_total",
    "Total number of tasks completed",
    ["queue", "status", "task_name"]
)

TASKS_RETRIED = Counter(
    "jobqueue_tasks_retried_total",
    "Total number of task retries",
    ["queue", "task_name"]
)

TASK_DURATION = Histogram(
    "jobqueue_task_duration_seconds",
    "Task execution duration in seconds",
    ["queue", "task_name"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0)
)

# Queue metrics
QUEUE_DEPTH = Gauge(
    "jobqueue_queue_depth",
    "Current number of tasks in queue",
    ["queue", "status"]
)

# Worker metrics
WORKERS_ACTIVE = Gauge(
    "jobqueue_workers_active",
    "Number of active workers",
    ["queue"]
)

WORKER_TASKS_PROCESSING = Gauge(
    "jobqueue_worker_tasks_processing",
    "Number of tasks currently being processed",
    ["worker_id"]
)

WORKER_TASKS_COMPLETED = Counter(
    "jobqueue_worker_tasks_completed_total",
    "Total tasks completed by worker",
    ["worker_id"]
)

WORKER_TASKS_FAILED = Counter(
    "jobqueue_worker_tasks_failed_total",
    "Total tasks failed by worker",
    ["worker_id"]
)

# Circuit breaker metrics
CIRCUIT_BREAKER_STATE = Gauge(
    "jobqueue_circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["task_name"]
)

CIRCUIT_BREAKER_FAILURES = Counter(
    "jobqueue_circuit_breaker_failures_total",
    "Total circuit breaker failures",
    ["task_name"]
)

# DLQ metrics
DLQ_DEPTH = Gauge(
    "jobqueue_dlq_depth",
    "Number of tasks in dead-letter queue",
    ["queue"]
)

# Scheduler metrics
SCHEDULED_JOBS = Gauge(
    "jobqueue_scheduled_jobs_total",
    "Total number of scheduled jobs",
    []
)

SCHEDULED_JOB_RUNS = Counter(
    "jobqueue_scheduled_job_runs_total",
    "Total scheduled job executions",
    ["job_name"]
)

# Broker metrics
BROKER_OPERATIONS = Counter(
    "jobqueue_broker_operations_total",
    "Total broker operations",
    ["operation", "status"]
)

BROKER_OPERATION_DURATION = Histogram(
    "jobqueue_broker_operation_duration_seconds",
    "Broker operation duration",
    ["operation"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
)


def record_task_enqueued(queue: str, priority: int, task_name: str) -> None:
    """Record a task enqueue."""
    TASKS_ENQUEUED.labels(queue=queue, priority=priority, task_name=task_name).inc()


def record_task_dequeued(queue: str) -> None:
    """Record a task dequeue."""
    TASKS_DEQUEUED.labels(queue=queue).inc()


def record_task_completed(queue: str, status: str, task_name: str, duration_seconds: float) -> None:
    """Record a task completion."""
    TASKS_COMPLETED.labels(queue=queue, status=status, task_name=task_name).inc()
    TASK_DURATION.labels(queue=queue, task_name=task_name).observe(duration_seconds)


def record_task_retried(queue: str, task_name: str) -> None:
    """Record a task retry."""
    TASKS_RETRIED.labels(queue=queue, task_name=task_name).inc()


def update_queue_depth(queue: str, pending: int, running: int) -> None:
    """Update queue depth gauges."""
    QUEUE_DEPTH.labels(queue=queue, status="pending").set(pending)
    QUEUE_DEPTH.labels(queue=queue, status="running").set(running)


def update_worker_metrics(worker_id: str, processing: int) -> None:
    """Update worker processing count."""
    WORKER_TASKS_PROCESSING.labels(worker_id=worker_id).set(processing)


def record_worker_task_completed(worker_id: str) -> None:
    """Record worker task completion."""
    WORKER_TASKS_COMPLETED.labels(worker_id=worker_id).inc()


def record_worker_task_failed(worker_id: str) -> None:
    """Record worker task failure."""
    WORKER_TASKS_FAILED.labels(worker_id=worker_id).inc()


def update_circuit_breaker_state(task_name: str, state: int) -> None:
    """Update circuit breaker state (0=closed, 1=open, 2=half_open)."""
    CIRCUIT_BREAKER_STATE.labels(task_name=task_name).set(state)


def record_circuit_breaker_failure(task_name: str) -> None:
    """Record circuit breaker failure."""
    CIRCUIT_BREAKER_FAILURES.labels(task_name=task_name).inc()


def update_dlq_depth(queue: str, depth: int) -> None:
    """Update DLQ depth."""
    DLQ_DEPTH.labels(queue=queue).set(depth)


def record_scheduled_job_run(job_name: str) -> None:
    """Record scheduled job execution."""
    SCHEDULED_JOB_RUNS.labels(job_name=job_name).inc()


def record_broker_operation(operation: str, status: str, duration_seconds: float) -> None:
    """Record broker operation."""
    BROKER_OPERATIONS.labels(operation=operation, status=status).inc()
    BROKER_OPERATION_DURATION.labels(operation=operation).observe(duration_seconds)
