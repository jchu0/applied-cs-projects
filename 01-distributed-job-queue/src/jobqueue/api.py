"""FastAPI application for the job queue API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from jobqueue.broker import Broker, InMemoryBroker
from jobqueue.models import (
    Task,
    TaskCreate,
    TaskResult,
    TaskStatus,
    QueueStats,
)

logger = structlog.get_logger()

# Prometheus metrics
TASKS_ENQUEUED = Counter(
    "jobqueue_tasks_enqueued_total",
    "Total number of tasks enqueued",
    ["queue", "priority"]
)
TASKS_COMPLETED = Counter(
    "jobqueue_tasks_completed_total",
    "Total number of tasks completed",
    ["queue", "status"]
)
TASK_DURATION = Histogram(
    "jobqueue_task_duration_seconds",
    "Task execution duration",
    ["queue"]
)

# Global broker instance
_broker: Broker | None = None


def get_broker() -> Broker:
    """Get the global broker instance."""
    if _broker is None:
        raise RuntimeError("Broker not initialized")
    return _broker


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _broker
    _broker = InMemoryBroker()
    logger.info("Job queue API started")
    yield
    logger.info("Job queue API shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Distributed Job Queue",
        description="A distributed job queue and scheduler system",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Task endpoints
    @app.post("/tasks", response_model=Task)
    async def create_task(task_create: TaskCreate):
        """Create and enqueue a new task."""
        broker = get_broker()

        task = Task(
            name=task_create.name,
            queue=task_create.queue,
            payload=task_create.payload,
            priority=task_create.priority,
            max_retries=task_create.max_retries,
            timeout_ms=task_create.timeout_ms,
            eta=task_create.eta,
            idempotency_key=task_create.idempotency_key,
            metadata=task_create.metadata,
        )

        enqueued_task = await broker.enqueue(task)

        TASKS_ENQUEUED.labels(
            queue=task.queue,
            priority=task.priority
        ).inc()

        return enqueued_task

    @app.get("/tasks/{task_id}", response_model=Task)
    async def get_task(task_id: str):
        """Get task by ID."""
        broker = get_broker()
        task = await broker.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.get("/tasks/{task_id}/result", response_model=TaskResult)
    async def get_task_result(task_id: str):
        """Get result for a completed task."""
        broker = get_broker()

        # First check if task exists
        task = await broker.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        # Check if task is completed
        if task.status not in (TaskStatus.SUCCESS, TaskStatus.FAILURE):
            raise HTTPException(
                status_code=202,
                detail=f"Task is still {task.status}"
            )

        result = await broker.get_result(task_id)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return result

    @app.delete("/tasks/{task_id}")
    async def cancel_task(task_id: str):
        """Cancel a pending task."""
        broker = get_broker()

        cancelled = await broker.cancel_task(task_id)
        if not cancelled:
            raise HTTPException(
                status_code=400,
                detail="Task cannot be cancelled (not found or already processed)"
            )
        return {"status": "cancelled", "task_id": task_id}

    @app.post("/tasks/{task_id}/retry", response_model=Task)
    async def retry_task(task_id: str):
        """Manually retry a failed task."""
        broker = get_broker()

        task = await broker.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.status not in (TaskStatus.FAILURE, TaskStatus.CANCELLED):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot retry task with status {task.status}"
            )

        await broker.requeue(task_id)
        return await broker.get_task(task_id)

    # Queue endpoints
    @app.get("/queues")
    async def list_queues():
        """List all queues."""
        broker = get_broker()
        if isinstance(broker, InMemoryBroker):
            queues = await broker.get_all_queues()
            return {"queues": queues}
        return {"queues": []}

    @app.get("/queues/{queue_name}/stats", response_model=QueueStats)
    async def get_queue_stats(queue_name: str):
        """Get statistics for a queue."""
        broker = get_broker()
        return await broker.get_queue_stats(queue_name)

    # Worker endpoints (for worker communication)
    @app.post("/internal/dequeue")
    async def dequeue_task(
        queues: list[str] = Query(...),
        timeout: float = Query(default=0, ge=0, le=30)
    ):
        """Dequeue a task for worker processing."""
        broker = get_broker()
        task = await broker.dequeue(queues, timeout)
        if not task:
            return {"task": None}
        return {"task": task.model_dump()}

    @app.post("/internal/acknowledge")
    async def acknowledge_task(result: TaskResult):
        """Acknowledge task completion."""
        broker = get_broker()
        await broker.acknowledge(result.task_id, result)

        TASKS_COMPLETED.labels(
            queue="unknown",  # Would need to track this
            status=result.status
        ).inc()

        if result.duration_ms:
            TASK_DURATION.labels(queue="unknown").observe(result.duration_ms / 1000)

        return {"status": "acknowledged"}

    @app.post("/internal/heartbeat")
    async def worker_heartbeat(worker_id: str, current_task: str | None = None):
        """Update worker heartbeat."""
        broker = get_broker()
        await broker.heartbeat(worker_id, current_task)
        return {"status": "ok"}

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
