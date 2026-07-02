"""FastAPI application for the job queue API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from jobqueue.broker import Broker, InMemoryBroker
from jobqueue.security import install_hardening
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

# Global broker instance (used as the default when no broker is injected).
_broker: Broker | None = None


def get_broker() -> Broker:
    """Get the global broker instance.

    This is the default dependency. When a broker is injected via
    ``create_app(broker=...)`` the app overrides this dependency so the
    injected instance is used instead (see :func:`create_app`).
    """
    if _broker is None:
        raise RuntimeError("Broker not initialized")
    return _broker


def _make_lifespan(broker: Broker | None):
    """Build a lifespan manager.

    When ``broker`` is provided (dependency injection, e.g. in tests) it is
    used as-is and the global broker is left untouched. Otherwise a default
    :class:`InMemoryBroker` is created for the process lifetime, preserving the
    original no-arg behaviour.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _broker
        if broker is None:
            _broker = InMemoryBroker()
            logger.info("Job queue API started", broker="InMemoryBroker")
        else:
            logger.info("Job queue API started", broker=type(broker).__name__)
        yield
        logger.info("Job queue API shutting down")

    return lifespan


def create_app(broker: Broker | None = None, scheduler: Any = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        broker: Optional broker instance to inject. When provided it is stored
            on ``app.state`` and wired in via a FastAPI dependency override, so
            the app talks to the given instance instead of the process-global
            default. When omitted, the app lazily creates an
            :class:`InMemoryBroker` on startup (the original behaviour).
        scheduler: Optional scheduler instance. Stored on ``app.state`` for
            callers that manage scheduling out of band. The HTTP API does not
            expose scheduling endpoints, so this is not otherwise used.
    """
    app = FastAPI(
        title="Distributed Job Queue",
        description="A distributed job queue and scheduler system",
        version="0.1.0",
        lifespan=_make_lifespan(broker),
    )

    # Expose injected dependencies on app.state for introspection/testing.
    app.state.broker = broker
    app.state.scheduler = scheduler

    # When a broker is injected, override the get_broker dependency so every
    # route uses it without touching the process-global instance.
    if broker is not None:
        app.dependency_overrides[get_broker] = lambda: broker

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
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": app.version,
        }

    # Metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # Task endpoints
    @app.post("/tasks", response_model=Task)
    async def create_task(task_create: TaskCreate, broker: Broker = Depends(get_broker)):
        """Create and enqueue a new task."""
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
    async def get_task(task_id: str, broker: Broker = Depends(get_broker)):
        """Get task by ID."""
        task = await broker.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.get("/tasks/{task_id}/result", response_model=TaskResult)
    async def get_task_result(task_id: str, broker: Broker = Depends(get_broker)):
        """Get result for a completed task."""
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
    async def cancel_task(task_id: str, broker: Broker = Depends(get_broker)):
        """Cancel a pending task."""
        cancelled = await broker.cancel_task(task_id)
        if not cancelled:
            raise HTTPException(
                status_code=400,
                detail="Task cannot be cancelled (not found or already processed)"
            )
        return {"status": "cancelled", "task_id": task_id}

    @app.post("/tasks/{task_id}/retry", response_model=Task)
    async def retry_task(task_id: str, broker: Broker = Depends(get_broker)):
        """Manually retry a failed task."""
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
    async def list_queues(broker: Broker = Depends(get_broker)):
        """List all queues."""
        if hasattr(broker, "get_all_queues"):
            queues = await broker.get_all_queues()
            return {"queues": queues}
        return {"queues": []}

    @app.get("/queues/{queue_name}/stats", response_model=QueueStats)
    async def get_queue_stats(queue_name: str, broker: Broker = Depends(get_broker)):
        """Get statistics for a queue."""
        return await broker.get_queue_stats(queue_name)

    # Worker endpoints (for worker communication)
    @app.post("/internal/dequeue")
    async def dequeue_task(
        queues: list[str] = Query(...),
        timeout: float = Query(default=0, ge=0, le=30),
        broker: Broker = Depends(get_broker),
    ):
        """Dequeue a task for worker processing."""
        task = await broker.dequeue(queues, timeout)
        if not task:
            return {"task": None}
        return {"task": task.model_dump()}

    @app.post("/internal/acknowledge")
    async def acknowledge_task(result: TaskResult, broker: Broker = Depends(get_broker)):
        """Acknowledge task completion."""
        await broker.acknowledge(result.task_id, result)

        TASKS_COMPLETED.labels(
            queue="unknown",  # Would need to track this
            status=result.status
        ).inc()

        if result.duration_ms:
            TASK_DURATION.labels(queue="unknown").observe(result.duration_ms / 1000)

        return {"status": "acknowledged"}

    @app.post("/internal/heartbeat")
    async def worker_heartbeat(
        worker_id: str,
        current_task: str | None = None,
        broker: Broker = Depends(get_broker),
    ):
        """Update worker heartbeat."""
        await broker.heartbeat(worker_id, current_task)
        return {"status": "ok"}

    # Production hardening: API-key auth, rate limiting, request timeout.
    # Health/readiness/root, /metrics, and docs stay open; everything else
    # (including the worker-facing /internal/* routes) is protected.
    install_hardening(app)

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
