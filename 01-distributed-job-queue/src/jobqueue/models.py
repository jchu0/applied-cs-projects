"""Task and result data models for the job queue system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class Task(BaseModel):
    """Represents a task to be executed by a worker."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Task name/type identifier")
    queue: str = Field(default="default", description="Target queue name")
    payload: dict[str, Any] = Field(default_factory=dict, description="Task arguments")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL)

    # Retry configuration
    retries: int = Field(default=0, description="Current retry count")
    max_retries: int = Field(default=3, description="Maximum retry attempts")

    # Timing
    timeout_ms: int = Field(default=30000, description="Task timeout in milliseconds")
    eta: datetime | None = Field(default=None, description="Earliest time to execute")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Deduplication
    idempotency_key: str | None = Field(default=None, description="Key for deduplication")

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = Field(default=TaskStatus.PENDING)

    # Worker tracking
    worker_id: str | None = None

    class Config:
        use_enum_values = True


class TaskResult(BaseModel):
    """Result of a task execution."""

    task_id: str
    status: TaskStatus
    result: Any = None
    error: str | None = None
    traceback: str | None = None
    started_at: datetime | None = None
    completed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float | None = None
    worker_id: str | None = None

    class Config:
        use_enum_values = True


class TaskCreate(BaseModel):
    """Request model for creating a new task."""

    name: str
    queue: str = "default"
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    timeout_ms: int = 30000
    eta: datetime | None = None
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueueStats(BaseModel):
    """Statistics for a queue."""

    name: str
    pending: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    total: int = 0


class WorkerInfo(BaseModel):
    """Information about a registered worker."""

    id: str
    queues: list[str]
    current_task: str | None = None
    last_heartbeat: datetime
    started_at: datetime
    tasks_completed: int = 0
    tasks_failed: int = 0
    status: str = "idle"
