"""Tests for task models."""

import pytest
from datetime import datetime, timezone

from jobqueue.models import (
    Task,
    TaskCreate,
    TaskResult,
    TaskStatus,
    TaskPriority,
    QueueStats,
)


class TestTask:
    """Tests for Task model."""

    def test_task_defaults(self):
        """Test task default values."""
        task = Task(name="test")

        assert task.id is not None
        assert task.name == "test"
        assert task.queue == "default"
        assert task.payload == {}
        assert task.priority == TaskPriority.NORMAL
        assert task.retries == 0
        assert task.max_retries == 3
        assert task.timeout_ms == 30000
        assert task.status == TaskStatus.PENDING

    def test_task_with_payload(self):
        """Test task with custom payload."""
        task = Task(
            name="send_email",
            queue="emails",
            payload={"to": "test@example.com"},
            priority=TaskPriority.HIGH,
        )

        assert task.name == "send_email"
        assert task.queue == "emails"
        assert task.payload == {"to": "test@example.com"}
        assert task.priority == TaskPriority.HIGH

    def test_task_with_eta(self):
        """Test task with scheduled execution time."""
        eta = datetime.now(timezone.utc)
        task = Task(name="scheduled", eta=eta)

        assert task.eta == eta

    def test_task_serialization(self):
        """Test task serialization to dict."""
        task = Task(name="test", payload={"key": "value"})
        data = task.model_dump()

        assert data["name"] == "test"
        assert data["payload"] == {"key": "value"}
        assert "id" in data


class TestTaskCreate:
    """Tests for TaskCreate model."""

    def test_task_create_minimal(self):
        """Test minimal task creation request."""
        create = TaskCreate(name="test")

        assert create.name == "test"
        assert create.queue == "default"
        assert create.priority == TaskPriority.NORMAL

    def test_task_create_full(self):
        """Test full task creation request."""
        create = TaskCreate(
            name="process",
            queue="heavy",
            payload={"data": [1, 2, 3]},
            priority=TaskPriority.CRITICAL,
            max_retries=5,
            timeout_ms=60000,
            idempotency_key="unique-key-123",
        )

        assert create.name == "process"
        assert create.queue == "heavy"
        assert create.max_retries == 5
        assert create.idempotency_key == "unique-key-123"


class TestTaskResult:
    """Tests for TaskResult model."""

    def test_success_result(self):
        """Test successful task result."""
        result = TaskResult(
            task_id="test-id",
            status=TaskStatus.SUCCESS,
            result={"output": 42},
            duration_ms=150.5,
        )

        assert result.task_id == "test-id"
        assert result.status == TaskStatus.SUCCESS
        assert result.result == {"output": 42}
        assert result.error is None

    def test_failure_result(self):
        """Test failed task result."""
        result = TaskResult(
            task_id="test-id",
            status=TaskStatus.FAILURE,
            error="Connection refused",
            traceback="Traceback...",
        )

        assert result.status == TaskStatus.FAILURE
        assert result.error == "Connection refused"
        assert result.traceback == "Traceback..."


class TestQueueStats:
    """Tests for QueueStats model."""

    def test_queue_stats(self):
        """Test queue statistics."""
        stats = QueueStats(
            name="default",
            pending=10,
            running=5,
            completed=100,
            failed=3,
            total=118,
        )

        assert stats.name == "default"
        assert stats.pending == 10
        assert stats.total == 118
