"""Tests for the FastAPI application (jobqueue.api).

These exercise the real routes exposed by ``create_app`` using an injected
mock broker. The API talks to a :class:`~jobqueue.broker.Broker` via the
``get_broker`` dependency, which ``create_app(broker=...)`` overrides, so no
real broker or Redis is needed here.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

# Skip if required dependencies are not available
pytest.importorskip("fastapi")
pytest.importorskip("redis")
pytest.importorskip("croniter")

from fastapi.testclient import TestClient

from jobqueue.api import create_app
from jobqueue.broker import Broker
from jobqueue.models import QueueStats, Task, TaskResult, TaskStatus


class TestAPI:
    """Test suite for API endpoints backed by an injected mock broker."""

    @pytest_asyncio.fixture
    async def mock_broker(self):
        """Create a mock broker matching the real Broker interface."""
        broker = AsyncMock(spec=Broker)
        # get_all_queues is not on the ABC but the routes probe for it.
        broker.get_all_queues = AsyncMock(return_value=[])
        return broker

    @pytest.fixture
    def client(self, mock_broker):
        """Create a test client with the mock broker injected.

        Used as a context manager so the app lifespan runs (which is a no-op
        for the injected-broker path but keeps behaviour realistic).
        """
        app = create_app(broker=mock_broker)
        with TestClient(app) as client:
            yield client

    def test_health_check(self, client):
        """Health check reports status, timestamp, and version."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_create_task(self, client, mock_broker):
        """POST /tasks enqueues a task and returns it."""
        task_data = {
            "name": "test_task",
            "queue": "default",
            "payload": {"data": "test"},
            "priority": 2,
            "max_retries": 3,
        }

        # enqueue echoes back the stored task; return a realistic Task.
        async def _enqueue(task):
            task.status = TaskStatus.QUEUED
            return task

        mock_broker.enqueue.side_effect = _enqueue

        response = client.post("/tasks", json=task_data)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_task"
        assert data["queue"] == "default"
        assert data["status"] == "queued"
        mock_broker.enqueue.assert_awaited_once()

    def test_create_task_validation(self, client):
        """Missing required fields yields 422."""
        response = client.post("/tasks", json={})
        assert response.status_code == 422

    def test_get_task(self, client, mock_broker):
        """GET /tasks/{id} returns the task."""
        task = Task(
            id="task-002",
            name="test_task",
            queue="default",
            payload={"data": "test"},
            status=TaskStatus.RUNNING,
        )
        mock_broker.get_task.return_value = task

        response = client.get("/tasks/task-002")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "task-002"
        assert data["status"] == "running"

    def test_get_task_not_found(self, client, mock_broker):
        """GET /tasks/{id} for an unknown id yields 404."""
        mock_broker.get_task.return_value = None

        response = client.get("/tasks/nonexistent")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_task_result(self, client, mock_broker):
        """GET /tasks/{id}/result returns the stored result for a done task."""
        task = Task(
            id="task-006",
            name="completed_task",
            queue="default",
            status=TaskStatus.SUCCESS,
            completed_at=datetime.now(timezone.utc),
        )
        result = TaskResult(
            task_id="task-006",
            status=TaskStatus.SUCCESS,
            result={"output": "success", "data": [1, 2, 3]},
        )
        mock_broker.get_task.return_value = task
        mock_broker.get_result.return_value = result

        response = client.get("/tasks/task-006/result")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-006"
        assert data["result"]["output"] == "success"
        assert data["result"]["data"] == [1, 2, 3]

    def test_get_task_result_not_ready(self, client, mock_broker):
        """Requesting a result for a still-running task yields 202."""
        task = Task(id="task-007", name="running", status=TaskStatus.RUNNING)
        mock_broker.get_task.return_value = task

        response = client.get("/tasks/task-007/result")
        assert response.status_code == 202

    def test_cancel_task(self, client, mock_broker):
        """DELETE /tasks/{id} cancels a pending task."""
        mock_broker.cancel_task.return_value = True

        response = client.delete("/tasks/task-003")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"
        assert data["task_id"] == "task-003"

    def test_cancel_task_failure(self, client, mock_broker):
        """DELETE /tasks/{id} on a non-cancellable task yields 400."""
        mock_broker.cancel_task.return_value = False

        response = client.delete("/tasks/task-004")
        assert response.status_code == 400
        assert "cannot be cancelled" in response.json()["detail"].lower()

    def test_retry_task(self, client, mock_broker):
        """POST /tasks/{id}/retry requeues a failed task."""
        failed = Task(id="task-005", name="test_task", status=TaskStatus.FAILURE)
        requeued = Task(id="task-005", name="test_task", status=TaskStatus.QUEUED)
        # get_task is called before and after requeue.
        mock_broker.get_task.side_effect = [failed, requeued]

        response = client.post("/tasks/task-005/retry")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "task-005"
        assert data["status"] == "queued"
        mock_broker.requeue.assert_awaited_once_with("task-005")

    def test_retry_task_not_found(self, client, mock_broker):
        """Retrying an unknown task yields 404."""
        mock_broker.get_task.return_value = None
        response = client.post("/tasks/nope/retry")
        assert response.status_code == 404

    def test_retry_task_wrong_status(self, client, mock_broker):
        """Retrying a task that is not FAILURE/CANCELLED yields 400."""
        running = Task(id="task-008", name="x", status=TaskStatus.RUNNING)
        mock_broker.get_task.return_value = running
        response = client.post("/tasks/task-008/retry")
        assert response.status_code == 400

    def test_list_queues(self, client, mock_broker):
        """GET /queues lists known queue names."""
        mock_broker.get_all_queues.return_value = ["default", "priority"]

        response = client.get("/queues")
        assert response.status_code == 200
        data = response.json()
        assert data["queues"] == ["default", "priority"]

    def test_get_queue_stats(self, client, mock_broker):
        """GET /queues/{name}/stats returns queue statistics."""
        stats = QueueStats(
            name="default", pending=10, running=5, completed=100, failed=2, total=117
        )
        mock_broker.get_queue_stats.return_value = stats

        response = client.get("/queues/default/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "default"
        assert data["pending"] == 10
        assert data["running"] == 5

    def test_internal_dequeue_empty(self, client, mock_broker):
        """POST /internal/dequeue returns {task: None} when nothing is ready."""
        mock_broker.dequeue.return_value = None

        response = client.post("/internal/dequeue?queues=default")
        assert response.status_code == 200
        assert response.json() == {"task": None}

    def test_internal_dequeue_returns_task(self, client, mock_broker):
        """POST /internal/dequeue returns a serialized task when available."""
        task = Task(id="task-009", name="worker_task", queue="default")
        mock_broker.dequeue.return_value = task

        response = client.post("/internal/dequeue?queues=default")
        assert response.status_code == 200
        data = response.json()
        assert data["task"]["id"] == "task-009"

    def test_internal_acknowledge(self, client, mock_broker):
        """POST /internal/acknowledge records a task result."""
        result = TaskResult(
            task_id="task-010",
            status=TaskStatus.SUCCESS,
            result={"ok": True},
            duration_ms=12.5,
        )
        response = client.post("/internal/acknowledge", json=result.model_dump(mode="json"))
        assert response.status_code == 200
        assert response.json() == {"status": "acknowledged"}
        mock_broker.acknowledge.assert_awaited_once()

    def test_internal_heartbeat(self, client, mock_broker):
        """POST /internal/heartbeat updates a worker heartbeat."""
        response = client.post("/internal/heartbeat?worker_id=worker-1")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        mock_broker.heartbeat.assert_awaited_once_with("worker-1", None)

    def test_metrics_endpoint(self, client):
        """GET /metrics returns Prometheus exposition text."""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus text format exposes the registered metric families.
        assert "jobqueue_tasks_enqueued_total" in response.text
