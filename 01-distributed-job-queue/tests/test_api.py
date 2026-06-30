"""Comprehensive tests for the API module."""

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# Skip if required dependencies are not available
pytest.importorskip("fastapi")
pytest.importorskip("redis")
pytest.importorskip("croniter")

from fastapi.testclient import TestClient

from jobqueue.api import create_app
from jobqueue.broker import Broker
from jobqueue.models import Task, TaskResult, TaskStatus, WorkerInfo
from jobqueue.scheduler import Scheduler


class TestAPI:
    """Test suite for API endpoints."""

    @pytest_asyncio.fixture
    async def mock_broker(self):
        """Create a mock broker for testing."""
        broker = AsyncMock(spec=Broker)
        broker.submit_task = AsyncMock()
        broker.get_task = AsyncMock()
        broker.list_tasks = AsyncMock(return_value=[])
        broker.cancel_task = AsyncMock()
        broker.get_queue_stats = AsyncMock(return_value={})
        broker.list_workers = AsyncMock(return_value=[])
        broker.get_worker_info = AsyncMock()
        broker.connect = AsyncMock()
        broker.close = AsyncMock()
        return broker

    @pytest_asyncio.fixture
    async def mock_scheduler(self):
        """Create a mock scheduler for testing."""
        scheduler = AsyncMock(spec=Scheduler)
        scheduler.schedule_task = AsyncMock()
        scheduler.cancel_scheduled_task = AsyncMock()
        scheduler.list_scheduled_tasks = AsyncMock(return_value=[])
        scheduler.start = AsyncMock()
        scheduler.stop = AsyncMock()
        return scheduler

    @pytest_asyncio.fixture
    async def app(self, mock_broker, mock_scheduler):
        """Create test application."""
        return create_app(broker=mock_broker, scheduler=mock_scheduler)

    @pytest_asyncio.fixture
    async def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_submit_task(self, client, mock_broker):
        """Test task submission endpoint."""
        task_data = {
            "name": "test_task",
            "queue": "default",
            "payload": {"data": "test"},
            "priority": 5,
            "timeout": 30,
            "max_retries": 3
        }

        mock_broker.submit_task.return_value = Task(
            id="task-001",
            name=task_data["name"],
            queue=task_data["queue"],
            payload=task_data["payload"],
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=task_data["priority"],
            timeout=task_data["timeout"],
            max_retries=task_data["max_retries"]
        )

        response = client.post("/tasks", json=task_data)
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "task-001"
        assert data["name"] == "test_task"
        assert data["status"] == "pending"

    def test_submit_task_validation(self, client):
        """Test task submission validation."""
        # Missing required fields
        response = client.post("/tasks", json={})
        assert response.status_code == 422

        # Invalid priority
        response = client.post("/tasks", json={
            "name": "test",
            "queue": "default",
            "priority": -1
        })
        assert response.status_code == 422

    def test_get_task(self, client, mock_broker):
        """Test get task endpoint."""
        task = Task(
            id="task-002",
            name="test_task",
            queue="default",
            payload={"data": "test"},
            status=TaskStatus.RUNNING,
            created_at=datetime.now(timezone.utc),
            started_at=datetime.now(timezone.utc),
            priority=5
        )
        mock_broker.get_task.return_value = task

        response = client.get("/tasks/task-002")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "task-002"
        assert data["status"] == "running"

    def test_get_task_not_found(self, client, mock_broker):
        """Test get task with non-existent ID."""
        mock_broker.get_task.return_value = None

        response = client.get("/tasks/nonexistent")
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_list_tasks(self, client, mock_broker):
        """Test list tasks endpoint."""
        tasks = [
            Task(
                id=f"task-{i:03d}",
                name="test_task",
                queue="default",
                payload={"index": i},
                status=TaskStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                priority=1
            )
            for i in range(5)
        ]
        mock_broker.list_tasks.return_value = tasks

        response = client.get("/tasks")
        assert response.status_code == 200
        data = response.json()
        assert len(data["tasks"]) == 5
        assert data["total"] == 5

    def test_list_tasks_with_filters(self, client, mock_broker):
        """Test list tasks with filters."""
        mock_broker.list_tasks.return_value = []

        response = client.get("/tasks?queue=priority&status=running&limit=10&offset=5")
        assert response.status_code == 200

        # Verify broker was called with correct filters
        mock_broker.list_tasks.assert_called_with(
            queue="priority",
            status=TaskStatus.RUNNING,
            limit=10,
            offset=5
        )

    def test_cancel_task(self, client, mock_broker):
        """Test cancel task endpoint."""
        mock_broker.cancel_task.return_value = True

        response = client.post("/tasks/task-003/cancel")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Task cancelled successfully"
        assert data["task_id"] == "task-003"

    def test_cancel_task_failure(self, client, mock_broker):
        """Test cancel task failure."""
        mock_broker.cancel_task.return_value = False

        response = client.post("/tasks/task-004/cancel")
        assert response.status_code == 400
        data = response.json()
        assert "could not be cancelled" in data["detail"].lower()

    def test_retry_task(self, client, mock_broker):
        """Test retry task endpoint."""
        original_task = Task(
            id="task-005",
            name="test_task",
            queue="default",
            payload={"data": "test"},
            status=TaskStatus.FAILED,
            created_at=datetime.now(timezone.utc),
            priority=5
        )

        new_task = Task(
            id="task-005-retry",
            name="test_task",
            queue="default",
            payload={"data": "test"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=5
        )

        mock_broker.get_task.return_value = original_task
        mock_broker.submit_task.return_value = new_task

        response = client.post("/tasks/task-005/retry")
        assert response.status_code == 200
        data = response.json()
        assert data["new_task_id"] == "task-005-retry"
        assert data["original_task_id"] == "task-005"

    def test_get_queue_stats(self, client, mock_broker):
        """Test get queue statistics endpoint."""
        stats = {
            "default": {
                "pending": 10,
                "running": 5,
                "completed": 100,
                "failed": 2
            },
            "priority": {
                "pending": 3,
                "running": 1,
                "completed": 50,
                "failed": 0
            }
        }
        mock_broker.get_queue_stats.return_value = stats

        response = client.get("/queues/stats")
        assert response.status_code == 200
        data = response.json()
        assert "default" in data
        assert data["default"]["pending"] == 10
        assert data["priority"]["running"] == 1

    def test_list_workers(self, client, mock_broker):
        """Test list workers endpoint."""
        workers = [
            WorkerInfo(
                id=f"worker-{i}",
                hostname=f"host-{i}",
                pid=1000 + i,
                queues=["default"],
                status="active",
                started_at=datetime.now(timezone.utc),
                last_heartbeat=datetime.now(timezone.utc),
                tasks_processed=10 * i,
                tasks_failed=i
            )
            for i in range(3)
        ]
        mock_broker.list_workers.return_value = workers

        response = client.get("/workers")
        assert response.status_code == 200
        data = response.json()
        assert len(data["workers"]) == 3
        assert data["total"] == 3
        assert data["active"] == 3

    def test_get_worker_info(self, client, mock_broker):
        """Test get worker info endpoint."""
        worker = WorkerInfo(
            id="worker-001",
            hostname="host-001",
            pid=1001,
            queues=["default", "priority"],
            status="active",
            started_at=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
            tasks_processed=100,
            tasks_failed=5,
            current_task_id="task-current"
        )
        mock_broker.get_worker_info.return_value = worker

        response = client.get("/workers/worker-001")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "worker-001"
        assert data["tasks_processed"] == 100
        assert data["current_task_id"] == "task-current"

    def test_schedule_task(self, client, mock_scheduler):
        """Test schedule task endpoint."""
        schedule_data = {
            "name": "scheduled_task",
            "queue": "default",
            "payload": {"data": "scheduled"},
            "cron": "0 * * * *",  # Every hour
            "timezone": "UTC"
        }

        mock_scheduler.schedule_task.return_value = "schedule-001"

        response = client.post("/schedule", json=schedule_data)
        assert response.status_code == 201
        data = response.json()
        assert data["schedule_id"] == "schedule-001"
        assert data["cron"] == "0 * * * *"

    def test_schedule_task_with_interval(self, client, mock_scheduler):
        """Test schedule task with interval."""
        schedule_data = {
            "name": "interval_task",
            "queue": "default",
            "payload": {"data": "interval"},
            "interval": 3600,  # Every hour in seconds
        }

        mock_scheduler.schedule_task.return_value = "schedule-002"

        response = client.post("/schedule", json=schedule_data)
        assert response.status_code == 201
        data = response.json()
        assert data["schedule_id"] == "schedule-002"
        assert data["interval"] == 3600

    def test_cancel_scheduled_task(self, client, mock_scheduler):
        """Test cancel scheduled task."""
        mock_scheduler.cancel_scheduled_task.return_value = True

        response = client.delete("/schedule/schedule-001")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Schedule cancelled successfully"

    def test_list_scheduled_tasks(self, client, mock_scheduler):
        """Test list scheduled tasks."""
        scheduled_tasks = [
            {
                "id": "schedule-001",
                "name": "task1",
                "cron": "0 * * * *",
                "next_run": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "schedule-002",
                "name": "task2",
                "interval": 3600,
                "next_run": datetime.now(timezone.utc).isoformat()
            }
        ]
        mock_scheduler.list_scheduled_tasks.return_value = scheduled_tasks

        response = client.get("/schedule")
        assert response.status_code == 200
        data = response.json()
        assert len(data["schedules"]) == 2
        assert data["total"] == 2

    def test_bulk_submit_tasks(self, client, mock_broker):
        """Test bulk task submission."""
        tasks_data = [
            {
                "name": f"bulk_task_{i}",
                "queue": "default",
                "payload": {"index": i},
                "priority": i
            }
            for i in range(3)
        ]

        submitted_tasks = [
            Task(
                id=f"task-bulk-{i}",
                name=task["name"],
                queue=task["queue"],
                payload=task["payload"],
                status=TaskStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                priority=task["priority"]
            )
            for i, task in enumerate(tasks_data)
        ]

        mock_broker.submit_task.side_effect = submitted_tasks

        response = client.post("/tasks/bulk", json={"tasks": tasks_data})
        assert response.status_code == 201
        data = response.json()
        assert len(data["task_ids"]) == 3
        assert data["submitted"] == 3

    def test_get_task_result(self, client, mock_broker):
        """Test get task result endpoint."""
        task = Task(
            id="task-006",
            name="completed_task",
            queue="default",
            payload={"data": "test"},
            status=TaskStatus.COMPLETED,
            created_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
            priority=5
        )

        result = TaskResult(
            task_id="task-006",
            result={"output": "success", "data": [1, 2, 3]},
            error=None
        )

        mock_broker.get_task.return_value = task
        mock_broker.get_task_result = AsyncMock(return_value=result)

        response = client.get("/tasks/task-006/result")
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "task-006"
        assert data["result"]["output"] == "success"
        assert data["result"]["data"] == [1, 2, 3]

    def test_pause_queue(self, client, mock_broker):
        """Test pause queue endpoint."""
        mock_broker.pause_queue = AsyncMock(return_value=True)

        response = client.post("/queues/default/pause")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Queue paused successfully"
        assert data["queue"] == "default"

    def test_resume_queue(self, client, mock_broker):
        """Test resume queue endpoint."""
        mock_broker.resume_queue = AsyncMock(return_value=True)

        response = client.post("/queues/default/resume")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Queue resumed successfully"
        assert data["queue"] == "default"

    def test_purge_queue(self, client, mock_broker):
        """Test purge queue endpoint."""
        mock_broker.purge_queue = AsyncMock(return_value=10)

        response = client.post("/queues/default/purge")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Queue purged successfully"
        assert data["tasks_removed"] == 10

    def test_metrics_endpoint(self, client, mock_broker):
        """Test metrics endpoint for Prometheus."""
        mock_broker.get_metrics = AsyncMock(return_value={
            "tasks_submitted_total": 1000,
            "tasks_completed_total": 900,
            "tasks_failed_total": 50,
            "tasks_pending": 50,
            "workers_active": 5
        })

        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "tasks_submitted_total 1000" in content
        assert "tasks_completed_total 900" in content
        assert "workers_active 5" in content