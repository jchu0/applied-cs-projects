"""Tests for the Worker class (jobqueue.worker).

These exercise the real worker behaviour against a mock broker that matches
the :class:`~jobqueue.broker.Broker` interface: ``dequeue`` to fetch work,
``acknowledge`` to record results, ``requeue`` for retries, and the optional
``move_to_dlq`` for permanently failed tasks.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from jobqueue.broker import Broker
from jobqueue.models import Task, TaskResult, TaskStatus
from jobqueue.worker import Worker


def _make_task(name: str, task_id: str = "task-1", **kwargs) -> Task:
    return Task(
        id=task_id,
        name=name,
        queue="test-queue",
        payload=kwargs.pop("payload", {"data": "test"}),
        status=TaskStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
        **kwargs,
    )


class TestWorker:
    """Test suite for Worker class."""

    @pytest_asyncio.fixture
    async def mock_broker(self):
        """Create a mock broker matching the real Broker interface."""
        broker = AsyncMock(spec=Broker)
        # move_to_dlq is optional on the base interface; add it explicitly so
        # DLQ-related tests can assert on it.
        broker.move_to_dlq = AsyncMock()
        return broker

    @pytest_asyncio.fixture
    async def worker(self, mock_broker):
        """Create a worker instance for testing."""
        return Worker(
            broker=mock_broker,
            queues=["test-queue"],
            concurrency=2,
            poll_interval=0.1,
            heartbeat_interval=1.0,
            enable_circuit_breaker=True,
            circuit_failure_threshold=3,
            circuit_reset_timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_worker_initialization(self, worker, mock_broker):
        """Worker stores its configuration correctly."""
        assert worker.broker == mock_broker
        assert worker.queues == ["test-queue"]
        assert worker.concurrency == 2
        assert worker.poll_interval == 0.1
        assert worker.heartbeat_interval == 1.0
        assert worker.enable_circuit_breaker is True
        assert worker.circuit_failure_threshold == 3
        assert worker.circuit_reset_timeout == 5.0
        assert worker.worker_id.startswith("worker-")

    @pytest.mark.asyncio
    async def test_register_handler(self, worker):
        """The task decorator registers a handler by name."""

        @worker.task("test_task")
        async def test_handler(task: Task):
            return {"result": "success"}

        assert "test_task" in worker._handlers
        assert worker._handlers["test_task"] == test_handler

    @pytest.mark.asyncio
    async def test_process_task_success(self, worker, mock_broker):
        """A successful handler acknowledges a SUCCESS result."""
        task = _make_task("test_task", task_id="task-123")

        @worker.task("test_task")
        async def test_handler(task_obj: Task):
            return {"result": "processed"}

        await worker._process_task(task)

        mock_broker.acknowledge.assert_awaited_once()
        call_args = mock_broker.acknowledge.call_args[0]
        assert call_args[0] == task.id
        result = call_args[1]
        assert isinstance(result, TaskResult)
        assert result.task_id == task.id
        assert result.status == TaskStatus.SUCCESS
        assert result.result == {"result": "processed"}
        assert worker._tasks_completed == 1

    @pytest.mark.asyncio
    async def test_process_task_failure_no_retries(self, worker, mock_broker):
        """A failing handler with no retries left is sent to the DLQ."""
        task = _make_task("failing_task", task_id="task-124", max_retries=0)

        @worker.task("failing_task")
        async def failing_handler(task_obj: Task):
            raise ValueError("Task processing failed")

        await worker._process_task(task)

        # max_retries=0 -> no requeue; broker exposes move_to_dlq so DLQ is used.
        mock_broker.requeue.assert_not_awaited()
        mock_broker.move_to_dlq.assert_awaited_once()
        dlq_args = mock_broker.move_to_dlq.call_args[0]
        assert dlq_args[0] == task.id
        assert "Task processing failed" in dlq_args[1]
        assert worker._tasks_failed == 1

    @pytest.mark.asyncio
    async def test_process_task_failure_acknowledges_when_no_dlq(self, mock_broker):
        """Without DLQ support, a terminal failure is acknowledged as FAILURE."""
        # Broker without move_to_dlq.
        broker = AsyncMock(spec=Broker)
        worker = Worker(broker=broker, queues=["test-queue"], use_dlq=False)
        task = _make_task("failing_task", task_id="task-131", max_retries=0)

        @worker.task("failing_task")
        async def failing_handler(task_obj: Task):
            raise ValueError("boom")

        await worker._process_task(task)

        broker.acknowledge.assert_awaited_once()
        result = broker.acknowledge.call_args[0][1]
        assert result.status == TaskStatus.FAILURE
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_process_task_retries_before_giving_up(self, worker, mock_broker, monkeypatch):
        """A failure with retries remaining requeues the task."""
        # Avoid real backoff sleep.
        monkeypatch.setattr("jobqueue.worker.Worker._calculate_backoff", lambda self, a: 0.0)

        task = _make_task("retry_task", task_id="task-128", max_retries=3, retries=0)

        @worker.task("retry_task")
        async def retry_handler(task_obj: Task):
            raise Exception("Retry me")

        await worker._process_task(task)

        mock_broker.requeue.assert_awaited_once_with(task.id)
        mock_broker.acknowledge.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_process_task_timeout(self, worker, mock_broker):
        """A handler exceeding its timeout is treated as a failure."""
        # timeout_ms=100 -> 0.1s; handler sleeps 0.5s. max_retries=0 -> terminal.
        task = _make_task("slow_task", task_id="task-125", timeout_ms=100, max_retries=0)

        @worker.task("slow_task")
        async def slow_handler(task_obj: Task):
            await asyncio.sleep(0.5)
            return {"result": "too_late"}

        await worker._process_task(task)

        mock_broker.move_to_dlq.assert_awaited_once()
        dlq_args = mock_broker.move_to_dlq.call_args[0]
        assert dlq_args[0] == task.id
        assert "timeout" in dlq_args[1].lower()

    @pytest.mark.asyncio
    async def test_no_handler_registered_fails(self, worker, mock_broker):
        """A task with no registered handler is treated as a failure."""
        task = _make_task("unknown_task", task_id="task-140", max_retries=0)

        await worker._process_task(task)

        mock_broker.move_to_dlq.assert_awaited_once()
        assert "No handler" in mock_broker.move_to_dlq.call_args[0][1]

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold(self, worker, mock_broker, monkeypatch):
        """Repeated failures open the per-task circuit breaker."""
        monkeypatch.setattr("jobqueue.worker.Worker._calculate_backoff", lambda self, a: 0.0)

        @worker.task("circuit_task")
        async def circuit_handler(task_obj: Task):
            raise Exception("Simulated failure")

        # circuit_failure_threshold=3 -> after 3 failures the breaker opens.
        for _ in range(3):
            task = _make_task("circuit_task", task_id="task-126", max_retries=5)
            await worker._process_task(task)

        breaker = worker._circuit_registry.get("circuit_task")
        assert breaker is not None
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_heartbeat_loop_calls_broker(self, worker, mock_broker):
        """The heartbeat loop periodically calls broker.heartbeat."""
        worker._running = True
        worker.heartbeat_interval = 0.05

        loop_task = asyncio.create_task(worker._heartbeat_loop())
        await asyncio.sleep(0.15)
        worker._running = False
        await asyncio.wait_for(loop_task, timeout=1.0)

        assert mock_broker.heartbeat.await_count >= 1
        # No task in flight -> current_task is None.
        mock_broker.heartbeat.assert_awaited_with(worker.worker_id, None)

    @pytest.mark.asyncio
    async def test_worker_registers_on_start(self, worker, mock_broker):
        """Starting the worker registers it with the broker."""
        mock_broker.dequeue.return_value = None

        start_task = asyncio.create_task(worker.start())
        await asyncio.sleep(0.2)
        await worker.stop()
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            pass

        mock_broker.register_worker.assert_awaited_once()
        worker_info = mock_broker.register_worker.call_args[0][0]
        assert worker_info.id == worker.worker_id
        assert worker_info.queues == ["test-queue"]

    @pytest.mark.asyncio
    async def test_task_result_serialization(self, worker, mock_broker):
        """The acknowledged result carries the handler's structured output."""
        task = _make_task(
            "serialization_task",
            task_id="task-130",
            payload={"complex": {"nested": ["data", 123, True]}},
        )

        @worker.task("serialization_task")
        async def serialization_handler(task_obj: Task):
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": task_obj.payload,
                "output": {"result": [1, 2, 3], "success": True},
            }

        await worker._process_task(task)

        mock_broker.acknowledge.assert_awaited_once()
        result = mock_broker.acknowledge.call_args[0][1]
        assert isinstance(result, TaskResult)
        assert result.task_id == task.id
        assert "timestamp" in result.result
        assert "input" in result.result
        assert "output" in result.result

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, worker, mock_broker):
        """Multiple tasks can be processed concurrently via _process_task."""
        tasks = [
            _make_task("concurrent_task", task_id=f"task-{i}", payload={"index": i})
            for i in range(5)
        ]
        processed = []

        @worker.task("concurrent_task")
        async def concurrent_handler(task_obj: Task):
            processed.append(task_obj.id)
            await asyncio.sleep(0.05)
            return {"index": task_obj.payload["index"]}

        await asyncio.gather(*[worker._process_task(t) for t in tasks])

        assert set(processed) == {f"task-{i}" for i in range(5)}
        assert mock_broker.acknowledge.await_count == 5
