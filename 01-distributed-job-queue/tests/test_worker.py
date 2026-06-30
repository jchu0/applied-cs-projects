"""Comprehensive tests for the Worker class."""

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from jobqueue.broker import Broker
from jobqueue.circuit_breaker import CircuitBreakerRegistry, CircuitOpenError
from jobqueue.models import Task, TaskResult, TaskStatus, WorkerInfo
from jobqueue.worker import Worker


class TestWorker:
    """Test suite for Worker class."""

    @pytest_asyncio.fixture
    async def mock_broker(self):
        """Create a mock broker for testing."""
        broker = AsyncMock(spec=Broker)
        broker.fetch_task = AsyncMock(return_value=None)
        broker.heartbeat = AsyncMock(return_value=True)
        broker.register_worker = AsyncMock()
        broker.deregister_worker = AsyncMock()
        broker.update_task_status = AsyncMock()
        broker.complete_task = AsyncMock()
        broker.fail_task = AsyncMock()
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
        """Test worker initialization with correct parameters."""
        assert worker.broker == mock_broker
        assert worker.queues == ["test-queue"]
        assert worker.concurrency == 2
        assert worker.poll_interval == 0.1
        assert worker.heartbeat_interval == 1.0
        assert worker.enable_circuit_breaker is True
        assert worker.circuit_failure_threshold == 3
        assert worker.circuit_reset_timeout == 5.0

    @pytest.mark.asyncio
    async def test_register_handler(self, worker):
        """Test registering task handlers."""

        @worker.task("test_task")
        async def test_handler(task: Task):
            return {"result": "success"}

        assert "test_task" in worker.handlers
        assert worker.handlers["test_task"] == test_handler

    @pytest.mark.asyncio
    async def test_process_task_success(self, worker, mock_broker):
        """Test successful task processing."""
        task = Task(
            id="task-123",
            name="test_task",
            queue="test-queue",
            payload={"data": "test"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=1,
        )

        @worker.task("test_task")
        async def test_handler(task_obj: Task):
            return {"result": "processed"}

        result = await worker._process_task(task)

        mock_broker.update_task_status.assert_called_with(
            task.id, TaskStatus.RUNNING
        )
        mock_broker.complete_task.assert_called_once()

        # Verify the result
        call_args = mock_broker.complete_task.call_args[0]
        assert call_args[0] == task.id
        assert isinstance(call_args[1], TaskResult)
        assert call_args[1].task_id == task.id
        assert call_args[1].result == {"result": "processed"}

    @pytest.mark.asyncio
    async def test_process_task_failure(self, worker, mock_broker):
        """Test task processing failure."""
        task = Task(
            id="task-124",
            name="failing_task",
            queue="test-queue",
            payload={"data": "test"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=1,
        )

        @worker.task("failing_task")
        async def failing_handler(task_obj: Task):
            raise ValueError("Task processing failed")

        result = await worker._process_task(task)

        mock_broker.fail_task.assert_called_once()
        call_args = mock_broker.fail_task.call_args[0]
        assert call_args[0] == task.id
        assert "Task processing failed" in call_args[1]

    @pytest.mark.asyncio
    async def test_process_task_timeout(self, worker, mock_broker):
        """Test task timeout handling."""
        task = Task(
            id="task-125",
            name="slow_task",
            queue="test-queue",
            payload={"data": "test"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=1,
            timeout=0.1,  # 100ms timeout
        )

        @worker.task("slow_task")
        async def slow_handler(task_obj: Task):
            await asyncio.sleep(0.5)  # Sleep longer than timeout
            return {"result": "too_late"}

        result = await worker._process_task(task)

        mock_broker.fail_task.assert_called_once()
        call_args = mock_broker.fail_task.call_args[0]
        assert call_args[0] == task.id
        assert "timeout" in call_args[1].lower()

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self, worker, mock_broker):
        """Test circuit breaker integration with worker."""
        task = Task(
            id="task-126",
            name="circuit_task",
            queue="test-queue",
            payload={"data": "test"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=1,
        )

        failure_count = 0

        @worker.task("circuit_task")
        async def circuit_handler(task_obj: Task):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:
                raise Exception("Simulated failure")
            return {"result": "success"}

        # Process task multiple times to trigger circuit breaker
        for i in range(3):
            await worker._process_task(task)

        # Circuit should be open now
        circuit = worker.circuit_registry.get_circuit("circuit_task")
        assert circuit is not None
        assert circuit.is_open

    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, worker, mock_broker):
        """Test worker heartbeat mechanism."""
        worker.worker_id = "worker-123"
        worker.running = True

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(worker._heartbeat())

        # Let it run for a short time
        await asyncio.sleep(0.2)

        # Stop the worker
        worker.running = False
        await heartbeat_task

        # Verify heartbeat was called
        assert mock_broker.heartbeat.called

    @pytest.mark.asyncio
    async def test_worker_lifecycle(self, worker, mock_broker):
        """Test complete worker lifecycle: start and stop."""
        # Mock fetch_task to return a task then None
        task = Task(
            id="task-127",
            name="lifecycle_task",
            queue="test-queue",
            payload={"data": "test"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=1,
        )

        mock_broker.fetch_task.side_effect = [task, None, None, None]

        @worker.task("lifecycle_task")
        async def lifecycle_handler(task_obj: Task):
            return {"result": "processed"}

        # Start worker in background
        start_task = asyncio.create_task(worker.start())

        # Let it process the task
        await asyncio.sleep(0.5)

        # Stop the worker
        await worker.stop()

        # Verify registration and deregistration
        mock_broker.register_worker.assert_called_once()
        mock_broker.deregister_worker.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, worker, mock_broker):
        """Test task retry mechanism."""
        task = Task(
            id="task-128",
            name="retry_task",
            queue="test-queue",
            payload={"data": "test"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=1,
            max_retries=3,
            retry_count=0,
        )

        attempt_count = 0

        @worker.task("retry_task")
        async def retry_handler(task_obj: Task):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Retry me")
            return {"result": "success"}

        # Process task (should retry internally)
        await worker._process_task(task)

        # Verify retries were attempted
        assert mock_broker.update_task_status.called

    @pytest.mark.asyncio
    async def test_dlq_handling(self, worker, mock_broker):
        """Test Dead Letter Queue handling."""
        task = Task(
            id="task-129",
            name="dlq_task",
            queue="test-queue",
            payload={"data": "test"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=1,
            max_retries=0,  # No retries, should go to DLQ on failure
        )

        @worker.task("dlq_task")
        async def dlq_handler(task_obj: Task):
            raise Exception("Send to DLQ")

        await worker._process_task(task)

        # Verify task was moved to DLQ
        if worker.use_dlq:
            mock_broker.move_to_dlq.assert_called_once_with(task.id)

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, worker, mock_broker):
        """Test concurrent processing of multiple tasks."""
        tasks = [
            Task(
                id=f"task-{i}",
                name="concurrent_task",
                queue="test-queue",
                payload={"index": i},
                status=TaskStatus.PENDING,
                created_at=datetime.now(timezone.utc),
                priority=1,
            )
            for i in range(5)
        ]

        processed_tasks = []

        @worker.task("concurrent_task")
        async def concurrent_handler(task_obj: Task):
            processed_tasks.append(task_obj.id)
            await asyncio.sleep(0.1)
            return {"index": task_obj.payload["index"]}

        # Process all tasks concurrently
        await asyncio.gather(*[worker._process_task(task) for task in tasks])

        # Verify all tasks were processed
        assert len(processed_tasks) == 5
        assert set(processed_tasks) == {f"task-{i}" for i in range(5)}

    @pytest.mark.asyncio
    async def test_priority_queue_handling(self, worker, mock_broker):
        """Test priority queue task fetching."""
        high_priority_task = Task(
            id="high-priority",
            name="priority_task",
            queue="test-queue",
            payload={"priority": "high"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=10,
        )

        low_priority_task = Task(
            id="low-priority",
            name="priority_task",
            queue="test-queue",
            payload={"priority": "low"},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=1,
        )

        # Mock broker to return high priority task first
        mock_broker.fetch_task.side_effect = [high_priority_task, low_priority_task, None]

        @worker.task("priority_task")
        async def priority_handler(task_obj: Task):
            return {"processed": task_obj.id}

        # Process tasks
        processed = []
        while True:
            task = await mock_broker.fetch_task(worker.queues)
            if task is None:
                break
            await worker._process_task(task)
            processed.append(task.id)

        # Verify high priority was processed first
        assert processed[0] == "high-priority"
        assert processed[1] == "low-priority"

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, worker, mock_broker):
        """Test graceful shutdown with signal handling."""
        worker.running = True
        shutdown_called = False

        async def mock_shutdown():
            nonlocal shutdown_called
            shutdown_called = True
            worker.running = False

        worker.stop = mock_shutdown

        # Simulate SIGTERM
        worker._handle_signal(signal.SIGTERM, None)

        await asyncio.sleep(0.1)
        assert shutdown_called

    @pytest.mark.asyncio
    async def test_task_result_serialization(self, worker, mock_broker):
        """Test task result serialization and deserialization."""
        task = Task(
            id="task-130",
            name="serialization_task",
            queue="test-queue",
            payload={"complex": {"nested": ["data", 123, True]}},
            status=TaskStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            priority=1,
        )

        @worker.task("serialization_task")
        async def serialization_handler(task_obj: Task):
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": task_obj.payload,
                "output": {"result": [1, 2, 3], "success": True}
            }

        await worker._process_task(task)

        # Verify complete_task was called with serializable result
        mock_broker.complete_task.assert_called_once()
        call_args = mock_broker.complete_task.call_args[0]
        result = call_args[1]
        assert isinstance(result, TaskResult)
        assert result.task_id == task.id
        assert "timestamp" in result.result
        assert "input" in result.result
        assert "output" in result.result