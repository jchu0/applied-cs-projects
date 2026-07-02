"""Integration tests for the distributed job queue system.

These wire together the real components -- :class:`InMemoryBroker`,
:class:`Worker`, and :class:`Scheduler` -- and drive full task lifecycles
without any external infrastructure (no Redis required). They exercise the
actual public API: ``broker.enqueue`` / ``broker.get_task``, the worker's
``start`` / ``stop`` loop and ``@worker.task`` handlers, retries, timeouts,
the circuit breaker, and scheduler job triggering.
"""

import asyncio
from datetime import datetime, timezone

import pytest
import pytest_asyncio

# Skip if required dependencies are not available
pytest.importorskip("croniter")
pytest.importorskip("structlog")

from jobqueue.broker import InMemoryBroker
from jobqueue.models import Task, TaskPriority, TaskStatus
from jobqueue.scheduler import Scheduler
from jobqueue.worker import Worker


async def _run_worker_briefly(worker: Worker, duration: float) -> None:
    """Start a worker, let it run for ``duration`` seconds, then stop it."""
    worker_task = asyncio.create_task(worker.start())
    await asyncio.sleep(duration)
    await worker.stop()
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


@pytest.mark.integration
class TestIntegration:
    """Integration test suite for the complete system."""

    @pytest_asyncio.fixture
    async def broker(self):
        """Create an in-memory broker for testing."""
        return InMemoryBroker(visibility_timeout_ms=30000)

    @pytest_asyncio.fixture
    async def worker(self, broker):
        """Create a worker bound to the broker."""
        worker = Worker(
            broker=broker,
            queues=["test-queue", "priority-queue"],
            concurrency=2,
            poll_interval=0.05,
            heartbeat_interval=1.0,
        )
        yield worker
        await worker.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_task_processing(self, broker, worker):
        """A submitted task is picked up, processed, and marked SUCCESS."""
        processed_tasks = []

        @worker.task("e2e_task")
        async def process_task(task: Task):
            processed_tasks.append(task.id)
            return {"processed": task.payload["data"]}

        task = await broker.enqueue(
            Task(name="e2e_task", queue="test-queue", payload={"data": "test_value"})
        )
        assert task.status == TaskStatus.QUEUED

        await _run_worker_briefly(worker, 0.5)

        updated = await broker.get_task(task.id)
        assert updated.status == TaskStatus.SUCCESS
        assert task.id in processed_tasks

        result = await broker.get_result(task.id)
        assert result is not None
        assert result.result == {"processed": "test_value"}

    @pytest.mark.asyncio
    async def test_priority_queue_processing(self, broker):
        """Higher-priority tasks (lower numeric value) are processed first."""
        processing_order = []

        # Single slot so ordering is deterministic.
        worker = Worker(
            broker=broker,
            queues=["priority-queue"],
            concurrency=1,
            poll_interval=0.02,
        )

        @worker.task("priority_task")
        async def process_priority_task(task: Task):
            processing_order.append(task.priority)
            await asyncio.sleep(0.02)
            return {"id": task.id}

        # Lower value == higher priority (CRITICAL=0 .. LOW=3).
        for prio in (TaskPriority.LOW, TaskPriority.CRITICAL, TaskPriority.NORMAL):
            await broker.enqueue(
                Task(name="priority_task", queue="priority-queue", priority=prio)
            )

        await _run_worker_briefly(worker, 0.6)

        assert len(processing_order) == 3
        # Processed in ascending priority-value order (highest priority first).
        assert processing_order == sorted(processing_order)

    @pytest.mark.asyncio
    async def test_task_retry_mechanism(self, broker):
        """A task that fails then succeeds completes after being requeued."""
        attempts = {"count": 0}

        worker = Worker(
            broker=broker,
            queues=["test-queue"],
            concurrency=1,
            poll_interval=0.02,
        )
        # Keep backoff negligible so retries happen quickly.
        worker._calculate_backoff = lambda attempt, base_delay=1.0: 0.0

        @worker.task("retry_task")
        async def failing_task(task: Task):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise Exception(f"Attempt {attempts['count']} failed")
            return {"success": True, "attempts": attempts["count"]}

        task = await broker.enqueue(
            Task(name="retry_task", queue="test-queue", max_retries=3)
        )

        await _run_worker_briefly(worker, 1.0)

        updated = await broker.get_task(task.id)
        assert updated.status == TaskStatus.SUCCESS
        assert attempts["count"] == 3

    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, broker):
        """A handler exceeding its timeout is failed, never completing."""
        completed = {"value": False}

        worker = Worker(
            broker=broker,
            queues=["test-queue"],
            concurrency=1,
            poll_interval=0.02,
        )
        worker._calculate_backoff = lambda attempt, base_delay=1.0: 0.0

        @worker.task("timeout_task")
        async def slow_task(task: Task):
            await asyncio.sleep(5.0)
            completed["value"] = True
            return {"status": "completed"}

        # 100ms timeout, no retries -> terminal failure.
        task = await broker.enqueue(
            Task(name="timeout_task", queue="test-queue", timeout_ms=100, max_retries=0)
        )

        await _run_worker_briefly(worker, 0.6)

        updated = await broker.get_task(task.id)
        assert updated.status == TaskStatus.FAILURE
        assert completed["value"] is False

    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self, broker):
        """The circuit breaker opens after repeated handler failures.

        Uses ``max_retries=0`` so each failure is terminal (no requeue), which
        keeps the test deterministic while still driving the per-task-type
        breaker past its failure threshold.
        """
        worker = Worker(
            broker=broker,
            queues=["circuit-queue"],
            concurrency=1,
            poll_interval=0.02,
            enable_circuit_breaker=True,
            circuit_failure_threshold=3,
            circuit_reset_timeout=10.0,
            use_dlq=False,
        )

        @worker.task("circuit_task")
        async def unreliable_task(task: Task):
            raise Exception("Service unavailable")

        # Enqueue exactly the threshold count so every task is consumed as a
        # terminal failure and none are left to hit (and requeue against) the
        # open circuit -- keeping the polling loop from spinning.
        for _ in range(3):
            await broker.enqueue(
                Task(name="circuit_task", queue="circuit-queue", max_retries=0)
            )

        await _run_worker_briefly(worker, 0.4)

        breaker = worker._circuit_registry.get("circuit_task")
        assert breaker is not None
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_worker_cancel_task(self, broker, worker):
        """A queued task can be cancelled through the broker."""
        task = await broker.enqueue(
            Task(name="cancel_me", queue="test-queue")
        )
        cancelled = await broker.cancel_task(task.id)
        assert cancelled is True

        updated = await broker.get_task(task.id)
        assert updated.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_scheduled_job_triggers_task(self, broker):
        """A scheduler job enqueues a task that a worker then processes."""
        scheduler = Scheduler(broker)
        scheduler.add_job(
            name="periodic",
            task_name="scheduled_task",
            interval_seconds=1,
            queue="test-queue",
            payload={"type": "scheduled"},
        )

        executed = []

        worker = Worker(
            broker=broker,
            queues=["test-queue"],
            concurrency=1,
            poll_interval=0.02,
        )

        @worker.task("scheduled_task")
        async def handler(task: Task):
            executed.append(task.id)
            return {"ok": True}

        # Manually trigger the job (deterministic, no wall-clock waiting).
        triggered = await scheduler.run_once("periodic")
        assert triggered is not None
        assert triggered.metadata["scheduled_job"] == "periodic"

        await _run_worker_briefly(worker, 0.4)

        assert triggered.id in executed
        updated = await broker.get_task(triggered.id)
        assert updated.status == TaskStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_concurrent_workers(self, broker):
        """Multiple workers drain a shared queue without dropping tasks."""
        processed = {}

        def make_worker(worker_id: str) -> Worker:
            w = Worker(
                broker=broker,
                queues=["concurrent-queue"],
                concurrency=1,
                poll_interval=0.02,
            )

            @w.task("concurrent_task")
            async def process(task: Task):
                processed[task.id] = worker_id
                await asyncio.sleep(0.05)
                return {"worker": worker_id}

            return w

        workers = [make_worker(f"worker-{i}") for i in range(3)]

        for i in range(9):
            await broker.enqueue(
                Task(name="concurrent_task", queue="concurrent-queue", payload={"index": i})
            )

        worker_tasks = [asyncio.create_task(w.start()) for w in workers]
        await asyncio.sleep(1.0)
        for w in workers:
            await w.stop()
        for t in worker_tasks:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        assert len(processed) == 9
