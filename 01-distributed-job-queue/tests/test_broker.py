"""Tests for broker implementations."""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta

from jobqueue.broker import InMemoryBroker
from jobqueue.models import Task, TaskResult, TaskStatus, TaskPriority


@pytest.fixture
def broker():
    """Create a fresh broker for each test."""
    return InMemoryBroker()


class TestInMemoryBroker:
    """Tests for InMemoryBroker."""

    async def test_enqueue_dequeue(self, broker):
        """Test basic enqueue and dequeue."""
        task = Task(name="test", payload={"key": "value"})

        # Enqueue
        enqueued = await broker.enqueue(task)
        assert enqueued.id == task.id
        assert enqueued.status == TaskStatus.QUEUED

        # Dequeue
        dequeued = await broker.dequeue(["default"])
        assert dequeued is not None
        assert dequeued.id == task.id
        assert dequeued.status == TaskStatus.RUNNING

    async def test_dequeue_empty_queue(self, broker):
        """Test dequeue from empty queue."""
        task = await broker.dequeue(["default"])
        assert task is None

    async def test_dequeue_with_timeout(self, broker):
        """Test dequeue with timeout."""
        # Start dequeue with timeout
        start = asyncio.get_event_loop().time()
        task = await broker.dequeue(["default"], timeout=0.1)
        elapsed = asyncio.get_event_loop().time() - start

        assert task is None
        assert elapsed >= 0.1

    async def test_priority_ordering(self, broker):
        """Test that tasks are dequeued by priority."""
        # Enqueue tasks with different priorities
        low = await broker.enqueue(Task(name="low", priority=TaskPriority.LOW))
        high = await broker.enqueue(Task(name="high", priority=TaskPriority.HIGH))
        normal = await broker.enqueue(Task(name="normal", priority=TaskPriority.NORMAL))

        # Dequeue should return highest priority first
        task1 = await broker.dequeue(["default"])
        task2 = await broker.dequeue(["default"])
        task3 = await broker.dequeue(["default"])

        assert task1.id == high.id
        assert task2.id == normal.id
        assert task3.id == low.id

    async def test_multiple_queues(self, broker):
        """Test tasks in different queues."""
        task1 = await broker.enqueue(Task(name="t1", queue="queue1"))
        task2 = await broker.enqueue(Task(name="t2", queue="queue2"))

        # Dequeue from queue1 only
        dequeued = await broker.dequeue(["queue1"])
        assert dequeued.id == task1.id

        # Queue2 task still available
        dequeued = await broker.dequeue(["queue2"])
        assert dequeued.id == task2.id

    async def test_acknowledge_success(self, broker):
        """Test acknowledging a successful task."""
        task = await broker.enqueue(Task(name="test"))
        dequeued = await broker.dequeue(["default"])

        result = TaskResult(
            task_id=task.id,
            status=TaskStatus.SUCCESS,
            result="done",
        )
        await broker.acknowledge(task.id, result)

        # Check task status updated
        updated = await broker.get_task(task.id)
        assert updated.status == TaskStatus.SUCCESS

        # Check result stored
        stored_result = await broker.get_result(task.id)
        assert stored_result.status == TaskStatus.SUCCESS
        assert stored_result.result == "done"

    async def test_requeue(self, broker):
        """Test requeuing a task."""
        task = await broker.enqueue(Task(name="test"))
        dequeued = await broker.dequeue(["default"])

        # Requeue
        await broker.requeue(task.id)

        # Check task status
        updated = await broker.get_task(task.id)
        assert updated.status == TaskStatus.QUEUED
        assert updated.retries == 1

        # Should be able to dequeue again
        dequeued2 = await broker.dequeue(["default"])
        assert dequeued2.id == task.id

    async def test_idempotency(self, broker):
        """Test task deduplication via idempotency key."""
        task1 = await broker.enqueue(Task(
            name="test",
            idempotency_key="unique-key"
        ))

        # Enqueue with same key returns existing task
        task2 = await broker.enqueue(Task(
            name="test",
            idempotency_key="unique-key"
        ))

        assert task1.id == task2.id

    async def test_cancel_task(self, broker):
        """Test cancelling a pending task."""
        task = await broker.enqueue(Task(name="test"))

        # Cancel
        cancelled = await broker.cancel_task(task.id)
        assert cancelled is True

        # Check status
        updated = await broker.get_task(task.id)
        assert updated.status == TaskStatus.CANCELLED

        # Cannot dequeue cancelled task
        dequeued = await broker.dequeue(["default"])
        assert dequeued is None

    async def test_eta_scheduling(self, broker):
        """Test task with future ETA."""
        future_time = datetime.now(timezone.utc) + timedelta(seconds=10)
        task = await broker.enqueue(Task(
            name="scheduled",
            eta=future_time
        ))

        # Should not be dequeued yet
        dequeued = await broker.dequeue(["default"])
        assert dequeued is None

    async def test_queue_stats(self, broker):
        """Test queue statistics."""
        # Enqueue some tasks
        await broker.enqueue(Task(name="t1"))
        await broker.enqueue(Task(name="t2"))
        task3 = await broker.enqueue(Task(name="t3"))

        # Process one
        await broker.dequeue(["default"])

        # Cancel one
        await broker.cancel_task(task3.id)

        stats = await broker.get_queue_stats("default")
        assert stats.pending == 1
        assert stats.running == 1
        assert stats.failed == 1  # cancelled counts as failed

    async def test_get_task(self, broker):
        """Test getting task by ID."""
        task = await broker.enqueue(Task(name="test"))

        fetched = await broker.get_task(task.id)
        assert fetched is not None
        assert fetched.id == task.id
        assert fetched.name == "test"

    async def test_get_nonexistent_task(self, broker):
        """Test getting nonexistent task."""
        task = await broker.get_task("nonexistent-id")
        assert task is None


class TestBrokerConcurrency:
    """Tests for concurrent broker operations."""

    async def test_concurrent_enqueue(self, broker):
        """Test concurrent task enqueueing."""
        async def enqueue_task(i):
            return await broker.enqueue(Task(name=f"task-{i}"))

        # Enqueue 100 tasks concurrently
        tasks = await asyncio.gather(*[enqueue_task(i) for i in range(100)])

        assert len(tasks) == 100
        assert len(set(t.id for t in tasks)) == 100  # All unique IDs

    async def test_concurrent_dequeue(self, broker):
        """Test concurrent task dequeueing."""
        # Enqueue tasks
        for i in range(10):
            await broker.enqueue(Task(name=f"task-{i}"))

        # Dequeue concurrently
        async def dequeue():
            return await broker.dequeue(["default"])

        results = await asyncio.gather(*[dequeue() for _ in range(10)])
        dequeued = [r for r in results if r is not None]

        # Each task should only be dequeued once
        assert len(dequeued) == 10
        assert len(set(t.id for t in dequeued)) == 10
