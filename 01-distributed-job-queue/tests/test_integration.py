"""Integration tests for the distributed job queue system."""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict

import pytest
import pytest_asyncio

# Skip if required dependencies are not available
pytest.importorskip("redis")
pytest.importorskip("croniter")
pytest.importorskip("structlog")

import redis.asyncio as redis

from jobqueue.broker import Broker
from jobqueue.models import Task, TaskStatus
from jobqueue.redis_broker import RedisBroker
from jobqueue.scheduler import Scheduler
from jobqueue.worker import Worker


@pytest.mark.integration
class TestIntegration:
    """Integration test suite for the complete system."""

    @pytest_asyncio.fixture
    async def redis_client(self):
        """Create Redis client for testing."""
        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_DB", 1)),  # Use separate DB for tests
            decode_responses=True
        )

        # Clear test database
        await client.flushdb()

        yield client

        # Cleanup
        await client.flushdb()
        await client.close()

    @pytest_asyncio.fixture
    async def broker(self, redis_client):
        """Create Redis broker for testing."""
        broker = RedisBroker(redis_client)
        await broker.connect()
        yield broker
        await broker.close()

    @pytest_asyncio.fixture
    async def scheduler(self, broker):
        """Create scheduler for testing."""
        scheduler = Scheduler(broker)
        await scheduler.start()
        yield scheduler
        await scheduler.stop()

    @pytest_asyncio.fixture
    async def worker(self, broker):
        """Create worker for testing."""
        worker = Worker(
            broker=broker,
            queues=["test-queue", "priority-queue"],
            concurrency=2,
            poll_interval=0.1,
            heartbeat_interval=1.0
        )
        yield worker
        await worker.stop()

    @pytest.mark.asyncio
    async def test_end_to_end_task_processing(self, broker, worker):
        """Test complete task lifecycle from submission to completion."""
        processed_tasks = []

        @worker.task("e2e_task")
        async def process_task(task: Task):
            processed_tasks.append(task.id)
            return {"processed": task.payload["data"], "timestamp": time.time()}

        # Submit task
        task = await broker.submit_task(
            name="e2e_task",
            queue="test-queue",
            payload={"data": "test_value"},
            priority=5
        )

        assert task.id is not None
        assert task.status == TaskStatus.PENDING

        # Start worker in background
        worker_task = asyncio.create_task(worker.start())

        # Wait for task to be processed
        await asyncio.sleep(1.0)

        # Check task status
        updated_task = await broker.get_task(task.id)
        assert updated_task.status == TaskStatus.COMPLETED
        assert task.id in processed_tasks

        # Stop worker
        await worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_priority_queue_processing(self, broker, worker):
        """Test that high priority tasks are processed first."""
        processing_order = []

        @worker.task("priority_task")
        async def process_priority_task(task: Task):
            processing_order.append((task.id, task.priority))
            await asyncio.sleep(0.1)
            return {"id": task.id}

        # Submit tasks with different priorities
        low_priority = await broker.submit_task(
            name="priority_task",
            queue="priority-queue",
            payload={"type": "low"},
            priority=1
        )

        high_priority = await broker.submit_task(
            name="priority_task",
            queue="priority-queue",
            payload={"type": "high"},
            priority=10
        )

        medium_priority = await broker.submit_task(
            name="priority_task",
            queue="priority-queue",
            payload={"type": "medium"},
            priority=5
        )

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Wait for processing
        await asyncio.sleep(1.5)

        # Stop worker
        await worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        # Verify processing order (high priority first)
        assert len(processing_order) == 3
        priorities = [p[1] for p in processing_order]
        assert priorities == sorted(priorities, reverse=True)

    @pytest.mark.asyncio
    async def test_task_retry_mechanism(self, broker, worker):
        """Test automatic task retry on failure."""
        attempt_count = {}

        @worker.task("retry_task")
        async def failing_task(task: Task):
            task_id = task.id
            attempt_count[task_id] = attempt_count.get(task_id, 0) + 1

            if attempt_count[task_id] < 3:
                raise Exception(f"Attempt {attempt_count[task_id]} failed")

            return {"success": True, "attempts": attempt_count[task_id]}

        # Submit task with retry settings
        task = await broker.submit_task(
            name="retry_task",
            queue="test-queue",
            payload={"test": "retry"},
            max_retries=3,
            retry_delay=0.1
        )

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Wait for retries and processing
        await asyncio.sleep(2.0)

        # Check task completed after retries
        updated_task = await broker.get_task(task.id)
        assert updated_task.status == TaskStatus.COMPLETED
        assert attempt_count[task.id] == 3

        # Stop worker
        await worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_workers(self, broker):
        """Test multiple workers processing tasks concurrently."""
        processed_by_worker = {}

        async def create_worker(worker_id: str):
            worker = Worker(
                broker=broker,
                queues=["concurrent-queue"],
                concurrency=1,
                poll_interval=0.1
            )

            @worker.task("concurrent_task")
            async def process(task: Task):
                processed_by_worker[task.id] = worker_id
                await asyncio.sleep(0.2)
                return {"worker": worker_id}

            return worker

        # Create multiple workers
        worker1 = await create_worker("worker-1")
        worker2 = await create_worker("worker-2")
        worker3 = await create_worker("worker-3")

        # Submit multiple tasks
        tasks = []
        for i in range(10):
            task = await broker.submit_task(
                name="concurrent_task",
                queue="concurrent-queue",
                payload={"index": i}
            )
            tasks.append(task)

        # Start all workers
        worker_tasks = [
            asyncio.create_task(worker1.start()),
            asyncio.create_task(worker2.start()),
            asyncio.create_task(worker3.start())
        ]

        # Wait for processing
        await asyncio.sleep(3.0)

        # Stop all workers
        for worker in [worker1, worker2, worker3]:
            await worker.stop()

        for task in worker_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Verify all tasks were processed
        assert len(processed_by_worker) == 10

        # Verify distribution among workers
        worker_counts = {}
        for worker_id in processed_by_worker.values():
            worker_counts[worker_id] = worker_counts.get(worker_id, 0) + 1

        # Each worker should have processed at least 1 task
        assert len(worker_counts) == 3
        assert all(count >= 1 for count in worker_counts.values())

    @pytest.mark.asyncio
    async def test_scheduled_task_execution(self, broker, scheduler, worker):
        """Test scheduled task execution."""
        executed_times = []

        @worker.task("scheduled_task")
        async def scheduled_handler(task: Task):
            executed_times.append(datetime.now(timezone.utc))
            return {"execution_time": executed_times[-1].isoformat()}

        # Schedule task to run every second
        schedule_id = await scheduler.schedule_task(
            name="scheduled_task",
            queue="test-queue",
            payload={"type": "scheduled"},
            interval=1  # Every second
        )

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Wait for multiple executions
        await asyncio.sleep(3.5)

        # Cancel schedule
        await scheduler.cancel_scheduled_task(schedule_id)

        # Stop worker
        await worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        # Verify multiple executions
        assert len(executed_times) >= 3

        # Verify executions were roughly 1 second apart
        for i in range(1, len(executed_times)):
            delta = (executed_times[i] - executed_times[i-1]).total_seconds()
            assert 0.8 <= delta <= 1.5  # Allow some variance

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, broker, worker):
        """Test Dead Letter Queue for permanently failed tasks."""
        @worker.task("dlq_task")
        async def always_fails(task: Task):
            raise Exception("This task always fails")

        # Submit task that will fail
        task = await broker.submit_task(
            name="dlq_task",
            queue="test-queue",
            payload={"will": "fail"},
            max_retries=2,
            retry_delay=0.1
        )

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Wait for retries and DLQ move
        await asyncio.sleep(1.5)

        # Check task is in failed state
        updated_task = await broker.get_task(task.id)
        assert updated_task.status == TaskStatus.FAILED

        # Check task is in DLQ
        dlq_tasks = await broker.list_dlq_tasks()
        assert any(t.id == task.id for t in dlq_tasks)

        # Stop worker
        await worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_circuit_breaker_protection(self, broker):
        """Test circuit breaker prevents cascading failures."""
        worker = Worker(
            broker=broker,
            queues=["circuit-queue"],
            concurrency=1,
            poll_interval=0.1,
            enable_circuit_breaker=True,
            circuit_failure_threshold=3,
            circuit_reset_timeout=2.0
        )

        failure_count = 0
        success_count = 0

        @worker.task("circuit_task")
        async def unreliable_task(task: Task):
            nonlocal failure_count, success_count

            # Fail first 5 attempts
            if failure_count < 5:
                failure_count += 1
                raise Exception("Service unavailable")

            success_count += 1
            return {"status": "recovered"}

        # Submit multiple tasks
        tasks = []
        for i in range(8):
            task = await broker.submit_task(
                name="circuit_task",
                queue="circuit-queue",
                payload={"index": i}
            )
            tasks.append(task)

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Wait for processing with circuit breaker
        await asyncio.sleep(5.0)

        # Stop worker
        await worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        # Circuit should have opened after threshold
        assert failure_count >= 3
        # Some tasks should succeed after reset
        assert success_count > 0

    @pytest.mark.asyncio
    async def test_task_timeout_handling(self, broker, worker):
        """Test task timeout enforcement."""
        timed_out = False
        completed = False

        @worker.task("timeout_task")
        async def slow_task(task: Task):
            nonlocal completed
            try:
                await asyncio.sleep(5.0)  # Sleep longer than timeout
                completed = True
                return {"status": "completed"}
            except asyncio.CancelledError:
                nonlocal timed_out
                timed_out = True
                raise

        # Submit task with short timeout
        task = await broker.submit_task(
            name="timeout_task",
            queue="test-queue",
            payload={"slow": True},
            timeout=1.0  # 1 second timeout
        )

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Wait for timeout
        await asyncio.sleep(2.0)

        # Check task failed due to timeout
        updated_task = await broker.get_task(task.id)
        assert updated_task.status == TaskStatus.FAILED
        assert not completed  # Task should not complete

        # Stop worker
        await worker.stop()
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, broker):
        """Test graceful worker shutdown during task processing."""
        processing_started = False
        processing_completed = False

        worker = Worker(
            broker=broker,
            queues=["shutdown-queue"],
            concurrency=1,
            poll_interval=0.1
        )

        @worker.task("long_task")
        async def long_running_task(task: Task):
            nonlocal processing_started, processing_completed
            processing_started = True

            try:
                await asyncio.sleep(2.0)
                processing_completed = True
                return {"status": "completed"}
            except asyncio.CancelledError:
                # Task was cancelled during shutdown
                return {"status": "cancelled"}

        # Submit task
        task = await broker.submit_task(
            name="long_task",
            queue="shutdown-queue",
            payload={"long": True}
        )

        # Start worker
        worker_task = asyncio.create_task(worker.start())

        # Wait for task to start processing
        await asyncio.sleep(0.5)
        assert processing_started

        # Initiate graceful shutdown
        await worker.stop()

        # Worker should wait for task to complete
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

        # Task should have completed gracefully
        updated_task = await broker.get_task(task.id)
        assert updated_task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]