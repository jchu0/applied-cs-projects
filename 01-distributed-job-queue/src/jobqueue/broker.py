"""Broker implementation for managing task queues."""

from __future__ import annotations

import asyncio
import heapq
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import structlog

from jobqueue.models import (
    Task,
    TaskCreate,
    TaskResult,
    TaskStatus,
    TaskPriority,
    QueueStats,
    WorkerInfo,
)

logger = structlog.get_logger()


class Broker(ABC):
    """Abstract base class for task brokers."""

    @abstractmethod
    async def enqueue(self, task: Task) -> Task:
        """Add a task to the queue."""
        pass

    @abstractmethod
    async def dequeue(self, queues: list[str], timeout: float = 0) -> Task | None:
        """Get the next task from the specified queues."""
        pass

    @abstractmethod
    async def acknowledge(self, task_id: str, result: TaskResult) -> None:
        """Acknowledge task completion."""
        pass

    @abstractmethod
    async def requeue(self, task_id: str) -> None:
        """Requeue a task for retry."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        pass

    @abstractmethod
    async def get_result(self, task_id: str) -> TaskResult | None:
        """Get result for a task."""
        pass

    @abstractmethod
    async def get_queue_stats(self, queue: str) -> QueueStats:
        """Get statistics for a queue."""
        pass

    @abstractmethod
    async def register_worker(self, worker_info: WorkerInfo) -> None:
        """Register a worker with the broker."""
        pass

    @abstractmethod
    async def heartbeat(self, worker_id: str, current_task: str | None = None) -> None:
        """Update worker heartbeat."""
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        pass


class InMemoryBroker(Broker):
    """In-memory broker implementation for development and testing."""

    def __init__(self, visibility_timeout_ms: int = 30000):
        # Priority queues: (priority, timestamp, task_id)
        self._queues: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
        self._tasks: dict[str, Task] = {}
        self._results: dict[str, TaskResult] = {}
        self._workers: dict[str, WorkerInfo] = {}

        # Visibility timeout for tasks being processed
        self._visibility_timeout_ms = visibility_timeout_ms
        self._visibility_locks: dict[str, tuple[str, float]] = {}  # task_id -> (worker_id, expires_at)

        # Deduplication
        self._idempotency_keys: dict[str, str] = {}  # key -> task_id

        # Locks for thread safety
        self._queue_lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()

        # Event for notifying waiting workers
        self._task_available = asyncio.Event()

        logger.info("InMemoryBroker initialized", visibility_timeout_ms=visibility_timeout_ms)

    async def enqueue(self, task: Task) -> Task:
        """Add a task to the queue."""
        async with self._task_lock:
            # Check for duplicate via idempotency key
            if task.idempotency_key:
                if task.idempotency_key in self._idempotency_keys:
                    existing_id = self._idempotency_keys[task.idempotency_key]
                    existing_task = self._tasks.get(existing_id)
                    if existing_task:
                        logger.info(
                            "Duplicate task detected",
                            task_id=task.id,
                            existing_id=existing_id,
                            idempotency_key=task.idempotency_key,
                        )
                        return existing_task
                self._idempotency_keys[task.idempotency_key] = task.id

            # Store task
            task.status = TaskStatus.QUEUED
            self._tasks[task.id] = task

        # Add to priority queue
        async with self._queue_lock:
            # Priority queue entry: (priority, timestamp, task_id)
            # Lower priority value = higher priority
            entry = (task.priority, time.time(), task.id)
            heapq.heappush(self._queues[task.queue], entry)

            # Signal waiting workers
            self._task_available.set()

        logger.info(
            "Task enqueued",
            task_id=task.id,
            queue=task.queue,
            priority=task.priority,
            name=task.name,
        )
        return task

    async def dequeue(self, queues: list[str], timeout: float = 0) -> Task | None:
        """Get the next task from the specified queues (priority-based)."""
        start_time = time.time()

        while True:
            async with self._queue_lock:
                # Check each queue in order, respecting priority within each queue
                best_task: Task | None = None
                best_priority = (float('inf'), float('inf'))
                best_queue: str | None = None

                for queue_name in queues:
                    queue = self._queues.get(queue_name, [])

                    # Find highest priority task that's ready
                    while queue:
                        priority, timestamp, task_id = queue[0]

                        # Check if task still exists and is valid
                        task = self._tasks.get(task_id)
                        if not task or task.status != TaskStatus.QUEUED:
                            heapq.heappop(queue)
                            continue

                        # Check ETA
                        if task.eta and task.eta > datetime.now(timezone.utc):
                            break  # Task not ready yet

                        # Check visibility lock
                        if task_id in self._visibility_locks:
                            _, expires_at = self._visibility_locks[task_id]
                            if time.time() < expires_at:
                                break  # Still locked
                            # Lock expired, remove it
                            del self._visibility_locks[task_id]

                        # Compare with current best
                        if (priority, timestamp) < best_priority:
                            best_priority = (priority, timestamp)
                            best_task = task
                            best_queue = queue_name
                        break

                if best_task and best_queue:
                    # Remove from queue
                    queue = self._queues[best_queue]
                    # Find and remove the task
                    for i, (p, t, tid) in enumerate(queue):
                        if tid == best_task.id:
                            queue.pop(i)
                            heapq.heapify(queue)
                            break

                    # Set visibility lock
                    expires_at = time.time() + (self._visibility_timeout_ms / 1000)
                    self._visibility_locks[best_task.id] = ("pending", expires_at)

                    # Update task status
                    best_task.status = TaskStatus.RUNNING
                    best_task.started_at = datetime.now(timezone.utc)

                    logger.info(
                        "Task dequeued",
                        task_id=best_task.id,
                        queue=best_queue,
                        name=best_task.name,
                    )
                    return best_task

            # No task available
            if timeout <= 0:
                return None

            # Wait for task or timeout
            elapsed = time.time() - start_time
            remaining = timeout - elapsed
            if remaining <= 0:
                return None

            # Clear event and wait
            self._task_available.clear()
            try:
                await asyncio.wait_for(
                    self._task_available.wait(),
                    timeout=min(remaining, 1.0)  # Check periodically
                )
            except asyncio.TimeoutError:
                pass

    async def acknowledge(self, task_id: str, result: TaskResult) -> None:
        """Acknowledge task completion."""
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning("Acknowledge for unknown task", task_id=task_id)
                return

            # Update task status
            task.status = result.status
            task.completed_at = result.completed_at

            # Store result
            self._results[task_id] = result

            # Remove visibility lock
            if task_id in self._visibility_locks:
                del self._visibility_locks[task_id]

            logger.info(
                "Task acknowledged",
                task_id=task_id,
                status=result.status,
                duration_ms=result.duration_ms,
            )

    async def requeue(self, task_id: str) -> None:
        """Requeue a task for retry."""
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task:
                logger.warning("Requeue for unknown task", task_id=task_id)
                return

            task.retries += 1
            task.status = TaskStatus.QUEUED
            task.started_at = None
            task.worker_id = None

            # Remove visibility lock
            if task_id in self._visibility_locks:
                del self._visibility_locks[task_id]

        # Re-add to queue
        async with self._queue_lock:
            entry = (task.priority, time.time(), task.id)
            heapq.heappush(self._queues[task.queue], entry)
            self._task_available.set()

        logger.info(
            "Task requeued",
            task_id=task_id,
            retries=task.retries,
            max_retries=task.max_retries,
        )

    async def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        return self._tasks.get(task_id)

    async def get_result(self, task_id: str) -> TaskResult | None:
        """Get result for a task."""
        return self._results.get(task_id)

    async def get_queue_stats(self, queue: str) -> QueueStats:
        """Get statistics for a queue."""
        pending = 0
        running = 0
        completed = 0
        failed = 0

        for task in self._tasks.values():
            if task.queue != queue:
                continue

            if task.status == TaskStatus.QUEUED:
                pending += 1
            elif task.status == TaskStatus.RUNNING:
                running += 1
            elif task.status == TaskStatus.SUCCESS:
                completed += 1
            elif task.status in (TaskStatus.FAILURE, TaskStatus.CANCELLED):
                failed += 1

        return QueueStats(
            name=queue,
            pending=pending,
            running=running,
            completed=completed,
            failed=failed,
            total=pending + running + completed + failed,
        )

    async def register_worker(self, worker_info: WorkerInfo) -> None:
        """Register a worker with the broker."""
        self._workers[worker_info.id] = worker_info
        logger.info("Worker registered", worker_id=worker_info.id, queues=worker_info.queues)

    async def heartbeat(self, worker_id: str, current_task: str | None = None) -> None:
        """Update worker heartbeat."""
        if worker_id in self._workers:
            self._workers[worker_id].last_heartbeat = datetime.now(timezone.utc)
            self._workers[worker_id].current_task = current_task

            # Extend visibility timeout for current task
            if current_task and current_task in self._visibility_locks:
                expires_at = time.time() + (self._visibility_timeout_ms / 1000)
                self._visibility_locks[current_task] = (worker_id, expires_at)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        async with self._task_lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.status not in (TaskStatus.PENDING, TaskStatus.QUEUED):
                return False

            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)

            # Remove from queue
            async with self._queue_lock:
                queue = self._queues.get(task.queue, [])
                self._queues[task.queue] = [
                    (p, t, tid) for p, t, tid in queue if tid != task_id
                ]
                heapq.heapify(self._queues[task.queue])

            logger.info("Task cancelled", task_id=task_id)
            return True

    async def get_all_queues(self) -> list[str]:
        """Get list of all queue names."""
        return list(self._queues.keys())

    async def get_all_workers(self) -> list[WorkerInfo]:
        """Get list of all registered workers."""
        return list(self._workers.values())
