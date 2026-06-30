"""Redis-backed broker implementation for persistent task storage."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

import redis.asyncio as redis
import structlog

from jobqueue.broker import Broker
from jobqueue.models import (
    Task,
    TaskResult,
    TaskStatus,
    TaskPriority,
    QueueStats,
    WorkerInfo,
)

logger = structlog.get_logger()


class RedisBroker(Broker):
    """Redis-backed broker for persistent task storage."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        visibility_timeout_ms: int = 30000,
        result_ttl_seconds: int = 3600,
        key_prefix: str = "jobqueue",
    ):
        self.redis_url = redis_url
        self.visibility_timeout_ms = visibility_timeout_ms
        self.result_ttl_seconds = result_ttl_seconds
        self.key_prefix = key_prefix

        self._redis: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None

    async def connect(self) -> None:
        """Connect to Redis."""
        self._redis = redis.from_url(self.redis_url, decode_responses=True)
        await self._redis.ping()
        logger.info("Connected to Redis", url=self.redis_url)

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            logger.info("Disconnected from Redis")

    def _key(self, *parts: str) -> str:
        """Generate a Redis key with prefix."""
        return f"{self.key_prefix}:{':'.join(parts)}"

    async def enqueue(self, task: Task) -> Task:
        """Add a task to the queue."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        # Check for duplicate via idempotency key
        if task.idempotency_key:
            existing_id = await self._redis.get(
                self._key("idempotency", task.idempotency_key)
            )
            if existing_id:
                existing_task = await self.get_task(existing_id)
                if existing_task:
                    logger.info(
                        "Duplicate task detected",
                        task_id=task.id,
                        existing_id=existing_id,
                    )
                    return existing_task

        # Update task status
        task.status = TaskStatus.QUEUED

        # Store task data
        task_data = task.model_dump_json()
        await self._redis.set(self._key("task", task.id), task_data)

        # Store idempotency key mapping
        if task.idempotency_key:
            await self._redis.set(
                self._key("idempotency", task.idempotency_key),
                task.id,
                ex=self.result_ttl_seconds,
            )

        # Add to priority queue
        # Score = priority * 1e12 + timestamp (for FIFO within priority)
        score = task.priority * 1e12 + time.time()
        await self._redis.zadd(
            self._key("queue", task.queue),
            {task.id: score}
        )

        # Track in queue set
        await self._redis.sadd(self._key("queues"), task.queue)

        # Publish notification
        await self._redis.publish(
            self._key("notifications"),
            json.dumps({"type": "task_enqueued", "queue": task.queue})
        )

        logger.info(
            "Task enqueued",
            task_id=task.id,
            queue=task.queue,
            priority=task.priority,
        )
        return task

    async def dequeue(self, queues: list[str], timeout: float = 0) -> Task | None:
        """Get the next task from the specified queues."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        start_time = time.time()

        while True:
            # Check each queue for available tasks
            for queue_name in queues:
                queue_key = self._key("queue", queue_name)
                processing_key = self._key("processing", queue_name)

                # Try to atomically move task from queue to processing
                # Using ZPOPMIN to get highest priority (lowest score)
                result = await self._redis.zpopmin(queue_key, count=1)

                if result:
                    task_id, score = result[0]

                    # Get task data
                    task_data = await self._redis.get(self._key("task", task_id))
                    if not task_data:
                        continue

                    task = Task.model_validate_json(task_data)

                    # Check ETA
                    if task.eta and task.eta > datetime.now(timezone.utc):
                        # Put back in queue
                        await self._redis.zadd(queue_key, {task_id: score})
                        continue

                    # Check if task is still valid
                    if task.status != TaskStatus.QUEUED:
                        continue

                    # Set visibility timeout
                    expires_at = time.time() + (self.visibility_timeout_ms / 1000)
                    await self._redis.zadd(
                        processing_key,
                        {task_id: expires_at}
                    )

                    # Update task status
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now(timezone.utc)
                    await self._redis.set(
                        self._key("task", task.id),
                        task.model_dump_json()
                    )

                    logger.info(
                        "Task dequeued",
                        task_id=task.id,
                        queue=queue_name,
                    )
                    return task

            # No task available
            if timeout <= 0:
                return None

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                return None

            # Wait a bit before retrying
            await asyncio.sleep(min(0.1, timeout - elapsed))

    async def acknowledge(self, task_id: str, result: TaskResult) -> None:
        """Acknowledge task completion."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        # Get task to find queue
        task = await self.get_task(task_id)
        if not task:
            logger.warning("Acknowledge for unknown task", task_id=task_id)
            return

        # Update task status
        task.status = result.status
        task.completed_at = result.completed_at
        await self._redis.set(
            self._key("task", task.id),
            task.model_dump_json()
        )

        # Store result with TTL
        await self._redis.set(
            self._key("result", task_id),
            result.model_dump_json(),
            ex=self.result_ttl_seconds,
        )

        # Remove from processing queue
        await self._redis.zrem(
            self._key("processing", task.queue),
            task_id
        )

        # Update stats
        if result.status == TaskStatus.SUCCESS:
            await self._redis.incr(self._key("stats", task.queue, "completed"))
        else:
            await self._redis.incr(self._key("stats", task.queue, "failed"))

        logger.info(
            "Task acknowledged",
            task_id=task_id,
            status=result.status,
        )

    async def requeue(self, task_id: str) -> None:
        """Requeue a task for retry."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        task = await self.get_task(task_id)
        if not task:
            return

        # Update task
        task.retries += 1
        task.status = TaskStatus.QUEUED
        task.started_at = None
        task.worker_id = None

        await self._redis.set(
            self._key("task", task.id),
            task.model_dump_json()
        )

        # Remove from processing
        await self._redis.zrem(
            self._key("processing", task.queue),
            task_id
        )

        # Add back to queue
        score = task.priority * 1e12 + time.time()
        await self._redis.zadd(
            self._key("queue", task.queue),
            {task_id: score}
        )

        logger.info(
            "Task requeued",
            task_id=task_id,
            retries=task.retries,
        )

    async def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        task_data = await self._redis.get(self._key("task", task_id))
        if not task_data:
            return None

        return Task.model_validate_json(task_data)

    async def get_result(self, task_id: str) -> TaskResult | None:
        """Get result for a task."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        result_data = await self._redis.get(self._key("result", task_id))
        if not result_data:
            return None

        return TaskResult.model_validate_json(result_data)

    async def get_queue_stats(self, queue: str) -> QueueStats:
        """Get statistics for a queue."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        pending = await self._redis.zcard(self._key("queue", queue))
        running = await self._redis.zcard(self._key("processing", queue))
        completed = int(await self._redis.get(self._key("stats", queue, "completed")) or 0)
        failed = int(await self._redis.get(self._key("stats", queue, "failed")) or 0)

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
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        await self._redis.hset(
            self._key("workers"),
            worker_info.id,
            worker_info.model_dump_json(),
        )
        logger.info("Worker registered", worker_id=worker_info.id)

    async def heartbeat(self, worker_id: str, current_task: str | None = None) -> None:
        """Update worker heartbeat."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        # Update worker info
        worker_data = await self._redis.hget(self._key("workers"), worker_id)
        if worker_data:
            worker = WorkerInfo.model_validate_json(worker_data)
            worker.last_heartbeat = datetime.now(timezone.utc)
            worker.current_task = current_task
            await self._redis.hset(
                self._key("workers"),
                worker_id,
                worker.model_dump_json(),
            )

        # Extend visibility timeout for current task
        if current_task:
            task = await self.get_task(current_task)
            if task:
                expires_at = time.time() + (self.visibility_timeout_ms / 1000)
                await self._redis.zadd(
                    self._key("processing", task.queue),
                    {current_task: expires_at}
                )

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        task = await self.get_task(task_id)
        if not task:
            return False

        if task.status not in (TaskStatus.PENDING, TaskStatus.QUEUED):
            return False

        # Update task
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.now(timezone.utc)
        await self._redis.set(
            self._key("task", task.id),
            task.model_dump_json()
        )

        # Remove from queue
        await self._redis.zrem(
            self._key("queue", task.queue),
            task_id
        )

        logger.info("Task cancelled", task_id=task_id)
        return True

    async def get_all_queues(self) -> list[str]:
        """Get list of all queue names."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        queues = await self._redis.smembers(self._key("queues"))
        return list(queues)

    async def get_all_workers(self) -> list[WorkerInfo]:
        """Get list of all registered workers."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        workers_data = await self._redis.hgetall(self._key("workers"))
        return [
            WorkerInfo.model_validate_json(data)
            for data in workers_data.values()
        ]

    async def recover_stuck_tasks(self) -> int:
        """Recover tasks stuck in processing (visibility timeout expired)."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        recovered = 0
        queues = await self.get_all_queues()

        for queue in queues:
            processing_key = self._key("processing", queue)

            # Find expired tasks
            now = time.time()
            expired = await self._redis.zrangebyscore(
                processing_key,
                "-inf",
                now
            )

            for task_id in expired:
                task = await self.get_task(task_id)
                if task and task.status == TaskStatus.RUNNING:
                    await self.requeue(task_id)
                    recovered += 1
                    logger.info(
                        "Recovered stuck task",
                        task_id=task_id,
                        queue=queue,
                    )

        return recovered

    async def move_to_dlq(self, task_id: str, error: str) -> None:
        """Move a task to the dead-letter queue."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        task = await self.get_task(task_id)
        if not task:
            return

        # Create DLQ entry
        dlq_entry = {
            "task_id": task_id,
            "task": task.model_dump(),
            "error": error,
            "moved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add to DLQ
        await self._redis.lpush(
            self._key("dlq", task.queue),
            json.dumps(dlq_entry)
        )

        # Remove from processing
        await self._redis.zrem(
            self._key("processing", task.queue),
            task_id
        )

        # Update task status
        task.status = TaskStatus.FAILURE
        await self._redis.set(
            self._key("task", task.id),
            task.model_dump_json()
        )

        logger.info(
            "Task moved to DLQ",
            task_id=task_id,
            queue=task.queue,
            error=error,
        )

    async def get_dlq_size(self, queue: str) -> int:
        """Get the number of tasks in the dead-letter queue."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        return await self._redis.llen(self._key("dlq", queue))

    async def get_dlq_tasks(self, queue: str, limit: int = 100) -> list[dict]:
        """Get tasks from the dead-letter queue."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        entries = await self._redis.lrange(
            self._key("dlq", queue),
            0,
            limit - 1
        )
        return [json.loads(entry) for entry in entries]

    async def retry_from_dlq(self, queue: str, task_id: str) -> bool:
        """Retry a task from the dead-letter queue."""
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        # Find and remove from DLQ
        dlq_key = self._key("dlq", queue)
        entries = await self._redis.lrange(dlq_key, 0, -1)

        for entry_data in entries:
            entry = json.loads(entry_data)
            if entry["task_id"] == task_id:
                await self._redis.lrem(dlq_key, 1, entry_data)

                # Reset task for retry
                task = await self.get_task(task_id)
                if task:
                    task.retries = 0
                    task.status = TaskStatus.QUEUED
                    await self._redis.set(
                        self._key("task", task.id),
                        task.model_dump_json()
                    )

                    # Add back to queue
                    score = task.priority * 1e12 + time.time()
                    await self._redis.zadd(
                        self._key("queue", queue),
                        {task_id: score}
                    )

                    logger.info("Task retried from DLQ", task_id=task_id)
                    return True

        return False
