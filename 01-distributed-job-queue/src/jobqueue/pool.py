"""Worker pool management for scaling workers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import structlog

from jobqueue.broker import Broker
from jobqueue.worker import Worker, TaskHandler

logger = structlog.get_logger()


class WorkerPool:
    """
    Manages a pool of workers for horizontal scaling.

    Features:
    - Dynamic worker scaling
    - Shared handler registration
    - Graceful shutdown
    - Pool-wide statistics
    """

    def __init__(
        self,
        broker: Broker,
        queues: list[str] | None = None,
        min_workers: int = 1,
        max_workers: int = 10,
        concurrency_per_worker: int = 4,
        **worker_kwargs: Any,
    ):
        self.broker = broker
        self.queues = queues or ["default"]
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.concurrency_per_worker = concurrency_per_worker
        self.worker_kwargs = worker_kwargs

        self._workers: list[Worker] = []
        self._worker_tasks: list[asyncio.Task] = []
        self._handlers: dict[str, TaskHandler] = {}
        self._running = False
        self._started_at: datetime | None = None

        logger.info(
            "Worker pool initialized",
            min_workers=min_workers,
            max_workers=max_workers,
            concurrency_per_worker=concurrency_per_worker,
        )

    def register_handler(self, task_name: str, handler: TaskHandler) -> None:
        """Register a handler for all workers in the pool."""
        self._handlers[task_name] = handler
        # Register with existing workers
        for worker in self._workers:
            worker.register_handler(task_name, handler)

    def task(self, name: str) -> Callable[[TaskHandler], TaskHandler]:
        """Decorator to register a task handler."""
        def decorator(func: TaskHandler) -> TaskHandler:
            self.register_handler(name, func)
            return func
        return decorator

    async def start(self, num_workers: int | None = None) -> None:
        """Start the worker pool."""
        if self._running:
            return

        self._running = True
        self._started_at = datetime.now(timezone.utc)

        # Determine initial worker count
        initial_count = num_workers or self.min_workers
        initial_count = max(self.min_workers, min(initial_count, self.max_workers))

        # Create and start workers
        for _ in range(initial_count):
            await self._add_worker()

        logger.info(
            "Worker pool started",
            num_workers=len(self._workers),
        )

    async def stop(self) -> None:
        """Stop all workers gracefully."""
        if not self._running:
            return

        self._running = False

        logger.info("Worker pool stopping", num_workers=len(self._workers))

        # Stop all workers
        for worker in self._workers:
            await worker.stop()

        # Wait for worker tasks
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        self._workers.clear()
        self._worker_tasks.clear()

        logger.info("Worker pool stopped")

    async def scale(self, target_workers: int) -> None:
        """Scale the pool to a target number of workers."""
        target_workers = max(self.min_workers, min(target_workers, self.max_workers))
        current_workers = len(self._workers)

        if target_workers == current_workers:
            return

        if target_workers > current_workers:
            # Scale up
            for _ in range(target_workers - current_workers):
                await self._add_worker()
            logger.info("Worker pool scaled up", new_count=len(self._workers))
        else:
            # Scale down
            workers_to_remove = current_workers - target_workers
            for _ in range(workers_to_remove):
                await self._remove_worker()
            logger.info("Worker pool scaled down", new_count=len(self._workers))

    async def _add_worker(self) -> Worker:
        """Add a new worker to the pool."""
        worker = Worker(
            broker=self.broker,
            queues=self.queues,
            concurrency=self.concurrency_per_worker,
            **self.worker_kwargs,
        )

        # Register all handlers
        for name, handler in self._handlers.items():
            worker.register_handler(name, handler)

        self._workers.append(worker)

        # Start worker in background
        task = asyncio.create_task(worker.start())
        self._worker_tasks.append(task)

        return worker

    async def _remove_worker(self) -> None:
        """Remove a worker from the pool."""
        if not self._workers:
            return

        worker = self._workers.pop()
        await worker.stop()

    def get_stats(self) -> dict[str, Any]:
        """Get pool-wide statistics."""
        total_completed = sum(w._tasks_completed for w in self._workers)
        total_failed = sum(w._tasks_failed for w in self._workers)

        return {
            "num_workers": len(self._workers),
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "concurrency_per_worker": self.concurrency_per_worker,
            "total_concurrency": len(self._workers) * self.concurrency_per_worker,
            "tasks_completed": total_completed,
            "tasks_failed": total_failed,
            "running": self._running,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "workers": [
                {
                    "id": w.worker_id,
                    "completed": w._tasks_completed,
                    "failed": w._tasks_failed,
                }
                for w in self._workers
            ],
        }

    def get_circuit_breaker_stats(self) -> dict[str, Any]:
        """Get circuit breaker stats from all workers."""
        all_stats = {}
        for worker in self._workers:
            stats = worker.get_circuit_breaker_stats()
            for name, breaker_stats in stats.items():
                if name not in all_stats:
                    all_stats[name] = breaker_stats
        return all_stats

    def reset_circuit_breakers(self) -> None:
        """Reset circuit breakers on all workers."""
        for worker in self._workers:
            worker.reset_circuit_breakers()

    @property
    def worker_count(self) -> int:
        """Get current number of workers."""
        return len(self._workers)

    @property
    def total_concurrency(self) -> int:
        """Get total concurrency across all workers."""
        return len(self._workers) * self.concurrency_per_worker
