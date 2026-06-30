"""Worker implementation for processing tasks."""

from __future__ import annotations

import asyncio
import random
import signal
import time
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine

import structlog

from jobqueue.broker import Broker
from jobqueue.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitOpenError
from jobqueue.models import Task, TaskResult, TaskStatus, WorkerInfo

logger = structlog.get_logger()

# Type alias for task handlers
TaskHandler = Callable[[Task], Coroutine[Any, Any, Any]]


class Worker:
    """Worker that processes tasks from the broker."""

    def __init__(
        self,
        broker: Broker,
        queues: list[str] | None = None,
        concurrency: int = 1,
        poll_interval: float = 1.0,
        heartbeat_interval: float = 5.0,
        enable_circuit_breaker: bool = True,
        circuit_failure_threshold: int = 5,
        circuit_reset_timeout: float = 60.0,
        use_dlq: bool = True,
    ):
        self.broker = broker
        self.queues = queues or ["default"]
        self.concurrency = concurrency
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.enable_circuit_breaker = enable_circuit_breaker
        self.circuit_failure_threshold = circuit_failure_threshold
        self.circuit_reset_timeout = circuit_reset_timeout
        self.use_dlq = use_dlq

        self.worker_id = f"worker-{uuid.uuid4().hex[:8]}"
        self._handlers: dict[str, TaskHandler] = {}
        self._running = False
        self._current_tasks: dict[str, Task] = {}  # slot_id -> task
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._started_at: datetime | None = None

        # Circuit breaker registry for per-task-type breakers
        self._circuit_registry = CircuitBreakerRegistry()

        logger.info(
            "Worker initialized",
            worker_id=self.worker_id,
            queues=self.queues,
            concurrency=concurrency,
        )

    def register_handler(self, task_name: str, handler: TaskHandler) -> None:
        """Register a handler for a task type."""
        self._handlers[task_name] = handler
        logger.info("Handler registered", task_name=task_name, worker_id=self.worker_id)

    def task(self, name: str) -> Callable[[TaskHandler], TaskHandler]:
        """Decorator to register a task handler."""
        def decorator(func: TaskHandler) -> TaskHandler:
            self.register_handler(name, func)
            return func
        return decorator

    async def start(self) -> None:
        """Start the worker."""
        self._running = True
        self._started_at = datetime.now(timezone.utc)

        # Register with broker
        worker_info = WorkerInfo(
            id=self.worker_id,
            queues=self.queues,
            last_heartbeat=datetime.now(timezone.utc),
            started_at=self._started_at,
        )
        await self.broker.register_worker(worker_info)

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        logger.info("Worker started", worker_id=self.worker_id)

        # Start background tasks
        tasks = [
            asyncio.create_task(self._heartbeat_loop()),
        ]

        # Start worker slots
        for slot_id in range(self.concurrency):
            tasks.append(asyncio.create_task(self._worker_loop(f"slot-{slot_id}")))

        # Wait for all tasks
        await asyncio.gather(*tasks)

    async def stop(self) -> None:
        """Stop the worker gracefully."""
        logger.info("Worker stopping", worker_id=self.worker_id)
        self._running = False

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the broker."""
        while self._running:
            try:
                # Get current task (if any)
                current_task = None
                if self._current_tasks:
                    current_task = list(self._current_tasks.values())[0].id

                await self.broker.heartbeat(self.worker_id, current_task)
            except Exception as e:
                logger.error("Heartbeat failed", error=str(e))

            await asyncio.sleep(self.heartbeat_interval)

    async def _worker_loop(self, slot_id: str) -> None:
        """Main worker loop for a single slot."""
        logger.info("Worker slot started", worker_id=self.worker_id, slot_id=slot_id)

        while self._running:
            try:
                # Poll for task
                task = await self.broker.dequeue(self.queues, timeout=self.poll_interval)

                if task:
                    self._current_tasks[slot_id] = task
                    try:
                        await self._process_task(task)
                    finally:
                        del self._current_tasks[slot_id]
                else:
                    # No task available, wait a bit
                    await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "Worker loop error",
                    error=str(e),
                    slot_id=slot_id,
                    traceback=traceback.format_exc(),
                )
                await asyncio.sleep(1)

        logger.info("Worker slot stopped", worker_id=self.worker_id, slot_id=slot_id)

    async def _process_task(self, task: Task) -> None:
        """Process a single task."""
        start_time = time.time()

        logger.info(
            "Processing task",
            task_id=task.id,
            task_name=task.name,
            worker_id=self.worker_id,
        )

        try:
            # Find handler
            handler = self._handlers.get(task.name)
            if not handler:
                raise ValueError(f"No handler registered for task: {task.name}")

            # Check circuit breaker
            if self.enable_circuit_breaker:
                breaker = await self._circuit_registry.get_or_create(
                    name=task.name,
                    failure_threshold=self.circuit_failure_threshold,
                    reset_timeout=self.circuit_reset_timeout,
                )

                # Execute through circuit breaker with timeout
                timeout_seconds = task.timeout_ms / 1000
                result_value = await breaker.call(
                    asyncio.wait_for,
                    handler(task),
                    timeout=timeout_seconds
                )
            else:
                # Execute with timeout directly
                timeout_seconds = task.timeout_ms / 1000
                result_value = await asyncio.wait_for(
                    handler(task),
                    timeout=timeout_seconds
                )

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Create success result
            result = TaskResult(
                task_id=task.id,
                status=TaskStatus.SUCCESS,
                result=result_value,
                started_at=task.started_at,
                duration_ms=duration_ms,
                worker_id=self.worker_id,
            )

            await self.broker.acknowledge(task.id, result)
            self._tasks_completed += 1

            logger.info(
                "Task completed",
                task_id=task.id,
                duration_ms=duration_ms,
            )

        except CircuitOpenError as e:
            # Circuit is open, requeue immediately without counting as failure
            logger.warning(
                "Circuit breaker open, requeuing task",
                task_id=task.id,
                task_name=task.name,
                error=str(e),
            )
            await self.broker.requeue(task.id)

        except asyncio.TimeoutError:
            await self._handle_task_failure(
                task,
                start_time,
                "Task timeout",
                timeout=True
            )

        except Exception as e:
            await self._handle_task_failure(
                task,
                start_time,
                str(e),
                tb=traceback.format_exc()
            )

    async def _handle_task_failure(
        self,
        task: Task,
        start_time: float,
        error: str,
        tb: str | None = None,
        timeout: bool = False
    ) -> None:
        """Handle task failure with retry logic."""
        duration_ms = (time.time() - start_time) * 1000

        logger.error(
            "Task failed",
            task_id=task.id,
            error=error,
            retries=task.retries,
            max_retries=task.max_retries,
        )

        # Check if we should retry
        if task.retries < task.max_retries:
            # Calculate backoff
            backoff = self._calculate_backoff(task.retries)
            logger.info(
                "Scheduling retry",
                task_id=task.id,
                retry=task.retries + 1,
                backoff_seconds=backoff,
            )

            # Wait for backoff then requeue
            await asyncio.sleep(backoff)
            await self.broker.requeue(task.id)

        else:
            # Max retries exceeded, mark as failed
            result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILURE,
                error=error,
                traceback=tb,
                started_at=task.started_at,
                duration_ms=duration_ms,
                worker_id=self.worker_id,
            )

            # Move to dead-letter queue if enabled and broker supports it
            if self.use_dlq and hasattr(self.broker, 'move_to_dlq'):
                await self.broker.move_to_dlq(task.id, error)
            else:
                await self.broker.acknowledge(task.id, result)

            self._tasks_failed += 1

    def get_circuit_breaker_stats(self) -> dict[str, Any]:
        """Get statistics for all circuit breakers."""
        return self._circuit_registry.get_all_stats()

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        self._circuit_registry.reset_all()
        logger.info("All circuit breakers reset", worker_id=self.worker_id)

    def _calculate_backoff(self, attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff with jitter."""
        delay = base_delay * (2 ** attempt)
        jitter = random.uniform(0, delay * 0.1)
        return min(delay + jitter, 60.0)  # Max 60 seconds


async def run_worker(
    broker: Broker,
    handlers: dict[str, TaskHandler],
    queues: list[str] | None = None,
    concurrency: int = 1,
) -> None:
    """Convenience function to run a worker with handlers."""
    worker = Worker(broker, queues=queues, concurrency=concurrency)

    for name, handler in handlers.items():
        worker.register_handler(name, handler)

    await worker.start()
