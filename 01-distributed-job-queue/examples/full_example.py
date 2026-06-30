#!/usr/bin/env python3
"""
Full example demonstrating all features of the distributed job queue.

Features demonstrated:
- Worker pool with multiple workers
- Scheduler with cron and interval jobs
- Delayed task execution
- Circuit breakers
- Task priorities and queues
- Retry logic
"""

import asyncio
import random

from jobqueue import (
    Task,
    TaskPriority,
    InMemoryBroker,
    WorkerPool,
    Scheduler,
    schedule_delayed,
)


async def main():
    """Run the full example."""
    print("=" * 60)
    print("Distributed Job Queue - Full Feature Demo")
    print("=" * 60)

    # Create broker
    broker = InMemoryBroker()

    # Create worker pool
    pool = WorkerPool(
        broker,
        queues=["default", "emails", "reports"],
        min_workers=2,
        max_workers=5,
        concurrency_per_worker=2,
        enable_circuit_breaker=True,
        circuit_failure_threshold=3,
    )

    # Create scheduler
    scheduler = Scheduler(broker, poll_interval=1.0)

    # Register task handlers
    @pool.task("add")
    async def add_numbers(task: Task) -> int:
        """Add two numbers."""
        a = task.payload.get("a", 0)
        b = task.payload.get("b", 0)
        await asyncio.sleep(0.1)
        return a + b

    @pool.task("send_email")
    async def send_email(task: Task) -> dict:
        """Send an email (simulated)."""
        to = task.payload.get("to")
        subject = task.payload.get("subject")

        # Simulate occasional failures
        if random.random() < 0.2:
            raise Exception("Email server unavailable")

        await asyncio.sleep(0.2)
        return {"sent": True, "to": to, "subject": subject}

    @pool.task("generate_report")
    async def generate_report(task: Task) -> dict:
        """Generate a report (simulated)."""
        report_type = task.payload.get("type", "daily")
        await asyncio.sleep(0.5)
        return {"report": report_type, "rows": random.randint(100, 1000)}

    @pool.task("health_check")
    async def health_check(task: Task) -> dict:
        """Perform health check."""
        return {"status": "healthy", "timestamp": task.created_at.isoformat()}

    @pool.task("flaky_task")
    async def flaky_task(task: Task) -> str:
        """A task that fails often (for circuit breaker demo)."""
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "success"

    # Start pool and scheduler
    pool_task = asyncio.create_task(pool.start(num_workers=3))
    scheduler_task = asyncio.create_task(scheduler.start())

    await asyncio.sleep(0.5)  # Let them start

    print("\n[1] Testing basic tasks...")
    # -----------------------------------------------

    # Enqueue some basic tasks
    task1 = await broker.enqueue(Task(
        name="add",
        payload={"a": 10, "b": 20},
    ))
    print(f"  Enqueued add task: {task1.id[:8]}")

    task2 = await broker.enqueue(Task(
        name="add",
        payload={"a": 100, "b": 200},
        priority=TaskPriority.HIGH,
    ))
    print(f"  Enqueued high-priority add task: {task2.id[:8]}")

    print("\n[2] Testing multiple queues...")
    # -----------------------------------------------

    task3 = await broker.enqueue(Task(
        name="send_email",
        queue="emails",
        payload={"to": "user@example.com", "subject": "Hello"},
        max_retries=3,
    ))
    print(f"  Enqueued email task: {task3.id[:8]}")

    task4 = await broker.enqueue(Task(
        name="generate_report",
        queue="reports",
        payload={"type": "weekly"},
    ))
    print(f"  Enqueued report task: {task4.id[:8]}")

    print("\n[3] Testing scheduled jobs...")
    # -----------------------------------------------

    # Add a job that runs every 3 seconds
    scheduler.add_job(
        name="periodic_health",
        task_name="health_check",
        interval_seconds=3,
        queue="default",
    )
    print("  Added periodic health check (every 3s)")

    # Manually trigger a job
    manual_task = await scheduler.run_once("periodic_health")
    if manual_task:
        print(f"  Manually triggered job: {manual_task.id[:8]}")

    print("\n[4] Testing delayed tasks...")
    # -----------------------------------------------

    delayed_task = await schedule_delayed(
        broker,
        task_name="add",
        delay_seconds=2,
        payload={"a": 1000, "b": 2000},
    )
    print(f"  Scheduled delayed task (2s): {delayed_task.id[:8]}")

    print("\n[5] Testing circuit breaker...")
    # -----------------------------------------------

    # Enqueue tasks that will fail often
    for i in range(5):
        await broker.enqueue(Task(
            name="flaky_task",
            payload={"attempt": i},
        ))
    print("  Enqueued 5 flaky tasks (expect circuit breaker to open)")

    # Wait for processing
    print("\n[Waiting 5 seconds for processing...]")
    await asyncio.sleep(5)

    print("\n[6] Checking results...")
    # -----------------------------------------------

    for task_id, name in [(task1.id, "add1"), (task2.id, "add2"), (task3.id, "email"), (task4.id, "report")]:
        result = await broker.get_result(task_id)
        if result:
            status = "OK" if result.status == "success" else result.status
            print(f"  {name}: {status} - {result.result}")
        else:
            task = await broker.get_task(task_id)
            print(f"  {name}: {task.status if task else 'not found'}")

    print("\n[7] Pool statistics...")
    # -----------------------------------------------

    stats = pool.get_stats()
    print(f"  Workers: {stats['num_workers']}")
    print(f"  Total concurrency: {stats['total_concurrency']}")
    print(f"  Tasks completed: {stats['tasks_completed']}")
    print(f"  Tasks failed: {stats['tasks_failed']}")

    print("\n[8] Circuit breaker stats...")
    # -----------------------------------------------

    cb_stats = pool.get_circuit_breaker_stats()
    for name, breaker in cb_stats.items():
        print(f"  {name}: {breaker['state']} (failures: {breaker['failure_count']})")

    print("\n[9] Queue statistics...")
    # -----------------------------------------------

    for queue_name in ["default", "emails", "reports"]:
        queue_stats = await broker.get_queue_stats(queue_name)
        print(f"  {queue_name}: {queue_stats.completed} completed, {queue_stats.failed} failed")

    print("\n[10] Testing dynamic scaling...")
    # -----------------------------------------------

    print(f"  Current workers: {pool.worker_count}")
    await pool.scale(5)
    print(f"  After scale up: {pool.worker_count}")
    await pool.scale(2)
    print(f"  After scale down: {pool.worker_count}")

    # Cleanup
    print("\n[Shutting down...]")
    await scheduler.stop()
    await pool.stop()
    scheduler_task.cancel()
    pool_task.cancel()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
