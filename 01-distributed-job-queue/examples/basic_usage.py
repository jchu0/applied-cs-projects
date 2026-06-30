#!/usr/bin/env python3
"""Basic usage example for the distributed job queue."""

import asyncio
import random

from jobqueue import Task, InMemoryBroker, Worker


async def main():
    """Run a basic example with broker and worker."""

    # Create broker
    broker = InMemoryBroker()

    # Create worker
    worker = Worker(broker, queues=["default", "emails"], concurrency=2)

    # Register task handlers
    @worker.task("add")
    async def add_numbers(task: Task) -> int:
        """Add two numbers."""
        a = task.payload.get("a", 0)
        b = task.payload.get("b", 0)
        await asyncio.sleep(0.1)  # Simulate work
        return a + b

    @worker.task("send_email")
    async def send_email(task: Task) -> dict:
        """Simulate sending an email."""
        to = task.payload.get("to")
        subject = task.payload.get("subject")

        # Simulate occasional failures for retry demo
        if random.random() < 0.3:
            raise Exception("Email server temporarily unavailable")

        await asyncio.sleep(0.2)  # Simulate network call
        return {"sent": True, "to": to, "subject": subject}

    @worker.task("slow_task")
    async def slow_task(task: Task) -> str:
        """A task that takes a while."""
        duration = task.payload.get("duration", 1)
        await asyncio.sleep(duration)
        return f"Completed after {duration}s"

    # Start worker in background
    worker_task = asyncio.create_task(worker.start())

    # Give worker time to start
    await asyncio.sleep(0.5)

    # Enqueue some tasks
    print("Enqueuing tasks...")

    # Simple math task
    task1 = await broker.enqueue(Task(
        name="add",
        payload={"a": 10, "b": 20},
    ))
    print(f"Enqueued: {task1.id} (add)")

    # Email task (might fail and retry)
    task2 = await broker.enqueue(Task(
        name="send_email",
        queue="emails",
        payload={
            "to": "user@example.com",
            "subject": "Hello World",
        },
        max_retries=3,
    ))
    print(f"Enqueued: {task2.id} (send_email)")

    # Another math task with higher priority
    task3 = await broker.enqueue(Task(
        name="add",
        payload={"a": 100, "b": 200},
        priority=1,  # HIGH priority
    ))
    print(f"Enqueued: {task3.id} (add - high priority)")

    # Wait for tasks to complete
    print("\nWaiting for tasks to complete...")
    await asyncio.sleep(3)

    # Check results
    print("\nResults:")
    for task_id in [task1.id, task2.id, task3.id]:
        result = await broker.get_result(task_id)
        if result:
            print(f"  {task_id[:8]}: {result.status} - {result.result}")
        else:
            task = await broker.get_task(task_id)
            print(f"  {task_id[:8]}: {task.status if task else 'not found'}")

    # Show queue stats
    print("\nQueue stats:")
    for queue_name in ["default", "emails"]:
        stats = await broker.get_queue_stats(queue_name)
        print(f"  {queue_name}: {stats.completed} completed, {stats.failed} failed")

    # Stop worker
    await worker.stop()
    await asyncio.sleep(0.5)
    worker_task.cancel()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
