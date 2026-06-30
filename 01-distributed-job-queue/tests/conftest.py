"""Pytest configuration and shared fixtures for tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import asyncio
import os
import uuid
import importlib.util
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock

# Check if required dependencies are available
REDIS_AVAILABLE = importlib.util.find_spec("redis") is not None
CRONITER_AVAILABLE = importlib.util.find_spec("croniter") is not None
STRUCTLOG_AVAILABLE = importlib.util.find_spec("structlog") is not None
DEPS_AVAILABLE = REDIS_AVAILABLE and CRONITER_AVAILABLE and STRUCTLOG_AVAILABLE

if DEPS_AVAILABLE:
    import redis.asyncio as redis
    from jobqueue.broker import Broker
    from jobqueue.models import Task, TaskStatus
    from jobqueue.redis_broker import RedisBroker
    from jobqueue.scheduler import Scheduler
    from jobqueue.worker import Worker


# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if DEPS_AVAILABLE:
    @pytest.fixture
    def task_factory():
        """Factory for creating test tasks."""
        def create_task(
            name: str = "test_task",
            queue: str = "test-queue",
            payload: dict = None,
            status: TaskStatus = TaskStatus.PENDING,
            priority: int = 5,
            **kwargs
        ) -> Task:
            """Create a test task with default values."""
            return Task(
                id=kwargs.get("id", f"task-{uuid.uuid4().hex[:8]}"),
                name=name,
                queue=queue,
                payload=payload or {"test": "data"},
                status=status,
                created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
                priority=priority,
                timeout=kwargs.get("timeout", 30),
                max_retries=kwargs.get("max_retries", 3),
                retry_count=kwargs.get("retry_count", 0),
                retry_delay=kwargs.get("retry_delay", 1.0),
                started_at=kwargs.get("started_at"),
                completed_at=kwargs.get("completed_at"),
            )

        return create_task


@pytest_asyncio.fixture
async def mock_redis_client():
    """Create mock Redis client for unit tests."""
    client = AsyncMock()

    # Mock common Redis operations
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.delete = AsyncMock(return_value=1)
    client.exists = AsyncMock(return_value=False)
    client.lpush = AsyncMock(return_value=1)
    client.rpop = AsyncMock(return_value=None)
    client.llen = AsyncMock(return_value=0)
    client.hset = AsyncMock(return_value=1)
    client.hget = AsyncMock(return_value=None)
    client.hgetall = AsyncMock(return_value={})
    client.zadd = AsyncMock(return_value=1)
    client.zrange = AsyncMock(return_value=[])
    client.zrem = AsyncMock(return_value=1)
    client.pubsub = AsyncMock()
    client.pipeline = AsyncMock()
    client.close = AsyncMock()

    # Mock pipeline operations
    pipeline = AsyncMock()
    pipeline.multi = AsyncMock(return_value=pipeline)
    pipeline.watch = AsyncMock(return_value=pipeline)
    pipeline.execute = AsyncMock(return_value=[True])
    client.pipeline.return_value = pipeline

    # Mock pubsub operations
    pubsub = AsyncMock()
    pubsub.subscribe = AsyncMock()
    pubsub.unsubscribe = AsyncMock()
    pubsub.listen = AsyncMock(return_value=AsyncMock())
    pubsub.close = AsyncMock()
    client.pubsub.return_value = pubsub

    yield client

    await client.close()


if DEPS_AVAILABLE:
    @pytest_asyncio.fixture
    async def real_redis_client():
        """Create real Redis client for integration tests."""
        # Skip if Redis is not available
        if not os.getenv("REDIS_INTEGRATION_TEST", "").lower() == "true":
            pytest.skip("Redis integration tests disabled")

        client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            db=int(os.getenv("REDIS_TEST_DB", 15)),  # Use separate DB for tests
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )

        # Test connection
        try:
            await client.ping()
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

        # Clear test database
        await client.flushdb()

        yield client

        # Cleanup
        await client.flushdb()
        await client.close()


    @pytest_asyncio.fixture
    async def redis_broker(real_redis_client):
        """Create Redis broker for integration tests."""
        broker = RedisBroker(real_redis_client)
        await broker.connect()

        yield broker

        await broker.close()


    @pytest.fixture
    def sample_tasks(task_factory):
        """Create sample tasks for testing."""
        return [
            task_factory(
                id=f"task-{i:03d}",
                name=f"task_type_{i % 3}",
                queue="queue1" if i % 2 == 0 else "queue2",
                payload={"index": i, "data": f"test_{i}"},
                priority=i % 10,
            )
            for i in range(10)
        ]


@pytest.fixture
def worker_config():
    """Default worker configuration for tests."""
    return {
        "queues": ["test-queue", "priority-queue"],
        "concurrency": 2,
        "poll_interval": 0.1,
        "heartbeat_interval": 1.0,
        "enable_circuit_breaker": True,
        "circuit_failure_threshold": 3,
        "circuit_reset_timeout": 5.0,
        "use_dlq": True,
    }


if DEPS_AVAILABLE:
    @pytest_asyncio.fixture
    async def mock_scheduler():
        """Create mock scheduler for unit tests."""
        scheduler = AsyncMock(spec=Scheduler)
        scheduler.schedule_task = AsyncMock(return_value="schedule-001")
        scheduler.cancel_scheduled_task = AsyncMock(return_value=True)
        scheduler.list_scheduled_tasks = AsyncMock(return_value=[])
        scheduler.start = AsyncMock()
        scheduler.stop = AsyncMock()
        return scheduler


@pytest.fixture
def performance_monitor():
    """Monitor for performance testing."""
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.operations = []

        def start(self):
            self.start_time = datetime.now(timezone.utc)

        def stop(self):
            self.end_time = datetime.now(timezone.utc)

        def record_operation(self, name: str, duration: float):
            self.operations.append({"name": name, "duration": duration})

        @property
        def total_duration(self):
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds()
            return 0

        @property
        def average_operation_time(self):
            if not self.operations:
                return 0
            return sum(op["duration"] for op in self.operations) / len(self.operations)

        @property
        def operations_per_second(self):
            if self.total_duration == 0:
                return 0
            return len(self.operations) / self.total_duration

    return PerformanceMonitor()


@pytest.fixture
def cleanup_registry():
    """Cleanup function registry for tests."""
    cleanups = []

    def add_cleanup(func):
        cleanups.append(func)

    yield add_cleanup

    # Execute cleanups in reverse order
    for cleanup in reversed(cleanups):
        try:
            if asyncio.iscoroutinefunction(cleanup):
                asyncio.run(cleanup())
            else:
                cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")


# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
pytest.mark.redis = pytest.mark.redis


# Test environment configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "redis: mark test as requiring Redis"
    )


# Async test timeout configuration
def pytest_collection_modifyitems(config, items):
    """Add timeout to async tests and skip if dependencies not available."""
    if not DEPS_AVAILABLE:
        skip_deps = pytest.mark.skip(reason="Required dependencies (redis, croniter, structlog) not installed")
        for item in items:
            item.add_marker(skip_deps)
        return

    for item in items:
        if "asyncio" in item.keywords:
            item.add_marker(pytest.mark.timeout(30))
