"""Distributed Job Queue + Scheduler System."""

import importlib.util

__version__ = "0.1.0"

__all__ = []

# Core models (require pydantic)
try:
    from jobqueue.models import Task, TaskStatus, TaskPriority, TaskResult
    __all__.extend(["Task", "TaskStatus", "TaskPriority", "TaskResult"])
except ImportError:
    pass

# Broker (core, no external deps beyond pydantic)
try:
    from jobqueue.broker import Broker, InMemoryBroker
    __all__.extend(["Broker", "InMemoryBroker"])
except ImportError:
    pass

# Worker (requires structlog)
try:
    from jobqueue.worker import Worker
    from jobqueue.pool import WorkerPool
    __all__.extend(["Worker", "WorkerPool"])
except ImportError:
    pass

# Scheduler (requires croniter)
try:
    from jobqueue.scheduler import Scheduler, ScheduledJob, schedule_delayed
    __all__.extend(["Scheduler", "ScheduledJob", "schedule_delayed"])
except ImportError:
    pass

# Circuit breaker
try:
    from jobqueue.circuit_breaker import CircuitBreaker, CircuitBreakerRegistry, CircuitOpenError
    __all__.extend(["CircuitBreaker", "CircuitBreakerRegistry", "CircuitOpenError"])
except ImportError:
    pass

# Deduplication
try:
    from jobqueue.deduplication import (
        DeduplicationLayer,
        DeduplicationStore,
        InMemoryDeduplicationStore,
        DeduplicationEntry,
    )
    __all__.extend([
        "DeduplicationLayer",
        "DeduplicationStore",
        "InMemoryDeduplicationStore",
        "DeduplicationEntry",
    ])
except ImportError:
    pass

# API (requires fastapi)
try:
    from jobqueue.api import create_app
    __all__.append("create_app")
except ImportError:
    pass

# Optional redis support
if importlib.util.find_spec("redis") is not None:
    try:
        from jobqueue.redis_broker import RedisBroker
        __all__.append("RedisBroker")
    except ImportError:
        pass
