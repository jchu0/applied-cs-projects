"""Worker registry for model routing."""

import time
import asyncio
from typing import Any
from dataclasses import asdict

from ..schemas import WorkerInfo, GPUInfo, CapacityInfo


class WorkerRegistry:
    """Manages worker registration and health."""

    def __init__(self, storage=None, health_checker=None):
        """Initialize registry.

        Args:
            storage: Storage backend
            health_checker: Health checker
        """
        self.storage = storage or InMemoryStorage()
        self.health_checker = health_checker

    async def register(self, worker: WorkerInfo) -> str:
        """Register a new worker.

        Args:
            worker: Worker info

        Returns:
            Worker ID
        """
        await self.storage.set(
            f"worker:{worker.worker_id}",
            self._worker_to_dict(worker),
            ttl=60  # Requires heartbeat
        )

        # Add to model index
        for model in worker.models:
            await self.storage.sadd(f"model:{model}:workers", worker.worker_id)

        return worker.worker_id

    async def deregister(self, worker_id: str):
        """Remove worker from registry.

        Args:
            worker_id: Worker ID
        """
        worker = await self.get_worker(worker_id)
        if worker:
            for model in worker.models:
                await self.storage.srem(f"model:{model}:workers", worker_id)
            await self.storage.delete(f"worker:{worker_id}")

    async def heartbeat(self, worker_id: str, status: dict):
        """Update worker status.

        Args:
            worker_id: Worker ID
            status: Status update
        """
        existing = await self.storage.get(f"worker:{worker_id}")
        if existing:
            existing.update(status)
            existing["last_heartbeat"] = time.time()
            await self.storage.set(
                f"worker:{worker_id}",
                existing,
                ttl=60
            )

    async def update_status(self, worker_id: str, status: str):
        """Update worker health status.

        Args:
            worker_id: Worker ID
            status: Health status
        """
        existing = await self.storage.get(f"worker:{worker_id}")
        if existing:
            existing["status"] = status
            await self.storage.set(f"worker:{worker_id}", existing, ttl=60)

    async def get_workers(
        self,
        model: str = None,
        status: str = "healthy"
    ) -> list[WorkerInfo]:
        """Get workers, optionally filtered.

        Args:
            model: Filter by model
            status: Filter by status

        Returns:
            List of workers
        """
        if model:
            worker_ids = await self.storage.smembers(f"model:{model}:workers")
        else:
            worker_ids = await self.storage.keys("worker:*")
            worker_ids = [k.replace("worker:", "") for k in worker_ids]

        workers = []
        for worker_id in worker_ids:
            data = await self.storage.get(f"worker:{worker_id}")
            if data and (status is None or data.get("status") == status):
                workers.append(self._dict_to_worker(data))

        return workers

    async def get_worker(self, worker_id: str) -> WorkerInfo | None:
        """Get specific worker.

        Args:
            worker_id: Worker ID

        Returns:
            Worker info or None
        """
        data = await self.storage.get(f"worker:{worker_id}")
        return self._dict_to_worker(data) if data else None

    async def update_load(
        self,
        worker_id: str,
        current_load: float,
        queue_depth: int,
        tokens_in_flight: int
    ):
        """Update worker load metrics.

        Args:
            worker_id: Worker ID
            current_load: Current load
            queue_depth: Queue depth
            tokens_in_flight: Tokens in flight
        """
        await self.heartbeat(worker_id, {
            "current_load": current_load,
            "queue_depth": queue_depth,
            "tokens_in_flight": tokens_in_flight
        })

    def _worker_to_dict(self, worker: WorkerInfo) -> dict:
        """Convert worker to dictionary.

        Args:
            worker: Worker info

        Returns:
            Dictionary representation
        """
        return {
            "worker_id": worker.worker_id,
            "host": worker.host,
            "port": worker.port,
            "models": worker.models,
            "gpu_info": asdict(worker.gpu_info),
            "current_load": worker.current_load,
            "queue_depth": worker.queue_depth,
            "tokens_in_flight": worker.tokens_in_flight,
            "token_budget": worker.token_budget,
            "status": worker.status,
            "last_heartbeat": worker.last_heartbeat,
            "performance_factor": worker.performance_factor,
            "worker_type": worker.worker_type
        }

    def _dict_to_worker(self, data: dict) -> WorkerInfo:
        """Convert dictionary to worker.

        Args:
            data: Dictionary data

        Returns:
            Worker info
        """
        gpu_data = data.get("gpu_info", {})
        gpu_info = GPUInfo(
            device_name=gpu_data.get("device_name", "Unknown"),
            memory_total_mb=gpu_data.get("memory_total_mb", 0),
            memory_used_mb=gpu_data.get("memory_used_mb", 0),
            utilization_percent=gpu_data.get("utilization_percent", 0),
            temperature_celsius=gpu_data.get("temperature_celsius", 0)
        )

        return WorkerInfo(
            worker_id=data["worker_id"],
            host=data["host"],
            port=data["port"],
            models=data["models"],
            gpu_info=gpu_info,
            current_load=data.get("current_load", 0),
            queue_depth=data.get("queue_depth", 0),
            tokens_in_flight=data.get("tokens_in_flight", 0),
            token_budget=data.get("token_budget", 100000),
            status=data.get("status", "healthy"),
            last_heartbeat=data.get("last_heartbeat", time.time()),
            performance_factor=data.get("performance_factor", 1.0),
            worker_type=data.get("worker_type", "standard")
        )


class InMemoryStorage:
    """In-memory storage for registry."""

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._sets: dict[str, set] = {}
        self._expiry: dict[str, float] = {}

    async def get(self, key: str) -> Any | None:
        """Get value by key."""
        self._clean_expired(key)
        return self._data.get(key)

    async def set(self, key: str, value: Any, ttl: int = None):
        """Set key-value pair."""
        self._data[key] = value
        if ttl:
            self._expiry[key] = time.time() + ttl

    async def delete(self, key: str):
        """Delete key."""
        self._data.pop(key, None)
        self._expiry.pop(key, None)

    async def keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return [k for k in self._data.keys() if k.startswith(prefix)]
        return []

    async def sadd(self, key: str, member: str):
        """Add to set."""
        if key not in self._sets:
            self._sets[key] = set()
        self._sets[key].add(member)

    async def srem(self, key: str, member: str):
        """Remove from set."""
        if key in self._sets:
            self._sets[key].discard(member)

    async def smembers(self, key: str) -> set:
        """Get set members."""
        return self._sets.get(key, set()).copy()

    def _clean_expired(self, key: str):
        """Clean expired entries."""
        if key in self._expiry and time.time() > self._expiry[key]:
            self._data.pop(key, None)
            self._expiry.pop(key, None)


class CapacityTracker:
    """Tracks capacity across all workers."""

    def __init__(self, registry: WorkerRegistry):
        """Initialize tracker.

        Args:
            registry: Worker registry
        """
        self.registry = registry

    async def get_capacity(self, model: str) -> CapacityInfo:
        """Get current capacity for a model.

        Args:
            model: Model name

        Returns:
            Capacity information
        """
        workers = await self.registry.get_workers(model=model)

        if not workers:
            return CapacityInfo(
                total_workers=0,
                healthy_workers=0,
                total_token_budget=0,
                available_tokens=0,
                utilization=0,
                gpu_memory_total_mb=0,
                gpu_memory_used_mb=0
            )

        total_token_budget = sum(w.token_budget for w in workers)
        tokens_in_flight = sum(w.tokens_in_flight for w in workers)
        available_tokens = total_token_budget - tokens_in_flight

        total_gpu_memory = sum(w.gpu_info.memory_total_mb for w in workers)
        used_gpu_memory = sum(w.gpu_info.memory_used_mb for w in workers)

        return CapacityInfo(
            total_workers=len(workers),
            healthy_workers=len([w for w in workers if w.status == "healthy"]),
            total_token_budget=total_token_budget,
            available_tokens=available_tokens,
            utilization=tokens_in_flight / total_token_budget if total_token_budget > 0 else 0,
            gpu_memory_total_mb=total_gpu_memory,
            gpu_memory_used_mb=used_gpu_memory
        )

    async def get_all_capacity(self) -> dict[str, CapacityInfo]:
        """Get capacity for all models.

        Returns:
            Capacity by model
        """
        workers = await self.registry.get_workers(status=None)

        # Get unique models
        models = set()
        for worker in workers:
            models.update(worker.models)

        capacity = {}
        for model in models:
            capacity[model] = await self.get_capacity(model)

        return capacity
