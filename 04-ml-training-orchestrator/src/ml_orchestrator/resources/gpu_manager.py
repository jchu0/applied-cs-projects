"""GPU resource management."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import uuid4
import structlog

from ml_orchestrator.core.exceptions import GPUNotAvailableError, ResourceError


logger = structlog.get_logger(__name__)


class GPUStatus(str, Enum):
    """GPU availability status."""

    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    MAINTENANCE = "maintenance"
    FAILED = "failed"


@dataclass
class GPUInfo:
    """Information about a GPU."""

    id: str = field(default_factory=lambda: str(uuid4()))
    node_id: str = ""
    device_index: int = 0
    name: str = "Unknown GPU"
    gpu_type: str = "unknown"
    memory_total_gb: float = 0.0
    memory_free_gb: float = 0.0
    memory_used_gb: float = 0.0
    utilization_percent: float = 0.0
    temperature_celsius: float = 0.0
    power_usage_watts: float = 0.0
    status: GPUStatus = GPUStatus.AVAILABLE
    allocated_to: Optional[str] = None  # Job ID
    allocated_at: Optional[datetime] = None
    labels: dict[str, str] = field(default_factory=dict)
    pcie_generation: int = 4
    pcie_width: int = 16
    compute_capability: tuple[int, int] = (8, 0)

    @property
    def memory_utilization(self) -> float:
        """Get memory utilization percentage."""
        if self.memory_total_gb <= 0:
            return 0.0
        return (self.memory_used_gb / self.memory_total_gb) * 100

    @property
    def is_available(self) -> bool:
        """Check if GPU is available for allocation."""
        return self.status == GPUStatus.AVAILABLE

    def matches_type(self, gpu_type: Optional[str]) -> bool:
        """Check if GPU matches requested type."""
        if not gpu_type:
            return True
        return self.gpu_type.upper() == gpu_type.upper()

    def has_memory(self, required_gb: Optional[float]) -> bool:
        """Check if GPU has enough memory."""
        if not required_gb:
            return True
        return self.memory_total_gb >= required_gb

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "node_id": self.node_id,
            "device_index": self.device_index,
            "name": self.name,
            "gpu_type": self.gpu_type,
            "memory_total_gb": self.memory_total_gb,
            "memory_free_gb": self.memory_free_gb,
            "memory_used_gb": self.memory_used_gb,
            "utilization_percent": self.utilization_percent,
            "status": self.status.value,
            "allocated_to": self.allocated_to,
        }


@dataclass
class GPUAllocation:
    """Record of GPU allocation."""

    id: str = field(default_factory=lambda: str(uuid4()))
    job_id: str = ""
    gpu_ids: list[str] = field(default_factory=list)
    node_id: str = ""
    requested_at: datetime = field(default_factory=datetime.utcnow)
    allocated_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    gpu_type: Optional[str] = None
    memory_per_gpu_gb: Optional[float] = None


class GPUPool:
    """Pool of GPUs for allocation."""

    def __init__(self, pool_id: str = "default"):
        self.pool_id = pool_id
        self._gpus: dict[str, GPUInfo] = {}
        self._allocations: dict[str, GPUAllocation] = {}
        self._lock = asyncio.Lock()

    async def add_gpu(self, gpu: GPUInfo) -> None:
        """Add a GPU to the pool."""
        async with self._lock:
            self._gpus[gpu.id] = gpu
            logger.info(
                "gpu_added_to_pool",
                pool_id=self.pool_id,
                gpu_id=gpu.id,
                gpu_type=gpu.gpu_type,
                memory_gb=gpu.memory_total_gb,
            )

    async def remove_gpu(self, gpu_id: str) -> Optional[GPUInfo]:
        """Remove a GPU from the pool."""
        async with self._lock:
            gpu = self._gpus.pop(gpu_id, None)
            if gpu:
                logger.info("gpu_removed_from_pool", pool_id=self.pool_id, gpu_id=gpu_id)
            return gpu

    async def get_gpu(self, gpu_id: str) -> Optional[GPUInfo]:
        """Get GPU by ID."""
        async with self._lock:
            return self._gpus.get(gpu_id)

    async def list_gpus(
        self,
        status: Optional[GPUStatus] = None,
        gpu_type: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> list[GPUInfo]:
        """List GPUs with optional filtering."""
        async with self._lock:
            gpus = list(self._gpus.values())

            if status:
                gpus = [g for g in gpus if g.status == status]
            if gpu_type:
                gpus = [g for g in gpus if g.matches_type(gpu_type)]
            if node_id:
                gpus = [g for g in gpus if g.node_id == node_id]

            return gpus

    async def get_available_count(
        self,
        gpu_type: Optional[str] = None,
        min_memory_gb: Optional[float] = None,
    ) -> int:
        """Get count of available GPUs matching criteria."""
        async with self._lock:
            count = 0
            for gpu in self._gpus.values():
                if not gpu.is_available:
                    continue
                if gpu_type and not gpu.matches_type(gpu_type):
                    continue
                if min_memory_gb and not gpu.has_memory(min_memory_gb):
                    continue
                count += 1
            return count

    async def allocate(
        self,
        job_id: str,
        count: int,
        gpu_type: Optional[str] = None,
        memory_gb: Optional[float] = None,
        node_id: Optional[str] = None,
        prefer_same_node: bool = True,
    ) -> GPUAllocation:
        """
        Allocate GPUs for a job.

        Args:
            job_id: Job requesting GPUs
            count: Number of GPUs to allocate
            gpu_type: Required GPU type
            memory_gb: Required GPU memory
            node_id: Prefer GPUs on this node
            prefer_same_node: Try to allocate all GPUs on same node

        Returns:
            GPUAllocation with allocated GPU IDs

        Raises:
            GPUNotAvailableError: If not enough GPUs available
        """
        async with self._lock:
            # Find suitable GPUs
            candidates = []
            for gpu in self._gpus.values():
                if not gpu.is_available:
                    continue
                if gpu_type and not gpu.matches_type(gpu_type):
                    continue
                if memory_gb and not gpu.has_memory(memory_gb):
                    continue
                candidates.append(gpu)

            if len(candidates) < count:
                raise GPUNotAvailableError(
                    gpu_type or "any",
                    count,
                )

            # Sort by preference
            if prefer_same_node and node_id:
                # Prefer GPUs on specified node
                candidates.sort(key=lambda g: (g.node_id != node_id, g.device_index))
            elif prefer_same_node:
                # Group by node
                from collections import Counter

                node_counts = Counter(g.node_id for g in candidates)
                # Find node with most available GPUs
                best_node = max(node_counts, key=node_counts.get) if node_counts else None
                candidates.sort(
                    key=lambda g: (g.node_id != best_node, g.device_index)
                )

            # Allocate
            selected = candidates[:count]
            now = datetime.utcnow()

            for gpu in selected:
                gpu.status = GPUStatus.ALLOCATED
                gpu.allocated_to = job_id
                gpu.allocated_at = now

            allocation = GPUAllocation(
                job_id=job_id,
                gpu_ids=[g.id for g in selected],
                node_id=selected[0].node_id if selected else "",
                allocated_at=now,
                gpu_type=gpu_type,
                memory_per_gpu_gb=memory_gb,
            )
            self._allocations[allocation.id] = allocation

            logger.info(
                "gpus_allocated",
                job_id=job_id,
                gpu_ids=allocation.gpu_ids,
                count=count,
            )

            return allocation

    async def release(self, job_id: str) -> int:
        """
        Release all GPUs allocated to a job.

        Returns:
            Number of GPUs released
        """
        async with self._lock:
            released = 0
            now = datetime.utcnow()

            for gpu in self._gpus.values():
                if gpu.allocated_to == job_id:
                    gpu.status = GPUStatus.AVAILABLE
                    gpu.allocated_to = None
                    gpu.allocated_at = None
                    released += 1

            # Update allocation records
            for allocation in self._allocations.values():
                if allocation.job_id == job_id and not allocation.released_at:
                    allocation.released_at = now

            if released > 0:
                logger.info("gpus_released", job_id=job_id, count=released)

            return released

    async def get_allocation(self, job_id: str) -> Optional[GPUAllocation]:
        """Get allocation for a job."""
        async with self._lock:
            for allocation in self._allocations.values():
                if allocation.job_id == job_id and not allocation.released_at:
                    return allocation
            return None

    async def update_gpu_stats(
        self,
        gpu_id: str,
        utilization: Optional[float] = None,
        memory_used_gb: Optional[float] = None,
        temperature: Optional[float] = None,
        power_watts: Optional[float] = None,
    ) -> bool:
        """Update GPU statistics."""
        async with self._lock:
            gpu = self._gpus.get(gpu_id)
            if not gpu:
                return False

            if utilization is not None:
                gpu.utilization_percent = utilization
            if memory_used_gb is not None:
                gpu.memory_used_gb = memory_used_gb
                gpu.memory_free_gb = gpu.memory_total_gb - memory_used_gb
            if temperature is not None:
                gpu.temperature_celsius = temperature
            if power_watts is not None:
                gpu.power_usage_watts = power_watts

            return True

    async def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        async with self._lock:
            total = len(self._gpus)
            available = sum(1 for g in self._gpus.values() if g.is_available)
            allocated = sum(
                1 for g in self._gpus.values() if g.status == GPUStatus.ALLOCATED
            )

            by_type: dict[str, dict[str, int]] = {}
            for gpu in self._gpus.values():
                if gpu.gpu_type not in by_type:
                    by_type[gpu.gpu_type] = {"total": 0, "available": 0}
                by_type[gpu.gpu_type]["total"] += 1
                if gpu.is_available:
                    by_type[gpu.gpu_type]["available"] += 1

            avg_utilization = 0.0
            utilized_gpus = [g for g in self._gpus.values() if g.status == GPUStatus.ALLOCATED]
            if utilized_gpus:
                avg_utilization = sum(g.utilization_percent for g in utilized_gpus) / len(
                    utilized_gpus
                )

            return {
                "pool_id": self.pool_id,
                "total_gpus": total,
                "available_gpus": available,
                "allocated_gpus": allocated,
                "by_type": by_type,
                "avg_utilization_percent": avg_utilization,
                "active_allocations": sum(
                    1 for a in self._allocations.values() if not a.released_at
                ),
            }


class GPUManager:
    """
    Manages GPU resources across multiple pools and nodes.

    Provides high-level GPU allocation, monitoring, and optimization.
    """

    def __init__(self):
        self._pools: dict[str, GPUPool] = {}
        self._default_pool = GPUPool("default")
        self._pools["default"] = self._default_pool
        self._lock = asyncio.Lock()
        self._topology_cache: dict[str, dict[str, Any]] = {}

    async def create_pool(self, pool_id: str) -> GPUPool:
        """Create a new GPU pool."""
        async with self._lock:
            if pool_id in self._pools:
                return self._pools[pool_id]
            pool = GPUPool(pool_id)
            self._pools[pool_id] = pool
            return pool

    async def get_pool(self, pool_id: str = "default") -> Optional[GPUPool]:
        """Get a GPU pool."""
        async with self._lock:
            return self._pools.get(pool_id)

    async def delete_pool(self, pool_id: str) -> bool:
        """Delete a GPU pool."""
        if pool_id == "default":
            return False
        async with self._lock:
            if pool_id in self._pools:
                del self._pools[pool_id]
                return True
            return False

    async def register_gpu(
        self,
        node_id: str,
        device_index: int,
        name: str,
        gpu_type: str,
        memory_gb: float,
        pool_id: str = "default",
        **kwargs: Any,
    ) -> GPUInfo:
        """Register a GPU from a node."""
        gpu = GPUInfo(
            node_id=node_id,
            device_index=device_index,
            name=name,
            gpu_type=gpu_type,
            memory_total_gb=memory_gb,
            memory_free_gb=memory_gb,
            **kwargs,
        )

        pool = await self.get_pool(pool_id)
        if pool:
            await pool.add_gpu(gpu)

        return gpu

    async def unregister_gpu(self, gpu_id: str, pool_id: str = "default") -> bool:
        """Unregister a GPU."""
        pool = await self.get_pool(pool_id)
        if pool:
            gpu = await pool.remove_gpu(gpu_id)
            return gpu is not None
        return False

    async def allocate_gpus(
        self,
        job_id: str,
        count: int,
        gpu_type: Optional[str] = None,
        memory_gb: Optional[float] = None,
        pool_id: str = "default",
        **kwargs: Any,
    ) -> GPUAllocation:
        """Allocate GPUs for a job."""
        pool = await self.get_pool(pool_id)
        if not pool:
            raise ResourceError(f"GPU pool not found: {pool_id}", "gpu")

        return await pool.allocate(
            job_id=job_id,
            count=count,
            gpu_type=gpu_type,
            memory_gb=memory_gb,
            **kwargs,
        )

    async def release_gpus(self, job_id: str, pool_id: str = "default") -> int:
        """Release GPUs for a job."""
        pool = await self.get_pool(pool_id)
        if not pool:
            return 0
        return await pool.release(job_id)

    async def get_job_gpus(
        self, job_id: str, pool_id: str = "default"
    ) -> list[GPUInfo]:
        """Get GPUs allocated to a job."""
        pool = await self.get_pool(pool_id)
        if not pool:
            return []

        allocation = await pool.get_allocation(job_id)
        if not allocation:
            return []

        gpus = []
        for gpu_id in allocation.gpu_ids:
            gpu = await pool.get_gpu(gpu_id)
            if gpu:
                gpus.append(gpu)
        return gpus

    async def get_available_gpus(
        self,
        gpu_type: Optional[str] = None,
        min_memory_gb: Optional[float] = None,
        pool_id: str = "default",
    ) -> list[GPUInfo]:
        """Get all available GPUs matching criteria."""
        pool = await self.get_pool(pool_id)
        if not pool:
            return []

        gpus = await pool.list_gpus(status=GPUStatus.AVAILABLE, gpu_type=gpu_type)
        if min_memory_gb:
            gpus = [g for g in gpus if g.has_memory(min_memory_gb)]
        return gpus

    async def get_gpu_topology(self, node_id: str) -> dict[str, Any]:
        """
        Get GPU topology for a node (NVLink connections, PCIe hierarchy).

        Returns cached topology if available.
        """
        if node_id in self._topology_cache:
            return self._topology_cache[node_id]

        # Build topology from registered GPUs
        pool = await self.get_pool("default")
        if not pool:
            return {}

        node_gpus = await pool.list_gpus(node_id=node_id)

        topology = {
            "node_id": node_id,
            "gpu_count": len(node_gpus),
            "gpus": [
                {
                    "id": g.id,
                    "device_index": g.device_index,
                    "pcie_gen": g.pcie_generation,
                    "pcie_width": g.pcie_width,
                }
                for g in sorted(node_gpus, key=lambda x: x.device_index)
            ],
            # Simplified topology - would need actual hardware info
            "nvlink_connections": [],
            "pcie_switches": [],
        }

        self._topology_cache[node_id] = topology
        return topology

    async def find_best_allocation(
        self,
        count: int,
        gpu_type: Optional[str] = None,
        memory_gb: Optional[float] = None,
        prefer_nvlink: bool = True,
    ) -> list[str]:
        """
        Find the best set of GPUs for an allocation.

        Considers:
        - GPU availability
        - NVLink connectivity for multi-GPU
        - Same-node preference
        - Memory requirements
        """
        pool = await self.get_pool("default")
        if not pool:
            return []

        available = await self.get_available_gpus(
            gpu_type=gpu_type, min_memory_gb=memory_gb
        )

        if len(available) < count:
            return []

        # Group by node
        by_node: dict[str, list[GPUInfo]] = {}
        for gpu in available:
            if gpu.node_id not in by_node:
                by_node[gpu.node_id] = []
            by_node[gpu.node_id].append(gpu)

        # Find best node (most GPUs available)
        best_node = max(by_node.keys(), key=lambda n: len(by_node[n]))

        if len(by_node[best_node]) >= count:
            # Can satisfy from single node
            gpus = sorted(by_node[best_node], key=lambda g: g.device_index)[:count]
            return [g.id for g in gpus]

        # Need GPUs from multiple nodes
        selected = []
        for node_id in sorted(by_node.keys(), key=lambda n: len(by_node[n]), reverse=True):
            for gpu in by_node[node_id]:
                selected.append(gpu.id)
                if len(selected) >= count:
                    return selected

        return selected

    async def get_stats(self) -> dict[str, Any]:
        """Get overall GPU manager statistics."""
        stats = {
            "pools": {},
            "total_gpus": 0,
            "available_gpus": 0,
            "allocated_gpus": 0,
        }

        async with self._lock:
            for pool_id, pool in self._pools.items():
                pool_stats = await pool.get_stats()
                stats["pools"][pool_id] = pool_stats
                stats["total_gpus"] += pool_stats["total_gpus"]
                stats["available_gpus"] += pool_stats["available_gpus"]
                stats["allocated_gpus"] += pool_stats["allocated_gpus"]

        return stats

    async def detect_failed_gpus(self) -> list[GPUInfo]:
        """Detect GPUs that may have failed."""
        failed = []
        pool = await self.get_pool("default")
        if not pool:
            return failed

        gpus = await pool.list_gpus()
        for gpu in gpus:
            # Check for signs of failure
            if gpu.status == GPUStatus.FAILED:
                failed.append(gpu)
            elif gpu.temperature_celsius > 90:  # Overheating
                failed.append(gpu)
            elif gpu.status == GPUStatus.ALLOCATED and gpu.utilization_percent == 0:
                # Allocated but not being used - could indicate crash
                if gpu.allocated_at:
                    age = datetime.utcnow() - gpu.allocated_at
                    if age > timedelta(minutes=5):
                        failed.append(gpu)

        return failed
