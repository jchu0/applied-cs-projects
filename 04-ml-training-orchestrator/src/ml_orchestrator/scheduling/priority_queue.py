"""Priority queue implementation for job scheduling."""

import heapq
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
import asyncio
import structlog

from ml_orchestrator.core.models import JobPriority, TrainingJob


logger = structlog.get_logger(__name__)


@dataclass(order=True)
class PriorityItem:
    """Item in priority queue with aging support."""

    effective_priority: float
    timestamp: datetime = field(compare=False)
    job_id: str = field(compare=False)
    base_priority: int = field(compare=False)


class PriorityQueue:
    """
    Thread-safe priority queue with aging for training jobs.

    Implements priority scheduling with aging to prevent starvation.
    Jobs that wait longer get higher effective priority.
    """

    def __init__(
        self,
        aging_factor: float = 0.1,
        max_age_hours: float = 24.0,
    ):
        """
        Initialize priority queue.

        Args:
            aging_factor: Priority boost per hour of waiting
            max_age_hours: Maximum age for priority calculation
        """
        self._heap: list[PriorityItem] = []
        self._job_map: dict[str, PriorityItem] = {}
        self._aging_factor = aging_factor
        self._max_age_hours = max_age_hours
        self._lock = asyncio.Lock()
        self._removed: set[str] = set()

    def _calculate_effective_priority(
        self, base_priority: int, queue_time: datetime
    ) -> float:
        """
        Calculate effective priority with aging.

        Higher values = higher priority (will be processed first).
        We use negative because heapq is a min-heap.
        """
        now = datetime.utcnow()
        age_hours = min(
            (now - queue_time).total_seconds() / 3600,
            self._max_age_hours,
        )
        age_boost = age_hours * self._aging_factor
        # Negative because heapq is min-heap - lower values come first
        return -(base_priority + age_boost * 100)

    async def push(self, job: TrainingJob) -> None:
        """Add a job to the queue."""
        async with self._lock:
            if job.id in self._job_map:
                # Update existing job
                await self._remove_internal(job.id)

            queue_time = job.queued_at or datetime.utcnow()
            effective_priority = self._calculate_effective_priority(
                job.priority.value, queue_time
            )

            item = PriorityItem(
                effective_priority=effective_priority,
                timestamp=queue_time,
                job_id=job.id,
                base_priority=job.priority.value,
            )

            heapq.heappush(self._heap, item)
            self._job_map[job.id] = item

            logger.debug(
                "job_queued",
                job_id=job.id,
                priority=job.priority.value,
                effective_priority=-effective_priority,
            )

    async def pop(self) -> Optional[str]:
        """Remove and return the highest priority job ID."""
        async with self._lock:
            while self._heap:
                item = heapq.heappop(self._heap)
                job_id = item.job_id

                if job_id in self._removed:
                    self._removed.discard(job_id)
                    continue

                if job_id in self._job_map:
                    del self._job_map[job_id]
                    return job_id

            return None

    async def peek(self) -> Optional[str]:
        """Return the highest priority job ID without removing it."""
        async with self._lock:
            for item in self._heap:
                if item.job_id not in self._removed and item.job_id in self._job_map:
                    return item.job_id
            return None

    async def _remove_internal(self, job_id: str) -> bool:
        """Internal remove without lock."""
        if job_id in self._job_map:
            del self._job_map[job_id]
            self._removed.add(job_id)
            return True
        return False

    async def remove(self, job_id: str) -> bool:
        """Remove a job from the queue."""
        async with self._lock:
            return await self._remove_internal(job_id)

    async def update_priority(self, job_id: str, new_priority: JobPriority) -> bool:
        """Update the priority of a queued job."""
        async with self._lock:
            if job_id not in self._job_map:
                return False

            old_item = self._job_map[job_id]
            await self._remove_internal(job_id)

            effective_priority = self._calculate_effective_priority(
                new_priority.value, old_item.timestamp
            )

            new_item = PriorityItem(
                effective_priority=effective_priority,
                timestamp=old_item.timestamp,
                job_id=job_id,
                base_priority=new_priority.value,
            )

            heapq.heappush(self._heap, new_item)
            self._job_map[job_id] = new_item
            # _remove_internal marked this id as removed; clear that tombstone so
            # the freshly re-pushed item isn't skipped by pop()/peek().
            self._removed.discard(job_id)

            return True

    async def refresh_priorities(self) -> None:
        """Refresh all effective priorities based on aging."""
        async with self._lock:
            if not self._job_map:
                return

            # Rebuild heap with updated priorities
            new_heap: list[PriorityItem] = []
            for job_id, item in self._job_map.items():
                effective_priority = self._calculate_effective_priority(
                    item.base_priority, item.timestamp
                )
                new_item = PriorityItem(
                    effective_priority=effective_priority,
                    timestamp=item.timestamp,
                    job_id=job_id,
                    base_priority=item.base_priority,
                )
                new_heap.append(new_item)
                self._job_map[job_id] = new_item

            heapq.heapify(new_heap)
            self._heap = new_heap
            self._removed.clear()

    async def contains(self, job_id: str) -> bool:
        """Check if job is in queue."""
        async with self._lock:
            return job_id in self._job_map

    async def size(self) -> int:
        """Get queue size."""
        async with self._lock:
            return len(self._job_map)

    async def is_empty(self) -> bool:
        """Check if queue is empty."""
        async with self._lock:
            return len(self._job_map) == 0

    async def clear(self) -> None:
        """Clear the queue."""
        async with self._lock:
            self._heap.clear()
            self._job_map.clear()
            self._removed.clear()

    async def get_all_job_ids(self) -> list[str]:
        """Get all job IDs in priority order."""
        async with self._lock:
            # Create sorted copy
            items = sorted(
                [item for item in self._job_map.values()],
                key=lambda x: x.effective_priority,
            )
            return [item.job_id for item in items]

    async def get_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        async with self._lock:
            if not self._job_map:
                return {
                    "size": 0,
                    "priority_distribution": {},
                    "avg_wait_time_seconds": 0,
                    "max_wait_time_seconds": 0,
                }

            now = datetime.utcnow()
            priority_counts: dict[int, int] = {}
            wait_times: list[float] = []

            for item in self._job_map.values():
                priority_counts[item.base_priority] = (
                    priority_counts.get(item.base_priority, 0) + 1
                )
                wait_time = (now - item.timestamp).total_seconds()
                wait_times.append(wait_time)

            return {
                "size": len(self._job_map),
                "priority_distribution": priority_counts,
                "avg_wait_time_seconds": sum(wait_times) / len(wait_times) if wait_times else 0,
                "max_wait_time_seconds": max(wait_times) if wait_times else 0,
            }


class MultiLevelQueue:
    """
    Multi-level priority queue with separate queues per priority level.

    Higher priority levels are served first, with round-robin within levels.
    """

    def __init__(self, levels: int = 5):
        self._levels = levels
        self._queues: list[list[str]] = [[] for _ in range(levels)]
        self._job_to_level: dict[str, int] = {}
        self._lock = asyncio.Lock()

    def _priority_to_level(self, priority: JobPriority) -> int:
        """Map priority to queue level."""
        if priority.value >= 100:
            return 0  # Highest priority
        elif priority.value >= 75:
            return 1
        elif priority.value >= 50:
            return 2
        elif priority.value >= 25:
            return 3
        else:
            return 4  # Lowest priority

    async def push(self, job_id: str, priority: JobPriority) -> None:
        """Add a job to the appropriate level queue."""
        async with self._lock:
            if job_id in self._job_to_level:
                # Remove from old level
                old_level = self._job_to_level[job_id]
                if job_id in self._queues[old_level]:
                    self._queues[old_level].remove(job_id)

            level = self._priority_to_level(priority)
            self._queues[level].append(job_id)
            self._job_to_level[job_id] = level

    async def pop(self) -> Optional[str]:
        """Remove and return highest priority job."""
        async with self._lock:
            for level in range(self._levels):
                if self._queues[level]:
                    job_id = self._queues[level].pop(0)
                    del self._job_to_level[job_id]
                    return job_id
            return None

    async def remove(self, job_id: str) -> bool:
        """Remove a job from the queue."""
        async with self._lock:
            if job_id not in self._job_to_level:
                return False
            level = self._job_to_level[job_id]
            if job_id in self._queues[level]:
                self._queues[level].remove(job_id)
            del self._job_to_level[job_id]
            return True

    async def size(self) -> int:
        """Get total queue size."""
        async with self._lock:
            return sum(len(q) for q in self._queues)

    async def is_empty(self) -> bool:
        """Check if all queues are empty."""
        return await self.size() == 0

    async def get_level_sizes(self) -> list[int]:
        """Get sizes of each level queue."""
        async with self._lock:
            return [len(q) for q in self._queues]
