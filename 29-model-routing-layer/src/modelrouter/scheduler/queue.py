"""Priority queue management for model routing."""

import heapq
import asyncio
import time
from typing import Any

from ..schemas import InferenceRequest, QueuedRequest


class QueueManager:
    """Manages priority queues per model."""

    def __init__(self):
        """Initialize queue manager."""
        self.queues: dict[str, list[QueuedRequest]] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def enqueue(self, request: InferenceRequest) -> str:
        """Add request to appropriate queue.

        Args:
            request: Inference request

        Returns:
            Request ID
        """
        model = request.model

        if model not in self.queues:
            self.queues[model] = []
            self.locks[model] = asyncio.Lock()

        # Compute priority score (lower = higher priority)
        score = self._compute_priority_score(request)

        queued = QueuedRequest(
            request=request,
            enqueue_time=time.time(),
            priority_score=score
        )

        async with self.locks[model]:
            heapq.heappush(self.queues[model], queued)

        return request.request_id

    async def dequeue(self, model: str) -> InferenceRequest | None:
        """Get highest priority request for model.

        Args:
            model: Model name

        Returns:
            Highest priority request or None
        """
        if model not in self.queues:
            return None

        async with self.locks[model]:
            if not self.queues[model]:
                return None

            queued = heapq.heappop(self.queues[model])
            return queued.request

    async def peek(self, model: str) -> InferenceRequest | None:
        """Peek at highest priority request without removing.

        Args:
            model: Model name

        Returns:
            Highest priority request or None
        """
        if model not in self.queues or not self.queues[model]:
            return None
        return self.queues[model][0].request

    async def remove(self, request_id: str, model: str) -> bool:
        """Remove specific request from queue.

        Args:
            request_id: Request ID
            model: Model name

        Returns:
            True if removed
        """
        if model not in self.queues:
            return False

        async with self.locks[model]:
            original_len = len(self.queues[model])
            self.queues[model] = [
                q for q in self.queues[model]
                if q.request.request_id != request_id
            ]
            heapq.heapify(self.queues[model])
            return len(self.queues[model]) < original_len

    def _compute_priority_score(self, request: InferenceRequest) -> float:
        """Compute priority score for queue ordering.

        Args:
            request: Inference request

        Returns:
            Priority score (lower = higher priority)
        """
        # Base priority (0-4)
        base = request.priority.value

        # Time factor (older requests get priority boost)
        age = time.time() - request.created_at
        age_bonus = -min(age / 60, 1.0)  # Up to -1 for 60s wait

        # SLA urgency
        if request.sla_deadline_ms:
            remaining = request.sla_deadline_ms - (time.time() - request.created_at) * 1000
            urgency = -max(0, 1 - remaining / request.sla_deadline_ms)
        else:
            urgency = 0

        return base + age_bonus + urgency

    async def get_queue_stats(self, model: str) -> dict[str, Any]:
        """Get queue statistics.

        Args:
            model: Model name

        Returns:
            Queue statistics
        """
        if model not in self.queues:
            return {"depth": 0, "oldest_ms": 0, "by_priority": {}}

        queue = self.queues[model]
        if not queue:
            return {"depth": 0, "oldest_ms": 0, "by_priority": {}}

        oldest = min(q.enqueue_time for q in queue)

        return {
            "depth": len(queue),
            "oldest_ms": int((time.time() - oldest) * 1000),
            "by_priority": self._count_by_priority(queue)
        }

    def _count_by_priority(self, queue: list[QueuedRequest]) -> dict[str, int]:
        """Count requests by priority.

        Args:
            queue: Queue to count

        Returns:
            Count by priority name
        """
        counts = {}
        for q in queue:
            name = q.request.priority.name
            counts[name] = counts.get(name, 0) + 1
        return counts

    async def get_all_stats(self) -> dict[str, dict]:
        """Get statistics for all queues.

        Returns:
            Stats by model
        """
        stats = {}
        for model in self.queues:
            stats[model] = await self.get_queue_stats(model)
        return stats
