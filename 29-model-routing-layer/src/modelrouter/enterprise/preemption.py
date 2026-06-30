"""Request preemption for priority handling."""

import logging
from typing import Any

from ..schemas import InferenceRequest, Priority
from ..scheduler.queue import QueueManager

logger = logging.getLogger(__name__)


class PreemptionManager:
    """Manages request preemption for priority handling."""

    def __init__(self, queue_manager: QueueManager):
        """Initialize preemption manager.

        Args:
            queue_manager: Queue manager
        """
        self.queue_manager = queue_manager
        self._in_flight: dict[str, InferenceRequest] = {}
        self._preempted: set[str] = set()

    def register_in_flight(self, request: InferenceRequest):
        """Register request as in-flight.

        Args:
            request: Request being processed
        """
        self._in_flight[request.request_id] = request

    def complete_request(self, request_id: str):
        """Mark request as completed.

        Args:
            request_id: Request ID
        """
        self._in_flight.pop(request_id, None)
        self._preempted.discard(request_id)

    async def maybe_preempt(
        self,
        high_priority_request: InferenceRequest,
        model: str
    ) -> bool:
        """Check if we should preempt lower priority work.

        Args:
            high_priority_request: High priority request
            model: Model name

        Returns:
            True if preemption occurred
        """
        # Only preempt for CRITICAL/HIGH
        if high_priority_request.priority.value > Priority.HIGH.value:
            return False

        # Find preemption candidates
        candidates = await self._find_preemption_candidates(
            model,
            high_priority_request.priority
        )

        if not candidates:
            return False

        # Preempt lowest priority request
        victim = max(candidates, key=lambda r: r.priority.value)
        await self._preempt(victim)

        logger.info(
            f"Preempted request {victim.request_id} "
            f"(priority {victim.priority.name}) "
            f"for {high_priority_request.request_id} "
            f"(priority {high_priority_request.priority.name})"
        )

        return True

    async def _find_preemption_candidates(
        self,
        model: str,
        priority: Priority
    ) -> list[InferenceRequest]:
        """Find requests that can be preempted.

        Args:
            model: Model name
            priority: Minimum priority to preempt

        Returns:
            List of preemption candidates
        """
        candidates = []

        for request in self._in_flight.values():
            if request.model != model:
                continue

            # Can only preempt lower priority
            if request.priority.value <= priority.value:
                continue

            # Don't preempt already preempted requests
            if request.request_id in self._preempted:
                continue

            candidates.append(request)

        return candidates

    async def _preempt(self, request: InferenceRequest):
        """Cancel and requeue preempted request.

        Args:
            request: Request to preempt
        """
        # Mark as preempted
        self._preempted.add(request.request_id)

        # Cancel execution (would signal worker in production)
        await self._cancel_execution(request.request_id)

        # Requeue with preemption marker
        request.metadata["preempted"] = True
        request.metadata["preempted_at"] = __import__("time").time()

        await self.queue_manager.enqueue(request)

        # Remove from in-flight
        self._in_flight.pop(request.request_id, None)

    async def _cancel_execution(self, request_id: str):
        """Cancel request execution.

        Args:
            request_id: Request ID
        """
        # Mock implementation - would signal worker
        pass

    def get_preempted_count(self) -> int:
        """Get count of preempted requests.

        Returns:
            Number of preempted requests
        """
        return len(self._preempted)

    def get_in_flight_count(self, model: str = None) -> int:
        """Get count of in-flight requests.

        Args:
            model: Optional model filter

        Returns:
            Number of in-flight requests
        """
        if model:
            return len([r for r in self._in_flight.values() if r.model == model])
        return len(self._in_flight)
