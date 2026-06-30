"""Human-in-the-loop (HITL) review for workflows.

A ``human_review`` node pauses a running workflow until a human approves or
rejects it. The :class:`HumanReviewStore` holds pending requests and the
``asyncio`` events that the executing node awaits; a UI or REST API resolves
them out-of-band (see ``aiworkflow.api``). For tests/CI the store can
auto-approve so flows don't block.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from ..nodes.base import BaseNodeExecutor
from ..schemas import Node, generate_id


@dataclass
class ReviewRequest:
    """A pending (or resolved) human-review request."""

    id: str
    node_id: str
    payload: dict
    prompt: str = ""
    run_id: Optional[str] = None
    status: str = "pending"  # pending | approved | rejected
    reviewer: Optional[str] = None
    comment: str = ""
    decision_data: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "run_id": self.run_id,
            "prompt": self.prompt,
            "payload": self.payload,
            "status": self.status,
            "reviewer": self.reviewer,
            "comment": self.comment,
            "decision_data": self.decision_data,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


class HumanReviewStore:
    """In-memory store of human-review requests and their resolution events."""

    def __init__(self, auto_approve: bool = False):
        """Initialize the store.

        Args:
            auto_approve: If True, every created review resolves immediately as
                approved (useful for tests and non-interactive runs).
        """
        self.auto_approve = auto_approve
        self._reviews: dict[str, ReviewRequest] = {}
        self._events: dict[str, asyncio.Event] = {}

    def create_review(
        self,
        node_id: str,
        payload: dict,
        prompt: str = "",
        run_id: Optional[str] = None,
    ) -> ReviewRequest:
        """Create a pending review request (auto-approved if configured)."""
        review_id = generate_id()
        request = ReviewRequest(
            id=review_id, node_id=node_id, payload=payload, prompt=prompt, run_id=run_id
        )
        self._reviews[review_id] = request
        self._events[review_id] = asyncio.Event()
        if self.auto_approve:
            self.approve(review_id, reviewer="auto", comment="auto-approved")
        return request

    def get(self, review_id: str) -> Optional[ReviewRequest]:
        return self._reviews.get(review_id)

    def list_pending(self) -> list[ReviewRequest]:
        return [r for r in self._reviews.values() if r.status == "pending"]

    def list_all(self) -> list[ReviewRequest]:
        return list(self._reviews.values())

    def approve(
        self, review_id: str, reviewer: Optional[str] = None, comment: str = "", data: dict = None
    ) -> ReviewRequest:
        """Approve a pending review and wake any node awaiting it."""
        return self._resolve(review_id, "approved", reviewer, comment, data)

    def reject(
        self, review_id: str, reviewer: Optional[str] = None, comment: str = "", data: dict = None
    ) -> ReviewRequest:
        """Reject a pending review and wake any node awaiting it."""
        return self._resolve(review_id, "rejected", reviewer, comment, data)

    def _resolve(self, review_id, status, reviewer, comment, data) -> ReviewRequest:
        request = self._reviews.get(review_id)
        if request is None:
            raise KeyError(f"No review request: {review_id}")
        if request.status != "pending":
            raise ValueError(f"Review {review_id} already resolved as {request.status}")
        request.status = status
        request.reviewer = reviewer
        request.comment = comment
        request.decision_data = data or {}
        request.resolved_at = datetime.utcnow()
        event = self._events.get(review_id)
        if event is not None:
            event.set()
        return request

    async def wait_for(self, review_id: str, timeout: Optional[float] = None) -> ReviewRequest:
        """Block until the review is resolved (or ``timeout`` seconds elapse)."""
        event = self._events.get(review_id)
        if event is None:
            raise KeyError(f"No review request: {review_id}")
        if timeout is not None:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        else:
            await event.wait()
        return self._reviews[review_id]


class HumanReviewNodeExecutor(BaseNodeExecutor):
    """Executor for ``human_review`` nodes — pauses the flow pending a decision.

    Config (``node.config.extra``):
        prompt: text shown to the reviewer.
        timeout_seconds: max time to wait for a decision (None = wait forever).
        fail_on_reject: if True (default), a rejection fails the node (and run);
            if False, the node completes with ``approved=False``.
    """

    def __init__(self, store: Optional[HumanReviewStore] = None, default_timeout: Optional[float] = None):
        self.store = store or HumanReviewStore()
        self.default_timeout = default_timeout

    async def execute(self, node: Node, inputs: dict) -> Any:
        config = node.config
        extra = config.extra if hasattr(config, "extra") else (config if isinstance(config, dict) else {})
        prompt = extra.get("prompt", "")
        fail_on_reject = extra.get("fail_on_reject", True)

        # `timeout_seconds` is a first-class NodeConfig field, so the parser
        # stores it on config.timeout_seconds (not in extra). Prefer an explicit
        # extra override, then the NodeConfig field, then the executor default.
        # A non-positive value means "wait indefinitely".
        timeout = extra.get("timeout_seconds")
        if timeout is None:
            timeout = getattr(config, "timeout_seconds", None)
        if timeout is None:
            timeout = self.default_timeout
        if timeout is not None and timeout <= 0:
            timeout = None

        request = self.store.create_review(
            node_id=node.id, payload=inputs, prompt=prompt, run_id=inputs.get("__run_id__")
        )

        try:
            resolved = await self.store.wait_for(request.id, timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Human review timed out for node '{node.id}' after {timeout}s"
            ) from exc

        approved = resolved.status == "approved"
        if fail_on_reject and not approved:
            raise RuntimeError(
                f"Human review rejected for node '{node.id}': {resolved.comment}"
            )

        return {
            "approved": approved,
            "status": resolved.status,
            "reviewer": resolved.reviewer,
            "comment": resolved.comment,
            "decision_data": resolved.decision_data,
            "payload": inputs,
        }
