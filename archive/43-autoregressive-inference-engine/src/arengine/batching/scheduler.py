"""Continuous batching scheduler for inference."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Status of a generation request."""
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass
class Request:
    """Single generation request."""
    request_id: str
    prompt_tokens: np.ndarray
    max_new_tokens: int
    status: RequestStatus = RequestStatus.PENDING
    output_tokens: List[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    finish_time: Optional[float] = None

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_tokens)

    @property
    def output_length(self) -> int:
        return len(self.output_tokens)

    @property
    def total_length(self) -> int:
        return self.prompt_length + self.output_length

    @property
    def is_done(self) -> bool:
        return self.output_length >= self.max_new_tokens


@dataclass
class Batch:
    """Batch of requests for processing."""
    requests: List[Request]
    padded_input: np.ndarray
    attention_mask: np.ndarray

    @property
    def batch_size(self) -> int:
        return len(self.requests)


class ContinuousBatcher:
    """
    Continuous batching for efficient inference.

    Dynamically adds/removes requests from running batch.
    Implements iteration-level scheduling.
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_seq_length: int = 2048,
        max_tokens_per_batch: int = 4096
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_length = max_seq_length
        self.max_tokens_per_batch = max_tokens_per_batch

        # Request queues
        self.pending_queue: List[Request] = []
        self.running_batch: List[Request] = []
        self.finished_requests: List[Request] = []

        # Statistics
        self.total_requests = 0
        self.total_tokens_generated = 0

    def add_request(
        self,
        request_id: str,
        prompt_tokens: np.ndarray,
        max_new_tokens: int
    ):
        """Add a new request to the queue."""
        request = Request(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens
        )
        self.pending_queue.append(request)
        self.total_requests += 1
        logger.debug(f"Added request {request_id}, queue size: {len(self.pending_queue)}")

    def schedule(self) -> Optional[Batch]:
        """
        Schedule next batch for processing.

        Returns batch of requests to process, or None if nothing to do.
        """
        # Remove finished requests from running batch
        still_running = []
        for req in self.running_batch:
            if req.is_done or req.status == RequestStatus.FINISHED:
                req.status = RequestStatus.FINISHED
                req.finish_time = time.time()
                self.finished_requests.append(req)
            else:
                still_running.append(req)
        self.running_batch = still_running

        # Calculate available capacity
        current_tokens = sum(r.total_length for r in self.running_batch)
        available_slots = self.max_batch_size - len(self.running_batch)
        available_tokens = self.max_tokens_per_batch - current_tokens

        # Add pending requests
        new_requests = []
        while (
            self.pending_queue and
            len(new_requests) < available_slots
        ):
            req = self.pending_queue[0]

            # Check if request fits
            req_tokens = req.prompt_length + req.max_new_tokens
            if req_tokens > available_tokens:
                break

            # Add to batch
            self.pending_queue.pop(0)
            req.status = RequestStatus.RUNNING
            req.start_time = time.time()
            new_requests.append(req)
            available_tokens -= req_tokens

        self.running_batch.extend(new_requests)

        if not self.running_batch:
            return None

        # Create padded batch
        return self._create_batch()

    def _create_batch(self) -> Batch:
        """Create padded batch from running requests."""
        # Find max length for padding
        max_len = max(r.total_length for r in self.running_batch)

        batch_size = len(self.running_batch)
        padded_input = np.zeros((batch_size, max_len), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.float32)

        for i, req in enumerate(self.running_batch):
            # Concatenate prompt and output
            tokens = np.concatenate([
                req.prompt_tokens,
                np.array(req.output_tokens, dtype=np.int64)
            ])
            length = len(tokens)

            padded_input[i, :length] = tokens
            attention_mask[i, :length] = 1.0

        return Batch(
            requests=self.running_batch.copy(),
            padded_input=padded_input,
            attention_mask=attention_mask
        )

    def update(self, new_tokens: np.ndarray, finished_mask: np.ndarray):
        """
        Update batch with generated tokens.

        Args:
            new_tokens: Generated tokens (batch_size,)
            finished_mask: Boolean mask for finished sequences
        """
        for i, req in enumerate(self.running_batch):
            if not finished_mask[i]:
                req.output_tokens.append(int(new_tokens[i]))
                self.total_tokens_generated += 1
            else:
                req.status = RequestStatus.FINISHED

    def get_results(self, request_id: str) -> Optional[Request]:
        """Get results for a completed request."""
        for req in self.finished_requests:
            if req.request_id == request_id:
                return req
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            "total_requests": self.total_requests,
            "pending": len(self.pending_queue),
            "running": len(self.running_batch),
            "finished": len(self.finished_requests),
            "tokens_generated": self.total_tokens_generated,
        }


class PrefillDecodeScheduler:
    """
    Scheduler that separates prefill and decode phases.

    Optimizes for different compute patterns:
    - Prefill: compute-bound, process full prompt
    - Decode: memory-bound, single token per iteration
    """

    def __init__(
        self,
        max_prefill_batch: int = 8,
        max_decode_batch: int = 128,
        chunk_size: int = 512
    ):
        self.max_prefill_batch = max_prefill_batch
        self.max_decode_batch = max_decode_batch
        self.chunk_size = chunk_size

        self.prefill_queue: List[Request] = []
        self.decode_queue: List[Request] = []
        self.finished: List[Request] = []

    def add_request(self, request: Request):
        """Add request to prefill queue."""
        self.prefill_queue.append(request)

    def schedule_prefill(self) -> Optional[Batch]:
        """Schedule prefill batch."""
        if not self.prefill_queue:
            return None

        # Select requests for prefill
        batch_requests = []
        total_tokens = 0

        for req in self.prefill_queue[:self.max_prefill_batch]:
            tokens = min(req.prompt_length, self.chunk_size)
            total_tokens += tokens
            batch_requests.append(req)

        if not batch_requests:
            return None

        # Remove from prefill queue
        self.prefill_queue = self.prefill_queue[len(batch_requests):]

        # Create batch with prompts
        max_len = max(r.prompt_length for r in batch_requests)
        padded = np.zeros((len(batch_requests), max_len), dtype=np.int64)
        mask = np.zeros((len(batch_requests), max_len), dtype=np.float32)

        for i, req in enumerate(batch_requests):
            length = req.prompt_length
            padded[i, :length] = req.prompt_tokens
            mask[i, :length] = 1.0

        return Batch(batch_requests, padded, mask)

    def move_to_decode(self, requests: List[Request]):
        """Move prefilled requests to decode queue."""
        for req in requests:
            req.status = RequestStatus.RUNNING
            self.decode_queue.append(req)

    def schedule_decode(self) -> Optional[Batch]:
        """Schedule decode batch."""
        if not self.decode_queue:
            return None

        # Select requests for decode
        batch_requests = self.decode_queue[:self.max_decode_batch]

        # Create batch with last tokens
        padded = np.zeros((len(batch_requests), 1), dtype=np.int64)
        for i, req in enumerate(batch_requests):
            if req.output_tokens:
                padded[i, 0] = req.output_tokens[-1]
            else:
                padded[i, 0] = req.prompt_tokens[-1]

        return Batch(batch_requests, padded, np.ones_like(padded, dtype=np.float32))

    def update_decode(self, new_tokens: np.ndarray, finished: np.ndarray):
        """Update decode queue with generated tokens."""
        still_running = []
        for i, req in enumerate(self.decode_queue[:len(new_tokens)]):
            if not finished[i]:
                req.output_tokens.append(int(new_tokens[i]))
                still_running.append(req)
            else:
                req.status = RequestStatus.FINISHED
                self.finished.append(req)

        self.decode_queue = still_running + self.decode_queue[len(new_tokens):]


class PriorityScheduler:
    """Priority-based scheduler for SLA compliance."""

    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.high_priority: List[Request] = []
        self.normal_priority: List[Request] = []
        self.low_priority: List[Request] = []
        self.running: List[Request] = []

    def add_request(self, request: Request, priority: str = "normal"):
        """Add request with priority."""
        if priority == "high":
            self.high_priority.append(request)
        elif priority == "low":
            self.low_priority.append(request)
        else:
            self.normal_priority.append(request)

    def schedule(self) -> List[Request]:
        """Schedule requests by priority."""
        # Clear finished
        self.running = [r for r in self.running if r.status == RequestStatus.RUNNING]

        available = self.max_batch_size - len(self.running)
        new_requests = []

        # High priority first
        while self.high_priority and len(new_requests) < available:
            new_requests.append(self.high_priority.pop(0))

        # Then normal
        while self.normal_priority and len(new_requests) < available:
            new_requests.append(self.normal_priority.pop(0))

        # Then low
        while self.low_priority and len(new_requests) < available:
            new_requests.append(self.low_priority.pop(0))

        for req in new_requests:
            req.status = RequestStatus.RUNNING
        self.running.extend(new_requests)

        return self.running


class TokenBudgetScheduler:
    """
    Token budget scheduler for fair sharing.

    Allocates compute based on token budgets.
    """

    def __init__(
        self,
        max_tokens_per_step: int = 4096,
        preemption_enabled: bool = True
    ):
        self.max_tokens_per_step = max_tokens_per_step
        self.preemption_enabled = preemption_enabled
        self.requests: Dict[str, Request] = {}
        self.token_budgets: Dict[str, int] = {}

    def add_request(self, request: Request, budget: int = 1000):
        """Add request with token budget."""
        self.requests[request.request_id] = request
        self.token_budgets[request.request_id] = budget

    def schedule(self) -> List[Request]:
        """Schedule based on remaining budget."""
        # Sort by budget (highest first)
        sorted_reqs = sorted(
            self.requests.values(),
            key=lambda r: self.token_budgets.get(r.request_id, 0),
            reverse=True
        )

        scheduled = []
        tokens_used = 0

        for req in sorted_reqs:
            if req.status == RequestStatus.FINISHED:
                continue

            req_tokens = 1  # One token per decode step
            if tokens_used + req_tokens <= self.max_tokens_per_step:
                scheduled.append(req)
                tokens_used += req_tokens
                self.token_budgets[req.request_id] -= req_tokens

        return scheduled

    def preempt(self, request_id: str) -> bool:
        """Preempt a running request."""
        if not self.preemption_enabled:
            return False

        if request_id in self.requests:
            self.requests[request_id].status = RequestStatus.PENDING
            return True
        return False
