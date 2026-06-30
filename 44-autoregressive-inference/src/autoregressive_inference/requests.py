"""Request management for autoregressive inference.

This module handles incoming inference requests with priority scheduling,
queue management, and request lifecycle tracking.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
import time
import heapq
import threading


class RequestStatus(Enum):
    """Status of an inference request."""
    PENDING = "pending"
    RUNNING_PREFILL = "running_prefill"
    RUNNING_DECODE = "running_decode"
    COMPLETED = "completed"
    FAILED = "failed"
    PREEMPTED = "preempted"


class RequestPriority(Enum):
    """Priority levels for requests."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class SamplingParams:
    """Parameters for token sampling."""
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    max_tokens: int = 256
    stop_sequences: List[str] = field(default_factory=list)
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass
class InferenceRequest:
    """A single inference request."""
    request_id: str
    prompt: str
    prompt_token_ids: List[int]
    sampling_params: SamplingParams
    priority: RequestPriority = RequestPriority.NORMAL

    # Runtime state
    status: RequestStatus = RequestStatus.PENDING
    output_token_ids: List[int] = field(default_factory=list)

    # Timing
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    # KV cache info
    kv_cache_block_ids: List[int] = field(default_factory=list)

    # Metrics
    prefill_tokens: int = 0
    decode_tokens: int = 0

    def __lt__(self, other):
        """For priority queue ordering - higher priority comes first."""
        return (self.priority.value, -self.arrival_time) > \
               (other.priority.value, -other.arrival_time)

    def __hash__(self):
        """Make request hashable by request_id."""
        return hash(self.request_id)

    def __eq__(self, other):
        """Equality based on request_id."""
        if not isinstance(other, InferenceRequest):
            return False
        return self.request_id == other.request_id

    def get_total_tokens(self) -> int:
        """Get total tokens generated so far."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def get_latency(self) -> Optional[float]:
        """Get request latency if completed."""
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def get_time_to_first_token(self) -> Optional[float]:
        """Get time from start to first generated token."""
        if self.start_time is None:
            return None
        if not self.output_token_ids:
            return None
        # Approximate: prefill time
        return self.start_time - self.arrival_time


class RequestManager:
    """Manage incoming requests with priority scheduling."""

    def __init__(self, max_queue_size: int = 1000):
        """Initialize request manager.

        Args:
            max_queue_size: Maximum number of pending requests.
        """
        self.max_queue_size = max_queue_size
        self.pending_queue: List[InferenceRequest] = []  # Min-heap
        self.running_requests: Dict[str, InferenceRequest] = {}
        self.completed_requests: Dict[str, InferenceRequest] = {}
        self._lock = threading.Lock()

        # Metrics
        self.total_requests = 0
        self.rejected_requests = 0

    def add_request(self, request: InferenceRequest) -> bool:
        """Add a new request to the queue.

        Args:
            request: The inference request to add.

        Returns:
            True if added successfully, False if queue is full.
        """
        with self._lock:
            if len(self.pending_queue) >= self.max_queue_size:
                self.rejected_requests += 1
                return False

            heapq.heappush(self.pending_queue, request)
            self.total_requests += 1
            return True

    def get_next_requests(self, max_requests: int) -> List[InferenceRequest]:
        """Get next batch of requests to process.

        Args:
            max_requests: Maximum number of requests to return.

        Returns:
            List of requests to process.
        """
        with self._lock:
            requests = []
            while self.pending_queue and len(requests) < max_requests:
                request = heapq.heappop(self.pending_queue)
                request.status = RequestStatus.RUNNING_PREFILL
                request.start_time = time.time()
                self.running_requests[request.request_id] = request
                requests.append(request)
            return requests

    def peek_pending(self, max_requests: int = 10) -> List[InferenceRequest]:
        """Peek at pending requests without removing them.

        Args:
            max_requests: Maximum number of requests to return.

        Returns:
            List of pending requests (copies).
        """
        with self._lock:
            # Return copies of the first N requests (heap is partially sorted)
            return list(sorted(self.pending_queue))[:max_requests]

    def complete_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Mark a request as completed.

        Args:
            request_id: The ID of the request to complete.

        Returns:
            The completed request, or None if not found.
        """
        with self._lock:
            if request_id in self.running_requests:
                request = self.running_requests.pop(request_id)
                request.status = RequestStatus.COMPLETED
                request.end_time = time.time()
                self.completed_requests[request_id] = request
                return request
            return None

    def fail_request(self, request_id: str, reason: str = "") -> Optional[InferenceRequest]:
        """Mark a request as failed.

        Args:
            request_id: The ID of the request that failed.
            reason: Optional reason for failure.

        Returns:
            The failed request, or None if not found.
        """
        with self._lock:
            if request_id in self.running_requests:
                request = self.running_requests.pop(request_id)
                request.status = RequestStatus.FAILED
                request.end_time = time.time()
                self.completed_requests[request_id] = request
                return request
            return None

    def preempt_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Preempt a running request back to queue.

        Args:
            request_id: The ID of the request to preempt.

        Returns:
            The preempted request, or None if not found.
        """
        with self._lock:
            if request_id in self.running_requests:
                request = self.running_requests.pop(request_id)
                request.status = RequestStatus.PREEMPTED
                heapq.heappush(self.pending_queue, request)
                return request
            return None

    def get_request(self, request_id: str) -> Optional[InferenceRequest]:
        """Get a request by ID from any queue.

        Args:
            request_id: The ID of the request.

        Returns:
            The request, or None if not found.
        """
        with self._lock:
            if request_id in self.running_requests:
                return self.running_requests[request_id]
            if request_id in self.completed_requests:
                return self.completed_requests[request_id]
            for req in self.pending_queue:
                if req.request_id == request_id:
                    return req
            return None

    def get_running_requests(self) -> List[InferenceRequest]:
        """Get list of all running requests.

        Returns:
            List of running requests.
        """
        with self._lock:
            return list(self.running_requests.values())

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary of statistics.
        """
        with self._lock:
            return {
                'pending': len(self.pending_queue),
                'running': len(self.running_requests),
                'completed': len(self.completed_requests),
                'total': self.total_requests,
                'rejected': self.rejected_requests
            }

    def clear_completed(self, older_than: Optional[float] = None) -> int:
        """Clear completed requests.

        Args:
            older_than: Only clear requests older than this timestamp.

        Returns:
            Number of requests cleared.
        """
        with self._lock:
            if older_than is None:
                count = len(self.completed_requests)
                self.completed_requests.clear()
                return count

            to_remove = []
            for req_id, req in self.completed_requests.items():
                if req.end_time and req.end_time < older_than:
                    to_remove.append(req_id)

            for req_id in to_remove:
                del self.completed_requests[req_id]

            return len(to_remove)
