"""Tests for request management functionality.

This module tests:
- InferenceRequest creation and lifecycle
- RequestManager queue operations
- Priority scheduling
- Request preemption
"""

import pytest
import time
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from autoregressive_inference.requests import (
    RequestStatus,
    RequestPriority,
    SamplingParams,
    InferenceRequest,
    RequestManager,
)


class TestSamplingParams:
    """Tests for SamplingParams dataclass."""

    def test_default_params(self):
        """Test default sampling parameters."""
        params = SamplingParams()
        assert params.temperature == 1.0
        assert params.top_k == 50
        assert params.top_p == 1.0
        assert params.max_tokens == 256
        assert params.repetition_penalty == 1.0
        assert params.frequency_penalty == 0.0
        assert params.presence_penalty == 0.0

    def test_custom_params(self):
        """Test custom sampling parameters."""
        params = SamplingParams(
            temperature=0.7,
            top_k=40,
            top_p=0.9,
            max_tokens=100,
            stop_sequences=["END", "STOP"],
            repetition_penalty=1.2
        )
        assert params.temperature == 0.7
        assert params.top_k == 40
        assert params.top_p == 0.9
        assert params.max_tokens == 100
        assert params.stop_sequences == ["END", "STOP"]
        assert params.repetition_penalty == 1.2


class TestInferenceRequest:
    """Tests for InferenceRequest dataclass."""

    def test_request_creation(self, simple_request):
        """Test basic request creation."""
        assert simple_request.request_id == "test-001"
        assert simple_request.prompt == "Hello, world!"
        assert simple_request.prompt_token_ids == [1, 2, 3, 4, 5]
        assert simple_request.status == RequestStatus.PENDING
        assert simple_request.priority == RequestPriority.NORMAL

    def test_request_initial_state(self, simple_request):
        """Test initial request state."""
        assert simple_request.output_token_ids == []
        assert simple_request.start_time is None
        assert simple_request.end_time is None
        assert simple_request.prefill_tokens == 0
        assert simple_request.decode_tokens == 0

    def test_request_priority_ordering(self, default_sampling_params):
        """Test that requests are ordered by priority."""
        low = InferenceRequest(
            request_id="low",
            prompt="Low",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params,
            priority=RequestPriority.LOW
        )
        high = InferenceRequest(
            request_id="high",
            prompt="High",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params,
            priority=RequestPriority.HIGH
        )
        # Higher priority should be "less than" for min-heap ordering
        assert high < low

    def test_request_arrival_time_ordering(self, default_sampling_params):
        """Test that requests with same priority are ordered by arrival time."""
        req1 = InferenceRequest(
            request_id="first",
            prompt="First",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params,
            priority=RequestPriority.NORMAL
        )
        time.sleep(0.001)
        req2 = InferenceRequest(
            request_id="second",
            prompt="Second",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params,
            priority=RequestPriority.NORMAL
        )
        # Earlier arrival should come first (be "less than")
        assert req1 < req2

    def test_request_equality(self, default_sampling_params):
        """Test request equality based on request_id."""
        req1 = InferenceRequest(
            request_id="same-id",
            prompt="First",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params
        )
        req2 = InferenceRequest(
            request_id="same-id",
            prompt="Second",
            prompt_token_ids=[2],
            sampling_params=default_sampling_params
        )
        assert req1 == req2

    def test_request_hash(self, default_sampling_params):
        """Test request hashing."""
        req1 = InferenceRequest(
            request_id="test-id",
            prompt="Test",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params
        )
        req2 = InferenceRequest(
            request_id="test-id",
            prompt="Different",
            prompt_token_ids=[2],
            sampling_params=default_sampling_params
        )
        # Same request_id should have same hash
        assert hash(req1) == hash(req2)

        # Can be used in sets
        request_set = {req1, req2}
        assert len(request_set) == 1

    def test_get_total_tokens(self, simple_request):
        """Test total tokens calculation."""
        assert simple_request.get_total_tokens() == 5

        simple_request.output_token_ids = [10, 11, 12]
        assert simple_request.get_total_tokens() == 8

    def test_get_latency_pending(self, simple_request):
        """Test latency for pending request."""
        assert simple_request.get_latency() is None

    def test_get_latency_running(self, simple_request):
        """Test latency for running request."""
        simple_request.start_time = time.time() - 1.0
        latency = simple_request.get_latency()
        assert latency is not None
        assert latency >= 1.0

    def test_get_latency_completed(self, simple_request):
        """Test latency for completed request."""
        simple_request.start_time = time.time() - 2.0
        simple_request.end_time = time.time() - 1.0
        latency = simple_request.get_latency()
        assert latency is not None
        assert 0.9 < latency < 1.1


class TestRequestManager:
    """Tests for RequestManager."""

    def test_manager_creation(self, request_manager):
        """Test basic manager creation."""
        assert request_manager.max_queue_size == 100
        assert request_manager.total_requests == 0
        assert request_manager.rejected_requests == 0

    def test_add_request(self, request_manager, simple_request):
        """Test adding a request."""
        success = request_manager.add_request(simple_request)
        assert success
        assert request_manager.total_requests == 1

        stats = request_manager.get_stats()
        assert stats['pending'] == 1

    def test_add_request_queue_full(self, default_sampling_params):
        """Test adding request when queue is full."""
        manager = RequestManager(max_queue_size=2)

        for i in range(2):
            req = InferenceRequest(
                request_id=f"req-{i}",
                prompt="Test",
                prompt_token_ids=[1],
                sampling_params=default_sampling_params
            )
            assert manager.add_request(req)

        # Third request should fail
        req = InferenceRequest(
            request_id="req-3",
            prompt="Test",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params
        )
        assert not manager.add_request(req)
        assert manager.rejected_requests == 1

    def test_get_next_requests(self, populated_request_manager):
        """Test getting next batch of requests."""
        requests = populated_request_manager.get_next_requests(3)

        assert len(requests) == 3
        for req in requests:
            assert req.status == RequestStatus.RUNNING_PREFILL
            assert req.start_time is not None

        stats = populated_request_manager.get_stats()
        assert stats['pending'] == 2
        assert stats['running'] == 3

    def test_get_next_requests_empty_queue(self, request_manager):
        """Test getting requests from empty queue."""
        requests = request_manager.get_next_requests(5)
        assert requests == []

    def test_get_next_requests_priority_order(self, request_manager, default_sampling_params):
        """Test that requests are returned in priority order."""
        # Add requests with different priorities
        low = InferenceRequest(
            request_id="low",
            prompt="Low",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params,
            priority=RequestPriority.LOW
        )
        high = InferenceRequest(
            request_id="high",
            prompt="High",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params,
            priority=RequestPriority.HIGH
        )
        normal = InferenceRequest(
            request_id="normal",
            prompt="Normal",
            prompt_token_ids=[1],
            sampling_params=default_sampling_params,
            priority=RequestPriority.NORMAL
        )

        request_manager.add_request(low)
        request_manager.add_request(high)
        request_manager.add_request(normal)

        requests = request_manager.get_next_requests(3)

        # Should be in priority order: high, normal, low
        assert requests[0].request_id == "high"
        assert requests[1].request_id == "normal"
        assert requests[2].request_id == "low"

    def test_complete_request(self, populated_request_manager):
        """Test completing a request."""
        # First get some requests
        requests = populated_request_manager.get_next_requests(2)

        # Complete one
        completed = populated_request_manager.complete_request(requests[0].request_id)

        assert completed is not None
        assert completed.status == RequestStatus.COMPLETED
        assert completed.end_time is not None

        stats = populated_request_manager.get_stats()
        assert stats['running'] == 1
        assert stats['completed'] == 1

    def test_complete_nonexistent_request(self, request_manager):
        """Test completing a request that doesn't exist."""
        result = request_manager.complete_request("nonexistent")
        assert result is None

    def test_fail_request(self, populated_request_manager):
        """Test failing a request."""
        requests = populated_request_manager.get_next_requests(1)

        failed = populated_request_manager.fail_request(requests[0].request_id)

        assert failed is not None
        assert failed.status == RequestStatus.FAILED

    def test_preempt_request(self, populated_request_manager):
        """Test preempting a running request."""
        requests = populated_request_manager.get_next_requests(2)
        initial_pending = populated_request_manager.get_stats()['pending']

        preempted = populated_request_manager.preempt_request(requests[0].request_id)

        assert preempted is not None
        assert preempted.status == RequestStatus.PREEMPTED

        stats = populated_request_manager.get_stats()
        assert stats['pending'] == initial_pending + 1
        assert stats['running'] == 1

    def test_preempt_nonexistent_request(self, request_manager):
        """Test preempting a request that doesn't exist."""
        result = request_manager.preempt_request("nonexistent")
        assert result is None

    def test_get_request(self, populated_request_manager):
        """Test getting a request by ID."""
        # Get from pending
        request = populated_request_manager.get_request("req-000")
        assert request is not None
        assert request.request_id == "req-000"

        # Move to running
        populated_request_manager.get_next_requests(1)

        # Get from running
        request = populated_request_manager.get_request("req-000")
        assert request is not None

        # Complete and get from completed
        populated_request_manager.complete_request("req-000")
        request = populated_request_manager.get_request("req-000")
        assert request is not None
        assert request.status == RequestStatus.COMPLETED

    def test_get_request_not_found(self, request_manager):
        """Test getting a request that doesn't exist."""
        result = request_manager.get_request("nonexistent")
        assert result is None

    def test_get_running_requests(self, populated_request_manager):
        """Test getting all running requests."""
        # Initially no running
        running = populated_request_manager.get_running_requests()
        assert running == []

        # Start some
        populated_request_manager.get_next_requests(3)
        running = populated_request_manager.get_running_requests()
        assert len(running) == 3

    def test_peek_pending(self, populated_request_manager):
        """Test peeking at pending requests."""
        pending = populated_request_manager.peek_pending(max_requests=3)

        assert len(pending) == 3
        # Should still be pending (not removed)
        stats = populated_request_manager.get_stats()
        assert stats['pending'] == 5

    def test_get_stats(self, populated_request_manager):
        """Test getting queue statistics."""
        stats = populated_request_manager.get_stats()

        assert stats['pending'] == 5
        assert stats['running'] == 0
        assert stats['completed'] == 0
        assert stats['total'] == 5
        assert stats['rejected'] == 0

    def test_clear_completed(self, populated_request_manager):
        """Test clearing completed requests."""
        # Complete some requests
        requests = populated_request_manager.get_next_requests(3)
        for req in requests:
            populated_request_manager.complete_request(req.request_id)

        # Clear all
        cleared = populated_request_manager.clear_completed()
        assert cleared == 3

        stats = populated_request_manager.get_stats()
        assert stats['completed'] == 0

    def test_clear_completed_older_than(self, populated_request_manager):
        """Test clearing completed requests older than timestamp."""
        requests = populated_request_manager.get_next_requests(2)

        # Complete first request
        populated_request_manager.complete_request(requests[0].request_id)
        time.sleep(0.01)  # Ensure cutoff is strictly after first completion
        cutoff = time.time()
        time.sleep(0.01)

        # Complete second request after cutoff
        populated_request_manager.complete_request(requests[1].request_id)

        # Clear only older ones
        cleared = populated_request_manager.clear_completed(older_than=cutoff)
        assert cleared == 1

        stats = populated_request_manager.get_stats()
        assert stats['completed'] == 1


class TestRequestManagerConcurrency:
    """Tests for thread-safety of RequestManager."""

    def test_concurrent_add_requests(self, default_sampling_params):
        """Test concurrent request additions."""
        import threading

        manager = RequestManager(max_queue_size=1000)
        errors = []

        def add_requests(thread_id):
            try:
                for i in range(100):
                    req = InferenceRequest(
                        request_id=f"thread-{thread_id}-req-{i}",
                        prompt="Test",
                        prompt_token_ids=[1],
                        sampling_params=default_sampling_params
                    )
                    manager.add_request(req)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=add_requests, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert manager.total_requests == 1000

    def test_concurrent_get_and_complete(self, default_sampling_params):
        """Test concurrent getting and completing requests."""
        import threading

        manager = RequestManager(max_queue_size=1000)
        completed_count = [0]
        errors = []
        lock = threading.Lock()

        # Add requests
        for i in range(100):
            req = InferenceRequest(
                request_id=f"req-{i}",
                prompt="Test",
                prompt_token_ids=[1],
                sampling_params=default_sampling_params
            )
            manager.add_request(req)

        def process_requests():
            try:
                while True:
                    requests = manager.get_next_requests(1)
                    if not requests:
                        break
                    for req in requests:
                        result = manager.complete_request(req.request_id)
                        if result:
                            with lock:
                                completed_count[0] += 1
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(target=process_requests)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert completed_count[0] == 100


class TestRequestStatus:
    """Tests for RequestStatus enum."""

    def test_all_statuses(self):
        """Test all request statuses exist."""
        assert RequestStatus.PENDING.value == "pending"
        assert RequestStatus.RUNNING_PREFILL.value == "running_prefill"
        assert RequestStatus.RUNNING_DECODE.value == "running_decode"
        assert RequestStatus.COMPLETED.value == "completed"
        assert RequestStatus.FAILED.value == "failed"
        assert RequestStatus.PREEMPTED.value == "preempted"


class TestRequestPriority:
    """Tests for RequestPriority enum."""

    def test_priority_values(self):
        """Test priority ordering values."""
        assert RequestPriority.LOW.value < RequestPriority.NORMAL.value
        assert RequestPriority.NORMAL.value < RequestPriority.HIGH.value
        assert RequestPriority.HIGH.value < RequestPriority.CRITICAL.value

    def test_priority_comparisons(self):
        """Test priority can be compared."""
        assert RequestPriority.HIGH.value > RequestPriority.LOW.value
