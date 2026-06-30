"""Tests for inference scheduler functionality.

This module tests:
- InferenceScheduler initialization and configuration
- Step execution (prefill and decode phases)
- Request lifecycle management
- Statistics and metrics
"""

import pytest
import time
import sys
from pathlib import Path

# Add src and tests to path
src_path = Path(__file__).parent.parent / "src"
tests_path = Path(__file__).parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(tests_path))

from autoregressive_inference.scheduler import InferenceScheduler
from autoregressive_inference.requests import (
    InferenceRequest,
    RequestManager,
    RequestStatus,
    RequestPriority,
    SamplingParams,
)
from autoregressive_inference.kv_cache import KVCacheConfig, PagedKVCacheManager
from autoregressive_inference.batching import ContinuousBatcher
from autoregressive_inference.sampling import TokenSampler

from conftest import create_request


class TestSchedulerCreation:
    """Tests for InferenceScheduler initialization."""

    def test_scheduler_creation_minimal(self):
        """Test minimal scheduler creation."""
        scheduler = InferenceScheduler()

        assert scheduler.model is None
        assert scheduler.kv_cache is None
        assert scheduler.request_manager is not None
        assert scheduler.batcher is not None
        assert scheduler.sampler is not None

    def test_scheduler_with_components(self, basic_scheduler):
        """Test scheduler with all components."""
        assert basic_scheduler.kv_cache is not None
        assert basic_scheduler.request_manager is not None
        assert basic_scheduler.batcher is not None
        assert basic_scheduler.sampler is not None

    def test_scheduler_eos_token(self):
        """Test scheduler with custom EOS token."""
        scheduler = InferenceScheduler(eos_token_id=50256)
        assert scheduler.eos_token_id == 50256


class TestAddRequest:
    """Tests for adding requests to scheduler."""

    def test_add_request(self, basic_scheduler, simple_request):
        """Test adding a single request."""
        success = basic_scheduler.add_request(simple_request)
        assert success

        stats = basic_scheduler.request_manager.get_stats()
        assert stats['pending'] == 1

    def test_add_multiple_requests(self, basic_scheduler, default_sampling_params):
        """Test adding multiple requests."""
        for i in range(5):
            req = create_request(f"req-{i}")
            basic_scheduler.add_request(req)

        stats = basic_scheduler.request_manager.get_stats()
        assert stats['pending'] == 5


class TestSchedulerStep:
    """Tests for scheduler step execution."""

    def test_step_empty(self, basic_scheduler):
        """Test step with no requests."""
        completed = basic_scheduler.step()
        assert completed == []
        assert basic_scheduler.get_step_count() == 1

    def test_step_with_pending(self, basic_scheduler, default_sampling_params):
        """Test step with pending requests."""
        # Add requests with max_tokens=1 so they complete immediately
        for i in range(3):
            req = InferenceRequest(
                request_id=f"req-{i}",
                prompt="Test",
                prompt_token_ids=[1, 2, 3],
                sampling_params=SamplingParams(max_tokens=1)
            )
            basic_scheduler.add_request(req)

        # First step: prefill
        completed = basic_scheduler.step()
        # May complete if max_tokens=1

        # Check status
        stats = basic_scheduler.request_manager.get_stats()
        assert stats['running'] + stats['completed'] > 0

    def test_step_prefill_to_decode(self, basic_scheduler, default_sampling_params):
        """Test that requests transition from prefill to decode."""
        req = create_request("test-req")
        basic_scheduler.add_request(req)

        # First step should do prefill
        basic_scheduler.step()

        # Request should be in decode state
        request = basic_scheduler.request_manager.get_request("test-req")
        # Either running decode or completed (if max_tokens=1)
        assert request.status in [RequestStatus.RUNNING_DECODE, RequestStatus.COMPLETED]

    def test_step_increments_counter(self, basic_scheduler):
        """Test that step counter increments."""
        assert basic_scheduler.get_step_count() == 0

        basic_scheduler.step()
        assert basic_scheduler.get_step_count() == 1

        basic_scheduler.step()
        assert basic_scheduler.get_step_count() == 2


class TestRequestCompletion:
    """Tests for request completion logic."""

    def test_completion_max_tokens(self, basic_scheduler):
        """Test completion when max_tokens reached."""
        req = InferenceRequest(
            request_id="max-tokens-test",
            prompt="Test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=2)
        )
        basic_scheduler.add_request(req)

        # Run steps until completion
        for _ in range(10):
            completed = basic_scheduler.step()
            if completed:
                break

        request = basic_scheduler.request_manager.get_request("max-tokens-test")
        assert request.status == RequestStatus.COMPLETED

    def test_completion_eos_token(self, basic_scheduler):
        """Test completion when EOS token generated."""
        # This would require a model that generates EOS
        # For now, test the _should_stop method directly
        req = create_request("eos-test")
        req.sampling_params.max_tokens = 1000  # High limit

        # Simulate generation
        req.output_token_ids = [10, 11, 12]

        # Check stopping conditions
        assert not basic_scheduler._should_stop(req, 100)  # Normal token
        assert basic_scheduler._should_stop(req, basic_scheduler.eos_token_id)  # EOS


class TestSchedulerRun:
    """Tests for scheduler run loop."""

    def test_run_with_max_steps(self, basic_scheduler, default_sampling_params):
        """Test run with maximum steps limit."""
        for i in range(3):
            req = create_request(f"req-{i}")
            basic_scheduler.add_request(req)

        basic_scheduler.run(max_steps=5)

        assert basic_scheduler.get_step_count() == 5

    def test_run_until_completion(self, basic_scheduler):
        """Test run until all requests complete."""
        # Add requests that will complete quickly
        for i in range(2):
            req = InferenceRequest(
                request_id=f"req-{i}",
                prompt="Test",
                prompt_token_ids=[1],
                sampling_params=SamplingParams(max_tokens=1)
            )
            basic_scheduler.add_request(req)

        basic_scheduler.run(max_steps=100)

        stats = basic_scheduler.request_manager.get_stats()
        assert stats['completed'] == 2
        assert stats['pending'] == 0
        assert stats['running'] == 0

    def test_stop(self, basic_scheduler):
        """Test stopping the scheduler."""
        basic_scheduler.running = True
        basic_scheduler.stop()
        assert not basic_scheduler.running


class TestSchedulerStats:
    """Tests for scheduler statistics."""

    def test_get_stats_initial(self, basic_scheduler):
        """Test initial statistics."""
        stats = basic_scheduler.get_stats()

        assert stats['pending'] == 0
        assert stats['running'] == 0
        assert stats['completed'] == 0
        assert stats['step_count'] == 0
        assert stats['total_prefill_tokens'] == 0
        assert stats['total_decode_tokens'] == 0

    def test_get_stats_after_steps(self, basic_scheduler):
        """Test statistics after processing."""
        for i in range(3):
            req = InferenceRequest(
                request_id=f"req-{i}",
                prompt="Test",
                prompt_token_ids=[1, 2, 3],
                sampling_params=SamplingParams(max_tokens=2)
            )
            basic_scheduler.add_request(req)

        # Run until completion
        basic_scheduler.run(max_steps=50)

        stats = basic_scheduler.get_stats()

        assert stats['step_count'] > 0
        assert stats['total_prefill_tokens'] > 0
        assert stats['completed'] == 3


class TestWaitForCompletion:
    """Tests for wait_for_completion method."""

    def test_wait_for_completion(self, basic_scheduler):
        """Test waiting for a specific request."""
        req = InferenceRequest(
            request_id="wait-test",
            prompt="Test",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=1)
        )
        basic_scheduler.add_request(req)

        completed_req = basic_scheduler.wait_for_completion("wait-test", timeout=5.0)

        assert completed_req is not None
        assert completed_req.status == RequestStatus.COMPLETED

    def test_wait_for_completion_timeout(self):
        """Test wait with timeout."""
        scheduler = InferenceScheduler()

        # Don't add any requests - should timeout
        result = scheduler.wait_for_completion("nonexistent", timeout=0.1)
        assert result is None


class TestPreemption:
    """Tests for request preemption."""

    def test_preempt_for_memory(self, basic_scheduler, default_sampling_params):
        """Test preemption when memory is needed."""
        # This tests the internal method
        # First add and start some requests
        for i in range(3):
            req = create_request(f"req-{i}")
            basic_scheduler.add_request(req)

        # Get them running
        basic_scheduler.step()

        # Simulate needing to preempt
        new_req = create_request("new-req")
        result = basic_scheduler._preempt_for_memory(new_req)

        # Should have preempted one
        assert result is True


class TestKVCacheIntegration:
    """Tests for KV cache integration with scheduler."""

    def test_kv_cache_allocation(self, basic_scheduler, default_sampling_params):
        """Test that KV cache is allocated for requests."""
        req = create_request("cache-test", prompt_length=10)
        basic_scheduler.add_request(req)

        # Run prefill
        basic_scheduler.step()

        # Check cache allocation
        blocks = basic_scheduler.kv_cache.get_blocks_for_request("cache-test")
        assert len(blocks) > 0

    def test_kv_cache_freed_on_completion(self, basic_scheduler):
        """Test that KV cache is freed when request completes."""
        req = InferenceRequest(
            request_id="free-test",
            prompt="Test",
            prompt_token_ids=[1, 2],
            sampling_params=SamplingParams(max_tokens=1)
        )
        basic_scheduler.add_request(req)

        initial_free = len(basic_scheduler.kv_cache.free_blocks)

        # Run until completion
        basic_scheduler.run(max_steps=50)

        # Cache should be freed
        final_free = len(basic_scheduler.kv_cache.free_blocks)
        assert final_free == initial_free


class TestSchedulerWithoutModel:
    """Tests for scheduler operating without a model (mock mode)."""

    def test_mock_prefill(self, basic_scheduler, default_sampling_params):
        """Test mock prefill execution."""
        req = create_request("mock-prefill")
        basic_scheduler.add_request(req)

        basic_scheduler.step()

        request = basic_scheduler.request_manager.get_request("mock-prefill")
        assert request.status == RequestStatus.RUNNING_DECODE
        assert request.prefill_tokens > 0

    def test_mock_decode(self, basic_scheduler, default_sampling_params):
        """Test mock decode execution."""
        req = create_request("mock-decode")
        basic_scheduler.add_request(req)

        # First step: prefill
        basic_scheduler.step()

        # Second step: decode
        basic_scheduler.step()

        request = basic_scheduler.request_manager.get_request("mock-decode")
        assert len(request.output_token_ids) >= 1


class TestTokenGeneration:
    """Tests for token generation tracking."""

    def test_prefill_token_tracking(self, basic_scheduler):
        """Test prefill token counting."""
        req = InferenceRequest(
            request_id="prefill-track",
            prompt="Test",
            prompt_token_ids=[1, 2, 3, 4, 5],  # 5 tokens
            sampling_params=SamplingParams(max_tokens=10)
        )
        basic_scheduler.add_request(req)

        basic_scheduler.step()

        stats = basic_scheduler.get_stats()
        assert stats['total_prefill_tokens'] == 5

    def test_decode_token_tracking(self, basic_scheduler):
        """Test decode token counting."""
        req = InferenceRequest(
            request_id="decode-track",
            prompt="Test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=5)
        )
        basic_scheduler.add_request(req)

        # Run until completion
        basic_scheduler.run(max_steps=50)

        stats = basic_scheduler.get_stats()
        assert stats['total_decode_tokens'] > 0


class TestSchedulerEdgeCases:
    """Tests for edge cases in scheduler."""

    def test_empty_prompt(self, basic_scheduler):
        """Test request with empty prompt."""
        req = InferenceRequest(
            request_id="empty-prompt",
            prompt="",
            prompt_token_ids=[],
            sampling_params=SamplingParams(max_tokens=5)
        )
        basic_scheduler.add_request(req)

        # Should not crash
        basic_scheduler.step()

    def test_single_token_prompt(self, basic_scheduler):
        """Test request with single token prompt."""
        req = InferenceRequest(
            request_id="single-token",
            prompt="A",
            prompt_token_ids=[1],
            sampling_params=SamplingParams(max_tokens=5)
        )
        basic_scheduler.add_request(req)

        basic_scheduler.run(max_steps=20)

        request = basic_scheduler.request_manager.get_request("single-token")
        assert request.status == RequestStatus.COMPLETED

    def test_zero_max_tokens(self, basic_scheduler):
        """Test request with max_tokens=0."""
        req = InferenceRequest(
            request_id="zero-tokens",
            prompt="Test",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=0)
        )
        basic_scheduler.add_request(req)

        basic_scheduler.run(max_steps=10)

        request = basic_scheduler.request_manager.get_request("zero-tokens")
        assert request.status == RequestStatus.COMPLETED
        assert len(request.output_token_ids) == 0


class TestSchedulerConcurrency:
    """Tests for concurrent scheduler operations."""

    def test_concurrent_request_addition(self, basic_scheduler, default_sampling_params):
        """Test adding requests while running."""
        import threading

        errors = []

        def add_requests():
            try:
                for i in range(10):
                    req = create_request(f"concurrent-{i}")
                    basic_scheduler.add_request(req)
                    time.sleep(0.01)
            except Exception as e:
                errors.append(e)

        def run_scheduler():
            try:
                basic_scheduler.run(max_steps=100)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=add_requests)
        t2 = threading.Thread(target=run_scheduler)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

        assert len(errors) == 0


class TestSchedulerWithSpeculative:
    """Tests for InferenceScheduler with speculative decoding integration."""

    @pytest.fixture
    def mock_speculative_decoder(self):
        """Create a mock speculative decoder."""
        from autoregressive_inference.speculative import (
            SpeculativeDecoder,
            SpeculativeStats,
        )

        class MockSpeculativeDecoder:
            """Mock speculative decoder for testing."""

            def __init__(self):
                self.stats = SpeculativeStats()
                self._call_count = 0

            def decode_step(self, context, sampling_params, request_id):
                """Return mock accepted tokens."""
                self._call_count += 1
                # Simulate accepting 2 tokens per step
                self.stats.total_drafted += 3
                self.stats.total_accepted += 2
                return [100, 101]  # Mock token IDs

        return MockSpeculativeDecoder()

    def test_scheduler_with_speculative_disabled(self, mock_speculative_decoder):
        """Test scheduler with speculative decoder but disabled."""
        scheduler = InferenceScheduler(
            speculative_decoder=mock_speculative_decoder,
            use_speculative=False,
        )

        assert scheduler.speculative_decoder is not None
        assert scheduler.use_speculative is False

    def test_scheduler_with_speculative_enabled(self, mock_speculative_decoder):
        """Test scheduler with speculative decoding enabled."""
        scheduler = InferenceScheduler(
            speculative_decoder=mock_speculative_decoder,
            use_speculative=True,
        )

        assert scheduler.speculative_decoder is not None
        assert scheduler.use_speculative is True

    def test_speculative_decode_execution(self, mock_speculative_decoder):
        """Test that speculative decode is actually called when enabled."""
        scheduler = InferenceScheduler(
            speculative_decoder=mock_speculative_decoder,
            use_speculative=True,
        )

        req = InferenceRequest(
            request_id="spec-test-1",
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(req)

        # Run two steps: first for prefill, second for decode
        scheduler.step()  # Prefill
        scheduler.step()  # Decode with speculative

        # Verify speculative decoder was called
        assert mock_speculative_decoder._call_count > 0

    def test_speculative_stats_in_scheduler_stats(self, mock_speculative_decoder):
        """Test that speculative stats are included in scheduler stats."""
        scheduler = InferenceScheduler(
            speculative_decoder=mock_speculative_decoder,
            use_speculative=True,
        )

        req = InferenceRequest(
            request_id="spec-test-2",
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=5),
        )
        scheduler.add_request(req)
        scheduler.run(max_steps=10)

        stats = scheduler.get_stats()

        assert 'speculative_enabled' in stats
        assert stats['speculative_enabled'] is True
        assert 'speculative_tokens' in stats
        assert 'speculative_acceptance_rate' in stats

    def test_speculative_token_counting(self, mock_speculative_decoder):
        """Test that speculative tokens are counted correctly."""
        scheduler = InferenceScheduler(
            speculative_decoder=mock_speculative_decoder,
            use_speculative=True,
        )

        req = InferenceRequest(
            request_id="spec-test-3",
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=10),
        )
        scheduler.add_request(req)
        scheduler.run(max_steps=20)

        stats = scheduler.get_stats()

        # Should have counted tokens from speculative decoding
        assert stats['speculative_tokens'] > 0
        assert stats['total_decode_tokens'] > 0

    def test_fallback_to_regular_decode_when_no_speculative(self):
        """Test that regular decode is used when no speculative decoder."""
        scheduler = InferenceScheduler(
            speculative_decoder=None,
            use_speculative=True,  # Enable flag but no decoder
        )

        # Should effectively be disabled
        assert scheduler.use_speculative is False

        req = InferenceRequest(
            request_id="no-spec-test",
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=5),
        )
        scheduler.add_request(req)
        scheduler.run(max_steps=20)

        # Should still complete using regular decode
        request = scheduler.request_manager.get_request("no-spec-test")
        assert request.status == RequestStatus.COMPLETED

    def test_speculative_request_completion(self, mock_speculative_decoder):
        """Test that requests complete properly with speculative decoding."""
        scheduler = InferenceScheduler(
            speculative_decoder=mock_speculative_decoder,
            use_speculative=True,
        )

        req = InferenceRequest(
            request_id="spec-complete-test",
            prompt="Test prompt",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=4),
        )
        scheduler.add_request(req)
        scheduler.run(max_steps=20)

        request = scheduler.request_manager.get_request("spec-complete-test")
        assert request.status == RequestStatus.COMPLETED
        assert len(request.output_token_ids) > 0

    def test_multiple_requests_with_speculative(self, mock_speculative_decoder):
        """Test multiple requests with speculative decoding."""
        scheduler = InferenceScheduler(
            speculative_decoder=mock_speculative_decoder,
            use_speculative=True,
        )

        # Add multiple requests
        for i in range(3):
            req = InferenceRequest(
                request_id=f"spec-multi-{i}",
                prompt=f"Test prompt {i}",
                prompt_token_ids=[1, 2, 3, 4 + i],
                sampling_params=SamplingParams(max_tokens=5),
            )
            scheduler.add_request(req)

        scheduler.run(max_steps=50)

        # All requests should complete
        for i in range(3):
            request = scheduler.request_manager.get_request(f"spec-multi-{i}")
            assert request.status == RequestStatus.COMPLETED
