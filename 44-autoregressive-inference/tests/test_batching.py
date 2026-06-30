"""Tests for continuous batching functionality.

This module tests:
- ContinuousBatcher batch formation
- BatchedInputs tensor creation
- Scheduling policies (FIFO, Priority, SJF)
- Request lifecycle in batching
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src and tests to path
src_path = Path(__file__).parent.parent / "src"
tests_path = Path(__file__).parent
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(tests_path))

from autoregressive_inference.batching import (
    ContinuousBatcher,
    BatchedInputs,
    FIFOPolicy,
    PriorityPolicy,
    ShortestJobFirstPolicy,
)
from autoregressive_inference.requests import (
    InferenceRequest,
    SamplingParams,
    RequestStatus,
    RequestPriority,
)

# Import helper from conftest
from conftest import create_request, create_running_request

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestContinuousBatcherCreation:
    """Tests for ContinuousBatcher initialization."""

    def test_batcher_creation(self, continuous_batcher):
        """Test basic batcher creation."""
        assert continuous_batcher.max_batch_size == 8
        assert continuous_batcher.max_prefill_tokens == 1024
        assert continuous_batcher.max_decode_tokens == 512

    def test_batcher_custom_params(self):
        """Test batcher with custom parameters."""
        batcher = ContinuousBatcher(
            max_batch_size=32,
            max_prefill_tokens=8192,
            max_decode_tokens=4096
        )
        assert batcher.max_batch_size == 32
        assert batcher.max_prefill_tokens == 8192
        assert batcher.max_decode_tokens == 4096


class TestBatchFormation:
    """Tests for batch formation logic."""

    def test_form_batches_empty(self, continuous_batcher):
        """Test batch formation with no requests."""
        prefill, decode = continuous_batcher.form_batches([], [])
        assert prefill == []
        assert decode == []

    def test_form_batches_prefill_only(self, continuous_batcher, sample_requests_for_batching):
        """Test batch formation with only pending requests."""
        pending = sample_requests_for_batching[:5]
        running = []

        prefill, decode = continuous_batcher.form_batches(pending, running)

        assert len(prefill) > 0
        assert len(decode) == 0
        assert len(prefill) <= continuous_batcher.max_batch_size

    def test_form_batches_decode_only(self, continuous_batcher, default_sampling_params):
        """Test batch formation with only decode requests."""
        # Create running requests in decode state
        running = []
        for i in range(5):
            req = create_running_request(f"decode-{i}", prompt_length=10, output_length=5)
            running.append(req)

        prefill, decode = continuous_batcher.form_batches([], running)

        assert len(prefill) == 0
        assert len(decode) == 5

    def test_form_batches_mixed(self, continuous_batcher, default_sampling_params):
        """Test batch formation with both prefill and decode requests."""
        # Create pending requests
        pending = [create_request(f"pending-{i}", prompt_length=50) for i in range(3)]

        # Create running decode requests
        running = [create_running_request(f"decode-{i}") for i in range(2)]

        prefill, decode = continuous_batcher.form_batches(pending, running)

        assert len(prefill) > 0
        assert len(decode) == 2

    def test_form_batches_respects_batch_size(self, continuous_batcher, default_sampling_params):
        """Test that batch formation respects max batch size."""
        # Create more requests than max batch size
        pending = [create_request(f"req-{i}", prompt_length=10) for i in range(20)]

        prefill, decode = continuous_batcher.form_batches(pending, [])

        assert len(prefill) <= continuous_batcher.max_batch_size

    def test_form_batches_respects_token_limit(self, default_sampling_params):
        """Test that batch formation respects token limit."""
        batcher = ContinuousBatcher(
            max_batch_size=100,
            max_prefill_tokens=200,  # Very low limit
            max_decode_tokens=100
        )

        # Create requests with 50 tokens each
        pending = [create_request(f"req-{i}", prompt_length=50) for i in range(10)]

        prefill, decode = batcher.form_batches(pending, [])

        # Should only fit 4 requests (4 * 50 = 200)
        assert len(prefill) <= 4

        # Total tokens should not exceed limit
        total_tokens = sum(len(r.prompt_token_ids) for r in prefill)
        assert total_tokens <= batcher.max_prefill_tokens

    def test_form_batches_decode_limits_prefill(self, default_sampling_params):
        """Test that decode tokens reduce prefill capacity."""
        batcher = ContinuousBatcher(
            max_batch_size=100,
            max_prefill_tokens=100,
            max_decode_tokens=100
        )

        # Create decode requests (each uses 1 token)
        running = [create_running_request(f"decode-{i}") for i in range(50)]

        # Create pending requests with 10 tokens each
        pending = [create_request(f"pending-{i}", prompt_length=10) for i in range(10)]

        prefill, decode = batcher.form_batches(pending, running)

        # Decode takes 50 tokens, leaving 50 for prefill
        # Should fit 5 requests of 10 tokens each
        total_tokens = sum(len(r.prompt_token_ids) for r in prefill) + len(decode)
        assert total_tokens <= batcher.max_prefill_tokens


class TestBatchedInputs:
    """Tests for BatchedInputs tensor creation."""

    def test_batched_inputs_empty(self):
        """Test empty batch."""
        batch = BatchedInputs()
        assert batch.batch_size == 0
        assert batch.is_empty

    def test_batched_inputs_prefill_only(self, default_sampling_params):
        """Test batch with only prefill requests."""
        requests = [create_request(f"req-{i}", prompt_length=10) for i in range(3)]
        batch = BatchedInputs(prefill_requests=requests)

        assert batch.batch_size == 3
        assert batch.num_prefill == 3
        assert batch.num_decode == 0
        assert not batch.is_empty

    def test_batched_inputs_decode_only(self, default_sampling_params):
        """Test batch with only decode requests."""
        requests = [create_running_request(f"req-{i}") for i in range(3)]
        batch = BatchedInputs(decode_requests=requests)

        assert batch.batch_size == 3
        assert batch.num_prefill == 0
        assert batch.num_decode == 3

    def test_get_input_ids_shape(self, default_sampling_params):
        """Test input_ids tensor shape."""
        requests = [
            create_request("r1", prompt_length=5),
            create_request("r2", prompt_length=10),
            create_request("r3", prompt_length=7),
        ]
        batch = BatchedInputs(prefill_requests=requests)

        input_ids = batch.get_input_ids()

        # Shape should be [batch_size, max_seq_len]
        assert input_ids.shape[0] == 3
        assert input_ids.shape[1] == 10  # Max length

    def test_get_input_ids_padding(self, default_sampling_params):
        """Test input_ids padding."""
        requests = [
            create_request("r1", prompt_length=3),
            create_request("r2", prompt_length=5),
        ]
        batch = BatchedInputs(prefill_requests=requests)

        input_ids = batch.get_input_ids()

        # First request should have padding (zeros) at end
        assert input_ids[0, 3] == 0
        assert input_ids[0, 4] == 0

        # Second request should have no padding
        assert input_ids[1, 4] != 0

    def test_get_input_ids_decode(self, default_sampling_params):
        """Test input_ids for decode requests (single token)."""
        # Create decode request with some output
        req = create_running_request("r1", output_length=5)
        batch = BatchedInputs(decode_requests=[req])

        input_ids = batch.get_input_ids()

        # Should be single token (last output token)
        assert input_ids.shape[0] == 1
        assert input_ids.shape[1] == 1
        assert input_ids[0, 0] == req.output_token_ids[-1]

    def test_get_position_ids_prefill(self, default_sampling_params):
        """Test position_ids for prefill."""
        requests = [
            create_request("r1", prompt_length=5),
            create_request("r2", prompt_length=3),
        ]
        batch = BatchedInputs(prefill_requests=requests)

        position_ids = batch.get_position_ids()

        # First request: [0, 1, 2, 3, 4, 0, 0] (padded)
        if HAS_TORCH:
            assert position_ids[0, 0].item() == 0
            assert position_ids[0, 4].item() == 4
        else:
            assert position_ids[0, 0] == 0
            assert position_ids[0, 4] == 4

    def test_get_position_ids_decode(self, default_sampling_params):
        """Test position_ids for decode."""
        # Decode request at position prompt_len + output_len
        req = create_running_request("r1", prompt_length=10, output_length=5)
        batch = BatchedInputs(decode_requests=[req])

        position_ids = batch.get_position_ids()

        # Position should be 10 + 5 = 15
        if HAS_TORCH:
            assert position_ids[0, 0].item() == 15
        else:
            assert position_ids[0, 0] == 15

    def test_get_attention_mask(self, default_sampling_params):
        """Test attention mask generation."""
        requests = [
            create_request("r1", prompt_length=5),
            create_request("r2", prompt_length=3),
        ]
        batch = BatchedInputs(prefill_requests=requests)

        mask = batch.get_attention_mask()

        # First request: 5 ones, then zeros
        if HAS_TORCH:
            assert mask[0, 0].item() == 1
            assert mask[0, 4].item() == 1
            # Second request: 3 ones
            assert mask[1, 0].item() == 1
            assert mask[1, 2].item() == 1
        else:
            assert mask[0, 0] == 1
            assert mask[0, 4] == 1
            assert mask[1, 0] == 1
            assert mask[1, 2] == 1

    def test_get_sequence_lengths(self, default_sampling_params):
        """Test sequence length extraction."""
        prefill_reqs = [
            create_request("p1", prompt_length=10),
            create_request("p2", prompt_length=20),
        ]
        decode_reqs = [create_running_request("d1")]
        batch = BatchedInputs(prefill_requests=prefill_reqs, decode_requests=decode_reqs)

        lengths = batch.get_sequence_lengths()

        assert lengths == [10, 20, 1]  # Prefill lengths, then 1 for decode

    def test_get_request_ids(self, default_sampling_params):
        """Test request ID extraction."""
        prefill_reqs = [
            create_request("prefill-1", prompt_length=10),
            create_request("prefill-2", prompt_length=10),
        ]
        decode_reqs = [
            create_running_request("decode-1"),
        ]
        batch = BatchedInputs(prefill_requests=prefill_reqs, decode_requests=decode_reqs)

        request_ids = batch.get_request_ids()

        assert request_ids == ["prefill-1", "prefill-2", "decode-1"]

    def test_total_prefill_tokens(self, default_sampling_params):
        """Test total prefill token count."""
        requests = [
            create_request("r1", prompt_length=10),
            create_request("r2", prompt_length=20),
            create_request("r3", prompt_length=15),
        ]
        batch = BatchedInputs(prefill_requests=requests)

        assert batch.total_prefill_tokens == 45


class TestCanAddRequest:
    """Tests for can_add_request logic."""

    def test_can_add_within_limits(self, continuous_batcher):
        """Test can add when within limits."""
        request = create_request("r1", prompt_length=100)
        assert continuous_batcher.can_add_request(request, current_batch_size=0, current_tokens=0)

    def test_cannot_add_batch_full(self, continuous_batcher):
        """Test cannot add when batch is full."""
        request = create_request("r1", prompt_length=10)
        assert not continuous_batcher.can_add_request(
            request,
            current_batch_size=continuous_batcher.max_batch_size,
            current_tokens=0
        )

    def test_cannot_add_token_limit(self, continuous_batcher):
        """Test cannot add when token limit exceeded."""
        request = create_request("r1", prompt_length=100)
        assert not continuous_batcher.can_add_request(
            request,
            current_batch_size=0,
            current_tokens=continuous_batcher.max_prefill_tokens
        )


class TestBatchStats:
    """Tests for batch statistics."""

    def test_get_batch_stats(self, continuous_batcher, default_sampling_params):
        """Test batch statistics calculation."""
        prefill_reqs = [
            create_request("p1", prompt_length=50),
            create_request("p2", prompt_length=30),
        ]
        decode_reqs = [create_running_request("d1"), create_running_request("d2")]

        stats = continuous_batcher.get_batch_stats(prefill_reqs, decode_reqs)

        assert stats['prefill_requests'] == 2
        assert stats['decode_requests'] == 2
        assert stats['total_requests'] == 4
        assert stats['prefill_tokens'] == 80
        assert stats['decode_tokens'] == 2
        assert stats['total_tokens'] == 82
        assert 0 < stats['batch_utilization'] <= 1


class TestEstimateBatchTime:
    """Tests for batch time estimation."""

    def test_estimate_batch_time(self, continuous_batcher, default_sampling_params):
        """Test batch time estimation."""
        prefill_reqs = [create_request("p1", prompt_length=100)]
        decode_reqs = [create_running_request(f"d{i}") for i in range(10)]

        time_est = continuous_batcher.estimate_batch_time(
            prefill_reqs,
            decode_reqs,
            tokens_per_second=1000.0
        )

        # 100 prefill tokens * 2 + 10 decode tokens = 210 effective tokens
        # 210 / 1000 = 0.21 seconds
        assert 0.2 < time_est < 0.25


class TestFIFOPolicy:
    """Tests for FIFO scheduling policy."""

    def test_fifo_order(self, default_sampling_params):
        """Test FIFO maintains arrival order."""
        policy = FIFOPolicy()

        # Create requests with different arrival times
        import time
        requests = []
        for i in range(5):
            req = create_request(f"req-{i}", prompt_length=10)
            req.arrival_time = time.time() + i * 0.001  # Stagger arrivals
            requests.append(req)

        prefill, decode = policy.select_requests(
            pending=requests,
            running=[],
            max_batch_size=10,
            max_tokens=1000
        )

        # Should be in arrival order
        for i, req in enumerate(prefill):
            assert req.request_id == f"req-{i}"

    def test_fifo_respects_limits(self, default_sampling_params):
        """Test FIFO respects batch size and token limits."""
        policy = FIFOPolicy()

        requests = [create_request(f"req-{i}", prompt_length=50) for i in range(10)]

        prefill, decode = policy.select_requests(
            pending=requests,
            running=[],
            max_batch_size=5,
            max_tokens=1000
        )

        assert len(prefill) <= 5


class TestPriorityPolicy:
    """Tests for priority-based scheduling policy."""

    def test_priority_order(self, default_sampling_params):
        """Test priority policy respects priorities."""
        policy = PriorityPolicy()

        requests = [
            create_request("low", prompt_length=10, priority=RequestPriority.LOW),
            create_request("high", prompt_length=10, priority=RequestPriority.HIGH),
            create_request("critical", prompt_length=10, priority=RequestPriority.CRITICAL),
            create_request("normal", prompt_length=10, priority=RequestPriority.NORMAL),
        ]

        prefill, decode = policy.select_requests(
            pending=requests,
            running=[],
            max_batch_size=10,
            max_tokens=1000
        )

        # Should be in priority order: critical, high, normal, low
        assert prefill[0].request_id == "critical"
        assert prefill[1].request_id == "high"
        assert prefill[2].request_id == "normal"
        assert prefill[3].request_id == "low"

    def test_priority_with_decode(self, default_sampling_params):
        """Test priority policy includes decode requests."""
        policy = PriorityPolicy()

        pending = [create_request("pending", prompt_length=10)]
        running = [create_running_request("running")]

        prefill, decode = policy.select_requests(
            pending=pending,
            running=running,
            max_batch_size=10,
            max_tokens=1000
        )

        assert len(prefill) == 1
        assert len(decode) == 1


class TestShortestJobFirstPolicy:
    """Tests for shortest job first scheduling policy."""

    def test_sjf_order(self, default_sampling_params):
        """Test SJF selects shortest prompts first."""
        policy = ShortestJobFirstPolicy()

        requests = [
            create_request("long", prompt_length=100),
            create_request("short", prompt_length=10),
            create_request("medium", prompt_length=50),
        ]

        prefill, decode = policy.select_requests(
            pending=requests,
            running=[],
            max_batch_size=10,
            max_tokens=1000
        )

        # Should be in length order: short, medium, long
        assert prefill[0].request_id == "short"
        assert prefill[1].request_id == "medium"
        assert prefill[2].request_id == "long"

    def test_sjf_respects_limits(self, default_sampling_params):
        """Test SJF respects token limits."""
        policy = ShortestJobFirstPolicy()

        requests = [
            create_request("r1", prompt_length=30),
            create_request("r2", prompt_length=40),
            create_request("r3", prompt_length=50),
        ]

        prefill, decode = policy.select_requests(
            pending=requests,
            running=[],
            max_batch_size=10,
            max_tokens=80  # Can only fit r1 + r2 (70 tokens)
        )

        assert len(prefill) == 2
        total_tokens = sum(len(r.prompt_token_ids) for r in prefill)
        assert total_tokens <= 80


class TestMergeForExecution:
    """Tests for merging batches for execution."""

    def test_merge_creates_batched_inputs(self, continuous_batcher, default_sampling_params):
        """Test merge_for_execution creates BatchedInputs."""
        prefill = [create_request("p1", prompt_length=10)]
        decode = [create_running_request("d1")]

        batch = continuous_batcher.merge_for_execution(prefill, decode)

        assert isinstance(batch, BatchedInputs)
        assert batch.num_prefill == 1
        assert batch.num_decode == 1
        assert batch.batch_size == 2


class TestBatchingEdgeCases:
    """Tests for edge cases in batching."""

    def test_single_request_batch(self, continuous_batcher, default_sampling_params):
        """Test batch with single request."""
        request = create_request("single", prompt_length=10)
        batch = BatchedInputs(prefill_requests=[request])

        assert batch.batch_size == 1
        input_ids = batch.get_input_ids()
        assert input_ids.shape[0] == 1

    def test_very_long_prompt(self, default_sampling_params):
        """Test handling of very long prompt."""
        batcher = ContinuousBatcher(max_batch_size=10, max_prefill_tokens=100)

        # Create request longer than token limit
        long_request = create_request("long", prompt_length=200)
        pending = [long_request]

        prefill, decode = batcher.form_batches(pending, [])

        # Should not include the oversized request (or include if it's first and fits)
        # Behavior depends on implementation - just ensure no crash
        assert len(prefill) <= 1

    def test_all_decode_state(self, continuous_batcher, default_sampling_params):
        """Test when all requests are in decode state."""
        running = [create_running_request(f"r{i}") for i in range(5)]

        prefill, decode = continuous_batcher.form_batches([], running)

        assert len(prefill) == 0
        assert len(decode) == 5

    def test_mixed_priorities_in_batch(self, continuous_batcher, default_sampling_params):
        """Test batch with mixed priority requests."""
        requests = [
            create_request("low", priority=RequestPriority.LOW),
            create_request("high", priority=RequestPriority.HIGH),
            create_request("normal", priority=RequestPriority.NORMAL),
        ]

        batch = BatchedInputs(prefill_requests=requests)

        # All should be in batch
        assert batch.batch_size == 3

        # Request IDs should be preserved
        ids = batch.get_request_ids()
        assert set(ids) == {"low", "high", "normal"}


class TestBatchingConcurrency:
    """Tests for concurrent batching operations."""

    def test_concurrent_batch_formation(self, default_sampling_params):
        """Test concurrent batch formation."""
        import threading

        batcher = ContinuousBatcher(max_batch_size=100, max_prefill_tokens=10000)
        results = []
        errors = []

        def form_batch(batch_id):
            try:
                pending = [create_request(f"req-{batch_id}-{i}") for i in range(5)]
                prefill, decode = batcher.form_batches(pending, [])
                results.append((batch_id, len(prefill)))
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t = threading.Thread(target=form_batch, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


class TestBatchInputsWithTorch:
    """Tests for BatchedInputs with torch tensors."""

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
    def test_torch_input_ids(self, default_sampling_params):
        """Test input_ids returns torch tensor when available."""
        requests = [create_request("r1", prompt_length=10)]
        batch = BatchedInputs(prefill_requests=requests)

        input_ids = batch.get_input_ids()

        assert isinstance(input_ids, torch.Tensor)
        assert input_ids.dtype == torch.long

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
    def test_torch_position_ids(self, default_sampling_params):
        """Test position_ids returns torch tensor when available."""
        requests = [create_request("r1", prompt_length=10)]
        batch = BatchedInputs(prefill_requests=requests)

        position_ids = batch.get_position_ids()

        assert isinstance(position_ids, torch.Tensor)

    @pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
    def test_torch_attention_mask(self, default_sampling_params):
        """Test attention_mask returns torch tensor when available."""
        requests = [create_request("r1", prompt_length=10)]
        batch = BatchedInputs(prefill_requests=requests)

        mask = batch.get_attention_mask()

        assert isinstance(mask, torch.Tensor)
