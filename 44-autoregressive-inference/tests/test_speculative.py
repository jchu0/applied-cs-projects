"""Tests for speculative decoding functionality.

This module tests:
- SpeculativeDecoder draft generation
- Verification logic
- Acceptance rate tracking
- TreeSpeculativeDecoder (tree-based speculation)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from autoregressive_inference.speculative import (
    SpeculativeDecoder,
    TreeSpeculativeDecoder,
    SpeculativeStats,
)
from autoregressive_inference.requests import SamplingParams

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestSpeculativeStats:
    """Tests for SpeculativeStats dataclass."""

    def test_stats_creation(self):
        """Test stats initialization."""
        stats = SpeculativeStats()
        assert stats.total_drafted == 0
        assert stats.total_accepted == 0
        assert stats.total_rejected == 0
        assert len(stats.acceptance_by_position) == 8

    def test_acceptance_rate_zero(self):
        """Test acceptance rate when no drafting."""
        stats = SpeculativeStats()
        assert stats.acceptance_rate == 0.0

    def test_acceptance_rate(self):
        """Test acceptance rate calculation."""
        stats = SpeculativeStats()
        stats.total_drafted = 100
        stats.total_accepted = 75
        assert stats.acceptance_rate == 0.75

    def test_avg_accepted_per_step(self):
        """Test average accepted per step calculation."""
        stats = SpeculativeStats()
        stats.total_drafted = 100
        stats.total_accepted = 75
        stats.acceptance_by_position[0] = 50
        stats.acceptance_by_position[1] = 30

        avg = stats.avg_accepted_per_step
        assert avg > 0


class TestSpeculativeDecoderCreation:
    """Tests for SpeculativeDecoder initialization."""

    def test_decoder_creation(self, speculative_decoder):
        """Test basic decoder creation."""
        assert speculative_decoder.target_model is None
        assert speculative_decoder.draft_model is None
        assert speculative_decoder.num_speculative == 4

    def test_decoder_custom_tokens(self):
        """Test decoder with custom speculative tokens."""
        decoder = SpeculativeDecoder(num_speculative_tokens=8)
        assert decoder.num_speculative == 8

    def test_decoder_stats_initialized(self, speculative_decoder):
        """Test that stats are initialized."""
        assert speculative_decoder.stats.total_drafted == 0
        assert speculative_decoder.stats.total_accepted == 0


class TestDraftGeneration:
    """Tests for draft token generation."""

    def test_draft_mock_generation(self, speculative_decoder, default_sampling_params):
        """Test mock draft generation without models."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        draft_tokens = speculative_decoder._draft(input_ids, default_sampling_params)

        # Should generate num_speculative tokens
        assert len(draft_tokens) == speculative_decoder.num_speculative
        # Mock returns 100 + i
        assert draft_tokens[0] == 100
        assert draft_tokens[1] == 101

    def test_draft_different_counts(self, default_sampling_params):
        """Test draft with different speculative counts."""
        for n in [2, 4, 8]:
            decoder = SpeculativeDecoder(num_speculative_tokens=n)

            if HAS_TORCH:
                input_ids = torch.tensor([[1, 2, 3]])
            else:
                input_ids = np.array([[1, 2, 3]])

            draft_tokens = decoder._draft(input_ids, default_sampling_params)
            assert len(draft_tokens) == n


class TestVerification:
    """Tests for draft verification."""

    def test_verify_mock(self, speculative_decoder, default_sampling_params):
        """Test mock verification without models."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        draft_tokens = [100, 101, 102, 103]

        accepted = speculative_decoder._verify(
            input_ids,
            draft_tokens,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        # Mock accepts first 2, modifies third
        assert len(accepted) == 3
        assert accepted[0] == 100
        assert accepted[1] == 101
        assert accepted[2] == 103  # Modified (rejected + 1)

    def test_verify_updates_stats(self, speculative_decoder, default_sampling_params):
        """Test that verification updates statistics."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        initial_rejected = speculative_decoder.stats.total_rejected

        speculative_decoder._verify(
            input_ids,
            [100, 101, 102, 103],
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        # Should have recorded a rejection
        assert speculative_decoder.stats.total_rejected > initial_rejected


class TestGenerateStep:
    """Tests for complete generate_step."""

    def test_generate_step_returns_tokens(self, speculative_decoder, default_sampling_params):
        """Test that generate_step returns accepted tokens."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        accepted = speculative_decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        assert len(accepted) > 0
        assert all(isinstance(t, int) for t in accepted)

    def test_generate_step_updates_stats(self, speculative_decoder, default_sampling_params):
        """Test that generate_step updates statistics."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        initial_drafted = speculative_decoder.stats.total_drafted

        speculative_decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        assert speculative_decoder.stats.total_drafted > initial_drafted


class TestAcceptanceRate:
    """Tests for acceptance rate tracking."""

    def test_get_acceptance_rate(self, speculative_decoder, default_sampling_params):
        """Test acceptance rate getter."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        # Do a few generate steps
        for _ in range(5):
            speculative_decoder.generate_step(
                input_ids,
                kv_cache=None,
                request_id="test",
                sampling_params=default_sampling_params
            )

        rate = speculative_decoder.get_acceptance_rate()
        assert 0 <= rate <= 1

    def test_get_stats(self, speculative_decoder, default_sampling_params):
        """Test detailed stats retrieval."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        speculative_decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        stats = speculative_decoder.get_stats()

        assert 'total_drafted' in stats
        assert 'total_accepted' in stats
        assert 'total_rejected' in stats
        assert 'acceptance_rate' in stats
        assert 'avg_accepted_per_step' in stats
        assert 'acceptance_by_position' in stats

    def test_reset_stats(self, speculative_decoder, default_sampling_params):
        """Test resetting statistics."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        # Accumulate some stats
        for _ in range(5):
            speculative_decoder.generate_step(
                input_ids,
                kv_cache=None,
                request_id="test",
                sampling_params=default_sampling_params
            )

        speculative_decoder.reset_stats()

        assert speculative_decoder.stats.total_drafted == 0
        assert speculative_decoder.stats.total_accepted == 0


class TestTreeSpeculativeDecoder:
    """Tests for TreeSpeculativeDecoder."""

    def test_tree_decoder_creation(self):
        """Test tree decoder initialization."""
        decoder = TreeSpeculativeDecoder(
            tree_width=2,
            tree_depth=4
        )
        assert decoder.tree_width == 2
        assert decoder.tree_depth == 4

    def test_tree_draft_generation(self, default_sampling_params):
        """Test tree draft generation."""
        decoder = TreeSpeculativeDecoder(tree_width=2, tree_depth=3)

        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        paths = decoder._draft_tree(input_ids, default_sampling_params)

        # Should have 2^3 = 8 paths
        assert len(paths) == 8

        # Each path should have 3 tokens
        for path in paths:
            assert len(path) == 3

    def test_tree_generate_step(self, default_sampling_params):
        """Test tree generate step."""
        decoder = TreeSpeculativeDecoder(tree_width=2, tree_depth=3)

        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        accepted = decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        assert len(accepted) > 0

    def test_tree_stats(self, default_sampling_params):
        """Test tree decoder statistics."""
        decoder = TreeSpeculativeDecoder(tree_width=2, tree_depth=3)

        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        stats = decoder.get_stats()

        assert 'tree_width' in stats
        assert 'tree_depth' in stats
        assert stats['tree_width'] == 2
        assert stats['tree_depth'] == 3


class TestSpeculativeDecoderEdgeCases:
    """Tests for edge cases in speculative decoding."""

    def test_empty_input(self, speculative_decoder, default_sampling_params):
        """Test with empty input sequence."""
        if HAS_TORCH:
            input_ids = torch.tensor([[]])
        else:
            input_ids = np.array([[]])

        # Should not crash
        try:
            accepted = speculative_decoder.generate_step(
                input_ids,
                kv_cache=None,
                request_id="test",
                sampling_params=default_sampling_params
            )
            assert isinstance(accepted, list)
        except (ValueError, IndexError):
            pass  # Some implementations may raise

    def test_single_token_speculation(self, default_sampling_params):
        """Test with single speculative token."""
        decoder = SpeculativeDecoder(num_speculative_tokens=1)

        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        accepted = decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        assert len(accepted) >= 1

    def test_many_speculative_tokens(self, default_sampling_params):
        """Test with many speculative tokens."""
        decoder = SpeculativeDecoder(num_speculative_tokens=16)

        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        accepted = decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        assert len(accepted) > 0


class TestSpeculativeDecoderWithSampling:
    """Tests for speculative decoding with different sampling parameters."""

    def test_greedy_sampling(self):
        """Test speculative decoding with greedy sampling."""
        decoder = SpeculativeDecoder(num_speculative_tokens=4)

        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        params = SamplingParams(temperature=0.0)

        accepted = decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=params
        )

        assert len(accepted) > 0

    def test_temperature_sampling(self):
        """Test speculative decoding with temperature sampling."""
        decoder = SpeculativeDecoder(num_speculative_tokens=4)

        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        params = SamplingParams(temperature=0.8, top_p=0.9)

        accepted = decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=params
        )

        assert len(accepted) > 0


class TestPositionTracking:
    """Tests for position-based acceptance tracking."""

    def test_acceptance_by_position_updated(self, speculative_decoder, default_sampling_params):
        """Test that acceptance by position is tracked."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        # Run several steps
        for _ in range(10):
            speculative_decoder.generate_step(
                input_ids,
                kv_cache=None,
                request_id="test",
                sampling_params=default_sampling_params
            )

        # Position 0 and 1 should have more acceptances (mock behavior)
        stats = speculative_decoder.get_stats()
        assert stats['acceptance_by_position'][0] > 0
        assert stats['acceptance_by_position'][1] > 0

    def test_rejection_by_position_tracked(self, speculative_decoder, default_sampling_params):
        """Test that rejection by position is tracked."""
        if HAS_TORCH:
            input_ids = torch.tensor([[1, 2, 3]])
        else:
            input_ids = np.array([[1, 2, 3]])

        for _ in range(10):
            speculative_decoder.generate_step(
                input_ids,
                kv_cache=None,
                request_id="test",
                sampling_params=default_sampling_params
            )

        # Position 2 should have rejections (mock behavior rejects at position 2)
        assert speculative_decoder.stats.rejection_by_position[2] > 0


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestSpeculativeDecoderTorch:
    """Tests for speculative decoding with PyTorch tensors."""

    def test_torch_input(self, speculative_decoder, default_sampling_params):
        """Test with PyTorch tensor input."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])

        accepted = speculative_decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        assert len(accepted) > 0
        assert all(isinstance(t, int) for t in accepted)

    def test_torch_batched_input(self, speculative_decoder, default_sampling_params):
        """Test with batched PyTorch input."""
        # Note: Current implementation may not support batched input
        input_ids = torch.tensor([[1, 2, 3]])

        accepted = speculative_decoder.generate_step(
            input_ids,
            kv_cache=None,
            request_id="test",
            sampling_params=default_sampling_params
        )

        assert len(accepted) > 0
