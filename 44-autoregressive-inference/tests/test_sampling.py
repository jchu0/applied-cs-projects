"""Tests for token sampling strategies.

This module tests:
- Greedy sampling (argmax)
- Temperature scaling
- Top-k filtering
- Top-p (nucleus) sampling
- Repetition penalty
- Frequency and presence penalties
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from autoregressive_inference.sampling import TokenSampler
from autoregressive_inference.requests import SamplingParams

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestGreedySampling:
    """Tests for greedy (argmax) sampling."""

    def test_greedy_with_clear_winner(self, token_sampler, skewed_logits):
        """Test greedy sampling picks the highest logit."""
        result = token_sampler.sample_greedy(skewed_logits)
        assert result == 42  # Token 42 has highest logit

    def test_greedy_via_temperature_zero(self, token_sampler, skewed_logits):
        """Test that temperature=0 produces greedy sampling."""
        params = SamplingParams(temperature=0.0)
        result = token_sampler.sample(skewed_logits, params)
        assert result == 42

    def test_greedy_deterministic(self, token_sampler, random_logits):
        """Test greedy sampling is deterministic."""
        result1 = token_sampler.sample_greedy(random_logits)
        result2 = token_sampler.sample_greedy(random_logits)
        assert result1 == result2

    def test_greedy_batched(self, token_sampler):
        """Test greedy sampling with batched input."""
        np.random.seed(42)
        logits = np.random.randn(4, 100).astype(np.float32)

        # Set different winners for each batch element
        logits[0, 10] = 100.0
        logits[1, 20] = 100.0
        logits[2, 30] = 100.0
        logits[3, 40] = 100.0

        results = token_sampler.sample_greedy(logits)

        assert len(results) == 4
        assert results[0] == 10
        assert results[1] == 20
        assert results[2] == 30
        assert results[3] == 40


class TestTemperatureSampling:
    """Tests for temperature-scaled sampling."""

    def test_temperature_one_samples(self, token_sampler, random_logits):
        """Test that temperature=1 produces varied samples."""
        params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0)

        # Sample multiple times
        samples = set()
        for _ in range(50):
            sample = token_sampler.sample(random_logits.copy(), params)
            samples.add(int(sample))

        # Should get varied results
        assert len(samples) > 1

    def test_low_temperature_concentrates(self, token_sampler):
        """Test that low temperature concentrates on top tokens."""
        # Create logits with a clear gradient
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 0] = 2.0
        logits[0, 1] = 1.5
        logits[0, 2] = 1.0

        params = SamplingParams(temperature=0.1, top_k=0, top_p=1.0)

        samples = []
        for _ in range(100):
            sample = token_sampler.sample(logits.copy(), params)
            samples.append(int(sample))

        # Most samples should be token 0
        token_0_count = samples.count(0)
        assert token_0_count > 90  # Very concentrated

    def test_high_temperature_spreads(self, token_sampler):
        """Test that high temperature spreads probability mass."""
        # Create logits with some gradient
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 0] = 2.0
        logits[0, 1] = 1.8
        logits[0, 2] = 1.6
        logits[0, 3] = 1.4
        logits[0, 4] = 1.2

        params = SamplingParams(temperature=2.0, top_k=0, top_p=1.0)

        samples = set()
        for _ in range(100):
            sample = token_sampler.sample(logits.copy(), params)
            samples.add(int(sample))

        # Should get more variety with high temperature
        assert len(samples) > 3

    def test_temperature_with_method(self, token_sampler, skewed_logits):
        """Test sample_with_temperature method."""
        # Temperature 0 should be greedy
        result = token_sampler.sample_with_temperature(skewed_logits, 0.0)
        assert result == 42

        # High temperature should vary
        samples = set()
        for _ in range(50):
            sample = token_sampler.sample_with_temperature(skewed_logits.copy(), 2.0)
            samples.add(int(sample))
        assert len(samples) > 1


class TestTopKSampling:
    """Tests for top-k filtering."""

    def test_top_k_limits_candidates(self, token_sampler):
        """Test that top-k limits to k candidates."""
        # Create uniform logits
        logits = np.zeros((1, 100), dtype=np.float32)

        params = SamplingParams(temperature=1.0, top_k=5, top_p=1.0)

        # With uniform logits and top_k=5, should only sample from 5 tokens
        samples = set()
        for _ in range(100):
            sample = token_sampler.sample(logits.copy(), params)
            samples.add(int(sample))

        assert len(samples) <= 5

    def test_top_k_with_gradient(self, token_sampler):
        """Test top-k with gradient logits."""
        logits = np.arange(100).astype(np.float32).reshape(1, -1)

        params = SamplingParams(temperature=1.0, top_k=3, top_p=1.0)

        samples = set()
        for _ in range(100):
            sample = token_sampler.sample(logits.copy(), params)
            samples.add(int(sample))

        # Should only sample from top 3 (97, 98, 99)
        assert samples.issubset({97, 98, 99})

    def test_top_k_zero_no_filtering(self, token_sampler, random_logits):
        """Test that top_k=0 applies no filtering."""
        params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0)

        samples = set()
        for _ in range(100):
            sample = token_sampler.sample(random_logits.copy(), params)
            samples.add(int(sample))

        # Should have variety across the vocabulary
        assert len(samples) > 10

    def test_top_k_larger_than_vocab(self, token_sampler):
        """Test top-k larger than vocabulary size."""
        logits = np.random.randn(1, 50).astype(np.float32)

        params = SamplingParams(temperature=1.0, top_k=100, top_p=1.0)

        # Should not crash, effectively no filtering
        sample = token_sampler.sample(logits, params)
        assert 0 <= sample < 50

    def test_top_k_filtering_method(self, token_sampler):
        """Test the _top_k_filtering internal method."""
        logits = np.arange(10).astype(np.float32).reshape(1, -1)

        filtered = token_sampler._top_k_filtering(logits.copy(), 3)

        # Only top 3 should remain finite
        finite_count = np.sum(np.isfinite(filtered))
        assert finite_count == 3

        # Top 3 values should be unchanged
        assert filtered[0, 9] == 9.0
        assert filtered[0, 8] == 8.0
        assert filtered[0, 7] == 7.0


class TestTopPSampling:
    """Tests for top-p (nucleus) sampling."""

    def test_top_p_limits_candidates(self, token_sampler):
        """Test that top-p limits based on cumulative probability."""
        # Create logits with clear probability distribution
        logits = np.full((1, 100), -10.0, dtype=np.float32)
        logits[0, 0] = 3.0   # ~50% probability
        logits[0, 1] = 2.0   # ~25% probability
        logits[0, 2] = 1.0   # ~12.5% probability

        params = SamplingParams(temperature=1.0, top_k=0, top_p=0.8)

        samples = set()
        for _ in range(100):
            sample = token_sampler.sample(logits.copy(), params)
            samples.add(int(sample))

        # With top_p=0.8, should mostly sample from tokens 0, 1, 2
        assert samples.issubset({0, 1, 2, 3})

    def test_top_p_one_no_filtering(self, token_sampler, random_logits):
        """Test that top_p=1.0 applies no filtering."""
        params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0)

        samples = set()
        for _ in range(100):
            sample = token_sampler.sample(random_logits.copy(), params)
            samples.add(int(sample))

        # Should have good variety
        assert len(samples) > 10

    def test_top_p_very_low(self, token_sampler, skewed_logits):
        """Test very low top_p concentrates heavily."""
        params = SamplingParams(temperature=1.0, top_k=0, top_p=0.1)

        samples = []
        for _ in range(100):
            sample = token_sampler.sample(skewed_logits.copy(), params)
            samples.append(int(sample))

        # Should mostly be token 42 (the dominant one)
        assert samples.count(42) > 80

    def test_top_p_keeps_at_least_one(self, token_sampler):
        """Test that top-p always keeps at least one token."""
        # Extreme case: one token has almost all probability
        logits = np.full((1, 100), -100.0, dtype=np.float32)
        logits[0, 50] = 0.0

        params = SamplingParams(temperature=1.0, top_k=0, top_p=0.001)

        sample = token_sampler.sample(logits, params)
        assert sample == 50

    def test_top_p_filtering_method(self, token_sampler):
        """Test the _top_p_filtering internal method."""
        # Create logits where softmax gives clear ordering
        logits = np.full((1, 10), -10.0, dtype=np.float32)
        logits[0, 0] = 2.0  # High prob
        logits[0, 1] = 1.0  # Medium prob
        logits[0, 2] = 0.0  # Low prob

        filtered = token_sampler._top_p_filtering(logits.copy(), 0.9)

        # Some tokens should be -inf
        inf_count = np.sum(np.isinf(filtered))
        assert inf_count > 0


class TestTopKPCombined:
    """Tests for combining top-k and top-p."""

    def test_top_k_and_top_p(self, token_sampler):
        """Test combined top-k and top-p filtering."""
        logits = np.zeros((1, 100), dtype=np.float32)
        # Create a distribution
        for i in range(10):
            logits[0, i] = 10 - i

        params = SamplingParams(temperature=1.0, top_k=5, top_p=0.9)

        samples = set()
        for _ in range(100):
            sample = token_sampler.sample(logits.copy(), params)
            samples.add(int(sample))

        # Should be limited by both top_k and top_p
        assert len(samples) <= 5
        assert all(s < 10 for s in samples)


class TestRepetitionPenalty:
    """Tests for repetition penalty."""

    def test_repetition_penalty_discourages_repeats(self, token_sampler):
        """Test that repetition penalty discourages previously used tokens."""
        # Create logits where token 42 is dominant
        logits = np.full((1, 100), -5.0, dtype=np.float32)
        logits[0, 42] = 5.0
        logits[0, 43] = 4.5

        # Without penalty, should strongly prefer 42
        params = SamplingParams(temperature=0.0, repetition_penalty=1.0)
        sample = token_sampler.sample(logits.copy(), params)
        assert sample == 42

        # With penalty and 42 in history, should prefer 43
        params = SamplingParams(temperature=0.0, repetition_penalty=2.0)
        sample = token_sampler.sample(logits.copy(), params, generated_ids=[42])
        assert sample == 43

    def test_repetition_penalty_one_no_effect(self, token_sampler, skewed_logits):
        """Test that repetition_penalty=1.0 has no effect."""
        params = SamplingParams(temperature=0.0, repetition_penalty=1.0)

        sample1 = token_sampler.sample(skewed_logits.copy(), params, generated_ids=[])
        sample2 = token_sampler.sample(skewed_logits.copy(), params, generated_ids=[42, 43, 44])

        # Should still pick 42 as it has highest logit
        assert sample1 == 42
        assert sample2 == 42

    def test_repetition_penalty_multiple_tokens(self, token_sampler):
        """Test repetition penalty with multiple repeated tokens."""
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 0] = 3.0
        logits[0, 1] = 2.9
        logits[0, 2] = 2.8
        logits[0, 3] = 2.7
        logits[0, 4] = 2.6

        params = SamplingParams(temperature=0.0, repetition_penalty=2.0)

        # Penalize tokens 0, 1, 2 - should pick 3
        sample = token_sampler.sample(logits.copy(), params, generated_ids=[0, 1, 2])
        assert sample == 3

    def test_repetition_penalty_negative_logits(self, token_sampler):
        """Test repetition penalty works correctly with negative logits."""
        logits = np.full((1, 100), -2.0, dtype=np.float32)
        logits[0, 10] = -0.5  # Best option
        logits[0, 11] = -0.6  # Second best

        params = SamplingParams(temperature=0.0, repetition_penalty=2.0)

        # Without history
        sample1 = token_sampler.sample(logits.copy(), params, generated_ids=[])
        assert sample1 == 10

        # With 10 in history (negative logits get multiplied, making them worse)
        sample2 = token_sampler.sample(logits.copy(), params, generated_ids=[10])
        assert sample2 == 11


class TestFrequencyPresencePenalty:
    """Tests for frequency and presence penalties."""

    def test_frequency_penalty(self, token_sampler):
        """Test frequency penalty proportional to count."""
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 0] = 5.0
        logits[0, 1] = 4.8

        params = SamplingParams(temperature=0.0, frequency_penalty=1.0, presence_penalty=0.0)

        # Token 0 appears 5 times, should be heavily penalized
        sample = token_sampler.sample(logits.copy(), params, generated_ids=[0, 0, 0, 0, 0])
        assert sample == 1  # Should prefer 1 now

    def test_presence_penalty(self, token_sampler):
        """Test presence penalty (flat penalty for appearing)."""
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 0] = 5.0
        logits[0, 1] = 4.5

        params = SamplingParams(temperature=0.0, frequency_penalty=0.0, presence_penalty=1.0)

        # Token 0 appears once - should get flat penalty
        sample = token_sampler.sample(logits.copy(), params, generated_ids=[0])
        # With presence penalty of 1.0, token 0 becomes 4.0, token 1 stays 4.5
        assert sample == 1

    def test_combined_penalties(self, token_sampler):
        """Test combined frequency and presence penalties."""
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 0] = 10.0
        logits[0, 1] = 5.0

        params = SamplingParams(
            temperature=0.0,
            frequency_penalty=0.5,
            presence_penalty=0.5
        )

        # Token 0 appears 10 times: penalty = 0.5*10 + 0.5 = 5.5
        # Token 0 logit becomes 10.0 - 5.5 = 4.5
        # Token 1 stays 5.0
        sample = token_sampler.sample(logits.copy(), params, generated_ids=[0]*10)
        assert sample == 1

    def test_no_penalties_without_history(self, token_sampler, skewed_logits):
        """Test that penalties don't apply without history."""
        params = SamplingParams(
            temperature=0.0,
            frequency_penalty=5.0,
            presence_penalty=5.0
        )

        sample = token_sampler.sample(skewed_logits.copy(), params, generated_ids=[])
        assert sample == 42  # No effect without history


class TestSamplerEdgeCases:
    """Tests for edge cases in sampling."""

    def test_single_token_vocab(self, token_sampler):
        """Test sampling from single-token vocabulary."""
        logits = np.array([[5.0]], dtype=np.float32)
        params = SamplingParams(temperature=1.0)

        sample = token_sampler.sample(logits, params)
        assert sample == 0

    def test_uniform_distribution(self, token_sampler, uniform_logits):
        """Test sampling from uniform distribution."""
        params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0)

        samples = set()
        for _ in range(200):
            sample = token_sampler.sample(uniform_logits.copy(), params)
            samples.add(int(sample))

        # Should sample from many tokens with uniform distribution
        assert len(samples) > 50  # Most of the 100 tokens

    def test_extreme_logits(self, token_sampler):
        """Test with extreme logit values."""
        logits = np.full((1, 100), -1000.0, dtype=np.float32)
        logits[0, 50] = 1000.0

        params = SamplingParams(temperature=1.0)
        sample = token_sampler.sample(logits, params)
        assert sample == 50

    def test_nan_handling(self, token_sampler):
        """Test behavior with NaN values."""
        logits = np.zeros((1, 100), dtype=np.float32)
        logits[0, 0] = 5.0
        logits[0, 50] = np.nan

        params = SamplingParams(temperature=0.0)

        # NaN should propagate - behavior depends on implementation
        # At minimum, shouldn't crash
        try:
            sample = token_sampler.sample(logits, params)
            # If we get here, check it's a valid token
            assert 0 <= sample < 100
        except (ValueError, RuntimeError):
            pass  # Some implementations may raise

    def test_reproducibility_with_seed(self):
        """Test that seeding produces reproducible results."""
        logits = np.random.randn(1, 100).astype(np.float32)
        params = SamplingParams(temperature=1.0)

        sampler1 = TokenSampler(seed=12345)
        sampler2 = TokenSampler(seed=12345)

        samples1 = [int(sampler1.sample(logits.copy(), params)) for _ in range(10)]
        samples2 = [int(sampler2.sample(logits.copy(), params)) for _ in range(10)]

        assert samples1 == samples2


class TestGetTopTokens:
    """Tests for get_top_tokens method."""

    def test_get_top_tokens_returns_correct_count(self, token_sampler, random_logits):
        """Test that get_top_tokens returns correct number of tokens."""
        top = token_sampler.get_top_tokens(random_logits, k=5)
        assert len(top) == 5

    def test_get_top_tokens_sorted(self, token_sampler, skewed_logits):
        """Test that top tokens are sorted by probability."""
        top = token_sampler.get_top_tokens(skewed_logits, k=3)

        assert top[0]['token_id'] == 42  # Highest
        assert top[1]['token_id'] == 43  # Second
        assert top[2]['token_id'] == 44  # Third

    def test_get_top_tokens_has_probabilities(self, token_sampler, random_logits):
        """Test that top tokens include probabilities."""
        top = token_sampler.get_top_tokens(random_logits, k=5)

        for item in top:
            assert 'token_id' in item
            assert 'logit' in item
            assert 'probability' in item
            assert 0 <= item['probability'] <= 1

    def test_get_top_tokens_probabilities_sum(self, token_sampler):
        """Test that top-k probabilities are reasonable."""
        # Create logits where top 5 have most probability
        logits = np.full((1, 100), -10.0, dtype=np.float32)
        for i in range(5):
            logits[0, i] = 5.0 - i

        top = token_sampler.get_top_tokens(logits, k=5)
        total_prob = sum(item['probability'] for item in top)

        # These 5 should have most of the probability
        assert total_prob > 0.95


class TestBatchedSampling:
    """Tests for batched sampling operations."""

    def test_batched_greedy(self, token_sampler, batch_logits):
        """Test batched greedy sampling."""
        # Set clear winners for each batch element
        batch_logits[0, 10] = 100.0
        batch_logits[1, 20] = 100.0
        batch_logits[2, 30] = 100.0
        batch_logits[3, 40] = 100.0

        params = SamplingParams(temperature=0.0)
        results = token_sampler.sample(batch_logits, params)

        assert len(results) == 4
        assert results[0] == 10
        assert results[1] == 20
        assert results[2] == 30
        assert results[3] == 40

    def test_batched_temperature(self, token_sampler):
        """Test batched temperature sampling."""
        np.random.seed(42)
        logits = np.random.randn(4, 100).astype(np.float32)

        params = SamplingParams(temperature=1.0, top_k=0, top_p=1.0)

        # Just verify it works and returns correct shape
        results = token_sampler.sample(logits, params)
        assert len(results) == 4

    def test_batched_top_k(self, token_sampler):
        """Test batched top-k sampling."""
        logits = np.zeros((4, 100), dtype=np.float32)
        # Set top 5 for each batch element
        for batch_idx in range(4):
            for i in range(5):
                logits[batch_idx, batch_idx * 10 + i] = 10.0 - i

        params = SamplingParams(temperature=1.0, top_k=3, top_p=1.0)

        results = token_sampler.sample(logits, params)

        # Each result should be from that batch element's top 3
        assert 0 <= results[0] < 5
        assert 10 <= results[1] < 15
        assert 20 <= results[2] < 25
        assert 30 <= results[3] < 35


@pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")
class TestTorchSampling:
    """Tests for torch tensor sampling."""

    def test_torch_greedy(self, token_sampler, torch_skewed_logits):
        """Test greedy sampling with torch tensors."""
        if torch_skewed_logits is None:
            pytest.skip("Torch not available")

        result = token_sampler.sample_greedy(torch_skewed_logits)
        assert result.item() == 42

    def test_torch_temperature(self, token_sampler, torch_random_logits):
        """Test temperature sampling with torch tensors."""
        if torch_random_logits is None:
            pytest.skip("Torch not available")

        params = SamplingParams(temperature=1.0)
        result = token_sampler.sample(torch_random_logits, params)

        # Result may be tensor or int depending on batch size
        result_val = result.item() if hasattr(result, 'item') else result
        assert 0 <= result_val < 1000

    def test_torch_top_k(self, token_sampler, torch_random_logits):
        """Test top-k with torch tensors."""
        if torch_random_logits is None:
            pytest.skip("Torch not available")

        params = SamplingParams(temperature=1.0, top_k=10)
        result = token_sampler.sample(torch_random_logits, params)

        # Result may be tensor or int depending on batch size
        result_val = result.item() if hasattr(result, 'item') else result
        assert 0 <= result_val < 1000

    def test_torch_batched(self, token_sampler):
        """Test batched torch tensor sampling."""
        logits = torch.randn(4, 100)
        logits[0, 10] = 100.0
        logits[1, 20] = 100.0
        logits[2, 30] = 100.0
        logits[3, 40] = 100.0

        params = SamplingParams(temperature=0.0)
        results = token_sampler.sample(logits, params)

        assert len(results) == 4
        assert results[0].item() == 10
        assert results[1].item() == 20
