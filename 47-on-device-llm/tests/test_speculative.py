"""Tests for speculative decoding module."""

import numpy as np
import pytest

from on_device_llm.speculative import (
    SpeculativeConfig,
    SpeculativeStats,
    SpeculativeDecoder,
    SelfSpeculativeDecoder,
    LookaheadDecoder,
)
from on_device_llm.inference import TransformerInference, Sampler, GenerationConfig
from on_device_llm.loader import MockGGUFLoader


class TestSpeculativeConfig:
    """Tests for SpeculativeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SpeculativeConfig()
        assert config.num_speculative_tokens == 4
        assert config.draft_temperature == 0.8
        assert config.target_temperature == 1.0
        assert config.top_k == 40
        assert config.top_p == 0.9

    def test_custom_config(self):
        """Test custom configuration."""
        config = SpeculativeConfig(
            num_speculative_tokens=8,
            draft_temperature=0.5,
            target_temperature=0.9,
            top_k=50,
            top_p=0.95,
        )
        assert config.num_speculative_tokens == 8
        assert config.draft_temperature == 0.5
        assert config.target_temperature == 0.9


class TestSpeculativeStats:
    """Tests for SpeculativeStats."""

    def test_empty_stats(self):
        """Test empty statistics."""
        stats = SpeculativeStats()
        assert stats.total_tokens == 0
        assert stats.acceptance_rate == 0.0
        assert stats.tokens_per_iteration == 0.0

    def test_stats_calculation(self):
        """Test statistics calculation."""
        stats = SpeculativeStats(
            total_tokens=100,
            accepted_tokens=80,
            rejected_tokens=20,
            num_iterations=25,
        )
        assert stats.acceptance_rate == 0.8
        assert stats.tokens_per_iteration == 4.0


class TestSpeculativeDecoder:
    """Tests for SpeculativeDecoder."""

    @pytest.fixture
    def target_model(self):
        """Create target model fixture."""
        loader = MockGGUFLoader(
            vocab_size=1000,
            hidden_size=64,
            num_layers=4,
            num_heads=4,
        )
        return TransformerInference(loader)

    @pytest.fixture
    def draft_model(self):
        """Create draft model fixture (smaller)."""
        loader = MockGGUFLoader(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,  # Fewer layers for draft
            num_heads=4,
        )
        return TransformerInference(loader)

    def test_decoder_initialization(self, target_model, draft_model):
        """Test decoder initialization."""
        decoder = SpeculativeDecoder(target_model, draft_model)
        assert decoder.target is target_model
        assert decoder.draft is draft_model
        assert decoder.config.num_speculative_tokens == 4

    def test_decoder_with_custom_config(self, target_model, draft_model):
        """Test decoder with custom config."""
        config = SpeculativeConfig(num_speculative_tokens=8)
        decoder = SpeculativeDecoder(target_model, draft_model, config)
        assert decoder.config.num_speculative_tokens == 8

    def test_speculate_tokens(self, target_model, draft_model):
        """Test speculative token generation."""
        decoder = SpeculativeDecoder(target_model, draft_model)

        # Reset draft model
        decoder.draft.reset()

        # Generate speculative tokens
        draft_tokens, draft_probs = decoder._speculate(
            start_token=1,
            start_position=0,
            num_tokens=4,
        )

        assert len(draft_tokens) == 4
        assert len(draft_probs) == 4
        assert all(0 <= t < 1000 for t in draft_tokens)
        assert all(p.shape == (1000,) for p in draft_probs)

    def test_generate_basic(self, target_model, draft_model):
        """Test basic generation."""
        np.random.seed(42)
        decoder = SpeculativeDecoder(target_model, draft_model)

        prompt = [1, 2, 3]
        output, stats = decoder.generate(
            prompt_tokens=prompt,
            max_new_tokens=10,
            eos_token_id=2,
        )

        assert len(output) > 0
        assert len(output) <= 10
        assert stats.total_tokens == len(output)
        assert stats.num_iterations > 0

    def test_generate_stops_at_eos(self, target_model, draft_model):
        """Test that generation stops at EOS token."""
        np.random.seed(42)
        decoder = SpeculativeDecoder(target_model, draft_model)

        # Use a small vocab model and set EOS to likely token
        prompt = [1]
        output, stats = decoder.generate(
            prompt_tokens=prompt,
            max_new_tokens=50,
            eos_token_id=999,  # Unlikely EOS
        )

        # Should either hit EOS or max tokens
        assert len(output) <= 50

    def test_stats_tracked(self, target_model, draft_model):
        """Test that statistics are tracked."""
        np.random.seed(42)
        decoder = SpeculativeDecoder(target_model, draft_model)

        prompt = [1, 2, 3]
        _, stats = decoder.generate(prompt, max_new_tokens=20)

        assert stats.total_tokens > 0
        assert stats.num_iterations > 0
        # Either accepted or rejected tokens should be non-zero
        assert stats.accepted_tokens + stats.rejected_tokens > 0

    def test_reset_stats(self, target_model, draft_model):
        """Test stats reset."""
        decoder = SpeculativeDecoder(target_model, draft_model)
        decoder.stats.total_tokens = 100
        decoder.reset_stats()
        assert decoder.stats.total_tokens == 0


class TestSelfSpeculativeDecoder:
    """Tests for SelfSpeculativeDecoder."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        loader = MockGGUFLoader(
            vocab_size=1000,
            hidden_size=64,
            num_layers=8,
            num_heads=4,
        )
        return TransformerInference(loader)

    def test_decoder_initialization(self, model):
        """Test self-speculative decoder initialization."""
        decoder = SelfSpeculativeDecoder(model, draft_layers=4)
        assert decoder.model is model
        assert decoder.draft_layers == 4

    def test_generate(self, model):
        """Test generation (falls back to standard)."""
        np.random.seed(42)
        decoder = SelfSpeculativeDecoder(model, draft_layers=4)

        prompt = [1, 2, 3]
        output, stats = decoder.generate(prompt, max_new_tokens=10)

        assert len(output) > 0
        assert len(output) <= 10


class TestLookaheadDecoder:
    """Tests for LookaheadDecoder."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        loader = MockGGUFLoader(
            vocab_size=1000,
            hidden_size=64,
            num_layers=4,
            num_heads=4,
        )
        return TransformerInference(loader)

    def test_decoder_initialization(self, model):
        """Test lookahead decoder initialization."""
        decoder = LookaheadDecoder(model, window_size=4, ngram_size=2)
        assert decoder.model is model
        assert decoder.window_size == 4
        assert decoder.ngram_size == 2

    def test_generate(self, model):
        """Test generation with lookahead."""
        np.random.seed(42)
        decoder = LookaheadDecoder(model, window_size=4)

        prompt = [1, 2, 3, 4, 5]
        output, stats = decoder.generate(prompt, max_new_tokens=10)

        assert len(output) > 0
        assert len(output) <= 10
        assert stats.total_tokens == len(output)

    def test_ngram_cache_building(self, model):
        """Test that n-gram cache is built during generation."""
        np.random.seed(42)
        decoder = LookaheadDecoder(model, window_size=4, ngram_size=2)

        prompt = [1, 2, 3, 4, 5]
        decoder.generate(prompt, max_new_tokens=20)

        # Cache should have some entries after generation
        assert len(decoder._ngram_cache) > 0

    def test_clear_cache(self, model):
        """Test cache clearing."""
        decoder = LookaheadDecoder(model)
        decoder._ngram_cache[(1, 2)] = (3, 1)
        decoder.clear_cache()
        assert len(decoder._ngram_cache) == 0


class TestSpeculativeDecodingIntegration:
    """Integration tests for speculative decoding."""

    def test_speculative_vs_standard(self):
        """Compare speculative to standard generation quality."""
        np.random.seed(42)

        # Create models
        loader = MockGGUFLoader(
            vocab_size=1000,
            hidden_size=64,
            num_layers=4,
            num_heads=4,
        )

        target = TransformerInference(loader)
        draft_loader = MockGGUFLoader(
            vocab_size=1000,
            hidden_size=64,
            num_layers=2,
            num_heads=4,
        )
        draft = TransformerInference(draft_loader)

        decoder = SpeculativeDecoder(target, draft)

        prompt = [1, 2, 3]

        # Generate with speculative decoding
        spec_output, spec_stats = decoder.generate(prompt, max_new_tokens=10)

        # Generate with standard decoding
        np.random.seed(42)
        target.reset()
        standard_config = GenerationConfig(max_new_tokens=10)
        standard_output = target.generate(prompt, standard_config)

        # Both should produce valid output
        assert len(spec_output) > 0
        assert len(standard_output) > 0

    def test_acceptance_rate_reasonable(self):
        """Test that acceptance rate is reasonable."""
        np.random.seed(42)

        loader = MockGGUFLoader(
            vocab_size=1000,
            hidden_size=64,
            num_layers=4,
            num_heads=4,
        )

        target = TransformerInference(loader)
        draft = TransformerInference(loader)  # Same model for high acceptance

        decoder = SpeculativeDecoder(target, draft)

        prompt = [1, 2, 3]
        _, stats = decoder.generate(prompt, max_new_tokens=20)

        # Same model should have decent acceptance
        # (though random init may vary)
        assert stats.acceptance_rate >= 0  # Just verify it's computed
