"""Tests for the inference engine and generation."""

import numpy as np
import pytest

from on_device_llm.loader import ModelConfig, MockGGUFLoader
from on_device_llm.memory import KVCache
from on_device_llm.inference import (
    TransformerInference,
    Sampler,
    GenerationConfig,
    BatchInference,
)
from on_device_llm.quantization import GGMLType


class TestModelConfig:
    """Tests for ModelConfig."""

    def test_model_config_creation(self, tiny_config):
        """Test creating a model config."""
        assert tiny_config.vocab_size == 256
        assert tiny_config.hidden_size == 64
        assert tiny_config.num_layers == 2
        assert tiny_config.num_heads == 4
        assert tiny_config.num_kv_heads == 4

    def test_head_dim_property(self, tiny_config):
        """Test head_dim is computed correctly."""
        expected = tiny_config.hidden_size // tiny_config.num_heads
        assert tiny_config.head_dim == expected
        assert tiny_config.head_dim == 16

    def test_gqa_config(self, gqa_config):
        """Test GQA configuration."""
        assert gqa_config.num_heads == 8
        assert gqa_config.num_kv_heads == 2
        # GQA ratio is 4:1
        assert gqa_config.num_heads // gqa_config.num_kv_heads == 4


class TestMockGGUFLoader:
    """Tests for MockGGUFLoader."""

    def test_mock_loader_creation(self, mock_loader):
        """Test creating a mock loader."""
        assert mock_loader.config is not None
        assert len(mock_loader.tensors) > 0

    def test_mock_loader_tensor_names(self, mock_loader):
        """Test mock loader has expected tensor names."""
        tensor_names = mock_loader.list_tensors()

        # Check for embedding
        assert 'token_embd.weight' in tensor_names

        # Check for layer tensors
        assert 'blk.0.attn_norm.weight' in tensor_names
        assert 'blk.0.attn_q.weight' in tensor_names
        assert 'blk.0.attn_k.weight' in tensor_names
        assert 'blk.0.attn_v.weight' in tensor_names
        assert 'blk.0.attn_output.weight' in tensor_names
        assert 'blk.0.ffn_norm.weight' in tensor_names
        assert 'blk.0.ffn_gate.weight' in tensor_names
        assert 'blk.0.ffn_up.weight' in tensor_names
        assert 'blk.0.ffn_down.weight' in tensor_names

        # Check for output
        assert 'output_norm.weight' in tensor_names
        assert 'output.weight' in tensor_names

    def test_mock_loader_tensor_shapes(self, mock_loader, tiny_config):
        """Test mock loader produces correct tensor shapes."""
        embed = mock_loader.get_tensor('token_embd.weight')
        assert embed.shape == (tiny_config.vocab_size, tiny_config.hidden_size)

        wq = mock_loader.get_tensor('blk.0.attn_q.weight')
        assert wq.shape == (
            tiny_config.num_heads * tiny_config.head_dim,
            tiny_config.hidden_size
        )

    def test_mock_loader_get_tensor_returns_copy(self, mock_loader):
        """Test get_tensor returns a copy (modifications don't affect original)."""
        tensor1 = mock_loader.get_tensor('token_embd.weight')
        tensor1[0, 0] = 999.0

        tensor2 = mock_loader.get_tensor('token_embd.weight')
        assert tensor2[0, 0] != 999.0

    def test_mock_loader_missing_tensor(self, mock_loader):
        """Test accessing non-existent tensor raises error."""
        with pytest.raises(KeyError):
            mock_loader.get_tensor('nonexistent.weight')


class TestKVCache:
    """Tests for KV cache."""

    def test_kv_cache_creation(self, kv_cache, tiny_config):
        """Test creating KV cache."""
        assert kv_cache.num_layers == tiny_config.num_layers
        assert kv_cache.num_heads == tiny_config.num_kv_heads
        assert kv_cache.head_dim == tiny_config.head_dim
        assert kv_cache.max_seq_len == tiny_config.max_seq_len
        assert kv_cache.length == 0

    def test_kv_cache_append(self, kv_cache, tiny_config):
        """Test appending to KV cache."""
        k = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)
        v = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)

        # Append to all layers
        for layer_idx in range(tiny_config.num_layers):
            kv_cache.append(layer_idx, k, v)

        assert kv_cache.length == 1

    def test_kv_cache_get(self, kv_cache, tiny_config):
        """Test getting from KV cache."""
        k = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)
        v = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)

        for layer_idx in range(tiny_config.num_layers):
            kv_cache.append(layer_idx, k, v)

        k_cache, v_cache = kv_cache.get(0)

        assert k_cache.shape == (1, tiny_config.num_kv_heads, tiny_config.head_dim)
        assert v_cache.shape == (1, tiny_config.num_kv_heads, tiny_config.head_dim)

    def test_kv_cache_multiple_tokens(self, kv_cache, tiny_config):
        """Test caching multiple tokens."""
        num_tokens = 5

        for _ in range(num_tokens):
            k = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)
            v = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)

            for layer_idx in range(tiny_config.num_layers):
                kv_cache.append(layer_idx, k, v)

        assert kv_cache.length == num_tokens

        k_cache, v_cache = kv_cache.get(0)
        assert k_cache.shape[0] == num_tokens

    def test_kv_cache_clear(self, kv_cache, tiny_config):
        """Test clearing KV cache."""
        k = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)
        v = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)

        for layer_idx in range(tiny_config.num_layers):
            kv_cache.append(layer_idx, k, v)

        kv_cache.clear()

        assert kv_cache.length == 0

    def test_kv_cache_memory_usage(self, kv_cache, tiny_config):
        """Test memory usage calculation."""
        expected_size = (
            2 *  # k and v
            tiny_config.num_layers *
            tiny_config.max_seq_len *
            tiny_config.num_kv_heads *
            tiny_config.head_dim *
            4  # float32
        )

        assert kv_cache.memory_usage() == expected_size

    def test_sliding_window_cache(self, sliding_window_cache, tiny_config):
        """Test sliding window KV cache."""
        window_size = sliding_window_cache.window_size
        assert window_size == 8

        # Add more tokens than window size
        for i in range(window_size + 5):
            k = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)
            v = np.random.randn(tiny_config.num_kv_heads, tiny_config.head_dim).astype(np.float32)

            for layer_idx in range(tiny_config.num_layers):
                sliding_window_cache.append(layer_idx, k, v)

        # Should be capped at window size
        assert sliding_window_cache.effective_length == window_size


class TestSampler:
    """Tests for token sampling."""

    def test_sampler_creation(self, sampler):
        """Test creating a sampler."""
        assert sampler.temperature == 1.0
        assert sampler.top_k == 10
        assert sampler.top_p == 0.9

    def test_sampler_sample_returns_valid_token(self, sampler):
        """Test sampling returns valid token index."""
        logits = np.random.randn(100).astype(np.float32)
        token = sampler.sample(logits)

        assert isinstance(token, int)
        assert 0 <= token < len(logits)

    def test_sampler_greedy(self, sampler):
        """Test greedy decoding."""
        logits = np.array([0.1, 0.2, 10.0, 0.3, 0.4], dtype=np.float32)
        token = sampler.greedy(logits)

        assert token == 2  # Highest logit

    def test_sampler_temperature_effect(self):
        """Test temperature affects sampling distribution."""
        logits = np.array([0.0, 1.0, 2.0, 1.0, 0.0] * 20, dtype=np.float32)

        # Low temperature should favor high logit tokens
        low_temp_sampler = Sampler(temperature=0.1, top_k=0, top_p=1.0)
        high_temp_sampler = Sampler(temperature=10.0, top_k=0, top_p=1.0)

        low_temp_tokens = [low_temp_sampler.sample(logits.copy()) for _ in range(100)]
        high_temp_tokens = [high_temp_sampler.sample(logits.copy()) for _ in range(100)]

        # Low temperature should have less variety
        low_temp_unique = len(set(low_temp_tokens))
        high_temp_unique = len(set(high_temp_tokens))

        assert low_temp_unique <= high_temp_unique

    def test_sampler_top_k_filtering(self):
        """Test top-k filtering."""
        logits = np.arange(100).astype(np.float32)

        sampler = Sampler(temperature=1.0, top_k=5, top_p=1.0)
        tokens = [sampler.sample(logits.copy()) for _ in range(50)]

        # All tokens should be from top-k
        for token in tokens:
            assert token >= 95  # Top 5 indices

    def test_sampler_reset(self, sampler):
        """Test sampler reset."""
        logits = np.random.randn(100).astype(np.float32)
        sampler.sample(logits)
        sampler.sample(logits)

        assert len(sampler._generated_tokens) == 2

        sampler.reset()

        assert len(sampler._generated_tokens) == 0

    def test_sampler_from_config(self, generation_config):
        """Test creating sampler from config."""
        sampler = Sampler.from_config(generation_config)

        assert sampler.temperature == generation_config.temperature
        assert sampler.top_k == generation_config.top_k
        assert sampler.top_p == generation_config.top_p


class TestGenerationConfig:
    """Tests for GenerationConfig."""

    def test_generation_config_defaults(self):
        """Test default generation config values."""
        config = GenerationConfig()

        assert config.max_new_tokens == 100
        assert config.temperature == 1.0
        assert config.top_k == 40
        assert config.top_p == 0.9
        assert config.do_sample == True

    def test_generation_config_validation_temperature(self):
        """Test temperature validation."""
        config = GenerationConfig(temperature=0)

        with pytest.raises(ValueError, match="temperature"):
            config.validate()

    def test_generation_config_validation_top_k(self):
        """Test top_k validation."""
        config = GenerationConfig(top_k=-1)

        with pytest.raises(ValueError, match="top_k"):
            config.validate()

    def test_generation_config_validation_top_p(self):
        """Test top_p validation."""
        config = GenerationConfig(top_p=1.5)

        with pytest.raises(ValueError, match="top_p"):
            config.validate()

    def test_generation_config_validation_valid(self):
        """Test valid config passes validation."""
        config = GenerationConfig(
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )
        config.validate()  # Should not raise


class TestTransformerInference:
    """Tests for TransformerInference engine."""

    def test_inference_engine_creation(self, inference_engine, tiny_config):
        """Test creating inference engine."""
        assert inference_engine.config == tiny_config
        assert inference_engine.kv_cache is not None
        assert inference_engine._embed_tokens is not None

    def test_inference_forward_shape(self, inference_engine, tiny_config):
        """Test forward pass output shape."""
        token_id = 0
        position = 0

        logits = inference_engine.forward(token_id, position)

        assert logits.shape == (tiny_config.vocab_size,)
        assert logits.dtype == np.float32

    def test_inference_forward_updates_cache(self, inference_engine):
        """Test forward pass updates KV cache."""
        initial_length = inference_engine.kv_cache.length

        inference_engine.forward(0, 0)

        assert inference_engine.kv_cache.length == initial_length + 1

    def test_inference_multiple_tokens(self, inference_engine, tiny_config):
        """Test inference with multiple tokens."""
        for i in range(5):
            token_id = i % tiny_config.vocab_size
            logits = inference_engine.forward(token_id, i)

            assert logits.shape == (tiny_config.vocab_size,)

        assert inference_engine.kv_cache.length == 5

    def test_inference_reset(self, inference_engine):
        """Test inference reset."""
        inference_engine.forward(0, 0)
        inference_engine.forward(1, 1)

        inference_engine.reset()

        assert inference_engine.kv_cache.length == 0

    def test_inference_memory_usage(self, inference_engine):
        """Test memory usage tracking."""
        mem_before = inference_engine.memory_usage()

        # Memory should be same since cache is pre-allocated
        inference_engine.forward(0, 0)
        mem_after = inference_engine.memory_usage()

        assert mem_before == mem_after

    def test_inference_gqa_support(self, inference_engine_gqa, gqa_config):
        """Test inference with Grouped Query Attention."""
        # GQA should work correctly
        logits = inference_engine_gqa.forward(0, 0)

        assert logits.shape == (gqa_config.vocab_size,)

    def test_inference_generate(self, inference_engine, tiny_config, generation_config):
        """Test text generation."""
        prompt_tokens = [1, 2, 3]
        generation_config.max_new_tokens = 5
        generation_config.eos_token_id = -1  # Disable early stopping

        output_tokens = inference_engine.generate(prompt_tokens, generation_config)

        assert len(output_tokens) == 5
        assert all(0 <= t < tiny_config.vocab_size for t in output_tokens)

    def test_inference_generate_eos_stop(self, inference_engine, tiny_config):
        """Test generation stops at EOS token."""
        prompt_tokens = [1, 2, 3]

        # Force model to produce EOS (token 0)
        config = GenerationConfig(
            max_new_tokens=100,
            eos_token_id=0,
            temperature=100.0,  # Very high temp for randomness
            do_sample=True
        )

        output_tokens = inference_engine.generate(prompt_tokens, config)

        # Should stop when EOS is generated
        assert len(output_tokens) <= 100

    def test_inference_greedy_generate(self, inference_engine, greedy_config):
        """Test greedy generation."""
        prompt_tokens = [1, 2, 3]

        output_tokens = inference_engine.generate(prompt_tokens, greedy_config)

        assert len(output_tokens) <= greedy_config.max_new_tokens


class TestBatchInference:
    """Tests for batch inference."""

    def test_batch_inference_creation(self, mock_loader):
        """Test creating batch inference engine."""
        batch = BatchInference(mock_loader, max_batch_size=4)

        assert batch.max_batch_size == 4
        assert len(batch._engines) == 4

    def test_batch_forward(self, mock_loader, tiny_config):
        """Test batch forward pass."""
        batch = BatchInference(mock_loader, max_batch_size=4)

        token_ids = [0, 1, 2]
        positions = [0, 0, 0]
        batch_indices = [0, 1, 2]

        results = batch.forward_batch(token_ids, positions, batch_indices)

        assert len(results) == 3
        assert all(r.shape == (tiny_config.vocab_size,) for r in results)

    def test_batch_forward_invalid_index(self, mock_loader):
        """Test batch forward with invalid batch index."""
        batch = BatchInference(mock_loader, max_batch_size=2)

        with pytest.raises(ValueError):
            batch.forward_batch([0], [0], [5])  # Index 5 > max_batch_size 2

    def test_batch_reset_all(self, mock_loader):
        """Test resetting all batch engines."""
        batch = BatchInference(mock_loader, max_batch_size=2)

        # Do some forward passes
        batch.forward_batch([0, 1], [0, 0], [0, 1])

        batch.reset()

        # All caches should be empty
        for engine in batch._engines:
            assert engine.kv_cache.length == 0

    def test_batch_reset_single(self, mock_loader):
        """Test resetting single batch engine."""
        batch = BatchInference(mock_loader, max_batch_size=2)

        batch.forward_batch([0, 1], [0, 0], [0, 1])

        batch.reset(batch_index=0)

        assert batch._engines[0].kv_cache.length == 0
        assert batch._engines[1].kv_cache.length == 1


class TestInferenceNumericalStability:
    """Tests for numerical stability of inference."""

    def test_forward_no_nan(self, inference_engine, tiny_config):
        """Test forward pass produces no NaN values."""
        for i in range(10):
            token_id = i % tiny_config.vocab_size
            logits = inference_engine.forward(token_id, i)

            assert not np.any(np.isnan(logits)), f"NaN at position {i}"
            assert not np.any(np.isinf(logits)), f"Inf at position {i}"

    def test_forward_reasonable_range(self, inference_engine, tiny_config):
        """Test forward pass produces reasonable logit values."""
        for i in range(10):
            token_id = i % tiny_config.vocab_size
            logits = inference_engine.forward(token_id, i)

            # Logits should be in a reasonable range
            assert np.max(np.abs(logits)) < 1000

    def test_long_sequence_stability(self, inference_engine, tiny_config):
        """Test stability over longer sequences."""
        max_len = min(tiny_config.max_seq_len - 1, 20)

        for i in range(max_len):
            token_id = i % tiny_config.vocab_size
            logits = inference_engine.forward(token_id, i)

            assert not np.any(np.isnan(logits))
            assert not np.any(np.isinf(logits))
