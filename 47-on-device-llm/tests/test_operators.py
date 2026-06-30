"""Tests for SIMD-optimized operators."""

import numpy as np
import pytest

from on_device_llm.operators import (
    matmul_f32,
    rmsnorm,
    softmax,
    rope_embed,
    silu,
    gelu,
    attention_scores,
    weighted_sum,
    layer_norm,
    apply_rope_batch,
    compute_causal_mask,
    HAS_NUMBA,
)


class TestMatmul:
    """Tests for matrix multiplication."""

    def test_matmul_basic(self, random_matrix_a, random_matrix_b):
        """Test basic matrix multiplication."""
        result = matmul_f32(random_matrix_a, random_matrix_b)

        expected = random_matrix_a @ random_matrix_b

        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_matmul_shape(self, random_matrix_a, random_matrix_b):
        """Test matmul output shape."""
        result = matmul_f32(random_matrix_a, random_matrix_b)

        # A is [16, 32], B is [32, 24] -> result is [16, 24]
        assert result.shape == (16, 24)

    def test_matmul_dtype(self, random_matrix_a, random_matrix_b):
        """Test matmul output dtype."""
        result = matmul_f32(random_matrix_a, random_matrix_b)

        assert result.dtype == np.float32

    def test_matmul_square_matrices(self):
        """Test matmul with square matrices."""
        a = np.random.randn(32, 32).astype(np.float32)
        b = np.random.randn(32, 32).astype(np.float32)

        result = matmul_f32(a, b)
        expected = a @ b

        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_matmul_vector_matrix(self):
        """Test matmul with 1xN vector and NxM matrix."""
        a = np.random.randn(1, 64).astype(np.float32)
        b = np.random.randn(64, 32).astype(np.float32)

        result = matmul_f32(a, b)
        expected = a @ b

        np.testing.assert_array_almost_equal(result, expected, decimal=4)

    def test_matmul_identity(self):
        """Test matmul with identity matrix."""
        a = np.random.randn(16, 16).astype(np.float32)
        identity = np.eye(16, dtype=np.float32)

        result = matmul_f32(a, identity)

        np.testing.assert_array_almost_equal(result, a, decimal=5)

    def test_matmul_zeros(self):
        """Test matmul with zero matrix."""
        a = np.random.randn(8, 8).astype(np.float32)
        zeros = np.zeros((8, 8), dtype=np.float32)

        result = matmul_f32(a, zeros)

        np.testing.assert_array_almost_equal(result, zeros, decimal=5)


class TestRMSNorm:
    """Tests for RMS normalization."""

    def test_rmsnorm_basic(self, random_tensor_1d, random_weights):
        """Test basic RMS normalization."""
        result = rmsnorm(random_tensor_1d, random_weights)

        assert result.shape == random_tensor_1d.shape
        assert result.dtype == np.float32

    def test_rmsnorm_unit_weights(self, random_tensor_1d):
        """Test RMSNorm with unit weights."""
        weights = np.ones_like(random_tensor_1d)
        result = rmsnorm(random_tensor_1d, weights)

        # Result should be normalized
        rms = np.sqrt(np.mean(result ** 2))
        np.testing.assert_almost_equal(rms, 1.0, decimal=5)

    def test_rmsnorm_zero_input(self, random_weights):
        """Test RMSNorm with zero input (tests eps stability)."""
        x = np.zeros(64, dtype=np.float32)
        result = rmsnorm(x, random_weights)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_rmsnorm_preserves_scale(self):
        """Test RMSNorm with known values."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        weights = np.ones(4, dtype=np.float32)

        result = rmsnorm(x, weights)

        # RMS of [1,2,3,4] is sqrt(30/4) = sqrt(7.5)
        expected_rms = np.sqrt(np.mean(x ** 2))
        expected = x / expected_rms

        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_rmsnorm_different_eps(self):
        """Test RMSNorm with different epsilon values."""
        x = np.random.randn(32).astype(np.float32) * 0.001  # Small values
        weights = np.ones(32, dtype=np.float32)

        result1 = rmsnorm(x, weights, eps=1e-5)
        result2 = rmsnorm(x, weights, eps=1e-8)

        # Both should be valid (no NaN/Inf)
        assert not np.any(np.isnan(result1))
        assert not np.any(np.isnan(result2))


class TestSoftmax:
    """Tests for softmax activation."""

    def test_softmax_basic(self, random_tensor_1d):
        """Test basic softmax."""
        result = softmax(random_tensor_1d)

        assert result.shape == random_tensor_1d.shape
        assert result.dtype == np.float32

    def test_softmax_sums_to_one(self, random_tensor_1d):
        """Test softmax outputs sum to 1."""
        result = softmax(random_tensor_1d)

        np.testing.assert_almost_equal(result.sum(), 1.0, decimal=5)

    def test_softmax_all_positive(self, random_tensor_1d):
        """Test softmax outputs are all positive."""
        result = softmax(random_tensor_1d)

        assert np.all(result >= 0)

    def test_softmax_numerical_stability(self):
        """Test softmax with large values (numerical stability)."""
        x = np.array([1000.0, 1001.0, 1002.0], dtype=np.float32)
        result = softmax(x)

        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        np.testing.assert_almost_equal(result.sum(), 1.0, decimal=5)

    def test_softmax_negative_values(self):
        """Test softmax with all negative values."""
        x = np.array([-10.0, -20.0, -5.0], dtype=np.float32)
        result = softmax(x)

        np.testing.assert_almost_equal(result.sum(), 1.0, decimal=5)
        assert np.argmax(result) == 2  # -5 is largest

    def test_softmax_uniform_input(self):
        """Test softmax with uniform input."""
        x = np.ones(10, dtype=np.float32)
        result = softmax(x)

        # Should be uniform distribution
        expected = np.ones(10) / 10
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_softmax_known_values(self):
        """Test softmax with known expected output."""
        x = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        result = softmax(x)

        # Middle value has e^1, others have e^0
        expected = np.array([1.0, np.e, 1.0]) / (2 + np.e)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)


class TestRoPE:
    """Tests for Rotary Position Embeddings."""

    def test_rope_basic(self, random_embedding):
        """Test basic RoPE embedding."""
        result = rope_embed(random_embedding, pos=0)

        assert result.shape == random_embedding.shape
        assert result.dtype == np.float32

    def test_rope_position_zero(self, random_embedding):
        """Test RoPE at position 0."""
        result = rope_embed(random_embedding, pos=0)

        # At position 0, cos(0)=1, sin(0)=0
        # So x[i] stays as x[i]*1 - x[i+1]*0 = x[i]
        # and x[i+1] stays as x[i]*0 + x[i+1]*1 = x[i+1]
        np.testing.assert_array_almost_equal(result, random_embedding, decimal=5)

    def test_rope_preserves_norm(self, random_embedding):
        """Test RoPE preserves norm of embedding pairs."""
        result = rope_embed(random_embedding, pos=5)

        # Each pair should have same norm before and after rotation
        for i in range(0, len(random_embedding), 2):
            original_norm = np.linalg.norm(random_embedding[i:i+2])
            result_norm = np.linalg.norm(result[i:i+2])
            np.testing.assert_almost_equal(original_norm, result_norm, decimal=5)

    def test_rope_different_positions(self, random_embedding):
        """Test RoPE produces different results for different positions."""
        result1 = rope_embed(random_embedding, pos=1)
        result2 = rope_embed(random_embedding, pos=10)

        assert not np.allclose(result1, result2)

    def test_rope_theta_effect(self, random_embedding):
        """Test different theta values produce different results."""
        result1 = rope_embed(random_embedding, pos=5, theta=10000.0)
        result2 = rope_embed(random_embedding, pos=5, theta=500000.0)

        assert not np.allclose(result1, result2)

    def test_rope_rotation_property(self):
        """Test RoPE rotation property: r(x, p) rotates by p*freq."""
        x = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)

        # After rotation, the vector should be rotated
        result = rope_embed(x, pos=1, theta=10000.0)

        # First pair rotates by freq = 1/theta^(0/4) = 1
        # Second pair rotates by freq = 1/theta^(2/4) = 1/sqrt(theta)
        assert not np.allclose(result, x)


class TestSiLU:
    """Tests for SiLU (Swish) activation."""

    def test_silu_basic(self, random_tensor_1d):
        """Test basic SiLU."""
        result = silu(random_tensor_1d)

        assert result.shape == random_tensor_1d.shape
        assert result.dtype == np.float32

    def test_silu_zero(self):
        """Test SiLU at zero."""
        x = np.array([0.0], dtype=np.float32)
        result = silu(x)

        np.testing.assert_almost_equal(result[0], 0.0, decimal=5)

    def test_silu_positive(self):
        """Test SiLU with positive values."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = silu(x)

        # SiLU(x) = x * sigmoid(x)
        expected = x * (1.0 / (1.0 + np.exp(-x)))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_silu_negative(self):
        """Test SiLU with negative values."""
        x = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        result = silu(x)

        # SiLU can be slightly negative for negative inputs
        expected = x * (1.0 / (1.0 + np.exp(-x)))
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_silu_numerical_stability(self):
        """Test SiLU with extreme values."""
        x = np.array([-100.0, 100.0], dtype=np.float32)
        result = silu(x)

        assert not np.any(np.isnan(result))
        # SiLU(-100) should be close to 0
        assert result[0] < 0.01
        # SiLU(100) should be close to 100
        assert result[1] > 99


class TestGELU:
    """Tests for GELU activation."""

    def test_gelu_basic(self, random_tensor_1d):
        """Test basic GELU."""
        result = gelu(random_tensor_1d)

        assert result.shape == random_tensor_1d.shape
        assert result.dtype == np.float32

    def test_gelu_zero(self):
        """Test GELU at zero."""
        x = np.array([0.0], dtype=np.float32)
        result = gelu(x)

        np.testing.assert_almost_equal(result[0], 0.0, decimal=5)

    def test_gelu_positive(self):
        """Test GELU with positive values."""
        x = np.array([1.0, 2.0], dtype=np.float32)
        result = gelu(x)

        # GELU is approximately x * Phi(x)
        # For positive values, GELU(x) < x
        assert np.all(result > 0)
        assert np.all(result < x)

    def test_gelu_symmetry(self):
        """Test GELU asymptotic behavior."""
        x_pos = np.array([5.0], dtype=np.float32)
        x_neg = np.array([-5.0], dtype=np.float32)

        result_pos = gelu(x_pos)
        result_neg = gelu(x_neg)

        # GELU(5) should be close to 5
        assert result_pos[0] > 4.9
        # GELU(-5) should be close to 0
        assert abs(result_neg[0]) < 0.01


class TestAttentionScores:
    """Tests for attention score computation."""

    def test_attention_scores_basic(self):
        """Test basic attention score computation."""
        q = np.random.randn(16).astype(np.float32)
        k_cache = np.random.randn(5, 16).astype(np.float32)

        scores = attention_scores(q, k_cache, head_dim=16)

        assert scores.shape == (5,)
        assert scores.dtype == np.float32

    def test_attention_scores_scaling(self):
        """Test attention scores are scaled by sqrt(head_dim)."""
        q = np.ones(16, dtype=np.float32)
        k_cache = np.ones((1, 16), dtype=np.float32)

        scores = attention_scores(q, k_cache, head_dim=16)

        # q.k = 16, scaled by 1/sqrt(16) = 1/4 = 4
        expected = 16.0 / np.sqrt(16)
        np.testing.assert_almost_equal(scores[0], expected, decimal=5)

    def test_attention_scores_empty_cache(self):
        """Test attention with empty cache."""
        q = np.random.randn(16).astype(np.float32)
        k_cache = np.zeros((0, 16), dtype=np.float32)

        scores = attention_scores(q, k_cache, head_dim=16)

        assert len(scores) == 0


class TestWeightedSum:
    """Tests for weighted sum computation."""

    def test_weighted_sum_basic(self):
        """Test basic weighted sum."""
        probs = np.array([0.5, 0.5], dtype=np.float32)
        v_cache = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        result = weighted_sum(probs, v_cache)

        expected = np.array([2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_weighted_sum_single_prob(self):
        """Test weighted sum with single probability of 1."""
        probs = np.array([1.0, 0.0], dtype=np.float32)
        v_cache = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)

        result = weighted_sum(probs, v_cache)

        np.testing.assert_array_almost_equal(result, v_cache[0], decimal=5)

    def test_weighted_sum_shape(self):
        """Test weighted sum output shape."""
        probs = np.random.randn(8).astype(np.float32)
        probs = np.abs(probs) / np.abs(probs).sum()
        v_cache = np.random.randn(8, 32).astype(np.float32)

        result = weighted_sum(probs, v_cache)

        assert result.shape == (32,)


class TestLayerNorm:
    """Tests for layer normalization."""

    def test_layer_norm_basic(self):
        """Test basic layer normalization."""
        x = np.random.randn(64).astype(np.float32)
        weight = np.ones(64, dtype=np.float32)
        bias = np.zeros(64, dtype=np.float32)

        result = layer_norm(x, weight, bias)

        # Should be approximately normalized
        np.testing.assert_almost_equal(np.mean(result), 0.0, decimal=4)
        np.testing.assert_almost_equal(np.std(result), 1.0, decimal=4)

    def test_layer_norm_with_scale_shift(self):
        """Test layer normalization with scale and shift."""
        x = np.random.randn(32).astype(np.float32)
        weight = np.full(32, 2.0, dtype=np.float32)
        bias = np.full(32, 1.0, dtype=np.float32)

        result = layer_norm(x, weight, bias)

        # Should be scaled by 2 and shifted by 1
        np.testing.assert_almost_equal(np.mean(result), 1.0, decimal=3)


class TestApplyRoPEBatch:
    """Tests for batch RoPE application."""

    def test_rope_batch_basic(self):
        """Test batch RoPE application."""
        x = np.random.randn(4, 16).astype(np.float32)
        positions = np.array([0, 1, 2, 3])

        result = apply_rope_batch(x, positions)

        assert result.shape == x.shape

    def test_rope_batch_matches_individual(self, random_embedding):
        """Test batch RoPE matches individual applications."""
        batch = np.stack([random_embedding, random_embedding])
        positions = np.array([5, 10])

        batch_result = apply_rope_batch(batch, positions)

        individual_0 = rope_embed(random_embedding, 5)
        individual_1 = rope_embed(random_embedding, 10)

        np.testing.assert_array_almost_equal(batch_result[0], individual_0, decimal=5)
        np.testing.assert_array_almost_equal(batch_result[1], individual_1, decimal=5)


class TestCausalMask:
    """Tests for causal attention mask."""

    def test_causal_mask_shape(self):
        """Test causal mask shape."""
        mask = compute_causal_mask(8)

        assert mask.shape == (8, 8)
        assert mask.dtype == np.float32

    def test_causal_mask_lower_triangular(self):
        """Test causal mask is lower triangular."""
        mask = compute_causal_mask(4)

        # Lower triangular should be 0, upper triangular should be -inf
        assert mask[0, 0] == 0
        assert mask[1, 0] == 0
        assert mask[1, 1] == 0
        assert mask[0, 1] == float('-inf')
        assert mask[0, 3] == float('-inf')

    def test_causal_mask_prevents_future_attention(self):
        """Test causal mask prevents attending to future tokens."""
        mask = compute_causal_mask(5)

        # Adding mask to scores should make future positions -inf
        scores = np.ones((5, 5), dtype=np.float32)
        masked_scores = scores + mask

        # Past and current positions should be finite
        assert all(np.isfinite(masked_scores[i, :i+1]).all() for i in range(5))

        # Future positions should be -inf
        for i in range(5):
            for j in range(i+1, 5):
                assert masked_scores[i, j] == float('-inf')


class TestNumbaAvailability:
    """Tests related to Numba availability."""

    def test_numba_status_is_bool(self):
        """Test HAS_NUMBA is a boolean."""
        assert isinstance(HAS_NUMBA, bool)

    def test_operators_work_without_numba(self):
        """Test operators work regardless of Numba availability."""
        # These should work with or without Numba
        x = np.random.randn(32).astype(np.float32)
        w = np.abs(np.random.randn(32).astype(np.float32)) + 0.1

        result = rmsnorm(x, w)
        assert not np.any(np.isnan(result))

        result = softmax(x)
        assert not np.any(np.isnan(result))

        result = silu(x)
        assert not np.any(np.isnan(result))


class TestOperatorConsistency:
    """Tests for operator consistency and correctness."""

    def test_rmsnorm_vs_manual(self):
        """Test RMSNorm matches manual computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        w = np.ones(4, dtype=np.float32)

        result = rmsnorm(x, w)

        # Manual computation
        rms = np.sqrt(np.mean(x ** 2) + 1e-5)
        expected = x / rms

        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_softmax_vs_scipy(self):
        """Test softmax matches expected formula."""
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        result = softmax(x)

        # Manual computation
        exp_x = np.exp(x - np.max(x))
        expected = exp_x / exp_x.sum()

        np.testing.assert_array_almost_equal(result, expected, decimal=5)
