"""Tests for gradient compression."""

import pytest
import numpy as np

from paramserver.enterprise.compression import (
    CompressionType,
    QuantizationCompressor,
    TopKCompressor,
    RandomKCompressor,
)


class TestQuantizationCompressor:
    """Tests for quantization compressor."""

    def test_create_default(self):
        """Test default creation."""
        compressor = QuantizationCompressor()
        assert compressor.bits == 8
        assert compressor.use_error_feedback is True

    def test_create_custom(self):
        """Test custom creation."""
        compressor = QuantizationCompressor(bits=4, use_error_feedback=False)
        assert compressor.bits == 4
        assert compressor.use_error_feedback is False

    def test_invalid_bits(self):
        """Test invalid bits raises error."""
        with pytest.raises(ValueError):
            QuantizationCompressor(bits=0)
        with pytest.raises(ValueError):
            QuantizationCompressor(bits=9)

    def test_compression_type(self):
        """Test compression type property."""
        compressor = QuantizationCompressor()
        assert compressor.compression_type == CompressionType.QUANTIZE

    def test_compress_decompress(self):
        """Test basic compression and decompression."""
        compressor = QuantizationCompressor(bits=8)
        gradient = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        compressed, metadata = compressor.compress(gradient)
        decompressed = compressor.decompress(compressed, metadata)

        # Should be close after round-trip
        np.testing.assert_allclose(decompressed, gradient, rtol=0.1)

    def test_compress_2d_array(self):
        """Test compression of 2D array."""
        compressor = QuantizationCompressor(bits=8)
        gradient = np.random.randn(10, 5).astype(np.float32)

        compressed, metadata = compressor.compress(gradient)
        decompressed = compressor.decompress(compressed, metadata)

        assert decompressed.shape == gradient.shape
        # 8-bit quantization can have ~1% error per value
        np.testing.assert_allclose(decompressed, gradient, rtol=0.15, atol=0.01)

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        compressor = QuantizationCompressor(bits=8)
        original = np.random.randn(100).astype(np.float32)
        compressed, _ = compressor.compress(original)

        ratio = compressor.get_compression_ratio(original, compressed)
        # 4 bytes per float32, 1 byte per uint8 = 4x compression
        assert ratio == pytest.approx(4.0, rel=0.1)

    def test_error_feedback(self):
        """Test error feedback accumulation."""
        compressor = QuantizationCompressor(bits=4, use_error_feedback=True)

        # First compression - use different values to trigger error feedback
        gradient1 = np.array([0.1, 0.5, 0.9, 1.3], dtype=np.float32)
        compressor.compress(gradient1)

        # Error buffer should be populated (not None when values differ)
        assert compressor._error_buffer is not None

        # Reset should clear buffer
        compressor.reset_error_buffer()
        assert compressor._error_buffer is None

    def test_constant_gradient(self):
        """Test compression of constant gradient."""
        compressor = QuantizationCompressor()
        gradient = np.ones(100, dtype=np.float32) * 5.0

        compressed, metadata = compressor.compress(gradient)
        decompressed = compressor.decompress(compressed, metadata)

        np.testing.assert_array_equal(decompressed, gradient)


class TestTopKCompressor:
    """Tests for top-K compressor."""

    def test_create_default(self):
        """Test default creation."""
        compressor = TopKCompressor()
        assert compressor.k_percent == 10.0
        assert compressor.use_error_feedback is True

    def test_create_custom(self):
        """Test custom creation."""
        compressor = TopKCompressor(k_percent=20.0, use_error_feedback=False)
        assert compressor.k_percent == 20.0
        assert compressor.use_error_feedback is False

    def test_invalid_k_percent(self):
        """Test invalid k_percent raises error."""
        with pytest.raises(ValueError):
            TopKCompressor(k_percent=0)
        with pytest.raises(ValueError):
            TopKCompressor(k_percent=101)

    def test_compression_type(self):
        """Test compression type property."""
        compressor = TopKCompressor()
        assert compressor.compression_type == CompressionType.TOP_K

    def test_compress_decompress(self):
        """Test basic compression and decompression."""
        compressor = TopKCompressor(k_percent=50.0)

        # Create gradient with some large and small values
        gradient = np.array([10.0, 1.0, 20.0, 2.0], dtype=np.float32)

        compressed, metadata = compressor.compress(gradient)
        decompressed = compressor.decompress(compressed, metadata)

        # Large values should be preserved, small values zeroed
        assert decompressed[0] == pytest.approx(10.0)  # Top 50%
        assert decompressed[2] == pytest.approx(20.0)  # Top 50%

    def test_sparse_representation(self):
        """Test that compression is actually sparse."""
        compressor = TopKCompressor(k_percent=10.0)
        gradient = np.random.randn(100).astype(np.float32)

        compressed, metadata = compressor.compress(gradient)

        # Compressed should be smaller than original
        # (indices + values for 10% of elements)
        assert compressed.nbytes < gradient.nbytes

    def test_100_percent(self):
        """Test keeping 100% of values."""
        compressor = TopKCompressor(k_percent=100.0)
        gradient = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        compressed, metadata = compressor.compress(gradient)
        decompressed = compressor.decompress(compressed, metadata)

        np.testing.assert_array_equal(decompressed, gradient)


class TestRandomKCompressor:
    """Tests for random-K compressor."""

    def test_create_default(self):
        """Test default creation."""
        compressor = RandomKCompressor()
        assert compressor.k_percent == 10.0

    def test_create_with_seed(self):
        """Test creation with seed for reproducibility."""
        compressor1 = RandomKCompressor(k_percent=50.0, seed=42)
        compressor2 = RandomKCompressor(k_percent=50.0, seed=42)

        gradient = np.random.randn(100).astype(np.float32)

        compressed1, _ = compressor1.compress(gradient)
        compressed2, _ = compressor2.compress(gradient)

        np.testing.assert_array_equal(compressed1, compressed2)

    def test_invalid_k_percent(self):
        """Test invalid k_percent raises error."""
        with pytest.raises(ValueError):
            RandomKCompressor(k_percent=0)
        with pytest.raises(ValueError):
            RandomKCompressor(k_percent=101)

    def test_compression_type(self):
        """Test compression type property."""
        compressor = RandomKCompressor()
        assert compressor.compression_type == CompressionType.RANDOM_K

    def test_compress_decompress(self):
        """Test basic compression and decompression."""
        compressor = RandomKCompressor(k_percent=50.0, seed=42)
        gradient = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        compressed, metadata = compressor.compress(gradient)
        decompressed = compressor.decompress(compressed, metadata)

        # Decompressed should have same shape
        assert decompressed.shape == gradient.shape

        # Non-zero values should be scaled by 2x (50% sampling)
        non_zero = decompressed[decompressed != 0]
        assert len(non_zero) == 2

    def test_scaling(self):
        """Test that values are scaled correctly."""
        compressor = RandomKCompressor(k_percent=10.0, seed=42)

        # Create array with all 1s
        gradient = np.ones(100, dtype=np.float32)

        compressed, metadata = compressor.compress(gradient)
        decompressed = compressor.decompress(compressed, metadata)

        # Sum should be approximately preserved after scaling
        # 10% of 100 = 10 values, each scaled by 10x
        assert np.sum(decompressed) == pytest.approx(100.0, rel=0.1)
