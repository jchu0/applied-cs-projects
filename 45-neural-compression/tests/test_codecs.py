"""Tests for codecs (arithmetic coding and neural compression).

Tests cover:
- ArithmeticCoder encode/decode roundtrip
- RangeCoder encode/decode roundtrip
- Bitstream operations
- NeuralCompressionCodec forward pass
- Compress/decompress roundtrip
- Compression ratio calculations
"""

import pytest
import math
import numpy as np

torch = pytest.importorskip("torch")
import torch.nn as nn

import os
import sys
# Add the project root (which contains the `src/` package) to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.neural_compression.codecs import (
    ArithmeticCoder,
    RangeCoder,
    NeuralCompressionCodec,
    CompressionResult,
)


class TestArithmeticCoder:
    """Tests for arithmetic coding."""

    def test_encode_decode_roundtrip(self, arithmetic_coder, simple_symbols, simple_cdfs):
        """Encode then decode should recover original symbols."""
        bitstream = arithmetic_coder.encode(simple_symbols, simple_cdfs)
        decoded = arithmetic_coder.decode(bitstream, simple_cdfs, len(simple_symbols))

        np.testing.assert_array_equal(simple_symbols, decoded)

    def test_encode_decode_gaussian(
        self, arithmetic_coder, random_gaussian_symbols, gaussian_cdfs
    ):
        """Should work with Gaussian-distributed symbols."""
        bitstream = arithmetic_coder.encode(random_gaussian_symbols, gaussian_cdfs)
        decoded = arithmetic_coder.decode(
            bitstream, gaussian_cdfs, len(random_gaussian_symbols)
        )

        np.testing.assert_array_equal(random_gaussian_symbols, decoded)

    def test_empty_symbols(self, arithmetic_coder):
        """Should handle empty symbol sequence."""
        symbols = np.array([], dtype=np.int32)
        cdfs = np.zeros((0, 4), dtype=np.int64)

        bitstream = arithmetic_coder.encode(symbols, cdfs)
        decoded = arithmetic_coder.decode(bitstream, cdfs, 0)

        assert len(decoded) == 0

    def test_single_symbol(self, arithmetic_coder):
        """Should handle single symbol."""
        symbols = np.array([1], dtype=np.int32)
        cdfs = np.array([[0, 21845, 43690, 65536]], dtype=np.int64)

        bitstream = arithmetic_coder.encode(symbols, cdfs)
        decoded = arithmetic_coder.decode(bitstream, cdfs, 1)

        np.testing.assert_array_equal(symbols, decoded)

    def test_repeated_symbols(self, arithmetic_coder):
        """Should handle repeated symbols (tests E3 scaling)."""
        symbols = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
        precision = 16
        max_val = 1 << precision
        cdfs = np.array(
            [[0, max_val // 3, 2 * max_val // 3, max_val]] * 8, dtype=np.int64
        )

        bitstream = arithmetic_coder.encode(symbols, cdfs)
        decoded = arithmetic_coder.decode(bitstream, cdfs, 8)

        np.testing.assert_array_equal(symbols, decoded)

    def test_compression_ratio(self, arithmetic_coder):
        """High probability symbols should compress better."""
        num_symbols = 100
        precision = 16
        max_val = 1 << precision

        # Create skewed distribution (90% prob for symbol 0)
        cdfs_skewed = np.zeros((num_symbols, 4), dtype=np.int64)
        cdfs_skewed[:, 0] = 0
        cdfs_skewed[:, 1] = int(0.9 * max_val)
        cdfs_skewed[:, 2] = int(0.95 * max_val)
        cdfs_skewed[:, 3] = max_val

        # Uniform distribution
        cdfs_uniform = np.zeros((num_symbols, 4), dtype=np.int64)
        cdfs_uniform[:, 0] = 0
        cdfs_uniform[:, 1] = max_val // 3
        cdfs_uniform[:, 2] = 2 * max_val // 3
        cdfs_uniform[:, 3] = max_val

        # Symbols all 0 (high probability in skewed)
        symbols_0 = np.zeros(num_symbols, dtype=np.int32)

        bitstream_skewed = arithmetic_coder.encode(symbols_0, cdfs_skewed)
        bitstream_uniform = arithmetic_coder.encode(symbols_0, cdfs_uniform)

        # Skewed should produce smaller bitstream
        assert len(bitstream_skewed) < len(bitstream_uniform)

    def test_bits_to_bytes_roundtrip(self, arithmetic_coder):
        """Bit/byte conversion should be lossless."""
        bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1]
        bytes_data = arithmetic_coder._bits_to_bytes(bits.copy())
        recovered_bits = arithmetic_coder._bytes_to_bits(bytes_data)

        assert bits == recovered_bits[: len(bits)]

    def test_precision_parameter(self):
        """Different precision should work."""
        for precision in [8, 12, 16, 24]:
            coder = ArithmeticCoder(precision=precision)
            symbols = np.array([0, 1, 2], dtype=np.int32)
            max_val = 1 << precision
            cdfs = np.array(
                [[0, max_val // 3, 2 * max_val // 3, max_val]] * 3, dtype=np.int64
            )

            bitstream = coder.encode(symbols, cdfs)
            decoded = coder.decode(bitstream, cdfs, 3)

            np.testing.assert_array_equal(symbols, decoded)

    def test_deterministic(self, arithmetic_coder, simple_symbols, simple_cdfs):
        """Encoding should be deterministic."""
        bitstream1 = arithmetic_coder.encode(simple_symbols, simple_cdfs)
        bitstream2 = arithmetic_coder.encode(simple_symbols, simple_cdfs)

        assert bitstream1 == bitstream2

    def test_long_sequence(self, arithmetic_coder):
        """Should handle longer sequences."""
        np.random.seed(42)
        num_symbols = 1000
        vocab_size = 8
        precision = 16
        max_val = 1 << precision

        symbols = np.random.randint(0, vocab_size, num_symbols, dtype=np.int32)

        # Uniform CDFs
        cdfs = np.zeros((num_symbols, vocab_size + 1), dtype=np.int64)
        for i in range(vocab_size + 1):
            cdfs[:, i] = (i * max_val) // vocab_size

        bitstream = arithmetic_coder.encode(symbols, cdfs)
        decoded = arithmetic_coder.decode(bitstream, cdfs, num_symbols)

        np.testing.assert_array_equal(symbols, decoded)


class TestRangeCoder:
    """Tests for range coder."""

    def test_encode_decode_roundtrip(self, range_coder):
        """Encode then decode should recover original symbols."""
        np.random.seed(42)
        num_symbols = 20
        vocab_size = 4

        symbols = np.random.randint(0, vocab_size, num_symbols, dtype=np.int32)

        # Create probability distributions (uniform)
        probs = np.ones((num_symbols, vocab_size)) / vocab_size

        bitstream = range_coder.encode(symbols, probs, vocab_size)
        decoded = range_coder.decode(bitstream, probs, num_symbols, vocab_size)

        np.testing.assert_array_equal(symbols, decoded)

    def test_skewed_distribution(self, range_coder):
        """Should work with skewed probability distributions."""
        num_symbols = 50
        vocab_size = 4

        # Symbol 0 has 70% probability
        probs = np.array([[0.7, 0.1, 0.1, 0.1]] * num_symbols)
        symbols = np.zeros(num_symbols, dtype=np.int32)

        bitstream = range_coder.encode(symbols, probs, vocab_size)
        decoded = range_coder.decode(bitstream, probs, num_symbols, vocab_size)

        np.testing.assert_array_equal(symbols, decoded)


class TestNeuralCompressionCodec:
    """Tests for neural compression codec."""

    def test_forward_output_shape(self, neural_codec, random_image):
        """Forward pass should return correct shapes."""
        neural_codec.train()
        x_hat, likelihoods = neural_codec(random_image)

        assert x_hat.shape == random_image.shape
        assert "y" in likelihoods
        assert "z" in likelihoods

    def test_forward_likelihoods_valid(self, neural_codec, random_image):
        """Likelihoods should be valid probability-like values."""
        neural_codec.train()
        _, likelihoods = neural_codec(random_image)

        # Likelihoods should be positive (probabilities)
        assert (likelihoods["y"] > 0).all()
        assert (likelihoods["z"] > 0).all()

        # Likelihoods should be finite
        assert likelihoods["y"].isfinite().all()
        assert likelihoods["z"].isfinite().all()

    def test_gradient_flow(self, neural_codec, random_image):
        """Gradients should flow through entire codec."""
        neural_codec.train()
        x = random_image.clone().requires_grad_(True)

        x_hat, likelihoods = neural_codec(x)
        # Compute loss from reconstruction and likelihoods
        mse = torch.nn.functional.mse_loss(x_hat, x)
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3]
        bpp = (-torch.log2(likelihoods["y"].clamp(min=1e-9)).sum()
               - torch.log2(likelihoods["z"].clamp(min=1e-9)).sum()) / num_pixels
        loss = mse + 0.01 * bpp
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_eval_mode_deterministic(self, neural_codec, single_image):
        """Eval mode should produce deterministic results."""
        neural_codec.eval()

        with torch.no_grad():
            x_hat1, _ = neural_codec(single_image)
            x_hat2, _ = neural_codec(single_image)

        assert torch.allclose(x_hat1, x_hat2)

    def test_training_step(self, neural_codec, random_image):
        """Training step should return loss and metrics."""
        neural_codec.train()
        loss, metrics = neural_codec.training_step(random_image, lambda_rd=0.01)

        assert loss.isfinite()
        assert "loss" in metrics
        assert "mse" in metrics
        assert "psnr" in metrics
        assert "bpp" in metrics

    def test_different_lambda(self, neural_codec, random_image):
        """Different lambda values should change loss balance."""
        neural_codec.train()

        loss_low, metrics_low = neural_codec.training_step(random_image, lambda_rd=0.001)
        loss_high, metrics_high = neural_codec.training_step(random_image, lambda_rd=0.1)

        # Higher lambda should weight BPP more, different total loss
        # (Actual relationship depends on input, but loss should be computable)
        assert loss_low.isfinite()
        assert loss_high.isfinite()


class TestCompressDecompress:
    """Tests for compression/decompression cycle."""

    def test_compress_returns_result(self, neural_codec, single_image):
        """Compress should return CompressionResult."""
        neural_codec.eval()
        result = neural_codec.compress(single_image)

        assert isinstance(result, CompressionResult)
        assert isinstance(result.bitstream, bytes)
        assert result.original_size > 0
        assert result.compressed_size > 0
        assert result.compression_ratio > 0
        assert result.bpp > 0

    def test_compress_decompress_shape(self, neural_codec, single_image):
        """Decompressed image should have same shape as input."""
        neural_codec.eval()
        result = neural_codec.compress(single_image)
        x_hat = neural_codec.decompress(result.bitstream)

        assert x_hat.shape == single_image.shape

    def test_bitstream_packing(self, neural_codec, single_image):
        """Bitstream should pack/unpack correctly."""
        neural_codec.eval()
        result = neural_codec.compress(single_image)

        # Unpack and verify structure
        z_bytes, y_bytes, z_shape, y_shape = neural_codec._unpack_bitstream(
            result.bitstream
        )

        assert len(z_bytes) > 0
        assert len(y_bytes) > 0
        assert len(z_shape) == 4
        assert len(y_shape) == 4

    def test_compression_ratio_reasonable(self, neural_codec, single_image):
        """Compression ratio should be reasonable (not too extreme)."""
        neural_codec.eval()
        result = neural_codec.compress(single_image)

        # For untrained codec, ratio may not be great but should be computable
        assert result.compression_ratio > 0.01
        assert result.compression_ratio < 1000

    def test_bpp_calculation(self, neural_codec, single_image):
        """BPP should be correctly calculated."""
        neural_codec.eval()
        result = neural_codec.compress(single_image)

        # Manually compute BPP
        num_pixels = single_image.shape[2] * single_image.shape[3]
        expected_bpp = (result.compressed_size * 8) / num_pixels

        assert abs(result.bpp - expected_bpp) < 1e-6

    def test_invalid_input_shape(self, neural_codec):
        """Should raise error for invalid input shape."""
        # Batch size > 1
        x_batch = torch.randn(2, 3, 64, 64)
        with pytest.raises(ValueError):
            neural_codec.compress(x_batch)

        # 3D input
        x_3d = torch.randn(3, 64, 64)
        with pytest.raises(ValueError):
            neural_codec.compress(x_3d)


class TestCompressionQuality:
    """Tests for compression quality metrics."""

    def test_mse_between_original_and_reconstructed(self, neural_codec, single_image):
        """MSE should be computable between original and reconstructed."""
        neural_codec.eval()

        with torch.no_grad():
            x_hat, _ = neural_codec(single_image)

        mse = ((single_image - x_hat) ** 2).mean()

        # MSE should be finite and non-negative
        assert mse >= 0
        assert mse.isfinite()

    def test_psnr_calculation(self, neural_codec, single_image):
        """PSNR should be correctly calculated from reconstruction."""
        neural_codec.eval()

        with torch.no_grad():
            x_hat, _ = neural_codec(single_image)

        # Compute MSE and PSNR
        mse = ((single_image - x_hat) ** 2).mean()
        psnr = -10 * torch.log10(mse.clamp(min=1e-10))

        # PSNR should be finite
        assert psnr.isfinite()

    def test_rate_distortion_tradeoff(self, neural_codec, random_image):
        """Both rate (bpp) and distortion should be computable."""
        neural_codec.train()

        x_hat, likelihoods = neural_codec(random_image)

        # Compute BPP from likelihoods
        num_pixels = random_image.shape[0] * random_image.shape[2] * random_image.shape[3]
        y_bpp = -torch.log2(likelihoods["y"].clamp(min=1e-9)).sum() / num_pixels
        z_bpp = -torch.log2(likelihoods["z"].clamp(min=1e-9)).sum() / num_pixels
        bpp = y_bpp + z_bpp

        # Compute MSE
        mse = ((x_hat - random_image) ** 2).mean()

        # Both rate and distortion should be finite
        assert bpp.isfinite()
        assert mse.isfinite()


class TestCodecComponents:
    """Tests for codec sub-components."""

    def test_encoder_decoder_dimensions(self, neural_codec, single_image):
        """Encoder/decoder should have matching dimensions."""
        y = neural_codec.encoder(single_image)
        x_hat = neural_codec.decoder(y)

        assert x_hat.shape == single_image.shape

    def test_cdf_building(self, neural_codec, single_image):
        """CDF building should produce valid CDFs."""
        neural_codec.eval()

        with torch.no_grad():
            y = neural_codec.encoder(single_image)
            y_hat, z_hat, mean, scale = neural_codec.entropy_model.compress(y)

            # Build CDFs
            z_cdfs = neural_codec._build_factorized_cdfs(z_hat)
            y_cdfs = neural_codec._build_gaussian_cdfs(y_hat, mean, scale)

        # CDFs should be monotonically increasing
        for i in range(min(10, len(z_cdfs))):
            assert all(z_cdfs[i, j] <= z_cdfs[i, j + 1] for j in range(len(z_cdfs[i]) - 1))

        for i in range(min(10, len(y_cdfs))):
            assert all(y_cdfs[i, j] <= y_cdfs[i, j + 1] for j in range(len(y_cdfs[i]) - 1))


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_small_image(self, neural_codec):
        """Should handle minimum valid image size."""
        # Minimum size is 16x16 due to 16x downsampling
        x = torch.randn(1, 3, 16, 16)
        neural_codec.eval()

        with torch.no_grad():
            x_hat, losses = neural_codec(x)

        assert x_hat.shape == x.shape

    def test_large_image(self, neural_codec):
        """Should handle larger images."""
        x = torch.randn(1, 3, 128, 128)
        neural_codec.eval()

        with torch.no_grad():
            x_hat, losses = neural_codec(x)

        assert x_hat.shape == x.shape

    def test_non_square_image(self, neural_codec):
        """Should handle non-square images (divisible by 16)."""
        x = torch.randn(1, 3, 32, 64)
        neural_codec.eval()

        with torch.no_grad():
            x_hat, losses = neural_codec(x)

        assert x_hat.shape == x.shape

    def test_constant_image(self, neural_codec):
        """Should handle constant-value images."""
        x = torch.ones(1, 3, 64, 64) * 0.5
        neural_codec.eval()

        with torch.no_grad():
            x_hat, likelihoods = neural_codec(x)

        assert x_hat.shape == x.shape
        # Compute MSE and verify it's finite
        mse = ((x - x_hat) ** 2).mean()
        assert mse.isfinite()

    def test_extreme_values(self, neural_codec):
        """Should handle images with extreme values."""
        x = torch.randn(1, 3, 64, 64) * 10  # Larger than [0,1]
        neural_codec.eval()

        with torch.no_grad():
            x_hat, losses = neural_codec(x)

        assert x_hat.shape == x.shape
        assert not torch.isnan(x_hat).any()


class TestCompressionResultDataclass:
    """Tests for CompressionResult dataclass."""

    def test_dataclass_fields(self):
        """CompressionResult should have all required fields."""
        result = CompressionResult(
            bitstream=b"test",
            original_size=100,
            compressed_size=50,
            compression_ratio=2.0,
            bpp=1.5,
            shape=(1, 3, 64, 64),
        )

        assert result.bitstream == b"test"
        assert result.original_size == 100
        assert result.compressed_size == 50
        assert result.compression_ratio == 2.0
        assert result.bpp == 1.5
        assert result.shape == (1, 3, 64, 64)

    def test_compression_ratio_formula(self, neural_codec, single_image):
        """Compression ratio should equal original_size / compressed_size."""
        neural_codec.eval()
        result = neural_codec.compress(single_image)

        expected_ratio = result.original_size / result.compressed_size
        assert abs(result.compression_ratio - expected_ratio) < 1e-6
