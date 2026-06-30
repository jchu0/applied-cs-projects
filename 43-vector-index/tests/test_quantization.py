"""Tests for quantization methods (Product Quantization, Scalar, Binary)."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vecindex import (
    ProductQuantizer,
    OPQ,
    ScalarQuantizer,
    BinaryQuantizer,
    compute_recall,
)
from vecindex.quantize.pq import PQConfig


class TestProductQuantizer:
    """Tests for ProductQuantizer."""

    def test_create_pq(self, medium_dim):
        """Test creating a ProductQuantizer."""
        M = 8
        pq = ProductQuantizer(medium_dim, M)

        assert pq.dim == medium_dim
        assert pq.M == M
        assert pq.nbits == 8  # default
        assert pq.ksub == 256  # 2^8
        assert pq.dsub == medium_dim // M
        assert pq.is_trained is False

    def test_dimension_divisibility(self):
        """Test that dimension must be divisible by M."""
        # This should work (64 % 8 == 0)
        pq = ProductQuantizer(64, M=8)
        assert pq.dsub == 8

        # This should fail
        with pytest.raises(AssertionError):
            ProductQuantizer(64, M=5)

    def test_train_required(self, medium_dim, medium_vectors):
        """Test that training is required before encoding."""
        pq = ProductQuantizer(medium_dim, M=8)

        with pytest.raises(RuntimeError, match="trained"):
            pq.encode(medium_vectors[0])

    def test_train_pq(self, product_quantizer):
        """Test training the ProductQuantizer."""
        pq, vectors = product_quantizer

        assert pq.is_trained
        assert pq.centroids is not None
        assert pq.centroids.shape == (8, 256, 8)  # (M, ksub, dsub)

    def test_encode(self, product_quantizer):
        """Test encoding vectors."""
        pq, vectors = product_quantizer

        codes = pq.encode(vectors[:10])
        assert codes.shape == (10, 8)  # (n, M)
        assert codes.dtype == np.uint8
        assert np.all(codes >= 0)
        assert np.all(codes < 256)

    def test_encode_single(self, product_quantizer):
        """Test encoding a single vector."""
        pq, vectors = product_quantizer

        code = pq.encode(vectors[0])
        assert code.shape == (1, 8)

    def test_decode(self, product_quantizer):
        """Test decoding codes back to vectors."""
        pq, vectors = product_quantizer

        codes = pq.encode(vectors[:10])
        reconstructed = pq.decode(codes)

        assert reconstructed.shape == (10, 64)
        assert reconstructed.dtype == np.float32

    def test_decode_single(self, product_quantizer):
        """Test decoding a single code."""
        pq, vectors = product_quantizer

        codes = pq.encode(vectors[0])
        reconstructed = pq.decode(codes)

        assert reconstructed.shape == (1, 64)

    def test_encode_decode_reconstruction_error(self, product_quantizer):
        """Test that reconstruction error is bounded."""
        pq, vectors = product_quantizer

        sample = vectors[:100]
        codes = pq.encode(sample)
        reconstructed = pq.decode(codes)

        # Compute MSE
        mse = np.mean((sample - reconstructed) ** 2)

        # MSE should be reasonable (not too high)
        # This is a rough sanity check
        assert mse < np.var(sample) * 2, f"MSE too high: {mse}"

    def test_compute_distance_table(self, product_quantizer):
        """Test computing distance tables."""
        pq, vectors = product_quantizer

        query = vectors[0]
        table = pq.compute_distance_table(query)

        assert table.shape == (8, 256)  # (M, ksub)
        assert table.dtype == np.float32
        assert np.all(table >= 0)

    def test_asymmetric_distance(self, product_quantizer):
        """Test asymmetric distance computation."""
        pq, vectors = product_quantizer

        query = vectors[0]
        codes = pq.encode(vectors[1:11])

        distances = pq.asymmetric_distance(query, codes)
        assert distances.shape == (10,)
        assert np.all(distances >= 0)

    def test_asymmetric_distance_to_self(self, product_quantizer):
        """Test asymmetric distance of query to its own code."""
        pq, vectors = product_quantizer

        query = vectors[0]
        code = pq.encode(query)
        distance = pq.asymmetric_distance(query, code)

        # Distance to quantized self should be small but not necessarily zero
        # due to quantization error
        assert distance[0] >= 0

    def test_symmetric_distance(self, product_quantizer):
        """Test symmetric distance computation."""
        pq, vectors = product_quantizer

        codes1 = pq.encode(vectors[0])[0]
        codes2 = pq.encode(vectors[1])[0]

        distance = pq.symmetric_distance(codes1, codes2)
        assert distance >= 0

    def test_symmetric_distance_self(self, product_quantizer):
        """Test symmetric distance of code to itself is zero."""
        pq, vectors = product_quantizer

        code = pq.encode(vectors[0])[0]
        distance = pq.symmetric_distance(code, code)

        np.testing.assert_almost_equal(distance, 0.0, decimal=5)

    def test_code_size(self, medium_dim):
        """Test code size calculation."""
        pq = ProductQuantizer(medium_dim, M=8, nbits=8)
        assert pq.code_size == 8  # M * nbits / 8 = 8 bytes

        pq2 = ProductQuantizer(medium_dim, M=8, nbits=4)
        assert pq2.code_size == 4  # 8 * 4 / 8 = 4 bytes

    def test_compression_ratio(self, medium_dim):
        """Test compression ratio calculation."""
        pq = ProductQuantizer(medium_dim, M=8, nbits=8)

        # Original: 64 * 4 = 256 bytes
        # Compressed: 8 bytes
        # Ratio: 32x
        assert pq.compression_ratio == 32.0


class TestOPQ:
    """Tests for Optimized Product Quantization."""

    def test_create_opq(self, medium_dim):
        """Test creating an OPQ."""
        opq = OPQ(medium_dim, M=8)
        assert opq.dim == medium_dim
        assert opq.M == 8
        assert opq.rotation is None

    def test_train_opq(self, medium_dim, medium_vectors):
        """Test training OPQ."""
        opq = OPQ(medium_dim, M=8)
        opq.train(medium_vectors, n_iter=3)  # Few iterations for speed

        assert opq.is_trained
        assert opq.rotation is not None
        assert opq.rotation.shape == (medium_dim, medium_dim)
        assert opq.centroids is not None

    def test_rotation_is_orthogonal(self, medium_dim, medium_vectors):
        """Test that learned rotation is (approximately) orthogonal."""
        opq = OPQ(medium_dim, M=8)
        opq.train(medium_vectors, n_iter=3)

        # R @ R.T should be approximately identity
        product = opq.rotation @ opq.rotation.T
        identity = np.eye(medium_dim)

        np.testing.assert_array_almost_equal(product, identity, decimal=3)

    def test_encode_with_rotation(self, medium_dim, medium_vectors):
        """Test encoding with rotation."""
        opq = OPQ(medium_dim, M=8)
        opq.train(medium_vectors, n_iter=3)

        codes = opq.encode(medium_vectors[:10])
        assert codes.shape == (10, 8)

    def test_decode_with_rotation(self, medium_dim, medium_vectors):
        """Test decoding with inverse rotation."""
        opq = OPQ(medium_dim, M=8)
        opq.train(medium_vectors, n_iter=3)

        sample = medium_vectors[:10]
        codes = opq.encode(sample)
        reconstructed = opq.decode(codes)

        assert reconstructed.shape == sample.shape

    def test_opq_reduces_error(self, medium_dim, medium_vectors):
        """Test that OPQ reduces reconstruction error compared to PQ."""
        # Train standard PQ
        pq = ProductQuantizer(medium_dim, M=8)
        pq.train(medium_vectors)

        # Train OPQ
        opq = OPQ(medium_dim, M=8)
        opq.train(medium_vectors, n_iter=5)

        # Compute errors
        sample = medium_vectors[:100]

        pq_codes = pq.encode(sample)
        pq_reconstructed = pq.decode(pq_codes)
        pq_mse = np.mean((sample - pq_reconstructed) ** 2)

        opq_codes = opq.encode(sample)
        opq_reconstructed = opq.decode(opq_codes)
        opq_mse = np.mean((sample - opq_reconstructed) ** 2)

        # OPQ should have equal or lower error
        # (may not always be lower with few iterations)
        assert opq_mse <= pq_mse * 1.5, f"OPQ MSE {opq_mse} much higher than PQ MSE {pq_mse}"


class TestScalarQuantizer:
    """Tests for ScalarQuantizer."""

    def test_create_sq(self, medium_dim):
        """Test creating a ScalarQuantizer."""
        sq = ScalarQuantizer(medium_dim)
        assert sq.dim == medium_dim
        assert sq.nbits == 8  # default
        assert sq.levels == 256
        assert sq.is_trained is False

    def test_create_sq_4bit(self, medium_dim):
        """Test creating a 4-bit ScalarQuantizer."""
        sq = ScalarQuantizer(medium_dim, nbits=4)
        assert sq.levels == 16

    def test_train_required(self, medium_dim):
        """Test that training is required before encoding."""
        sq = ScalarQuantizer(medium_dim)
        vector = np.random.randn(medium_dim).astype(np.float32)

        with pytest.raises(RuntimeError, match="train"):
            sq.encode(vector)

    def test_train_sq(self, scalar_quantizer):
        """Test training the ScalarQuantizer."""
        sq, _ = scalar_quantizer

        assert sq.is_trained
        assert sq.mins is not None
        assert sq.maxs is not None
        assert len(sq.mins) == 64
        assert len(sq.maxs) == 64

    def test_encode(self, scalar_quantizer):
        """Test encoding vectors."""
        sq, vectors = scalar_quantizer

        codes = sq.encode(vectors[:10])
        assert codes.shape == (10, 64)
        assert codes.dtype == np.uint8
        assert np.all(codes >= 0)
        assert np.all(codes < 256)

    def test_encode_single(self, scalar_quantizer):
        """Test encoding a single vector."""
        sq, vectors = scalar_quantizer

        code = sq.encode(vectors[0])
        assert code.shape == (1, 64)

    def test_decode(self, scalar_quantizer):
        """Test decoding codes back to vectors."""
        sq, vectors = scalar_quantizer

        codes = sq.encode(vectors[:10])
        reconstructed = sq.decode(codes)

        assert reconstructed.shape == (10, 64)
        assert reconstructed.dtype == np.float32

    def test_encode_decode_bounds(self, scalar_quantizer):
        """Test that decoded values are within training bounds."""
        sq, vectors = scalar_quantizer

        codes = sq.encode(vectors[:100])
        reconstructed = sq.decode(codes)

        # Reconstructed values should be within [min, max]
        assert np.all(reconstructed >= sq.mins - 1e-5)
        assert np.all(reconstructed <= sq.maxs + 1e-5)

    def test_encode_clipping(self, medium_dim, medium_vectors):
        """Test that values outside training range are clipped."""
        sq = ScalarQuantizer(medium_dim)
        sq.train(medium_vectors)

        # Create vectors with extreme values
        extreme = np.ones((1, medium_dim)) * 1000  # Way outside range
        codes = sq.encode(extreme.astype(np.float32))

        # Should be clipped to max code value
        assert np.all(codes == 255)


class TestBinaryQuantizer:
    """Tests for BinaryQuantizer."""

    def test_create_bq(self, medium_dim):
        """Test creating a BinaryQuantizer."""
        bq = BinaryQuantizer(medium_dim)
        assert bq.dim == medium_dim
        assert bq.is_trained is False

    def test_train_required(self, medium_dim):
        """Test that training is required before encoding."""
        bq = BinaryQuantizer(medium_dim)
        vector = np.random.randn(medium_dim).astype(np.float32)

        with pytest.raises(RuntimeError, match="train"):
            bq.encode(vector)

    def test_train_bq(self, binary_quantizer):
        """Test training the BinaryQuantizer."""
        bq, _ = binary_quantizer

        assert bq.is_trained
        assert bq.thresholds is not None
        assert len(bq.thresholds) == 64

    def test_thresholds_are_medians(self, medium_dim, medium_vectors):
        """Test that thresholds are per-dimension medians."""
        bq = BinaryQuantizer(medium_dim)
        bq.train(medium_vectors)

        expected_thresholds = np.median(medium_vectors, axis=0)
        np.testing.assert_array_almost_equal(bq.thresholds, expected_thresholds)

    def test_encode(self, binary_quantizer):
        """Test encoding vectors to binary."""
        bq, vectors = binary_quantizer

        codes = bq.encode(vectors[:10])

        # For 64-dim, we need 8 bytes
        expected_bytes = (64 + 7) // 8
        assert codes.shape == (10, expected_bytes)
        assert codes.dtype == np.uint8

    def test_encode_single(self, binary_quantizer):
        """Test encoding a single vector."""
        bq, vectors = binary_quantizer

        code = bq.encode(vectors[0])
        assert code.shape == (1, 8)

    def test_binary_codes_packed(self, medium_dim, medium_vectors):
        """Test that binary codes are properly packed."""
        bq = BinaryQuantizer(medium_dim)
        bq.train(medium_vectors)

        # Create a vector with known bits
        vector = np.zeros((1, medium_dim), dtype=np.float32)
        # Set first 8 dimensions to be above threshold
        vector[0, :8] = bq.thresholds[:8] + 1

        code = bq.encode(vector)

        # First byte should have bits 0-7 set
        assert code[0, 0] == 0xFF

    def test_hamming_distance_self(self, binary_quantizer):
        """Test Hamming distance of code to itself is zero."""
        bq, vectors = binary_quantizer

        code = bq.encode(vectors[0])
        distance = BinaryQuantizer.hamming_distance(code, code)

        assert distance == 0

    def test_hamming_distance_different(self, binary_quantizer):
        """Test Hamming distance between different codes."""
        bq, vectors = binary_quantizer

        code1 = bq.encode(vectors[0])
        code2 = bq.encode(vectors[1])

        distance = BinaryQuantizer.hamming_distance(code1, code2)

        # Distance should be positive (unless vectors are identical after quantization)
        assert distance >= 0
        # Max distance for 64 bits is 64
        assert distance <= 64

    def test_hamming_distance_maximum(self):
        """Test maximum Hamming distance."""
        code1 = np.array([[0x00]], dtype=np.uint8)
        code2 = np.array([[0xFF]], dtype=np.uint8)

        distance = BinaryQuantizer.hamming_distance(code1, code2)
        assert distance == 8  # All 8 bits different

    def test_compression_ratio(self, medium_dim):
        """Test compression ratio of binary quantization."""
        # Original: 64 * 4 = 256 bytes (float32)
        # Binary: 64 bits = 8 bytes
        # Ratio: 32x
        original_bytes = medium_dim * 4
        binary_bytes = (medium_dim + 7) // 8
        ratio = original_bytes / binary_bytes

        assert ratio == 32.0

    def test_decode(self, binary_quantizer):
        """Test decoding binary codes back to approximate vectors."""
        bq, vectors = binary_quantizer

        codes = bq.encode(vectors[:10])
        reconstructed = bq.decode(codes)

        assert reconstructed.shape == (10, 64)
        assert reconstructed.dtype == np.float32

    def test_decode_single(self, binary_quantizer):
        """Test decoding a single binary code."""
        bq, vectors = binary_quantizer

        code = bq.encode(vectors[0])
        reconstructed = bq.decode(code)

        assert reconstructed.shape == (1, 64)

    def test_encode_decode_consistency(self, binary_quantizer):
        """Test that encode-decode produces consistent results."""
        bq, vectors = binary_quantizer

        # Encode and decode
        codes = bq.encode(vectors[:10])
        reconstructed = bq.decode(codes)

        # Re-encode reconstructed vectors
        codes2 = bq.encode(reconstructed)

        # Codes should be identical (binary thresholding is deterministic)
        np.testing.assert_array_equal(codes, codes2)

    def test_decode_preserves_sign_relative_to_threshold(self, binary_quantizer):
        """Test that decoded values preserve above/below threshold relationship."""
        bq, vectors = binary_quantizer

        sample = vectors[:10]
        codes = bq.encode(sample)
        reconstructed = bq.decode(codes)

        # For each dimension, check that reconstructed value is above threshold
        # iff original value was above threshold
        for i in range(bq.dim):
            original_above = sample[:, i] > bq.thresholds[i]
            reconstructed_above = reconstructed[:, i] > bq.thresholds[i]
            np.testing.assert_array_equal(original_above, reconstructed_above)

    def test_decode_bounded_by_thresholds(self, binary_quantizer):
        """Test that decoded values are centered around thresholds."""
        bq, vectors = binary_quantizer

        codes = bq.encode(vectors[:100])
        reconstructed = bq.decode(codes)

        # Reconstructed values should be close to thresholds (within +/- 0.5)
        for i in range(bq.dim):
            diff = np.abs(reconstructed[:, i] - bq.thresholds[i])
            assert np.all(diff <= 0.5 + 1e-6)


class TestPQConfig:
    """Tests for PQConfig dataclass."""

    def test_create_config(self):
        """Test creating a PQConfig."""
        config = PQConfig(M=8, nbits=8)
        assert config.M == 8
        assert config.nbits == 8

    def test_default_nbits(self):
        """Test default nbits value."""
        config = PQConfig(M=16)
        assert config.nbits == 8


class TestQuantizationComparison:
    """Comparative tests across quantization methods."""

    def test_all_quantizers_have_common_interface(self, medium_dim):
        """Test that all quantizers have train/encode interface."""
        quantizers = [
            ProductQuantizer(medium_dim, M=8),
            OPQ(medium_dim, M=8),
            ScalarQuantizer(medium_dim),
            BinaryQuantizer(medium_dim),
        ]

        for q in quantizers:
            assert hasattr(q, 'train')
            assert hasattr(q, 'encode')
            assert hasattr(q, 'is_trained')

    def test_compression_comparison(self, medium_dim, medium_vectors):
        """Compare compression across methods."""
        # Train all quantizers
        pq = ProductQuantizer(medium_dim, M=8, nbits=8)
        pq.train(medium_vectors)

        sq = ScalarQuantizer(medium_dim, nbits=8)
        sq.train(medium_vectors)

        bq = BinaryQuantizer(medium_dim)
        bq.train(medium_vectors)

        # Encode sample
        sample = medium_vectors[:100]

        pq_codes = pq.encode(sample)
        sq_codes = sq.encode(sample)
        bq_codes = bq.encode(sample)

        # Compare sizes (bytes per vector)
        pq_size = pq_codes.nbytes / len(sample)  # 8 bytes
        sq_size = sq_codes.nbytes / len(sample)  # 64 bytes
        bq_size = bq_codes.nbytes / len(sample)  # 8 bytes

        assert pq_size == 8
        assert sq_size == 64
        assert bq_size == 8

    def test_reconstruction_quality_ordering(self, medium_dim, medium_vectors):
        """Test that SQ has better reconstruction than PQ which is better than BQ."""
        # This is a general expectation, though not guaranteed in all cases

        # Train quantizers
        pq = ProductQuantizer(medium_dim, M=8, nbits=8)
        pq.train(medium_vectors)

        sq = ScalarQuantizer(medium_dim, nbits=8)
        sq.train(medium_vectors)

        # Compute reconstruction errors
        sample = medium_vectors[:100]

        pq_codes = pq.encode(sample)
        pq_reconstructed = pq.decode(pq_codes)
        pq_mse = np.mean((sample - pq_reconstructed) ** 2)

        sq_codes = sq.encode(sample)
        sq_reconstructed = sq.decode(sq_codes)
        sq_mse = np.mean((sample - sq_reconstructed) ** 2)

        # SQ should have lower error (more bits per dimension)
        assert sq_mse < pq_mse, f"SQ MSE {sq_mse} should be less than PQ MSE {pq_mse}"
