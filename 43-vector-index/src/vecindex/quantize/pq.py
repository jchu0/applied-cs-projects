"""Product Quantization for vector compression."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..core.vectors import kmeans, l2_distance

logger = logging.getLogger(__name__)


@dataclass
class PQConfig:
    """Configuration for Product Quantization."""
    M: int           # Number of subquantizers
    nbits: int = 8   # Bits per subquantizer


class ProductQuantizer:
    """
    Product Quantization for vector compression.

    Splits vectors into M subvectors and quantizes each independently.
    Reduces storage from d*32 bits to M*nbits bits per vector.
    """

    def __init__(self, dim: int, M: int, nbits: int = 8):
        """
        Args:
            dim: Vector dimension
            M: Number of subquantizers
            nbits: Bits per code (2^nbits centroids per subquantizer)
        """
        self.dim = dim
        self.M = M
        self.nbits = nbits
        self.ksub = 2 ** nbits
        self.dsub = dim // M

        assert dim % M == 0, f"Dimension {dim} must be divisible by M={M}"

        self.centroids = None  # (M, ksub, dsub)
        self.is_trained = False

    def train(self, vectors: np.ndarray, max_iter: int = 100):
        """
        Train PQ centroids.

        Args:
            vectors: Training vectors (n, d)
            max_iter: K-means iterations
        """
        vectors = np.asarray(vectors, dtype=np.float32)

        self.centroids = np.zeros((self.M, self.ksub, self.dsub), dtype=np.float32)

        for m in range(self.M):
            # Extract subvectors
            sub_vectors = vectors[:, m * self.dsub:(m + 1) * self.dsub]

            # Train k-means for this subquantizer
            self.centroids[m], _ = kmeans(sub_vectors, self.ksub, max_iter=max_iter)

        self.is_trained = True
        logger.info(f"Trained PQ with M={self.M}, ksub={self.ksub}")

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors to PQ codes.

        Args:
            vectors: Input vectors (n, d)

        Returns:
            Codes (n, M)
        """
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before encoding")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        n = vectors.shape[0]
        codes = np.zeros((n, self.M), dtype=np.uint8)

        for m in range(self.M):
            sub_vectors = vectors[:, m * self.dsub:(m + 1) * self.dsub]
            distances = l2_distance(sub_vectors, self.centroids[m])
            if distances.ndim == 1:
                distances = distances.reshape(1, -1)
            codes[:, m] = np.argmin(distances, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes to approximate vectors.

        Args:
            codes: PQ codes (n, M)

        Returns:
            Reconstructed vectors (n, d)
        """
        codes = np.asarray(codes)
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)

        n = codes.shape[0]
        vectors = np.zeros((n, self.dim), dtype=np.float32)

        for m in range(self.M):
            vectors[:, m * self.dsub:(m + 1) * self.dsub] = self.centroids[m, codes[:, m]]

        return vectors

    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """
        Precompute distances from query to all centroids.

        Args:
            query: Query vector (d,)

        Returns:
            Distance table (M, ksub)
        """
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 2:
            query = query[0]

        table = np.zeros((self.M, self.ksub), dtype=np.float32)

        for m in range(self.M):
            sub_query = query[m * self.dsub:(m + 1) * self.dsub]
            table[m] = np.sum((self.centroids[m] - sub_query) ** 2, axis=1)

        return table

    def asymmetric_distance(
        self,
        query: np.ndarray,
        codes: np.ndarray
    ) -> np.ndarray:
        """
        Compute asymmetric distance (ADC).

        Distance from exact query to encoded vectors.

        Args:
            query: Query vector
            codes: PQ codes (n, M)

        Returns:
            Distances (n,)
        """
        table = self.compute_distance_table(query)

        codes = np.asarray(codes)
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)

        distances = np.sum([table[m, codes[:, m]] for m in range(self.M)], axis=0)
        return np.sqrt(distances)

    def symmetric_distance(
        self,
        codes1: np.ndarray,
        codes2: np.ndarray
    ) -> float:
        """
        Compute symmetric distance (SDC).

        Distance between two encoded vectors.

        Args:
            codes1: First PQ code (M,)
            codes2: Second PQ code (M,)

        Returns:
            Distance
        """
        distance = 0.0
        for m in range(self.M):
            c1 = self.centroids[m, codes1[m]]
            c2 = self.centroids[m, codes2[m]]
            distance += np.sum((c1 - c2) ** 2)

        return np.sqrt(distance)

    @property
    def code_size(self) -> int:
        """Size of code in bytes."""
        return self.M * self.nbits // 8

    @property
    def compression_ratio(self) -> float:
        """Compression ratio vs float32."""
        original = self.dim * 4  # bytes
        compressed = self.M * self.nbits / 8
        return original / compressed


class OPQ(ProductQuantizer):
    """
    Optimized Product Quantization.

    Learns rotation matrix to minimize quantization error.
    """

    def __init__(self, dim: int, M: int, nbits: int = 8):
        super().__init__(dim, M, nbits)
        self.rotation = None  # Rotation matrix (d, d)

    def train(self, vectors: np.ndarray, n_iter: int = 10):
        """
        Train OPQ with alternating optimization.

        Args:
            vectors: Training vectors
            n_iter: Number of optimization iterations
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        n, d = vectors.shape

        # Initialize rotation with identity
        self.rotation = np.eye(d, dtype=np.float32)
        rotated = vectors.copy()

        for i in range(n_iter):
            # Train PQ on rotated vectors
            super().train(rotated)

            # Encode and decode
            codes = self.encode(rotated)
            reconstructed = self.decode(codes)

            # Update rotation using Procrustes
            U, _, Vt = np.linalg.svd(vectors.T @ reconstructed)
            self.rotation = (U @ Vt).T

            # Rotate vectors
            rotated = vectors @ self.rotation.T

            # Compute error
            error = np.mean(np.sum((rotated - reconstructed) ** 2, axis=1))
            logger.info(f"OPQ iteration {i+1}, error: {error:.4f}")

        # Final PQ training
        super().train(rotated)
        self.is_trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate then encode."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        rotated = vectors @ self.rotation.T
        return super().encode(rotated)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode then inverse rotate."""
        reconstructed = super().decode(codes)
        return reconstructed @ self.rotation


class ScalarQuantizer:
    """
    Scalar Quantization for vector compression.

    Quantizes each dimension independently.
    """

    def __init__(self, dim: int, nbits: int = 8):
        """
        Args:
            dim: Vector dimension
            nbits: Bits per dimension
        """
        self.dim = dim
        self.nbits = nbits
        self.levels = 2 ** nbits

        self.mins = None
        self.maxs = None
        self.is_trained = False

    def train(self, vectors: np.ndarray):
        """Learn quantization ranges."""
        vectors = np.asarray(vectors, dtype=np.float32)

        self.mins = vectors.min(axis=0)
        self.maxs = vectors.max(axis=0)

        # Avoid division by zero
        self.ranges = self.maxs - self.mins
        self.ranges = np.where(self.ranges == 0, 1, self.ranges)

        self.is_trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Quantize vectors."""
        if not self.is_trained:
            raise RuntimeError("Must train before encoding")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Normalize to [0, 1]
        normalized = (vectors - self.mins) / self.ranges

        # Quantize to [0, levels-1]
        codes = np.clip(normalized * (self.levels - 1), 0, self.levels - 1)

        if self.nbits <= 8:
            return codes.astype(np.uint8)
        else:
            return codes.astype(np.uint16)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Dequantize codes."""
        codes = np.asarray(codes, dtype=np.float32)
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)

        # Reverse quantization
        normalized = codes / (self.levels - 1)
        vectors = normalized * self.ranges + self.mins

        return vectors


class BinaryQuantizer:
    """
    Binary Quantization using thresholding.

    Compresses vectors to binary codes for Hamming distance search.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.thresholds = None
        self.is_trained = False

    def train(self, vectors: np.ndarray):
        """Learn thresholds (median per dimension)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        self.thresholds = np.median(vectors, axis=0)
        self.is_trained = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode to binary codes.

        Args:
            vectors: Input vectors (n, d)

        Returns:
            Binary codes packed into uint8
        """
        if not self.is_trained:
            raise RuntimeError("Must train before encoding")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Threshold
        bits = (vectors > self.thresholds).astype(np.uint8)

        # Pack into bytes
        n_bytes = (self.dim + 7) // 8
        packed = np.zeros((vectors.shape[0], n_bytes), dtype=np.uint8)

        for i in range(self.dim):
            byte_idx = i // 8
            bit_idx = i % 8
            packed[:, byte_idx] |= bits[:, i] << bit_idx

        return packed

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode binary codes back to approximate vectors.

        Note: Binary quantization is lossy - decoded vectors will be approximations
        based on the learned thresholds. Values above threshold become threshold + delta,
        values below become threshold - delta, where delta is estimated from training.

        Args:
            codes: Binary codes packed into uint8 (n, n_bytes)

        Returns:
            Approximate vectors (n, d)
        """
        codes = np.asarray(codes, dtype=np.uint8)
        if codes.ndim == 1:
            codes = codes.reshape(1, -1)

        n = codes.shape[0]
        vectors = np.zeros((n, self.dim), dtype=np.float32)

        # Unpack bits
        for i in range(self.dim):
            byte_idx = i // 8
            bit_idx = i % 8
            bit_values = (codes[:, byte_idx] >> bit_idx) & 1

            # Reconstruct: use threshold as midpoint
            # Set high bits to threshold + 0.5, low bits to threshold - 0.5
            # This provides a reasonable approximation for binary codes
            vectors[:, i] = np.where(
                bit_values == 1,
                self.thresholds[i] + 0.5,
                self.thresholds[i] - 0.5
            )

        return vectors

    @staticmethod
    def hamming_distance(codes1: np.ndarray, codes2: np.ndarray) -> int:
        """Compute Hamming distance between binary codes."""
        xor = np.bitwise_xor(codes1, codes2)
        return int(np.sum([bin(b).count('1') for b in xor.flatten()]))


def compute_recall(
    ground_truth: np.ndarray,
    results: np.ndarray,
    k: int
) -> float:
    """
    Compute recall@k.

    Args:
        ground_truth: True nearest neighbors (nq, k)
        results: Retrieved results (nq, k)
        k: Number of results

    Returns:
        Recall value [0, 1]
    """
    if ground_truth.ndim == 1:
        ground_truth = ground_truth.reshape(1, -1)
    if results.ndim == 1:
        results = results.reshape(1, -1)

    nq = ground_truth.shape[0]
    correct = 0

    for i in range(nq):
        gt_set = set(ground_truth[i, :k])
        result_set = set(results[i, :k])
        correct += len(gt_set & result_set)

    return correct / (nq * k)
