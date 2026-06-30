"""Pytest fixtures for VecIndex test suite."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vecindex import (
    MetricType,
    FlatIndex,
    IVFIndex,
    HNSWIndex,
    IVFPQIndex,
    ProductQuantizer,
    ScalarQuantizer,
    BinaryQuantizer,
)


# Random seed for reproducibility
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def small_dim():
    """Small dimension for quick tests."""
    return 16


@pytest.fixture
def medium_dim():
    """Medium dimension for typical tests."""
    return 64


@pytest.fixture
def large_dim():
    """Large dimension for stress tests."""
    return 128


@pytest.fixture
def small_vectors(small_dim):
    """Small vector dataset (100 vectors)."""
    return np.random.randn(100, small_dim).astype(np.float32)


@pytest.fixture
def medium_vectors(medium_dim):
    """Medium vector dataset (1000 vectors)."""
    return np.random.randn(1000, medium_dim).astype(np.float32)


@pytest.fixture
def large_vectors(large_dim):
    """Large vector dataset (5000 vectors)."""
    return np.random.randn(5000, large_dim).astype(np.float32)


@pytest.fixture
def query_vectors(medium_dim):
    """Query vectors for search tests."""
    return np.random.randn(10, medium_dim).astype(np.float32)


@pytest.fixture
def single_query(medium_dim):
    """Single query vector."""
    return np.random.randn(medium_dim).astype(np.float32)


@pytest.fixture
def clustered_vectors(medium_dim):
    """Clustered vectors for IVF testing."""
    n_clusters = 10
    vectors_per_cluster = 100
    vectors = []

    for i in range(n_clusters):
        # Create cluster center
        center = np.random.randn(medium_dim).astype(np.float32) * 10
        # Add noisy points around center
        cluster_vectors = center + np.random.randn(vectors_per_cluster, medium_dim).astype(np.float32) * 0.5
        vectors.append(cluster_vectors)

    return np.vstack(vectors)


@pytest.fixture
def normalized_vectors(medium_dim):
    """Unit-normalized vectors for cosine similarity."""
    vectors = np.random.randn(500, medium_dim).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


@pytest.fixture
def flat_index_l2(medium_dim, medium_vectors):
    """Pre-built FlatIndex with L2 distance."""
    index = FlatIndex(medium_dim, MetricType.L2)
    index.add(medium_vectors)
    return index, medium_vectors


@pytest.fixture
def flat_index_ip(medium_dim, normalized_vectors):
    """Pre-built FlatIndex with inner product."""
    index = FlatIndex(medium_dim, MetricType.IP)
    index.add(normalized_vectors)
    return index, normalized_vectors


@pytest.fixture
def flat_index_cosine(medium_dim, medium_vectors):
    """Pre-built FlatIndex with cosine similarity."""
    index = FlatIndex(medium_dim, MetricType.COSINE)
    index.add(medium_vectors)
    return index, medium_vectors


@pytest.fixture
def ivf_index(medium_dim, clustered_vectors):
    """Pre-built IVFIndex."""
    nlist = 10
    nprobe = 3
    index = IVFIndex(medium_dim, nlist, MetricType.L2, nprobe)
    index.train(clustered_vectors)
    index.add(clustered_vectors)
    return index, clustered_vectors


@pytest.fixture
def hnsw_index(small_dim, small_vectors):
    """Pre-built HNSWIndex."""
    index = HNSWIndex(small_dim, M=8, ef_construction=100, ef_search=30, metric=MetricType.L2)
    index.add(small_vectors)
    return index, small_vectors


@pytest.fixture
def ivfpq_index(medium_dim, medium_vectors):
    """Pre-built IVFPQIndex."""
    nlist = 10
    M = 8  # 8 subquantizers for 64-dim vectors (8 dims each)
    nbits = 8
    index = IVFPQIndex(medium_dim, nlist, M, nbits, MetricType.L2, nprobe=3)
    index.train(medium_vectors)
    index.add(medium_vectors)
    return index, medium_vectors


@pytest.fixture
def product_quantizer(medium_dim, medium_vectors):
    """Pre-trained ProductQuantizer."""
    M = 8  # 8 subquantizers
    pq = ProductQuantizer(medium_dim, M, nbits=8)
    pq.train(medium_vectors)
    return pq, medium_vectors


@pytest.fixture
def scalar_quantizer(medium_dim, medium_vectors):
    """Pre-trained ScalarQuantizer."""
    sq = ScalarQuantizer(medium_dim, nbits=8)
    sq.train(medium_vectors)
    return sq, medium_vectors


@pytest.fixture
def binary_quantizer(medium_dim, medium_vectors):
    """Pre-trained BinaryQuantizer."""
    bq = BinaryQuantizer(medium_dim)
    bq.train(medium_vectors)
    return bq, medium_vectors


def compute_ground_truth_l2(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Compute ground truth k-NN using brute force L2 distance."""
    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Compute all pairwise distances
    dists = np.linalg.norm(vectors - query, axis=1)

    # Get top-k indices
    return np.argsort(dists)[:k]


def compute_ground_truth_ip(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Compute ground truth k-NN using brute force inner product."""
    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Compute inner products (higher is better)
    scores = np.dot(vectors, query.T).flatten()

    # Get top-k indices (highest scores)
    return np.argsort(scores)[::-1][:k]


def compute_ground_truth_cosine(vectors: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """Compute ground truth k-NN using brute force cosine similarity."""
    if query.ndim == 1:
        query = query.reshape(1, -1)

    # Normalize
    v_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)

    # Compute cosine similarities (higher is better)
    scores = np.dot(v_norm, q_norm.T).flatten()

    # Get top-k indices (highest scores)
    return np.argsort(scores)[::-1][:k]
