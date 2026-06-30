"""Tests for SIMD-optimized vector operations."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vecindex.simd import (
    simd_l2_distance,
    simd_inner_product,
    simd_cosine_similarity,
    simd_topk,
    simd_l2_batch,
    simd_ip_batch,
    batch_search,
    SIMDVectorOps,
    SIMD_AVAILABLE,
)
from vecindex.core import l2_distance, inner_product, cosine_similarity


class TestSIMDDistances:
    """Test SIMD distance computations."""

    def test_l2_single_query(self):
        """Test L2 distance for single query."""
        np.random.seed(42)
        query = np.random.randn(128).astype(np.float32)
        database = np.random.randn(1000, 128).astype(np.float32)

        result = simd_l2_distance(query, database)
        expected = l2_distance(query, database)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_l2_batch_query(self):
        """Test L2 distance for batch query."""
        np.random.seed(42)
        queries = np.random.randn(10, 128).astype(np.float32)
        database = np.random.randn(1000, 128).astype(np.float32)

        result = simd_l2_distance(queries, database)
        expected = l2_distance(queries, database)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_inner_product_single(self):
        """Test inner product for single query."""
        np.random.seed(42)
        query = np.random.randn(128).astype(np.float32)
        database = np.random.randn(1000, 128).astype(np.float32)

        result = simd_inner_product(query, database)
        expected = inner_product(query, database)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_inner_product_batch(self):
        """Test inner product for batch query."""
        np.random.seed(42)
        queries = np.random.randn(10, 128).astype(np.float32)
        database = np.random.randn(1000, 128).astype(np.float32)

        result = simd_inner_product(queries, database)
        expected = inner_product(queries, database)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_cosine_similarity_single(self):
        """Test cosine similarity for single query."""
        np.random.seed(42)
        query = np.random.randn(128).astype(np.float32)
        database = np.random.randn(1000, 128).astype(np.float32)

        result = simd_cosine_similarity(query, database)
        expected = cosine_similarity(query, database)

        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_cosine_similarity_batch(self):
        """Test cosine similarity for batch query."""
        np.random.seed(42)
        queries = np.random.randn(10, 128).astype(np.float32)
        database = np.random.randn(1000, 128).astype(np.float32)

        result = simd_cosine_similarity(queries, database)
        expected = cosine_similarity(queries, database)

        np.testing.assert_allclose(result, expected, rtol=1e-4)


class TestSIMDTopK:
    """Test SIMD top-k selection."""

    def test_topk_single(self):
        """Test top-k for single query."""
        np.random.seed(42)
        distances = np.random.randn(1000).astype(np.float32)
        k = 10

        indices, dists = simd_topk(distances, k)

        assert len(indices) == k
        assert len(dists) == k

        # Should be sorted
        for i in range(1, k):
            assert dists[i] >= dists[i-1]

        # Should be correct values
        sorted_idx = np.argsort(distances)[:k]
        np.testing.assert_array_equal(np.sort(indices), np.sort(sorted_idx))

    def test_topk_batch(self):
        """Test top-k for batch query."""
        np.random.seed(42)
        distances = np.random.randn(10, 1000).astype(np.float32)
        k = 10

        indices, dists = simd_topk(distances, k)

        assert indices.shape == (10, k)
        assert dists.shape == (10, k)

        # Check each query
        for i in range(10):
            sorted_idx = np.argsort(distances[i])[:k]
            np.testing.assert_array_equal(
                np.sort(indices[i]),
                np.sort(sorted_idx)
            )

    def test_topk_k_larger_than_n(self):
        """Test top-k when k > n."""
        distances = np.random.randn(5).astype(np.float32)
        k = 10

        indices, dists = simd_topk(distances, k)

        assert len(indices) == 5
        assert len(dists) == 5


class TestSIMDBatchSearch:
    """Test combined distance + top-k search."""

    def test_l2_batch_search(self):
        """Test L2 batch search."""
        np.random.seed(42)
        queries = np.random.randn(10, 128).astype(np.float32)
        database = np.random.randn(1000, 128).astype(np.float32)
        k = 10

        indices, distances = simd_l2_batch(queries, database, k)

        assert indices.shape == (10, k)
        assert distances.shape == (10, k)

        # Verify results match naive computation
        for i in range(10):
            dists = simd_l2_distance(queries[i], database)
            sorted_idx = np.argsort(dists)[:k]
            np.testing.assert_array_equal(
                np.sort(indices[i]),
                np.sort(sorted_idx)
            )

    def test_ip_batch_search(self):
        """Test inner product batch search."""
        np.random.seed(42)
        queries = np.random.randn(10, 128).astype(np.float32)
        database = np.random.randn(1000, 128).astype(np.float32)
        k = 10

        indices, similarities = simd_ip_batch(queries, database, k)

        assert indices.shape == (10, k)
        assert similarities.shape == (10, k)

        # Should have highest similarities
        for i in range(10):
            sims = simd_inner_product(queries[i], database)
            sorted_idx = np.argsort(-sims)[:k]  # Descending
            np.testing.assert_array_equal(
                np.sort(indices[i]),
                np.sort(sorted_idx)
            )

    def test_batch_search_l2(self):
        """Test batch_search with L2 metric."""
        np.random.seed(42)
        queries = np.random.randn(50, 64).astype(np.float32)
        database = np.random.randn(5000, 64).astype(np.float32)
        k = 10

        indices, distances = batch_search(
            queries, database, k,
            metric="l2", batch_size=16
        )

        assert indices.shape == (50, k)
        assert distances.shape == (50, k)

    def test_batch_search_ip(self):
        """Test batch_search with inner product metric."""
        np.random.seed(42)
        queries = np.random.randn(50, 64).astype(np.float32)
        database = np.random.randn(5000, 64).astype(np.float32)
        k = 10

        indices, distances = batch_search(
            queries, database, k,
            metric="ip", batch_size=16
        )

        assert indices.shape == (50, k)
        assert distances.shape == (50, k)

    def test_batch_search_cosine(self):
        """Test batch_search with cosine metric."""
        np.random.seed(42)
        queries = np.random.randn(20, 64).astype(np.float32)
        database = np.random.randn(1000, 64).astype(np.float32)
        k = 10

        indices, distances = batch_search(
            queries, database, k,
            metric="cosine", batch_size=8
        )

        assert indices.shape == (20, k)
        assert distances.shape == (20, k)


class TestSIMDVectorOps:
    """Test SIMDVectorOps class."""

    def test_l2_search(self):
        """Test L2 search."""
        np.random.seed(42)
        database = np.random.randn(1000, 128).astype(np.float32)
        ops = SIMDVectorOps(database, metric="l2")

        assert ops.ntotal == 1000
        assert ops.dim == 128

        queries = np.random.randn(10, 128).astype(np.float32)
        indices, distances = ops.search(queries, k=10)

        assert indices.shape == (10, 10)
        assert distances.shape == (10, 10)

    def test_ip_search(self):
        """Test inner product search."""
        np.random.seed(42)
        database = np.random.randn(1000, 128).astype(np.float32)
        ops = SIMDVectorOps(database, metric="ip")

        queries = np.random.randn(10, 128).astype(np.float32)
        indices, similarities = ops.search(queries, k=10)

        assert indices.shape == (10, 10)

    def test_cosine_search(self):
        """Test cosine search."""
        np.random.seed(42)
        database = np.random.randn(1000, 128).astype(np.float32)
        ops = SIMDVectorOps(database, metric="cosine")

        queries = np.random.randn(10, 128).astype(np.float32)
        indices, similarities = ops.search(queries, k=10)

        assert indices.shape == (10, 10)

    def test_range_search(self):
        """Test range search."""
        np.random.seed(42)
        database = np.random.randn(100, 32).astype(np.float32)
        ops = SIMDVectorOps(database, metric="l2")

        query = np.random.randn(1, 32).astype(np.float32)
        lims, indices, distances = ops.range_search(query, radius=5.0)

        assert len(lims) == 2  # nq + 1
        assert lims[0] == 0
        assert all(d <= 5.0 for d in distances)

    def test_add_vectors(self):
        """Test adding vectors."""
        np.random.seed(42)
        database = np.random.randn(100, 32).astype(np.float32)
        ops = SIMDVectorOps(database, metric="l2")

        assert ops.ntotal == 100

        new_vectors = np.random.randn(50, 32).astype(np.float32)
        ops.add(new_vectors)

        assert ops.ntotal == 150

    def test_single_query(self):
        """Test single query vector."""
        np.random.seed(42)
        database = np.random.randn(100, 32).astype(np.float32)
        ops = SIMDVectorOps(database, metric="l2")

        query = np.random.randn(32).astype(np.float32)
        indices, distances = ops.search(query, k=5)

        assert len(indices.flatten()) == 5


class TestSIMDAvailability:
    """Test SIMD availability flag."""

    def test_simd_available_flag(self):
        """Test that SIMD_AVAILABLE is a boolean."""
        assert isinstance(SIMD_AVAILABLE, bool)

    def test_fallback_works(self):
        """Test that fallback works when numba unavailable."""
        # Even without numba, should produce correct results
        np.random.seed(42)
        query = np.random.randn(64).astype(np.float32)
        database = np.random.randn(100, 64).astype(np.float32)

        result = simd_l2_distance(query, database)
        assert result.shape == (100,)


class TestSIMDEdgeCases:
    """Test edge cases for SIMD operations."""

    def test_empty_database(self):
        """Test with empty database."""
        database = np.empty((0, 64), dtype=np.float32)
        query = np.random.randn(64).astype(np.float32)

        result = simd_l2_distance(query, database)
        assert len(result) == 0

    def test_single_vector_database(self):
        """Test with single vector in database."""
        database = np.random.randn(1, 64).astype(np.float32)
        query = np.random.randn(64).astype(np.float32)

        result = simd_l2_distance(query, database)
        # Use size for numpy arrays/scalars compatibility
        assert np.asarray(result).size == 1

    def test_high_dimensional(self):
        """Test with high-dimensional vectors."""
        np.random.seed(42)
        database = np.random.randn(100, 1024).astype(np.float32)
        queries = np.random.randn(5, 1024).astype(np.float32)

        indices, distances = simd_l2_batch(queries, database, k=10)

        assert indices.shape == (5, 10)
        assert distances.shape == (5, 10)

    def test_low_dimensional(self):
        """Test with low-dimensional vectors."""
        np.random.seed(42)
        database = np.random.randn(100, 2).astype(np.float32)
        queries = np.random.randn(5, 2).astype(np.float32)

        indices, distances = simd_l2_batch(queries, database, k=10)

        assert indices.shape == (5, 10)

    def test_non_contiguous_array(self):
        """Test with non-contiguous array."""
        np.random.seed(42)
        database = np.random.randn(100, 64).astype(np.float32)[:, ::2]  # Non-contiguous
        query = np.random.randn(64).astype(np.float32)[::2]

        # Should not raise
        result = simd_l2_distance(query, database)
        assert len(result) == 100

    def test_float64_input(self):
        """Test with float64 input (should be converted)."""
        np.random.seed(42)
        database = np.random.randn(100, 64).astype(np.float64)
        query = np.random.randn(64).astype(np.float64)

        result = simd_l2_distance(query, database)
        assert result.dtype == np.float32
