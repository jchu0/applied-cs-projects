"""Tests for search functionality and utilities."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vecindex import (
    MetricType,
    SearchResult,
    FlatIndex,
    IVFIndex,
    HNSWIndex,
    l2_distance,
    inner_product,
    cosine_similarity,
    normalize_vectors,
    BatchSearcher,
    HybridSearcher,
    RerankSearcher,
    build_index,
    benchmark_index,
    IndexFactory,
    compute_recall,
)
from vecindex.core.vectors import compute_distance, topk
from conftest import (
    compute_ground_truth_l2,
    compute_ground_truth_ip,
    compute_ground_truth_cosine,
)


class TestDistanceMetrics:
    """Tests for distance/similarity metrics."""

    def test_l2_distance_single(self, medium_dim):
        """Test L2 distance with single vectors."""
        a = np.random.randn(medium_dim).astype(np.float32)
        b = np.random.randn(10, medium_dim).astype(np.float32)

        dists = l2_distance(a, b)
        assert dists.shape == (10,)

        # Verify correctness
        for i in range(10):
            expected = np.linalg.norm(a - b[i])
            np.testing.assert_almost_equal(dists[i], expected, decimal=5)

    def test_l2_distance_batch(self, medium_dim):
        """Test L2 distance with batched queries."""
        a = np.random.randn(5, medium_dim).astype(np.float32)
        b = np.random.randn(10, medium_dim).astype(np.float32)

        dists = l2_distance(a, b)
        assert dists.shape == (5, 10)

        # Verify correctness
        for i in range(5):
            for j in range(10):
                expected = np.linalg.norm(a[i] - b[j])
                np.testing.assert_almost_equal(dists[i, j], expected, decimal=5)

    def test_l2_distance_self(self, medium_dim):
        """Test L2 distance of vector with itself is zero."""
        a = np.random.randn(medium_dim).astype(np.float32)
        dist = l2_distance(a, a.reshape(1, -1))
        # Handle both scalar and array cases
        dist_val = float(dist) if dist.ndim == 0 else dist[0]
        np.testing.assert_almost_equal(dist_val, 0.0, decimal=5)

    def test_inner_product_single(self, medium_dim):
        """Test inner product with single vector."""
        a = np.random.randn(medium_dim).astype(np.float32)
        b = np.random.randn(10, medium_dim).astype(np.float32)

        scores = inner_product(a, b)
        assert scores.shape == (10,)

        # Verify correctness
        for i in range(10):
            expected = np.dot(a, b[i])
            np.testing.assert_almost_equal(scores[i], expected, decimal=5)

    def test_inner_product_normalized(self, normalized_vectors):
        """Test inner product on normalized vectors."""
        a = normalized_vectors[0]
        b = normalized_vectors[1:11]

        scores = inner_product(a, b)

        # All scores should be in [-1, 1] for normalized vectors
        assert np.all(scores >= -1.0 - 1e-5)
        assert np.all(scores <= 1.0 + 1e-5)

    def test_cosine_similarity_single(self, medium_dim):
        """Test cosine similarity with single vector."""
        a = np.random.randn(medium_dim).astype(np.float32)
        b = np.random.randn(10, medium_dim).astype(np.float32)

        sims = cosine_similarity(a, b)
        assert sims.shape == (10,)

        # All similarities should be in [-1, 1]
        assert np.all(sims >= -1.0 - 1e-5)
        assert np.all(sims <= 1.0 + 1e-5)

    def test_cosine_similarity_self(self, medium_dim):
        """Test cosine similarity of vector with itself is 1."""
        a = np.random.randn(medium_dim).astype(np.float32)
        sim = cosine_similarity(a, a.reshape(1, -1))
        # Handle both scalar and array cases
        sim_val = float(sim) if sim.ndim == 0 else sim[0]
        np.testing.assert_almost_equal(sim_val, 1.0, decimal=5)

    def test_cosine_similarity_orthogonal(self, medium_dim):
        """Test cosine similarity of orthogonal vectors is 0."""
        # Create orthogonal vectors
        a = np.zeros(medium_dim, dtype=np.float32)
        a[0] = 1.0
        b = np.zeros(medium_dim, dtype=np.float32)
        b[1] = 1.0

        sim = cosine_similarity(a, b.reshape(1, -1))
        # Handle both scalar and array cases
        sim_val = float(sim) if sim.ndim == 0 else sim[0]
        np.testing.assert_almost_equal(sim_val, 0.0, decimal=5)

    def test_normalize_vectors(self, medium_vectors):
        """Test vector normalization."""
        normalized = normalize_vectors(medium_vectors)

        # All vectors should have unit norm
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones_like(norms), decimal=5)


class TestComputeDistance:
    """Tests for the unified compute_distance function."""

    def test_compute_distance_l2(self, medium_dim):
        """Test compute_distance with L2 metric."""
        a = np.random.randn(medium_dim).astype(np.float32)
        b = np.random.randn(10, medium_dim).astype(np.float32)

        dists = compute_distance(a, b, MetricType.L2)
        expected = l2_distance(a, b)
        np.testing.assert_array_almost_equal(dists, expected, decimal=5)

    def test_compute_distance_ip(self, medium_dim):
        """Test compute_distance with inner product metric."""
        a = np.random.randn(medium_dim).astype(np.float32)
        b = np.random.randn(10, medium_dim).astype(np.float32)

        dists = compute_distance(a, b, MetricType.IP)
        expected = -inner_product(a, b)  # Negated for lower is better
        np.testing.assert_array_almost_equal(dists, expected, decimal=5)

    def test_compute_distance_cosine(self, medium_dim):
        """Test compute_distance with cosine metric."""
        a = np.random.randn(medium_dim).astype(np.float32)
        b = np.random.randn(10, medium_dim).astype(np.float32)

        dists = compute_distance(a, b, MetricType.COSINE)
        expected = -cosine_similarity(a, b)  # Negated for lower is better
        np.testing.assert_array_almost_equal(dists, expected, decimal=5)


class TestTopK:
    """Tests for the topk function."""

    def test_topk_1d(self):
        """Test topk with 1D array."""
        distances = np.array([5.0, 2.0, 8.0, 1.0, 3.0])
        indices, dists = topk(distances, k=3)

        np.testing.assert_array_equal(indices, [3, 1, 4])  # Indices of smallest
        np.testing.assert_array_equal(dists, [1.0, 2.0, 3.0])

    def test_topk_2d(self):
        """Test topk with 2D array (batch)."""
        distances = np.array([
            [5.0, 2.0, 8.0, 1.0],
            [3.0, 7.0, 1.0, 4.0],
        ])
        indices, dists = topk(distances, k=2)

        np.testing.assert_array_equal(indices[0], [3, 1])
        np.testing.assert_array_equal(indices[1], [2, 0])

    def test_topk_k_larger_than_n(self):
        """Test topk with k larger than array size."""
        distances = np.array([5.0, 2.0, 8.0])
        indices, dists = topk(distances, k=10)

        assert len(indices) == 3  # Should return all elements


class TestSearchWithMetrics:
    """Tests for search with different distance metrics."""

    def test_search_l2(self, medium_dim, medium_vectors):
        """Test search with L2 distance."""
        index = FlatIndex(medium_dim, MetricType.L2)
        index.add(medium_vectors)

        query = np.random.randn(medium_dim).astype(np.float32)
        k = 10
        result = index.search(query, k)

        # Verify against ground truth
        ground_truth = compute_ground_truth_l2(medium_vectors, query, k)
        np.testing.assert_array_equal(result.ids, ground_truth)

    def test_search_inner_product(self, medium_dim, normalized_vectors):
        """Test search with inner product."""
        index = FlatIndex(medium_dim, MetricType.IP)
        index.add(normalized_vectors)

        query = normalized_vectors[0]
        k = 10
        result = index.search(query, k)

        # Should find itself as most similar
        assert result.ids[0] == 0

    def test_search_cosine(self, medium_dim, medium_vectors):
        """Test search with cosine similarity."""
        index = FlatIndex(medium_dim, MetricType.COSINE)
        index.add(medium_vectors)

        query = medium_vectors[0]
        k = 10
        result = index.search(query, k)

        # Should find itself as most similar
        assert result.ids[0] == 0


class TestBatchSearcher:
    """Tests for BatchSearcher utility."""

    def test_batch_search(self, flat_index_l2, query_vectors):
        """Test batch search."""
        index, vectors = flat_index_l2
        searcher = BatchSearcher(index)

        k = 5
        result = searcher.search(query_vectors, k)

        assert result.ids.shape == (len(query_vectors), k)
        assert result.distances.shape == (len(query_vectors), k)

    def test_batch_search_single_query(self, flat_index_l2, single_query):
        """Test batch search with single query."""
        index, _ = flat_index_l2
        searcher = BatchSearcher(index)

        result = searcher.search(single_query, k=5)
        assert len(result.ids) == 1
        assert len(result.ids[0]) == 5

    def test_search_with_filter(self, flat_index_l2, single_query):
        """Test filtered search."""
        index, _ = flat_index_l2
        searcher = BatchSearcher(index)

        # Only allow certain IDs
        filter_ids = np.array([0, 5, 10, 15, 20, 25, 30])
        result = searcher.search_with_filter(single_query, k=3, filter_ids=filter_ids)

        # All returned IDs should be in filter set
        for vid in result.ids:
            assert vid in filter_ids


class TestHybridSearcher:
    """Tests for HybridSearcher utility."""

    def test_add_index(self, flat_index_l2):
        """Test adding indexes to hybrid searcher."""
        index, _ = flat_index_l2
        searcher = HybridSearcher()

        searcher.add_index("dense", index, weight=1.0)
        assert "dense" in searcher.indexes
        assert searcher.weights["dense"] == 1.0

    def test_hybrid_search(self, medium_dim, medium_vectors):
        """Test hybrid search with multiple indexes."""
        # Create two indexes
        index1 = FlatIndex(medium_dim, MetricType.L2)
        index1.add(medium_vectors)

        index2 = FlatIndex(medium_dim, MetricType.COSINE)
        index2.add(medium_vectors)

        searcher = HybridSearcher()
        searcher.add_index("l2", index1, weight=1.0)
        searcher.add_index("cosine", index2, weight=1.0)

        query = np.random.randn(medium_dim).astype(np.float32)
        queries = {"l2": query, "cosine": query}

        result = searcher.search(queries, k=10)
        assert len(result.ids) <= 10


class TestRerankSearcher:
    """Tests for RerankSearcher utility."""

    def test_rerank_search(self, medium_dim, medium_vectors):
        """Test two-stage search with re-ranking."""
        # First stage: IVF (approximate)
        first_stage = IVFIndex(medium_dim, nlist=10, nprobe=2)
        first_stage.train(medium_vectors)
        first_stage.add(medium_vectors)

        # Create re-ranker
        reranker = RerankSearcher(first_stage, medium_vectors, MetricType.L2)

        query = np.random.randn(medium_dim).astype(np.float32)
        result = reranker.search(query, k=10, rerank_k=50)

        assert len(result.ids) == 10

        # Results should be sorted by distance
        for i in range(len(result.distances) - 1):
            assert result.distances[i] <= result.distances[i + 1]


class TestBuildIndex:
    """Tests for the build_index convenience function."""

    def test_build_flat_index(self, medium_vectors):
        """Test building a flat index."""
        index = build_index(medium_vectors, index_type="flat")
        assert isinstance(index, FlatIndex)
        assert index.ntotal == len(medium_vectors)

    def test_build_ivf_index(self, medium_vectors):
        """Test building an IVF index."""
        index = build_index(medium_vectors, index_type="ivf", nlist=10, nprobe=3)
        assert isinstance(index, IVFIndex)
        assert index.ntotal == len(medium_vectors)
        assert index.is_trained

    def test_build_hnsw_index(self, medium_vectors):
        """Test building an HNSW index."""
        index = build_index(medium_vectors, index_type="hnsw", M=16)
        assert isinstance(index, HNSWIndex)
        assert index.ntotal == len(medium_vectors)

    def test_build_ivfpq_index(self, medium_vectors):
        """Test building an IVFPQ index."""
        index = build_index(
            medium_vectors,
            index_type="ivfpq",
            nlist=10,
            M=8,
            nbits=8
        )
        from vecindex import IVFPQIndex
        assert isinstance(index, IVFPQIndex)
        assert index.ntotal == len(medium_vectors)

    def test_build_index_with_metric(self, medium_vectors):
        """Test building index with custom metric."""
        index = build_index(medium_vectors, index_type="flat", metric=MetricType.COSINE)
        assert index.metric == MetricType.COSINE

    def test_build_unknown_type(self, medium_vectors):
        """Test that unknown index type raises error."""
        with pytest.raises(ValueError, match="Unknown"):
            build_index(medium_vectors, index_type="unknown")


class TestIndexFactory:
    """Tests for IndexFactory."""

    def test_create_flat(self, medium_dim):
        """Test creating Flat index."""
        index = IndexFactory.create("Flat", medium_dim)
        assert isinstance(index, FlatIndex)

    def test_create_ivf_flat(self, medium_dim):
        """Test creating IVF,Flat index."""
        index = IndexFactory.create("IVF100,Flat", medium_dim)
        assert isinstance(index, IVFIndex)
        assert index.nlist == 100

    def test_create_ivf_pq(self, medium_dim):
        """Test creating IVF,PQ index."""
        from vecindex import IVFPQIndex
        index = IndexFactory.create("IVF100,PQ8", medium_dim)
        assert isinstance(index, IVFPQIndex)
        assert index.nlist == 100
        assert index.M == 8

    def test_create_hnsw(self, medium_dim):
        """Test creating HNSW index."""
        index = IndexFactory.create("HNSW32", medium_dim)
        assert isinstance(index, HNSWIndex)
        assert index.M == 32

    def test_unknown_index_string(self, medium_dim):
        """Test that unknown index string raises error."""
        with pytest.raises(ValueError, match="Unknown"):
            IndexFactory.create("UnknownIndex", medium_dim)


class TestBenchmarkIndex:
    """Tests for benchmark_index function."""

    def test_benchmark(self, medium_dim, medium_vectors):
        """Test benchmarking an index."""
        index = FlatIndex(medium_dim, MetricType.L2)
        index.add(medium_vectors)

        # Create queries and ground truth
        queries = np.random.randn(5, medium_dim).astype(np.float32)
        k = 5

        # Compute ground truth
        ground_truth = []
        for q in queries:
            gt = compute_ground_truth_l2(medium_vectors, q, k)
            ground_truth.append(gt)
        ground_truth = np.array(ground_truth)

        # Benchmark
        metrics = benchmark_index(index, queries, ground_truth, k)

        assert "recall@5" in metrics
        assert "qps" in metrics
        assert "latency_ms" in metrics
        assert metrics["recall@5"] == 1.0  # Flat should be exact
        assert metrics["qps"] > 0
        assert metrics["latency_ms"] > 0


class TestComputeRecall:
    """Tests for compute_recall function."""

    def test_perfect_recall(self):
        """Test recall when results match ground truth exactly."""
        ground_truth = np.array([[0, 1, 2, 3, 4]])
        results = np.array([[0, 1, 2, 3, 4]])
        recall = compute_recall(ground_truth, results, k=5)
        assert recall == 1.0

    def test_zero_recall(self):
        """Test recall when no results match."""
        ground_truth = np.array([[0, 1, 2, 3, 4]])
        results = np.array([[5, 6, 7, 8, 9]])
        recall = compute_recall(ground_truth, results, k=5)
        assert recall == 0.0

    def test_partial_recall(self):
        """Test recall with partial match."""
        ground_truth = np.array([[0, 1, 2, 3, 4]])
        results = np.array([[0, 1, 5, 6, 7]])  # 2/5 correct
        recall = compute_recall(ground_truth, results, k=5)
        assert recall == 0.4

    def test_batch_recall(self):
        """Test recall with multiple queries."""
        ground_truth = np.array([
            [0, 1, 2, 3, 4],
            [5, 6, 7, 8, 9]
        ])
        results = np.array([
            [0, 1, 2, 3, 4],  # 5/5 correct
            [5, 6, 10, 11, 12]  # 2/5 correct
        ])
        recall = compute_recall(ground_truth, results, k=5)
        assert recall == 0.7  # 7/10 correct


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test creating a SearchResult."""
        ids = np.array([1, 2, 3])
        distances = np.array([0.1, 0.2, 0.3])
        result = SearchResult(ids, distances)

        np.testing.assert_array_equal(result.ids, ids)
        np.testing.assert_array_equal(result.distances, distances)
        assert result.num_results == 3

    def test_search_result_empty(self):
        """Test empty SearchResult."""
        result = SearchResult(np.array([]), np.array([]))
        assert result.num_results == 0
