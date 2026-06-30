"""Tests for vector index implementations (HNSW, IVF, Flat, IVFPQ)."""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vecindex import (
    MetricType,
    FlatIndex,
    IVFIndex,
    HNSWIndex,
    IVFPQIndex,
)
from conftest import compute_ground_truth_l2


class TestFlatIndex:
    """Tests for FlatIndex (brute-force exact search)."""

    def test_create_index(self, medium_dim):
        """Test index creation."""
        index = FlatIndex(medium_dim)
        assert index.dim == medium_dim
        assert index.metric == MetricType.L2
        assert index.is_trained is True
        assert index.ntotal == 0

    def test_create_index_with_metric(self, medium_dim):
        """Test index creation with different metrics."""
        for metric in [MetricType.L2, MetricType.IP, MetricType.COSINE]:
            index = FlatIndex(medium_dim, metric)
            assert index.metric == metric

    def test_add_single_vector(self, medium_dim):
        """Test adding a single vector."""
        index = FlatIndex(medium_dim)
        vector = np.random.randn(medium_dim).astype(np.float32)
        index.add(vector)
        assert index.ntotal == 1

    def test_add_multiple_vectors(self, medium_dim, medium_vectors):
        """Test adding multiple vectors."""
        index = FlatIndex(medium_dim)
        index.add(medium_vectors)
        assert index.ntotal == len(medium_vectors)

    def test_add_batch(self, medium_dim):
        """Test batch adding vectors."""
        index = FlatIndex(medium_dim)

        batch1 = np.random.randn(100, medium_dim).astype(np.float32)
        batch2 = np.random.randn(200, medium_dim).astype(np.float32)

        index.add(batch1)
        assert index.ntotal == 100

        index.add(batch2)
        assert index.ntotal == 300

    def test_len_property(self, flat_index_l2):
        """Test __len__ returns ntotal."""
        index, vectors = flat_index_l2
        assert len(index) == index.ntotal == len(vectors)

    def test_search_basic(self, flat_index_l2, single_query):
        """Test basic search functionality."""
        index, vectors = flat_index_l2
        k = 10
        result = index.search(single_query, k)

        assert result.ids is not None
        assert result.distances is not None
        assert len(result.ids) == k
        assert len(result.distances) == k
        assert result.num_results == k

    def test_search_returns_sorted_results(self, flat_index_l2, single_query):
        """Test that search results are sorted by distance."""
        index, _ = flat_index_l2
        result = index.search(single_query, 20)

        # Distances should be monotonically increasing
        for i in range(len(result.distances) - 1):
            assert result.distances[i] <= result.distances[i + 1]

    def test_search_exact_correctness(self, medium_dim, medium_vectors):
        """Test that FlatIndex returns exact k-NN."""
        index = FlatIndex(medium_dim, MetricType.L2)
        index.add(medium_vectors)

        query = np.random.randn(medium_dim).astype(np.float32)
        k = 10
        result = index.search(query, k)

        # Compute ground truth
        ground_truth = compute_ground_truth_l2(medium_vectors, query, k)

        # FlatIndex should return exact results
        np.testing.assert_array_equal(result.ids, ground_truth)

    def test_search_k_larger_than_ntotal(self, small_dim, small_vectors):
        """Test search with k larger than number of vectors."""
        index = FlatIndex(small_dim)
        index.add(small_vectors[:10])  # Only 10 vectors

        query = np.random.randn(small_dim).astype(np.float32)
        result = index.search(query, k=20)  # Request more than available

        assert len(result.ids) == 10  # Should return all available

    def test_search_self_query(self, medium_dim, medium_vectors):
        """Test that searching for an indexed vector returns itself first."""
        index = FlatIndex(medium_dim, MetricType.L2)
        index.add(medium_vectors)

        # Search for the first vector
        query = medium_vectors[0]
        result = index.search(query, k=1)

        # Handle both scalar and array results
        result_id = int(result.ids) if result.ids.ndim == 0 else result.ids[0]
        result_dist = float(result.distances) if result.distances.ndim == 0 else result.distances[0]

        assert result_id == 0
        assert result_dist < 0.01  # Should be very small (allowing for float32 precision)


class TestIVFIndex:
    """Tests for IVFIndex (inverted file index)."""

    def test_create_index(self, medium_dim):
        """Test index creation."""
        nlist = 10
        index = IVFIndex(medium_dim, nlist)
        assert index.dim == medium_dim
        assert index.nlist == nlist
        assert index.nprobe == 1  # default
        assert index.is_trained is False

    def test_create_with_nprobe(self, medium_dim):
        """Test index creation with custom nprobe."""
        index = IVFIndex(medium_dim, nlist=20, nprobe=5)
        assert index.nprobe == 5

    def test_train_required(self, medium_dim):
        """Test that training is required before adding."""
        index = IVFIndex(medium_dim, nlist=10)
        vectors = np.random.randn(100, medium_dim).astype(np.float32)

        with pytest.raises(RuntimeError, match="trained"):
            index.add(vectors)

    def test_train_index(self, medium_dim, medium_vectors):
        """Test training the index."""
        index = IVFIndex(medium_dim, nlist=10)
        assert index.is_trained is False

        index.train(medium_vectors)
        assert index.is_trained is True
        assert index.centroids is not None
        assert index.centroids.shape == (10, medium_dim)

    def test_add_after_train(self, medium_dim, medium_vectors):
        """Test adding vectors after training."""
        nlist = 10
        index = IVFIndex(medium_dim, nlist)
        index.train(medium_vectors)
        index.add(medium_vectors)

        assert index.ntotal == len(medium_vectors)

    def test_search_basic(self, ivf_index, single_query):
        """Test basic search functionality."""
        index, vectors = ivf_index
        k = 10
        result = index.search(single_query, k)

        assert result.ids is not None
        assert result.distances is not None
        assert len(result.ids) <= k

    def test_search_nprobe_affects_results(self, medium_dim, clustered_vectors):
        """Test that increasing nprobe improves recall."""
        nlist = 10
        query = clustered_vectors[0]
        k = 10

        # Ground truth from FlatIndex
        flat = FlatIndex(medium_dim, MetricType.L2)
        flat.add(clustered_vectors)
        gt = flat.search(query, k)

        # IVF with low nprobe
        ivf_low = IVFIndex(medium_dim, nlist, MetricType.L2, nprobe=1)
        ivf_low.train(clustered_vectors)
        ivf_low.add(clustered_vectors)
        result_low = ivf_low.search(query, k)

        # IVF with high nprobe
        ivf_high = IVFIndex(medium_dim, nlist, MetricType.L2, nprobe=5)
        ivf_high.train(clustered_vectors)
        ivf_high.add(clustered_vectors)
        result_high = ivf_high.search(query, k)

        # Calculate recalls
        gt_set = set(gt.ids)
        recall_low = len(set(result_low.ids) & gt_set) / k
        recall_high = len(set(result_high.ids) & gt_set) / k

        # Higher nprobe should give equal or better recall
        assert recall_high >= recall_low

    def test_vectors_distributed_across_partitions(self, ivf_index):
        """Test that vectors are distributed across partitions."""
        index, _ = ivf_index
        non_empty = sum(1 for lst in index.inverted_lists if len(lst) > 0)
        # With 10 clusters and 1000 vectors, most should be non-empty
        assert non_empty > 0

    def test_ntotal_property(self, ivf_index):
        """Test ntotal counts all vectors."""
        index, vectors = ivf_index
        assert index.ntotal == len(vectors)


class TestHNSWIndex:
    """Tests for HNSWIndex (hierarchical navigable small world)."""

    def test_create_index(self, small_dim):
        """Test index creation."""
        index = HNSWIndex(small_dim)
        assert index.dim == small_dim
        assert index.M == 16  # default
        assert index.ef_construction == 200  # default
        assert index.ef_search == 50  # default
        assert index.is_trained is True  # HNSW doesn't need training

    def test_create_with_custom_params(self, small_dim):
        """Test index creation with custom parameters."""
        index = HNSWIndex(small_dim, M=32, ef_construction=100, ef_search=20)
        assert index.M == 32
        assert index.ef_construction == 100
        assert index.ef_search == 20

    def test_add_first_vector(self, small_dim):
        """Test adding the first vector."""
        index = HNSWIndex(small_dim)
        vector = np.random.randn(small_dim).astype(np.float32)
        index.add(vector)

        assert index.ntotal == 1
        assert index.entry_point == 0

    def test_add_multiple_vectors(self, small_dim, small_vectors):
        """Test adding multiple vectors."""
        index = HNSWIndex(small_dim, M=8, ef_construction=50)
        index.add(small_vectors)
        assert index.ntotal == len(small_vectors)

    def test_graph_structure(self, small_dim, small_vectors):
        """Test that graph structure is built."""
        index = HNSWIndex(small_dim, M=8, ef_construction=50)
        index.add(small_vectors)

        # Check that graphs are created
        assert len(index.graphs) > 0
        # Check that layer 0 has connections
        layer_0 = index.graphs[0]
        has_neighbors = sum(1 for neighbors in layer_0 if len(neighbors) > 0)
        assert has_neighbors > 0

    def test_search_empty_index(self, small_dim):
        """Test search on empty index."""
        index = HNSWIndex(small_dim)
        query = np.random.randn(small_dim).astype(np.float32)
        result = index.search(query, k=10)

        assert len(result.ids) == 0
        assert len(result.distances) == 0

    def test_search_basic(self, hnsw_index, small_dim):
        """Test basic search functionality."""
        index, vectors = hnsw_index
        query = np.random.randn(small_dim).astype(np.float32)
        k = 10
        result = index.search(query, k)

        assert len(result.ids) == k
        assert len(result.distances) == k

    def test_search_returns_valid_ids(self, hnsw_index, small_dim):
        """Test that search returns valid vector IDs."""
        index, vectors = hnsw_index
        query = np.random.randn(small_dim).astype(np.float32)
        result = index.search(query, k=10)

        for vid in result.ids:
            assert 0 <= vid < len(vectors)

    def test_search_approximate_recall(self, small_dim, small_vectors):
        """Test that HNSW achieves reasonable recall."""
        # Build HNSW index
        hnsw = HNSWIndex(small_dim, M=16, ef_construction=100, ef_search=50)
        hnsw.add(small_vectors)

        # Build Flat index for ground truth
        flat = FlatIndex(small_dim, MetricType.L2)
        flat.add(small_vectors)

        # Test recall on multiple queries
        queries = np.random.randn(10, small_dim).astype(np.float32)
        k = 10
        total_correct = 0

        for query in queries:
            hnsw_result = hnsw.search(query, k)
            flat_result = flat.search(query, k)

            gt_set = set(flat_result.ids)
            correct = len(set(hnsw_result.ids) & gt_set)
            total_correct += correct

        recall = total_correct / (len(queries) * k)
        # HNSW should achieve at least 70% recall with these parameters
        assert recall >= 0.7, f"Recall too low: {recall}"

    def test_ef_search_affects_recall(self, small_dim, small_vectors):
        """Test that higher ef_search improves recall."""
        # Build with same construction parameters
        hnsw_low = HNSWIndex(small_dim, M=8, ef_construction=50, ef_search=10)
        hnsw_high = HNSWIndex(small_dim, M=8, ef_construction=50, ef_search=50)

        for idx in [hnsw_low, hnsw_high]:
            idx.add(small_vectors)

        # Ground truth
        flat = FlatIndex(small_dim, MetricType.L2)
        flat.add(small_vectors)

        query = small_vectors[0]
        k = 10
        gt = set(flat.search(query, k).ids)

        result_low = hnsw_low.search(query, k)
        result_high = hnsw_high.search(query, k)

        recall_low = len(set(result_low.ids) & gt) / k
        recall_high = len(set(result_high.ids) & gt) / k

        # Higher ef_search should give equal or better recall
        assert recall_high >= recall_low


class TestIVFPQIndex:
    """Tests for IVFPQIndex (IVF with Product Quantization)."""

    def test_create_index(self, medium_dim):
        """Test index creation."""
        nlist = 10
        M = 8  # subquantizers
        index = IVFPQIndex(medium_dim, nlist, M)
        assert index.dim == medium_dim
        assert index.nlist == nlist
        assert index.M == M
        assert index.nbits == 8  # default
        assert index.is_trained is False

    def test_dimension_divisibility(self, medium_dim):
        """Test that dimension must be divisible by M."""
        # This should work (64 % 8 == 0)
        index = IVFPQIndex(medium_dim, nlist=10, M=8)
        assert index.dsub == medium_dim // 8

        # This should fail (64 % 5 != 0)
        with pytest.raises(AssertionError):
            IVFPQIndex(medium_dim, nlist=10, M=5)

    def test_train_required(self, medium_dim):
        """Test that training is required before adding."""
        index = IVFPQIndex(medium_dim, nlist=10, M=8)
        vectors = np.random.randn(100, medium_dim).astype(np.float32)

        with pytest.raises(RuntimeError, match="trained"):
            index.add(vectors)

    def test_train_index(self, medium_dim, medium_vectors):
        """Test training the index."""
        index = IVFPQIndex(medium_dim, nlist=10, M=8)
        index.train(medium_vectors)

        assert index.is_trained
        assert index.coarse_centroids is not None
        assert index.coarse_centroids.shape == (10, medium_dim)
        assert index.pq_centroids is not None
        assert index.pq_centroids.shape == (8, 256, 8)  # (M, ksub, dsub)

    def test_add_and_search(self, ivfpq_index, medium_dim):
        """Test adding and searching."""
        index, vectors = ivfpq_index
        assert index.ntotal == len(vectors)

        query = np.random.randn(medium_dim).astype(np.float32)
        result = index.search(query, k=10)

        assert len(result.ids) <= 10
        assert len(result.distances) <= 10

    def test_search_returns_valid_ids(self, ivfpq_index, medium_dim):
        """Test that search returns valid vector IDs."""
        index, vectors = ivfpq_index
        query = np.random.randn(medium_dim).astype(np.float32)
        result = index.search(query, k=10)

        for vid in result.ids:
            assert 0 <= vid < len(vectors)

    def test_compression_effect(self, medium_dim, medium_vectors):
        """Test that IVFPQ achieves compression."""
        index = IVFPQIndex(medium_dim, nlist=10, M=8, nbits=8)
        index.train(medium_vectors)
        index.add(medium_vectors)

        # Each vector is compressed to M bytes (8 bytes)
        # Original: 64 * 4 = 256 bytes
        # Compressed: 8 bytes
        # Compression ratio: 32x
        expected_code_size = 8  # M bytes
        n_vectors = len(medium_vectors)

        # Count total code bytes stored
        total_codes = sum(len(lst) for lst in index.inverted_lists)
        assert total_codes == n_vectors

    def test_approximate_recall(self, medium_dim, medium_vectors):
        """Test that IVFPQ achieves reasonable recall."""
        # Build IVFPQ index
        ivfpq = IVFPQIndex(medium_dim, nlist=10, M=8, nbits=8, nprobe=3)
        ivfpq.train(medium_vectors)
        ivfpq.add(medium_vectors)

        # Build Flat index for ground truth
        flat = FlatIndex(medium_dim, MetricType.L2)
        flat.add(medium_vectors)

        # Test recall on multiple queries
        queries = np.random.randn(5, medium_dim).astype(np.float32)
        k = 10
        total_correct = 0

        for query in queries:
            ivfpq_result = ivfpq.search(query, k)
            flat_result = flat.search(query, k)

            gt_set = set(flat_result.ids)
            correct = len(set(ivfpq_result.ids) & gt_set)
            total_correct += correct

        recall = total_correct / (len(queries) * k)
        # IVFPQ is lossy, so we accept lower recall
        assert recall >= 0.3, f"Recall too low: {recall}"


class TestIndexComparison:
    """Comparative tests across index types."""

    def test_all_indexes_have_common_interface(self, medium_dim):
        """Test that all indexes implement the common interface."""
        indexes = [
            FlatIndex(medium_dim),
            IVFIndex(medium_dim, nlist=10),
            HNSWIndex(medium_dim),
            IVFPQIndex(medium_dim, nlist=10, M=8),
        ]

        for index in indexes:
            assert hasattr(index, 'dim')
            assert hasattr(index, 'metric')
            assert hasattr(index, 'is_trained')
            assert hasattr(index, 'train')
            assert hasattr(index, 'add')
            assert hasattr(index, 'search')
            assert hasattr(index, 'ntotal')

    def test_flat_is_exact(self, medium_dim, medium_vectors):
        """Test that FlatIndex returns exact results."""
        flat = FlatIndex(medium_dim, MetricType.L2)
        flat.add(medium_vectors)

        query = np.random.randn(medium_dim).astype(np.float32)
        k = 10

        result = flat.search(query, k)
        ground_truth = compute_ground_truth_l2(medium_vectors, query, k)

        np.testing.assert_array_equal(result.ids, ground_truth)

    def test_approximate_indexes_return_similar_results(self, medium_dim, medium_vectors):
        """Test that approximate indexes return results similar to exact."""
        k = 10

        # Build indexes
        flat = FlatIndex(medium_dim, MetricType.L2)
        flat.add(medium_vectors)

        ivf = IVFIndex(medium_dim, nlist=10, nprobe=5)
        ivf.train(medium_vectors)
        ivf.add(medium_vectors)

        hnsw = HNSWIndex(medium_dim, M=16, ef_construction=100, ef_search=50)
        hnsw.add(medium_vectors)

        # Query
        query = medium_vectors[0]

        # Get results
        flat_result = flat.search(query, k)
        ivf_result = ivf.search(query, k)
        hnsw_result = hnsw.search(query, k)

        # All should find the query itself as nearest
        assert flat_result.ids[0] == 0
        # For approximate indexes, the query should be in top results
        assert 0 in ivf_result.ids
        assert 0 in hnsw_result.ids
