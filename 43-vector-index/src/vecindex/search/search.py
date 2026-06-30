"""Search utilities for vector indexes."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..core.vectors import SearchResult, MetricType, l2_distance, normalize_vectors
from ..index.indexes import Index, FlatIndex

logger = logging.getLogger(__name__)


@dataclass
class SearchParams:
    """Parameters for search operations."""
    k: int = 10
    nprobe: int = 1
    ef_search: int = 50
    rerank: bool = False
    rerank_k: int = 100


class BatchSearcher:
    """
    Batch search utility for efficient querying.

    Supports batching, filtering, and re-ranking.
    """

    def __init__(self, index: Index):
        self.index = index

    def search(
        self,
        queries: np.ndarray,
        k: int,
        batch_size: int = 1000
    ) -> SearchResult:
        """
        Batch search with automatic batching.

        Args:
            queries: Query vectors (nq, d)
            k: Number of results per query
            batch_size: Queries per batch

        Returns:
            SearchResult with all results
        """
        queries = np.asarray(queries)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        nq = queries.shape[0]
        all_ids = []
        all_dists = []

        for start in range(0, nq, batch_size):
            end = min(start + batch_size, nq)
            batch = queries[start:end]

            # Search each query
            for query in batch:
                result = self.index.search(query, k)
                all_ids.append(result.ids)
                all_dists.append(result.distances)

        return SearchResult(
            np.array(all_ids),
            np.array(all_dists)
        )

    def search_with_filter(
        self,
        query: np.ndarray,
        k: int,
        filter_ids: np.ndarray
    ) -> SearchResult:
        """
        Search with ID filtering.

        Only returns results from filter_ids set.

        Args:
            query: Query vector
            k: Number of results
            filter_ids: Allowed IDs

        Returns:
            Filtered SearchResult
        """
        # Get more candidates
        result = self.index.search(query, k * 10)

        # Filter
        filter_set = set(filter_ids)
        filtered_ids = []
        filtered_dists = []

        for i, vid in enumerate(result.ids):
            if vid in filter_set:
                filtered_ids.append(vid)
                filtered_dists.append(result.distances[i])

            if len(filtered_ids) >= k:
                break

        return SearchResult(
            np.array(filtered_ids),
            np.array(filtered_dists)
        )


class HybridSearcher:
    """
    Hybrid search combining multiple indexes.

    Supports combining dense and sparse retrieval.
    """

    def __init__(self):
        self.indexes = {}
        self.weights = {}

    def add_index(self, name: str, index: Index, weight: float = 1.0):
        """Add an index with a weight."""
        self.indexes[name] = index
        self.weights[name] = weight

    def search(self, queries: Dict[str, np.ndarray], k: int) -> SearchResult:
        """
        Hybrid search across indexes.

        Args:
            queries: Dict mapping index name to query
            k: Number of results

        Returns:
            Combined SearchResult
        """
        # Collect results from each index
        all_results = {}
        for name, index in self.indexes.items():
            if name in queries:
                result = index.search(queries[name], k * 2)
                all_results[name] = result

        # Score fusion
        scores = {}  # id -> total score

        for name, result in all_results.items():
            weight = self.weights[name]
            for i, vid in enumerate(result.ids):
                # Convert distance to score (reciprocal rank)
                score = weight / (i + 1)
                vid_int = int(vid)
                scores[vid_int] = scores.get(vid_int, 0) + score

        # Sort by score
        sorted_items = sorted(scores.items(), key=lambda x: -x[1])[:k]

        ids = np.array([item[0] for item in sorted_items])
        dists = np.array([1.0 / item[1] for item in sorted_items])

        return SearchResult(ids, dists)


class RerankSearcher:
    """
    Two-stage search with re-ranking.

    First stage: fast approximate search
    Second stage: exact re-ranking of candidates
    """

    def __init__(
        self,
        first_stage: Index,
        vectors: np.ndarray,
        metric: MetricType = MetricType.L2
    ):
        """
        Args:
            first_stage: Fast index for candidate retrieval
            vectors: Original vectors for re-ranking
            metric: Distance metric
        """
        self.first_stage = first_stage
        self.vectors = vectors
        self.metric = metric

    def search(
        self,
        query: np.ndarray,
        k: int,
        rerank_k: int = 100
    ) -> SearchResult:
        """
        Two-stage search with re-ranking.

        Args:
            query: Query vector
            k: Final number of results
            rerank_k: Candidates for re-ranking

        Returns:
            Re-ranked SearchResult
        """
        # First stage: get candidates
        candidates = self.first_stage.search(query, rerank_k)

        # Second stage: exact re-ranking
        candidate_vectors = self.vectors[candidates.ids]

        query = np.asarray(query)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Compute exact distances
        from ..core.vectors import compute_distance
        distances = compute_distance(query, candidate_vectors, self.metric)
        if distances.ndim == 1:
            distances = distances.reshape(-1)

        # Sort and return top k
        sorted_idx = np.argsort(distances)[:k]
        ids = candidates.ids[sorted_idx]
        dists = distances[sorted_idx]

        return SearchResult(ids, dists)


class RangeSearcher:
    """Search for all vectors within a distance threshold."""

    def __init__(self, index: Index, vectors: np.ndarray):
        self.index = index
        self.vectors = vectors

    def search(
        self,
        query: np.ndarray,
        radius: float,
        max_results: int = 1000
    ) -> SearchResult:
        """
        Range search.

        Args:
            query: Query vector
            radius: Distance threshold
            max_results: Maximum results to return

        Returns:
            SearchResult with all vectors within radius
        """
        # Get initial candidates
        result = self.index.search(query, max_results)

        # Filter by radius
        mask = result.distances <= radius
        ids = result.ids[mask]
        dists = result.distances[mask]

        return SearchResult(ids, dists)


def build_index(
    vectors: np.ndarray,
    index_type: str = "flat",
    metric: MetricType = MetricType.L2,
    **kwargs
) -> Index:
    """
    Build an index from vectors.

    Args:
        vectors: Input vectors
        index_type: Type of index ('flat', 'ivf', 'hnsw', 'ivfpq')
        metric: Distance metric
        **kwargs: Index-specific parameters

    Returns:
        Trained index with vectors added
    """
    from ..index.indexes import FlatIndex, IVFIndex, HNSWIndex, IVFPQIndex

    vectors = np.asarray(vectors, dtype=np.float32)
    dim = vectors.shape[1]

    if index_type == "flat":
        index = FlatIndex(dim, metric)
    elif index_type == "ivf":
        nlist = kwargs.get("nlist", 100)
        nprobe = kwargs.get("nprobe", 10)
        index = IVFIndex(dim, nlist, metric, nprobe)
        index.train(vectors)
    elif index_type == "hnsw":
        M = kwargs.get("M", 16)
        ef_construction = kwargs.get("ef_construction", 200)
        ef_search = kwargs.get("ef_search", 50)
        index = HNSWIndex(dim, M, ef_construction, ef_search, metric)
    elif index_type == "ivfpq":
        nlist = kwargs.get("nlist", 100)
        M = kwargs.get("M", 8)
        nbits = kwargs.get("nbits", 8)
        index = IVFPQIndex(dim, nlist, M, nbits, metric)
        index.train(vectors)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(vectors)
    return index


def benchmark_index(
    index: Index,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10
) -> Dict[str, float]:
    """
    Benchmark index performance.

    Args:
        index: Index to benchmark
        queries: Query vectors
        ground_truth: True nearest neighbors
        k: Number of results

    Returns:
        Dict with metrics (recall, qps)
    """
    import time

    queries = np.asarray(queries)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)

    nq = queries.shape[0]

    # Measure time
    start = time.perf_counter()
    all_results = []
    for query in queries:
        result = index.search(query, k)
        all_results.append(result.ids)
    elapsed = time.perf_counter() - start

    # Compute recall
    from ..quantize.pq import compute_recall
    results_array = np.array(all_results)
    recall = compute_recall(ground_truth, results_array, k)

    return {
        "recall@%d" % k: recall,
        "qps": nq / elapsed,
        "latency_ms": elapsed * 1000 / nq,
    }


class IndexFactory:
    """Factory for creating indexes with standard configurations."""

    @staticmethod
    def create(index_string: str, dim: int) -> Index:
        """
        Create index from description string.

        Examples:
            "Flat" -> FlatIndex
            "IVF100,Flat" -> IVF with 100 clusters
            "HNSW32" -> HNSW with M=32
            "IVF100,PQ8" -> IVF with PQ8

        Args:
            index_string: Index description
            dim: Vector dimension

        Returns:
            Index instance
        """
        from ..index.indexes import FlatIndex, IVFIndex, HNSWIndex, IVFPQIndex

        parts = index_string.split(",")

        if index_string == "Flat":
            return FlatIndex(dim)

        elif parts[0].startswith("IVF"):
            nlist = int(parts[0][3:])

            if len(parts) == 2 and parts[1] == "Flat":
                return IVFIndex(dim, nlist)
            elif len(parts) == 2 and parts[1].startswith("PQ"):
                M = int(parts[1][2:])
                return IVFPQIndex(dim, nlist, M)
            else:
                return IVFIndex(dim, nlist)

        elif parts[0].startswith("HNSW"):
            M = int(parts[0][4:])
            return HNSWIndex(dim, M=M)

        else:
            raise ValueError(f"Unknown index string: {index_string}")
