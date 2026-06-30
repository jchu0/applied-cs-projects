"""Vector index implementations for similarity search."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import heapq

from ..core.vectors import (
    MetricType, SearchResult, VectorStore,
    compute_distance, topk, kmeans, l2_distance
)

logger = logging.getLogger(__name__)


class Index(ABC):
    """Base class for vector indexes."""

    def __init__(self, dim: int, metric: MetricType = MetricType.L2):
        self.dim = dim
        self.metric = metric
        self.is_trained = False

    @abstractmethod
    def train(self, vectors: np.ndarray):
        """Train the index."""
        pass

    @abstractmethod
    def add(self, vectors: np.ndarray):
        """Add vectors to the index."""
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> SearchResult:
        """Search for nearest neighbors."""
        pass

    @property
    @abstractmethod
    def ntotal(self) -> int:
        """Total number of indexed vectors."""
        pass

    def __len__(self) -> int:
        return self.ntotal


class FlatIndex(Index):
    """
    Flat (brute-force) index.

    Exact search by computing all distances. O(n) per query.
    """

    def __init__(self, dim: int, metric: MetricType = MetricType.L2):
        super().__init__(dim, metric)
        self.store = VectorStore(dim)
        self.is_trained = True

    def train(self, vectors: np.ndarray):
        """No training needed for flat index."""
        pass

    def add(self, vectors: np.ndarray):
        """Add vectors to index."""
        self.store.add(np.asarray(vectors))

    def search(self, query: np.ndarray, k: int) -> SearchResult:
        """
        Brute-force search.

        Args:
            query: Query vector(s)
            k: Number of results

        Returns:
            SearchResult with IDs and distances
        """
        query = np.asarray(query)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        distances = compute_distance(query, self.store.vectors, self.metric)
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)

        indices, dists = topk(distances, k)
        ids = self.store.ids[indices]

        return SearchResult(ids.squeeze(), dists.squeeze())

    @property
    def ntotal(self) -> int:
        return self.store.ntotal


class IVFIndex(Index):
    """
    Inverted File Index.

    Clusters vectors into partitions and searches only relevant partitions.
    O(n/nlist) per query with nprobe partitions.
    """

    def __init__(
        self,
        dim: int,
        nlist: int,
        metric: MetricType = MetricType.L2,
        nprobe: int = 1
    ):
        """
        Args:
            dim: Vector dimension
            nlist: Number of clusters/partitions
            metric: Distance metric
            nprobe: Number of partitions to search
        """
        super().__init__(dim, metric)
        self.nlist = nlist
        self.nprobe = nprobe

        # Centroids and inverted lists
        self.centroids = None
        self.inverted_lists = [[] for _ in range(nlist)]  # (id, vector) pairs

        self._next_id = 0

    def train(self, vectors: np.ndarray):
        """
        Train centroids using k-means.

        Args:
            vectors: Training vectors
        """
        vectors = np.asarray(vectors)
        self.centroids, _ = kmeans(vectors, self.nlist)
        self.is_trained = True

    def add(self, vectors: np.ndarray):
        """
        Add vectors to appropriate partitions.

        Args:
            vectors: Vectors to add
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding")

        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Find nearest centroid for each vector
        distances = l2_distance(vectors, self.centroids)
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
        assignments = np.argmin(distances, axis=1)

        # Add to inverted lists
        for i, vec in enumerate(vectors):
            list_id = assignments[i]
            self.inverted_lists[list_id].append((self._next_id, vec))
            self._next_id += 1

    def search(self, query: np.ndarray, k: int) -> SearchResult:
        """
        Search using inverted file.

        Args:
            query: Query vector
            k: Number of results

        Returns:
            SearchResult
        """
        query = np.asarray(query)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Find nearest centroids
        centroid_dists = l2_distance(query, self.centroids)
        if centroid_dists.ndim == 1:
            centroid_dists = centroid_dists.reshape(1, -1)

        # Get top nprobe partitions
        partition_indices, _ = topk(centroid_dists, self.nprobe)
        if partition_indices.ndim == 1:
            partition_indices = partition_indices.reshape(1, -1)

        # Search in selected partitions
        all_ids = []
        all_dists = []

        for partitions in partition_indices:
            candidates_ids = []
            candidates_vecs = []

            for p in partitions:
                for vid, vec in self.inverted_lists[p]:
                    candidates_ids.append(vid)
                    candidates_vecs.append(vec)

            if not candidates_ids:
                all_ids.append(np.array([]))
                all_dists.append(np.array([]))
                continue

            candidates_vecs = np.array(candidates_vecs)
            candidates_ids = np.array(candidates_ids)

            distances = compute_distance(query, candidates_vecs, self.metric)
            if distances.ndim == 1:
                distances = distances.reshape(1, -1)

            idx, dists = topk(distances[0], k)
            all_ids.append(candidates_ids[idx])
            all_dists.append(dists)

        return SearchResult(
            np.array(all_ids[0]) if len(all_ids) == 1 else np.array(all_ids),
            np.array(all_dists[0]) if len(all_dists) == 1 else np.array(all_dists)
        )

    @property
    def ntotal(self) -> int:
        return sum(len(lst) for lst in self.inverted_lists)


class HNSWIndex(Index):
    """
    Hierarchical Navigable Small World graph index.

    Builds multi-layer proximity graph for efficient approximate search.
    O(log n) per query.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        metric: MetricType = MetricType.L2
    ):
        """
        Args:
            dim: Vector dimension
            M: Number of connections per node
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of dynamic candidate list during search
            metric: Distance metric
        """
        super().__init__(dim, metric)
        self.M = M
        self.M0 = 2 * M  # Max connections at layer 0
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.ml = 1 / np.log(M)

        # Graph structure
        self.vectors = []
        self.graphs = []  # graphs[layer][node] = [neighbors]
        self.entry_point = -1
        self.max_layer = -1

        self.is_trained = True

    def train(self, vectors: np.ndarray):
        """HNSW doesn't need separate training."""
        pass

    def _get_random_level(self) -> int:
        """Get random layer for new node."""
        return int(-np.log(np.random.random()) * self.ml)

    def _distance(self, id1: int, id2: int) -> float:
        """Compute distance between two indexed vectors."""
        return float(compute_distance(
            self.vectors[id1].reshape(1, -1),
            self.vectors[id2].reshape(1, -1),
            self.metric
        ))

    def _distance_to_query(self, query: np.ndarray, id: int) -> float:
        """Compute distance from query to indexed vector."""
        return float(compute_distance(
            query.reshape(1, -1),
            self.vectors[id].reshape(1, -1),
            self.metric
        ))

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int
    ) -> List[Tuple[float, int]]:
        """
        Search a single layer.

        Returns list of (distance, id) tuples.
        """
        visited = set(entry_points)
        candidates = []
        results = []

        # Initialize with entry points
        for ep in entry_points:
            dist = self._distance_to_query(query, ep)
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(results, (-dist, ep))

        while candidates:
            dist_c, c = heapq.heappop(candidates)

            # Get furthest result
            if results:
                dist_f = -results[0][0]
                if dist_c > dist_f:
                    break

            # Explore neighbors
            if layer < len(self.graphs) and c < len(self.graphs[layer]):
                for neighbor in self.graphs[layer][c]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist_n = self._distance_to_query(query, neighbor)

                        if len(results) < ef or dist_n < -results[0][0]:
                            heapq.heappush(candidates, (dist_n, neighbor))
                            heapq.heappush(results, (-dist_n, neighbor))

                            if len(results) > ef:
                                heapq.heappop(results)

        return [(- d, i) for d, i in results]

    def _select_neighbors(
        self,
        query_id: int,
        candidates: List[Tuple[float, int]],
        M: int
    ) -> List[int]:
        """Select M best neighbors from candidates."""
        candidates = sorted(candidates, key=lambda x: x[0])
        return [c[1] for c in candidates[:M]]

    def add(self, vectors: np.ndarray):
        """
        Add vectors to HNSW index.

        Args:
            vectors: Vectors to add
        """
        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        for vec in vectors:
            self._add_single(vec)

    def _add_single(self, vector: np.ndarray):
        """Add single vector to index."""
        node_id = len(self.vectors)
        self.vectors.append(vector.copy())

        node_level = self._get_random_level()

        # Extend graph structure if needed
        while len(self.graphs) <= node_level:
            self.graphs.append([])

        for layer in self.graphs:
            while len(layer) <= node_id:
                layer.append([])

        # First node
        if self.entry_point == -1:
            self.entry_point = node_id
            self.max_layer = node_level
            return

        # Find entry point for each layer
        ep = [self.entry_point]

        # Descend from top to node_level + 1
        for layer in range(self.max_layer, node_level, -1):
            results = self._search_layer(vector, ep, 1, layer)
            ep = [results[0][1]] if results else ep

        # Insert at layers node_level down to 0
        for layer in range(min(node_level, self.max_layer), -1, -1):
            results = self._search_layer(vector, ep, self.ef_construction, layer)

            # Select neighbors
            M = self.M0 if layer == 0 else self.M
            neighbors = self._select_neighbors(node_id, results, M)

            # Add bidirectional connections
            self.graphs[layer][node_id] = neighbors
            for neighbor in neighbors:
                self.graphs[layer][neighbor].append(node_id)

                # Trim if exceeds M
                if len(self.graphs[layer][neighbor]) > M:
                    # Keep closest
                    neighbor_neighbors = self.graphs[layer][neighbor]
                    dists = [
                        (self._distance(neighbor, nn), nn)
                        for nn in neighbor_neighbors
                    ]
                    dists.sort()
                    self.graphs[layer][neighbor] = [d[1] for d in dists[:M]]

            ep = [r[1] for r in results]

        # Update entry point if new node has higher level
        if node_level > self.max_layer:
            self.entry_point = node_id
            self.max_layer = node_level

    def search(self, query: np.ndarray, k: int) -> SearchResult:
        """
        Search for nearest neighbors in HNSW.

        Args:
            query: Query vector
            k: Number of results

        Returns:
            SearchResult
        """
        if self.entry_point == -1:
            return SearchResult(np.array([]), np.array([]))

        query = np.asarray(query)
        if query.ndim == 2:
            query = query[0]

        # Descend from top layer
        ep = [self.entry_point]
        for layer in range(self.max_layer, 0, -1):
            results = self._search_layer(query, ep, 1, layer)
            ep = [results[0][1]] if results else ep

        # Search at layer 0
        results = self._search_layer(query, ep, self.ef_search, 0)

        # Get top k
        results.sort()
        results = results[:k]

        ids = np.array([r[1] for r in results])
        dists = np.array([r[0] for r in results])

        return SearchResult(ids, dists)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)


class IVFPQIndex(Index):
    """
    IVF with Product Quantization.

    Combines IVF partitioning with PQ compression for memory efficiency.
    """

    def __init__(
        self,
        dim: int,
        nlist: int,
        M: int,  # Number of subquantizers
        nbits: int = 8,
        metric: MetricType = MetricType.L2,
        nprobe: int = 1
    ):
        """
        Args:
            dim: Vector dimension
            nlist: Number of clusters
            M: Number of subquantizers
            nbits: Bits per subquantizer (2^nbits centroids)
            metric: Distance metric
            nprobe: Number of partitions to search
        """
        super().__init__(dim, metric)
        self.nlist = nlist
        self.M = M
        self.nbits = nbits
        self.nprobe = nprobe
        self.ksub = 2 ** nbits  # Centroids per subquantizer

        assert dim % M == 0, "Dimension must be divisible by M"
        self.dsub = dim // M

        # Centroids
        self.coarse_centroids = None  # (nlist, dim)
        self.pq_centroids = None      # (M, ksub, dsub)

        # Inverted lists with codes
        self.inverted_lists = [[] for _ in range(nlist)]  # (id, codes) pairs
        self._next_id = 0

    def train(self, vectors: np.ndarray):
        """
        Train coarse and PQ centroids.

        Args:
            vectors: Training vectors
        """
        vectors = np.asarray(vectors)

        # Train coarse centroids
        self.coarse_centroids, _ = kmeans(vectors, self.nlist)

        # Compute residuals
        distances = l2_distance(vectors, self.coarse_centroids)
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
        assignments = np.argmin(distances, axis=1)
        residuals = vectors - self.coarse_centroids[assignments]

        # Train PQ on residuals
        self.pq_centroids = np.zeros((self.M, self.ksub, self.dsub))
        for m in range(self.M):
            sub_vectors = residuals[:, m * self.dsub:(m + 1) * self.dsub]
            self.pq_centroids[m], _ = kmeans(sub_vectors, self.ksub)

        self.is_trained = True

    def _encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors using PQ."""
        n = vectors.shape[0]
        codes = np.zeros((n, self.M), dtype=np.uint8)

        for m in range(self.M):
            sub_vectors = vectors[:, m * self.dsub:(m + 1) * self.dsub]
            distances = l2_distance(sub_vectors, self.pq_centroids[m])
            if distances.ndim == 1:
                distances = distances.reshape(1, -1)
            codes[:, m] = np.argmin(distances, axis=1)

        return codes

    def add(self, vectors: np.ndarray):
        """Add vectors with PQ encoding."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding")

        vectors = np.asarray(vectors)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        # Assign to coarse centroids
        distances = l2_distance(vectors, self.coarse_centroids)
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
        assignments = np.argmin(distances, axis=1)

        # Compute residuals and encode
        residuals = vectors - self.coarse_centroids[assignments]
        codes = self._encode(residuals)

        # Add to inverted lists
        for i in range(len(vectors)):
            list_id = assignments[i]
            self.inverted_lists[list_id].append((self._next_id, codes[i]))
            self._next_id += 1

    def search(self, query: np.ndarray, k: int) -> SearchResult:
        """Search with asymmetric distance computation."""
        query = np.asarray(query)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Find nearest coarse centroids
        coarse_dists = l2_distance(query, self.coarse_centroids)
        if coarse_dists.ndim == 1:
            coarse_dists = coarse_dists.reshape(1, -1)
        partition_indices, _ = topk(coarse_dists, self.nprobe)
        if partition_indices.ndim == 1:
            partition_indices = partition_indices.reshape(1, -1)

        # Precompute distance tables
        query_residuals = query[0] - self.coarse_centroids[partition_indices[0]]

        # Search in selected partitions
        candidates = []
        for p_idx, p in enumerate(partition_indices[0]):
            residual = query_residuals[p_idx]

            # Build distance table for this partition
            dist_table = np.zeros((self.M, self.ksub))
            for m in range(self.M):
                sub_query = residual[m * self.dsub:(m + 1) * self.dsub]
                dist_table[m] = np.sum(
                    (self.pq_centroids[m] - sub_query) ** 2,
                    axis=1
                )

            # Score candidates
            for vid, codes in self.inverted_lists[p]:
                dist = sum(dist_table[m, codes[m]] for m in range(self.M))
                candidates.append((dist, vid))

        if not candidates:
            return SearchResult(np.array([]), np.array([]))

        # Get top k
        candidates.sort()
        candidates = candidates[:k]

        ids = np.array([c[1] for c in candidates])
        dists = np.array([np.sqrt(c[0]) for c in candidates])

        return SearchResult(ids, dists)

    @property
    def ntotal(self) -> int:
        return sum(len(lst) for lst in self.inverted_lists)
