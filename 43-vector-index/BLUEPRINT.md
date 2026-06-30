# Project 42: High-Performance Vector Index (FAISS-lite)

## Executive Summary

A high-performance approximate nearest neighbor (ANN) search system implementing HNSW, IVF, and product quantization algorithms. Optimized for SIMD and GPU acceleration, this project enables sub-millisecond search across billion-scale vector collections with configurable recall/speed tradeoffs.

> **Concepts covered:** [§04 Vector stores](../../04-ai-engineering/03-vector-databases/vector-stores/vector-stores.md) · [§04 Embeddings](../../04-ai-engineering/03-vector-databases/embeddings/embeddings.md) · [§01 Rust SIMD](../../01-software-engineering/rust/) (for the SIMD-accelerated index paths). Used by RAG projects [25](../25-rag-baseline/), [26](../26-advanced-rag/), [27](../27-micro-model-orchestrated-rag/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture Overview

### System Design

```
+------------------------------------------------------------------+
|                    Vector Index Architecture                      |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Index Builder     |     | Search Engine     |     | Quantizer | |
|  | (Train/Add)       |<--->| (Query/Filter)    |<--->| (PQ/OPQ)  | |
|  +-------------------+     +-------------------+     +-----------+ |
|         |                          |                       |       |
|         v                          v                       v       |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Index Storage     |     | SIMD Kernels      |     | GPU       | |
|  | (mmap/memory)     |     | (AVX512/NEON)     |     | Kernels   | |
|  +-------------------+     +-------------------+     +-----------+ |
|                                    |                               |
|  +----------------------------------------------------------+     |
|  |                    Index Types                            |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  |  | HNSW   |  |  IVF   |  |   PQ   |  | Flat   |           |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  +----------------------------------------------------------+     |
+------------------------------------------------------------------+
```

### Core Components

#### 1. Base Index Interface

```python
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import numpy as np
from dataclasses import dataclass
from enum import Enum

class DistanceMetric(Enum):
    L2 = "l2"                    # Euclidean distance
    IP = "inner_product"         # Inner product (cosine with normalized)
    COSINE = "cosine"            # Cosine similarity

@dataclass
class SearchResult:
    """Result of k-NN search."""
    ids: np.ndarray      # [n_queries, k] - vector IDs
    distances: np.ndarray # [n_queries, k] - distances

@dataclass
class IndexConfig:
    """Configuration for index building."""
    dim: int
    metric: DistanceMetric = DistanceMetric.L2
    # HNSW params
    M: int = 16                  # Max connections per node
    ef_construction: int = 200   # Search width during construction
    # IVF params
    nlist: int = 100             # Number of clusters
    nprobe: int = 10             # Clusters to search
    # PQ params
    m_pq: int = 8                # Number of subquantizers
    nbits: int = 8               # Bits per subquantizer

class VectorIndex(ABC):
    """Abstract base class for vector indices."""

    def __init__(self, config: IndexConfig):
        self.config = config
        self.dim = config.dim
        self.metric = config.metric
        self.is_trained = False
        self.ntotal = 0

    @abstractmethod
    def train(self, vectors: np.ndarray) -> None:
        """Train the index (e.g., learn centroids for IVF)."""
        pass

    @abstractmethod
    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """Add vectors to the index."""
        pass

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        """Search for k nearest neighbors."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass

    def remove(self, ids: np.ndarray) -> None:
        """Remove vectors by ID (optional)."""
        raise NotImplementedError("This index does not support removal")
```

#### 2. HNSW Index

Hierarchical Navigable Small World graph for fast approximate search.

```python
import heapq
import random
from typing import Set, Dict
import threading

class HNSWIndex(VectorIndex):
    """
    Hierarchical Navigable Small World graph index.

    Properties:
    - O(log N) search complexity
    - High recall (>95% typical)
    - Supports incremental updates
    """

    def __init__(self, config: IndexConfig):
        super().__init__(config)

        self.M = config.M                    # Max connections per layer
        self.M0 = 2 * config.M               # Max connections at layer 0
        self.ef_construction = config.ef_construction
        self.ef_search = 50                   # Search beam width

        # Multi-level probability
        self.ml = 1.0 / np.log(self.M)

        # Graph structure: level -> node_id -> neighbors
        self.graphs: List[Dict[int, List[int]]] = []
        self.max_level = -1
        self.entry_point = -1

        # Vector storage
        self.vectors: List[np.ndarray] = []
        self.id_to_idx: Dict[int, int] = {}
        self.idx_to_id: Dict[int, int] = {}

        # Thread safety
        self._lock = threading.RLock()

    def train(self, vectors: np.ndarray) -> None:
        """HNSW doesn't require training."""
        self.is_trained = True

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """Add vectors to the HNSW graph."""
        if not self.is_trained:
            self.train(vectors)

        n = len(vectors)
        if ids is None:
            ids = np.arange(self.ntotal, self.ntotal + n)

        with self._lock:
            for i in range(n):
                self._insert_one(vectors[i], ids[i])

    def _insert_one(self, vector: np.ndarray, vec_id: int) -> None:
        """Insert a single vector into the graph."""
        # Assign internal index
        idx = len(self.vectors)
        self.vectors.append(vector.copy())
        self.id_to_idx[vec_id] = idx
        self.idx_to_id[idx] = vec_id

        # Determine level for this node
        level = int(-np.log(random.random()) * self.ml)

        # Ensure we have enough graph levels
        while len(self.graphs) <= level:
            self.graphs.append({})

        # Initialize adjacency lists
        for l in range(level + 1):
            self.graphs[l][idx] = []

        if self.entry_point == -1:
            # First node
            self.entry_point = idx
            self.max_level = level
            self.ntotal = 1
            return

        # Search for entry point at each level
        curr_node = self.entry_point
        curr_dist = self._distance(vector, self.vectors[curr_node])

        # Traverse from top to insertion level
        for l in range(self.max_level, level, -1):
            changed = True
            while changed:
                changed = False
                if curr_node in self.graphs[l]:
                    for neighbor in self.graphs[l][curr_node]:
                        dist = self._distance(vector, self.vectors[neighbor])
                        if dist < curr_dist:
                            curr_dist = dist
                            curr_node = neighbor
                            changed = True

        # Insert at each level from insertion level down to 0
        for l in range(min(level, self.max_level), -1, -1):
            # Search layer for nearest neighbors
            candidates = self._search_layer(vector, curr_node, self.ef_construction, l)

            # Select M neighbors
            max_conn = self.M if l > 0 else self.M0
            neighbors = self._select_neighbors(vector, candidates, max_conn)

            # Add bidirectional edges
            self.graphs[l][idx] = neighbors
            for neighbor in neighbors:
                self.graphs[l][neighbor].append(idx)
                # Prune if too many connections
                if len(self.graphs[l][neighbor]) > max_conn:
                    self.graphs[l][neighbor] = self._select_neighbors(
                        self.vectors[neighbor],
                        [(self._distance(self.vectors[neighbor], self.vectors[n]), n)
                         for n in self.graphs[l][neighbor]],
                        max_conn
                    )

            if candidates:
                curr_node = candidates[0][1]

        # Update entry point if new node is at higher level
        if level > self.max_level:
            self.entry_point = idx
            self.max_level = level

        self.ntotal += 1

    def _search_layer(self,
                      query: np.ndarray,
                      entry: int,
                      ef: int,
                      level: int) -> List[Tuple[float, int]]:
        """Search a single layer of the graph."""
        visited = {entry}
        candidates = [(self._distance(query, self.vectors[entry]), entry)]
        results = [(-candidates[0][0], entry)]  # Max-heap for results

        heapq.heapify(candidates)

        while candidates:
            curr_dist, curr_node = heapq.heappop(candidates)

            # Stop if current is worse than worst result
            if -results[0][0] < curr_dist and len(results) >= ef:
                break

            # Explore neighbors
            if curr_node in self.graphs[level]:
                for neighbor in self.graphs[level][curr_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist = self._distance(query, self.vectors[neighbor])

                        if len(results) < ef or dist < -results[0][0]:
                            heapq.heappush(candidates, (dist, neighbor))
                            heapq.heappush(results, (-dist, neighbor))
                            if len(results) > ef:
                                heapq.heappop(results)

        return [(−d, n) for d, n in results]

    def _select_neighbors(self,
                          query: np.ndarray,
                          candidates: List[Tuple[float, int]],
                          M: int) -> List[int]:
        """Select M best neighbors using simple heuristic."""
        # Sort by distance
        candidates = sorted(candidates, key=lambda x: x[0])
        return [c[1] for c in candidates[:M]]

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        """Search for k nearest neighbors."""
        n_queries = len(queries)
        ids = np.zeros((n_queries, k), dtype=np.int64)
        distances = np.full((n_queries, k), np.inf)

        for i in range(n_queries):
            results = self._search_one(queries[i], k)
            for j, (dist, idx) in enumerate(results[:k]):
                ids[i, j] = self.idx_to_id.get(idx, -1)
                distances[i, j] = dist

        return SearchResult(ids=ids, distances=distances)

    def _search_one(self, query: np.ndarray, k: int) -> List[Tuple[float, int]]:
        """Search for k nearest neighbors of a single query."""
        if self.entry_point == -1:
            return []

        # Start from entry point
        curr_node = self.entry_point
        curr_dist = self._distance(query, self.vectors[curr_node])

        # Traverse from top to level 1
        for l in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                if curr_node in self.graphs[l]:
                    for neighbor in self.graphs[l][curr_node]:
                        dist = self._distance(query, self.vectors[neighbor])
                        if dist < curr_dist:
                            curr_dist = dist
                            curr_node = neighbor
                            changed = True

        # Search layer 0 with ef
        candidates = self._search_layer(query, curr_node, max(k, self.ef_search), 0)
        return sorted(candidates)[:k]

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between vectors."""
        if self.metric == DistanceMetric.L2:
            return np.sum((a - b) ** 2)
        elif self.metric == DistanceMetric.IP:
            return -np.dot(a, b)
        else:  # Cosine
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def save(self, path: str) -> None:
        """Save index to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'vectors': self.vectors,
                'graphs': self.graphs,
                'max_level': self.max_level,
                'entry_point': self.entry_point,
                'id_to_idx': self.id_to_idx,
                'idx_to_id': self.idx_to_id,
                'ntotal': self.ntotal
            }, f)

    def load(self, path: str) -> None:
        """Load index from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.config = data['config']
            self.vectors = data['vectors']
            self.graphs = data['graphs']
            self.max_level = data['max_level']
            self.entry_point = data['entry_point']
            self.id_to_idx = data['id_to_idx']
            self.idx_to_id = data['idx_to_id']
            self.ntotal = data['ntotal']
            self.is_trained = True
```

#### 3. IVF Index

Inverted file index with clustering for coarse quantization.

```python
class IVFIndex(VectorIndex):
    """
    Inverted File Index with k-means clustering.

    Properties:
    - Requires training on representative data
    - Good for very large datasets
    - Configurable recall/speed via nprobe
    """

    def __init__(self, config: IndexConfig):
        super().__init__(config)

        self.nlist = config.nlist
        self.nprobe = config.nprobe

        # Centroids learned during training
        self.centroids: Optional[np.ndarray] = None  # [nlist, dim]

        # Inverted lists: cluster_id -> (vectors, ids)
        self.invlists: List[Tuple[List[np.ndarray], List[int]]] = []

    def train(self, vectors: np.ndarray) -> None:
        """Train centroids using k-means."""
        n = len(vectors)
        k = min(self.nlist, n)

        # Initialize centroids (k-means++)
        centroids = self._kmeans_plusplus_init(vectors, k)

        # K-means iterations
        for iteration in range(100):
            # Assign vectors to nearest centroid
            assignments = self._assign_to_centroids(vectors, centroids)

            # Recompute centroids
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k)

            for i, cluster_id in enumerate(assignments):
                new_centroids[cluster_id] += vectors[i]
                counts[cluster_id] += 1

            # Avoid division by zero
            counts = np.maximum(counts, 1)
            new_centroids /= counts[:, np.newaxis]

            # Check convergence
            diff = np.abs(new_centroids - centroids).sum()
            centroids = new_centroids

            if diff < 1e-6:
                break

        self.centroids = centroids
        self.invlists = [([], []) for _ in range(k)]
        self.is_trained = True

    def _kmeans_plusplus_init(self,
                               vectors: np.ndarray,
                               k: int) -> np.ndarray:
        """K-means++ initialization."""
        n = len(vectors)
        centroids = [vectors[np.random.randint(n)]]

        for _ in range(1, k):
            # Compute distances to nearest centroid
            dists = np.array([
                min(self._distance_batch(v.reshape(1, -1), np.array(centroids))[0])
                for v in vectors
            ])

            # Sample proportional to squared distance
            probs = dists ** 2
            probs /= probs.sum()
            idx = np.random.choice(n, p=probs)
            centroids.append(vectors[idx])

        return np.array(centroids)

    def _assign_to_centroids(self,
                              vectors: np.ndarray,
                              centroids: np.ndarray) -> np.ndarray:
        """Assign each vector to nearest centroid."""
        # Compute all distances
        dists = self._distance_batch(vectors, centroids)
        return np.argmin(dists, axis=1)

    def _distance_batch(self,
                        queries: np.ndarray,
                        targets: np.ndarray) -> np.ndarray:
        """Compute pairwise distances."""
        if self.metric == DistanceMetric.L2:
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            queries_sq = np.sum(queries ** 2, axis=1, keepdims=True)
            targets_sq = np.sum(targets ** 2, axis=1)
            cross = queries @ targets.T
            return queries_sq + targets_sq - 2 * cross
        elif self.metric == DistanceMetric.IP:
            return -queries @ targets.T
        else:  # Cosine
            queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
            targets_norm = targets / np.linalg.norm(targets, axis=1, keepdims=True)
            return 1 - queries_norm @ targets_norm.T

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """Add vectors to inverted lists."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors")

        n = len(vectors)
        if ids is None:
            ids = np.arange(self.ntotal, self.ntotal + n)

        # Assign to clusters
        assignments = self._assign_to_centroids(vectors, self.centroids)

        # Add to inverted lists
        for i in range(n):
            cluster_id = assignments[i]
            self.invlists[cluster_id][0].append(vectors[i])
            self.invlists[cluster_id][1].append(ids[i])

        self.ntotal += n

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        """Search using inverted file."""
        n_queries = len(queries)
        all_ids = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.full((n_queries, k), np.inf)

        for i in range(n_queries):
            query = queries[i]

            # Find nearest centroids
            centroid_dists = self._distance_batch(
                query.reshape(1, -1), self.centroids
            )[0]
            probe_clusters = np.argsort(centroid_dists)[:self.nprobe]

            # Search in selected clusters
            candidates = []
            for cluster_id in probe_clusters:
                vectors, ids = self.invlists[cluster_id]
                if not vectors:
                    continue

                vectors_arr = np.array(vectors)
                dists = self._distance_batch(
                    query.reshape(1, -1), vectors_arr
                )[0]

                for j, (d, vid) in enumerate(zip(dists, ids)):
                    candidates.append((d, vid))

            # Select top-k
            candidates.sort()
            for j, (dist, vid) in enumerate(candidates[:k]):
                all_ids[i, j] = vid
                all_distances[i, j] = dist

        return SearchResult(ids=all_ids, distances=all_distances)

    def save(self, path: str) -> None:
        """Save index to disk."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'centroids': self.centroids,
                'invlists': self.invlists,
                'ntotal': self.ntotal
            }, f)

    def load(self, path: str) -> None:
        """Load index from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.config = data['config']
            self.centroids = data['centroids']
            self.invlists = data['invlists']
            self.ntotal = data['ntotal']
            self.is_trained = True
```

#### 4. Product Quantization

Compress vectors into compact codes for memory-efficient search.

```python
class ProductQuantizer:
    """
    Product Quantization for vector compression.

    Splits vectors into m subspaces and quantizes each independently.
    """

    def __init__(self, dim: int, m: int = 8, nbits: int = 8):
        """
        Args:
            dim: Vector dimension
            m: Number of subquantizers
            nbits: Bits per subquantizer (k = 2^nbits centroids)
        """
        assert dim % m == 0, f"dim ({dim}) must be divisible by m ({m})"

        self.dim = dim
        self.m = m
        self.nbits = nbits
        self.k = 2 ** nbits  # Centroids per subquantizer
        self.dsub = dim // m  # Subvector dimension

        # Codebooks: [m, k, dsub]
        self.codebooks: Optional[np.ndarray] = None

    def train(self, vectors: np.ndarray) -> None:
        """Train codebooks using k-means."""
        n = len(vectors)
        self.codebooks = np.zeros((self.m, self.k, self.dsub), dtype=np.float32)

        for i in range(self.m):
            # Extract subvectors
            start = i * self.dsub
            end = start + self.dsub
            subvectors = vectors[:, start:end]

            # K-means clustering
            centroids = self._train_subquantizer(subvectors)
            self.codebooks[i] = centroids

    def _train_subquantizer(self, subvectors: np.ndarray) -> np.ndarray:
        """Train one subquantizer with k-means."""
        n = len(subvectors)
        k = min(self.k, n)

        # Random initialization
        idx = np.random.choice(n, k, replace=False)
        centroids = subvectors[idx].copy()

        for _ in range(50):  # K-means iterations
            # Assign
            dists = np.sum(
                (subvectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
                axis=2
            )
            assignments = np.argmin(dists, axis=1)

            # Update
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k)
            for i, cluster_id in enumerate(assignments):
                new_centroids[cluster_id] += subvectors[i]
                counts[cluster_id] += 1

            counts = np.maximum(counts, 1)
            new_centroids /= counts[:, np.newaxis]
            centroids = new_centroids

        return centroids

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors into PQ codes."""
        n = len(vectors)
        codes = np.zeros((n, self.m), dtype=np.uint8)

        for i in range(self.m):
            start = i * self.dsub
            end = start + self.dsub
            subvectors = vectors[:, start:end]

            # Find nearest centroid
            dists = np.sum(
                (subvectors[:, np.newaxis, :] - self.codebooks[i][np.newaxis, :, :]) ** 2,
                axis=2
            )
            codes[:, i] = np.argmin(dists, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode PQ codes back to approximate vectors."""
        n = len(codes)
        vectors = np.zeros((n, self.dim), dtype=np.float32)

        for i in range(self.m):
            start = i * self.dsub
            end = start + self.dsub
            vectors[:, start:end] = self.codebooks[i][codes[:, i]]

        return vectors

    def compute_distance_tables(self, query: np.ndarray) -> np.ndarray:
        """
        Precompute distance tables for fast search.

        Returns: [m, k] table where table[i, j] = distance from
                 query subvector i to centroid j
        """
        tables = np.zeros((self.m, self.k), dtype=np.float32)

        for i in range(self.m):
            start = i * self.dsub
            end = start + self.dsub
            subquery = query[start:end]

            # Distance to all centroids in this subspace
            tables[i] = np.sum(
                (subquery - self.codebooks[i]) ** 2,
                axis=1
            )

        return tables

    def asymmetric_distance(self,
                             query: np.ndarray,
                             codes: np.ndarray) -> np.ndarray:
        """
        Compute asymmetric distances from query to encoded vectors.
        Uses ADC (Asymmetric Distance Computation) for accuracy.
        """
        tables = self.compute_distance_tables(query)

        # Sum up distances from tables
        n = len(codes)
        distances = np.zeros(n, dtype=np.float32)

        for i in range(self.m):
            distances += tables[i, codes[:, i]]

        return distances


class IVFPQIndex(VectorIndex):
    """Combined IVF + PQ index for billion-scale search."""

    def __init__(self, config: IndexConfig):
        super().__init__(config)

        self.nlist = config.nlist
        self.nprobe = config.nprobe

        # Coarse quantizer (IVF)
        self.coarse_centroids: Optional[np.ndarray] = None

        # Fine quantizer (PQ)
        self.pq = ProductQuantizer(config.dim, config.m_pq, config.nbits)

        # Inverted lists store PQ codes instead of full vectors
        self.invlists: List[Tuple[np.ndarray, np.ndarray]] = []

    def train(self, vectors: np.ndarray) -> None:
        """Train both coarse and fine quantizers."""
        # Train coarse quantizer
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.nlist, random_state=42)
        kmeans.fit(vectors)
        self.coarse_centroids = kmeans.cluster_centers_

        # Compute residuals for PQ training
        assignments = kmeans.predict(vectors)
        residuals = vectors - self.coarse_centroids[assignments]

        # Train PQ on residuals
        self.pq.train(residuals)

        self.invlists = [(np.array([]), np.array([])) for _ in range(self.nlist)]
        self.is_trained = True

    def add(self, vectors: np.ndarray, ids: Optional[np.ndarray] = None) -> None:
        """Add vectors (stored as PQ codes)."""
        if not self.is_trained:
            raise RuntimeError("Index must be trained first")

        n = len(vectors)
        if ids is None:
            ids = np.arange(self.ntotal, self.ntotal + n)

        # Assign to coarse clusters
        dists = np.sum(
            (vectors[:, np.newaxis, :] - self.coarse_centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )
        assignments = np.argmin(dists, axis=1)

        # Compute residuals and encode
        residuals = vectors - self.coarse_centroids[assignments]
        codes = self.pq.encode(residuals)

        # Add to inverted lists
        for i in range(n):
            cluster_id = assignments[i]
            existing_codes, existing_ids = self.invlists[cluster_id]

            if len(existing_codes) == 0:
                new_codes = codes[i:i+1]
                new_ids = np.array([ids[i]])
            else:
                new_codes = np.vstack([existing_codes, codes[i:i+1]])
                new_ids = np.append(existing_ids, ids[i])

            self.invlists[cluster_id] = (new_codes, new_ids)

        self.ntotal += n

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        """Search using IVF + PQ."""
        n_queries = len(queries)
        all_ids = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.full((n_queries, k), np.inf)

        for i in range(n_queries):
            query = queries[i]

            # Find nearest coarse centroids
            coarse_dists = np.sum(
                (query - self.coarse_centroids) ** 2,
                axis=1
            )
            probe_clusters = np.argsort(coarse_dists)[:self.nprobe]

            # Search in selected clusters
            candidates = []
            for cluster_id in probe_clusters:
                codes, ids = self.invlists[cluster_id]
                if len(codes) == 0:
                    continue

                # Compute residual
                residual = query - self.coarse_centroids[cluster_id]

                # Use ADC for distance computation
                dists = self.pq.asymmetric_distance(residual, codes)

                for d, vid in zip(dists, ids):
                    candidates.append((d, vid))

            # Select top-k
            candidates.sort()
            for j, (dist, vid) in enumerate(candidates[:k]):
                all_ids[i, j] = vid
                all_distances[i, j] = dist

        return SearchResult(ids=all_ids, distances=all_distances)

    def save(self, path: str) -> None:
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'coarse_centroids': self.coarse_centroids,
                'pq_codebooks': self.pq.codebooks,
                'invlists': self.invlists,
                'ntotal': self.ntotal
            }, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.config = data['config']
            self.coarse_centroids = data['coarse_centroids']
            self.pq.codebooks = data['pq_codebooks']
            self.invlists = data['invlists']
            self.ntotal = data['ntotal']
            self.is_trained = True
```

#### 5. SIMD Optimized Distance Computation

```python
import numpy as np
from typing import Optional

# Check for SIMD availability
try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(parallel=True, fastmath=True)
    def l2_distance_simd(queries: np.ndarray,
                         database: np.ndarray) -> np.ndarray:
        """
        Compute L2 distances with SIMD optimization via Numba.

        Args:
            queries: [n_queries, dim]
            database: [n_database, dim]

        Returns:
            [n_queries, n_database] distance matrix
        """
        n_queries = queries.shape[0]
        n_database = database.shape[0]
        dim = queries.shape[1]

        result = np.empty((n_queries, n_database), dtype=np.float32)

        for i in prange(n_queries):
            for j in range(n_database):
                dist = 0.0
                for d in range(dim):
                    diff = queries[i, d] - database[j, d]
                    dist += diff * diff
                result[i, j] = dist

        return result

    @njit(parallel=True, fastmath=True)
    def inner_product_simd(queries: np.ndarray,
                           database: np.ndarray) -> np.ndarray:
        """Compute inner products with SIMD optimization."""
        n_queries = queries.shape[0]
        n_database = database.shape[0]
        dim = queries.shape[1]

        result = np.empty((n_queries, n_database), dtype=np.float32)

        for i in prange(n_queries):
            for j in range(n_database):
                dot = 0.0
                for d in range(dim):
                    dot += queries[i, d] * database[j, d]
                result[i, j] = -dot  # Negative for distance

        return result

else:
    def l2_distance_simd(queries, database):
        """Fallback without SIMD."""
        queries_sq = np.sum(queries ** 2, axis=1, keepdims=True)
        database_sq = np.sum(database ** 2, axis=1)
        cross = queries @ database.T
        return queries_sq + database_sq - 2 * cross

    def inner_product_simd(queries, database):
        return -queries @ database.T


class SIMDDistanceComputer:
    """Distance computation with automatic SIMD selection."""

    def __init__(self, metric: DistanceMetric):
        self.metric = metric

    def compute(self,
                queries: np.ndarray,
                database: np.ndarray) -> np.ndarray:
        """Compute pairwise distances."""
        # Ensure contiguous float32
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        database = np.ascontiguousarray(database, dtype=np.float32)

        if self.metric == DistanceMetric.L2:
            return l2_distance_simd(queries, database)
        elif self.metric == DistanceMetric.IP:
            return inner_product_simd(queries, database)
        else:  # Cosine
            queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)
            database_norm = database / np.linalg.norm(database, axis=1, keepdims=True)
            return inner_product_simd(queries_norm, database_norm) + 1
```

### Enterprise Features

#### Multi-Index Search with Metadata Filtering

```python
from typing import Dict, Any, Callable
import json

class FilteredVectorIndex:
    """Vector index with metadata filtering support."""

    def __init__(self, base_index: VectorIndex):
        self.index = base_index
        self.metadata: Dict[int, Dict[str, Any]] = {}

    def add_with_metadata(self,
                          vectors: np.ndarray,
                          ids: np.ndarray,
                          metadata: List[Dict[str, Any]]) -> None:
        """Add vectors with associated metadata."""
        self.index.add(vectors, ids)

        for vec_id, meta in zip(ids, metadata):
            self.metadata[int(vec_id)] = meta

    def search_with_filter(self,
                           queries: np.ndarray,
                           k: int,
                           filter_fn: Callable[[Dict], bool]) -> SearchResult:
        """Search with metadata filtering."""
        # Over-fetch to account for filtering
        fetch_k = min(k * 10, self.index.ntotal)

        results = self.index.search(queries, fetch_k)

        # Filter results
        n_queries = len(queries)
        filtered_ids = np.zeros((n_queries, k), dtype=np.int64)
        filtered_dists = np.full((n_queries, k), np.inf)

        for i in range(n_queries):
            count = 0
            for j in range(fetch_k):
                vec_id = results.ids[i, j]
                if vec_id == -1:
                    continue

                meta = self.metadata.get(int(vec_id), {})
                if filter_fn(meta):
                    filtered_ids[i, count] = vec_id
                    filtered_dists[i, count] = results.distances[i, j]
                    count += 1
                    if count >= k:
                        break

        return SearchResult(ids=filtered_ids, distances=filtered_dists)


class MultiIndexManager:
    """Manage multiple indices for sharding and routing."""

    def __init__(self):
        self.indices: Dict[str, VectorIndex] = {}
        self.routing_metadata: Dict[str, Dict] = {}

    def create_index(self,
                     name: str,
                     config: IndexConfig,
                     index_type: str = 'hnsw') -> None:
        """Create a new index."""
        if index_type == 'hnsw':
            index = HNSWIndex(config)
        elif index_type == 'ivf':
            index = IVFIndex(config)
        elif index_type == 'ivfpq':
            index = IVFPQIndex(config)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        self.indices[name] = index
        self.routing_metadata[name] = {'type': index_type}

    def search_all(self,
                   queries: np.ndarray,
                   k: int,
                   index_names: Optional[List[str]] = None) -> SearchResult:
        """Search across multiple indices and merge results."""
        if index_names is None:
            index_names = list(self.indices.keys())

        all_results = []
        for name in index_names:
            results = self.indices[name].search(queries, k)
            all_results.append(results)

        # Merge and re-rank
        return self._merge_results(all_results, k)

    def _merge_results(self,
                       results: List[SearchResult],
                       k: int) -> SearchResult:
        """Merge results from multiple indices."""
        n_queries = results[0].ids.shape[0]

        merged_ids = np.zeros((n_queries, k), dtype=np.int64)
        merged_dists = np.full((n_queries, k), np.inf)

        for i in range(n_queries):
            # Gather all candidates
            candidates = []
            for result in results:
                for j in range(result.ids.shape[1]):
                    if result.distances[i, j] < np.inf:
                        candidates.append((
                            result.distances[i, j],
                            result.ids[i, j]
                        ))

            # Sort and select top-k
            candidates.sort()
            for j, (dist, vec_id) in enumerate(candidates[:k]):
                merged_ids[i, j] = vec_id
                merged_dists[i, j] = dist

        return SearchResult(ids=merged_ids, distances=merged_dists)


class MMapIndex:
    """Memory-mapped index for on-disk storage."""

    def __init__(self, path: str, config: IndexConfig):
        self.path = path
        self.config = config

    def create(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Create memory-mapped index file."""
        n, dim = vectors.shape

        # Create mmap file
        fp = np.memmap(
            self.path,
            dtype=np.float32,
            mode='w+',
            shape=(n, dim)
        )
        fp[:] = vectors
        fp.flush()

        # Save metadata
        meta_path = self.path + '.meta'
        with open(meta_path, 'w') as f:
            json.dump({
                'n': n,
                'dim': dim,
                'ids': ids.tolist()
            }, f)

    def search(self, queries: np.ndarray, k: int) -> SearchResult:
        """Search in memory-mapped index."""
        # Load metadata
        with open(self.path + '.meta', 'r') as f:
            meta = json.load(f)

        n, dim = meta['n'], meta['dim']
        ids = np.array(meta['ids'])

        # Memory-map vectors
        vectors = np.memmap(
            self.path,
            dtype=np.float32,
            mode='r',
            shape=(n, dim)
        )

        # Compute distances
        dists = l2_distance_simd(
            queries.astype(np.float32),
            np.array(vectors, dtype=np.float32)
        )

        # Select top-k
        n_queries = len(queries)
        result_ids = np.zeros((n_queries, k), dtype=np.int64)
        result_dists = np.zeros((n_queries, k), dtype=np.float32)

        for i in range(n_queries):
            top_k_idx = np.argsort(dists[i])[:k]
            result_ids[i] = ids[top_k_idx]
            result_dists[i] = dists[i, top_k_idx]

        return SearchResult(ids=result_ids, distances=result_dists)
```

## API Reference

### Index Creation

```python
# Create HNSW index
config = IndexConfig(dim=128, metric=DistanceMetric.L2, M=32)
index = HNSWIndex(config)

# Create IVF-PQ index for large scale
config = IndexConfig(dim=128, nlist=1000, nprobe=50, m_pq=16)
index = IVFPQIndex(config)
```

### Training and Adding

```python
# Train on sample data
train_data = np.random.randn(10000, 128).astype(np.float32)
index.train(train_data)

# Add vectors
vectors = np.random.randn(100000, 128).astype(np.float32)
ids = np.arange(100000)
index.add(vectors, ids)
```

### Searching

```python
# Basic search
queries = np.random.randn(10, 128).astype(np.float32)
results = index.search(queries, k=10)

print(results.ids)       # [10, 10] array of neighbor IDs
print(results.distances)  # [10, 10] array of distances

# Search with filtering
filtered_index = FilteredVectorIndex(index)
results = filtered_index.search_with_filter(
    queries, k=10,
    filter_fn=lambda m: m.get('category') == 'document'
)
```

### Persistence

```python
# Save index
index.save('/path/to/index.bin')

# Load index
index = HNSWIndex(config)
index.load('/path/to/index.bin')
```

## Implementation Phases

### Phase 1: Core Flat Index (Week 1)
- Base index interface
- Flat (brute-force) index
- Distance metrics (L2, IP, cosine)
- Basic persistence

### Phase 2: HNSW Index (Weeks 2-3)
- Graph construction algorithm
- Multi-layer structure
- Search algorithm
- Incremental updates

### Phase 3: IVF Index (Week 4)
- K-means training
- Inverted file structure
- Multi-probe search
- Cluster assignment

### Phase 4: Product Quantization (Weeks 5-6)
- PQ training
- Encoding/decoding
- ADC distance computation
- IVF+PQ combination

### Phase 5: SIMD Optimization (Weeks 7-8)
- Numba/JIT compilation
- AVX-512 intrinsics
- Batched operations
- Memory layout optimization

### Phase 6: GPU Acceleration (Weeks 9-10)
- CUDA distance kernels
- GPU index search
- Hybrid CPU/GPU search

### Phase 7: Enterprise Features (Weeks 11-14)
- Metadata filtering
- Multi-index management
- Memory-mapped indices
- Auto-training

## Testing Strategy

### Unit Tests

```python
class TestHNSWIndex:
    def test_add_and_search(self):
        config = IndexConfig(dim=32, M=16)
        index = HNSWIndex(config)

        vectors = np.random.randn(100, 32).astype(np.float32)
        index.train(vectors)
        index.add(vectors)

        # Query should return itself as nearest
        results = index.search(vectors[:1], k=1)
        assert results.ids[0, 0] == 0

    def test_recall(self):
        config = IndexConfig(dim=64, M=32)
        index = HNSWIndex(config)

        # Add vectors
        vectors = np.random.randn(1000, 64).astype(np.float32)
        index.train(vectors)
        index.add(vectors)

        # Compute recall@10
        queries = vectors[:100]
        results = index.search(queries, k=10)

        # Brute-force ground truth
        dists = l2_distance_simd(queries, vectors)
        gt = np.argsort(dists, axis=1)[:, :10]

        recall = 0
        for i in range(100):
            recall += len(set(results.ids[i]) & set(gt[i]))
        recall /= 1000

        assert recall > 0.95  # Should achieve >95% recall


class TestProductQuantization:
    def test_encode_decode(self):
        pq = ProductQuantizer(dim=128, m=8, nbits=8)

        vectors = np.random.randn(1000, 128).astype(np.float32)
        pq.train(vectors)

        codes = pq.encode(vectors)
        decoded = pq.decode(codes)

        # Check reconstruction error
        error = np.mean((vectors - decoded) ** 2)
        assert error < 0.5  # Reasonable reconstruction

    def test_adc_distance(self):
        pq = ProductQuantizer(dim=64, m=8, nbits=8)

        vectors = np.random.randn(100, 64).astype(np.float32)
        pq.train(vectors)
        codes = pq.encode(vectors)

        query = vectors[0]
        adc_dists = pq.asymmetric_distance(query, codes)

        # ADC distance to itself should be small
        assert adc_dists[0] < 0.1
```

### Performance Benchmarks

```python
class TestPerformance:
    def test_hnsw_throughput(self):
        """Benchmark HNSW search QPS."""
        config = IndexConfig(dim=128, M=32)
        index = HNSWIndex(config)

        vectors = np.random.randn(100000, 128).astype(np.float32)
        index.train(vectors)
        index.add(vectors)

        queries = np.random.randn(1000, 128).astype(np.float32)

        import time
        start = time.time()
        for _ in range(10):
            index.search(queries, k=10)
        elapsed = time.time() - start

        qps = 10000 / elapsed
        print(f"HNSW QPS: {qps:.0f}")
        assert qps > 1000  # Target >1000 QPS

    def test_ivfpq_memory(self):
        """Verify IVF-PQ memory efficiency."""
        config = IndexConfig(dim=128, nlist=100, m_pq=16, nbits=8)
        index = IVFPQIndex(config)

        vectors = np.random.randn(100000, 128).astype(np.float32)
        index.train(vectors)
        index.add(vectors)

        # PQ codes should be 16 bytes per vector
        # vs 512 bytes for float32
        # ~32x compression
        pass
```

## Stretch Goals

### Multi-Modal Vectors

```python
class MultiModalIndex:
    """Index supporting multiple vector modalities."""

    def __init__(self, modality_dims: Dict[str, int]):
        self.modalities = modality_dims
        self.indices = {
            name: HNSWIndex(IndexConfig(dim=dim))
            for name, dim in modality_dims.items()
        }

    def add(self, vectors: Dict[str, np.ndarray], ids: np.ndarray):
        for name, vecs in vectors.items():
            self.indices[name].add(vecs, ids)

    def search(self,
               queries: Dict[str, np.ndarray],
               k: int,
               weights: Dict[str, float]) -> SearchResult:
        """Weighted multi-modal search."""
        # Search each modality
        results = {}
        for name, query in queries.items():
            results[name] = self.indices[name].search(query, k * 3)

        # Weighted score fusion
        # ...
        pass
```

### Dynamic Updates

```python
class DynamicHNSWIndex(HNSWIndex):
    """HNSW with efficient delete support."""

    def __init__(self, config: IndexConfig):
        super().__init__(config)
        self.deleted: Set[int] = set()

    def remove(self, ids: np.ndarray) -> None:
        """Mark vectors as deleted (lazy deletion)."""
        for vec_id in ids:
            self.deleted.add(int(vec_id))

    def compact(self) -> None:
        """Rebuild index without deleted vectors."""
        # Rebuild graph excluding deleted nodes
        pass
```

## Performance Targets

| Metric | Target | Index Type |
|--------|--------|------------|
| Search QPS | >10,000 | HNSW, 1M vectors |
| Recall@10 | >95% | HNSW |
| Memory | <100 bytes/vector | IVF-PQ |
| Build time | <1 hour | 1B vectors |
| Index size | <50GB | 1B vectors @ 128d |

## Dependencies

- NumPy
- (Optional) Numba for SIMD
- (Optional) scikit-learn for k-means
- (Optional) CUDA toolkit for GPU

## References

- FAISS: https://github.com/facebookresearch/faiss
- Efficient and robust approximate nearest neighbor search using HNSW graphs
- Product Quantization for Nearest Neighbor Search
- Billion-scale similarity search with GPUs
