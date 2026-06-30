"""SIMD-optimized vector operations using Numba.

Provides JIT-compiled, vectorized implementations of distance computations
and search operations for improved performance on large vector datasets.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import numba for SIMD optimizations
try:
    from numba import jit, prange, float32, float64, int64
    from numba import vectorize, guvectorize
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    logger.warning("Numba not available, falling back to numpy implementations")

SIMD_AVAILABLE = NUMBA_AVAILABLE


def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """Ensure array is C-contiguous for optimal SIMD access."""
    if not arr.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(arr)
    return arr


def _ensure_aligned(arr: np.ndarray, alignment: int = 32) -> np.ndarray:
    """Ensure array is memory-aligned for SIMD operations."""
    if arr.ctypes.data % alignment != 0:
        # Create aligned copy
        aligned = np.empty(arr.shape, dtype=arr.dtype)
        aligned[:] = arr
        return aligned
    return arr


if NUMBA_AVAILABLE:
    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _l2_distance_simd(query: np.ndarray, database: np.ndarray) -> np.ndarray:
        """
        SIMD-optimized L2 distance computation.

        Uses parallel processing and fast math for vectorized operations.
        """
        nq, d = query.shape
        nb = database.shape[0]
        distances = np.empty((nq, nb), dtype=np.float32)

        for i in prange(nq):
            for j in prange(nb):
                dist = 0.0
                for k in range(d):
                    diff = query[i, k] - database[j, k]
                    dist += diff * diff
                distances[i, j] = np.sqrt(dist)

        return distances

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _l2_distance_single_simd(query: np.ndarray, database: np.ndarray) -> np.ndarray:
        """SIMD L2 distance for single query vector."""
        d = query.shape[0]
        nb = database.shape[0]
        distances = np.empty(nb, dtype=np.float32)

        for j in prange(nb):
            dist = 0.0
            for k in range(d):
                diff = query[k] - database[j, k]
                dist += diff * diff
            distances[j] = np.sqrt(dist)

        return distances

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _inner_product_simd(query: np.ndarray, database: np.ndarray) -> np.ndarray:
        """SIMD-optimized inner product computation."""
        nq, d = query.shape
        nb = database.shape[0]
        similarities = np.empty((nq, nb), dtype=np.float32)

        for i in prange(nq):
            for j in prange(nb):
                sim = 0.0
                for k in range(d):
                    sim += query[i, k] * database[j, k]
                similarities[i, j] = sim

        return similarities

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _inner_product_single_simd(query: np.ndarray, database: np.ndarray) -> np.ndarray:
        """SIMD inner product for single query vector."""
        d = query.shape[0]
        nb = database.shape[0]
        similarities = np.empty(nb, dtype=np.float32)

        for j in prange(nb):
            sim = 0.0
            for k in range(d):
                sim += query[k] * database[j, k]
            similarities[j] = sim

        return similarities

    @jit(nopython=True, fastmath=True, cache=True)
    def _normalize_vector(v: np.ndarray) -> np.ndarray:
        """Normalize a single vector."""
        norm = 0.0
        for i in range(len(v)):
            norm += v[i] * v[i]
        norm = np.sqrt(norm) + 1e-8
        result = np.empty_like(v)
        for i in range(len(v)):
            result[i] = v[i] / norm
        return result

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _cosine_similarity_simd(query: np.ndarray, database: np.ndarray) -> np.ndarray:
        """SIMD-optimized cosine similarity computation."""
        nq, d = query.shape
        nb = database.shape[0]
        similarities = np.empty((nq, nb), dtype=np.float32)

        for i in prange(nq):
            # Query norm
            q_norm = 0.0
            for k in range(d):
                q_norm += query[i, k] * query[i, k]
            q_norm = np.sqrt(q_norm) + 1e-8

            for j in prange(nb):
                # Database norm
                db_norm = 0.0
                dot = 0.0
                for k in range(d):
                    dot += query[i, k] * database[j, k]
                    db_norm += database[j, k] * database[j, k]
                db_norm = np.sqrt(db_norm) + 1e-8

                similarities[i, j] = dot / (q_norm * db_norm)

        return similarities

    @jit(nopython=True, fastmath=True, cache=True)
    def _topk_single(distances: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k smallest distances for single query."""
        n = len(distances)
        k = min(k, n)

        # Partial sort using selection
        indices = np.arange(n, dtype=np.int64)

        for i in range(k):
            min_idx = i
            for j in range(i + 1, n):
                if distances[indices[j]] < distances[indices[min_idx]]:
                    min_idx = j
            # Swap
            indices[i], indices[min_idx] = indices[min_idx], indices[i]

        return indices[:k].copy(), distances[indices[:k]].copy()

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _topk_batch(distances: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k smallest distances for batch of queries."""
        nq, n = distances.shape
        k = min(k, n)

        result_indices = np.empty((nq, k), dtype=np.int64)
        result_distances = np.empty((nq, k), dtype=np.float32)

        for q in prange(nq):
            indices = np.arange(n, dtype=np.int64)

            for i in range(k):
                min_idx = i
                for j in range(i + 1, n):
                    if distances[q, indices[j]] < distances[q, indices[min_idx]]:
                        min_idx = j
                indices[i], indices[min_idx] = indices[min_idx], indices[i]

            for i in range(k):
                result_indices[q, i] = indices[i]
                result_distances[q, i] = distances[q, indices[i]]

        return result_indices, result_distances

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _batch_l2_search(
        queries: np.ndarray,
        database: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combined L2 distance + top-k search for batch."""
        nq, d = queries.shape
        nb = database.shape[0]
        k = min(k, nb)

        result_indices = np.empty((nq, k), dtype=np.int64)
        result_distances = np.empty((nq, k), dtype=np.float32)

        for i in prange(nq):
            # Compute distances
            distances = np.empty(nb, dtype=np.float32)
            for j in range(nb):
                dist = 0.0
                for dim in range(d):
                    diff = queries[i, dim] - database[j, dim]
                    dist += diff * diff
                distances[j] = np.sqrt(dist)

            # Top-k selection
            indices = np.arange(nb, dtype=np.int64)
            for ki in range(k):
                min_idx = ki
                for j in range(ki + 1, nb):
                    if distances[indices[j]] < distances[indices[min_idx]]:
                        min_idx = j
                indices[ki], indices[min_idx] = indices[min_idx], indices[ki]

            for ki in range(k):
                result_indices[i, ki] = indices[ki]
                result_distances[i, ki] = distances[indices[ki]]

        return result_indices, result_distances

    @jit(nopython=True, fastmath=True, parallel=True, cache=True)
    def _batch_ip_search(
        queries: np.ndarray,
        database: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Combined inner product + top-k search for batch."""
        nq, d = queries.shape
        nb = database.shape[0]
        k = min(k, nb)

        result_indices = np.empty((nq, k), dtype=np.int64)
        result_similarities = np.empty((nq, k), dtype=np.float32)

        for i in prange(nq):
            # Compute similarities (negative for top-k)
            similarities = np.empty(nb, dtype=np.float32)
            for j in range(nb):
                sim = 0.0
                for dim in range(d):
                    sim += queries[i, dim] * database[j, dim]
                similarities[j] = -sim  # Negate for min selection

            # Top-k selection
            indices = np.arange(nb, dtype=np.int64)
            for ki in range(k):
                min_idx = ki
                for j in range(ki + 1, nb):
                    if similarities[indices[j]] < similarities[indices[min_idx]]:
                        min_idx = j
                indices[ki], indices[min_idx] = indices[min_idx], indices[ki]

            for ki in range(k):
                result_indices[i, ki] = indices[ki]
                result_similarities[i, ki] = -similarities[indices[ki]]

        return result_indices, result_similarities


# Public API functions with fallback to numpy
def simd_l2_distance(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Compute L2 distance using SIMD operations.

    Args:
        query: Query vectors (n, d) or (d,)
        database: Database vectors (m, d)

    Returns:
        Distance matrix (n, m) or (m,)
    """
    query = _ensure_contiguous(query.astype(np.float32))
    database = _ensure_contiguous(database.astype(np.float32))

    if NUMBA_AVAILABLE:
        if query.ndim == 1:
            return _l2_distance_single_simd(query, database)
        else:
            return _l2_distance_simd(query, database)
    else:
        # Fallback to numpy
        if query.ndim == 1:
            query = query.reshape(1, -1)
        a_norm = np.sum(query ** 2, axis=1, keepdims=True)
        b_norm = np.sum(database ** 2, axis=1, keepdims=True).T
        dist = a_norm + b_norm - 2 * np.dot(query, database.T)
        dist = np.maximum(dist, 0)
        return np.sqrt(dist).squeeze()


def simd_inner_product(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Compute inner product using SIMD operations.

    Args:
        query: Query vectors (n, d) or (d,)
        database: Database vectors (m, d)

    Returns:
        Similarity matrix (n, m) or (m,)
    """
    query = _ensure_contiguous(query.astype(np.float32))
    database = _ensure_contiguous(database.astype(np.float32))

    if NUMBA_AVAILABLE:
        if query.ndim == 1:
            return _inner_product_single_simd(query, database)
        else:
            return _inner_product_simd(query, database)
    else:
        if query.ndim == 1:
            query = query.reshape(1, -1)
        return np.dot(query, database.T).squeeze()


def simd_cosine_similarity(query: np.ndarray, database: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity using SIMD operations.

    Args:
        query: Query vectors (n, d) or (d,)
        database: Database vectors (m, d)

    Returns:
        Similarity matrix (n, m) or (m,)
    """
    query = _ensure_contiguous(query.astype(np.float32))
    database = _ensure_contiguous(database.astype(np.float32))

    if query.ndim == 1:
        query = query.reshape(1, -1)

    if NUMBA_AVAILABLE:
        return _cosine_similarity_simd(query, database).squeeze()
    else:
        a_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
        b_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
        return np.dot(a_norm, b_norm.T).squeeze()


def simd_topk(
    distances: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top-k smallest distances using SIMD operations.

    Args:
        distances: Distance values (n,) or (nq, n)
        k: Number of results

    Returns:
        Tuple of (indices, distances)
    """
    distances = _ensure_contiguous(distances.astype(np.float32))

    if NUMBA_AVAILABLE:
        if distances.ndim == 1:
            return _topk_single(distances, k)
        else:
            return _topk_batch(distances, k)
    else:
        # Fallback to numpy
        if distances.ndim == 1:
            k = min(k, len(distances))
            idx = np.argpartition(distances, k-1)[:k]
            idx = idx[np.argsort(distances[idx])]
            return idx.astype(np.int64), distances[idx].astype(np.float32)
        else:
            nq = distances.shape[0]
            k = min(k, distances.shape[1])
            indices = np.zeros((nq, k), dtype=np.int64)
            dists = np.zeros((nq, k), dtype=np.float32)
            for i in range(nq):
                idx = np.argpartition(distances[i], k-1)[:k]
                idx = idx[np.argsort(distances[i, idx])]
                indices[i] = idx
                dists[i] = distances[i, idx]
            return indices, dists


def simd_l2_batch(
    queries: np.ndarray,
    database: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combined L2 distance + top-k search using SIMD.

    More efficient than separate distance and topk calls.

    Args:
        queries: Query vectors (nq, d)
        database: Database vectors (nb, d)
        k: Number of nearest neighbors

    Returns:
        Tuple of (indices, distances)
    """
    queries = _ensure_contiguous(queries.astype(np.float32))
    database = _ensure_contiguous(database.astype(np.float32))

    if queries.ndim == 1:
        queries = queries.reshape(1, -1)

    if NUMBA_AVAILABLE:
        return _batch_l2_search(queries, database, k)
    else:
        distances = simd_l2_distance(queries, database)
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
        return simd_topk(distances, k)


def simd_ip_batch(
    queries: np.ndarray,
    database: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combined inner product + top-k search using SIMD.

    Args:
        queries: Query vectors (nq, d)
        database: Database vectors (nb, d)
        k: Number of nearest neighbors

    Returns:
        Tuple of (indices, similarities)
    """
    queries = _ensure_contiguous(queries.astype(np.float32))
    database = _ensure_contiguous(database.astype(np.float32))

    if queries.ndim == 1:
        queries = queries.reshape(1, -1)

    if NUMBA_AVAILABLE:
        return _batch_ip_search(queries, database, k)
    else:
        similarities = simd_inner_product(queries, database)
        if similarities.ndim == 1:
            similarities = similarities.reshape(1, -1)
        # Negate and use topk
        idx, dists = simd_topk(-similarities, k)
        return idx, -dists


def batch_search(
    queries: np.ndarray,
    database: np.ndarray,
    k: int,
    metric: str = "l2",
    batch_size: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Memory-efficient batch search with chunking.

    Processes queries in batches to manage memory usage for large datasets.

    Args:
        queries: Query vectors (nq, d)
        database: Database vectors (nb, d)
        k: Number of nearest neighbors
        metric: Distance metric ("l2", "ip", "cosine")
        batch_size: Batch size for query processing

    Returns:
        Tuple of (indices, distances/similarities)
    """
    queries = _ensure_contiguous(queries.astype(np.float32))
    database = _ensure_contiguous(database.astype(np.float32))

    if queries.ndim == 1:
        queries = queries.reshape(1, -1)

    nq = queries.shape[0]
    all_indices = []
    all_distances = []

    for start in range(0, nq, batch_size):
        end = min(start + batch_size, nq)
        batch_queries = queries[start:end]

        if metric == "l2":
            indices, distances = simd_l2_batch(batch_queries, database, k)
        elif metric == "ip":
            indices, distances = simd_ip_batch(batch_queries, database, k)
        elif metric == "cosine":
            # Normalize and use inner product
            norms = np.linalg.norm(batch_queries, axis=1, keepdims=True) + 1e-8
            normalized = batch_queries / norms
            db_norms = np.linalg.norm(database, axis=1, keepdims=True) + 1e-8
            normalized_db = database / db_norms
            indices, distances = simd_ip_batch(normalized, normalized_db, k)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        all_indices.append(indices)
        all_distances.append(distances)

    return np.vstack(all_indices), np.vstack(all_distances)


class SIMDVectorOps:
    """
    Class-based interface for SIMD vector operations.

    Provides cached precomputation and optimized search methods.
    """

    def __init__(self, database: np.ndarray, metric: str = "l2"):
        """
        Initialize with database vectors.

        Args:
            database: Database vectors (nb, d)
            metric: Distance metric ("l2", "ip", "cosine")
        """
        self.database = _ensure_contiguous(database.astype(np.float32))
        self.metric = metric
        self._db_norms: Optional[np.ndarray] = None

        # Precompute norms for cosine similarity
        if metric == "cosine":
            self._db_norms = np.linalg.norm(self.database, axis=1, keepdims=True) + 1e-8
            self._normalized_db = self.database / self._db_norms

    @property
    def ntotal(self) -> int:
        """Number of vectors in database."""
        return self.database.shape[0]

    @property
    def dim(self) -> int:
        """Vector dimension."""
        return self.database.shape[1]

    def search(
        self,
        queries: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            queries: Query vectors (nq, d) or (d,)
            k: Number of neighbors

        Returns:
            Tuple of (indices, distances)
        """
        queries = _ensure_contiguous(queries.astype(np.float32))

        if self.metric == "l2":
            return simd_l2_batch(queries, self.database, k)
        elif self.metric == "ip":
            return simd_ip_batch(queries, self.database, k)
        elif self.metric == "cosine":
            if queries.ndim == 1:
                queries = queries.reshape(1, -1)
            q_norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
            normalized_q = queries / q_norms
            return simd_ip_batch(normalized_q, self._normalized_db, k)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def range_search(
        self,
        queries: np.ndarray,
        radius: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for all neighbors within radius.

        Args:
            queries: Query vectors (nq, d) or (d,)
            radius: Search radius

        Returns:
            Tuple of (lims, indices, distances) where lims[i] gives
            the start of results for query i
        """
        queries = _ensure_contiguous(queries.astype(np.float32))
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)

        nq = queries.shape[0]

        if self.metric == "l2":
            distances = simd_l2_distance(queries, self.database)
        elif self.metric == "ip":
            distances = -simd_inner_product(queries, self.database)
        elif self.metric == "cosine":
            q_norms = np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
            normalized_q = queries / q_norms
            distances = -simd_cosine_similarity(normalized_q, self._normalized_db)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        if distances.ndim == 1:
            distances = distances.reshape(1, -1)

        # Collect results within radius
        all_indices = []
        all_distances = []
        lims = [0]

        for i in range(nq):
            mask = distances[i] <= radius
            indices = np.where(mask)[0]
            dists = distances[i, mask]

            # Sort by distance
            order = np.argsort(dists)
            all_indices.extend(indices[order])
            all_distances.extend(dists[order])
            lims.append(len(all_indices))

        return (
            np.array(lims, dtype=np.int64),
            np.array(all_indices, dtype=np.int64),
            np.array(all_distances, dtype=np.float32)
        )

    def add(self, vectors: np.ndarray):
        """Add vectors to the database."""
        vectors = _ensure_contiguous(vectors.astype(np.float32))
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        self.database = np.vstack([self.database, vectors])

        if self.metric == "cosine":
            v_norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
            normalized_v = vectors / v_norms
            self._normalized_db = np.vstack([self._normalized_db, normalized_v])
