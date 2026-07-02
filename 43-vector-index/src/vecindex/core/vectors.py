"""Core vector operations for similarity search."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Distance/similarity metric types."""
    L2 = "l2"                    # Euclidean distance
    IP = "inner_product"         # Inner product (similarity)
    COSINE = "cosine"            # Cosine similarity


@dataclass
class SearchResult:
    """Result of a vector search."""
    ids: np.ndarray       # Vector IDs
    distances: np.ndarray # Distances/similarities

    @property
    def num_results(self) -> int:
        return len(self.ids)


def l2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute L2 (Euclidean) distance between vectors.

    Args:
        a: Query vectors (n, d) or (d,)
        b: Database vectors (m, d)

    Returns:
        Distance matrix (n, m) or (m,)
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)

    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    a_norm = np.sum(a ** 2, axis=1, keepdims=True)
    b_norm = np.sum(b ** 2, axis=1, keepdims=True).T
    dist = a_norm + b_norm - 2 * np.dot(a, b.T)

    # Numerical stability
    dist = np.maximum(dist, 0)

    return np.sqrt(dist).squeeze()


def inner_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute inner product similarity.

    Args:
        a: Query vectors (n, d) or (d,)
        b: Database vectors (m, d)

    Returns:
        Similarity matrix (n, m) or (m,)
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)

    return np.dot(a, b.T).squeeze()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity.

    Args:
        a: Query vectors
        b: Database vectors

    Returns:
        Similarity matrix
    """
    if a.ndim == 1:
        a = a.reshape(1, -1)

    # Normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)

    return np.dot(a_norm, b_norm.T).squeeze()


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2 normalize vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    return vectors / norms


def compute_distance(
    a: np.ndarray,
    b: np.ndarray,
    metric: MetricType
) -> np.ndarray:
    """
    Compute distance/similarity between vectors.

    Args:
        a: Query vectors
        b: Database vectors
        metric: Distance metric

    Returns:
        Distance/similarity values
    """
    if metric == MetricType.L2:
        return l2_distance(a, b)
    elif metric == MetricType.IP:
        # Negate for consistent "lower is better"
        return -inner_product(a, b)
    elif metric == MetricType.COSINE:
        # Negate for consistent "lower is better"
        return -cosine_similarity(a, b)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def topk(
    distances: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get top-k smallest distances.

    Args:
        distances: Distance values (n,) or (nq, n)
        k: Number of results

    Returns:
        Tuple of (indices, distances)
    """
    if distances.ndim == 1:
        k = min(k, len(distances))
        idx = np.argpartition(distances, k-1)[:k]
        idx = idx[np.argsort(distances[idx])]
        return idx, distances[idx]
    else:
        # Batch version
        nq = distances.shape[0]
        k = min(k, distances.shape[1])

        indices = np.zeros((nq, k), dtype=np.int64)
        dists = np.zeros((nq, k))

        for i in range(nq):
            idx = np.argpartition(distances[i], k-1)[:k]
            idx = idx[np.argsort(distances[i, idx])]
            indices[i] = idx
            dists[i] = distances[i, idx]

        return indices, dists


class VectorStore:
    """
    Simple in-memory vector storage.

    Supports add, remove, and retrieval operations.
    """

    def __init__(self, dim: int, dtype: np.dtype = np.float32):
        self.dim = dim
        self.dtype = dtype
        self.vectors = np.empty((0, dim), dtype=dtype)
        self.ids = np.empty(0, dtype=np.int64)
        self._next_id = 0

    @property
    def ntotal(self) -> int:
        """Total number of vectors."""
        return len(self.ids)

    def add(self, vectors: np.ndarray) -> np.ndarray:
        """
        Add vectors to store.

        Args:
            vectors: Vectors to add (n, d)

        Returns:
            Assigned IDs
        """
        vectors = np.asarray(vectors, dtype=self.dtype)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        n = vectors.shape[0]
        new_ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)

        self.vectors = np.vstack([self.vectors, vectors])
        self.ids = np.concatenate([self.ids, new_ids])
        self._next_id += n

        return new_ids

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray):
        """Add vectors with specific IDs."""
        vectors = np.asarray(vectors, dtype=self.dtype)
        ids = np.asarray(ids, dtype=np.int64)

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        self.vectors = np.vstack([self.vectors, vectors])
        self.ids = np.concatenate([self.ids, ids])
        self._next_id = max(self._next_id, ids.max() + 1)

    def remove(self, ids: np.ndarray):
        """Remove vectors by ID."""
        ids = np.asarray(ids)
        mask = ~np.isin(self.ids, ids)
        self.vectors = self.vectors[mask]
        self.ids = self.ids[mask]

    def get(self, ids: np.ndarray) -> np.ndarray:
        """Get vectors by ID."""
        ids = np.asarray(ids)
        idx = np.searchsorted(self.ids, ids)
        return self.vectors[idx]

    def clear(self):
        """Clear all vectors."""
        self.vectors = np.empty((0, self.dim), dtype=self.dtype)
        self.ids = np.empty(0, dtype=np.int64)
        self._next_id = 0


def kmeans(
    vectors: np.ndarray,
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means clustering.

    Args:
        vectors: Input vectors (n, d)
        k: Number of clusters
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Tuple of (centroids, assignments)
    """
    n, d = vectors.shape

    if k > n:
        raise ValueError(
            f"Cannot form {k} clusters from only {n} training vectors "
            f"(need at least k={k} vectors)"
        )

    # Initialize centroids randomly
    idx = np.random.choice(n, k, replace=False)
    centroids = vectors[idx].copy()

    for _ in range(max_iter):
        # Assign to nearest centroid
        distances = l2_distance(vectors, centroids)
        if distances.ndim == 1:
            distances = distances.reshape(1, -1)
        assignments = np.argmin(distances, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = assignments == i
            if mask.sum() > 0:
                new_centroids[i] = vectors[mask].mean(axis=0)
            else:
                new_centroids[i] = centroids[i]

        # Check convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return centroids, assignments


def pca(
    vectors: np.ndarray,
    n_components: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Principal Component Analysis for dimensionality reduction.

    Args:
        vectors: Input vectors (n, d)
        n_components: Number of components

    Returns:
        Tuple of (transformed, components, mean)
    """
    mean = vectors.mean(axis=0)
    centered = vectors - mean

    # Covariance matrix
    cov = np.cov(centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx[:n_components]].T

    # Transform
    transformed = np.dot(centered, components.T)

    return transformed, components, mean
