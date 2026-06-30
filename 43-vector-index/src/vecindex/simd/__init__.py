"""SIMD-optimized vector operations.

This module provides SIMD-accelerated implementations of common vector
operations using Numba JIT compilation with explicit vectorization.
"""

from .ops import (
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

__all__ = [
    "simd_l2_distance",
    "simd_inner_product",
    "simd_cosine_similarity",
    "simd_topk",
    "simd_l2_batch",
    "simd_ip_batch",
    "batch_search",
    "SIMDVectorOps",
    "SIMD_AVAILABLE",
]
