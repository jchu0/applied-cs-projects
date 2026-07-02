"""VecIndex - FAISS-lite vector similarity search library."""

from .core import (
    MetricType,
    SearchResult,
    VectorStore,
    l2_distance,
    inner_product,
    cosine_similarity,
    normalize_vectors,
    kmeans,
)
from .index import (
    Index,
    FlatIndex,
    IVFIndex,
    HNSWIndex,
    IVFPQIndex,
)
from .quantize import (
    ProductQuantizer,
    OPQ,
    ScalarQuantizer,
    BinaryQuantizer,
    compute_recall,
)
from .search import (
    SearchParams,
    RangeSearcher,
    BatchSearcher,
    HybridSearcher,
    RerankSearcher,
    build_index,
    benchmark_index,
    IndexFactory,
)
from .simd import (
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

__version__ = "0.1.0"

__all__ = [
    # Core
    "MetricType",
    "SearchResult",
    "VectorStore",
    "l2_distance",
    "inner_product",
    "cosine_similarity",
    "normalize_vectors",
    "kmeans",
    # Index
    "Index",
    "FlatIndex",
    "IVFIndex",
    "HNSWIndex",
    "IVFPQIndex",
    # Quantize
    "ProductQuantizer",
    "OPQ",
    "ScalarQuantizer",
    "BinaryQuantizer",
    "compute_recall",
    # Search
    "SearchParams",
    "RangeSearcher",
    "BatchSearcher",
    "HybridSearcher",
    "RerankSearcher",
    "build_index",
    "benchmark_index",
    "IndexFactory",
    # SIMD
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
