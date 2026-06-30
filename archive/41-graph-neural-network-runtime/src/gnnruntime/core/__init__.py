"""Core graph structures for GNN runtime."""

from .graph import (
    EdgeIndex,
    Graph,
    MessagePassing,
    SparseTensor,
    normalize_adj,
    gcn_norm,
    compute_attention_weights,
)

__all__ = [
    "EdgeIndex",
    "Graph",
    "MessagePassing",
    "SparseTensor",
    "normalize_adj",
    "gcn_norm",
    "compute_attention_weights",
]
