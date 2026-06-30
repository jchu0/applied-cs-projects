"""Optimized sparse operations for GNN computation."""

from typing import Optional, Tuple
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


def scatter_add(
    src: 'torch.Tensor',
    index: 'torch.Tensor',
    dim_size: int,
    dim: int = 0,
) -> 'torch.Tensor':
    """
    Scatter-add operation.

    Args:
        src: Source tensor [E, F].
        index: Index tensor [E].
        dim_size: Size of output dimension.
        dim: Dimension to scatter into.

    Returns:
        Output tensor [dim_size, F].
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for scatter operations")

    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
    return out


def scatter_mean(
    src: 'torch.Tensor',
    index: 'torch.Tensor',
    dim_size: int,
    dim: int = 0,
) -> 'torch.Tensor':
    """
    Scatter-mean operation.

    Args:
        src: Source tensor [E, F].
        index: Index tensor [E].
        dim_size: Size of output dimension.
        dim: Dimension to scatter into.

    Returns:
        Output tensor [dim_size, F].
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for scatter operations")

    out = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)

    # Count elements per index
    ones = torch.ones(index.size(0), device=src.device)
    count = torch.zeros(dim_size, device=src.device)
    count.scatter_add_(0, index, ones)
    count = count.clamp(min=1)

    return out / count.unsqueeze(-1)


def scatter_max(
    src: 'torch.Tensor',
    index: 'torch.Tensor',
    dim_size: int,
    dim: int = 0,
) -> Tuple['torch.Tensor', 'torch.Tensor']:
    """
    Scatter-max operation.

    Args:
        src: Source tensor [E, F].
        index: Index tensor [E].
        dim_size: Size of output dimension.
        dim: Dimension to scatter into.

    Returns:
        Tuple of (max values [dim_size, F], argmax indices [dim_size, F]).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for scatter operations")

    out = torch.full((dim_size, src.size(-1)), float('-inf'),
                    device=src.device, dtype=src.dtype)
    argmax = torch.zeros(dim_size, src.size(-1), device=src.device, dtype=torch.long)

    # Use scatter_reduce for max
    out.scatter_reduce_(
        dim,
        index.unsqueeze(-1).expand_as(src),
        src,
        reduce='amax',
    )
    out[out == float('-inf')] = 0

    return out, argmax


def spmm(
    rowptr: 'torch.Tensor',
    col: 'torch.Tensor',
    value: Optional['torch.Tensor'],
    mat: 'torch.Tensor',
) -> 'torch.Tensor':
    """
    Sparse matrix-matrix multiplication.

    Computes: out = sparse(rowptr, col, value) @ mat

    Args:
        rowptr: CSR row pointers [num_rows + 1].
        col: CSR column indices [nnz].
        value: Optional edge weights [nnz].
        mat: Dense matrix [num_cols, feat_dim].

    Returns:
        Result matrix [num_rows, feat_dim].
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for spmm")

    num_rows = rowptr.size(0) - 1
    feat_dim = mat.size(1)
    out = torch.zeros(num_rows, feat_dim, device=mat.device, dtype=mat.dtype)

    for row in range(num_rows):
        start = rowptr[row].item()
        end = rowptr[row + 1].item()

        if start == end:
            continue

        cols = col[start:end]
        vals = value[start:end] if value is not None else torch.ones(end - start, device=mat.device)

        # out[row] = sum(val * mat[c] for c, val in zip(cols, vals))
        out[row] = (mat[cols] * vals.unsqueeze(-1)).sum(dim=0)

    return out


def spmm_coo(
    src: 'torch.Tensor',
    dst: 'torch.Tensor',
    value: Optional['torch.Tensor'],
    mat: 'torch.Tensor',
    num_rows: int,
) -> 'torch.Tensor':
    """
    SpMM using COO format.

    Args:
        src: Source indices [nnz].
        dst: Destination indices [nnz].
        value: Optional edge weights [nnz].
        mat: Dense matrix [num_cols, feat_dim].
        num_rows: Number of output rows.

    Returns:
        Result matrix [num_rows, feat_dim].
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for spmm")

    feat_dim = mat.size(1)
    out = torch.zeros(num_rows, feat_dim, device=mat.device, dtype=mat.dtype)

    # Gather source features
    messages = mat[src]  # [nnz, feat_dim]

    if value is not None:
        messages = messages * value.unsqueeze(-1)

    # Scatter to destinations
    out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

    return out


class SparseOps:
    """
    Collection of optimized sparse operations.

    Provides both PyTorch implementations and numpy fallbacks.
    """

    @staticmethod
    def segment_csr(
        src: 'torch.Tensor',
        indptr: 'torch.Tensor',
        reduce: str = 'sum',
    ) -> 'torch.Tensor':
        """
        Segment reduction using CSR format.

        Args:
            src: Source tensor [nnz, F].
            indptr: Segment pointers [num_segments + 1].
            reduce: Reduction type ('sum', 'mean', 'max', 'min').

        Returns:
            Reduced tensor [num_segments, F].
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        num_segments = indptr.size(0) - 1
        out = torch.zeros(num_segments, src.size(-1),
                         device=src.device, dtype=src.dtype)

        for i in range(num_segments):
            start = indptr[i].item()
            end = indptr[i + 1].item()

            if start == end:
                continue

            segment = src[start:end]

            if reduce == 'sum':
                out[i] = segment.sum(dim=0)
            elif reduce == 'mean':
                out[i] = segment.mean(dim=0)
            elif reduce == 'max':
                out[i] = segment.max(dim=0)[0]
            elif reduce == 'min':
                out[i] = segment.min(dim=0)[0]

        return out

    @staticmethod
    def csr_to_coo(
        rowptr: 'torch.Tensor',
        col: 'torch.Tensor',
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Convert CSR to COO format."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        num_rows = rowptr.size(0) - 1
        row = torch.zeros_like(col)

        for i in range(num_rows):
            start = rowptr[i].item()
            end = rowptr[i + 1].item()
            row[start:end] = i

        return row, col

    @staticmethod
    def coo_to_csr(
        row: 'torch.Tensor',
        col: 'torch.Tensor',
        num_rows: int,
    ) -> Tuple['torch.Tensor', 'torch.Tensor']:
        """Convert COO to CSR format."""
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        # Count edges per row
        rowptr = torch.zeros(num_rows + 1, dtype=torch.long, device=row.device)
        for r in row:
            rowptr[r + 1] += 1
        rowptr = rowptr.cumsum(dim=0)

        # Sort by row
        sort_idx = torch.argsort(row)
        col_sorted = col[sort_idx]

        return rowptr, col_sorted

    @staticmethod
    def edge_softmax(
        edge_score: 'torch.Tensor',
        edge_index: 'torch.Tensor',
        num_nodes: int,
    ) -> 'torch.Tensor':
        """
        Compute softmax over edges grouped by destination node.

        Args:
            edge_score: Edge scores [E].
            edge_index: Edge indices [2, E].
            num_nodes: Number of nodes.

        Returns:
            Softmax weights [E].
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        dst = edge_index[1]

        # Max for numerical stability
        score_max = torch.zeros(num_nodes, device=edge_score.device, dtype=edge_score.dtype)
        score_max.fill_(float('-inf'))
        score_max.scatter_reduce_(0, dst, edge_score, reduce='amax')
        score_max[score_max == float('-inf')] = 0

        # Subtract max and exp
        score = (edge_score - score_max[dst]).exp()

        # Sum per destination
        score_sum = torch.zeros(num_nodes, device=edge_score.device, dtype=edge_score.dtype)
        score_sum.scatter_add_(0, dst, score)

        return score / (score_sum[dst] + 1e-16)


class SparseOpsNumpy:
    """NumPy implementations of sparse operations for CPU-only execution."""

    @staticmethod
    def spmm_csr(
        rowptr: np.ndarray,
        col: np.ndarray,
        value: Optional[np.ndarray],
        mat: np.ndarray,
    ) -> np.ndarray:
        """SpMM with CSR format using NumPy."""
        num_rows = len(rowptr) - 1
        feat_dim = mat.shape[1]
        out = np.zeros((num_rows, feat_dim), dtype=mat.dtype)

        for row in range(num_rows):
            start = rowptr[row]
            end = rowptr[row + 1]

            if start == end:
                continue

            cols = col[start:end]
            if value is not None:
                vals = value[start:end]
                out[row] = (mat[cols] * vals[:, np.newaxis]).sum(axis=0)
            else:
                out[row] = mat[cols].sum(axis=0)

        return out

    @staticmethod
    def scatter_add_np(
        src: np.ndarray,
        index: np.ndarray,
        dim_size: int,
    ) -> np.ndarray:
        """Scatter-add using NumPy."""
        feat_dim = src.shape[1] if src.ndim > 1 else 1
        out = np.zeros((dim_size, feat_dim), dtype=src.dtype)

        for i, idx in enumerate(index):
            if src.ndim > 1:
                out[idx] += src[i]
            else:
                out[idx, 0] += src[i]

        return out if src.ndim > 1 else out.squeeze(-1)

    @staticmethod
    def edge_softmax_np(
        edge_score: np.ndarray,
        dst: np.ndarray,
        num_nodes: int,
    ) -> np.ndarray:
        """Edge softmax using NumPy."""
        # Compute max per destination
        score_max = np.full(num_nodes, float('-inf'))
        np.maximum.at(score_max, dst, edge_score)
        score_max[score_max == float('-inf')] = 0

        # Exp of (score - max)
        score = np.exp(edge_score - score_max[dst])

        # Sum per destination
        score_sum = np.zeros(num_nodes)
        np.add.at(score_sum, dst, score)

        return score / (score_sum[dst] + 1e-16)


class FusedOps:
    """
    Fused operations for better performance.

    Combines multiple operations to reduce memory bandwidth.
    """

    @staticmethod
    def gather_scatter_add(
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
        weight: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        """
        Fused gather from source nodes and scatter-add to destinations.

        This is the core operation for GNN message passing.

        Args:
            x: Node features [N, F].
            edge_index: Edge indices [2, E].
            weight: Optional edge weights [E].

        Returns:
            Aggregated features [N, F].
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required")

        src, dst = edge_index
        num_nodes = x.size(0)

        # Gather
        messages = x[src]

        # Weight if provided
        if weight is not None:
            messages = messages * weight.unsqueeze(-1)

        # Scatter-add
        out = torch.zeros(num_nodes, x.size(-1), device=x.device, dtype=x.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        return out

    @staticmethod
    def fused_gcn_aggregate(
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
        norm: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Fused GCN aggregation with normalization.

        Computes: out = D^{-1/2} A D^{-1/2} x

        Args:
            x: Node features [N, F].
            edge_index: Edge indices [2, E].
            norm: Pre-computed normalization [E].

        Returns:
            Normalized aggregated features [N, F].
        """
        return FusedOps.gather_scatter_add(x, edge_index, norm)

    @staticmethod
    def fused_gat_aggregate(
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
        alpha: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Fused GAT aggregation with attention weights.

        Args:
            x: Node features [N, F].
            edge_index: Edge indices [2, E].
            alpha: Attention weights [E].

        Returns:
            Attention-weighted aggregated features [N, F].
        """
        return FusedOps.gather_scatter_add(x, edge_index, alpha)
