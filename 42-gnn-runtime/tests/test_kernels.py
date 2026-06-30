"""Tests for sparse operations and kernels."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnn_runtime.kernels import SparseOpsNumpy

# Try to import torch-dependent classes
try:
    import torch
    from gnn_runtime.kernels import (
        scatter_add,
        scatter_mean,
        scatter_max,
        spmm,
        spmm_coo,
        SparseOps,
        FusedOps,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestSparseOpsNumpy:
    """Tests for NumPy sparse operations."""

    def test_spmm_csr(self):
        """Test SpMM with CSR format."""
        rowptr = np.array([0, 2, 3, 4])
        col = np.array([0, 1, 1, 0])
        value = np.array([1.0, 2.0, 1.0, 1.0])
        mat = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        out = SparseOpsNumpy.spmm_csr(rowptr, col, value, mat)

        assert out.shape == (3, 2)
        # Row 0: 1*[1,0] + 2*[0,1] = [1, 2]
        np.testing.assert_array_almost_equal(out[0], [1.0, 2.0])

    def test_spmm_csr_no_values(self):
        """Test SpMM without edge values."""
        rowptr = np.array([0, 2, 3, 4])
        col = np.array([0, 1, 1, 0])
        mat = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        out = SparseOpsNumpy.spmm_csr(rowptr, col, None, mat)

        assert out.shape == (3, 2)

    def test_scatter_add_np(self):
        """Test scatter add with NumPy."""
        src = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        index = np.array([0, 1, 0])

        out = SparseOpsNumpy.scatter_add_np(src, index, dim_size=2)

        assert out.shape == (2, 2)
        np.testing.assert_array_almost_equal(out[0], [6.0, 8.0])  # 1+5, 2+6
        np.testing.assert_array_almost_equal(out[1], [3.0, 4.0])

    def test_edge_softmax_np(self):
        """Test edge softmax with NumPy."""
        edge_score = np.array([1.0, 2.0, 1.0])
        dst = np.array([0, 0, 1])

        out = SparseOpsNumpy.edge_softmax_np(edge_score, dst, num_nodes=2)

        assert len(out) == 3
        # Softmax over edges to node 0
        assert abs(out[0] + out[1] - 1.0) < 1e-5
        # Edge to node 1 should be 1.0
        assert abs(out[2] - 1.0) < 1e-5


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestScatterOperations:
    """Tests for PyTorch scatter operations."""

    def test_scatter_add(self):
        """Test scatter add."""
        src = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        index = torch.tensor([0, 1, 0])

        out = scatter_add(src, index, dim_size=2)

        assert out.shape == (2, 2)
        torch.testing.assert_close(out[0], torch.tensor([6.0, 8.0]))
        torch.testing.assert_close(out[1], torch.tensor([3.0, 4.0]))

    def test_scatter_mean(self):
        """Test scatter mean."""
        src = torch.tensor([
            [2.0, 4.0],
            [4.0, 6.0],
            [6.0, 8.0],
        ])
        index = torch.tensor([0, 0, 1])

        out = scatter_mean(src, index, dim_size=2)

        assert out.shape == (2, 2)
        torch.testing.assert_close(out[0], torch.tensor([3.0, 5.0]))  # Mean of [2,4] and [4,6]
        torch.testing.assert_close(out[1], torch.tensor([6.0, 8.0]))

    def test_scatter_max(self):
        """Test scatter max."""
        src = torch.tensor([
            [1.0, 3.0],
            [2.0, 1.0],
            [0.0, 2.0],
        ])
        index = torch.tensor([0, 0, 0])

        out, _ = scatter_max(src, index, dim_size=1)

        assert out.shape == (1, 2)
        torch.testing.assert_close(out[0], torch.tensor([2.0, 3.0]))


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestSpMM:
    """Tests for SpMM operations."""

    def test_spmm_basic(self):
        """Test basic SpMM."""
        rowptr = torch.tensor([0, 2, 3, 4])
        col = torch.tensor([0, 1, 1, 0])
        value = torch.tensor([1.0, 2.0, 1.0, 1.0])
        mat = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])

        out = spmm(rowptr, col, value, mat)

        assert out.shape == (3, 2)
        torch.testing.assert_close(out[0], torch.tensor([1.0, 2.0]))

    def test_spmm_coo(self):
        """Test SpMM with COO format."""
        src = torch.tensor([0, 0, 1, 2])
        dst = torch.tensor([0, 1, 1, 0])
        value = torch.tensor([1.0, 2.0, 1.0, 1.0])
        mat = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        out = spmm_coo(src, dst, value, mat, num_rows=2)

        assert out.shape == (2, 2)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestSparseOps:
    """Tests for SparseOps class."""

    def test_segment_csr_sum(self):
        """Test segment reduction with sum."""
        src = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        indptr = torch.tensor([0, 2, 3])

        out = SparseOps.segment_csr(src, indptr, reduce='sum')

        assert out.shape == (2, 2)
        torch.testing.assert_close(out[0], torch.tensor([4.0, 6.0]))
        torch.testing.assert_close(out[1], torch.tensor([5.0, 6.0]))

    def test_segment_csr_mean(self):
        """Test segment reduction with mean."""
        src = torch.tensor([
            [2.0, 4.0],
            [4.0, 6.0],
            [6.0, 8.0],
        ])
        indptr = torch.tensor([0, 2, 3])

        out = SparseOps.segment_csr(src, indptr, reduce='mean')

        assert out.shape == (2, 2)
        torch.testing.assert_close(out[0], torch.tensor([3.0, 5.0]))

    def test_csr_to_coo(self):
        """Test CSR to COO conversion."""
        rowptr = torch.tensor([0, 2, 3, 4])
        col = torch.tensor([0, 1, 1, 0])

        row, col_out = SparseOps.csr_to_coo(rowptr, col)

        assert len(row) == len(col)
        torch.testing.assert_close(row, torch.tensor([0, 0, 1, 2]))

    def test_coo_to_csr(self):
        """Test COO to CSR conversion."""
        row = torch.tensor([0, 0, 1, 2])
        col = torch.tensor([0, 1, 1, 0])

        rowptr, col_sorted = SparseOps.coo_to_csr(row, col, num_rows=3)

        assert len(rowptr) == 4
        assert rowptr[0] == 0
        assert rowptr[-1] == 4

    def test_edge_softmax(self):
        """Test edge softmax."""
        edge_score = torch.tensor([1.0, 2.0, 1.0])
        edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])

        out = SparseOps.edge_softmax(edge_score, edge_index, num_nodes=2)

        assert len(out) == 3
        # Softmax over edges to node 0 (indices 0, 1)
        assert abs(out[0].item() + out[1].item() - 1.0) < 1e-5


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestFusedOps:
    """Tests for fused operations."""

    def test_gather_scatter_add(self):
        """Test fused gather and scatter-add."""
        x = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        edge_index = torch.tensor([[0, 1], [1, 2]])

        out = FusedOps.gather_scatter_add(x, edge_index)

        assert out.shape == (3, 2)
        torch.testing.assert_close(out[1], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(out[2], torch.tensor([0.0, 1.0]))

    def test_gather_scatter_with_weight(self):
        """Test gather-scatter with edge weights."""
        x = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        weight = torch.tensor([0.5, 2.0])

        out = FusedOps.gather_scatter_add(x, edge_index, weight)

        assert out.shape == (2, 2)
        torch.testing.assert_close(out[0], torch.tensor([0.0, 2.0]))
        torch.testing.assert_close(out[1], torch.tensor([0.5, 0.0]))

    def test_fused_gcn_aggregate(self):
        """Test fused GCN aggregation."""
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        norm = torch.ones(4)

        out = FusedOps.fused_gcn_aggregate(x, edge_index, norm)

        assert out.shape == (5, 8)

    def test_fused_gat_aggregate(self):
        """Test fused GAT aggregation."""
        x = torch.randn(5, 8)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        alpha = torch.softmax(torch.randn(4), dim=0)

        out = FusedOps.fused_gat_aggregate(x, edge_index, alpha)

        assert out.shape == (5, 8)
