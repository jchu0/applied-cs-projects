"""Tests for message passing module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnn_runtime.message_passing import MessagePassingNumpy

# Try to import torch-dependent classes
try:
    import torch
    from gnn_runtime.message_passing import (
        MessagePassingEngine,
        CopyMessage,
        ConcatMessage,
        SumAggregate,
        MeanAggregate,
        MaxAggregate,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestMessagePassingNumpy:
    """Tests for NumPy-based message passing."""

    def test_propagate_sum(self):
        """Test sum aggregation with NumPy."""
        edge_index = np.array([[0, 1, 1], [1, 0, 2]])
        node_features = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])

        out = MessagePassingNumpy.propagate_sum(edge_index, node_features)

        assert out.shape == (3, 2)
        # Node 0 receives from node 1
        np.testing.assert_array_almost_equal(out[0], [0.0, 1.0])
        # Node 1 receives from node 0
        np.testing.assert_array_almost_equal(out[1], [1.0, 0.0])
        # Node 2 receives from node 1
        np.testing.assert_array_almost_equal(out[2], [0.0, 1.0])

    def test_propagate_mean(self):
        """Test mean aggregation with NumPy."""
        edge_index = np.array([[0, 1, 2], [1, 1, 1]])
        node_features = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ])

        out = MessagePassingNumpy.propagate_mean(edge_index, node_features)

        assert out.shape == (3, 2)
        # Node 1 receives from 0, 1, 2 with mean
        expected_mean = (node_features[0] + node_features[1] + node_features[2]) / 3
        np.testing.assert_array_almost_equal(out[1], expected_mean)

    def test_propagate_with_flow(self):
        """Test different flow directions."""
        edge_index = np.array([[0, 1], [1, 2]])
        node_features = np.array([
            [1.0],
            [2.0],
            [3.0],
        ])

        # Source to target
        out_s2t = MessagePassingNumpy.propagate_sum(
            edge_index, node_features, flow='source_to_target'
        )

        # Target to source
        out_t2s = MessagePassingNumpy.propagate_sum(
            edge_index, node_features, flow='target_to_source'
        )

        assert out_s2t.shape == out_t2s.shape


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestMessagePassingEngine:
    """Tests for PyTorch message passing engine."""

    def test_sum_aggregation(self):
        """Test sum aggregation."""
        engine = MessagePassingEngine(device='cpu')
        edge_index = torch.tensor([[0, 1], [1, 2]])
        features = torch.randn(3, 16)

        out = engine.propagate(edge_index, features, aggregate_fn='sum')
        assert out.shape == (3, 16)

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        engine = MessagePassingEngine(device='cpu')
        edge_index = torch.tensor([[0, 1, 2], [1, 1, 1]])
        features = torch.randn(3, 8)

        out = engine.propagate(edge_index, features, aggregate_fn='mean')
        assert out.shape == (3, 8)

    def test_max_aggregation(self):
        """Test max aggregation."""
        engine = MessagePassingEngine(device='cpu')
        edge_index = torch.tensor([[0, 1], [1, 2]])
        features = torch.randn(3, 4)

        out = engine.propagate(edge_index, features, aggregate_fn='max')
        assert out.shape == (3, 4)

    def test_custom_message_fn(self):
        """Test custom message function."""
        engine = MessagePassingEngine(device='cpu')
        edge_index = torch.tensor([[0, 1], [1, 2]])
        features = torch.randn(3, 8)

        def message_fn(src, dst, edge_feat):
            return src + dst  # Sum of source and destination

        out = engine.propagate(edge_index, features, message_fn=message_fn)
        assert out.shape == (3, 8)

    def test_flow_direction(self):
        """Test different flow directions."""
        engine = MessagePassingEngine(device='cpu')
        edge_index = torch.tensor([[0], [1]])
        features = torch.tensor([[1.0], [2.0]])

        out_s2t = engine.propagate(
            edge_index, features, flow='source_to_target'
        )
        out_t2s = engine.propagate(
            edge_index, features, flow='target_to_source'
        )

        # Different results for different flows
        assert out_s2t.shape == out_t2s.shape

    def test_attention_propagation(self):
        """Test attention-weighted message passing."""
        engine = MessagePassingEngine(device='cpu')
        edge_index = torch.tensor([[0, 1, 2], [1, 1, 1]])
        features = torch.randn(3, 8)
        attention = torch.softmax(torch.randn(3), dim=0)

        out = engine.propagate_with_attention(
            edge_index, features, attention
        )
        assert out.shape == (3, 8)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestMessageFunctions:
    """Tests for message function classes."""

    def test_copy_message(self):
        """Test CopyMessage function."""
        msg_fn = CopyMessage()
        src = torch.randn(10, 8)
        dst = torch.randn(10, 8)

        out = msg_fn(src, dst)
        torch.testing.assert_close(out, src)

    def test_concat_message(self):
        """Test ConcatMessage function."""
        msg_fn = ConcatMessage()
        src = torch.randn(10, 8)
        dst = torch.randn(10, 8)

        out = msg_fn(src, dst)
        assert out.shape == (10, 16)

    def test_concat_with_edge_features(self):
        """Test ConcatMessage with edge features."""
        msg_fn = ConcatMessage()
        src = torch.randn(10, 8)
        dst = torch.randn(10, 8)
        edge = torch.randn(10, 4)

        out = msg_fn(src, dst, edge)
        assert out.shape == (10, 20)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestAggregateFunctions:
    """Tests for aggregate function classes."""

    def test_sum_aggregate(self):
        """Test SumAggregate function."""
        agg_fn = SumAggregate()
        messages = torch.randn(5, 8)
        index = torch.tensor([0, 0, 1, 1, 1])

        out = agg_fn(messages, index, num_nodes=2)
        assert out.shape == (2, 8)

    def test_mean_aggregate(self):
        """Test MeanAggregate function."""
        agg_fn = MeanAggregate()
        messages = torch.ones(4, 2)
        index = torch.tensor([0, 0, 1, 1])

        out = agg_fn(messages, index, num_nodes=2)
        torch.testing.assert_close(out, torch.ones(2, 2))

    def test_max_aggregate(self):
        """Test MaxAggregate function."""
        agg_fn = MaxAggregate()
        messages = torch.tensor([
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 4.0],
        ])
        index = torch.tensor([0, 0, 0])

        out = agg_fn(messages, index, num_nodes=1)
        expected = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(out, expected)
