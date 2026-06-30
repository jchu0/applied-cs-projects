"""Tests for GNN layer implementations."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try to import torch-dependent classes
try:
    import torch
    import torch.nn as nn
    from gnn_runtime.layers import (
        GNNLayer,
        GCNLayer,
        GATLayer,
        GraphSAGELayer,
        GNNModel,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def create_simple_graph():
    """Create a simple graph for testing."""
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn(3, 16)
    return x, edge_index


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestGNNLayer:
    """Tests for generic GNNLayer."""

    def test_gcn_message_type(self):
        """Test GNN layer with GCN message type."""
        x, edge_index = create_simple_graph()
        layer = GNNLayer(16, 8, message_type='gcn')

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_gat_message_type(self):
        """Test GNN layer with GAT message type."""
        x, edge_index = create_simple_graph()
        layer = GNNLayer(16, 8, message_type='gat')

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_sum_aggregation(self):
        """Test sum aggregation."""
        x, edge_index = create_simple_graph()
        layer = GNNLayer(16, 8, aggregate_type='sum')

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        x, edge_index = create_simple_graph()
        layer = GNNLayer(16, 8, aggregate_type='mean')

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_without_bias(self):
        """Test layer without bias."""
        x, edge_index = create_simple_graph()
        layer = GNNLayer(16, 8, bias=False)

        assert layer.bias is None
        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_without_normalization(self):
        """Test layer without normalization."""
        x, edge_index = create_simple_graph()
        layer = GNNLayer(16, 8, normalize=False)

        out = layer(x, edge_index)
        assert out.shape == (3, 8)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestGCNLayer:
    """Tests for GCNLayer."""

    def test_forward(self):
        """Test GCN forward pass."""
        x, edge_index = create_simple_graph()
        layer = GCNLayer(16, 8)

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_with_edge_weight(self):
        """Test GCN with edge weights."""
        x, edge_index = create_simple_graph()
        edge_weight = torch.ones(edge_index.size(1))
        layer = GCNLayer(16, 8)

        out = layer(x, edge_index, edge_weight)
        assert out.shape == (3, 8)

    def test_improved_gcn(self):
        """Test improved GCN normalization."""
        x, edge_index = create_simple_graph()
        layer = GCNLayer(16, 8, improved=True)

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_cached_norm(self):
        """Test cached normalization."""
        x, edge_index = create_simple_graph()
        layer = GCNLayer(16, 8, cached=True)

        # First call computes and caches norm
        out1 = layer(x, edge_index)
        # Second call uses cached norm
        out2 = layer(x, edge_index)

        assert out1.shape == out2.shape

    def test_gradient_flow(self):
        """Test gradient computation."""
        x, edge_index = create_simple_graph()
        x.requires_grad = True
        layer = GCNLayer(16, 8)

        out = layer(x, edge_index)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestGATLayer:
    """Tests for GATLayer."""

    def test_single_head(self):
        """Test single-head attention."""
        x, edge_index = create_simple_graph()
        layer = GATLayer(16, 8, heads=1)

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_multi_head_concat(self):
        """Test multi-head attention with concatenation."""
        x, edge_index = create_simple_graph()
        layer = GATLayer(16, 8, heads=4, concat=True)

        out = layer(x, edge_index)
        assert out.shape == (3, 32)  # 8 * 4 heads

    def test_multi_head_mean(self):
        """Test multi-head attention with mean."""
        x, edge_index = create_simple_graph()
        layer = GATLayer(16, 8, heads=4, concat=False)

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_with_dropout(self):
        """Test GAT with dropout."""
        x, edge_index = create_simple_graph()
        layer = GATLayer(16, 8, dropout=0.5)

        layer.train()
        out_train = layer(x, edge_index)

        layer.eval()
        out_eval = layer(x, edge_index)

        assert out_train.shape == out_eval.shape

    def test_negative_slope(self):
        """Test custom negative slope for LeakyReLU."""
        x, edge_index = create_simple_graph()
        layer = GATLayer(16, 8, negative_slope=0.1)

        out = layer(x, edge_index)
        assert out.shape == (3, 8)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestGraphSAGELayer:
    """Tests for GraphSAGELayer."""

    def test_mean_aggregator(self):
        """Test mean aggregator."""
        x, edge_index = create_simple_graph()
        layer = GraphSAGELayer(16, 8, aggregator='mean')

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_max_aggregator(self):
        """Test max aggregator."""
        x, edge_index = create_simple_graph()
        layer = GraphSAGELayer(16, 8, aggregator='max')

        out = layer(x, edge_index)
        assert out.shape == (3, 8)

    def test_with_normalization(self):
        """Test L2 normalization."""
        x, edge_index = create_simple_graph()
        layer = GraphSAGELayer(16, 8, normalize=True)

        out = layer(x, edge_index)
        # Check L2 norm is approximately 1
        norms = torch.norm(out, dim=1)
        torch.testing.assert_close(norms, torch.ones(3), atol=1e-5, rtol=1e-5)

    def test_without_root_weight(self):
        """Test without root node transformation."""
        x, edge_index = create_simple_graph()
        layer = GraphSAGELayer(16, 8, root_weight=False)

        out = layer(x, edge_index)
        assert out.shape == (3, 8)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestGNNModel:
    """Tests for multi-layer GNN model."""

    def test_gcn_model(self):
        """Test GCN model."""
        x, edge_index = create_simple_graph()
        model = GNNModel(16, 32, 10, num_layers=2, layer_type='gcn')

        out = model(x, edge_index)
        assert out.shape == (3, 10)

    def test_gat_model(self):
        """Test GAT model."""
        x, edge_index = create_simple_graph()
        model = GNNModel(16, 32, 10, num_layers=2, layer_type='gat')

        out = model(x, edge_index)
        assert out.shape == (3, 10)

    def test_sage_model(self):
        """Test GraphSAGE model."""
        x, edge_index = create_simple_graph()
        model = GNNModel(16, 32, 10, num_layers=2, layer_type='sage')

        out = model(x, edge_index)
        assert out.shape == (3, 10)

    def test_deep_model(self):
        """Test deep GNN model."""
        x, edge_index = create_simple_graph()
        model = GNNModel(16, 32, 10, num_layers=5, layer_type='gcn')

        out = model(x, edge_index)
        assert out.shape == (3, 10)

    def test_with_dropout(self):
        """Test model with dropout."""
        x, edge_index = create_simple_graph()
        model = GNNModel(16, 32, 10, num_layers=2, dropout=0.5)

        model.train()
        out_train = model(x, edge_index)

        model.eval()
        out_eval = model(x, edge_index)

        assert out_train.shape == out_eval.shape

    def test_jumping_knowledge_cat(self):
        """Test jumping knowledge with concatenation."""
        x, edge_index = create_simple_graph()
        model = GNNModel(16, 32, 10, num_layers=2, jk='cat')

        out = model(x, edge_index)
        assert out.shape == (3, 10)

    def test_jumping_knowledge_max(self):
        """Test jumping knowledge with max pooling."""
        x, edge_index = create_simple_graph()
        model = GNNModel(16, 32, 10, num_layers=2, jk='max')

        out = model(x, edge_index)
        assert out.shape == (3, 10)

    def test_encode(self):
        """Test getting node embeddings."""
        x, edge_index = create_simple_graph()
        model = GNNModel(16, 32, 10, num_layers=2)

        embeddings = model.encode(x, edge_index)
        assert embeddings.shape == (3, 32)

    def test_training_loop(self):
        """Test training loop."""
        x, edge_index = create_simple_graph()
        y = torch.randint(0, 10, (3,))

        model = GNNModel(16, 32, 10, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for _ in range(3):
            optimizer.zero_grad()
            out = model(x, edge_index)
            loss = nn.functional.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

        assert loss.item() > 0
