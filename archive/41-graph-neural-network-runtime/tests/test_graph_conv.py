"""Unit tests for graph convolutional layers."""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnnruntime.layers.conv import (
    GNNLayer, GCNConv, GATConv, GraphSAGEConv,
    EdgeConv, ChebConv, SGConv
)
from gnnruntime.core.graph import EdgeIndex, normalize_adj, gcn_norm


class TestGCNConv(unittest.TestCase):
    """Test Graph Convolutional Network layer."""

    def setUp(self):
        np.random.seed(42)
        self.in_channels = 16
        self.out_channels = 32
        self.num_nodes = 100

    def test_init(self):
        """Test GCN layer initialization."""
        layer = GCNConv(self.in_channels, self.out_channels)

        self.assertEqual(layer.in_channels, self.in_channels)
        self.assertEqual(layer.out_channels, self.out_channels)
        self.assertTrue(layer.add_self_loops)
        self.assertTrue(layer.normalize)

    def test_forward_basic(self):
        """Test basic forward pass."""
        layer = GCNConv(self.in_channels, self.out_channels)

        # Create sample graph
        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes, num_edges=500)

        # Forward pass
        output = layer.forward(x, edge_index)

        # Check output shape
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))

    def test_self_loops(self):
        """Test self-loop addition."""
        layer = GCNConv(self.in_channels, self.out_channels, add_self_loops=True)

        x = np.random.randn(10, self.in_channels).astype(np.float32)
        # Small graph without self-loops
        edge_index = EdgeIndex([[0, 1, 2], [1, 2, 0]])

        output = layer.forward(x, edge_index)

        # Should process successfully with self-loops added
        self.assertEqual(output.shape, (10, self.out_channels))

    def test_normalization(self):
        """Test adjacency matrix normalization."""
        layer = GCNConv(self.in_channels, self.out_channels, normalize=True)

        # Create a simple graph
        x = np.ones((5, self.in_channels)).astype(np.float32)
        edge_index = EdgeIndex([
            [0, 1, 2, 3, 4],
            [1, 2, 3, 4, 0]
        ])

        # Get normalized adjacency
        normalized = gcn_norm(edge_index, num_nodes=5)

        # Check normalization properties
        self.assertIsNotNone(normalized)

    def test_no_bias(self):
        """Test layer without bias."""
        layer = GCNConv(self.in_channels, self.out_channels, bias=False)

        self.assertIsNone(layer.bias)

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes)

        output = layer.forward(x, edge_index)
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))

    def test_different_aggregations(self):
        """Test different aggregation functions."""
        for aggr in ['add', 'mean', 'max']:
            layer = GCNConv(
                self.in_channels,
                self.out_channels,
                aggr=aggr
            )

            x = np.random.randn(50, self.in_channels).astype(np.float32)
            edge_index = self._create_random_edges(50)

            output = layer.forward(x, edge_index)
            self.assertEqual(output.shape, (50, self.out_channels))

    def _create_random_edges(self, num_nodes, num_edges=None):
        """Helper to create random edge index."""
        if num_edges is None:
            num_edges = num_nodes * 3

        source = np.random.randint(0, num_nodes, num_edges)
        target = np.random.randint(0, num_nodes, num_edges)

        return EdgeIndex([source, target])


class TestGATConv(unittest.TestCase):
    """Test Graph Attention Network layer."""

    def setUp(self):
        np.random.seed(42)
        self.in_channels = 16
        self.out_channels = 32
        self.heads = 8
        self.num_nodes = 50

    def test_init(self):
        """Test GAT layer initialization."""
        layer = GATConv(
            self.in_channels,
            self.out_channels,
            heads=self.heads
        )

        self.assertEqual(layer.in_channels, self.in_channels)
        self.assertEqual(layer.out_channels, self.out_channels)
        self.assertEqual(layer.heads, self.heads)

    def test_multi_head_attention(self):
        """Test multi-head attention mechanism."""
        layer = GATConv(
            self.in_channels,
            self.out_channels,
            heads=self.heads,
            concat=True
        )

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes)

        output = layer.forward(x, edge_index)

        # With concatenation, output has heads * out_channels
        expected_channels = self.heads * self.out_channels
        self.assertEqual(output.shape, (self.num_nodes, expected_channels))

    def test_attention_weights(self):
        """Test attention weight computation."""
        layer = GATConv(
            self.in_channels,
            self.out_channels,
            heads=4,
            return_attention_weights=True
        )

        x = np.random.randn(20, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(20, num_edges=50)

        output, attention_weights = layer.forward(x, edge_index)

        # Check attention weights
        self.assertEqual(len(attention_weights), 50)  # One per edge
        self.assertTrue(np.all(attention_weights >= 0))
        # Attention weights should sum to 1 for each target node
        # (approximately, due to numerical precision)

    def test_dropout(self):
        """Test attention dropout."""
        layer = GATConv(
            self.in_channels,
            self.out_channels,
            heads=self.heads,
            dropout=0.5
        )

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes)

        # Training mode (with dropout)
        layer.training = True
        output_train = layer.forward(x, edge_index)

        # Eval mode (no dropout)
        layer.training = False
        output_eval = layer.forward(x, edge_index)

        # Outputs should be different due to dropout
        # But same shape
        self.assertEqual(output_train.shape, output_eval.shape)

    def test_edge_features(self):
        """Test GAT with edge features."""
        layer = GATConv(
            self.in_channels,
            self.out_channels,
            heads=self.heads,
            edge_dim=8
        )

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes, num_edges=100)
        edge_attr = np.random.randn(100, 8).astype(np.float32)

        output = layer.forward(x, edge_index, edge_attr=edge_attr)

        expected_channels = self.heads * self.out_channels
        self.assertEqual(output.shape, (self.num_nodes, expected_channels))

    def test_negative_slope(self):
        """Test LeakyReLU negative slope parameter."""
        layer = GATConv(
            self.in_channels,
            self.out_channels,
            heads=self.heads,
            negative_slope=0.1
        )

        self.assertEqual(layer.negative_slope, 0.1)

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes)

        output = layer.forward(x, edge_index)
        self.assertIsNotNone(output)

    def _create_random_edges(self, num_nodes, num_edges=None):
        """Helper to create random edge index."""
        if num_edges is None:
            num_edges = num_nodes * 3

        source = np.random.randint(0, num_nodes, num_edges)
        target = np.random.randint(0, num_nodes, num_edges)

        return EdgeIndex([source, target])


class TestGraphSAGEConv(unittest.TestCase):
    """Test GraphSAGE convolutional layer."""

    def setUp(self):
        np.random.seed(42)
        self.in_channels = 16
        self.out_channels = 32
        self.num_nodes = 100

    def test_init(self):
        """Test GraphSAGE layer initialization."""
        layer = GraphSAGEConv(self.in_channels, self.out_channels)

        self.assertEqual(layer.in_channels, self.in_channels)
        self.assertEqual(layer.out_channels, self.out_channels)
        self.assertEqual(layer.aggr, 'mean')

    def test_aggregators(self):
        """Test different aggregation methods."""
        aggregators = ['mean', 'max', 'lstm', 'pool']

        for aggr in aggregators:
            layer = GraphSAGEConv(
                self.in_channels,
                self.out_channels,
                aggr=aggr
            )

            x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
            edge_index = self._create_random_edges(self.num_nodes)

            output = layer.forward(x, edge_index)
            self.assertEqual(output.shape, (self.num_nodes, self.out_channels))

    def test_normalize_embedding(self):
        """Test L2 normalization of embeddings."""
        layer = GraphSAGEConv(
            self.in_channels,
            self.out_channels,
            normalize_emb=True
        )

        x = np.random.randn(50, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(50)

        output = layer.forward(x, edge_index)

        # Check L2 normalization
        norms = np.linalg.norm(output, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_root_weight(self):
        """Test root node self-connection."""
        layer = GraphSAGEConv(
            self.in_channels,
            self.out_channels,
            root_weight=True
        )

        # Check root weight matrix exists
        self.assertIsNotNone(layer.lin_root)

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes)

        output = layer.forward(x, edge_index)
        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))

    def test_lstm_aggregator(self):
        """Test LSTM aggregation (if supported)."""
        layer = GraphSAGEConv(
            self.in_channels,
            self.out_channels,
            aggr='lstm'
        )

        x = np.random.randn(30, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(30)

        output = layer.forward(x, edge_index)
        self.assertEqual(output.shape, (30, self.out_channels))

    def _create_random_edges(self, num_nodes, num_edges=None):
        """Helper to create random edge index."""
        if num_edges is None:
            num_edges = num_nodes * 3

        source = np.random.randint(0, num_nodes, num_edges)
        target = np.random.randint(0, num_nodes, num_edges)

        return EdgeIndex([source, target])


class TestEdgeConv(unittest.TestCase):
    """Test Edge Convolutional layer."""

    def setUp(self):
        np.random.seed(42)
        self.in_channels = 16
        self.out_channels = 32
        self.num_nodes = 50

    def test_init(self):
        """Test EdgeConv layer initialization."""
        # Mock MLP
        nn = Mock()
        layer = EdgeConv(nn, aggr='max')

        self.assertEqual(layer.aggr, 'max')
        self.assertIsNotNone(layer.nn)

    def test_dynamic_graph(self):
        """Test dynamic graph construction."""
        from gnnruntime.layers.conv import DynamicEdgeConv

        layer = DynamicEdgeConv(
            self.in_channels,
            self.out_channels,
            k=10  # k-nearest neighbors
        )

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)

        # Should construct k-NN graph dynamically
        output = layer.forward(x)

        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))

    def test_edge_features(self):
        """Test edge feature computation."""
        nn = lambda edge_feat: edge_feat @ np.random.randn(
            edge_feat.shape[-1], self.out_channels
        ).astype(np.float32)

        layer = EdgeConv(nn)

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes, num_edges=150)

        output = layer.forward(x, edge_index)

        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))

    def _create_random_edges(self, num_nodes, num_edges=None):
        """Helper to create random edge index."""
        if num_edges is None:
            num_edges = num_nodes * 3

        source = np.random.randint(0, num_nodes, num_edges)
        target = np.random.randint(0, num_nodes, num_edges)

        return EdgeIndex([source, target])


class TestChebConv(unittest.TestCase):
    """Test Chebyshev Spectral Graph Convolution."""

    def setUp(self):
        np.random.seed(42)
        self.in_channels = 16
        self.out_channels = 32
        self.K = 3  # Chebyshev filter size
        self.num_nodes = 50

    def test_init(self):
        """Test ChebConv initialization."""
        layer = ChebConv(self.in_channels, self.out_channels, K=self.K)

        self.assertEqual(layer.in_channels, self.in_channels)
        self.assertEqual(layer.out_channels, self.out_channels)
        self.assertEqual(layer.K, self.K)

    def test_chebyshev_polynomials(self):
        """Test Chebyshev polynomial computation."""
        layer = ChebConv(self.in_channels, self.out_channels, K=self.K)

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes)

        # Compute with normalization
        output = layer.forward(x, edge_index, lambda_max=2.0)

        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))

    def test_lambda_max_computation(self):
        """Test largest eigenvalue computation."""
        layer = ChebConv(self.in_channels, self.out_channels, K=self.K)

        x = np.random.randn(20, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(20)

        # Should compute lambda_max if not provided
        output = layer.forward(x, edge_index, lambda_max=None)

        self.assertEqual(output.shape, (20, self.out_channels))

    def test_different_K_values(self):
        """Test different Chebyshev filter sizes."""
        for K in [1, 2, 3, 5, 10]:
            layer = ChebConv(self.in_channels, self.out_channels, K=K)

            x = np.random.randn(30, self.in_channels).astype(np.float32)
            edge_index = self._create_random_edges(30)

            output = layer.forward(x, edge_index)
            self.assertEqual(output.shape, (30, self.out_channels))

    def _create_random_edges(self, num_nodes, num_edges=None):
        """Helper to create random edge index."""
        if num_edges is None:
            num_edges = num_nodes * 3

        source = np.random.randint(0, num_nodes, num_edges)
        target = np.random.randint(0, num_nodes, num_edges)

        # Make symmetric for Chebyshev
        edge_index = EdgeIndex([
            np.concatenate([source, target]),
            np.concatenate([target, source])
        ])

        return edge_index


class TestSGConv(unittest.TestCase):
    """Test Simplified Graph Convolution."""

    def setUp(self):
        np.random.seed(42)
        self.in_channels = 16
        self.out_channels = 32
        self.K = 2  # Number of hops
        self.num_nodes = 50

    def test_init(self):
        """Test SGConv initialization."""
        layer = SGConv(self.in_channels, self.out_channels, K=self.K)

        self.assertEqual(layer.in_channels, self.in_channels)
        self.assertEqual(layer.out_channels, self.out_channels)
        self.assertEqual(layer.K, self.K)

    def test_multi_hop_aggregation(self):
        """Test K-hop aggregation."""
        layer = SGConv(self.in_channels, self.out_channels, K=self.K)

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes)

        output = layer.forward(x, edge_index)

        self.assertEqual(output.shape, (self.num_nodes, self.out_channels))

    def test_cached_computation(self):
        """Test cached adjacency matrix powers."""
        layer = SGConv(
            self.in_channels,
            self.out_channels,
            K=self.K,
            cached=True
        )

        x = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        edge_index = self._create_random_edges(self.num_nodes)

        # First forward should cache
        output1 = layer.forward(x, edge_index)

        # Second forward should use cache
        x2 = np.random.randn(self.num_nodes, self.in_channels).astype(np.float32)
        output2 = layer.forward(x2, edge_index)

        self.assertEqual(output1.shape, output2.shape)

    def _create_random_edges(self, num_nodes, num_edges=None):
        """Helper to create random edge index."""
        if num_edges is None:
            num_edges = num_nodes * 3

        source = np.random.randint(0, num_nodes, num_edges)
        target = np.random.randint(0, num_nodes, num_edges)

        return EdgeIndex([source, target])


class TestLayerNormalization(unittest.TestCase):
    """Test layer normalization for GNN layers."""

    def setUp(self):
        np.random.seed(42)

    def test_batch_norm(self):
        """Test batch normalization."""
        from gnnruntime.layers.norm import BatchNorm

        norm = BatchNorm(32)

        x = np.random.randn(100, 32).astype(np.float32)
        output = norm(x)

        # Check normalized statistics
        mean = np.mean(output, axis=0)
        std = np.std(output, axis=0)

        np.testing.assert_allclose(mean, 0, atol=1e-5)
        np.testing.assert_allclose(std, 1, atol=1e-5)

    def test_layer_norm(self):
        """Test layer normalization."""
        from gnnruntime.layers.norm import LayerNorm

        norm = LayerNorm(32)

        x = np.random.randn(100, 32).astype(np.float32)
        output = norm(x)

        # Check normalized statistics per sample
        mean = np.mean(output, axis=1)
        std = np.std(output, axis=1)

        np.testing.assert_allclose(mean, 0, atol=1e-5)
        np.testing.assert_allclose(std, 1, atol=1e-5)

    def test_graph_norm(self):
        """Test graph-specific normalization."""
        from gnnruntime.layers.norm import GraphNorm

        norm = GraphNorm(32)

        x = np.random.randn(100, 32).astype(np.float32)
        batch = np.array([0] * 50 + [1] * 50)  # Two graphs

        output = norm(x, batch)

        # Check normalization per graph
        graph1 = output[:50]
        graph2 = output[50:]

        self.assertAlmostEqual(np.mean(graph1), 0, places=5)
        self.assertAlmostEqual(np.mean(graph2), 0, places=5)


if __name__ == '__main__':
    unittest.main()