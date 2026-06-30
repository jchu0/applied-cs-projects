"""Unit tests for graph sampling and neighbor aggregation."""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnnruntime.sampler.neighbor import (
    NeighborSampler, UniformSampler, LayerwiseSampler,
    RandomWalkSampler, GraphSAINTSampler, ClusterGCNSampler,
    ImportanceSampler, AdaptiveSampler
)
from gnnruntime.core.graph import Graph, EdgeIndex


class TestNeighborSampler(unittest.TestCase):
    """Test neighbor sampling functionality."""

    def setUp(self):
        np.random.seed(42)
        self.num_nodes = 100
        self.edge_index = self._create_random_edges()

    def test_uniform_sampling(self):
        """Test uniform neighbor sampling."""
        sampler = UniformSampler(
            edge_index=self.edge_index,
            num_neighbors=[10, 5],  # 10 neighbors in first hop, 5 in second
            num_nodes=self.num_nodes
        )

        # Sample neighbors for nodes [0, 1, 2]
        target_nodes = [0, 1, 2]
        sampled_nodes, sampled_edges = sampler.sample(target_nodes)

        # Check sampling worked
        self.assertIn(0, sampled_nodes)
        self.assertIn(1, sampled_nodes)
        self.assertIn(2, sampled_nodes)

        # Check we don't oversample
        self.assertLessEqual(len(sampled_nodes), 3 + 3*10 + 3*10*5)  # Max possible

    def test_layerwise_sampling(self):
        """Test layer-wise neighbor sampling."""
        sampler = LayerwiseSampler(
            edge_index=self.edge_index,
            num_layers=2,
            num_neighbors=5,
            num_nodes=self.num_nodes
        )

        target_nodes = list(range(10))
        layers = sampler.sample_layers(target_nodes)

        # Should have 2 layers
        self.assertEqual(len(layers), 2)

        # Each layer should have nodes and edges
        for layer in layers:
            self.assertIn('nodes', layer)
            self.assertIn('edges', layer)

    def test_importance_sampling(self):
        """Test importance-weighted neighbor sampling."""
        # Create edge weights (importance scores)
        num_edges = self.edge_index.shape[1]
        edge_weights = np.random.rand(num_edges).astype(np.float32)

        sampler = ImportanceSampler(
            edge_index=self.edge_index,
            edge_weights=edge_weights,
            num_neighbors=10,
            num_nodes=self.num_nodes
        )

        target_nodes = [0, 5, 10]
        sampled_nodes, sampled_edges, sample_weights = sampler.sample(target_nodes)

        # Check importance weights are returned
        self.assertEqual(len(sample_weights), len(sampled_edges[0]))

        # High-weight edges should be more likely sampled
        self.assertTrue(np.all(sample_weights >= 0))

    def test_sampling_without_replacement(self):
        """Test sampling without replacement."""
        sampler = UniformSampler(
            edge_index=self.edge_index,
            num_neighbors=5,
            replace=False,
            num_nodes=self.num_nodes
        )

        node_id = 0
        neighbors = sampler.sample_neighbors(node_id, num_samples=5)

        # Check no duplicates
        self.assertEqual(len(neighbors), len(set(neighbors)))

    def test_sampling_with_replacement(self):
        """Test sampling with replacement."""
        sampler = UniformSampler(
            edge_index=self.edge_index,
            num_neighbors=20,  # More than actual neighbors
            replace=True,
            num_nodes=self.num_nodes
        )

        node_id = 0
        neighbors = sampler.sample_neighbors(node_id, num_samples=20)

        # Should have exactly 20 samples (with possible duplicates)
        self.assertEqual(len(neighbors), 20)

    def _create_random_edges(self):
        """Helper to create random edge index."""
        num_edges = self.num_nodes * 5
        source = np.random.randint(0, self.num_nodes, num_edges)
        target = np.random.randint(0, self.num_nodes, num_edges)
        return np.array([source, target])


class TestRandomWalkSampler(unittest.TestCase):
    """Test random walk sampling."""

    def setUp(self):
        np.random.seed(42)
        self.num_nodes = 50
        self.edge_index = self._create_connected_graph()

    def test_random_walk(self):
        """Test basic random walk sampling."""
        sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            walk_length=10,
            num_nodes=self.num_nodes
        )

        start_nodes = [0, 5, 10]
        walks = sampler.sample(start_nodes)

        # Check walk properties
        self.assertEqual(len(walks), 3)  # One walk per start node
        for walk in walks:
            self.assertEqual(len(walk), 10)  # Walk length
            # Check consecutive nodes are connected
            for i in range(len(walk) - 1):
                self.assertTrue(self._are_connected(walk[i], walk[i+1]))

    def test_node2vec_walks(self):
        """Test Node2Vec-style biased random walks."""
        sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            walk_length=20,
            p=0.5,  # Return parameter
            q=2.0,  # In-out parameter
            num_nodes=self.num_nodes
        )

        start_node = 0
        walks = sampler.sample_node2vec([start_node], num_walks=5)

        # Should generate 5 walks
        self.assertEqual(len(walks), 5)

        for walk in walks:
            self.assertEqual(walk[0], start_node)
            self.assertLessEqual(len(walk), 20)

    def test_restart_probability(self):
        """Test random walk with restart."""
        sampler = RandomWalkSampler(
            edge_index=self.edge_index,
            walk_length=100,
            restart_prob=0.15,
            num_nodes=self.num_nodes
        )

        start_node = 10
        walk = sampler.sample_with_restart(start_node)

        # Count returns to start node
        returns = sum(1 for node in walk if node == start_node)

        # With restart probability, should return to start occasionally
        self.assertGreater(returns, 1)

    def _create_connected_graph(self):
        """Helper to create a connected graph."""
        # Create a ring graph with random edges
        edges = []

        # Ring edges
        for i in range(self.num_nodes):
            edges.append([i, (i + 1) % self.num_nodes])
            edges.append([(i + 1) % self.num_nodes, i])

        # Random edges
        for _ in range(self.num_nodes * 2):
            src = np.random.randint(0, self.num_nodes)
            dst = np.random.randint(0, self.num_nodes)
            if src != dst:
                edges.append([src, dst])
                edges.append([dst, src])

        edges = np.array(edges).T
        return edges

    def _are_connected(self, node1, node2):
        """Check if two nodes are connected."""
        for i in range(self.edge_index.shape[1]):
            if (self.edge_index[0, i] == node1 and self.edge_index[1, i] == node2):
                return True
        return False


class TestGraphSAINTSampler(unittest.TestCase):
    """Test GraphSAINT sampling methods."""

    def setUp(self):
        np.random.seed(42)
        self.num_nodes = 200
        self.edge_index = self._create_clustered_graph()

    def test_node_sampling(self):
        """Test GraphSAINT node sampling."""
        sampler = GraphSAINTSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            sample_coverage=3,  # Each node sampled ~3 times
            method='node'
        )

        subgraphs = sampler.sample_subgraphs(num_samples=10, size=50)

        self.assertEqual(len(subgraphs), 10)

        for subgraph in subgraphs:
            self.assertLessEqual(len(subgraph['nodes']), 50)
            # Check subgraph is connected
            self.assertGreater(len(subgraph['edges'][0]), 0)

    def test_edge_sampling(self):
        """Test GraphSAINT edge sampling."""
        sampler = GraphSAINTSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            sample_coverage=2,
            method='edge'
        )

        subgraphs = sampler.sample_subgraphs(num_samples=5, size=100)

        for subgraph in subgraphs:
            # Check edge-induced subgraph
            edge_count = len(subgraph['edges'][0])
            self.assertLessEqual(edge_count, 100)

    def test_random_walk_sampling(self):
        """Test GraphSAINT random walk sampling."""
        sampler = GraphSAINTSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            method='rw',
            walk_length=20
        )

        subgraphs = sampler.sample_subgraphs(num_samples=5, size=30)

        for subgraph in subgraphs:
            # Random walk subgraphs should be connected
            nodes = subgraph['nodes']
            self.assertLessEqual(len(nodes), 30)

    def test_normalization_coefficients(self):
        """Test computation of normalization coefficients."""
        sampler = GraphSAINTSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            method='node'
        )

        subgraph = sampler.sample_single_subgraph(size=50)
        norm_coeffs = sampler.compute_norm_coefficients(subgraph)

        # Check normalization coefficients
        self.assertIn('node_norm', norm_coeffs)
        self.assertIn('edge_norm', norm_coeffs)

        # Coefficients should be positive
        self.assertTrue(np.all(norm_coeffs['node_norm'] > 0))
        self.assertTrue(np.all(norm_coeffs['edge_norm'] > 0))

    def _create_clustered_graph(self):
        """Helper to create a graph with community structure."""
        edges = []
        num_clusters = 4
        nodes_per_cluster = self.num_nodes // num_clusters

        # Create dense clusters
        for c in range(num_clusters):
            cluster_start = c * nodes_per_cluster
            cluster_end = (c + 1) * nodes_per_cluster

            # Intra-cluster edges (dense)
            for i in range(cluster_start, cluster_end):
                for j in range(i + 1, min(i + 5, cluster_end)):
                    edges.append([i, j])
                    edges.append([j, i])

        # Inter-cluster edges (sparse)
        for _ in range(self.num_nodes):
            src = np.random.randint(0, self.num_nodes)
            dst = np.random.randint(0, self.num_nodes)
            if src != dst:
                edges.append([src, dst])

        return np.array(edges).T


class TestClusterGCNSampler(unittest.TestCase):
    """Test ClusterGCN sampling."""

    def setUp(self):
        np.random.seed(42)
        self.num_nodes = 300
        self.edge_index = self._create_clustered_graph()

    def test_graph_clustering(self):
        """Test graph clustering for ClusterGCN."""
        sampler = ClusterGCNSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            num_parts=10
        )

        # Perform clustering
        clusters = sampler.cluster_graph()

        self.assertEqual(len(clusters), 10)

        # Check all nodes are assigned
        all_nodes = set()
        for cluster in clusters:
            all_nodes.update(cluster)
        self.assertEqual(len(all_nodes), self.num_nodes)

        # Check clusters are roughly balanced
        cluster_sizes = [len(c) for c in clusters]
        avg_size = self.num_nodes / 10
        for size in cluster_sizes:
            self.assertGreater(size, avg_size * 0.5)
            self.assertLess(size, avg_size * 1.5)

    def test_cluster_batch_sampling(self):
        """Test batch sampling of clusters."""
        sampler = ClusterGCNSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            num_parts=10,
            batch_size=3
        )

        # Sample batch of clusters
        batch = sampler.sample_cluster_batch()

        # Should combine 3 clusters
        self.assertLessEqual(len(batch['nodes']), self.num_nodes * 0.4)

    def test_between_cluster_edges(self):
        """Test handling of between-cluster edges."""
        sampler = ClusterGCNSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            num_parts=5
        )

        clusters = sampler.cluster_graph()

        # Check between-cluster edge computation
        for i, cluster in enumerate(clusters):
            subgraph = sampler.get_cluster_subgraph(cluster, include_between=True)

            # Should include some between-cluster edges
            edges_in_subgraph = subgraph['edges']
            self.assertGreater(len(edges_in_subgraph[0]), 0)

    def _create_clustered_graph(self):
        """Helper to create clustered graph."""
        edges = []
        num_clusters = 6
        nodes_per_cluster = self.num_nodes // num_clusters

        for c in range(num_clusters):
            cluster_start = c * nodes_per_cluster
            cluster_end = (c + 1) * nodes_per_cluster

            # Dense within cluster
            for i in range(cluster_start, cluster_end):
                for j in range(i + 1, min(i + 8, cluster_end)):
                    edges.append([i, j])
                    edges.append([j, i])

            # Sparse between clusters
            if c < num_clusters - 1:
                for _ in range(5):
                    src = np.random.randint(cluster_start, cluster_end)
                    dst = np.random.randint(cluster_end, min(cluster_end + nodes_per_cluster, self.num_nodes))
                    edges.append([src, dst])
                    edges.append([dst, src])

        return np.array(edges).T


class TestAdaptiveSampler(unittest.TestCase):
    """Test adaptive sampling strategies."""

    def setUp(self):
        np.random.seed(42)
        self.num_nodes = 100
        self.edge_index = self._create_heterogeneous_graph()

    def test_variance_reduction_sampling(self):
        """Test variance reduction in sampling."""
        sampler = AdaptiveSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            method='variance_reduction'
        )

        # Initialize node features for variance computation
        node_features = np.random.randn(self.num_nodes, 32).astype(np.float32)

        # Sample with variance reduction
        target_nodes = list(range(10))
        sampled_nodes, sampled_edges, weights = sampler.sample_adaptive(
            target_nodes,
            node_features,
            num_neighbors=5
        )

        # Check importance weights for variance reduction
        self.assertEqual(len(weights), len(sampled_edges[0]))
        self.assertTrue(np.all(weights > 0))

    def test_fastgcn_sampling(self):
        """Test FastGCN importance sampling."""
        sampler = AdaptiveSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            method='fastgcn'
        )

        # Sample with importance based on node degrees
        layer_sizes = [50, 25, 10]
        layers = sampler.sample_fastgcn(layer_sizes)

        self.assertEqual(len(layers), 3)

        # Check layer sizes
        for i, layer in enumerate(layers):
            self.assertLessEqual(len(layer['nodes']), layer_sizes[i])

    def test_ladies_sampling(self):
        """Test LADIES layer-dependent sampling."""
        sampler = AdaptiveSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            method='ladies'
        )

        # Layer-dependent importance sampling
        target_nodes = list(range(20))
        node_features = np.random.randn(self.num_nodes, 16).astype(np.float32)

        samples = sampler.sample_ladies(
            target_nodes,
            node_features,
            num_layers=2,
            layer_sizes=[30, 15]
        )

        self.assertEqual(len(samples), 2)

        # Check layer-wise sampling
        for i, layer in enumerate(samples):
            self.assertIn('nodes', layer)
            self.assertIn('importance_weights', layer)

    def test_adaptive_neighbor_selection(self):
        """Test adaptive neighbor count selection."""
        sampler = AdaptiveSampler(
            edge_index=self.edge_index,
            num_nodes=self.num_nodes,
            method='adaptive_k'
        )

        # Adaptively choose number of neighbors based on degree
        target_nodes = list(range(self.num_nodes))
        degrees = sampler.compute_degrees()

        adaptive_k = sampler.compute_adaptive_k(
            target_nodes,
            degrees,
            min_k=2,
            max_k=20,
            budget=500
        )

        # High-degree nodes should have fewer samples
        for node in target_nodes:
            if degrees[node] > 20:
                self.assertLess(adaptive_k[node], 10)

    def _create_heterogeneous_graph(self):
        """Helper to create graph with varying node degrees."""
        edges = []

        # Create hub nodes (high degree)
        hubs = list(range(5))
        for hub in hubs:
            for _ in range(30):
                target = np.random.randint(5, self.num_nodes)
                edges.append([hub, target])
                edges.append([target, hub])

        # Create normal connectivity
        for i in range(5, self.num_nodes):
            for _ in range(3):
                target = np.random.randint(0, self.num_nodes)
                if target != i:
                    edges.append([i, target])

        return np.array(edges).T


if __name__ == '__main__':
    unittest.main()