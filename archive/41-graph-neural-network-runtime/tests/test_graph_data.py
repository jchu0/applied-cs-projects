"""Unit tests for graph data structures and operations."""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os
import json

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnnruntime.core.graph import (
    Graph, EdgeIndex, HeteroGraph, TemporalGraph,
    to_undirected, to_directed, add_self_loops,
    remove_self_loops, degree, k_hop_subgraph,
    induced_subgraph, sample_negative_edges
)
from gnnruntime.data.batch import Batch, DataLoader, collate_fn


class TestGraph(unittest.TestCase):
    """Test Graph data structure."""

    def setUp(self):
        np.random.seed(42)

    def test_graph_creation(self):
        """Test basic graph creation."""
        num_nodes = 100
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        x = np.random.randn(num_nodes, 16).astype(np.float32)

        graph = Graph(
            x=x,
            edge_index=edge_index,
            edge_attr=None,
            y=None,
            num_nodes=num_nodes
        )

        self.assertEqual(graph.num_nodes, num_nodes)
        self.assertEqual(graph.num_edges, 3)
        self.assertEqual(graph.x.shape, (num_nodes, 16))

    def test_edge_index_validation(self):
        """Test edge index validation."""
        # Invalid edge indices
        with self.assertRaises(ValueError):
            edge_index = np.array([[0, 1, 200], [1, 2, 0]])  # Node 200 doesn't exist
            graph = Graph(
                edge_index=edge_index,
                num_nodes=100
            )

    def test_graph_properties(self):
        """Test graph property computations."""
        edge_index = np.array([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2]
        ])
        graph = Graph(edge_index=edge_index, num_nodes=4)

        # Test degree computation
        degrees = graph.degree()
        expected_degrees = [1, 2, 2, 1]  # Node degrees
        np.testing.assert_array_equal(degrees, expected_degrees)

        # Test is_undirected - this graph has all edges with their reverse
        self.assertTrue(graph.is_undirected())

    def test_add_self_loops(self):
        """Test adding self-loops to graph."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        graph = Graph(edge_index=edge_index, num_nodes=3)

        graph_with_loops = graph.add_self_loops()

        # Should have original edges + 3 self-loops
        self.assertEqual(graph_with_loops.num_edges, 6)

        # Check self-loops exist
        edge_set = set(zip(graph_with_loops.edge_index[0], graph_with_loops.edge_index[1]))
        self.assertIn((0, 0), edge_set)
        self.assertIn((1, 1), edge_set)
        self.assertIn((2, 2), edge_set)

    def test_remove_self_loops(self):
        """Test removing self-loops from graph."""
        edge_index = np.array([
            [0, 1, 2, 0, 1, 2],
            [0, 2, 0, 1, 1, 2]  # Has self-loops
        ])
        graph = Graph(edge_index=edge_index, num_nodes=3)

        graph_no_loops = graph.remove_self_loops()

        # Should have removed 3 self-loops
        self.assertEqual(graph_no_loops.num_edges, 3)

        # Check no self-loops exist
        edge_set = set(zip(graph_no_loops.edge_index[0], graph_no_loops.edge_index[1]))
        self.assertNotIn((0, 0), edge_set)
        self.assertNotIn((1, 1), edge_set)
        self.assertNotIn((2, 2), edge_set)

    def test_to_undirected(self):
        """Test converting to undirected graph."""
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        graph = Graph(edge_index=edge_index, num_nodes=3)

        undirected = graph.to_undirected()

        # Should have symmetric edges
        self.assertEqual(undirected.num_edges, 6)
        self.assertTrue(undirected.is_undirected())

    def test_subgraph_extraction(self):
        """Test subgraph extraction."""
        edge_index = np.array([
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3]
        ])
        x = np.random.randn(5, 8).astype(np.float32)
        graph = Graph(x=x, edge_index=edge_index, num_nodes=5)

        # Extract subgraph with nodes [1, 2, 3]
        node_idx = [1, 2, 3]
        subgraph = graph.subgraph(node_idx)

        self.assertEqual(subgraph.num_nodes, 3)
        # Should only have edges between nodes 1, 2, 3
        self.assertTrue(np.all(subgraph.edge_index < 3))

    def test_graph_with_features(self):
        """Test graph with node and edge features."""
        num_nodes = 10
        num_edges = 20

        x = np.random.randn(num_nodes, 16).astype(np.float32)
        edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        edge_attr = np.random.randn(num_edges, 8).astype(np.float32)
        y = np.random.randint(0, 4, num_nodes)  # Node labels

        graph = Graph(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            num_nodes=num_nodes
        )

        self.assertEqual(graph.x.shape, (num_nodes, 16))
        self.assertEqual(graph.edge_attr.shape, (num_edges, 8))
        self.assertEqual(graph.y.shape, (num_nodes,))


class TestEdgeIndex(unittest.TestCase):
    """Test EdgeIndex data structure."""

    def setUp(self):
        np.random.seed(42)

    def test_edge_index_creation(self):
        """Test EdgeIndex creation and validation."""
        edge_list = [[0, 1, 2], [1, 2, 0]]
        edge_index = EdgeIndex(edge_list)

        self.assertEqual(edge_index.shape, (2, 3))
        self.assertEqual(edge_index.num_edges, 3)

    def test_edge_index_operations(self):
        """Test EdgeIndex operations."""
        edge_index = EdgeIndex([[0, 1, 2], [1, 2, 0]])

        # Test transpose
        transposed = edge_index.t()
        np.testing.assert_array_equal(transposed[0], [1, 2, 0])
        np.testing.assert_array_equal(transposed[1], [0, 1, 2])

        # Test concatenation
        other = EdgeIndex([[3, 4], [4, 5]])
        combined = edge_index.cat(other)
        self.assertEqual(combined.num_edges, 5)

    def test_coalesce(self):
        """Test removing duplicate edges."""
        # Has duplicate edges: (0,1) at idx 0 and 2, (1,2) at idx 1 and 4
        edge_index = EdgeIndex([
            [0, 1, 0, 2, 1],
            [1, 2, 1, 3, 2]
        ])

        coalesced = edge_index.coalesce()

        # Should remove duplicates: (0,1), (1,2), (2,3) = 3 unique edges
        self.assertEqual(coalesced.num_edges, 3)

    def test_to_dense_adj(self):
        """Test conversion to dense adjacency matrix."""
        edge_index = EdgeIndex([[0, 1, 2], [1, 2, 0]])
        adj = edge_index.to_dense_adj(num_nodes=3)

        expected = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0]
        ])
        np.testing.assert_array_equal(adj, expected)


class TestHeteroGraph(unittest.TestCase):
    """Test Heterogeneous Graph data structure."""

    def setUp(self):
        np.random.seed(42)

    def test_hetero_graph_creation(self):
        """Test creating heterogeneous graph."""
        # Define node types
        node_types = {
            'user': 100,
            'item': 200,
            'category': 50
        }

        # Define edge types
        edge_types = {
            ('user', 'rates', 'item'): np.array([[0, 1, 2], [0, 1, 2]]),
            ('item', 'belongs_to', 'category'): np.array([[0, 1], [0, 1]]),
            ('user', 'follows', 'user'): np.array([[0, 1], [1, 2]])
        }

        # Node features
        x_dict = {
            'user': np.random.randn(100, 32).astype(np.float32),
            'item': np.random.randn(200, 64).astype(np.float32),
            'category': np.random.randn(50, 16).astype(np.float32)
        }

        hetero_graph = HeteroGraph(
            x_dict=x_dict,
            edge_index_dict=edge_types,
            node_types=node_types
        )

        self.assertEqual(len(hetero_graph.node_types), 3)
        self.assertEqual(len(hetero_graph.edge_types), 3)
        self.assertEqual(hetero_graph.num_nodes('user'), 100)

    def test_hetero_graph_operations(self):
        """Test operations on heterogeneous graphs."""
        hetero_graph = self._create_sample_hetero_graph()

        # Test getting subgraph for specific edge type
        edge_type = ('user', 'rates', 'item')
        subgraph = hetero_graph.edge_type_subgraph(edge_type)

        self.assertIn('user', subgraph.node_types)
        self.assertIn('item', subgraph.node_types)

        # Test getting node features
        user_features = hetero_graph.x_dict['user']
        self.assertEqual(user_features.shape[0], 50)

    def test_hetero_to_homo_conversion(self):
        """Test converting heterogeneous to homogeneous graph."""
        hetero_graph = self._create_sample_hetero_graph()

        # Convert to homogeneous
        homo_graph = hetero_graph.to_homogeneous()

        # Should combine all nodes and edges
        total_nodes = sum(hetero_graph.num_nodes(nt) for nt in hetero_graph.node_types)
        self.assertEqual(homo_graph.num_nodes, total_nodes)

    def _create_sample_hetero_graph(self):
        """Helper to create sample heterogeneous graph."""
        node_types = {'user': 50, 'item': 100}
        edge_types = {
            ('user', 'rates', 'item'): np.array([
                np.random.randint(0, 50, 200),
                np.random.randint(0, 100, 200)
            ])
        }
        x_dict = {
            'user': np.random.randn(50, 32).astype(np.float32),
            'item': np.random.randn(100, 64).astype(np.float32)
        }

        return HeteroGraph(x_dict, edge_types, node_types)


class TestTemporalGraph(unittest.TestCase):
    """Test Temporal/Dynamic Graph data structure."""

    def setUp(self):
        np.random.seed(42)

    def test_temporal_graph_creation(self):
        """Test creating temporal graph."""
        num_nodes = 100
        num_edges = 500
        num_timestamps = 10

        # Create edges with timestamps
        edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        timestamps = np.random.randint(0, num_timestamps, num_edges)

        temporal_graph = TemporalGraph(
            edge_index=edge_index,
            timestamps=timestamps,
            num_nodes=num_nodes
        )

        self.assertEqual(temporal_graph.num_nodes, num_nodes)
        self.assertEqual(temporal_graph.num_edges, num_edges)
        self.assertEqual(temporal_graph.num_timestamps, num_timestamps)

    def test_snapshot_extraction(self):
        """Test extracting graph snapshots at specific times."""
        temporal_graph = self._create_sample_temporal_graph()

        # Extract snapshot at time t=5
        snapshot = temporal_graph.snapshot(t=5)

        # Should only have edges at time 5
        self.assertTrue(all(temporal_graph.timestamps[i] == 5
                          for i in snapshot.edge_indices))

    def test_temporal_subgraph(self):
        """Test extracting subgraph for time range."""
        temporal_graph = self._create_sample_temporal_graph()

        # Extract subgraph for time range [3, 7]
        subgraph = temporal_graph.temporal_subgraph(start_time=3, end_time=7)

        # All edges should be in time range
        edge_times = subgraph.timestamps
        self.assertTrue(np.all(edge_times >= 3))
        self.assertTrue(np.all(edge_times <= 7))

    def test_temporal_aggregation(self):
        """Test aggregating temporal edges."""
        temporal_graph = self._create_sample_temporal_graph()

        # Aggregate all edges (ignore time)
        static_graph = temporal_graph.to_static()

        # Should have all edges
        self.assertEqual(static_graph.num_edges, temporal_graph.num_edges)

    def _create_sample_temporal_graph(self):
        """Helper to create sample temporal graph."""
        num_nodes = 50
        num_edges = 200
        num_timestamps = 10

        edge_index = np.random.randint(0, num_nodes, (2, num_edges))
        timestamps = np.random.randint(0, num_timestamps, num_edges)

        return TemporalGraph(
            edge_index=edge_index,
            timestamps=timestamps,
            num_nodes=num_nodes
        )


class TestBatch(unittest.TestCase):
    """Test graph batching functionality."""

    def setUp(self):
        np.random.seed(42)

    def test_batch_creation(self):
        """Test creating batch from multiple graphs."""
        graphs = []
        for i in range(3):
            num_nodes = 10 + i * 5
            edge_index = np.array([
                np.random.randint(0, num_nodes, 20),
                np.random.randint(0, num_nodes, 20)
            ])
            x = np.random.randn(num_nodes, 8).astype(np.float32)
            graphs.append(Graph(x=x, edge_index=edge_index, num_nodes=num_nodes))

        batch = Batch.from_graph_list(graphs)

        # Check batch properties
        total_nodes = sum(g.num_nodes for g in graphs)
        total_edges = sum(g.num_edges for g in graphs)

        self.assertEqual(batch.num_nodes, total_nodes)
        self.assertEqual(batch.num_edges, total_edges)
        self.assertEqual(len(batch.batch), total_nodes)

    def test_batch_indexing(self):
        """Test accessing individual graphs from batch."""
        graphs = [self._create_random_graph(20) for _ in range(5)]
        batch = Batch.from_graph_list(graphs)

        # Access individual graphs
        for i in range(5):
            graph_i = batch[i]
            self.assertEqual(graph_i.num_nodes, graphs[i].num_nodes)

    def test_batch_to_list(self):
        """Test converting batch back to list of graphs."""
        original_graphs = [self._create_random_graph(15 + i * 5) for i in range(4)]
        batch = Batch.from_graph_list(original_graphs)

        # Convert back to list
        recovered_graphs = batch.to_graph_list()

        self.assertEqual(len(recovered_graphs), len(original_graphs))
        for orig, recov in zip(original_graphs, recovered_graphs):
            self.assertEqual(orig.num_nodes, recov.num_nodes)
            np.testing.assert_array_equal(orig.x.shape, recov.x.shape)

    def _create_random_graph(self, num_nodes):
        """Helper to create random graph."""
        edge_index = np.random.randint(0, num_nodes, (2, num_nodes * 3))
        x = np.random.randn(num_nodes, 16).astype(np.float32)
        return Graph(x=x, edge_index=edge_index, num_nodes=num_nodes)


class TestDataLoader(unittest.TestCase):
    """Test graph data loader."""

    def setUp(self):
        np.random.seed(42)
        self.dataset = [self._create_random_graph(20 + i) for i in range(100)]

    def test_dataloader_creation(self):
        """Test creating data loader."""
        loader = DataLoader(
            self.dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0
        )

        self.assertEqual(loader.batch_size, 32)
        self.assertEqual(len(loader.dataset), 100)

    def test_dataloader_iteration(self):
        """Test iterating through data loader."""
        loader = DataLoader(self.dataset, batch_size=10)

        batch_count = 0
        total_graphs = 0

        for batch in loader:
            batch_count += 1
            self.assertIsInstance(batch, Batch)
            self.assertLessEqual(batch.num_graphs, 10)
            total_graphs += batch.num_graphs

        self.assertEqual(batch_count, 10)  # 100 graphs / 10 per batch
        self.assertEqual(total_graphs, 100)

    def test_dataloader_shuffle(self):
        """Test data loader shuffling."""
        loader1 = DataLoader(self.dataset, batch_size=10, shuffle=True, seed=42)
        loader2 = DataLoader(self.dataset, batch_size=10, shuffle=True, seed=43)

        batches1 = list(loader1)
        batches2 = list(loader2)

        # With different seeds, order should be different
        different = False
        for b1, b2 in zip(batches1, batches2):
            if not np.array_equal(b1.x, b2.x):
                different = True
                break

        self.assertTrue(different)

    def test_collate_fn(self):
        """Test custom collate function."""
        def custom_collate(graph_list):
            # Custom batching logic
            batch = Batch.from_graph_list(graph_list)
            batch.custom_field = len(graph_list)
            return batch

        loader = DataLoader(
            self.dataset[:20],
            batch_size=5,
            collate_fn=custom_collate
        )

        for batch in loader:
            self.assertEqual(batch.custom_field, 5)

    def _create_random_graph(self, num_nodes):
        """Helper to create random graph."""
        edge_index = np.random.randint(0, num_nodes, (2, num_nodes * 2))
        x = np.random.randn(num_nodes, 16).astype(np.float32)
        y = np.random.randint(0, 4, 1)  # Graph label
        return Graph(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)


class TestGraphUtilities(unittest.TestCase):
    """Test graph utility functions."""

    def setUp(self):
        np.random.seed(42)

    def test_k_hop_subgraph(self):
        """Test k-hop subgraph extraction."""
        # Create a chain graph: 0-1-2-3-4
        edge_index = np.array([
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3]
        ])

        # Get 2-hop subgraph around node 2
        subgraph_nodes, subgraph_edges = k_hop_subgraph(
            node_idx=2,
            num_hops=2,
            edge_index=edge_index
        )

        # Should include nodes 0, 1, 2, 3, 4 (all nodes within 2 hops)
        expected_nodes = {0, 1, 2, 3, 4}
        self.assertEqual(set(subgraph_nodes), expected_nodes)

    def test_induced_subgraph(self):
        """Test induced subgraph extraction."""
        edge_index = np.array([
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]
        ])

        # Get induced subgraph for nodes [1, 2, 3]
        node_idx = [1, 2, 3]
        subgraph_edges = induced_subgraph(node_idx, edge_index)

        # Should only have edges between nodes 1, 2, 3
        for src, dst in zip(subgraph_edges[0], subgraph_edges[1]):
            self.assertIn(src, node_idx)
            self.assertIn(dst, node_idx)

    def test_negative_sampling(self):
        """Test negative edge sampling."""
        num_nodes = 10
        edge_index = np.array([
            [0, 1, 2, 3],
            [1, 2, 3, 4]
        ])

        neg_edges = sample_negative_edges(
            edge_index,
            num_nodes=num_nodes,
            num_neg_samples=10
        )

        # Check negative edges don't exist in original
        pos_edges = set(zip(edge_index[0], edge_index[1]))
        neg_edges_set = set(zip(neg_edges[0], neg_edges[1]))

        self.assertEqual(len(pos_edges & neg_edges_set), 0)

    def test_degree_computation(self):
        """Test node degree computation."""
        edge_index = np.array([
            [0, 1, 1, 2, 2, 2],
            [1, 0, 2, 0, 1, 3]
        ])

        # Compute in-degree and out-degree
        in_degree = degree(edge_index[1], num_nodes=4)
        out_degree = degree(edge_index[0], num_nodes=4)

        # In-degrees: count dst occurrences
        # dst = [1, 0, 2, 0, 1, 3] -> node 0: 2, node 1: 2, node 2: 1, node 3: 1
        expected_in = [2, 2, 1, 1]
        # Out-degrees: count src occurrences
        # src = [0, 1, 1, 2, 2, 2] -> node 0: 1, node 1: 2, node 2: 3, node 3: 0
        expected_out = [1, 2, 3, 0]

        np.testing.assert_array_equal(in_degree, expected_in)
        np.testing.assert_array_equal(out_degree, expected_out)


if __name__ == '__main__':
    unittest.main()