"""Tests for graph storage module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnn_runtime.graph import GraphStorage, GraphFormat, PartitionedGraph


class TestGraphStorage:
    """Tests for GraphStorage class."""

    def test_from_edge_list_basic(self):
        """Test basic graph construction from edge list."""
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        assert graph.num_nodes == 3
        assert graph.num_edges == 3
        assert len(graph.csr_indptr) == 4

    def test_from_edge_list_with_num_nodes(self):
        """Test graph construction with explicit num_nodes."""
        src = np.array([0, 1])
        dst = np.array([1, 2])
        graph = GraphStorage.from_edge_list(src, dst, num_nodes=5)

        assert graph.num_nodes == 5
        assert graph.num_edges == 2

    def test_from_edge_list_undirected(self):
        """Test undirected graph construction."""
        src = np.array([0, 1])
        dst = np.array([1, 2])
        graph = GraphStorage.from_edge_list(src, dst, directed=False)

        assert graph.num_nodes == 3
        assert graph.num_edges == 4  # Doubled for undirected

    def test_from_adjacency_matrix(self):
        """Test graph construction from adjacency matrix."""
        adj = np.array([
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ])
        graph = GraphStorage.from_adjacency_matrix(adj)

        assert graph.num_nodes == 3
        assert graph.num_edges == 3

    def test_get_neighbors_out(self):
        """Test getting outgoing neighbors."""
        src = np.array([0, 0, 1])
        dst = np.array([1, 2, 2])
        graph = GraphStorage.from_edge_list(src, dst)

        neighbors = graph.get_neighbors(0, 'out')
        assert set(neighbors) == {1, 2}

        neighbors = graph.get_neighbors(1, 'out')
        assert set(neighbors) == {2}

        neighbors = graph.get_neighbors(2, 'out')
        assert len(neighbors) == 0

    def test_get_neighbors_in(self):
        """Test getting incoming neighbors."""
        src = np.array([0, 1])
        dst = np.array([1, 0])
        graph = GraphStorage.from_edge_list(src, dst)
        graph.to_csc()

        in_neighbors = graph.get_neighbors(0, 'in')
        assert 1 in in_neighbors

        in_neighbors = graph.get_neighbors(1, 'in')
        assert 0 in in_neighbors

    def test_csc_conversion(self):
        """Test CSR to CSC conversion."""
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        assert graph.csc_indptr is None
        graph.to_csc()
        assert graph.csc_indptr is not None
        assert graph.csc_indices is not None

    def test_degree_out(self):
        """Test out-degree calculation."""
        src = np.array([0, 0, 1])
        dst = np.array([1, 2, 2])
        graph = GraphStorage.from_edge_list(src, dst)

        assert graph.degree(0, 'out') == 2
        assert graph.degree(1, 'out') == 1
        assert graph.degree(2, 'out') == 0

    def test_degree_in(self):
        """Test in-degree calculation."""
        src = np.array([0, 0, 1])
        dst = np.array([1, 2, 2])
        graph = GraphStorage.from_edge_list(src, dst)

        assert graph.degree(0, 'in') == 0
        assert graph.degree(1, 'in') == 1
        assert graph.degree(2, 'in') == 2

    def test_degrees_array(self):
        """Test getting all degrees."""
        src = np.array([0, 0, 1])
        dst = np.array([1, 2, 2])
        graph = GraphStorage.from_edge_list(src, dst)

        out_degrees = graph.degrees('out')
        assert list(out_degrees) == [2, 1, 0]

    def test_has_edge(self):
        """Test edge existence check."""
        src = np.array([0, 1])
        dst = np.array([1, 2])
        graph = GraphStorage.from_edge_list(src, dst)

        assert graph.has_edge(0, 1)
        assert graph.has_edge(1, 2)
        assert not graph.has_edge(0, 2)
        assert not graph.has_edge(2, 0)

    def test_to_coo(self):
        """Test conversion to COO format."""
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        coo_src, coo_dst = graph.to_coo()
        assert len(coo_src) == 3
        assert len(coo_dst) == 3

    def test_add_self_loops(self):
        """Test adding self-loops."""
        src = np.array([0, 1])
        dst = np.array([1, 2])
        graph = GraphStorage.from_edge_list(src, dst, num_nodes=3)

        graph_with_loops = graph.add_self_loops()
        assert graph_with_loops.num_edges >= graph.num_edges
        assert graph_with_loops.has_edge(0, 0)
        assert graph_with_loops.has_edge(1, 1)
        assert graph_with_loops.has_edge(2, 2)

    def test_subgraph(self):
        """Test subgraph extraction."""
        src = np.array([0, 0, 1, 1, 2])
        dst = np.array([1, 2, 2, 3, 3])
        graph = GraphStorage.from_edge_list(src, dst)

        subgraph = graph.subgraph(np.array([0, 1, 2]))
        assert subgraph.num_nodes == 3
        assert subgraph.has_edge(0, 1)
        assert subgraph.has_edge(0, 2)
        assert subgraph.has_edge(1, 2)

    def test_to_edge_index(self):
        """Test conversion to edge_index format."""
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        edge_index = graph.to_edge_index()
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] == 3

    def test_node_features(self):
        """Test node feature storage."""
        src = np.array([0, 1])
        dst = np.array([1, 2])
        graph = GraphStorage.from_edge_list(src, dst, num_nodes=3)

        features = np.random.randn(3, 16)
        graph.node_features['x'] = features

        assert 'x' in graph.node_features
        assert graph.node_features['x'].shape == (3, 16)

    def test_edge_data(self):
        """Test edge weight storage."""
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 0])
        weights = np.array([0.5, 0.3, 0.8])
        graph = GraphStorage.from_edge_list(src, dst, edge_data=weights)

        assert graph.edge_data is not None
        assert len(graph.edge_data) == 3

    def test_empty_graph(self):
        """Test empty graph handling."""
        src = np.array([], dtype=np.int64)
        dst = np.array([], dtype=np.int64)
        graph = GraphStorage.from_edge_list(src, dst, num_nodes=3)

        assert graph.num_nodes == 3
        assert graph.num_edges == 0


class TestPartitionedGraph:
    """Tests for PartitionedGraph class."""

    def test_balanced_partitioning(self):
        """Test balanced partitioning."""
        src = np.array([0, 1, 2, 3, 4, 5])
        dst = np.array([1, 2, 3, 4, 5, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        partitioned = PartitionedGraph(graph, num_partitions=2)
        partitioned.partition_balanced()

        sizes = partitioned.partition_sizes()
        assert len(sizes) == 2
        assert sum(sizes) == graph.num_nodes

    def test_metis_partitioning(self):
        """Test METIS-style partitioning."""
        src = np.array([0, 1, 2, 3, 4, 5, 0, 2, 4])
        dst = np.array([1, 2, 3, 4, 5, 0, 3, 4, 1])
        graph = GraphStorage.from_edge_list(src, dst)

        partitioned = PartitionedGraph(graph, num_partitions=2)
        partitioned.partition_metis()

        sizes = partitioned.partition_sizes()
        assert sum(sizes) == graph.num_nodes

    def test_get_partition_without_halo(self):
        """Test getting partition without halo nodes."""
        src = np.array([0, 1, 2, 3])
        dst = np.array([1, 2, 3, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        partitioned = PartitionedGraph(graph, num_partitions=2)
        partitioned.partition_balanced()

        subgraph, nodes, halo = partitioned.get_partition(0, include_halo=False)
        assert halo is None

    def test_get_partition_with_halo(self):
        """Test getting partition with halo nodes."""
        src = np.array([0, 1, 2, 3])
        dst = np.array([1, 2, 3, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        partitioned = PartitionedGraph(graph, num_partitions=2)
        partitioned.partition_balanced()

        subgraph, nodes, halo = partitioned.get_partition(0, include_halo=True)
        assert isinstance(halo, set)

    def test_edge_cut_ratio(self):
        """Test edge cut ratio calculation."""
        src = np.array([0, 1, 2, 3])
        dst = np.array([1, 2, 3, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        partitioned = PartitionedGraph(graph, num_partitions=2)
        partitioned.partition_balanced()

        ratio = partitioned.edge_cut_ratio()
        assert 0 <= ratio <= 1

    def test_halo_nodes(self):
        """Test halo node identification."""
        src = np.array([0, 1, 2, 3])
        dst = np.array([1, 2, 3, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        partitioned = PartitionedGraph(graph, num_partitions=2)
        partitioned.partition_balanced()

        halo = partitioned.get_halo_nodes(0)
        assert isinstance(halo, set)


class TestGraphFormat:
    """Tests for GraphFormat enum."""

    def test_formats(self):
        """Test format enum values."""
        assert GraphFormat.CSR.value == "csr"
        assert GraphFormat.CSC.value == "csc"
        assert GraphFormat.COO.value == "coo"
        assert GraphFormat.HYBRID.value == "hybrid"
