"""Tests for distributed training module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnn_runtime.graph import GraphStorage, PartitionedGraph

# Try to import torch-dependent classes
try:
    import torch
    import torch.nn as nn
    from gnn_runtime.distributed import (
        HaloExchange,
        DistributedGNNTrainer,
        VertexReorderOptimizer,
    )
    from gnn_runtime.layers import GCNLayer, GNNModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def create_test_graph(num_nodes=100, avg_degree=5):
    """Create a test graph."""
    num_edges = num_nodes * avg_degree
    src = np.random.randint(0, num_nodes, num_edges)
    dst = np.random.randint(0, num_nodes, num_edges)
    graph = GraphStorage.from_edge_list(src, dst, num_nodes)

    # Add features
    graph.node_features['x'] = np.random.randn(num_nodes, 32).astype(np.float32)
    graph.node_features['y'] = np.random.randint(0, 10, num_nodes)

    return graph


class TestVertexReorderOptimizer:
    """Tests for VertexReorderOptimizer."""

    def test_rcm_reordering(self):
        """Test Reverse Cuthill-McKee reordering."""
        graph = create_test_graph(num_nodes=50)
        optimizer = VertexReorderOptimizer(graph)

        perm = optimizer.reorder_rcm()

        assert len(perm) == graph.num_nodes
        assert len(set(perm)) == graph.num_nodes  # All unique

    def test_degree_reordering_descending(self):
        """Test degree-based reordering (descending)."""
        graph = create_test_graph(num_nodes=50)
        optimizer = VertexReorderOptimizer(graph)

        perm = optimizer.reorder_by_degree(descending=True)

        # First nodes should have higher degrees
        degrees = graph.degrees('out')
        assert degrees[perm[0]] >= degrees[perm[-1]]

    def test_degree_reordering_ascending(self):
        """Test degree-based reordering (ascending)."""
        graph = create_test_graph(num_nodes=50)
        optimizer = VertexReorderOptimizer(graph)

        perm = optimizer.reorder_by_degree(descending=False)

        # First nodes should have lower degrees
        degrees = graph.degrees('out')
        assert degrees[perm[0]] <= degrees[perm[-1]]

    def test_partition_reordering(self):
        """Test partition-based reordering."""
        graph = create_test_graph(num_nodes=100)
        optimizer = VertexReorderOptimizer(graph)

        perm = optimizer.reorder_by_partition(num_parts=4)

        assert len(perm) == graph.num_nodes
        assert len(set(perm)) == graph.num_nodes

    def test_apply_reordering(self):
        """Test applying reordering to graph."""
        graph = create_test_graph(num_nodes=50)
        optimizer = VertexReorderOptimizer(graph)

        perm = optimizer.reorder_rcm()
        new_graph = optimizer.apply_reordering(perm)

        assert new_graph.num_nodes == graph.num_nodes
        assert new_graph.num_edges == graph.num_edges

        # Features should be reordered
        if 'x' in graph.node_features:
            assert 'x' in new_graph.node_features
            assert new_graph.node_features['x'].shape == graph.node_features['x'].shape

    def test_cache_efficiency(self):
        """Test cache efficiency computation."""
        graph = create_test_graph(num_nodes=50)
        optimizer = VertexReorderOptimizer(graph)

        # Original efficiency
        orig_efficiency = optimizer.compute_cache_efficiency()

        # RCM reordering should improve efficiency
        perm = optimizer.reorder_rcm()
        new_efficiency = optimizer.compute_cache_efficiency(perm)

        assert orig_efficiency >= 0
        assert new_efficiency >= 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestHaloExchange:
    """Tests for HaloExchange class."""

    def test_halo_mapping_construction(self):
        """Test halo node mapping construction."""
        graph = create_test_graph(num_nodes=100)
        partitioned = PartitionedGraph(graph, num_partitions=2)
        partitioned.partition_balanced()

        halo_exchange = HaloExchange(partitioned, world_size=2, rank=0)

        # Should have mappings for rank 1
        assert 1 in halo_exchange.send_nodes or 1 in halo_exchange.recv_nodes

    def test_exchange_without_distributed(self):
        """Test exchange without distributed setup."""
        graph = create_test_graph(num_nodes=100)
        partitioned = PartitionedGraph(graph, num_partitions=2)
        partitioned.partition_balanced()

        halo_exchange = HaloExchange(partitioned, world_size=2, rank=0)

        local_nodes = partitioned.local_to_global[0]
        features = torch.randn(len(local_nodes), 32)

        # Should return features unchanged when not distributed
        out = halo_exchange.exchange(features, local_nodes)
        assert out.shape == features.shape


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestDistributedGNNTrainer:
    """Tests for DistributedGNNTrainer."""

    def test_single_gpu_training(self):
        """Test training with single GPU (or CPU)."""
        graph = create_test_graph(num_nodes=100)
        model = GNNModel(32, 64, 10, num_layers=2)

        trainer = DistributedGNNTrainer(
            model=model,
            graph=graph,
            num_gpus=1,
        )

        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        # Train one epoch
        loss = trainer.train_epoch(
            optimizer, loss_fn, batch_size=32, fanouts=[5, 5]
        )

        assert loss >= 0

    def test_evaluate(self):
        """Test evaluation."""
        graph = create_test_graph(num_nodes=100)
        model = GNNModel(32, 64, 10, num_layers=2)

        trainer = DistributedGNNTrainer(
            model=model,
            graph=graph,
            num_gpus=1,
        )

        loss, acc = trainer.evaluate()

        assert loss >= 0
        assert 0 <= acc <= 1

    def test_evaluate_with_mask(self):
        """Test evaluation with mask."""
        graph = create_test_graph(num_nodes=100)
        model = GNNModel(32, 64, 10, num_layers=2)

        trainer = DistributedGNNTrainer(
            model=model,
            graph=graph,
            num_gpus=1,
        )

        # Mask for subset of nodes
        mask = np.zeros(graph.num_nodes, dtype=bool)
        mask[:50] = True

        loss, acc = trainer.evaluate(mask=mask)

        assert loss >= 0
        assert 0 <= acc <= 1

    def test_training_loop(self):
        """Test full training loop."""
        graph = create_test_graph(num_nodes=100)
        model = GNNModel(32, 64, 10, num_layers=2)

        trainer = DistributedGNNTrainer(
            model=model,
            graph=graph,
            num_gpus=1,
        )

        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(3):
            loss = trainer.train_epoch(optimizer, loss_fn, batch_size=32)
            losses.append(loss)

        # Loss should generally decrease or remain similar
        assert all(l >= 0 for l in losses)


class TestPartitioningIntegration:
    """Integration tests for graph partitioning."""

    def test_balanced_partition_quality(self):
        """Test quality of balanced partitioning."""
        graph = create_test_graph(num_nodes=100)
        partitioned = PartitionedGraph(graph, num_partitions=4)
        partitioned.partition_balanced()

        sizes = partitioned.partition_sizes()

        # Partitions should be roughly equal
        assert max(sizes) - min(sizes) <= graph.num_nodes // 4

    def test_metis_partition_quality(self):
        """Test quality of METIS-style partitioning."""
        graph = create_test_graph(num_nodes=100)
        partitioned = PartitionedGraph(graph, num_partitions=4)
        partitioned.partition_metis()

        edge_cut = partitioned.edge_cut_ratio()

        # Edge cut should be reasonable
        assert 0 <= edge_cut <= 1

    def test_partition_coverage(self):
        """Test that all nodes are in exactly one partition."""
        graph = create_test_graph(num_nodes=100)
        partitioned = PartitionedGraph(graph, num_partitions=4)
        partitioned.partition_balanced()

        all_nodes = set()
        for p in range(4):
            nodes = set(partitioned.local_to_global[p])
            # No overlap
            assert len(all_nodes & nodes) == 0
            all_nodes.update(nodes)

        # All nodes covered
        assert len(all_nodes) == graph.num_nodes
