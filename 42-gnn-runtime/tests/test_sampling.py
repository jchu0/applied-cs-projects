"""Tests for sampling module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gnn_runtime.graph import GraphStorage

# Try to import torch-dependent classes
try:
    import torch
    from gnn_runtime.sampling import (
        NeighborSampler,
        PPRSampler,
        LayerSampler,
        ClusterSampler,
        SampledSubgraph,
    )
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def create_test_graph():
    """Create a simple test graph."""
    src = np.array([0, 0, 1, 1, 2, 2, 3, 4])
    dst = np.array([1, 2, 2, 3, 3, 4, 4, 0])
    return GraphStorage.from_edge_list(src, dst)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestNeighborSampler:
    """Tests for NeighborSampler class."""

    def test_basic_sampling(self):
        """Test basic neighbor sampling."""
        graph = create_test_graph()
        sampler = NeighborSampler(graph, fanouts=[2], device='cpu')

        seeds = torch.tensor([3])
        subgraph = sampler.sample(seeds)

        assert isinstance(subgraph, SampledSubgraph)
        assert len(subgraph.node_ids) >= 1
        assert 3 in subgraph.node_ids.tolist()

    def test_multi_hop_sampling(self):
        """Test multi-hop sampling."""
        graph = create_test_graph()
        sampler = NeighborSampler(graph, fanouts=[2, 2], device='cpu')

        seeds = torch.tensor([4])
        subgraph = sampler.sample(seeds)

        # Should have at least 2 layers
        assert len(subgraph.layer_sizes) >= 2

    def test_sampling_all_neighbors(self):
        """Test sampling all neighbors (fanout=-1)."""
        graph = create_test_graph()
        sampler = NeighborSampler(graph, fanouts=[-1], device='cpu')

        seeds = torch.tensor([0])
        subgraph = sampler.sample(seeds)

        # Should include all neighbors of node 0
        assert subgraph.edge_index.shape[1] >= 2

    def test_sampling_batch(self):
        """Test sampling with batch of seeds."""
        graph = create_test_graph()
        sampler = NeighborSampler(graph, fanouts=[2], device='cpu')

        seeds = torch.tensor([0, 1, 2])
        subgraph = sampler.sample(seeds)

        assert subgraph.batch_size == 3
        for seed in [0, 1, 2]:
            assert seed in subgraph.node_ids.tolist()

    def test_sampling_with_replacement(self):
        """Test sampling with replacement."""
        graph = create_test_graph()
        sampler = NeighborSampler(
            graph, fanouts=[10], replace=True, device='cpu'
        )

        seeds = torch.tensor([0])
        subgraph = sampler.sample(seeds)

        # Should work even with high fanout
        assert len(subgraph.node_ids) >= 1

    def test_node_mapping(self):
        """Test node ID mapping."""
        graph = create_test_graph()
        sampler = NeighborSampler(graph, fanouts=[2], device='cpu')

        seeds = torch.tensor([3, 4])
        subgraph = sampler.sample(seeds)

        # Check that mapping is consistent
        for global_id, local_id in subgraph.node_mapping.items():
            assert subgraph.node_ids[local_id].item() == global_id


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestPPRSampler:
    """Tests for PPRSampler class."""

    def test_ppr_sampling(self):
        """Test PPR-based sampling."""
        graph = create_test_graph()
        sampler = PPRSampler(graph, alpha=0.15)

        seeds = torch.tensor([0])
        subgraph = sampler.sample(seeds, top_k=3)

        assert len(subgraph.node_ids) <= 3
        assert 0 in subgraph.node_ids.tolist()

    def test_ppr_scores(self):
        """Test PPR score computation."""
        graph = create_test_graph()
        sampler = PPRSampler(graph, alpha=0.15)

        seeds = torch.tensor([0])
        scores = sampler.compute_ppr_scores(seeds)

        assert len(scores) == graph.num_nodes
        assert scores.sum() > 0
        # Seed node should have high score
        assert scores[0] > 0

    def test_ppr_convergence(self):
        """Test PPR convergence."""
        graph = create_test_graph()
        sampler = PPRSampler(graph, alpha=0.15, epsilon=1e-6, max_iters=200)

        seeds = torch.tensor([0, 1])
        scores = sampler.compute_ppr_scores(seeds)

        # Scores should sum approximately to 1
        assert 0.9 < scores.sum() < 1.1


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestLayerSampler:
    """Tests for LayerSampler class."""

    def test_layer_sampling(self):
        """Test layer-wise sampling."""
        graph = create_test_graph()
        sampler = LayerSampler(graph, num_layers=2, device='cpu')

        seeds = torch.tensor([0])
        subgraph = sampler.sample(seeds)

        assert len(subgraph.layer_sizes) >= 2

    def test_custom_fanouts(self):
        """Test layer sampling with custom fanouts."""
        graph = create_test_graph()
        sampler = LayerSampler(
            graph, num_layers=2, fanouts=[5, 3], device='cpu'
        )

        seeds = torch.tensor([0, 1])
        subgraph = sampler.sample(seeds)

        assert subgraph.batch_size == 2


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestClusterSampler:
    """Tests for ClusterSampler class."""

    def test_cluster_sampling(self):
        """Test cluster-based sampling."""
        # Create larger graph for clustering
        src = np.arange(100)
        dst = (np.arange(100) + 1) % 100
        graph = GraphStorage.from_edge_list(src, dst)

        sampler = ClusterSampler(graph, num_clusters=4, device='cpu')

        subgraph = sampler.sample_cluster(0)
        assert len(subgraph.node_ids) > 0

    def test_random_cluster_sampling(self):
        """Test random cluster sampling."""
        src = np.arange(100)
        dst = (np.arange(100) + 1) % 100
        graph = GraphStorage.from_edge_list(src, dst)

        sampler = ClusterSampler(graph, num_clusters=4, device='cpu')

        subgraph = sampler.sample_random_clusters(2)
        assert len(subgraph.node_ids) > 0


class TestSampledSubgraph:
    """Tests for SampledSubgraph dataclass."""

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_subgraph_attributes(self):
        """Test SampledSubgraph attributes."""
        subgraph = SampledSubgraph(
            node_ids=torch.tensor([0, 1, 2]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
            layer_sizes=[3],
            node_mapping={0: 0, 1: 1, 2: 2},
            batch_size=3,
        )

        assert len(subgraph.node_ids) == 3
        assert subgraph.edge_index.shape == (2, 2)
        assert subgraph.batch_size == 3
