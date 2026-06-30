"""Neighbor sampling for mini-batch GNN training."""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from .graph import GraphStorage


@dataclass
class SampledSubgraph:
    """
    Result of neighbor sampling.

    Attributes:
        node_ids: Node IDs in original graph.
        edge_index: Edge indices in sampled subgraph [2, E'].
        layer_sizes: Number of nodes at each layer (from target to source).
        node_mapping: Mapping from original node IDs to local IDs.
        batch_size: Number of target (seed) nodes.
    """
    node_ids: 'torch.Tensor'
    edge_index: 'torch.Tensor'
    layer_sizes: List[int]
    node_mapping: Dict[int, int]
    batch_size: int = 0


class NeighborSampler:
    """
    Multi-hop neighbor sampler for mini-batch GNN training.

    Samples k-hop neighborhoods around seed nodes for efficient
    mini-batch training on large graphs.
    """

    def __init__(
        self,
        graph: GraphStorage,
        fanouts: List[int],
        replace: bool = False,
        device: str = 'cpu',
    ):
        """
        Initialize neighbor sampler.

        Args:
            graph: Graph storage object.
            fanouts: Number of neighbors to sample per hop. Use -1 for all neighbors.
            replace: Whether to sample with replacement.
            device: Device for output tensors.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for NeighborSampler")

        self.graph = graph
        self.fanouts = fanouts
        self.replace = replace
        self.device = device

        # Convert graph to torch tensors for efficient GPU sampling
        self.csr_indptr = torch.from_numpy(graph.csr_indptr).to(device)
        self.csr_indices = torch.from_numpy(graph.csr_indices).to(device)

    def sample(self, seed_nodes: 'torch.Tensor') -> SampledSubgraph:
        """
        Sample k-hop neighborhood around seed nodes.

        Args:
            seed_nodes: Target nodes for this batch [B].

        Returns:
            SampledSubgraph with all info needed for mini-batch GNN.
        """
        seed_nodes = seed_nodes.to(self.device)
        batch_nodes = seed_nodes.clone()
        layer_sizes = [len(seed_nodes)]
        all_edges_src = []
        all_edges_dst = []

        frontier = seed_nodes
        sampled_nodes: Set[int] = set(seed_nodes.cpu().numpy().tolist())

        # Sample each hop (from target layer to source layers)
        for fanout in self.fanouts:
            neighbors, edges_src, edges_dst = self._sample_neighbors(frontier, fanout)

            # Add new nodes to batch
            new_nodes = [n for n in neighbors.cpu().numpy() if n not in sampled_nodes]
            if new_nodes:
                new_nodes_tensor = torch.tensor(new_nodes, device=self.device, dtype=torch.long)
                batch_nodes = torch.cat([batch_nodes, new_nodes_tensor])
                sampled_nodes.update(new_nodes)

            all_edges_src.append(edges_src)
            all_edges_dst.append(edges_dst)

            # Update frontier for next hop
            frontier = neighbors
            layer_sizes.append(len(batch_nodes))

        # Build edge index with local node IDs
        node_mapping = {n.item(): i for i, n in enumerate(batch_nodes)}

        if all_edges_src:
            edges_src = torch.cat(all_edges_src)
            edges_dst = torch.cat(all_edges_dst)

            # Map to local IDs
            local_src = torch.tensor(
                [node_mapping[s.item()] for s in edges_src],
                device=self.device, dtype=torch.long
            )
            local_dst = torch.tensor(
                [node_mapping[d.item()] for d in edges_dst],
                device=self.device, dtype=torch.long
            )
            edge_index = torch.stack([local_src, local_dst])
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)

        return SampledSubgraph(
            node_ids=batch_nodes,
            edge_index=edge_index,
            layer_sizes=layer_sizes,
            node_mapping=node_mapping,
            batch_size=len(seed_nodes),
        )

    def _sample_neighbors(
        self,
        nodes: 'torch.Tensor',
        fanout: int,
    ) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
        """
        Sample neighbors for a set of nodes.

        Args:
            nodes: Nodes to sample neighbors for.
            fanout: Number of neighbors to sample. -1 for all.

        Returns:
            Tuple of (unique_neighbors, edge_sources, edge_destinations).
        """
        all_neighbors = []
        edges_src = []
        edges_dst = []

        for node in nodes:
            node_id = node.item()
            start = self.csr_indptr[node_id].item()
            end = self.csr_indptr[node_id + 1].item()
            degree = end - start

            if degree == 0:
                continue

            neighbors = self.csr_indices[start:end]

            if fanout == -1 or degree <= fanout:
                # Take all neighbors
                sampled = neighbors
            else:
                # Random sample
                if self.replace:
                    idx = torch.randint(0, degree, (fanout,), device=self.device)
                else:
                    idx = torch.randperm(degree, device=self.device)[:fanout]
                sampled = neighbors[idx]

            all_neighbors.append(sampled)
            # Edges go from neighbor (src) to node (dst) for message passing
            edges_src.extend(sampled.tolist())
            edges_dst.extend([node_id] * len(sampled))

        if all_neighbors:
            neighbors = torch.cat(all_neighbors)
            neighbors = torch.unique(neighbors)
            edges_src = torch.tensor(edges_src, device=self.device, dtype=torch.long)
            edges_dst = torch.tensor(edges_dst, device=self.device, dtype=torch.long)
        else:
            neighbors = torch.tensor([], dtype=torch.long, device=self.device)
            edges_src = torch.tensor([], dtype=torch.long, device=self.device)
            edges_dst = torch.tensor([], dtype=torch.long, device=self.device)

        return neighbors, edges_src, edges_dst


class PPRSampler:
    """
    Personalized PageRank based sampling.

    Samples nodes based on PPR scores from seed nodes, capturing
    important structural information beyond k-hop neighborhoods.
    """

    def __init__(
        self,
        graph: GraphStorage,
        alpha: float = 0.15,
        epsilon: float = 1e-5,
        max_iters: int = 100,
    ):
        """
        Initialize PPR sampler.

        Args:
            graph: Graph storage object.
            alpha: Teleport probability.
            epsilon: Convergence threshold.
            max_iters: Maximum power iterations.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for PPRSampler")

        self.graph = graph
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iters = max_iters

    def sample(
        self,
        seed_nodes: 'torch.Tensor',
        top_k: int = 100,
    ) -> SampledSubgraph:
        """
        Sample nodes using PPR scores from seed nodes.

        Args:
            seed_nodes: Seed nodes for PPR computation.
            top_k: Number of top-scoring nodes to include.

        Returns:
            Sampled subgraph based on PPR scores.
        """
        num_nodes = self.graph.num_nodes
        ppr_scores = np.zeros(num_nodes, dtype=np.float64)

        # Initialize with seed nodes
        seed_list = seed_nodes.cpu().numpy()
        for seed in seed_list:
            ppr_scores[seed] = 1.0 / len(seed_list)

        # Power iteration
        for iteration in range(self.max_iters):
            new_scores = np.zeros(num_nodes, dtype=np.float64)

            for node in range(num_nodes):
                neighbors = self.graph.get_neighbors(node, 'out')
                if len(neighbors) > 0:
                    # Distribute score to neighbors
                    contrib = (1 - self.alpha) * ppr_scores[node] / len(neighbors)
                    for neighbor in neighbors:
                        new_scores[neighbor] += contrib

            # Add teleport probability
            for seed in seed_list:
                new_scores[seed] += self.alpha / len(seed_list)

            # Check convergence
            diff = np.abs(new_scores - ppr_scores).sum()
            ppr_scores = new_scores

            if diff < self.epsilon:
                break

        # Select top-k nodes
        top_nodes = np.argsort(ppr_scores)[-top_k:][::-1].copy()

        # Build subgraph from selected nodes
        node_set = set(top_nodes)
        edges_src = []
        edges_dst = []

        for node in top_nodes:
            for neighbor in self.graph.get_neighbors(node, 'out'):
                if neighbor in node_set:
                    edges_src.append(node)
                    edges_dst.append(neighbor)

        node_mapping = {int(n): i for i, n in enumerate(top_nodes)}

        device = seed_nodes.device
        edge_index_np = np.array([
            [node_mapping[s] for s in edges_src],
            [node_mapping[d] for d in edges_dst]
        ]) if edges_src else np.zeros((2, 0), dtype=np.int64)

        return SampledSubgraph(
            node_ids=torch.tensor(top_nodes, device=device, dtype=torch.long),
            edge_index=torch.tensor(edge_index_np, device=device, dtype=torch.long),
            layer_sizes=[len(top_nodes)],
            node_mapping=node_mapping,
            batch_size=len(seed_nodes),
        )

    def compute_ppr_scores(
        self,
        seed_nodes: 'torch.Tensor',
    ) -> np.ndarray:
        """
        Compute full PPR scores from seed nodes.

        Args:
            seed_nodes: Seed nodes for PPR.

        Returns:
            PPR scores for all nodes [num_nodes].
        """
        num_nodes = self.graph.num_nodes
        ppr_scores = np.zeros(num_nodes, dtype=np.float64)

        seed_list = seed_nodes.cpu().numpy()
        for seed in seed_list:
            ppr_scores[seed] = 1.0 / len(seed_list)

        for _ in range(self.max_iters):
            new_scores = np.zeros(num_nodes, dtype=np.float64)

            for node in range(num_nodes):
                neighbors = self.graph.get_neighbors(node, 'out')
                if len(neighbors) > 0:
                    contrib = (1 - self.alpha) * ppr_scores[node] / len(neighbors)
                    for neighbor in neighbors:
                        new_scores[neighbor] += contrib

            for seed in seed_list:
                new_scores[seed] += self.alpha / len(seed_list)

            diff = np.abs(new_scores - ppr_scores).sum()
            ppr_scores = new_scores

            if diff < self.epsilon:
                break

        return ppr_scores


class LayerSampler:
    """
    Layer-wise sampling for deep GNNs.

    Samples different number of neighbors at each layer,
    typically more at deeper layers.
    """

    def __init__(
        self,
        graph: GraphStorage,
        num_layers: int,
        fanouts: Optional[List[int]] = None,
        device: str = 'cpu',
    ):
        """
        Initialize layer sampler.

        Args:
            graph: Graph storage object.
            num_layers: Number of GNN layers.
            fanouts: Fanouts per layer. Defaults to [25, 10, ...].
            device: Device for output tensors.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for LayerSampler")

        self.graph = graph
        self.num_layers = num_layers
        self.device = device

        if fanouts is None:
            # Default: decreasing fanouts for deeper layers
            self.fanouts = [max(25 // (2 ** i), 5) for i in range(num_layers)]
        else:
            self.fanouts = fanouts

        self.sampler = NeighborSampler(graph, self.fanouts, device=device)

    def sample(self, seed_nodes: 'torch.Tensor') -> SampledSubgraph:
        """Sample using layer-wise fanouts."""
        return self.sampler.sample(seed_nodes)


class ClusterSampler:
    """
    Cluster-based sampling for large-scale GNN training.

    Partitions graph into clusters and samples entire clusters,
    useful for very large graphs.
    """

    def __init__(
        self,
        graph: GraphStorage,
        num_clusters: int,
        device: str = 'cpu',
    ):
        """
        Initialize cluster sampler.

        Args:
            graph: Graph storage object.
            num_clusters: Number of clusters to partition into.
            device: Device for output tensors.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for ClusterSampler")

        self.graph = graph
        self.num_clusters = num_clusters
        self.device = device

        # Compute clusters (simple balanced partitioning)
        self.clusters = self._compute_clusters()

    def _compute_clusters(self) -> List[np.ndarray]:
        """Compute graph clusters."""
        nodes_per_cluster = self.graph.num_nodes // self.num_clusters
        clusters = []

        for i in range(self.num_clusters):
            start = i * nodes_per_cluster
            if i == self.num_clusters - 1:
                # Last cluster gets remaining nodes
                end = self.graph.num_nodes
            else:
                end = start + nodes_per_cluster
            clusters.append(np.arange(start, end))

        return clusters

    def sample_cluster(self, cluster_idx: int) -> SampledSubgraph:
        """
        Sample a single cluster as a subgraph.

        Args:
            cluster_idx: Index of cluster to sample.

        Returns:
            Sampled subgraph for the cluster.
        """
        node_ids = self.clusters[cluster_idx]
        subgraph = self.graph.subgraph(node_ids)

        node_mapping = {int(n): i for i, n in enumerate(node_ids)}
        edge_index = subgraph.to_edge_index()

        return SampledSubgraph(
            node_ids=torch.tensor(node_ids, device=self.device, dtype=torch.long),
            edge_index=torch.tensor(edge_index, device=self.device, dtype=torch.long),
            layer_sizes=[len(node_ids)],
            node_mapping=node_mapping,
            batch_size=len(node_ids),
        )

    def sample_random_clusters(self, num_clusters: int) -> SampledSubgraph:
        """
        Sample multiple random clusters combined.

        Args:
            num_clusters: Number of clusters to sample.

        Returns:
            Sampled subgraph combining multiple clusters.
        """
        selected = np.random.choice(self.num_clusters, num_clusters, replace=False)
        node_ids = np.concatenate([self.clusters[i] for i in selected])

        subgraph = self.graph.subgraph(node_ids)
        node_mapping = {int(n): i for i, n in enumerate(node_ids)}
        edge_index = subgraph.to_edge_index()

        return SampledSubgraph(
            node_ids=torch.tensor(node_ids, device=self.device, dtype=torch.long),
            edge_index=torch.tensor(edge_index, device=self.device, dtype=torch.long),
            layer_sizes=[len(node_ids)],
            node_mapping=node_mapping,
            batch_size=len(node_ids),
        )
