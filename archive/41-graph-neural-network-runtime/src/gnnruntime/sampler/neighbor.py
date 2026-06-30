"""Neighbor sampling for scalable GNN training."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..core.graph import Graph, EdgeIndex

logger = logging.getLogger(__name__)


@dataclass
class SampledSubgraph:
    """Result of neighbor sampling."""
    x: np.ndarray  # Node features for sampled nodes
    edge_index: EdgeIndex  # Edges between sampled nodes
    node_idx: np.ndarray  # Original node indices
    batch_size: int  # Number of target nodes
    num_sampled_nodes: List[int]  # Nodes sampled at each hop


class NeighborSampler:
    """
    Mini-batch neighbor sampling for large graphs.

    Samples k-hop neighborhoods for scalable training on large graphs.
    Used by GraphSAGE and similar architectures.
    """

    def __init__(
        self,
        graph: Graph,
        num_neighbors: List[int],
        replace: bool = False
    ):
        """
        Args:
            graph: Input graph
            num_neighbors: Number of neighbors to sample at each hop
            replace: Whether to sample with replacement
        """
        self.graph = graph
        self.num_neighbors = num_neighbors
        self.replace = replace
        self.num_hops = len(num_neighbors)

        # Build adjacency lists for efficient sampling
        self._build_adj_lists()

    def _build_adj_lists(self):
        """Build adjacency lists from edge index."""
        self.adj_lists = [[] for _ in range(self.graph.num_nodes)]

        # Handle both EdgeIndex and numpy array
        edge_index = self.graph.edge_index
        if hasattr(edge_index, 'src'):
            src_arr = edge_index.src
            dst_arr = edge_index.dst
        else:
            src_arr = edge_index[0]
            dst_arr = edge_index[1]

        for i in range(self.graph.num_edges):
            src = src_arr[i]
            dst = dst_arr[i]
            self.adj_lists[dst].append(src)  # Incoming edges

        self.adj_lists = [np.array(adj) for adj in self.adj_lists]

    def sample(self, node_idx: np.ndarray) -> SampledSubgraph:
        """
        Sample k-hop neighborhood for given nodes.

        Args:
            node_idx: Target node indices

        Returns:
            SampledSubgraph with sampled nodes and edges
        """
        node_idx = np.asarray(node_idx)
        batch_size = len(node_idx)

        # Track all sampled nodes
        all_nodes = [node_idx]
        num_sampled = [len(node_idx)]

        # Sample hop by hop
        frontier = node_idx
        for hop in range(self.num_hops):
            num_neighbors = self.num_neighbors[hop]
            sampled = self._sample_neighbors(frontier, num_neighbors)
            all_nodes.append(sampled)
            num_sampled.append(len(sampled))
            frontier = sampled

        # Combine all sampled nodes
        combined = np.concatenate(all_nodes)
        unique_nodes = np.unique(combined)

        # Create node mapping
        mapping = np.full(self.graph.num_nodes, -1)
        mapping[unique_nodes] = np.arange(len(unique_nodes))

        # Extract subgraph edges
        # Handle both EdgeIndex and numpy array
        edge_index = self.graph.edge_index
        if hasattr(edge_index, 'src'):
            src_arr = edge_index.src
            dst_arr = edge_index.dst
        else:
            src_arr = edge_index[0]
            dst_arr = edge_index[1]

        src_list = []
        dst_list = []
        for i in range(self.graph.num_edges):
            src = src_arr[i]
            dst = dst_arr[i]
            if mapping[src] >= 0 and mapping[dst] >= 0:
                src_list.append(mapping[src])
                dst_list.append(mapping[dst])

        edge_index = EdgeIndex([np.array(src_list), np.array(dst_list)])

        # Get features
        x = self.graph.x[unique_nodes]

        return SampledSubgraph(
            x=x,
            edge_index=edge_index,
            node_idx=unique_nodes,
            batch_size=batch_size,
            num_sampled_nodes=num_sampled
        )

    def _sample_neighbors(self, nodes: np.ndarray, num_neighbors: int) -> np.ndarray:
        """Sample neighbors for a set of nodes."""
        sampled = []
        for node in nodes:
            neighbors = self.adj_lists[node]
            if len(neighbors) == 0:
                continue

            if len(neighbors) <= num_neighbors:
                sampled.extend(neighbors)
            else:
                idx = np.random.choice(
                    len(neighbors),
                    num_neighbors,
                    replace=self.replace
                )
                sampled.extend(neighbors[idx])

        return np.unique(sampled)


class ClusterSampler:
    """
    Cluster-based graph sampling.

    Partitions graph into clusters and samples cluster-induced subgraphs.
    Reduces between-cluster edges for efficient training.
    """

    def __init__(
        self,
        graph: Graph,
        num_clusters: int
    ):
        """
        Args:
            graph: Input graph
            num_clusters: Number of clusters
        """
        self.graph = graph
        self.num_clusters = num_clusters

        # Partition graph
        self._partition_graph()

    def _partition_graph(self):
        """Partition graph using random clustering."""
        # Simple random partition (production would use METIS)
        self.cluster_assignment = np.random.randint(
            0, self.num_clusters, self.graph.num_nodes
        )

        # Build cluster membership
        self.cluster_nodes = [
            np.where(self.cluster_assignment == i)[0]
            for i in range(self.num_clusters)
        ]

    def sample(self, cluster_idx: int) -> Graph:
        """
        Get subgraph for a cluster.

        Args:
            cluster_idx: Cluster index

        Returns:
            Subgraph induced by cluster nodes
        """
        nodes = self.cluster_nodes[cluster_idx]
        mask = np.zeros(self.graph.num_nodes, dtype=bool)
        mask[nodes] = True

        return self.graph.subgraph(mask)

    def __len__(self) -> int:
        return self.num_clusters

    def __iter__(self):
        indices = np.random.permutation(self.num_clusters)
        for i in indices:
            yield self.sample(i)


class GraphSAINTSampler:
    """
    GraphSAINT random walk sampler.

    Samples subgraphs using random walks to maintain edge distribution.
    Supports both Graph object and raw edge_index input.
    """

    def __init__(
        self,
        graph: Graph = None,
        edge_index: np.ndarray = None,
        num_nodes: int = None,
        walk_length: int = 2,
        num_steps: int = 1,
        budget: int = 1000,
        sample_coverage: int = 1,
        method: str = 'node'
    ):
        """
        Args:
            graph: Input graph (alternative to edge_index)
            edge_index: Edge connectivity array
            num_nodes: Number of nodes
            walk_length: Length of random walks
            num_steps: Number of random walk steps
            budget: Maximum number of nodes per subgraph
            sample_coverage: How many times each node should be sampled on average
            method: Sampling method ('node', 'edge', 'rw')
        """
        if graph is not None:
            self.num_nodes = graph.num_nodes
            self.edge_index = graph.edge_index
        else:
            self.edge_index = np.asarray(edge_index)
            self.num_nodes = num_nodes if num_nodes is not None else int(self.edge_index.max()) + 1

        self.graph = graph
        self.walk_length = walk_length
        self.num_steps = num_steps
        self.budget = budget
        self.sample_coverage = sample_coverage
        self.method = method

        # Build adjacency lists
        self._build_adj_lists()

    def _build_adj_lists(self):
        """Build adjacency lists for random walks."""
        self.adj_lists = [[] for _ in range(self.num_nodes)]

        if self.graph is not None:
            for i in range(self.graph.num_edges):
                src = self.graph.edge_index.src[i]
                dst = self.graph.edge_index.dst[i]
                self.adj_lists[src].append(dst)
        else:
            for i in range(self.edge_index.shape[1]):
                src = self.edge_index[0, i]
                dst = self.edge_index[1, i]
                self.adj_lists[src].append(dst)

        self.adj_lists = [np.array(adj) if adj else np.array([]) for adj in self.adj_lists]

    def sample_subgraphs(self, num_samples: int, size: int) -> List[dict]:
        """Sample multiple subgraphs."""
        return [self.sample_single_subgraph(size) for _ in range(num_samples)]

    def sample_single_subgraph(self, size: int) -> dict:
        """Sample a single subgraph."""
        if self.method == 'node':
            return self._sample_node_subgraph(size)
        elif self.method == 'edge':
            return self._sample_edge_subgraph(size)
        else:  # 'rw'
            return self._sample_rw_subgraph(size)

    def _sample_node_subgraph(self, size: int) -> dict:
        """Node sampling."""
        nodes = np.random.choice(self.num_nodes, min(size, self.num_nodes), replace=False)
        node_set = set(nodes.tolist())

        mask = np.array([
            self.edge_index[0, i] in node_set and self.edge_index[1, i] in node_set
            for i in range(self.edge_index.shape[1])
        ])

        return {'nodes': nodes, 'edges': self.edge_index[:, mask]}

    def _sample_edge_subgraph(self, size: int) -> dict:
        """Edge sampling."""
        num_edges = self.edge_index.shape[1]
        selected_edges = np.random.choice(num_edges, min(size, num_edges), replace=False)
        edges = self.edge_index[:, selected_edges]
        nodes = np.unique(edges)

        return {'nodes': nodes, 'edges': edges}

    def _sample_rw_subgraph(self, size: int) -> dict:
        """Random walk sampling."""
        visited = set()
        start_nodes = np.random.choice(self.num_nodes, min(size // self.walk_length, self.num_nodes), replace=False)

        for start in start_nodes:
            if len(visited) >= size:
                break

            current = start
            visited.add(current)

            for _ in range(self.walk_length):
                neighbors = self.adj_lists[current]
                if len(neighbors) == 0:
                    break
                current = np.random.choice(neighbors)
                visited.add(current)

                if len(visited) >= size:
                    break

        nodes = np.array(list(visited))
        node_set = visited

        mask = np.array([
            self.edge_index[0, i] in node_set and self.edge_index[1, i] in node_set
            for i in range(self.edge_index.shape[1])
        ])

        return {'nodes': nodes, 'edges': self.edge_index[:, mask]}

    def compute_norm_coefficients(self, subgraph: dict) -> dict:
        """Compute normalization coefficients for GraphSAINT loss."""
        nodes = subgraph['nodes']
        edges = subgraph['edges']

        # Node normalization: inverse of sampling probability
        node_norm = np.ones(len(nodes)) * (self.num_nodes / len(nodes))

        # Edge normalization
        num_edges = edges.shape[1] if edges.ndim > 1 else 0
        total_edges = self.edge_index.shape[1]
        edge_norm = np.ones(num_edges) * (total_edges / max(num_edges, 1))

        return {'node_norm': node_norm, 'edge_norm': edge_norm}

    def sample(self) -> Graph:
        """Sample a subgraph using random walks (legacy API)."""
        if self.graph is None:
            raise ValueError("Graph object required for sample() method")

        start_nodes = np.random.choice(
            self.num_nodes,
            min(self.budget // self.walk_length, self.num_nodes),
            replace=False
        )

        visited = set(start_nodes)
        for start in start_nodes:
            if len(visited) >= self.budget:
                break

            node = start
            for _ in range(self.walk_length):
                neighbors = self.adj_lists[node]
                if len(neighbors) == 0:
                    break
                node = np.random.choice(neighbors)
                visited.add(node)

                if len(visited) >= self.budget:
                    break

        nodes = np.array(list(visited))
        mask = np.zeros(self.num_nodes, dtype=bool)
        mask[nodes] = True

        return self.graph.subgraph(mask)


class ShaDowKHopSampler:
    """
    ShaDow-GNN k-hop subgraph sampler.

    Extracts k-hop ego-networks centered at target nodes.
    """

    def __init__(
        self,
        graph: Graph,
        depth: int = 2
    ):
        """
        Args:
            graph: Input graph
            depth: Number of hops
        """
        self.graph = graph
        self.depth = depth

        # Build adjacency lists
        self._build_adj_lists()

    def _build_adj_lists(self):
        """Build adjacency lists."""
        self.adj_lists = [set() for _ in range(self.graph.num_nodes)]
        for i in range(self.graph.num_edges):
            src = self.graph.edge_index.src[i]
            dst = self.graph.edge_index.dst[i]
            self.adj_lists[src].add(dst)
            self.adj_lists[dst].add(src)

    def sample(self, node_idx: int) -> Graph:
        """
        Extract k-hop subgraph around a node.

        Args:
            node_idx: Center node

        Returns:
            k-hop neighborhood subgraph
        """
        visited = {node_idx}
        frontier = {node_idx}

        for _ in range(self.depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in self.adj_lists[node]:
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)
            frontier = next_frontier

        # Create subgraph
        nodes = np.array(list(visited))
        mask = np.zeros(self.graph.num_nodes, dtype=bool)
        mask[nodes] = True

        return self.graph.subgraph(mask)


class LayerDependentSampler:
    """
    Layer-dependent sampling with importance sampling.

    Samples different number of neighbors at different layers
    based on node importance.
    """

    def __init__(
        self,
        graph: Graph,
        num_neighbors: List[int]
    ):
        """
        Args:
            graph: Input graph
            num_neighbors: Neighbors per layer
        """
        self.graph = graph
        self.num_neighbors = num_neighbors
        self.num_layers = len(num_neighbors)

        # Compute node importance (degree-based)
        self._compute_importance()
        self._build_adj_lists()

    def _compute_importance(self):
        """Compute node importance scores."""
        degree = self.graph.degree(direction='both')
        self.importance = degree / degree.sum()

    def _build_adj_lists(self):
        """Build adjacency lists with importance."""
        self.adj_lists = [[] for _ in range(self.graph.num_nodes)]
        self.adj_importance = [[] for _ in range(self.graph.num_nodes)]

        for i in range(self.graph.num_edges):
            src = self.graph.edge_index.src[i]
            dst = self.graph.edge_index.dst[i]
            self.adj_lists[dst].append(src)
            self.adj_importance[dst].append(self.importance[src])

        # Normalize probabilities
        for i in range(self.graph.num_nodes):
            self.adj_lists[i] = np.array(self.adj_lists[i])
            probs = np.array(self.adj_importance[i])
            if len(probs) > 0:
                self.adj_importance[i] = probs / probs.sum()
            else:
                self.adj_importance[i] = np.array([])

    def sample_layer(
        self,
        nodes: np.ndarray,
        layer: int
    ) -> Tuple[np.ndarray, EdgeIndex]:
        """
        Sample neighbors for one layer.

        Args:
            nodes: Current layer nodes
            layer: Layer index

        Returns:
            Tuple of (sampled_nodes, edges)
        """
        num_neighbors = self.num_neighbors[layer]
        sampled = []
        src_list = []
        dst_list = []

        for i, node in enumerate(nodes):
            neighbors = self.adj_lists[node]
            probs = self.adj_importance[node]

            if len(neighbors) == 0:
                continue

            # Importance sampling
            if len(neighbors) <= num_neighbors:
                selected = neighbors
            else:
                idx = np.random.choice(
                    len(neighbors),
                    num_neighbors,
                    replace=False,
                    p=probs
                )
                selected = neighbors[idx]

            sampled.extend(selected)
            for s in selected:
                src_list.append(s)
                dst_list.append(node)

        sampled_nodes = np.unique(sampled)
        edge_index = EdgeIndex([np.array(src_list), np.array(dst_list)])

        return sampled_nodes, edge_index


def random_walk(
    graph: Graph,
    start: int,
    walk_length: int
) -> np.ndarray:
    """
    Perform random walk on graph.

    Args:
        graph: Input graph
        start: Starting node
        walk_length: Length of walk

    Returns:
        Array of visited nodes
    """
    # Build adjacency list
    adj = [[] for _ in range(graph.num_nodes)]
    for i in range(graph.num_edges):
        src = graph.edge_index.src[i]
        dst = graph.edge_index.dst[i]
        adj[src].append(dst)

    walk = [start]
    current = start

    for _ in range(walk_length - 1):
        neighbors = adj[current]
        if not neighbors:
            break
        current = np.random.choice(neighbors)
        walk.append(current)

    return np.array(walk)


def node2vec_walk(
    graph: Graph,
    start: int,
    walk_length: int,
    p: float = 1.0,
    q: float = 1.0
) -> np.ndarray:
    """
    Node2vec biased random walk.

    Args:
        graph: Input graph
        start: Starting node
        walk_length: Length of walk
        p: Return parameter
        q: In-out parameter

    Returns:
        Array of visited nodes
    """
    # Build adjacency sets
    adj = [set() for _ in range(graph.num_nodes)]
    for i in range(graph.num_edges):
        src = graph.edge_index.src[i]
        dst = graph.edge_index.dst[i]
        adj[src].add(dst)

    walk = [start]
    if walk_length == 1:
        return np.array(walk)

    # First step (uniform)
    neighbors = list(adj[start])
    if not neighbors:
        return np.array(walk)

    current = np.random.choice(neighbors)
    walk.append(current)

    for _ in range(walk_length - 2):
        prev = walk[-2]
        neighbors = list(adj[current])
        if not neighbors:
            break

        # Compute transition probabilities
        probs = []
        for neighbor in neighbors:
            if neighbor == prev:
                probs.append(1.0 / p)  # Return
            elif neighbor in adj[prev]:
                probs.append(1.0)  # BFS
            else:
                probs.append(1.0 / q)  # DFS

        probs = np.array(probs)
        probs = probs / probs.sum()

        current = np.random.choice(neighbors, p=probs)
        walk.append(current)

    return np.array(walk)


class UniformSampler:
    """
    Uniform neighbor sampler for mini-batch GNN training.

    Samples neighbors uniformly at random for each layer.
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        num_neighbors,  # Can be int or List[int]
        num_nodes: int,
        replace: bool = False
    ):
        """
        Args:
            edge_index: Edge connectivity (2, num_edges)
            num_neighbors: Number of neighbors per hop (int for single hop, list for multi-hop)
            num_nodes: Total number of nodes
            replace: Sample with replacement
        """
        self.edge_index = np.asarray(edge_index)
        # Handle both int and list
        if isinstance(num_neighbors, int):
            self._num_neighbors_list = [num_neighbors]
        else:
            self._num_neighbors_list = list(num_neighbors)
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.replace = replace
        self.num_hops = len(self._num_neighbors_list)

        # Build adjacency lists (incoming edges)
        self._build_adj_lists()

    def _build_adj_lists(self):
        """Build adjacency lists from edge index."""
        self.adj_lists = [[] for _ in range(self.num_nodes)]
        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i]
            dst = self.edge_index[1, i]
            self.adj_lists[dst].append(src)  # Incoming edges for message passing

        self.adj_lists = [np.array(adj) for adj in self.adj_lists]

    def sample_neighbors(self, node_id: int, num_samples: int = None) -> np.ndarray:
        """
        Sample neighbors for a single node.

        Args:
            node_id: Node to sample neighbors for
            num_samples: Number of neighbors to sample (default: first num_neighbors value)

        Returns:
            Array of sampled neighbor indices
        """
        if num_samples is None:
            num_samples = self._num_neighbors_list[0] if isinstance(self.num_neighbors, list) else self.num_neighbors

        neighbors = self.adj_lists[node_id]
        if len(neighbors) == 0:
            return np.array([], dtype=np.int64)

        # With replacement, always sample exactly num_samples
        if self.replace:
            idx = np.random.choice(len(neighbors), num_samples, replace=True)
            return neighbors[idx]

        # Without replacement, sample min(neighbors, num_samples)
        if len(neighbors) <= num_samples:
            return neighbors

        idx = np.random.choice(len(neighbors), num_samples, replace=False)
        return neighbors[idx]

    def sample(self, target_nodes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample k-hop neighborhood for target nodes.

        Args:
            target_nodes: Target node indices

        Returns:
            Tuple of (sampled_nodes, sampled_edge_index)
        """
        target_nodes = np.asarray(target_nodes)
        all_nodes = set(target_nodes.tolist())

        src_list = []
        dst_list = []

        current_layer = target_nodes

        for hop in range(self.num_hops):
            num_neighbors = self._num_neighbors_list[hop]
            next_layer = set()

            for node in current_layer:
                neighbors = self.adj_lists[node]
                if len(neighbors) == 0:
                    continue

                if len(neighbors) <= num_neighbors:
                    sampled = neighbors
                else:
                    idx = np.random.choice(
                        len(neighbors),
                        num_neighbors,
                        replace=self.replace
                    )
                    sampled = neighbors[idx]

                for neighbor in sampled:
                    src_list.append(neighbor)
                    dst_list.append(node)
                    if neighbor not in all_nodes:
                        next_layer.add(neighbor)
                        all_nodes.add(neighbor)

            current_layer = np.array(list(next_layer))

        sampled_nodes = np.array(sorted(all_nodes))

        if src_list:
            sampled_edges = np.stack([np.array(src_list), np.array(dst_list)])
        else:
            sampled_edges = np.zeros((2, 0), dtype=np.int64)

        return sampled_nodes, sampled_edges


class LayerwiseSampler:
    """
    Layer-wise neighbor sampler with fixed fanout per layer.
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        fanouts: List[int] = None,
        num_nodes: int = None,
        num_layers: int = None,
        num_neighbors: int = None
    ):
        """
        Args:
            edge_index: Edge connectivity
            fanouts: Number of neighbors per layer (alternative to num_layers + num_neighbors)
            num_nodes: Total number of nodes
            num_layers: Number of layers (used with num_neighbors)
            num_neighbors: Neighbors per layer (used with num_layers)
        """
        self.edge_index = np.asarray(edge_index)

        # Support both fanouts list and num_layers + num_neighbors
        if fanouts is not None:
            self.fanouts = fanouts
        elif num_layers is not None and num_neighbors is not None:
            self.fanouts = [num_neighbors] * num_layers
        else:
            self.fanouts = [5]  # Default

        self.num_nodes = num_nodes if num_nodes is not None else int(edge_index.max()) + 1

        self._build_adj_lists()

    def _build_adj_lists(self):
        """Build adjacency lists."""
        self.adj_lists = [[] for _ in range(self.num_nodes)]
        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i]
            dst = self.edge_index[1, i]
            self.adj_lists[dst].append(src)

        self.adj_lists = [np.array(adj) for adj in self.adj_lists]

    def sample(self, seed_nodes: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Sample neighbors layer by layer.

        Returns list of (nodes, edges) per layer.
        """
        layers = []
        current_nodes = np.asarray(seed_nodes)

        for fanout in self.fanouts:
            src_list = []
            dst_list = []
            next_nodes = set()

            for node in current_nodes:
                neighbors = self.adj_lists[node]
                if len(neighbors) == 0:
                    continue

                if len(neighbors) <= fanout:
                    sampled = neighbors
                else:
                    idx = np.random.choice(len(neighbors), fanout, replace=False)
                    sampled = neighbors[idx]

                for neighbor in sampled:
                    src_list.append(neighbor)
                    dst_list.append(node)
                    next_nodes.add(neighbor)

            if src_list:
                edges = np.stack([np.array(src_list), np.array(dst_list)])
            else:
                edges = np.zeros((2, 0), dtype=np.int64)

            layers.append((current_nodes, edges))
            current_nodes = np.array(list(next_nodes))

        return layers

    def sample_layers(self, seed_nodes: List[int]) -> List[dict]:
        """
        Sample neighbors layer by layer.

        Returns list of dicts with 'nodes' and 'edges' per layer.
        """
        raw_layers = self.sample(np.asarray(seed_nodes))
        return [{'nodes': nodes, 'edges': edges} for nodes, edges in raw_layers]


class RandomWalkSampler:
    """Random walk based sampling for node embeddings."""

    def __init__(
        self,
        edge_index: np.ndarray,
        walk_length: int,
        num_nodes: int,
        num_walks: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        restart_prob: float = 0.0
    ):
        self.edge_index = np.asarray(edge_index)
        self.walk_length = walk_length
        self.num_nodes = num_nodes
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.restart_prob = restart_prob

        self._build_adj_lists()
        self._build_adj_sets()

    def _build_adj_lists(self):
        """Build adjacency lists."""
        self.adj_lists = [[] for _ in range(self.num_nodes)]
        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i]
            dst = self.edge_index[1, i]
            self.adj_lists[src].append(dst)

        self.adj_lists = [np.array(adj) if adj else np.array([]) for adj in self.adj_lists]

    def _build_adj_sets(self):
        """Build adjacency sets for node2vec."""
        self.adj_sets = [set(adj) for adj in self.adj_lists]

    def sample(self, start_nodes) -> List[np.ndarray]:
        """
        Generate random walks from start nodes.

        Returns: List of walks, one per start node
        """
        start_nodes = np.asarray(start_nodes)
        walks = []

        for node in start_nodes:
            walk = self._random_walk(node)
            walks.append(walk)

        return walks

    def _random_walk(self, start: int) -> np.ndarray:
        """Perform a single random walk."""
        walk = [start]
        current = start

        for _ in range(self.walk_length - 1):
            neighbors = self.adj_lists[current]
            if len(neighbors) == 0:
                break
            current = np.random.choice(neighbors)
            walk.append(current)

        return np.array(walk)

    def sample_node2vec(self, start_nodes: List[int], num_walks: int) -> List[np.ndarray]:
        """Generate Node2Vec biased random walks."""
        walks = []
        for node in start_nodes:
            for _ in range(num_walks):
                walk = self._node2vec_walk(node)
                walks.append(walk)
        return walks

    def _node2vec_walk(self, start: int) -> np.ndarray:
        """Perform Node2Vec biased random walk."""
        walk = [start]
        neighbors = self.adj_lists[start]
        if len(neighbors) == 0:
            return np.array(walk)

        current = np.random.choice(neighbors)
        walk.append(current)

        while len(walk) < self.walk_length:
            prev = walk[-2]
            neighbors = self.adj_lists[current]
            if len(neighbors) == 0:
                break

            probs = []
            for neighbor in neighbors:
                if neighbor == prev:
                    probs.append(1.0 / self.p)
                elif neighbor in self.adj_sets[prev]:
                    probs.append(1.0)
                else:
                    probs.append(1.0 / self.q)

            probs = np.array(probs)
            probs = probs / probs.sum()
            current = np.random.choice(neighbors, p=probs)
            walk.append(current)

        return np.array(walk)

    def sample_with_restart(self, start_node: int) -> np.ndarray:
        """Random walk with restart probability."""
        walk = [start_node]
        current = start_node

        for _ in range(self.walk_length - 1):
            if np.random.random() < self.restart_prob:
                current = start_node
            else:
                neighbors = self.adj_lists[current]
                if len(neighbors) == 0:
                    current = start_node
                else:
                    current = np.random.choice(neighbors)
            walk.append(current)

        return np.array(walk)


class ClusterGCNSampler:
    """Cluster-GCN sampling using graph partitioning."""

    def __init__(
        self,
        edge_index: np.ndarray,
        num_nodes: int,
        num_parts: int = 10,
        num_clusters: int = None,
        batch_size: int = 1
    ):
        self.edge_index = np.asarray(edge_index)
        self.num_nodes = num_nodes
        self.num_parts = num_parts if num_clusters is None else num_clusters
        self.batch_size = batch_size
        self._clusters = None

    def cluster_graph(self) -> List[np.ndarray]:
        """Partition graph into clusters."""
        # Balanced random partitioning
        indices = np.random.permutation(self.num_nodes)
        cluster_size = self.num_nodes // self.num_parts

        self._clusters = []
        for i in range(self.num_parts):
            start = i * cluster_size
            if i == self.num_parts - 1:
                end = self.num_nodes
            else:
                end = start + cluster_size
            self._clusters.append(indices[start:end])

        return self._clusters

    def sample_cluster_batch(self) -> dict:
        """Sample a batch of clusters."""
        if self._clusters is None:
            self.cluster_graph()

        selected = np.random.choice(self.num_parts, self.batch_size, replace=False)
        nodes = np.concatenate([self._clusters[i] for i in selected])
        node_set = set(nodes.tolist())

        mask = np.array([
            self.edge_index[0, i] in node_set and self.edge_index[1, i] in node_set
            for i in range(self.edge_index.shape[1])
        ])

        return {'nodes': nodes, 'edges': self.edge_index[:, mask]}

    def get_cluster_subgraph(self, cluster: np.ndarray, include_between: bool = False) -> dict:
        """Get subgraph for a cluster."""
        node_set = set(cluster.tolist())

        if include_between:
            # Include edges where at least one endpoint is in cluster
            mask = np.array([
                self.edge_index[0, i] in node_set or self.edge_index[1, i] in node_set
                for i in range(self.edge_index.shape[1])
            ])
        else:
            # Only edges within cluster
            mask = np.array([
                self.edge_index[0, i] in node_set and self.edge_index[1, i] in node_set
                for i in range(self.edge_index.shape[1])
            ])

        return {'nodes': cluster, 'edges': self.edge_index[:, mask]}

    def sample(self, cluster_ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Sample subgraph from specified clusters."""
        if self._clusters is None:
            self.cluster_graph()

        nodes = np.concatenate([self._clusters[c] for c in cluster_ids])
        node_set = set(nodes.tolist())

        mask = np.array([
            self.edge_index[0, i] in node_set and self.edge_index[1, i] in node_set
            for i in range(self.edge_index.shape[1])
        ])

        return nodes, self.edge_index[:, mask]

    def __iter__(self):
        """Iterate over clusters."""
        if self._clusters is None:
            self.cluster_graph()
        perm = np.random.permutation(self.num_parts)
        for cluster_id in perm:
            yield self.sample([cluster_id])


class ImportanceSampler:
    """Importance-based neighbor sampling."""

    def __init__(
        self,
        edge_index: np.ndarray,
        num_neighbors: int,
        num_nodes: int,
        edge_weights: Optional[np.ndarray] = None,
        node_importance: Optional[np.ndarray] = None
    ):
        self.edge_index = np.asarray(edge_index)
        self.num_neighbors = [num_neighbors] if isinstance(num_neighbors, int) else num_neighbors
        self.num_nodes = num_nodes
        self.edge_weights = edge_weights

        # Compute importance if not provided (degree-based)
        if node_importance is None:
            degree = np.zeros(num_nodes)
            np.add.at(degree, edge_index[0], 1)
            np.add.at(degree, edge_index[1], 1)
            self.importance = degree / (degree.sum() + 1e-8)
        else:
            self.importance = node_importance

        self._build_adj_lists()

    def _build_adj_lists(self):
        """Build adjacency lists with importance weights."""
        self.adj_lists = [[] for _ in range(self.num_nodes)]
        self.adj_weights = [[] for _ in range(self.num_nodes)]
        self.adj_edge_idx = [[] for _ in range(self.num_nodes)]

        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i]
            dst = self.edge_index[1, i]
            self.adj_lists[dst].append(src)
            self.adj_edge_idx[dst].append(i)
            if self.edge_weights is not None:
                self.adj_weights[dst].append(self.edge_weights[i])
            else:
                self.adj_weights[dst].append(self.importance[src])

        for i in range(self.num_nodes):
            self.adj_lists[i] = np.array(self.adj_lists[i])
            self.adj_edge_idx[i] = np.array(self.adj_edge_idx[i])
            weights = np.array(self.adj_weights[i])
            if len(weights) > 0:
                self.adj_weights[i] = weights / (weights.sum() + 1e-8)
            else:
                self.adj_weights[i] = weights

    def sample(self, target_nodes: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample using importance weights.

        Returns:
            Tuple of (sampled_nodes, sampled_edges, sample_weights)
        """
        target_nodes = np.asarray(target_nodes)
        all_nodes = set(target_nodes.tolist())

        src_list = []
        dst_list = []
        weight_list = []
        current_layer = target_nodes

        for num_neighbors in self.num_neighbors:
            next_layer = set()

            for node in current_layer:
                neighbors = self.adj_lists[node]
                weights = self.adj_weights[node]
                edge_indices = self.adj_edge_idx[node]

                if len(neighbors) == 0:
                    continue

                if len(neighbors) <= num_neighbors:
                    sampled = neighbors
                    sampled_weights = weights if len(weights) > 0 else np.ones(len(sampled))
                else:
                    idx = np.random.choice(
                        len(neighbors),
                        num_neighbors,
                        replace=False,
                        p=weights
                    )
                    sampled = neighbors[idx]
                    sampled_weights = weights[idx]

                for j, neighbor in enumerate(sampled):
                    src_list.append(neighbor)
                    dst_list.append(node)
                    weight_list.append(sampled_weights[j] if len(sampled_weights) > j else 1.0)
                    if neighbor not in all_nodes:
                        next_layer.add(neighbor)
                        all_nodes.add(neighbor)

            current_layer = np.array(list(next_layer))

        sampled_nodes = np.array(sorted(all_nodes))

        if src_list:
            sampled_edges = np.stack([np.array(src_list), np.array(dst_list)])
            sample_weights = np.array(weight_list)
        else:
            sampled_edges = np.zeros((2, 0), dtype=np.int64)
            sample_weights = np.array([])

        return sampled_nodes, sampled_edges, sample_weights


class AdaptiveSampler:
    """Adaptive neighbor sampling with various methods."""

    def __init__(
        self,
        edge_index: np.ndarray,
        num_nodes: int,
        num_neighbors: List[int] = None,
        temperature: float = 1.0,
        method: str = 'adaptive_k'
    ):
        self.edge_index = np.asarray(edge_index)
        self.num_nodes = num_nodes
        self.num_neighbors = num_neighbors or [10]
        self.temperature = temperature
        self.method = method

        # Initialize uniform sampling probabilities
        self.sample_probs = np.ones(num_nodes) / num_nodes

        self._build_adj_lists()
        self._degrees = None

    def _build_adj_lists(self):
        """Build adjacency lists."""
        self.adj_lists = [[] for _ in range(self.num_nodes)]
        for i in range(self.edge_index.shape[1]):
            src = self.edge_index[0, i]
            dst = self.edge_index[1, i]
            self.adj_lists[dst].append(src)

        self.adj_lists = [np.array(adj) for adj in self.adj_lists]

    def compute_degrees(self) -> np.ndarray:
        """Compute node degrees."""
        if self._degrees is None:
            self._degrees = np.zeros(self.num_nodes, dtype=np.int64)
            np.add.at(self._degrees, self.edge_index[0], 1)
            np.add.at(self._degrees, self.edge_index[1], 1)
        return self._degrees

    def compute_adaptive_k(
        self,
        target_nodes: List[int],
        degrees: np.ndarray,
        min_k: int = 2,
        max_k: int = 20,
        budget: int = 500
    ) -> dict:
        """Compute adaptive number of neighbors per node."""
        target_nodes = np.asarray(target_nodes)
        adaptive_k = {}

        # Inverse of degree for sampling (high degree -> fewer samples)
        inv_degrees = 1.0 / (degrees + 1)
        total_inv = inv_degrees[target_nodes].sum()

        for node in target_nodes:
            # Allocate budget proportionally to inverse degree
            proportion = inv_degrees[node] / total_inv
            k = int(budget * proportion)
            k = max(min_k, min(k, max_k))
            adaptive_k[node] = k

        return adaptive_k

    def sample_adaptive(
        self,
        target_nodes: List[int],
        node_features: np.ndarray,
        num_neighbors: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample with adaptive importance weights."""
        target_nodes = np.asarray(target_nodes)
        all_nodes = set(target_nodes.tolist())

        src_list = []
        dst_list = []
        weight_list = []

        for node in target_nodes:
            neighbors = self.adj_lists[node]
            if len(neighbors) == 0:
                continue

            # Compute importance based on feature variance
            if len(node_features.shape) > 1:
                neighbor_features = node_features[neighbors]
                variance = np.var(neighbor_features, axis=0).sum()
                weights = np.ones(len(neighbors)) * (1 + variance)
            else:
                weights = np.ones(len(neighbors))

            weights = weights / (weights.sum() + 1e-8)

            if len(neighbors) <= num_neighbors:
                sampled = neighbors
                sampled_weights = weights
            else:
                idx = np.random.choice(len(neighbors), num_neighbors, replace=False, p=weights)
                sampled = neighbors[idx]
                sampled_weights = weights[idx]

            for j, neighbor in enumerate(sampled):
                src_list.append(neighbor)
                dst_list.append(node)
                weight_list.append(sampled_weights[j])
                all_nodes.add(neighbor)

        sampled_nodes = np.array(sorted(all_nodes))
        sampled_edges = np.stack([np.array(src_list), np.array(dst_list)]) if src_list else np.zeros((2, 0))
        weights = np.array(weight_list)

        return sampled_nodes, sampled_edges, weights

    def sample_fastgcn(self, layer_sizes: List[int]) -> List[dict]:
        """FastGCN importance sampling."""
        degrees = self.compute_degrees()
        importance = degrees / (degrees.sum() + 1e-8)

        layers = []
        for size in layer_sizes:
            # Sample nodes proportional to degree
            probs = importance / (importance.sum() + 1e-8)
            nodes = np.random.choice(self.num_nodes, min(size, self.num_nodes), replace=False, p=probs)

            node_set = set(nodes.tolist())
            mask = np.array([
                self.edge_index[0, i] in node_set and self.edge_index[1, i] in node_set
                for i in range(self.edge_index.shape[1])
            ])

            layers.append({'nodes': nodes, 'edges': self.edge_index[:, mask]})

        return layers

    def sample_ladies(
        self,
        target_nodes: List[int],
        node_features: np.ndarray,
        num_layers: int,
        layer_sizes: List[int]
    ) -> List[dict]:
        """LADIES layer-dependent importance sampling."""
        layers = []
        current_nodes = np.asarray(target_nodes)

        for i, size in enumerate(layer_sizes):
            # Compute layer-dependent importance
            if len(node_features.shape) > 1:
                feature_norm = np.linalg.norm(node_features, axis=1)
            else:
                feature_norm = np.abs(node_features)

            importance = feature_norm / (feature_norm.sum() + 1e-8)

            # Sample nodes for this layer
            valid_probs = importance.copy()
            valid_probs = valid_probs / (valid_probs.sum() + 1e-8)

            nodes = np.random.choice(self.num_nodes, min(size, self.num_nodes), replace=False, p=valid_probs)

            node_set = set(nodes.tolist())
            mask = np.array([
                self.edge_index[0, i] in node_set and self.edge_index[1, i] in node_set
                for i in range(self.edge_index.shape[1])
            ])

            layers.append({
                'nodes': nodes,
                'edges': self.edge_index[:, mask],
                'importance_weights': importance[nodes]
            })

            current_nodes = nodes

        return layers

    def update_probs(self, node_losses: np.ndarray):
        """Update sampling probabilities based on node losses."""
        scaled = node_losses / self.temperature
        exp_losses = np.exp(scaled - scaled.max())
        self.sample_probs = exp_losses / exp_losses.sum()

    def sample(self, target_nodes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """Sample using adaptive probabilities."""
        target_nodes = np.asarray(target_nodes)
        all_nodes = set(target_nodes.tolist())

        src_list = []
        dst_list = []
        current_layer = target_nodes

        for num_neighbors in self.num_neighbors:
            next_layer = set()

            for node in current_layer:
                neighbors = self.adj_lists[node]

                if len(neighbors) == 0:
                    continue

                probs = self.sample_probs[neighbors]
                probs = probs / (probs.sum() + 1e-8)

                if len(neighbors) <= num_neighbors:
                    sampled = neighbors
                else:
                    idx = np.random.choice(len(neighbors), num_neighbors, replace=False, p=probs)
                    sampled = neighbors[idx]

                for neighbor in sampled:
                    src_list.append(neighbor)
                    dst_list.append(node)
                    if neighbor not in all_nodes:
                        next_layer.add(neighbor)
                        all_nodes.add(neighbor)

            current_layer = np.array(list(next_layer))

        sampled_nodes = np.array(sorted(all_nodes))
        sampled_edges = np.stack([np.array(src_list), np.array(dst_list)]) if src_list else np.zeros((2, 0))

        return sampled_nodes, sampled_edges
