"""Graph storage engine with CSR/CSC formats and partitioning."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Union
from enum import Enum
import numpy as np


class GraphFormat(Enum):
    """Supported graph storage formats."""
    CSR = "csr"       # Compressed Sparse Row (efficient for outgoing edges)
    CSC = "csc"       # Compressed Sparse Column (efficient for incoming edges)
    COO = "coo"       # Coordinate format (flexible, easy construction)
    HYBRID = "hybrid" # CSR + CSC for bidirectional access


@dataclass
class GraphStorage:
    """
    Core graph storage with multiple format support.

    Supports CSR (row-major) and CSC (column-major) formats for efficient
    neighbor queries in both directions.

    Attributes:
        num_nodes: Number of nodes in the graph.
        num_edges: Number of edges in the graph.
        csr_indptr: CSR row pointers [num_nodes + 1].
        csr_indices: CSR column indices [num_edges].
        csc_indptr: Optional CSC column pointers [num_nodes + 1].
        csc_indices: Optional CSC row indices [num_edges].
        edge_data: Optional edge weights [num_edges].
        node_features: Dict mapping feature names to [num_nodes, feature_dim] arrays.
        edge_features: Dict mapping feature names to [num_edges, feature_dim] arrays.
    """
    num_nodes: int
    num_edges: int

    # CSR format (row-major, efficient for outgoing edges)
    csr_indptr: np.ndarray    # [num_nodes + 1]
    csr_indices: np.ndarray   # [num_edges]

    # CSC format (col-major, efficient for incoming edges)
    csc_indptr: Optional[np.ndarray] = None
    csc_indices: Optional[np.ndarray] = None

    # Edge data
    edge_data: Optional[np.ndarray] = None

    # Feature storage
    node_features: Dict[str, np.ndarray] = field(default_factory=dict)
    edge_features: Dict[str, np.ndarray] = field(default_factory=dict)

    # Partition info
    partition_ids: Optional[np.ndarray] = None
    num_partitions: int = 1

    @classmethod
    def from_edge_list(
        cls,
        src: np.ndarray,
        dst: np.ndarray,
        num_nodes: Optional[int] = None,
        edge_data: Optional[np.ndarray] = None,
        directed: bool = True,
    ) -> 'GraphStorage':
        """
        Construct graph from edge list.

        Args:
            src: Source node IDs [num_edges].
            dst: Destination node IDs [num_edges].
            num_nodes: Number of nodes (inferred if None).
            edge_data: Optional edge weights [num_edges].
            directed: Whether the graph is directed.

        Returns:
            GraphStorage instance with CSR format.
        """
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)

        if num_nodes is None:
            num_nodes = max(src.max(), dst.max()) + 1 if len(src) > 0 else 0

        num_edges = len(src)

        if not directed:
            # Add reverse edges
            src, dst = np.concatenate([src, dst]), np.concatenate([dst, src])
            if edge_data is not None:
                edge_data = np.concatenate([edge_data, edge_data])
            num_edges = len(src)

        # Build CSR format
        csr_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        for s in src:
            csr_indptr[s + 1] += 1
        csr_indptr = np.cumsum(csr_indptr)

        # Sort edges by source node
        sort_idx = np.argsort(src)
        csr_indices = dst[sort_idx].astype(np.int64)

        if edge_data is not None:
            edge_data = edge_data[sort_idx]

        return cls(
            num_nodes=num_nodes,
            num_edges=num_edges,
            csr_indptr=csr_indptr,
            csr_indices=csr_indices,
            edge_data=edge_data,
        )

    @classmethod
    def from_adjacency_matrix(
        cls,
        adj: np.ndarray,
        threshold: float = 0.0,
    ) -> 'GraphStorage':
        """
        Construct graph from dense adjacency matrix.

        Args:
            adj: Dense adjacency matrix [num_nodes, num_nodes].
            threshold: Minimum value to create an edge.

        Returns:
            GraphStorage instance.
        """
        src, dst = np.where(adj > threshold)
        edge_data = adj[src, dst] if not np.allclose(adj[src, dst], 1.0) else None
        return cls.from_edge_list(src, dst, adj.shape[0], edge_data)

    def to_csc(self) -> None:
        """Build CSC format from CSR (transpose)."""
        if self.csc_indptr is not None:
            return  # Already built

        # Count incoming edges per node
        csc_indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
        for dst in self.csr_indices:
            csc_indptr[dst + 1] += 1
        csc_indptr = np.cumsum(csc_indptr)

        # Fill CSC indices
        csc_indices = np.zeros(self.num_edges, dtype=np.int64)
        counts = np.zeros(self.num_nodes, dtype=np.int64)

        for src in range(self.num_nodes):
            for idx in range(self.csr_indptr[src], self.csr_indptr[src + 1]):
                dst = self.csr_indices[idx]
                pos = csc_indptr[dst] + counts[dst]
                csc_indices[pos] = src
                counts[dst] += 1

        self.csc_indptr = csc_indptr
        self.csc_indices = csc_indices

    def to_coo(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to COO format (edge list)."""
        src = []
        dst = []
        for node in range(self.num_nodes):
            start = self.csr_indptr[node]
            end = self.csr_indptr[node + 1]
            neighbors = self.csr_indices[start:end]
            src.extend([node] * len(neighbors))
            dst.extend(neighbors)
        return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)

    def get_neighbors(self, node_id: int, direction: str = 'out') -> np.ndarray:
        """
        Get neighbors of a node.

        Args:
            node_id: Node ID to query.
            direction: 'out' for outgoing, 'in' for incoming neighbors.

        Returns:
            Array of neighbor node IDs.
        """
        if direction == 'out':
            start = self.csr_indptr[node_id]
            end = self.csr_indptr[node_id + 1]
            return self.csr_indices[start:end].copy()
        else:  # incoming
            if self.csc_indptr is None:
                self.to_csc()
            start = self.csc_indptr[node_id]
            end = self.csc_indptr[node_id + 1]
            return self.csc_indices[start:end].copy()

    def degree(self, node_id: int, direction: str = 'out') -> int:
        """
        Get degree of a node.

        Args:
            node_id: Node ID to query.
            direction: 'out' for out-degree, 'in' for in-degree.

        Returns:
            Degree of the node.
        """
        if direction == 'out':
            return int(self.csr_indptr[node_id + 1] - self.csr_indptr[node_id])
        else:
            if self.csc_indptr is None:
                self.to_csc()
            return int(self.csc_indptr[node_id + 1] - self.csc_indptr[node_id])

    def degrees(self, direction: str = 'out') -> np.ndarray:
        """
        Get degrees of all nodes.

        Args:
            direction: 'out' for out-degree, 'in' for in-degree.

        Returns:
            Array of degrees [num_nodes].
        """
        if direction == 'out':
            return np.diff(self.csr_indptr)
        else:
            if self.csc_indptr is None:
                self.to_csc()
            return np.diff(self.csc_indptr)

    def has_edge(self, src: int, dst: int) -> bool:
        """Check if edge exists."""
        neighbors = self.get_neighbors(src, 'out')
        return dst in neighbors

    def add_self_loops(self) -> 'GraphStorage':
        """Return new graph with self-loops added."""
        src, dst = self.to_coo()

        # Find nodes without self-loops
        self_loop_nodes = set(src[src == dst])
        missing = [i for i in range(self.num_nodes) if i not in self_loop_nodes]

        if missing:
            src = np.concatenate([src, np.array(missing, dtype=np.int64)])
            dst = np.concatenate([dst, np.array(missing, dtype=np.int64)])

        new_graph = GraphStorage.from_edge_list(src, dst, self.num_nodes)
        new_graph.node_features = self.node_features.copy()
        return new_graph

    def subgraph(self, node_ids: np.ndarray) -> 'GraphStorage':
        """
        Extract induced subgraph.

        Args:
            node_ids: Node IDs to include in subgraph.

        Returns:
            New GraphStorage for the subgraph.
        """
        node_set = set(node_ids)
        node_map = {n: i for i, n in enumerate(node_ids)}

        edges_src = []
        edges_dst = []

        for node in node_ids:
            for neighbor in self.get_neighbors(node, 'out'):
                if neighbor in node_set:
                    edges_src.append(node_map[node])
                    edges_dst.append(node_map[neighbor])

        subgraph = GraphStorage.from_edge_list(
            np.array(edges_src, dtype=np.int64),
            np.array(edges_dst, dtype=np.int64),
            num_nodes=len(node_ids),
        )

        # Copy features for selected nodes
        for key, feat in self.node_features.items():
            subgraph.node_features[key] = feat[node_ids]

        return subgraph

    def to_edge_index(self) -> np.ndarray:
        """Convert to PyG-style edge_index [2, num_edges]."""
        src, dst = self.to_coo()
        return np.stack([src, dst])


class PartitionedGraph:
    """
    Graph partitioned across multiple devices/machines.

    Supports balanced partitioning with halo (ghost) nodes for
    distributed GNN training.
    """

    def __init__(self, graph: GraphStorage, num_partitions: int):
        self.global_graph = graph
        self.num_partitions = num_partitions
        self.partitions: List[GraphStorage] = []
        self.node_to_partition: np.ndarray = np.zeros(graph.num_nodes, dtype=np.int32)
        self.local_to_global: List[np.ndarray] = []
        self.global_to_local: np.ndarray = np.zeros(graph.num_nodes, dtype=np.int64)
        self._partitioned = False

    def partition_balanced(self) -> None:
        """Perform balanced (round-robin) partitioning."""
        nodes_per_partition = self.global_graph.num_nodes // self.num_partitions

        for i in range(self.global_graph.num_nodes):
            partition_id = min(i // nodes_per_partition, self.num_partitions - 1)
            self.node_to_partition[i] = partition_id

        self._build_local_mappings()
        self._partitioned = True

    def partition_metis(self) -> None:
        """
        Partition using METIS-style algorithm.

        This simplified implementation does balanced partitioning.
        For production, use actual METIS via pymetis.
        """
        # Simple degree-aware partitioning
        degrees = self.global_graph.degrees('out')
        sorted_nodes = np.argsort(degrees)[::-1]  # High degree first

        partition_sizes = np.zeros(self.num_partitions, dtype=np.int64)
        target_size = self.global_graph.num_nodes // self.num_partitions

        for node in sorted_nodes:
            # Assign to smallest partition
            best_partition = np.argmin(partition_sizes)
            self.node_to_partition[node] = best_partition
            partition_sizes[best_partition] += 1

        self._build_local_mappings()
        self._partitioned = True

    def _build_local_mappings(self) -> None:
        """Build local-to-global and global-to-local mappings."""
        self.local_to_global = []

        for p in range(self.num_partitions):
            local_nodes = np.where(self.node_to_partition == p)[0]
            self.local_to_global.append(local_nodes)

            for local_idx, global_idx in enumerate(local_nodes):
                self.global_to_local[global_idx] = local_idx

    def get_partition(
        self,
        partition_id: int,
        include_halo: bool = True,
    ) -> Tuple[GraphStorage, np.ndarray, Optional[Set[int]]]:
        """
        Get subgraph for a partition with optional halo nodes.

        Args:
            partition_id: ID of the partition.
            include_halo: Whether to include halo (ghost) nodes.

        Returns:
            Tuple of (subgraph, all_node_ids, halo_node_ids).
        """
        if not self._partitioned:
            raise RuntimeError("Graph not partitioned. Call partition_balanced() or partition_metis() first.")

        local_nodes = self.local_to_global[partition_id]

        if not include_halo:
            subgraph = self.global_graph.subgraph(local_nodes)
            return subgraph, local_nodes, None

        # Find halo nodes (neighbors in other partitions)
        halo_nodes: Set[int] = set()
        for node in local_nodes:
            neighbors = self.global_graph.get_neighbors(node, 'out')
            for neighbor in neighbors:
                if self.node_to_partition[neighbor] != partition_id:
                    halo_nodes.add(int(neighbor))

        # Combine local and halo nodes
        all_nodes = np.concatenate([local_nodes, np.array(list(halo_nodes), dtype=np.int64)])

        subgraph = self.global_graph.subgraph(all_nodes)
        return subgraph, all_nodes, halo_nodes

    def get_halo_nodes(self, partition_id: int) -> Set[int]:
        """Get halo nodes for a partition."""
        if not self._partitioned:
            raise RuntimeError("Graph not partitioned.")

        local_nodes = set(self.local_to_global[partition_id])
        halo_nodes: Set[int] = set()

        for node in local_nodes:
            neighbors = self.global_graph.get_neighbors(node, 'out')
            for neighbor in neighbors:
                if self.node_to_partition[neighbor] != partition_id:
                    halo_nodes.add(int(neighbor))

        return halo_nodes

    def edge_cut_ratio(self) -> float:
        """Calculate edge cut ratio (fraction of edges crossing partitions)."""
        if not self._partitioned:
            return 0.0

        cut_edges = 0
        src, dst = self.global_graph.to_coo()

        for s, d in zip(src, dst):
            if self.node_to_partition[s] != self.node_to_partition[d]:
                cut_edges += 1

        return cut_edges / self.global_graph.num_edges if self.global_graph.num_edges > 0 else 0.0

    def partition_sizes(self) -> np.ndarray:
        """Get number of nodes in each partition."""
        sizes = np.zeros(self.num_partitions, dtype=np.int64)
        for p in range(self.num_partitions):
            sizes[p] = len(self.local_to_global[p])
        return sizes
