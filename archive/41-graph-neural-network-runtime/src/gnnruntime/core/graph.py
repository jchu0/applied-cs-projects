"""Core graph data structures for GNN runtime."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class EdgeIndex:
    """
    Edge index representation for sparse graphs.

    Stores edges as (source, target) pairs in COO format.
    """

    def __init__(self, data):
        """
        Initialize EdgeIndex from various formats.

        Args:
            data: Can be:
                - Tuple/list of (src, dst) arrays
                - 2D array of shape (2, num_edges)
                - Two separate arrays for src and dst
        """
        if isinstance(data, (list, tuple)) and len(data) == 2:
            self.src = np.asarray(data[0])
            self.dst = np.asarray(data[1])
        elif isinstance(data, np.ndarray):
            if data.shape[0] == 2:
                self.src = np.asarray(data[0])
                self.dst = np.asarray(data[1])
            else:
                raise ValueError(f"Invalid EdgeIndex shape: {data.shape}")
        else:
            raise ValueError(f"Cannot create EdgeIndex from {type(data)}")

        assert len(self.src) == len(self.dst), "Source and destination must have same length"

    @property
    def num_edges(self) -> int:
        return len(self.src)

    @property
    def shape(self):
        """Return shape as (2, num_edges) tuple."""
        return (2, len(self.src))

    def t(self) -> 'EdgeIndex':
        """Transpose (alias for transpose method)."""
        return self.transpose()

    def __getitem__(self, idx):
        """Index into edge_index like a 2D array."""
        if isinstance(idx, int):
            if idx == 0:
                return self.src
            elif idx == 1:
                return self.dst
            else:
                raise IndexError(f"Index {idx} out of range for EdgeIndex")
        return np.stack([self.src, self.dst])[idx]

    def coalesce(self) -> 'EdgeIndex':
        """Remove duplicate edges and sort."""
        if self.num_edges == 0:
            return EdgeIndex([self.src.copy(), self.dst.copy()])

        edges = np.stack([self.src, self.dst], axis=1)
        unique_edges = np.unique(edges, axis=0)
        return EdgeIndex([unique_edges[:, 0], unique_edges[:, 1]])

    def cat(self, other: 'EdgeIndex') -> 'EdgeIndex':
        """Concatenate with another EdgeIndex."""
        new_src = np.concatenate([self.src, other.src])
        new_dst = np.concatenate([self.dst, other.dst])
        return EdgeIndex([new_src, new_dst])

    def to_dense_adj(self, num_nodes: int = None) -> np.ndarray:
        """Convert to dense adjacency matrix."""
        if num_nodes is None:
            num_nodes = max(self.src.max(), self.dst.max()) + 1 if self.num_edges > 0 else 0
        return self.to_dense(num_nodes)

    def to_dense(self, num_nodes: int) -> np.ndarray:
        """Convert to dense adjacency matrix."""
        adj = np.zeros((num_nodes, num_nodes))
        adj[self.src, self.dst] = 1
        return adj

    @classmethod
    def from_dense(cls, adj: np.ndarray) -> 'EdgeIndex':
        """Create from dense adjacency matrix."""
        src, dst = np.nonzero(adj)
        return cls([src, dst])

    def transpose(self) -> 'EdgeIndex':
        """Get transposed edge index (reverse edges)."""
        return EdgeIndex([self.dst.copy(), self.src.copy()])

    def add_self_loops(self, num_nodes: int) -> 'EdgeIndex':
        """Add self-loops to the graph."""
        loop_index = np.arange(num_nodes)
        src = np.concatenate([self.src, loop_index])
        dst = np.concatenate([self.dst, loop_index])
        return EdgeIndex([src, dst])

    def remove_self_loops(self) -> 'EdgeIndex':
        """Remove self-loops from the graph."""
        mask = self.src != self.dst
        return EdgeIndex([self.src[mask], self.dst[mask]])


class Graph:
    """
    Graph data structure for GNN operations.

    Supports node features, edge features, and various graph operations.
    Accepts edge_index as numpy array (2, num_edges) format.
    """

    def __init__(
        self,
        x: Optional[np.ndarray] = None,
        edge_index: Optional[np.ndarray] = None,
        edge_attr: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        num_nodes: Optional[int] = None,
        batch: Optional[np.ndarray] = None,
        ptr: Optional[np.ndarray] = None,
        **kwargs
    ):
        # Handle edge_index - can be numpy array or EdgeIndex
        if edge_index is not None:
            if isinstance(edge_index, EdgeIndex):
                self.edge_index = np.stack([edge_index.src, edge_index.dst])
            else:
                self.edge_index = np.asarray(edge_index)
                if len(self.edge_index.shape) == 1:
                    # Assume flattened format
                    half = len(self.edge_index) // 2
                    self.edge_index = np.stack([self.edge_index[:half], self.edge_index[half:]])
        else:
            self.edge_index = np.zeros((2, 0), dtype=np.int64)

        # Handle node features
        if x is not None:
            self.x = np.asarray(x)
        else:
            self.x = None

        # Determine number of nodes
        if num_nodes is not None:
            self._num_nodes = num_nodes
        elif self.x is not None:
            self._num_nodes = self.x.shape[0]
        elif self.edge_index.shape[1] > 0:
            self._num_nodes = int(self.edge_index.max()) + 1
        else:
            self._num_nodes = 0

        # Validate edge indices
        if self.edge_index.shape[1] > 0 and self._num_nodes > 0:
            if self.edge_index.max() >= self._num_nodes:
                raise ValueError(f"Edge index contains node {self.edge_index.max()} but num_nodes is {self._num_nodes}")

        self.edge_attr = edge_attr
        self.y = y
        self.batch = batch
        self.ptr = ptr

        # Store any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def num_features(self) -> int:
        if self.x is None:
            return 0
        return self.x.shape[1] if len(self.x.shape) > 1 else 1

    @property
    def num_graphs(self) -> int:
        if self.batch is None:
            return 1
        return len(np.unique(self.batch))

    def degree(self, direction: str = 'in') -> np.ndarray:
        """Compute node degrees."""
        deg = np.zeros(self.num_nodes, dtype=np.int64)
        if direction == 'in':
            np.add.at(deg, self.edge_index[1], 1)
        elif direction == 'out':
            np.add.at(deg, self.edge_index[0], 1)
        else:
            np.add.at(deg, self.edge_index[0], 1)
            np.add.at(deg, self.edge_index[1], 1)
        return deg

    def is_undirected(self) -> bool:
        """Check if graph is undirected."""
        edge_set = set(zip(self.edge_index[0].tolist(), self.edge_index[1].tolist()))
        for src, dst in edge_set:
            if src != dst and (dst, src) not in edge_set:
                return False
        return True

    def add_self_loops(self) -> 'Graph':
        """Add self-loops to the graph."""
        loop_index = np.arange(self.num_nodes)
        new_edge_index = np.concatenate([
            self.edge_index,
            np.stack([loop_index, loop_index])
        ], axis=1)

        return Graph(
            x=self.x.copy() if self.x is not None else None,
            edge_index=new_edge_index,
            edge_attr=None,  # Can't preserve edge_attr with new edges
            y=self.y.copy() if self.y is not None else None,
            num_nodes=self.num_nodes
        )

    def remove_self_loops(self) -> 'Graph':
        """Remove self-loops from the graph."""
        mask = self.edge_index[0] != self.edge_index[1]
        new_edge_index = self.edge_index[:, mask]

        return Graph(
            x=self.x.copy() if self.x is not None else None,
            edge_index=new_edge_index,
            edge_attr=self.edge_attr[mask] if self.edge_attr is not None else None,
            y=self.y.copy() if self.y is not None else None,
            num_nodes=self.num_nodes
        )

    def to_undirected(self) -> 'Graph':
        """Convert to undirected graph by adding reverse edges."""
        src, dst = self.edge_index[0], self.edge_index[1]
        new_src = np.concatenate([src, dst])
        new_dst = np.concatenate([dst, src])

        # Remove duplicates
        edges = np.stack([new_src, new_dst], axis=1)
        edges = np.unique(edges, axis=0)
        new_edge_index = edges.T

        return Graph(
            x=self.x.copy() if self.x is not None else None,
            edge_index=new_edge_index,
            edge_attr=None,
            y=self.y.copy() if self.y is not None else None,
            num_nodes=self.num_nodes
        )

    def neighbors(self, node_idx: int) -> np.ndarray:
        """Get neighbors of a node."""
        mask = self.edge_index[0] == node_idx
        return self.edge_index[1, mask]

    def subgraph(self, node_idx: Union[np.ndarray, List[int]]) -> 'Graph':
        """Extract subgraph containing specified nodes.

        Args:
            node_idx: Node indices (array or list) or boolean mask
        """
        node_idx = np.asarray(node_idx)

        # Handle boolean mask
        if node_idx.dtype == bool:
            node_idx = np.where(node_idx)[0]

        # Create node mask and mapping
        node_mask = np.zeros(self.num_nodes, dtype=bool)
        node_mask[node_idx] = True

        mapping = np.full(self.num_nodes, -1)
        mapping[node_idx] = np.arange(len(node_idx))

        # Filter edges
        edge_mask = node_mask[self.edge_index[0]] & node_mask[self.edge_index[1]]
        new_edge_index = np.stack([
            mapping[self.edge_index[0, edge_mask]],
            mapping[self.edge_index[1, edge_mask]]
        ])

        return Graph(
            x=self.x[node_idx] if self.x is not None else None,
            edge_index=new_edge_index,
            edge_attr=self.edge_attr[edge_mask] if self.edge_attr is not None else None,
            y=self.y[node_idx] if self.y is not None else None,
            num_nodes=len(node_idx)
        )

    def clone(self) -> 'Graph':
        """Create a deep copy of the graph."""
        return Graph(
            x=self.x.copy() if self.x is not None else None,
            edge_index=self.edge_index.copy(),
            edge_attr=self.edge_attr.copy() if self.edge_attr is not None else None,
            y=self.y.copy() if self.y is not None else None,
            num_nodes=self.num_nodes,
            batch=self.batch.copy() if self.batch is not None else None,
            ptr=self.ptr.copy() if self.ptr is not None else None
        )

    def coalesce(self) -> 'Graph':
        """Remove duplicate edges and sort."""
        if self.num_edges == 0:
            return self.clone()

        # Get unique edges
        edges = np.stack([self.edge_index[0], self.edge_index[1]], axis=1)
        unique_edges, indices = np.unique(edges, axis=0, return_index=True)

        new_edge_index = unique_edges.T

        return Graph(
            x=self.x.copy() if self.x is not None else None,
            edge_index=new_edge_index,
            edge_attr=self.edge_attr[indices] if self.edge_attr is not None else None,
            y=self.y.copy() if self.y is not None else None,
            num_nodes=self.num_nodes
        )


class MessagePassing:
    """
    Base class for message passing neural networks.

    Implements the aggregate-update paradigm:
    1. Message: compute messages on edges
    2. Aggregate: aggregate messages at nodes
    3. Update: update node features
    """

    def __init__(self, aggr: str = 'add'):
        """
        Args:
            aggr: Aggregation method ('add', 'mean', 'max')
        """
        self.aggr = aggr

    def propagate(
        self,
        edge_index: EdgeIndex,
        x: np.ndarray,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Propagate messages through the graph.

        Args:
            edge_index: Graph connectivity
            x: Node features
            edge_attr: Edge features

        Returns:
            Aggregated messages at each node
        """
        num_nodes = x.shape[0]

        # Get source and target features
        x_i = x[edge_index.dst]  # Target node features
        x_j = x[edge_index.src]  # Source node features

        # Compute messages
        messages = self.message(x_j, x_i, edge_attr)

        # Aggregate messages
        out = self.aggregate(messages, edge_index.dst, num_nodes)

        # Update
        out = self.update(out)

        return out

    def message(
        self,
        x_j: np.ndarray,
        x_i: np.ndarray,
        edge_attr: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Compute messages from source to target.

        Args:
            x_j: Source node features
            x_i: Target node features
            edge_attr: Edge features

        Returns:
            Messages for each edge
        """
        return x_j

    def aggregate(
        self,
        messages: np.ndarray,
        index: np.ndarray,
        num_nodes: int
    ) -> np.ndarray:
        """
        Aggregate messages at nodes.

        Args:
            messages: Messages for each edge
            index: Target node indices
            num_nodes: Total number of nodes

        Returns:
            Aggregated features for each node
        """
        out = np.zeros((num_nodes, messages.shape[1]))

        if self.aggr == 'add':
            np.add.at(out, index, messages)
        elif self.aggr == 'mean':
            np.add.at(out, index, messages)
            count = np.zeros(num_nodes)
            np.add.at(count, index, 1)
            count = np.maximum(count, 1)
            out = out / count[:, None]
        elif self.aggr == 'max':
            # Initialize with -inf
            out = np.full((num_nodes, messages.shape[1]), -np.inf)
            for i, idx in enumerate(index):
                out[idx] = np.maximum(out[idx], messages[i])
            out = np.where(np.isinf(out), 0, out)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggr}")

        return out

    def update(self, aggr_out: np.ndarray) -> np.ndarray:
        """
        Update node features after aggregation.

        Args:
            aggr_out: Aggregated messages

        Returns:
            Updated node features
        """
        return aggr_out


def normalize_adj(edge_index, num_nodes: int, add_self_loops: bool = True) -> Tuple[EdgeIndex, np.ndarray]:
    """
    Compute normalized adjacency matrix coefficients.

    Implements symmetric normalization: D^{-1/2} A D^{-1/2}

    Args:
        edge_index: Graph connectivity (EdgeIndex or numpy array)
        num_nodes: Number of nodes
        add_self_loops: Whether to add self-loops

    Returns:
        Tuple of (edge_index, edge_weight)
    """
    # Convert to EdgeIndex if needed
    if isinstance(edge_index, np.ndarray):
        edge_index = EdgeIndex([edge_index[0], edge_index[1]])

    if add_self_loops:
        edge_index = edge_index.add_self_loops(num_nodes)

    # Compute degree
    deg = np.zeros(num_nodes)
    np.add.at(deg, edge_index.dst, 1)

    # D^{-1/2}
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0

    # Normalize
    edge_weight = deg_inv_sqrt[edge_index.src] * deg_inv_sqrt[edge_index.dst]

    return edge_index, edge_weight


def gcn_norm(edge_index, num_nodes: int) -> Tuple[EdgeIndex, np.ndarray]:
    """
    GCN-style normalization (symmetric).

    Args:
        edge_index: Graph connectivity (EdgeIndex or numpy array)
        num_nodes: Number of nodes

    Returns:
        Tuple of (edge_index with self-loops, edge_weight)
    """
    return normalize_adj(edge_index, num_nodes, add_self_loops=True)


def compute_attention_weights(
    query: np.ndarray,
    key: np.ndarray,
    edge_index: EdgeIndex
) -> np.ndarray:
    """
    Compute attention weights for edges.

    Args:
        query: Query vectors (num_nodes, dim)
        key: Key vectors (num_nodes, dim)
        edge_index: Graph connectivity

    Returns:
        Attention weights for each edge
    """
    q = query[edge_index.dst]  # Target queries
    k = key[edge_index.src]    # Source keys

    # Dot product attention
    attn = np.sum(q * k, axis=-1)

    # Softmax over neighbors
    # Group by target node
    num_nodes = query.shape[0]
    exp_attn = np.exp(attn - np.max(attn))

    # Sum of exp for each target node
    sum_exp = np.zeros(num_nodes)
    np.add.at(sum_exp, edge_index.dst, exp_attn)

    # Normalize
    weights = exp_attn / sum_exp[edge_index.dst]

    return weights


class SparseTensor:
    """
    Sparse tensor representation using COO format.

    Efficient for sparse matrix operations in GNNs.
    """

    def __init__(
        self,
        row: np.ndarray,
        col: np.ndarray,
        value: Optional[np.ndarray] = None,
        sparse_sizes: Optional[Tuple[int, int]] = None
    ):
        self.row = np.asarray(row)
        self.col = np.asarray(col)

        if value is None:
            self.value = np.ones(len(row))
        else:
            self.value = np.asarray(value)

        if sparse_sizes is None:
            self.sparse_sizes = (int(row.max()) + 1, int(col.max()) + 1)
        else:
            self.sparse_sizes = sparse_sizes

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return len(self.row)

    def to_dense(self) -> np.ndarray:
        """Convert to dense matrix."""
        dense = np.zeros(self.sparse_sizes)
        dense[self.row, self.col] = self.value
        return dense

    def matmul(self, other: np.ndarray) -> np.ndarray:
        """Sparse-dense matrix multiplication."""
        # A @ B where A is sparse
        out = np.zeros((self.sparse_sizes[0], other.shape[1]))

        for i in range(self.nnz):
            out[self.row[i]] += self.value[i] * other[self.col[i]]

        return out

    def transpose(self) -> 'SparseTensor':
        """Transpose the sparse tensor."""
        return SparseTensor(
            self.col.copy(),
            self.row.copy(),
            self.value.copy(),
            (self.sparse_sizes[1], self.sparse_sizes[0])
        )

    @classmethod
    def from_edge_index(
        cls,
        edge_index: EdgeIndex,
        edge_weight: Optional[np.ndarray] = None,
        num_nodes: int = None
    ) -> 'SparseTensor':
        """Create from edge index."""
        if num_nodes is None:
            num_nodes = max(edge_index.src.max(), edge_index.dst.max()) + 1

        return cls(
            edge_index.src,
            edge_index.dst,
            edge_weight,
            (num_nodes, num_nodes)
        )


class HeteroGraph:
    """
    Heterogeneous graph with multiple node and edge types.

    Supports different node types with different feature dimensions
    and multiple relation types between node types.
    """

    def __init__(
        self,
        x_dict: Dict[str, np.ndarray],
        edge_index_dict: Dict[Tuple[str, str, str], np.ndarray],
        node_types: Optional[Dict[str, int]] = None,
        edge_attr_dict: Optional[Dict[Tuple[str, str, str], np.ndarray]] = None
    ):
        self.x_dict = x_dict
        self.edge_index_dict = edge_index_dict
        self.edge_attr_dict = edge_attr_dict

        # node_types can be dict (node_type -> count) or list
        if node_types is None:
            self._node_types = list(x_dict.keys())
            self._node_counts = {nt: x_dict[nt].shape[0] for nt in self._node_types}
        elif isinstance(node_types, dict):
            self._node_types = list(node_types.keys())
            self._node_counts = node_types
        else:
            self._node_types = list(node_types)
            self._node_counts = {nt: x_dict[nt].shape[0] for nt in self._node_types}

    @property
    def node_types(self) -> List[str]:
        """Get list of node types."""
        return self._node_types

    @property
    def edge_types(self) -> List[Tuple[str, str, str]]:
        """Get list of edge types."""
        return list(self.edge_index_dict.keys())

    @property
    def num_node_types(self) -> int:
        return len(self._node_types)

    @property
    def num_edge_types(self) -> int:
        return len(self.edge_index_dict)

    def num_nodes(self, node_type: Optional[str] = None) -> int:
        """Get number of nodes of a type or total."""
        if node_type is None:
            return sum(self._node_counts.values())
        return self._node_counts.get(node_type, self.x_dict[node_type].shape[0])

    def num_edges(self, edge_type: Optional[Tuple[str, str, str]] = None) -> int:
        """Get number of edges of a type or total."""
        if edge_type is None:
            return sum(e.shape[1] for e in self.edge_index_dict.values())
        return self.edge_index_dict[edge_type].shape[1]

    def get_node_features(self, node_type: str) -> np.ndarray:
        """Get features for a node type."""
        return self.x_dict[node_type]

    def get_edge_index(self, edge_type: Tuple[str, str, str]) -> np.ndarray:
        """Get edge index for an edge type."""
        return self.edge_index_dict[edge_type]

    def edge_type_subgraph(self, edge_type: Tuple[str, str, str]) -> 'HeteroGraph':
        """Get subgraph for a specific edge type."""
        src_type, rel, dst_type = edge_type
        node_types_subset = {src_type, dst_type}

        new_x_dict = {nt: self.x_dict[nt] for nt in node_types_subset if nt in self.x_dict}
        new_edge_dict = {edge_type: self.edge_index_dict[edge_type]}
        new_node_counts = {nt: self._node_counts[nt] for nt in node_types_subset if nt in self._node_counts}

        return HeteroGraph(new_x_dict, new_edge_dict, new_node_counts)

    def to_homogeneous(self) -> 'Graph':
        """Convert to homogeneous graph."""
        # Compute node offsets
        offsets = {}
        current_offset = 0
        for node_type in self._node_types:
            offsets[node_type] = current_offset
            current_offset += self.num_nodes(node_type)

        total_nodes = current_offset

        # Combine features (zero-pad to max dim)
        max_dim = max(x.shape[1] if len(x.shape) > 1 else 1 for x in self.x_dict.values())
        combined_x = np.zeros((total_nodes, max_dim))
        for node_type in self._node_types:
            if node_type in self.x_dict:
                x = self.x_dict[node_type]
                if len(x.shape) == 1:
                    x = x.reshape(-1, 1)
                start = offsets[node_type]
                end = start + x.shape[0]
                combined_x[start:end, :x.shape[1]] = x

        # Combine edges
        src_list = []
        dst_list = []
        for (src_type, _, dst_type), edge_index in self.edge_index_dict.items():
            src_list.append(edge_index[0] + offsets[src_type])
            dst_list.append(edge_index[1] + offsets[dst_type])

        if src_list:
            combined_src = np.concatenate(src_list)
            combined_dst = np.concatenate(dst_list)
            combined_edge_index = np.stack([combined_src, combined_dst])
        else:
            combined_edge_index = np.zeros((2, 0), dtype=np.int64)

        return Graph(
            x=combined_x,
            edge_index=combined_edge_index,
            num_nodes=total_nodes
        )


class TemporalGraph:
    """
    Temporal/dynamic graph with time-stamped edges.

    Supports evolving graphs where edges have timestamps.
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        timestamps: np.ndarray,
        num_nodes: Optional[int] = None,
        x: Optional[np.ndarray] = None,
        edge_attr: Optional[np.ndarray] = None
    ):
        self.edge_index = np.asarray(edge_index)
        self.timestamps = np.asarray(timestamps)
        self.edge_attr = edge_attr

        if len(self.edge_index.shape) == 1:
            half = len(self.edge_index) // 2
            self.edge_index = np.stack([self.edge_index[:half], self.edge_index[half:]])

        if num_nodes is not None:
            self._num_nodes = num_nodes
        elif x is not None:
            self._num_nodes = x.shape[0]
        elif self.edge_index.shape[1] > 0:
            self._num_nodes = int(self.edge_index.max()) + 1
        else:
            self._num_nodes = 0

        if x is not None:
            self.x = np.asarray(x)
        else:
            self.x = np.eye(self._num_nodes)  # Default identity features

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    @property
    def num_timestamps(self) -> int:
        """Number of unique timestamps."""
        return len(np.unique(self.timestamps))

    def snapshot(self, t: int) -> 'TemporalSnapshot':
        """Get graph snapshot at specific time t."""
        mask = self.timestamps == t
        edge_indices = np.where(mask)[0]

        return TemporalSnapshot(
            edge_index=self.edge_index[:, mask],
            edge_indices=edge_indices,
            num_nodes=self._num_nodes
        )

    def get_snapshot(self, t_start: float, t_end: float) -> Graph:
        """Get graph snapshot for a time window."""
        mask = (self.timestamps >= t_start) & (self.timestamps < t_end)
        filtered_edges = self.edge_index[:, mask]

        return Graph(
            x=self.x.copy() if self.x is not None else None,
            edge_index=filtered_edges,
            edge_attr=self.edge_attr[mask] if self.edge_attr is not None else None,
            num_nodes=self._num_nodes
        )

    def temporal_subgraph(self, start_time: int, end_time: int) -> 'TemporalGraph':
        """Extract subgraph for time range [start_time, end_time]."""
        mask = (self.timestamps >= start_time) & (self.timestamps <= end_time)

        return TemporalGraph(
            edge_index=self.edge_index[:, mask],
            timestamps=self.timestamps[mask],
            num_nodes=self._num_nodes,
            x=self.x.copy() if self.x is not None else None,
            edge_attr=self.edge_attr[mask] if self.edge_attr is not None else None
        )

    def to_static(self) -> Graph:
        """Convert to static graph (ignore timestamps)."""
        return Graph(
            x=self.x.copy() if self.x is not None else None,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            num_nodes=self._num_nodes
        )

    def get_events_before(self, t: float) -> 'TemporalGraph':
        """Get all events before time t."""
        mask = self.timestamps < t
        return TemporalGraph(
            edge_index=self.edge_index[:, mask],
            timestamps=self.timestamps[mask],
            num_nodes=self._num_nodes,
            x=self.x.copy() if self.x is not None else None,
            edge_attr=self.edge_attr[mask] if self.edge_attr is not None else None
        )


class TemporalSnapshot:
    """Snapshot of a temporal graph at a specific time."""

    def __init__(self, edge_index: np.ndarray, edge_indices: np.ndarray, num_nodes: int):
        self.edge_index = edge_index
        self.edge_indices = edge_indices
        self.num_nodes = num_nodes


# Utility functions
def to_undirected(edge_index: np.ndarray, num_nodes: Optional[int] = None) -> np.ndarray:
    """Convert directed edges to undirected by adding reverse edges."""
    src, dst = edge_index[0], edge_index[1]

    # Add reverse edges
    new_src = np.concatenate([src, dst])
    new_dst = np.concatenate([dst, src])

    # Remove duplicates
    edges = np.stack([new_src, new_dst], axis=0)
    edges = np.unique(edges, axis=1)

    return edges


def to_directed(edge_index: np.ndarray) -> np.ndarray:
    """Keep only one direction of edges (lower to higher index)."""
    src, dst = edge_index[0], edge_index[1]
    mask = src < dst
    return np.stack([src[mask], dst[mask]])


def add_self_loops(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Add self-loops to edge index."""
    loop_index = np.arange(num_nodes)
    src = np.concatenate([edge_index[0], loop_index])
    dst = np.concatenate([edge_index[1], loop_index])
    return np.stack([src, dst])


def remove_self_loops(edge_index: np.ndarray) -> np.ndarray:
    """Remove self-loops from edge index."""
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def degree(index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Compute node degrees from edge indices.

    Args:
        index: 1D array of node indices (e.g., edge_index[0] or edge_index[1])
        num_nodes: Total number of nodes

    Returns:
        Array of degrees for each node
    """
    deg = np.zeros(num_nodes, dtype=np.int64)
    np.add.at(deg, index, 1)
    return deg


def k_hop_subgraph(
    node_idx: Union[int, np.ndarray],
    num_hops: int,
    edge_index: np.ndarray,
    num_nodes: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract k-hop subgraph around given nodes.

    Args:
        node_idx: Center node(s)
        num_hops: Number of hops
        edge_index: Edge index (2, num_edges)
        num_nodes: Total number of nodes (optional, inferred if not provided)

    Returns:
        Tuple of (subset, sub_edge_index):
        - subset: Node indices in subgraph
        - sub_edge_index: Edge index for subgraph (remapped)
    """
    if isinstance(node_idx, int):
        node_idx = np.array([node_idx])
    else:
        node_idx = np.asarray(node_idx)

    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1

    # Build adjacency set
    adj = [set() for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj[src].add(dst)
        adj[dst].add(src)

    # BFS for k hops
    visited = set(node_idx.tolist())
    frontier = set(node_idx.tolist())

    for _ in range(num_hops):
        next_frontier = set()
        for node in frontier:
            for neighbor in adj[node]:
                if neighbor not in visited:
                    next_frontier.add(neighbor)
                    visited.add(neighbor)
        frontier = next_frontier

    subset = np.array(sorted(visited))

    # Create mapping
    mapping = np.full(num_nodes, -1)
    mapping[subset] = np.arange(len(subset))

    # Filter edges
    mask = np.isin(edge_index[0], subset) & np.isin(edge_index[1], subset)
    sub_edge_index = np.stack([
        mapping[edge_index[0, mask]],
        mapping[edge_index[1, mask]]
    ])

    return subset, sub_edge_index


def induced_subgraph(
    node_idx: Union[np.ndarray, List[int]],
    edge_index: np.ndarray
) -> np.ndarray:
    """
    Extract induced subgraph from node indices.

    Args:
        node_idx: Node indices (array or list)
        edge_index: Edge index (2, num_edges)

    Returns:
        sub_edge_index: Filtered edge index (original node ids, not remapped)
    """
    node_idx = np.asarray(node_idx)

    # If boolean mask
    if node_idx.dtype == bool:
        node_set = set(np.where(node_idx)[0].tolist())
    else:
        node_set = set(node_idx.tolist())

    # Filter edges
    mask = np.array([
        edge_index[0, i] in node_set and edge_index[1, i] in node_set
        for i in range(edge_index.shape[1])
    ])

    return edge_index[:, mask]


def sample_negative_edges(
    edge_index: np.ndarray,
    num_nodes: int,
    num_neg_samples: int
) -> np.ndarray:
    """Sample negative edges (non-existing edges)."""
    # Create set of existing edges
    existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    neg_src = []
    neg_dst = []

    while len(neg_src) < num_neg_samples:
        src = np.random.randint(0, num_nodes)
        dst = np.random.randint(0, num_nodes)
        if src != dst and (src, dst) not in existing:
            neg_src.append(src)
            neg_dst.append(dst)
            existing.add((src, dst))

    return np.stack([np.array(neg_src), np.array(neg_dst)])
