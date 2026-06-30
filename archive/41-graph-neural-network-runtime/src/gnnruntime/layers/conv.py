"""Graph convolutional layers for GNN runtime."""

import numpy as np
import logging
from typing import Optional, Tuple
from abc import ABC, abstractmethod

from ..core.graph import EdgeIndex, MessagePassing, gcn_norm, normalize_adj

logger = logging.getLogger(__name__)


class GNNLayer(ABC):
    """Base class for GNN layers."""

    @abstractmethod
    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass."""
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def reset_parameters(self):
        """Reset layer parameters."""
        pass


class GCNConv(GNNLayer, MessagePassing):
    """
    Graph Convolutional Network layer.

    Implements: X' = D^{-1/2} A D^{-1/2} X W

    Reference: Kipf & Welling, "Semi-Supervised Classification with
    Graph Convolutional Networks" (ICLR 2017)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
        aggr: str = 'add'
    ):
        MessagePassing.__init__(self, aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        # Initialize weights
        self.weight = np.random.randn(in_channels, out_channels) * np.sqrt(2.0 / in_channels)
        self._bias = np.zeros(out_channels) if bias else None

        # Cached normalization
        self._cached_edge_index = None
        self._cached_edge_weight = None

    @property
    def bias(self):
        """Alias for backward compatibility."""
        return self._bias

    @property
    def bias_vec(self):
        """Alias for backward compatibility."""
        return self._bias

    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity
            edge_attr: Not used (for API compatibility)

        Returns:
            Updated node features (num_nodes, out_channels)
        """
        num_nodes = x.shape[0]

        # Get normalization coefficients
        if self.normalize:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes)
        else:
            if self.add_self_loops:
                edge_index = edge_index.add_self_loops(num_nodes)
            edge_weight = np.ones(edge_index.num_edges)

        # Linear transform
        x = x @ self.weight

        # Message passing with edge weights
        out = np.zeros_like(x)
        np.add.at(out, edge_index.dst, edge_weight[:, None] * x[edge_index.src])

        # Add bias
        if self._bias is not None:
            out = out + self._bias

        return out


class GATConv(GNNLayer, MessagePassing):
    """
    Graph Attention Network layer.

    Implements attention mechanism over neighbors.

    Reference: Veličković et al., "Graph Attention Networks" (ICLR 2018)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        return_attention_weights: bool = False,
        edge_dim: Optional[int] = None
    ):
        MessagePassing.__init__(self, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.return_attention_weights = return_attention_weights
        self.edge_dim = edge_dim
        self.training = True  # Training mode flag

        # Linear transformations
        self.weight = np.random.randn(in_channels, heads * out_channels) * np.sqrt(2.0 / in_channels)

        # Edge feature projection if needed
        if edge_dim is not None:
            self.edge_proj = np.random.randn(edge_dim, heads * out_channels) * np.sqrt(2.0 / edge_dim)
        else:
            self.edge_proj = None

        # Attention parameters
        self.att_src = np.random.randn(1, heads, out_channels) * 0.01
        self.att_dst = np.random.randn(1, heads, out_channels) * 0.01

        # Bias
        if bias and concat:
            self.bias_vec = np.zeros(heads * out_channels)
        elif bias:
            self.bias_vec = np.zeros(out_channels)
        else:
            self.bias_vec = None

    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex,
        edge_attr: Optional[np.ndarray] = None
    ):
        """
        Forward pass with attention mechanism.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity
            edge_attr: Optional edge features

        Returns:
            Updated node features, or (output, attention_weights) if return_attention_weights=True
        """
        num_nodes = x.shape[0]
        num_original_edges = edge_index.num_edges

        # Add self-loops
        if self.add_self_loops:
            edge_index = edge_index.add_self_loops(num_nodes)
            if edge_attr is not None:
                # Pad edge_attr for self-loops
                self_loop_attr = np.zeros((num_nodes, edge_attr.shape[1]))
                edge_attr = np.vstack([edge_attr, self_loop_attr])

        # Linear transform and reshape for multi-head
        h = x @ self.weight
        h = h.reshape(num_nodes, self.heads, self.out_channels)

        # Compute attention scores
        alpha_src = (h * self.att_src).sum(axis=-1)  # (num_nodes, heads)
        alpha_dst = (h * self.att_dst).sum(axis=-1)

        # Edge attention
        alpha = alpha_src[edge_index.src] + alpha_dst[edge_index.dst]

        # Add edge features if present
        if edge_attr is not None and self.edge_proj is not None:
            edge_h = edge_attr @ self.edge_proj
            edge_h = edge_h.reshape(-1, self.heads, self.out_channels)
            alpha = alpha + (edge_h * self.att_src[0]).sum(axis=-1)

        # LeakyReLU
        alpha = np.where(alpha > 0, alpha, alpha * self.negative_slope)

        # Softmax over neighbors
        alpha_exp = np.exp(alpha - alpha.max())
        alpha_sum = np.zeros((num_nodes, self.heads))
        np.add.at(alpha_sum, edge_index.dst, alpha_exp)
        alpha_normalized = alpha_exp / (alpha_sum[edge_index.dst] + 1e-8)

        # Apply dropout (only during training)
        if self.dropout > 0 and self.training:
            mask = np.random.random(alpha_normalized.shape) > self.dropout
            alpha_normalized = alpha_normalized * mask / (1 - self.dropout)

        # Aggregate with attention
        out = np.zeros((num_nodes, self.heads, self.out_channels))
        for i in range(edge_index.num_edges):
            src, dst = edge_index.src[i], edge_index.dst[i]
            out[dst] += alpha_normalized[i, :, None] * h[src]

        # Concat or average heads
        if self.concat:
            out = out.reshape(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(axis=1)

        # Bias
        if self.bias_vec is not None:
            out = out + self.bias_vec

        if self.return_attention_weights:
            # Return attention weights averaged over heads, only for original edges
            attention_weights = alpha_normalized[:num_original_edges].mean(axis=1)
            return out, attention_weights

        return out


class SAGEConv(GNNLayer, MessagePassing):
    """
    GraphSAGE convolutional layer.

    Samples and aggregates features from neighbors.

    Reference: Hamilton et al., "Inductive Representation Learning on
    Large Graphs" (NeurIPS 2017)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean',
        normalize: bool = False,
        normalize_emb: bool = False,
        root_weight: bool = True,
        bias: bool = True
    ):
        # Handle lstm and pool aggregators
        actual_aggr = aggr
        if aggr in ('lstm', 'pool'):
            actual_aggr = 'mean'  # Fall back to mean for now
            self._lstm_or_pool = aggr
        else:
            self._lstm_or_pool = None

        MessagePassing.__init__(self, aggr=actual_aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize_output = normalize or normalize_emb
        self.root_weight = root_weight

        # Linear transformations
        self.lin_l = np.random.randn(in_channels, out_channels) * np.sqrt(2.0 / in_channels)
        if root_weight:
            self._lin_root = np.random.randn(in_channels, out_channels) * np.sqrt(2.0 / in_channels)
        else:
            self._lin_root = None

        self.bias_vec = np.zeros(out_channels) if bias else None

        # LSTM state if needed
        if aggr == 'lstm':
            self._lstm_weight = np.random.randn(in_channels, 4 * in_channels) * 0.1

    @property
    def lin_r(self):
        """Backward compatibility."""
        return self._lin_root

    @property
    def lin_root(self):
        """Root weight matrix."""
        return self._lin_root

    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity
            edge_attr: Not used

        Returns:
            Updated node features (num_nodes, out_channels)
        """
        # Aggregate neighbor features
        out = self.propagate(edge_index, x)

        # Transform aggregated
        out = out @ self.lin_l

        # Add root node features
        if self._lin_root is not None:
            out = out + x @ self._lin_root

        # Bias
        if self.bias_vec is not None:
            out = out + self.bias_vec

        # L2 normalize
        if self.normalize_output:
            out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-8)

        return out


class GINConv(GNNLayer, MessagePassing):
    """
    Graph Isomorphism Network layer.

    Reference: Xu et al., "How Powerful are Graph Neural Networks?" (ICLR 2019)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        eps: float = 0.0,
        train_eps: bool = False
    ):
        MessagePassing.__init__(self, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.initial_eps = eps
        self.eps = eps

        # MLP
        hidden = max(in_channels, out_channels)
        self.mlp_weight1 = np.random.randn(in_channels, hidden) * np.sqrt(2.0 / in_channels)
        self.mlp_weight2 = np.random.randn(hidden, out_channels) * np.sqrt(2.0 / hidden)
        self.mlp_bias1 = np.zeros(hidden)
        self.mlp_bias2 = np.zeros(out_channels)

    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Node features
            edge_index: Graph connectivity
            edge_attr: Not used

        Returns:
            Updated node features
        """
        # Aggregate neighbors
        out = self.propagate(edge_index, x)

        # (1 + eps) * x + aggregated
        out = (1 + self.eps) * x + out

        # MLP
        out = out @ self.mlp_weight1 + self.mlp_bias1
        out = np.maximum(out, 0)  # ReLU
        out = out @ self.mlp_weight2 + self.mlp_bias2

        return out


class EdgeConv(GNNLayer, MessagePassing):
    """
    EdgeConv layer for point cloud processing.

    Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds"
    (TOG 2019)
    """

    def __init__(
        self,
        nn_or_in_channels,
        out_channels: int = None,
        aggr: str = 'max'
    ):
        MessagePassing.__init__(self, aggr=aggr)

        # Support both nn function and in_channels signature
        if callable(nn_or_in_channels) and out_channels is None:
            # nn is a callable
            self.nn = nn_or_in_channels
            self.in_channels = None
            self.out_channels = None
            self.weight1 = None
            self.bias1 = None
        else:
            # in_channels, out_channels signature
            in_channels = nn_or_in_channels
            self.nn = None
            self.in_channels = in_channels
            self.out_channels = out_channels

            # MLP for edge features
            self.weight1 = np.random.randn(2 * in_channels, out_channels) * np.sqrt(2.0 / (2 * in_channels))
            self.bias1 = np.zeros(out_channels)

    def message(
        self,
        x_j: np.ndarray,
        x_i: np.ndarray,
        edge_attr: Optional[np.ndarray]
    ) -> np.ndarray:
        """Compute edge features from node pairs."""
        # Concatenate (x_i, x_j - x_i)
        edge_features = np.concatenate([x_i, x_j - x_i], axis=-1)

        if self.nn is not None:
            return self.nn(edge_features)
        else:
            # MLP
            out = edge_features @ self.weight1 + self.bias1
            out = np.maximum(out, 0)  # ReLU
            return out

    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex = None,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass."""
        if edge_index is None:
            # Handle case where edge_index might be computed dynamically
            raise ValueError("edge_index required for EdgeConv")
        return self.propagate(edge_index, x, edge_attr)


class DynamicEdgeConv(GNNLayer, MessagePassing):
    """
    Dynamic EdgeConv that constructs k-NN graph on the fly.

    Reference: Wang et al., "Dynamic Graph CNN for Learning on Point Clouds"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 10,
        aggr: str = 'max'
    ):
        MessagePassing.__init__(self, aggr=aggr)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        # MLP for edge features
        self.weight1 = np.random.randn(2 * in_channels, out_channels) * np.sqrt(2.0 / (2 * in_channels))
        self.bias1 = np.zeros(out_channels)

    def _compute_knn_graph(self, x: np.ndarray) -> EdgeIndex:
        """Compute k-nearest neighbors graph."""
        num_nodes = x.shape[0]
        k = min(self.k, num_nodes - 1)

        # Compute pairwise distances
        diff = x[:, None, :] - x[None, :, :]
        dist = np.sum(diff ** 2, axis=-1)

        # Get k nearest neighbors for each node
        src_list = []
        dst_list = []

        for i in range(num_nodes):
            distances = dist[i]
            distances[i] = np.inf  # Exclude self
            nearest = np.argsort(distances)[:k]

            src_list.extend([i] * k)
            dst_list.extend(nearest.tolist())

        return EdgeIndex([np.array(src_list), np.array(dst_list)])

    def message(
        self,
        x_j: np.ndarray,
        x_i: np.ndarray,
        edge_attr: Optional[np.ndarray]
    ) -> np.ndarray:
        """Compute edge features from node pairs."""
        edge_features = np.concatenate([x_i, x_j - x_i], axis=-1)
        out = edge_features @ self.weight1 + self.bias1
        out = np.maximum(out, 0)  # ReLU
        return out

    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex = None,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass with dynamic graph construction."""
        # Construct k-NN graph
        edge_index = self._compute_knn_graph(x)

        return self.propagate(edge_index, x, edge_attr)


class Linear(GNNLayer):
    """Simple linear layer for GNN."""

    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        self.weight = np.random.randn(in_channels, out_channels) * np.sqrt(2.0 / in_channels)
        self.bias_vec = np.zeros(out_channels) if bias else None

    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex = None,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass (ignores graph structure)."""
        out = x @ self.weight
        if self.bias_vec is not None:
            out = out + self.bias_vec
        return out


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(x, 0)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax activation."""
    exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Log softmax activation."""
    return x - np.log(np.exp(x - x.max(axis=axis, keepdims=True)).sum(axis=axis, keepdims=True))


def dropout(x: np.ndarray, p: float = 0.5, training: bool = True) -> np.ndarray:
    """Dropout regularization."""
    if not training or p == 0:
        return x
    mask = np.random.random(x.shape) > p
    return x * mask / (1 - p)


class ChebConv(GNNLayer, MessagePassing):
    """
    Chebyshev spectral graph convolution.

    Uses Chebyshev polynomials for spectral graph convolution.

    Reference: Defferrard et al., "Convolutional Neural Networks on
    Graphs with Fast Localized Spectral Filtering" (NeurIPS 2016)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 2,
        normalization: str = 'sym',
        bias: bool = True
    ):
        MessagePassing.__init__(self, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalization = normalization

        # Chebyshev coefficients
        self.lins = [
            np.random.randn(in_channels, out_channels) * np.sqrt(2.0 / in_channels)
            for _ in range(K)
        ]

        self.bias_vec = np.zeros(out_channels) if bias else None

    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex,
        edge_attr: Optional[np.ndarray] = None,
        lambda_max: Optional[float] = None
    ) -> np.ndarray:
        """Forward pass using Chebyshev polynomials.

        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Graph connectivity
            edge_attr: Not used
            lambda_max: Largest eigenvalue of the Laplacian (optional)
        """
        num_nodes = x.shape[0]

        # Compute normalized Laplacian
        edge_index_norm, edge_weight = gcn_norm(edge_index, num_nodes)

        # If lambda_max provided, scale the Laplacian
        if lambda_max is not None and lambda_max != 2.0:
            edge_weight = edge_weight * (2.0 / lambda_max)

        # Chebyshev recurrence
        Tx_0 = x
        out = Tx_0 @ self.lins[0]

        if self.K > 1:
            # Propagate: L * x where L = I - D^{-1/2} A D^{-1/2}
            Tx_1 = x - self._propagate(x, edge_index_norm, edge_weight)
            out = out + Tx_1 @ self.lins[1]

        for k in range(2, self.K):
            # Tx_k = 2 * L * Tx_{k-1} - Tx_{k-2}
            Tx_2 = 2 * (Tx_1 - self._propagate(Tx_1, edge_index_norm, edge_weight)) - Tx_0
            out = out + Tx_2 @ self.lins[k]
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias_vec is not None:
            out = out + self.bias_vec

        return out

    def _propagate(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex,
        edge_weight: np.ndarray
    ) -> np.ndarray:
        """Propagate with normalized adjacency."""
        out = np.zeros_like(x)
        np.add.at(out, edge_index.dst, edge_weight[:, None] * x[edge_index.src])
        return out


class SGConv(GNNLayer, MessagePassing):
    """
    Simplified Graph Convolution.

    Pre-computes K-hop aggregation for efficiency.

    Reference: Wu et al., "Simplifying Graph Convolutional Networks" (ICML 2019)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 1,
        cached: bool = False,
        add_self_loops: bool = True,
        bias: bool = True
    ):
        MessagePassing.__init__(self, aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.cached = cached
        self.add_self_loops = add_self_loops

        self.weight = np.random.randn(in_channels, out_channels) * np.sqrt(2.0 / in_channels)
        self.bias_vec = np.zeros(out_channels) if bias else None

        self._cached_x = None

    def forward(
        self,
        x: np.ndarray,
        edge_index: EdgeIndex,
        edge_attr: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Forward pass with K-hop aggregation."""
        if self.cached and self._cached_x is not None:
            out = self._cached_x
        else:
            num_nodes = x.shape[0]
            edge_index_norm, edge_weight = gcn_norm(edge_index, num_nodes)

            # Apply K hops of propagation
            out = x
            for _ in range(self.K):
                new_out = np.zeros_like(out)
                np.add.at(new_out, edge_index_norm.dst, edge_weight[:, None] * out[edge_index_norm.src])
                out = new_out

            if self.cached:
                self._cached_x = out

        # Linear transform
        out = out @ self.weight

        if self.bias_vec is not None:
            out = out + self.bias_vec

        return out


# Alias for GraphSAGE - tests expect GraphSAGEConv name
GraphSAGEConv = SAGEConv
