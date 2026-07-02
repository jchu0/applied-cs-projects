"""GNN layer implementations."""

from __future__ import annotations

from typing import Optional, List, Callable
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

from .message_passing import MessagePassingEngine


def _check_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch required for GNN layers")


class _ModuleFallback:
    """Base used in place of ``nn.Module`` when PyTorch is unavailable.

    Lets class bodies be defined at import time; any attempt to actually
    instantiate a layer without PyTorch raises via ``_check_torch()``.
    """


# Base class for the layer definitions below. When torch is present this is the
# real ``nn.Module``; otherwise it is a plain placeholder so import still works.
_Base = nn.Module if HAS_TORCH else _ModuleFallback


class GNNLayer(_Base):
    """
    Generic GNN layer with configurable message passing.

    Supports GCN, GAT, and GraphSAGE-style message passing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        message_type: str = 'gcn',
        aggregate_type: str = 'sum',
        normalize: bool = True,
        bias: bool = True,
    ):
        """
        Initialize GNN layer.

        Args:
            in_channels: Input feature dimension.
            out_channels: Output feature dimension.
            message_type: Type of message function ('gcn', 'gat', 'sage').
            aggregate_type: Aggregation function ('sum', 'mean', 'max').
            normalize: Whether to apply normalization.
            bias: Whether to use bias.
        """
        _check_torch()
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.message_type = message_type
        self.aggregate_type = aggregate_type
        self.normalize = normalize

        # Linear transformation
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # GAT attention parameters
        if message_type == 'gat':
            self.att_src = nn.Parameter(torch.zeros(1, out_channels))
            self.att_dst = nn.Parameter(torch.zeros(1, out_channels))
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)

        self.mp_engine = MessagePassingEngine(device='cpu')
        self.reset_parameters()

    def reset_parameters(self):
        """Reset learnable parameters."""
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
        edge_weight: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        """
        Forward pass.

        Args:
            x: Node features [N, in_channels].
            edge_index: Edge indices [2, E].
            edge_weight: Optional edge weights [E].

        Returns:
            Updated node features [N, out_channels].
        """
        # Transform features
        x = self.lin(x)

        # Compute normalization for GCN
        if self.normalize and self.message_type == 'gcn':
            edge_weight = self._compute_gcn_norm(edge_index, x.size(0), x.device)

        # Message function based on type
        if self.message_type == 'gcn':
            def message_fn(src, dst, edge_feat):
                if edge_weight is not None:
                    return src * edge_weight.unsqueeze(-1)
                return src
        elif self.message_type == 'gat':
            def message_fn(src, dst, edge_feat):
                alpha_src = (src * self.att_src).sum(dim=-1)
                alpha_dst = (dst * self.att_dst).sum(dim=-1)
                alpha = F.leaky_relu(alpha_src + alpha_dst, 0.2)
                alpha = F.softmax(alpha, dim=0)
                return src * alpha.unsqueeze(-1)
        else:  # Simple aggregation
            message_fn = None

        # Perform message passing
        out = self.mp_engine.propagate(
            edge_index, x,
            message_fn=message_fn,
            aggregate_fn=self.aggregate_type,
        )

        if self.bias is not None:
            out = out + self.bias

        return out

    def _compute_gcn_norm(
        self,
        edge_index: 'torch.Tensor',
        num_nodes: int,
        device: 'torch.device',
    ) -> 'torch.Tensor':
        """Compute symmetric normalization D^{-1/2} A D^{-1/2}."""
        src, dst = edge_index

        # Compute degrees
        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))

        # D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Norm = D_src^{-1/2} * D_dst^{-1/2}
        return deg_inv_sqrt[src] * deg_inv_sqrt[dst]


class GCNLayer(_Base):
    """
    Graph Convolutional Network layer.

    Implements: H' = D^{-1/2} A D^{-1/2} H W
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: bool = True,
        normalize: bool = True,
        bias: bool = True,
    ):
        _check_torch()
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self._cached_edge_index = None
        self._cached_norm = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self._cached_edge_index = None
        self._cached_norm = None

    def forward(
        self,
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
        edge_weight: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        num_nodes = x.size(0)

        if self.normalize:
            if self.cached and self._cached_norm is not None:
                norm = self._cached_norm
            else:
                norm = self._compute_norm(edge_index, num_nodes, x.device)
                if self.cached:
                    self._cached_norm = norm
        else:
            norm = None

        # Transform
        x = self.lin(x)

        # Message passing
        src, dst = edge_index
        out = torch.zeros_like(x)

        if norm is not None:
            messages = x[src] * norm.unsqueeze(-1)
        else:
            messages = x[src]

        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        if self.bias is not None:
            out = out + self.bias

        return out

    def _compute_norm(
        self,
        edge_index: 'torch.Tensor',
        num_nodes: int,
        device: 'torch.device',
    ) -> 'torch.Tensor':
        src, dst = edge_index

        deg = torch.zeros(num_nodes, device=device)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))

        if self.improved:
            deg = deg + 1

        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[src] * deg_inv_sqrt[dst]


class GATLayer(_Base):
    """
    Graph Attention Network layer.

    Implements multi-head attention mechanism for graph-structured data.
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
    ):
        _check_torch()
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # Per-head linear transformation
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)

        # Attention parameters
        self.att_src = nn.Parameter(torch.zeros(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.zeros(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.zeros(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
    ) -> 'torch.Tensor':
        num_nodes = x.size(0)
        src, dst = edge_index

        # Linear transformation
        x = self.lin(x).view(-1, self.heads, self.out_channels)  # [N, H, F]

        # Compute attention scores
        alpha_src = (x * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # [N, H]

        # Edge attention
        alpha = alpha_src[src] + alpha_dst[dst]  # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)

        # Softmax over neighbors
        alpha = self._edge_softmax(alpha, dst, num_nodes)

        if self.dropout > 0 and self.training:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Message passing
        out = torch.zeros(num_nodes, self.heads, self.out_channels,
                         device=x.device, dtype=x.dtype)

        messages = x[src] * alpha.unsqueeze(-1)
        out.scatter_add_(0, dst.view(-1, 1, 1).expand_as(messages), messages)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def _edge_softmax(
        self,
        alpha: 'torch.Tensor',
        index: 'torch.Tensor',
        num_nodes: int,
    ) -> 'torch.Tensor':
        """Compute softmax over edges with same destination."""
        # Subtract max for numerical stability
        alpha_max = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_max.scatter_reduce_(0, index.unsqueeze(-1).expand_as(alpha), alpha, reduce='amax')
        alpha = alpha - alpha_max[index]

        # Exp and sum
        alpha = alpha.exp()
        alpha_sum = torch.zeros(num_nodes, alpha.size(1), device=alpha.device)
        alpha_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(alpha), alpha)

        return alpha / (alpha_sum[index] + 1e-16)


class GraphSAGELayer(_Base):
    """
    GraphSAGE layer.

    Implements sampling and aggregation with mean/max/LSTM aggregators.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggregator: str = 'mean',
        normalize: bool = True,
        root_weight: bool = True,
        bias: bool = True,
    ):
        _check_torch()
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggregator = aggregator
        self.normalize = normalize
        self.root_weight = root_weight

        # Neighbor transformation
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)

        # Root node transformation
        if root_weight:
            self.lin_root = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.register_parameter('lin_root', None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        if self.lin_root is not None:
            nn.init.xavier_uniform_(self.lin_root.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
    ) -> 'torch.Tensor':
        num_nodes = x.size(0)
        src, dst = edge_index

        # Aggregate neighbors
        if self.aggregator == 'mean':
            out = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
            out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(1)), x[src])

            # Compute degree for mean
            deg = torch.zeros(num_nodes, device=x.device)
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
            deg = deg.clamp(min=1)
            out = out / deg.unsqueeze(-1)

        elif self.aggregator == 'max':
            out = torch.full((num_nodes, x.size(1)), float('-inf'),
                            device=x.device, dtype=x.dtype)
            out.scatter_reduce_(0, dst.unsqueeze(-1).expand(-1, x.size(1)),
                               x[src], reduce='amax')
            out[out == float('-inf')] = 0

        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        # Transform aggregated neighbors
        out = self.lin_neigh(out)

        # Add root node
        if self.lin_root is not None:
            out = out + self.lin_root(x)

        if self.bias is not None:
            out = out + self.bias

        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)

        return out


class GNNModel(_Base):
    """
    Multi-layer GNN model.

    Stacks multiple GNN layers with activation and dropout.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        layer_type: str = 'gcn',
        dropout: float = 0.5,
        jk: Optional[str] = None,
    ):
        """
        Initialize GNN model.

        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden layer dimension.
            out_channels: Output dimension (e.g., num_classes).
            num_layers: Number of GNN layers.
            layer_type: Type of GNN layer ('gcn', 'gat', 'sage').
            dropout: Dropout rate.
            jk: Jumping knowledge mode ('cat', 'max', 'lstm', None).
        """
        _check_torch()
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.jk = jk

        # Create layers
        self.layers = nn.ModuleList()

        layer_class = {
            'gcn': GCNLayer,
            'gat': GATLayer,
            'sage': GraphSAGELayer,
        }.get(layer_type, GCNLayer)

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            out_ch = hidden_channels
            is_last_layer = (i == num_layers - 1)

            if layer_type == 'gat':
                # Use concat=False for last layer to get expected output dimension
                concat = not is_last_layer
                self.layers.append(layer_class(in_ch, out_ch, heads=4, concat=concat))
                # GAT with concat multiplies channels by heads (only for non-last layers)
                if concat:
                    hidden_channels = out_ch * 4
            else:
                self.layers.append(layer_class(in_ch, out_ch))

        # Final classifier
        jk_channels = hidden_channels * num_layers if jk == 'cat' else hidden_channels
        self.classifier = nn.Linear(jk_channels, out_channels)

        # Jumping knowledge
        if jk == 'lstm':
            self.jk_lstm = nn.LSTM(
                hidden_channels, hidden_channels,
                bidirectional=True, batch_first=True
            )
            self.jk_linear = nn.Linear(2 * hidden_channels, hidden_channels)

    def forward(
        self,
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Forward pass through all layers."""
        xs = []

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

            if self.jk is not None:
                xs.append(x)

        # Jumping knowledge aggregation
        if self.jk == 'cat':
            x = torch.cat(xs, dim=-1)
        elif self.jk == 'max':
            x = torch.stack(xs, dim=-1).max(dim=-1)[0]
        elif self.jk == 'lstm':
            x = torch.stack(xs, dim=1)  # [N, L, F]
            _, (h, _) = self.jk_lstm(x)
            x = self.jk_linear(torch.cat([h[0], h[1]], dim=-1))

        return self.classifier(x)

    def encode(
        self,
        x: 'torch.Tensor',
        edge_index: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Get node embeddings before classification."""
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
