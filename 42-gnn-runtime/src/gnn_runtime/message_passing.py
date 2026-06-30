"""Message passing engine for GNN computation."""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


class MessageFunction(ABC):
    """Abstract message function for edges."""

    @abstractmethod
    def __call__(
        self,
        src_features: 'torch.Tensor',
        dst_features: 'torch.Tensor',
        edge_features: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        """
        Compute messages on edges.

        Args:
            src_features: Source node features [E, F].
            dst_features: Destination node features [E, F].
            edge_features: Optional edge features [E, Fe].

        Returns:
            Edge messages [E, F'].
        """
        pass


class AggregateFunction(ABC):
    """Abstract aggregation function for neighborhoods."""

    @abstractmethod
    def __call__(
        self,
        messages: 'torch.Tensor',
        index: 'torch.Tensor',
        num_nodes: int,
    ) -> 'torch.Tensor':
        """
        Aggregate messages to nodes.

        Args:
            messages: Messages to aggregate [E, F].
            index: Target node indices [E].
            num_nodes: Total number of nodes.

        Returns:
            Aggregated messages [N, F].
        """
        pass


class CopyMessage(MessageFunction):
    """Simply copy source features as messages."""

    def __call__(
        self,
        src_features: 'torch.Tensor',
        dst_features: 'torch.Tensor',
        edge_features: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        return src_features


class ConcatMessage(MessageFunction):
    """Concatenate source and destination features."""

    def __call__(
        self,
        src_features: 'torch.Tensor',
        dst_features: 'torch.Tensor',
        edge_features: Optional['torch.Tensor'] = None,
    ) -> 'torch.Tensor':
        if not HAS_TORCH:
            raise ImportError("PyTorch required for message passing")

        if edge_features is not None:
            return torch.cat([src_features, dst_features, edge_features], dim=-1)
        return torch.cat([src_features, dst_features], dim=-1)


class SumAggregate(AggregateFunction):
    """Sum aggregation."""

    def __call__(
        self,
        messages: 'torch.Tensor',
        index: 'torch.Tensor',
        num_nodes: int,
    ) -> 'torch.Tensor':
        if not HAS_TORCH:
            raise ImportError("PyTorch required for message passing")

        out = torch.zeros(num_nodes, messages.size(-1),
                         device=messages.device, dtype=messages.dtype)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)
        return out


class MeanAggregate(AggregateFunction):
    """Mean aggregation."""

    def __call__(
        self,
        messages: 'torch.Tensor',
        index: 'torch.Tensor',
        num_nodes: int,
    ) -> 'torch.Tensor':
        if not HAS_TORCH:
            raise ImportError("PyTorch required for message passing")

        out = torch.zeros(num_nodes, messages.size(-1),
                         device=messages.device, dtype=messages.dtype)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)

        # Compute degrees
        ones = torch.ones(index.size(0), device=messages.device)
        degree = torch.zeros(num_nodes, device=messages.device)
        degree.scatter_add_(0, index, ones)
        degree = degree.clamp(min=1)  # Avoid division by zero

        return out / degree.unsqueeze(-1)


class MaxAggregate(AggregateFunction):
    """Max aggregation."""

    def __call__(
        self,
        messages: 'torch.Tensor',
        index: 'torch.Tensor',
        num_nodes: int,
    ) -> 'torch.Tensor':
        if not HAS_TORCH:
            raise ImportError("PyTorch required for message passing")

        out = torch.full((num_nodes, messages.size(-1)), float('-inf'),
                        device=messages.device, dtype=messages.dtype)
        out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(messages),
                           messages, reduce='amax')
        out[out == float('-inf')] = 0
        return out


class MessagePassingEngine:
    """
    High-performance message passing engine for GNNs.

    Supports configurable message and aggregation functions with
    efficient scatter operations.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initialize message passing engine.

        Args:
            device: Device for computation ('cpu' or 'cuda').
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MessagePassingEngine")

        self.device = device
        self._aggregate_fns: Dict[str, AggregateFunction] = {
            'sum': SumAggregate(),
            'mean': MeanAggregate(),
            'max': MaxAggregate(),
        }

    def propagate(
        self,
        edge_index: 'torch.Tensor',
        node_features: 'torch.Tensor',
        edge_features: Optional['torch.Tensor'] = None,
        message_fn: Optional[Callable] = None,
        aggregate_fn: str = 'sum',
        flow: str = 'source_to_target',
    ) -> 'torch.Tensor':
        """
        Perform message passing on graph.

        Args:
            edge_index: Edge indices [2, E].
            node_features: Node features [N, F].
            edge_features: Optional edge features [E, Fe].
            message_fn: Function to compute messages.
            aggregate_fn: Aggregation type ('sum', 'mean', 'max').
            flow: Direction of message flow ('source_to_target' or 'target_to_source').

        Returns:
            Aggregated messages for each node [N, F'].
        """
        if flow == 'source_to_target':
            src, dst = edge_index[0], edge_index[1]
        else:
            dst, src = edge_index[0], edge_index[1]

        # Gather source and destination features
        src_features = node_features[src]  # [E, F]
        dst_features = node_features[dst]  # [E, F]

        # Compute messages
        if message_fn is not None:
            messages = message_fn(src_features, dst_features, edge_features)
        else:
            messages = src_features

        # Aggregate messages to nodes
        num_nodes = node_features.size(0)
        agg_fn = self._aggregate_fns.get(aggregate_fn)

        if agg_fn is None:
            raise ValueError(f"Unknown aggregation: {aggregate_fn}")

        return agg_fn(messages, dst, num_nodes)

    def propagate_with_attention(
        self,
        edge_index: 'torch.Tensor',
        node_features: 'torch.Tensor',
        attention_weights: 'torch.Tensor',
        flow: str = 'source_to_target',
    ) -> 'torch.Tensor':
        """
        Perform attention-weighted message passing.

        Args:
            edge_index: Edge indices [2, E].
            node_features: Node features [N, F].
            attention_weights: Attention weights per edge [E].
            flow: Direction of message flow.

        Returns:
            Attention-weighted aggregated messages [N, F].
        """
        if flow == 'source_to_target':
            src, dst = edge_index[0], edge_index[1]
        else:
            dst, src = edge_index[0], edge_index[1]

        # Gather source features and weight by attention
        src_features = node_features[src]  # [E, F]
        weighted_messages = src_features * attention_weights.unsqueeze(-1)

        # Sum aggregate
        num_nodes = node_features.size(0)
        out = torch.zeros(num_nodes, node_features.size(-1),
                         device=self.device, dtype=node_features.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_messages), weighted_messages)

        return out

    def multi_head_propagate(
        self,
        edge_index: 'torch.Tensor',
        node_features: 'torch.Tensor',
        attention_weights: 'torch.Tensor',
        num_heads: int,
    ) -> 'torch.Tensor':
        """
        Multi-head attention message passing.

        Args:
            edge_index: Edge indices [2, E].
            node_features: Node features [N, H, F/H].
            attention_weights: Attention weights [E, H].
            num_heads: Number of attention heads.

        Returns:
            Multi-head aggregated messages [N, H, F/H].
        """
        src, dst = edge_index[0], edge_index[1]

        # Gather source features [E, H, F/H]
        src_features = node_features[src]

        # Weight by attention [E, H, F/H]
        weighted = src_features * attention_weights.unsqueeze(-1)

        # Aggregate per head
        num_nodes = node_features.size(0)
        out = torch.zeros_like(node_features)

        for h in range(num_heads):
            head_messages = weighted[:, h, :]
            head_out = torch.zeros(num_nodes, node_features.size(-1),
                                  device=self.device, dtype=node_features.dtype)
            head_out.scatter_add_(0, dst.unsqueeze(-1).expand_as(head_messages), head_messages)
            out[:, h, :] = head_out

        return out


class MessagePassingNumpy:
    """
    NumPy-based message passing for CPU-only execution.

    Useful for testing and environments without PyTorch.
    """

    @staticmethod
    def propagate_sum(
        edge_index: np.ndarray,
        node_features: np.ndarray,
        flow: str = 'source_to_target',
    ) -> np.ndarray:
        """
        Perform sum-aggregation message passing with NumPy.

        Args:
            edge_index: Edge indices [2, E].
            node_features: Node features [N, F].
            flow: Direction of message flow.

        Returns:
            Aggregated messages [N, F].
        """
        if flow == 'source_to_target':
            src, dst = edge_index[0], edge_index[1]
        else:
            dst, src = edge_index[0], edge_index[1]

        num_nodes, feat_dim = node_features.shape
        out = np.zeros((num_nodes, feat_dim), dtype=node_features.dtype)

        # Gather and aggregate
        for e in range(len(src)):
            out[dst[e]] += node_features[src[e]]

        return out

    @staticmethod
    def propagate_mean(
        edge_index: np.ndarray,
        node_features: np.ndarray,
        flow: str = 'source_to_target',
    ) -> np.ndarray:
        """Mean aggregation message passing."""
        if flow == 'source_to_target':
            src, dst = edge_index[0], edge_index[1]
        else:
            dst, src = edge_index[0], edge_index[1]

        num_nodes, feat_dim = node_features.shape
        out = np.zeros((num_nodes, feat_dim), dtype=node_features.dtype)
        degree = np.zeros(num_nodes, dtype=np.float64)

        for e in range(len(src)):
            out[dst[e]] += node_features[src[e]]
            degree[dst[e]] += 1

        # Normalize
        degree = np.maximum(degree, 1)
        out = out / degree[:, np.newaxis]

        return out
