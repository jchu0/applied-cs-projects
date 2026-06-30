# Project 41: Graph Neural Network Runtime (DGL-lite / PyG-lite)

> **Concepts covered:** §03 ml-engineering — `02-deep-learning`, `06-cuda-optimization`

## Executive Summary

A high-performance graph neural network runtime system optimized for GPU sparse operations, efficient graph partitioning, and scalable message passing. This project implements the core primitives needed for training and inference of GNNs on large-scale graphs with millions to billions of edges.

## Architecture Overview

### System Design

```
+------------------------------------------------------------------+
|                    GNN Runtime Architecture                       |
+------------------------------------------------------------------+
|                                                                    |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Graph Storage     |     | Message Passing   |     | Neighbor  | |
|  | Engine            |<--->| Engine            |<--->| Sampler   | |
|  +-------------------+     +-------------------+     +-----------+ |
|         |                          |                       |       |
|         v                          v                       v       |
|  +-------------------+     +-------------------+     +-----------+ |
|  | Feature Store     |     | GPU Kernel Pool   |     | Partition | |
|  | (Node/Edge)       |     | (SpMM, Scatter)   |     | Manager   | |
|  +-------------------+     +-------------------+     +-----------+ |
|                                    |                               |
|                                    v                               |
|  +----------------------------------------------------------+     |
|  |                Multi-GPU Execution Engine                 |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  |  | GPU 0  |  | GPU 1  |  | GPU 2  |  | GPU 3  |           |     |
|  |  +--------+  +--------+  +--------+  +--------+           |     |
|  +----------------------------------------------------------+     |
+------------------------------------------------------------------+
```

### Core Components

#### 1. Graph Storage Engine

Efficient storage formats for sparse graphs with vertex/edge features.

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum

class GraphFormat(Enum):
    CSR = "csr"           # Compressed Sparse Row
    CSC = "csc"           # Compressed Sparse Column
    COO = "coo"           # Coordinate format
    HYBRID = "hybrid"     # CSR + CSC for bidirectional

@dataclass
class GraphStorage:
    """Core graph storage with multiple format support."""
    num_nodes: int
    num_edges: int

    # CSR format (row-major, efficient for outgoing edges)
    csr_indptr: np.ndarray      # [num_nodes + 1]
    csr_indices: np.ndarray     # [num_edges]

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
    def from_edge_list(cls,
                       src: np.ndarray,
                       dst: np.ndarray,
                       num_nodes: Optional[int] = None,
                       edge_data: Optional[np.ndarray] = None) -> 'GraphStorage':
        """Construct graph from edge list."""
        if num_nodes is None:
            num_nodes = max(src.max(), dst.max()) + 1

        num_edges = len(src)

        # Build CSR
        csr_indptr = np.zeros(num_nodes + 1, dtype=np.int64)
        for s in src:
            csr_indptr[s + 1] += 1
        csr_indptr = np.cumsum(csr_indptr)

        # Sort edges by source
        sort_idx = np.argsort(src)
        csr_indices = dst[sort_idx]

        if edge_data is not None:
            edge_data = edge_data[sort_idx]

        return cls(
            num_nodes=num_nodes,
            num_edges=num_edges,
            csr_indptr=csr_indptr,
            csr_indices=csr_indices,
            edge_data=edge_data
        )

    def to_csc(self) -> None:
        """Build CSC format from CSR (transpose)."""
        # Transpose CSR to get CSC
        csc_indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)

        # Count incoming edges per node
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

    def get_neighbors(self, node_id: int, direction: str = 'out') -> np.ndarray:
        """Get neighbors of a node."""
        if direction == 'out':
            start = self.csr_indptr[node_id]
            end = self.csr_indptr[node_id + 1]
            return self.csr_indices[start:end]
        else:  # incoming
            if self.csc_indptr is None:
                self.to_csc()
            start = self.csc_indptr[node_id]
            end = self.csc_indptr[node_id + 1]
            return self.csc_indices[start:end]

    def degree(self, node_id: int, direction: str = 'out') -> int:
        """Get degree of a node."""
        if direction == 'out':
            return self.csr_indptr[node_id + 1] - self.csr_indptr[node_id]
        else:
            if self.csc_indptr is None:
                self.to_csc()
            return self.csc_indptr[node_id + 1] - self.csc_indptr[node_id]


class PartitionedGraph:
    """Graph partitioned across multiple devices/machines."""

    def __init__(self, graph: GraphStorage, num_partitions: int):
        self.global_graph = graph
        self.num_partitions = num_partitions
        self.partitions: List[GraphStorage] = []
        self.node_to_partition: np.ndarray = np.zeros(graph.num_nodes, dtype=np.int32)
        self.local_to_global: List[np.ndarray] = []
        self.global_to_local: np.ndarray = np.zeros(graph.num_nodes, dtype=np.int64)

    def partition_metis(self) -> None:
        """Partition graph using METIS-style algorithm."""
        # Simplified balanced partitioning
        nodes_per_partition = self.global_graph.num_nodes // self.num_partitions

        for i in range(self.global_graph.num_nodes):
            partition_id = min(i // nodes_per_partition, self.num_partitions - 1)
            self.node_to_partition[i] = partition_id

        # Build local graphs for each partition
        for p in range(self.num_partitions):
            local_nodes = np.where(self.node_to_partition == p)[0]
            self.local_to_global.append(local_nodes)

            for local_idx, global_idx in enumerate(local_nodes):
                self.global_to_local[global_idx] = local_idx

    def get_partition(self, partition_id: int) -> Tuple[GraphStorage, np.ndarray]:
        """Get subgraph for a partition with halo nodes."""
        local_nodes = self.local_to_global[partition_id]

        # Find halo nodes (neighbors in other partitions)
        halo_nodes = set()
        for node in local_nodes:
            neighbors = self.global_graph.get_neighbors(node, 'out')
            for neighbor in neighbors:
                if self.node_to_partition[neighbor] != partition_id:
                    halo_nodes.add(neighbor)

        # Build local subgraph
        all_nodes = np.concatenate([local_nodes, np.array(list(halo_nodes))])

        return self._extract_subgraph(all_nodes), all_nodes

    def _extract_subgraph(self, nodes: np.ndarray) -> GraphStorage:
        """Extract subgraph induced by node set."""
        node_set = set(nodes)
        node_map = {n: i for i, n in enumerate(nodes)}

        edges_src = []
        edges_dst = []

        for node in nodes:
            for neighbor in self.global_graph.get_neighbors(node, 'out'):
                if neighbor in node_set:
                    edges_src.append(node_map[node])
                    edges_dst.append(node_map[neighbor])

        return GraphStorage.from_edge_list(
            np.array(edges_src, dtype=np.int64),
            np.array(edges_dst, dtype=np.int64),
            num_nodes=len(nodes)
        )
```

#### 2. Message Passing Engine

Core GNN computation primitive implementing AGGREGATE and MESSAGE functions.

```python
import torch
import torch.nn as nn
from typing import Callable, Optional
from abc import ABC, abstractmethod

class MessageFunction(ABC):
    """Abstract message function for edges."""

    @abstractmethod
    def __call__(self,
                 src_features: torch.Tensor,
                 dst_features: torch.Tensor,
                 edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

class AggregateFunction(ABC):
    """Abstract aggregation function for neighborhoods."""

    @abstractmethod
    def __call__(self,
                 messages: torch.Tensor,
                 index: torch.Tensor,
                 num_nodes: int) -> torch.Tensor:
        pass

class MessagePassingEngine:
    """High-performance message passing for GNNs."""

    def __init__(self, device: str = 'cuda'):
        self.device = device

    def propagate(self,
                  edge_index: torch.Tensor,
                  node_features: torch.Tensor,
                  edge_features: Optional[torch.Tensor] = None,
                  message_fn: Optional[Callable] = None,
                  aggregate_fn: str = 'sum',
                  flow: str = 'source_to_target') -> torch.Tensor:
        """
        Perform message passing on graph.

        Args:
            edge_index: [2, E] tensor of edges
            node_features: [N, F] node feature tensor
            edge_features: Optional [E, Fe] edge features
            message_fn: Function to compute messages
            aggregate_fn: 'sum', 'mean', 'max'
            flow: Direction of message flow

        Returns:
            Aggregated messages for each node
        """
        if flow == 'source_to_target':
            src, dst = edge_index[0], edge_index[1]
        else:
            dst, src = edge_index[0], edge_index[1]

        # Gather source features
        src_features = node_features[src]  # [E, F]
        dst_features = node_features[dst]  # [E, F]

        # Compute messages
        if message_fn is not None:
            messages = message_fn(src_features, dst_features, edge_features)
        else:
            messages = src_features

        # Aggregate messages
        num_nodes = node_features.size(0)
        out = self._aggregate(messages, dst, num_nodes, aggregate_fn)

        return out

    def _aggregate(self,
                   messages: torch.Tensor,
                   index: torch.Tensor,
                   num_nodes: int,
                   agg_type: str) -> torch.Tensor:
        """Aggregate messages to nodes using scatter operations."""
        out = torch.zeros(num_nodes, messages.size(-1),
                         device=self.device, dtype=messages.dtype)

        if agg_type == 'sum':
            out.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)
        elif agg_type == 'mean':
            out.scatter_add_(0, index.unsqueeze(-1).expand_as(messages), messages)
            # Compute degrees
            ones = torch.ones(index.size(0), device=self.device)
            degree = torch.zeros(num_nodes, device=self.device)
            degree.scatter_add_(0, index, ones)
            degree = degree.clamp(min=1)  # Avoid division by zero
            out = out / degree.unsqueeze(-1)
        elif agg_type == 'max':
            out.fill_(float('-inf'))
            out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(messages),
                               messages, reduce='amax')
            out[out == float('-inf')] = 0

        return out


class GNNLayer(nn.Module):
    """Generic GNN layer with configurable message passing."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 message_type: str = 'gcn',
                 aggregate_type: str = 'sum',
                 normalize: bool = True,
                 bias: bool = True):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.message_type = message_type
        self.aggregate_type = aggregate_type
        self.normalize = normalize

        # Learnable parameters
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # For attention-based messages (GAT)
        if message_type == 'gat':
            self.att_src = nn.Parameter(torch.zeros(1, out_channels))
            self.att_dst = nn.Parameter(torch.zeros(1, out_channels))
            nn.init.xavier_uniform_(self.att_src)
            nn.init.xavier_uniform_(self.att_dst)

        self.mp_engine = MessagePassingEngine()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [N, in_channels] node features
            edge_index: [2, E] edge indices
            edge_weight: Optional [E] edge weights

        Returns:
            [N, out_channels] updated node features
        """
        # Transform features
        x = self.lin(x)

        # Normalization for GCN
        if self.normalize and self.message_type == 'gcn':
            edge_weight = self._compute_gcn_norm(edge_index, x.size(0))

        # Message function based on type
        if self.message_type == 'gcn':
            def message_fn(src, dst, edge_feat):
                if edge_weight is not None:
                    return src * edge_weight.unsqueeze(-1)
                return src
        elif self.message_type == 'gat':
            def message_fn(src, dst, edge_feat):
                # Attention scores
                alpha_src = (src * self.att_src).sum(dim=-1)
                alpha_dst = (dst * self.att_dst).sum(dim=-1)
                alpha = torch.nn.functional.leaky_relu(alpha_src + alpha_dst)
                alpha = torch.softmax(alpha, dim=0)  # Simplified
                return src * alpha.unsqueeze(-1)
        else:  # Simple sum
            message_fn = None

        # Message passing
        out = self.mp_engine.propagate(
            edge_index, x,
            message_fn=message_fn,
            aggregate_fn=self.aggregate_type
        )

        if self.bias is not None:
            out = out + self.bias

        return out

    def _compute_gcn_norm(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute symmetric normalization for GCN."""
        src, dst = edge_index

        # Compute degrees
        deg = torch.zeros(num_nodes, device=edge_index.device)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))

        # D^{-1/2}
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Norm = D_src^{-1/2} * D_dst^{-1/2}
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]

        return norm
```

#### 3. Neighbor Sampler

Efficient sampling for mini-batch training on large graphs.

```python
import numpy as np
import torch
from typing import List, Tuple, Optional
from dataclasses import dataclass
import random

@dataclass
class SampledSubgraph:
    """Result of neighbor sampling."""
    # Node IDs in original graph
    node_ids: torch.Tensor
    # Edge indices in sampled subgraph
    edge_index: torch.Tensor
    # Number of nodes at each layer
    layer_sizes: List[int]
    # Mapping from original to sampled IDs
    node_mapping: dict

class NeighborSampler:
    """Multi-hop neighbor sampler for mini-batch GNN training."""

    def __init__(self,
                 graph: 'GraphStorage',
                 fanouts: List[int],
                 replace: bool = False,
                 device: str = 'cuda'):
        """
        Args:
            graph: Graph storage object
            fanouts: Number of neighbors to sample per hop [-1 for all]
            replace: Sample with replacement
        """
        self.graph = graph
        self.fanouts = fanouts
        self.replace = replace
        self.device = device

        # Convert to torch for GPU operations
        self.csr_indptr = torch.from_numpy(graph.csr_indptr).to(device)
        self.csr_indices = torch.from_numpy(graph.csr_indices).to(device)

    def sample(self, seed_nodes: torch.Tensor) -> SampledSubgraph:
        """
        Sample k-hop neighborhood around seed nodes.

        Args:
            seed_nodes: Target nodes for batch

        Returns:
            Sampled subgraph with all info needed for GNN
        """
        batch_nodes = seed_nodes.clone()
        layer_sizes = [len(seed_nodes)]
        all_edges_src = []
        all_edges_dst = []

        frontier = seed_nodes
        sampled_nodes = set(seed_nodes.cpu().numpy().tolist())

        # Sample each hop
        for fanout in self.fanouts:
            # Get neighbors for frontier nodes
            neighbors, edges_src, edges_dst = self._sample_neighbors(
                frontier, fanout
            )

            # Add new nodes to batch
            new_nodes = [n for n in neighbors.cpu().numpy() if n not in sampled_nodes]
            if new_nodes:
                new_nodes_tensor = torch.tensor(new_nodes, device=self.device)
                batch_nodes = torch.cat([batch_nodes, new_nodes_tensor])
                sampled_nodes.update(new_nodes)

            all_edges_src.append(edges_src)
            all_edges_dst.append(edges_dst)

            # Update frontier
            frontier = neighbors
            layer_sizes.append(len(batch_nodes))

        # Build edge index with local node IDs
        node_mapping = {n.item(): i for i, n in enumerate(batch_nodes)}

        if all_edges_src:
            edges_src = torch.cat(all_edges_src)
            edges_dst = torch.cat(all_edges_dst)

            # Map to local IDs
            local_src = torch.tensor([node_mapping[s.item()] for s in edges_src],
                                    device=self.device)
            local_dst = torch.tensor([node_mapping[d.item()] for d in edges_dst],
                                    device=self.device)
            edge_index = torch.stack([local_src, local_dst])
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=self.device)

        return SampledSubgraph(
            node_ids=batch_nodes,
            edge_index=edge_index,
            layer_sizes=layer_sizes,
            node_mapping=node_mapping
        )

    def _sample_neighbors(self,
                          nodes: torch.Tensor,
                          fanout: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample neighbors for a set of nodes."""
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
            edges_src.extend([node] * len(sampled))
            edges_dst.extend(sampled.tolist())

        if all_neighbors:
            neighbors = torch.cat(all_neighbors)
            edges_src = torch.tensor(edges_src, device=self.device)
            edges_dst = torch.tensor(edges_dst, device=self.device)
        else:
            neighbors = torch.tensor([], dtype=torch.long, device=self.device)
            edges_src = torch.tensor([], dtype=torch.long, device=self.device)
            edges_dst = torch.tensor([], dtype=torch.long, device=self.device)

        return neighbors, edges_src, edges_dst


class PPRSampler:
    """Personalized PageRank based sampling."""

    def __init__(self,
                 graph: 'GraphStorage',
                 alpha: float = 0.15,
                 epsilon: float = 1e-5,
                 max_iters: int = 100):
        self.graph = graph
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_iters = max_iters

    def sample(self,
               seed_nodes: torch.Tensor,
               top_k: int = 100) -> SampledSubgraph:
        """
        Sample nodes using PPR scores from seed nodes.
        """
        num_nodes = self.graph.num_nodes
        ppr_scores = np.zeros(num_nodes)

        # Initialize with seed nodes
        seed_list = seed_nodes.cpu().numpy()
        for seed in seed_list:
            ppr_scores[seed] = 1.0 / len(seed_list)

        # Power iteration
        for _ in range(self.max_iters):
            new_scores = np.zeros(num_nodes)

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
        top_nodes = np.argsort(ppr_scores)[-top_k:][::-1]

        # Build subgraph from selected nodes
        node_set = set(top_nodes)
        edges_src = []
        edges_dst = []

        for node in top_nodes:
            for neighbor in self.graph.get_neighbors(node, 'out'):
                if neighbor in node_set:
                    edges_src.append(node)
                    edges_dst.append(neighbor)

        node_mapping = {n: i for i, n in enumerate(top_nodes)}

        device = seed_nodes.device
        return SampledSubgraph(
            node_ids=torch.tensor(top_nodes, device=device),
            edge_index=torch.tensor([
                [node_mapping[s] for s in edges_src],
                [node_mapping[d] for d in edges_dst]
            ], device=device),
            layer_sizes=[len(top_nodes)],
            node_mapping=node_mapping
        )
```

#### 4. GPU Kernels

High-performance CUDA kernels for sparse operations.

```python
import torch
from torch.utils.cpp_extension import load_inline

# CUDA kernel for Sparse Matrix-Matrix Multiplication (SpMM)
spmm_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void spmm_csr_kernel(
    const int64_t* __restrict__ rowptr,
    const int64_t* __restrict__ col,
    const scalar_t* __restrict__ values,
    const scalar_t* __restrict__ mat,
    scalar_t* __restrict__ out,
    int num_rows,
    int num_cols,
    int feat_dim
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int feat = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows && feat < feat_dim) {
        scalar_t sum = 0;
        int start = rowptr[row];
        int end = rowptr[row + 1];

        for (int i = start; i < end; i++) {
            int c = col[i];
            scalar_t val = values ? values[i] : 1.0;
            sum += val * mat[c * feat_dim + feat];
        }

        out[row * feat_dim + feat] = sum;
    }
}

torch::Tensor spmm_csr_cuda(
    torch::Tensor rowptr,
    torch::Tensor col,
    torch::Tensor values,
    torch::Tensor mat
) {
    auto num_rows = rowptr.size(0) - 1;
    auto feat_dim = mat.size(1);

    auto out = torch::zeros({num_rows, feat_dim}, mat.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (num_rows + threads.x - 1) / threads.x,
        (feat_dim + threads.y - 1) / threads.y
    );

    AT_DISPATCH_FLOATING_TYPES(mat.scalar_type(), "spmm_csr_cuda", ([&] {
        spmm_csr_kernel<scalar_t><<<blocks, threads>>>(
            rowptr.data_ptr<int64_t>(),
            col.data_ptr<int64_t>(),
            values.defined() ? values.data_ptr<scalar_t>() : nullptr,
            mat.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            num_rows,
            mat.size(0),
            feat_dim
        );
    }));

    return out;
}
"""

# Scatter-gather operations for message passing
scatter_cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void scatter_add_kernel(
    const scalar_t* __restrict__ src,
    const int64_t* __restrict__ index,
    scalar_t* __restrict__ out,
    int num_edges,
    int feat_dim
) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    int feat = blockIdx.y * blockDim.y + threadIdx.y;

    if (edge < num_edges && feat < feat_dim) {
        int64_t dst = index[edge];
        atomicAdd(&out[dst * feat_dim + feat], src[edge * feat_dim + feat]);
    }
}

torch::Tensor scatter_add_cuda(
    torch::Tensor src,
    torch::Tensor index,
    int num_nodes
) {
    auto num_edges = src.size(0);
    auto feat_dim = src.size(1);

    auto out = torch::zeros({num_nodes, feat_dim}, src.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (num_edges + threads.x - 1) / threads.x,
        (feat_dim + threads.y - 1) / threads.y
    );

    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "scatter_add_cuda", ([&] {
        scatter_add_kernel<scalar_t><<<blocks, threads>>>(
            src.data_ptr<scalar_t>(),
            index.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            num_edges,
            feat_dim
        );
    }));

    return out;
}

template <typename scalar_t>
__global__ void gather_kernel(
    const scalar_t* __restrict__ src,
    const int64_t* __restrict__ index,
    scalar_t* __restrict__ out,
    int num_indices,
    int feat_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int feat = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < num_indices && feat < feat_dim) {
        int64_t src_idx = index[idx];
        out[idx * feat_dim + feat] = src[src_idx * feat_dim + feat];
    }
}

torch::Tensor gather_cuda(
    torch::Tensor src,
    torch::Tensor index
) {
    auto num_indices = index.size(0);
    auto feat_dim = src.size(1);

    auto out = torch::empty({num_indices, feat_dim}, src.options());

    dim3 threads(16, 16);
    dim3 blocks(
        (num_indices + threads.x - 1) / threads.x,
        (feat_dim + threads.y - 1) / threads.y
    );

    AT_DISPATCH_FLOATING_TYPES(src.scalar_type(), "gather_cuda", ([&] {
        gather_kernel<scalar_t><<<blocks, threads>>>(
            src.data_ptr<scalar_t>(),
            index.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            num_indices,
            feat_dim
        );
    }));

    return out;
}
"""


class GPUKernelPool:
    """Pool of optimized GPU kernels for GNN operations."""

    def __init__(self):
        self._compiled = False
        self._spmm_fn = None
        self._scatter_add_fn = None
        self._gather_fn = None

    def compile(self):
        """JIT compile CUDA kernels."""
        if self._compiled:
            return

        # Compile SpMM
        spmm_module = load_inline(
            name='spmm_cuda',
            cpp_sources='torch::Tensor spmm_csr_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);',
            cuda_sources=spmm_cuda_source,
            functions=['spmm_csr_cuda'],
            verbose=False
        )
        self._spmm_fn = spmm_module.spmm_csr_cuda

        # Compile scatter/gather
        scatter_module = load_inline(
            name='scatter_cuda',
            cpp_sources='''
                torch::Tensor scatter_add_cuda(torch::Tensor, torch::Tensor, int);
                torch::Tensor gather_cuda(torch::Tensor, torch::Tensor);
            ''',
            cuda_sources=scatter_cuda_source,
            functions=['scatter_add_cuda', 'gather_cuda'],
            verbose=False
        )
        self._scatter_add_fn = scatter_module.scatter_add_cuda
        self._gather_fn = scatter_module.gather_cuda

        self._compiled = True

    def spmm(self,
             rowptr: torch.Tensor,
             col: torch.Tensor,
             values: torch.Tensor,
             mat: torch.Tensor) -> torch.Tensor:
        """Sparse matrix-matrix multiplication."""
        if not self._compiled:
            self.compile()
        return self._spmm_fn(rowptr, col, values, mat)

    def scatter_add(self,
                    src: torch.Tensor,
                    index: torch.Tensor,
                    num_nodes: int) -> torch.Tensor:
        """Scatter-add operation."""
        if not self._compiled:
            self.compile()
        return self._scatter_add_fn(src, index, num_nodes)

    def gather(self,
               src: torch.Tensor,
               index: torch.Tensor) -> torch.Tensor:
        """Gather operation."""
        if not self._compiled:
            self.compile()
        return self._gather_fn(src, index)
```

### Enterprise Features

#### Multi-GPU Distributed Training

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import List, Optional
import os

class DistributedGNNTrainer:
    """Multi-GPU distributed GNN training with graph partitioning."""

    def __init__(self,
                 model: torch.nn.Module,
                 graph: 'GraphStorage',
                 num_gpus: int,
                 partition_method: str = 'metis'):
        self.model = model
        self.graph = graph
        self.num_gpus = num_gpus
        self.partition_method = partition_method

        # Initialize distributed environment
        self._init_distributed()

        # Partition graph
        self.partitioned_graph = PartitionedGraph(graph, num_gpus)
        if partition_method == 'metis':
            self.partitioned_graph.partition_metis()

        # Wrap model in DDP
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device(f'cuda:{self.local_rank}')
        self.model = self.model.to(self.device)
        self.model = DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank]
        )

    def _init_distributed(self):
        """Initialize distributed training environment."""
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

    def train_epoch(self,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: torch.nn.Module,
                    batch_size: int = 1024) -> float:
        """Train one epoch with distributed mini-batches."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Get local partition
        local_graph, local_nodes = self.partitioned_graph.get_partition(self.local_rank)

        # Create sampler for local partition
        sampler = NeighborSampler(local_graph, fanouts=[10, 10], device=str(self.device))

        # Training loop
        local_node_ids = torch.arange(len(self.partitioned_graph.local_to_global[self.local_rank]))

        for i in range(0, len(local_node_ids), batch_size):
            batch_seeds = local_node_ids[i:i+batch_size].to(self.device)

            # Sample subgraph
            subgraph = sampler.sample(batch_seeds)

            # Get features for sampled nodes
            global_ids = local_nodes[subgraph.node_ids.cpu().numpy()]
            features = torch.tensor(
                self.graph.node_features['x'][global_ids],
                device=self.device, dtype=torch.float32
            )
            labels = torch.tensor(
                self.graph.node_features['y'][global_ids[:len(batch_seeds)]],
                device=self.device, dtype=torch.long
            )

            # Forward pass
            optimizer.zero_grad()
            out = self.model(features, subgraph.edge_index)
            loss = loss_fn(out[:len(batch_seeds)], labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        # Synchronize loss across GPUs
        loss_tensor = torch.tensor([total_loss, num_batches], device=self.device)
        dist.all_reduce(loss_tensor)

        return loss_tensor[0].item() / loss_tensor[1].item()

    def synchronize_halo_nodes(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Synchronize halo node embeddings across partitions."""
        # Gather all embeddings
        all_embeddings = [torch.zeros_like(embeddings) for _ in range(self.num_gpus)]
        dist.all_gather(all_embeddings, embeddings)

        # Update halo nodes with correct embeddings
        # Implementation depends on specific halo node mapping
        return embeddings


class VertexReorderOptimizer:
    """Optimize graph layout for cache efficiency."""

    def __init__(self, graph: 'GraphStorage'):
        self.graph = graph

    def reorder_rcm(self) -> np.ndarray:
        """Reverse Cuthill-McKee ordering for bandwidth reduction."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        # Build scipy sparse matrix
        adj = csr_matrix(
            (np.ones(self.graph.num_edges),
             self.graph.csr_indices,
             self.graph.csr_indptr),
            shape=(self.graph.num_nodes, self.graph.num_nodes)
        )

        # Get RCM ordering
        perm = reverse_cuthill_mckee(adj)
        return perm

    def reorder_metis(self, num_parts: int = 8) -> np.ndarray:
        """METIS-based reordering for locality."""
        # Group nodes by partition, then order within partition by degree
        partition_ids = np.zeros(self.graph.num_nodes, dtype=np.int32)
        nodes_per_part = self.graph.num_nodes // num_parts

        for i in range(self.graph.num_nodes):
            partition_ids[i] = min(i // nodes_per_part, num_parts - 1)

        # Compute degrees
        degrees = np.diff(self.graph.csr_indptr)

        # Sort within each partition by degree
        perm = []
        for p in range(num_parts):
            part_nodes = np.where(partition_ids == p)[0]
            part_degrees = degrees[part_nodes]
            sorted_idx = np.argsort(part_degrees)[::-1]
            perm.extend(part_nodes[sorted_idx])

        return np.array(perm)

    def apply_reordering(self, perm: np.ndarray) -> 'GraphStorage':
        """Apply reordering to graph."""
        inv_perm = np.argsort(perm)

        # Reorder CSR
        new_indptr = np.zeros_like(self.graph.csr_indptr)
        new_indices = []

        for new_id in range(self.graph.num_nodes):
            old_id = perm[new_id]
            start = self.graph.csr_indptr[old_id]
            end = self.graph.csr_indptr[old_id + 1]

            # Map old neighbors to new IDs
            old_neighbors = self.graph.csr_indices[start:end]
            new_neighbors = inv_perm[old_neighbors]

            new_indptr[new_id + 1] = new_indptr[new_id] + len(new_neighbors)
            new_indices.extend(sorted(new_neighbors))  # Sort for cache efficiency

        # Reorder features
        new_node_features = {}
        for key, feat in self.graph.node_features.items():
            new_node_features[key] = feat[perm]

        return GraphStorage(
            num_nodes=self.graph.num_nodes,
            num_edges=self.graph.num_edges,
            csr_indptr=new_indptr,
            csr_indices=np.array(new_indices, dtype=np.int64),
            node_features=new_node_features
        )
```

## API Reference

### Graph Construction API

```python
# Create graph from edge list
graph = GraphStorage.from_edge_list(
    src=np.array([0, 1, 2, 1]),
    dst=np.array([1, 2, 0, 3]),
    num_nodes=4
)

# Add features
graph.node_features['x'] = np.random.randn(4, 128)
graph.node_features['y'] = np.array([0, 1, 0, 1])

# Build CSC for incoming edges
graph.to_csc()
```

### GNN Training API

```python
# Create model
model = GNNLayer(128, 64, message_type='gcn')

# Create sampler
sampler = NeighborSampler(graph, fanouts=[10, 5])

# Training loop
for epoch in range(100):
    for batch_seeds in dataloader:
        subgraph = sampler.sample(batch_seeds)
        features = graph.node_features['x'][subgraph.node_ids]
        out = model(features, subgraph.edge_index)
        loss = criterion(out[:len(batch_seeds)], labels)
        loss.backward()
        optimizer.step()
```

### Distributed API

```python
# Initialize distributed training
trainer = DistributedGNNTrainer(
    model=model,
    graph=graph,
    num_gpus=4,
    partition_method='metis'
)

# Train
for epoch in range(100):
    loss = trainer.train_epoch(optimizer, loss_fn)
    print(f"Epoch {epoch}: Loss = {loss:.4f}")
```

## Implementation Phases

### Phase 1: Core Graph Storage (Weeks 1-2)
- CSR/CSC format implementation
- Edge list to sparse conversion
- Basic neighbor queries
- Feature storage

### Phase 2: Message Passing (Weeks 3-4)
- Aggregation functions (sum, mean, max)
- Message functions (GCN, GAT, GraphSAGE)
- Basic GNN layers
- CPU implementation

### Phase 3: Neighbor Sampling (Weeks 5-6)
- K-hop sampling
- Uniform sampling
- PPR-based sampling
- Mini-batch construction

### Phase 4: GPU Kernels (Weeks 7-9)
- SpMM CUDA kernel
- Scatter/gather operations
- Optimized aggregation
- Memory management

### Phase 5: Multi-GPU (Weeks 10-12)
- Graph partitioning
- Distributed training
- Halo node synchronization
- Vertex reordering

### Phase 6: Enterprise Features (Weeks 13-16)
- Batched mini-graphs
- Dynamic graph updates
- Advanced partitioning
- GNN compiler optimizations

## Testing Strategy

### Unit Tests

```python
import pytest
import torch
import numpy as np

class TestGraphStorage:
    def test_from_edge_list(self):
        src = np.array([0, 1, 2])
        dst = np.array([1, 2, 0])
        graph = GraphStorage.from_edge_list(src, dst)

        assert graph.num_nodes == 3
        assert graph.num_edges == 3
        assert len(graph.csr_indptr) == 4

    def test_get_neighbors(self):
        src = np.array([0, 0, 1])
        dst = np.array([1, 2, 2])
        graph = GraphStorage.from_edge_list(src, dst)

        neighbors = graph.get_neighbors(0, 'out')
        assert set(neighbors) == {1, 2}

    def test_csc_conversion(self):
        src = np.array([0, 1])
        dst = np.array([1, 0])
        graph = GraphStorage.from_edge_list(src, dst)
        graph.to_csc()

        assert graph.csc_indptr is not None
        in_neighbors = graph.get_neighbors(0, 'in')
        assert 1 in in_neighbors


class TestMessagePassing:
    def test_sum_aggregation(self):
        engine = MessagePassingEngine(device='cpu')
        edge_index = torch.tensor([[0, 1], [1, 2]])
        features = torch.randn(3, 16)

        out = engine.propagate(edge_index, features, aggregate_fn='sum')
        assert out.shape == (3, 16)

    def test_gcn_normalization(self):
        layer = GNNLayer(16, 8, message_type='gcn')
        edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
        x = torch.randn(3, 16)

        out = layer(x, edge_index)
        assert out.shape == (3, 8)


class TestNeighborSampler:
    def test_sampling(self):
        src = np.array([0, 0, 1, 1, 2])
        dst = np.array([1, 2, 2, 3, 3])
        graph = GraphStorage.from_edge_list(src, dst)

        sampler = NeighborSampler(graph, fanouts=[2], device='cpu')
        seeds = torch.tensor([3])
        subgraph = sampler.sample(seeds)

        assert len(subgraph.node_ids) > 1
        assert 3 in subgraph.node_ids.tolist()
```

### Integration Tests

```python
class TestEndToEndTraining:
    def test_single_gpu_training(self):
        # Create synthetic graph
        num_nodes = 1000
        num_edges = 5000
        src = np.random.randint(0, num_nodes, num_edges)
        dst = np.random.randint(0, num_nodes, num_edges)

        graph = GraphStorage.from_edge_list(src, dst, num_nodes)
        graph.node_features['x'] = np.random.randn(num_nodes, 64).astype(np.float32)
        graph.node_features['y'] = np.random.randint(0, 10, num_nodes)

        # Create model and train
        model = GNNLayer(64, 32)
        sampler = NeighborSampler(graph, fanouts=[5, 5], device='cpu')
        optimizer = torch.optim.Adam(model.parameters())

        # One training step
        seeds = torch.arange(32)
        subgraph = sampler.sample(seeds)
        features = torch.tensor(graph.node_features['x'][subgraph.node_ids.numpy()])
        labels = torch.tensor(graph.node_features['y'][seeds.numpy()])

        out = model(features, subgraph.edge_index)
        loss = torch.nn.functional.cross_entropy(out[:32], labels)
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
```

### Performance Benchmarks

```python
class TestPerformance:
    def test_spmm_performance(self):
        """Benchmark SpMM kernel."""
        # Create large sparse matrix
        num_nodes = 100000
        avg_degree = 50

        # Measure time for 100 iterations
        import time
        start = time.time()
        for _ in range(100):
            # SpMM operation
            pass
        elapsed = time.time() - start

        throughput = 100 * num_nodes * avg_degree / elapsed
        print(f"SpMM throughput: {throughput/1e9:.2f} GFLOPS")

    def test_sampling_throughput(self):
        """Benchmark neighbor sampling."""
        # Should achieve >1M nodes/sec
        pass
```

## Stretch Goals

### Dynamic Graph Support

```python
class DynamicGraph:
    """Graph with efficient incremental updates."""

    def __init__(self, initial_graph: GraphStorage):
        self.graph = initial_graph
        self.pending_additions = []
        self.pending_deletions = []
        self.version = 0

    def add_edge(self, src: int, dst: int):
        """Add edge (batched, applied on rebuild)."""
        self.pending_additions.append((src, dst))

    def remove_edge(self, src: int, dst: int):
        """Remove edge (batched, applied on rebuild)."""
        self.pending_deletions.append((src, dst))

    def rebuild(self):
        """Apply pending changes and rebuild indices."""
        # Apply changes
        # Rebuild CSR/CSC
        self.version += 1
        self.pending_additions = []
        self.pending_deletions = []
```

### GNN Compiler

```python
class GNNCompiler:
    """Compile GNN models for optimized execution."""

    def compile(self, model: torch.nn.Module, sample_input: dict) -> 'CompiledGNN':
        """
        Compile GNN model with optimizations:
        - Operator fusion
        - Memory planning
        - Kernel selection
        """
        # Trace model
        # Optimize graph
        # Generate optimized code
        pass
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| SpMM throughput | >100 GFLOPS | On V100 GPU |
| Sampling rate | >1M nodes/sec | With fanout [10, 10] |
| Training throughput | >100K nodes/sec | Full GCN pipeline |
| Memory efficiency | <2x theoretical | Compared to dense |
| Multi-GPU scaling | >80% efficiency | Up to 8 GPUs |

## Dependencies

- PyTorch >= 2.0
- CUDA >= 11.0
- NumPy
- SciPy (for sparse operations)
- (Optional) METIS for partitioning
- (Optional) PyTorch Geometric for reference

## References

- DGL: Deep Graph Library
- PyTorch Geometric
- GraphSAGE paper
- GAT paper
- Cluster-GCN for mini-batch training
