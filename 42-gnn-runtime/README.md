# GNN Runtime

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A high-performance graph neural network runtime system optimized for GPU sparse operations, efficient graph partitioning, and scalable message passing. This project implements the core primitives needed for training and inference of GNNs on large-scale graphs with millions to billions of edges.

## Features

- **Efficient Graph Storage**
  - CSR/CSC sparse matrix formats
  - Hybrid format for bidirectional operations
  - Memory-mapped support for large graphs
  - Node and edge feature storage

- **Message Passing Engine**
  - Configurable aggregation functions (sum, mean, max)
  - GCN, GAT, and GraphSAGE layer implementations
  - Edge feature support in message computation
  - GPU-accelerated operations

- **Neighbor Sampling**
  - Multi-hop uniform sampling
  - PPR-based importance sampling
  - Efficient mini-batch graph construction
  - Parallel sampling implementation

- **GPU Kernels**
  - SpMM CUDA kernel (CSR format)
  - Scatter-add and gather operations
  - Memory-coalesced access patterns
  - JIT-compiled kernels

- **Multi-GPU Training**
  - Graph partitioning (METIS-style)
  - Distributed training with DDP
  - Halo node synchronization
  - Vertex reordering for cache efficiency

## Quick Start

### Installation

```bash
cd projects/41-gnn-runtime
pip install -e ".[full]"
```

### Basic Usage

```python
from gnn_runtime import GraphStorage, GNNLayer, NeighborSampler
import numpy as np
import torch

# Create graph from edge list
src = np.array([0, 1, 2, 1])
dst = np.array([1, 2, 0, 3])
graph = GraphStorage.from_edge_list(src, dst, num_nodes=4)

# Add node features
graph.node_features['x'] = np.random.randn(4, 128).astype(np.float32)
graph.node_features['y'] = np.array([0, 1, 0, 1])

# Build CSC for incoming edge queries
graph.to_csc()

# Create GNN layer
model = GNNLayer(128, 64, message_type='gcn')

# Create sampler for mini-batch training
sampler = NeighborSampler(graph, fanouts=[10, 5], device='cuda')

# Sample and forward pass
seeds = torch.tensor([0, 1])
subgraph = sampler.sample(seeds)
features = torch.tensor(graph.node_features['x'][subgraph.node_ids.numpy()])
output = model(features, subgraph.edge_index)
```

### Training Loop

```python
from gnn_runtime import GraphStorage, GNNLayer, NeighborSampler
import torch
import torch.nn.functional as F

# Create model and optimizer
model = GNNLayer(128, 64, message_type='gcn')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Create sampler
sampler = NeighborSampler(graph, fanouts=[10, 5], device='cuda')

# Training loop
for epoch in range(100):
    for batch_seeds in dataloader:
        # Sample subgraph
        subgraph = sampler.sample(batch_seeds)

        # Get features
        features = torch.tensor(
            graph.node_features['x'][subgraph.node_ids.numpy()],
            device='cuda'
        )
        labels = torch.tensor(
            graph.node_features['y'][batch_seeds.numpy()],
            device='cuda'
        )

        # Forward pass
        optimizer.zero_grad()
        out = model(features, subgraph.edge_index)
        loss = F.cross_entropy(out[:len(batch_seeds)], labels)

        # Backward pass
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
```

### Distributed Training

```python
from gnn_runtime import DistributedGNNTrainer

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

## Architecture

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

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| SpMM throughput | >100 GFLOPS | On V100 GPU |
| Sampling rate | >1M nodes/sec | With fanout [10, 10] |
| Training throughput | >100K nodes/sec | Full GCN pipeline |
| Memory efficiency | <2x theoretical | Compared to dense |
| Multi-GPU scaling | >80% efficiency | Up to 8 GPUs |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_graph_storage.py -v

# Run with coverage
pytest tests/ --cov=gnn_runtime --cov-report=html
```

## Documentation

- [Architecture Overview](docs/ARCHITECTURE.md) - System design and components
- [API Reference](docs/API.md) - Complete API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions

## Dependencies

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.0
- NumPy
- SciPy (for sparse operations)
- (Optional) METIS for graph partitioning

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- DGL: Deep Graph Library
- PyTorch Geometric
- GraphSAGE: Inductive Representation Learning on Large Graphs
- GAT: Graph Attention Networks
- Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks
