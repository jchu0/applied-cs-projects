# Graph Neural Network Runtime - API Documentation

## Table of Contents

1. [Graph Layers API](#graph-layers-api)
2. [Graph Data API](#graph-data-api)
3. [Sampling API](#sampling-api)
4. [Batch Processing API](#batch-processing-api)
5. [Utilities](#utilities)

## Graph Layers API

### GCNConv

Graph Convolutional Network layer implementing spectral graph convolution.

```python
from gnnruntime.layers import GCNConv

# Create GCN layer
gcn = GCNConv(
    in_channels=128,
    out_channels=64,
    add_self_loops=True,
    normalize=True,
    bias=True,
    aggr='add'
)

# Forward pass
out = gcn(x, edge_index)

# With edge weights
out = gcn(x, edge_index, edge_weight=edge_weight)
```

#### Parameters

- `in_channels` (int): Input feature dimensionality
- `out_channels` (int): Output feature dimensionality
- `add_self_loops` (bool): Add self-loops to graph. Default: True
- `normalize` (bool): Apply symmetric normalization. Default: True
- `bias` (bool): Add learnable bias. Default: True
- `aggr` (str): Aggregation method ['add', 'mean', 'max']. Default: 'add'

#### Methods

**`forward(x, edge_index, edge_weight=None)`**
- Perform graph convolution
- Parameters:
  - `x` (Tensor): Node features [num_nodes, in_channels]
  - `edge_index` (Tensor): Edge indices [2, num_edges]
  - `edge_weight` (Tensor, optional): Edge weights [num_edges]
- Returns: Node embeddings [num_nodes, out_channels]

**`reset_parameters()`**
- Reset layer parameters to initial values

### GATConv

Graph Attention Network layer with multi-head attention mechanism.

```python
from gnnruntime.layers import GATConv

# Create GAT layer
gat = GATConv(
    in_channels=128,
    out_channels=64,
    heads=8,
    concat=True,
    negative_slope=0.2,
    dropout=0.6,
    add_self_loops=True,
    edge_dim=None,
    bias=True
)

# Forward pass
out = gat(x, edge_index)

# Get attention weights
out, attention_weights = gat(x, edge_index, return_attention_weights=True)
```

#### Parameters

- `in_channels` (int): Input feature dimensionality
- `out_channels` (int): Output feature dimensionality per head
- `heads` (int): Number of attention heads. Default: 1
- `concat` (bool): Concatenate heads (True) or average (False). Default: True
- `negative_slope` (float): LeakyReLU angle. Default: 0.2
- `dropout` (float): Dropout probability. Default: 0.0
- `add_self_loops` (bool): Add self-loops. Default: True
- `edge_dim` (int, optional): Edge feature dimensionality
- `bias` (bool): Add learnable bias. Default: True

#### Methods

**`forward(x, edge_index, edge_attr=None, return_attention_weights=False)`**
- Apply graph attention
- Parameters:
  - `x` (Tensor): Node features [num_nodes, in_channels]
  - `edge_index` (Tensor): Edge indices [2, num_edges]
  - `edge_attr` (Tensor, optional): Edge features [num_edges, edge_dim]
  - `return_attention_weights` (bool): Return attention weights
- Returns:
  - Node embeddings [num_nodes, heads * out_channels] if concat
  - (embeddings, attention_weights) if return_attention_weights=True

### GraphSAGEConv

GraphSAGE convolution for inductive learning.

```python
from gnnruntime.layers import GraphSAGEConv

# Create GraphSAGE layer
sage = GraphSAGEConv(
    in_channels=128,
    out_channels=64,
    aggr='mean',
    normalize_emb=True,
    root_weight=True,
    bias=True
)

# Forward pass
out = sage(x, edge_index)

# With sampling
sampled_nodes, sampled_edges = sample_neighbors(node_idx, num_neighbors=25)
out = sage(x[sampled_nodes], sampled_edges)
```

#### Parameters

- `in_channels` (int): Input feature dimensionality
- `out_channels` (int): Output feature dimensionality
- `aggr` (str): Aggregation method ['mean', 'max', 'lstm', 'pool']. Default: 'mean'
- `normalize_emb` (bool): L2 normalize embeddings. Default: False
- `root_weight` (bool): Add root node transformation. Default: True
- `bias` (bool): Add learnable bias. Default: True

#### Methods

**`forward(x, edge_index)`**
- Apply GraphSAGE convolution
- Parameters:
  - `x` (Tensor): Node features [num_nodes, in_channels]
  - `edge_index` (Tensor): Edge indices [2, num_edges]
- Returns: Node embeddings [num_nodes, out_channels]

### EdgeConv

Dynamic edge convolution layer.

```python
from gnnruntime.layers import EdgeConv

# Define edge MLP
class EdgeMLP:
    def __call__(self, edge_features):
        # edge_features: [num_edges, 2 * in_channels]
        return mlp(edge_features)  # [num_edges, out_channels]

edge_mlp = EdgeMLP()

# Create EdgeConv layer
edge_conv = EdgeConv(
    nn=edge_mlp,
    aggr='max'
)

# Forward pass
out = edge_conv(x, edge_index)

# Dynamic graph construction
from gnnruntime.layers import DynamicEdgeConv

dynamic_conv = DynamicEdgeConv(
    in_channels=128,
    out_channels=64,
    k=20  # k-nearest neighbors
)

out = dynamic_conv(x)  # Automatically constructs k-NN graph
```

#### Parameters

- `nn` (callable): Neural network for edge features
- `aggr` (str): Aggregation method. Default: 'max'

#### Dynamic EdgeConv Parameters

- `in_channels` (int): Input dimensionality
- `out_channels` (int): Output dimensionality
- `k` (int): Number of nearest neighbors

### ChebConv

Chebyshev spectral graph convolution.

```python
from gnnruntime.layers import ChebConv

# Create Chebyshev convolution
cheb = ChebConv(
    in_channels=128,
    out_channels=64,
    K=3,  # Chebyshev polynomial order
    normalization='sym',
    bias=True
)

# Forward pass
out = cheb(x, edge_index, lambda_max=2.0)

# Auto-compute lambda_max
out = cheb(x, edge_index, lambda_max=None)
```

#### Parameters

- `in_channels` (int): Input dimensionality
- `out_channels` (int): Output dimensionality
- `K` (int): Chebyshev filter size. Default: 1
- `normalization` (str): Normalization type ['sym', 'rw', None]
- `bias` (bool): Add learnable bias. Default: True

### SGConv

Simplified Graph Convolution (SGC).

```python
from gnnruntime.layers import SGConv

# Create SGC layer
sgc = SGConv(
    in_channels=128,
    out_channels=64,
    K=2,  # Number of hops
    cached=True,
    add_self_loops=True,
    bias=True
)

# Forward pass
out = sgc(x, edge_index)
```

#### Parameters

- `in_channels` (int): Input dimensionality
- `out_channels` (int): Output dimensionality
- `K` (int): Number of propagation hops. Default: 1
- `cached` (bool): Cache normalized adjacency. Default: False
- `add_self_loops` (bool): Add self-loops. Default: True
- `bias` (bool): Add learnable bias. Default: True

## Graph Data API

### Graph

Basic graph data structure.

```python
from gnnruntime.core import Graph

# Create graph
graph = Graph(
    x=node_features,           # [num_nodes, num_features]
    edge_index=edge_index,     # [2, num_edges]
    edge_attr=edge_features,   # [num_edges, edge_features]
    y=labels,                  # [num_nodes] or [1] for graph
    pos=positions,             # [num_nodes, dims]
    num_nodes=100
)

# Access properties
print(f"Nodes: {graph.num_nodes}")
print(f"Edges: {graph.num_edges}")
print(f"Features: {graph.num_features}")
print(f"Directed: {not graph.is_undirected()}")

# Graph operations
graph = graph.add_self_loops()
graph = graph.remove_self_loops()
graph = graph.to_undirected()

# Subgraph extraction
node_idx = [0, 1, 2, 5, 10]
subgraph = graph.subgraph(node_idx)

# Node degrees
degrees = graph.degree()
in_degrees = graph.in_degree()
out_degrees = graph.out_degree()
```

#### Properties

- `num_nodes`: Number of nodes
- `num_edges`: Number of edges
- `num_features`: Node feature dimensionality
- `has_self_loops`: Whether graph has self-loops
- `is_directed`: Whether graph is directed

#### Methods

**`subgraph(node_idx)`**
- Extract induced subgraph
- Parameters:
  - `node_idx`: Indices of nodes to include
- Returns: New Graph object

**`add_self_loops()`**
- Add self-loop edges
- Returns: New Graph with self-loops

**`to_undirected()`**
- Convert to undirected graph
- Returns: Undirected Graph

### EdgeIndex

Edge connectivity representation.

```python
from gnnruntime.core import EdgeIndex

# Create edge index
edge_index = EdgeIndex([[0, 1, 2], [1, 2, 0]])

# Properties
num_edges = edge_index.num_edges
num_nodes = edge_index.num_nodes

# Operations
transposed = edge_index.t()
coalesced = edge_index.coalesce()  # Remove duplicates

# Convert to dense adjacency
adj_matrix = edge_index.to_dense_adj(num_nodes=10)

# Concatenate edge indices
combined = edge_index.cat(other_edge_index)
```

### HeteroGraph

Heterogeneous graph with multiple node and edge types.

```python
from gnnruntime.core import HeteroGraph

# Create heterogeneous graph
hetero_graph = HeteroGraph(
    x_dict={
        'user': user_features,     # [num_users, user_dim]
        'item': item_features,     # [num_items, item_dim]
        'category': cat_features   # [num_cats, cat_dim]
    },
    edge_index_dict={
        ('user', 'rates', 'item'): rating_edges,
        ('item', 'belongs', 'category'): category_edges,
        ('user', 'follows', 'user'): follow_edges
    },
    num_nodes_dict={
        'user': 1000,
        'item': 5000,
        'category': 50
    }
)

# Access by type
user_features = hetero_graph['user'].x
rating_edges = hetero_graph['user', 'rates', 'item'].edge_index

# Convert to homogeneous
homo_graph = hetero_graph.to_homogeneous()

# Get subgraph for specific relation
subgraph = hetero_graph.edge_type_subgraph(('user', 'rates', 'item'))
```

### TemporalGraph

Graph with temporal edges.

```python
from gnnruntime.core import TemporalGraph

# Create temporal graph
temporal_graph = TemporalGraph(
    edge_index=edge_index,     # [2, num_edges]
    timestamps=timestamps,     # [num_edges]
    x=node_features,          # [num_nodes, features]
    num_nodes=100
)

# Get snapshot at time t
snapshot = temporal_graph.snapshot(t=5)

# Get temporal subgraph
subgraph = temporal_graph.temporal_subgraph(
    start_time=0,
    end_time=10
)

# Aggregate to static graph
static_graph = temporal_graph.to_static()

# Temporal statistics
time_range = temporal_graph.time_range()
edge_frequency = temporal_graph.edge_frequency()
```

## Sampling API

### NeighborSampler

Sample neighbors for mini-batch training.

```python
from gnnruntime.sampler import NeighborSampler

# Create sampler
sampler = NeighborSampler(
    edge_index=graph.edge_index,
    num_neighbors=[25, 10],  # Sample 25 neighbors in layer 1, 10 in layer 2
    batch_size=1024,
    num_nodes=graph.num_nodes,
    replace=False,
    directed=True
)

# Sample batch
target_nodes = [0, 10, 20, 30]
batch = sampler.sample(target_nodes)

# Iterate through training nodes
train_loader = sampler.loader(
    node_idx=train_nodes,
    shuffle=True,
    num_workers=4
)

for batch in train_loader:
    # batch contains sampled subgraph
    out = model(batch.x, batch.edge_index)
```

#### Parameters

- `edge_index`: Graph connectivity
- `num_neighbors`: List of neighbors per layer
- `batch_size`: Batch size
- `num_nodes`: Total number of nodes
- `replace`: Sample with replacement
- `directed`: Use directed edges only

### GraphSAINTSampler

GraphSAINT sampling methods.

```python
from gnnruntime.sampler import GraphSAINTSampler

# Node sampling
node_sampler = GraphSAINTSampler(
    edge_index=graph.edge_index,
    num_nodes=graph.num_nodes,
    sample_coverage=3,  # Each node sampled ~3 times
    method='node'
)

# Edge sampling
edge_sampler = GraphSAINTSampler(
    edge_index=graph.edge_index,
    num_nodes=graph.num_nodes,
    sample_coverage=2,
    method='edge'
)

# Random walk sampling
rw_sampler = GraphSAINTSampler(
    edge_index=graph.edge_index,
    num_nodes=graph.num_nodes,
    method='rw',
    walk_length=20
)

# Sample subgraphs
subgraphs = sampler.sample_subgraphs(
    num_samples=10,
    size=500  # Subgraph size
)

for subgraph in subgraphs:
    # Get normalization coefficients
    norm = sampler.compute_norm_coefficients(subgraph)

    # Apply to loss
    loss = criterion(output, labels)
    loss = (loss * norm['node_norm']).mean()
```

### ClusterGCNSampler

Cluster-based sampling.

```python
from gnnruntime.sampler import ClusterGCNSampler

# Create sampler
sampler = ClusterGCNSampler(
    edge_index=graph.edge_index,
    num_nodes=graph.num_nodes,
    num_parts=10,  # Number of clusters
    batch_size=3    # Clusters per batch
)

# Perform clustering
clusters = sampler.cluster_graph()

# Sample cluster batches
for _ in range(num_epochs):
    for batch in sampler:
        # batch contains cluster subgraph
        out = model(batch.x, batch.edge_index)
```

### RandomWalkSampler

Random walk based sampling.

```python
from gnnruntime.sampler import RandomWalkSampler

# Basic random walk
sampler = RandomWalkSampler(
    edge_index=graph.edge_index,
    walk_length=20,
    num_nodes=graph.num_nodes
)

# Node2Vec biased walks
node2vec_sampler = RandomWalkSampler(
    edge_index=graph.edge_index,
    walk_length=80,
    p=1.0,  # Return parameter
    q=1.0,  # In-out parameter
    num_nodes=graph.num_nodes
)

# Sample walks
start_nodes = [0, 10, 20]
walks = sampler.sample(start_nodes)

# Sample with restart
walk = sampler.sample_with_restart(
    start_node=0,
    restart_prob=0.15
)
```

## Batch Processing API

### Batch

Batch multiple graphs together.

```python
from gnnruntime.data import Batch

# Create batch from list of graphs
graphs = [graph1, graph2, graph3]
batch = Batch.from_graph_list(graphs)

# Access batched data
x = batch.x              # Concatenated features
edge_index = batch.edge_index  # Shifted edge indices
batch_vector = batch.batch      # Graph assignment
ptr = batch.ptr          # Graph boundaries

# Convert back to list
graphs_recovered = batch.to_graph_list()

# Access individual graph
graph_0 = batch[0]

# Batch properties
num_graphs = batch.num_graphs
total_nodes = batch.num_nodes
total_edges = batch.num_edges
```

### DataLoader

Load and batch graph data.

```python
from gnnruntime.data import DataLoader

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    collate_fn=None
)

# Iterate through batches
for batch in loader:
    # Process batch
    out = model(batch.x, batch.edge_index, batch.batch)
    loss = criterion(out, batch.y)

# Custom collate function
def custom_collate(graph_list):
    batch = Batch.from_graph_list(graph_list)
    # Custom processing
    return batch

loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=custom_collate
)
```

## Utilities

### Graph Utilities

```python
from gnnruntime.utils import (
    k_hop_subgraph,
    induced_subgraph,
    sample_negative_edges,
    degree,
    contains_self_loops,
    remove_isolated_nodes,
    to_networkx,
    from_networkx
)

# K-hop subgraph
nodes, edges = k_hop_subgraph(
    node_idx=0,
    num_hops=2,
    edge_index=edge_index,
    num_nodes=100
)

# Induced subgraph
node_subset = [0, 1, 2, 5, 10]
sub_edges = induced_subgraph(node_subset, edge_index)

# Sample negative edges
neg_edges = sample_negative_edges(
    edge_index,
    num_nodes=100,
    num_neg_samples=1000
)

# Compute degrees
node_degrees = degree(edge_index[1], num_nodes=100)

# NetworkX conversion
nx_graph = to_networkx(graph)
graph = from_networkx(nx_graph)
```

### Metrics

```python
from gnnruntime.metrics import (
    accuracy,
    f1_score,
    auc_score,
    average_precision,
    homophily,
    clustering_coefficient
)

# Classification metrics
acc = accuracy(predictions, labels)
f1 = f1_score(predictions, labels, num_classes=4)

# Link prediction metrics
auc = auc_score(scores, labels)
ap = average_precision(scores, labels)

# Graph metrics
h = homophily(edge_index, labels, num_nodes)
cc = clustering_coefficient(edge_index, num_nodes)
```

### Transforms

```python
from gnnruntime.transforms import (
    AddSelfLoops,
    RemoveSelfLoops,
    ToUndirected,
    NormalizeFeatures,
    AddPositionalEncoding,
    AddRandomWalkPE,
    DropEdges,
    DropNodes,
    Compose
)

# Single transform
transform = NormalizeFeatures()
graph = transform(graph)

# Compose transforms
transform = Compose([
    RemoveSelfLoops(),
    ToUndirected(),
    NormalizeFeatures(),
    AddPositionalEncoding(dim=16)
])

graph = transform(graph)

# Augmentation
augment = Compose([
    DropEdges(p=0.1),
    DropNodes(p=0.1)
])

aug_graph = augment(graph)
```

## Examples

### Node Classification

```python
import numpy as np
from gnnruntime.layers import GCNConv
from gnnruntime.data import DataLoader

# Load data
dataset = load_dataset("Cora")
train_loader = DataLoader(dataset.train, batch_size=32)

# Build model
class GCN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = np.maximum(x, 0)  # ReLU
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_features, 64, dataset.num_classes)

# Training loop
for epoch in range(200):
    for batch in train_loader:
        out = model.forward(batch.x, batch.edge_index)
        loss = cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        # Backward and optimize
```

### Graph Classification

```python
from gnnruntime.layers import GraphSAGEConv
from gnnruntime.nn import global_mean_pool

class GraphClassifier:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.conv1 = GraphSAGEConv(input_dim, hidden_dim)
        self.conv2 = GraphSAGEConv(hidden_dim, hidden_dim)
        self.classifier = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        x = self.conv1(x, edge_index)
        x = np.maximum(x, 0)
        x = self.conv2(x, edge_index)

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        # Classification
        return self.classifier(x)

model = GraphClassifier(dataset.num_features, 128, dataset.num_classes)
```

### Link Prediction

```python
from gnnruntime.layers import GATConv

class LinkPredictor:
    def __init__(self, input_dim, hidden_dim):
        self.conv1 = GATConv(input_dim, hidden_dim, heads=8)
        self.conv2 = GATConv(hidden_dim * 8, hidden_dim, heads=1)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = np.maximum(x, 0)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        # Dot product decoder
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, pos_edge, neg_edge):
        z = self.encode(x, edge_index)
        pos_score = self.decode(z, pos_edge)
        neg_score = self.decode(z, neg_edge)
        return pos_score, neg_score

model = LinkPredictor(dataset.num_features, 64)
```

### Heterogeneous GNN

```python
from gnnruntime.nn import HeteroConv

class HeteroGNN:
    def __init__(self, metadata, hidden_dim):
        self.convs = []
        for _ in range(2):
            conv_dict = {}
            for edge_type in metadata[1]:
                src, _, dst = edge_type
                conv_dict[edge_type] = GraphSAGEConv(
                    -1, hidden_dim
                )
            self.convs.append(HeteroConv(conv_dict))

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: np.maximum(x, 0) for key, x in x_dict.items()}
        return x_dict

model = HeteroGNN(hetero_graph.metadata(), hidden_dim=64)
```

## Error Handling

All API methods may raise the following exceptions:

- `ValueError`: Invalid parameters or graph structure
- `RuntimeError`: Runtime errors during execution
- `IndexError`: Invalid node/edge indices
- `TypeError`: Type mismatch in inputs
- `MemoryError`: Insufficient memory

Example error handling:

```python
try:
    out = layer(x, edge_index)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Execution error: {e}")
except MemoryError:
    # Try with smaller batch
    out = layer(x[:half], edge_index_subset)
```