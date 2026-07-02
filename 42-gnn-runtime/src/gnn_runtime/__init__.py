"""GNN Runtime - High-performance Graph Neural Network execution engine."""

from .graph import (
    GraphStorage,
    GraphFormat,
    PartitionedGraph,
)
from .message_passing import (
    MessagePassingEngine,
    MessagePassingNumpy,
    MessageFunction,
    AggregateFunction,
)
from .sampling import (
    NeighborSampler,
    PPRSampler,
    SampledSubgraph,
)
from .layers import (
    GNNLayer,
    GCNLayer,
    GATLayer,
    GraphSAGELayer,
    GNNModel,
)
from .kernels import (
    SparseOps,
    SparseOpsNumpy,
    scatter_add,
    scatter_mean,
    scatter_max,
    spmm,
)
from .distributed import (
    DistributedGNNTrainer,
    HaloExchange,
    VertexReorderOptimizer,
)

__version__ = "0.1.0"
__all__ = [
    # Graph storage
    "GraphStorage",
    "GraphFormat",
    "PartitionedGraph",
    # Message passing
    "MessagePassingEngine",
    "MessagePassingNumpy",
    "MessageFunction",
    "AggregateFunction",
    # Sampling
    "NeighborSampler",
    "PPRSampler",
    "SampledSubgraph",
    # Layers
    "GNNLayer",
    "GCNLayer",
    "GATLayer",
    "GraphSAGELayer",
    "GNNModel",
    # Kernels
    "SparseOps",
    "SparseOpsNumpy",
    "scatter_add",
    "scatter_mean",
    "scatter_max",
    "spmm",
    # Distributed
    "DistributedGNNTrainer",
    "HaloExchange",
    "VertexReorderOptimizer",
]
