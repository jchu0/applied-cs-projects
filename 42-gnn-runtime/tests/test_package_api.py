"""Tests that the documented public API is exported from the package root.

These names are advertised in the README's Features section and must be
importable as ``from gnn_runtime import <name>`` regardless of whether PyTorch
is installed. The NumPy-only classes in particular are the framework-free entry
points, so they must resolve without torch.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import gnn_runtime

# The public API surface the README documents.
DOCUMENTED_PUBLIC_API = [
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


def test_documented_api_in_all():
    for name in DOCUMENTED_PUBLIC_API:
        assert name in gnn_runtime.__all__, f"{name} missing from __all__"


def test_documented_api_importable_from_root():
    for name in DOCUMENTED_PUBLIC_API:
        assert hasattr(gnn_runtime, name), f"{name} not exported from package root"


def test_all_entries_resolve():
    # Every name in __all__ must actually be an attribute of the package.
    for name in gnn_runtime.__all__:
        assert hasattr(gnn_runtime, name), f"__all__ lists {name} but it is not present"


def test_numpy_only_classes_are_torch_free():
    # SparseOpsNumpy and MessagePassingNumpy are the framework-free entry
    # points; they must be plain classes, not torch-guarded aliases.
    from gnn_runtime import MessagePassingNumpy, SparseOpsNumpy

    assert MessagePassingNumpy is not None
    assert SparseOpsNumpy is not None
    assert hasattr(MessagePassingNumpy, "propagate_sum")
    assert hasattr(MessagePassingNumpy, "propagate_mean")


def test_version_exposed():
    assert isinstance(gnn_runtime.__version__, str)
    assert gnn_runtime.__version__
