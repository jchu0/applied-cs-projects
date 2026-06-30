"""DistAutograd - Distributed automatic differentiation system."""

from .core import (
    DistributedTensor,
    DistributedContext,
    GradBucket,
    ProcessGroup,
    DeviceMesh,
)
from .distributed import (
    DistributedDataParallel,
    GradReducer,
    AllReduceStrategy,
    Reducer,
)
from .pipeline import (
    PipelineParallel,
    PipelineStage,
    MicroBatch,
    PipelineSchedule,
)
from .rpc import (
    RPCAutograd,
    RemoteGradient,
    DistAutogradContext,
    rpc_sync,
    rpc_async,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "DistributedTensor",
    "DistributedContext",
    "GradBucket",
    "ProcessGroup",
    "DeviceMesh",
    # Distributed
    "DistributedDataParallel",
    "GradReducer",
    "AllReduceStrategy",
    "Reducer",
    # Pipeline
    "PipelineParallel",
    "PipelineStage",
    "MicroBatch",
    "PipelineSchedule",
    # RPC
    "RPCAutograd",
    "RemoteGradient",
    "DistAutogradContext",
    "rpc_sync",
    "rpc_async",
]
