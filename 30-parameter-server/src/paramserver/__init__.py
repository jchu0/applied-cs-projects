"""Large-scale parameter server for distributed ML training.

This module provides a distributed parameter server implementation with:
- Model sharding across multiple servers
- Async/sync optimization with push/pull pipelines
- Multiple consistency models (Hogwild!, SSP, BSP)
- Gradient compression and mixed precision
- Fault-tolerant checkpointing
"""

from paramserver.schemas import (
    ShardInfo,
    GradientUpdate,
    WorkerInfo,
    ParameterMetadata,
)
from paramserver.server.parameter_server import ParameterServer
from paramserver.server.cluster import ParameterServerCluster
from paramserver.server.sharding import (
    ShardingStrategy,
    UniformSharding,
    RoundRobinSharding,
    SizeBalancedSharding,
)
from paramserver.worker.worker import (
    Worker,
    MockGradientComputer,
    DataBatchGenerator,
)
from paramserver.transport import (
    serve_parameter_server,
    RemoteParameterServer,
)
from paramserver.optimizer.base import UpdateEngine
from paramserver.optimizer.sgd import SGDEngine
from paramserver.optimizer.adam import AdamEngine
from paramserver.optimizer.lars import LARSEngine
from paramserver.optimizer.schedulers import (
    LRScheduler,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    WarmupLR,
    CosineWarmupLR,
    PolynomialLR,
    OneCycleLR,
)
from paramserver.consistency.base import ConsistencyModel
from paramserver.consistency.hogwild import HogwildConsistency
from paramserver.consistency.bsp import BSPConsistency
from paramserver.consistency.ssp import SSPConsistency
from paramserver.consistency.manager import ConsistencyManager, ConsistencyType
from paramserver.fault_tolerance.checkpoint import Checkpoint, CheckpointManager
from paramserver.fault_tolerance.replica import ReplicaManager, ReplicationStrategy
from paramserver.fault_tolerance.health import HealthMonitor, HealthStatus
from paramserver.enterprise.compression import (
    CompressionType,
    GradientCompressor,
    QuantizationCompressor,
    TopKCompressor,
    RandomKCompressor,
)
from paramserver.enterprise.mixed_precision import (
    MixedPrecisionManager,
    PrecisionMode,
)
from paramserver.enterprise.staleness import (
    StalenessController,
    AdaptiveSSP,
)
from paramserver.enterprise.metrics import (
    MetricsCollector,
    PerformanceMetrics,
)

__all__ = [
    # Schemas
    "ShardInfo",
    "GradientUpdate",
    "WorkerInfo",
    "ParameterMetadata",
    # Server
    "ParameterServer",
    "ParameterServerCluster",
    "ShardingStrategy",
    "UniformSharding",
    "RoundRobinSharding",
    "SizeBalancedSharding",
    # Worker
    "Worker",
    "MockGradientComputer",
    "DataBatchGenerator",
    # Transport
    "serve_parameter_server",
    "RemoteParameterServer",
    # Optimizer
    "UpdateEngine",
    "SGDEngine",
    "AdamEngine",
    "LARSEngine",
    # Schedulers
    "LRScheduler",
    "StepLR",
    "MultiStepLR",
    "ExponentialLR",
    "CosineAnnealingLR",
    "WarmupLR",
    "CosineWarmupLR",
    "PolynomialLR",
    "OneCycleLR",
    # Consistency
    "ConsistencyModel",
    "HogwildConsistency",
    "BSPConsistency",
    "SSPConsistency",
    "ConsistencyManager",
    "ConsistencyType",
    # Fault Tolerance
    "Checkpoint",
    "CheckpointManager",
    "ReplicaManager",
    "ReplicationStrategy",
    "HealthMonitor",
    "HealthStatus",
    # Enterprise - Compression
    "CompressionType",
    "GradientCompressor",
    "QuantizationCompressor",
    "TopKCompressor",
    "RandomKCompressor",
    # Enterprise - Mixed Precision
    "MixedPrecisionManager",
    "PrecisionMode",
    # Enterprise - Staleness
    "StalenessController",
    "AdaptiveSSP",
    # Enterprise - Metrics
    "MetricsCollector",
    "PerformanceMetrics",
]

__version__ = "0.1.0"
