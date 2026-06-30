"""
Data Lakehouse - Production-grade lakehouse with Delta Lake

This package provides:
- Medallion architecture (Bronze -> Silver -> Gold)
- ACID transactions with Delta Lake
- Time travel and schema evolution
- Data quality validation
"""

import importlib.util

# Check if pyspark is available
PYSPARK_AVAILABLE = importlib.util.find_spec("pyspark") is not None

# Core modules that don't require pyspark
from lakehouse.config import LakehouseConfig, Layer, DeltaTableConfig
from lakehouse.delta_log import (
    DeltaLog,
    Action,
    AddFile,
    RemoveFile,
    Metadata,
    Protocol,
    CommitInfo,
    SetTransaction,
    TableState,
    Snapshot,
)
from lakehouse.optimizer import (
    TableOptimizer,
    PartitionOptimizer,
    CompactionScheduler,
    StorageReport,
    OptimizationMetrics,
)
from lakehouse.enterprise import (
    WorkflowOrchestrator,
    LakehouseMonitor,
    CostOptimizer,
    TaskConfig,
    DAGConfig,
    TableMetrics,
    Alert,
    CostReport,
)
from lakehouse.lineage import (
    LineageTracker,
    LineageGraph,
    LineageType,
    ColumnLineage,
    TableNode,
    LineageEdge,
    create_medallion_lineage,
)

# Modules that require pyspark - import conditionally
if PYSPARK_AVAILABLE:
    from lakehouse.processor import LakehouseProcessor
    from lakehouse.quality import QualityEngine
    from lakehouse.streaming import StreamingProcessor, ChangeDataFeedProcessor
else:
    # Provide placeholder classes that raise helpful errors
    LakehouseProcessor = None
    QualityEngine = None
    StreamingProcessor = None
    ChangeDataFeedProcessor = None

__version__ = "0.1.0"

__all__ = [
    "LakehouseConfig",
    "Layer",
    "DeltaTableConfig",
    "LakehouseProcessor",
    "DeltaLog",
    "Action",
    "AddFile",
    "RemoveFile",
    "Metadata",
    "Protocol",
    "CommitInfo",
    "SetTransaction",
    "TableState",
    "Snapshot",
    "QualityEngine",
    "StreamingProcessor",
    "ChangeDataFeedProcessor",
    "TableOptimizer",
    "PartitionOptimizer",
    "CompactionScheduler",
    "StorageReport",
    "OptimizationMetrics",
    "WorkflowOrchestrator",
    "LakehouseMonitor",
    "CostOptimizer",
    "TaskConfig",
    "DAGConfig",
    "TableMetrics",
    "Alert",
    "CostReport",
    "LineageTracker",
    "LineageGraph",
    "LineageType",
    "ColumnLineage",
    "TableNode",
    "LineageEdge",
    "create_medallion_lineage",
]
