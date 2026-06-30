"""AI Workflow Engine - DSL-based workflow orchestration for AI pipelines."""

from .schemas import (
    NodeType,
    NodeStatus,
    RunStatus,
    NodeConfig,
    RetryConfig,
    Node,
    FlowDefinition,
    NodeExecution,
    FlowRun,
    generate_id,
)

# Optional imports requiring yaml
try:
    from .compiler import FlowParser, FlowValidator, DAGBuilder, FlowOptimizer
    from .executor import Scheduler, AsyncScheduler
    from .nodes import (
        BaseNodeExecutor,
        LLMNodeExecutor,
        RetrievalNodeExecutor,
        BranchNodeExecutor,
        TransformNodeExecutor,
        ToolNodeExecutor,
        NodeExecutorRegistry,
    )
    from .retry import (
        RetryStrategy,
        ExponentialBackoffRetry,
        LLMOutputRetry,
        ConstantRetry,
        AdaptiveRetry,
        RetryManager,
    )
    from .versioning import FlowVersion, FlowVersionManager, MigrationManager
    from .engine import WorkflowEngine, create_engine, EXAMPLE_FLOW
    _HAS_YAML = True
except ImportError:
    FlowParser = None
    FlowValidator = None
    DAGBuilder = None
    FlowOptimizer = None
    Scheduler = None
    AsyncScheduler = None
    BaseNodeExecutor = None
    LLMNodeExecutor = None
    RetrievalNodeExecutor = None
    BranchNodeExecutor = None
    TransformNodeExecutor = None
    ToolNodeExecutor = None
    NodeExecutorRegistry = None
    RetryStrategy = None
    ExponentialBackoffRetry = None
    LLMOutputRetry = None
    ConstantRetry = None
    AdaptiveRetry = None
    RetryManager = None
    FlowVersion = None
    FlowVersionManager = None
    MigrationManager = None
    WorkflowEngine = None
    create_engine = None
    EXAMPLE_FLOW = None
    _HAS_YAML = False

__version__ = "0.1.0"

__all__ = [
    # Schemas
    "NodeType",
    "NodeStatus",
    "RunStatus",
    "NodeConfig",
    "RetryConfig",
    "Node",
    "FlowDefinition",
    "NodeExecution",
    "FlowRun",
    "generate_id",
    # Compiler
    "FlowParser",
    "FlowValidator",
    "DAGBuilder",
    "FlowOptimizer",
    # Executor
    "Scheduler",
    "AsyncScheduler",
    # Nodes
    "BaseNodeExecutor",
    "LLMNodeExecutor",
    "RetrievalNodeExecutor",
    "BranchNodeExecutor",
    "TransformNodeExecutor",
    "ToolNodeExecutor",
    "NodeExecutorRegistry",
    # Retry
    "RetryStrategy",
    "ExponentialBackoffRetry",
    "LLMOutputRetry",
    "ConstantRetry",
    "AdaptiveRetry",
    "RetryManager",
    # Versioning
    "FlowVersion",
    "FlowVersionManager",
    "MigrationManager",
    # Engine
    "WorkflowEngine",
    "create_engine",
    "EXAMPLE_FLOW",
]
