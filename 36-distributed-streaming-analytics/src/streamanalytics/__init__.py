"""StreamAnalytics - Distributed real-time streaming analytics engine."""

from .core import (
    Event,
    DataStream,
    StreamExecutionEnvironment,
    KeyedStream,
)
from .operators import (
    MapOperator,
    FilterOperator,
    FlatMapOperator,
    ReduceOperator,
    AggregateOperator,
)
from .windowing import (
    Window,
    TumblingWindow,
    SlidingWindow,
    SessionWindow,
    WindowAssigner,
    WindowFunction,
)
from .state import (
    StateBackend,
    MemoryStateBackend,
    RocksDBStateBackend,
    KeyedState,
    ValueState,
    ListState,
    MapState,
    Checkpoint,
)
from .sql import (
    SQLParser,
    SQLLexer,
    LogicalPlanBuilder,
    LogicalOptimizer,
    PhysicalPlanner,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Event",
    "DataStream",
    "StreamExecutionEnvironment",
    "KeyedStream",
    # Operators
    "MapOperator",
    "FilterOperator",
    "FlatMapOperator",
    "ReduceOperator",
    "AggregateOperator",
    # Windowing
    "Window",
    "TumblingWindow",
    "SlidingWindow",
    "SessionWindow",
    "WindowAssigner",
    "WindowFunction",
    # State
    "StateBackend",
    "MemoryStateBackend",
    "RocksDBStateBackend",
    "KeyedState",
    "ValueState",
    "ListState",
    "MapState",
    "Checkpoint",
    # SQL
    "SQLParser",
    "SQLLexer",
    "LogicalPlanBuilder",
    "LogicalOptimizer",
    "PhysicalPlanner",
]
