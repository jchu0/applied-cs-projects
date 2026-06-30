"""Pytest fixtures for StreamAnalytics tests."""

import os
import sys
import time
import tempfile
import shutil
from typing import List, Dict, Any

import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from streamanalytics import (
    Event,
    DataStream,
    StreamExecutionEnvironment,
    KeyedStream,
    MapOperator,
    FilterOperator,
    FlatMapOperator,
    ReduceOperator,
    AggregateOperator,
    Window,
    TumblingWindow,
    SlidingWindow,
    SessionWindow,
    StateBackend,
    MemoryStateBackend,
    RocksDBStateBackend,
    KeyedState,
    ValueState,
    ListState,
    MapState,
    Checkpoint,
)
from streamanalytics.windowing import (
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
    GlobalWindowAssigner,
    CountTrigger,
    EventTimeTrigger,
    ProcessingTimeTrigger,
    WindowedStream,
)
from streamanalytics.state import (
    StateDescriptor,
    FileCheckpointStorage,
    CheckpointCoordinator,
    ReducingState,
    AggregatingState,
)


# --- Execution Environment Fixtures ---

@pytest.fixture
def env() -> StreamExecutionEnvironment:
    """Create a fresh StreamExecutionEnvironment."""
    return StreamExecutionEnvironment.get_execution_environment(parallelism=1)


@pytest.fixture
def parallel_env() -> StreamExecutionEnvironment:
    """Create a StreamExecutionEnvironment with parallelism > 1."""
    return StreamExecutionEnvironment.get_execution_environment(parallelism=4)


# --- Sample Data Fixtures ---

@pytest.fixture
def sample_integers() -> List[int]:
    """Sample list of integers for testing."""
    return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@pytest.fixture
def sample_strings() -> List[str]:
    """Sample list of strings for testing."""
    return ["hello", "world", "stream", "analytics", "test"]


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """Sample list of records for keyed operations."""
    return [
        {"user_id": "user1", "amount": 100, "category": "A"},
        {"user_id": "user2", "amount": 200, "category": "B"},
        {"user_id": "user1", "amount": 150, "category": "A"},
        {"user_id": "user3", "amount": 300, "category": "A"},
        {"user_id": "user2", "amount": 250, "category": "B"},
        {"user_id": "user1", "amount": 50, "category": "C"},
    ]


@pytest.fixture
def timestamped_events() -> List[Event]:
    """Sample timestamped events for window testing."""
    base_time = 1000.0  # Fixed base time for predictable tests
    return [
        Event(value={"key": "a", "value": 1}, timestamp=base_time + 0.0, key="a"),
        Event(value={"key": "a", "value": 2}, timestamp=base_time + 0.5, key="a"),
        Event(value={"key": "b", "value": 3}, timestamp=base_time + 1.0, key="b"),
        Event(value={"key": "a", "value": 4}, timestamp=base_time + 1.5, key="a"),
        Event(value={"key": "b", "value": 5}, timestamp=base_time + 2.0, key="b"),
        Event(value={"key": "a", "value": 6}, timestamp=base_time + 3.0, key="a"),
    ]


@pytest.fixture
def word_count_data() -> List[str]:
    """Sample data for word count example."""
    return [
        "hello world",
        "hello stream",
        "world of streaming",
        "hello hello world",
    ]


# --- State Backend Fixtures ---

@pytest.fixture
def memory_backend() -> MemoryStateBackend:
    """Create a MemoryStateBackend."""
    return MemoryStateBackend()


@pytest.fixture
def temp_dir():
    """Create a temporary directory that is cleaned up after test."""
    dir_path = tempfile.mkdtemp()
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def rocksdb_backend(temp_dir) -> RocksDBStateBackend:
    """Create a RocksDBStateBackend with temporary directory."""
    return RocksDBStateBackend(db_path=temp_dir)


@pytest.fixture
def checkpoint_storage(temp_dir) -> FileCheckpointStorage:
    """Create a FileCheckpointStorage with temporary directory."""
    storage_path = os.path.join(temp_dir, "checkpoints")
    return FileCheckpointStorage(storage_path)


@pytest.fixture
def checkpoint_coordinator(checkpoint_storage) -> CheckpointCoordinator:
    """Create a CheckpointCoordinator."""
    return CheckpointCoordinator(
        storage=checkpoint_storage,
        interval_ms=1000,
        min_pause_ms=100,
        timeout_ms=10000
    )


# --- Window Assigner Fixtures ---

@pytest.fixture
def tumbling_assigner() -> TumblingWindowAssigner:
    """Create a TumblingWindowAssigner with 1 second windows."""
    return TumblingWindowAssigner(size_ms=1000)


@pytest.fixture
def sliding_assigner() -> SlidingWindowAssigner:
    """Create a SlidingWindowAssigner with 2 second windows, 1 second slide."""
    return SlidingWindowAssigner(size_ms=2000, slide_ms=1000)


@pytest.fixture
def session_assigner() -> SessionWindowAssigner:
    """Create a SessionWindowAssigner with 500ms gap."""
    return SessionWindowAssigner(gap_ms=500)


@pytest.fixture
def global_assigner() -> GlobalWindowAssigner:
    """Create a GlobalWindowAssigner."""
    return GlobalWindowAssigner()


# --- Trigger Fixtures ---

@pytest.fixture
def count_trigger() -> CountTrigger:
    """Create a CountTrigger that fires every 3 elements."""
    return CountTrigger(count=3)


@pytest.fixture
def event_time_trigger() -> EventTimeTrigger:
    """Create an EventTimeTrigger."""
    return EventTimeTrigger()


@pytest.fixture
def processing_time_trigger() -> ProcessingTimeTrigger:
    """Create a ProcessingTimeTrigger."""
    return ProcessingTimeTrigger()


# --- State Descriptor Fixtures ---

@pytest.fixture
def value_state_descriptor() -> StateDescriptor:
    """Create a value state descriptor."""
    return StateDescriptor(name="test_value", default_value=0)


@pytest.fixture
def list_state_descriptor() -> StateDescriptor:
    """Create a list state descriptor."""
    return StateDescriptor(name="test_list", default_value=[])


@pytest.fixture
def map_state_descriptor() -> StateDescriptor:
    """Create a map state descriptor."""
    return StateDescriptor(name="test_map", default_value={})


# --- Operator Fixtures ---

@pytest.fixture
def double_map_operator() -> MapOperator:
    """Create a MapOperator that doubles values."""
    return MapOperator(func=lambda x: x * 2, name="double")


@pytest.fixture
def even_filter_operator() -> FilterOperator:
    """Create a FilterOperator that keeps even numbers."""
    return FilterOperator(predicate=lambda x: x % 2 == 0, name="even_filter")


@pytest.fixture
def split_flatmap_operator() -> FlatMapOperator:
    """Create a FlatMapOperator that splits strings by space."""
    return FlatMapOperator(func=lambda s: iter(s.split()), name="split")


# --- Helper Functions ---

def create_test_events(values: List[Any], base_time: float = 1000.0) -> List[Event]:
    """Create a list of events with sequential timestamps."""
    return [
        Event(value=v, timestamp=base_time + i * 0.1)
        for i, v in enumerate(values)
    ]


def run_stream_sync(stream: DataStream) -> List[Any]:
    """Helper to run a stream synchronously."""
    import asyncio
    return asyncio.run(stream.execute())


# Export helper functions
__all__ = [
    'create_test_events',
    'run_stream_sync',
]
