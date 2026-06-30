"""Core streaming data structures."""

import asyncio
import time
import logging
from typing import Any, Callable, List, Iterator, Optional, Dict, TypeVar, Generic
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
R = TypeVar('R')


@dataclass
class Event(Generic[T]):
    """A timestamped event in the stream."""
    value: T
    timestamp: float = field(default_factory=time.time)
    key: Optional[Any] = None
    watermark: Optional[float] = None


class DataStream(Generic[T]):
    """
    A stream of events for processing.

    Supports:
    - Map, filter, flatMap transformations
    - KeyBy for partitioning
    - Window operations
    - Reduce/aggregate operations
    """

    def __init__(self, env: 'StreamExecutionEnvironment', name: str = "stream"):
        self.env = env
        self.name = name
        self._operators: List[Callable] = []
        self._source: Optional[Callable] = None
        self._sink: Optional[Callable] = None
        self._parallelism = 1

    def set_parallelism(self, parallelism: int) -> 'DataStream[T]':
        """Set parallelism for this stream."""
        self._parallelism = parallelism
        return self

    def map(self, func: Callable[[T], R]) -> 'DataStream[R]':
        """Apply function to each element."""
        new_stream = DataStream[R](self.env, f"{self.name}_map")
        new_stream._source = self._source
        new_stream._operators = self._operators + [
            lambda event: Event(func(event.value), event.timestamp, event.key)
        ]
        return new_stream

    def filter(self, predicate: Callable[[T], bool]) -> 'DataStream[T]':
        """Filter elements by predicate."""
        new_stream = DataStream[T](self.env, f"{self.name}_filter")
        new_stream._source = self._source
        new_stream._operators = self._operators + [
            lambda event: event if predicate(event.value) else None
        ]
        return new_stream

    def flat_map(self, func: Callable[[T], Iterator[R]]) -> 'DataStream[R]':
        """Apply function returning iterator to each element."""
        new_stream = DataStream[R](self.env, f"{self.name}_flatmap")
        new_stream._source = self._source

        def flat_map_op(event):
            results = []
            for value in func(event.value):
                results.append(Event(value, event.timestamp, event.key))
            return results

        new_stream._operators = self._operators + [flat_map_op]
        return new_stream

    def key_by(self, key_selector: Callable[[T], K]) -> 'KeyedStream[T, K]':
        """Partition stream by key."""
        keyed = KeyedStream[T, K](self.env, f"{self.name}_keyed", key_selector)
        keyed._source = self._source
        keyed._operators = self._operators.copy()
        return keyed

    def union(self, *streams: 'DataStream[T]') -> 'DataStream[T]':
        """Union multiple streams."""
        new_stream = DataStream[T](self.env, f"{self.name}_union")

        async def union_source():
            tasks = []
            if self._source:
                tasks.append(self._source())
            for s in streams:
                if s._source:
                    tasks.append(s._source())

            for coro in asyncio.as_completed(tasks):
                async for event in await coro:
                    yield event

        new_stream._source = union_source
        return new_stream

    def sink(self, sink_func: Callable[[T], None]) -> 'DataStream[T]':
        """Add a sink for output."""
        self._sink = sink_func
        return self

    async def execute(self) -> List[T]:
        """Execute the stream pipeline."""
        results = []

        if self._source is None:
            return results

        async for event in self._source():
            current = event

            # Apply operators
            for op in self._operators:
                if current is None:
                    break
                result = op(current)
                if result is None:
                    current = None
                elif isinstance(result, list):
                    # Handle flat_map
                    for r in result:
                        if self._sink:
                            self._sink(r.value)
                        results.append(r.value)
                    current = None
                else:
                    current = result

            if current is not None:
                if self._sink:
                    self._sink(current.value)
                results.append(current.value)

        return results


class KeyedStream(DataStream[T], Generic[T, K]):
    """A stream partitioned by key."""

    def __init__(self, env: 'StreamExecutionEnvironment', name: str, key_selector: Callable[[T], K]):
        super().__init__(env, name)
        self.key_selector = key_selector
        self._state_backend = None

    def reduce(self, reduce_func: Callable[[T, T], T]) -> DataStream[T]:
        """Reduce keyed stream."""
        new_stream = DataStream[T](self.env, f"{self.name}_reduce")
        new_stream._source = self._source

        state: Dict[K, T] = {}

        def reduce_op(event):
            key = self.key_selector(event.value)
            if key in state:
                state[key] = reduce_func(state[key], event.value)
            else:
                state[key] = event.value
            return Event(state[key], event.timestamp, key)

        new_stream._operators = self._operators + [reduce_op]
        return new_stream

    def sum(self, field: str = None) -> DataStream[T]:
        """Sum by key."""
        if field:
            def reduce_func(a, b):
                result = dict(a) if isinstance(a, dict) else a
                if isinstance(a, dict):
                    result[field] = a.get(field, 0) + b.get(field, 0)
                return result
        else:
            def reduce_func(a, b):
                return a + b

        return self.reduce(reduce_func)

    def count(self) -> DataStream[int]:
        """Count by key."""
        counts: Dict[K, int] = {}

        def count_op(event):
            key = self.key_selector(event.value)
            counts[key] = counts.get(key, 0) + 1
            return Event(counts[key], event.timestamp, key)

        new_stream = DataStream[int](self.env, f"{self.name}_count")
        new_stream._source = self._source
        new_stream._operators = self._operators + [count_op]
        return new_stream

    def window(self, assigner: 'WindowAssigner') -> 'WindowedStream[T, K]':
        """Apply windowing to keyed stream."""
        from ..windowing import WindowedStream
        return WindowedStream(self, assigner)

    def process(self, process_func: Callable[[K, Iterator[T]], Iterator[R]]) -> DataStream[R]:
        """Process keyed elements."""
        new_stream = DataStream[R](self.env, f"{self.name}_process")
        new_stream._source = self._source

        groups: Dict[K, List[T]] = defaultdict(list)

        def process_op(event):
            key = self.key_selector(event.value)
            groups[key].append(event.value)

            results = []
            for value in process_func(key, iter(groups[key])):
                results.append(Event(value, event.timestamp, key))
            return results

        new_stream._operators = self._operators + [process_op]
        return new_stream


class StreamExecutionEnvironment:
    """
    Execution environment for stream processing.

    Manages:
    - Stream creation
    - Configuration
    - Execution
    """

    def __init__(self, parallelism: int = 1):
        self.parallelism = parallelism
        self._streams: List[DataStream] = []
        self._checkpointing_enabled = False
        self._checkpoint_interval = 60000  # ms
        self._state_backend = None

    @staticmethod
    def get_execution_environment(parallelism: int = 1) -> 'StreamExecutionEnvironment':
        """Get or create execution environment."""
        return StreamExecutionEnvironment(parallelism)

    def from_collection(self, data: List[T]) -> DataStream[T]:
        """Create stream from collection."""
        stream = DataStream[T](self, "collection_source")

        async def source():
            for item in data:
                yield Event(item)

        stream._source = source
        self._streams.append(stream)
        return stream

    def from_elements(self, *elements: T) -> DataStream[T]:
        """Create stream from elements."""
        return self.from_collection(list(elements))

    def add_source(self, source_func: Callable[[], Iterator[T]]) -> DataStream[T]:
        """Add custom source."""
        stream = DataStream[T](self, "custom_source")

        async def async_source():
            for item in source_func():
                yield Event(item)

        stream._source = async_source
        self._streams.append(stream)
        return stream

    def enable_checkpointing(self, interval: int):
        """Enable checkpointing with interval in ms."""
        self._checkpointing_enabled = True
        self._checkpoint_interval = interval
        return self

    def set_state_backend(self, backend):
        """Set state backend for fault tolerance."""
        self._state_backend = backend
        return self

    async def execute(self, job_name: str = "stream_job") -> List[Any]:
        """Execute all streams."""
        logger.info(f"Executing job: {job_name}")

        all_results = []
        for stream in self._streams:
            results = await stream.execute()
            all_results.extend(results)

        return all_results

    def execute_sync(self, job_name: str = "stream_job") -> List[Any]:
        """Execute synchronously."""
        return asyncio.run(self.execute(job_name))


# Import here to avoid circular imports
from ..windowing import WindowAssigner
