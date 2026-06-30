"""Window implementations for stream processing."""

import time
import logging
from typing import Any, Callable, List, Iterator, TypeVar, Generic, Dict, Optional, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
R = TypeVar('R')
ACC = TypeVar('ACC')


@dataclass
class Window:
    """Base window class."""
    start: float
    end: float

    def max_timestamp(self) -> float:
        """Get maximum timestamp for this window."""
        return self.end - 1

    def __hash__(self):
        return hash((self.start, self.end))

    def __eq__(self, other):
        if not isinstance(other, Window):
            return False
        return self.start == other.start and self.end == other.end


@dataclass(eq=False)
class TumblingWindow(Window):
    """Non-overlapping fixed-size window."""
    pass


@dataclass(eq=False)
class SlidingWindow(Window):
    """Overlapping fixed-size window."""
    pass


@dataclass(eq=False)
class SessionWindow(Window):
    """Dynamic window based on activity gaps."""
    pass


@dataclass(eq=False)
class GlobalWindow(Window):
    """Single global window containing all elements."""

    def __init__(self):
        super().__init__(float('-inf'), float('inf'))


class WindowAssigner(ABC, Generic[T]):
    """Assigns elements to windows."""

    @abstractmethod
    def assign_windows(self, element: T, timestamp: float) -> List[Window]:
        """Assign windows for an element."""
        pass

    @abstractmethod
    def get_default_trigger(self) -> 'Trigger':
        """Get default trigger for this assigner."""
        pass

    def is_event_time(self) -> bool:
        """Whether this uses event time."""
        return True


class TumblingWindowAssigner(WindowAssigner[T]):
    """Assigns elements to tumbling windows."""

    def __init__(self, size_ms: int):
        self.size_ms = size_ms

    def assign_windows(self, element: T, timestamp: float) -> List[Window]:
        timestamp_ms = int(timestamp * 1000)
        start = timestamp_ms - (timestamp_ms % self.size_ms)
        return [TumblingWindow(start / 1000, (start + self.size_ms) / 1000)]

    def get_default_trigger(self) -> 'Trigger':
        return EventTimeTrigger()


class SlidingWindowAssigner(WindowAssigner[T]):
    """Assigns elements to sliding windows."""

    def __init__(self, size_ms: int, slide_ms: int):
        self.size_ms = size_ms
        self.slide_ms = slide_ms

    def assign_windows(self, element: T, timestamp: float) -> List[Window]:
        timestamp_ms = int(timestamp * 1000)
        windows = []

        # Find all windows this element belongs to
        last_start = timestamp_ms - (timestamp_ms % self.slide_ms)
        first_start = last_start - self.size_ms + self.slide_ms

        start = first_start
        while start <= last_start:
            if start + self.size_ms > timestamp_ms:
                windows.append(SlidingWindow(start / 1000, (start + self.size_ms) / 1000))
            start += self.slide_ms

        return windows

    def get_default_trigger(self) -> 'Trigger':
        return EventTimeTrigger()


class SessionWindowAssigner(WindowAssigner[T]):
    """Assigns elements to session windows."""

    def __init__(self, gap_ms: int):
        self.gap_ms = gap_ms

    def assign_windows(self, element: T, timestamp: float) -> List[Window]:
        # Session windows are created dynamically
        timestamp_ms = int(timestamp * 1000)
        return [SessionWindow(timestamp_ms / 1000, (timestamp_ms + self.gap_ms) / 1000)]

    def get_default_trigger(self) -> 'Trigger':
        return EventTimeTrigger()


class GlobalWindowAssigner(WindowAssigner[T]):
    """Assigns all elements to a global window."""

    def assign_windows(self, element: T, timestamp: float) -> List[Window]:
        return [GlobalWindow()]

    def get_default_trigger(self) -> 'Trigger':
        return ProcessingTimeTrigger()


class Trigger(ABC):
    """Determines when a window should fire."""

    @abstractmethod
    def on_element(self, element: Any, timestamp: float, window: Window, key: Any = None) -> bool:
        """Called for each element added to window."""
        pass

    @abstractmethod
    def on_event_time(self, time: float, window: Window) -> bool:
        """Called when event time passes."""
        pass

    @abstractmethod
    def on_processing_time(self, time: float, window: Window) -> bool:
        """Called when processing time passes."""
        pass


class EventTimeTrigger(Trigger):
    """Fires when watermark passes window end."""

    def on_element(self, element: Any, timestamp: float, window: Window, key: Any = None) -> bool:
        return False

    def on_event_time(self, time: float, window: Window) -> bool:
        return time >= window.end

    def on_processing_time(self, time: float, window: Window) -> bool:
        return False


class ProcessingTimeTrigger(Trigger):
    """Fires based on processing time."""

    def on_element(self, element: Any, timestamp: float, window: Window, key: Any = None) -> bool:
        return False

    def on_event_time(self, time: float, window: Window) -> bool:
        return False

    def on_processing_time(self, time: float, window: Window) -> bool:
        return time >= window.end


class CountTrigger(Trigger):
    """Fires after a certain count."""

    def __init__(self, count: int):
        self.count = count
        self._counts: Dict[Tuple[Any, Window], int] = defaultdict(int)

    def on_element(self, element: Any, timestamp: float, window: Window, key: Any = None) -> bool:
        # Use (key, window) as the count key to support per-key counting in keyed streams
        count_key = (key, window)
        self._counts[count_key] += 1
        if self._counts[count_key] >= self.count:
            self._counts[count_key] = 0
            return True
        return False

    def on_event_time(self, time: float, window: Window) -> bool:
        return False

    def on_processing_time(self, time: float, window: Window) -> bool:
        return False


class WindowFunction(ABC, Generic[T, R]):
    """Function applied to window contents."""

    @abstractmethod
    def apply(self, key: K, window: Window, elements: Iterator[T]) -> Iterator[R]:
        """Apply function to window elements."""
        pass


class ReduceFunction(WindowFunction[T, T], Generic[T, K]):
    """Reduce function for windows."""

    def __init__(self, reduce_func: Callable[[T, T], T]):
        self.reduce_func = reduce_func

    def apply(self, key: K, window: Window, elements: Iterator[T]) -> Iterator[T]:
        result = None
        for elem in elements:
            if result is None:
                result = elem
            else:
                result = self.reduce_func(result, elem)
        if result is not None:
            yield result


class AggregateFunction(WindowFunction[T, R], Generic[T, K, ACC, R]):
    """Aggregate function for windows."""

    def __init__(
        self,
        create_accumulator: Callable[[], ACC],
        add: Callable[[ACC, T], ACC],
        get_result: Callable[[ACC], R],
        merge: Callable[[ACC, ACC], ACC] = None
    ):
        self.create_accumulator = create_accumulator
        self.add = add
        self.get_result = get_result
        self.merge = merge

    def apply(self, key: K, window: Window, elements: Iterator[T]) -> Iterator[R]:
        acc = self.create_accumulator()
        for elem in elements:
            acc = self.add(acc, elem)
        yield self.get_result(acc)


class ProcessWindowFunction(WindowFunction[T, R], Generic[T, K, R]):
    """Process function with full window context."""

    def __init__(self, process_func: Callable[[K, Window, Iterator[T]], Iterator[R]]):
        self.process_func = process_func

    def apply(self, key: K, window: Window, elements: Iterator[T]) -> Iterator[R]:
        yield from self.process_func(key, window, elements)


class WindowedStream(Generic[T, K]):
    """A keyed stream with window assignments."""

    def __init__(self, keyed_stream: 'KeyedStream[T, K]', assigner: WindowAssigner[T]):
        self.keyed_stream = keyed_stream
        self.assigner = assigner
        self.trigger = assigner.get_default_trigger()
        self._window_contents: Dict[Tuple[K, Window], List[T]] = defaultdict(list)
        self._watermark = 0.0

    def trigger_with(self, trigger: Trigger) -> 'WindowedStream[T, K]':
        """Set custom trigger."""
        self.trigger = trigger
        return self

    def reduce(self, reduce_func: Callable[[T, T], T]) -> 'DataStream[T]':
        """Apply reduce function to window."""
        from ..core.stream import DataStream, Event

        new_stream = DataStream[T](self.keyed_stream.env, f"{self.keyed_stream.name}_window_reduce")
        new_stream._source = self.keyed_stream._source

        def window_reduce_op(event):
            key = self.keyed_stream.key_selector(event.value)
            windows = self.assigner.assign_windows(event.value, event.timestamp)

            results = []
            for window in windows:
                window_key = (key, window)
                self._window_contents[window_key].append(event.value)

                # Check if window should fire
                if self.trigger.on_element(event.value, event.timestamp, window, key=key):
                    # Apply reduce
                    elements = self._window_contents[window_key]
                    result = elements[0]
                    for elem in elements[1:]:
                        result = reduce_func(result, elem)
                    results.append(Event(result, event.timestamp, key))
                    self._window_contents[window_key] = []

            return results if results else None

        new_stream._operators = self.keyed_stream._operators + [window_reduce_op]
        return new_stream

    def aggregate(
        self,
        create_accumulator: Callable[[], ACC],
        add: Callable[[ACC, T], ACC],
        get_result: Callable[[ACC], R]
    ) -> 'DataStream[R]':
        """Apply aggregate function to window."""
        from ..core.stream import DataStream, Event

        new_stream = DataStream[R](self.keyed_stream.env, f"{self.keyed_stream.name}_window_aggregate")
        new_stream._source = self.keyed_stream._source

        accumulators: Dict[Tuple[K, Window], ACC] = {}

        def window_aggregate_op(event):
            key = self.keyed_stream.key_selector(event.value)
            windows = self.assigner.assign_windows(event.value, event.timestamp)

            results = []
            for window in windows:
                window_key = (key, window)

                if window_key not in accumulators:
                    accumulators[window_key] = create_accumulator()

                accumulators[window_key] = add(accumulators[window_key], event.value)
                self._window_contents[window_key].append(event.value)

                # Check if window should fire
                if self.trigger.on_element(event.value, event.timestamp, window, key=key):
                    result = get_result(accumulators[window_key])
                    results.append(Event(result, event.timestamp, key))
                    del accumulators[window_key]
                    self._window_contents[window_key] = []

            return results if results else None

        new_stream._operators = self.keyed_stream._operators + [window_aggregate_op]
        return new_stream

    def process(self, func: Callable[[K, Window, Iterator[T]], Iterator[R]]) -> 'DataStream[R]':
        """Apply process function to window."""
        from ..core.stream import DataStream, Event

        new_stream = DataStream[R](self.keyed_stream.env, f"{self.keyed_stream.name}_window_process")
        new_stream._source = self.keyed_stream._source

        def window_process_op(event):
            key = self.keyed_stream.key_selector(event.value)
            windows = self.assigner.assign_windows(event.value, event.timestamp)

            results = []
            for window in windows:
                window_key = (key, window)
                self._window_contents[window_key].append(event.value)

                # Check if window should fire
                if self.trigger.on_element(event.value, event.timestamp, window, key=key):
                    for result in func(key, window, iter(self._window_contents[window_key])):
                        results.append(Event(result, event.timestamp, key))
                    self._window_contents[window_key] = []

            return results if results else None

        new_stream._operators = self.keyed_stream._operators + [window_process_op]
        return new_stream

    def sum(self, field: str = None) -> 'DataStream[T]':
        """Sum elements in window."""
        if field:
            def add_func(a, b):
                result = dict(a) if isinstance(a, dict) else {'value': a}
                if isinstance(b, dict):
                    result[field] = result.get(field, 0) + b.get(field, 0)
                return result
        else:
            def add_func(a, b):
                return a + b

        return self.reduce(add_func)

    def count(self) -> 'DataStream[int]':
        """Count elements in window."""
        return self.aggregate(
            create_accumulator=lambda: 0,
            add=lambda acc, _: acc + 1,
            get_result=lambda acc: acc
        )

    def min(self, field: str = None) -> 'DataStream[T]':
        """Get minimum element in window."""
        if field:
            def min_func(a, b):
                return a if a.get(field, float('inf')) < b.get(field, float('inf')) else b
        else:
            def min_func(a, b):
                return min(a, b)

        return self.reduce(min_func)

    def max(self, field: str = None) -> 'DataStream[T]':
        """Get maximum element in window."""
        if field:
            def max_func(a, b):
                return a if a.get(field, float('-inf')) > b.get(field, float('-inf')) else b
        else:
            def max_func(a, b):
                return max(a, b)

        return self.reduce(max_func)


# Import here to avoid circular imports
from ..core.stream import KeyedStream
