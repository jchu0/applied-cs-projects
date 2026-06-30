"""Stream operators for transformations."""

import logging
from typing import Any, Callable, List, Iterator, TypeVar, Generic, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')
K = TypeVar('K')
ACC = TypeVar('ACC')


class Operator(ABC, Generic[T, R]):
    """Base class for stream operators."""

    def __init__(self, name: str = "operator"):
        self.name = name
        self._parallelism = 1

    @abstractmethod
    def process(self, element: T) -> Iterator[R]:
        """Process an element and yield results."""
        pass

    def open(self):
        """Called when operator is initialized."""
        pass

    def close(self):
        """Called when operator is closed."""
        pass

    def set_parallelism(self, parallelism: int) -> 'Operator':
        """Set parallelism for this operator."""
        self._parallelism = parallelism
        return self


class MapOperator(Operator[T, R]):
    """Map operator applies function to each element."""

    def __init__(self, func: Callable[[T], R], name: str = "map"):
        super().__init__(name)
        self.func = func

    def process(self, element: T) -> Iterator[R]:
        yield self.func(element)


class FilterOperator(Operator[T, T]):
    """Filter operator keeps elements matching predicate."""

    def __init__(self, predicate: Callable[[T], bool], name: str = "filter"):
        super().__init__(name)
        self.predicate = predicate

    def process(self, element: T) -> Iterator[T]:
        if self.predicate(element):
            yield element


class FlatMapOperator(Operator[T, R]):
    """FlatMap operator applies function returning multiple elements."""

    def __init__(self, func: Callable[[T], Iterator[R]], name: str = "flatmap"):
        super().__init__(name)
        self.func = func

    def process(self, element: T) -> Iterator[R]:
        yield from self.func(element)


class ReduceOperator(Operator[T, T], Generic[T, K]):
    """Reduce operator for keyed streams."""

    def __init__(
        self,
        reduce_func: Callable[[T, T], T],
        key_selector: Callable[[T], K],
        name: str = "reduce"
    ):
        super().__init__(name)
        self.reduce_func = reduce_func
        self.key_selector = key_selector
        self._state: Dict[K, T] = {}

    def process(self, element: T) -> Iterator[T]:
        key = self.key_selector(element)
        if key in self._state:
            self._state[key] = self.reduce_func(self._state[key], element)
        else:
            self._state[key] = element
        yield self._state[key]

    def get_state(self) -> Dict[K, T]:
        """Get current state."""
        return self._state.copy()


@dataclass
class Accumulator(Generic[ACC, T, R]):
    """Accumulator for aggregate operations."""
    value: ACC


class AggregateOperator(Operator[T, R], Generic[T, K, ACC, R]):
    """Aggregate operator for complex aggregations."""

    def __init__(
        self,
        key_selector: Callable[[T], K],
        create_accumulator: Callable[[], ACC],
        add: Callable[[ACC, T], ACC],
        get_result: Callable[[ACC], R],
        merge: Callable[[ACC, ACC], ACC] = None,
        name: str = "aggregate"
    ):
        super().__init__(name)
        self.key_selector = key_selector
        self.create_accumulator = create_accumulator
        self.add = add
        self.get_result = get_result
        self.merge = merge
        self._accumulators: Dict[K, ACC] = {}

    def process(self, element: T) -> Iterator[R]:
        key = self.key_selector(element)

        if key not in self._accumulators:
            self._accumulators[key] = self.create_accumulator()

        self._accumulators[key] = self.add(self._accumulators[key], element)
        yield self.get_result(self._accumulators[key])

    def merge_accumulators(self, key: K, other_acc: ACC):
        """Merge accumulators for parallel execution."""
        if self.merge is None:
            raise NotImplementedError("Merge function not provided")

        if key not in self._accumulators:
            self._accumulators[key] = other_acc
        else:
            self._accumulators[key] = self.merge(self._accumulators[key], other_acc)


class KeyByOperator(Operator[T, T], Generic[T, K]):
    """Key-by operator for partitioning."""

    def __init__(self, key_selector: Callable[[T], K], name: str = "keyby"):
        super().__init__(name)
        self.key_selector = key_selector

    def process(self, element: T) -> Iterator[T]:
        # Keying is handled at the stream level
        yield element

    def get_key(self, element: T) -> K:
        """Get key for element."""
        return self.key_selector(element)


class UnionOperator(Operator[T, T]):
    """Union operator merges multiple streams."""

    def __init__(self, name: str = "union"):
        super().__init__(name)

    def process(self, element: T) -> Iterator[T]:
        yield element


class ProcessOperator(Operator[T, R], Generic[T, K, R]):
    """Process operator for complex processing with state access."""

    def __init__(
        self,
        process_func: Callable[[K, T, 'ProcessContext'], Iterator[R]],
        key_selector: Callable[[T], K],
        name: str = "process"
    ):
        super().__init__(name)
        self.process_func = process_func
        self.key_selector = key_selector
        self._context = ProcessContext()

    def process(self, element: T) -> Iterator[R]:
        key = self.key_selector(element)
        self._context.set_current_key(key)
        yield from self.process_func(key, element, self._context)


class ProcessContext:
    """Context for process functions."""

    def __init__(self):
        self._current_key = None
        self._timers: Dict[str, float] = {}
        self._state: Dict[Any, Any] = {}

    def set_current_key(self, key: Any):
        """Set current key."""
        self._current_key = key

    def get_current_key(self) -> Any:
        """Get current key."""
        return self._current_key

    def register_timer(self, name: str, timestamp: float):
        """Register a timer."""
        self._timers[name] = timestamp

    def get_state(self, name: str) -> Any:
        """Get state by name."""
        key = (self._current_key, name)
        return self._state.get(key)

    def update_state(self, name: str, value: Any):
        """Update state."""
        key = (self._current_key, name)
        self._state[key] = value


class CoMapOperator(Operator[Any, R], Generic[T, R]):
    """Co-map operator for connected streams."""

    def __init__(
        self,
        map1: Callable[[Any], R],
        map2: Callable[[Any], R],
        name: str = "comap"
    ):
        super().__init__(name)
        self.map1 = map1
        self.map2 = map2

    def process_first(self, element: Any) -> Iterator[R]:
        """Process element from first stream."""
        yield self.map1(element)

    def process_second(self, element: Any) -> Iterator[R]:
        """Process element from second stream."""
        yield self.map2(element)

    def process(self, element: Any) -> Iterator[R]:
        # Default to first stream processing
        yield from self.process_first(element)


class AsyncMapOperator(Operator[T, R]):
    """Async map operator for I/O operations."""

    def __init__(
        self,
        async_func: Callable[[T], R],
        timeout: float = 60.0,
        capacity: int = 100,
        name: str = "async_map"
    ):
        super().__init__(name)
        self.async_func = async_func
        self.timeout = timeout
        self.capacity = capacity

    def process(self, element: T) -> Iterator[R]:
        # Simplified sync version
        yield self.async_func(element)


class SinkOperator(Operator[T, None]):
    """Sink operator for output."""

    def __init__(self, sink_func: Callable[[T], None], name: str = "sink"):
        super().__init__(name)
        self.sink_func = sink_func

    def process(self, element: T) -> Iterator[None]:
        self.sink_func(element)
        return
        yield  # Make it a generator
