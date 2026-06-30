"""Windowing and event-time processing for streams."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


@dataclass
class Window:
    """Represents a time window."""

    start: int  # milliseconds
    end: int  # milliseconds

    @property
    def duration_ms(self) -> int:
        """Window duration in milliseconds."""
        return self.end - self.start

    def contains(self, timestamp: int) -> bool:
        """Check if timestamp falls in window."""
        return self.start <= timestamp < self.end

    def __repr__(self) -> str:
        start_dt = datetime.fromtimestamp(self.start / 1000)
        end_dt = datetime.fromtimestamp(self.end / 1000)
        return f"Window({start_dt.isoformat()} - {end_dt.isoformat()})"


@dataclass
class WindowedValue(Generic[T]):
    """A value with its window context."""

    window: Window
    value: T
    timestamp: int


class WindowAssigner:
    """Base class for window assigners."""

    def assign_windows(self, timestamp: int) -> List[Window]:
        """Assign windows for a timestamp."""
        raise NotImplementedError


class TumblingWindowAssigner(WindowAssigner):
    """Assign events to non-overlapping fixed-size windows."""

    def __init__(self, size_ms: int):
        self.size_ms = size_ms

    def assign_windows(self, timestamp: int) -> List[Window]:
        """Assign to a single tumbling window."""
        window_start = (timestamp // self.size_ms) * self.size_ms
        window_end = window_start + self.size_ms
        return [Window(window_start, window_end)]


class SlidingWindowAssigner(WindowAssigner):
    """Assign events to overlapping sliding windows."""

    def __init__(self, size_ms: int, slide_ms: int):
        self.size_ms = size_ms
        self.slide_ms = slide_ms

    def assign_windows(self, timestamp: int) -> List[Window]:
        """Assign to multiple overlapping windows."""
        windows = []

        # Find the first window that contains this timestamp
        last_start = (timestamp // self.slide_ms) * self.slide_ms

        # Go back to find all windows containing this timestamp
        start = last_start - self.size_ms + self.slide_ms
        while start <= last_start:
            window = Window(start, start + self.size_ms)
            if window.contains(timestamp):
                windows.append(window)
            start += self.slide_ms

        return windows


class SessionWindowAssigner(WindowAssigner):
    """Assign events to session windows based on activity gaps."""

    def __init__(self, gap_ms: int):
        self.gap_ms = gap_ms

    def assign_windows(self, timestamp: int) -> List[Window]:
        """Assign to a session window (will be merged later)."""
        return [Window(timestamp, timestamp + self.gap_ms)]

    def merge_windows(self, windows: List[Window]) -> List[Window]:
        """Merge overlapping session windows."""
        if not windows:
            return []

        sorted_windows = sorted(windows, key=lambda w: w.start)
        merged = [sorted_windows[0]]

        for window in sorted_windows[1:]:
            last = merged[-1]
            if window.start <= last.end:
                # Merge windows
                merged[-1] = Window(last.start, max(last.end, window.end))
            else:
                merged.append(window)

        return merged


class Watermark:
    """Watermark for tracking event-time progress."""

    def __init__(self, timestamp: int):
        self.timestamp = timestamp

    def __repr__(self) -> str:
        dt = datetime.fromtimestamp(self.timestamp / 1000)
        return f"Watermark({dt.isoformat()})"


class WatermarkGenerator:
    """Generate watermarks based on event timestamps."""

    def __init__(self, max_out_of_orderness_ms: int = 0):
        self.max_out_of_orderness_ms = max_out_of_orderness_ms
        self.current_max_timestamp = 0

    def on_event(self, timestamp: int) -> None:
        """Update based on event timestamp."""
        if timestamp > self.current_max_timestamp:
            self.current_max_timestamp = timestamp

    def get_watermark(self) -> Watermark:
        """Get current watermark."""
        return Watermark(self.current_max_timestamp - self.max_out_of_orderness_ms)


class BoundedOutOfOrdernessGenerator(WatermarkGenerator):
    """Watermark generator for bounded out-of-orderness."""

    def __init__(self, max_out_of_orderness_ms: int):
        super().__init__(max_out_of_orderness_ms)


class IdleAwareWatermarkGenerator(WatermarkGenerator):
    """Watermark generator that handles idle partitions."""

    def __init__(
        self,
        max_out_of_orderness_ms: int,
        idle_timeout_ms: int,
    ):
        super().__init__(max_out_of_orderness_ms)
        self.idle_timeout_ms = idle_timeout_ms
        self.last_record_time = 0

    def on_event(self, timestamp: int) -> None:
        """Update based on event timestamp."""
        super().on_event(timestamp)
        self.last_record_time = int(datetime.now().timestamp() * 1000)

    def get_watermark(self) -> Watermark:
        """Get watermark, advancing for idle sources."""
        now = int(datetime.now().timestamp() * 1000)

        if now - self.last_record_time > self.idle_timeout_ms:
            # Source is idle - advance watermark to current time
            return Watermark(now - self.max_out_of_orderness_ms)

        return super().get_watermark()


@dataclass
class WindowState(Generic[T]):
    """State for a window."""

    window: Window
    elements: List[T] = field(default_factory=list)
    result: Optional[Any] = None


class WindowOperator(Generic[T, R]):
    """Process windowed streams."""

    def __init__(
        self,
        assigner: WindowAssigner,
        aggregator: Callable[[List[T]], R],
    ):
        self.assigner = assigner
        self.aggregator = aggregator
        self.window_states: Dict[str, Dict[int, WindowState[T]]] = {}
        self.watermark_generator = BoundedOutOfOrdernessGenerator(0)

    def process_element(
        self,
        key: str,
        value: T,
        timestamp: int,
    ) -> List[WindowedValue[R]]:
        """
        Process an element and return any fired windows.

        Args:
            key: Element key
            value: Element value
            timestamp: Event timestamp

        Returns:
            List of window results that fired
        """
        # Update watermark
        self.watermark_generator.on_event(timestamp)

        # Assign to windows
        windows = self.assigner.assign_windows(timestamp)

        # Ensure key exists
        if key not in self.window_states:
            self.window_states[key] = {}

        # Add element to windows
        for window in windows:
            window_id = window.start
            if window_id not in self.window_states[key]:
                self.window_states[key][window_id] = WindowState(window)

            self.window_states[key][window_id].elements.append(value)

        # Windows fire only on explicit advance_watermark, not on element processing
        return []

    def _fire_windows(self, key: str) -> List[WindowedValue[R]]:
        """Fire windows that are ready based on watermark."""
        results = []
        watermark = self.watermark_generator.get_watermark()

        if key not in self.window_states:
            return results

        fired_windows = []
        for window_id, state in self.window_states[key].items():
            # Window fires when watermark passes end of window
            if watermark.timestamp >= state.window.end:
                result = self.aggregator(state.elements)
                results.append(WindowedValue(
                    window=state.window,
                    value=result,
                    timestamp=state.window.end,
                ))
                fired_windows.append(window_id)

        # Clean up fired windows
        for window_id in fired_windows:
            del self.window_states[key][window_id]

        return results

    def advance_watermark(self, timestamp: int) -> List[WindowedValue[R]]:
        """
        Manually advance watermark and fire windows.

        Args:
            timestamp: New watermark timestamp

        Returns:
            All fired window results
        """
        self.watermark_generator.current_max_timestamp = timestamp + self.watermark_generator.max_out_of_orderness_ms

        results = []
        for key in list(self.window_states.keys()):
            results.extend(self._fire_windows(key))

        return results


# Aggregation functions

def count_aggregator(elements: List[Any]) -> int:
    """Count elements in window."""
    return len(elements)


def sum_aggregator(elements: List[float]) -> float:
    """Sum elements in window."""
    return sum(elements)


def avg_aggregator(elements: List[float]) -> float:
    """Average elements in window."""
    return sum(elements) / len(elements) if elements else 0


def min_aggregator(elements: List[float]) -> float:
    """Minimum element in window."""
    return min(elements) if elements else 0


def max_aggregator(elements: List[float]) -> float:
    """Maximum element in window."""
    return max(elements) if elements else 0


@dataclass
class AggregationResult:
    """Full aggregation result."""

    count: int
    sum: float
    min: float
    max: float
    avg: float


def full_aggregator(elements: List[float]) -> AggregationResult:
    """Compute all standard aggregations."""
    if not elements:
        return AggregationResult(0, 0, 0, 0, 0)

    return AggregationResult(
        count=len(elements),
        sum=sum(elements),
        min=min(elements),
        max=max(elements),
        avg=sum(elements) / len(elements),
    )
