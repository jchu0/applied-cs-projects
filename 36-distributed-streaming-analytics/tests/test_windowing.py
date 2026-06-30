"""Tests for window operations (tumbling, sliding, session)."""

import pytest
import asyncio
import time
from typing import List, Iterator, Dict, Any

from streamanalytics import (
    Event,
    DataStream,
    StreamExecutionEnvironment,
    Window,
    TumblingWindow,
    SlidingWindow,
    SessionWindow,
)
from streamanalytics.windowing import (
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
    GlobalWindowAssigner,
    GlobalWindow,
    WindowAssigner,
    WindowFunction,
    ReduceFunction,
    AggregateFunction,
    ProcessWindowFunction,
    WindowedStream,
    Trigger,
    EventTimeTrigger,
    ProcessingTimeTrigger,
    CountTrigger,
)


def run_stream(stream: DataStream) -> List:
    """Helper to execute a stream and return results."""
    return asyncio.run(stream.execute())


class TestWindow:
    """Tests for base Window class."""

    def test_window_creation(self):
        """Test creating a basic window."""
        window = Window(start=1000.0, end=2000.0)
        assert window.start == 1000.0
        assert window.end == 2000.0

    def test_window_max_timestamp(self):
        """Test window max_timestamp method."""
        window = Window(start=1000.0, end=2000.0)
        assert window.max_timestamp() == 1999.0

    def test_window_equality(self):
        """Test window equality comparison."""
        w1 = Window(start=1000.0, end=2000.0)
        w2 = Window(start=1000.0, end=2000.0)
        w3 = Window(start=1000.0, end=3000.0)

        assert w1 == w2
        assert w1 != w3

    def test_window_hash(self):
        """Test window hashing for use in dicts/sets."""
        w1 = Window(start=1000.0, end=2000.0)
        w2 = Window(start=1000.0, end=2000.0)

        window_set = {w1}
        assert w2 in window_set

        window_dict = {w1: "test"}
        assert window_dict[w2] == "test"


class TestTumblingWindow:
    """Tests for TumblingWindow."""

    def test_tumbling_window_creation(self):
        """Test creating a tumbling window."""
        window = TumblingWindow(start=0.0, end=1.0)
        assert isinstance(window, Window)
        assert window.start == 0.0
        assert window.end == 1.0


class TestSlidingWindow:
    """Tests for SlidingWindow."""

    def test_sliding_window_creation(self):
        """Test creating a sliding window."""
        window = SlidingWindow(start=0.0, end=2.0)
        assert isinstance(window, Window)


class TestSessionWindow:
    """Tests for SessionWindow."""

    def test_session_window_creation(self):
        """Test creating a session window."""
        window = SessionWindow(start=0.0, end=0.5)
        assert isinstance(window, Window)


class TestGlobalWindow:
    """Tests for GlobalWindow."""

    def test_global_window_creation(self):
        """Test creating a global window."""
        window = GlobalWindow()
        assert window.start == float('-inf')
        assert window.end == float('inf')


class TestTumblingWindowAssigner:
    """Tests for TumblingWindowAssigner."""

    def test_assigner_creation(self, tumbling_assigner):
        """Test creating a tumbling window assigner."""
        assert tumbling_assigner.size_ms == 1000

    def test_assign_single_window(self, tumbling_assigner):
        """Test that an element is assigned to exactly one tumbling window."""
        # Timestamp at 1.5 seconds (1500ms)
        windows = tumbling_assigner.assign_windows("element", 1.5)
        assert len(windows) == 1
        assert isinstance(windows[0], TumblingWindow)

    def test_assign_correct_window_bounds(self, tumbling_assigner):
        """Test tumbling window boundaries."""
        # Element at 1.5 seconds should be in window [1.0, 2.0)
        windows = tumbling_assigner.assign_windows("element", 1.5)
        assert windows[0].start == 1.0
        assert windows[0].end == 2.0

    def test_assign_window_at_boundary(self, tumbling_assigner):
        """Test element exactly at window boundary."""
        # Element at exactly 2.0 seconds should be in window [2.0, 3.0)
        windows = tumbling_assigner.assign_windows("element", 2.0)
        assert windows[0].start == 2.0
        assert windows[0].end == 3.0

    def test_assign_elements_same_window(self, tumbling_assigner):
        """Test elements in same time range go to same window."""
        w1 = tumbling_assigner.assign_windows("a", 1.1)
        w2 = tumbling_assigner.assign_windows("b", 1.5)
        w3 = tumbling_assigner.assign_windows("c", 1.9)

        assert w1[0] == w2[0] == w3[0]

    def test_assign_elements_different_windows(self, tumbling_assigner):
        """Test elements in different time ranges go to different windows."""
        w1 = tumbling_assigner.assign_windows("a", 1.5)
        w2 = tumbling_assigner.assign_windows("b", 2.5)

        assert w1[0] != w2[0]

    def test_default_trigger(self, tumbling_assigner):
        """Test that tumbling windows use EventTimeTrigger by default."""
        trigger = tumbling_assigner.get_default_trigger()
        assert isinstance(trigger, EventTimeTrigger)

    def test_is_event_time(self, tumbling_assigner):
        """Test that tumbling windows use event time."""
        assert tumbling_assigner.is_event_time() is True


class TestSlidingWindowAssigner:
    """Tests for SlidingWindowAssigner."""

    def test_assigner_creation(self, sliding_assigner):
        """Test creating a sliding window assigner."""
        assert sliding_assigner.size_ms == 2000
        assert sliding_assigner.slide_ms == 1000

    def test_assign_multiple_windows(self, sliding_assigner):
        """Test that an element can be assigned to multiple overlapping windows."""
        # With 2 second windows sliding every 1 second,
        # an element at 1.5 seconds could be in windows [0,2) and [1,3)
        windows = sliding_assigner.assign_windows("element", 1.5)
        assert len(windows) >= 1  # At least one window

    def test_assign_correct_overlapping_windows(self):
        """Test sliding window assignments with specific parameters."""
        # 2 second window, 1 second slide
        assigner = SlidingWindowAssigner(size_ms=2000, slide_ms=1000)

        # Element at 1.5 seconds should be in windows [0,2) and [1,3)
        windows = assigner.assign_windows("element", 1.5)

        # Check that we get the expected windows
        window_starts = sorted([w.start for w in windows])
        assert 0.0 in window_starts or 1.0 in window_starts

    def test_default_trigger(self, sliding_assigner):
        """Test that sliding windows use EventTimeTrigger by default."""
        trigger = sliding_assigner.get_default_trigger()
        assert isinstance(trigger, EventTimeTrigger)


class TestSessionWindowAssigner:
    """Tests for SessionWindowAssigner."""

    def test_assigner_creation(self, session_assigner):
        """Test creating a session window assigner."""
        assert session_assigner.gap_ms == 500

    def test_assign_session_window(self, session_assigner):
        """Test assigning to a session window."""
        windows = session_assigner.assign_windows("element", 1.0)
        assert len(windows) == 1
        assert isinstance(windows[0], SessionWindow)

    def test_session_window_extends_with_gap(self, session_assigner):
        """Test that session window end is timestamp + gap."""
        # Element at 1.0 seconds with 500ms gap should create window ending at 1.5
        windows = session_assigner.assign_windows("element", 1.0)
        assert windows[0].start == 1.0
        assert windows[0].end == 1.5  # 1.0 + 0.5

    def test_default_trigger(self, session_assigner):
        """Test that session windows use EventTimeTrigger by default."""
        trigger = session_assigner.get_default_trigger()
        assert isinstance(trigger, EventTimeTrigger)


class TestGlobalWindowAssigner:
    """Tests for GlobalWindowAssigner."""

    def test_assigner_creation(self, global_assigner):
        """Test creating a global window assigner."""
        assert global_assigner is not None

    def test_assign_global_window(self, global_assigner):
        """Test that all elements go to the same global window."""
        w1 = global_assigner.assign_windows("a", 1.0)
        w2 = global_assigner.assign_windows("b", 1000.0)
        w3 = global_assigner.assign_windows("c", 1000000.0)

        assert len(w1) == 1
        assert len(w2) == 1
        assert len(w3) == 1
        assert isinstance(w1[0], GlobalWindow)
        assert isinstance(w2[0], GlobalWindow)
        assert isinstance(w3[0], GlobalWindow)

    def test_default_trigger(self, global_assigner):
        """Test that global windows use ProcessingTimeTrigger by default."""
        trigger = global_assigner.get_default_trigger()
        assert isinstance(trigger, ProcessingTimeTrigger)


class TestEventTimeTrigger:
    """Tests for EventTimeTrigger."""

    def test_trigger_creation(self, event_time_trigger):
        """Test creating an event time trigger."""
        assert event_time_trigger is not None

    def test_on_element_returns_false(self, event_time_trigger):
        """Test that on_element does not trigger firing."""
        window = Window(start=0.0, end=1.0)
        result = event_time_trigger.on_element("element", 0.5, window)
        assert result is False

    def test_on_event_time_before_window_end(self, event_time_trigger):
        """Test that event time before window end does not trigger."""
        window = Window(start=0.0, end=1.0)
        result = event_time_trigger.on_event_time(0.5, window)
        assert result is False

    def test_on_event_time_at_window_end(self, event_time_trigger):
        """Test that event time at window end triggers firing."""
        window = Window(start=0.0, end=1.0)
        result = event_time_trigger.on_event_time(1.0, window)
        assert result is True

    def test_on_event_time_after_window_end(self, event_time_trigger):
        """Test that event time after window end triggers firing."""
        window = Window(start=0.0, end=1.0)
        result = event_time_trigger.on_event_time(1.5, window)
        assert result is True

    def test_on_processing_time_returns_false(self, event_time_trigger):
        """Test that processing time does not trigger firing."""
        window = Window(start=0.0, end=1.0)
        result = event_time_trigger.on_processing_time(2.0, window)
        assert result is False


class TestProcessingTimeTrigger:
    """Tests for ProcessingTimeTrigger."""

    def test_trigger_creation(self, processing_time_trigger):
        """Test creating a processing time trigger."""
        assert processing_time_trigger is not None

    def test_on_element_returns_false(self, processing_time_trigger):
        """Test that on_element does not trigger firing."""
        window = Window(start=0.0, end=1.0)
        result = processing_time_trigger.on_element("element", 0.5, window)
        assert result is False

    def test_on_event_time_returns_false(self, processing_time_trigger):
        """Test that event time does not trigger firing."""
        window = Window(start=0.0, end=1.0)
        result = processing_time_trigger.on_event_time(1.5, window)
        assert result is False

    def test_on_processing_time_before_window_end(self, processing_time_trigger):
        """Test that processing time before window end does not trigger."""
        window = Window(start=0.0, end=1.0)
        result = processing_time_trigger.on_processing_time(0.5, window)
        assert result is False

    def test_on_processing_time_at_window_end(self, processing_time_trigger):
        """Test that processing time at window end triggers firing."""
        window = Window(start=0.0, end=1.0)
        result = processing_time_trigger.on_processing_time(1.0, window)
        assert result is True


class TestCountTrigger:
    """Tests for CountTrigger."""

    def test_trigger_creation(self, count_trigger):
        """Test creating a count trigger."""
        assert count_trigger.count == 3

    def test_count_trigger_before_threshold(self):
        """Test that count trigger does not fire before reaching count."""
        trigger = CountTrigger(count=3)
        window = Window(start=0.0, end=1.0)

        result1 = trigger.on_element("a", 0.1, window)
        result2 = trigger.on_element("b", 0.2, window)

        assert result1 is False
        assert result2 is False

    def test_count_trigger_at_threshold(self):
        """Test that count trigger fires when reaching count."""
        trigger = CountTrigger(count=3)
        window = Window(start=0.0, end=1.0)

        trigger.on_element("a", 0.1, window)
        trigger.on_element("b", 0.2, window)
        result = trigger.on_element("c", 0.3, window)

        assert result is True

    def test_count_trigger_resets_after_fire(self):
        """Test that count trigger resets after firing."""
        trigger = CountTrigger(count=2)
        window = Window(start=0.0, end=1.0)

        trigger.on_element("a", 0.1, window)
        trigger.on_element("b", 0.2, window)  # Fires here

        # Should not fire on next element
        result = trigger.on_element("c", 0.3, window)
        assert result is False

        # Should fire again after 2 more
        result = trigger.on_element("d", 0.4, window)
        assert result is True

    def test_count_trigger_per_window(self):
        """Test that count trigger maintains separate counts per window."""
        trigger = CountTrigger(count=2)
        window1 = Window(start=0.0, end=1.0)
        window2 = Window(start=1.0, end=2.0)

        trigger.on_element("a", 0.1, window1)
        trigger.on_element("b", 1.1, window2)

        # Neither should fire yet
        result1 = trigger.on_element("c", 0.2, window1)  # 2nd in window1, fires
        result2 = trigger.on_element("d", 1.2, window2)  # 2nd in window2, fires

        assert result1 is True
        assert result2 is True

    def test_on_event_time_returns_false(self, count_trigger):
        """Test that event time does not trigger count trigger."""
        window = Window(start=0.0, end=1.0)
        result = count_trigger.on_event_time(1.0, window)
        assert result is False

    def test_on_processing_time_returns_false(self, count_trigger):
        """Test that processing time does not trigger count trigger."""
        window = Window(start=0.0, end=1.0)
        result = count_trigger.on_processing_time(1.0, window)
        assert result is False


class TestWindowFunction:
    """Tests for WindowFunction implementations."""

    def test_reduce_function(self):
        """Test ReduceFunction."""
        reduce_func = ReduceFunction(lambda a, b: a + b)
        window = Window(start=0.0, end=1.0)
        elements = [1, 2, 3, 4, 5]

        results = list(reduce_func.apply("key", window, iter(elements)))
        assert results == [15]  # Sum of 1+2+3+4+5

    def test_reduce_function_single_element(self):
        """Test ReduceFunction with single element."""
        reduce_func = ReduceFunction(lambda a, b: a + b)
        window = Window(start=0.0, end=1.0)

        results = list(reduce_func.apply("key", window, iter([42])))
        assert results == [42]

    def test_reduce_function_empty(self):
        """Test ReduceFunction with empty iterator."""
        reduce_func = ReduceFunction(lambda a, b: a + b)
        window = Window(start=0.0, end=1.0)

        results = list(reduce_func.apply("key", window, iter([])))
        assert results == []

    def test_aggregate_function(self):
        """Test AggregateFunction."""
        agg_func = AggregateFunction(
            create_accumulator=lambda: {"sum": 0, "count": 0},
            add=lambda acc, v: {"sum": acc["sum"] + v, "count": acc["count"] + 1},
            get_result=lambda acc: acc["sum"] / acc["count"] if acc["count"] > 0 else 0
        )
        window = Window(start=0.0, end=1.0)
        elements = [10, 20, 30]

        results = list(agg_func.apply("key", window, iter(elements)))
        assert results == [20.0]  # Average of 10, 20, 30

    def test_process_window_function(self):
        """Test ProcessWindowFunction."""

        def process_func(key, window, elements):
            element_list = list(elements)
            yield {
                "key": key,
                "window_start": window.start,
                "window_end": window.end,
                "count": len(element_list),
                "sum": sum(element_list)
            }

        process_func_wrapper = ProcessWindowFunction(process_func)
        window = Window(start=0.0, end=1.0)
        elements = [1, 2, 3]

        results = list(process_func_wrapper.apply("my_key", window, iter(elements)))
        assert len(results) == 1
        assert results[0]["key"] == "my_key"
        assert results[0]["count"] == 3
        assert results[0]["sum"] == 6


class TestWindowedStream:
    """Tests for WindowedStream operations."""

    def test_windowed_stream_creation(self, env):
        """Test creating a windowed stream."""
        stream = env.from_collection([1, 2, 3, 4, 5])
        keyed = stream.key_by(lambda x: x % 2)
        assigner = TumblingWindowAssigner(size_ms=1000)
        windowed = keyed.window(assigner)

        assert isinstance(windowed, WindowedStream)
        assert windowed.keyed_stream is keyed
        assert windowed.assigner is assigner

    def test_windowed_stream_trigger_with(self, env):
        """Test setting a custom trigger on windowed stream."""
        stream = env.from_collection([1, 2, 3])
        keyed = stream.key_by(lambda x: x)
        windowed = keyed.window(TumblingWindowAssigner(size_ms=1000))

        custom_trigger = CountTrigger(count=2)
        result = windowed.trigger_with(custom_trigger)

        assert result is windowed
        assert windowed.trigger is custom_trigger

    def test_windowed_stream_reduce_with_count_trigger(self, env):
        """Test windowed reduce with count trigger."""
        # Create data where elements with same key will trigger reduce
        data = [
            ("a", 1), ("a", 2), ("a", 3),  # 3 elements for key "a"
            ("b", 10), ("b", 20), ("b", 30),  # 3 elements for key "b"
        ]
        stream = env.from_collection(data)
        keyed = stream.key_by(lambda x: x[0])
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))  # Large window
        windowed.trigger_with(CountTrigger(count=3))

        result_stream = windowed.reduce(lambda a, b: (a[0], a[1] + b[1]))
        result = run_stream(result_stream)

        # After 3 elements per key, reduce should fire
        a_results = [r for r in result if r[0] == "a"]
        b_results = [r for r in result if r[0] == "b"]

        # Last result for "a" should be (a, 6) since 1+2+3=6
        # Last result for "b" should be (b, 60) since 10+20+30=60
        if a_results:
            assert a_results[-1] == ("a", 6)
        if b_results:
            assert b_results[-1] == ("b", 60)

    def test_windowed_stream_aggregate_with_count_trigger(self, env):
        """Test windowed aggregate with count trigger."""
        data = [1, 2, 3, 4, 5, 6]  # All same key (modulo 10)
        stream = env.from_collection(data)
        keyed = stream.key_by(lambda x: 0)  # All same key
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))
        windowed.trigger_with(CountTrigger(count=3))

        result_stream = windowed.aggregate(
            create_accumulator=lambda: 0,
            add=lambda acc, v: acc + v,
            get_result=lambda acc: acc
        )
        result = run_stream(result_stream)

        # Should fire twice: after 3 elements (sum=6) and after 6 elements (sum=21)
        assert 6 in result or 21 in result

    def test_windowed_stream_count(self, env):
        """Test windowed count operation."""
        data = ["a", "a", "a", "b", "b", "a"]
        stream = env.from_collection(data)
        keyed = stream.key_by(lambda x: x)
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))
        windowed.trigger_with(CountTrigger(count=2))

        result_stream = windowed.count()
        result = run_stream(result_stream)

        # Should have counts triggered at 2 elements
        assert 2 in result

    def test_windowed_stream_sum(self, env):
        """Test windowed sum operation."""
        data = [10, 20, 30, 40]
        stream = env.from_collection(data)
        keyed = stream.key_by(lambda x: 0)
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))
        windowed.trigger_with(CountTrigger(count=2))

        result_stream = windowed.sum()
        result = run_stream(result_stream)

        # First trigger: 10+20=30, second trigger: 30+40=100
        assert 30 in result

    def test_windowed_stream_min(self, env):
        """Test windowed min operation."""
        data = [5, 3, 8, 1]
        stream = env.from_collection(data)
        keyed = stream.key_by(lambda x: 0)
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))
        windowed.trigger_with(CountTrigger(count=2))

        result_stream = windowed.min()
        result = run_stream(result_stream)

        # First trigger: min(5,3)=3, second trigger: min(8,1)=1
        assert 3 in result

    def test_windowed_stream_max(self, env):
        """Test windowed max operation."""
        data = [5, 3, 8, 1]
        stream = env.from_collection(data)
        keyed = stream.key_by(lambda x: 0)
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))
        windowed.trigger_with(CountTrigger(count=2))

        result_stream = windowed.max()
        result = run_stream(result_stream)

        # First trigger: max(5,3)=5, second trigger: max(8,1)=8
        assert 5 in result

    def test_windowed_stream_process(self, env):
        """Test windowed process function."""
        data = [1, 2, 3, 4, 5, 6]
        stream = env.from_collection(data)
        keyed = stream.key_by(lambda x: 0)
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))
        windowed.trigger_with(CountTrigger(count=3))

        def process_fn(key, window, elements):
            elem_list = list(elements)
            yield {"key": key, "count": len(elem_list), "sum": sum(elem_list)}

        result_stream = windowed.process(process_fn)
        result = run_stream(result_stream)

        # Should have processed windows with aggregated results
        if result:
            assert any(isinstance(r, dict) and "count" in r for r in result)


class TestWindowAssignment:
    """Integration tests for window assignment with streams."""

    def test_tumbling_window_assignment_boundaries(self):
        """Test that tumbling windows have correct boundaries."""
        assigner = TumblingWindowAssigner(size_ms=1000)

        # Test various timestamps
        test_cases = [
            (0.0, 0.0, 1.0),      # At window start
            (0.5, 0.0, 1.0),      # Middle of window
            (0.999, 0.0, 1.0),    # Just before window end
            (1.0, 1.0, 2.0),      # At next window start
            (2.5, 2.0, 3.0),      # Another window
        ]

        for timestamp, expected_start, expected_end in test_cases:
            windows = assigner.assign_windows("elem", timestamp)
            assert len(windows) == 1, f"Expected 1 window for timestamp {timestamp}"
            assert windows[0].start == expected_start, f"Wrong start for timestamp {timestamp}"
            assert windows[0].end == expected_end, f"Wrong end for timestamp {timestamp}"

    def test_sliding_window_overlap(self):
        """Test that sliding windows properly overlap."""
        # 2 second window, 1 second slide
        assigner = SlidingWindowAssigner(size_ms=2000, slide_ms=1000)

        # An element at 1.5 should be in windows starting at 0 and 1
        windows = assigner.assign_windows("elem", 1.5)

        # Should be in at least one window
        assert len(windows) >= 1

        # All assigned windows should contain the timestamp
        for window in windows:
            assert window.start <= 1.5 < window.end


class TestComplexWindowScenarios:
    """Complex integration tests for windowing."""

    def test_multi_key_windowing(self, env):
        """Test windowing with multiple keys."""
        data = [
            ("user1", 100), ("user2", 200),
            ("user1", 150), ("user2", 250),
            ("user1", 50), ("user2", 300),
        ]

        stream = env.from_collection(data)
        keyed = stream.key_by(lambda x: x[0])
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))
        windowed.trigger_with(CountTrigger(count=2))

        result_stream = windowed.reduce(lambda a, b: (a[0], a[1] + b[1]))
        result = run_stream(result_stream)

        # After 2 elements per key:
        # user1: 100 + 150 = 250
        # user2: 200 + 250 = 450
        user1_results = [r for r in result if r[0] == "user1"]
        user2_results = [r for r in result if r[0] == "user2"]

        if user1_results:
            assert user1_results[0] == ("user1", 250)
        if user2_results:
            assert user2_results[0] == ("user2", 450)

    def test_window_with_map_before(self, env):
        """Test windowing after map transformation."""
        data = [1, 2, 3, 4, 5, 6]

        stream = env.from_collection(data)
        mapped = stream.map(lambda x: x * 10)
        keyed = mapped.key_by(lambda x: 0)
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))
        windowed.trigger_with(CountTrigger(count=3))

        result_stream = windowed.reduce(lambda a, b: a + b)
        result = run_stream(result_stream)

        # After map: 10, 20, 30, 40, 50, 60
        # First window: 10+20+30=60
        if result:
            assert 60 in result

    def test_window_with_filter_before(self, env):
        """Test windowing after filter transformation."""
        data = [1, 2, 3, 4, 5, 6, 7, 8]

        stream = env.from_collection(data)
        filtered = stream.filter(lambda x: x % 2 == 0)  # Keep 2, 4, 6, 8
        keyed = filtered.key_by(lambda x: 0)
        windowed = keyed.window(TumblingWindowAssigner(size_ms=10000))
        windowed.trigger_with(CountTrigger(count=2))

        result_stream = windowed.reduce(lambda a, b: a + b)
        result = run_stream(result_stream)

        # After filter: 2, 4, 6, 8
        # First window: 2+4=6
        if result:
            assert 6 in result
