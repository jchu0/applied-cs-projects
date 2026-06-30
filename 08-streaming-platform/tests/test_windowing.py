"""Tests for windowing and event-time processing."""

import pytest
from datetime import datetime

pytest.importorskip("confluent_kafka")

from streaming.windowing import (
    Window,
    WindowedValue,
    WindowAssigner,
    TumblingWindowAssigner,
    SlidingWindowAssigner,
    SessionWindowAssigner,
    Watermark,
    WatermarkGenerator,
    BoundedOutOfOrdernessGenerator,
    IdleAwareWatermarkGenerator,
    WindowState,
    WindowOperator,
    count_aggregator,
    sum_aggregator,
    avg_aggregator,
    min_aggregator,
    max_aggregator,
    full_aggregator,
    AggregationResult,
)


# --- Window Tests ---


class TestWindow:
    """Tests for Window class."""

    def test_window_creation(self):
        """Test window creation."""
        window = Window(start=1000, end=2000)

        assert window.start == 1000
        assert window.end == 2000

    def test_window_duration(self):
        """Test window duration calculation."""
        window = Window(start=1000, end=6000)

        assert window.duration_ms == 5000

    def test_window_contains_timestamp(self):
        """Test timestamp containment check."""
        window = Window(start=1000, end=2000)

        assert window.contains(1000) is True  # Inclusive start
        assert window.contains(1500) is True
        assert window.contains(1999) is True
        assert window.contains(2000) is False  # Exclusive end
        assert window.contains(999) is False

    def test_window_repr(self):
        """Test window string representation."""
        window = Window(start=1704067200000, end=1704067260000)  # 2024-01-01 00:00:00

        repr_str = repr(window)

        assert "Window(" in repr_str
        assert "-" in repr_str


class TestWindowedValue:
    """Tests for WindowedValue class."""

    def test_windowed_value_creation(self):
        """Test windowed value creation."""
        window = Window(start=1000, end=2000)
        windowed = WindowedValue(window=window, value=100, timestamp=1500)

        assert windowed.window == window
        assert windowed.value == 100
        assert windowed.timestamp == 1500


# --- TumblingWindowAssigner Tests ---


class TestTumblingWindowAssigner:
    """Tests for TumblingWindowAssigner class."""

    def test_assign_to_single_window(self):
        """Test that event is assigned to exactly one window."""
        assigner = TumblingWindowAssigner(size_ms=60000)  # 1 minute windows

        windows = assigner.assign_windows(1704067230000)  # 30 seconds into window

        assert len(windows) == 1
        assert windows[0].duration_ms == 60000

    def test_window_alignment(self):
        """Test windows are aligned to size boundaries."""
        assigner = TumblingWindowAssigner(size_ms=60000)

        # Different timestamps in same minute
        windows1 = assigner.assign_windows(1704067200000)  # Start of minute
        windows2 = assigner.assign_windows(1704067230000)  # Middle of minute
        windows3 = assigner.assign_windows(1704067259999)  # End of minute

        assert windows1[0].start == windows2[0].start == windows3[0].start
        assert windows1[0].end == windows2[0].end == windows3[0].end

    def test_adjacent_windows_non_overlapping(self):
        """Test adjacent windows don't overlap."""
        assigner = TumblingWindowAssigner(size_ms=60000)

        windows1 = assigner.assign_windows(1704067200000)  # First minute
        windows2 = assigner.assign_windows(1704067260000)  # Second minute

        assert windows1[0].end == windows2[0].start

    def test_various_window_sizes(self):
        """Test various window sizes."""
        for size_ms in [1000, 5000, 60000, 300000, 3600000]:
            assigner = TumblingWindowAssigner(size_ms=size_ms)
            windows = assigner.assign_windows(0)

            assert len(windows) == 1
            assert windows[0].duration_ms == size_ms


# --- SlidingWindowAssigner Tests ---


class TestSlidingWindowAssigner:
    """Tests for SlidingWindowAssigner class."""

    def test_assign_to_multiple_windows(self):
        """Test event is assigned to overlapping windows."""
        assigner = SlidingWindowAssigner(size_ms=60000, slide_ms=30000)

        windows = assigner.assign_windows(1704067230000)

        # With 1-minute windows sliding every 30 seconds,
        # each event should be in 2 windows
        assert len(windows) == 2

    def test_window_overlap(self):
        """Test windows properly overlap."""
        assigner = SlidingWindowAssigner(size_ms=10000, slide_ms=5000)

        windows = assigner.assign_windows(7500)

        # Should be in windows [0, 10000) and [5000, 15000)
        assert len(windows) == 2
        window_starts = [w.start for w in windows]
        assert 0 in window_starts
        assert 5000 in window_starts

    def test_window_containment(self):
        """Test all assigned windows contain the timestamp."""
        assigner = SlidingWindowAssigner(size_ms=60000, slide_ms=20000)
        timestamp = 1704067250000

        windows = assigner.assign_windows(timestamp)

        for window in windows:
            assert window.contains(timestamp)

    def test_size_equals_slide(self):
        """Test when size equals slide (behaves like tumbling)."""
        assigner = SlidingWindowAssigner(size_ms=60000, slide_ms=60000)

        windows = assigner.assign_windows(1704067230000)

        assert len(windows) == 1


# --- SessionWindowAssigner Tests ---


class TestSessionWindowAssigner:
    """Tests for SessionWindowAssigner class."""

    def test_initial_window_creation(self):
        """Test initial session window creation."""
        assigner = SessionWindowAssigner(gap_ms=30000)

        windows = assigner.assign_windows(1704067200000)

        assert len(windows) == 1
        assert windows[0].start == 1704067200000
        assert windows[0].end == 1704067230000  # start + gap

    def test_merge_overlapping_sessions(self):
        """Test merging overlapping session windows."""
        assigner = SessionWindowAssigner(gap_ms=30000)

        # Create windows that should merge
        window1 = Window(start=0, end=30000)
        window2 = Window(start=20000, end=50000)
        window3 = Window(start=100000, end=130000)

        merged = assigner.merge_windows([window1, window2, window3])

        assert len(merged) == 2
        # First two should be merged
        assert merged[0].start == 0
        assert merged[0].end == 50000
        # Third is separate
        assert merged[1].start == 100000

    def test_merge_adjacent_sessions(self):
        """Test merging adjacent session windows."""
        assigner = SessionWindowAssigner(gap_ms=30000)

        window1 = Window(start=0, end=30000)
        window2 = Window(start=30000, end=60000)

        merged = assigner.merge_windows([window1, window2])

        # Adjacent windows should merge
        assert len(merged) == 1
        assert merged[0].start == 0
        assert merged[0].end == 60000

    def test_no_merge_separate_sessions(self):
        """Test separate sessions are not merged."""
        assigner = SessionWindowAssigner(gap_ms=30000)

        window1 = Window(start=0, end=30000)
        window2 = Window(start=60000, end=90000)  # Gap of 30000

        merged = assigner.merge_windows([window1, window2])

        assert len(merged) == 2

    def test_merge_empty_list(self):
        """Test merging empty list."""
        assigner = SessionWindowAssigner(gap_ms=30000)

        merged = assigner.merge_windows([])

        assert merged == []


# --- Watermark Tests ---


class TestWatermark:
    """Tests for Watermark class."""

    def test_watermark_creation(self):
        """Test watermark creation."""
        watermark = Watermark(timestamp=1704067200000)

        assert watermark.timestamp == 1704067200000

    def test_watermark_repr(self):
        """Test watermark string representation."""
        watermark = Watermark(timestamp=1704067200000)

        assert "Watermark(" in repr(watermark)


# --- WatermarkGenerator Tests ---


class TestWatermarkGenerator:
    """Tests for WatermarkGenerator class."""

    def test_initial_watermark(self):
        """Test initial watermark is zero."""
        generator = WatermarkGenerator()

        watermark = generator.get_watermark()

        assert watermark.timestamp == 0

    def test_watermark_advances(self):
        """Test watermark advances with events."""
        generator = WatermarkGenerator()

        generator.on_event(1000)
        generator.on_event(2000)
        generator.on_event(3000)

        watermark = generator.get_watermark()
        assert watermark.timestamp == 3000

    def test_watermark_ignores_old_events(self):
        """Test watermark doesn't go backwards."""
        generator = WatermarkGenerator()

        generator.on_event(3000)
        generator.on_event(1000)  # Old event

        watermark = generator.get_watermark()
        assert watermark.timestamp == 3000


class TestBoundedOutOfOrdernessGenerator:
    """Tests for BoundedOutOfOrdernessGenerator class."""

    def test_watermark_with_out_of_orderness(self):
        """Test watermark respects out-of-orderness bound."""
        generator = BoundedOutOfOrdernessGenerator(max_out_of_orderness_ms=5000)

        generator.on_event(10000)

        watermark = generator.get_watermark()
        assert watermark.timestamp == 5000  # 10000 - 5000

    def test_multiple_events_with_out_of_orderness(self):
        """Test multiple events with out-of-orderness."""
        generator = BoundedOutOfOrdernessGenerator(max_out_of_orderness_ms=10000)

        generator.on_event(15000)
        generator.on_event(20000)
        generator.on_event(18000)  # Out of order

        watermark = generator.get_watermark()
        assert watermark.timestamp == 10000  # 20000 - 10000


class TestIdleAwareWatermarkGenerator:
    """Tests for IdleAwareWatermarkGenerator class."""

    def test_normal_operation(self):
        """Test normal operation with active events."""
        generator = IdleAwareWatermarkGenerator(
            max_out_of_orderness_ms=5000,
            idle_timeout_ms=30000,
        )

        generator.on_event(10000)

        watermark = generator.get_watermark()
        # Should be bounded by out-of-orderness, not idle
        assert watermark.timestamp == 5000


# --- WindowOperator Tests ---


class TestWindowOperator:
    """Tests for WindowOperator class."""

    def test_process_element_tumbling(self):
        """Test processing elements with tumbling window."""
        assigner = TumblingWindowAssigner(size_ms=10000)
        operator = WindowOperator(assigner, count_aggregator)

        # Process events within same window
        result1 = operator.process_element("user1", 100, 1000)
        result2 = operator.process_element("user1", 200, 2000)
        result3 = operator.process_element("user1", 300, 3000)

        # No windows should fire yet (watermark hasn't passed window end)
        assert len(result1) == 0
        assert len(result2) == 0
        assert len(result3) == 0

    def test_window_fires_on_watermark_advance(self):
        """Test window fires when watermark passes window end."""
        assigner = TumblingWindowAssigner(size_ms=10000)
        operator = WindowOperator(assigner, count_aggregator)

        operator.process_element("user1", 100, 1000)
        operator.process_element("user1", 200, 5000)
        operator.process_element("user1", 300, 8000)

        # Advance watermark past window end
        results = operator.advance_watermark(15000)

        assert len(results) == 1
        assert results[0].value == 3  # count

    def test_multiple_keys(self):
        """Test processing multiple keys."""
        assigner = TumblingWindowAssigner(size_ms=10000)
        operator = WindowOperator(assigner, sum_aggregator)

        operator.process_element("user1", 100.0, 1000)
        operator.process_element("user2", 200.0, 2000)
        operator.process_element("user1", 50.0, 3000)
        operator.process_element("user2", 75.0, 4000)

        results = operator.advance_watermark(15000)

        # Should have results for both keys
        assert len(results) == 2
        values = {r.value for r in results}
        assert 150.0 in values  # user1: 100 + 50
        assert 275.0 in values  # user2: 200 + 75

    def test_multiple_windows(self):
        """Test events in different windows."""
        assigner = TumblingWindowAssigner(size_ms=10000)
        operator = WindowOperator(assigner, count_aggregator)

        # Window 1: [0, 10000)
        operator.process_element("user1", 1, 1000)
        operator.process_element("user1", 2, 5000)

        # Window 2: [10000, 20000)
        operator.process_element("user1", 3, 15000)

        # Advance watermark to fire first window only
        results = operator.advance_watermark(12000)

        assert len(results) == 1
        assert results[0].value == 2  # First window count

    def test_window_cleanup(self):
        """Test windows are cleaned up after firing."""
        assigner = TumblingWindowAssigner(size_ms=10000)
        operator = WindowOperator(assigner, count_aggregator)

        operator.process_element("user1", 1, 1000)
        operator.advance_watermark(15000)

        # Window should be cleaned up
        assert "user1" not in operator.window_states or len(operator.window_states["user1"]) == 0


# --- Aggregator Tests ---


class TestAggregators:
    """Tests for aggregation functions."""

    def test_count_aggregator(self):
        """Test count aggregation."""
        elements = [1, 2, 3, 4, 5]

        result = count_aggregator(elements)

        assert result == 5

    def test_count_aggregator_empty(self):
        """Test count aggregation with empty list."""
        result = count_aggregator([])

        assert result == 0

    def test_sum_aggregator(self):
        """Test sum aggregation."""
        elements = [1.0, 2.0, 3.0, 4.0, 5.0]

        result = sum_aggregator(elements)

        assert result == 15.0

    def test_sum_aggregator_empty(self):
        """Test sum aggregation with empty list."""
        result = sum_aggregator([])

        assert result == 0

    def test_avg_aggregator(self):
        """Test average aggregation."""
        elements = [10.0, 20.0, 30.0]

        result = avg_aggregator(elements)

        assert result == 20.0

    def test_avg_aggregator_empty(self):
        """Test average aggregation with empty list."""
        result = avg_aggregator([])

        assert result == 0

    def test_min_aggregator(self):
        """Test min aggregation."""
        elements = [5.0, 2.0, 8.0, 1.0, 9.0]

        result = min_aggregator(elements)

        assert result == 1.0

    def test_min_aggregator_empty(self):
        """Test min aggregation with empty list."""
        result = min_aggregator([])

        assert result == 0

    def test_max_aggregator(self):
        """Test max aggregation."""
        elements = [5.0, 2.0, 8.0, 1.0, 9.0]

        result = max_aggregator(elements)

        assert result == 9.0

    def test_max_aggregator_empty(self):
        """Test max aggregation with empty list."""
        result = max_aggregator([])

        assert result == 0

    def test_full_aggregator(self):
        """Test full aggregation result."""
        elements = [10.0, 20.0, 30.0, 40.0, 50.0]

        result = full_aggregator(elements)

        assert isinstance(result, AggregationResult)
        assert result.count == 5
        assert result.sum == 150.0
        assert result.min == 10.0
        assert result.max == 50.0
        assert result.avg == 30.0

    def test_full_aggregator_empty(self):
        """Test full aggregation with empty list."""
        result = full_aggregator([])

        assert result.count == 0
        assert result.sum == 0
        assert result.min == 0
        assert result.max == 0
        assert result.avg == 0


# --- Integration Tests ---


class TestWindowingIntegration:
    """Integration tests for windowing."""

    def test_tumbling_window_aggregation(self):
        """Test complete tumbling window aggregation flow."""
        assigner = TumblingWindowAssigner(size_ms=5000)
        operator = WindowOperator(assigner, sum_aggregator)

        # Simulate stream of events
        events = [
            ("key1", 10.0, 1000),
            ("key1", 20.0, 2000),
            ("key1", 30.0, 3000),  # All in window [0, 5000)
            ("key1", 40.0, 6000),  # In window [5000, 10000)
        ]

        for key, value, timestamp in events:
            operator.process_element(key, value, timestamp)

        # Fire all windows
        results = operator.advance_watermark(15000)

        # Should have 2 windows
        assert len(results) == 2
        values = sorted([r.value for r in results])
        assert values[0] == 40.0  # Second window
        assert values[1] == 60.0  # First window (10+20+30)

    def test_sliding_window_aggregation(self):
        """Test sliding window aggregation with overlapping results."""
        assigner = SlidingWindowAssigner(size_ms=10000, slide_ms=5000)
        operator = WindowOperator(assigner, count_aggregator)

        # Events at various times
        operator.process_element("key1", 1, 2000)
        operator.process_element("key1", 2, 7000)
        operator.process_element("key1", 3, 12000)

        # Event at 2000 is in windows [0, 10000) and [-5000, 5000) if negative allowed
        # Event at 7000 is in windows [0, 10000) and [5000, 15000)
        # Event at 12000 is in windows [5000, 15000) and [10000, 20000)

        results = operator.advance_watermark(20000)

        # Should have multiple window results due to overlap
        assert len(results) >= 2

    def test_multi_key_windowing(self):
        """Test windowing with multiple keys."""
        assigner = TumblingWindowAssigner(size_ms=10000)
        operator = WindowOperator(assigner, full_aggregator)

        # Multiple keys in same window
        operator.process_element("user1", 100.0, 1000)
        operator.process_element("user2", 200.0, 2000)
        operator.process_element("user3", 300.0, 3000)
        operator.process_element("user1", 150.0, 4000)
        operator.process_element("user2", 250.0, 5000)

        results = operator.advance_watermark(15000)

        # Should have 3 results (one per key)
        assert len(results) == 3

        # Find user1's result
        user1_result = next(r for r in results if r.value.sum == 250.0)  # 100 + 150
        assert user1_result.value.count == 2
        assert user1_result.value.avg == 125.0
