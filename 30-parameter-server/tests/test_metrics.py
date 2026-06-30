"""Tests for performance metrics."""

import pytest
import time

from paramserver.enterprise.metrics import (
    MetricsCollector,
    PerformanceMetrics,
)


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics dataclass."""

    def test_create_default(self):
        """Test default creation."""
        metrics = PerformanceMetrics()

        assert metrics.throughput == 0.0
        assert metrics.latency_ms == 0.0
        assert metrics.push_count == 0
        assert metrics.pull_count == 0
        assert metrics.bytes_sent == 0
        assert metrics.bytes_received == 0
        assert metrics.timestamp > 0

    def test_create_with_values(self):
        """Test creation with values."""
        metrics = PerformanceMetrics(
            throughput=100.0,
            latency_ms=5.0,
            push_count=50,
            pull_count=30,
            bytes_sent=1000,
            bytes_received=2000,
        )

        assert metrics.throughput == 100.0
        assert metrics.latency_ms == 5.0
        assert metrics.push_count == 50
        assert metrics.pull_count == 30


class TestMetricsCollectorInit:
    """Tests for MetricsCollector initialization."""

    def test_create_default(self):
        """Test default creation."""
        collector = MetricsCollector()
        assert collector.window_size == 1000

    def test_create_custom(self):
        """Test custom creation."""
        collector = MetricsCollector(window_size=500)
        assert collector.window_size == 500


class TestRecordOperations:
    """Tests for recording operations."""

    def test_record_push(self):
        """Test recording push operation."""
        collector = MetricsCollector()

        collector.record_push(latency_ms=10.0, bytes_count=100)

        stats = collector.get_stats()
        assert stats["total_pushes"] == 1
        assert stats["bytes_sent"] == 100

    def test_record_pull(self):
        """Test recording pull operation."""
        collector = MetricsCollector()

        collector.record_pull(latency_ms=5.0, bytes_count=200)

        stats = collector.get_stats()
        assert stats["total_pulls"] == 1
        assert stats["bytes_received"] == 200

    def test_record_multiple(self):
        """Test recording multiple operations."""
        collector = MetricsCollector()

        for i in range(10):
            collector.record_push(latency_ms=float(i), bytes_count=100)
            collector.record_pull(latency_ms=float(i), bytes_count=200)

        stats = collector.get_stats()
        assert stats["total_pushes"] == 10
        assert stats["total_pulls"] == 10
        assert stats["bytes_sent"] == 1000
        assert stats["bytes_received"] == 2000


class TestLatencyCalculation:
    """Tests for latency calculation."""

    def test_get_push_latency(self):
        """Test getting push latency."""
        collector = MetricsCollector()

        collector.record_push(latency_ms=10.0)
        collector.record_push(latency_ms=20.0)
        collector.record_push(latency_ms=30.0)

        latency = collector.get_push_latency()

        assert latency == pytest.approx(20.0)

    def test_get_pull_latency(self):
        """Test getting pull latency."""
        collector = MetricsCollector()

        collector.record_pull(latency_ms=5.0)
        collector.record_pull(latency_ms=15.0)

        latency = collector.get_pull_latency()

        assert latency == pytest.approx(10.0)

    def test_latency_empty(self):
        """Test latency with no samples."""
        collector = MetricsCollector()

        assert collector.get_push_latency() == 0.0
        assert collector.get_pull_latency() == 0.0


class TestThroughput:
    """Tests for throughput calculation."""

    def test_get_throughput(self):
        """Test getting throughput."""
        collector = MetricsCollector()

        # Record 100 operations
        for _ in range(100):
            collector.record_push(latency_ms=1.0)

        throughput = collector.get_throughput()

        # Should be positive
        assert throughput > 0

    def test_throughput_empty(self):
        """Test throughput with no operations."""
        collector = MetricsCollector()

        # Small sleep to ensure non-zero time
        time.sleep(0.01)

        throughput = collector.get_throughput()

        assert throughput == 0.0


class TestWindowSize:
    """Tests for sliding window behavior."""

    def test_window_limits_samples(self):
        """Test that window limits samples."""
        collector = MetricsCollector(window_size=5)

        # Record more than window size
        for i in range(10):
            collector.record_push(latency_ms=float(i))

        # Average should be from last 5 samples (5, 6, 7, 8, 9)
        latency = collector.get_push_latency()

        assert latency == pytest.approx(7.0)


class TestGetMetrics:
    """Tests for getting metrics snapshot."""

    def test_get_metrics(self):
        """Test getting metrics snapshot."""
        collector = MetricsCollector()

        collector.record_push(latency_ms=10.0, bytes_count=100)
        collector.record_pull(latency_ms=20.0, bytes_count=200)

        metrics = collector.get_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.push_count == 1
        assert metrics.pull_count == 1
        assert metrics.bytes_sent == 100
        assert metrics.bytes_received == 200
        assert metrics.latency_ms == pytest.approx(15.0)


class TestGetStats:
    """Tests for getting statistics."""

    def test_get_stats(self):
        """Test getting statistics dictionary."""
        collector = MetricsCollector()

        for _ in range(5):
            collector.record_push(latency_ms=10.0, bytes_count=100)
            collector.record_pull(latency_ms=5.0, bytes_count=200)

        stats = collector.get_stats()

        assert "elapsed_seconds" in stats
        assert "total_pushes" in stats
        assert "total_pulls" in stats
        assert "throughput_ops_sec" in stats
        assert "avg_push_latency_ms" in stats
        assert "avg_pull_latency_ms" in stats
        assert "bytes_sent" in stats
        assert "bytes_received" in stats
        assert "bandwidth_sent_mb_sec" in stats
        assert "bandwidth_recv_mb_sec" in stats

        assert stats["total_pushes"] == 5
        assert stats["total_pulls"] == 5
        assert stats["avg_push_latency_ms"] == pytest.approx(10.0)
        assert stats["avg_pull_latency_ms"] == pytest.approx(5.0)


class TestReset:
    """Tests for resetting metrics."""

    def test_reset(self):
        """Test resetting all metrics."""
        collector = MetricsCollector()

        for _ in range(10):
            collector.record_push(latency_ms=10.0, bytes_count=100)

        collector.reset()

        stats = collector.get_stats()
        assert stats["total_pushes"] == 0
        assert stats["bytes_sent"] == 0
        assert collector.get_push_latency() == 0.0
