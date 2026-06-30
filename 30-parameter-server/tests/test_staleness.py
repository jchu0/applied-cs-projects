"""Tests for staleness control."""

import pytest

from paramserver.enterprise.staleness import (
    StalenessController,
    AdaptiveSSP,
)


class TestStalenessControllerInit:
    """Tests for StalenessController initialization."""

    def test_create_default(self):
        """Test default creation."""
        controller = StalenessController()
        assert controller.max_staleness == 5
        assert controller.min_staleness == 0

    def test_create_custom(self):
        """Test custom creation."""
        controller = StalenessController(max_staleness=10, min_staleness=2)
        assert controller.max_staleness == 10
        assert controller.min_staleness == 2


class TestWorkerRegistration:
    """Tests for worker registration."""

    def test_register_worker(self):
        """Test registering a worker."""
        controller = StalenessController()
        controller.register_worker(0, initial_clock=100)

        assert controller.get_staleness(0) == 0

    def test_register_multiple_workers(self):
        """Test registering multiple workers."""
        controller = StalenessController()
        controller.register_worker(0, initial_clock=100)
        controller.register_worker(1, initial_clock=98)
        controller.register_worker(2, initial_clock=102)

        assert controller.get_staleness(1) == 4  # 102 - 98

    def test_update_clock(self):
        """Test updating worker clock."""
        controller = StalenessController()
        controller.register_worker(0, initial_clock=100)

        controller.update_clock(0, 110)

        stats = controller.get_stats()
        assert stats["max_clock"] == 110


class TestStalenessCalculation:
    """Tests for staleness calculation."""

    def test_get_staleness(self):
        """Test getting staleness."""
        controller = StalenessController()
        controller.register_worker(0, initial_clock=100)
        controller.register_worker(1, initial_clock=95)

        staleness = controller.get_staleness(1)

        assert staleness == 5  # 100 - 95

    def test_staleness_unregistered(self):
        """Test staleness for unregistered worker."""
        controller = StalenessController()

        staleness = controller.get_staleness(99)

        assert staleness == 0

    def test_get_slowest_worker(self):
        """Test getting slowest worker."""
        controller = StalenessController()
        controller.register_worker(0, initial_clock=100)
        controller.register_worker(1, initial_clock=95)
        controller.register_worker(2, initial_clock=105)

        slowest = controller.get_slowest_worker()

        assert slowest == 1

    def test_get_fastest_worker(self):
        """Test getting fastest worker."""
        controller = StalenessController()
        controller.register_worker(0, initial_clock=100)
        controller.register_worker(1, initial_clock=95)
        controller.register_worker(2, initial_clock=105)

        fastest = controller.get_fastest_worker()

        assert fastest == 2


class TestStalenessThreshold:
    """Tests for staleness threshold."""

    def test_is_too_stale(self):
        """Test checking if worker is too stale."""
        controller = StalenessController(max_staleness=5)
        controller.register_worker(0, initial_clock=100)
        controller.register_worker(1, initial_clock=90)

        assert controller.is_too_stale(1) is True
        assert controller.is_too_stale(0) is False

    def test_can_proceed(self):
        """Test checking if worker can proceed."""
        controller = StalenessController(max_staleness=5)
        controller.register_worker(0, initial_clock=100)
        controller.register_worker(1, initial_clock=96)

        assert controller.can_proceed(1) is True

    def test_set_threshold(self):
        """Test setting threshold."""
        controller = StalenessController(max_staleness=10, min_staleness=2)

        controller.set_threshold(5)
        assert controller.get_threshold() == 5

        # Should clamp to max
        controller.set_threshold(15)
        assert controller.get_threshold() == 10

        # Should clamp to min
        controller.set_threshold(1)
        assert controller.get_threshold() == 2


class TestStalenessStats:
    """Tests for staleness statistics."""

    def test_get_stats_empty(self):
        """Test stats with no workers."""
        controller = StalenessController()

        stats = controller.get_stats()

        assert stats["num_workers"] == 0
        assert stats["max_staleness"] == 0

    def test_get_stats(self):
        """Test getting statistics."""
        controller = StalenessController()
        controller.register_worker(0, initial_clock=100)
        controller.register_worker(1, initial_clock=95)

        stats = controller.get_stats()

        assert stats["num_workers"] == 2
        assert stats["min_clock"] == 95
        assert stats["max_clock"] == 100
        assert stats["clock_spread"] == 5


class TestAdaptiveSSPInit:
    """Tests for AdaptiveSSP initialization."""

    def test_create_default(self):
        """Test default creation."""
        controller = AdaptiveSSP()
        assert controller.target_staleness == 2.0
        assert controller.adapt_rate == 0.1

    def test_create_custom(self):
        """Test custom creation."""
        controller = AdaptiveSSP(
            max_staleness=20,
            target_staleness=5.0,
            adapt_rate=0.2,
            check_interval=50,
        )
        assert controller.max_staleness == 20
        assert controller.target_staleness == 5.0
        assert controller.adapt_rate == 0.2
        assert controller.check_interval == 50


class TestAdaptiveSSPRecording:
    """Tests for recording steps."""

    def test_record_step(self):
        """Test recording a step."""
        controller = AdaptiveSSP()
        controller.register_worker(0)

        controller.record_step(0, loss=0.5)

        stats = controller.get_stats()
        assert stats["step_count"] == 1

    def test_record_multiple_steps(self):
        """Test recording multiple steps."""
        controller = AdaptiveSSP()
        controller.register_worker(0)

        for i in range(10):
            controller.record_step(0, loss=0.5 - i * 0.01)

        stats = controller.get_stats()
        assert stats["step_count"] == 10


class TestAdaptiveThreshold:
    """Tests for adaptive threshold adjustment."""

    def test_threshold_stable(self):
        """Test threshold stays stable when staleness is at target."""
        controller = AdaptiveSSP(
            max_staleness=10,
            target_staleness=2.0,
            check_interval=10,
        )
        controller.register_worker(0, initial_clock=100)
        controller.register_worker(1, initial_clock=98)

        initial_threshold = controller.get_threshold()

        # Record steps (staleness ~ 2 which is at target)
        for i in range(10):
            controller.record_step(0)

        # Threshold should be stable
        assert abs(controller.get_threshold() - initial_threshold) <= 1

    def test_threshold_adapts_to_high_staleness(self):
        """Test threshold decreases when staleness is too high."""
        controller = AdaptiveSSP(
            max_staleness=10,
            min_staleness=1,
            target_staleness=2.0,
            check_interval=10,
        )
        controller.register_worker(0, initial_clock=100)
        controller.register_worker(1, initial_clock=90)  # Very stale

        initial_threshold = controller.get_threshold()

        # Record steps (high staleness)
        for i in range(20):
            controller.record_step(1)  # Record from slow worker

        # Threshold should decrease
        assert controller.get_threshold() <= initial_threshold


class TestAdaptiveSSPStats:
    """Tests for AdaptiveSSP statistics."""

    def test_get_stats(self):
        """Test getting adaptive stats."""
        controller = AdaptiveSSP(target_staleness=3.0)
        controller.register_worker(0)

        for i in range(5):
            controller.record_step(0, loss=0.5)

        stats = controller.get_stats()

        assert "target_staleness" in stats
        assert stats["target_staleness"] == 3.0
        assert "average_staleness" in stats
        assert stats["step_count"] == 5
