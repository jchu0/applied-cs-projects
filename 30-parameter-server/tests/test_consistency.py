"""Tests for consistency models."""

import pytest

from paramserver.consistency.base import ConsistencyModel
from paramserver.consistency.hogwild import HogwildConsistency


class TestConsistencyModelBase:
    """Tests for ConsistencyModel base class."""

    def test_abstract_methods(self):
        """Test that ConsistencyModel cannot be instantiated."""
        with pytest.raises(TypeError):
            ConsistencyModel()

    def test_name_property(self):
        """Test name property returns class name."""
        model = HogwildConsistency()
        assert model.name == "HogwildConsistency"


class TestHogwildConsistency:
    """Tests for Hogwild! consistency model."""

    def test_always_allows_update(self):
        """Test that Hogwild always allows updates."""
        model = HogwildConsistency()

        # Should always return True regardless of versions/clocks
        assert model.can_apply(param_version=0, worker_clock=0) is True
        assert model.can_apply(param_version=100, worker_clock=0) is True
        assert model.can_apply(param_version=0, worker_clock=100) is True
        assert model.can_apply(param_version=50, worker_clock=100) is True
        assert model.can_apply(param_version=100, worker_clock=50) is True

    def test_with_stale_worker(self):
        """Test that stale workers can still apply updates."""
        model = HogwildConsistency()

        # Worker is very behind
        assert model.can_apply(param_version=1000, worker_clock=5) is True

    def test_with_future_clock(self):
        """Test with worker clock ahead of param version."""
        model = HogwildConsistency()

        assert model.can_apply(param_version=5, worker_clock=1000) is True

    def test_name(self):
        """Test consistency model name."""
        model = HogwildConsistency()
        assert model.name == "HogwildConsistency"
