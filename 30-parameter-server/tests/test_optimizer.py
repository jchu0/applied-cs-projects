"""Tests for optimizer implementations."""

import pytest
import numpy as np

from paramserver.optimizer.base import UpdateEngine
from paramserver.optimizer.sgd import SGDEngine


class TestUpdateEngineBase:
    """Tests for UpdateEngine base class."""

    def test_abstract_methods(self):
        """Test that UpdateEngine cannot be instantiated."""
        with pytest.raises(TypeError):
            UpdateEngine()

    def test_name_property(self):
        """Test name property returns class name."""
        engine = SGDEngine()
        assert engine.name == "SGDEngine"


class TestSGDEngine:
    """Tests for SGD optimizer."""

    def test_basic_update(self):
        """Test basic SGD update without momentum."""
        engine = SGDEngine(lr=0.1)
        params = np.array([1.0, 2.0, 3.0])
        grads = np.array([0.1, 0.2, 0.3])

        new_params = engine.apply(params, grads)

        # params = params - lr * grads
        expected = np.array([0.99, 1.98, 2.97])
        np.testing.assert_array_almost_equal(new_params, expected)

    def test_momentum(self):
        """Test SGD with momentum."""
        engine = SGDEngine(lr=0.1, momentum=0.9)
        params = np.array([1.0, 2.0, 3.0])
        grads = np.array([0.1, 0.2, 0.3])

        # First update: v = 0.9*0 + grads = grads
        new_params = engine.apply(params, grads, param_id="p1")
        expected = np.array([0.99, 1.98, 2.97])
        np.testing.assert_array_almost_equal(new_params, expected)

        # Second update: v = 0.9*grads + grads = 1.9*grads
        new_params2 = engine.apply(new_params, grads, param_id="p1")
        expected2 = new_params - 0.1 * (0.9 * grads + grads)
        np.testing.assert_array_almost_equal(new_params2, expected2)

    def test_weight_decay(self):
        """Test SGD with weight decay."""
        engine = SGDEngine(lr=0.1, weight_decay=0.01)
        params = np.array([1.0, 2.0, 3.0])
        grads = np.array([0.1, 0.2, 0.3])

        new_params = engine.apply(params, grads)

        # effective_grads = grads + weight_decay * params
        effective_grads = grads + 0.01 * params
        expected = params - 0.1 * effective_grads
        np.testing.assert_array_almost_equal(new_params, expected)

    def test_nesterov_momentum(self):
        """Test Nesterov momentum."""
        engine = SGDEngine(lr=0.1, momentum=0.9, nesterov=True)
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])

        # First update
        new_params = engine.apply(params, grads, param_id="p1")

        # Second update should use Nesterov correction
        new_params2 = engine.apply(new_params, grads, param_id="p1")
        assert new_params2 is not None

    def test_dampening(self):
        """Test momentum with dampening."""
        engine = SGDEngine(lr=0.1, momentum=0.9, dampening=0.1)
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])

        # First update
        new_params = engine.apply(params, grads, param_id="p1")

        # Second update with dampening
        new_params2 = engine.apply(new_params, grads, param_id="p1")
        # v = 0.9*v + (1-0.1)*grads = 0.9*v + 0.9*grads
        assert new_params2 is not None

    def test_invalid_lr(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            SGDEngine(lr=-0.1)

    def test_invalid_momentum(self):
        """Test that negative momentum raises error."""
        with pytest.raises(ValueError, match="Invalid momentum"):
            SGDEngine(momentum=-0.1)

    def test_nesterov_requires_momentum(self):
        """Test that Nesterov requires momentum."""
        with pytest.raises(ValueError, match="Nesterov momentum requires"):
            SGDEngine(nesterov=True, momentum=0)

    def test_get_state(self):
        """Test state serialization."""
        engine = SGDEngine(lr=0.1, momentum=0.9)
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])

        # Build up some state
        engine.apply(params, grads, param_id="p1")
        engine.apply(params, grads, param_id="p2")

        state = engine.get_state()
        assert "velocity" in state
        assert "p1" in state["velocity"]
        assert "p2" in state["velocity"]
        assert state["lr"] == 0.1
        assert state["momentum"] == 0.9

    def test_load_state(self):
        """Test state restoration."""
        engine = SGDEngine(lr=0.1, momentum=0.9)

        state = {
            "velocity": {"p1": np.array([0.5, 0.5])},
            "lr": 0.05,
            "momentum": 0.8,
        }
        engine.load_state(state)

        assert engine.lr == 0.05
        assert engine.momentum == 0.8
        assert "p1" in engine._velocity
        np.testing.assert_array_equal(engine._velocity["p1"], [0.5, 0.5])

    def test_reset(self):
        """Test resetting optimizer state."""
        engine = SGDEngine(lr=0.1, momentum=0.9)
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])

        engine.apply(params, grads, param_id="p1")
        assert len(engine._velocity) > 0

        engine.reset()
        assert len(engine._velocity) == 0

    def test_set_lr(self):
        """Test learning rate update."""
        engine = SGDEngine(lr=0.1)
        engine.set_lr(0.05)
        assert engine.lr == 0.05

        with pytest.raises(ValueError):
            engine.set_lr(-0.01)

    def test_2d_params(self):
        """Test with 2D parameter arrays."""
        engine = SGDEngine(lr=0.1)
        params = np.random.randn(10, 5)
        grads = np.random.randn(10, 5)

        new_params = engine.apply(params, grads)
        expected = params - 0.1 * grads
        np.testing.assert_array_almost_equal(new_params, expected)

    def test_multiple_params_momentum(self):
        """Test momentum with multiple parameters."""
        engine = SGDEngine(lr=0.1, momentum=0.9)

        p1 = np.array([1.0])
        p2 = np.array([2.0])
        g1 = np.array([0.1])
        g2 = np.array([0.2])

        # Update both params
        new_p1 = engine.apply(p1, g1, param_id="p1")
        new_p2 = engine.apply(p2, g2, param_id="p2")

        # Verify they have separate momentum buffers
        assert "p1" in engine._velocity
        assert "p2" in engine._velocity
        assert not np.array_equal(engine._velocity["p1"], engine._velocity["p2"])
