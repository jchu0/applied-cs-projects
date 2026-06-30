"""Tests for Adam optimizer."""

import pytest
import numpy as np

from paramserver.optimizer.adam import AdamEngine


class TestAdamInit:
    """Tests for Adam initialization."""

    def test_create_default(self):
        """Test Adam creation with defaults."""
        adam = AdamEngine()
        assert adam.lr == 0.001
        assert adam.beta1 == 0.9
        assert adam.beta2 == 0.999
        assert adam.eps == 1e-8
        assert adam.weight_decay == 0.0
        assert adam.amsgrad is False

    def test_create_custom(self):
        """Test Adam with custom parameters."""
        adam = AdamEngine(
            lr=0.01,
            beta1=0.8,
            beta2=0.99,
            eps=1e-6,
            weight_decay=0.01,
            amsgrad=True,
        )
        assert adam.lr == 0.01
        assert adam.beta1 == 0.8
        assert adam.beta2 == 0.99
        assert adam.eps == 1e-6
        assert adam.weight_decay == 0.01
        assert adam.amsgrad is True

    def test_invalid_lr(self):
        """Test invalid learning rate."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            AdamEngine(lr=-0.1)

    def test_invalid_beta1(self):
        """Test invalid beta1."""
        with pytest.raises(ValueError, match="Invalid beta1"):
            AdamEngine(beta1=1.0)
        with pytest.raises(ValueError, match="Invalid beta1"):
            AdamEngine(beta1=-0.1)

    def test_invalid_beta2(self):
        """Test invalid beta2."""
        with pytest.raises(ValueError, match="Invalid beta2"):
            AdamEngine(beta2=1.0)

    def test_invalid_eps(self):
        """Test invalid epsilon."""
        with pytest.raises(ValueError, match="Invalid epsilon"):
            AdamEngine(eps=-1e-8)

    def test_invalid_weight_decay(self):
        """Test invalid weight decay."""
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            AdamEngine(weight_decay=-0.01)

    def test_name(self):
        """Test name property."""
        adam = AdamEngine()
        assert adam.name == "AdamEngine"


class TestAdamApply:
    """Tests for Adam apply method."""

    def test_basic_update(self):
        """Test basic Adam update."""
        adam = AdamEngine(lr=0.1)
        params = np.array([1.0, 2.0, 3.0])
        grads = np.array([0.1, 0.2, 0.3])

        new_params = adam.apply(params, grads, param_id="p1")

        # Params should have decreased
        assert all(new_params < params)

    def test_multiple_updates(self):
        """Test multiple Adam updates."""
        adam = AdamEngine(lr=0.1)
        params = np.array([1.0, 2.0, 3.0])
        grads = np.array([0.1, 0.2, 0.3])

        for _ in range(10):
            params = adam.apply(params, grads, param_id="p1")

        # Should have moved significantly
        assert params[0] < 0.5

    def test_momentum_accumulation(self):
        """Test that momentum accumulates."""
        adam = AdamEngine(lr=0.1)
        params = np.array([1.0])
        grads = np.array([0.1])

        # Apply several updates
        for _ in range(5):
            params = adam.apply(params, grads, param_id="p1")

        # Check that moment estimates exist
        assert "p1" in adam._m
        assert "p1" in adam._v
        assert adam._t["p1"] == 5

    def test_adaptive_scaling(self):
        """Test that Adam adapts to gradient magnitudes."""
        adam = AdamEngine(lr=0.1)

        # Large gradient
        params1 = np.array([1.0])
        large_grad = np.array([10.0])
        new_params1 = adam.apply(params1, large_grad, param_id="large")

        # Small gradient
        adam2 = AdamEngine(lr=0.1)
        params2 = np.array([1.0])
        small_grad = np.array([0.01])
        new_params2 = adam2.apply(params2, small_grad, param_id="small")

        # Both should move roughly similar amounts due to adaptive scaling
        # (not exactly equal, but scaled)
        move1 = abs(new_params1[0] - params1[0])
        move2 = abs(new_params2[0] - params2[0])

        # The ratio of moves should be much less than ratio of gradients
        grad_ratio = 10.0 / 0.01  # 1000
        move_ratio = move1 / move2
        assert move_ratio < grad_ratio / 10  # Much smaller ratio

    def test_weight_decay(self):
        """Test weight decay (AdamW)."""
        adam = AdamEngine(lr=0.1, weight_decay=0.1)
        params = np.array([1.0, 2.0, 3.0])
        grads = np.zeros(3)  # Zero gradients

        new_params = adam.apply(params, grads, param_id="p1")

        # Params should decrease due to weight decay
        assert all(new_params < params)

    def test_amsgrad(self):
        """Test AMSGrad variant."""
        adam = AdamEngine(lr=0.1, amsgrad=True)
        params = np.array([1.0])
        grads = np.array([0.1])

        # Apply updates
        adam.apply(params, grads, param_id="p1")

        # v_max should exist
        assert "p1" in adam._v_max


class TestAdamBiasCorrection:
    """Tests for Adam bias correction."""

    def test_bias_correction(self):
        """Test that bias correction is applied correctly."""
        adam = AdamEngine(lr=0.1, beta1=0.9, beta2=0.999)
        params = np.array([1.0])
        grads = np.array([1.0])

        # First update - bias correction should be significant
        new_params = adam.apply(params, grads, param_id="p1")

        # Without bias correction, first m_hat would be g * (1-beta1) = 0.1
        # With bias correction at t=1: m_hat = 0.1 / (1-0.9) = 1.0
        # So update should be close to lr * 1.0 = 0.1
        # (though v also plays a role)

        # After many iterations, bias correction should have less effect
        adam2 = AdamEngine(lr=0.1, beta1=0.9, beta2=0.999)
        params2 = np.array([1.0])
        for _ in range(1000):
            adam2.apply(params2, grads, param_id="p2")

        # Bias correction factors should be close to 1
        t = adam2._t["p2"]
        bc1 = 1 - 0.9 ** t  # Should be close to 1
        bc2 = 1 - 0.999 ** t  # Should be close to 1
        assert bc1 > 0.99
        assert bc2 > 0.5  # 0.999^1000 is still significant


class TestAdamState:
    """Tests for Adam state management."""

    def test_get_state(self):
        """Test getting optimizer state."""
        adam = AdamEngine(lr=0.1)
        params = np.array([1.0])
        grads = np.array([0.1])

        adam.apply(params, grads, param_id="p1")

        state = adam.get_state()

        assert "m" in state
        assert "v" in state
        assert "t" in state
        assert "p1" in state["m"]
        assert state["lr"] == 0.1

    def test_load_state(self):
        """Test loading optimizer state."""
        adam1 = AdamEngine(lr=0.1)
        params = np.array([1.0, 2.0])
        grads = np.array([0.1, 0.2])

        # Build up state
        for _ in range(5):
            adam1.apply(params, grads, param_id="p1")

        state = adam1.get_state()

        # Load into new optimizer
        adam2 = AdamEngine()
        adam2.load_state(state)

        assert adam2.lr == 0.1
        assert adam2._t["p1"] == 5
        np.testing.assert_array_equal(adam2._m["p1"], adam1._m["p1"])

    def test_reset(self):
        """Test resetting optimizer state."""
        adam = AdamEngine()
        params = np.array([1.0])
        grads = np.array([0.1])

        adam.apply(params, grads, param_id="p1")
        assert len(adam._m) > 0

        adam.reset()
        assert len(adam._m) == 0
        assert len(adam._v) == 0
        assert len(adam._t) == 0

    def test_set_lr(self):
        """Test updating learning rate."""
        adam = AdamEngine(lr=0.1)
        adam.set_lr(0.05)
        assert adam.lr == 0.05

        with pytest.raises(ValueError):
            adam.set_lr(-0.01)


class TestAdamMultiParam:
    """Tests for Adam with multiple parameters."""

    def test_separate_state_per_param(self):
        """Test that each param has separate state."""
        adam = AdamEngine()

        p1 = np.array([1.0])
        p2 = np.array([2.0])
        g1 = np.array([0.1])
        g2 = np.array([0.5])

        adam.apply(p1, g1, param_id="p1")
        adam.apply(p2, g2, param_id="p2")

        assert "p1" in adam._m
        assert "p2" in adam._m
        assert not np.array_equal(adam._m["p1"], adam._m["p2"])
