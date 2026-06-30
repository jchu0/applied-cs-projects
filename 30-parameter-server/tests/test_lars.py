"""Tests for LARS optimizer."""

import pytest
import numpy as np

from paramserver.optimizer.lars import LARSEngine


class TestLARSInit:
    """Tests for LARS initialization."""

    def test_create_default(self):
        """Test LARS creation with defaults."""
        lars = LARSEngine()
        assert lars.lr == 0.1
        assert lars.momentum == 0.9
        assert lars.weight_decay == 0.0001
        assert lars.trust_coefficient == 0.001

    def test_create_custom(self):
        """Test LARS with custom parameters."""
        lars = LARSEngine(
            lr=0.5,
            momentum=0.95,
            weight_decay=0.01,
            trust_coefficient=0.01,
            eps=1e-6,
        )
        assert lars.lr == 0.5
        assert lars.momentum == 0.95
        assert lars.weight_decay == 0.01
        assert lars.trust_coefficient == 0.01
        assert lars.eps == 1e-6

    def test_invalid_lr(self):
        """Test invalid learning rate."""
        with pytest.raises(ValueError, match="Invalid learning rate"):
            LARSEngine(lr=-0.1)

    def test_invalid_momentum(self):
        """Test invalid momentum."""
        with pytest.raises(ValueError, match="Invalid momentum"):
            LARSEngine(momentum=-0.1)

    def test_invalid_weight_decay(self):
        """Test invalid weight decay."""
        with pytest.raises(ValueError, match="Invalid weight_decay"):
            LARSEngine(weight_decay=-0.1)

    def test_invalid_trust_coefficient(self):
        """Test invalid trust coefficient."""
        with pytest.raises(ValueError, match="Invalid trust_coefficient"):
            LARSEngine(trust_coefficient=-0.1)

    def test_name(self):
        """Test name property."""
        lars = LARSEngine()
        assert lars.name == "LARSEngine"


class TestLARSApply:
    """Tests for LARS apply method."""

    def test_basic_update(self):
        """Test basic LARS update."""
        lars = LARSEngine(lr=0.1, momentum=0, weight_decay=0)
        params = np.array([1.0, 2.0, 3.0])
        grads = np.array([0.1, 0.2, 0.3])

        new_params = lars.apply(params, grads, param_id="p1")

        # Params should have decreased
        assert all(new_params < params)

    def test_layer_wise_scaling(self):
        """Test that LARS scales updates based on layer norms."""
        lars1 = LARSEngine(lr=0.1, momentum=0, weight_decay=0)
        lars2 = LARSEngine(lr=0.1, momentum=0, weight_decay=0)

        # Large params, small grad
        large_params = np.ones(100) * 10.0
        small_grad = np.ones(100) * 0.01

        # Small params, large grad
        small_params = np.ones(100) * 0.1
        large_grad = np.ones(100) * 1.0

        new_large = lars1.apply(large_params, small_grad, param_id="large")
        new_small = lars2.apply(small_params, large_grad, param_id="small")

        # Local LR should compensate for the difference
        local_lr_large = lars1.compute_local_lr(large_params, small_grad)
        local_lr_small = lars2.compute_local_lr(small_params, large_grad)

        # Large params / small grad should have larger local LR
        assert local_lr_large > local_lr_small

    def test_momentum(self):
        """Test LARS with momentum."""
        lars = LARSEngine(lr=0.1, momentum=0.9, weight_decay=0)
        params = np.array([1.0, 2.0, 3.0])
        grads = np.array([0.1, 0.2, 0.3])

        # First update
        params = lars.apply(params, grads, param_id="p1")

        # Second update should have momentum from first
        params2 = lars.apply(params, grads, param_id="p1")

        # Velocity should exist
        assert "p1" in lars._velocity

    def test_weight_decay(self):
        """Test LARS with weight decay."""
        lars = LARSEngine(lr=0.1, momentum=0, weight_decay=0.1)
        params = np.array([1.0, 2.0, 3.0])
        grads = np.zeros(3)  # Zero gradients

        new_params = lars.apply(params, grads, param_id="p1")

        # Params should decrease due to weight decay
        assert all(new_params < params)

    def test_zero_norms(self):
        """Test LARS with zero norms."""
        lars = LARSEngine(lr=0.1, momentum=0, weight_decay=0)

        # Zero params
        zero_params = np.zeros(3)
        grads = np.array([0.1, 0.2, 0.3])
        new_params = lars.apply(zero_params, grads, param_id="p1")
        # Should use local_lr = 1.0
        assert new_params is not None

        # Zero grads
        params = np.array([1.0, 2.0, 3.0])
        zero_grads = np.zeros(3)
        new_params = lars.apply(params, zero_grads, param_id="p2")
        # Should use local_lr = 1.0
        assert new_params is not None


class TestLARSLocalLR:
    """Tests for local learning rate computation."""

    def test_compute_local_lr(self):
        """Test computing local learning rate."""
        lars = LARSEngine(trust_coefficient=0.001, weight_decay=0)

        params = np.ones(100) * 10.0  # norm = 100
        grads = np.ones(100) * 1.0    # norm = 10

        local_lr = lars.compute_local_lr(params, grads)

        # local_lr = trust_coeff * ||params|| / ||grads||
        # = 0.001 * 100 / 10 = 0.01
        assert abs(local_lr - 0.01) < 0.001

    def test_local_lr_with_weight_decay(self):
        """Test local LR includes weight decay in grad norm."""
        lars = LARSEngine(trust_coefficient=0.001, weight_decay=0.1)

        params = np.ones(100) * 10.0
        grads = np.ones(100) * 1.0

        local_lr = lars.compute_local_lr(params, grads)

        # grad_with_decay = grads + 0.1 * params = 1 + 1 = 2
        # grad_with_decay_norm = sqrt(100 * 4) = 20
        # local_lr = 0.001 * 100 / 20 = 0.005
        assert abs(local_lr - 0.005) < 0.001


class TestLARSState:
    """Tests for LARS state management."""

    def test_get_state(self):
        """Test getting optimizer state."""
        lars = LARSEngine(lr=0.1, momentum=0.9)
        params = np.array([1.0])
        grads = np.array([0.1])

        lars.apply(params, grads, param_id="p1")

        state = lars.get_state()

        assert "velocity" in state
        assert "p1" in state["velocity"]
        assert state["lr"] == 0.1
        assert state["momentum"] == 0.9

    def test_load_state(self):
        """Test loading optimizer state."""
        lars1 = LARSEngine(lr=0.1, momentum=0.9)
        params = np.array([1.0])
        grads = np.array([0.1])

        # Build up state
        for _ in range(5):
            lars1.apply(params, grads, param_id="p1")

        state = lars1.get_state()

        # Load into new optimizer
        lars2 = LARSEngine()
        lars2.load_state(state)

        assert lars2.lr == 0.1
        assert lars2.momentum == 0.9
        np.testing.assert_array_equal(
            lars2._velocity["p1"],
            lars1._velocity["p1"],
        )

    def test_reset(self):
        """Test resetting optimizer state."""
        lars = LARSEngine(momentum=0.9)
        params = np.array([1.0])
        grads = np.array([0.1])

        lars.apply(params, grads, param_id="p1")
        assert len(lars._velocity) > 0

        lars.reset()
        assert len(lars._velocity) == 0

    def test_set_lr(self):
        """Test updating learning rate."""
        lars = LARSEngine(lr=0.1)
        lars.set_lr(0.05)
        assert lars.lr == 0.05

        with pytest.raises(ValueError):
            lars.set_lr(-0.01)


class TestLARSLargeBatch:
    """Tests simulating large batch training scenarios."""

    def test_large_batch_stability(self):
        """Test that LARS provides stability with large updates."""
        lars = LARSEngine(lr=1.0, trust_coefficient=0.001)

        params = np.random.randn(1000)
        # Simulate large batch gradient (scaled up)
        grads = np.random.randn(1000) * 10.0

        # Update should be reasonable despite large gradients
        new_params = lars.apply(params, grads, param_id="p1")

        # Change should not be explosive
        change = np.linalg.norm(new_params - params)
        param_norm = np.linalg.norm(params)
        assert change < param_norm * 2  # Should not double

    def test_multiple_layers(self):
        """Test LARS with layers of different sizes."""
        lars = LARSEngine(lr=0.1, momentum=0)

        # Small layer
        small_params = np.random.randn(10)
        small_grads = np.random.randn(10)
        new_small = lars.apply(small_params, small_grads, param_id="small")

        # Large layer
        large_params = np.random.randn(1000)
        large_grads = np.random.randn(1000)
        new_large = lars.apply(large_params, large_grads, param_id="large")

        # Both should have reasonable updates
        assert new_small is not None
        assert new_large is not None
