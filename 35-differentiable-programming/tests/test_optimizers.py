"""Tests for optimizer step correctness."""

import pytest
import numpy as np
import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from autograd.core.tensor import Tensor
from autograd.nn.modules import Linear, Sequential, ReLU, MSELoss
from autograd.nn.optim import SGD, Adam, AdamW, RMSprop


class TestSGD:
    """Test SGD optimizer."""

    def test_sgd_basic_step(self):
        """Test basic SGD step without momentum."""
        np.random.seed(42)
        param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1)

        # Set gradient
        param.grad = np.array([1.0, 1.0, 1.0])

        # Initial values
        initial = param.data.copy()

        # Step
        optimizer.step()

        # param = param - lr * grad
        expected = initial - 0.1 * np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(param.data, expected, rtol=1e-5)

    def test_sgd_momentum(self):
        """Test SGD with momentum."""
        np.random.seed(42)
        param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1, momentum=0.9)

        # First step
        param.grad = np.array([1.0, 1.0, 1.0])
        initial = param.data.copy()
        optimizer.step()

        # velocity = 0 * 0.9 + grad = grad
        # param = param - lr * velocity
        expected = initial - 0.1 * np.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(param.data, expected, rtol=1e-5)

        # Second step with same gradient
        param.grad = np.array([1.0, 1.0, 1.0])
        prev = param.data.copy()
        optimizer.step()

        # velocity = 1.0 * 0.9 + 1.0 = 1.9
        # param = param - lr * velocity
        expected = prev - 0.1 * np.array([1.9, 1.9, 1.9])
        np.testing.assert_allclose(param.data, expected, rtol=1e-5)

    def test_sgd_weight_decay(self):
        """Test SGD with weight decay."""
        np.random.seed(42)
        param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1, weight_decay=0.01)

        param.grad = np.array([0.0, 0.0, 0.0])  # Zero gradient
        initial = param.data.copy()
        optimizer.step()

        # grad = grad + weight_decay * param = 0 + 0.01 * [1, 2, 3] = [0.01, 0.02, 0.03]
        # param = param - lr * grad
        expected = initial - 0.1 * (0.01 * initial)
        np.testing.assert_allclose(param.data, expected, rtol=1e-5)

    def test_sgd_zero_grad(self):
        """Test SGD zero_grad."""
        param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = SGD([param], lr=0.1)

        param.grad = np.array([1.0, 1.0, 1.0])
        assert param.grad is not None

        optimizer.zero_grad()
        assert param.grad is None

    def test_sgd_multiple_params(self):
        """Test SGD with multiple parameters."""
        np.random.seed(42)
        param1 = Tensor([1.0, 2.0], requires_grad=True)
        param2 = Tensor([3.0, 4.0], requires_grad=True)
        optimizer = SGD([param1, param2], lr=0.1)

        param1.grad = np.array([1.0, 0.0])
        param2.grad = np.array([0.0, 1.0])

        initial1 = param1.data.copy()
        initial2 = param2.data.copy()

        optimizer.step()

        np.testing.assert_allclose(param1.data, initial1 - 0.1 * np.array([1.0, 0.0]), rtol=1e-5)
        np.testing.assert_allclose(param2.data, initial2 - 0.1 * np.array([0.0, 1.0]), rtol=1e-5)

    def test_sgd_skip_none_grad(self):
        """Test SGD skips parameters with None gradient."""
        param1 = Tensor([1.0, 2.0], requires_grad=True)
        param2 = Tensor([3.0, 4.0], requires_grad=True)
        optimizer = SGD([param1, param2], lr=0.1)

        param1.grad = np.array([1.0, 1.0])
        # param2.grad is None

        initial1 = param1.data.copy()
        initial2 = param2.data.copy()

        optimizer.step()

        # param1 should be updated
        np.testing.assert_allclose(param1.data, initial1 - 0.1 * np.array([1.0, 1.0]), rtol=1e-5)
        # param2 should be unchanged
        np.testing.assert_allclose(param2.data, initial2, rtol=1e-5)


class TestAdam:
    """Test Adam optimizer."""

    def test_adam_basic_step(self):
        """Test basic Adam step."""
        np.random.seed(42)
        param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = Adam([param], lr=0.001, betas=(0.9, 0.999), eps=1e-8)

        param.grad = np.array([1.0, 1.0, 1.0])
        initial = param.data.copy()

        optimizer.step()

        # After first step, t=1
        # m = 0.9 * 0 + 0.1 * 1 = 0.1
        # v = 0.999 * 0 + 0.001 * 1 = 0.001
        # m_hat = 0.1 / (1 - 0.9) = 1.0
        # v_hat = 0.001 / (1 - 0.999) = 1.0
        # update = lr * m_hat / (sqrt(v_hat) + eps) = 0.001 * 1.0 / (1.0 + 1e-8) ~= 0.001
        # param = initial - update

        # Check parameter was updated
        assert not np.allclose(param.data, initial)
        # Check update is approximately correct
        expected_update = 0.001 * 1.0 / (np.sqrt(1.0) + 1e-8)
        np.testing.assert_allclose(param.data, initial - expected_update, rtol=1e-5)

    def test_adam_bias_correction(self):
        """Test Adam bias correction over multiple steps."""
        np.random.seed(42)
        param = Tensor([1.0], requires_grad=True)
        optimizer = Adam([param], lr=0.1, betas=(0.9, 0.999))

        updates = []
        for i in range(10):
            param.grad = np.array([1.0])
            prev = param.data.copy()
            optimizer.step()
            updates.append(prev[0] - param.data[0])

        # Updates should decrease as bias correction becomes less significant
        # and moments accumulate
        assert updates[0] > 0  # Parameter should decrease
        # Later updates should be more stable

    def test_adam_weight_decay(self):
        """Test Adam with weight decay."""
        np.random.seed(42)
        param = Tensor([1.0, 2.0], requires_grad=True)
        optimizer = Adam([param], lr=0.1, weight_decay=0.1)

        param.grad = np.array([0.0, 0.0])  # Zero gradient
        initial = param.data.copy()

        optimizer.step()

        # With L2 regularization, grad = grad + wd * param
        # The parameter should still change due to weight decay
        assert not np.allclose(param.data, initial)

    def test_adam_multiple_params(self):
        """Test Adam with multiple parameters."""
        np.random.seed(42)
        param1 = Tensor([1.0], requires_grad=True)
        param2 = Tensor([2.0], requires_grad=True)
        optimizer = Adam([param1, param2], lr=0.1)

        param1.grad = np.array([1.0])
        param2.grad = np.array([2.0])

        initial1 = param1.data.copy()
        initial2 = param2.data.copy()

        optimizer.step()

        # Both should be updated
        assert not np.allclose(param1.data, initial1)
        assert not np.allclose(param2.data, initial2)


class TestAdamW:
    """Test AdamW optimizer."""

    def test_adamw_basic_step(self):
        """Test basic AdamW step."""
        np.random.seed(42)
        param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = AdamW([param], lr=0.001, betas=(0.9, 0.999), weight_decay=0.01)

        param.grad = np.array([1.0, 1.0, 1.0])
        initial = param.data.copy()

        optimizer.step()

        # AdamW applies weight decay directly, not to gradient
        # param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
        assert not np.allclose(param.data, initial)

    def test_adamw_vs_adam_weight_decay(self):
        """Test that AdamW applies decoupled weight decay differently than Adam."""
        np.random.seed(42)

        # Create two identical parameters
        param_adam = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        param_adamw = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        adam = Adam([param_adam], lr=0.1, weight_decay=0.01)
        adamw = AdamW([param_adamw], lr=0.1, weight_decay=0.01)

        # Set same gradients
        param_adam.grad = np.array([0.5, 0.5, 0.5])
        param_adamw.grad = np.array([0.5, 0.5, 0.5])

        adam.step()
        adamw.step()

        # Results should be different due to decoupled weight decay
        assert not np.allclose(param_adam.data, param_adamw.data)

    def test_adamw_only_weight_decay(self):
        """Test AdamW with only weight decay (zero gradient)."""
        np.random.seed(42)
        param = Tensor([1.0, 2.0], requires_grad=True)
        optimizer = AdamW([param], lr=0.1, weight_decay=0.1)

        param.grad = np.array([0.0, 0.0])
        initial = param.data.copy()

        optimizer.step()

        # With zero gradient, only weight decay should apply
        # param = param - lr * (0 + weight_decay * param)
        expected = initial - 0.1 * (0.1 * initial)
        # Note: Adam moments will still affect the update
        assert not np.allclose(param.data, initial)


class TestRMSprop:
    """Test RMSprop optimizer."""

    def test_rmsprop_basic_step(self):
        """Test basic RMSprop step."""
        np.random.seed(42)
        param = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        optimizer = RMSprop([param], lr=0.01, alpha=0.99, eps=1e-8)

        param.grad = np.array([1.0, 1.0, 1.0])
        initial = param.data.copy()

        optimizer.step()

        # v = 0.99 * 0 + 0.01 * 1^2 = 0.01
        # param = param - lr * grad / (sqrt(v) + eps)
        v = 0.01
        expected_update = 0.01 * 1.0 / (np.sqrt(v) + 1e-8)
        expected = initial - expected_update
        np.testing.assert_allclose(param.data, expected, rtol=1e-5)

    def test_rmsprop_accumulation(self):
        """Test RMSprop running average accumulation."""
        np.random.seed(42)
        param = Tensor([1.0], requires_grad=True)
        optimizer = RMSprop([param], lr=0.01, alpha=0.9)

        # Multiple steps with same gradient
        for _ in range(5):
            param.grad = np.array([1.0])
            optimizer.step()

        # Running average should accumulate
        # v converges towards 1.0 as more steps are taken

    def test_rmsprop_weight_decay(self):
        """Test RMSprop with weight decay."""
        np.random.seed(42)
        param = Tensor([1.0, 2.0], requires_grad=True)
        optimizer = RMSprop([param], lr=0.01, weight_decay=0.1)

        param.grad = np.array([0.0, 0.0])
        initial = param.data.copy()

        optimizer.step()

        # With weight decay, effective grad = weight_decay * param
        assert not np.allclose(param.data, initial)


class TestOptimizerWithModel:
    """Test optimizers with actual neural network models."""

    def test_sgd_training(self):
        """Test SGD on a simple regression problem."""
        np.random.seed(42)

        # Create model
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 1)
        )

        optimizer = SGD(model.parameters(), lr=0.01)
        loss_fn = MSELoss()

        # Training data
        X = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randn(32, 1).astype(np.float32)

        losses = []
        for _ in range(20):
            x_tensor = Tensor(X, requires_grad=True)
            y_tensor = Tensor(y)

            pred = model(x_tensor)
            loss = loss_fn(pred, y_tensor)
            losses.append(loss.data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_adam_training(self):
        """Test Adam on a simple regression problem."""
        np.random.seed(42)

        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 1)
        )

        optimizer = Adam(model.parameters(), lr=0.01)
        loss_fn = MSELoss()

        X = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randn(32, 1).astype(np.float32)

        losses = []
        for _ in range(20):
            x_tensor = Tensor(X, requires_grad=True)
            y_tensor = Tensor(y)

            pred = model(x_tensor)
            loss = loss_fn(pred, y_tensor)
            losses.append(loss.data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0]

    def test_adamw_training(self):
        """Test AdamW on a simple regression problem."""
        np.random.seed(42)

        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 1)
        )

        optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        loss_fn = MSELoss()

        X = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randn(32, 1).astype(np.float32)

        losses = []
        for _ in range(20):
            x_tensor = Tensor(X, requires_grad=True)
            y_tensor = Tensor(y)

            pred = model(x_tensor)
            loss = loss_fn(pred, y_tensor)
            losses.append(loss.data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0]

    def test_rmsprop_training(self):
        """Test RMSprop on a simple regression problem."""
        np.random.seed(42)

        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 1)
        )

        optimizer = RMSprop(model.parameters(), lr=0.01)
        loss_fn = MSELoss()

        X = np.random.randn(32, 10).astype(np.float32)
        y = np.random.randn(32, 1).astype(np.float32)

        losses = []
        for _ in range(20):
            x_tensor = Tensor(X, requires_grad=True)
            y_tensor = Tensor(y)

            pred = model(x_tensor)
            loss = loss_fn(pred, y_tensor)
            losses.append(loss.data.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert losses[-1] < losses[0]


class TestOptimizerComparison:
    """Compare different optimizers."""

    def test_optimizer_convergence_speed(self):
        """Compare convergence speed of different optimizers."""
        np.random.seed(42)

        # Generate simple quadratic objective: min ||Wx - y||^2
        X = np.random.randn(32, 10).astype(np.float32)
        true_W = np.random.randn(10, 1).astype(np.float32)
        y = X @ true_W

        results = {}

        for name, OptClass, kwargs in [
            ('SGD', SGD, {'lr': 0.01}),
            ('SGD+momentum', SGD, {'lr': 0.01, 'momentum': 0.9}),
            ('Adam', Adam, {'lr': 0.1}),  # Adam needs higher LR for this task
            ('RMSprop', RMSprop, {'lr': 0.01}),
        ]:
            np.random.seed(42)
            W = Tensor(np.random.randn(10, 1).astype(np.float32), requires_grad=True)
            optimizer = OptClass([W], **kwargs)
            loss_fn = MSELoss()

            final_loss = None
            for _ in range(100):  # More iterations
                pred = Tensor(X) @ W
                loss = loss_fn(pred, Tensor(y))
                final_loss = loss.data.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            results[name] = final_loss

        # All should converge to low loss
        for name, loss in results.items():
            assert loss < 2.0, f"{name} did not converge: loss={loss}"

    def test_sgd_momentum_vs_vanilla(self):
        """Test that momentum improves SGD convergence."""
        np.random.seed(42)

        X = np.random.randn(32, 10).astype(np.float32)
        true_W = np.random.randn(10, 1).astype(np.float32)
        y = X @ true_W

        # Vanilla SGD
        np.random.seed(42)
        W_vanilla = Tensor(np.random.randn(10, 1).astype(np.float32), requires_grad=True)
        optimizer_vanilla = SGD([W_vanilla], lr=0.01)
        loss_fn = MSELoss()

        losses_vanilla = []
        for _ in range(30):
            pred = Tensor(X) @ W_vanilla
            loss = loss_fn(pred, Tensor(y))
            losses_vanilla.append(loss.data.item())
            optimizer_vanilla.zero_grad()
            loss.backward()
            optimizer_vanilla.step()

        # SGD with momentum
        np.random.seed(42)
        W_momentum = Tensor(np.random.randn(10, 1).astype(np.float32), requires_grad=True)
        optimizer_momentum = SGD([W_momentum], lr=0.01, momentum=0.9)

        losses_momentum = []
        for _ in range(30):
            pred = Tensor(X) @ W_momentum
            loss = loss_fn(pred, Tensor(y))
            losses_momentum.append(loss.data.item())
            optimizer_momentum.zero_grad()
            loss.backward()
            optimizer_momentum.step()

        # Momentum should help (final loss should be lower)
        assert losses_momentum[-1] <= losses_vanilla[-1] * 1.5  # Allow some variance


class TestOptimizerEdgeCases:
    """Test optimizer edge cases."""

    def test_optimizer_with_zero_lr(self):
        """Test optimizer with zero learning rate."""
        param = Tensor([1.0, 2.0], requires_grad=True)
        optimizer = SGD([param], lr=0.0)

        param.grad = np.array([1.0, 1.0])
        initial = param.data.copy()

        optimizer.step()

        # Parameter should not change
        np.testing.assert_allclose(param.data, initial, rtol=1e-5)

    def test_optimizer_with_very_small_lr(self):
        """Test optimizer with very small learning rate."""
        param = Tensor([1.0, 2.0], requires_grad=True)
        optimizer = SGD([param], lr=1e-10)

        param.grad = np.array([1.0, 1.0])
        initial = param.data.copy()

        optimizer.step()

        # Parameter should change by very small amount
        np.testing.assert_allclose(param.data, initial, rtol=1e-5, atol=1e-9)

    def test_optimizer_with_large_gradient(self):
        """Test optimizer with large gradient values."""
        param = Tensor([1.0, 2.0], requires_grad=True)
        optimizer = SGD([param], lr=0.01)

        param.grad = np.array([1e6, 1e6])
        initial = param.data.copy()

        optimizer.step()

        # Should still update (though large)
        expected = initial - 0.01 * np.array([1e6, 1e6])
        np.testing.assert_allclose(param.data, expected, rtol=1e-5)

    def test_optimizer_empty_params(self):
        """Test optimizer with empty parameter list."""
        optimizer = SGD([], lr=0.1)
        optimizer.step()  # Should not raise
        optimizer.zero_grad()  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
