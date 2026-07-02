"""Tests for neural network modules."""

import pytest
import numpy as np
import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from autograd.core.tensor import Tensor, no_grad
from autograd.nn.modules import (
    Module,
    Linear,
    Conv2d,
    BatchNorm1d,
    LayerNorm,
    Dropout,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Sequential,
    MSELoss,
    CrossEntropyLoss,
)


# Numerical gradient helper
def numerical_gradient(f, x_data, eps=1e-5):
    """
    Compute numerical gradient using central difference.
    Uses float64 internally for precision.
    The function f should return a Tensor (not call .sum()).
    """
    x_data_64 = x_data.astype(np.float64)
    x_flat = x_data_64.flatten().copy()
    grad_flat = np.zeros_like(x_flat, dtype=np.float64)

    for i in range(len(x_flat)):
        orig_val = x_flat[i]

        # f(x + eps)
        x_flat[i] = orig_val + eps
        x_plus = Tensor(x_flat.reshape(x_data.shape).astype(np.float32))
        result_plus = f(x_plus)
        f_plus = float(result_plus.data.astype(np.float64).sum())

        # f(x - eps)
        x_flat[i] = orig_val - eps
        x_minus = Tensor(x_flat.reshape(x_data.shape).astype(np.float32))
        result_minus = f(x_minus)
        f_minus = float(result_minus.data.astype(np.float64).sum())

        # Restore
        x_flat[i] = orig_val

        # Central difference
        grad_flat[i] = (f_plus - f_minus) / (2 * eps)

    return grad_flat.reshape(x_data.shape).astype(x_data.dtype)


class TestLinear:
    """Test Linear layer."""

    def test_linear_forward(self):
        """Test Linear forward pass."""
        np.random.seed(42)
        layer = Linear(10, 5)
        x = Tensor.randn((4, 10))

        y = layer(x)
        assert y.shape == (4, 5)

    def test_linear_no_bias(self):
        """Test Linear without bias."""
        np.random.seed(42)
        layer = Linear(10, 5, bias=False)
        x = Tensor.randn((4, 10))

        assert layer.bias is None
        y = layer(x)
        assert y.shape == (4, 5)

    def test_linear_gradient(self):
        """Test Linear backward pass."""
        np.random.seed(42)
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert layer.weight.grad is not None
        assert layer.bias.grad is not None
        assert x.grad.shape == x.shape
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape

    def test_linear_numerical_gradient(self):
        """Test Linear with numerical gradient checking."""
        np.random.seed(42)
        layer = Linear(4, 3)
        x_data = np.random.randn(2, 4).astype(np.float32)
        x = Tensor(x_data.copy(), requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        # Numerical gradient for input - pass numpy array, function returns tensor
        def f(inp):
            return layer(inp)
        num_grad = numerical_gradient(f, x_data)
        np.testing.assert_allclose(x.grad, num_grad, rtol=0.05, atol=0.01)

    def test_linear_weight_initialization(self):
        """Test Xavier weight initialization."""
        np.random.seed(42)
        layer = Linear(100, 100)

        # Xavier init should have std close to sqrt(2/(in+out))
        expected_std = np.sqrt(2.0 / (100 + 100))
        actual_std = layer.weight.data.std()

        assert abs(actual_std - expected_std) < 0.1

    def test_linear_parameters(self):
        """Test Linear parameters iterator."""
        layer = Linear(10, 5)
        params = list(layer.parameters())

        assert len(params) == 2  # weight and bias
        assert params[0].shape == (10, 5)
        assert params[1].shape == (5,)


class TestConv2d:
    """Test Conv2d layer."""

    def test_conv2d_forward(self):
        """Test Conv2d forward pass."""
        np.random.seed(42)
        layer = Conv2d(3, 16, kernel_size=3, padding=1)
        x = Tensor.randn((2, 3, 8, 8))

        y = layer(x)
        assert y.shape == (2, 16, 8, 8)  # Same size with padding=1

    def test_conv2d_forward_no_padding(self):
        """Test Conv2d forward pass without padding."""
        np.random.seed(42)
        layer = Conv2d(3, 16, kernel_size=3, padding=0)
        x = Tensor.randn((2, 3, 8, 8))

        y = layer(x)
        assert y.shape == (2, 16, 6, 6)  # Reduced by 2 in each dimension

    def test_conv2d_stride(self):
        """Test Conv2d with stride."""
        np.random.seed(42)
        layer = Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        x = Tensor.randn((2, 3, 8, 8))

        y = layer(x)
        assert y.shape == (2, 16, 4, 4)  # Halved with stride=2

    def test_conv2d_parameters(self):
        """Test Conv2d parameters."""
        layer = Conv2d(3, 16, kernel_size=3)
        params = list(layer.parameters())

        assert len(params) == 2  # weight and bias
        assert params[0].shape == (16, 3, 3, 3)  # out, in, k, k
        assert params[1].shape == (16,)

    def test_conv2d_no_bias(self):
        """Test Conv2d without bias."""
        layer = Conv2d(3, 16, kernel_size=3, bias=False)
        params = list(layer.parameters())

        assert len(params) == 1  # only weight
        assert layer.bias is None

    def test_conv2d_gradient_exists(self):
        """Conv2d backward populates input and parameter gradients (non-zero)."""
        np.random.seed(0)
        layer = Conv2d(2, 3, kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(2, 2, 5, 5).astype(np.float32), requires_grad=True)

        y = layer(x)
        y.sum().backward()

        assert x.grad is not None and x.grad.shape == x.shape
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad is not None
        assert layer.bias.grad.shape == layer.bias.shape
        # The stub returned all zeros; a real backward must not.
        assert not np.allclose(x.grad, 0)
        assert not np.allclose(layer.weight.grad, 0)

    def test_conv2d_numerical_gradient_dx(self):
        """Finite-difference check of Conv2d input gradient (dx)."""
        np.random.seed(1)
        layer = Conv2d(2, 3, kernel_size=3, stride=1, padding=1)
        x_data = np.random.randn(2, 2, 5, 5).astype(np.float32)

        x = Tensor(x_data.copy(), requires_grad=True)
        y = layer(x)
        y.sum().backward()

        def f(inp):
            return layer(inp)

        num_grad = numerical_gradient(f, x_data)
        np.testing.assert_allclose(x.grad, num_grad, rtol=0.05, atol=0.02)

    def test_conv2d_numerical_gradient_dw(self):
        """Finite-difference check of Conv2d weight gradient (dW)."""
        np.random.seed(2)
        layer = Conv2d(2, 3, kernel_size=3, stride=1, padding=0)
        x_data = np.random.randn(1, 2, 5, 5).astype(np.float32)
        w_data = layer.weight.data.copy()

        x = Tensor(x_data.copy(), requires_grad=True)
        y = layer(x)
        y.sum().backward()
        analytic_dw = layer.weight.grad.copy()

        # Numerical gradient of sum(conv(x, W)) w.r.t. each weight element.
        def f_w(w):
            probe = Conv2d(2, 3, kernel_size=3, stride=1, padding=0)
            probe.weight.data = w.data
            if probe.bias is not None:
                probe.bias.data = layer.bias.data.copy()
            return probe(Tensor(x_data.copy()))

        num_dw = numerical_gradient(f_w, w_data)
        np.testing.assert_allclose(analytic_dw, num_dw, rtol=0.05, atol=0.02)

    def test_conv2d_numerical_gradient_db(self):
        """Finite-difference check of Conv2d bias gradient (db)."""
        np.random.seed(3)
        layer = Conv2d(2, 3, kernel_size=3, stride=2, padding=1)
        x_data = np.random.randn(2, 2, 6, 6).astype(np.float32)
        b_data = layer.bias.data.copy()

        x = Tensor(x_data.copy(), requires_grad=True)
        y = layer(x)
        y.sum().backward()
        analytic_db = layer.bias.grad.copy()

        def f_b(b):
            probe = Conv2d(2, 3, kernel_size=3, stride=2, padding=1)
            probe.weight.data = layer.weight.data.copy()
            probe.bias.data = b.data
            return probe(Tensor(x_data.copy()))

        num_db = numerical_gradient(f_b, b_data)
        np.testing.assert_allclose(analytic_db, num_db, rtol=0.05, atol=0.02)

    def test_conv2d_trainable(self):
        """Conv2d can be trained: loss decreases with SGD on its parameters."""
        np.random.seed(4)
        layer = Conv2d(1, 1, kernel_size=3, padding=1)
        x_data = np.random.randn(1, 1, 6, 6).astype(np.float32)
        target = np.random.randn(1, 1, 6, 6).astype(np.float32)
        loss_fn = MSELoss()

        losses = []
        for _ in range(30):
            x = Tensor(x_data.copy())
            layer.zero_grad()
            pred = layer(x)
            loss = loss_fn(pred, Tensor(target))
            loss.backward()
            for p in layer.parameters():
                if p.grad is not None:
                    p.data -= 0.05 * p.grad
            losses.append(loss.item())

        assert losses[-1] < losses[0]


class TestBatchNorm1d:
    """Test BatchNorm1d layer."""

    def test_batchnorm_forward_train(self):
        """Test BatchNorm1d forward pass in training mode."""
        np.random.seed(42)
        layer = BatchNorm1d(10)
        layer.train()

        x = Tensor(np.random.randn(32, 10).astype(np.float32) * 5 + 3, requires_grad=True)
        y = layer(x)

        # Output should be normalized (approximately)
        assert y.shape == x.shape
        np.testing.assert_allclose(y.data.mean(axis=0), 0, atol=0.1)
        np.testing.assert_allclose(y.data.std(axis=0), 1, atol=0.2)

    def test_batchnorm_forward_eval(self):
        """Test BatchNorm1d forward pass in eval mode."""
        np.random.seed(42)
        layer = BatchNorm1d(10)

        # First train to accumulate running stats
        layer.train()
        for _ in range(10):
            x = Tensor(np.random.randn(32, 10).astype(np.float32) * 5 + 3)
            _ = layer(x)

        # Then eval
        layer.eval()
        x = Tensor(np.random.randn(32, 10).astype(np.float32) * 5 + 3)
        y = layer(x)

        assert y.shape == x.shape

    def test_batchnorm_gradient(self):
        """Test BatchNorm1d backward pass."""
        np.random.seed(42)
        layer = BatchNorm1d(10)
        layer.train()

        x = Tensor(np.random.randn(32, 10).astype(np.float32), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.gamma.grad is not None
        assert layer.beta.grad is not None

    def test_batchnorm_parameters(self):
        """Test BatchNorm1d parameters."""
        layer = BatchNorm1d(10)
        params = list(layer.parameters())

        assert len(params) == 2  # gamma and beta


class TestLayerNorm:
    """Test LayerNorm layer."""

    def test_layernorm_forward(self):
        """Test LayerNorm forward pass."""
        np.random.seed(42)
        layer = LayerNorm(10)

        x = Tensor(np.random.randn(4, 10).astype(np.float32) * 5 + 3, requires_grad=True)
        y = layer(x)

        assert y.shape == x.shape
        # Each sample should be normalized
        np.testing.assert_allclose(y.data.mean(axis=-1), 0, atol=1e-5)
        np.testing.assert_allclose(y.data.std(axis=-1), 1, atol=0.2)

    def test_layernorm_gradient(self):
        """Test LayerNorm backward pass."""
        np.random.seed(42)
        layer = LayerNorm(10)

        x = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert layer.gamma.grad is not None
        assert layer.beta.grad is not None
        assert layer.gamma.grad.shape == layer.gamma.shape
        assert layer.beta.grad.shape == layer.beta.shape

    def test_layernorm_numerical_gradient(self):
        """Test LayerNorm input gradient with numerical gradient checking."""
        np.random.seed(42)
        layer = LayerNorm(6)
        # Non-trivial affine parameters so the mean/variance terms matter
        layer.gamma.data = np.random.randn(6).astype(np.float32)
        layer.beta.data = np.random.randn(6).astype(np.float32)

        x_data = np.random.randn(3, 6).astype(np.float32)
        x = Tensor(x_data.copy(), requires_grad=True)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        def f(inp):
            return layer(inp)
        num_grad = numerical_gradient(f, x_data)
        np.testing.assert_allclose(x.grad, num_grad, rtol=0.05, atol=0.01)

    def test_layernorm_parameter_numerical_gradient(self):
        """Test LayerNorm gamma/beta gradients with numerical gradient checking."""
        np.random.seed(42)
        x_data = np.random.randn(3, 6).astype(np.float32)

        layer = LayerNorm(6)
        x = Tensor(x_data.copy(), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.gamma.grad is not None
        assert layer.beta.grad is not None

        # Numerical gradient w.r.t. gamma
        def f_gamma(gamma):
            probe = LayerNorm(6)
            probe.gamma.data = gamma.data
            probe.beta.data = layer.beta.data.copy()
            return probe(Tensor(x_data.copy()))

        num_dgamma = numerical_gradient(f_gamma, layer.gamma.data)
        np.testing.assert_allclose(layer.gamma.grad, num_dgamma, rtol=0.05, atol=0.01)

        # Numerical gradient w.r.t. beta
        def f_beta(beta):
            probe = LayerNorm(6)
            probe.gamma.data = layer.gamma.data.copy()
            probe.beta.data = beta.data
            return probe(Tensor(x_data.copy()))

        num_dbeta = numerical_gradient(f_beta, layer.beta.data)
        np.testing.assert_allclose(layer.beta.grad, num_dbeta, rtol=0.05, atol=0.01)


class TestDropout:
    """Test Dropout layer."""

    def test_dropout_train(self):
        """Test Dropout in training mode."""
        np.random.seed(42)
        layer = Dropout(p=0.5)
        layer.train()

        x = Tensor.ones((100, 100))
        y = layer(x)

        # About 50% should be zero
        zero_frac = (y.data == 0).mean()
        assert 0.4 < zero_frac < 0.6

        # Non-zero elements should be scaled by 1/(1-p) = 2
        non_zero_vals = y.data[y.data != 0]
        np.testing.assert_allclose(non_zero_vals, 2.0, rtol=1e-5)

    def test_dropout_eval(self):
        """Test Dropout in eval mode - should be identity."""
        layer = Dropout(p=0.5)
        layer.eval()

        x = Tensor.ones((10, 10))
        y = layer(x)

        np.testing.assert_allclose(y.data, x.data, rtol=1e-5)

    def test_dropout_gradient(self):
        """Test Dropout gradient."""
        np.random.seed(42)
        layer = Dropout(p=0.5)
        layer.train()

        x = Tensor(np.random.randn(10, 10).astype(np.float32), requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        # Gradient should be scaled and masked same as forward
        assert (x.grad == 0).any()


class TestActivationModules:
    """Test activation function modules."""

    def test_relu_module(self):
        """Test ReLU module."""
        layer = ReLU()
        x = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
        y = layer(x)

        expected = np.array([0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(y.data, expected, rtol=1e-5)

        y.sum().backward()
        expected_grad = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(x.grad, expected_grad, rtol=1e-5)

    def test_sigmoid_module(self):
        """Test Sigmoid module."""
        layer = Sigmoid()
        x = Tensor([0.0], requires_grad=True)
        y = layer(x)

        np.testing.assert_allclose(y.data, [0.5], rtol=1e-5)

        y.backward()
        np.testing.assert_allclose(x.grad, [0.25], rtol=1e-5)

    def test_tanh_module(self):
        """Test Tanh module."""
        layer = Tanh()
        x = Tensor([0.0], requires_grad=True)
        y = layer(x)

        np.testing.assert_allclose(y.data, [0.0], rtol=1e-5)

        y.backward()
        np.testing.assert_allclose(x.grad, [1.0], rtol=1e-5)

    def test_softmax_module(self):
        """Test Softmax module."""
        layer = Softmax(axis=-1)
        x = Tensor([[1.0, 2.0, 3.0]])
        y = layer(x)

        # Should sum to 1
        np.testing.assert_allclose(y.data.sum(), 1.0, rtol=1e-5)


class TestSequential:
    """Test Sequential container."""

    def test_sequential_forward(self):
        """Test Sequential forward pass."""
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )

        x = Tensor.randn((4, 10))
        y = model(x)

        assert y.shape == (4, 5)

    def test_sequential_gradient(self):
        """Test Sequential backward pass."""
        np.random.seed(42)
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )

        x = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Check all parameters have gradients
        for param in model.parameters():
            assert param.grad is not None

    def test_sequential_parameters(self):
        """Test Sequential parameters."""
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )

        params = list(model.parameters())
        # 2 Linear layers with weight + bias each = 4 parameters
        assert len(params) == 4

    def test_sequential_train_eval(self):
        """Test Sequential train/eval mode propagation."""
        model = Sequential(
            Linear(10, 20),
            Dropout(0.5),
            Linear(20, 5)
        )

        model.train()
        assert model._training

        model.eval()
        assert not model._training

    def test_sequential_zero_grad(self):
        """Test Sequential zero_grad."""
        np.random.seed(42)
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 5)
        )

        # Forward and backward
        x = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        y = model(x)
        y.sum().backward()

        # Zero gradients
        model.zero_grad()

        for param in model.parameters():
            assert param.grad is None


class TestMSELoss:
    """Test MSE loss."""

    def test_mse_loss_forward(self):
        """Test MSE loss forward pass."""
        loss_fn = MSELoss()
        pred = Tensor([[1.0, 2.0], [3.0, 4.0]])
        target = Tensor([[1.0, 2.0], [3.0, 4.0]])

        loss = loss_fn(pred, target)
        np.testing.assert_allclose(loss.data, 0.0, rtol=1e-5)

    def test_mse_loss_value(self):
        """Test MSE loss computes correct value."""
        loss_fn = MSELoss()
        pred = Tensor([[1.0, 2.0]])
        target = Tensor([[0.0, 0.0]])

        loss = loss_fn(pred, target)
        # MSE = mean((1-0)^2 + (2-0)^2) = (1 + 4) / 2 = 2.5
        np.testing.assert_allclose(loss.data, 2.5, rtol=1e-5)

    def test_mse_loss_gradient(self):
        """Test MSE loss gradient."""
        loss_fn = MSELoss()
        pred = Tensor([[1.0, 2.0]], requires_grad=True)
        target = Tensor([[0.0, 0.0]])

        loss = loss_fn(pred, target)
        loss.backward()

        # d(MSE)/d(pred) = 2 * (pred - target) / n
        # = 2 * [1, 2] / 2 = [1, 2]
        expected_grad = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(pred.grad, expected_grad, rtol=1e-5)

    def test_mse_loss_numerical_gradient(self):
        """Test MSE loss with numerical gradient checking."""
        np.random.seed(42)
        loss_fn = MSELoss()
        pred_data = np.random.randn(4, 5).astype(np.float32)
        target_data = np.random.randn(4, 5).astype(np.float32)
        target = Tensor(target_data)

        pred = Tensor(pred_data.copy(), requires_grad=True)
        loss = loss_fn(pred, target)
        loss.backward()

        # MSE loss returns a scalar, so we can pass the loss function directly
        def f(x):
            return loss_fn(x, target)
        num_grad = numerical_gradient(f, pred_data)
        np.testing.assert_allclose(pred.grad, num_grad, rtol=0.1, atol=0.02)


class TestCrossEntropyLoss:
    """Test CrossEntropy loss."""

    def test_crossentropy_forward(self):
        """Test CrossEntropy forward pass."""
        loss_fn = CrossEntropyLoss()
        pred = Tensor([[10.0, 0.0, 0.0]])  # Strongly predicting class 0
        target = Tensor([0])  # Correct class

        loss = loss_fn(pred, target)
        assert loss.data < 0.1  # Should be very small

    def test_crossentropy_gradient(self):
        """Test CrossEntropy gradient."""
        np.random.seed(42)
        loss_fn = CrossEntropyLoss()

        pred = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
        target = Tensor(np.array([0, 1, 2, 0]))

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_crossentropy_onehot(self):
        """Test CrossEntropy with one-hot targets."""
        loss_fn = CrossEntropyLoss()

        pred = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        target = Tensor([[0.0, 0.0, 1.0]])  # One-hot

        loss = loss_fn(pred, target)
        loss.backward()

        assert pred.grad is not None

    def test_crossentropy_numerical_gradient(self):
        """Test CrossEntropy with numerical gradient checking."""
        np.random.seed(42)
        loss_fn = CrossEntropyLoss()

        pred_data = np.random.randn(4, 3).astype(np.float32)
        target = Tensor(np.array([0, 1, 2, 0]))

        pred = Tensor(pred_data.copy(), requires_grad=True)
        loss = loss_fn(pred, target)
        loss.backward()

        def f(x):
            return loss_fn(x, target)
        num_grad = numerical_gradient(f, pred_data)
        np.testing.assert_allclose(pred.grad, num_grad, rtol=0.05, atol=0.01)


class TestModuleBase:
    """Test Module base class."""

    def test_module_train_mode(self):
        """Test train mode setting."""
        layer = Linear(10, 5)

        layer.train()
        assert layer._training

        layer.eval()
        assert not layer._training

    def test_module_zero_grad(self):
        """Test zero_grad on module."""
        np.random.seed(42)
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)

        y = layer(x)
        y.sum().backward()

        assert layer.weight.grad is not None
        assert layer.bias.grad is not None

        layer.zero_grad()

        assert layer.weight.grad is None
        assert layer.bias.grad is None


class TestDeepNetwork:
    """Test deeper networks."""

    def test_deep_network_forward(self):
        """Test forward pass through deep network."""
        np.random.seed(42)
        model = Sequential(
            Linear(10, 32),
            ReLU(),
            Linear(32, 32),
            ReLU(),
            Linear(32, 16),
            ReLU(),
            Linear(16, 5)
        )

        x = Tensor.randn((4, 10))
        y = model(x)

        assert y.shape == (4, 5)

    def test_deep_network_gradient_flow(self):
        """Test gradient flow through deep network."""
        np.random.seed(42)
        model = Sequential(
            Linear(10, 32),
            ReLU(),
            Linear(32, 32),
            ReLU(),
            Linear(32, 16),
            ReLU(),
            Linear(16, 5)
        )

        x = Tensor(np.random.randn(4, 10).astype(np.float32), requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # All parameters should have gradients
        for i, param in enumerate(model.parameters()):
            assert param.grad is not None, f"Parameter {i} has no gradient"
            # Gradient should not be zero (with high probability)
            assert not np.allclose(param.grad, 0), f"Parameter {i} has zero gradient"

    def test_training_loop(self):
        """Test a simple training loop."""
        np.random.seed(42)

        # Create simple network
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 1)
        )

        # Create data
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)

        loss_fn = MSELoss()
        lr = 0.01

        initial_loss = None
        final_loss = None

        # Training loop
        for epoch in range(10):
            x_tensor = Tensor(X, requires_grad=True)
            y_tensor = Tensor(y)

            # Forward
            pred = model(x_tensor)
            loss = loss_fn(pred, y_tensor)

            if epoch == 0:
                initial_loss = loss.data.copy()

            if epoch == 9:
                final_loss = loss.data.copy()

            # Backward
            model.zero_grad()
            loss.backward()

            # Update (manual SGD)
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= lr * param.grad

        # Loss should decrease
        assert final_loss < initial_loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
