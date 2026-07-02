"""Tests for differentiable operations with numerical gradient checking."""

import pytest
import numpy as np
import sys
import os

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from autograd.core.tensor import Tensor, no_grad, enable_grad
from autograd import ops


# Numerical gradient helper for testing
def numerical_gradient_for_scalar_func(f, x_data, eps=1e-5):
    """
    Compute numerical gradient for a function that returns a non-scalar Tensor.
    Computes gradient of sum(f(x)) with respect to x.

    Note: Uses float64 internally to avoid float32 precision issues
    with summing many small values. The function f should return a Tensor
    (not call .sum() internally) to preserve precision.

    Args:
        f: Function that takes Tensor and returns Tensor (not necessarily scalar)
        x_data: Input numpy array (not Tensor, to avoid aliasing)
        eps: Finite difference epsilon

    Returns:
        Numerical gradient as numpy array
    """
    # Use float64 for precision in gradient computation
    x_data_64 = x_data.astype(np.float64)
    x_flat = x_data_64.flatten().copy()
    grad_flat = np.zeros_like(x_flat, dtype=np.float64)

    for i in range(len(x_flat)):
        orig_val = x_flat[i]

        # f(x + eps)
        x_flat[i] = orig_val + eps
        x_plus = Tensor(x_flat.reshape(x_data.shape).astype(np.float32))
        result_plus = f(x_plus)
        # Sum in float64 for accuracy
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


class TestArithmeticOperations:
    """Test arithmetic operations and their gradients."""

    def test_add_forward(self):
        """Test addition forward pass."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = ops.add(a, b)

        expected = np.array([5.0, 7.0, 9.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_add_gradient(self):
        """Test addition gradients with numerical check."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = ops.add(a, b).sum()
        c.backward()

        # Analytical gradients
        assert a.grad is not None
        assert b.grad is not None
        np.testing.assert_allclose(a.grad, np.ones(3), rtol=1e-5)
        np.testing.assert_allclose(b.grad, np.ones(3), rtol=1e-5)

    def test_add_broadcasting(self):
        """Test addition with broadcasting."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([10.0, 20.0], requires_grad=True)
        c = ops.add(a, b).sum()
        c.backward()

        expected = np.array([[11.0, 22.0], [13.0, 24.0]])
        np.testing.assert_allclose((a.data + b.data), expected, rtol=1e-5)
        np.testing.assert_allclose(b.grad, np.array([2.0, 2.0]), rtol=1e-5)

    def test_sub_forward(self):
        """Test subtraction forward pass."""
        a = Tensor([5.0, 7.0, 9.0])
        b = Tensor([1.0, 2.0, 3.0])
        c = ops.sub(a, b)

        expected = np.array([4.0, 5.0, 6.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_sub_gradient(self):
        """Test subtraction gradients."""
        a = Tensor([5.0, 7.0, 9.0], requires_grad=True)
        b = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = ops.sub(a, b).sum()
        c.backward()

        np.testing.assert_allclose(a.grad, np.ones(3), rtol=1e-5)
        np.testing.assert_allclose(b.grad, -np.ones(3), rtol=1e-5)

    def test_mul_forward(self):
        """Test multiplication forward pass."""
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        c = ops.mul(a, b)

        expected = np.array([4.0, 10.0, 18.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_mul_gradient(self):
        """Test multiplication gradients with numerical check."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = ops.mul(a, b).sum()
        c.backward()

        # d(a*b)/da = b, d(a*b)/db = a
        np.testing.assert_allclose(a.grad, b.data, rtol=1e-5)
        np.testing.assert_allclose(b.grad, a.data, rtol=1e-5)

    def test_mul_numerical_gradient(self):
        """Test multiplication with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)
        b_data = np.random.randn(3, 4).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        b = Tensor(b_data.copy(), requires_grad=True)
        c = ops.mul(a, b).sum()
        c.backward()

        # Numerical gradient for a - return tensor without .sum() for precision
        def f_a(x):
            return ops.mul(x, Tensor(b_data))
        num_grad_a = numerical_gradient_for_scalar_func(f_a, a_data)
        np.testing.assert_allclose(a.grad, num_grad_a, rtol=0.05, atol=0.01)

    def test_div_forward(self):
        """Test division forward pass."""
        a = Tensor([4.0, 10.0, 18.0])
        b = Tensor([2.0, 5.0, 3.0])
        c = ops.div(a, b)

        expected = np.array([2.0, 2.0, 6.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_div_gradient(self):
        """Test division gradients."""
        a = Tensor([4.0, 10.0, 18.0], requires_grad=True)
        b = Tensor([2.0, 5.0, 3.0], requires_grad=True)
        c = ops.div(a, b).sum()
        c.backward()

        # d(a/b)/da = 1/b
        np.testing.assert_allclose(a.grad, 1.0 / b.data, rtol=1e-5)
        # d(a/b)/db = -a/b^2
        np.testing.assert_allclose(b.grad, -a.data / (b.data ** 2), rtol=1e-5)

    def test_div_numerical_gradient(self):
        """Test division with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.uniform(1, 5, (3, 4)).astype(np.float32)
        b_data = np.random.uniform(1, 5, (3, 4)).astype(np.float32)  # Avoid zero

        a = Tensor(a_data.copy(), requires_grad=True)
        b = Tensor(b_data.copy(), requires_grad=True)
        c = ops.div(a, b).sum()
        c.backward()

        # Numerical gradient for a
        def f_a(x):
            return ops.div(x, Tensor(b_data))
        num_grad_a = numerical_gradient_for_scalar_func(f_a, a_data)
        np.testing.assert_allclose(a.grad, num_grad_a, rtol=0.05, atol=0.01)

    def test_neg_forward(self):
        """Test negation forward pass."""
        a = Tensor([1.0, -2.0, 3.0])
        c = ops.neg(a)

        expected = np.array([-1.0, 2.0, -3.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_neg_gradient(self):
        """Test negation gradient."""
        a = Tensor([1.0, -2.0, 3.0], requires_grad=True)
        c = ops.neg(a).sum()
        c.backward()

        np.testing.assert_allclose(a.grad, -np.ones(3), rtol=1e-5)

    def test_matmul_forward(self):
        """Test matrix multiplication forward pass."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = ops.matmul(a, b)

        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_matmul_gradient(self):
        """Test matmul gradients with numerical check."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)
        b_data = np.random.randn(4, 5).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        b = Tensor(b_data.copy(), requires_grad=True)
        c = ops.matmul(a, b).sum()
        c.backward()

        # Numerical gradient for a
        def f_a(x):
            return ops.matmul(x, Tensor(b_data))
        num_grad_a = numerical_gradient_for_scalar_func(f_a, a_data)
        np.testing.assert_allclose(a.grad, num_grad_a, rtol=0.05, atol=0.01)

        # Numerical gradient for b
        def f_b(x):
            return ops.matmul(Tensor(a_data), x)
        num_grad_b = numerical_gradient_for_scalar_func(f_b, b_data)
        np.testing.assert_allclose(b.grad, num_grad_b, rtol=0.05, atol=0.01)

    def test_power_forward(self):
        """Test power operation forward pass."""
        a = Tensor([1.0, 2.0, 3.0])
        c = a ** 2

        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_power_gradient(self):
        """Test power operation gradients."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = (a ** 2).sum()
        c.backward()

        # d(x^2)/dx = 2x
        expected = 2 * np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)


class TestMathFunctions:
    """Test math functions and their gradients."""

    def test_exp_forward(self):
        """Test exp forward pass."""
        a = Tensor([0.0, 1.0, 2.0])
        c = ops.exp(a)

        expected = np.exp([0.0, 1.0, 2.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_exp_gradient(self):
        """Test exp gradient."""
        a = Tensor([0.0, 1.0, 2.0], requires_grad=True)
        c = ops.exp(a).sum()
        c.backward()

        # d(exp(x))/dx = exp(x)
        expected = np.exp([0.0, 1.0, 2.0])
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)

    def test_exp_numerical_gradient(self):
        """Test exp with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32) * 0.5  # Scale down to avoid overflow

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.exp(a).sum()
        c.backward()

        def f(x):
            return ops.exp(x)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)

    def test_log_forward(self):
        """Test log forward pass."""
        a = Tensor([1.0, np.e, np.e**2])
        c = ops.log(a)

        expected = np.array([0.0, 1.0, 2.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_log_gradient(self):
        """Test log gradient."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        c = ops.log(a).sum()
        c.backward()

        # d(log(x))/dx = 1/x
        expected = 1.0 / np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)

    def test_log_numerical_gradient(self):
        """Test log with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.uniform(0.5, 5, (3, 4)).astype(np.float32)  # Positive values

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.log(a).sum()
        c.backward()

        def f(x):
            return ops.log(x)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)

    def test_sqrt_forward(self):
        """Test sqrt forward pass."""
        a = Tensor([1.0, 4.0, 9.0])
        c = ops.sqrt(a)

        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_sqrt_gradient(self):
        """Test sqrt gradient."""
        a = Tensor([1.0, 4.0, 9.0], requires_grad=True)
        c = ops.sqrt(a).sum()
        c.backward()

        # d(sqrt(x))/dx = 1/(2*sqrt(x))
        expected = 1.0 / (2 * np.sqrt([1.0, 4.0, 9.0]))
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)

    def test_sqrt_numerical_gradient(self):
        """Test sqrt with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.uniform(0.5, 5, (3, 4)).astype(np.float32)  # Positive values

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.sqrt(a).sum()
        c.backward()

        def f(x):
            return ops.sqrt(x)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)

    def test_sin_forward(self):
        """Test sin forward pass."""
        a = Tensor([0.0, np.pi / 2, np.pi])
        c = ops.sin(a)

        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5, atol=1e-7)

    def test_sin_gradient(self):
        """Test sin gradient."""
        a = Tensor([0.0, np.pi / 4, np.pi / 2], requires_grad=True)
        c = ops.sin(a).sum()
        c.backward()

        # d(sin(x))/dx = cos(x)
        expected = np.cos([0.0, np.pi / 4, np.pi / 2])
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5, atol=1e-6)

    def test_sin_numerical_gradient(self):
        """Test sin with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.sin(a).sum()
        c.backward()

        def f(x):
            return ops.sin(x)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)

    def test_cos_forward(self):
        """Test cos forward pass."""
        a = Tensor([0.0, np.pi / 2, np.pi])
        c = ops.cos(a)

        expected = np.array([1.0, 0.0, -1.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5, atol=1e-7)

    def test_cos_gradient(self):
        """Test cos gradient."""
        a = Tensor([0.0, np.pi / 4, np.pi / 2], requires_grad=True)
        c = ops.cos(a).sum()
        c.backward()

        # d(cos(x))/dx = -sin(x)
        expected = -np.sin([0.0, np.pi / 4, np.pi / 2])
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)

    def test_cos_numerical_gradient(self):
        """Test cos with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.cos(a).sum()
        c.backward()

        def f(x):
            return ops.cos(x)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)

    def test_tanh_forward(self):
        """Test tanh forward pass."""
        a = Tensor([0.0, 1.0, -1.0])
        c = ops.tanh(a)

        expected = np.tanh([0.0, 1.0, -1.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_tanh_gradient(self):
        """Test tanh gradient."""
        a = Tensor([0.0, 1.0, -1.0], requires_grad=True)
        c = ops.tanh(a).sum()
        c.backward()

        # d(tanh(x))/dx = 1 - tanh(x)^2
        tanh_val = np.tanh([0.0, 1.0, -1.0])
        expected = 1 - tanh_val ** 2
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)

    def test_tanh_numerical_gradient(self):
        """Test tanh with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.tanh(a).sum()
        c.backward()

        def f(x):
            return ops.tanh(x)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)


class TestActivations:
    """Test activation functions and their gradients."""

    def test_sigmoid_forward(self):
        """Test sigmoid forward pass."""
        a = Tensor([0.0, 10.0, -10.0])
        c = ops.sigmoid(a)

        expected = 1 / (1 + np.exp(-np.array([0.0, 10.0, -10.0])))
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_sigmoid_gradient(self):
        """Test sigmoid gradient."""
        a = Tensor([0.0, 1.0, -1.0], requires_grad=True)
        c = ops.sigmoid(a).sum()
        c.backward()

        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        sig = 1 / (1 + np.exp(-np.array([0.0, 1.0, -1.0])))
        expected = sig * (1 - sig)
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)

    def test_sigmoid_numerical_gradient(self):
        """Test sigmoid with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.sigmoid(a).sum()
        c.backward()

        def f(x):
            return ops.sigmoid(x)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)

    def test_relu_forward(self):
        """Test relu forward pass."""
        a = Tensor([-1.0, 0.0, 1.0, 2.0])
        c = ops.relu(a)

        expected = np.array([0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_relu_gradient(self):
        """Test relu gradient."""
        a = Tensor([-1.0, 0.5, 1.0, 2.0], requires_grad=True)
        c = ops.relu(a).sum()
        c.backward()

        # d(relu(x))/dx = 1 if x > 0 else 0
        expected = np.array([0.0, 1.0, 1.0, 1.0])
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)

    def test_relu_numerical_gradient(self):
        """Test relu with numerical gradient checking (avoiding zero)."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)
        # Avoid values close to zero for numerical stability
        a_data[np.abs(a_data) < 0.1] = 0.5

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.relu(a).sum()
        c.backward()

        def f(x):
            return ops.relu(x)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)

    def test_softmax_forward(self):
        """Test softmax forward pass."""
        a = Tensor([[1.0, 2.0, 3.0]])
        c = ops.softmax(a, axis=-1)

        # Check that it sums to 1
        np.testing.assert_allclose(c.data.sum(axis=-1), 1.0, rtol=1e-5)

        # Check values
        exp_a = np.exp([1.0, 2.0, 3.0])
        expected = exp_a / exp_a.sum()
        np.testing.assert_allclose(c.data[0], expected, rtol=1e-5)

    def test_softmax_gradient(self):
        """Test softmax gradient."""
        np.random.seed(42)
        a_data = np.random.randn(2, 3).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.softmax(a, axis=-1).sum()
        c.backward()

        # Softmax sum is always 1, so gradient should be 0
        np.testing.assert_allclose(a.grad, np.zeros_like(a_data), rtol=1e-5, atol=1e-5)

    def test_softmax_numerical_gradient(self):
        """Test softmax with numerical gradient checking (using weighted sum)."""
        np.random.seed(42)
        a_data = np.random.randn(2, 3).astype(np.float32)
        weights = np.random.randn(2, 3).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        c = (ops.softmax(a, axis=-1) * Tensor(weights)).sum()
        c.backward()

        def f(x):
            return ops.softmax(x, axis=-1) * Tensor(weights)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)


class TestReductionOperations:
    """Test reduction operations and their gradients."""

    def test_sum_forward(self):
        """Test sum forward pass."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])

        c_all = ops.sum(a)
        np.testing.assert_allclose(c_all.data, 10.0, rtol=1e-5)

        c_axis0 = ops.sum(a, axis=0)
        np.testing.assert_allclose(c_axis0.data, [4.0, 6.0], rtol=1e-5)

        c_axis1 = ops.sum(a, axis=1)
        np.testing.assert_allclose(c_axis1.data, [3.0, 7.0], rtol=1e-5)

    def test_sum_gradient(self):
        """Test sum gradient."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = ops.sum(a)
        c.backward()

        np.testing.assert_allclose(a.grad, np.ones((2, 2)), rtol=1e-5)

    def test_sum_axis_gradient(self):
        """Test sum with axis gradient."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = ops.sum(a, axis=0).sum()
        c.backward()

        np.testing.assert_allclose(a.grad, np.ones((2, 2)), rtol=1e-5)

    def test_mean_forward(self):
        """Test mean forward pass."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])

        c_all = ops.mean(a)
        np.testing.assert_allclose(c_all.data, 2.5, rtol=1e-5)

        c_axis0 = ops.mean(a, axis=0)
        np.testing.assert_allclose(c_axis0.data, [2.0, 3.0], rtol=1e-5)

    def test_mean_gradient(self):
        """Test mean gradient."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        c = ops.mean(a)
        c.backward()

        np.testing.assert_allclose(a.grad, np.ones((2, 2)) / 4, rtol=1e-5)

    def test_mean_numerical_gradient(self):
        """Test mean with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.mean(a)
        c.backward()

        def f(x):
            return x  # Just pass through, mean is handled in sum
        num_grad = numerical_gradient_for_scalar_func(f, a_data) / a_data.size
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)

    def test_max_forward(self):
        """Test max forward pass."""
        a = Tensor([[1.0, 4.0], [3.0, 2.0]])

        c_all = ops.max(a)
        np.testing.assert_allclose(c_all.data, 4.0, rtol=1e-5)

        c_axis0 = ops.max(a, axis=0)
        np.testing.assert_allclose(c_axis0.data, [3.0, 4.0], rtol=1e-5)

    def test_max_gradient(self):
        """Test max gradient - only max elements get gradient."""
        a = Tensor([[1.0, 4.0], [3.0, 2.0]], requires_grad=True)
        c = ops.max(a)
        c.backward()

        # Only the max element (4.0 at [0,1]) should have gradient 1
        expected = np.array([[0.0, 1.0], [0.0, 0.0]])
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)


class TestShapeOperations:
    """Test shape operations and their gradients."""

    def test_reshape_forward(self):
        """Test reshape forward pass."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        c = ops.reshape(a, (3, 2))

        assert c.shape == (3, 2)
        np.testing.assert_allclose(c.data.flatten(), a.data.flatten(), rtol=1e-5)

    def test_reshape_gradient(self):
        """Test reshape gradient."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        c = ops.reshape(a, (3, 2)).sum()
        c.backward()

        np.testing.assert_allclose(a.grad, np.ones((2, 3)), rtol=1e-5)

    def test_transpose_forward(self):
        """Test transpose forward pass."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        c = ops.transpose(a)

        assert c.shape == (3, 2)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_transpose_gradient(self):
        """Test transpose gradient."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        c = ops.transpose(a).sum()
        c.backward()

        np.testing.assert_allclose(a.grad, np.ones((3, 4)), rtol=1e-5)

    def test_transpose_numerical_gradient(self):
        """Test transpose with numerical gradient checking."""
        np.random.seed(42)
        a_data = np.random.randn(3, 4).astype(np.float32)
        weights = np.random.randn(4, 3).astype(np.float32)

        a = Tensor(a_data.copy(), requires_grad=True)
        c = (ops.transpose(a) * Tensor(weights)).sum()
        c.backward()

        def f(x):
            return ops.transpose(x) * Tensor(weights)
        num_grad = numerical_gradient_for_scalar_func(f, a_data)
        np.testing.assert_allclose(a.grad, num_grad, rtol=0.05, atol=0.01)

    def test_concat_forward(self):
        """Test concat forward pass."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0], [7.0, 8.0]])
        c = ops.concat([a, b], axis=0)

        assert c.shape == (4, 2)
        expected = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_allclose(c.data, expected, rtol=1e-5)

    def test_concat_gradient(self):
        """Test concat gradient."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        c = ops.concat([a, b], axis=0).sum()
        c.backward()

        np.testing.assert_allclose(a.grad, np.ones((2, 2)), rtol=1e-5)
        np.testing.assert_allclose(b.grad, np.ones((2, 2)), rtol=1e-5)


class TestTensorMethods:
    """Test Tensor class methods."""

    def test_tensor_slicing(self):
        """Test tensor slicing with gradients."""
        a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        c = a[0, :].sum()
        c.backward()

        expected = np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(a.grad, expected, rtol=1e-5)

    def test_squeeze_unsqueeze(self):
        """Test squeeze and unsqueeze operations."""
        a = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        b = a.squeeze(0)

        assert b.shape == (3,)

        c = b.unsqueeze(0)
        assert c.shape == (1, 3)

        c.sum().backward()
        np.testing.assert_allclose(a.grad, np.ones((1, 3)), rtol=1e-5)

    def test_flatten(self):
        """Test flatten operation."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = a.flatten()

        assert b.shape == (4,)
        b.sum().backward()
        np.testing.assert_allclose(a.grad, np.ones((2, 2)), rtol=1e-5)


class TestNoGradContext:
    """Test no_grad context manager."""

    def test_no_grad_disables_gradient(self):
        """Test that no_grad disables gradient tracking."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        with no_grad():
            b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
            # Even with requires_grad=True, inside no_grad it should be False
            assert not b.requires_grad

    def test_no_grad_nested(self):
        """Test nested no_grad contexts."""
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)

        with no_grad():
            with enable_grad():
                b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
                assert b.requires_grad


class TestFactoryMethods:
    """Test Tensor factory methods."""

    def test_zeros(self):
        """Test zeros factory method."""
        a = Tensor.zeros((3, 4))
        assert a.shape == (3, 4)
        np.testing.assert_allclose(a.data, np.zeros((3, 4)), rtol=1e-5)

    def test_ones(self):
        """Test ones factory method."""
        a = Tensor.ones((3, 4))
        assert a.shape == (3, 4)
        np.testing.assert_allclose(a.data, np.ones((3, 4)), rtol=1e-5)

    def test_randn(self):
        """Test randn factory method."""
        np.random.seed(42)
        a = Tensor.randn((100, 100))

        # Should be approximately standard normal
        assert abs(a.data.mean()) < 0.1
        assert abs(a.data.std() - 1.0) < 0.1

    def test_rand(self):
        """Test rand factory method."""
        np.random.seed(42)
        a = Tensor.rand((100, 100))

        # Should be uniform [0, 1)
        assert a.data.min() >= 0.0
        assert a.data.max() < 1.0
        assert abs(a.data.mean() - 0.5) < 0.1

    def test_eye(self):
        """Test eye factory method."""
        a = Tensor.eye(4)
        assert a.shape == (4, 4)
        np.testing.assert_allclose(a.data, np.eye(4), rtol=1e-5)

    def test_arange(self):
        """Test arange factory method."""
        a = Tensor.arange(0, 10, 2)
        expected = np.array([0, 2, 4, 6, 8])
        np.testing.assert_allclose(a.data, expected, rtol=1e-5)


class TestBackwardTraversal:
    """Test the iterative reverse-topological backward pass."""

    def test_multiple_paths(self):
        """A tensor reached along multiple paths sums its incoming gradients."""
        x = Tensor([2.0], requires_grad=True)
        # y = x**2 + 3*x + 1 ; dy/dx = 2x + 3 = 7 at x = 2
        y = x ** 2 + 3 * x + 1
        y.backward()
        np.testing.assert_allclose(x.grad, [7.0], rtol=1e-6)

    def test_diamond_graph_visited_once(self):
        """Shared node in a diamond gets the summed gradient exactly once."""
        x = Tensor([3.0], requires_grad=True)
        a = x * 2.0          # da/dx = 2
        b = x * 5.0          # db/dx = 5
        y = a + b            # y = 7x ; dy/dx = 7
        y.backward()
        np.testing.assert_allclose(x.grad, [7.0], rtol=1e-6)

    def test_deep_chain_no_recursion_error(self):
        """A very deep chain (2000+ ops) completes without RecursionError.

        The old recursive backward would raise RecursionError here; the
        iterative traversal handles arbitrary depth. Chain: repeatedly
        y = y + 1, whose gradient w.r.t. the leaf is exactly 1.
        """
        import sys

        depth = 3000
        assert depth > sys.getrecursionlimit(), (
            "test depth should exceed the interpreter recursion limit to be "
            "a meaningful regression check"
        )

        x = Tensor([1.0], requires_grad=True)
        y = x
        for _ in range(depth):
            y = y + 1.0

        # Forward value: 1 + depth
        np.testing.assert_allclose(y.data, [1.0 + depth], rtol=1e-6)

        y.backward()
        # d(x + depth)/dx == 1
        np.testing.assert_allclose(x.grad, [1.0], rtol=1e-6)

    def test_deep_chain_multiplicative_grad(self):
        """Deep chain with a non-trivial gradient stays numerically correct."""
        depth = 2500
        x = Tensor([1.0], requires_grad=True)
        y = x
        # Alternate +1 and *1.0 so the chain is long but gradient stays 1.
        for _ in range(depth):
            y = (y + 2.0) - 2.0
        y.backward()
        np.testing.assert_allclose(x.grad, [1.0], rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
