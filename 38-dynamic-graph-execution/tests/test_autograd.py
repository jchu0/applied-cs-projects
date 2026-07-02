"""Tests for automatic differentiation correctness in DynaGraph."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynagraph import Tensor, Parameter, no_grad, enable_grad, is_grad_enabled
from dynagraph.autograd.autograd import (
    Function, FunctionContext, backward, grad, GradientTape,
    ReLU, Sigmoid, Tanh, Softmax, MatMul, Add, Mul,
    jacobian, hessian,
)


class TestBasicGradients:
    """Tests for basic gradient computation."""

    def test_scalar_gradient(self):
        """Test gradient for scalar output."""
        x = Tensor([2.0], requires_grad=True)
        y = x * x  # y = x^2

        y.backward()

        # dy/dx = 2x = 4
        expected = np.array([4.0])
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_vector_gradient(self):
        """Test gradient for vector operations."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = (x * x).sum()  # y = sum(x^2)

        y.backward()

        # dy/dx = 2x
        expected = np.array([2.0, 4.0, 6.0])
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_matrix_gradient(self):
        """Test gradient for matrix operations."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = (x * 2).sum()

        y.backward()

        expected = np.full((2, 2), 2.0)
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_addition_gradient(self):
        """Test gradient of addition."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        y = (a + b).sum()

        y.backward()

        np.testing.assert_allclose(a.grad, np.ones(2), rtol=1e-5)
        np.testing.assert_allclose(b.grad, np.ones(2), rtol=1e-5)

    def test_subtraction_gradient(self):
        """Test gradient of subtraction."""
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        y = (a - b).sum()

        y.backward()

        np.testing.assert_allclose(a.grad, np.ones(2), rtol=1e-5)
        np.testing.assert_allclose(b.grad, -np.ones(2), rtol=1e-5)

    def test_multiplication_gradient(self):
        """Test gradient of multiplication."""
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        y = (a * b).sum()

        y.backward()

        # d(ab)/da = b, d(ab)/db = a
        np.testing.assert_allclose(a.grad, np.array([4.0, 5.0]), rtol=1e-5)
        np.testing.assert_allclose(b.grad, np.array([2.0, 3.0]), rtol=1e-5)

    def test_division_gradient(self):
        """Test gradient of division."""
        a = Tensor([4.0, 9.0], requires_grad=True)
        b = Tensor([2.0, 3.0], requires_grad=True)
        y = (a / b).sum()

        y.backward()

        # d(a/b)/da = 1/b
        # d(a/b)/db = -a/b^2
        np.testing.assert_allclose(a.grad, np.array([0.5, 1/3]), rtol=1e-5)
        np.testing.assert_allclose(b.grad, np.array([-1.0, -1.0]), rtol=1e-5)

    def test_power_gradient(self):
        """Test gradient of power operation."""
        x = Tensor([2.0, 3.0], requires_grad=True)
        y = (x ** 3).sum()  # y = x^3

        y.backward()

        # dy/dx = 3x^2
        expected = 3 * np.array([4.0, 9.0])
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_negation_gradient(self):
        """Test gradient of negation."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = (-x).sum()

        y.backward()

        np.testing.assert_allclose(x.grad, -np.ones(2), rtol=1e-5)


class TestMatrixGradients:
    """Tests for matrix operation gradients."""

    def test_matmul_gradient(self):
        """Test gradient of matrix multiplication."""
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        y = (a @ b).sum()

        y.backward()

        # Verify gradients exist and have correct shape
        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == a.shape
        assert b.grad.shape == b.shape

    def test_matmul_gradient_values(self):
        """Test exact values of matmul gradient."""
        # Simple case: 1x2 @ 2x1 = 1x1
        a = Tensor([[2.0, 3.0]], requires_grad=True)
        b = Tensor([[4.0], [5.0]], requires_grad=True)
        y = (a @ b).sum()

        y.backward()

        # d(a@b)/da = grad @ b.T
        # d(a@b)/db = a.T @ grad
        np.testing.assert_allclose(a.grad, np.array([[4.0, 5.0]]), rtol=1e-5)
        np.testing.assert_allclose(b.grad, np.array([[2.0], [3.0]]), rtol=1e-5)

    def test_transpose_gradient(self):
        """Test gradient through transpose."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.T.sum()

        y.backward()

        np.testing.assert_allclose(x.grad, np.ones((2, 2)), rtol=1e-5)

    def test_view_gradient(self):
        """Test gradient through reshape/view."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.view(4).sum()

        y.backward()

        np.testing.assert_allclose(x.grad, np.ones((2, 2)), rtol=1e-5)


class TestReductionGradients:
    """Tests for reduction operation gradients."""

    def test_sum_gradient(self):
        """Test gradient of sum."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum()

        y.backward()

        np.testing.assert_allclose(x.grad, np.ones((2, 2)), rtol=1e-5)

    def test_sum_axis_gradient(self):
        """Test gradient of sum along axis."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum(axis=0).sum()

        y.backward()

        np.testing.assert_allclose(x.grad, np.ones((2, 2)), rtol=1e-5)

    def test_mean_gradient(self):
        """Test gradient of mean."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.mean()

        y.backward()

        expected = np.ones((2, 2)) / 4
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_max_gradient(self):
        """Test gradient of max."""
        x = Tensor([1.0, 3.0, 2.0], requires_grad=True)
        y = x.max()

        y.backward()

        # Gradient flows only through max element
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_min_gradient(self):
        """Test gradient of min."""
        x = Tensor([1.0, 3.0, 2.0], requires_grad=True)
        y = x.min()

        y.backward()

        # Gradient flows only through min element
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)


class TestActivationGradients:
    """Tests for activation function gradients."""

    def test_relu_gradient_positive(self):
        """Test ReLU gradient for positive values."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = ReLU.apply(x)
        y.sum().backward()

        expected = np.ones(3)
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_relu_gradient_negative(self):
        """Test ReLU gradient for negative values."""
        x = Tensor([-1.0, -2.0, -3.0], requires_grad=True)
        y = ReLU.apply(x)
        y.sum().backward()

        expected = np.zeros(3)
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_relu_gradient_mixed(self):
        """Test ReLU gradient for mixed values."""
        x = Tensor([-1.0, 2.0, -3.0, 4.0], requires_grad=True)
        y = ReLU.apply(x)
        y.sum().backward()

        expected = np.array([0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)

    def test_sigmoid_gradient(self):
        """Test sigmoid gradient."""
        x = Tensor([0.0], requires_grad=True)
        y = Sigmoid.apply(x)
        y.backward()

        # sigmoid(0) = 0.5, gradient = 0.5 * (1 - 0.5) = 0.25
        np.testing.assert_allclose(x.grad, np.array([0.25]), rtol=1e-5)

    def test_tanh_gradient(self):
        """Test tanh gradient."""
        x = Tensor([0.0], requires_grad=True)
        y = Tanh.apply(x)
        y.backward()

        # tanh(0) = 0, gradient = 1 - 0^2 = 1
        np.testing.assert_allclose(x.grad, np.array([1.0]), rtol=1e-5)

    def test_softmax_gradient(self):
        """Test softmax gradient."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = Softmax.apply(x)
        loss = (y * Tensor([1.0, 0.0, 0.0])).sum()
        loss.backward()

        # Gradient should exist and have correct shape
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestBroadcastingGradients:
    """Tests for broadcasting gradient handling."""

    def test_scalar_broadcast_gradient(self):
        """Test gradient with scalar broadcasting."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = (x + 1.0).sum()

        y.backward()

        np.testing.assert_allclose(x.grad, np.ones((2, 2)), rtol=1e-5)

    def test_vector_broadcast_gradient(self):
        """Test gradient with vector broadcasting."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([1.0, 2.0], requires_grad=True)
        y = (x + b).sum()

        y.backward()

        np.testing.assert_allclose(x.grad, np.ones((2, 2)), rtol=1e-5)
        # b is broadcast, so gradient is summed
        np.testing.assert_allclose(b.grad, np.array([2.0, 2.0]), rtol=1e-5)

    def test_mul_broadcast_gradient(self):
        """Test multiplication with broadcasting gradient."""
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        s = Tensor([2.0], requires_grad=True)
        y = (x * s).sum()

        y.backward()

        np.testing.assert_allclose(x.grad, np.full((2, 2), 2.0), rtol=1e-5)
        # s gradient is sum of all x values
        np.testing.assert_allclose(s.grad, np.array([10.0]), rtol=1e-5)


class TestChainRule:
    """Tests for chain rule in backpropagation."""

    def test_simple_chain(self):
        """Test simple chain: y = 2 * (x^2)."""
        x = Tensor([3.0], requires_grad=True)
        y = x ** 2
        z = y * 2

        z.backward()

        # dz/dx = 2 * 2x = 4x = 12
        np.testing.assert_allclose(x.grad, np.array([12.0]), rtol=1e-5)

    def test_multi_step_chain(self):
        """Test multi-step chain."""
        x = Tensor([2.0], requires_grad=True)
        y = x * 3      # y = 3x
        z = y + 1      # z = 3x + 1
        w = z ** 2     # w = (3x + 1)^2

        w.backward()

        # dw/dx = 2(3x + 1) * 3 = 6(3x + 1) = 6 * 7 = 42
        np.testing.assert_allclose(x.grad, np.array([42.0]), rtol=1e-5)

    def test_diamond_graph(self):
        """Test diamond-shaped computation graph."""
        x = Tensor([2.0], requires_grad=True)
        y1 = x * 2    # y1 = 2x
        y2 = x * 3    # y2 = 3x
        z = y1 + y2   # z = 5x

        z.backward()

        # dz/dx = 5
        np.testing.assert_allclose(x.grad, np.array([5.0]), rtol=1e-5)

    def test_multiple_paths(self):
        """Test gradient accumulation from multiple paths."""
        x = Tensor([1.0], requires_grad=True)
        y = x + x + x  # y = 3x

        y.backward()

        np.testing.assert_allclose(x.grad, np.array([3.0]), rtol=1e-5)


class TestGradientTape:
    """Tests for GradientTape API."""

    def test_gradient_tape_basic(self):
        """Test basic GradientTape usage."""
        x = Tensor([2.0, 3.0])

        with GradientTape() as tape:
            tape.watch(x)
            y = x ** 2
            z = y.sum()

        grad = tape.gradient(z, x)

        np.testing.assert_allclose(grad, np.array([4.0, 6.0]), rtol=1e-5)

    def test_gradient_tape_persistent(self):
        """Test persistent GradientTape."""
        x = Tensor([2.0])

        with GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = x ** 2
            z = x ** 3

        grad_y = tape.gradient(y, x)
        grad_z = tape.gradient(z, x)

        np.testing.assert_allclose(grad_y, np.array([4.0]), rtol=1e-5)
        np.testing.assert_allclose(grad_z, np.array([12.0]), rtol=1e-5)

    def test_gradient_tape_multiple_inputs(self):
        """Test GradientTape with multiple inputs."""
        x = Tensor([1.0, 2.0])
        y = Tensor([3.0, 4.0])

        with GradientTape() as tape:
            tape.watch(x)
            tape.watch(y)
            z = (x + y).sum()

        grads = tape.gradient(z, [x, y])

        np.testing.assert_allclose(grads[0], np.ones(2), rtol=1e-5)
        np.testing.assert_allclose(grads[1], np.ones(2), rtol=1e-5)


class TestGradientContext:
    """Tests for gradient context managers."""

    def test_no_grad_context(self):
        """Test no_grad context prevents gradient tracking."""
        x = Tensor([1.0, 2.0], requires_grad=True)

        with no_grad():
            y = x * 2
            assert not y.requires_grad

    def test_enable_grad_context(self):
        """Test enable_grad context enables gradient tracking."""
        x = Tensor([1.0, 2.0], requires_grad=True)

        with no_grad():
            with enable_grad():
                y = x * 2
                assert y.requires_grad

    def test_nested_grad_contexts(self):
        """Test nested gradient contexts."""
        x = Tensor([1.0], requires_grad=True)

        assert is_grad_enabled()

        with no_grad():
            assert not is_grad_enabled()
            with enable_grad():
                assert is_grad_enabled()
            assert not is_grad_enabled()

        assert is_grad_enabled()

    def test_no_grad_preserves_existing_grads(self):
        """Test that no_grad doesn't affect existing gradients."""
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward()

        original_grad = x.grad.copy()

        with no_grad():
            z = x * 2  # Should not affect x.grad

        np.testing.assert_allclose(x.grad, original_grad, rtol=1e-5)


class TestGradFunction:
    """Tests for the grad() function."""

    def test_grad_single_input(self):
        """Test grad with single input."""
        x = Tensor([2.0], requires_grad=True)
        y = x ** 2

        g = grad(y, x)

        np.testing.assert_allclose(g, np.array([4.0]), rtol=1e-5)

    def test_grad_multiple_inputs(self):
        """Test grad with multiple inputs."""
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x * y

        grads = grad(z, [x, y])

        np.testing.assert_allclose(grads[0], np.array([3.0]), rtol=1e-5)
        np.testing.assert_allclose(grads[1], np.array([2.0]), rtol=1e-5)

    def test_grad_with_custom_output_grad(self):
        """Test grad with custom output gradient."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x * 2

        g = grad(y, x, grad_outputs=np.array([1.0, 2.0]))

        np.testing.assert_allclose(g, np.array([2.0, 4.0]), rtol=1e-5)


class TestCustomFunction:
    """Tests for custom differentiable functions."""

    def test_custom_square_function(self):
        """Test custom square function with backward."""
        class Square(Function):
            @staticmethod
            def forward(ctx, x):
                ctx.save_for_backward(x)
                return Tensor(x.data ** 2, requires_grad=True)

            @staticmethod
            def backward(ctx, grad_output):
                x, = ctx.saved_tensors
                return 2 * x * grad_output

        x = Tensor([3.0], requires_grad=True)
        y = Square.apply(x)
        y.backward()

        np.testing.assert_allclose(x.grad, np.array([6.0]), rtol=1e-5)

    def test_custom_function_chain(self):
        """Test chaining custom functions."""
        class Double(Function):
            @staticmethod
            def forward(ctx, x):
                return Tensor(x.data * 2, requires_grad=True)

            @staticmethod
            def backward(ctx, grad_output):
                return grad_output * 2

        x = Tensor([1.0], requires_grad=True)
        y = Double.apply(x)
        z = Double.apply(y)
        z.backward()

        np.testing.assert_allclose(x.grad, np.array([4.0]), rtol=1e-5)


class TestEdgeCases:
    """Tests for edge cases in autograd."""

    def test_zero_gradient(self):
        """Test computation with zero gradient."""
        x = Tensor([0.0], requires_grad=True)
        y = x * 0  # Gradient should be 0

        y.backward()

        np.testing.assert_allclose(x.grad, np.array([0.0]), rtol=1e-5)

    def test_large_values(self):
        """Test gradient computation with large values."""
        x = Tensor([1e6], requires_grad=True)
        y = x / 1e6
        y.backward()

        np.testing.assert_allclose(x.grad, np.array([1e-6]), rtol=1e-5)

    def test_small_values(self):
        """Test gradient computation with small values."""
        x = Tensor([1e-6], requires_grad=True)
        y = x * 1e6
        y.backward()

        np.testing.assert_allclose(x.grad, np.array([1e6]), rtol=1e-5)

    def test_clone_gradient(self):
        """Test gradient flows through clone."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = x.clone()
        z = y.sum()
        z.backward()

        np.testing.assert_allclose(x.grad, np.ones(2), rtol=1e-5)

    def test_getitem_gradient(self):
        """Test gradient flows through indexing."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x[1]  # Select middle element
        y.backward()

        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(x.grad, expected, rtol=1e-5)


class TestJacobian:
    """Reverse-mode Jacobian (regression tests: previously returned all-zeros)."""

    def test_jacobian_elementwise_square(self):
        """d/dx of elementwise x*x is diag(2x)."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * x
        jac = jacobian(y, x)
        np.testing.assert_allclose(jac, np.diag([2.0, 4.0, 6.0]), atol=1e-5)

    def test_jacobian_is_not_zero(self):
        """Guard against the old placeholder that always returned zeros."""
        x = Tensor([0.5, -1.5], requires_grad=True)
        jac = jacobian(x * x, x)
        assert np.any(jac != 0.0)

    def test_gradient_tape_jacobian(self):
        """GradientTape.jacobian delegates to the same correct implementation."""
        x = Tensor([2.0, 3.0], requires_grad=True)
        with GradientTape() as tape:
            tape.watch(x)
            y = x * x
        jac = tape.jacobian(y, x)
        np.testing.assert_allclose(jac, np.diag([4.0, 6.0]), atol=1e-5)


class TestHessian:
    """Finite-difference Hessian of the reverse-mode gradient."""

    def test_hessian_sum_of_squares(self):
        """Hessian of sum(x^2) is 2*I."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = (x * x).sum()
        H = hessian(y, x, func=lambda t: (t * t).sum())
        np.testing.assert_allclose(H, 2.0 * np.eye(3), atol=1e-3)

    def test_hessian_off_diagonal(self):
        """Hessian of (sum x)^2 is 2 on every entry (pure cross terms)."""
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = (x.sum()) * (x.sum())
        H = hessian(y, x, func=lambda t: (t.sum()) * (t.sum()))
        np.testing.assert_allclose(H, 2.0 * np.ones((3, 3)), atol=1e-3)

    def test_hessian_without_func_raises(self):
        """Without a recompute func the second derivative is unrecoverable;
        it must raise rather than silently return zeros (the old behavior)."""
        x = Tensor([1.0, 2.0], requires_grad=True)
        y = (x * x).sum()
        with pytest.raises(NotImplementedError):
            hessian(y, x)


class TestPublicAPIExports:
    """The autograd package must re-export the public built-in ops the README
    documents alongside the activations (Add, Mul, MatMul), plus FunctionContext,
    so `from dynagraph.autograd import ...` matches the docs."""

    def test_builtin_ops_reexported(self):
        from dynagraph import autograd as ag

        for name in ("Add", "Mul", "MatMul", "ReLU", "Sigmoid", "Tanh",
                     "Softmax", "FunctionContext"):
            assert name in ag.__all__, f"{name} missing from dynagraph.autograd.__all__"
            assert hasattr(ag, name), f"{name} not importable from dynagraph.autograd"

    def test_reexported_ops_are_functions(self):
        from dynagraph.autograd import Add, Mul, MatMul, Function

        for op in (Add, Mul, MatMul):
            assert issubclass(op, Function)

    def test_reexported_matmul_apply_matches_operator(self):
        from dynagraph.autograd import MatMul

        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        out = MatMul.apply(a, b)
        np.testing.assert_allclose(out.data, (a @ b).data, rtol=1e-6)
