"""Test fixtures and helpers for autograd tests."""

import numpy as np
import torch
import sys
import os
from typing import Tuple, List, Callable, Optional
import random

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from autograd.core.tensor import Tensor
from autograd.nn.modules import Module


def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def generate_random_tensor(
    shape: Tuple[int, ...],
    requires_grad: bool = True,
    low: float = -1.0,
    high: float = 1.0
) -> Tensor:
    """Generate a random tensor for testing."""
    data = np.random.uniform(low, high, shape).astype(np.float32)
    return Tensor(data, requires_grad=requires_grad)


def generate_random_data(
    shape: Tuple[int, ...],
    low: float = -1.0,
    high: float = 1.0
) -> np.ndarray:
    """Generate random numpy array."""
    return np.random.uniform(low, high, shape).astype(np.float32)


def numerical_gradient(
    f: Callable,
    x: Tensor,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute numerical gradient using finite differences.

    Args:
        f: Function that takes a Tensor and returns a scalar
        x: Input tensor
        eps: Small epsilon for finite differences

    Returns:
        Numerical gradient with same shape as x
    """
    grad = np.zeros_like(x.data)
    flat_x = x.data.flatten()
    flat_grad = grad.flatten()

    for i in range(len(flat_x)):
        # Save original value
        orig = flat_x[i]

        # f(x + eps)
        flat_x[i] = orig + eps
        x_plus = Tensor(x.data.copy())
        f_plus = f(x_plus)

        # f(x - eps)
        flat_x[i] = orig - eps
        x_minus = Tensor(x.data.copy())
        f_minus = f(x_minus)

        # Restore original value
        flat_x[i] = orig

        # Compute gradient
        flat_grad[i] = (f_plus.data - f_minus.data) / (2 * eps)

    return grad


def check_gradient(
    f: Callable,
    inputs: List[Tensor],
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> bool:
    """
    Check if analytical gradients match numerical gradients.

    Args:
        f: Function to test
        inputs: List of input tensors
        eps: Epsilon for numerical gradient
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        True if gradients match within tolerance
    """
    # Compute analytical gradients
    output = f(*inputs)
    output.backward()

    # Check each input
    for inp in inputs:
        if inp.requires_grad:
            # Compute numerical gradient
            def scalar_f(x):
                return f(x).sum() if len(inputs) == 1 else f(*[x if i == inp else inputs[i] for i in inputs]).sum()

            num_grad = numerical_gradient(scalar_f, inp, eps)

            # Compare
            if not np.allclose(inp.grad, num_grad, atol=atol, rtol=rtol):
                print(f"Gradient mismatch!")
                print(f"Analytical: {inp.grad}")
                print(f"Numerical: {num_grad}")
                print(f"Difference: {np.abs(inp.grad - num_grad)}")
                return False

    return True


def compare_with_pytorch(
    our_func: Callable,
    pytorch_func: Callable,
    inputs_np: List[np.ndarray],
    check_grad: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-3
) -> bool:
    """
    Compare our implementation with PyTorch.

    Args:
        our_func: Our autograd function
        pytorch_func: Equivalent PyTorch function
        inputs_np: List of numpy arrays as inputs
        check_grad: Whether to compare gradients
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        True if outputs and gradients match
    """
    # Create our tensors
    our_inputs = [Tensor(inp, requires_grad=True) for inp in inputs_np]

    # Create PyTorch tensors
    torch_inputs = [torch.tensor(inp, requires_grad=True, dtype=torch.float32)
                   for inp in inputs_np]

    # Forward pass
    our_output = our_func(*our_inputs)
    torch_output = pytorch_func(*torch_inputs)

    # Compare forward outputs
    if not np.allclose(our_output.data, torch_output.detach().numpy(), atol=atol, rtol=rtol):
        print(f"Forward pass mismatch!")
        print(f"Our output: {our_output.data}")
        print(f"PyTorch output: {torch_output.detach().numpy()}")
        return False

    if check_grad:
        # Backward pass
        our_output.sum().backward()
        torch_output.sum().backward()

        # Compare gradients
        for our_inp, torch_inp in zip(our_inputs, torch_inputs):
            if our_inp.requires_grad:
                if not np.allclose(our_inp.grad, torch_inp.grad.numpy(), atol=atol, rtol=rtol):
                    print(f"Gradient mismatch!")
                    print(f"Our gradient: {our_inp.grad}")
                    print(f"PyTorch gradient: {torch_inp.grad.numpy()}")
                    return False

    return True


class SimpleNet(Module):
    """Simple neural network for testing."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        from autograd.nn.modules import Linear, ReLU

        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SimpleCNN(Module):
    """Simple CNN for testing."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        from autograd.nn.modules import Conv2d, MaxPool2d, ReLU, Linear

        self.conv1 = Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2)

        self.conv2 = Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2)

        # Assuming input is 28x28
        self.fc = Linear(32 * 7 * 7, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


def generate_classification_data(
    num_samples: int,
    input_dim: int,
    num_classes: int,
    noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data."""
    # Generate random centers for each class
    centers = np.random.randn(num_classes, input_dim) * 2

    X = []
    y = []

    for _ in range(num_samples):
        # Random class
        class_idx = np.random.randint(num_classes)

        # Generate point near class center
        point = centers[class_idx] + np.random.randn(input_dim) * noise

        X.append(point)
        y.append(class_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_regression_data(
    num_samples: int,
    input_dim: int,
    noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data."""
    # Random linear relationship
    true_weights = np.random.randn(input_dim, 1)
    true_bias = np.random.randn(1)

    X = np.random.randn(num_samples, input_dim).astype(np.float32)
    y = X @ true_weights + true_bias + np.random.randn(num_samples, 1) * noise

    return X, y.astype(np.float32)


def assert_tensors_equal(
    t1: Tensor,
    t2: Tensor,
    atol: float = 1e-6,
    rtol: float = 1e-5
):
    """Assert two tensors are equal within tolerance."""
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"
    assert np.allclose(t1.data, t2.data, atol=atol, rtol=rtol), \
        f"Data mismatch: max diff = {np.max(np.abs(t1.data - t2.data))}"


def assert_gradients_equal(
    t1: Tensor,
    t2: Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4
):
    """Assert two tensor gradients are equal within tolerance."""
    assert t1.grad is not None and t2.grad is not None, "Missing gradients"
    assert t1.grad.shape == t2.grad.shape, f"Gradient shape mismatch: {t1.grad.shape} vs {t2.grad.shape}"
    assert np.allclose(t1.grad, t2.grad, atol=atol, rtol=rtol), \
        f"Gradient mismatch: max diff = {np.max(np.abs(t1.grad - t2.grad))}"


def create_mock_optimizer(params: List[Tensor], lr: float = 0.01):
    """Create a simple mock optimizer for testing."""
    class MockOptimizer:
        def __init__(self, params, lr):
            self.params = list(params)
            self.lr = lr

        def step(self):
            for param in self.params:
                if param.grad is not None:
                    param.data -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                param.grad = None

    return MockOptimizer(params, lr)


def finite_difference_jacobian(
    f: Callable[[Tensor], Tensor],
    x: Tensor,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Compute Jacobian matrix using finite differences.

    Args:
        f: Function that takes a tensor and returns a tensor
        x: Input tensor
        eps: Small epsilon for finite differences

    Returns:
        Jacobian matrix
    """
    x_flat = x.data.flatten()
    f_x = f(x)
    output_size = f_x.data.size
    input_size = x_flat.size

    jacobian = np.zeros((output_size, input_size))

    for i in range(input_size):
        # Save original value
        orig = x_flat[i]

        # f(x + eps)
        x_flat[i] = orig + eps
        x_plus = Tensor(x.data.copy())
        f_plus = f(x_plus).data.flatten()

        # f(x - eps)
        x_flat[i] = orig - eps
        x_minus = Tensor(x.data.copy())
        f_minus = f(x_minus).data.flatten()

        # Restore original value
        x_flat[i] = orig

        # Compute column of Jacobian
        jacobian[:, i] = (f_plus - f_minus) / (2 * eps)

    return jacobian