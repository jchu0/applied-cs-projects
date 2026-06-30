"""Tests for LazyTensor operations and lazy evaluation."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorlib.core.tensor import (
    LazyTensor, Device, DeviceType, ShapeSpec, TensorSpec, Tracer,
    array, zeros, ones, randn, rand, arange, linspace, eye, full
)
from tensorlib.core.primitives import (
    add, sub, mul, div, neg, power, matmul,
    exp, log, sqrt, sin, cos, tanh,
    reduce_sum, reduce_mean, reduce_max, reduce_min,
    reshape, transpose, broadcast_to, expand_dims, squeeze,
    relu, sigmoid, softmax
)


class TestLazyTensorCreation:
    """Test LazyTensor creation and factory functions."""

    def test_array_from_list(self):
        """Test creating tensor from Python list."""
        data = [[1, 2, 3], [4, 5, 6]]
        tensor = array(data)

        assert tensor.shape == (2, 3)
        assert tensor.ndim == 2
        assert tensor.size == 6
        np.testing.assert_array_equal(tensor.numpy(), np.array(data))

    def test_array_from_numpy(self):
        """Test creating tensor from NumPy array."""
        np_data = np.random.randn(10, 20).astype(np.float32)
        tensor = array(np_data)

        assert tensor.shape == (10, 20)
        assert tensor.dtype == np.float32
        np.testing.assert_array_almost_equal(tensor.numpy(), np_data)

    def test_array_with_dtype(self):
        """Test creating tensor with specific dtype."""
        tensor = array([1, 2, 3], dtype=np.float64)
        assert tensor.dtype == np.float64

    def test_zeros(self):
        """Test creating zero tensor."""
        tensor = zeros((3, 4, 5))

        assert tensor.shape == (3, 4, 5)
        assert np.all(tensor.numpy() == 0)

    def test_zeros_with_dtype(self):
        """Test creating zero tensor with specific dtype."""
        tensor = zeros((2, 2), dtype=np.float64)
        assert tensor.dtype == np.float64

    def test_ones(self):
        """Test creating ones tensor."""
        tensor = ones((2, 3))

        assert tensor.shape == (2, 3)
        assert np.all(tensor.numpy() == 1)

    def test_randn(self):
        """Test creating random normal tensor."""
        np.random.seed(42)
        tensor = randn(100, 100)

        assert tensor.shape == (100, 100)
        # Should have roughly zero mean and unit variance
        assert abs(tensor.numpy().mean()) < 0.3
        assert abs(tensor.numpy().std() - 1.0) < 0.3

    def test_rand(self):
        """Test creating random uniform tensor."""
        tensor = rand(100, 100)

        assert tensor.shape == (100, 100)
        # Values should be in [0, 1)
        assert tensor.numpy().min() >= 0
        assert tensor.numpy().max() < 1

    def test_arange(self):
        """Test creating arange tensor."""
        tensor = arange(0, 10, 2)
        np.testing.assert_array_equal(tensor.numpy(), np.array([0, 2, 4, 6, 8], dtype=np.float32))

    def test_arange_single_arg(self):
        """Test arange with single argument."""
        tensor = arange(5)
        np.testing.assert_array_equal(tensor.numpy(), np.array([0, 1, 2, 3, 4], dtype=np.float32))

    def test_linspace(self):
        """Test creating linspace tensor."""
        tensor = linspace(0, 1, 5)
        expected = np.array([0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        np.testing.assert_array_almost_equal(tensor.numpy(), expected)

    def test_eye(self):
        """Test creating identity matrix."""
        tensor = eye(4)

        assert tensor.shape == (4, 4)
        np.testing.assert_array_equal(tensor.numpy(), np.eye(4, dtype=np.float32))

    def test_eye_rectangular(self):
        """Test creating rectangular identity matrix."""
        tensor = eye(3, 5)

        assert tensor.shape == (3, 5)
        np.testing.assert_array_equal(tensor.numpy(), np.eye(3, 5, dtype=np.float32))

    def test_full(self):
        """Test creating tensor filled with value."""
        tensor = full((3, 4), 7.5)

        assert tensor.shape == (3, 4)
        assert np.all(tensor.numpy() == 7.5)


class TestLazyEvaluation:
    """Test lazy evaluation behavior."""

    def test_lazy_tensor_not_materialized(self):
        """Test that lazy operations don't materialize immediately."""
        x = randn(10, 10)
        y = randn(10, 10)

        # This creates a lazy computation
        z = x + y

        # z should be lazy (not materialized yet)
        assert not z.is_materialized

    def test_materialize_on_demand(self):
        """Test that materialize() forces evaluation."""
        x = array([1.0, 2.0, 3.0])
        y = array([4.0, 5.0, 6.0])

        z = x + y
        assert not z.is_materialized

        result = z.materialize()
        assert z.is_materialized
        np.testing.assert_array_equal(result, [5.0, 7.0, 9.0])

    def test_numpy_materializes(self):
        """Test that numpy() forces materialization."""
        x = array([1.0, 2.0, 3.0])
        y = x * 2

        result = y.numpy()
        assert y.is_materialized
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_already_materialized(self):
        """Test that materialized tensors don't recompute."""
        x = array([1.0, 2.0, 3.0])

        # Directly created tensors are already materialized
        assert x.is_materialized

        # Calling materialize again should return same value
        result1 = x.materialize()
        result2 = x.materialize()
        np.testing.assert_array_equal(result1, result2)


class TestArithmeticOperations:
    """Test arithmetic operations on LazyTensor."""

    def test_add(self):
        """Test addition."""
        x = array([1.0, 2.0, 3.0])
        y = array([4.0, 5.0, 6.0])

        z = x + y
        np.testing.assert_array_equal(z.numpy(), [5.0, 7.0, 9.0])

    def test_add_scalar(self):
        """Test addition with scalar."""
        x = array([1.0, 2.0, 3.0])
        z = x + 5.0

        np.testing.assert_array_equal(z.numpy(), [6.0, 7.0, 8.0])

    def test_radd(self):
        """Test reverse addition."""
        x = array([1.0, 2.0, 3.0])
        z = 10.0 + x

        np.testing.assert_array_equal(z.numpy(), [11.0, 12.0, 13.0])

    def test_sub(self):
        """Test subtraction."""
        x = array([5.0, 6.0, 7.0])
        y = array([1.0, 2.0, 3.0])

        z = x - y
        np.testing.assert_array_equal(z.numpy(), [4.0, 4.0, 4.0])

    def test_rsub(self):
        """Test reverse subtraction."""
        x = array([1.0, 2.0, 3.0])
        z = 10.0 - x

        np.testing.assert_array_equal(z.numpy(), [9.0, 8.0, 7.0])

    def test_mul(self):
        """Test multiplication."""
        x = array([1.0, 2.0, 3.0])
        y = array([4.0, 5.0, 6.0])

        z = x * y
        np.testing.assert_array_equal(z.numpy(), [4.0, 10.0, 18.0])

    def test_mul_scalar(self):
        """Test multiplication with scalar."""
        x = array([1.0, 2.0, 3.0])
        z = x * 3.0

        np.testing.assert_array_equal(z.numpy(), [3.0, 6.0, 9.0])

    def test_div(self):
        """Test division."""
        x = array([10.0, 20.0, 30.0])
        y = array([2.0, 4.0, 5.0])

        z = x / y
        np.testing.assert_array_equal(z.numpy(), [5.0, 5.0, 6.0])

    def test_rdiv(self):
        """Test reverse division."""
        x = array([1.0, 2.0, 4.0])
        z = 8.0 / x

        np.testing.assert_array_equal(z.numpy(), [8.0, 4.0, 2.0])

    def test_neg(self):
        """Test negation."""
        x = array([1.0, -2.0, 3.0])
        z = -x

        np.testing.assert_array_equal(z.numpy(), [-1.0, 2.0, -3.0])

    def test_power(self):
        """Test power operation."""
        x = array([2.0, 3.0, 4.0])
        z = x ** 2

        np.testing.assert_array_equal(z.numpy(), [4.0, 9.0, 16.0])


class TestMatrixOperations:
    """Test matrix operations."""

    def test_matmul(self):
        """Test matrix multiplication."""
        x = array([[1, 2], [3, 4]], dtype=np.float32)
        y = array([[5, 6], [7, 8]], dtype=np.float32)

        z = x @ y
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(z.numpy(), expected)

    def test_matmul_shapes(self):
        """Test matrix multiplication with different shapes."""
        x = randn(32, 64)
        y = randn(64, 128)

        z = x @ y
        assert z.shape == (32, 128)

    def test_matmul_batch(self):
        """Test batched matrix multiplication."""
        x = randn(10, 32, 64)
        y = randn(10, 64, 128)

        z = matmul(x, y)
        assert z.shape == (10, 32, 128)

    def test_transpose(self):
        """Test transpose."""
        x = array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        z = x.T

        assert z.shape == (3, 2)
        np.testing.assert_array_equal(z.numpy(), np.array([[1, 4], [2, 5], [3, 6]]))

    def test_transpose_explicit(self):
        """Test explicit transpose with axes."""
        x = randn(2, 3, 4)
        z = x.transpose(2, 0, 1)

        assert z.shape == (4, 2, 3)


class TestTranscendentalOperations:
    """Test transcendental math operations."""

    def test_exp(self):
        """Test exponential."""
        x = array([0, 1, 2], dtype=np.float32)
        z = exp(x)

        expected = np.exp([0, 1, 2]).astype(np.float32)
        np.testing.assert_array_almost_equal(z.numpy(), expected)

    def test_log(self):
        """Test logarithm."""
        x = array([1, np.e, np.e**2], dtype=np.float32)
        z = log(x)

        expected = np.array([0, 1, 2], dtype=np.float32)
        np.testing.assert_array_almost_equal(z.numpy(), expected, decimal=5)

    def test_sqrt(self):
        """Test square root."""
        x = array([1, 4, 9, 16], dtype=np.float32)
        z = sqrt(x)

        expected = np.array([1, 2, 3, 4], dtype=np.float32)
        np.testing.assert_array_equal(z.numpy(), expected)

    def test_sin(self):
        """Test sine."""
        x = array([0, np.pi/2, np.pi], dtype=np.float32)
        z = sin(x)

        expected = np.sin([0, np.pi/2, np.pi]).astype(np.float32)
        np.testing.assert_array_almost_equal(z.numpy(), expected, decimal=5)

    def test_cos(self):
        """Test cosine."""
        x = array([0, np.pi/2, np.pi], dtype=np.float32)
        z = cos(x)

        expected = np.cos([0, np.pi/2, np.pi]).astype(np.float32)
        np.testing.assert_array_almost_equal(z.numpy(), expected, decimal=5)

    def test_tanh(self):
        """Test hyperbolic tangent."""
        x = array([-1, 0, 1], dtype=np.float32)
        z = tanh(x)

        expected = np.tanh([-1, 0, 1]).astype(np.float32)
        np.testing.assert_array_almost_equal(z.numpy(), expected)


class TestReductions:
    """Test reduction operations."""

    def test_sum_global(self):
        """Test global sum."""
        x = array([[1, 2], [3, 4]], dtype=np.float32)
        z = x.sum()

        np.testing.assert_almost_equal(z.numpy(), 10.0)

    def test_sum_axis(self):
        """Test sum along axis."""
        x = array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        z0 = x.sum(axis=0)
        np.testing.assert_array_equal(z0.numpy(), [5, 7, 9])

        z1 = x.sum(axis=1)
        np.testing.assert_array_equal(z1.numpy(), [6, 15])

    def test_sum_keepdims(self):
        """Test sum with keepdims."""
        x = array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        z = x.sum(axis=1, keepdims=True)

        assert z.shape == (2, 1)
        np.testing.assert_array_equal(z.numpy(), [[6], [15]])

    def test_mean(self):
        """Test mean."""
        x = array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        z = x.mean()

        np.testing.assert_almost_equal(z.numpy(), 3.5)

    def test_mean_axis(self):
        """Test mean along axis."""
        x = array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        z = x.mean(axis=0)

        np.testing.assert_array_almost_equal(z.numpy(), [2.5, 3.5, 4.5])

    def test_max(self):
        """Test max."""
        x = array([[1, 5, 3], [4, 2, 6]], dtype=np.float32)
        z = x.max()

        np.testing.assert_equal(z.numpy(), 6)

    def test_max_axis(self):
        """Test max along axis."""
        x = array([[1, 5, 3], [4, 2, 6]], dtype=np.float32)

        z0 = x.max(axis=0)
        np.testing.assert_array_equal(z0.numpy(), [4, 5, 6])

        z1 = x.max(axis=1)
        np.testing.assert_array_equal(z1.numpy(), [5, 6])

    def test_min(self):
        """Test min."""
        x = array([[1, 5, 3], [4, 2, 6]], dtype=np.float32)
        z = x.min()

        np.testing.assert_equal(z.numpy(), 1)


class TestShapeOperations:
    """Test shape manipulation operations."""

    def test_reshape(self):
        """Test reshape."""
        x = array(np.arange(12), dtype=np.float32)
        z = x.reshape(3, 4)

        assert z.shape == (3, 4)

    def test_reshape_tuple(self):
        """Test reshape with tuple."""
        x = array(np.arange(24), dtype=np.float32)
        z = x.reshape((2, 3, 4))

        assert z.shape == (2, 3, 4)

    def test_reshape_with_infer(self):
        """Test reshape with -1 for inference."""
        x = array(np.arange(24), dtype=np.float32)
        z = x.reshape(4, -1)

        # The implementation stores the shape spec as-is
        # and resolution happens at materialize time
        result = z.numpy()
        assert result.shape == (4, 6)

    def test_flatten(self):
        """Test flatten."""
        x = randn(2, 3, 4)
        z = x.flatten()

        # Flatten uses reshape(-1), result shape resolved at materialize time
        result = z.numpy()
        assert result.shape == (24,)

    def test_squeeze(self):
        """Test squeeze."""
        x = randn(1, 3, 1, 4)
        z = x.squeeze()

        assert z.shape == (3, 4)

    def test_squeeze_axis(self):
        """Test squeeze specific axis."""
        x = randn(1, 3, 1, 4)
        z = x.squeeze(axis=0)

        assert z.shape == (3, 1, 4)

    def test_expand_dims(self):
        """Test expand_dims."""
        x = randn(3, 4)
        z = expand_dims(x, axis=0)

        assert z.shape == (1, 3, 4)

    def test_broadcast_to(self):
        """Test broadcast_to."""
        x = array([1, 2, 3], dtype=np.float32)
        z = broadcast_to(x, (4, 3))

        assert z.shape == (4, 3)
        expected = np.array([[1, 2, 3]] * 4)
        np.testing.assert_array_equal(z.numpy(), expected)


class TestNeuralNetworkOperations:
    """Test neural network operations."""

    def test_relu(self):
        """Test ReLU activation."""
        x = array([-2, -1, 0, 1, 2], dtype=np.float32)
        z = relu(x)

        np.testing.assert_array_equal(z.numpy(), [0, 0, 0, 1, 2])

    def test_sigmoid(self):
        """Test sigmoid activation."""
        x = array([0], dtype=np.float32)
        z = sigmoid(x)

        np.testing.assert_array_almost_equal(z.numpy(), [0.5])

    def test_sigmoid_range(self):
        """Test sigmoid output range."""
        x = randn(100)
        z = sigmoid(x)

        assert z.numpy().min() >= 0
        assert z.numpy().max() <= 1

    def test_softmax(self):
        """Test softmax activation."""
        x = array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
        z = softmax(x, axis=-1)

        # Sum along last axis should be 1
        sums = z.numpy().sum(axis=-1)
        np.testing.assert_array_almost_equal(sums, [1.0, 1.0])

    def test_softmax_numerical_stability(self):
        """Test softmax with large values."""
        x = array([1000, 1001, 1002], dtype=np.float32)
        z = softmax(x)

        # Should not overflow
        assert not np.any(np.isnan(z.numpy()))
        assert not np.any(np.isinf(z.numpy()))
        np.testing.assert_almost_equal(z.numpy().sum(), 1.0)


class TestBroadcasting:
    """Test broadcasting behavior."""

    def test_broadcast_add_scalar(self):
        """Test broadcasting with scalar."""
        x = array([[1, 2], [3, 4]], dtype=np.float32)
        z = x + 10

        np.testing.assert_array_equal(z.numpy(), [[11, 12], [13, 14]])

    def test_broadcast_add_row(self):
        """Test broadcasting row vector."""
        x = array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = array([10, 20, 30], dtype=np.float32)
        z = x + y

        expected = [[11, 22, 33], [14, 25, 36]]
        np.testing.assert_array_equal(z.numpy(), expected)

    def test_broadcast_add_column(self):
        """Test broadcasting column vector."""
        x = array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        y = array([[10], [20]], dtype=np.float32)
        z = x + y

        expected = [[11, 12, 13], [24, 25, 26]]
        np.testing.assert_array_equal(z.numpy(), expected)


class TestChainingOperations:
    """Test chaining multiple operations."""

    def test_chain_arithmetic(self):
        """Test chaining arithmetic operations."""
        x = array([1, 2, 3], dtype=np.float32)

        z = ((x + 1) * 2 - 3) / 2
        expected = [(1+1)*2 - 3, (2+1)*2 - 3, (3+1)*2 - 3]
        expected = [e / 2 for e in expected]

        np.testing.assert_array_almost_equal(z.numpy(), expected)

    def test_chain_lazy(self):
        """Test that chained operations are lazy."""
        x = randn(10, 10)
        y = randn(10, 10)

        # Create chain of operations
        z = ((x + y) * 2 - 1) / 2

        # Should still be lazy
        assert not z.is_materialized

        # Now materialize
        result = z.numpy()
        assert z.is_materialized

    def test_mlp_forward(self):
        """Test simple MLP forward pass."""
        batch_size = 32
        in_features = 64
        hidden = 128
        out_features = 10

        # Create inputs and weights
        x = randn(batch_size, in_features)
        w1 = randn(in_features, hidden)
        b1 = zeros((hidden,))
        w2 = randn(hidden, out_features)
        b2 = zeros((out_features,))

        # Forward pass
        h = x @ w1 + b1
        h = relu(h)
        out = h @ w2 + b2
        out = softmax(out)

        assert out.shape == (batch_size, out_features)
        # Softmax outputs should sum to 1
        sums = out.numpy().sum(axis=-1)
        np.testing.assert_array_almost_equal(sums, np.ones(batch_size), decimal=5)


class TestDeviceAndSpec:
    """Test Device and TensorSpec classes."""

    def test_device_creation(self):
        """Test creating devices."""
        device = Device(DeviceType.GPU, 0)

        assert device.device_type == DeviceType.GPU
        assert device.device_id == 0
        assert str(device) == "gpu:0"

    def test_device_hash(self):
        """Test device hashing for dict keys."""
        d1 = Device(DeviceType.GPU, 0)
        d2 = Device(DeviceType.GPU, 0)
        d3 = Device(DeviceType.GPU, 1)

        assert hash(d1) == hash(d2)
        assert hash(d1) != hash(d3)

    def test_shape_spec(self):
        """Test ShapeSpec."""
        spec = ShapeSpec((32, 64, 128))

        assert spec.rank == 3
        assert spec.numel == 32 * 64 * 128
        assert spec[0] == 32
        assert len(spec) == 3

    def test_shape_spec_dynamic(self):
        """Test ShapeSpec with dynamic dimensions."""
        spec = ShapeSpec((32, -1, 128))

        assert spec.rank == 3
        assert spec.numel == -1  # Unknown due to dynamic dim

    def test_tracer(self):
        """Test Tracer for symbolic execution."""
        spec = TensorSpec(ShapeSpec((32, 64)), np.float32)
        tracer = Tracer(spec, "input_0")

        assert tracer.shape == (32, 64)
        assert tracer.dtype == np.float32
        assert tracer.name == "input_0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
