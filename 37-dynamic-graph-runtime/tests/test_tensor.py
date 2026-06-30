"""Tests for symbolic tensor representation."""

import unittest
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dynamicgraph.core.tensor import (
    SymbolicTensor, TensorMetadata, TensorFactory
)


class TestTensorMetadata(unittest.TestCase):
    """Tests for TensorMetadata."""

    def test_metadata_creation(self):
        """Test basic metadata creation."""
        metadata = TensorMetadata(
            dtype=np.float32,
            shape=(10, 20, 30),
            device="cuda:0",
            requires_grad=True,
            is_parameter=True
        )

        self.assertEqual(metadata.dtype, np.float32)
        self.assertEqual(metadata.shape, (10, 20, 30))
        self.assertEqual(metadata.device, "cuda:0")
        self.assertTrue(metadata.requires_grad)
        self.assertTrue(metadata.is_parameter)

    def test_metadata_defaults(self):
        """Test metadata default values."""
        metadata = TensorMetadata()

        self.assertIsNone(metadata.dtype)
        self.assertIsNone(metadata.shape)
        self.assertEqual(metadata.device, "cpu")
        self.assertFalse(metadata.requires_grad)
        self.assertFalse(metadata.is_parameter)
        self.assertFalse(metadata.is_buffer)


class TestSymbolicTensor(unittest.TestCase):
    """Tests for SymbolicTensor class."""

    def test_tensor_creation(self):
        """Test basic tensor creation."""
        tensor = SymbolicTensor()
        self.assertIsNotNone(tensor.node_id)
        self.assertIsNotNone(tensor.metadata)
        self.assertIsNone(tensor.concrete_value)
        self.assertIsNone(tensor.source)

    def test_tensor_with_concrete_value(self):
        """Test tensor with concrete NumPy value."""
        arr = np.random.randn(3, 4, 5).astype(np.float32)
        tensor = SymbolicTensor(concrete_value=arr)

        self.assertTrue(tensor.is_concrete())
        self.assertFalse(tensor.is_symbolic())
        self.assertEqual(tensor.shape, (3, 4, 5))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(tensor.concrete_value, arr)

    def test_tensor_properties(self):
        """Test tensor property accessors."""
        metadata = TensorMetadata(
            dtype=np.float64,
            shape=(2, 3, 4),
            device="cuda:1",
            requires_grad=True
        )
        tensor = SymbolicTensor(metadata=metadata)

        self.assertEqual(tensor.dtype, np.float64)
        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertEqual(tensor.device, "cuda:1")
        self.assertTrue(tensor.requires_grad)

        # Test numel and ndim
        self.assertEqual(tensor.numel(), 24)
        self.assertEqual(tensor.ndim(), 3)

        # Test size
        self.assertEqual(tensor.size(), (2, 3, 4))
        self.assertEqual(tensor.size(0), 2)
        self.assertEqual(tensor.size(1), 3)
        self.assertEqual(tensor.size(2), 4)
        self.assertEqual(tensor.size(-1), 4)

    def test_gradient_operations(self):
        """Test gradient-related operations."""
        tensor = SymbolicTensor()

        # Initially no gradient requirement
        self.assertFalse(tensor.requires_grad)
        self.assertIsNone(tensor.grad)

        # Set requires_grad
        tensor.requires_grad = True
        self.assertTrue(tensor.requires_grad)

        # Set gradient
        grad_tensor = SymbolicTensor()
        tensor.grad = grad_tensor
        self.assertEqual(tensor.grad, grad_tensor)

        # Cannot set gradient without requires_grad
        tensor2 = SymbolicTensor()
        with self.assertRaises(RuntimeError):
            tensor2.grad = grad_tensor

    def test_detach_operation(self):
        """Test detach operation."""
        metadata = TensorMetadata(
            dtype=np.float32,
            shape=(10, 20),
            requires_grad=True,
            is_parameter=True
        )
        arr = np.random.randn(10, 20).astype(np.float32)
        tensor = SymbolicTensor(
            node_id="original",
            metadata=metadata,
            concrete_value=arr,
            source="input"
        )

        # Detach
        detached = tensor.detach()

        self.assertNotEqual(detached.node_id, tensor.node_id)
        self.assertFalse(detached.requires_grad)
        self.assertFalse(detached.metadata.is_parameter)
        self.assertEqual(detached.shape, tensor.shape)
        self.assertEqual(detached.dtype, tensor.dtype)

        if tensor.concrete_value is not None:
            np.testing.assert_array_equal(detached.concrete_value, tensor.concrete_value)
            # Verify it's a copy
            self.assertIsNot(detached.concrete_value, tensor.concrete_value)

    def test_clone_operation(self):
        """Test clone operation."""
        metadata = TensorMetadata(
            dtype=np.float32,
            shape=(5, 5),
            requires_grad=True,
            is_parameter=True
        )
        tensor = SymbolicTensor(
            node_id="original",
            metadata=metadata,
            source="weight"
        )

        # Clone
        cloned = tensor.clone()

        self.assertNotEqual(cloned.node_id, tensor.node_id)
        self.assertEqual(cloned.requires_grad, tensor.requires_grad)
        self.assertEqual(cloned.metadata.is_parameter, tensor.metadata.is_parameter)
        self.assertEqual(cloned.shape, tensor.shape)
        self.assertEqual(cloned.dtype, tensor.dtype)

    def test_device_operations(self):
        """Test device movement operations."""
        tensor = SymbolicTensor()
        self.assertEqual(tensor.device, "cpu")

        # Move to GPU
        gpu_tensor = tensor.to("cuda:0")
        self.assertEqual(gpu_tensor.device, "cuda:0")
        self.assertNotEqual(gpu_tensor.node_id, tensor.node_id)

        # Original tensor unchanged
        self.assertEqual(tensor.device, "cpu")

    def test_arithmetic_operations(self):
        """Test arithmetic operator overloads."""
        metadata1 = TensorMetadata(
            dtype=np.float32,
            shape=(3, 4),
            requires_grad=True
        )
        metadata2 = TensorMetadata(
            dtype=np.float32,
            shape=(3, 4),
            requires_grad=False
        )

        tensor1 = SymbolicTensor(metadata=metadata1, source="t1")
        tensor2 = SymbolicTensor(metadata=metadata2, source="t2")

        # Addition
        result = tensor1 + tensor2
        self.assertEqual(result.shape, (3, 4))
        self.assertTrue(result.requires_grad)  # Propagates from tensor1

        # Subtraction
        result = tensor1 - tensor2
        self.assertEqual(result.shape, (3, 4))

        # Multiplication
        result = tensor1 * tensor2
        self.assertEqual(result.shape, (3, 4))

        # Division
        result = tensor1 / tensor2
        self.assertEqual(result.shape, (3, 4))

        # Scalar operations
        result = tensor1 + 5.0
        self.assertEqual(result.shape, (3, 4))

        result = 2.0 * tensor1
        self.assertEqual(result.shape, (3, 4))

    def test_broadcasting(self):
        """Test shape broadcasting in operations."""
        # Test basic broadcasting rules
        test_cases = [
            ((3, 4), (3, 4), (3, 4)),  # Same shape
            ((3, 4), (1, 4), (3, 4)),  # Broadcast dim 0
            ((3, 4), (3, 1), (3, 4)),  # Broadcast dim 1
            ((3, 4), (4,), (3, 4)),    # Broadcast scalar-like
            ((3, 1, 4), (3, 4), (3, 3, 4)),  # Different ranks
        ]

        for shape1, shape2, expected in test_cases:
            result_shape = SymbolicTensor._broadcast_shapes(shape1, shape2)
            self.assertEqual(result_shape, expected,
                           f"Failed for {shape1} x {shape2}")

        # Test incompatible shapes
        with self.assertRaises(ValueError):
            SymbolicTensor._broadcast_shapes((3, 4), (5, 4))

    def test_dtype_promotion(self):
        """Test dtype promotion in operations."""
        test_cases = [
            (np.float32, np.float32, np.float32),
            (np.float32, np.float64, np.float64),
            (np.int32, np.float32, np.float64),
            (np.int32, np.int64, np.int64),
        ]

        for dtype1, dtype2, expected in test_cases:
            result = SymbolicTensor._promote_dtypes(dtype1, dtype2)
            self.assertEqual(result, expected,
                           f"Failed for {dtype1} x {dtype2}")

    def test_matrix_multiplication(self):
        """Test matrix multiplication operator."""
        tensor1 = SymbolicTensor(
            metadata=TensorMetadata(shape=(3, 4), dtype=np.float32)
        )
        tensor2 = SymbolicTensor(
            metadata=TensorMetadata(shape=(4, 5), dtype=np.float32)
        )

        result = tensor1 @ tensor2
        self.assertEqual(result.shape, (3, 5))

    def test_unary_operations(self):
        """Test unary operations."""
        tensor = SymbolicTensor(
            metadata=TensorMetadata(
                shape=(3, 4),
                dtype=np.float32,
                requires_grad=True
            )
        )

        # Negation
        neg_tensor = -tensor
        self.assertEqual(neg_tensor.shape, tensor.shape)
        self.assertEqual(neg_tensor.dtype, tensor.dtype)
        self.assertEqual(neg_tensor.requires_grad, tensor.requires_grad)


class TestTensorFactory(unittest.TestCase):
    """Tests for TensorFactory."""

    def test_zeros(self):
        """Test creating zero tensors."""
        tensor = TensorFactory.zeros(
            shape=(2, 3, 4),
            dtype=np.float64,
            device="cuda:0",
            requires_grad=True
        )

        self.assertEqual(tensor.shape, (2, 3, 4))
        self.assertEqual(tensor.dtype, np.float64)
        self.assertEqual(tensor.device, "cuda:0")
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(tensor.is_concrete())

        np.testing.assert_array_equal(
            tensor.concrete_value,
            np.zeros((2, 3, 4), dtype=np.float64)
        )

    def test_ones(self):
        """Test creating ones tensors."""
        tensor = TensorFactory.ones(
            shape=(5, 5),
            dtype=np.float32
        )

        self.assertEqual(tensor.shape, (5, 5))
        self.assertEqual(tensor.dtype, np.float32)
        np.testing.assert_array_equal(
            tensor.concrete_value,
            np.ones((5, 5), dtype=np.float32)
        )

    def test_randn(self):
        """Test creating random normal tensors."""
        # Use larger size for more stable statistics
        tensor = TensorFactory.randn(
            shape=(100, 100),
            dtype=np.float32,
            requires_grad=True
        )

        self.assertEqual(tensor.shape, (100, 100))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.requires_grad)
        self.assertTrue(tensor.is_concrete())

        # Check that values are roughly normal (with larger sample)
        mean = np.mean(tensor.concrete_value)
        std = np.std(tensor.concrete_value)
        self.assertAlmostEqual(mean, 0.0, places=1)
        self.assertAlmostEqual(std, 1.0, places=1)

    def test_from_numpy(self):
        """Test creating tensor from NumPy array."""
        arr = np.random.randn(3, 4, 5).astype(np.float32)
        tensor = TensorFactory.from_numpy(
            arr,
            device="cuda:1",
            requires_grad=True
        )

        self.assertEqual(tensor.shape, arr.shape)
        self.assertEqual(tensor.dtype, arr.dtype)
        self.assertEqual(tensor.device, "cuda:1")
        self.assertTrue(tensor.requires_grad)
        np.testing.assert_array_equal(tensor.concrete_value, arr)

        # Verify it's a copy
        arr[0, 0, 0] = 999
        self.assertNotEqual(tensor.concrete_value[0, 0, 0], 999)


class TestSymbolicOperations(unittest.TestCase):
    """Tests for symbolic operations without concrete values."""

    def test_pure_symbolic_tensors(self):
        """Test operations on purely symbolic tensors."""
        # Create symbolic tensors without concrete values
        t1 = SymbolicTensor(
            metadata=TensorMetadata(shape=(10, 20), dtype=np.float32)
        )
        t2 = SymbolicTensor(
            metadata=TensorMetadata(shape=(20, 30), dtype=np.float32)
        )

        self.assertTrue(t1.is_symbolic())
        self.assertFalse(t1.is_concrete())

        # Operations should still work symbolically
        t3 = t1 @ t2  # Matrix multiplication
        self.assertEqual(t3.shape, (10, 30))
        self.assertEqual(t3.dtype, np.float32)

    def test_mixed_symbolic_concrete(self):
        """Test operations mixing symbolic and concrete tensors."""
        symbolic = SymbolicTensor(
            metadata=TensorMetadata(shape=(3, 4), dtype=np.float32)
        )
        concrete = TensorFactory.ones((3, 4), dtype=np.float32)

        # Operations should work
        result = symbolic + concrete
        self.assertEqual(result.shape, (3, 4))

        result = concrete * symbolic
        self.assertEqual(result.shape, (3, 4))

    def test_shape_inference(self):
        """Test shape inference in symbolic operations."""
        t1 = SymbolicTensor(metadata=TensorMetadata(shape=(5, 10)))
        t2 = SymbolicTensor(metadata=TensorMetadata(shape=(10, 15)))
        t3 = SymbolicTensor(metadata=TensorMetadata(shape=(15, 20)))

        # Chain matrix multiplications
        result = t1 @ t2 @ t3
        self.assertEqual(result.shape, (5, 20))

        # Test with broadcasting
        t4 = SymbolicTensor(metadata=TensorMetadata(shape=(1, 20)))
        result = result + t4
        self.assertEqual(result.shape, (5, 20))

    def test_unknown_shapes(self):
        """Test handling of unknown shapes."""
        # Tensor with unknown shape
        t1 = SymbolicTensor(metadata=TensorMetadata(dtype=np.float32))
        self.assertIsNone(t1.shape)
        self.assertIsNone(t1.numel())
        self.assertIsNone(t1.ndim())

        # Operations with unknown shapes
        t2 = SymbolicTensor(metadata=TensorMetadata(shape=(10, 20)))
        result = t1 + t2

        # Result shape is unknown when one operand has unknown shape
        self.assertIsNone(result.shape)


if __name__ == "__main__":
    unittest.main()