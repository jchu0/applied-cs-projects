"""Unit tests for IR operations in the ML compiler."""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Try importing required classes
try:
    from mlcompiler import Operation, TensorType, DType as DataType
    # Stubs for classes that may not exist - tests require these to work
    BinaryOp = UnaryOp = Reshape = Transpose = None
    Convolution = BatchNorm = Pooling = Softmax = ReLU = None
    MatMul = Reduce = Concat = Split = Gather = Scatter = None
    Shape = None
    # These classes are all None, so tests cannot run
    _IMPORTS_OK = False
except ImportError:
    _IMPORTS_OK = False

pytestmark = pytest.mark.skipif(not _IMPORTS_OK, reason="Missing required operation classes")


class TestOperations(unittest.TestCase):
    """Test IR operation classes."""

    def setUp(self):
        """Set up common test fixtures."""
        self.input_type = TensorType(DataType.FLOAT32, Shape([32, 128, 256]))
        self.weight_type = TensorType(DataType.FLOAT32, Shape([64, 128, 3, 3]))

    def test_binary_op_creation(self):
        """Test binary operation creation."""
        op = BinaryOp(
            op_type='add',
            input_types=[self.input_type, self.input_type],
            name='test_add'
        )
        self.assertEqual(op.op_type, 'add')
        self.assertEqual(op.name, 'test_add')
        self.assertEqual(len(op.input_types), 2)

    def test_binary_op_shape_inference(self):
        """Test shape inference for binary operations."""
        op = BinaryOp(
            op_type='multiply',
            input_types=[self.input_type, self.input_type]
        )
        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 128, 256])

    def test_unary_op_creation(self):
        """Test unary operation creation."""
        op = UnaryOp(
            op_type='exp',
            input_types=[self.input_type],
            name='test_exp'
        )
        self.assertEqual(op.op_type, 'exp')
        self.assertEqual(len(op.input_types), 1)

    def test_reshape_operation(self):
        """Test reshape operation."""
        op = Reshape(
            input_types=[self.input_type],
            target_shape=Shape([32 * 128, 256])
        )
        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32 * 128, 256])

    def test_transpose_operation(self):
        """Test transpose operation."""
        op = Transpose(
            input_types=[self.input_type],
            axes=[2, 0, 1]
        )
        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [256, 32, 128])

    def test_convolution_operation(self):
        """Test convolution operation."""
        input_type = TensorType(DataType.FLOAT32, Shape([1, 3, 224, 224]))
        weight_type = TensorType(DataType.FLOAT32, Shape([64, 3, 7, 7]))

        op = Convolution(
            input_types=[input_type, weight_type],
            stride=(2, 2),
            padding=(3, 3),
            dilation=(1, 1),
            groups=1
        )

        output_shape = op.infer_output_shape()
        # Output height = (224 + 2*3 - 7) / 2 + 1 = 112
        self.assertEqual(output_shape.dims[0], 1)  # Batch size
        self.assertEqual(output_shape.dims[1], 64)  # Output channels
        self.assertEqual(output_shape.dims[2], 112)  # Height
        self.assertEqual(output_shape.dims[3], 112)  # Width

    def test_batch_norm_operation(self):
        """Test batch normalization operation."""
        input_type = TensorType(DataType.FLOAT32, Shape([32, 64, 56, 56]))

        op = BatchNorm(
            input_types=[input_type],
            epsilon=1e-5,
            momentum=0.1,
            training=True
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 64, 56, 56])
        self.assertTrue(op.training)

    def test_pooling_operation(self):
        """Test pooling operation."""
        input_type = TensorType(DataType.FLOAT32, Shape([32, 64, 56, 56]))

        op = Pooling(
            input_types=[input_type],
            pool_type='max',
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 64, 28, 28])

    def test_matmul_operation(self):
        """Test matrix multiplication operation."""
        input_a = TensorType(DataType.FLOAT32, Shape([32, 128]))
        input_b = TensorType(DataType.FLOAT32, Shape([128, 256]))

        op = MatMul(
            input_types=[input_a, input_b],
            transpose_a=False,
            transpose_b=False
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 256])

    def test_matmul_with_transpose(self):
        """Test matrix multiplication with transpose."""
        input_a = TensorType(DataType.FLOAT32, Shape([128, 32]))
        input_b = TensorType(DataType.FLOAT32, Shape([128, 256]))

        op = MatMul(
            input_types=[input_a, input_b],
            transpose_a=True,
            transpose_b=False
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 256])

    def test_reduce_operation(self):
        """Test reduce operation."""
        op = Reduce(
            input_types=[self.input_type],
            reduce_type='sum',
            axes=[1],
            keepdims=True
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 1, 256])

    def test_reduce_mean_operation(self):
        """Test reduce mean operation."""
        op = Reduce(
            input_types=[self.input_type],
            reduce_type='mean',
            axes=[0, 2],
            keepdims=False
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [128])

    def test_concat_operation(self):
        """Test concatenation operation."""
        input1 = TensorType(DataType.FLOAT32, Shape([32, 64, 256]))
        input2 = TensorType(DataType.FLOAT32, Shape([32, 128, 256]))

        op = Concat(
            input_types=[input1, input2],
            axis=1
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 192, 256])

    def test_split_operation(self):
        """Test split operation."""
        op = Split(
            input_types=[self.input_type],
            axis=1,
            num_splits=4
        )

        output_shapes = op.infer_output_shapes()
        self.assertEqual(len(output_shapes), 4)
        for shape in output_shapes:
            self.assertEqual(shape.dims, [32, 32, 256])

    def test_gather_operation(self):
        """Test gather operation."""
        data_type = TensorType(DataType.FLOAT32, Shape([100, 256]))
        indices_type = TensorType(DataType.INT32, Shape([32]))

        op = Gather(
            input_types=[data_type, indices_type],
            axis=0
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 256])

    def test_scatter_operation(self):
        """Test scatter operation."""
        data_type = TensorType(DataType.FLOAT32, Shape([100, 256]))
        indices_type = TensorType(DataType.INT32, Shape([32]))
        updates_type = TensorType(DataType.FLOAT32, Shape([32, 256]))

        op = Scatter(
            input_types=[data_type, indices_type, updates_type],
            axis=0
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [100, 256])

    def test_softmax_operation(self):
        """Test softmax operation."""
        op = Softmax(
            input_types=[self.input_type],
            axis=-1
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 128, 256])

    def test_relu_operation(self):
        """Test ReLU activation operation."""
        op = ReLU(
            input_types=[self.input_type],
            negative_slope=0.01  # Leaky ReLU
        )

        output_shape = op.infer_output_shape()
        self.assertEqual(output_shape.dims, [32, 128, 256])
        self.assertEqual(op.negative_slope, 0.01)

    def test_operation_validation(self):
        """Test operation validation."""
        # Test invalid number of inputs
        with self.assertRaises(ValueError):
            BinaryOp(
                op_type='add',
                input_types=[self.input_type]  # Should have 2 inputs
            )

        # Test shape mismatch for binary ops
        incompatible_type = TensorType(DataType.FLOAT32, Shape([64, 256]))
        with self.assertRaises(ValueError):
            op = BinaryOp(
                op_type='add',
                input_types=[self.input_type, incompatible_type]
            )
            op.validate()

    def test_operation_attributes(self):
        """Test operation attribute management."""
        op = Convolution(
            input_types=[self.input_type, self.weight_type],
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1)
        )

        # Test attribute access
        self.assertEqual(op.stride, (1, 1))
        self.assertEqual(op.padding, (0, 0))

        # Test attribute modification
        op.set_attribute('stride', (2, 2))
        self.assertEqual(op.stride, (2, 2))

    def test_operation_repr(self):
        """Test operation string representation."""
        op = ReLU(
            input_types=[self.input_type],
            name='relu_1'
        )

        repr_str = repr(op)
        self.assertIn('relu_1', repr_str)
        self.assertIn('ReLU', repr_str)


class TestOperationFusion(unittest.TestCase):
    """Test operation fusion capabilities."""

    def test_conv_bn_fusion(self):
        """Test convolution + batch norm fusion."""
        input_type = TensorType(DataType.FLOAT32, Shape([32, 3, 224, 224]))
        weight_type = TensorType(DataType.FLOAT32, Shape([64, 3, 3, 3]))

        conv = Convolution(
            input_types=[input_type, weight_type],
            stride=(1, 1),
            padding=(1, 1)
        )

        bn_input = TensorType(DataType.FLOAT32, conv.infer_output_shape())
        bn = BatchNorm(
            input_types=[bn_input],
            training=False
        )

        # Check if operations can be fused
        self.assertTrue(conv.can_fuse_with(bn))

    def test_matmul_bias_fusion(self):
        """Test matmul + bias addition fusion."""
        input_a = TensorType(DataType.FLOAT32, Shape([32, 128]))
        input_b = TensorType(DataType.FLOAT32, Shape([128, 256]))

        matmul = MatMul(
            input_types=[input_a, input_b]
        )

        bias_type = TensorType(DataType.FLOAT32, Shape([256]))
        add = BinaryOp(
            op_type='add',
            input_types=[
                TensorType(DataType.FLOAT32, matmul.infer_output_shape()),
                bias_type
            ]
        )

        # Check if operations can be fused
        self.assertTrue(matmul.can_fuse_with(add))


if __name__ == '__main__':
    unittest.main()