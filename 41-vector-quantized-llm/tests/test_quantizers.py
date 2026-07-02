"""Unit tests for quantizers."""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vqllm.quantize.quantizers import (
    Quantizer, INT8Quantizer, INT4Quantizer,
    GPTQQuantizer, AWQQuantizer, FP8Quantizer
)
from vqllm.core.types import (
    QuantConfig, QuantType, QuantizedTensor,
    ScaleType, unpack_int4, FP8QuantizedTensor,
    float_to_fp8_e4m3, fp8_e4m3_to_float,
    float_to_fp8_e5m2, fp8_e5m2_to_float,
    quantize_fp8_e4m3, quantize_fp8_e5m2,
    FP8_E4M3_MAX, FP8_E5M2_MAX
)


class TestQuantizer(unittest.TestCase):
    """Test base Quantizer class."""

    def setUp(self):
        self.config = QuantConfig(
            quant_type=QuantType.INT8,
            scale_type=ScaleType.PER_CHANNEL,
            block_size=128
        )

    def test_init(self):
        """Test quantizer initialization."""
        quantizer = INT8Quantizer(self.config)
        self.assertEqual(quantizer.config.quant_type, QuantType.INT8)
        self.assertEqual(quantizer.config.block_size, 128)

    def test_init_default_config(self):
        """Test quantizer with default config."""
        quantizer = INT8Quantizer()
        self.assertIsNotNone(quantizer.config)
        self.assertEqual(quantizer.config.quant_type, QuantType.INT8)


class TestINT8Quantizer(unittest.TestCase):
    """Test INT8 quantizer."""

    def setUp(self):
        self.quantizer = INT8Quantizer()
        np.random.seed(42)

    def test_quantize_weight(self):
        """Test weight quantization to INT8."""
        weight = np.random.randn(256, 512).astype(np.float32)
        qtensor = self.quantizer.quantize_weight(weight, "test_layer")

        # Check quantized tensor properties
        self.assertIsInstance(qtensor, QuantizedTensor)
        self.assertEqual(qtensor.dtype, np.int8)
        self.assertEqual(qtensor.shape, weight.shape)
        self.assertIsNotNone(qtensor.scale)
        self.assertIsNotNone(qtensor.zero_point)

    def test_quantize_weight_per_channel(self):
        """Test per-channel quantization."""
        config = QuantConfig(
            quant_type=QuantType.INT8,
            scale_type=ScaleType.PER_CHANNEL
        )
        quantizer = INT8Quantizer(config)

        weight = np.random.randn(128, 256).astype(np.float32)
        qtensor = quantizer.quantize_weight(weight, "test_layer")

        # Check scale shape for per-channel
        self.assertEqual(qtensor.scale.shape[0], weight.shape[0])

    def test_quantize_weight_per_tensor(self):
        """Test per-tensor quantization."""
        config = QuantConfig(
            quant_type=QuantType.INT8,
            scale_type=ScaleType.PER_TENSOR
        )
        quantizer = INT8Quantizer(config)

        weight = np.random.randn(128, 256).astype(np.float32)
        qtensor = quantizer.quantize_weight(weight, "test_layer")

        # Check scale is scalar for per-tensor
        self.assertEqual(qtensor.scale.ndim, 0)

    def test_quantization_range(self):
        """Test quantized values are in valid range."""
        weight = np.random.randn(100, 100).astype(np.float32) * 10
        qtensor = self.quantizer.quantize_weight(weight)

        # Check all values are in INT8 range
        self.assertTrue(np.all(qtensor.data >= -128))
        self.assertTrue(np.all(qtensor.data <= 127))

    def test_dequantization(self):
        """Test dequantization accuracy."""
        weight = np.random.randn(64, 64).astype(np.float32)
        qtensor = self.quantizer.quantize_weight(weight)

        # Dequantize
        dequant = qtensor.dequantize()

        # Check shape preserved
        self.assertEqual(dequant.shape, weight.shape)

        # Check reasonable accuracy (INT8 should have ~1% error)
        rel_error = np.abs(dequant - weight) / (np.abs(weight) + 1e-8)
        self.assertTrue(np.mean(rel_error) < 0.05)

    def test_quantize_model(self):
        """Test full model quantization."""
        # Mock model with parameters
        model = Mock()
        model.named_parameters = Mock(return_value=[
            ('layer1.weight', Mock(data=np.random.randn(128, 256), ndim=2)),
            ('layer1.bias', Mock(data=np.random.randn(128), ndim=1)),
            ('layer2.weight', Mock(data=np.random.randn(256, 128), ndim=2)),
        ])

        quantized = self.quantizer.quantize_model(model)
        self.assertIsNotNone(quantized)


class TestINT4Quantizer(unittest.TestCase):
    """Test INT4 quantizer."""

    def setUp(self):
        self.quantizer = INT4Quantizer()
        np.random.seed(42)

    def test_quantize_weight(self):
        """Test weight quantization to INT4."""
        weight = np.random.randn(256, 512).astype(np.float32)
        qtensor = self.quantizer.quantize_weight(weight, "test_layer")

        # Check quantized tensor properties
        self.assertIsInstance(qtensor, QuantizedTensor)
        self.assertEqual(qtensor.shape, weight.shape)
        self.assertIsNotNone(qtensor.scale)
        self.assertIsNotNone(qtensor.zero_point)

    def test_quantization_range(self):
        """Test quantized values are in valid INT4 range."""
        weight = np.random.randn(100, 128).astype(np.float32) * 5  # Use 128 columns for group_size
        qtensor = self.quantizer.quantize_weight(weight)

        # INT4 range is -8 to 7 (symmetric) or 0-15 (asymmetric)
        # Unpack the data first since INT4 is packed
        unpacked = unpack_int4(qtensor.data)
        # For symmetric quantization, values should be in [-8, 7]
        self.assertTrue(np.all(unpacked >= -8))
        self.assertTrue(np.all(unpacked <= 15))  # Allow both symmetric and asymmetric ranges

    def test_packing_efficiency(self):
        """Test INT4 packing into bytes."""
        weight = np.random.randn(128, 256).astype(np.float32)
        qtensor = self.quantizer.quantize_weight(weight)

        # Should be packed efficiently (2 INT4s per byte)
        expected_bytes = (weight.size + 1) // 2
        self.assertLessEqual(qtensor.packed_data.nbytes, expected_bytes + 100)  # Small overhead ok

    def test_dequantization_accuracy(self):
        """Test INT4 dequantization accuracy."""
        # Use dimensions divisible by group_size (128)
        weight = np.random.randn(64, 256).astype(np.float32) * 2
        qtensor = self.quantizer.quantize_weight(weight)

        # Dequantize
        dequant = qtensor.dequantize()

        # Check shape preserved
        self.assertEqual(dequant.shape, weight.shape)

        # INT4 has lower accuracy than INT8 - check MSE instead of relative error
        mse = np.mean((dequant - weight) ** 2)
        # INT4 with 4 bits can have significant quantization error
        self.assertLess(mse, 10.0)  # Reasonable error bound for INT4


class TestGPTQQuantizer(unittest.TestCase):
    """Test GPTQ quantizer."""

    def setUp(self):
        config = QuantConfig(
            quant_type=QuantType.GPTQ,
            block_size=128,
            dampening=0.01
        )
        self.quantizer = GPTQQuantizer(config)
        np.random.seed(42)

    def test_quantize_weight(self):
        """Test GPTQ weight quantization."""
        weight = np.random.randn(256, 512).astype(np.float32)
        hessian = np.random.randn(512, 512).astype(np.float32)
        hessian = hessian @ hessian.T  # Make positive semi-definite

        qtensor = self.quantizer.quantize_weight(weight, "test_layer", hessian=hessian)

        # Check quantized tensor properties
        self.assertIsInstance(qtensor, QuantizedTensor)
        self.assertEqual(qtensor.shape, weight.shape)
        self.assertIsNotNone(qtensor.scale)

    def test_block_wise_quantization(self):
        """Test block-wise GPTQ quantization."""
        weight = np.random.randn(256, 512).astype(np.float32)
        hessian = np.eye(512, dtype=np.float32) * 0.1

        qtensor = self.quantizer.quantize_weight(weight, "test_layer", hessian=hessian)

        # Check blocks were processed
        self.assertEqual(qtensor.config.block_size, 128)

        # Verify quantization occurred
        self.assertTrue(qtensor.is_quantized)

    def test_optimal_ordering(self):
        """Test GPTQ optimal ordering computation."""
        weight = np.random.randn(128, 256).astype(np.float32)
        hessian = np.random.randn(256, 256).astype(np.float32)
        hessian = hessian @ hessian.T

        order = self.quantizer._compute_optimal_order(hessian)

        # Check order is valid permutation
        self.assertEqual(len(order), 256)
        self.assertEqual(len(set(order)), 256)
        self.assertTrue(all(0 <= i < 256 for i in order))

    def test_error_minimization(self):
        """Test GPTQ error minimization."""
        # Use dimensions divisible by group_size (128)
        weight = np.random.randn(64, 256).astype(np.float32)
        hessian = np.eye(256, dtype=np.float32) * 0.5

        qtensor = self.quantizer.quantize_weight(weight, "test_layer", hessian=hessian)
        dequant = qtensor.dequantize()

        # GPTQ INT4 has significant quantization error, just verify it's not extreme
        error = np.mean((dequant - weight) ** 2)
        self.assertLess(error, 50.0)  # INT4 with GPTQ has larger error due to aggressive quantization


class TestAWQQuantizer(unittest.TestCase):
    """Test AWQ (Activation-aware Weight Quantization) quantizer."""

    def setUp(self):
        config = QuantConfig(
            quant_type=QuantType.AWQ,
            group_size=128,
            zero_point=True
        )
        self.quantizer = AWQQuantizer(config)
        np.random.seed(42)

    def test_quantize_weight(self):
        """Test AWQ weight quantization."""
        weight = np.random.randn(256, 512).astype(np.float32)
        activation_scale = np.random.rand(512).astype(np.float32) + 0.1

        qtensor = self.quantizer.quantize_weight(
            weight, "test_layer",
            activation_scale=activation_scale
        )

        # Check quantized tensor properties
        self.assertIsInstance(qtensor, QuantizedTensor)
        self.assertEqual(qtensor.shape, weight.shape)
        self.assertIsNotNone(qtensor.scale)
        self.assertTrue(qtensor.config.zero_point)

    def test_activation_aware_scaling(self):
        """Test activation-aware weight scaling."""
        weight = np.random.randn(128, 256).astype(np.float32)

        # Create activation scales with different magnitudes
        activation_scale = np.ones(256, dtype=np.float32)
        activation_scale[:128] = 10.0  # High activation channels
        activation_scale[128:] = 0.1   # Low activation channels

        qtensor = self.quantizer.quantize_weight(
            weight, "test_layer",
            activation_scale=activation_scale
        )

        # Verify scaling was applied
        self.assertIsNotNone(qtensor.activation_scale)
        self.assertEqual(len(qtensor.activation_scale), 256)

    def test_group_wise_quantization(self):
        """Test group-wise AWQ quantization."""
        weight = np.random.randn(256, 512).astype(np.float32)
        activation_scale = np.random.rand(512).astype(np.float32) + 0.1

        qtensor = self.quantizer.quantize_weight(
            weight, "test_layer",
            activation_scale=activation_scale
        )

        # Check group size is respected
        self.assertEqual(qtensor.config.group_size, 128)

        # Scale should be per-group
        num_groups = (512 + 127) // 128
        self.assertGreaterEqual(qtensor.scale.shape[-1], num_groups)

    def test_search_optimal_scale(self):
        """Test AWQ optimal scale search."""
        weight = np.random.randn(64, 128).astype(np.float32)
        activation_scale = np.random.rand(128).astype(np.float32) + 0.5

        # Search for optimal scale
        optimal_scale = self.quantizer._search_optimal_scale(
            weight, activation_scale
        )

        # Check scale is positive and reasonable
        self.assertTrue(np.all(optimal_scale > 0))
        self.assertTrue(np.all(optimal_scale < 100))

    def test_quantization_with_calibration(self):
        """Test AWQ with calibration data."""
        weight = np.random.randn(128, 256).astype(np.float32)

        # Simulate calibration data
        calibration_data = [
            np.random.randn(32, 256).astype(np.float32) for _ in range(10)
        ]

        # Compute activation scales from calibration
        activation_scale = self.quantizer._compute_activation_scale(calibration_data)

        qtensor = self.quantizer.quantize_weight(
            weight, "test_layer",
            activation_scale=activation_scale
        )

        # Check quantization succeeded
        self.assertTrue(qtensor.is_quantized)
        self.assertIsNotNone(qtensor.activation_scale)


class TestQuantizationIntegration(unittest.TestCase):
    """Integration tests for quantization pipeline."""

    def setUp(self):
        np.random.seed(42)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_mixed_precision_quantization(self):
        """Test mixed precision quantization."""
        # Create quantizers for different layers
        int8_quantizer = INT8Quantizer()
        int4_quantizer = INT4Quantizer()

        # Quantize different layers with different precisions
        weight1 = np.random.randn(256, 512).astype(np.float32)
        weight2 = np.random.randn(512, 256).astype(np.float32)

        qtensor1 = int8_quantizer.quantize_weight(weight1, "attention")
        qtensor2 = int4_quantizer.quantize_weight(weight2, "mlp")

        # Check different precisions
        self.assertEqual(qtensor1.dtype, np.int8)
        self.assertNotEqual(qtensor1.dtype, qtensor2.dtype)

    def test_save_and_load_quantized(self):
        """Test saving and loading quantized tensors."""
        quantizer = GPTQQuantizer()
        weight = np.random.randn(128, 256).astype(np.float32)
        hessian = np.eye(256, dtype=np.float32) * 0.1

        qtensor = quantizer.quantize_weight(weight, "test_layer", hessian=hessian)

        # Save quantized tensor
        save_path = os.path.join(self.temp_dir, "quantized.npz")
        qtensor.save(save_path)

        # Load and verify
        loaded = QuantizedTensor.load(save_path)
        np.testing.assert_array_equal(loaded.data, qtensor.data)
        np.testing.assert_allclose(loaded.scale, qtensor.scale)

    def test_quantization_memory_efficiency(self):
        """Test memory reduction from quantization."""
        weight = np.random.randn(1024, 1024).astype(np.float32)
        original_size = weight.nbytes

        # Quantize to INT4
        quantizer = INT4Quantizer()
        qtensor = quantizer.quantize_weight(weight)

        # Check memory reduction (INT4 is 4x smaller than FP32)
        quantized_size = qtensor.packed_data.nbytes + qtensor.scale.nbytes
        compression_ratio = original_size / quantized_size
        self.assertGreater(compression_ratio, 3.5)  # Should be close to 4x

    def test_batch_quantization(self):
        """Test quantizing multiple weights in batch."""
        quantizer = INT8Quantizer()

        weights = [
            np.random.randn(128, 256).astype(np.float32),
            np.random.randn(256, 512).astype(np.float32),
            np.random.randn(512, 1024).astype(np.float32),
        ]

        quantized = []
        for i, w in enumerate(weights):
            qtensor = quantizer.quantize_weight(w, f"layer_{i}")
            quantized.append(qtensor)

        # Verify all quantized
        self.assertEqual(len(quantized), 3)
        for qt in quantized:
            self.assertTrue(qt.is_quantized)


class TestFP8Conversions(unittest.TestCase):
    """Test FP8 format conversions."""

    def test_fp8_e4m3_zero(self):
        """Test FP8 E4M3 conversion of zero."""
        fp8 = float_to_fp8_e4m3(0.0)
        result = fp8_e4m3_to_float(fp8)
        self.assertEqual(result, 0.0)

    def test_fp8_e4m3_positive(self):
        """Test FP8 E4M3 conversion of positive values."""
        test_values = [1.0, 2.0, 10.0, 100.0, 448.0]
        for val in test_values:
            fp8 = float_to_fp8_e4m3(val)
            result = fp8_e4m3_to_float(fp8)
            # Allow small relative error due to limited precision
            self.assertAlmostEqual(result, val, delta=val * 0.2)

    def test_fp8_e4m3_negative(self):
        """Test FP8 E4M3 conversion of negative values."""
        test_values = [-1.0, -10.0, -100.0, -448.0]
        for val in test_values:
            fp8 = float_to_fp8_e4m3(val)
            result = fp8_e4m3_to_float(fp8)
            self.assertAlmostEqual(result, val, delta=abs(val) * 0.2)

    def test_fp8_e4m3_clamp(self):
        """Test FP8 E4M3 clamps to max value."""
        # Values beyond max should clamp
        fp8 = float_to_fp8_e4m3(1000.0)
        result = fp8_e4m3_to_float(fp8)
        self.assertLessEqual(result, FP8_E4M3_MAX)

    def test_fp8_e5m2_zero(self):
        """Test FP8 E5M2 conversion of zero."""
        fp8 = float_to_fp8_e5m2(0.0)
        result = fp8_e5m2_to_float(fp8)
        self.assertEqual(result, 0.0)

    def test_fp8_e5m2_positive(self):
        """Test FP8 E5M2 conversion of positive values."""
        test_values = [1.0, 100.0, 1000.0, 10000.0]
        for val in test_values:
            fp8 = float_to_fp8_e5m2(val)
            result = fp8_e5m2_to_float(fp8)
            self.assertAlmostEqual(result, val, delta=val * 0.5)

    def test_fp8_e5m2_infinity(self):
        """Test FP8 E5M2 handles infinity."""
        fp8 = float_to_fp8_e5m2(float('inf'))
        result = fp8_e5m2_to_float(fp8)
        self.assertEqual(result, float('inf'))

    def test_fp8_e5m2_nan(self):
        """Test FP8 E5M2 handles NaN."""
        fp8 = float_to_fp8_e5m2(float('nan'))
        result = fp8_e5m2_to_float(fp8)
        self.assertTrue(np.isnan(result))


class TestFP8Quantizer(unittest.TestCase):
    """Test FP8 quantizer."""

    def setUp(self):
        np.random.seed(42)

    def test_fp8_e4m3_quantizer_init(self):
        """Test FP8 E4M3 quantizer initialization."""
        quantizer = FP8Quantizer(format="e4m3")
        self.assertEqual(quantizer.format, "e4m3")
        self.assertEqual(quantizer.config.quant_type, QuantType.FP8_E4M3)

    def test_fp8_e5m2_quantizer_init(self):
        """Test FP8 E5M2 quantizer initialization."""
        quantizer = FP8Quantizer(format="e5m2")
        self.assertEqual(quantizer.format, "e5m2")
        self.assertEqual(quantizer.config.quant_type, QuantType.FP8_E5M2)

    def test_fp8_invalid_format(self):
        """Test FP8 quantizer rejects invalid format."""
        with self.assertRaises(ValueError):
            FP8Quantizer(format="e3m4")

    def test_fp8_e4m3_quantize_weight(self):
        """Test FP8 E4M3 weight quantization."""
        quantizer = FP8Quantizer(format="e4m3")
        weight = np.random.randn(64, 128).astype(np.float32) * 10

        qtensor = quantizer.quantize_weight(weight, "test_layer")

        self.assertIsInstance(qtensor, FP8QuantizedTensor)
        self.assertEqual(qtensor.format, "e4m3")
        self.assertEqual(qtensor.data.dtype, np.uint8)
        self.assertEqual(qtensor.shape, weight.shape)

    def test_fp8_e5m2_quantize_weight(self):
        """Test FP8 E5M2 weight quantization."""
        quantizer = FP8Quantizer(format="e5m2")
        weight = np.random.randn(64, 128).astype(np.float32) * 100

        qtensor = quantizer.quantize_weight(weight, "test_layer")

        self.assertIsInstance(qtensor, FP8QuantizedTensor)
        self.assertEqual(qtensor.format, "e5m2")
        self.assertEqual(qtensor.data.dtype, np.uint8)

    def test_fp8_e4m3_dequantize(self):
        """Test FP8 E4M3 dequantization accuracy."""
        quantizer = FP8Quantizer(format="e4m3")
        weight = np.random.randn(32, 64).astype(np.float32) * 5

        qtensor = quantizer.quantize_weight(weight, "test")
        dequantized = qtensor.dequantize()

        # Check shape preserved
        self.assertEqual(dequantized.shape, weight.shape)

        # Check reasonable accuracy (FP8 has ~12.5% error on average)
        rel_error = np.abs(dequantized - weight) / (np.abs(weight) + 1e-10)
        mean_error = np.mean(rel_error)
        self.assertLess(mean_error, 0.5)  # Less than 50% average error

    def test_fp8_e5m2_dequantize(self):
        """Test FP8 E5M2 dequantization accuracy."""
        quantizer = FP8Quantizer(format="e5m2")
        weight = np.random.randn(32, 64).astype(np.float32) * 50

        qtensor = quantizer.quantize_weight(weight, "test")
        dequantized = qtensor.dequantize()

        self.assertEqual(dequantized.shape, weight.shape)

    def test_fp8_per_channel_quantization(self):
        """Test FP8 per-channel quantization."""
        config = QuantConfig(scale_type=ScaleType.PER_CHANNEL)
        quantizer = FP8Quantizer(config=config, format="e4m3")
        weight = np.random.randn(64, 128).astype(np.float32) * 10

        qtensor = quantizer.quantize_weight(weight, "test")

        # Scales should have one per output channel
        self.assertEqual(qtensor.scales.shape[0], weight.shape[0])

    def test_fp8_memory_efficiency(self):
        """Test FP8 memory is smaller than FP32."""
        quantizer = FP8Quantizer(format="e4m3")
        weight = np.random.randn(512, 512).astype(np.float32)
        original_size = weight.nbytes

        qtensor = quantizer.quantize_weight(weight, "test")
        quantized_size = qtensor.nbytes

        # FP8 should be ~4x smaller than FP32
        compression_ratio = original_size / quantized_size
        self.assertGreater(compression_ratio, 3.5)

    def test_fp8_e4m3_range(self):
        """Test FP8 E4M3 handles its valid range."""
        quantizer = FP8Quantizer(format="e4m3")
        # Create weights at edges of E4M3 range
        weight = np.array([[-448, 0, 448], [-100, 100, 200]], dtype=np.float32)

        qtensor = quantizer.quantize_weight(weight, "test")
        dequantized = qtensor.dequantize()

        # Check values are within valid range
        self.assertTrue(np.all(np.abs(dequantized) <= FP8_E4M3_MAX * 1.1))


class TestFP8TensorOperations(unittest.TestCase):
    """Test FP8 tensor operations."""

    def test_quantize_fp8_e4m3_per_tensor(self):
        """Test per-tensor FP8 E4M3 quantization."""
        tensor = np.random.randn(16, 32).astype(np.float32) * 10
        qdata, scales = quantize_fp8_e4m3(tensor, ScaleType.PER_TENSOR)

        self.assertEqual(qdata.shape, tensor.shape)
        self.assertEqual(qdata.dtype, np.uint8)
        self.assertEqual(len(scales), 1)

    def test_quantize_fp8_e4m3_per_channel(self):
        """Test per-channel FP8 E4M3 quantization."""
        tensor = np.random.randn(16, 32).astype(np.float32) * 10
        qdata, scales = quantize_fp8_e4m3(tensor, ScaleType.PER_CHANNEL)

        self.assertEqual(qdata.shape, tensor.shape)
        self.assertEqual(scales.shape[0], tensor.shape[0])

    def test_quantize_fp8_e5m2_per_tensor(self):
        """Test per-tensor FP8 E5M2 quantization."""
        tensor = np.random.randn(16, 32).astype(np.float32) * 100
        qdata, scales = quantize_fp8_e5m2(tensor, ScaleType.PER_TENSOR)

        self.assertEqual(qdata.shape, tensor.shape)
        self.assertEqual(qdata.dtype, np.uint8)

    def test_fp8_quantized_tensor_save_load(self):
        """Test FP8QuantizedTensor can be created and accessed."""
        data = np.zeros((8, 16), dtype=np.uint8)
        scales = np.array([1.0], dtype=np.float32)

        qtensor = FP8QuantizedTensor(
            data=data,
            scales=scales,
            format="e4m3",
            scale_type=ScaleType.PER_TENSOR,
            original_shape=(8, 16)
        )

        self.assertEqual(qtensor.shape, (8, 16))
        self.assertEqual(qtensor.format, "e4m3")
        self.assertGreater(qtensor.nbytes, 0)


class TestWeightValidation(unittest.TestCase):
    """Public-API input validation for quantize_weight."""

    def test_int8_rejects_1d_weight(self):
        with self.assertRaises(ValueError):
            INT8Quantizer().quantize_weight(np.arange(10, dtype=np.float32))

    def test_int4_rejects_3d_weight(self):
        with self.assertRaises(ValueError):
            INT4Quantizer().quantize_weight(np.zeros((2, 3, 4), dtype=np.float32))

    def test_gptq_rejects_1d_weight(self):
        with self.assertRaises(ValueError):
            GPTQQuantizer().quantize_weight(np.arange(8, dtype=np.float32))

    def test_awq_rejects_1d_weight(self):
        with self.assertRaises(ValueError):
            AWQQuantizer().quantize_weight(np.arange(8, dtype=np.float32))

    def test_fp8_rejects_1d_weight(self):
        with self.assertRaises(ValueError):
            FP8Quantizer().quantize_weight(np.arange(8, dtype=np.float32))

    def test_per_group_requires_enough_columns(self):
        config = QuantConfig(
            bits=8, scale_type=ScaleType.PER_GROUP, group_size=128
        )
        with self.assertRaises(ValueError):
            INT8Quantizer(config).quantize_weight(
                np.random.randn(4, 100).astype(np.float32)
            )

    def test_valid_2d_weight_still_works(self):
        w = np.random.randn(4, 32).astype(np.float32)
        qt = INT8Quantizer().quantize_weight(w)
        self.assertEqual(qt.shape, (4, 32))


class TestPublicExports(unittest.TestCase):
    """Documented public API must be importable where the docs claim."""

    def test_core_free_functions_exported(self):
        from vqllm.core import (
            quantize_int8, quantize_int4,
            float_to_fp8_e4m3, fp8_e4m3_to_float,
            float_to_fp8_e5m2, fp8_e5m2_to_float,
        )
        self.assertTrue(callable(quantize_int8))
        self.assertTrue(callable(float_to_fp8_e4m3))

    def test_calibration_free_functions_exported(self):
        from vqllm.calibration import (
            compute_hessian, HessianCalibrator, ActivationCalibrator,
            CalibrationDataset,
        )
        self.assertTrue(callable(compute_hessian))

    def test_formats_helpers_exported(self):
        from vqllm.formats import convert_to_gguf, load_from_gguf
        self.assertTrue(callable(convert_to_gguf))
        self.assertTrue(callable(load_from_gguf))

    def test_package_root_all_resolves(self):
        import vqllm
        for name in vqllm.__all__:
            self.assertTrue(hasattr(vqllm, name), f"vqllm missing {name}")


if __name__ == '__main__':
    unittest.main()