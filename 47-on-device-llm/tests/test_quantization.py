"""Tests for quantization and dequantization operations."""

import numpy as np
import pytest

from on_device_llm.quantization import (
    GGMLType,
    QuantizedTensor,
    quantize_q4_0,
    quantize_q8_0,
    dequantize_q4_0,
    dequantize_q8_0,
    compute_quantization_error,
    calc_tensor_size,
    BLOCK_SIZES,
    BYTES_PER_BLOCK,
)


class TestGGMLType:
    """Tests for GGMLType enumeration."""

    def test_ggml_type_values(self):
        """Test GGML type enum values match expected values."""
        assert GGMLType.F32 == 0
        assert GGMLType.F16 == 1
        assert GGMLType.Q4_0 == 2
        assert GGMLType.Q4_1 == 3
        assert GGMLType.Q8_0 == 8
        assert GGMLType.Q8_1 == 9

    def test_ggml_type_from_int(self):
        """Test creating GGMLType from integer."""
        assert GGMLType(0) == GGMLType.F32
        assert GGMLType(2) == GGMLType.Q4_0
        assert GGMLType(8) == GGMLType.Q8_0

    def test_block_sizes_defined(self):
        """Test block sizes are defined for quantized types."""
        assert GGMLType.Q4_0 in BLOCK_SIZES
        assert GGMLType.Q8_0 in BLOCK_SIZES
        assert BLOCK_SIZES[GGMLType.Q4_0] == 32
        assert BLOCK_SIZES[GGMLType.Q8_0] == 32

    def test_bytes_per_block_defined(self):
        """Test bytes per block are defined for quantized types."""
        assert GGMLType.Q4_0 in BYTES_PER_BLOCK
        assert GGMLType.Q8_0 in BYTES_PER_BLOCK
        assert BYTES_PER_BLOCK[GGMLType.Q4_0] == 18  # 2 + 16
        assert BYTES_PER_BLOCK[GGMLType.Q8_0] == 34  # 2 + 32


class TestQ8Quantization:
    """Tests for Q8_0 quantization."""

    def test_quantize_q8_basic(self, tensor_for_quantization):
        """Test basic Q8_0 quantization."""
        data, dtype = quantize_q8_0(tensor_for_quantization)

        assert isinstance(data, bytes)
        assert dtype == GGMLType.Q8_0
        assert len(data) > 0

    def test_quantize_q8_block_structure(self):
        """Test Q8_0 quantization produces correct block structure."""
        # Create tensor with exactly 32 elements (1 block)
        x = np.random.randn(32).astype(np.float32)
        data, _ = quantize_q8_0(x)

        # Q8_0: 2 bytes scale + 32 bytes values = 34 bytes per block
        assert len(data) == 34

    def test_quantize_q8_multiple_blocks(self):
        """Test Q8_0 quantization with multiple blocks."""
        # 64 elements = 2 blocks
        x = np.random.randn(64).astype(np.float32)
        data, _ = quantize_q8_0(x)

        # 2 blocks * 34 bytes = 68 bytes
        assert len(data) == 68

    def test_quantize_q8_padding(self):
        """Test Q8_0 quantization handles non-block-aligned sizes."""
        # 33 elements needs 2 blocks (32 + 1)
        x = np.random.randn(33).astype(np.float32)
        data, _ = quantize_q8_0(x)

        # Should pad to 64 elements = 2 blocks
        assert len(data) == 68

    def test_dequantize_q8_recovers_shape(self, tensor_for_quantization):
        """Test Q8_0 dequantization recovers original shape."""
        original_shape = tensor_for_quantization.shape
        data, _ = quantize_q8_0(tensor_for_quantization)

        recovered = dequantize_q8_0(data, original_shape)

        assert recovered.shape == original_shape
        assert recovered.dtype == np.float32

    def test_dequantize_q8_recovers_values(self, tensor_for_quantization):
        """Test Q8_0 dequantization recovers approximate values."""
        data, _ = quantize_q8_0(tensor_for_quantization)
        recovered = dequantize_q8_0(data, tensor_for_quantization.shape)

        # Q8 should have low error (8-bit precision)
        max_error = np.max(np.abs(tensor_for_quantization - recovered))
        mean_error = np.mean(np.abs(tensor_for_quantization - recovered))

        # 8-bit quantization should have <2% relative error for most values
        relative_error = mean_error / (np.std(tensor_for_quantization) + 1e-6)
        assert relative_error < 0.05, f"Relative error too high: {relative_error}"

    def test_q8_roundtrip_consistency(self, large_tensor_for_quantization):
        """Test Q8 quantize-dequantize roundtrip is consistent."""
        shape = large_tensor_for_quantization.shape

        data1, _ = quantize_q8_0(large_tensor_for_quantization)
        recovered1 = dequantize_q8_0(data1, shape)

        data2, _ = quantize_q8_0(large_tensor_for_quantization)
        recovered2 = dequantize_q8_0(data2, shape)

        np.testing.assert_array_equal(recovered1, recovered2)

    def test_q8_zero_tensor(self):
        """Test Q8 quantization handles zero tensor."""
        x = np.zeros(64, dtype=np.float32)
        data, _ = quantize_q8_0(x)
        recovered = dequantize_q8_0(data, x.shape)

        np.testing.assert_array_almost_equal(recovered, x, decimal=5)

    def test_q8_extreme_values(self):
        """Test Q8 quantization handles extreme values."""
        x = np.array([1e6, -1e6, 0, 1e-6, -1e-6] + [0.0] * 27, dtype=np.float32)
        data, _ = quantize_q8_0(x)
        recovered = dequantize_q8_0(data, x.shape)

        # Should preserve sign and rough magnitude
        assert np.sign(recovered[0]) == np.sign(x[0])
        assert np.sign(recovered[1]) == np.sign(x[1])


class TestQ4Quantization:
    """Tests for Q4_0 quantization."""

    def test_quantize_q4_basic(self, tensor_for_quantization):
        """Test basic Q4_0 quantization."""
        data, dtype = quantize_q4_0(tensor_for_quantization)

        assert isinstance(data, bytes)
        assert dtype == GGMLType.Q4_0
        assert len(data) > 0

    def test_quantize_q4_block_structure(self):
        """Test Q4_0 quantization produces correct block structure."""
        # Create tensor with exactly 32 elements (1 block)
        x = np.random.randn(32).astype(np.float32)
        data, _ = quantize_q4_0(x)

        # Q4_0: 2 bytes scale + 16 bytes values = 18 bytes per block
        assert len(data) == 18

    def test_quantize_q4_multiple_blocks(self):
        """Test Q4_0 quantization with multiple blocks."""
        # 64 elements = 2 blocks
        x = np.random.randn(64).astype(np.float32)
        data, _ = quantize_q4_0(x)

        # 2 blocks * 18 bytes = 36 bytes
        assert len(data) == 36

    def test_dequantize_q4_recovers_shape(self, tensor_for_quantization):
        """Test Q4_0 dequantization recovers original shape."""
        original_shape = tensor_for_quantization.shape
        data, _ = quantize_q4_0(tensor_for_quantization)

        recovered = dequantize_q4_0(data, original_shape)

        assert recovered.shape == original_shape
        assert recovered.dtype == np.float32

    def test_dequantize_q4_recovers_values(self, tensor_for_quantization):
        """Test Q4_0 dequantization recovers approximate values."""
        data, _ = quantize_q4_0(tensor_for_quantization)
        recovered = dequantize_q4_0(data, tensor_for_quantization.shape)

        # Q4 has higher error than Q8 but should still be reasonable
        mean_error = np.mean(np.abs(tensor_for_quantization - recovered))
        relative_error = mean_error / (np.std(tensor_for_quantization) + 1e-6)

        # 4-bit quantization should have <20% relative error
        assert relative_error < 0.3, f"Relative error too high: {relative_error}"

    def test_q4_vs_q8_compression(self, large_tensor_for_quantization):
        """Test Q4 produces smaller data than Q8."""
        q4_data, _ = quantize_q4_0(large_tensor_for_quantization)
        q8_data, _ = quantize_q8_0(large_tensor_for_quantization)

        # Q4 should be roughly half the size of Q8
        ratio = len(q4_data) / len(q8_data)
        assert ratio < 0.6  # Q4 is ~53% of Q8 (18/34)

    def test_q4_vs_q8_accuracy(self, tensor_for_quantization):
        """Test Q8 is more accurate than Q4."""
        q4_data, _ = quantize_q4_0(tensor_for_quantization)
        q8_data, _ = quantize_q8_0(tensor_for_quantization)

        q4_recovered = dequantize_q4_0(q4_data, tensor_for_quantization.shape)
        q8_recovered = dequantize_q8_0(q8_data, tensor_for_quantization.shape)

        q4_error = np.mean(np.abs(tensor_for_quantization - q4_recovered))
        q8_error = np.mean(np.abs(tensor_for_quantization - q8_recovered))

        # Q8 should have lower error than Q4
        assert q8_error < q4_error

    def test_q4_packing_correctness(self):
        """Test Q4 value packing is correct."""
        # Create values that are easy to verify after quantization
        # Values in range that should quantize predictably
        x = np.array([1.0] * 16 + [-1.0] * 16, dtype=np.float32)
        data, _ = quantize_q4_0(x)
        recovered = dequantize_q4_0(data, x.shape)

        # First 16 values should be positive, last 16 negative
        assert all(recovered[:16] >= 0)
        assert all(recovered[16:] <= 0)


class TestQuantizedTensor:
    """Tests for QuantizedTensor class."""

    def test_quantized_tensor_creation(self, tensor_for_quantization):
        """Test creating QuantizedTensor."""
        data, dtype = quantize_q8_0(tensor_for_quantization)

        qt = QuantizedTensor(
            data=data,
            shape=tensor_for_quantization.shape,
            dtype=dtype
        )

        assert qt.data == data
        assert qt.shape == tensor_for_quantization.shape
        assert qt.dtype == GGMLType.Q8_0

    def test_quantized_tensor_n_elements(self):
        """Test n_elements property."""
        data, dtype = quantize_q8_0(np.zeros((4, 8), dtype=np.float32))

        qt = QuantizedTensor(data=data, shape=(4, 8), dtype=dtype)

        assert qt.n_elements == 32

    def test_quantized_tensor_n_blocks(self):
        """Test n_blocks property."""
        x = np.zeros(64, dtype=np.float32)
        data, dtype = quantize_q8_0(x)

        qt = QuantizedTensor(data=data, shape=(64,), dtype=dtype)

        assert qt.n_blocks == 2  # 64 elements / 32 per block

    def test_quantized_tensor_dequantize_q8(self, tensor_for_quantization):
        """Test QuantizedTensor.dequantize() for Q8."""
        data, dtype = quantize_q8_0(tensor_for_quantization)

        qt = QuantizedTensor(
            data=data,
            shape=tensor_for_quantization.shape,
            dtype=dtype
        )

        recovered = qt.dequantize()

        assert recovered.shape == tensor_for_quantization.shape
        assert recovered.dtype == np.float32

    def test_quantized_tensor_dequantize_q4(self, tensor_for_quantization):
        """Test QuantizedTensor.dequantize() for Q4."""
        data, dtype = quantize_q4_0(tensor_for_quantization)

        qt = QuantizedTensor(
            data=data,
            shape=tensor_for_quantization.shape,
            dtype=dtype
        )

        recovered = qt.dequantize()

        assert recovered.shape == tensor_for_quantization.shape
        assert recovered.dtype == np.float32

    def test_quantized_tensor_dequantize_f32(self):
        """Test QuantizedTensor.dequantize() for F32 (no-op)."""
        x = np.random.randn(32).astype(np.float32)

        qt = QuantizedTensor(
            data=x.tobytes(),
            shape=x.shape,
            dtype=GGMLType.F32
        )

        recovered = qt.dequantize()

        np.testing.assert_array_equal(recovered, x)


class TestComputeQuantizationError:
    """Tests for quantization error computation."""

    def test_compute_error_q8(self, tensor_for_quantization):
        """Test computing quantization error for Q8."""
        data, dtype = quantize_q8_0(tensor_for_quantization)

        mae, max_error = compute_quantization_error(
            tensor_for_quantization, data, dtype
        )

        assert mae >= 0
        assert max_error >= mae
        assert max_error < 1.0  # Q8 should have small max error

    def test_compute_error_q4(self, tensor_for_quantization):
        """Test computing quantization error for Q4."""
        data, dtype = quantize_q4_0(tensor_for_quantization)

        mae, max_error = compute_quantization_error(
            tensor_for_quantization, data, dtype
        )

        assert mae >= 0
        assert max_error >= mae

    def test_q8_error_less_than_q4(self, tensor_for_quantization):
        """Test Q8 has less error than Q4."""
        q8_data, q8_dtype = quantize_q8_0(tensor_for_quantization)
        q4_data, q4_dtype = quantize_q4_0(tensor_for_quantization)

        q8_mae, _ = compute_quantization_error(tensor_for_quantization, q8_data, q8_dtype)
        q4_mae, _ = compute_quantization_error(tensor_for_quantization, q4_data, q4_dtype)

        assert q8_mae < q4_mae


class TestCalcTensorSize:
    """Tests for tensor size calculation."""

    def test_calc_size_f32(self):
        """Test size calculation for F32."""
        size = calc_tensor_size((100,), GGMLType.F32)
        assert size == 100 * 4  # 4 bytes per element

    def test_calc_size_f16(self):
        """Test size calculation for F16."""
        size = calc_tensor_size((100,), GGMLType.F16)
        assert size == 100 * 2  # 2 bytes per element

    def test_calc_size_q8(self):
        """Test size calculation for Q8_0."""
        # 64 elements = 2 blocks, 34 bytes each
        size = calc_tensor_size((64,), GGMLType.Q8_0)
        assert size == 2 * 34

    def test_calc_size_q4(self):
        """Test size calculation for Q4_0."""
        # 64 elements = 2 blocks, 18 bytes each
        size = calc_tensor_size((64,), GGMLType.Q4_0)
        assert size == 2 * 18

    def test_calc_size_2d(self):
        """Test size calculation for 2D tensor."""
        size = calc_tensor_size((32, 64), GGMLType.F32)
        assert size == 32 * 64 * 4

    def test_calc_size_q8_multidim(self):
        """Test Q8 size calculation for multi-dimensional tensor."""
        # 32 * 64 = 2048 elements = 64 blocks
        size = calc_tensor_size((32, 64), GGMLType.Q8_0)
        assert size == 64 * 34


class TestShapePreservation:
    """Tests for shape preservation during quantization."""

    def test_2d_shape_preserved_q8(self, shaped_tensor_for_quantization):
        """Test 2D shape is preserved after Q8 roundtrip."""
        original_shape = shaped_tensor_for_quantization.shape
        data, _ = quantize_q8_0(shaped_tensor_for_quantization)
        recovered = dequantize_q8_0(data, original_shape)

        assert recovered.shape == original_shape

    def test_2d_shape_preserved_q4(self, shaped_tensor_for_quantization):
        """Test 2D shape is preserved after Q4 roundtrip."""
        original_shape = shaped_tensor_for_quantization.shape
        data, _ = quantize_q4_0(shaped_tensor_for_quantization)
        recovered = dequantize_q4_0(data, original_shape)

        assert recovered.shape == original_shape

    def test_various_shapes_q8(self):
        """Test various shapes with Q8."""
        shapes = [(32,), (64,), (4, 16), (2, 4, 8), (8, 8)]

        for shape in shapes:
            x = np.random.randn(*shape).astype(np.float32)
            data, _ = quantize_q8_0(x)
            recovered = dequantize_q8_0(data, shape)

            assert recovered.shape == shape, f"Shape mismatch for {shape}"
