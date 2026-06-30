"""Tests for mixed precision training."""

import pytest
import numpy as np

from paramserver.enterprise.mixed_precision import (
    MixedPrecisionManager,
    PrecisionMode,
)


class TestPrecisionMode:
    """Tests for PrecisionMode enum."""

    def test_enum_values(self):
        """Test enum values."""
        assert PrecisionMode.FP32.value == "fp32"
        assert PrecisionMode.FP16.value == "fp16"
        assert PrecisionMode.MIXED.value == "mixed"
        assert PrecisionMode.DYNAMIC.value == "dynamic"


class TestMixedPrecisionManagerInit:
    """Tests for MixedPrecisionManager initialization."""

    def test_create_default(self):
        """Test default creation."""
        manager = MixedPrecisionManager()
        assert manager.mode == PrecisionMode.MIXED
        assert manager.loss_scale == 65536.0
        assert manager.dynamic_scaling is True

    def test_create_custom(self):
        """Test custom creation."""
        manager = MixedPrecisionManager(
            mode=PrecisionMode.FP16,
            initial_loss_scale=1024.0,
            dynamic_scaling=False,
        )
        assert manager.mode == PrecisionMode.FP16
        assert manager.loss_scale == 1024.0
        assert manager.dynamic_scaling is False


class TestPrecisionConversion:
    """Tests for precision conversion."""

    def test_to_fp16(self):
        """Test converting to FP16."""
        manager = MixedPrecisionManager()
        tensor = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        fp16 = manager.to_fp16(tensor)

        assert fp16.dtype == np.float16
        np.testing.assert_array_equal(fp16, tensor.astype(np.float16))

    def test_to_fp32(self):
        """Test converting to FP32."""
        manager = MixedPrecisionManager()
        tensor = np.array([1.0, 2.0, 3.0], dtype=np.float16)

        fp32 = manager.to_fp32(tensor)

        assert fp32.dtype == np.float32


class TestLossScaling:
    """Tests for loss scaling."""

    def test_scale_loss_fp32_mode(self):
        """Test loss scaling in FP32 mode."""
        manager = MixedPrecisionManager(mode=PrecisionMode.FP32)
        loss = 0.5

        scaled = manager.scale_loss(loss)

        # Should not scale in FP32 mode
        assert scaled == loss

    def test_scale_loss_mixed_mode(self):
        """Test loss scaling in mixed mode."""
        manager = MixedPrecisionManager(
            mode=PrecisionMode.MIXED,
            initial_loss_scale=1000.0,
        )
        loss = 0.5

        scaled = manager.scale_loss(loss)

        assert scaled == loss * 1000.0

    def test_unscale_gradients(self):
        """Test gradient unscaling."""
        manager = MixedPrecisionManager(
            mode=PrecisionMode.MIXED,
            initial_loss_scale=1000.0,
        )
        gradients = {
            "w1": np.array([1000.0, 2000.0], dtype=np.float32),
            "w2": np.array([500.0], dtype=np.float32),
        }

        unscaled = manager.unscale_gradients(gradients)

        np.testing.assert_array_almost_equal(
            unscaled["w1"], [1.0, 2.0]
        )
        np.testing.assert_array_almost_equal(
            unscaled["w2"], [0.5]
        )


class TestOverflowDetection:
    """Tests for overflow detection."""

    def test_no_overflow(self):
        """Test no overflow detected."""
        manager = MixedPrecisionManager()
        gradients = {
            "w1": np.array([1.0, 2.0], dtype=np.float32),
        }

        overflow = manager.check_overflow(gradients)

        assert overflow is False

    def test_nan_overflow(self):
        """Test NaN detection."""
        manager = MixedPrecisionManager()
        gradients = {
            "w1": np.array([1.0, np.nan], dtype=np.float32),
        }

        overflow = manager.check_overflow(gradients)

        assert overflow is True

    def test_inf_overflow(self):
        """Test infinity detection."""
        manager = MixedPrecisionManager()
        gradients = {
            "w1": np.array([1.0, np.inf], dtype=np.float32),
        }

        overflow = manager.check_overflow(gradients)

        assert overflow is True


class TestDynamicScaling:
    """Tests for dynamic loss scaling."""

    def test_scale_decrease_on_overflow(self):
        """Test scale decreases on overflow."""
        manager = MixedPrecisionManager(
            initial_loss_scale=1000.0,
            scale_factor=2.0,
        )

        manager.update_scale(overflow=True)

        assert manager.loss_scale == 500.0

    def test_scale_increase_after_window(self):
        """Test scale increases after stable window."""
        manager = MixedPrecisionManager(
            initial_loss_scale=100.0,
            scale_window=3,
            scale_factor=2.0,
        )

        # No overflow for scale_window steps
        for _ in range(3):
            manager.update_scale(overflow=False)

        assert manager.loss_scale == 200.0

    def test_scale_capped_at_max(self):
        """Test scale is capped at maximum."""
        manager = MixedPrecisionManager(
            initial_loss_scale=65536.0,
            scale_window=1,
        )

        manager.update_scale(overflow=False)

        assert manager.loss_scale == 65536.0

    def test_scale_capped_at_min(self):
        """Test scale is capped at minimum."""
        manager = MixedPrecisionManager(
            initial_loss_scale=2.0,
            scale_factor=4.0,
        )

        manager.update_scale(overflow=True)

        assert manager.loss_scale >= 1.0


class TestStep:
    """Tests for step processing."""

    def test_step_success(self):
        """Test successful step."""
        manager = MixedPrecisionManager(initial_loss_scale=10.0)
        gradients = {
            "w1": np.array([10.0, 20.0], dtype=np.float32),
        }

        result = manager.step(gradients)

        assert result is not None
        np.testing.assert_array_almost_equal(
            result["w1"], [1.0, 2.0]
        )

    def test_step_overflow_skipped(self):
        """Test step is skipped on overflow."""
        manager = MixedPrecisionManager(initial_loss_scale=10.0)
        gradients = {
            "w1": np.array([10.0, np.inf], dtype=np.float32),
        }

        result = manager.step(gradients)

        assert result is None


class TestPrecisionForParam:
    """Tests for parameter precision selection."""

    def test_fp16_mode(self):
        """Test FP16 mode returns FP16."""
        manager = MixedPrecisionManager(mode=PrecisionMode.FP16)

        dtype = manager.get_precision_for_param("weights")

        assert dtype == np.float16

    def test_fp32_mode(self):
        """Test FP32 mode returns FP32."""
        manager = MixedPrecisionManager(mode=PrecisionMode.FP32)

        dtype = manager.get_precision_for_param("weights")

        assert dtype == np.float32

    def test_mixed_mode(self):
        """Test mixed mode returns FP32 for params."""
        manager = MixedPrecisionManager(mode=PrecisionMode.MIXED)

        dtype = manager.get_precision_for_param("weights")

        # Parameters kept in FP32 for accuracy
        assert dtype == np.float32


class TestStats:
    """Tests for statistics."""

    def test_get_stats(self):
        """Test getting statistics."""
        manager = MixedPrecisionManager(
            mode=PrecisionMode.MIXED,
            initial_loss_scale=1000.0,
        )

        # Simulate some steps
        manager.update_scale(overflow=False)
        manager.update_scale(overflow=True)
        manager.update_scale(overflow=False)

        stats = manager.get_stats()

        assert stats["mode"] == "mixed"
        assert stats["total_steps"] == 3
        assert stats["overflow_count"] == 1
        assert stats["overflow_rate"] == pytest.approx(1/3)
