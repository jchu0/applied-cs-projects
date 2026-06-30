"""Unit tests for calibration module."""

import unittest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check if torch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from vqllm.calibration.calibrate import (
    CalibrationDataset, Calibrator, HessianCalibrator,
    ActivationCalibrator, compute_hessian, collect_activations
)
from vqllm.core.types import QuantConfig, QuantType


class TestCalibrationDataset(unittest.TestCase):
    """Test calibration dataset handling."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_create_dataset(self):
        """Test calibration dataset creation."""
        # Create sample data
        data = [
            {"input_ids": np.random.randint(0, 1000, size=(128,))},
            {"input_ids": np.random.randint(0, 1000, size=(128,))},
            {"input_ids": np.random.randint(0, 1000, size=(128,))},
        ]

        dataset = CalibrationDataset(data, max_samples=2)

        # Check dataset properties
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["input_ids"].shape, (128,))

    def test_dataset_from_text(self):
        """Test creating dataset from text."""
        texts = [
            "This is a sample text for calibration.",
            "Another example sentence for testing.",
            "Quantization calibration is important."
        ]

        dataset = CalibrationDataset.from_texts(texts, max_length=32)

        # Check tokenization
        self.assertGreater(len(dataset), 0)
        self.assertIn("input_ids", dataset[0])

    def test_dataset_batching(self):
        """Test batching calibration data."""
        data = [
            {"input_ids": np.random.randint(0, 1000, size=(64,))}
            for _ in range(10)
        ]

        dataset = CalibrationDataset(data)
        batch = dataset.get_batch(batch_size=4, indices=[0, 2, 4, 6])

        # Check batch shape
        self.assertEqual(batch["input_ids"].shape[0], 4)
        self.assertEqual(batch["input_ids"].shape[1], 64)

    def test_dataset_shuffle(self):
        """Test dataset shuffling."""
        data = [
            {"input_ids": np.array([i] * 32)}
            for i in range(10)
        ]

        dataset = CalibrationDataset(data, shuffle=True, seed=42)
        first_sample = dataset[0]["input_ids"][0]

        # Should not be the first element if shuffled
        self.assertNotEqual(first_sample, 0)

    def test_save_load_dataset(self):
        """Test saving and loading calibration dataset."""
        data = [
            {"input_ids": np.random.randint(0, 1000, size=(128,))}
            for _ in range(5)
        ]

        dataset = CalibrationDataset(data)

        # Save dataset
        save_path = os.path.join(self.temp_dir, "calib_data.npz")
        dataset.save(save_path)

        # Load and verify
        loaded = CalibrationDataset.load(save_path)
        self.assertEqual(len(loaded), len(dataset))
        np.testing.assert_array_equal(loaded[0]["input_ids"], dataset[0]["input_ids"])


class TestCalibrator(unittest.TestCase):
    """Test base calibrator functionality."""

    def setUp(self):
        self.config = QuantConfig(quant_type=QuantType.INT8)
        self.calibrator = Calibrator(self.config)
        np.random.seed(42)

    def test_calibrator_init(self):
        """Test calibrator initialization."""
        self.assertEqual(self.calibrator.config.quant_type, QuantType.INT8)
        self.assertIsNotNone(self.calibrator.statistics)

    def test_collect_statistics(self):
        """Test collecting calibration statistics."""
        # Mock model
        model = Mock()
        layer_output = np.random.randn(32, 128, 768).astype(np.float32)
        model.forward = Mock(return_value=layer_output)

        # Calibration data
        data = [
            {"input_ids": np.random.randint(0, 1000, size=(32, 128))}
            for _ in range(5)
        ]

        stats = self.calibrator.collect_statistics(model, data)

        # Check statistics collected
        self.assertIsNotNone(stats)
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("percentiles", stats)

    def test_compute_scale_factors(self):
        """Test computing scale factors from statistics."""
        stats = {
            "min": np.array([-10.0, -8.0, -12.0]),
            "max": np.array([10.0, 12.0, 8.0]),
            "mean": np.array([0.0, 1.0, -1.0]),
            "std": np.array([2.0, 3.0, 2.5])
        }

        scales = self.calibrator.compute_scale_factors(stats)

        # Check scales are positive
        self.assertTrue(np.all(scales > 0))

        # Check scales are reasonable
        self.assertTrue(np.all(scales < 1000))

    def test_percentile_calibration(self):
        """Test percentile-based calibration."""
        data = np.random.randn(1000, 512).astype(np.float32) * 5

        # Compute percentiles for calibration
        percentiles = [0.1, 1.0, 5.0, 95.0, 99.0, 99.9]
        results = {}

        for p in percentiles:
            results[f"p{p}"] = np.percentile(np.abs(data), p, axis=0)

        # Check percentiles increase monotonically
        for i in range(len(percentiles) - 1):
            p1, p2 = percentiles[i], percentiles[i + 1]
            self.assertTrue(np.all(results[f"p{p1}"] <= results[f"p{p2}"]))


class TestHessianCalibrator(unittest.TestCase):
    """Test Hessian-based calibration for GPTQ."""

    def setUp(self):
        config = QuantConfig(
            quant_type=QuantType.GPTQ,
            dampening=0.01
        )
        self.calibrator = HessianCalibrator(config)
        np.random.seed(42)

    def test_compute_hessian(self):
        """Test Hessian computation."""
        # Mock layer inputs
        inputs = [
            np.random.randn(32, 512).astype(np.float32)
            for _ in range(10)
        ]

        hessian = compute_hessian(inputs, dampening=0.01)

        # Check Hessian properties
        self.assertEqual(hessian.shape, (512, 512))

        # Should be symmetric
        np.testing.assert_allclose(hessian, hessian.T, rtol=1e-5)

        # Should be positive semi-definite (all eigenvalues >= 0)
        eigenvalues = np.linalg.eigvalsh(hessian)
        self.assertTrue(np.all(eigenvalues >= -1e-6))

    def test_batch_hessian_update(self):
        """Test incremental Hessian updates."""
        hessian = np.zeros((256, 256), dtype=np.float32)

        # Update with batches
        for _ in range(5):
            batch = np.random.randn(32, 256).astype(np.float32)
            hessian = self.calibrator.update_hessian(hessian, batch)

        # Check updated Hessian
        self.assertFalse(np.allclose(hessian, 0))
        self.assertEqual(hessian.shape, (256, 256))

    def test_hessian_regularization(self):
        """Test Hessian regularization with dampening."""
        inputs = [np.random.randn(32, 128).astype(np.float32)]

        # Compute with different dampening values
        hessian_low = compute_hessian(inputs, dampening=0.001)
        hessian_high = compute_hessian(inputs, dampening=0.1)

        # Higher dampening should increase diagonal elements
        diag_low = np.diag(hessian_low)
        diag_high = np.diag(hessian_high)
        self.assertTrue(np.all(diag_high > diag_low))

    def test_layer_wise_hessian(self):
        """Test layer-wise Hessian collection."""
        model = Mock()

        # Mock multiple layers
        layers = {
            "layer1": Mock(weight=np.random.randn(256, 512)),
            "layer2": Mock(weight=np.random.randn(512, 256)),
            "layer3": Mock(weight=np.random.randn(256, 128)),
        }

        hessians = self.calibrator.collect_layer_hessians(model, layers)

        # Check Hessians for each layer
        self.assertEqual(len(hessians), 3)
        self.assertIn("layer1", hessians)
        self.assertEqual(hessians["layer1"].shape[0], 512)  # Input dimension


class TestActivationCalibrator(unittest.TestCase):
    """Test activation-based calibration for AWQ."""

    def setUp(self):
        config = QuantConfig(
            quant_type=QuantType.AWQ,
            group_size=128
        )
        self.calibrator = ActivationCalibrator(config)
        np.random.seed(42)

    def test_collect_activations(self):
        """Test activation collection."""
        # Mock model forward pass
        activations = [
            np.random.randn(32, 128, 512).astype(np.float32)
            for _ in range(10)
        ]

        collected = collect_activations(activations)

        # Check collected statistics
        self.assertIn("mean", collected)
        self.assertIn("std", collected)
        self.assertIn("max", collected)

        # Check dimensions
        self.assertEqual(collected["mean"].shape[-1], 512)

    def test_compute_activation_scale(self):
        """Test computing activation scales."""
        activations = [
            np.random.randn(32, 512).astype(np.float32) * (i + 1)
            for i in range(5)
        ]

        scales = self.calibrator.compute_activation_scales(activations)

        # Check scale properties
        self.assertEqual(len(scales), 512)
        self.assertTrue(np.all(scales > 0))

        # Scales should reflect activation magnitudes
        mean_act = np.mean([np.abs(a).mean() for a in activations])
        self.assertGreater(np.mean(scales), 0.1 * mean_act)

    def test_channel_wise_statistics(self):
        """Test per-channel activation statistics."""
        # Create activations with different channel magnitudes
        activations = np.random.randn(100, 256).astype(np.float32)
        activations[:, :128] *= 10  # First half channels have higher magnitude

        stats = self.calibrator.compute_channel_statistics(activations)

        # Check statistics reflect the difference
        mean_first_half = np.mean(stats["std"][:128])
        mean_second_half = np.mean(stats["std"][128:])
        self.assertGreater(mean_first_half, mean_second_half)

    def test_smooth_activation_scales(self):
        """Test smoothing of activation scales."""
        # Create noisy scales
        scales = np.random.randn(256).astype(np.float32) ** 2 + 1.0
        scales[::10] *= 100  # Add outliers

        smoothed = self.calibrator.smooth_scales(scales, alpha=0.5)

        # Check smoothing reduces variance
        self.assertLess(np.std(smoothed), np.std(scales))

        # Check outliers are dampened
        outlier_indices = np.arange(0, 256, 10)
        for idx in outlier_indices:
            self.assertLess(smoothed[idx], scales[idx])


class TestCalibrationIntegration(unittest.TestCase):
    """Integration tests for calibration pipeline."""

    def setUp(self):
        np.random.seed(42)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_calibration(self):
        """Test full calibration pipeline."""
        # Create config
        config = QuantConfig(
            quant_type=QuantType.GPTQ,
            block_size=128,
            dampening=0.01
        )

        # Create calibrator
        calibrator = HessianCalibrator(config)

        # Mock model and data
        model = Mock()
        calibration_data = [
            {"input_ids": np.random.randint(0, 1000, size=(32, 128))}
            for _ in range(20)
        ]

        # Run calibration
        calib_result = calibrator.calibrate(model, calibration_data)

        # Check calibration results
        self.assertIsNotNone(calib_result)
        self.assertIn("hessians", calib_result)
        self.assertIn("statistics", calib_result)

    def test_calibration_cache(self):
        """Test caching calibration results."""
        config = QuantConfig(quant_type=QuantType.AWQ)
        calibrator = ActivationCalibrator(config)

        # First calibration
        activations = [np.random.randn(32, 512) for _ in range(10)]
        result1 = calibrator.compute_activation_scales(activations)

        # Save to cache
        cache_path = os.path.join(self.temp_dir, "calib_cache.npz")
        calibrator.save_cache(cache_path, result1)

        # Load from cache
        result2 = calibrator.load_cache(cache_path)

        # Should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_calibration_with_different_configs(self):
        """Test calibration with various configurations."""
        configs = [
            QuantConfig(quant_type=QuantType.INT8),
            QuantConfig(quant_type=QuantType.INT4),
            QuantConfig(quant_type=QuantType.GPTQ, block_size=64),
            QuantConfig(quant_type=QuantType.AWQ, group_size=256),
        ]

        for config in configs:
            calibrator = Calibrator.create(config)
            self.assertIsNotNone(calibrator)
            self.assertEqual(calibrator.config.quant_type, config.quant_type)

    @unittest.skipUnless(HAS_TORCH, "torch not available")
    def test_multi_gpu_calibration(self):
        """Test calibration with multi-GPU simulation."""
        with patch('torch.cuda.device_count', return_value=2):
            config = QuantConfig(
                quant_type=QuantType.GPTQ,
                use_multi_gpu=True
            )

            calibrator = HessianCalibrator(config)

            # Mock distributed calibration
            results = calibrator.distributed_calibrate(
                model=Mock(),
                data=[{"input_ids": np.random.randint(0, 1000, (32, 128))} for _ in range(10)],
                num_gpus=2
            )

            self.assertIsNotNone(results)


if __name__ == '__main__':
    unittest.main()