"""Calibration methods for quantization."""

import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..core.types import QuantConfig, QuantType

logger = logging.getLogger(__name__)


class CalibrationDataset:
    """Dataset for calibration data."""

    def __init__(
        self,
        data: List[Dict[str, np.ndarray]],
        max_samples: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ):
        self.data = data[:max_samples] if max_samples else data
        if shuffle:
            rng = np.random.RandomState(seed)
            indices = rng.permutation(len(self.data))
            self.data = [self.data[i] for i in indices]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.data[idx]

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        return iter(self.data)

    @classmethod
    def from_texts(cls, texts: List[str], max_length: int = 128) -> 'CalibrationDataset':
        """Create dataset from text strings (simple tokenization)."""
        data = []
        for text in texts:
            # Simple character-based tokenization
            tokens = [ord(c) % 1000 for c in text[:max_length]]
            if len(tokens) < max_length:
                tokens = tokens + [0] * (max_length - len(tokens))
            data.append({"input_ids": np.array(tokens, dtype=np.int64)})
        return cls(data)

    def get_batch(
        self,
        batch_size: int,
        indices: Optional[List[int]] = None
    ) -> Dict[str, np.ndarray]:
        """Get a batch of samples."""
        if indices is None:
            indices = list(range(min(batch_size, len(self.data))))

        batch_data = [self.data[i] for i in indices]

        # Stack arrays
        result = {}
        for key in batch_data[0].keys():
            result[key] = np.stack([d[key] for d in batch_data])

        return result

    def save(self, path: str):
        """Save dataset to file."""
        arrays = {}
        for i, sample in enumerate(self.data):
            for key, value in sample.items():
                arrays[f"{i}_{key}"] = value
        arrays["__len__"] = np.array([len(self.data)])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> 'CalibrationDataset':
        """Load dataset from file."""
        loaded = np.load(path)
        length = int(loaded["__len__"][0])

        data = []
        for i in range(length):
            sample = {}
            for key in loaded.files:
                if key.startswith(f"{i}_"):
                    field_name = key[len(f"{i}_"):]
                    sample[field_name] = loaded[key]
            if sample:
                data.append(sample)

        return cls(data)


@dataclass
class CalibrationData:
    """Container for calibration data."""
    activations: List[np.ndarray] = field(default_factory=list)
    scales: Dict[str, np.ndarray] = field(default_factory=dict)
    zeros: Dict[str, np.ndarray] = field(default_factory=dict)


class Calibrator:
    """Base calibrator class."""

    def __init__(self, config: QuantConfig):
        self.config = config
        self.statistics: Dict[str, Any] = {}

    def collect_statistics(
        self,
        model: Any,
        data: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Collect calibration statistics from model."""
        all_outputs = []

        for sample in data:
            if hasattr(model, 'forward'):
                output = model.forward(sample.get("input_ids", sample))
                if isinstance(output, np.ndarray):
                    all_outputs.append(output.flatten())

        if all_outputs:
            concat = np.concatenate(all_outputs)
            self.statistics = {
                "min": np.array([concat.min()]),
                "max": np.array([concat.max()]),
                "mean": np.array([concat.mean()]),
                "std": np.array([concat.std()]),
                "percentiles": {
                    p: np.percentile(np.abs(concat), p)
                    for p in [0.1, 1.0, 5.0, 95.0, 99.0, 99.9]
                }
            }

        return self.statistics

    def compute_scale_factors(
        self,
        stats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute scale factors from statistics."""
        min_vals = stats.get("min", np.array([0]))
        max_vals = stats.get("max", np.array([1]))

        abs_max = np.maximum(np.abs(min_vals), np.abs(max_vals))
        scales = abs_max / 127.0
        scales = np.where(scales == 0, 1.0, scales)

        return scales

    def calibrate(
        self,
        data: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate quantization parameters."""
        all_data = np.concatenate([d.flatten() for d in data])

        min_val = all_data.min()
        max_val = all_data.max()

        abs_max = max(abs(min_val), abs(max_val))
        bits = self.config.bits if hasattr(self.config, 'bits') else 8
        scale = abs_max / (2 ** (bits - 1) - 1)
        zero = np.array([0])

        return np.array([scale]), zero

    @staticmethod
    def create(config: QuantConfig) -> 'Calibrator':
        """Factory method to create calibrator based on config."""
        if config.quant_type == QuantType.GPTQ:
            return HessianCalibrator(config)
        elif config.quant_type == QuantType.AWQ:
            return ActivationCalibrator(config)
        else:
            return Calibrator(config)


class HessianCalibrator(Calibrator):
    """Hessian-based calibration for GPTQ."""

    def __init__(self, config: QuantConfig):
        super().__init__(config)
        self.dampening = getattr(config, 'dampening', 0.01)

    def update_hessian(
        self,
        hessian: np.ndarray,
        batch: np.ndarray
    ) -> np.ndarray:
        """Update Hessian with new batch."""
        # H = X^T @ X
        batch_hessian = batch.T @ batch
        return hessian + batch_hessian

    def collect_layer_hessians(
        self,
        model: Any,
        layers: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Collect Hessian for each layer."""
        hessians = {}

        for name, layer in layers.items():
            if hasattr(layer, 'weight'):
                weight = layer.weight
                if isinstance(weight, np.ndarray):
                    in_dim = weight.shape[1] if len(weight.shape) > 1 else weight.shape[0]
                    # Initialize with identity scaled by dampening
                    hessian = np.eye(in_dim, dtype=np.float32) * self.dampening
                    hessians[name] = hessian

        return hessians

    def calibrate(
        self,
        model: Any,
        data: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Run full calibration."""
        stats = self.collect_statistics(model, data)

        return {
            "hessians": {},
            "statistics": stats
        }

    def distributed_calibrate(
        self,
        model: Any,
        data: List[Dict[str, np.ndarray]],
        num_gpus: int = 1
    ) -> Dict[str, Any]:
        """Distributed calibration across GPUs."""
        # Simulate distributed calibration
        return self.calibrate(model, data)


class ActivationCalibrator(Calibrator):
    """Activation-based calibration for AWQ."""

    def __init__(self, config: QuantConfig):
        super().__init__(config)
        self.cache: Dict[str, np.ndarray] = {}

    def compute_activation_scales(
        self,
        activations: List[np.ndarray]
    ) -> np.ndarray:
        """Compute per-channel activation scales."""
        if not activations:
            return np.array([1.0])

        # Stack activations
        stacked = np.vstack([a.reshape(-1, a.shape[-1]) if len(a.shape) > 1 else a for a in activations])

        # Compute max absolute value per channel
        scales = np.abs(stacked).max(axis=0)
        scales = np.where(scales == 0, 1.0, scales)

        return scales

    def compute_channel_statistics(
        self,
        activations: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute per-channel statistics."""
        return {
            "mean": np.mean(activations, axis=0),
            "std": np.std(activations, axis=0),
            "max": np.max(np.abs(activations), axis=0),
            "min": np.min(activations, axis=0)
        }

    def smooth_scales(
        self,
        scales: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """Smooth scales to reduce outlier impact."""
        mean_scale = np.mean(scales)
        smoothed = alpha * scales + (1 - alpha) * mean_scale
        return smoothed

    def save_cache(self, path: str, data: np.ndarray):
        """Save calibration cache."""
        np.savez(path, scales=data)

    def load_cache(self, path: str) -> np.ndarray:
        """Load calibration cache."""
        loaded = np.load(path)
        return loaded["scales"]


def compute_hessian(
    inputs: List[np.ndarray],
    dampening: float = 0.01
) -> np.ndarray:
    """Compute Hessian from input activations."""
    if not inputs:
        return np.array([[]])

    # Get feature dimension
    first = inputs[0]
    if len(first.shape) > 1:
        feature_dim = first.shape[-1]
    else:
        feature_dim = len(first)

    # Initialize Hessian
    hessian = np.zeros((feature_dim, feature_dim), dtype=np.float32)

    # Accumulate H = sum(X^T @ X)
    for inp in inputs:
        if len(inp.shape) == 1:
            inp = inp.reshape(1, -1)
        elif len(inp.shape) > 2:
            inp = inp.reshape(-1, inp.shape[-1])

        hessian += inp.T @ inp

    # Average and add dampening
    hessian /= len(inputs)
    hessian += dampening * np.eye(feature_dim, dtype=np.float32)

    return hessian


def collect_activations(
    activations: List[np.ndarray]
) -> Dict[str, np.ndarray]:
    """Collect and aggregate activation statistics."""
    if not activations:
        return {"mean": np.array([]), "std": np.array([]), "max": np.array([])}

    # Get feature dimension
    feature_dim = activations[0].shape[-1]

    # Stack all activations
    all_acts = []
    for act in activations:
        if len(act.shape) > 2:
            act = act.reshape(-1, act.shape[-1])
        elif len(act.shape) == 1:
            act = act.reshape(1, -1)
        all_acts.append(act)

    stacked = np.vstack(all_acts)

    return {
        "mean": np.mean(stacked, axis=0),
        "std": np.std(stacked, axis=0),
        "max": np.max(np.abs(stacked), axis=0)
    }


# Legacy calibrators for backward compatibility
class MinMaxCalibrator(Calibrator):
    """Min-max calibration."""

    def calibrate(self, data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Find min/max values for quantization."""
        all_data = np.concatenate([d.flatten() for d in data])

        min_val = all_data.min()
        max_val = all_data.max()

        abs_max = max(abs(min_val), abs(max_val))
        bits = getattr(self.config, 'bits', 8)
        scale = abs_max / (2 ** (bits - 1) - 1)
        zero = np.array([0])

        return np.array([scale]), zero


class PercentileCalibrator(Calibrator):
    """Percentile-based calibration."""

    def __init__(self, config: QuantConfig, percentile: float = 99.99):
        super().__init__(config)
        self.percentile = percentile

    def calibrate(self, data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Find percentile values for quantization."""
        all_data = np.concatenate([d.flatten() for d in data])

        low = np.percentile(all_data, 100 - self.percentile)
        high = np.percentile(all_data, self.percentile)

        abs_max = max(abs(low), abs(high))
        bits = getattr(self.config, 'bits', 8)
        scale = abs_max / (2 ** (bits - 1) - 1)
        zero = np.array([0])

        return np.array([scale]), zero


class MSECalibrator(Calibrator):
    """MSE-based calibration that minimizes quantization error."""

    def __init__(self, config: QuantConfig, num_bins: int = 2048):
        super().__init__(config)
        self.num_bins = num_bins

    def calibrate(self, data: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Find optimal quantization parameters using MSE minimization."""
        all_data = np.concatenate([d.flatten() for d in data])

        # Initial scale based on min/max
        abs_max = np.abs(all_data).max()
        bits = getattr(self.config, 'bits', 8)
        qmax = 2 ** (bits - 1) - 1

        # Grid search for optimal scale
        best_scale = abs_max / qmax
        best_mse = float('inf')

        # Try different scale factors
        for factor in np.linspace(0.5, 1.0, 20):
            scale = abs_max * factor / qmax
            quantized = np.clip(np.round(all_data / scale), -qmax - 1, qmax)
            dequantized = quantized * scale
            mse = np.mean((all_data - dequantized) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_scale = scale

        zero = np.array([0])
        return np.array([best_scale]), zero


def collect_calibration_data(
    model: Any,
    dataloader: Any,
    num_samples: int = 128
) -> CalibrationData:
    """Collect calibration data from model."""
    calib_data = CalibrationData()
    activations = {}

    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break

        if hasattr(model, '__call__'):
            model(batch)
        else:
            model.forward(batch)

        count += len(batch) if hasattr(batch, '__len__') else 1

    for name, acts in activations.items():
        calib_data.activations.extend(acts)

    return calib_data


def calibrate_model(
    model: Any,
    calib_data: CalibrationData,
    calibrator: Calibrator
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Calibrate model with collected data."""
    results = {}

    if hasattr(calib_data, 'activations') and isinstance(calib_data.activations, dict):
        for name, acts in calib_data.activations.items():
            scale, zero = calibrator.calibrate(acts)
            results[name] = (scale, zero)

    return results
