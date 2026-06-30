"""Test fixtures and helpers for vector quantized LLM tests."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os


class ModelFactory:
    """Factory for creating test models."""

    @staticmethod
    def create_transformer_model(
        num_layers: int = 12,
        hidden_size: int = 768,
        num_heads: int = 12,
        vocab_size: int = 50257,
        max_seq_length: int = 512
    ) -> Dict[str, Any]:
        """Create a mock transformer model."""
        model = {
            "config": {
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "num_heads": num_heads,
                "vocab_size": vocab_size,
                "max_seq_length": max_seq_length,
                "intermediate_size": hidden_size * 4,
            },
            "layers": {}
        }

        # Create layers
        for i in range(num_layers):
            layer_prefix = f"layer_{i}"

            # Attention weights
            model["layers"][f"{layer_prefix}.attention.q_proj"] = {
                "weight": np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02,
                "bias": np.zeros(hidden_size, dtype=np.float32)
            }
            model["layers"][f"{layer_prefix}.attention.k_proj"] = {
                "weight": np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02,
                "bias": np.zeros(hidden_size, dtype=np.float32)
            }
            model["layers"][f"{layer_prefix}.attention.v_proj"] = {
                "weight": np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02,
                "bias": np.zeros(hidden_size, dtype=np.float32)
            }
            model["layers"][f"{layer_prefix}.attention.out_proj"] = {
                "weight": np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.02,
                "bias": np.zeros(hidden_size, dtype=np.float32)
            }

            # MLP weights
            model["layers"][f"{layer_prefix}.mlp.fc1"] = {
                "weight": np.random.randn(hidden_size, hidden_size * 4).astype(np.float32) * 0.02,
                "bias": np.zeros(hidden_size * 4, dtype=np.float32)
            }
            model["layers"][f"{layer_prefix}.mlp.fc2"] = {
                "weight": np.random.randn(hidden_size * 4, hidden_size).astype(np.float32) * 0.02,
                "bias": np.zeros(hidden_size, dtype=np.float32)
            }

            # Layer norms
            model["layers"][f"{layer_prefix}.ln1"] = {
                "weight": np.ones(hidden_size, dtype=np.float32),
                "bias": np.zeros(hidden_size, dtype=np.float32)
            }
            model["layers"][f"{layer_prefix}.ln2"] = {
                "weight": np.ones(hidden_size, dtype=np.float32),
                "bias": np.zeros(hidden_size, dtype=np.float32)
            }

        # Embeddings and output
        model["layers"]["embeddings"] = {
            "weight": np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.02
        }
        model["layers"]["lm_head"] = {
            "weight": np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.02
        }

        return model

    @staticmethod
    def create_llama_model(
        num_layers: int = 32,
        hidden_size: int = 4096,
        num_heads: int = 32,
        vocab_size: int = 32000
    ) -> Dict[str, Any]:
        """Create a mock LLaMA-style model."""
        model = ModelFactory.create_transformer_model(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            vocab_size=vocab_size
        )

        # Add RMSNorm instead of LayerNorm
        for i in range(num_layers):
            layer_prefix = f"layer_{i}"
            model["layers"][f"{layer_prefix}.rms_norm"] = {
                "weight": np.ones(hidden_size, dtype=np.float32)
            }

        # Add RoPE embeddings
        model["rope_embeddings"] = {
            "freqs": np.random.randn(hidden_size // num_heads).astype(np.float32)
        }

        return model

    @staticmethod
    def create_small_model() -> Dict[str, Any]:
        """Create a small model for quick testing."""
        return ModelFactory.create_transformer_model(
            num_layers=2,
            hidden_size=128,
            num_heads=4,
            vocab_size=1000,
            max_seq_length=64
        )


class DataGenerator:
    """Generate test data for calibration and inference."""

    @staticmethod
    def generate_text_data(
        num_samples: int,
        seq_length: int,
        vocab_size: int = 50257,
        batch_size: Optional[int] = None
    ) -> List[Dict[str, np.ndarray]]:
        """Generate random text data."""
        data = []
        for _ in range(num_samples):
            if batch_size:
                input_ids = np.random.randint(0, vocab_size, (batch_size, seq_length))
                attention_mask = np.ones((batch_size, seq_length), dtype=np.int32)
            else:
                input_ids = np.random.randint(0, vocab_size, seq_length)
                attention_mask = np.ones(seq_length, dtype=np.int32)

            data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

        return data

    @staticmethod
    def generate_calibration_data(
        num_samples: int = 100,
        seq_length: int = 128,
        hidden_size: int = 768
    ) -> List[np.ndarray]:
        """Generate calibration data (activations)."""
        activations = []
        for _ in range(num_samples):
            # Generate activations with different distributions
            if np.random.rand() < 0.3:
                # Normal distribution
                act = np.random.randn(seq_length, hidden_size).astype(np.float32)
            elif np.random.rand() < 0.6:
                # Skewed distribution
                act = np.random.gamma(2, 2, (seq_length, hidden_size)).astype(np.float32)
            else:
                # Sparse activations
                act = np.random.randn(seq_length, hidden_size).astype(np.float32)
                mask = np.random.rand(seq_length, hidden_size) > 0.7
                act[mask] = 0

            activations.append(act)

        return activations

    @staticmethod
    def generate_hessian_data(
        input_dim: int,
        num_samples: int = 100
    ) -> np.ndarray:
        """Generate Hessian matrix for GPTQ calibration."""
        # Generate input samples
        X = np.random.randn(num_samples, input_dim).astype(np.float32)

        # Compute Hessian approximation
        H = X.T @ X / num_samples

        # Add dampening for stability
        H += np.eye(input_dim) * 0.01

        return H

    @staticmethod
    def generate_activation_scales(
        num_channels: int,
        distribution: str = "uniform"
    ) -> np.ndarray:
        """Generate activation scales for AWQ."""
        if distribution == "uniform":
            scales = np.ones(num_channels, dtype=np.float32)
        elif distribution == "random":
            scales = np.random.rand(num_channels).astype(np.float32) + 0.1
        elif distribution == "peaked":
            # Some channels have much higher activation
            scales = np.ones(num_channels, dtype=np.float32) * 0.1
            peak_channels = np.random.choice(num_channels, size=num_channels // 10, replace=False)
            scales[peak_channels] = 1.0
        else:
            scales = np.ones(num_channels, dtype=np.float32)

        return scales


class QuantizationTestHelper:
    """Helper functions for quantization testing."""

    @staticmethod
    def compute_quantization_error(
        original: np.ndarray,
        quantized: np.ndarray
    ) -> Dict[str, float]:
        """Compute various quantization error metrics."""
        # Absolute error
        abs_error = np.abs(original - quantized)

        # Relative error
        rel_error = abs_error / (np.abs(original) + 1e-8)

        # Metrics
        metrics = {
            "mse": np.mean((original - quantized) ** 2),
            "mae": np.mean(abs_error),
            "max_abs_error": np.max(abs_error),
            "mean_rel_error": np.mean(rel_error),
            "max_rel_error": np.max(rel_error),
            "snr": 10 * np.log10(np.var(original) / (np.var(original - quantized) + 1e-8))
        }

        return metrics

    @staticmethod
    def verify_quantized_tensor(qtensor: Any) -> bool:
        """Verify quantized tensor properties."""
        checks = []

        # Check data type
        if hasattr(qtensor, "dtype"):
            checks.append(qtensor.dtype in [np.int8, np.int4, np.uint8, np.uint4])

        # Check scale exists
        checks.append(hasattr(qtensor, "scale") and qtensor.scale is not None)

        # Check scale is positive
        if hasattr(qtensor, "scale"):
            checks.append(np.all(qtensor.scale > 0))

        # Check zero point if exists
        if hasattr(qtensor, "zero_point") and qtensor.zero_point is not None:
            if qtensor.dtype == np.int8:
                checks.append(np.all(np.abs(qtensor.zero_point) <= 127))

        # Check shape preserved
        if hasattr(qtensor, "shape") and hasattr(qtensor, "original_shape"):
            checks.append(qtensor.shape == qtensor.original_shape)

        return all(checks)

    @staticmethod
    def generate_edge_case_weights() -> List[np.ndarray]:
        """Generate edge case weights for testing."""
        edge_cases = []

        # All zeros
        edge_cases.append(np.zeros((256, 256), dtype=np.float32))

        # All ones
        edge_cases.append(np.ones((256, 256), dtype=np.float32))

        # Very small values
        edge_cases.append(np.random.randn(256, 256).astype(np.float32) * 1e-8)

        # Very large values
        edge_cases.append(np.random.randn(256, 256).astype(np.float32) * 1e8)

        # Sparse matrix
        sparse = np.zeros((256, 256), dtype=np.float32)
        sparse[np.random.rand(256, 256) > 0.9] = np.random.randn()
        edge_cases.append(sparse)

        # Diagonal matrix
        edge_cases.append(np.eye(256, dtype=np.float32))

        # Rank-1 matrix
        u = np.random.randn(256, 1).astype(np.float32)
        v = np.random.randn(1, 256).astype(np.float32)
        edge_cases.append(u @ v)

        return edge_cases


class BenchmarkHelper:
    """Helper for benchmarking tests."""

    @staticmethod
    def create_benchmark_suite() -> Dict[str, Dict]:
        """Create benchmark test suite."""
        return {
            "small": {
                "batch_size": 1,
                "seq_length": 128,
                "hidden_size": 768,
                "num_layers": 12,
                "expected_latency_ms": 10
            },
            "medium": {
                "batch_size": 4,
                "seq_length": 512,
                "hidden_size": 1024,
                "num_layers": 24,
                "expected_latency_ms": 50
            },
            "large": {
                "batch_size": 8,
                "seq_length": 2048,
                "hidden_size": 4096,
                "num_layers": 32,
                "expected_latency_ms": 200
            }
        }

    @staticmethod
    def measure_memory_usage(func, *args, **kwargs) -> Dict[str, float]:
        """Measure memory usage of a function."""
        import tracemalloc

        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        # Run function
        result = func(*args, **kwargs)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Calculate memory usage
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
        total_memory = sum(stat.size_diff for stat in top_stats)

        return {
            "result": result,
            "memory_bytes": total_memory,
            "memory_mb": total_memory / (1024 * 1024)
        }

    @staticmethod
    def profile_quantization(quantizer, weight: np.ndarray) -> Dict[str, Any]:
        """Profile quantization operation."""
        import time

        # Measure time
        start = time.perf_counter()
        qtensor = quantizer.quantize_weight(weight)
        quantization_time = time.perf_counter() - start

        # Measure dequantization
        start = time.perf_counter()
        dequant = qtensor.dequantize()
        dequantization_time = time.perf_counter() - start

        # Calculate compression
        original_size = weight.nbytes
        quantized_size = qtensor.memory_usage() if hasattr(qtensor, "memory_usage") else 0
        compression_ratio = original_size / max(quantized_size, 1)

        return {
            "quantization_time": quantization_time,
            "dequantization_time": dequantization_time,
            "compression_ratio": compression_ratio,
            "original_size_bytes": original_size,
            "quantized_size_bytes": quantized_size
        }


class TestConfig:
    """Test configuration settings."""

    # Tolerances for different quantization types
    TOLERANCES = {
        "int8": {
            "mse": 0.01,
            "rel_error": 0.05,
            "snr_db": 30
        },
        "int4": {
            "mse": 0.05,
            "rel_error": 0.15,
            "snr_db": 20
        },
        "gptq": {
            "mse": 0.02,
            "rel_error": 0.08,
            "snr_db": 25
        },
        "awq": {
            "mse": 0.02,
            "rel_error": 0.08,
            "snr_db": 25
        }
    }

    # Performance baselines
    PERFORMANCE_BASELINES = {
        "int8_latency_ms": 10,
        "int4_latency_ms": 8,
        "gptq_latency_ms": 12,
        "awq_latency_ms": 11,
        "memory_reduction_factor": 3.5
    }

    # Test data sizes
    TEST_SIZES = {
        "quick": {
            "num_samples": 10,
            "seq_length": 64,
            "batch_size": 2
        },
        "normal": {
            "num_samples": 100,
            "seq_length": 128,
            "batch_size": 4
        },
        "thorough": {
            "num_samples": 500,
            "seq_length": 512,
            "batch_size": 8
        }
    }


# Export test data for reuse
SAMPLE_WEIGHTS = {
    "small": np.random.randn(128, 128).astype(np.float32),
    "medium": np.random.randn(768, 768).astype(np.float32),
    "large": np.random.randn(4096, 4096).astype(np.float32),
}

SAMPLE_ACTIVATIONS = DataGenerator.generate_calibration_data(
    num_samples=10,
    seq_length=128,
    hidden_size=768
)

SAMPLE_INPUT_IDS = np.random.randint(0, 50257, (4, 128))

SAMPLE_HESSIAN = DataGenerator.generate_hessian_data(768, num_samples=100)