"""Pytest fixtures for on-device LLM tests."""

import sys
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import pytest

from on_device_llm.loader import ModelConfig, MockGGUFLoader
from on_device_llm.memory import MemoryPool, KVCache
from on_device_llm.quantization import GGMLType
from on_device_llm.inference import TransformerInference, GenerationConfig, Sampler


# =============================================================================
# Model Configuration Fixtures
# =============================================================================


@pytest.fixture
def tiny_config() -> ModelConfig:
    """Tiny model config for fast unit tests."""
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        max_seq_len=32,
        rope_theta=10000.0,
        norm_eps=1e-5
    )


@pytest.fixture
def small_config() -> ModelConfig:
    """Small model config for integration tests."""
    return ModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=4,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=256,
        max_seq_len=64,
        rope_theta=10000.0,
        norm_eps=1e-5
    )


@pytest.fixture
def gqa_config() -> ModelConfig:
    """Config with Grouped Query Attention (GQA)."""
    return ModelConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_heads=8,
        num_kv_heads=2,  # 4x fewer KV heads than Q heads
        intermediate_size=128,
        max_seq_len=32,
        rope_theta=10000.0,
        norm_eps=1e-5
    )


# =============================================================================
# Model Loader Fixtures
# =============================================================================


@pytest.fixture
def mock_loader(tiny_config: ModelConfig) -> MockGGUFLoader:
    """Mock GGUF loader with tiny model."""
    return MockGGUFLoader(tiny_config, dtype=GGMLType.F32)


@pytest.fixture
def mock_loader_small(small_config: ModelConfig) -> MockGGUFLoader:
    """Mock GGUF loader with small model."""
    return MockGGUFLoader(small_config, dtype=GGMLType.F32)


@pytest.fixture
def mock_loader_gqa(gqa_config: ModelConfig) -> MockGGUFLoader:
    """Mock GGUF loader with GQA model."""
    return MockGGUFLoader(gqa_config, dtype=GGMLType.F32)


# =============================================================================
# Memory Fixtures
# =============================================================================


@pytest.fixture
def memory_pool() -> MemoryPool:
    """Memory pool for testing."""
    return MemoryPool(size_bytes=1024 * 1024, alignment=64)  # 1MB


@pytest.fixture
def kv_cache(tiny_config: ModelConfig) -> KVCache:
    """KV cache for testing."""
    return KVCache(
        num_layers=tiny_config.num_layers,
        num_heads=tiny_config.num_kv_heads,
        head_dim=tiny_config.head_dim,
        max_seq_len=tiny_config.max_seq_len
    )


@pytest.fixture
def sliding_window_cache(tiny_config: ModelConfig) -> KVCache:
    """KV cache with sliding window."""
    return KVCache(
        num_layers=tiny_config.num_layers,
        num_heads=tiny_config.num_kv_heads,
        head_dim=tiny_config.head_dim,
        max_seq_len=tiny_config.max_seq_len,
        window_size=8
    )


# =============================================================================
# Inference Fixtures
# =============================================================================


@pytest.fixture
def inference_engine(mock_loader: MockGGUFLoader) -> TransformerInference:
    """Transformer inference engine for testing."""
    return TransformerInference(mock_loader)


@pytest.fixture
def inference_engine_gqa(mock_loader_gqa: MockGGUFLoader) -> TransformerInference:
    """Inference engine with GQA support."""
    return TransformerInference(mock_loader_gqa)


@pytest.fixture
def generation_config() -> GenerationConfig:
    """Default generation config for testing."""
    return GenerationConfig(
        max_new_tokens=10,
        temperature=1.0,
        top_k=10,
        top_p=0.9,
        eos_token_id=0,
        do_sample=True
    )


@pytest.fixture
def greedy_config() -> GenerationConfig:
    """Greedy generation config."""
    return GenerationConfig(
        max_new_tokens=10,
        temperature=1.0,
        do_sample=False,
        eos_token_id=0
    )


@pytest.fixture
def sampler() -> Sampler:
    """Sampler for testing."""
    return Sampler(temperature=1.0, top_k=10, top_p=0.9)


# =============================================================================
# Tensor Fixtures
# =============================================================================


@pytest.fixture
def random_tensor_1d() -> np.ndarray:
    """Random 1D tensor for testing."""
    np.random.seed(42)
    return np.random.randn(64).astype(np.float32)


@pytest.fixture
def random_tensor_2d() -> np.ndarray:
    """Random 2D tensor for testing."""
    np.random.seed(42)
    return np.random.randn(32, 64).astype(np.float32)


@pytest.fixture
def random_weights() -> np.ndarray:
    """Random weight tensor for normalization."""
    np.random.seed(42)
    return np.abs(np.random.randn(64).astype(np.float32)) + 0.1


@pytest.fixture
def random_matrix_a() -> np.ndarray:
    """Random matrix A for matmul tests."""
    np.random.seed(42)
    return np.random.randn(16, 32).astype(np.float32)


@pytest.fixture
def random_matrix_b() -> np.ndarray:
    """Random matrix B for matmul tests."""
    np.random.seed(43)
    return np.random.randn(32, 24).astype(np.float32)


@pytest.fixture
def random_embedding() -> np.ndarray:
    """Random embedding for RoPE tests (even dimension)."""
    np.random.seed(42)
    return np.random.randn(16).astype(np.float32)


# =============================================================================
# Quantization Fixtures
# =============================================================================


@pytest.fixture
def tensor_for_quantization() -> np.ndarray:
    """Tensor suitable for quantization testing."""
    np.random.seed(42)
    # Use a moderate range of values
    return np.random.randn(128).astype(np.float32) * 2.0


@pytest.fixture
def large_tensor_for_quantization() -> np.ndarray:
    """Large tensor for quantization stress testing."""
    np.random.seed(42)
    return np.random.randn(1024).astype(np.float32) * 2.0


@pytest.fixture
def shaped_tensor_for_quantization() -> np.ndarray:
    """2D tensor for quantization with shape preservation."""
    np.random.seed(42)
    return np.random.randn(32, 64).astype(np.float32) * 2.0


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    yield
    # Reset to non-deterministic after test
    np.random.seed(None)


@pytest.fixture(autouse=True)
def reset_random_state():
    """Reset random state before each test."""
    np.random.seed(42)
