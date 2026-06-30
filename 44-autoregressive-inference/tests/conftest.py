"""Pytest fixtures for autoregressive inference tests."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Try to import torch, set flag for conditional tests
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np

from autoregressive_inference.requests import (
    RequestStatus,
    RequestPriority,
    SamplingParams,
    InferenceRequest,
    RequestManager,
)
from autoregressive_inference.kv_cache import (
    KVCacheConfig,
    KVCacheBlock,
    PagedKVCacheManager,
    SlidingWindowCache,
)
from autoregressive_inference.batching import (
    ContinuousBatcher,
    BatchedInputs,
)
from autoregressive_inference.sampling import TokenSampler
from autoregressive_inference.scheduler import InferenceScheduler
from autoregressive_inference.speculative import SpeculativeDecoder


# Skip markers
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="Requires PyTorch")


# ============================================================================
# KV Cache Fixtures
# ============================================================================

@pytest.fixture
def small_kv_config():
    """Small KV cache configuration for testing."""
    return KVCacheConfig(
        num_layers=2,
        num_heads=4,
        head_dim=64,
        max_seq_len=256,
        block_size=16,
        dtype="float32",
        device="cpu"
    )


@pytest.fixture
def medium_kv_config():
    """Medium KV cache configuration for testing."""
    return KVCacheConfig(
        num_layers=4,
        num_heads=8,
        head_dim=64,
        max_seq_len=512,
        block_size=16,
        dtype="float32",
        device="cpu"
    )


@pytest.fixture
def kv_cache_block(small_kv_config):
    """Create a single KV cache block."""
    return KVCacheBlock(block_id=0, config=small_kv_config)


@pytest.fixture
def paged_kv_cache(small_kv_config):
    """Create a paged KV cache manager with 10 blocks."""
    return PagedKVCacheManager(config=small_kv_config, num_blocks=10)


@pytest.fixture
def large_paged_kv_cache(medium_kv_config):
    """Create a larger paged KV cache manager."""
    return PagedKVCacheManager(config=medium_kv_config, num_blocks=100)


@pytest.fixture
def sliding_window_cache(small_kv_config):
    """Create a sliding window cache."""
    return SlidingWindowCache(config=small_kv_config, window_size=32)


# ============================================================================
# Request Fixtures
# ============================================================================

@pytest.fixture
def default_sampling_params():
    """Default sampling parameters."""
    return SamplingParams(
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        max_tokens=100
    )


@pytest.fixture
def greedy_sampling_params():
    """Greedy sampling parameters."""
    return SamplingParams(
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        max_tokens=100
    )


@pytest.fixture
def creative_sampling_params():
    """Creative sampling parameters with high temperature."""
    return SamplingParams(
        temperature=1.5,
        top_k=100,
        top_p=0.95,
        max_tokens=200,
        repetition_penalty=1.2
    )


@pytest.fixture
def simple_request(default_sampling_params):
    """Create a simple inference request."""
    return InferenceRequest(
        request_id="test-001",
        prompt="Hello, world!",
        prompt_token_ids=[1, 2, 3, 4, 5],
        sampling_params=default_sampling_params,
        priority=RequestPriority.NORMAL
    )


@pytest.fixture
def high_priority_request(default_sampling_params):
    """Create a high priority request."""
    return InferenceRequest(
        request_id="test-high",
        prompt="Urgent request",
        prompt_token_ids=[1, 2, 3],
        sampling_params=default_sampling_params,
        priority=RequestPriority.HIGH
    )


@pytest.fixture
def long_prompt_request(default_sampling_params):
    """Create a request with a long prompt."""
    return InferenceRequest(
        request_id="test-long",
        prompt="Long prompt " * 50,
        prompt_token_ids=list(range(1, 201)),  # 200 tokens
        sampling_params=default_sampling_params,
        priority=RequestPriority.NORMAL
    )


@pytest.fixture
def request_manager():
    """Create a request manager."""
    return RequestManager(max_queue_size=100)


@pytest.fixture
def populated_request_manager(request_manager, default_sampling_params):
    """Create a request manager with some requests."""
    for i in range(5):
        request = InferenceRequest(
            request_id=f"req-{i:03d}",
            prompt=f"Request {i}",
            prompt_token_ids=[1, 2, 3, 4, 5],
            sampling_params=default_sampling_params,
            priority=RequestPriority.NORMAL
        )
        request_manager.add_request(request)
    return request_manager


# ============================================================================
# Batching Fixtures
# ============================================================================

@pytest.fixture
def continuous_batcher():
    """Create a continuous batcher."""
    return ContinuousBatcher(
        max_batch_size=8,
        max_prefill_tokens=1024,
        max_decode_tokens=512
    )


@pytest.fixture
def large_batcher():
    """Create a larger batcher for stress testing."""
    return ContinuousBatcher(
        max_batch_size=64,
        max_prefill_tokens=4096,
        max_decode_tokens=2048
    )


@pytest.fixture
def sample_requests_for_batching(default_sampling_params):
    """Create a list of requests for batching tests."""
    requests = []
    for i in range(10):
        request = InferenceRequest(
            request_id=f"batch-req-{i:03d}",
            prompt=f"Request {i}",
            prompt_token_ids=list(range(1, 51 + i * 10)),  # Varying lengths
            sampling_params=default_sampling_params,
            priority=RequestPriority.NORMAL
        )
        requests.append(request)
    return requests


# ============================================================================
# Sampling Fixtures
# ============================================================================

@pytest.fixture
def token_sampler():
    """Create a token sampler with fixed seed."""
    return TokenSampler(seed=42)


@pytest.fixture
def random_logits():
    """Create random logits for testing."""
    np.random.seed(42)
    return np.random.randn(1, 1000).astype(np.float32)


@pytest.fixture
def skewed_logits():
    """Create skewed logits with a clear winner."""
    logits = np.full((1, 1000), -10.0, dtype=np.float32)
    logits[0, 42] = 10.0  # Token 42 is strongly preferred
    logits[0, 43] = 5.0   # Token 43 is second
    logits[0, 44] = 2.0   # Token 44 is third
    return logits


@pytest.fixture
def uniform_logits():
    """Create uniform logits."""
    return np.zeros((1, 100), dtype=np.float32)


@pytest.fixture
def batch_logits():
    """Create batched logits."""
    np.random.seed(42)
    return np.random.randn(4, 1000).astype(np.float32)


# ============================================================================
# Scheduler Fixtures
# ============================================================================

@pytest.fixture
def basic_scheduler(paged_kv_cache, request_manager, continuous_batcher, token_sampler):
    """Create a basic scheduler without a model."""
    return InferenceScheduler(
        model=None,
        kv_cache_manager=paged_kv_cache,
        request_manager=request_manager,
        batcher=continuous_batcher,
        sampler=token_sampler
    )


# ============================================================================
# Speculative Decoding Fixtures
# ============================================================================

@pytest.fixture
def speculative_decoder():
    """Create a speculative decoder without models (for mock testing)."""
    return SpeculativeDecoder(
        target_model=None,
        draft_model=None,
        num_speculative_tokens=4
    )


# ============================================================================
# Tensor Fixtures (with torch if available)
# ============================================================================

@pytest.fixture
def torch_random_logits():
    """Create random logits as torch tensor."""
    if HAS_TORCH:
        torch.manual_seed(42)
        return torch.randn(1, 1000)
    return None


@pytest.fixture
def torch_skewed_logits():
    """Create skewed logits as torch tensor."""
    if HAS_TORCH:
        logits = torch.full((1, 1000), -10.0)
        logits[0, 42] = 10.0
        logits[0, 43] = 5.0
        logits[0, 44] = 2.0
        return logits
    return None


@pytest.fixture
def torch_kv_tensors(small_kv_config):
    """Create sample KV tensors."""
    if HAS_TORCH:
        k = torch.randn(small_kv_config.num_heads, small_kv_config.head_dim)
        v = torch.randn(small_kv_config.num_heads, small_kv_config.head_dim)
        return k, v
    return None, None


@pytest.fixture
def numpy_kv_tensors(small_kv_config):
    """Create sample KV tensors as numpy arrays."""
    np.random.seed(42)
    k = np.random.randn(small_kv_config.num_heads, small_kv_config.head_dim).astype(np.float32)
    v = np.random.randn(small_kv_config.num_heads, small_kv_config.head_dim).astype(np.float32)
    return k, v


# ============================================================================
# Helper Functions
# ============================================================================

def create_request(
    request_id: str,
    prompt_length: int = 10,
    priority: RequestPriority = RequestPriority.NORMAL,
    max_tokens: int = 100
) -> InferenceRequest:
    """Helper to create a request with specified parameters."""
    return InferenceRequest(
        request_id=request_id,
        prompt="Test prompt",
        prompt_token_ids=list(range(1, prompt_length + 1)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        priority=priority
    )


def create_running_request(
    request_id: str,
    prompt_length: int = 10,
    output_length: int = 5
) -> InferenceRequest:
    """Helper to create a request in running/decode state."""
    request = create_request(request_id, prompt_length)
    request.status = RequestStatus.RUNNING_DECODE
    request.output_token_ids = list(range(100, 100 + output_length))
    return request
