"""Autoregressive Inference Engine.

A high-performance LLM inference engine implementing continuous batching,
KV cache management, and speculative decoding from first principles.
"""

from .requests import (
    RequestStatus,
    RequestPriority,
    SamplingParams,
    InferenceRequest,
    RequestManager,
)
from .kv_cache import (
    KVCacheConfig,
    KVCacheBlock,
    PagedKVCacheManager,
    SlidingWindowCache,
)
from .batching import (
    ContinuousBatcher,
    BatchedInputs,
    SchedulingPolicy,
    FIFOPolicy,
    PriorityPolicy,
    ShortestJobFirstPolicy,
)
from .sampling import TokenSampler
from .scheduler import InferenceScheduler
from .speculative import (
    SpeculativeDecoder,
    TreeSpeculativeDecoder,
    SpeculativeStats,
    DraftModel,
    TargetModel,
)

__version__ = "0.1.0"

__all__ = [
    # Requests
    "RequestStatus",
    "RequestPriority",
    "SamplingParams",
    "InferenceRequest",
    "RequestManager",
    # KV Cache
    "KVCacheConfig",
    "KVCacheBlock",
    "PagedKVCacheManager",
    "SlidingWindowCache",
    # Batching
    "ContinuousBatcher",
    "BatchedInputs",
    "SchedulingPolicy",
    "FIFOPolicy",
    "PriorityPolicy",
    "ShortestJobFirstPolicy",
    # Sampling
    "TokenSampler",
    # Scheduler
    "InferenceScheduler",
    # Speculative
    "SpeculativeDecoder",
    "TreeSpeculativeDecoder",
    "SpeculativeStats",
    "DraftModel",
    "TargetModel",
]
