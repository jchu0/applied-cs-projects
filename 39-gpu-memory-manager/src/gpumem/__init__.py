"""GPUMem - Unified GPU memory management system."""

from .core import (
    MemoryBlock,
    MemoryPool,
    DeviceType,
    MemoryStats,
    MemoryConfig,
)
from .allocator import (
    Allocator,
    CachingAllocator,
    PoolAllocator,
    BuddyAllocator,
    SlabAllocator,
    # Multi-GPU Distribution
    LoadBalanceStrategy,
    GPUDeviceState,
    MultiGPUAllocator,
    DistributedTensor,
    DistributedTensorManager,
    GradientSynchronizer,
)
from .cache import (
    MemoryCache,
    TensorCache,
    DefragmentStrategy,
    CachePolicy,
)
from .profiler import (
    MemoryProfiler,
    AllocationTrace,
    MemorySnapshot,
    MemoryTimeline,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "MemoryBlock",
    "MemoryPool",
    "DeviceType",
    "MemoryStats",
    "MemoryConfig",
    # Allocator
    "Allocator",
    "CachingAllocator",
    "PoolAllocator",
    "BuddyAllocator",
    "SlabAllocator",
    # Multi-GPU Distribution
    "LoadBalanceStrategy",
    "GPUDeviceState",
    "MultiGPUAllocator",
    "DistributedTensor",
    "DistributedTensorManager",
    "GradientSynchronizer",
    # Cache
    "MemoryCache",
    "TensorCache",
    "DefragmentStrategy",
    "CachePolicy",
    # Profiler
    "MemoryProfiler",
    "AllocationTrace",
    "MemorySnapshot",
    "MemoryTimeline",
]
