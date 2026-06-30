# GPU Memory Manager - Technical Blueprint

## Executive Summary

A high-performance GPU memory allocator inspired by PyTorch's caching allocator, designed to minimize fragmentation, maximize memory reuse, and support stream-aware allocation for concurrent kernel execution. This system addresses the unique challenges of GPU memory management including high allocation latency, limited memory capacity, and the need for stream synchronization.

> **Concepts covered:** [§03 PyTorch + CUDA](../../03-machine-learning-engineering/06-cuda-optimization/pytorch-cuda/pytorch-cuda.md) (caching allocator, streams) · [§03 CUDA basics](../../03-machine-learning-engineering/06-cuda-optimization/cuda-basics/cuda-basics.md). The memory substrate beneath [Project 19 (GEMM kernels)](../19-gpu-kernel-optimization/), [Project 44 (autoregressive inference KV cache)](../44-autoregressive-inference/), [Project 48 (multi-GPU scheduler)](../48-multi-gpu-kernel-scheduler/), and [Project 40 (distributed autograd)](../40-distributed-autograd/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## System Architecture

### High-Level Architecture

```
+----------------------------------------------------------+
|                    User Applications                      |
|  (PyTorch, TensorFlow, Custom CUDA kernels)              |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                   Allocation API Layer                    |
|  alloc(size, stream) / free(ptr, stream) / stats()       |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    Caching Layer                          |
|  +-------------+  +-------------+  +------------------+  |
|  | Free Pools  |  | Active Set  |  | Stream Tracker   |  |
|  | (by size)   |  | (allocated) |  | (events/sync)    |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                   Block Allocator                         |
|  +-------------+  +-------------+  +------------------+  |
|  | Segment     |  | Block       |  | Fragmentation    |  |
|  | Manager     |  | Splitter    |  | Monitor          |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    CUDA Runtime                           |
|  cudaMalloc() / cudaFree() / cudaEventRecord()          |
+----------------------------------------------------------+
                              |
                              v
+----------------------------------------------------------+
|                    GPU Device Memory                      |
+----------------------------------------------------------+
```

### Core Design Principles

1. **Lazy Deallocation**: Never call cudaFree immediately; cache for reuse
2. **Stream Isolation**: Prevent use-after-free across streams
3. **Size Bucketing**: Fast allocation through size-based free lists
4. **Block Coalescing**: Merge adjacent free blocks to reduce fragmentation
5. **Memory Pressure Handling**: Graceful degradation under OOM conditions

## Component Design

### 1. Block and Segment Representation

```cpp
#include <cstdint>
#include <cstddef>
#include <set>
#include <map>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <cuda_runtime.h>

// Forward declarations
struct Block;
struct Segment;
struct StreamContext;

// Memory statistics
struct MemoryStats {
    size_t allocated_bytes;
    size_t reserved_bytes;
    size_t active_bytes;
    size_t num_allocations;
    size_t num_segments;
    size_t num_free_blocks;
    size_t peak_allocated_bytes;
    size_t peak_reserved_bytes;
    size_t num_alloc_retries;
    size_t num_ooms;
};

// Block states
enum class BlockState {
    FREE,      // In free pool, available for allocation
    ACTIVE,    // Currently allocated to user
    PENDING,   // Freed but waiting for stream sync
};

// A block represents a contiguous region within a segment
struct Block {
    void* ptr;                      // Device pointer
    size_t size;                    // Size in bytes
    size_t requested_size;          // Original requested size
    BlockState state;
    Segment* segment;               // Parent segment
    Block* prev;                    // Previous block in segment
    Block* next;                    // Next block in segment
    cudaStream_t stream;            // Allocation stream
    cudaEvent_t event;              // Sync event for pending frees
    int device;                     // GPU device ID

    // For efficient free pool lookup
    size_t size_class;              // Bucketed size class

    Block(void* p, size_t s, Segment* seg)
        : ptr(p), size(s), requested_size(s), state(BlockState::FREE),
          segment(seg), prev(nullptr), next(nullptr),
          stream(nullptr), event(nullptr), device(0), size_class(0) {}
};

// A segment is a large allocation from cudaMalloc
struct Segment {
    void* ptr;                      // Base device pointer
    size_t size;                    // Total segment size
    int device;                     // GPU device
    cudaStream_t stream;            // Allocation stream
    std::vector<Block*> blocks;     // Blocks in this segment
    bool is_small;                  // Small vs large segment pool

    Segment(void* p, size_t s, int dev, cudaStream_t str, bool small)
        : ptr(p), size(s), device(dev), stream(str), is_small(small) {}
};

// Size classes for bucketing (powers of 2, plus intermediate sizes)
class SizeClassifier {
public:
    static const size_t kMinBlockSize = 512;           // Minimum allocation
    static const size_t kSmallSize = 1048576;          // 1MB threshold
    static const size_t kSmallBuffer = 2097152;        // 2MB small segment
    static const size_t kLargeBuffer = 20971520;       // 20MB large segment
    static const size_t kRoundLarge = 2097152;         // Round large to 2MB

    static size_t round_size(size_t size) {
        if (size < kMinBlockSize) {
            return kMinBlockSize;
        } else if (size < kSmallSize) {
            // Round to power of 2
            size_t power = 1;
            while (power < size) {
                power *= 2;
            }
            return power;
        } else {
            // Round up to kRoundLarge boundary
            return ((size + kRoundLarge - 1) / kRoundLarge) * kRoundLarge;
        }
    }

    static size_t get_size_class(size_t size) {
        // Convert size to class index for free pool lookup
        if (size <= kMinBlockSize) return 0;

        size_t rounded = round_size(size);
        if (rounded < kSmallSize) {
            // log2 based classes
            size_t log_size = 0;
            size_t temp = rounded;
            while (temp > kMinBlockSize) {
                temp >>= 1;
                log_size++;
            }
            return log_size;
        } else {
            // 2MB based classes
            return 20 + (rounded / kRoundLarge);
        }
    }
};
```

### 2. Block Allocator

```cpp
class BlockAllocator {
public:
    // Allocation strategies
    enum class Strategy {
        BEST_FIT,    // Find smallest block that fits
        FIRST_FIT,   // Find first block that fits
        BUDDY,       // Buddy system allocation
    };

private:
    Strategy strategy_;

    // Free pools organized by size class
    // Maps size_class -> set of blocks ordered by size
    std::map<size_t, std::set<Block*, BlockSizeCompare>> free_pools_;

    // All active allocations: ptr -> Block
    std::unordered_map<void*, Block*> active_blocks_;

    // All segments
    std::vector<Segment*> segments_;

    // Fragmentation tracking
    size_t total_free_bytes_;
    size_t largest_free_block_;

    std::mutex mutex_;

public:
    BlockAllocator(Strategy strategy = Strategy::BEST_FIT)
        : strategy_(strategy), total_free_bytes_(0), largest_free_block_(0) {}

    // Allocate a block of given size
    Block* allocate(size_t size, int device, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);

        size_t alloc_size = SizeClassifier::round_size(size);
        size_t size_class = SizeClassifier::get_size_class(alloc_size);

        // Try to find existing free block
        Block* block = find_free_block(alloc_size, size_class, stream);

        if (!block) {
            // Need new segment
            block = allocate_segment(alloc_size, device, stream);
        }

        if (block) {
            // Split if much larger than needed
            if (block->size >= alloc_size + SizeClassifier::kMinBlockSize) {
                split_block(block, alloc_size);
            }

            block->state = BlockState::ACTIVE;
            block->requested_size = size;
            block->stream = stream;
            active_blocks_[block->ptr] = block;

            total_free_bytes_ -= block->size;
        }

        return block;
    }

    // Free a block (mark pending or return to pool)
    void free(Block* block, cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (block->stream != stream) {
            // Cross-stream free - need synchronization
            mark_pending(block, stream);
        } else {
            // Same stream - can reuse immediately
            return_to_pool(block);
        }

        active_blocks_.erase(block->ptr);
    }

    // Process pending frees that have completed
    void process_pending() {
        std::lock_guard<std::mutex> lock(mutex_);

        for (auto* segment : segments_) {
            for (auto* block : segment->blocks) {
                if (block->state == BlockState::PENDING) {
                    // Check if event completed
                    cudaError_t err = cudaEventQuery(block->event);
                    if (err == cudaSuccess) {
                        return_to_pool(block);
                    }
                }
            }
        }
    }

private:
    Block* find_free_block(size_t size, size_t size_class, cudaStream_t stream) {
        // Search in same size class first, then larger classes
        for (auto it = free_pools_.lower_bound(size_class);
             it != free_pools_.end(); ++it) {

            for (auto* block : it->second) {
                if (block->size >= size) {
                    // Found suitable block
                    it->second.erase(block);
                    if (it->second.empty()) {
                        free_pools_.erase(it);
                    }
                    return block;
                }
            }
        }

        return nullptr;
    }

    Block* allocate_segment(size_t size, int device, cudaStream_t stream) {
        // Determine segment size
        bool is_small = size <= SizeClassifier::kSmallSize;
        size_t segment_size = is_small ?
            SizeClassifier::kSmallBuffer :
            std::max(size, SizeClassifier::kLargeBuffer);

        // Allocate from CUDA
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, segment_size);

        if (err != cudaSuccess) {
            // Try to free cached memory and retry
            release_cached_memory();
            err = cudaMalloc(&ptr, segment_size);

            if (err != cudaSuccess) {
                return nullptr;  // OOM
            }
        }

        // Create segment and initial block
        auto* segment = new Segment(ptr, segment_size, device, stream, is_small);
        auto* block = new Block(ptr, segment_size, segment);
        segment->blocks.push_back(block);
        segments_.push_back(segment);

        total_free_bytes_ += segment_size;
        update_largest_free();

        return block;
    }

    void split_block(Block* block, size_t size) {
        // Create new block for remainder
        void* new_ptr = static_cast<char*>(block->ptr) + size;
        size_t new_size = block->size - size;

        auto* new_block = new Block(new_ptr, new_size, block->segment);
        new_block->state = BlockState::FREE;

        // Update linked list
        new_block->prev = block;
        new_block->next = block->next;
        if (block->next) {
            block->next->prev = new_block;
        }
        block->next = new_block;
        block->size = size;

        // Add to segment
        block->segment->blocks.push_back(new_block);

        // Add remainder to free pool
        add_to_free_pool(new_block);
    }

    void return_to_pool(Block* block) {
        block->state = BlockState::FREE;
        block->stream = nullptr;

        // Try to coalesce with neighbors
        block = coalesce(block);

        // Add to appropriate free pool
        add_to_free_pool(block);

        total_free_bytes_ += block->size;
        update_largest_free();
    }

    Block* coalesce(Block* block) {
        // Merge with previous if free
        if (block->prev && block->prev->state == BlockState::FREE) {
            Block* prev = block->prev;
            remove_from_free_pool(prev);

            prev->size += block->size;
            prev->next = block->next;
            if (block->next) {
                block->next->prev = prev;
            }

            // Remove block from segment
            auto& blocks = block->segment->blocks;
            blocks.erase(std::find(blocks.begin(), blocks.end(), block));
            delete block;

            block = prev;
        }

        // Merge with next if free
        if (block->next && block->next->state == BlockState::FREE) {
            Block* next = block->next;
            remove_from_free_pool(next);

            block->size += next->size;
            block->next = next->next;
            if (next->next) {
                next->next->prev = block;
            }

            // Remove next from segment
            auto& blocks = block->segment->blocks;
            blocks.erase(std::find(blocks.begin(), blocks.end(), next));
            delete next;
        }

        return block;
    }

    void mark_pending(Block* block, cudaStream_t stream) {
        block->state = BlockState::PENDING;

        // Create event if needed
        if (!block->event) {
            cudaEventCreate(&block->event);
        }

        // Record event on the stream that freed this block
        cudaEventRecord(block->event, stream);
    }

    void add_to_free_pool(Block* block) {
        size_t size_class = SizeClassifier::get_size_class(block->size);
        block->size_class = size_class;
        free_pools_[size_class].insert(block);
    }

    void remove_from_free_pool(Block* block) {
        auto it = free_pools_.find(block->size_class);
        if (it != free_pools_.end()) {
            it->second.erase(block);
            if (it->second.empty()) {
                free_pools_.erase(it);
            }
        }
    }

    void release_cached_memory() {
        // Release empty segments back to CUDA
        for (auto it = segments_.begin(); it != segments_.end();) {
            Segment* segment = *it;

            // Check if segment has only one free block
            if (segment->blocks.size() == 1 &&
                segment->blocks[0]->state == BlockState::FREE) {

                Block* block = segment->blocks[0];
                remove_from_free_pool(block);
                total_free_bytes_ -= block->size;

                cudaFree(segment->ptr);
                delete block;
                delete segment;

                it = segments_.erase(it);
            } else {
                ++it;
            }
        }

        update_largest_free();
    }

    void update_largest_free() {
        largest_free_block_ = 0;
        for (const auto& [size_class, blocks] : free_pools_) {
            if (!blocks.empty()) {
                largest_free_block_ = std::max(
                    largest_free_block_,
                    (*blocks.rbegin())->size
                );
            }
        }
    }
};

// Comparator for ordering blocks by size in free pools
struct BlockSizeCompare {
    bool operator()(const Block* a, const Block* b) const {
        if (a->size != b->size) {
            return a->size < b->size;
        }
        return a->ptr < b->ptr;  // Tie-break by address
    }
};
```

### 3. Stream-Aware Caching Layer

```cpp
class CachingAllocator {
private:
    BlockAllocator block_allocator_;

    // Per-device state
    struct DeviceState {
        int device;
        MemoryStats stats;
        std::mutex mutex;
    };
    std::vector<DeviceState> device_states_;

    // Stream tracking for synchronization
    std::unordered_map<cudaStream_t, std::vector<Block*>> stream_allocations_;

    // Global mutex for cross-device operations
    std::mutex global_mutex_;

    // Configuration
    size_t max_split_size_;
    bool release_on_oom_;

public:
    CachingAllocator() : max_split_size_(0), release_on_oom_(true) {
        int num_devices;
        cudaGetDeviceCount(&num_devices);
        device_states_.resize(num_devices);
        for (int i = 0; i < num_devices; i++) {
            device_states_[i].device = i;
        }
    }

    void* allocate(size_t size, int device, cudaStream_t stream) {
        if (size == 0) {
            return nullptr;
        }

        // Process pending frees before allocation
        block_allocator_.process_pending();

        Block* block = block_allocator_.allocate(size, device, stream);

        if (!block) {
            // OOM handling
            if (release_on_oom_) {
                // Synchronize and release cached memory
                cudaDeviceSynchronize();
                block_allocator_.process_pending();

                // Retry allocation
                block = block_allocator_.allocate(size, device, stream);
            }

            if (!block) {
                device_states_[device].stats.num_ooms++;
                throw std::bad_alloc();
            }

            device_states_[device].stats.num_alloc_retries++;
        }

        // Update statistics
        DeviceState& state = device_states_[device];
        state.stats.allocated_bytes += block->size;
        state.stats.active_bytes += size;
        state.stats.num_allocations++;
        state.stats.peak_allocated_bytes = std::max(
            state.stats.peak_allocated_bytes,
            state.stats.allocated_bytes
        );

        // Track stream allocation
        stream_allocations_[stream].push_back(block);

        return block->ptr;
    }

    void deallocate(void* ptr, cudaStream_t stream) {
        if (!ptr) {
            return;
        }

        Block* block = block_allocator_.get_block(ptr);
        if (!block) {
            throw std::runtime_error("Invalid pointer for deallocation");
        }

        int device = block->device;
        DeviceState& state = device_states_[device];

        // Update statistics
        state.stats.allocated_bytes -= block->size;
        state.stats.active_bytes -= block->requested_size;

        // Remove from stream tracking
        auto& stream_blocks = stream_allocations_[block->stream];
        stream_blocks.erase(
            std::find(stream_blocks.begin(), stream_blocks.end(), block)
        );

        // Free the block
        block_allocator_.free(block, stream);
    }

    // Synchronize stream and release its allocations
    void synchronize_stream(cudaStream_t stream) {
        cudaStreamSynchronize(stream);
        block_allocator_.process_pending();
    }

    // Release all cached memory (for memory pressure)
    void empty_cache() {
        std::lock_guard<std::mutex> lock(global_mutex_);

        // Synchronize all streams
        for (auto& [stream, blocks] : stream_allocations_) {
            if (stream) {
                cudaStreamSynchronize(stream);
            }
        }

        block_allocator_.process_pending();
        block_allocator_.release_all_cached();

        // Update statistics
        for (auto& state : device_states_) {
            state.stats.reserved_bytes = state.stats.allocated_bytes;
        }
    }

    // Get memory statistics for device
    MemoryStats get_stats(int device) const {
        return device_states_[device].stats;
    }

    // Get allocation snapshot
    std::string get_memory_snapshot() const {
        std::stringstream ss;
        ss << "GPU Memory Allocator Snapshot\n";
        ss << "============================\n\n";

        for (const auto& state : device_states_) {
            ss << "Device " << state.device << ":\n";
            ss << "  Allocated: " << format_bytes(state.stats.allocated_bytes) << "\n";
            ss << "  Reserved:  " << format_bytes(state.stats.reserved_bytes) << "\n";
            ss << "  Active:    " << format_bytes(state.stats.active_bytes) << "\n";
            ss << "  Peak:      " << format_bytes(state.stats.peak_allocated_bytes) << "\n";
            ss << "  Allocs:    " << state.stats.num_allocations << "\n";
            ss << "  OOMs:      " << state.stats.num_ooms << "\n";
            ss << "\n";
        }

        return ss.str();
    }

private:
    static std::string format_bytes(size_t bytes) {
        if (bytes >= 1024 * 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
        } else if (bytes >= 1024 * 1024) {
            return std::to_string(bytes / (1024 * 1024)) + " MB";
        } else if (bytes >= 1024) {
            return std::to_string(bytes / 1024) + " KB";
        }
        return std::to_string(bytes) + " B";
    }
};
```

### 4. Memory Pressure Handling

```cpp
class MemoryPressureHandler {
public:
    enum class PressureLevel {
        NONE,      // < 70% utilization
        LOW,       // 70-85% utilization
        MEDIUM,    // 85-95% utilization
        HIGH,      // 95-99% utilization
        CRITICAL   // > 99% utilization
    };

private:
    CachingAllocator& allocator_;
    size_t total_memory_;
    std::vector<std::function<void(PressureLevel)>> callbacks_;

public:
    MemoryPressureHandler(CachingAllocator& allocator, int device)
        : allocator_(allocator) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        total_memory_ = prop.totalGlobalMem;
    }

    PressureLevel get_pressure_level() const {
        auto stats = allocator_.get_stats(0);
        double utilization = static_cast<double>(stats.allocated_bytes) / total_memory_;

        if (utilization >= 0.99) return PressureLevel::CRITICAL;
        if (utilization >= 0.95) return PressureLevel::HIGH;
        if (utilization >= 0.85) return PressureLevel::MEDIUM;
        if (utilization >= 0.70) return PressureLevel::LOW;
        return PressureLevel::NONE;
    }

    void register_callback(std::function<void(PressureLevel)> callback) {
        callbacks_.push_back(callback);
    }

    void check_and_notify() {
        PressureLevel level = get_pressure_level();

        if (level >= PressureLevel::MEDIUM) {
            // Notify registered callbacks
            for (auto& callback : callbacks_) {
                callback(level);
            }

            // Take automatic action
            if (level >= PressureLevel::HIGH) {
                allocator_.process_pending();
            }
            if (level >= PressureLevel::CRITICAL) {
                allocator_.empty_cache();
            }
        }
    }

    // Proactive memory management
    void ensure_available(size_t required_bytes) {
        auto stats = allocator_.get_stats(0);
        size_t available = total_memory_ - stats.allocated_bytes;

        if (available < required_bytes) {
            // Try to free memory
            allocator_.process_pending();

            stats = allocator_.get_stats(0);
            available = total_memory_ - stats.allocated_bytes;

            if (available < required_bytes) {
                // More aggressive: release cached
                allocator_.empty_cache();
            }
        }
    }
};
```

### 5. Multi-GPU Support

```cpp
class MultiGPUAllocator {
private:
    std::vector<std::unique_ptr<CachingAllocator>> allocators_;
    std::unordered_map<void*, int> ptr_to_device_;

    // Peer-to-peer access matrix
    std::vector<std::vector<bool>> p2p_access_;

public:
    MultiGPUAllocator() {
        int num_devices;
        cudaGetDeviceCount(&num_devices);

        // Create per-device allocators
        for (int i = 0; i < num_devices; i++) {
            allocators_.push_back(std::make_unique<CachingAllocator>());
        }

        // Initialize P2P access matrix
        p2p_access_.resize(num_devices, std::vector<bool>(num_devices, false));
        for (int i = 0; i < num_devices; i++) {
            p2p_access_[i][i] = true;
            for (int j = i + 1; j < num_devices; j++) {
                int can_access;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                if (can_access) {
                    cudaSetDevice(i);
                    cudaDeviceEnablePeerAccess(j, 0);
                    cudaSetDevice(j);
                    cudaDeviceEnablePeerAccess(i, 0);
                    p2p_access_[i][j] = true;
                    p2p_access_[j][i] = true;
                }
            }
        }
    }

    void* allocate(size_t size, int device, cudaStream_t stream = nullptr) {
        cudaSetDevice(device);
        void* ptr = allocators_[device]->allocate(size, device, stream);
        ptr_to_device_[ptr] = device;
        return ptr;
    }

    void deallocate(void* ptr, cudaStream_t stream = nullptr) {
        int device = ptr_to_device_[ptr];
        cudaSetDevice(device);
        allocators_[device]->deallocate(ptr, stream);
        ptr_to_device_.erase(ptr);
    }

    // Copy with P2P if available
    void copy_between_devices(void* dst, int dst_device,
                              void* src, int src_device,
                              size_t size, cudaStream_t stream = nullptr) {
        if (p2p_access_[src_device][dst_device]) {
            // Direct P2P copy
            cudaMemcpyPeerAsync(dst, dst_device, src, src_device, size, stream);
        } else {
            // Stage through CPU
            void* host_buffer;
            cudaMallocHost(&host_buffer, size);

            cudaMemcpyAsync(host_buffer, src, size, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            cudaMemcpyAsync(dst, host_buffer, size, cudaMemcpyHostToDevice, stream);

            cudaFreeHost(host_buffer);
        }
    }

    // Get combined stats
    MemoryStats get_total_stats() const {
        MemoryStats total = {};
        for (int i = 0; i < allocators_.size(); i++) {
            auto stats = allocators_[i]->get_stats(i);
            total.allocated_bytes += stats.allocated_bytes;
            total.reserved_bytes += stats.reserved_bytes;
            total.active_bytes += stats.active_bytes;
            total.num_allocations += stats.num_allocations;
            total.num_ooms += stats.num_ooms;
        }
        return total;
    }
};
```

### 6. Unified Memory Support

```cpp
class UnifiedMemoryAllocator {
private:
    // Track unified memory allocations
    struct UnifiedAllocation {
        void* ptr;
        size_t size;
        bool prefetched_to_device;
        int preferred_device;
    };
    std::unordered_map<void*, UnifiedAllocation> allocations_;

    std::mutex mutex_;

public:
    void* allocate(size_t size, int preferred_device = -1) {
        std::lock_guard<std::mutex> lock(mutex_);

        void* ptr;
        cudaError_t err = cudaMallocManaged(&ptr, size);

        if (err != cudaSuccess) {
            throw std::bad_alloc();
        }

        // Set memory advice
        if (preferred_device >= 0) {
            cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, preferred_device);
            cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, preferred_device);
        }

        allocations_[ptr] = {ptr, size, false, preferred_device};

        return ptr;
    }

    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (allocations_.find(ptr) != allocations_.end()) {
            cudaFree(ptr);
            allocations_.erase(ptr);
        }
    }

    // Prefetch to device
    void prefetch_to_device(void* ptr, int device, cudaStream_t stream = nullptr) {
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            cudaMemPrefetchAsync(ptr, it->second.size, device, stream);
            it->second.prefetched_to_device = true;
        }
    }

    // Prefetch to CPU
    void prefetch_to_cpu(void* ptr, cudaStream_t stream = nullptr) {
        auto it = allocations_.find(ptr);
        if (it != allocations_.end()) {
            cudaMemPrefetchAsync(ptr, it->second.size, cudaCpuDeviceId, stream);
            it->second.prefetched_to_device = false;
        }
    }
};
```

## Enterprise Features

### Memory Profiler

```cpp
class MemoryProfiler {
public:
    struct AllocationRecord {
        void* ptr;
        size_t size;
        std::chrono::time_point<std::chrono::steady_clock> timestamp;
        std::string callstack;
        cudaStream_t stream;
    };

private:
    std::vector<AllocationRecord> records_;
    bool enabled_;
    std::mutex mutex_;

public:
    MemoryProfiler() : enabled_(false) {}

    void enable() { enabled_ = true; }
    void disable() { enabled_ = false; }

    void record_alloc(void* ptr, size_t size, cudaStream_t stream) {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        records_.push_back({
            ptr,
            size,
            std::chrono::steady_clock::now(),
            capture_callstack(),
            stream
        });
    }

    void record_free(void* ptr) {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        // Find and mark allocation
        for (auto& record : records_) {
            if (record.ptr == ptr && record.callstack != "freed") {
                record.callstack = "freed";
                break;
            }
        }
    }

    // Generate memory timeline
    std::string generate_timeline() const {
        std::stringstream ss;
        ss << "Memory Timeline\n";
        ss << "===============\n\n";

        for (const auto& record : records_) {
            auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                record.timestamp.time_since_epoch()
            ).count();

            ss << "[" << time_ms << "ms] "
               << (record.callstack == "freed" ? "FREE " : "ALLOC")
               << " " << record.ptr
               << " (" << record.size << " bytes)\n";
        }

        return ss.str();
    }

    // Find memory leaks
    std::vector<AllocationRecord> find_leaks() const {
        std::vector<AllocationRecord> leaks;
        for (const auto& record : records_) {
            if (record.callstack != "freed") {
                leaks.push_back(record);
            }
        }
        return leaks;
    }

private:
    std::string capture_callstack() {
        // Platform-specific stack capture
        return "stack trace";
    }
};
```

### Fragmentation Analysis

```cpp
class FragmentationAnalyzer {
public:
    struct FragmentationReport {
        double fragmentation_ratio;  // 0.0 = none, 1.0 = severe
        size_t largest_free_block;
        size_t total_free_bytes;
        size_t num_free_blocks;
        std::vector<std::pair<size_t, int>> free_block_histogram;
    };

    static FragmentationReport analyze(const BlockAllocator& allocator) {
        FragmentationReport report;

        // Collect free block sizes
        std::vector<size_t> free_sizes;
        size_t total_free = 0;
        size_t largest = 0;

        for (auto* segment : allocator.get_segments()) {
            for (auto* block : segment->blocks) {
                if (block->state == BlockState::FREE) {
                    free_sizes.push_back(block->size);
                    total_free += block->size;
                    largest = std::max(largest, block->size);
                }
            }
        }

        report.total_free_bytes = total_free;
        report.largest_free_block = largest;
        report.num_free_blocks = free_sizes.size();

        // Calculate fragmentation ratio
        if (total_free > 0) {
            report.fragmentation_ratio = 1.0 -
                static_cast<double>(largest) / total_free;
        } else {
            report.fragmentation_ratio = 0.0;
        }

        // Build histogram
        std::map<size_t, int> histogram;
        for (size_t size : free_sizes) {
            size_t bucket = round_to_bucket(size);
            histogram[bucket]++;
        }

        for (auto& [bucket, count] : histogram) {
            report.free_block_histogram.push_back({bucket, count});
        }

        return report;
    }

private:
    static size_t round_to_bucket(size_t size) {
        // Round to power of 2 for histogram
        size_t bucket = 1;
        while (bucket < size) {
            bucket *= 2;
        }
        return bucket;
    }
};
```

## Development Phases

### Phase 1: Core Allocation (Weeks 1-3)
- Block and segment data structures
- Basic best-fit allocation
- Block splitting and coalescing
- Free pool management

### Phase 2: Caching Layer (Weeks 4-5)
- Size class bucketing
- Memory reuse without cudaFree
- Statistics tracking
- Basic OOM handling

### Phase 3: Stream Awareness (Weeks 6-7)
- Stream-based allocation tracking
- Event-based synchronization
- Pending free management
- Cross-stream safety

### Phase 4: Memory Pressure (Weeks 8-9)
- Pressure level detection
- Callback system
- Automatic cache release
- Memory snapshots

### Phase 5: Enterprise Features (Weeks 10-11)
- Memory profiler
- Fragmentation analysis
- IPC statistics
- Per-type caching

### Phase 6: Stretch Goals (Week 12+)
- Multi-GPU with P2P
- Unified memory support
- CPU offloading
- Custom allocation strategies

## Testing Strategy

### Unit Tests
- Block split/coalesce correctness
- Size class computation
- Free pool operations
- Statistics accuracy

### Integration Tests
- Allocation/deallocation cycles
- Multi-stream scenarios
- OOM recovery
- Cache behavior

### Stress Tests
- High allocation rate
- Fragmentation resistance
- Memory pressure handling
- Multi-threaded access

### Performance Tests
- Allocation latency
- Throughput benchmarks
- Fragmentation over time
- Memory overhead

## Performance Targets

| Metric | Target |
|--------|--------|
| Allocation latency (cached) | < 1 microsecond |
| Cache hit rate | > 95% |
| Memory overhead | < 5% |
| Fragmentation ratio | < 20% |
| Multi-GPU copy efficiency | > 80% of peak bandwidth |

## Dependencies

- **CUDA Toolkit**: 11.0+
- **C++17**: For std::optional, structured bindings
- **pthread or std::mutex**: Thread safety

## References

- PyTorch CUDACachingAllocator source code
- CUDA Best Practices Guide: Memory Management
- "Efficient Memory Management for Large Language Model Serving with PagedAttention"
- jemalloc: A General Purpose Memory Allocator
