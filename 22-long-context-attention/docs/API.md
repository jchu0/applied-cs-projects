# Long-Context Attention API Documentation

## Table of Contents

- [Core API](#core-api)
- [Flash Attention](#flash-attention)
- [Block Sparse Attention](#block-sparse-attention)
- [Sliding Window Attention](#sliding-window-attention)
- [KV Cache](#kv-cache)
- [RoPE Embeddings](#rope-embeddings)
- [Auto-Selector](#auto-selector)
- [Configuration Types](#configuration-types)
- [Examples](#examples)

## Core API

### `Attention`

The main attention interface that provides a unified API for all attention mechanisms.

```rust
pub struct Attention {
    config: AttentionConfig,
    implementation: Box<dyn AttentionMechanism>,
}

impl Attention {
    /// Create a new attention instance with automatic selection
    pub fn new(config: AttentionConfig) -> Self;

    /// Create with specific Flash Attention implementation
    pub fn new_flash(config: AttentionConfig, flash_config: FlashAttentionConfig) -> Self;

    /// Create with Block Sparse implementation
    pub fn new_sparse(config: AttentionConfig, sparse_config: SparsityConfig) -> Self;

    /// Create with Sliding Window implementation
    pub fn new_sliding(config: AttentionConfig, window_config: SlidingWindowConfig) -> Self;

    /// Forward pass
    pub fn forward(
        &mut self,
        q: &ArrayView4<f32>,  // [batch, heads, seq_len, head_dim]
        k: &ArrayView4<f32>,  // [batch, heads, seq_len, head_dim]
        v: &ArrayView4<f32>,  // [batch, heads, seq_len, head_dim]
    ) -> Array4<f32>;

    /// Forward pass with attention mask
    pub fn forward_masked(
        &mut self,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        v: &ArrayView4<f32>,
        mask: &ArrayView3<f32>,  // [batch, seq_len, seq_len]
    ) -> Array4<f32>;

    /// Backward pass
    pub fn backward(
        &mut self,
        d_out: &ArrayView4<f32>,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        v: &ArrayView4<f32>,
    ) -> (Array4<f32>, Array4<f32>, Array4<f32>);
}
```

#### Example

```rust
use long_context_attention::{Attention, AttentionConfig, AttentionType};

let config = AttentionConfig {
    attention_type: AttentionType::Auto,
    num_heads: 8,
    head_dim: 64,
    max_seq_len: 2048,
    dropout: 0.1,
    use_bias: false,
};

let mut attention = Attention::new(config);

// Forward pass
let output = attention.forward(&q, &k, &v);
```

## Flash Attention

### `FlashAttention`

Implementation of the Flash Attention algorithm for memory-efficient exact attention.

```rust
pub struct FlashAttention {
    config: FlashAttentionConfig,
    max_seq_len: usize,
    head_dim: usize,
}

impl FlashAttention {
    /// Create a new Flash Attention instance
    pub fn new(config: FlashAttentionConfig, max_seq_len: usize, head_dim: usize) -> Self;

    /// Forward pass with optional mask
    pub fn forward(
        &mut self,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        v: &ArrayView4<f32>,
        mask: Option<&ArrayView3<f32>>,
    ) -> Array4<f32>;

    /// Backward pass
    pub fn backward(
        &mut self,
        d_out: &ArrayView4<f32>,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        v: &ArrayView4<f32>,
    ) -> (Array4<f32>, Array4<f32>, Array4<f32>);

    /// Estimate memory usage in bytes
    pub fn estimate_memory_usage(&self, batch_size: usize, seq_len: usize, num_heads: usize) -> usize;

    /// Set custom block size
    pub fn set_block_size(&mut self, block_size: BlockSize);

    /// Enable/disable Triton kernels
    pub fn set_use_triton(&mut self, use_triton: bool);
}
```

### `FlashAttentionConfig`

```rust
pub struct FlashAttentionConfig {
    /// Block size for tiling
    pub block_size: BlockSize,

    /// Use Triton kernels if available
    pub use_triton: bool,

    /// Apply causal mask
    pub causal: bool,

    /// Backward pass strategy
    pub backward: Backward,
}

pub enum BlockSize {
    B32,   // 32x32 blocks
    B64,   // 64x64 blocks
    B128,  // 128x128 blocks
}

pub enum Backward {
    Store,      // Store intermediate values
    Recompute,  // Recompute in backward pass
}
```

#### Example

```rust
use long_context_attention::flash::{FlashAttention, FlashAttentionConfig, BlockSize, Backward};

let config = FlashAttentionConfig {
    block_size: BlockSize::B64,
    use_triton: true,
    causal: true,
    backward: Backward::Recompute,
};

let mut flash_attn = FlashAttention::new(config, 2048, 64);
let output = flash_attn.forward(&q, &k, &v, None);
```

## Block Sparse Attention

### `BlockSparseAttention`

Attention with structured sparsity patterns for efficient long-context processing.

```rust
pub struct BlockSparseAttention {
    config: SparsityConfig,
    max_seq_len: usize,
    head_dim: usize,
}

impl BlockSparseAttention {
    /// Create a new Block Sparse Attention instance
    pub fn new(config: SparsityConfig, max_seq_len: usize, head_dim: usize) -> Self;

    /// Forward pass
    pub fn forward(
        &mut self,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        v: &ArrayView4<f32>,
        mask: Option<&ArrayView3<f32>>,
    ) -> Array4<f32>;

    /// Forward with adaptive sparsity
    pub fn forward_adaptive(
        &mut self,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        v: &ArrayView4<f32>,
    ) -> Array4<f32>;

    /// Generate sparsity pattern
    pub fn generate_pattern(&self, seq_len: usize) -> Array2<bool>;

    /// Get current sparsity ratio
    pub fn sparsity_ratio(&self) -> f32;

    /// Update sparsity pattern
    pub fn update_pattern(&mut self, pattern: BlockPattern);
}
```

### `SparsityConfig`

```rust
pub struct SparsityConfig {
    /// Block size for sparsity
    pub block_size: usize,

    /// Target sparsity ratio (0.0 = dense, 1.0 = fully sparse)
    pub sparsity_ratio: f32,

    /// Sparsity pattern type
    pub pattern: BlockPattern,

    /// Local attention window size
    pub local_window: usize,

    /// Global token positions
    pub global_tokens: Vec<usize>,
}

pub enum BlockPattern {
    Fixed,                    // Predefined pattern
    Random,                   // Random sampling
    LocalWindow,             // Local attention windows
    GlobalLocal,             // Global + local attention
    Adaptive,                // Content-based dynamic
    Strided { stride: usize }, // Strided pattern
}
```

#### Example

```rust
use long_context_attention::block_sparse::{BlockSparseAttention, SparsityConfig, BlockPattern};

let config = SparsityConfig {
    block_size: 64,
    sparsity_ratio: 0.9,
    pattern: BlockPattern::GlobalLocal,
    local_window: 256,
    global_tokens: vec![0, 512, 1024],
};

let mut sparse_attn = BlockSparseAttention::new(config, 2048, 64);
let pattern = sparse_attn.generate_pattern(2048);
let output = sparse_attn.forward(&q, &k, &v, None);
```

## Sliding Window Attention

### `SlidingWindowAttention`

Attention restricted to local windows for linear complexity.

```rust
pub struct SlidingWindowAttention {
    config: SlidingWindowConfig,
    max_seq_len: usize,
    head_dim: usize,
}

impl SlidingWindowAttention {
    /// Create a new Sliding Window Attention instance
    pub fn new(config: SlidingWindowConfig, max_seq_len: usize, head_dim: usize) -> Self;

    /// Forward pass
    pub fn forward(
        &mut self,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        v: &ArrayView4<f32>,
    ) -> Array4<f32>;

    /// Compute window boundaries
    pub fn compute_windows(&self, seq_len: usize) -> Vec<(usize, usize)>;

    /// Get window size
    pub fn window_size(&self) -> usize;

    /// Update window configuration
    pub fn update_config(&mut self, config: SlidingWindowConfig);
}
```

### `SlidingWindowConfig`

```rust
pub struct SlidingWindowConfig {
    /// Window size in tokens
    pub window_size: usize,

    /// Overlap between windows
    pub overlap: usize,

    /// Apply causal masking
    pub use_causal_mask: bool,
}
```

#### Example

```rust
use long_context_attention::sliding_window::{SlidingWindowAttention, SlidingWindowConfig};

let config = SlidingWindowConfig {
    window_size: 256,
    overlap: 64,
    use_causal_mask: true,
};

let mut window_attn = SlidingWindowAttention::new(config, 2048, 64);
let output = window_attn.forward(&q, &k, &v);
```

## KV Cache

### `KVCache`

Key-Value cache management for autoregressive generation.

```rust
pub struct KVCache {
    config: CacheConfig,
}

impl KVCache {
    /// Create a new KV cache
    pub fn new(config: CacheConfig) -> Self;

    /// Append new K, V tensors
    pub fn append(
        &mut self,
        k: &ArrayView4<f32>,
        v: &ArrayView4<f32>,
        position: usize,
    );

    /// Retrieve cached K, V tensors
    pub fn get(&self, start: usize, end: usize) -> (Array4<f32>, Array4<f32>);

    /// Get last n tokens
    pub fn get_last(&self, n: usize) -> (Array4<f32>, Array4<f32>);

    /// Clear cache
    pub fn clear(&mut self);

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool;

    /// Get current cache size
    pub fn current_size(&self) -> usize;

    /// Get maximum sequence length
    pub fn max_seq_len(&self) -> usize;
}
```

### `CacheConfig`

```rust
pub struct CacheConfig {
    /// Maximum sequence length to cache
    pub max_seq_len: usize,

    /// Maximum batch size
    pub max_batch_size: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Cache eviction policy
    pub eviction_policy: EvictionPolicy,

    /// Compression method
    pub compression: CompressionMethod,
}

pub enum EvictionPolicy {
    FIFO,           // First In First Out
    LRU,            // Least Recently Used
    SlidingWindow,  // Keep last N tokens
}

pub enum CompressionMethod {
    None,           // No compression
    Quantize8Bit,   // 8-bit quantization
    Quantize4Bit,   // 4-bit quantization
    Pruning,        // Magnitude-based pruning
}
```

#### Example

```rust
use long_context_attention::kv_cache::{KVCache, CacheConfig, EvictionPolicy, CompressionMethod};

let config = CacheConfig {
    max_seq_len: 2048,
    max_batch_size: 4,
    num_heads: 8,
    head_dim: 64,
    eviction_policy: EvictionPolicy::LRU,
    compression: CompressionMethod::Quantize8Bit,
};

let mut cache = KVCache::new(config);

// During generation
cache.append(&k_new, &v_new, current_position);
let (k_full, v_full) = cache.get(0, current_position + 1);
```

## RoPE Embeddings

### `RoPE`

Rotary Position Embeddings for encoding position information.

```rust
pub struct RoPE {
    config: RoPEConfig,
}

impl RoPE {
    /// Create a new RoPE instance
    pub fn new(config: RoPEConfig) -> Self;

    /// Apply RoPE to Q and K tensors
    pub fn apply(
        &mut self,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        offset: usize,
    ) -> (Array4<f32>, Array4<f32>);

    /// Apply to specific position
    pub fn apply_to_position(
        &mut self,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        position: usize,
    ) -> (Array4<f32>, Array4<f32>);

    /// Apply inverse RoPE
    pub fn apply_inverse(
        &mut self,
        q: &ArrayView4<f32>,
        k: &ArrayView4<f32>,
        offset: usize,
    ) -> (Array4<f32>, Array4<f32>);

    /// Precompute frequencies for caching
    pub fn precompute_freqs(&mut self);
}
```

### `RoPEConfig`

```rust
pub struct RoPEConfig {
    /// Dimension of embeddings
    pub dim: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Base for frequency computation
    pub base: f32,

    /// Frequency scaling method
    pub freq_scale: FrequencyScale,
}

pub enum FrequencyScale {
    Linear,                     // Standard RoPE
    NTKScaling { factor: f32 }, // Neural Tangent Kernel scaling
    Yarn { factor: f32, beta: f32 }, // YaRN scaling
}
```

#### Example

```rust
use long_context_attention::rope::{RoPE, RoPEConfig, FrequencyScale};

let config = RoPEConfig {
    dim: 64,
    max_seq_len: 4096,
    base: 10000.0,
    freq_scale: FrequencyScale::NTKScaling { factor: 2.0 },
};

let mut rope = RoPE::new(config);
rope.precompute_freqs();

let (q_rotated, k_rotated) = rope.apply(&q, &k, position_offset);
```

## Auto-Selector

### `AutoSelector`

Automatically selects the optimal attention mechanism based on runtime conditions.

```rust
pub struct AutoSelector {
    model_config: ModelConfig,
}

impl AutoSelector {
    /// Create a new auto-selector
    pub fn new(model_config: ModelConfig) -> Self;

    /// Select attention mechanism
    pub fn select_attention(&mut self, criteria: SelectionCriteria) -> AttentionSelection;

    /// Update performance profile
    pub fn update_profile(
        &mut self,
        attention_type: AttentionType,
        seq_len: usize,
        batch_size: usize,
        latency_ms: f64,
        memory_mb: f64,
    );

    /// Estimate memory usage
    pub fn estimate_memory(
        &self,
        seq_len: usize,
        batch_size: usize,
        attention_type: &AttentionType,
    ) -> usize;

    /// Enable adaptive mode
    pub fn set_adaptive_mode(&mut self, enabled: bool);
}
```

### `SelectionCriteria`

```rust
pub struct SelectionCriteria {
    /// Sequence length
    pub sequence_length: usize,

    /// Batch size
    pub batch_size: usize,

    /// Available memory in bytes
    pub available_memory: usize,

    /// Optimization target
    pub optimize_for: &'static str,  // "throughput", "memory", "quality", "auto"
}
```

### `AttentionSelection`

```rust
pub struct AttentionSelection {
    /// Selected attention type
    pub attention_type: AttentionType,

    /// Configuration parameters
    pub config: HashMap<String, Value>,

    /// Estimated performance metrics
    pub estimated_latency_ms: f64,
    pub estimated_memory_mb: f64,
}
```

#### Example

```rust
use long_context_attention::auto_select::{AutoSelector, SelectionCriteria, ModelConfig};

let model_config = ModelConfig {
    vocab_size: 50000,
    hidden_dim: 768,
    num_heads: 12,
    head_dim: 64,
    num_layers: 12,
    max_seq_len: 4096,
};

let mut selector = AutoSelector::new(model_config);

let criteria = SelectionCriteria {
    sequence_length: 2048,
    batch_size: 4,
    available_memory: 8 * 1024 * 1024 * 1024, // 8GB
    optimize_for: "throughput",
};

let selection = selector.select_attention(criteria);
println!("Selected: {:?}", selection.attention_type);
```

## Configuration Types

### `AttentionConfig`

```rust
pub struct AttentionConfig {
    /// Attention mechanism type
    pub attention_type: AttentionType,

    /// Number of attention heads
    pub num_heads: usize,

    /// Dimension per head
    pub head_dim: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,

    /// Dropout probability
    pub dropout: f32,

    /// Use bias in projections
    pub use_bias: bool,
}
```

### `AttentionType`

```rust
pub enum AttentionType {
    Standard,      // Standard scaled dot-product attention
    Flash,         // Flash Attention
    BlockSparse,   // Block sparse attention
    SlidingWindow, // Sliding window attention
    Auto,          // Automatic selection
}
```

### `ModelConfig`

```rust
pub struct ModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,

    /// Hidden dimension
    pub hidden_dim: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Head dimension
    pub head_dim: usize,

    /// Number of transformer layers
    pub num_layers: usize,

    /// Maximum sequence length
    pub max_seq_len: usize,
}
```

## Examples

### Basic Usage

```rust
use long_context_attention::{Attention, AttentionConfig, AttentionType};
use ndarray::Array4;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn main() {
    // Configuration
    let config = AttentionConfig {
        attention_type: AttentionType::Auto,
        num_heads: 8,
        head_dim: 64,
        max_seq_len: 2048,
        dropout: 0.0,
        use_bias: false,
    };

    // Create attention layer
    let mut attention = Attention::new(config);

    // Generate random inputs
    let batch_size = 2;
    let seq_len = 1024;
    let q = Array4::random((batch_size, 8, seq_len, 64), Uniform::new(-0.1, 0.1));
    let k = Array4::random((batch_size, 8, seq_len, 64), Uniform::new(-0.1, 0.1));
    let v = Array4::random((batch_size, 8, seq_len, 64), Uniform::new(-0.1, 0.1));

    // Forward pass
    let output = attention.forward(&q.view(), &k.view(), &v.view());

    println!("Output shape: {:?}", output.shape());
}
```

### Autoregressive Generation with KV Cache

```rust
use long_context_attention::{
    Attention, AttentionConfig, AttentionType,
    kv_cache::{KVCache, CacheConfig, EvictionPolicy, CompressionMethod},
};

fn generate_tokens(prompt_tokens: Vec<i32>, max_new_tokens: usize) {
    // Setup attention
    let attention_config = AttentionConfig {
        attention_type: AttentionType::Flash,
        num_heads: 8,
        head_dim: 64,
        max_seq_len: 2048,
        dropout: 0.0,
        use_bias: false,
    };

    let mut attention = Attention::new(attention_config);

    // Setup KV cache
    let cache_config = CacheConfig {
        max_seq_len: 2048,
        max_batch_size: 1,
        num_heads: 8,
        head_dim: 64,
        eviction_policy: EvictionPolicy::SlidingWindow,
        compression: CompressionMethod::None,
    };

    let mut kv_cache = KVCache::new(cache_config);

    // Process prompt
    let prompt_len = prompt_tokens.len();
    // ... compute prompt K, V ...
    // kv_cache.append(&k_prompt, &v_prompt, 0);

    // Generate new tokens
    for i in 0..max_new_tokens {
        let position = prompt_len + i;

        // Get new token embeddings (would come from model)
        // let q_new = ...
        // let k_new = ...
        // let v_new = ...

        // Update cache
        // kv_cache.append(&k_new, &v_new, position);

        // Get full K, V from cache
        // let (k_full, v_full) = kv_cache.get(0, position + 1);

        // Compute attention
        // let output = attention.forward(&q_new.view(), &k_full.view(), &v_full.view());

        // ... rest of generation logic ...
    }
}
```

### Long Document Processing

```rust
use long_context_attention::{
    block_sparse::{BlockSparseAttention, SparsityConfig, BlockPattern},
};

fn process_long_document(document_tokens: Vec<i32>) {
    let seq_len = document_tokens.len();

    // Use block sparse for very long documents
    let config = SparsityConfig {
        block_size: 128,
        sparsity_ratio: 0.95, // 95% sparse
        pattern: BlockPattern::GlobalLocal,
        local_window: 512,
        global_tokens: (0..seq_len).step_by(1024).collect(), // Every 1024th token is global
    };

    let mut attention = BlockSparseAttention::new(config, seq_len, 64);

    // Process document in chunks if needed
    // ...
}
```

### Custom Attention Pattern

```rust
use long_context_attention::{
    block_sparse::{BlockSparseAttention, SparsityConfig, BlockPattern},
};

fn create_custom_pattern() {
    let config = SparsityConfig {
        block_size: 64,
        sparsity_ratio: 0.8,
        pattern: BlockPattern::Adaptive, // Will adapt based on content
        local_window: 256,
        global_tokens: vec![0], // First token always global
    };

    let mut attention = BlockSparseAttention::new(config, 2048, 64);

    // Generate and inspect pattern
    let pattern = attention.generate_pattern(2048);

    // Pattern is a boolean matrix indicating which blocks are active
    println!("Active blocks: {}", pattern.iter().filter(|&&x| x).count());
}
```