# Long-Context Attention API Documentation

This crate (`long-context-attention`) is a CPU-only, pure-Rust library. All
attention kernels operate on flat `&[f32]` buffers described by a
[`TensorShape`], and every fallible entry point returns [`crate::Result`].
There is no GPU backend, binary, or server component.

The convention throughout is a **row-major** logical layout of
`[batch, seq_len, num_heads, head_dim]`, and all `forward` methods take
borrowed input buffers plus explicit query/key/value shapes.

## Table of Contents

- [Common Types](#common-types)
- [Configuration](#configuration)
- [Standard Attention](#standard-attention)
- [Flash Attention](#flash-attention)
- [Linear Attention](#linear-attention)
- [Sliding Window Attention](#sliding-window-attention)
- [Block Sparse Attention](#block-sparse-attention)
- [RoPE and Positional Bias](#rope-and-positional-bias)
- [KV Cache](#kv-cache)
- [Auto-Selection](#auto-selection)
- [Streaming Inference](#streaming-inference)
- [Examples](#examples)

## Common Types

### `TensorShape`

Describes the logical shape of a flat attention buffer.

```rust
pub struct TensorShape {
    pub batch: usize,
    pub seq_len: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl TensorShape {
    pub fn new(batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> Self;
    pub fn num_elements(&self) -> usize;
}
```

### `AttentionOutput`

Result of a `forward` call. `output` is a flat buffer laid out as
`[batch, seq_len, num_heads, head_dim]`.

```rust
pub struct AttentionOutput {
    pub output: Vec<f32>,
    pub shape: TensorShape,
    pub attention_weights: Option<Vec<f32>>,
}
```

### `Error` / `Result`

```rust
pub type Result<T> = std::result::Result<T, Error>;

pub enum Error {
    InvalidConfig(String),
    DimensionMismatch { expected: usize, actual: usize },
    OutOfMemory { required: usize, available: usize },
    CacheError(String),
    SequenceTooLong { length: usize, max: usize },
    InvalidMask,
    ShapeMismatch { name: &'static str, expected: usize, actual: usize },
}
```

### Free functions

```rust
/// Validate q/k/v (and optional mask) buffer lengths against their shapes.
pub fn validate_forward_inputs(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    q_shape: TensorShape,
    kv_shape: TensorShape,
    attention_mask: Option<&[f32]>,
) -> Result<()>;

/// In-place row-wise softmax over a [num_rows, num_cols] score buffer.
pub fn softmax(scores: &mut [f32], num_rows: usize, num_cols: usize);

/// Apply a causal mask in place to a [q_len, kv_len] score buffer.
pub fn apply_causal_mask(scores: &mut [f32], q_len: usize, kv_len: usize);

/// Expand KV heads to Q heads for grouped-query / multi-query attention.
pub fn expand_kv_heads(/* see source */);
```

## Configuration

### `AttentionConfig`

Shared configuration for every attention implementation. Uses a builder-style
API. See defaults in the `Default` impl.

```rust
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub num_kv_heads: Option<usize>,   // None => same as num_heads
    pub max_seq_len: usize,

    pub attention_type: AttentionType,
    pub window_size: usize,

    pub block_size: usize,
    pub num_global_tokens: usize,
    pub num_random_blocks: usize,

    pub use_alibi: bool,
    pub use_rope: bool,
    pub causal: bool,
    pub dropout: f32,

    pub dtype: DataType,
}

impl AttentionConfig {
    pub fn new(num_heads: usize, head_dim: usize) -> Self;

    pub fn with_attention_type(self, attention_type: AttentionType) -> Self;
    pub fn with_window_size(self, window_size: usize) -> Self;
    pub fn with_causal(self, causal: bool) -> Self;
    pub fn with_max_seq_len(self, max_seq_len: usize) -> Self;
    pub fn with_num_kv_heads(self, num_kv_heads: usize) -> Self;
    pub fn with_rope(self, use_rope: bool) -> Self;
    pub fn with_alibi(self, use_alibi: bool) -> Self;

    pub fn effective_num_kv_heads(&self) -> usize;
    pub fn scale(&self) -> f32;            // head_dim^-0.5
    pub fn validate(&self) -> Result<()>;
}
```

### `AttentionType`

```rust
pub enum AttentionType {
    Standard,      // Standard O(n^2) attention
    Flash,         // Memory-efficient FlashAttention
    SlidingWindow, // Sliding window attention
    BlockSparse,   // Block sparse (Longformer / BigBird style)
    Linear,        // Linear attention
    Auto,          // Auto-select based on sequence length (default)
}
```

### `DataType`

```rust
pub enum DataType { Float16, Float32, BFloat16 } // Default: Float32
```

## Standard Attention

`StandardAttention` computes dense scaled dot-product attention.

```rust
pub struct StandardAttention { /* private */ }

impl StandardAttention {
    pub fn new(config: AttentionConfig) -> Self;

    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
        attention_mask: Option<&[f32]>,
    ) -> Result<AttentionOutput>;
}
```

## Flash Attention

`FlashAttention` computes exact attention with blocked (tiled) accumulation
for reduced peak memory.

```rust
pub struct FlashAttention { /* private */ }

impl FlashAttention {
    /// Default block sizes (64 x 64).
    pub fn new(config: AttentionConfig) -> Self;

    /// Custom tiling block sizes (block_m over queries, block_n over keys).
    pub fn with_block_sizes(config: AttentionConfig, block_m: usize, block_n: usize) -> Self;

    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
        attention_mask: Option<&[f32]>,
    ) -> Result<AttentionOutput>;

    /// Estimate working-set memory in bytes.
    pub fn estimate_memory(&self, batch: usize, seq_len: usize) -> usize;
}
```

## Linear Attention

`LinearAttention` provides linear-complexity attention via a feature map.

```rust
pub struct LinearAttention { /* private */ }

impl LinearAttention {
    pub fn new(config: AttentionConfig) -> Self;

    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
        attention_mask: Option<&[f32]>,
    ) -> Result<AttentionOutput>;
}
```

## Sliding Window Attention

`SlidingWindowAttention` restricts each query to a local window
(`config.window_size`). `DilatedSlidingWindowAttention` adds a dilation stride.

```rust
pub struct SlidingWindowAttention { /* private */ }

impl SlidingWindowAttention {
    pub fn new(config: AttentionConfig) -> Self;

    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
        attention_mask: Option<&[f32]>,
    ) -> Result<AttentionOutput>;

    pub fn window_size(&self) -> usize;
    pub fn estimate_memory(&self, batch: usize, seq_len: usize) -> usize;
}

pub struct DilatedSlidingWindowAttention { /* private */ }

impl DilatedSlidingWindowAttention {
    pub fn new(config: AttentionConfig, dilation: usize) -> Self;
    pub fn forward(/* same signature as above */) -> Result<AttentionOutput>;
}
```

## Block Sparse Attention

`BlockSparseAttention` computes attention over a structured block pattern.
`SparsePattern` is the block-level mask; `BigBirdPattern` builds
global/window/random block index sets.

Note: `forward` uses `q_shape` for the key/value layout as well, so all three
buffers must match `q_shape`'s element count.

```rust
pub struct BlockSparseAttention { /* private */ }

impl BlockSparseAttention {
    pub fn new(config: AttentionConfig) -> Self;

    /// Build the block-level sparsity pattern for a given sequence length.
    pub fn build_pattern(&self, seq_len: usize) -> SparsePattern;

    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,          // ignored; q_shape is used for KV
        attention_mask: Option<&[f32]>, // ignored
    ) -> Result<AttentionOutput>;

    pub fn estimate_memory(&self, batch: usize, seq_len: usize) -> usize;
}

pub struct SparsePattern { /* private */ }

impl SparsePattern {
    pub fn new(num_blocks: usize, block_size: usize) -> Self;
    pub fn set(&mut self, i: usize, j: usize, value: bool);
    pub fn get(&self, i: usize, j: usize) -> bool;
    pub fn count_active(&self) -> usize;
    pub fn sparsity(&self) -> f32;
}

pub struct BigBirdPattern { /* private */ }

impl BigBirdPattern {
    pub fn new(num_global: usize, window_size: usize, num_random: usize) -> Self;
    pub fn build(&self, seq_len: usize) -> Vec<Vec<usize>>;
}
```

## RoPE and Positional Bias

### `RotaryEmbedding`

Rotary Position Embeddings applied in place to a flat buffer.

```rust
pub struct RotaryEmbedding { /* private */ }

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self;       // base = 10000.0
    pub fn with_base(head_dim: usize, max_seq_len: usize, base: f32) -> Self;

    /// Apply RoPE in place. `position_ids` defaults to 0..seq_len when None.
    pub fn apply(&self, x: &mut [f32], shape: TensorShape, position_ids: Option<&[usize]>);

    pub fn head_dim(&self) -> usize;
    pub fn max_seq_len(&self) -> usize;
}
```

### `ALiBiPositionalBias`

```rust
pub struct ALiBiPositionalBias { /* private */ }

impl ALiBiPositionalBias {
    pub fn new(num_heads: usize, max_seq_len: usize) -> Self;

    /// Additive bias buffer for a [q_len, kv_len] score matrix per head.
    pub fn compute_bias(&self, q_len: usize, kv_len: usize) -> Vec<f32>;
    pub fn slopes(&self) -> &[f32];
}
```

### Free function

```rust
/// Classic (non-rotary) sinusoidal position embedding vector.
pub fn sinusoidal_position_embedding(position: usize, dim: usize) -> Vec<f32>;
```

## KV Cache

Three cache backends are provided. All buffers are laid out for
`[batch, seq_len, num_heads, head_dim]`.

### `ContinuousKVCache`

Contiguous per-layer cache that appends new tokens.

```rust
pub struct ContinuousKVCache { /* private */ }

impl ContinuousKVCache {
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self;

    /// Append KV for `layer`; returns the full cached (key, value) slices.
    pub fn update(
        &mut self,
        layer: usize,
        key: &[f32],
        value: &[f32],
        seq_len: usize,
    ) -> Result<(&[f32], &[f32])>;

    pub fn get(&self, layer: usize) -> Option<(&[f32], &[f32], usize)>;
    pub fn reset(&mut self);
    pub fn current_len(&self) -> usize;
}
```

### `PagedKVCache`

Block-paged cache (vLLM-style) with per-sequence block tables.

```rust
pub struct PagedKVCache { /* private */ }

impl PagedKVCache {
    pub fn new(num_blocks: usize, block_size: usize, num_heads: usize, head_dim: usize) -> Self;

    pub fn allocate_sequence(&mut self, seq_id: usize, initial_tokens: usize) -> Result<()>;
    pub fn update(/* see source */) -> Result<()>;
    pub fn gather(&self, seq_id: usize) -> Result<(Vec<f32>, Vec<f32>)>;
    pub fn free_sequence(&mut self, seq_id: usize);
    pub fn num_free_blocks(&self) -> usize;
    pub fn num_sequences(&self) -> usize;
}
```

### `QuantizedKVCache`

8-bit quantized cache with per-token scales.

```rust
pub struct QuantizedKVCache { /* private */ }

impl QuantizedKVCache {
    pub fn new(batch_size: usize, max_seq_len: usize, num_heads: usize, head_dim: usize) -> Self;

    pub fn update(/* see source */) -> Result<...>;
    pub fn get(&self) -> (Vec<f32>, Vec<f32>);   // dequantized
    pub fn reset(&mut self);
    pub fn memory_savings(&self) -> f32;
}
```

## Auto-Selection

`AutoSelectAttention` dispatches to Standard / Flash / SlidingWindow /
BlockSparse based on sequence length and configured thresholds.

```rust
pub struct AutoSelectAttention { /* private */ }

impl AutoSelectAttention {
    pub fn new(config: AttentionConfig) -> Self;
    pub fn with_thresholds(config: AttentionConfig, thresholds: SelectionThresholds) -> Self;
    pub fn for_gpu(config: AttentionConfig, arch: GpuArchitecture) -> Self;

    /// Which implementation would be chosen for `seq_len`.
    pub fn select_impl(&self, seq_len: usize) -> AttentionType;

    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
        attention_mask: Option<&[f32]>,
    ) -> Result<AttentionOutput>;

    pub fn estimate_memory(&self, batch: usize, seq_len: usize) -> usize;
}

pub struct SelectionThresholds {
    pub flash_min: usize,
    pub sliding_window_min: usize,
    pub block_sparse_min: usize,
}

impl SelectionThresholds {
    pub fn for_gpu(arch: GpuArchitecture) -> Self;
}

pub enum GpuArchitecture { A100, H100, V100, Generic }
```

`GpuArchitecture` only tunes the length thresholds used for algorithm
selection; the crate itself performs all computation on the CPU.

### Profiling and memory estimation helpers

```rust
pub struct AttentionProfiler { /* private */ }

impl AttentionProfiler {
    pub fn new() -> Self;
    pub fn record(&mut self, seq_len: usize, impl_type: AttentionType, time_ms: f64);
    pub fn average_time(&self, seq_len: usize, impl_type: AttentionType) -> Option<f64>;
    pub fn best_impl(&self, seq_len: usize) -> Option<AttentionType>;
    pub fn clear(&mut self);
    pub fn timings(&self) -> &[(usize, AttentionType, f64)];
}

pub struct MemoryEstimator;

impl MemoryEstimator {
    pub fn standard(batch: usize, seq_len: usize, num_heads: usize, head_dim: usize) -> usize;
    pub fn flash(/* see source */) -> usize;
    pub fn sliding_window(/* see source */) -> usize;
    pub fn transformer(/* see source */) -> usize;
    pub fn fits_in_memory(/* see source */) -> bool;
    pub fn max_seq_len(/* see source */) -> usize;
}
```

## Streaming Inference

`StreamingInference` drives prefill + incremental decode over a KV cache;
build it with `StreamingInferenceBuilder`.

```rust
pub struct StreamingInference { /* private */ }

impl StreamingInference {
    pub fn new(config: StreamingConfig) -> Self;

    pub fn prefill(/* see source */) -> Result<...>;
    pub fn decode_step(/* see source */) -> Result<...>;
    pub fn forward(/* see source */) -> Result<...>;

    pub fn reset(&mut self);
    pub fn current_seq_len(&self) -> usize;
    pub fn is_prefill_done(&self) -> bool;
    pub fn remaining_capacity(&self) -> usize;
}

pub struct StreamingInferenceBuilder { /* private */ }

impl StreamingInferenceBuilder {
    pub fn new() -> Self;
    pub fn num_layers(self, n: usize) -> Self;
    pub fn batch_size(self, n: usize) -> Self;
    pub fn max_seq_len(self, n: usize) -> Self;
    pub fn num_heads(self, n: usize) -> Self;
    pub fn head_dim(self, n: usize) -> Self;
    pub fn causal(self, c: bool) -> Self;
    pub fn standard_attention(self) -> Self;
    pub fn flash_attention(self) -> Self;
    pub fn sliding_window(self, window_size: usize) -> Self;
    pub fn continuous_cache(self) -> Self;
    pub fn paged_cache(self, block_size: usize, num_blocks: usize) -> Self;
    pub fn quantized_cache(self) -> Self;
    pub fn build(self) -> StreamingInference;
}

pub enum AttentionStrategy { /* see source */ }
pub enum CacheStrategy { /* see source */ }

pub struct StreamingConfig { /* fields; builder-style with_* setters */ }

impl StreamingConfig {
    pub fn new(/* see source */) -> Self;
    pub fn with_attention_strategy(self, strategy: AttentionStrategy) -> Self;
    pub fn with_cache_strategy(self, strategy: CacheStrategy) -> Self;
    pub fn with_causal(self, causal: bool) -> Self;
}
```

## Examples

### Standard forward pass

```rust
use long_context_attention::{AttentionConfig, StandardAttention, TensorShape};

let batch = 2;
let seq_len = 128;
let num_heads = 8;
let head_dim = 64;

let config = AttentionConfig::new(num_heads, head_dim).with_causal(true);
let attn = StandardAttention::new(config);

let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
let n = shape.num_elements();
let q = vec![0.01_f32; n];
let k = vec![0.01_f32; n];
let v = vec![0.01_f32; n];

let out = attn.forward(&q, &k, &v, shape, shape, None).unwrap();
assert_eq!(out.output.len(), n);
```

### Auto-selecting an implementation

```rust
use long_context_attention::{AttentionConfig, AutoSelectAttention, TensorShape};

let config = AttentionConfig::new(8, 64).with_max_seq_len(65_536);
let attn = AutoSelectAttention::new(config);

// Inspect which kernel would run for a given length.
let chosen = attn.select_impl(32_768);
println!("selected: {chosen:?}");

let shape = TensorShape::new(1, 4096, 8, 64);
let n = shape.num_elements();
let (q, k, v) = (vec![0.0_f32; n], vec![0.0_f32; n], vec![0.0_f32; n]);
let out = attn.forward(&q, &k, &v, shape, shape, None).unwrap();
```

### Applying RoPE in place

```rust
use long_context_attention::{RotaryEmbedding, TensorShape};

let head_dim = 64;
let rope = RotaryEmbedding::new(head_dim, 4096);

let shape = TensorShape::new(1, 128, 8, head_dim);
let mut q = vec![0.01_f32; shape.num_elements()];
rope.apply(&mut q, shape, None); // positions 0..seq_len
```

### Autoregressive decode with a KV cache

```rust
use long_context_attention::ContinuousKVCache;

let (num_layers, batch, max_seq, heads, head_dim) = (1, 1, 2048, 8, 64);
let mut cache = ContinuousKVCache::new(num_layers, batch, max_seq, heads, head_dim);

// Append one token's KV to layer 0; get back the full cached slices.
let step = heads * head_dim;
let (k_new, v_new) = (vec![0.0_f32; step], vec![0.0_f32; step]);
let (k_full, v_full) = cache.update(0, &k_new, &v_new, 1).unwrap();
assert_eq!(k_full.len(), v_full.len());
```
