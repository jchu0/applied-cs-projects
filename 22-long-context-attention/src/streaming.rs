//! Streaming inference for autoregressive generation.
//!
//! This module provides the infrastructure for token-by-token generation
//! with efficient KV cache management.

use crate::attention::{AttentionOutput, StandardAttention};
use crate::config::{AttentionConfig, TensorShape};
use crate::flash::FlashAttention;
use crate::kv_cache::{ContinuousKVCache, PagedKVCache, QuantizedKVCache};
use crate::sliding_window::SlidingWindowAttention;
use crate::{Error, Result};

/// Attention strategy for streaming inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionStrategy {
    /// Standard O(n^2) attention.
    Standard,
    /// FlashAttention with O(n) memory.
    Flash,
    /// Sliding window for long contexts.
    SlidingWindow { window_size: usize },
}

/// KV cache strategy for streaming inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheStrategy {
    /// Continuous pre-allocated buffer.
    Continuous,
    /// Paged allocation (vLLM-style).
    Paged { block_size: usize, num_blocks: usize },
    /// INT8 quantized cache.
    Quantized,
}

/// Configuration for streaming inference.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Number of layers.
    pub num_layers: usize,
    /// Batch size.
    pub batch_size: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension per head.
    pub head_dim: usize,
    /// Whether to use causal masking.
    pub causal: bool,
    /// Attention strategy.
    pub attention_strategy: AttentionStrategy,
    /// KV cache strategy.
    pub cache_strategy: CacheStrategy,
}

impl StreamingConfig {
    /// Create new streaming config.
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            num_layers,
            batch_size,
            max_seq_len,
            num_heads,
            head_dim,
            causal: true,
            attention_strategy: AttentionStrategy::Standard,
            cache_strategy: CacheStrategy::Continuous,
        }
    }

    /// Set attention strategy.
    pub fn with_attention_strategy(mut self, strategy: AttentionStrategy) -> Self {
        self.attention_strategy = strategy;
        self
    }

    /// Set cache strategy.
    pub fn with_cache_strategy(mut self, strategy: CacheStrategy) -> Self {
        self.cache_strategy = strategy;
        self
    }

    /// Set causal masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }
}

/// Result of a single inference step.
#[derive(Debug)]
pub struct InferenceStepResult {
    /// Output tensor [batch, seq_len, num_heads, head_dim].
    pub output: Vec<f32>,
    /// Output shape.
    pub shape: TensorShape,
    /// Current total sequence length (including KV cache).
    pub total_seq_len: usize,
    /// Whether this was a prefill step.
    pub is_prefill: bool,
}

/// KV cache wrapper supporting multiple strategies.
enum KVCacheImpl {
    Continuous(ContinuousKVCache),
    Paged(PagedKVCache),
    Quantized(QuantizedKVCache),
}

/// Streaming inference engine for autoregressive generation.
pub struct StreamingInference {
    config: StreamingConfig,
    attention_config: AttentionConfig,
    cache: KVCacheImpl,
    /// Current sequence length per batch item.
    current_seq_len: Vec<usize>,
    /// Whether prefill has been done.
    prefill_done: bool,
}

impl StreamingInference {
    /// Create new streaming inference engine.
    pub fn new(config: StreamingConfig) -> Self {
        let attention_config = AttentionConfig::new(config.num_heads, config.head_dim)
            .with_causal(config.causal);

        let cache = match config.cache_strategy {
            CacheStrategy::Continuous => KVCacheImpl::Continuous(ContinuousKVCache::new(
                config.num_layers,
                config.batch_size,
                config.max_seq_len,
                config.num_heads,
                config.head_dim,
            )),
            CacheStrategy::Paged { block_size, num_blocks } => {
                KVCacheImpl::Paged(PagedKVCache::new(
                    num_blocks,
                    block_size,
                    config.num_heads,
                    config.head_dim,
                ))
            }
            CacheStrategy::Quantized => KVCacheImpl::Quantized(QuantizedKVCache::new(
                config.batch_size,
                config.max_seq_len,
                config.num_heads,
                config.head_dim,
            )),
        };

        Self {
            config,
            attention_config,
            cache,
            current_seq_len: vec![0; 1], // Will be resized as needed
            prefill_done: false,
        }
    }

    /// Prefill: process initial prompt tokens.
    ///
    /// This should be called once at the start of generation with the full prompt.
    pub fn prefill(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        seq_len: usize,
        layer: usize,
    ) -> Result<InferenceStepResult> {
        if seq_len > self.config.max_seq_len {
            return Err(Error::SequenceTooLong {
                length: seq_len,
                max: self.config.max_seq_len,
            });
        }

        // Update KV cache
        self.update_cache(layer, key, value, seq_len)?;

        // Compute attention
        let q_shape = TensorShape::new(
            self.config.batch_size,
            seq_len,
            self.config.num_heads,
            self.config.head_dim,
        );

        let output = self.compute_attention(query, key, value, q_shape, q_shape)?;

        if layer == self.config.num_layers - 1 {
            self.current_seq_len = vec![seq_len; self.config.batch_size];
            self.prefill_done = true;
        }

        Ok(InferenceStepResult {
            output: output.output,
            shape: output.shape,
            total_seq_len: seq_len,
            is_prefill: true,
        })
    }

    /// Decode: process a single new token.
    ///
    /// This should be called repeatedly after prefill for each new token.
    pub fn decode_step(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        layer: usize,
    ) -> Result<InferenceStepResult> {
        if !self.prefill_done {
            return Err(Error::InvalidConfig(
                "Must call prefill before decode".to_string(),
            ));
        }

        let current_len = self.current_seq_len.first().copied().unwrap_or(0);
        let new_len = current_len + 1;

        if new_len > self.config.max_seq_len {
            return Err(Error::SequenceTooLong {
                length: new_len,
                max: self.config.max_seq_len,
            });
        }

        // Update KV cache with new token
        self.update_cache(layer, key, value, 1)?;

        // Get full KV cache
        let (cached_k, cached_v) = self.get_cache(layer)?;

        // Query shape: [batch, 1, num_heads, head_dim]
        let q_shape = TensorShape::new(
            self.config.batch_size,
            1,
            self.config.num_heads,
            self.config.head_dim,
        );

        // KV shape: [batch, total_seq_len, num_heads, head_dim]
        let kv_shape = TensorShape::new(
            self.config.batch_size,
            new_len,
            self.config.num_heads,
            self.config.head_dim,
        );

        let output = self.compute_attention(query, &cached_k, &cached_v, q_shape, kv_shape)?;

        if layer == self.config.num_layers - 1 {
            for len in &mut self.current_seq_len {
                *len = new_len;
            }
        }

        Ok(InferenceStepResult {
            output: output.output,
            shape: output.shape,
            total_seq_len: new_len,
            is_prefill: false,
        })
    }

    /// Combined prefill and decode for a batch of tokens.
    ///
    /// Automatically determines whether to use prefill or decode based on state.
    pub fn forward(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        seq_len: usize,
        layer: usize,
    ) -> Result<InferenceStepResult> {
        if !self.prefill_done {
            self.prefill(query, key, value, seq_len, layer)
        } else if seq_len == 1 {
            self.decode_step(query, key, value, layer)
        } else {
            // Multi-token decode (chunked prefill)
            self.chunked_decode(query, key, value, seq_len, layer)
        }
    }

    /// Chunked decode for multiple tokens at once.
    fn chunked_decode(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        seq_len: usize,
        layer: usize,
    ) -> Result<InferenceStepResult> {
        let current_len = self.current_seq_len.first().copied().unwrap_or(0);
        let new_len = current_len + seq_len;

        if new_len > self.config.max_seq_len {
            return Err(Error::SequenceTooLong {
                length: new_len,
                max: self.config.max_seq_len,
            });
        }

        // Update KV cache
        self.update_cache(layer, key, value, seq_len)?;

        // Get full KV cache
        let (cached_k, cached_v) = self.get_cache(layer)?;

        let q_shape = TensorShape::new(
            self.config.batch_size,
            seq_len,
            self.config.num_heads,
            self.config.head_dim,
        );

        let kv_shape = TensorShape::new(
            self.config.batch_size,
            new_len,
            self.config.num_heads,
            self.config.head_dim,
        );

        let output = self.compute_attention(query, &cached_k, &cached_v, q_shape, kv_shape)?;

        if layer == self.config.num_layers - 1 {
            for len in &mut self.current_seq_len {
                *len = new_len;
            }
        }

        Ok(InferenceStepResult {
            output: output.output,
            shape: output.shape,
            total_seq_len: new_len,
            is_prefill: false,
        })
    }

    /// Update KV cache for a layer.
    fn update_cache(
        &mut self,
        layer: usize,
        key: &[f32],
        value: &[f32],
        seq_len: usize,
    ) -> Result<()> {
        match &mut self.cache {
            KVCacheImpl::Continuous(cache) => {
                cache.update(layer, key, value, seq_len)?;
            }
            KVCacheImpl::Paged(cache) => {
                // For paged cache, we need to handle sequences differently
                // Simplified: use sequence 0 for single-batch
                if !self.prefill_done {
                    cache.allocate_sequence(0, self.config.max_seq_len)?;
                }
                cache.update(0, key, value)?;
            }
            KVCacheImpl::Quantized(cache) => {
                cache.update(key, value, seq_len)?;
            }
        }
        Ok(())
    }

    /// Get cached KV for a layer.
    fn get_cache(&self, layer: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        match &self.cache {
            KVCacheImpl::Continuous(cache) => {
                if let Some((k, v, _len)) = cache.get(layer) {
                    Ok((k.to_vec(), v.to_vec()))
                } else {
                    Err(Error::CacheError(format!("Layer {} not found", layer)))
                }
            }
            KVCacheImpl::Paged(cache) => cache.gather(0),
            KVCacheImpl::Quantized(cache) => Ok(cache.get()),
        }
    }

    /// Compute attention based on strategy.
    fn compute_attention(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
    ) -> Result<AttentionOutput> {
        match self.config.attention_strategy {
            AttentionStrategy::Standard => {
                let attention = StandardAttention::new(self.attention_config.clone());
                attention.forward(query, key, value, q_shape, kv_shape, None)
            }
            AttentionStrategy::Flash => {
                let attention = FlashAttention::new(self.attention_config.clone());
                attention.forward(query, key, value, q_shape, kv_shape, None)
            }
            AttentionStrategy::SlidingWindow { window_size } => {
                // Create config with window size
                let config = self.attention_config.clone().with_window_size(window_size);
                let attention = SlidingWindowAttention::new(config);
                attention.forward(query, key, value, q_shape, kv_shape, None)
            }
        }
    }

    /// Reset the cache and state for new generation.
    pub fn reset(&mut self) {
        match &mut self.cache {
            KVCacheImpl::Continuous(cache) => cache.reset(),
            KVCacheImpl::Paged(cache) => {
                cache.free_sequence(0);
            }
            KVCacheImpl::Quantized(cache) => cache.reset(),
        }
        self.current_seq_len = vec![0; self.config.batch_size];
        self.prefill_done = false;
    }

    /// Get current sequence length.
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len.first().copied().unwrap_or(0)
    }

    /// Check if prefill has been done.
    pub fn is_prefill_done(&self) -> bool {
        self.prefill_done
    }

    /// Get remaining capacity.
    pub fn remaining_capacity(&self) -> usize {
        self.config.max_seq_len - self.current_seq_len()
    }
}

/// Builder for streaming inference with fluent API.
pub struct StreamingInferenceBuilder {
    num_layers: usize,
    batch_size: usize,
    max_seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    causal: bool,
    attention_strategy: AttentionStrategy,
    cache_strategy: CacheStrategy,
}

impl StreamingInferenceBuilder {
    /// Create new builder.
    pub fn new() -> Self {
        Self {
            num_layers: 1,
            batch_size: 1,
            max_seq_len: 2048,
            num_heads: 8,
            head_dim: 64,
            causal: true,
            attention_strategy: AttentionStrategy::Standard,
            cache_strategy: CacheStrategy::Continuous,
        }
    }

    /// Set number of layers.
    pub fn num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Set maximum sequence length.
    pub fn max_seq_len(mut self, n: usize) -> Self {
        self.max_seq_len = n;
        self
    }

    /// Set number of attention heads.
    pub fn num_heads(mut self, n: usize) -> Self {
        self.num_heads = n;
        self
    }

    /// Set head dimension.
    pub fn head_dim(mut self, n: usize) -> Self {
        self.head_dim = n;
        self
    }

    /// Set causal masking.
    pub fn causal(mut self, c: bool) -> Self {
        self.causal = c;
        self
    }

    /// Use standard attention.
    pub fn standard_attention(mut self) -> Self {
        self.attention_strategy = AttentionStrategy::Standard;
        self
    }

    /// Use flash attention.
    pub fn flash_attention(mut self) -> Self {
        self.attention_strategy = AttentionStrategy::Flash;
        self
    }

    /// Use sliding window attention.
    pub fn sliding_window(mut self, window_size: usize) -> Self {
        self.attention_strategy = AttentionStrategy::SlidingWindow { window_size };
        self
    }

    /// Use continuous KV cache.
    pub fn continuous_cache(mut self) -> Self {
        self.cache_strategy = CacheStrategy::Continuous;
        self
    }

    /// Use paged KV cache.
    pub fn paged_cache(mut self, block_size: usize, num_blocks: usize) -> Self {
        self.cache_strategy = CacheStrategy::Paged { block_size, num_blocks };
        self
    }

    /// Use quantized KV cache.
    pub fn quantized_cache(mut self) -> Self {
        self.cache_strategy = CacheStrategy::Quantized;
        self
    }

    /// Build the streaming inference engine.
    pub fn build(self) -> StreamingInference {
        let config = StreamingConfig {
            num_layers: self.num_layers,
            batch_size: self.batch_size,
            max_seq_len: self.max_seq_len,
            num_heads: self.num_heads,
            head_dim: self.head_dim,
            causal: self.causal,
            attention_strategy: self.attention_strategy,
            cache_strategy: self.cache_strategy,
        };
        StreamingInference::new(config)
    }
}

impl Default for StreamingInferenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_config_new() {
        let config = StreamingConfig::new(12, 1, 2048, 8, 64);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.max_seq_len, 2048);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
        assert!(config.causal);
    }

    #[test]
    fn test_streaming_config_with_strategies() {
        let config = StreamingConfig::new(12, 1, 2048, 8, 64)
            .with_attention_strategy(AttentionStrategy::Flash)
            .with_cache_strategy(CacheStrategy::Quantized)
            .with_causal(false);

        assert_eq!(config.attention_strategy, AttentionStrategy::Flash);
        assert_eq!(config.cache_strategy, CacheStrategy::Quantized);
        assert!(!config.causal);
    }

    #[test]
    fn test_streaming_inference_new() {
        let config = StreamingConfig::new(2, 1, 256, 4, 16);
        let inference = StreamingInference::new(config);

        assert_eq!(inference.current_seq_len(), 0);
        assert!(!inference.is_prefill_done());
        assert_eq!(inference.remaining_capacity(), 256);
    }

    #[test]
    fn test_streaming_inference_prefill() {
        let config = StreamingConfig::new(1, 1, 256, 2, 8);
        let mut inference = StreamingInference::new(config);

        let seq_len = 10;
        let size = 1 * seq_len * 2 * 8;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let result = inference.prefill(&query, &key, &value, seq_len, 0).unwrap();

        assert!(result.is_prefill);
        assert_eq!(result.total_seq_len, seq_len);
        assert_eq!(result.output.len(), size);
        assert!(inference.is_prefill_done());
        assert_eq!(inference.current_seq_len(), seq_len);
    }

    #[test]
    fn test_streaming_inference_decode() {
        let config = StreamingConfig::new(1, 1, 256, 2, 8);
        let mut inference = StreamingInference::new(config);

        // Prefill
        let prefill_len = 5;
        let prefill_size = 1 * prefill_len * 2 * 8;
        let prefill_q: Vec<f32> = (0..prefill_size).map(|i| (i as f32) * 0.01).collect();
        inference.prefill(&prefill_q, &prefill_q, &prefill_q, prefill_len, 0).unwrap();

        // Decode single token
        let decode_size = 1 * 1 * 2 * 8;
        let decode_q: Vec<f32> = (0..decode_size).map(|i| (i as f32) * 0.01).collect();

        let result = inference.decode_step(&decode_q, &decode_q, &decode_q, 0).unwrap();

        assert!(!result.is_prefill);
        assert_eq!(result.total_seq_len, prefill_len + 1);
        assert_eq!(result.shape.seq_len, 1);
    }

    #[test]
    fn test_streaming_inference_multiple_decode_steps() {
        let config = StreamingConfig::new(1, 1, 256, 2, 8);
        let mut inference = StreamingInference::new(config);

        // Prefill
        let prefill_len = 5;
        let prefill_size = 1 * prefill_len * 2 * 8;
        let prefill_q: Vec<f32> = (0..prefill_size).map(|i| (i as f32) * 0.01).collect();
        inference.prefill(&prefill_q, &prefill_q, &prefill_q, prefill_len, 0).unwrap();

        // Multiple decode steps
        let decode_size = 1 * 1 * 2 * 8;
        let decode_q: Vec<f32> = (0..decode_size).map(|i| (i as f32) * 0.01).collect();

        for i in 0..10 {
            let result = inference.decode_step(&decode_q, &decode_q, &decode_q, 0).unwrap();
            assert_eq!(result.total_seq_len, prefill_len + i + 1);
        }

        assert_eq!(inference.current_seq_len(), prefill_len + 10);
    }

    #[test]
    fn test_streaming_inference_reset() {
        let config = StreamingConfig::new(1, 1, 256, 2, 8);
        let mut inference = StreamingInference::new(config);

        // Prefill
        let prefill_len = 10;
        let prefill_size = 1 * prefill_len * 2 * 8;
        let prefill_q: Vec<f32> = (0..prefill_size).map(|i| (i as f32) * 0.01).collect();
        inference.prefill(&prefill_q, &prefill_q, &prefill_q, prefill_len, 0).unwrap();

        assert!(inference.is_prefill_done());
        assert_eq!(inference.current_seq_len(), prefill_len);

        // Reset
        inference.reset();

        assert!(!inference.is_prefill_done());
        assert_eq!(inference.current_seq_len(), 0);
        assert_eq!(inference.remaining_capacity(), 256);
    }

    #[test]
    fn test_streaming_inference_sequence_too_long() {
        let config = StreamingConfig::new(1, 1, 10, 2, 8);
        let mut inference = StreamingInference::new(config);

        let seq_len = 15; // Exceeds max_seq_len of 10
        let size = 1 * seq_len * 2 * 8;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let result = inference.prefill(&query, &query, &query, seq_len, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_inference_decode_without_prefill() {
        let config = StreamingConfig::new(1, 1, 256, 2, 8);
        let mut inference = StreamingInference::new(config);

        let decode_size = 1 * 1 * 2 * 8;
        let decode_q: Vec<f32> = (0..decode_size).map(|i| (i as f32) * 0.01).collect();

        let result = inference.decode_step(&decode_q, &decode_q, &decode_q, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_streaming_inference_forward_auto() {
        let config = StreamingConfig::new(1, 1, 256, 2, 8);
        let mut inference = StreamingInference::new(config);

        // First call with multiple tokens -> prefill
        let prefill_len = 5;
        let prefill_size = 1 * prefill_len * 2 * 8;
        let prefill_q: Vec<f32> = (0..prefill_size).map(|i| (i as f32) * 0.01).collect();

        let result1 = inference.forward(&prefill_q, &prefill_q, &prefill_q, prefill_len, 0).unwrap();
        assert!(result1.is_prefill);

        // Second call with single token -> decode
        let decode_size = 1 * 1 * 2 * 8;
        let decode_q: Vec<f32> = (0..decode_size).map(|i| (i as f32) * 0.01).collect();

        let result2 = inference.forward(&decode_q, &decode_q, &decode_q, 1, 0).unwrap();
        assert!(!result2.is_prefill);
    }

    #[test]
    fn test_streaming_inference_with_flash_attention() {
        let config = StreamingConfig::new(1, 1, 256, 2, 8)
            .with_attention_strategy(AttentionStrategy::Flash);
        let mut inference = StreamingInference::new(config);

        let seq_len = 10;
        let size = 1 * seq_len * 2 * 8;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let result = inference.prefill(&query, &query, &query, seq_len, 0).unwrap();
        assert_eq!(result.total_seq_len, seq_len);
    }

    #[test]
    fn test_streaming_inference_with_sliding_window() {
        let config = StreamingConfig::new(1, 1, 256, 2, 8)
            .with_attention_strategy(AttentionStrategy::SlidingWindow { window_size: 64 });
        let mut inference = StreamingInference::new(config);

        let seq_len = 10;
        let size = 1 * seq_len * 2 * 8;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let result = inference.prefill(&query, &query, &query, seq_len, 0).unwrap();
        assert_eq!(result.total_seq_len, seq_len);
    }

    #[test]
    fn test_streaming_inference_with_quantized_cache() {
        let config = StreamingConfig::new(1, 1, 256, 2, 8)
            .with_cache_strategy(CacheStrategy::Quantized);
        let mut inference = StreamingInference::new(config);

        let seq_len = 10;
        let size = 1 * seq_len * 2 * 8;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        let result = inference.prefill(&query, &query, &query, seq_len, 0).unwrap();
        assert_eq!(result.total_seq_len, seq_len);
    }

    #[test]
    fn test_streaming_inference_builder() {
        let inference = StreamingInferenceBuilder::new()
            .num_layers(12)
            .batch_size(2)
            .max_seq_len(4096)
            .num_heads(16)
            .head_dim(64)
            .flash_attention()
            .quantized_cache()
            .build();

        assert_eq!(inference.remaining_capacity(), 4096);
    }

    #[test]
    fn test_streaming_inference_builder_default() {
        let builder = StreamingInferenceBuilder::default();
        let inference = builder.build();

        assert_eq!(inference.remaining_capacity(), 2048);
    }

    #[test]
    fn test_streaming_inference_builder_sliding_window() {
        let inference = StreamingInferenceBuilder::new()
            .sliding_window(512)
            .build();

        // Verify it was created successfully
        assert_eq!(inference.current_seq_len(), 0);
    }

    #[test]
    fn test_streaming_inference_builder_paged_cache() {
        let inference = StreamingInferenceBuilder::new()
            .paged_cache(64, 128)
            .build();

        assert_eq!(inference.current_seq_len(), 0);
    }

    #[test]
    fn test_inference_step_result() {
        let result = InferenceStepResult {
            output: vec![1.0, 2.0, 3.0],
            shape: TensorShape::new(1, 1, 1, 3),
            total_seq_len: 10,
            is_prefill: false,
        };

        assert_eq!(result.output.len(), 3);
        assert_eq!(result.total_seq_len, 10);
        assert!(!result.is_prefill);
    }

    #[test]
    fn test_attention_strategy_enum() {
        assert_eq!(AttentionStrategy::Standard, AttentionStrategy::Standard);
        assert_ne!(AttentionStrategy::Flash, AttentionStrategy::Standard);

        let sw = AttentionStrategy::SlidingWindow { window_size: 256 };
        if let AttentionStrategy::SlidingWindow { window_size } = sw {
            assert_eq!(window_size, 256);
        }
    }

    #[test]
    fn test_cache_strategy_enum() {
        assert_eq!(CacheStrategy::Continuous, CacheStrategy::Continuous);

        let paged = CacheStrategy::Paged { block_size: 64, num_blocks: 128 };
        if let CacheStrategy::Paged { block_size, num_blocks } = paged {
            assert_eq!(block_size, 64);
            assert_eq!(num_blocks, 128);
        }
    }

    #[test]
    fn test_streaming_inference_multi_layer() {
        let num_layers = 4;
        let config = StreamingConfig::new(num_layers, 1, 256, 2, 8);
        let mut inference = StreamingInference::new(config);

        let seq_len = 10;
        let size = 1 * seq_len * 2 * 8;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();

        // Process through all layers
        for layer in 0..num_layers {
            let result = inference.prefill(&query, &query, &query, seq_len, layer).unwrap();
            assert_eq!(result.total_seq_len, seq_len);
        }

        assert!(inference.is_prefill_done());
    }

    #[test]
    fn test_streaming_remaining_capacity_decreases() {
        let config = StreamingConfig::new(1, 1, 100, 2, 8);
        let mut inference = StreamingInference::new(config);

        assert_eq!(inference.remaining_capacity(), 100);

        let prefill_len = 30;
        let prefill_size = 1 * prefill_len * 2 * 8;
        let prefill_q: Vec<f32> = (0..prefill_size).map(|i| (i as f32) * 0.01).collect();
        inference.prefill(&prefill_q, &prefill_q, &prefill_q, prefill_len, 0).unwrap();

        assert_eq!(inference.remaining_capacity(), 70);

        // Decode 10 tokens
        let decode_size = 1 * 1 * 2 * 8;
        let decode_q: Vec<f32> = (0..decode_size).map(|i| (i as f32) * 0.01).collect();
        for _ in 0..10 {
            inference.decode_step(&decode_q, &decode_q, &decode_q, 0).unwrap();
        }

        assert_eq!(inference.remaining_capacity(), 60);
    }
}
