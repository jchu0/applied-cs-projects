//! KV cache implementations for efficient inference.

use crate::{Error, Result};

/// KV cache entry for a single layer.
#[derive(Debug, Clone)]
pub struct KVCacheEntry {
    /// Key cache [batch, seq_len, num_heads, head_dim].
    pub key: Vec<f32>,
    /// Value cache [batch, seq_len, num_heads, head_dim].
    pub value: Vec<f32>,
    /// Current sequence length.
    pub current_len: usize,
}

/// Continuous KV cache with pre-allocated buffer.
pub struct ContinuousKVCache {
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Batch size.
    batch_size: usize,
    /// Number of heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Cache entries per layer.
    entries: Vec<KVCacheEntry>,
}

impl ContinuousKVCache {
    /// Create new continuous KV cache.
    pub fn new(
        num_layers: usize,
        batch_size: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let size = batch_size * max_seq_len * num_heads * head_dim;
        let mut entries = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            entries.push(KVCacheEntry {
                key: vec![0.0; size],
                value: vec![0.0; size],
                current_len: 0,
            });
        }

        Self {
            max_seq_len,
            batch_size,
            num_heads,
            head_dim,
            entries,
        }
    }

    /// Update cache for a layer.
    pub fn update(
        &mut self,
        layer: usize,
        key: &[f32],
        value: &[f32],
        seq_len: usize,
    ) -> Result<(&[f32], &[f32])> {
        if layer >= self.entries.len() {
            return Err(Error::CacheError(format!("Invalid layer: {}", layer)));
        }

        let entry = &mut self.entries[layer];
        let new_len = entry.current_len + seq_len;

        if new_len > self.max_seq_len {
            return Err(Error::SequenceTooLong {
                length: new_len,
                max: self.max_seq_len,
            });
        }

        // Copy new KV to cache
        let offset = entry.current_len * self.num_heads * self.head_dim;
        let size = seq_len * self.num_heads * self.head_dim;

        for b in 0..self.batch_size {
            let batch_offset = b * self.max_seq_len * self.num_heads * self.head_dim;
            let src_offset = b * seq_len * self.num_heads * self.head_dim;

            for i in 0..size {
                entry.key[batch_offset + offset + i] = key[src_offset + i];
                entry.value[batch_offset + offset + i] = value[src_offset + i];
            }
        }

        entry.current_len = new_len;

        // Return full cache
        let cache_size = new_len * self.num_heads * self.head_dim * self.batch_size;
        Ok((&entry.key[..cache_size], &entry.value[..cache_size]))
    }

    /// Get current cache for a layer.
    pub fn get(&self, layer: usize) -> Option<(&[f32], &[f32], usize)> {
        self.entries.get(layer).map(|entry| {
            let size = entry.current_len * self.num_heads * self.head_dim * self.batch_size;
            (&entry.key[..size], &entry.value[..size], entry.current_len)
        })
    }

    /// Reset cache.
    pub fn reset(&mut self) {
        for entry in &mut self.entries {
            entry.current_len = 0;
        }
    }

    /// Get current sequence length.
    pub fn current_len(&self) -> usize {
        self.entries.first().map(|e| e.current_len).unwrap_or(0)
    }
}

/// Paged KV cache for memory-efficient batched inference.
pub struct PagedKVCache {
    /// Block size in tokens.
    block_size: usize,
    /// Total number of blocks.
    _num_blocks: usize,
    /// Number of heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Physical key cache blocks.
    key_cache: Vec<f32>,
    /// Physical value cache blocks.
    value_cache: Vec<f32>,
    /// Block tables for each sequence: seq_id -> Vec<block_id>.
    block_tables: std::collections::HashMap<usize, Vec<usize>>,
    /// Free block list.
    free_blocks: Vec<usize>,
    /// Sequence lengths.
    seq_lengths: std::collections::HashMap<usize, usize>,
}

impl PagedKVCache {
    /// Create new paged KV cache.
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let block_elements = block_size * num_heads * head_dim;
        let total_elements = num_blocks * block_elements;

        Self {
            block_size,
            _num_blocks: num_blocks,
            num_heads,
            head_dim,
            key_cache: vec![0.0; total_elements],
            value_cache: vec![0.0; total_elements],
            block_tables: std::collections::HashMap::new(),
            free_blocks: (0..num_blocks).collect(),
            seq_lengths: std::collections::HashMap::new(),
        }
    }

    /// Allocate blocks for a new sequence.
    pub fn allocate_sequence(&mut self, seq_id: usize, initial_tokens: usize) -> Result<()> {
        let num_blocks_needed = (initial_tokens + self.block_size - 1) / self.block_size;

        if self.free_blocks.len() < num_blocks_needed {
            return Err(Error::OutOfMemory {
                required: num_blocks_needed,
                available: self.free_blocks.len(),
            });
        }

        let mut blocks = Vec::with_capacity(num_blocks_needed);
        for _ in 0..num_blocks_needed {
            blocks.push(self.free_blocks.pop().unwrap());
        }

        self.block_tables.insert(seq_id, blocks);
        self.seq_lengths.insert(seq_id, 0);

        Ok(())
    }

    /// Update cache for a sequence.
    pub fn update(
        &mut self,
        seq_id: usize,
        key: &[f32],
        value: &[f32],
    ) -> Result<()> {
        let blocks = self.block_tables.get_mut(&seq_id).ok_or_else(|| {
            Error::CacheError(format!("Sequence {} not found", seq_id))
        })?;

        let current_len = *self.seq_lengths.get(&seq_id).unwrap_or(&0);
        let new_tokens = key.len() / (self.num_heads * self.head_dim);
        let new_len = current_len + new_tokens;

        // Check if we need more blocks
        let blocks_needed = (new_len + self.block_size - 1) / self.block_size;
        while blocks.len() < blocks_needed {
            if self.free_blocks.is_empty() {
                return Err(Error::OutOfMemory {
                    required: blocks_needed - blocks.len(),
                    available: 0,
                });
            }
            blocks.push(self.free_blocks.pop().unwrap());
        }

        // Write new tokens to cache
        let block_elements = self.block_size * self.num_heads * self.head_dim;

        for t in 0..new_tokens {
            let pos = current_len + t;
            let block_idx = pos / self.block_size;
            let block_offset = pos % self.block_size;

            let physical_block = blocks[block_idx];
            let cache_offset =
                physical_block * block_elements + block_offset * self.num_heads * self.head_dim;
            let src_offset = t * self.num_heads * self.head_dim;

            for i in 0..(self.num_heads * self.head_dim) {
                self.key_cache[cache_offset + i] = key[src_offset + i];
                self.value_cache[cache_offset + i] = value[src_offset + i];
            }
        }

        self.seq_lengths.insert(seq_id, new_len);

        Ok(())
    }

    /// Gather KV cache for a sequence.
    pub fn gather(&self, seq_id: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let blocks = self.block_tables.get(&seq_id).ok_or_else(|| {
            Error::CacheError(format!("Sequence {} not found", seq_id))
        })?;

        let seq_len = *self.seq_lengths.get(&seq_id).unwrap_or(&0);
        let output_size = seq_len * self.num_heads * self.head_dim;
        let mut key_output = vec![0.0; output_size];
        let mut value_output = vec![0.0; output_size];

        let block_elements = self.block_size * self.num_heads * self.head_dim;
        let token_elements = self.num_heads * self.head_dim;

        for pos in 0..seq_len {
            let block_idx = pos / self.block_size;
            let block_offset = pos % self.block_size;
            let physical_block = blocks[block_idx];

            let cache_offset = physical_block * block_elements + block_offset * token_elements;
            let output_offset = pos * token_elements;

            for i in 0..token_elements {
                key_output[output_offset + i] = self.key_cache[cache_offset + i];
                value_output[output_offset + i] = self.value_cache[cache_offset + i];
            }
        }

        Ok((key_output, value_output))
    }

    /// Free blocks for a completed sequence.
    pub fn free_sequence(&mut self, seq_id: usize) {
        if let Some(blocks) = self.block_tables.remove(&seq_id) {
            self.free_blocks.extend(blocks);
        }
        self.seq_lengths.remove(&seq_id);
    }

    /// Get number of free blocks.
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    /// Get number of active sequences.
    pub fn num_sequences(&self) -> usize {
        self.block_tables.len()
    }
}

/// Quantized KV cache using INT8.
pub struct QuantizedKVCache {
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Batch size.
    batch_size: usize,
    /// Number of heads.
    num_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Quantized key cache (INT8).
    key_cache: Vec<i8>,
    /// Quantized value cache (INT8).
    value_cache: Vec<i8>,
    /// Per-token key scales.
    key_scales: Vec<f32>,
    /// Per-token value scales.
    value_scales: Vec<f32>,
    /// Current sequence length.
    current_len: usize,
}

impl QuantizedKVCache {
    /// Create new quantized KV cache.
    pub fn new(
        batch_size: usize,
        max_seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Self {
        let size = batch_size * max_seq_len * num_heads * head_dim;
        let scale_size = batch_size * max_seq_len * num_heads;

        Self {
            max_seq_len,
            batch_size,
            num_heads,
            head_dim,
            key_cache: vec![0; size],
            value_cache: vec![0; size],
            key_scales: vec![0.0; scale_size],
            value_scales: vec![0.0; scale_size],
            current_len: 0,
        }
    }

    /// Update cache with new KV.
    pub fn update(
        &mut self,
        key: &[f32],
        value: &[f32],
        seq_len: usize,
    ) -> Result<()> {
        let new_len = self.current_len + seq_len;

        if new_len > self.max_seq_len {
            return Err(Error::SequenceTooLong {
                length: new_len,
                max: self.max_seq_len,
            });
        }

        // Quantize and store
        for b in 0..self.batch_size {
            for t in 0..seq_len {
                let global_t = self.current_len + t;

                for h in 0..self.num_heads {
                    // Find max absolute value for this head's token
                    let mut k_max: f32 = 0.0;
                    let mut v_max: f32 = 0.0;

                    for d in 0..self.head_dim {
                        let idx = b * seq_len * self.num_heads * self.head_dim
                            + t * self.num_heads * self.head_dim
                            + h * self.head_dim
                            + d;

                        k_max = k_max.max(key[idx].abs());
                        v_max = v_max.max(value[idx].abs());
                    }

                    let k_scale = k_max / 127.0;
                    let v_scale = v_max / 127.0;

                    // Store scales
                    let scale_idx = b * self.max_seq_len * self.num_heads
                        + global_t * self.num_heads
                        + h;
                    self.key_scales[scale_idx] = k_scale.max(1e-5);
                    self.value_scales[scale_idx] = v_scale.max(1e-5);

                    // Quantize and store values
                    for d in 0..self.head_dim {
                        let src_idx = b * seq_len * self.num_heads * self.head_dim
                            + t * self.num_heads * self.head_dim
                            + h * self.head_dim
                            + d;

                        let dst_idx = b * self.max_seq_len * self.num_heads * self.head_dim
                            + global_t * self.num_heads * self.head_dim
                            + h * self.head_dim
                            + d;

                        self.key_cache[dst_idx] =
                            (key[src_idx] / self.key_scales[scale_idx]).round().clamp(-128.0, 127.0)
                                as i8;
                        self.value_cache[dst_idx] =
                            (value[src_idx] / self.value_scales[scale_idx]).round().clamp(-128.0, 127.0)
                                as i8;
                    }
                }
            }
        }

        self.current_len = new_len;
        Ok(())
    }

    /// Dequantize and return cache.
    pub fn get(&self) -> (Vec<f32>, Vec<f32>) {
        let size = self.batch_size * self.current_len * self.num_heads * self.head_dim;
        let mut key = vec![0.0; size];
        let mut value = vec![0.0; size];

        for b in 0..self.batch_size {
            for t in 0..self.current_len {
                for h in 0..self.num_heads {
                    let scale_idx = b * self.max_seq_len * self.num_heads
                        + t * self.num_heads
                        + h;
                    let k_scale = self.key_scales[scale_idx];
                    let v_scale = self.value_scales[scale_idx];

                    for d in 0..self.head_dim {
                        let cache_idx = b * self.max_seq_len * self.num_heads * self.head_dim
                            + t * self.num_heads * self.head_dim
                            + h * self.head_dim
                            + d;

                        let out_idx = b * self.current_len * self.num_heads * self.head_dim
                            + t * self.num_heads * self.head_dim
                            + h * self.head_dim
                            + d;

                        key[out_idx] = self.key_cache[cache_idx] as f32 * k_scale;
                        value[out_idx] = self.value_cache[cache_idx] as f32 * v_scale;
                    }
                }
            }
        }

        (key, value)
    }

    /// Reset cache.
    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    /// Get memory savings compared to FP32.
    pub fn memory_savings(&self) -> f32 {
        // INT8 + scale vs FP32: (1 + 4/head_dim) / 4
        let ratio = (1.0 + 4.0 / self.head_dim as f32) / 4.0;
        1.0 - ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== ContinuousKVCache Tests ==========

    #[test]
    fn test_continuous_cache() {
        let mut cache = ContinuousKVCache::new(2, 1, 1024, 8, 64);

        let seq_len = 10;
        let size = seq_len * 8 * 64;
        let key: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let value = key.clone();

        let (cached_k, cached_v) = cache.update(0, &key, &value, seq_len).unwrap();
        assert_eq!(cached_k.len(), size);
        assert_eq!(cached_v.len(), size);
        assert_eq!(cache.current_len(), seq_len);
    }

    #[test]
    fn test_continuous_cache_new() {
        let cache = ContinuousKVCache::new(4, 2, 512, 8, 64);

        assert_eq!(cache.current_len(), 0);

        let (k, v, len) = cache.get(0).unwrap();
        assert_eq!(len, 0);
        assert_eq!(k.len(), 0);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_continuous_cache_incremental_update() {
        let mut cache = ContinuousKVCache::new(2, 1, 256, 4, 16);

        let token_size = 4 * 16;

        // First update: 5 tokens
        let key1: Vec<f32> = (0..5 * token_size).map(|i| i as f32).collect();
        let value1 = key1.clone();
        let (k, v) = cache.update(0, &key1, &value1, 5).unwrap();
        assert_eq!(k.len(), 5 * token_size);
        assert_eq!(cache.current_len(), 5);

        // Second update: 3 more tokens
        let key2: Vec<f32> = (0..3 * token_size).map(|i| (i + 100) as f32).collect();
        let value2 = key2.clone();
        let (k, v) = cache.update(0, &key2, &value2, 3).unwrap();
        assert_eq!(k.len(), 8 * token_size);
        assert_eq!(cache.current_len(), 8);
    }

    #[test]
    fn test_continuous_cache_reset() {
        let mut cache = ContinuousKVCache::new(2, 1, 256, 4, 16);

        let token_size = 4 * 16;
        let key: Vec<f32> = (0..10 * token_size).map(|i| i as f32).collect();
        let value = key.clone();
        cache.update(0, &key, &value, 10).unwrap();

        assert_eq!(cache.current_len(), 10);

        cache.reset();
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn test_continuous_cache_sequence_too_long() {
        let mut cache = ContinuousKVCache::new(2, 1, 10, 2, 4);

        let token_size = 2 * 4;
        let key: Vec<f32> = (0..5 * token_size).map(|i| i as f32).collect();
        let value = key.clone();
        cache.update(0, &key, &value, 5).unwrap();

        // Try to exceed max_seq_len
        let key2: Vec<f32> = (0..10 * token_size).map(|i| i as f32).collect();
        let value2 = key2.clone();
        let result = cache.update(0, &key2, &value2, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_continuous_cache_invalid_layer() {
        let mut cache = ContinuousKVCache::new(2, 1, 256, 4, 16);

        let token_size = 4 * 16;
        let key: Vec<f32> = (0..5 * token_size).map(|i| i as f32).collect();
        let value = key.clone();

        let result = cache.update(5, &key, &value, 5); // Layer 5 doesn't exist
        assert!(result.is_err());
    }

    #[test]
    fn test_continuous_cache_multi_layer() {
        let mut cache = ContinuousKVCache::new(4, 1, 256, 4, 16);

        let token_size = 4 * 16;
        let seq_len = 10;

        // Update each layer
        for layer in 0..4 {
            let key: Vec<f32> = (0..seq_len * token_size)
                .map(|i| (i + layer * 100) as f32)
                .collect();
            let value = key.clone();
            cache.update(layer, &key, &value, seq_len).unwrap();
        }

        // Verify each layer has correct data
        for layer in 0..4 {
            let (k, v, len) = cache.get(layer).unwrap();
            assert_eq!(len, seq_len);
            // First value should be layer * 100
            assert_eq!(k[0], (layer * 100) as f32);
        }
    }

    // ========== PagedKVCache Tests ==========

    #[test]
    fn test_paged_cache() {
        let mut cache = PagedKVCache::new(16, 64, 8, 64);

        cache.allocate_sequence(0, 128).unwrap();
        assert_eq!(cache.num_sequences(), 1);

        let token_size = 8 * 64;
        let key = vec![1.0; token_size];
        let value = vec![2.0; token_size];

        cache.update(0, &key, &value).unwrap();
        let (k, v) = cache.gather(0).unwrap();

        assert_eq!(k.len(), token_size);
        assert_eq!(v.len(), token_size);
        assert_eq!(k[0], 1.0);
        assert_eq!(v[0], 2.0);

        cache.free_sequence(0);
        assert_eq!(cache.num_sequences(), 0);
    }

    #[test]
    fn test_paged_cache_new() {
        let cache = PagedKVCache::new(32, 128, 8, 64);

        assert_eq!(cache.num_sequences(), 0);
        assert_eq!(cache.num_free_blocks(), 32);
    }

    #[test]
    fn test_paged_cache_allocation() {
        let mut cache = PagedKVCache::new(16, 64, 4, 32);

        // Allocate sequence requiring 2 blocks (100 tokens / 64 = 2)
        cache.allocate_sequence(0, 100).unwrap();
        assert_eq!(cache.num_sequences(), 1);
        assert_eq!(cache.num_free_blocks(), 14);

        // Allocate another sequence
        cache.allocate_sequence(1, 50).unwrap();
        assert_eq!(cache.num_sequences(), 2);
        assert_eq!(cache.num_free_blocks(), 13);
    }

    #[test]
    fn test_paged_cache_out_of_memory() {
        let mut cache = PagedKVCache::new(4, 64, 4, 32);

        // Allocate all blocks
        cache.allocate_sequence(0, 256).unwrap(); // 4 blocks

        // Try to allocate more
        let result = cache.allocate_sequence(1, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_paged_cache_incremental_update() {
        let mut cache = PagedKVCache::new(16, 4, 2, 8);

        cache.allocate_sequence(0, 4).unwrap();

        let token_size = 2 * 8;

        // Add tokens one by one
        for i in 0..8 {
            let key = vec![(i + 1) as f32; token_size];
            let value = vec![(i + 10) as f32; token_size];
            cache.update(0, &key, &value).unwrap();
        }

        let (k, v) = cache.gather(0).unwrap();
        assert_eq!(k.len(), 8 * token_size);
        assert_eq!(k[0], 1.0);
        assert_eq!(v[0], 10.0);
    }

    #[test]
    fn test_paged_cache_multiple_sequences() {
        let mut cache = PagedKVCache::new(32, 32, 4, 16);

        let token_size = 4 * 16;

        // Allocate 3 sequences
        for seq_id in 0..3 {
            cache.allocate_sequence(seq_id, 64).unwrap();
            let key = vec![(seq_id + 1) as f32; token_size];
            let value = vec![(seq_id + 10) as f32; token_size];
            cache.update(seq_id, &key, &value).unwrap();
        }

        assert_eq!(cache.num_sequences(), 3);

        // Verify each sequence has correct data
        for seq_id in 0..3 {
            let (k, v) = cache.gather(seq_id).unwrap();
            assert_eq!(k[0], (seq_id + 1) as f32);
            assert_eq!(v[0], (seq_id + 10) as f32);
        }
    }

    #[test]
    fn test_paged_cache_free_and_reuse() {
        let mut cache = PagedKVCache::new(8, 32, 2, 8);

        let token_size = 2 * 8;

        // Fill up cache
        for seq_id in 0..4 {
            cache.allocate_sequence(seq_id, 64).unwrap();
        }
        assert_eq!(cache.num_free_blocks(), 0);

        // Free one sequence
        cache.free_sequence(1);
        assert_eq!(cache.num_sequences(), 3);
        assert_eq!(cache.num_free_blocks(), 2);

        // Allocate new sequence in freed space
        cache.allocate_sequence(4, 32).unwrap();
        assert_eq!(cache.num_sequences(), 4);
    }

    #[test]
    fn test_paged_cache_gather_nonexistent() {
        let cache = PagedKVCache::new(8, 32, 2, 8);

        let result = cache.gather(999);
        assert!(result.is_err());
    }

    // ========== QuantizedKVCache Tests ==========

    #[test]
    fn test_quantized_cache() {
        let mut cache = QuantizedKVCache::new(1, 1024, 8, 64);

        let seq_len = 10;
        let size = seq_len * 8 * 64;
        let key: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let value = key.clone();

        cache.update(&key, &value, seq_len).unwrap();
        let (dequant_k, dequant_v) = cache.get();

        assert_eq!(dequant_k.len(), size);
        assert_eq!(dequant_v.len(), size);

        // INT8 symmetric quantization: the absolute error is bounded by one
        // quantization step (max_abs / 127), which for these magnitudes is
        // well above 0.1 — assert the mathematically correct INT8 bound.
        let max_abs = key.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let tol = max_abs / 127.0 + 1e-4;
        for i in 0..size {
            let error = (dequant_k[i] - key[i]).abs();
            assert!(error <= tol, "Error too large: {} (tol {})", error, tol);
        }

        // Check memory savings
        assert!(cache.memory_savings() > 0.5); // Should save >50%
    }

    #[test]
    fn test_quantized_cache_new() {
        let cache = QuantizedKVCache::new(2, 512, 4, 32);

        let (k, v) = cache.get();
        assert_eq!(k.len(), 0);
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn test_quantized_cache_incremental_update() {
        let mut cache = QuantizedKVCache::new(1, 256, 2, 8);

        let token_size = 2 * 8;

        // First update
        let key1: Vec<f32> = (0..5 * token_size).map(|i| (i as f32) * 0.1).collect();
        let value1 = key1.clone();
        cache.update(&key1, &value1, 5).unwrap();

        let (k1, v1) = cache.get();
        assert_eq!(k1.len(), 5 * token_size);

        // Second update
        let key2: Vec<f32> = (0..3 * token_size).map(|i| (i as f32) * 0.1).collect();
        let value2 = key2.clone();
        cache.update(&key2, &value2, 3).unwrap();

        let (k2, v2) = cache.get();
        assert_eq!(k2.len(), 8 * token_size);
    }

    #[test]
    fn test_quantized_cache_reset() {
        let mut cache = QuantizedKVCache::new(1, 256, 2, 8);

        let token_size = 2 * 8;
        let key: Vec<f32> = (0..10 * token_size).map(|i| (i as f32) * 0.1).collect();
        let value = key.clone();
        cache.update(&key, &value, 10).unwrap();

        let (k, _) = cache.get();
        assert_eq!(k.len(), 10 * token_size);

        cache.reset();
        let (k2, _) = cache.get();
        assert_eq!(k2.len(), 0);
    }

    #[test]
    fn test_quantized_cache_sequence_too_long() {
        let mut cache = QuantizedKVCache::new(1, 10, 2, 4);

        let token_size = 2 * 4;
        let key: Vec<f32> = (0..5 * token_size).map(|i| (i as f32) * 0.1).collect();
        let value = key.clone();
        cache.update(&key, &value, 5).unwrap();

        // Try to exceed max_seq_len
        let key2: Vec<f32> = (0..10 * token_size).map(|i| (i as f32) * 0.1).collect();
        let value2 = key2.clone();
        let result = cache.update(&key2, &value2, 10);

        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_cache_precision() {
        let mut cache = QuantizedKVCache::new(1, 256, 2, 4);

        let token_size = 2 * 4;
        let seq_len = 5;

        // Test with various value ranges
        let test_cases = vec![
            (0.001, "very small"),
            (0.1, "small"),
            (1.0, "unit"),
            (10.0, "large"),
        ];

        for (scale, name) in test_cases {
            cache.reset();
            let key: Vec<f32> = (0..seq_len * token_size)
                .map(|i| (i as f32) * scale)
                .collect();
            let value = key.clone();
            cache.update(&key, &value, seq_len).unwrap();

            let (dequant_k, _) = cache.get();

            // Calculate relative error
            let mut max_rel_error = 0.0f32;
            for (i, &orig) in key.iter().enumerate() {
                if orig.abs() > 1e-6 {
                    let rel_error = ((dequant_k[i] - orig) / orig).abs();
                    max_rel_error = max_rel_error.max(rel_error);
                }
            }

            assert!(
                max_rel_error < 0.1,
                "Relative error too large for {} values: {}",
                name,
                max_rel_error
            );
        }
    }

    #[test]
    fn test_quantized_cache_memory_savings() {
        let cache = QuantizedKVCache::new(1, 256, 8, 64);
        let savings = cache.memory_savings();

        // INT8 + scale vs FP32: should save ~75%
        assert!(savings > 0.5, "Memory savings should be > 50%");
        assert!(savings < 0.9, "Memory savings should be < 90%");
    }

    #[test]
    fn test_quantized_cache_negative_values() {
        let mut cache = QuantizedKVCache::new(1, 256, 2, 4);

        let token_size = 2 * 4;
        let seq_len = 10;

        // Mix of positive and negative values
        let key: Vec<f32> = (0..seq_len * token_size)
            .map(|i| if i % 2 == 0 { i as f32 * 0.1 } else { -(i as f32) * 0.1 })
            .collect();
        let value = key.clone();
        cache.update(&key, &value, seq_len).unwrap();

        let (dequant_k, _) = cache.get();

        // Check that negative values are preserved correctly
        for (i, &orig) in key.iter().enumerate() {
            let error = (dequant_k[i] - orig).abs();
            assert!(error < 0.15, "Error too large at {}: {} vs {}", i, dequant_k[i], orig);
        }
    }

    // ========== KVCacheEntry Tests ==========

    #[test]
    fn test_kv_cache_entry_clone() {
        let entry = KVCacheEntry {
            key: vec![1.0, 2.0, 3.0],
            value: vec![4.0, 5.0, 6.0],
            current_len: 1,
        };

        let cloned = entry.clone();
        assert_eq!(cloned.key, entry.key);
        assert_eq!(cloned.value, entry.value);
        assert_eq!(cloned.current_len, entry.current_len);
    }
}
