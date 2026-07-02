//! Block sparse attention (Longformer-style) implementation.

use crate::attention::{validate_forward_inputs, AttentionOutput};
use crate::config::{AttentionConfig, TensorShape};
use crate::Result;
use rand::Rng;

/// Block sparse attention pattern.
#[derive(Debug, Clone)]
pub struct SparsePattern {
    /// Number of blocks.
    pub num_blocks: usize,
    /// Block size.
    pub block_size: usize,
    /// Pattern matrix (num_blocks x num_blocks), true = attend.
    pub pattern: Vec<bool>,
}

impl SparsePattern {
    /// Create a new sparse pattern.
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            num_blocks,
            block_size,
            pattern: vec![false; num_blocks * num_blocks],
        }
    }

    /// Set pattern at (i, j).
    pub fn set(&mut self, i: usize, j: usize, value: bool) {
        if i < self.num_blocks && j < self.num_blocks {
            self.pattern[i * self.num_blocks + j] = value;
        }
    }

    /// Get pattern at (i, j).
    pub fn get(&self, i: usize, j: usize) -> bool {
        if i < self.num_blocks && j < self.num_blocks {
            self.pattern[i * self.num_blocks + j]
        } else {
            false
        }
    }

    /// Count active blocks.
    pub fn count_active(&self) -> usize {
        self.pattern.iter().filter(|&&x| x).count()
    }

    /// Get sparsity ratio.
    pub fn sparsity(&self) -> f32 {
        let total = self.num_blocks * self.num_blocks;
        if total == 0 {
            0.0
        } else {
            1.0 - (self.count_active() as f32 / total as f32)
        }
    }
}

/// Block sparse attention implementation.
pub struct BlockSparseAttention {
    config: AttentionConfig,
}

impl BlockSparseAttention {
    /// Create new block sparse attention.
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }

    /// Build attention pattern.
    pub fn build_pattern(&self, seq_len: usize) -> SparsePattern {
        let block_size = self.config.block_size;
        let num_blocks = (seq_len + block_size - 1) / block_size;
        let window_blocks = self.config.window_size / block_size;

        let mut pattern = SparsePattern::new(num_blocks, block_size);

        for i in 0..num_blocks {
            // Local window
            let start = if i >= window_blocks {
                i - window_blocks
            } else {
                0
            };
            let end = if self.config.causal {
                i + 1
            } else {
                (i + window_blocks + 1).min(num_blocks)
            };

            for j in start..end {
                pattern.set(i, j, true);
            }

            // Random blocks: sample without replacement from the eligible,
            // not-yet-attended blocks. Rejection sampling here loops forever
            // when fewer eligible blocks exist than num_random_blocks (e.g.
            // row 0 under causal masking, whose only eligible block is the
            // local window itself).
            if self.config.num_random_blocks > 0 {
                let mut rng = rand::thread_rng();
                let mut candidates: Vec<usize> = (0..num_blocks)
                    .filter(|&j| !pattern.get(i, j) && (!self.config.causal || j <= i))
                    .collect();
                let take = self.config.num_random_blocks.min(candidates.len());
                for _ in 0..take {
                    let idx = rng.gen_range(0..candidates.len());
                    let j = candidates.swap_remove(idx);
                    pattern.set(i, j, true);
                }
            }
        }

        // Global tokens attend to all / are attended by all
        let num_global_blocks = self.config.num_global_tokens / block_size;
        for g in 0..num_global_blocks.min(num_blocks) {
            for i in 0..num_blocks {
                pattern.set(g, i, true); // Global attends to all
                pattern.set(i, g, true); // All attend to global
            }
        }

        pattern
    }

    /// Compute block sparse attention.
    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        _kv_shape: TensorShape,
        _attention_mask: Option<&[f32]>,
    ) -> Result<AttentionOutput> {
        // Block sparse indexes key/value with the query sequence length, so all
        // three buffers must match q_shape's element count.
        validate_forward_inputs(query, key, value, q_shape, q_shape, None)?;

        let batch = q_shape.batch;
        let seq_len = q_shape.seq_len;
        let num_heads = q_shape.num_heads;
        let head_dim = q_shape.head_dim;
        let block_size = self.config.block_size;

        let pattern = self.build_pattern(seq_len);
        let num_blocks = pattern.num_blocks;
        let scale = self.config.scale();

        let mut output = vec![0.0; batch * seq_len * num_heads * head_dim];

        for b in 0..batch {
            for h in 0..num_heads {
                // Process each query block
                for qi_block in 0..num_blocks {
                    let qi_start = qi_block * block_size;
                    let qi_end = (qi_start + block_size).min(seq_len);

                    // Find all key blocks this query block attends to
                    let mut attended_blocks = Vec::new();
                    for ki_block in 0..num_blocks {
                        if pattern.get(qi_block, ki_block) {
                            attended_blocks.push(ki_block);
                        }
                    }

                    // For each query position in this block
                    for qi in qi_start..qi_end {
                        // Collect all key positions
                        let mut key_positions = Vec::new();
                        for &ki_block in &attended_blocks {
                            let ki_start = ki_block * block_size;
                            let ki_end = (ki_start + block_size).min(seq_len);
                            for ki in ki_start..ki_end {
                                // Apply causal mask
                                if !self.config.causal || ki <= qi {
                                    key_positions.push(ki);
                                }
                            }
                        }

                        if key_positions.is_empty() {
                            continue;
                        }

                        // Compute attention scores
                        let mut scores = vec![0.0; key_positions.len()];

                        for (pi, &ki) in key_positions.iter().enumerate() {
                            let mut score = 0.0;

                            for d in 0..head_dim {
                                let q_idx = b * seq_len * num_heads * head_dim
                                    + qi * num_heads * head_dim
                                    + h * head_dim
                                    + d;
                                let k_idx = b * seq_len * num_heads * head_dim
                                    + ki * num_heads * head_dim
                                    + h * head_dim
                                    + d;

                                score += query[q_idx] * key[k_idx];
                            }

                            scores[pi] = score * scale;
                        }

                        // Softmax
                        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let mut sum = 0.0;
                        for score in &mut scores {
                            *score = (*score - max_score).exp();
                            sum += *score;
                        }
                        if sum > 0.0 {
                            for score in &mut scores {
                                *score /= sum;
                            }
                        }

                        // Apply to values
                        for d in 0..head_dim {
                            let mut out_val = 0.0;

                            for (pi, &ki) in key_positions.iter().enumerate() {
                                let v_idx = b * seq_len * num_heads * head_dim
                                    + ki * num_heads * head_dim
                                    + h * head_dim
                                    + d;

                                out_val += scores[pi] * value[v_idx];
                            }

                            let out_idx = b * seq_len * num_heads * head_dim
                                + qi * num_heads * head_dim
                                + h * head_dim
                                + d;

                            output[out_idx] = out_val;
                        }
                    }
                }
            }
        }

        Ok(AttentionOutput {
            output,
            shape: q_shape,
            attention_weights: None,
        })
    }

    /// Estimate memory usage.
    pub fn estimate_memory(&self, batch: usize, seq_len: usize) -> usize {
        let pattern = self.build_pattern(seq_len);
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let block_size = self.config.block_size;

        // Memory for active blocks
        let active_blocks = pattern.count_active();
        let attention_memory = batch * num_heads * active_blocks * block_size * block_size * 4;

        // Memory for output
        let output_memory = batch * seq_len * num_heads * head_dim * 4;

        attention_memory + output_memory
    }
}

/// BigBird attention pattern with global, local, and random attention.
pub struct BigBirdPattern {
    /// Number of global tokens at start.
    pub num_global: usize,
    /// Local window size.
    pub window_size: usize,
    /// Number of random connections per position.
    pub num_random: usize,
}

impl BigBirdPattern {
    /// Create new BigBird pattern.
    pub fn new(num_global: usize, window_size: usize, num_random: usize) -> Self {
        Self {
            num_global,
            window_size,
            num_random,
        }
    }

    /// Build pattern for sequence length.
    pub fn build(&self, seq_len: usize) -> Vec<Vec<usize>> {
        let mut attention_map = vec![Vec::new(); seq_len];
        let mut rng = rand::thread_rng();

        for i in 0..seq_len {
            let mut attended = Vec::new();

            // Global tokens
            for g in 0..self.num_global.min(seq_len) {
                attended.push(g);
            }

            // Local window
            let start = if i >= self.window_size / 2 {
                i - self.window_size / 2
            } else {
                0
            };
            let end = (i + self.window_size / 2 + 1).min(seq_len);
            for j in start..end {
                if !attended.contains(&j) {
                    attended.push(j);
                }
            }

            // Random
            for _ in 0..self.num_random {
                let j = rng.gen_range(0..seq_len);
                if !attended.contains(&j) {
                    attended.push(j);
                }
            }

            attended.sort();
            attention_map[i] = attended;
        }

        attention_map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_pattern() {
        let mut pattern = SparsePattern::new(4, 64);
        pattern.set(0, 0, true);
        pattern.set(0, 1, true);
        pattern.set(1, 1, true);

        assert!(pattern.get(0, 0));
        assert!(pattern.get(0, 1));
        assert!(pattern.get(1, 1));
        assert!(!pattern.get(2, 2));
        assert_eq!(pattern.count_active(), 3);
    }

    #[test]
    fn test_sparse_pattern_new() {
        let pattern = SparsePattern::new(8, 32);
        assert_eq!(pattern.num_blocks, 8);
        assert_eq!(pattern.block_size, 32);
        assert_eq!(pattern.pattern.len(), 64);
        assert_eq!(pattern.count_active(), 0);
    }

    #[test]
    fn test_sparse_pattern_sparsity() {
        let mut pattern = SparsePattern::new(4, 16);
        // Total: 16 blocks, set 4 active
        pattern.set(0, 0, true);
        pattern.set(1, 1, true);
        pattern.set(2, 2, true);
        pattern.set(3, 3, true);

        let sparsity = pattern.sparsity();
        assert!((sparsity - 0.75).abs() < 0.01); // 12/16 = 0.75
    }

    #[test]
    fn test_sparse_pattern_bounds() {
        let mut pattern = SparsePattern::new(4, 64);

        // Out of bounds should not panic
        pattern.set(10, 10, true);
        assert!(!pattern.get(10, 10));

        // In bounds
        pattern.set(3, 3, true);
        assert!(pattern.get(3, 3));
    }

    #[test]
    fn test_block_sparse_attention() {
        let config = AttentionConfig::new(2, 4)
            .with_causal(true)
            .with_window_size(128);

        let mut config = config;
        config.block_size = 4;
        config.num_global_tokens = 8;
        config.num_random_blocks = 1;

        let attention = BlockSparseAttention::new(config);

        let batch = 1;
        let seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        assert_eq!(output.output.len(), size);
    }

    #[test]
    fn test_block_sparse_short_buffer_errors() {
        let mut config = AttentionConfig::new(2, 4).with_causal(true);
        config.block_size = 4;
        config.num_global_tokens = 8;
        config.num_random_blocks = 0;

        let attention = BlockSparseAttention::new(config);

        let batch = 1;
        let seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = vec![0.1; size];
        let key: Vec<f32> = vec![0.1; size];
        let value: Vec<f32> = vec![0.1; size - 8]; // too short

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let result = attention.forward(&query, &key, &value, shape, shape, None);

        assert!(matches!(
            result,
            Err(crate::Error::ShapeMismatch { name: "value", .. })
        ));
    }

    #[test]
    fn test_block_sparse_valid_input_still_ok() {
        let mut config = AttentionConfig::new(2, 4).with_causal(true);
        config.block_size = 4;
        config.num_global_tokens = 8;
        config.num_random_blocks = 0;

        let attention = BlockSparseAttention::new(config);

        let batch = 1;
        let seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        assert_eq!(output.output.len(), size);
    }

    #[test]
    fn test_block_sparse_attention_output_shape() {
        let mut config = AttentionConfig::new(4, 8).with_causal(true);
        config.block_size = 8;
        config.num_global_tokens = 8;
        config.num_random_blocks = 0;

        let attention = BlockSparseAttention::new(config);

        let batch = 2;
        let seq_len = 32;
        let num_heads = 4;
        let head_dim = 8;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        assert_eq!(output.shape.batch, batch);
        assert_eq!(output.shape.seq_len, seq_len);
        assert_eq!(output.shape.num_heads, num_heads);
        assert_eq!(output.shape.head_dim, head_dim);
    }

    #[test]
    fn test_block_sparse_attention_numerical_stability() {
        let mut config = AttentionConfig::new(2, 4).with_causal(true);
        config.block_size = 4;
        config.num_global_tokens = 4;
        config.num_random_blocks = 0;

        let attention = BlockSparseAttention::new(config);

        let batch = 1;
        let seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;

        // Large values to test stability
        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 10.0).collect();
        let key = query.clone();
        let value: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        // All outputs should be finite
        for val in &output.output {
            assert!(val.is_finite(), "Output contains non-finite values");
        }
    }

    #[test]
    fn test_block_sparse_attention_batch_processing() {
        let mut config = AttentionConfig::new(2, 4).with_causal(true);
        config.block_size = 4;
        config.num_global_tokens = 4;
        config.num_random_blocks = 0;

        let attention = BlockSparseAttention::new(config);

        let batch = 4;
        let seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        assert_eq!(output.output.len(), size);
        assert_eq!(output.shape.batch, batch);
    }

    #[test]
    fn test_block_sparse_attention_causal() {
        let mut config = AttentionConfig::new(1, 4).with_causal(true);
        config.block_size = 4;
        config.num_global_tokens = 4;
        config.num_random_blocks = 0;
        config.window_size = 8;

        let attention = BlockSparseAttention::new(config);

        let batch = 1;
        let seq_len = 8;
        let num_heads = 1;
        let head_dim = 4;

        // Create values that increase with position
        let mut query = vec![0.0; batch * seq_len * num_heads * head_dim];
        let mut key = vec![0.0; batch * seq_len * num_heads * head_dim];
        let mut value = vec![0.0; batch * seq_len * num_heads * head_dim];

        for s in 0..seq_len {
            for d in 0..head_dim {
                let idx = s * num_heads * head_dim + d;
                query[idx] = 0.1;
                key[idx] = 0.1;
                value[idx] = s as f32;
            }
        }

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        // First position should only see itself (value 0)
        let first_val = output.output[0];
        assert!((first_val - 0.0).abs() < 0.5);
    }

    #[test]
    fn test_build_pattern_local_window() {
        let mut config = AttentionConfig::new(2, 4).with_causal(true);
        config.block_size = 4;
        config.window_size = 8; // 2 blocks
        config.num_global_tokens = 0;
        config.num_random_blocks = 0;

        let attention = BlockSparseAttention::new(config);
        let pattern = attention.build_pattern(16);

        // Check diagonal blocks are set
        assert!(pattern.get(0, 0)); // Block 0 attends to block 0
        assert!(pattern.get(1, 0)); // Block 1 attends to block 0
        assert!(pattern.get(1, 1)); // Block 1 attends to block 1
    }

    #[test]
    fn test_build_pattern_global_tokens() {
        let mut config = AttentionConfig::new(2, 4).with_causal(false);
        config.block_size = 8;
        config.window_size = 8;
        config.num_global_tokens = 8; // 1 block
        config.num_random_blocks = 0;

        let attention = BlockSparseAttention::new(config);
        let pattern = attention.build_pattern(32);

        // Global block (block 0) should attend to all
        for j in 0..4 {
            assert!(pattern.get(0, j), "Global block should attend to block {}", j);
        }

        // All blocks should attend to global block
        for i in 0..4 {
            assert!(pattern.get(i, 0), "Block {} should attend to global block", i);
        }
    }

    #[test]
    fn test_pattern_sparsity() {
        let config = AttentionConfig::new(8, 64)
            .with_window_size(512)
            .with_causal(true);

        let mut config = config;
        config.block_size = 64;
        config.num_global_tokens = 64;
        config.num_random_blocks = 2;

        let attention = BlockSparseAttention::new(config);
        let pattern = attention.build_pattern(4096);

        // Should be sparse
        assert!(pattern.sparsity() > 0.5);
    }

    #[test]
    fn test_block_sparse_memory_estimate() {
        let mut config = AttentionConfig::new(8, 64);
        config.block_size = 64;
        config.window_size = 256;
        config.num_global_tokens = 64;
        config.num_random_blocks = 2;

        let attention = BlockSparseAttention::new(config);
        let memory = attention.estimate_memory(1, 4096);

        // Should be significantly less than full O(n^2) attention
        let full_attention_memory = 1 * 8 * 4096 * 4096 * 4;
        assert!(memory < full_attention_memory / 2);
    }

    #[test]
    fn test_bigbird_pattern() {
        let pattern = BigBirdPattern::new(4, 8, 3);
        let attention_map = pattern.build(64);

        assert_eq!(attention_map.len(), 64);

        // Each position should attend to at least global tokens
        for attended in &attention_map {
            assert!(attended.len() >= 4);
        }
    }

    #[test]
    fn test_bigbird_pattern_global_tokens() {
        let pattern = BigBirdPattern::new(8, 16, 0);
        let attention_map = pattern.build(32);

        // Each position should include global tokens (0-7)
        for (i, attended) in attention_map.iter().enumerate() {
            for g in 0..8 {
                assert!(
                    attended.contains(&g),
                    "Position {} should attend to global token {}",
                    i,
                    g
                );
            }
        }
    }

    #[test]
    fn test_bigbird_pattern_local_window() {
        let pattern = BigBirdPattern::new(0, 4, 0);
        let attention_map = pattern.build(16);

        // Check position 8 attends to window [6, 10]
        let pos8 = &attention_map[8];
        assert!(pos8.contains(&6) || pos8.contains(&7));
        assert!(pos8.contains(&8));
        assert!(pos8.contains(&9) || pos8.contains(&10));
    }

    #[test]
    fn test_bigbird_pattern_sorted() {
        let pattern = BigBirdPattern::new(4, 8, 5);
        let attention_map = pattern.build(64);

        for attended in &attention_map {
            // Positions should be sorted
            for window in attended.windows(2) {
                assert!(window[0] <= window[1], "Attended positions should be sorted");
            }
        }
    }

    #[test]
    fn test_block_sparse_bidirectional() {
        let mut config = AttentionConfig::new(2, 4).with_causal(false);
        config.block_size = 4;
        config.window_size = 16;
        config.num_global_tokens = 4;
        config.num_random_blocks = 0;

        let attention = BlockSparseAttention::new(config);

        let batch = 1;
        let seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|_| 1.0).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        // All outputs should be finite
        for val in &output.output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_different_block_sizes() {
        for block_size in [2, 4, 8, 16] {
            let mut config = AttentionConfig::new(2, 4).with_causal(true);
            config.block_size = block_size;
            config.window_size = block_size * 2;
            config.num_global_tokens = block_size;
            config.num_random_blocks = 0;

            let attention = BlockSparseAttention::new(config);

            let seq_len = block_size * 4;
            let size = 1 * seq_len * 2 * 4;
            let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
            let key = query.clone();
            let value = query.clone();

            let shape = TensorShape::new(1, seq_len, 2, 4);

            let output = attention
                .forward(&query, &key, &value, shape, shape, None)
                .unwrap();

            assert_eq!(output.output.len(), size);
        }
    }
}
