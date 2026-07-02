//! Flash attention implementation with memory-efficient tiling.

use crate::attention::{validate_forward_inputs, AttentionOutput};
use crate::config::{AttentionConfig, TensorShape};
use crate::Result;

/// Flash attention implementation.
///
/// Uses tiling and online softmax for O(n) memory instead of O(n^2).
pub struct FlashAttention {
    config: AttentionConfig,
    /// Block size for tiling in M dimension (queries).
    block_m: usize,
    /// Block size for tiling in N dimension (keys).
    block_n: usize,
}

impl FlashAttention {
    /// Create new flash attention with default block sizes.
    pub fn new(config: AttentionConfig) -> Self {
        Self {
            config,
            block_m: 64,
            block_n: 64,
        }
    }

    /// Create with custom block sizes.
    pub fn with_block_sizes(config: AttentionConfig, block_m: usize, block_n: usize) -> Self {
        Self {
            config,
            block_m,
            block_n,
        }
    }

    /// Compute flash attention.
    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
        attention_mask: Option<&[f32]>,
    ) -> Result<AttentionOutput> {
        validate_forward_inputs(query, key, value, q_shape, kv_shape, attention_mask)?;

        let batch = q_shape.batch;
        let q_len = q_shape.seq_len;
        let kv_len = kv_shape.seq_len;
        let num_heads = q_shape.num_heads;
        let head_dim = q_shape.head_dim;

        let scale = self.config.scale();
        let mut output = vec![0.0; batch * q_len * num_heads * head_dim];

        // Also track logsumexp for potential backward pass
        let mut logsumexp = vec![0.0; batch * num_heads * q_len];

        for b in 0..batch {
            for h in 0..num_heads {
                // Process query blocks
                let num_q_blocks = (q_len + self.block_m - 1) / self.block_m;

                for q_block in 0..num_q_blocks {
                    let q_start = q_block * self.block_m;
                    let q_end = (q_start + self.block_m).min(q_len);
                    let q_block_size = q_end - q_start;

                    // Initialize accumulators for this query block
                    let mut m_i = vec![f32::NEG_INFINITY; q_block_size]; // Running max
                    let mut l_i = vec![0.0; q_block_size]; // Running sum
                    let mut acc = vec![0.0; q_block_size * head_dim]; // Accumulator

                    // Process key/value blocks
                    let num_kv_blocks = (kv_len + self.block_n - 1) / self.block_n;
                    let end_kv_block = if self.config.causal {
                        ((q_end + self.block_n - 1) / self.block_n).min(num_kv_blocks)
                    } else {
                        num_kv_blocks
                    };

                    for kv_block in 0..end_kv_block {
                        let k_start = kv_block * self.block_n;
                        let k_end = (k_start + self.block_n).min(kv_len);
                        let k_block_size = k_end - k_start;

                        // Compute QK^T for this tile
                        let mut qk = vec![0.0; q_block_size * k_block_size];

                        for qi in 0..q_block_size {
                            let global_qi = q_start + qi;

                            for ki in 0..k_block_size {
                                let global_ki = k_start + ki;

                                // Apply causal mask
                                if self.config.causal && global_ki > global_qi {
                                    qk[qi * k_block_size + ki] = f32::NEG_INFINITY;
                                    continue;
                                }

                                let mut score = 0.0;

                                for d in 0..head_dim {
                                    let q_idx = b * q_len * num_heads * head_dim
                                        + global_qi * num_heads * head_dim
                                        + h * head_dim
                                        + d;
                                    let k_idx = b * kv_len * num_heads * head_dim
                                        + global_ki * num_heads * head_dim
                                        + h * head_dim
                                        + d;

                                    score += query[q_idx] * key[k_idx];
                                }

                                qk[qi * k_block_size + ki] = score * scale;
                            }
                        }

                        // Online softmax update
                        for qi in 0..q_block_size {
                            // Find max in this block
                            let mut m_ij = f32::NEG_INFINITY;
                            for ki in 0..k_block_size {
                                m_ij = m_ij.max(qk[qi * k_block_size + ki]);
                            }

                            // Compute exp and local sum
                            let mut l_ij = 0.0;
                            for ki in 0..k_block_size {
                                let idx = qi * k_block_size + ki;
                                qk[idx] = (qk[idx] - m_ij).exp();
                                l_ij += qk[idx];
                            }

                            // Update running statistics
                            let m_new = m_i[qi].max(m_ij);
                            let alpha = (m_i[qi] - m_new).exp();
                            let beta = (m_ij - m_new).exp();
                            let l_new = alpha * l_i[qi] + beta * l_ij;

                            // Rescale accumulator
                            if l_new > 0.0 {
                                let rescale = alpha * l_i[qi] / l_new;
                                for d in 0..head_dim {
                                    acc[qi * head_dim + d] *= rescale;
                                }

                                // Add contribution from this KV block
                                for ki in 0..k_block_size {
                                    let global_ki = k_start + ki;
                                    let p = qk[qi * k_block_size + ki] * beta / l_new;

                                    for d in 0..head_dim {
                                        let v_idx = b * kv_len * num_heads * head_dim
                                            + global_ki * num_heads * head_dim
                                            + h * head_dim
                                            + d;

                                        acc[qi * head_dim + d] += p * value[v_idx];
                                    }
                                }
                            }

                            // Update state
                            m_i[qi] = m_new;
                            l_i[qi] = l_new;
                        }
                    }

                    // Write output and logsumexp
                    for qi in 0..q_block_size {
                        let global_qi = q_start + qi;

                        for d in 0..head_dim {
                            let out_idx = b * q_len * num_heads * head_dim
                                + global_qi * num_heads * head_dim
                                + h * head_dim
                                + d;

                            output[out_idx] = acc[qi * head_dim + d];
                        }

                        // Store logsumexp for backward
                        let lse_idx = b * num_heads * q_len + h * q_len + global_qi;
                        logsumexp[lse_idx] = m_i[qi] + l_i[qi].ln();
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
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;

        // Memory for intermediate tile: block_m * block_n
        let tile_memory = self.block_m * self.block_n * 4; // f32

        // Memory for accumulator per query block
        let acc_memory = self.block_m * head_dim * 4;

        // Memory for output
        let output_memory = batch * seq_len * num_heads * head_dim * 4;

        // Memory for logsumexp
        let lse_memory = batch * num_heads * seq_len * 4;

        // Total is dominated by output, not O(n^2)
        tile_memory + acc_memory + output_memory + lse_memory
    }
}

/// Linear attention using kernel feature maps.
///
/// Approximates softmax attention with O(n) complexity using
/// feature maps: Attention(Q,K,V) ≈ φ(Q)(φ(K)^T V)
pub struct LinearAttention {
    _config: AttentionConfig,
}

impl LinearAttention {
    /// Create new linear attention.
    pub fn new(config: AttentionConfig) -> Self {
        Self { _config: config }
    }

    /// Compute linear attention.
    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
    ) -> Result<AttentionOutput> {
        validate_forward_inputs(query, key, value, q_shape, kv_shape, None)?;

        let batch = q_shape.batch;
        let q_len = q_shape.seq_len;
        let kv_len = kv_shape.seq_len;
        let num_heads = q_shape.num_heads;
        let head_dim = q_shape.head_dim;

        let mut output = vec![0.0; batch * q_len * num_heads * head_dim];

        // Apply ELU+1 feature map: φ(x) = elu(x) + 1
        let feature_map = |x: f32| -> f32 {
            if x >= 0.0 {
                x + 1.0
            } else {
                x.exp()
            }
        };

        for b in 0..batch {
            for h in 0..num_heads {
                // Compute K^T V matrix: [head_dim, head_dim]
                let mut kv_matrix = vec![0.0; head_dim * head_dim];
                let mut k_sum = vec![0.0; head_dim];

                for ki in 0..kv_len {
                    // Apply feature map to key
                    let mut k_mapped = vec![0.0; head_dim];
                    for d in 0..head_dim {
                        let k_idx = b * kv_len * num_heads * head_dim
                            + ki * num_heads * head_dim
                            + h * head_dim
                            + d;
                        k_mapped[d] = feature_map(key[k_idx]);
                        k_sum[d] += k_mapped[d];
                    }

                    // Outer product k * v^T
                    for d1 in 0..head_dim {
                        for d2 in 0..head_dim {
                            let v_idx = b * kv_len * num_heads * head_dim
                                + ki * num_heads * head_dim
                                + h * head_dim
                                + d2;

                            kv_matrix[d1 * head_dim + d2] += k_mapped[d1] * value[v_idx];
                        }
                    }
                }

                // Compute output for each query
                for qi in 0..q_len {
                    // Apply feature map to query
                    let mut q_mapped = vec![0.0; head_dim];
                    let mut normalizer = 0.0;

                    for d in 0..head_dim {
                        let q_idx = b * q_len * num_heads * head_dim
                            + qi * num_heads * head_dim
                            + h * head_dim
                            + d;
                        q_mapped[d] = feature_map(query[q_idx]);
                        normalizer += q_mapped[d] * k_sum[d];
                    }

                    // Compute q * (K^T V)
                    for d2 in 0..head_dim {
                        let mut out_val = 0.0;
                        for d1 in 0..head_dim {
                            out_val += q_mapped[d1] * kv_matrix[d1 * head_dim + d2];
                        }

                        let out_idx = b * q_len * num_heads * head_dim
                            + qi * num_heads * head_dim
                            + h * head_dim
                            + d2;

                        output[out_idx] = if normalizer > 1e-6 {
                            out_val / normalizer
                        } else {
                            out_val
                        };
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::StandardAttention;

    #[test]
    fn test_flash_attention() {
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let attention = FlashAttention::new(config);

        let batch = 1;
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
    }

    #[test]
    fn test_flash_attention_output_shape() {
        let config = AttentionConfig::new(4, 8).with_causal(false);
        let attention = FlashAttention::new(config);

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

        assert_eq!(output.output.len(), size);
        assert_eq!(output.shape.batch, batch);
        assert_eq!(output.shape.seq_len, seq_len);
        assert_eq!(output.shape.num_heads, num_heads);
        assert_eq!(output.shape.head_dim, head_dim);
    }

    #[test]
    fn test_flash_attention_matches_standard() {
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let flash = FlashAttention::new(config.clone());
        let standard = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 8;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let flash_output = flash
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();
        let standard_output = standard
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        // Compare outputs with tolerance
        for (f, s) in flash_output.output.iter().zip(standard_output.output.iter()) {
            assert!(
                (f - s).abs() < 1e-3,
                "Flash and standard outputs differ: {} vs {}",
                f,
                s
            );
        }
    }

    #[test]
    fn test_flash_attention_bidirectional() {
        let config = AttentionConfig::new(2, 4).with_causal(false);
        let attention = FlashAttention::new(config);

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

        // With uniform inputs, output should be uniform
        for val in &output.output {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_flash_attention_causal_masking() {
        let config = AttentionConfig::new(1, 4).with_causal(true);
        let attention = FlashAttention::new(config);

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

        // First position can only see itself
        let first_pos_val = output.output[0];
        assert!((first_pos_val - 0.0).abs() < 0.1);

        // Later positions see more, should have higher values
        let mid_pos_val = output.output[(seq_len / 2) * num_heads * head_dim];
        assert!(mid_pos_val > first_pos_val);
    }

    #[test]
    fn test_flash_attention_numerical_stability() {
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let attention = FlashAttention::new(config);

        let batch = 1;
        let seq_len = 32;
        let num_heads = 2;
        let head_dim = 4;

        // Large values to test stability
        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 5.0).collect();
        let key = query.clone();
        let value: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        // No NaN or Inf
        for val in &output.output {
            assert!(val.is_finite(), "Output contains non-finite values");
        }
    }

    #[test]
    fn test_flash_attention_custom_block_sizes() {
        let config = AttentionConfig::new(2, 4).with_causal(true);

        // Test different block sizes
        for (block_m, block_n) in [(16, 16), (32, 32), (64, 64), (32, 64)] {
            let attention = FlashAttention::with_block_sizes(config.clone(), block_m, block_n);

            let batch = 1;
            let seq_len = 64;
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
        }
    }

    #[test]
    fn test_flash_attention_batch_processing() {
        let config = AttentionConfig::new(4, 8).with_causal(true);
        let attention = FlashAttention::new(config);

        let batch = 4;
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

        assert_eq!(output.output.len(), size);
        assert_eq!(output.shape.batch, batch);
    }

    #[test]
    fn test_flash_attention_longer_sequence() {
        let config = AttentionConfig::new(4, 16).with_causal(true);
        let attention = FlashAttention::new(config);

        let batch = 1;
        let seq_len = 256;
        let num_heads = 4;
        let head_dim = 16;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| ((i % 100) as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        assert_eq!(output.output.len(), size);

        // All outputs should be finite
        for val in &output.output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_flash_memory_estimate() {
        let config = AttentionConfig::new(32, 128);
        let attention = FlashAttention::new(config);

        let memory = attention.estimate_memory(1, 32768);

        // Should be O(n), not O(n^2)
        // O(n^2) would be ~4GB for 32K seq, O(n) should be ~500MB
        assert!(memory < 1_000_000_000);
    }

    #[test]
    fn test_flash_memory_estimate_scaling() {
        let config = AttentionConfig::new(8, 64);
        let attention = FlashAttention::new(config);

        let mem_1k = attention.estimate_memory(1, 1024);
        let mem_2k = attention.estimate_memory(1, 2048);
        let mem_4k = attention.estimate_memory(1, 4096);

        // Memory should scale linearly (roughly 2x for 2x seq_len)
        let ratio_2k_1k = mem_2k as f64 / mem_1k as f64;
        let ratio_4k_2k = mem_4k as f64 / mem_2k as f64;

        // Allow some tolerance for fixed overhead
        assert!(ratio_2k_1k < 3.0, "Memory should scale sub-quadratically");
        assert!(ratio_4k_2k < 3.0, "Memory should scale sub-quadratically");
    }

    #[test]
    fn test_linear_attention() {
        let config = AttentionConfig::new(2, 4);
        let attention = LinearAttention::new(config);

        let batch = 1;
        let seq_len = 32;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let q_shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let kv_shape = q_shape;

        let output = attention
            .forward(&query, &key, &value, q_shape, kv_shape)
            .unwrap();

        assert_eq!(output.output.len(), size);
    }

    #[test]
    fn test_linear_attention_output_shape() {
        let config = AttentionConfig::new(4, 8);
        let attention = LinearAttention::new(config);

        let batch = 2;
        let seq_len = 64;
        let num_heads = 4;
        let head_dim = 8;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention.forward(&query, &key, &value, shape, shape).unwrap();

        assert_eq!(output.shape.batch, batch);
        assert_eq!(output.shape.seq_len, seq_len);
        assert_eq!(output.shape.num_heads, num_heads);
        assert_eq!(output.shape.head_dim, head_dim);
    }

    #[test]
    fn test_linear_attention_numerical_stability() {
        let config = AttentionConfig::new(2, 4);
        let attention = LinearAttention::new(config);

        let batch = 1;
        let seq_len = 64;
        let num_heads = 2;
        let head_dim = 4;

        // Mix of positive and negative values
        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size)
            .map(|i| if i % 2 == 0 { (i as f32) * 0.1 } else { -(i as f32) * 0.1 })
            .collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention.forward(&query, &key, &value, shape, shape).unwrap();

        // All outputs should be finite
        for val in &output.output {
            assert!(val.is_finite(), "Output contains non-finite values");
        }
    }

    #[test]
    fn test_linear_attention_batch_processing() {
        let config = AttentionConfig::new(2, 4);
        let attention = LinearAttention::new(config);

        let batch = 4;
        let seq_len = 32;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention.forward(&query, &key, &value, shape, shape).unwrap();

        assert_eq!(output.output.len(), size);
        assert_eq!(output.shape.batch, batch);
    }

    #[test]
    fn test_linear_attention_different_qkv_lengths() {
        let config = AttentionConfig::new(2, 4);
        let attention = LinearAttention::new(config);

        let batch = 1;
        let q_len = 16;
        let kv_len = 32;
        let num_heads = 2;
        let head_dim = 4;

        let q_size = batch * q_len * num_heads * head_dim;
        let kv_size = batch * kv_len * num_heads * head_dim;

        let query: Vec<f32> = (0..q_size).map(|i| (i as f32) * 0.01).collect();
        let key: Vec<f32> = (0..kv_size).map(|i| (i as f32) * 0.01).collect();
        let value: Vec<f32> = (0..kv_size).map(|i| (i as f32) * 0.01).collect();

        let q_shape = TensorShape::new(batch, q_len, num_heads, head_dim);
        let kv_shape = TensorShape::new(batch, kv_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, q_shape, kv_shape)
            .unwrap();

        assert_eq!(output.output.len(), q_size);
        assert_eq!(output.shape.seq_len, q_len);
    }

    #[test]
    fn test_flash_vs_linear_attention_shapes() {
        let config = AttentionConfig::new(2, 4);

        let batch = 1;
        let seq_len = 32;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        // Both should produce same shape output
        let flash = FlashAttention::new(config.clone());
        let linear = LinearAttention::new(config);

        let flash_out = flash
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();
        let linear_out = linear
            .forward(&query, &key, &value, shape, shape)
            .unwrap();

        assert_eq!(flash_out.output.len(), linear_out.output.len());
        assert_eq!(flash_out.shape.seq_len, linear_out.shape.seq_len);
    }

    #[test]
    fn test_flash_attention_short_buffer_errors() {
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let attention = FlashAttention::new(config);

        let batch = 1;
        let seq_len = 16;
        let num_heads = 2;
        let head_dim = 4;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = vec![0.01; size - 3]; // too short
        let key: Vec<f32> = vec![0.01; size];
        let value: Vec<f32> = vec![0.01; size];

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let result = attention.forward(&query, &key, &value, shape, shape, None);

        assert!(matches!(
            result,
            Err(crate::Error::ShapeMismatch { name: "query", .. })
        ));
    }

    #[test]
    fn test_flash_attention_valid_input_still_ok() {
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let attention = FlashAttention::new(config);

        let batch = 1;
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
    }

    #[test]
    fn test_linear_attention_short_buffer_errors() {
        let config = AttentionConfig::new(2, 4);
        let attention = LinearAttention::new(config);

        let batch = 1;
        let seq_len = 8;
        let num_heads = 2;
        let head_dim = 4;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = vec![0.01; size];
        let key: Vec<f32> = vec![0.01; size];
        let value: Vec<f32> = vec![0.01; size - 1]; // too short

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let result = attention.forward(&query, &key, &value, shape, shape);

        assert!(matches!(
            result,
            Err(crate::Error::ShapeMismatch { name: "value", .. })
        ));
    }

    #[test]
    fn test_flash_attention_online_softmax_correctness() {
        // Test that online softmax produces same result as standard softmax
        let config = AttentionConfig::new(1, 4).with_causal(true);
        let flash = FlashAttention::with_block_sizes(config.clone(), 4, 4); // Small blocks
        let standard = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 12; // Multiple of block size
        let num_heads = 1;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.05).collect();
        let key = query.clone();
        let value = query.clone();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let flash_out = flash
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();
        let standard_out = standard
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        // Should match within tolerance
        for (f, s) in flash_out.output.iter().zip(standard_out.output.iter()) {
            assert!(
                (f - s).abs() < 1e-2,
                "Online softmax differs from standard: {} vs {}",
                f,
                s
            );
        }
    }
}
