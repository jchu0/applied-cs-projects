//! Sliding window attention for linear memory complexity.

use crate::attention::AttentionOutput;
use crate::config::{AttentionConfig, TensorShape};
use crate::Result;

/// Sliding window attention implementation.
///
/// Each token attends only to tokens within a fixed window,
/// reducing memory from O(n^2) to O(n * window_size).
pub struct SlidingWindowAttention {
    config: AttentionConfig,
}

impl SlidingWindowAttention {
    /// Create new sliding window attention.
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }

    /// Compute sliding window attention.
    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
        attention_mask: Option<&[f32]>,
    ) -> Result<AttentionOutput> {
        let batch = q_shape.batch;
        let q_len = q_shape.seq_len;
        let kv_len = kv_shape.seq_len;
        let num_heads = q_shape.num_heads;
        let head_dim = q_shape.head_dim;
        let window_size = self.config.window_size;

        let scale = self.config.scale();
        let mut output = vec![0.0; batch * q_len * num_heads * head_dim];

        // Process each batch and head
        for b in 0..batch {
            for h in 0..num_heads {
                for qi in 0..q_len {
                    // Determine window bounds for this query position
                    let window_start = if qi >= window_size {
                        qi - window_size + 1
                    } else {
                        0
                    };
                    let window_end = if self.config.causal {
                        (qi + 1).min(kv_len)
                    } else {
                        (qi + window_size).min(kv_len)
                    };

                    // Compute attention scores within window
                    let window_len = window_end - window_start;
                    let mut scores = vec![0.0; window_len];

                    for (wi, ki) in (window_start..window_end).enumerate() {
                        let mut score = 0.0;

                        for d in 0..head_dim {
                            let q_idx = b * q_len * num_heads * head_dim
                                + qi * num_heads * head_dim
                                + h * head_dim
                                + d;
                            let k_idx = b * kv_len * num_heads * head_dim
                                + ki * num_heads * head_dim
                                + h * head_dim
                                + d;

                            score += query[q_idx] * key[k_idx];
                        }

                        scores[wi] = score * scale;

                        // Apply attention mask if provided
                        if let Some(mask) = attention_mask {
                            let mask_idx = qi * kv_len + ki;
                            if mask_idx < mask.len() {
                                scores[wi] += mask[mask_idx];
                            }
                        }
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

                        for (wi, ki) in (window_start..window_end).enumerate() {
                            let v_idx = b * kv_len * num_heads * head_dim
                                + ki * num_heads * head_dim
                                + h * head_dim
                                + d;

                            out_val += scores[wi] * value[v_idx];
                        }

                        let out_idx = b * q_len * num_heads * head_dim
                            + qi * num_heads * head_dim
                            + h * head_dim
                            + d;

                        output[out_idx] = out_val;
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

    /// Get the window size.
    pub fn window_size(&self) -> usize {
        self.config.window_size
    }

    /// Estimate memory usage.
    pub fn estimate_memory(&self, batch: usize, seq_len: usize) -> usize {
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let window_size = self.config.window_size.min(seq_len);

        // Memory for attention scores within window
        let attention_memory = batch * num_heads * seq_len * window_size * 4; // f32

        // Memory for output
        let output_memory = batch * seq_len * num_heads * head_dim * 4;

        attention_memory + output_memory
    }
}

/// Dilated sliding window attention.
///
/// Similar to sliding window but with dilation factor for
/// longer range dependencies without increasing compute.
pub struct DilatedSlidingWindowAttention {
    config: AttentionConfig,
    dilation: usize,
}

impl DilatedSlidingWindowAttention {
    /// Create new dilated sliding window attention.
    pub fn new(config: AttentionConfig, dilation: usize) -> Self {
        Self { config, dilation }
    }

    /// Compute dilated sliding window attention.
    pub fn forward(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        q_shape: TensorShape,
        kv_shape: TensorShape,
    ) -> Result<AttentionOutput> {
        let batch = q_shape.batch;
        let q_len = q_shape.seq_len;
        let kv_len = kv_shape.seq_len;
        let num_heads = q_shape.num_heads;
        let head_dim = q_shape.head_dim;
        let window_size = self.config.window_size;

        let scale = self.config.scale();
        let mut output = vec![0.0; batch * q_len * num_heads * head_dim];

        for b in 0..batch {
            for h in 0..num_heads {
                for qi in 0..q_len {
                    // Collect positions within dilated window
                    let mut positions = Vec::new();
                    let _effective_window = window_size * self.dilation;

                    for i in 0..window_size {
                        let offset = i * self.dilation;
                        if qi >= offset {
                            let pos = qi - offset;
                            if pos < kv_len {
                                positions.push(pos);
                            }
                        }
                    }

                    if !self.config.causal {
                        for i in 1..window_size {
                            let offset = i * self.dilation;
                            let pos = qi + offset;
                            if pos < kv_len {
                                positions.push(pos);
                            }
                        }
                    }

                    // Compute attention scores
                    let mut scores = vec![0.0; positions.len()];

                    for (pi, &ki) in positions.iter().enumerate() {
                        let mut score = 0.0;

                        for d in 0..head_dim {
                            let q_idx = b * q_len * num_heads * head_dim
                                + qi * num_heads * head_dim
                                + h * head_dim
                                + d;
                            let k_idx = b * kv_len * num_heads * head_dim
                                + ki * num_heads * head_dim
                                + h * head_dim
                                + d;

                            score += query[q_idx] * key[k_idx];
                        }

                        scores[pi] = score * scale;
                    }

                    // Softmax
                    if !scores.is_empty() {
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
                    }

                    // Apply to values
                    for d in 0..head_dim {
                        let mut out_val = 0.0;

                        for (pi, &ki) in positions.iter().enumerate() {
                            let v_idx = b * kv_len * num_heads * head_dim
                                + ki * num_heads * head_dim
                                + h * head_dim
                                + d;

                            out_val += scores[pi] * value[v_idx];
                        }

                        let out_idx = b * q_len * num_heads * head_dim
                            + qi * num_heads * head_dim
                            + h * head_dim
                            + d;

                        output[out_idx] = out_val;
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

    #[test]
    fn test_sliding_window_attention() {
        let config = AttentionConfig::new(2, 4)
            .with_window_size(2)
            .with_causal(true);

        let attention = SlidingWindowAttention::new(config);

        let batch = 1;
        let seq_len = 8;
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
    fn test_memory_estimate() {
        let config = AttentionConfig::new(8, 64).with_window_size(1024);
        let attention = SlidingWindowAttention::new(config);

        let memory = attention.estimate_memory(1, 16384);

        // Should be much less than O(n^2)
        // Full attention would be: 8 * 16384 * 16384 * 4 = ~8GB
        // Window attention: 8 * 16384 * 1024 * 4 = ~512MB
        assert!(memory < 1_000_000_000); // Less than 1GB
    }

    #[test]
    fn test_dilated_sliding_window() {
        let config = AttentionConfig::new(2, 4)
            .with_window_size(3)
            .with_causal(true);

        let attention = DilatedSlidingWindowAttention::new(config, 2);

        let batch = 1;
        let seq_len = 12;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let key = query.clone();
        let value = query.clone();

        let q_shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let kv_shape = q_shape;

        let output = attention
            .forward(&query, &key, &value, q_shape, kv_shape)
            .unwrap();

        assert_eq!(output.output.len(), size);
    }
}
