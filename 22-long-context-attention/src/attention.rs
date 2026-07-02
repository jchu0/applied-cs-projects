//! Core attention implementations.

use crate::config::{AttentionConfig, TensorShape};
use crate::{Error, Result};

/// Attention output with optional metadata.
#[derive(Debug)]
pub struct AttentionOutput {
    /// Output tensor [batch, seq_len, num_heads, head_dim].
    pub output: Vec<f32>,
    /// Output shape.
    pub shape: TensorShape,
    /// Attention weights (optional, for debugging).
    pub attention_weights: Option<Vec<f32>>,
}

/// Validate that query/key/value (and optional mask) buffers match their
/// declared tensor shapes before any indexing occurs.
///
/// This guards the public `forward` entry points: a caller-supplied buffer
/// whose length disagrees with its declared shape would otherwise cause
/// out-of-bounds indexing (a panic, or worse). Instead we return a clean
/// [`Error::ShapeMismatch`] / [`Error::DimensionMismatch`].
///
/// Checks performed:
/// - `query.len() == q_shape.num_elements()`
/// - `key.len() == value.len() == kv_shape.num_elements()`
/// - query and KV share `num_heads` and `head_dim` (per-head invariants)
/// - the mask, if provided, has at least `q_len * kv_len` entries
pub fn validate_forward_inputs(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    q_shape: TensorShape,
    kv_shape: TensorShape,
    attention_mask: Option<&[f32]>,
) -> Result<()> {
    // Per-head invariants: query and KV must agree on head count and dim,
    // otherwise the shared index arithmetic reads out of the other buffer.
    if q_shape.num_heads != kv_shape.num_heads {
        return Err(Error::DimensionMismatch {
            expected: q_shape.num_heads,
            actual: kv_shape.num_heads,
        });
    }
    if q_shape.head_dim != kv_shape.head_dim {
        return Err(Error::DimensionMismatch {
            expected: q_shape.head_dim,
            actual: kv_shape.head_dim,
        });
    }
    if q_shape.batch != kv_shape.batch {
        return Err(Error::DimensionMismatch {
            expected: q_shape.batch,
            actual: kv_shape.batch,
        });
    }

    let q_expected = q_shape.num_elements();
    if query.len() != q_expected {
        return Err(Error::ShapeMismatch {
            name: "query",
            expected: q_expected,
            actual: query.len(),
        });
    }

    let kv_expected = kv_shape.num_elements();
    if key.len() != kv_expected {
        return Err(Error::ShapeMismatch {
            name: "key",
            expected: kv_expected,
            actual: key.len(),
        });
    }
    if value.len() != kv_expected {
        return Err(Error::ShapeMismatch {
            name: "value",
            expected: kv_expected,
            actual: value.len(),
        });
    }

    if let Some(mask) = attention_mask {
        let mask_expected = q_shape.seq_len * kv_shape.seq_len;
        if mask.len() < mask_expected {
            return Err(Error::ShapeMismatch {
                name: "attention_mask",
                expected: mask_expected,
                actual: mask.len(),
            });
        }
    }

    Ok(())
}

/// Standard O(n^2) attention implementation.
pub struct StandardAttention {
    config: AttentionConfig,
}

impl StandardAttention {
    /// Create new standard attention.
    pub fn new(config: AttentionConfig) -> Self {
        Self { config }
    }

    /// Compute attention.
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

        // Validate buffer lengths against declared shapes before indexing.
        validate_forward_inputs(query, key, value, q_shape, kv_shape, attention_mask)?;

        let scale = self.config.scale();
        let mut output = vec![0.0; batch * q_len * num_heads * head_dim];

        // Process each batch and head
        for b in 0..batch {
            for h in 0..num_heads {
                // Compute attention scores: [q_len, kv_len]
                let mut scores = vec![0.0; q_len * kv_len];

                for qi in 0..q_len {
                    for ki in 0..kv_len {
                        let mut score = 0.0;

                        // Dot product of query and key
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

                        scores[qi * kv_len + ki] = score * scale;
                    }
                }

                // Apply causal mask
                if self.config.causal {
                    for qi in 0..q_len {
                        for ki in 0..kv_len {
                            // For causal, position qi can only attend to positions <= qi
                            // In prefill, kv_len == q_len, so this is qi >= ki
                            // In decode, q_len=1, kv_len=past+1, so always valid
                            if ki > qi + (kv_len - q_len) {
                                scores[qi * kv_len + ki] = f32::NEG_INFINITY;
                            }
                        }
                    }
                }

                // Apply attention mask if provided
                if let Some(mask) = attention_mask {
                    for qi in 0..q_len {
                        for ki in 0..kv_len {
                            let mask_idx = qi * kv_len + ki;
                            if mask_idx < mask.len() {
                                scores[qi * kv_len + ki] += mask[mask_idx];
                            }
                        }
                    }
                }

                // Softmax over keys for each query
                for qi in 0..q_len {
                    // Find max for numerical stability
                    let mut max_score = f32::NEG_INFINITY;
                    for ki in 0..kv_len {
                        max_score = max_score.max(scores[qi * kv_len + ki]);
                    }

                    // Compute exp and sum
                    let mut sum = 0.0;
                    for ki in 0..kv_len {
                        let idx = qi * kv_len + ki;
                        scores[idx] = (scores[idx] - max_score).exp();
                        sum += scores[idx];
                    }

                    // Normalize
                    if sum > 0.0 {
                        for ki in 0..kv_len {
                            scores[qi * kv_len + ki] /= sum;
                        }
                    }
                }

                // Apply attention to values: output[qi] = sum_ki(attn[qi,ki] * value[ki])
                for qi in 0..q_len {
                    for d in 0..head_dim {
                        let mut out_val = 0.0;

                        for ki in 0..kv_len {
                            let attn_weight = scores[qi * kv_len + ki];
                            let v_idx = b * kv_len * num_heads * head_dim
                                + ki * num_heads * head_dim
                                + h * head_dim
                                + d;

                            out_val += attn_weight * value[v_idx];
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

/// Expand KV heads for grouped query attention.
pub fn expand_kv_heads(
    kv: &[f32],
    kv_shape: TensorShape,
    target_num_heads: usize,
) -> (Vec<f32>, TensorShape) {
    let batch = kv_shape.batch;
    let seq_len = kv_shape.seq_len;
    let num_kv_heads = kv_shape.num_heads;
    let head_dim = kv_shape.head_dim;

    if num_kv_heads == target_num_heads {
        return (kv.to_vec(), kv_shape);
    }

    let num_groups = target_num_heads / num_kv_heads;
    let mut expanded = vec![0.0; batch * seq_len * target_num_heads * head_dim];

    for b in 0..batch {
        for s in 0..seq_len {
            for kv_h in 0..num_kv_heads {
                for g in 0..num_groups {
                    let target_h = kv_h * num_groups + g;

                    for d in 0..head_dim {
                        let src_idx = b * seq_len * num_kv_heads * head_dim
                            + s * num_kv_heads * head_dim
                            + kv_h * head_dim
                            + d;

                        let dst_idx = b * seq_len * target_num_heads * head_dim
                            + s * target_num_heads * head_dim
                            + target_h * head_dim
                            + d;

                        expanded[dst_idx] = kv[src_idx];
                    }
                }
            }
        }
    }

    let new_shape = TensorShape::new(batch, seq_len, target_num_heads, head_dim);
    (expanded, new_shape)
}

/// Compute softmax along the last dimension.
pub fn softmax(scores: &mut [f32], num_rows: usize, num_cols: usize) {
    for row in 0..num_rows {
        let start = row * num_cols;
        let end = start + num_cols;
        let row_slice = &mut scores[start..end];

        // Find max
        let max = row_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Exp and sum
        let mut sum = 0.0;
        for val in row_slice.iter_mut() {
            *val = (*val - max).exp();
            sum += *val;
        }

        // Normalize
        if sum > 0.0 {
            for val in row_slice.iter_mut() {
                *val /= sum;
            }
        }
    }
}

/// Apply causal mask to attention scores.
pub fn apply_causal_mask(scores: &mut [f32], q_len: usize, kv_len: usize) {
    for qi in 0..q_len {
        for ki in 0..kv_len {
            // qi can attend to ki if ki <= qi + offset
            // offset = kv_len - q_len (handles prefill vs decode)
            if ki > qi + (kv_len - q_len) {
                scores[qi * kv_len + ki] = f32::NEG_INFINITY;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_attention() {
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let attention = StandardAttention::new(config);

        // Simple test: batch=1, seq_len=3, num_heads=2, head_dim=4
        let batch = 1;
        let seq_len = 3;
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
        assert_eq!(output.shape.seq_len, seq_len);
    }

    #[test]
    fn test_standard_attention_output_shape() {
        let config = AttentionConfig::new(4, 8).with_causal(false);
        let attention = StandardAttention::new(config);

        let batch = 2;
        let q_len = 5;
        let kv_len = 7;
        let num_heads = 4;
        let head_dim = 8;

        let q_size = batch * q_len * num_heads * head_dim;
        let kv_size = batch * kv_len * num_heads * head_dim;

        let query: Vec<f32> = (0..q_size).map(|i| (i as f32) * 0.01).collect();
        let key: Vec<f32> = (0..kv_size).map(|i| (i as f32) * 0.01).collect();
        let value: Vec<f32> = (0..kv_size).map(|i| (i as f32) * 0.01).collect();

        let q_shape = TensorShape::new(batch, q_len, num_heads, head_dim);
        let kv_shape = TensorShape::new(batch, kv_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, q_shape, kv_shape, None)
            .unwrap();

        assert_eq!(output.output.len(), q_size);
        assert_eq!(output.shape.batch, batch);
        assert_eq!(output.shape.seq_len, q_len);
        assert_eq!(output.shape.num_heads, num_heads);
        assert_eq!(output.shape.head_dim, head_dim);
    }

    #[test]
    fn test_standard_attention_bidirectional() {
        let config = AttentionConfig::new(2, 4).with_causal(false);
        let attention = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 4;
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

        // With uniform inputs in bidirectional attention, output should be uniform
        for val in &output.output {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_standard_attention_causal_masking() {
        let config = AttentionConfig::new(1, 4).with_causal(true);
        let attention = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 4;
        let num_heads = 1;
        let head_dim = 4;

        // Create distinct values for each position
        let mut query = vec![0.0; batch * seq_len * num_heads * head_dim];
        let mut key = vec![0.0; batch * seq_len * num_heads * head_dim];
        let mut value = vec![0.0; batch * seq_len * num_heads * head_dim];

        for s in 0..seq_len {
            for d in 0..head_dim {
                let idx = s * num_heads * head_dim + d;
                query[idx] = 0.1;
                key[idx] = 0.1;
                value[idx] = s as f32; // Each position has distinct value
            }
        }

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        // First position can only see itself, should have value 0
        let first_pos_val = output.output[0];
        assert!((first_pos_val - 0.0).abs() < 0.1);

        // Last position can see all, should have weighted average > 0
        let last_pos_val = output.output[(seq_len - 1) * num_heads * head_dim];
        assert!(last_pos_val > 0.0);
    }

    #[test]
    fn test_standard_attention_with_mask() {
        let config = AttentionConfig::new(2, 4).with_causal(false);
        let attention = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 3;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();
        let key = query.clone();
        let value = query.clone();

        // Mask that blocks attention to position 2
        let mut mask = vec![0.0; seq_len * seq_len];
        mask[2] = f32::NEG_INFINITY;
        mask[5] = f32::NEG_INFINITY;
        mask[8] = f32::NEG_INFINITY;

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, Some(&mask))
            .unwrap();

        assert_eq!(output.output.len(), size);
    }

    #[test]
    fn test_standard_attention_dimension_mismatch() {
        let config = AttentionConfig::new(2, 4);
        let attention = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 3;
        let num_heads = 2;
        let head_dim = 4;

        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let key = query.clone();
        let value = query.clone();

        let q_shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let kv_shape = TensorShape::new(batch, seq_len, 4, head_dim); // Wrong num_heads

        let result = attention.forward(&query, &key, &value, q_shape, kv_shape, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_standard_attention_numerical_stability() {
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let attention = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 8;
        let num_heads = 2;
        let head_dim = 4;

        // Large values to test numerical stability
        let size = batch * seq_len * num_heads * head_dim;
        let query: Vec<f32> = (0..size).map(|i| (i as f32) * 10.0).collect();
        let key = query.clone();
        let value: Vec<f32> = (0..size).map(|i| (i as f32) * 0.1).collect();

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let output = attention
            .forward(&query, &key, &value, shape, shape, None)
            .unwrap();

        // Check no NaN or Inf in output
        for val in &output.output {
            assert!(val.is_finite(), "Output contains non-finite values");
        }
    }

    #[test]
    fn test_standard_attention_batch_processing() {
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let attention = StandardAttention::new(config);

        let batch = 4;
        let seq_len = 5;
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
    fn test_expand_kv_heads() {
        let batch = 1;
        let seq_len = 4;
        let num_kv_heads = 2;
        let head_dim = 8;

        let kv: Vec<f32> = (0..batch * seq_len * num_kv_heads * head_dim)
            .map(|i| i as f32)
            .collect();

        let kv_shape = TensorShape::new(batch, seq_len, num_kv_heads, head_dim);
        let (expanded, new_shape) = expand_kv_heads(&kv, kv_shape, 8);

        assert_eq!(new_shape.num_heads, 8);
        assert_eq!(expanded.len(), batch * seq_len * 8 * head_dim);
    }

    #[test]
    fn test_expand_kv_heads_no_expansion() {
        let batch = 2;
        let seq_len = 4;
        let num_heads = 4;
        let head_dim = 8;

        let kv: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
            .map(|i| i as f32)
            .collect();

        let kv_shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let (expanded, new_shape) = expand_kv_heads(&kv, kv_shape, num_heads);

        // Should be identical when no expansion needed
        assert_eq!(new_shape.num_heads, num_heads);
        assert_eq!(expanded, kv);
    }

    #[test]
    fn test_expand_kv_heads_correctness() {
        let batch = 1;
        let seq_len = 2;
        let num_kv_heads = 2;
        let target_heads = 4;
        let head_dim = 2;

        let kv: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]; // 1*2*2*2 = 8

        let kv_shape = TensorShape::new(batch, seq_len, num_kv_heads, head_dim);
        let (expanded, new_shape) = expand_kv_heads(&kv, kv_shape, target_heads);

        assert_eq!(new_shape.num_heads, 4);
        // Each KV head should be duplicated twice
        // Head 0 duplicated to heads 0,1; Head 1 duplicated to heads 2,3
        assert_eq!(expanded.len(), batch * seq_len * target_heads * head_dim);
    }

    #[test]
    fn test_softmax() {
        let mut scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        softmax(&mut scores, 2, 3);

        // Check rows sum to 1
        let sum1: f32 = scores[0..3].iter().sum();
        let sum2: f32 = scores[3..6].iter().sum();

        assert!((sum1 - 1.0).abs() < 1e-5);
        assert!((sum2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_uniform_input() {
        let mut scores = vec![1.0, 1.0, 1.0, 1.0];
        softmax(&mut scores, 1, 4);

        // Uniform input should give uniform output
        for score in &scores {
            assert!((score - 0.25).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Test with large values that could cause overflow without proper handling
        let mut scores = vec![1000.0, 1001.0, 1002.0];
        softmax(&mut scores, 1, 3);

        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // All values should be finite
        for score in &scores {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_softmax_negative_infinity_handling() {
        let mut scores = vec![1.0, f32::NEG_INFINITY, 2.0];
        softmax(&mut scores, 1, 3);

        // Position with -inf should be 0
        assert!((scores[1] - 0.0).abs() < 1e-10);
        // Other positions should sum to 1
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_causal_mask() {
        let mut scores = vec![0.0; 9]; // 3x3
        apply_causal_mask(&mut scores, 3, 3);

        // Upper triangle should be -inf (causal: qi can't attend to ki > qi)
        assert!(scores[1].is_infinite()); // [0,1] masked
        assert!(scores[2].is_infinite()); // [0,2] masked
        assert!(scores[5].is_infinite()); // [1,2] masked
        assert!(scores[0].is_finite()); // [0,0] on the diagonal, kept
    }

    #[test]
    fn test_causal_mask_diagonal() {
        let mut scores = vec![1.0; 16]; // 4x4
        apply_causal_mask(&mut scores, 4, 4);

        // Diagonal and below should be preserved
        assert!((scores[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((scores[5] - 1.0).abs() < 1e-5); // [1,1]
        assert!((scores[10] - 1.0).abs() < 1e-5); // [2,2]
        assert!((scores[15] - 1.0).abs() < 1e-5); // [3,3]

        // Above diagonal should be -inf
        assert!(scores[1].is_infinite()); // [0,1]
        assert!(scores[2].is_infinite()); // [0,2]
        assert!(scores[3].is_infinite()); // [0,3]
    }

    #[test]
    fn test_causal_mask_asymmetric() {
        // q_len=2, kv_len=4 (decode scenario with past)
        let mut scores = vec![1.0; 8]; // 2x4
        apply_causal_mask(&mut scores, 2, 4);

        // First query can see positions 0,1,2 (offset = 4-2=2, so ki <= 0+2)
        assert!((scores[0] - 1.0).abs() < 1e-5); // [0,0]
        assert!((scores[1] - 1.0).abs() < 1e-5); // [0,1]
        assert!((scores[2] - 1.0).abs() < 1e-5); // [0,2]
        assert!(scores[3].is_infinite()); // [0,3]

        // Second query can see all
        assert!((scores[4] - 1.0).abs() < 1e-5); // [1,0]
        assert!((scores[5] - 1.0).abs() < 1e-5); // [1,1]
        assert!((scores[6] - 1.0).abs() < 1e-5); // [1,2]
        assert!((scores[7] - 1.0).abs() < 1e-5); // [1,3]
    }

    #[test]
    fn test_standard_attention_short_query_buffer_errors() {
        // A query buffer shorter than its declared shape must return Err,
        // not panic on out-of-bounds indexing.
        let config = AttentionConfig::new(2, 4);
        let attention = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 4;
        let size = batch * seq_len * num_heads * head_dim;

        // Deliberately too short.
        let query: Vec<f32> = vec![0.1; size - 1];
        let key: Vec<f32> = vec![0.1; size];
        let value: Vec<f32> = vec![0.1; size];

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let result = attention.forward(&query, &key, &value, shape, shape, None);

        assert!(matches!(result, Err(Error::ShapeMismatch { name: "query", .. })));
    }

    #[test]
    fn test_standard_attention_short_kv_buffer_errors() {
        let config = AttentionConfig::new(2, 4);
        let attention = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 4;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = vec![0.1; size];
        let key: Vec<f32> = vec![0.1; size - 2]; // too short
        let value: Vec<f32> = vec![0.1; size];

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let result = attention.forward(&query, &key, &value, shape, shape, None);

        assert!(matches!(result, Err(Error::ShapeMismatch { name: "key", .. })));
    }

    #[test]
    fn test_standard_attention_short_mask_errors() {
        let config = AttentionConfig::new(2, 4).with_causal(false);
        let attention = StandardAttention::new(config);

        let batch = 1;
        let seq_len = 3;
        let num_heads = 2;
        let head_dim = 4;
        let size = batch * seq_len * num_heads * head_dim;

        let query: Vec<f32> = vec![0.1; size];
        let key = query.clone();
        let value = query.clone();
        let mask = vec![0.0; seq_len * seq_len - 1]; // too short

        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);
        let result = attention.forward(&query, &key, &value, shape, shape, Some(&mask));

        assert!(matches!(
            result,
            Err(Error::ShapeMismatch { name: "attention_mask", .. })
        ));
    }

    #[test]
    fn test_standard_attention_valid_input_still_ok() {
        // Ensure validation does not reject correctly-sized inputs.
        let config = AttentionConfig::new(2, 4).with_causal(true);
        let attention = StandardAttention::new(config);

        let batch = 2;
        let seq_len = 5;
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
    fn test_validate_forward_inputs_head_dim_mismatch() {
        let q_shape = TensorShape::new(1, 4, 2, 4);
        let kv_shape = TensorShape::new(1, 4, 2, 8); // head_dim differs
        let q = vec![0.0; q_shape.num_elements()];
        let kv = vec![0.0; kv_shape.num_elements()];

        let result = validate_forward_inputs(&q, &kv, &kv, q_shape, kv_shape, None);
        assert!(matches!(result, Err(Error::DimensionMismatch { .. })));
    }

    #[test]
    fn test_attention_output_structure() {
        let output = AttentionOutput {
            output: vec![1.0, 2.0, 3.0],
            shape: TensorShape::new(1, 1, 1, 3),
            attention_weights: Some(vec![0.5, 0.5]),
        };

        assert_eq!(output.output.len(), 3);
        assert!(output.attention_weights.is_some());
    }
}
