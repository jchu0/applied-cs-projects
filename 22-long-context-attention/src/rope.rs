//! Rotary Position Embedding (RoPE) implementation.

use crate::config::TensorShape;

/// Rotary position embedding.
pub struct RotaryEmbedding {
    /// Dimension of each head.
    head_dim: usize,
    /// Maximum sequence length.
    max_seq_len: usize,
    /// Base for frequency computation.
    base: f32,
    /// Precomputed cosine values.
    cos_cache: Vec<f32>,
    /// Precomputed sine values.
    sin_cache: Vec<f32>,
}

impl RotaryEmbedding {
    /// Create new rotary embedding.
    pub fn new(head_dim: usize, max_seq_len: usize) -> Self {
        Self::with_base(head_dim, max_seq_len, 10000.0)
    }

    /// Create with custom base.
    pub fn with_base(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        let mut rope = Self {
            head_dim,
            max_seq_len,
            base,
            cos_cache: Vec::new(),
            sin_cache: Vec::new(),
        };
        rope.precompute_cache();
        rope
    }

    /// Precompute sin/cos cache.
    fn precompute_cache(&mut self) {
        let dim = self.head_dim;
        let half_dim = dim / 2;

        // Compute frequencies
        let mut inv_freq = vec![0.0; half_dim];
        for i in 0..half_dim {
            inv_freq[i] = 1.0 / self.base.powf((2 * i) as f32 / dim as f32);
        }

        // Compute position embeddings
        self.cos_cache = vec![0.0; self.max_seq_len * half_dim];
        self.sin_cache = vec![0.0; self.max_seq_len * half_dim];

        for pos in 0..self.max_seq_len {
            for i in 0..half_dim {
                let freq = pos as f32 * inv_freq[i];
                self.cos_cache[pos * half_dim + i] = freq.cos();
                self.sin_cache[pos * half_dim + i] = freq.sin();
            }
        }
    }

    /// Apply rotary embedding to tensor.
    pub fn apply(
        &self,
        x: &mut [f32],
        shape: TensorShape,
        position_ids: Option<&[usize]>,
    ) {
        let batch = shape.batch;
        let seq_len = shape.seq_len;
        let num_heads = shape.num_heads;
        let head_dim = shape.head_dim;
        let half_dim = head_dim / 2;

        for b in 0..batch {
            for s in 0..seq_len {
                let pos = if let Some(ids) = position_ids {
                    ids[b * seq_len + s]
                } else {
                    s
                };

                for h in 0..num_heads {
                    // Apply rotation to pairs of elements
                    for i in 0..half_dim {
                        let idx1 = b * seq_len * num_heads * head_dim
                            + s * num_heads * head_dim
                            + h * head_dim
                            + i;
                        let idx2 = idx1 + half_dim;

                        let x1 = x[idx1];
                        let x2 = x[idx2];

                        let cos = self.cos_cache[pos * half_dim + i];
                        let sin = self.sin_cache[pos * half_dim + i];

                        // Rotation: [cos, -sin; sin, cos] * [x1; x2]
                        x[idx1] = x1 * cos - x2 * sin;
                        x[idx2] = x1 * sin + x2 * cos;
                    }
                }
            }
        }
    }

    /// Get the head dimension.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Get the maximum sequence length.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// ALiBi (Attention with Linear Biases) positional encoding.
pub struct ALiBiPositionalBias {
    /// Number of heads.
    num_heads: usize,
    /// Maximum sequence length.
    _max_seq_len: usize,
    /// Slope for each head.
    slopes: Vec<f32>,
}

impl ALiBiPositionalBias {
    /// Create new ALiBi positional bias.
    pub fn new(num_heads: usize, max_seq_len: usize) -> Self {
        let slopes = Self::compute_slopes(num_heads);
        Self {
            num_heads,
            _max_seq_len: max_seq_len,
            slopes,
        }
    }

    /// Compute slopes for each head.
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        // Slopes are powers of 2^(-8/n) for n heads
        let ratio = 2.0_f32.powf(-8.0 / num_heads as f32);
        let mut slopes = Vec::with_capacity(num_heads);

        for i in 0..num_heads {
            slopes.push(ratio.powi(i as i32 + 1));
        }

        slopes
    }

    /// Compute bias for attention scores.
    pub fn compute_bias(&self, q_len: usize, kv_len: usize) -> Vec<f32> {
        let mut bias = vec![0.0; self.num_heads * q_len * kv_len];

        for h in 0..self.num_heads {
            let slope = self.slopes[h];

            for qi in 0..q_len {
                for ki in 0..kv_len {
                    // Distance between query and key positions
                    let distance = (qi as i32 - ki as i32).abs() as f32;
                    let bias_val = -slope * distance;

                    let idx = h * q_len * kv_len + qi * kv_len + ki;
                    bias[idx] = bias_val;
                }
            }
        }

        bias
    }

    /// Get slopes.
    pub fn slopes(&self) -> &[f32] {
        &self.slopes
    }
}

/// Compute sinusoidal position embedding.
pub fn sinusoidal_position_embedding(position: usize, dim: usize) -> Vec<f32> {
    let mut embedding = vec![0.0; dim];
    let half_dim = dim / 2;

    for i in 0..half_dim {
        let freq = 1.0 / 10000.0_f32.powf((2 * i) as f32 / dim as f32);
        let angle = position as f32 * freq;
        embedding[i] = angle.sin();
        embedding[i + half_dim] = angle.cos();
    }

    embedding
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== RotaryEmbedding Tests ==========

    #[test]
    fn test_rotary_embedding() {
        let rope = RotaryEmbedding::new(64, 1024);
        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_seq_len(), 1024);

        // Test application
        let batch = 1;
        let seq_len = 4;
        let num_heads = 2;
        let head_dim = 64;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let mut x: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let original = x.clone();
        rope.apply(&mut x, shape, None);

        // Values should be modified
        assert!(x.iter().zip(original.iter()).any(|(a, b)| (a - b).abs() > 1e-6));
    }

    #[test]
    fn test_rotary_embedding_new() {
        let rope = RotaryEmbedding::new(128, 2048);
        assert_eq!(rope.head_dim(), 128);
        assert_eq!(rope.max_seq_len(), 2048);

        // Caches should be populated
        let half_dim = 128 / 2;
        assert_eq!(rope.cos_cache.len(), 2048 * half_dim);
        assert_eq!(rope.sin_cache.len(), 2048 * half_dim);
    }

    #[test]
    fn test_rotary_embedding_with_base() {
        let rope = RotaryEmbedding::with_base(64, 512, 50000.0);
        assert_eq!(rope.head_dim(), 64);
        assert_eq!(rope.max_seq_len(), 512);
        assert_eq!(rope.base, 50000.0);
    }

    #[test]
    fn test_rotary_embedding_position_0() {
        let rope = RotaryEmbedding::new(4, 100);

        let batch = 1;
        let seq_len = 1;
        let num_heads = 1;
        let head_dim = 4;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let original = x.clone();

        rope.apply(&mut x, shape, Some(&[0]));

        // At position 0, cos=1, sin=0, so no rotation
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "Position 0 should not rotate");
        }
    }

    #[test]
    fn test_rotary_embedding_different_positions() {
        let rope = RotaryEmbedding::new(8, 256);

        let batch = 1;
        let seq_len = 1;
        let num_heads = 1;
        let head_dim = 8;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut x1 = original.clone();
        let mut x2 = original.clone();

        rope.apply(&mut x1, shape, Some(&[10]));
        rope.apply(&mut x2, shape, Some(&[100]));

        // Different positions should give different results
        let mut different = false;
        for (a, b) in x1.iter().zip(x2.iter()) {
            if (a - b).abs() > 1e-5 {
                different = true;
                break;
            }
        }
        assert!(different, "Different positions should produce different outputs");
    }

    #[test]
    fn test_rotary_embedding_preserves_magnitude() {
        let rope = RotaryEmbedding::new(4, 256);

        let batch = 1;
        let seq_len = 1;
        let num_heads = 1;
        let head_dim = 4;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let mut x: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
        let half_dim = head_dim / 2;

        // Calculate initial magnitude of first pair
        let mag_before = (x[0].powi(2) + x[half_dim].powi(2)).sqrt();

        rope.apply(&mut x, shape, Some(&[10]));

        // Calculate magnitude after rotation
        let mag_after = (x[0].powi(2) + x[half_dim].powi(2)).sqrt();

        assert!(
            (mag_before - mag_after).abs() < 1e-5,
            "Rotation should preserve magnitude"
        );
    }

    #[test]
    fn test_rotary_embedding_batch_processing() {
        let rope = RotaryEmbedding::new(8, 256);

        let batch = 4;
        let seq_len = 8;
        let num_heads = 2;
        let head_dim = 8;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let mut x: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let original = x.clone();
        rope.apply(&mut x, shape, None);

        // Values should be modified
        assert!(x.iter().zip(original.iter()).any(|(a, b)| (a - b).abs() > 1e-6));

        // All values should be finite
        for val in &x {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_rotary_embedding_custom_position_ids() {
        let rope = RotaryEmbedding::new(4, 256);

        let batch = 1;
        let seq_len = 3;
        let num_heads = 1;
        let head_dim = 4;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let mut x: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
            .map(|i| (i as f32) * 0.1)
            .collect();

        // Custom positions: [5, 10, 15] instead of [0, 1, 2]
        let position_ids = vec![5, 10, 15];
        rope.apply(&mut x, shape, Some(&position_ids));

        // All values should be finite
        for val in &x {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_rotary_embedding_numerical_stability() {
        let rope = RotaryEmbedding::new(128, 8192);

        let batch = 1;
        let seq_len = 1;
        let num_heads = 8;
        let head_dim = 128;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let mut x: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
            .map(|i| (i as f32) * 10.0) // Large values
            .collect();

        // Test at high position
        rope.apply(&mut x, shape, Some(&[8000]));

        // All values should be finite
        for val in &x {
            assert!(val.is_finite(), "RoPE produced non-finite value");
        }
    }

    #[test]
    fn test_rotary_embedding_consistency() {
        let rope = RotaryEmbedding::new(8, 256);

        let batch = 1;
        let seq_len = 1;
        let num_heads = 1;
        let head_dim = 8;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Apply to same position twice
        let mut x1 = original.clone();
        let mut x2 = original.clone();

        rope.apply(&mut x1, shape, Some(&[50]));
        rope.apply(&mut x2, shape, Some(&[50]));

        // Results should be identical
        for (a, b) in x1.iter().zip(x2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ========== ALiBiPositionalBias Tests ==========

    #[test]
    fn test_alibi() {
        let alibi = ALiBiPositionalBias::new(8, 1024);
        assert_eq!(alibi.slopes().len(), 8);

        // Compute bias
        let bias = alibi.compute_bias(4, 4);
        assert_eq!(bias.len(), 8 * 4 * 4);

        // Diagonal should be 0 (distance = 0)
        for h in 0..8 {
            for i in 0..4 {
                let idx = h * 16 + i * 4 + i;
                assert_eq!(bias[idx], 0.0);
            }
        }
    }

    #[test]
    fn test_alibi_new() {
        let alibi = ALiBiPositionalBias::new(16, 2048);
        assert_eq!(alibi.num_heads, 16);
        // _max_seq_len is private/unused field
        assert_eq!(alibi.slopes.len(), 16);
    }

    #[test]
    fn test_alibi_slopes_positive() {
        let alibi = ALiBiPositionalBias::new(8, 1024);

        // All slopes should be positive
        for &slope in alibi.slopes() {
            assert!(slope > 0.0, "Slopes should be positive");
        }
    }

    #[test]
    fn test_alibi_slopes_decreasing() {
        let alibi = ALiBiPositionalBias::new(8, 1024);

        // Slopes should be decreasing
        for i in 1..alibi.slopes.len() {
            assert!(
                alibi.slopes[i] < alibi.slopes[i - 1],
                "Slopes should be decreasing"
            );
        }
    }

    #[test]
    fn test_alibi_distance_penalty() {
        let alibi = ALiBiPositionalBias::new(1, 256);
        let bias = alibi.compute_bias(8, 8);

        let slope = alibi.slopes[0];

        // Check specific distances
        // bias[0,0] = 0 (distance 0)
        assert!((bias[0] - 0.0).abs() < 1e-6);

        // bias[0,1] = -slope * 1
        assert!((bias[1] - (-slope)).abs() < 1e-5);

        // bias[0,3] = -slope * 3
        assert!((bias[3] - (-slope * 3.0)).abs() < 1e-5);
    }

    #[test]
    fn test_alibi_increasing_penalty() {
        let alibi = ALiBiPositionalBias::new(1, 256);
        let bias = alibi.compute_bias(16, 16);

        // Penalty should increase (become more negative) with distance
        for j in 0..15 {
            let current = bias[j];
            let next = bias[j + 1];
            assert!(
                current >= next,
                "Penalty should increase with distance: {} vs {}",
                current,
                next
            );
        }
    }

    #[test]
    fn test_alibi_symmetry() {
        let alibi = ALiBiPositionalBias::new(1, 256);
        let bias = alibi.compute_bias(8, 8);

        // bias[i][j] should equal bias[j][i] (symmetric distance)
        for i in 0..8 {
            for j in 0..8 {
                let idx_ij = i * 8 + j;
                let idx_ji = j * 8 + i;
                assert!(
                    (bias[idx_ij] - bias[idx_ji]).abs() < 1e-6,
                    "ALiBi bias should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_alibi_multi_head() {
        let alibi = ALiBiPositionalBias::new(4, 256);
        let bias = alibi.compute_bias(4, 4);

        // Each head should have different biases due to different slopes
        let head0_bias = &bias[0..16];
        let head1_bias = &bias[16..32];

        let mut different = false;
        for (a, b) in head0_bias.iter().zip(head1_bias.iter()) {
            if (a - b).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "Different heads should have different biases");
    }

    #[test]
    fn test_alibi_asymmetric_lengths() {
        let alibi = ALiBiPositionalBias::new(2, 256);
        let bias = alibi.compute_bias(4, 8);

        assert_eq!(bias.len(), 2 * 4 * 8);

        // All values should be finite
        for &b in &bias {
            assert!(b.is_finite());
        }
    }

    #[test]
    fn test_alibi_large_sequence() {
        let alibi = ALiBiPositionalBias::new(8, 4096);
        let bias = alibi.compute_bias(1024, 1024);

        assert_eq!(bias.len(), 8 * 1024 * 1024);

        // Spot check: diagonal should be 0
        for h in 0..8 {
            for i in 0..1024 {
                let idx = h * 1024 * 1024 + i * 1024 + i;
                assert!((bias[idx] - 0.0).abs() < 1e-6);
            }
        }
    }

    // ========== Sinusoidal Embedding Tests ==========

    #[test]
    fn test_sinusoidal_embedding() {
        let embedding = sinusoidal_position_embedding(10, 64);
        assert_eq!(embedding.len(), 64);

        // Check normalization (values should be in [-1, 1])
        for &val in &embedding {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_sinusoidal_embedding_different_positions() {
        let emb1 = sinusoidal_position_embedding(0, 32);
        let emb2 = sinusoidal_position_embedding(100, 32);

        // Different positions should give different embeddings
        let mut different = false;
        for (a, b) in emb1.iter().zip(emb2.iter()) {
            if (a - b).abs() > 1e-6 {
                different = true;
                break;
            }
        }
        assert!(different, "Different positions should have different embeddings");
    }

    #[test]
    fn test_sinusoidal_embedding_different_dimensions() {
        for dim in [16, 32, 64, 128, 256] {
            let embedding = sinusoidal_position_embedding(50, dim);
            assert_eq!(embedding.len(), dim);

            // All values should be valid sin/cos outputs
            for &val in &embedding {
                assert!(val >= -1.0 && val <= 1.0);
            }
        }
    }

    #[test]
    fn test_sinusoidal_embedding_position_0() {
        let embedding = sinusoidal_position_embedding(0, 32);

        // At position 0:
        // sin(0 * freq) = 0 for all frequencies
        // cos(0 * freq) = 1 for all frequencies
        let half_dim = 16;

        for i in 0..half_dim {
            assert!((embedding[i] - 0.0).abs() < 1e-6, "sin(0) should be 0");
            assert!((embedding[i + half_dim] - 1.0).abs() < 1e-6, "cos(0) should be 1");
        }
    }

    #[test]
    fn test_sinusoidal_embedding_consistency() {
        // Same inputs should give same outputs
        let emb1 = sinusoidal_position_embedding(42, 64);
        let emb2 = sinusoidal_position_embedding(42, 64);

        for (a, b) in emb1.iter().zip(emb2.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    // ========== Integration Tests ==========

    #[test]
    fn test_rope_and_alibi_compatible() {
        // Verify RoPE and ALiBi can work together on same data
        let rope = RotaryEmbedding::new(64, 1024);
        let alibi = ALiBiPositionalBias::new(8, 1024);

        let batch = 1;
        let seq_len = 16;
        let num_heads = 8;
        let head_dim = 64;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        let mut x: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
            .map(|i| (i as f32) * 0.01)
            .collect();

        rope.apply(&mut x, shape, None);

        let bias = alibi.compute_bias(seq_len, seq_len);

        // Both should produce valid outputs
        for val in &x {
            assert!(val.is_finite());
        }
        for val in &bias {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_all_embeddings_numerical_stability() {
        // Test all embedding types at edge cases
        let rope = RotaryEmbedding::new(128, 8192);
        let alibi = ALiBiPositionalBias::new(32, 8192);

        let batch = 1;
        let seq_len = 1;
        let num_heads = 32;
        let head_dim = 128;
        let shape = TensorShape::new(batch, seq_len, num_heads, head_dim);

        // Test RoPE at various positions
        for pos in [0, 1, 100, 1000, 8000] {
            let mut x: Vec<f32> = (0..batch * seq_len * num_heads * head_dim)
                .map(|i| (i as f32) * 0.1)
                .collect();

            rope.apply(&mut x, shape, Some(&[pos]));

            for val in &x {
                assert!(val.is_finite(), "Non-finite at position {}", pos);
            }
        }

        // Test ALiBi at various sequence lengths
        for seq_len in [8, 64, 256, 1024] {
            let bias = alibi.compute_bias(seq_len, seq_len);
            for &b in &bias {
                assert!(b.is_finite(), "Non-finite ALiBi at seq_len {}", seq_len);
            }
        }

        // Test sinusoidal at various positions and dimensions
        for pos in [0, 1, 100, 1000] {
            for dim in [32, 64, 128] {
                let emb = sinusoidal_position_embedding(pos, dim);
                for &val in &emb {
                    assert!(val.is_finite());
                    assert!(val >= -1.0 && val <= 1.0);
                }
            }
        }
    }
}
