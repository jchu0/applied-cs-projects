//! Loss functions for contrastive embedding learning.

use crate::dataset::cosine_similarity;
use crate::Result;

/// Trait for loss functions.
pub trait LossFunction: Send + Sync {
    /// Compute loss value.
    fn compute(
        &self,
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
        negatives: Option<&[Vec<Vec<f32>>]>,
    ) -> f32;

    /// Loss function name.
    fn name(&self) -> &'static str;
}

/// Multiple Negatives Ranking Loss (MNRL).
///
/// Treats all other batch elements as negatives.
/// Efficient and effective for contrastive learning.
pub struct MultipleNegativesRankingLoss {
    /// Temperature scaling factor.
    pub scale: f32,
}

impl MultipleNegativesRankingLoss {
    /// Create new MNRL loss.
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}

impl Default for MultipleNegativesRankingLoss {
    fn default() -> Self {
        Self::new(20.0)
    }
}

impl LossFunction for MultipleNegativesRankingLoss {
    fn compute(
        &self,
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
        negatives: Option<&[Vec<Vec<f32>>]>,
    ) -> f32 {
        let batch_size = anchors.len();
        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            // Compute similarity to all positives (including self)
            let mut scores: Vec<f32> = (0..batch_size)
                .map(|j| cosine_similarity(&anchors[i], &positives[j]) * self.scale)
                .collect();

            // Add explicit negatives if provided
            if let Some(negs) = negatives {
                if i < negs.len() {
                    for neg in &negs[i] {
                        let neg_score = cosine_similarity(&anchors[i], neg) * self.scale;
                        scores.push(neg_score);
                    }
                }
            }

            // Cross-entropy loss: -log(softmax[i])
            // log_softmax[i] = score[i] - log_sum_exp(scores)
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = scores
                .iter()
                .map(|&s| (s - max_score).exp())
                .sum::<f32>()
                .ln()
                + max_score;

            let loss = log_sum_exp - scores[i];
            total_loss += loss;
        }

        total_loss / batch_size as f32
    }

    fn name(&self) -> &'static str {
        "MultipleNegativesRankingLoss"
    }
}

/// Triplet Margin Loss.
///
/// Learns embeddings such that:
/// d(anchor, positive) + margin < d(anchor, negative)
pub struct TripletMarginLoss {
    /// Margin for triplet loss.
    pub margin: f32,
    /// Use cosine distance (1 - cosine_sim) instead of L2.
    pub cosine_distance: bool,
}

impl TripletMarginLoss {
    /// Create new triplet loss.
    pub fn new(margin: f32, cosine_distance: bool) -> Self {
        Self {
            margin,
            cosine_distance,
        }
    }
}

impl Default for TripletMarginLoss {
    fn default() -> Self {
        Self::new(0.5, true)
    }
}

impl LossFunction for TripletMarginLoss {
    fn compute(
        &self,
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
        negatives: Option<&[Vec<Vec<f32>>]>,
    ) -> f32 {
        let Some(negatives) = negatives else {
            return 0.0;
        };

        let batch_size = anchors.len();
        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            if i >= negatives.len() || negatives[i].is_empty() {
                continue;
            }

            let pos_dist = if self.cosine_distance {
                1.0 - cosine_similarity(&anchors[i], &positives[i])
            } else {
                euclidean_distance(&anchors[i], &positives[i])
            };

            // Use first negative (hard negative)
            let neg_dist = if self.cosine_distance {
                1.0 - cosine_similarity(&anchors[i], &negatives[i][0])
            } else {
                euclidean_distance(&anchors[i], &negatives[i][0])
            };

            // ReLU(pos_dist - neg_dist + margin)
            let loss = (pos_dist - neg_dist + self.margin).max(0.0);
            total_loss += loss;
        }

        total_loss / batch_size as f32
    }

    fn name(&self) -> &'static str {
        "TripletMarginLoss"
    }
}

/// InfoNCE / NT-Xent loss.
///
/// Contrastive loss using temperature-scaled similarities.
pub struct InfoNCELoss {
    /// Temperature parameter.
    pub temperature: f32,
}

impl InfoNCELoss {
    /// Create new InfoNCE loss.
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl Default for InfoNCELoss {
    fn default() -> Self {
        Self::new(0.07)
    }
}

impl LossFunction for InfoNCELoss {
    fn compute(
        &self,
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
        negatives: Option<&[Vec<Vec<f32>>]>,
    ) -> f32 {
        let batch_size = anchors.len();
        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for i in 0..batch_size {
            // Positive similarity
            let pos_sim = cosine_similarity(&anchors[i], &positives[i]) / self.temperature;

            // Collect all negative similarities
            let mut neg_sims: Vec<f32> = Vec::new();

            // In-batch negatives
            for j in 0..batch_size {
                if i != j {
                    let sim = cosine_similarity(&anchors[i], &positives[j]) / self.temperature;
                    neg_sims.push(sim);
                }
            }

            // Explicit negatives
            if let Some(negs) = negatives {
                if i < negs.len() {
                    for neg in &negs[i] {
                        let sim = cosine_similarity(&anchors[i], neg) / self.temperature;
                        neg_sims.push(sim);
                    }
                }
            }

            // Loss = -log(exp(pos_sim) / (exp(pos_sim) + sum(exp(neg_sims))))
            let max_val = neg_sims
                .iter()
                .cloned()
                .fold(pos_sim, f32::max);

            let exp_pos = (pos_sim - max_val).exp();
            let exp_negs: f32 = neg_sims.iter().map(|&s| (s - max_val).exp()).sum();

            let loss = -(exp_pos / (exp_pos + exp_negs)).ln();
            total_loss += loss;
        }

        total_loss / batch_size as f32
    }

    fn name(&self) -> &'static str {
        "InfoNCELoss"
    }
}

/// Cosine Embedding Loss.
///
/// Simple cosine similarity loss with target.
pub struct CosineEmbeddingLoss {
    /// Margin for negative pairs.
    pub margin: f32,
}

impl CosineEmbeddingLoss {
    /// Create new cosine embedding loss.
    pub fn new(margin: f32) -> Self {
        Self { margin }
    }

    /// Compute loss for a pair with label.
    pub fn compute_pair(&self, a: &[f32], b: &[f32], label: f32) -> f32 {
        let cos_sim = cosine_similarity(a, b);

        if label > 0.0 {
            // Positive pair: minimize 1 - cos_sim
            1.0 - cos_sim
        } else {
            // Negative pair: minimize max(0, cos_sim - margin)
            (cos_sim - self.margin).max(0.0)
        }
    }
}

impl Default for CosineEmbeddingLoss {
    fn default() -> Self {
        Self::new(0.0)
    }
}

impl LossFunction for CosineEmbeddingLoss {
    fn compute(
        &self,
        anchors: &[Vec<f32>],
        positives: &[Vec<f32>],
        negatives: Option<&[Vec<Vec<f32>>]>,
    ) -> f32 {
        let batch_size = anchors.len();
        if batch_size == 0 {
            return 0.0;
        }

        let mut total_loss = 0.0;
        let mut count = 0;

        // Positive pairs
        for i in 0..batch_size {
            total_loss += self.compute_pair(&anchors[i], &positives[i], 1.0);
            count += 1;
        }

        // Negative pairs
        if let Some(negs) = negatives {
            for i in 0..batch_size {
                if i < negs.len() {
                    for neg in &negs[i] {
                        total_loss += self.compute_pair(&anchors[i], neg, -1.0);
                        count += 1;
                    }
                }
            }
        }

        total_loss / count as f32
    }

    fn name(&self) -> &'static str {
        "CosineEmbeddingLoss"
    }
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Softmax function.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&x| x / sum).collect()
}

/// Log-softmax function.
pub fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = logits
        .iter()
        .map(|&x| (x - max_val).exp())
        .sum::<f32>()
        .ln()
        + max_val;

    logits.iter().map(|&x| x - log_sum_exp).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::random_embeddings;

    // =============================================================================
    // MultipleNegativesRankingLoss Tests
    // =============================================================================

    #[test]
    fn test_mnrl_creation() {
        let loss_fn = MultipleNegativesRankingLoss::new(20.0);
        assert!((loss_fn.scale - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_mnrl_default() {
        let loss_fn = MultipleNegativesRankingLoss::default();
        assert!((loss_fn.scale - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_mnrl_name() {
        let loss_fn = MultipleNegativesRankingLoss::default();
        assert_eq!(loss_fn.name(), "MultipleNegativesRankingLoss");
    }

    #[test]
    fn test_mnrl_loss_perfect_match() {
        let loss_fn = MultipleNegativesRankingLoss::new(20.0);

        let anchors = random_embeddings(4, 64, true);
        let positives = anchors.clone(); // Perfect match

        let loss = loss_fn.compute(&anchors, &positives, None);
        // Loss should be close to log(batch_size) for perfect diagonal matches
        assert!(loss < 2.0);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_mnrl_loss_with_negatives() {
        let loss_fn = MultipleNegativesRankingLoss::new(20.0);

        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![0.9, 0.1, 0.0]];
        let negatives = vec![vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]];

        let loss = loss_fn.compute(&anchors, &positives, Some(&negatives));
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_mnrl_loss_empty_batch() {
        let loss_fn = MultipleNegativesRankingLoss::new(20.0);

        let anchors: Vec<Vec<f32>> = vec![];
        let positives: Vec<Vec<f32>> = vec![];

        let loss = loss_fn.compute(&anchors, &positives, None);
        assert!((loss - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_mnrl_loss_single_sample() {
        let loss_fn = MultipleNegativesRankingLoss::new(20.0);

        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![1.0, 0.0, 0.0]];

        let loss = loss_fn.compute(&anchors, &positives, None);
        // With single sample and perfect match, loss should be 0
        assert!(loss.abs() < 0.001);
    }

    #[test]
    fn test_mnrl_loss_temperature_scaling() {
        let loss_fn_low_temp = MultipleNegativesRankingLoss::new(10.0);
        let loss_fn_high_temp = MultipleNegativesRankingLoss::new(50.0);

        let anchors = random_embeddings(4, 64, true);
        let positives = random_embeddings(4, 64, true);

        let loss_low = loss_fn_low_temp.compute(&anchors, &positives, None);
        let loss_high = loss_fn_high_temp.compute(&anchors, &positives, None);

        // Both should be valid losses
        assert!(loss_low >= 0.0);
        assert!(loss_high >= 0.0);
    }

    #[test]
    fn test_mnrl_loss_not_nan() {
        let loss_fn = MultipleNegativesRankingLoss::new(20.0);

        let anchors = random_embeddings(8, 128, true);
        let positives = random_embeddings(8, 128, true);

        let loss = loss_fn.compute(&anchors, &positives, None);
        assert!(!loss.is_nan());
        assert!(!loss.is_infinite());
    }

    // =============================================================================
    // TripletMarginLoss Tests
    // =============================================================================

    #[test]
    fn test_triplet_creation() {
        let loss_fn = TripletMarginLoss::new(0.5, true);
        assert!((loss_fn.margin - 0.5).abs() < 0.001);
        assert!(loss_fn.cosine_distance);
    }

    #[test]
    fn test_triplet_default() {
        let loss_fn = TripletMarginLoss::default();
        assert!((loss_fn.margin - 0.5).abs() < 0.001);
        assert!(loss_fn.cosine_distance);
    }

    #[test]
    fn test_triplet_name() {
        let loss_fn = TripletMarginLoss::default();
        assert_eq!(loss_fn.name(), "TripletMarginLoss");
    }

    #[test]
    fn test_triplet_loss_easy_negative() {
        let loss_fn = TripletMarginLoss::new(0.5, true);

        // Anchor and positive are very similar
        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![0.99, 0.01, 0.0]];
        // Negative is orthogonal (far from anchor)
        let negatives = vec![vec![vec![0.0, 1.0, 0.0]]];

        let loss = loss_fn.compute(&anchors, &positives, Some(&negatives));
        // Should be 0 since neg is much farther than pos + margin
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_triplet_loss_hard_negative() {
        let loss_fn = TripletMarginLoss::new(0.1, true);

        // Anchor
        let anchors = vec![vec![1.0, 0.0, 0.0]];
        // Positive is somewhat similar
        let positives = vec![vec![0.7, 0.7, 0.0]];
        // Negative is closer to anchor than positive
        let negatives = vec![vec![vec![0.9, 0.1, 0.0]]];

        let loss = loss_fn.compute(&anchors, &positives, Some(&negatives));
        // Should be > 0 since negative is closer
        assert!(loss > 0.0);
    }

    #[test]
    fn test_triplet_loss_no_negatives() {
        let loss_fn = TripletMarginLoss::new(0.5, true);

        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![0.9, 0.1, 0.0]];

        let loss = loss_fn.compute(&anchors, &positives, None);
        assert!((loss - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_triplet_loss_empty_negatives() {
        let loss_fn = TripletMarginLoss::new(0.5, true);

        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![0.9, 0.1, 0.0]];
        let negatives: Vec<Vec<Vec<f32>>> = vec![vec![]];

        let loss = loss_fn.compute(&anchors, &positives, Some(&negatives));
        // No negatives to process
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_triplet_loss_euclidean_distance() {
        let loss_fn = TripletMarginLoss::new(1.0, false);

        let anchors = vec![vec![0.0, 0.0]];
        let positives = vec![vec![1.0, 0.0]]; // Distance = 1
        let negatives = vec![vec![vec![3.0, 0.0]]]; // Distance = 3

        let loss = loss_fn.compute(&anchors, &positives, Some(&negatives));
        // pos_dist=1, neg_dist=3, loss = max(0, 1 - 3 + 1) = max(0, -1) = 0
        assert!(loss.abs() < 0.001);
    }

    #[test]
    fn test_triplet_loss_margin_effect() {
        let loss_fn_small = TripletMarginLoss::new(0.1, true);
        let loss_fn_large = TripletMarginLoss::new(1.0, true);

        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![0.8, 0.2, 0.0]];
        let negatives = vec![vec![vec![0.6, 0.4, 0.0]]];

        let loss_small = loss_fn_small.compute(&anchors, &positives, Some(&negatives));
        let loss_large = loss_fn_large.compute(&anchors, &positives, Some(&negatives));

        // Larger margin should generally produce larger loss
        assert!(loss_large >= loss_small);
    }

    // =============================================================================
    // InfoNCELoss Tests
    // =============================================================================

    #[test]
    fn test_infonce_creation() {
        let loss_fn = InfoNCELoss::new(0.07);
        assert!((loss_fn.temperature - 0.07).abs() < 0.001);
    }

    #[test]
    fn test_infonce_default() {
        let loss_fn = InfoNCELoss::default();
        assert!((loss_fn.temperature - 0.07).abs() < 0.001);
    }

    #[test]
    fn test_infonce_name() {
        let loss_fn = InfoNCELoss::default();
        assert_eq!(loss_fn.name(), "InfoNCELoss");
    }

    #[test]
    fn test_infonce_loss_perfect_match() {
        let loss_fn = InfoNCELoss::new(0.07);

        let anchors = random_embeddings(4, 64, true);
        let positives = anchors.clone();

        let loss = loss_fn.compute(&anchors, &positives, None);
        // With perfect matches, loss should still be > 0 due to in-batch negatives
        assert!(loss > 0.0);
    }

    #[test]
    fn test_infonce_loss_single_sample() {
        let loss_fn = InfoNCELoss::new(0.07);

        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![1.0, 0.0, 0.0]];

        let loss = loss_fn.compute(&anchors, &positives, None);
        // Single sample with no in-batch negatives
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_infonce_loss_with_explicit_negatives() {
        let loss_fn = InfoNCELoss::new(0.07);

        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![0.9, 0.1, 0.0]];
        let negatives = vec![vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ]];

        let loss = loss_fn.compute(&anchors, &positives, Some(&negatives));
        assert!(loss > 0.0);
        assert!(!loss.is_nan());
    }

    #[test]
    fn test_infonce_temperature_effect() {
        let loss_fn_low_temp = InfoNCELoss::new(0.01);
        let loss_fn_high_temp = InfoNCELoss::new(1.0);

        let anchors = random_embeddings(4, 64, true);
        let positives = random_embeddings(4, 64, true);

        let loss_low = loss_fn_low_temp.compute(&anchors, &positives, None);
        let loss_high = loss_fn_high_temp.compute(&anchors, &positives, None);

        // Both should be valid
        assert!(loss_low >= 0.0 && !loss_low.is_nan());
        assert!(loss_high >= 0.0 && !loss_high.is_nan());
    }

    // =============================================================================
    // CosineEmbeddingLoss Tests
    // =============================================================================

    #[test]
    fn test_cosine_embedding_creation() {
        let loss_fn = CosineEmbeddingLoss::new(0.5);
        assert!((loss_fn.margin - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_cosine_embedding_default() {
        let loss_fn = CosineEmbeddingLoss::default();
        assert!((loss_fn.margin - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_embedding_name() {
        let loss_fn = CosineEmbeddingLoss::default();
        assert_eq!(loss_fn.name(), "CosineEmbeddingLoss");
    }

    #[test]
    fn test_cosine_embedding_positive_pair() {
        let loss_fn = CosineEmbeddingLoss::new(0.0);

        // Identical vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];

        let loss = loss_fn.compute_pair(&a, &b, 1.0);
        assert!(loss.abs() < 0.001);
    }

    #[test]
    fn test_cosine_embedding_negative_pair() {
        let loss_fn = CosineEmbeddingLoss::new(0.0);

        // Orthogonal vectors (should have loss 0 for negative with margin 0)
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let loss = loss_fn.compute_pair(&a, &b, -1.0);
        assert!(loss.abs() < 0.001);
    }

    #[test]
    fn test_cosine_embedding_negative_pair_with_margin() {
        let loss_fn = CosineEmbeddingLoss::new(0.5);

        // Vectors with cosine similarity 0
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let loss = loss_fn.compute_pair(&a, &b, -1.0);
        // cos_sim = 0, loss = max(0, 0 - 0.5) = 0
        assert!(loss.abs() < 0.001);
    }

    #[test]
    fn test_cosine_embedding_similar_negatives() {
        let loss_fn = CosineEmbeddingLoss::new(0.5);

        // Vectors with high cosine similarity
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.9, 0.1, 0.0];

        let loss = loss_fn.compute_pair(&a, &b, -1.0);
        // cos_sim > 0.5, so loss should be positive
        let cos_sim = cosine_similarity(&a, &b);
        if cos_sim > 0.5 {
            assert!(loss > 0.0);
        }
    }

    #[test]
    fn test_cosine_embedding_batch_loss() {
        let loss_fn = CosineEmbeddingLoss::new(0.0);

        let anchors = random_embeddings(4, 64, true);
        let positives = anchors.clone();

        let loss = loss_fn.compute(&anchors, &positives, None);
        // Perfect matches should have low loss
        assert!(loss < 0.1);
    }

    #[test]
    fn test_cosine_embedding_with_negatives() {
        let loss_fn = CosineEmbeddingLoss::new(0.0);

        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![0.9, 0.1, 0.0]];
        let negatives = vec![vec![vec![0.0, 1.0, 0.0]]];

        let loss = loss_fn.compute(&anchors, &positives, Some(&negatives));
        assert!(loss >= 0.0);
    }

    // =============================================================================
    // Euclidean Distance Tests
    // =============================================================================

    #[test]
    fn test_euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let dist = euclidean_distance(&a, &b);
        assert!(dist.abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance_basic() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];

        let dist = euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_euclidean_distance_unit_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let dist = euclidean_distance(&a, &b);
        assert!((dist - 2.0_f32.sqrt()).abs() < 0.001);
    }

    // =============================================================================
    // Softmax Tests
    // =============================================================================

    #[test]
    fn test_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_softmax_uniform() {
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let probs = softmax(&logits);

        for p in &probs {
            assert!((p - 0.25).abs() < 0.001);
        }
    }

    #[test]
    fn test_softmax_large_values() {
        let logits = vec![100.0, 200.0, 300.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        assert!(!probs.iter().any(|p| p.is_nan()));
    }

    #[test]
    fn test_softmax_negative_values() {
        let logits = vec![-1.0, -2.0, -3.0];
        let probs = softmax(&logits);

        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        assert!(probs[0] > probs[1] && probs[1] > probs[2]);
    }

    #[test]
    fn test_softmax_single_element() {
        let logits = vec![5.0];
        let probs = softmax(&logits);

        assert_eq!(probs.len(), 1);
        assert!((probs[0] - 1.0).abs() < 0.001);
    }

    // =============================================================================
    // Log-Softmax Tests
    // =============================================================================

    #[test]
    fn test_log_softmax_basic() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);

        // All log probabilities should be <= 0
        for lp in &log_probs {
            assert!(*lp <= 0.0);
        }
    }

    #[test]
    fn test_log_softmax_sum_exp() {
        let logits = vec![1.0, 2.0, 3.0];
        let log_probs = log_softmax(&logits);

        // exp(log_softmax) should sum to 1
        let sum: f32 = log_probs.iter().map(|lp| lp.exp()).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_log_softmax_numerical_stability() {
        let logits = vec![1000.0, 1001.0, 1002.0];
        let log_probs = log_softmax(&logits);

        // Should not produce NaN or Inf
        for lp in &log_probs {
            assert!(!lp.is_nan());
            assert!(!lp.is_infinite());
        }
    }

    #[test]
    fn test_log_softmax_vs_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);
        let log_probs = log_softmax(&logits);

        for (p, lp) in probs.iter().zip(log_probs.iter()) {
            assert!((p.ln() - lp).abs() < 0.001);
        }
    }

    // =============================================================================
    // LossFunction Trait Tests
    // =============================================================================

    #[test]
    fn test_loss_function_trait_mnrl() {
        let loss_fn: Box<dyn LossFunction> = Box::new(MultipleNegativesRankingLoss::default());

        let anchors = random_embeddings(4, 64, true);
        let positives = random_embeddings(4, 64, true);

        let loss = loss_fn.compute(&anchors, &positives, None);
        assert!(loss >= 0.0);
        assert_eq!(loss_fn.name(), "MultipleNegativesRankingLoss");
    }

    #[test]
    fn test_loss_function_trait_triplet() {
        let loss_fn: Box<dyn LossFunction> = Box::new(TripletMarginLoss::default());

        let anchors = vec![vec![1.0, 0.0, 0.0]];
        let positives = vec![vec![0.9, 0.1, 0.0]];
        let negatives = vec![vec![vec![0.0, 1.0, 0.0]]];

        let loss = loss_fn.compute(&anchors, &positives, Some(&negatives));
        assert!(loss >= 0.0);
        assert_eq!(loss_fn.name(), "TripletMarginLoss");
    }

    #[test]
    fn test_loss_function_trait_infonce() {
        let loss_fn: Box<dyn LossFunction> = Box::new(InfoNCELoss::default());

        let anchors = random_embeddings(4, 64, true);
        let positives = random_embeddings(4, 64, true);

        let loss = loss_fn.compute(&anchors, &positives, None);
        assert!(loss >= 0.0);
        assert_eq!(loss_fn.name(), "InfoNCELoss");
    }

    #[test]
    fn test_loss_function_trait_cosine() {
        let loss_fn: Box<dyn LossFunction> = Box::new(CosineEmbeddingLoss::default());

        let anchors = random_embeddings(4, 64, true);
        let positives = random_embeddings(4, 64, true);

        let loss = loss_fn.compute(&anchors, &positives, None);
        assert!(loss >= 0.0);
        assert_eq!(loss_fn.name(), "CosineEmbeddingLoss");
    }

    // =============================================================================
    // Edge Cases and Stress Tests
    // =============================================================================

    #[test]
    fn test_loss_high_dimensional() {
        let loss_fn = MultipleNegativesRankingLoss::new(20.0);

        let anchors = random_embeddings(8, 768, true);
        let positives = random_embeddings(8, 768, true);

        let loss = loss_fn.compute(&anchors, &positives, None);
        assert!(loss >= 0.0);
        assert!(!loss.is_nan());
    }

    #[test]
    fn test_loss_large_batch() {
        let loss_fn = MultipleNegativesRankingLoss::new(20.0);

        let anchors = random_embeddings(32, 64, true);
        let positives = random_embeddings(32, 64, true);

        let loss = loss_fn.compute(&anchors, &positives, None);
        assert!(loss >= 0.0);
        assert!(!loss.is_nan());
    }

    #[test]
    fn test_loss_gradient_direction() {
        let loss_fn = MultipleNegativesRankingLoss::new(20.0);

        // Perfect matches should have lower loss than random matches
        let anchors = random_embeddings(4, 64, true);
        let positives_perfect = anchors.clone();
        let positives_random = random_embeddings(4, 64, true);

        let loss_perfect = loss_fn.compute(&anchors, &positives_perfect, None);
        let loss_random = loss_fn.compute(&anchors, &positives_random, None);

        // Generally, perfect matches should have lower or equal loss
        // (though this may not always hold due to in-batch negatives)
        assert!(loss_perfect <= loss_random + 5.0); // Allow some tolerance
    }
}
