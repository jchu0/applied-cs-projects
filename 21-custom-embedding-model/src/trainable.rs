//! A small but **genuinely trainable** embedding model with hand-derived backprop.
//!
//! The legacy [`crate::model::BiEncoder`] + [`crate::trainer::EmbeddingTrainer`]
//! path is a simulation: the projection weights are random and never learn, and
//! the trainer's weight update adds *random noise* instead of gradients
//! (`trainer.rs`). This module is the real thing — analytic gradients of a
//! contrastive (logistic) loss flow back into a token-embedding table, so the
//! model actually learns to pull relevant pairs together and push irrelevant
//! pairs apart.
//!
//! It is intentionally dependency-free (no `candle`/`tch`/`burn`) so the gradient
//! math is fully visible; a production system would swap the hand-rolled backprop
//! for an autodiff backend, keeping this objective and pooling.

use rand::prelude::*;

/// A labelled training pair: two token sequences and whether they are relevant.
#[derive(Debug, Clone)]
pub struct TrainingPair {
    /// Token ids of the first text.
    pub a: Vec<usize>,
    /// Token ids of the second text.
    pub b: Vec<usize>,
    /// `1.0` if the pair is relevant (positive), `0.0` if not (negative).
    pub label: f32,
}

impl TrainingPair {
    /// A relevant (positive) pair.
    pub fn positive(a: Vec<usize>, b: Vec<usize>) -> Self {
        Self { a, b, label: 1.0 }
    }

    /// An irrelevant (negative) pair.
    pub fn negative(a: Vec<usize>, b: Vec<usize>) -> Self {
        Self { a, b, label: 0.0 }
    }
}

/// Mean-pooled token-embedding model trained with a logistic contrastive loss.
///
/// `embed(tokens)` mean-pools the rows of a `vocab_size × dim` embedding table;
/// `similarity` is the dot product of two pooled embeddings; training performs
/// real SGD on those table rows.
#[derive(Debug, Clone)]
pub struct TrainableEmbedder {
    vocab_size: usize,
    dim: usize,
    table: Vec<Vec<f32>>,
}

impl TrainableEmbedder {
    /// Create a model with small random initial embeddings.
    pub fn new(vocab_size: usize, dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let table = (0..vocab_size)
            .map(|_| (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        Self { vocab_size, dim, table }
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Mean-pool the embeddings of the given tokens (out-of-range tokens are skipped).
    pub fn embed(&self, tokens: &[usize]) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        let mut n = 0usize;
        for &t in tokens {
            if t < self.vocab_size {
                for d in 0..self.dim {
                    v[d] += self.table[t][d];
                }
                n += 1;
            }
        }
        if n > 0 {
            let inv = 1.0 / n as f32;
            for d in 0..self.dim {
                v[d] *= inv;
            }
        }
        v
    }

    /// Dot-product similarity between two token sequences.
    pub fn similarity(&self, a: &[usize], b: &[usize]) -> f32 {
        dot(&self.embed(a), &self.embed(b))
    }

    /// Train over `epochs` passes of the pairs; returns the mean loss per epoch.
    pub fn fit(&mut self, pairs: &[TrainingPair], epochs: usize, lr: f32) -> Vec<f32> {
        (0..epochs).map(|_| self.train_epoch(pairs, lr)).collect()
    }

    /// One epoch of SGD over the pairs; returns the mean (pre-update) loss.
    pub fn train_epoch(&mut self, pairs: &[TrainingPair], lr: f32) -> f32 {
        if pairs.is_empty() {
            return 0.0;
        }
        let mut total = 0.0f32;
        for p in pairs {
            total += self.train_pair(p, lr);
        }
        total / pairs.len() as f32
    }

    /// One real SGD step on a single pair; returns the loss evaluated before the step.
    ///
    /// With `s = <embed(a), embed(b)>` and `p = sigmoid(s)`, the logistic loss is
    /// `-[y·ln p + (1-y)·ln(1-p)]` and `dL/ds = p - y`. Because `s` is bilinear in
    /// the pooled embeddings, the gradient w.r.t. a token row `t ∈ a` is
    /// `(p - y) · (1/|a|) · embed(b)` (and symmetrically for `t ∈ b`).
    fn train_pair(&mut self, pair: &TrainingPair, lr: f32) -> f32 {
        let ua = self.embed(&pair.a);
        let ub = self.embed(&pair.b);
        let s = dot(&ua, &ub);
        let pred = sigmoid(s);

        let eps = 1e-7;
        let loss = -(pair.label * (pred + eps).ln() + (1.0 - pair.label) * (1.0 - pred + eps).ln());

        let dl_ds = pred - pair.label;
        let na = pair.a.iter().filter(|&&t| t < self.vocab_size).count().max(1) as f32;
        let nb = pair.b.iter().filter(|&&t| t < self.vocab_size).count().max(1) as f32;

        // Gradients are evaluated at the pre-step point: `ua`/`ub` are captured
        // copies, so mutating the table below does not affect them.
        for &t in &pair.a {
            if t < self.vocab_size {
                for d in 0..self.dim {
                    self.table[t][d] -= lr * dl_ds * (1.0 / na) * ub[d];
                }
            }
        }
        for &t in &pair.b {
            if t < self.vocab_size {
                for d in 0..self.dim {
                    self.table[t][d] -= lr * dl_ds * (1.0 / nb) * ua[d];
                }
            }
        }
        loss
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vocab: topic-A tokens {0,1}, topic-B tokens {2,3}.
    fn topic_pairs() -> Vec<TrainingPair> {
        vec![
            TrainingPair::positive(vec![0, 1], vec![1, 0]), // same topic A
            TrainingPair::positive(vec![2, 3], vec![3, 2]), // same topic B
            TrainingPair::negative(vec![0, 1], vec![2, 3]), // A vs B
            TrainingPair::negative(vec![2, 3], vec![0, 1]), // B vs A
        ]
    }

    #[test]
    fn embed_of_empty_is_zero() {
        let m = TrainableEmbedder::new(4, 8);
        assert!(m.embed(&[]).iter().all(|&x| x == 0.0));
    }

    #[test]
    fn training_reduces_loss() {
        let mut m = TrainableEmbedder::new(4, 8);
        let pairs = topic_pairs();
        let losses = m.fit(&pairs, 300, 0.5);
        let first = *losses.first().unwrap();
        let last = *losses.last().unwrap();
        assert!(last < first, "loss did not decrease: {first} -> {last}");
        assert!(last < 0.25, "final loss too high: {last}");
    }

    #[test]
    fn training_separates_positives_from_negatives() {
        let mut m = TrainableEmbedder::new(4, 8);
        let pairs = topic_pairs();
        m.fit(&pairs, 300, 0.5);
        let pos = m.similarity(&[0, 1], &[1, 0]); // same topic
        let neg = m.similarity(&[0, 1], &[2, 3]); // different topic
        assert!(pos > neg, "positive sim {pos} should exceed negative sim {neg}");
    }

    #[test]
    fn gradient_is_not_noise_deterministic_direction() {
        // Two models trained on the same data reach the same separation ordering,
        // unlike a noise-based update. (Init differs, but the learned ordering is
        // a property of the gradient signal, not randomness.)
        let pairs = topic_pairs();
        let mut a = TrainableEmbedder::new(4, 8);
        let mut b = TrainableEmbedder::new(4, 8);
        a.fit(&pairs, 300, 0.5);
        b.fit(&pairs, 300, 0.5);
        for m in [&a, &b] {
            assert!(m.similarity(&[0, 1], &[1, 0]) > m.similarity(&[0, 1], &[2, 3]));
        }
    }
}
