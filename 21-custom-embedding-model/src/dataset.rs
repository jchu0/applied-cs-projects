//! Dataset pipeline for embedding training.

use crate::{Error, Result, EMBEDDING_DIM};
use rand::prelude::*;
use std::collections::{HashMap, HashSet};

/// Single embedding training example.
#[derive(Debug, Clone)]
pub struct EmbeddingExample {
    /// Anchor ID.
    pub anchor_id: String,
    /// Anchor text.
    pub anchor_text: String,
    /// Positive ID.
    pub positive_id: String,
    /// Positive text.
    pub positive_text: String,
    /// Negative IDs.
    pub negative_ids: Vec<String>,
    /// Negative texts.
    pub negative_texts: Vec<String>,
    /// Domain tag.
    pub domain: Option<String>,
    /// Difficulty score (0=easy, 1=hard).
    pub difficulty: f32,
}

impl EmbeddingExample {
    /// Create new example.
    pub fn new(
        anchor_id: impl Into<String>,
        anchor_text: impl Into<String>,
        positive_id: impl Into<String>,
        positive_text: impl Into<String>,
    ) -> Self {
        Self {
            anchor_id: anchor_id.into(),
            anchor_text: anchor_text.into(),
            positive_id: positive_id.into(),
            positive_text: positive_text.into(),
            negative_ids: Vec::new(),
            negative_texts: Vec::new(),
            domain: None,
            difficulty: 0.5,
        }
    }

    /// Add negative example.
    pub fn add_negative(&mut self, id: impl Into<String>, text: impl Into<String>) {
        self.negative_ids.push(id.into());
        self.negative_texts.push(text.into());
    }

    /// Number of negatives.
    pub fn num_negatives(&self) -> usize {
        self.negative_ids.len()
    }
}

/// Collated batch for training.
#[derive(Debug, Clone)]
pub struct EmbeddingBatch {
    /// Anchor embeddings.
    pub anchor_embeddings: Vec<Vec<f32>>,
    /// Positive embeddings.
    pub positive_embeddings: Vec<Vec<f32>>,
    /// Negative embeddings (batch_size x num_negatives).
    pub negative_embeddings: Vec<Vec<Vec<f32>>>,
    /// Labels (indices of positives).
    pub labels: Vec<usize>,
}

impl EmbeddingBatch {
    /// Create new batch.
    pub fn new(batch_size: usize) -> Self {
        Self {
            anchor_embeddings: Vec::with_capacity(batch_size),
            positive_embeddings: Vec::with_capacity(batch_size),
            negative_embeddings: Vec::with_capacity(batch_size),
            labels: (0..batch_size).collect(),
        }
    }

    /// Batch size.
    pub fn len(&self) -> usize {
        self.anchor_embeddings.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.anchor_embeddings.is_empty()
    }

    /// Add example to batch.
    pub fn add(
        &mut self,
        anchor: Vec<f32>,
        positive: Vec<f32>,
        negatives: Vec<Vec<f32>>,
    ) {
        self.anchor_embeddings.push(anchor);
        self.positive_embeddings.push(positive);
        self.negative_embeddings.push(negatives);
    }
}

/// Hard negative miner using approximate nearest neighbor search.
pub struct HardNegativeMiner {
    /// Corpus embeddings.
    embeddings: Vec<Vec<f32>>,
    /// Corpus IDs.
    ids: Vec<String>,
    /// ID to index mapping.
    id_to_idx: HashMap<String, usize>,
}

impl HardNegativeMiner {
    /// Create new miner.
    pub fn new() -> Self {
        Self {
            embeddings: Vec::new(),
            ids: Vec::new(),
            id_to_idx: HashMap::new(),
        }
    }

    /// Build index from corpus embeddings.
    pub fn build_index(
        &mut self,
        embeddings: Vec<Vec<f32>>,
        ids: Vec<String>,
    ) -> Result<()> {
        if embeddings.len() != ids.len() {
            return Err(Error::DimensionMismatch {
                expected: ids.len(),
                got: embeddings.len(),
            });
        }

        self.id_to_idx.clear();
        for (i, id) in ids.iter().enumerate() {
            self.id_to_idx.insert(id.clone(), i);
        }

        self.embeddings = embeddings;
        self.ids = ids;

        Ok(())
    }

    /// Mine hard negatives for anchor embeddings.
    pub fn mine(
        &self,
        anchor_embs: &[Vec<f32>],
        positive_ids: &[String],
        k: usize,
    ) -> Vec<Vec<usize>> {
        let mut results = Vec::with_capacity(anchor_embs.len());

        for (anchor, pos_id) in anchor_embs.iter().zip(positive_ids) {
            // Compute similarities to all corpus embeddings
            let mut similarities: Vec<(usize, f32)> = self.embeddings
                .iter()
                .enumerate()
                .map(|(idx, emb)| (idx, cosine_similarity(anchor, emb)))
                .collect();

            // Sort by similarity descending
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Get positive index to exclude
            let pos_idx = self.id_to_idx.get(pos_id);

            // Collect top-k negatives (excluding positive)
            let negatives: Vec<usize> = similarities
                .into_iter()
                .filter(|(idx, _)| {
                    pos_idx.map_or(true, |&pi| *idx != pi)
                })
                .take(k)
                .map(|(idx, _)| idx)
                .collect();

            results.push(negatives);
        }

        results
    }

    /// Get embeddings by indices.
    pub fn get_embeddings(&self, indices: &[usize]) -> Vec<Vec<f32>> {
        indices
            .iter()
            .filter_map(|&idx| self.embeddings.get(idx).cloned())
            .collect()
    }

    /// Get IDs by indices.
    pub fn get_ids(&self, indices: &[usize]) -> Vec<String> {
        indices
            .iter()
            .filter_map(|&idx| self.ids.get(idx).cloned())
            .collect()
    }

    /// Corpus size.
    pub fn size(&self) -> usize {
        self.embeddings.len()
    }
}

impl Default for HardNegativeMiner {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory bank for in-batch negative sampling.
pub struct MemoryBank {
    /// Memory buffer.
    buffer: Vec<Vec<f32>>,
    /// Current pointer.
    ptr: usize,
    /// Maximum size.
    max_size: usize,
    /// Whether buffer is full.
    is_full: bool,
}

impl MemoryBank {
    /// Create new memory bank.
    pub fn new(max_size: usize, embedding_dim: usize) -> Self {
        Self {
            buffer: vec![vec![0.0; embedding_dim]; max_size],
            ptr: 0,
            max_size,
            is_full: false,
        }
    }

    /// Update memory bank with new embeddings (FIFO).
    pub fn update(&mut self, embeddings: &[Vec<f32>]) {
        for emb in embeddings {
            self.buffer[self.ptr] = emb.clone();
            self.ptr += 1;

            if self.ptr >= self.max_size {
                self.ptr = 0;
                self.is_full = true;
            }
        }
    }

    /// Get all valid embeddings from memory bank.
    pub fn get_embeddings(&self) -> &[Vec<f32>] {
        if self.is_full {
            &self.buffer
        } else {
            &self.buffer[..self.ptr]
        }
    }

    /// Current size.
    pub fn size(&self) -> usize {
        if self.is_full {
            self.max_size
        } else {
            self.ptr
        }
    }

    /// Clear memory bank.
    pub fn clear(&mut self) {
        self.ptr = 0;
        self.is_full = false;
    }
}

/// In-batch negative sampler.
pub struct InBatchNegativeSampler {
    /// Memory bank for past embeddings.
    memory_bank: MemoryBank,
    /// Use memory bank negatives.
    use_memory: bool,
}

impl InBatchNegativeSampler {
    /// Create new sampler.
    pub fn new(memory_size: usize, embedding_dim: usize, use_memory: bool) -> Self {
        Self {
            memory_bank: MemoryBank::new(memory_size, embedding_dim),
            use_memory,
        }
    }

    /// Get negative candidates for batch.
    pub fn get_negatives(
        &self,
        batch_embeddings: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        let mut negatives = batch_embeddings.to_vec();

        if self.use_memory {
            negatives.extend(self.memory_bank.get_embeddings().iter().cloned());
        }

        negatives
    }

    /// Update memory bank.
    pub fn update_memory(&mut self, embeddings: &[Vec<f32>]) {
        if self.use_memory {
            self.memory_bank.update(embeddings);
        }
    }
}

/// Batch sampler with curriculum learning.
pub struct CurriculumSampler {
    /// Examples sorted by difficulty.
    examples: Vec<(usize, f32)>, // (index, difficulty)
    /// Current difficulty threshold.
    difficulty_threshold: f32,
    /// Difficulty increase rate per epoch.
    increase_rate: f32,
}

impl CurriculumSampler {
    /// Create new curriculum sampler.
    pub fn new(difficulties: Vec<f32>, increase_rate: f32) -> Self {
        let mut examples: Vec<_> = difficulties
            .into_iter()
            .enumerate()
            .map(|(i, d)| (i, d))
            .collect();
        examples.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        Self {
            examples,
            difficulty_threshold: 0.3,
            increase_rate,
        }
    }

    /// Get indices for current epoch.
    pub fn get_epoch_indices(&self) -> Vec<usize> {
        self.examples
            .iter()
            .filter(|(_, d)| *d <= self.difficulty_threshold)
            .map(|(i, _)| *i)
            .collect()
    }

    /// Advance to next epoch.
    pub fn next_epoch(&mut self) {
        self.difficulty_threshold = (self.difficulty_threshold + self.increase_rate).min(1.0);
    }

    /// Current difficulty threshold.
    pub fn threshold(&self) -> f32 {
        self.difficulty_threshold
    }
}

/// Random negative sampler.
pub struct RandomNegativeSampler {
    /// Random number generator.
    rng: rand::rngs::StdRng,
}

impl RandomNegativeSampler {
    /// Create new random sampler with seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Sample random negatives.
    pub fn sample(
        &mut self,
        corpus_size: usize,
        exclude: &HashSet<usize>,
        k: usize,
    ) -> Vec<usize> {
        let mut negatives = Vec::with_capacity(k);

        while negatives.len() < k {
            let idx = self.rng.gen_range(0..corpus_size);
            if !exclude.contains(&idx) && !negatives.contains(&idx) {
                negatives.push(idx);
            }
        }

        negatives
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute dot product similarity.
pub fn dot_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Normalize vector to unit length.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // EmbeddingExample Tests
    // =============================================================================

    #[test]
    fn test_embedding_example_creation() {
        let example = EmbeddingExample::new(
            "anchor_001",
            "This is the anchor text",
            "positive_001",
            "This is the positive text",
        );

        assert_eq!(example.anchor_id, "anchor_001");
        assert_eq!(example.anchor_text, "This is the anchor text");
        assert_eq!(example.positive_id, "positive_001");
        assert_eq!(example.positive_text, "This is the positive text");
        assert!(example.negative_ids.is_empty());
        assert!(example.negative_texts.is_empty());
        assert_eq!(example.domain, None);
        assert!((example.difficulty - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_embedding_example_add_negative() {
        let mut example = EmbeddingExample::new(
            "a1", "anchor text", "p1", "positive text"
        );

        assert_eq!(example.num_negatives(), 0);

        example.add_negative("n1", "negative text 1");
        assert_eq!(example.num_negatives(), 1);
        assert_eq!(example.negative_ids[0], "n1");
        assert_eq!(example.negative_texts[0], "negative text 1");

        example.add_negative("n2", "negative text 2");
        assert_eq!(example.num_negatives(), 2);
        assert_eq!(example.negative_ids[1], "n2");
    }

    #[test]
    fn test_embedding_example_multiple_negatives() {
        let mut example = EmbeddingExample::new(
            "a1", "anchor", "p1", "positive"
        );

        for i in 0..10 {
            example.add_negative(format!("n{}", i), format!("negative {}", i));
        }

        assert_eq!(example.num_negatives(), 10);
    }

    #[test]
    fn test_embedding_example_with_domain() {
        let mut example = EmbeddingExample::new(
            "a1", "anchor", "p1", "positive"
        );
        example.domain = Some("medical".to_string());

        assert_eq!(example.domain, Some("medical".to_string()));
    }

    #[test]
    fn test_embedding_example_difficulty() {
        let mut example = EmbeddingExample::new(
            "a1", "anchor", "p1", "positive"
        );

        // Default difficulty
        assert!((example.difficulty - 0.5).abs() < 0.001);

        // Set easy difficulty
        example.difficulty = 0.0;
        assert!((example.difficulty - 0.0).abs() < 0.001);

        // Set hard difficulty
        example.difficulty = 1.0;
        assert!((example.difficulty - 1.0).abs() < 0.001);
    }

    // =============================================================================
    // EmbeddingBatch Tests
    // =============================================================================

    #[test]
    fn test_embedding_batch_creation() {
        let batch = EmbeddingBatch::new(8);

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
        assert_eq!(batch.labels.len(), 8);
    }

    #[test]
    fn test_embedding_batch_add() {
        let mut batch = EmbeddingBatch::new(4);

        let anchor = vec![1.0, 0.0, 0.0, 0.0];
        let positive = vec![0.9, 0.1, 0.0, 0.0];
        let negatives = vec![
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        batch.add(anchor.clone(), positive.clone(), negatives.clone());

        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
        assert_eq!(batch.anchor_embeddings[0], anchor);
        assert_eq!(batch.positive_embeddings[0], positive);
        assert_eq!(batch.negative_embeddings[0].len(), 2);
    }

    #[test]
    fn test_embedding_batch_multiple_adds() {
        let mut batch = EmbeddingBatch::new(4);

        for i in 0..4 {
            let anchor = vec![i as f32; 4];
            let positive = vec![(i + 1) as f32; 4];
            batch.add(anchor, positive, vec![]);
        }

        assert_eq!(batch.len(), 4);
        assert_eq!(batch.anchor_embeddings.len(), 4);
        assert_eq!(batch.positive_embeddings.len(), 4);
    }

    #[test]
    fn test_embedding_batch_labels() {
        let batch = EmbeddingBatch::new(5);

        assert_eq!(batch.labels, vec![0, 1, 2, 3, 4]);
    }

    // =============================================================================
    // Cosine Similarity Tests
    // =============================================================================

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_partial() {
        let a = vec![1.0, 1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        // cos(45 degrees) = sqrt(2)/2 ≈ 0.707
        let expected = 1.0 / 2.0_f32.sqrt();
        assert!((cosine_similarity(&a, &b) - expected).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_high_dimensional() {
        let dim = 768;
        let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        let b = a.clone();
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
    }

    // =============================================================================
    // Dot Similarity Tests
    // =============================================================================

    #[test]
    fn test_dot_similarity_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((dot_similarity(&a, &b) - 32.0).abs() < 0.001);
    }

    #[test]
    fn test_dot_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(dot_similarity(&a, &b).abs() < 0.001);
    }

    // =============================================================================
    // Normalize Tests
    // =============================================================================

    #[test]
    fn test_normalize_basic() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize(&mut v);

        // Should remain zero
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_normalize_already_unit() {
        let mut v = vec![1.0, 0.0, 0.0];
        normalize(&mut v);

        assert!((v[0] - 1.0).abs() < 0.001);
        assert!(v[1].abs() < 0.001);
    }

    #[test]
    fn test_normalize_high_dimensional() {
        let dim = 768;
        let mut v: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
        normalize(&mut v);

        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    // =============================================================================
    // HardNegativeMiner Tests
    // =============================================================================

    #[test]
    fn test_hard_negative_miner_new() {
        let miner = HardNegativeMiner::new();
        assert_eq!(miner.size(), 0);
    }

    #[test]
    fn test_hard_negative_miner_default() {
        let miner = HardNegativeMiner::default();
        assert_eq!(miner.size(), 0);
    }

    #[test]
    fn test_hard_negative_miner_build_index() {
        let mut miner = HardNegativeMiner::new();

        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ids = vec!["a".into(), "b".into(), "c".into(), "d".into()];

        miner.build_index(embeddings, ids).unwrap();
        assert_eq!(miner.size(), 4);
    }

    #[test]
    fn test_hard_negative_miner_build_index_dimension_mismatch() {
        let mut miner = HardNegativeMiner::new();

        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let ids = vec!["a".into(), "b".into(), "c".into()]; // 3 ids, 2 embeddings

        let result = miner.build_index(embeddings, ids);
        assert!(result.is_err());
    }

    #[test]
    fn test_hard_negative_miner_mine() {
        let mut miner = HardNegativeMiner::new();

        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ids = vec!["a".into(), "b".into(), "c".into(), "d".into()];

        miner.build_index(embeddings, ids).unwrap();

        let anchor = vec![vec![1.0, 0.0, 0.0]];
        let pos_ids = vec!["a".into()];

        let negatives = miner.mine(&anchor, &pos_ids, 2);
        assert_eq!(negatives[0].len(), 2);
        assert_eq!(negatives[0][0], 1); // "b" is most similar after "a"
    }

    #[test]
    fn test_hard_negative_miner_mine_multiple_anchors() {
        let mut miner = HardNegativeMiner::new();

        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ids = vec!["a".into(), "b".into(), "c".into(), "d".into()];

        miner.build_index(embeddings, ids).unwrap();

        let anchors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let pos_ids = vec!["a".into(), "c".into()];

        let negatives = miner.mine(&anchors, &pos_ids, 2);
        assert_eq!(negatives.len(), 2);
        assert_eq!(negatives[0].len(), 2);
        assert_eq!(negatives[1].len(), 2);
    }

    #[test]
    fn test_hard_negative_miner_get_embeddings() {
        let mut miner = HardNegativeMiner::new();

        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ids = vec!["a".into(), "b".into(), "c".into()];

        miner.build_index(embeddings.clone(), ids).unwrap();

        let retrieved = miner.get_embeddings(&[0, 2]);
        assert_eq!(retrieved.len(), 2);
        assert_eq!(retrieved[0], embeddings[0]);
        assert_eq!(retrieved[1], embeddings[2]);
    }

    #[test]
    fn test_hard_negative_miner_get_ids() {
        let mut miner = HardNegativeMiner::new();

        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let ids = vec!["a".into(), "b".into(), "c".into()];

        miner.build_index(embeddings, ids).unwrap();

        let retrieved_ids = miner.get_ids(&[0, 2]);
        assert_eq!(retrieved_ids.len(), 2);
        assert_eq!(retrieved_ids[0], "a");
        assert_eq!(retrieved_ids[1], "c");
    }

    // =============================================================================
    // MemoryBank Tests
    // =============================================================================

    #[test]
    fn test_memory_bank_creation() {
        let bank = MemoryBank::new(100, 64);
        assert_eq!(bank.size(), 0);
        assert!(!bank.is_full);
    }

    #[test]
    fn test_memory_bank_update() {
        let mut bank = MemoryBank::new(10, 4);
        let emb = vec![vec![1.0, 0.0, 0.0, 0.0]];

        bank.update(&emb);
        assert_eq!(bank.size(), 1);
        assert!(!bank.is_full);
    }

    #[test]
    fn test_memory_bank_wrap_around() {
        let mut bank = MemoryBank::new(10, 4);
        let emb = vec![vec![1.0, 0.0, 0.0, 0.0]];

        for _ in 0..10 {
            bank.update(&emb);
        }
        assert!(bank.is_full);
        assert_eq!(bank.size(), 10);

        // Add more - should wrap around
        bank.update(&emb);
        assert!(bank.is_full);
        assert_eq!(bank.size(), 10);
    }

    #[test]
    fn test_memory_bank_get_embeddings() {
        let mut bank = MemoryBank::new(10, 4);

        let emb1 = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let emb2 = vec![vec![0.0, 1.0, 0.0, 0.0]];

        bank.update(&emb1);
        bank.update(&emb2);

        let embeddings = bank.get_embeddings();
        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0], emb1[0]);
        assert_eq!(embeddings[1], emb2[0]);
    }

    #[test]
    fn test_memory_bank_clear() {
        let mut bank = MemoryBank::new(10, 4);
        let emb = vec![vec![1.0, 0.0, 0.0, 0.0]];

        for _ in 0..10 {
            bank.update(&emb);
        }
        assert!(bank.is_full);

        bank.clear();
        assert_eq!(bank.size(), 0);
        assert!(!bank.is_full);
    }

    // =============================================================================
    // InBatchNegativeSampler Tests
    // =============================================================================

    #[test]
    fn test_in_batch_sampler_creation() {
        let sampler = InBatchNegativeSampler::new(1000, 64, true);
        assert!(sampler.use_memory);
    }

    #[test]
    fn test_in_batch_sampler_get_negatives_no_memory() {
        let sampler = InBatchNegativeSampler::new(1000, 4, false);

        let batch_embs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        let negatives = sampler.get_negatives(&batch_embs);
        assert_eq!(negatives.len(), 2);
    }

    #[test]
    fn test_in_batch_sampler_with_memory() {
        let mut sampler = InBatchNegativeSampler::new(1000, 4, true);

        let batch1 = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        sampler.update_memory(&batch1);

        let batch2 = vec![
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let negatives = sampler.get_negatives(&batch2);
        // Should include batch2 + memory bank (batch1)
        assert_eq!(negatives.len(), 3);
    }

    // =============================================================================
    // CurriculumSampler Tests
    // =============================================================================

    #[test]
    fn test_curriculum_sampler_creation() {
        let difficulties = vec![0.1, 0.5, 0.9, 0.2, 0.8];
        let sampler = CurriculumSampler::new(difficulties, 0.1);

        assert!((sampler.threshold() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_curriculum_sampler_get_indices() {
        let difficulties = vec![0.1, 0.5, 0.9, 0.2, 0.8];
        let sampler = CurriculumSampler::new(difficulties, 0.1);

        let indices = sampler.get_epoch_indices();
        // Should include indices with difficulty <= 0.3 (indices 0 and 3)
        assert!(indices.contains(&0)); // difficulty 0.1
        assert!(indices.contains(&3)); // difficulty 0.2
        assert!(!indices.contains(&1)); // difficulty 0.5
    }

    #[test]
    fn test_curriculum_sampler_next_epoch() {
        let difficulties = vec![0.1, 0.5, 0.9, 0.2, 0.8];
        let mut sampler = CurriculumSampler::new(difficulties, 0.2);

        let initial_threshold = sampler.threshold();
        sampler.next_epoch();

        assert!((sampler.threshold() - (initial_threshold + 0.2)).abs() < 0.001);
    }

    #[test]
    fn test_curriculum_sampler_threshold_cap() {
        let difficulties = vec![0.1, 0.5, 0.9];
        let mut sampler = CurriculumSampler::new(difficulties, 0.5);

        for _ in 0..10 {
            sampler.next_epoch();
        }

        // Threshold should be capped at 1.0
        assert!(sampler.threshold() <= 1.0);
    }

    // =============================================================================
    // RandomNegativeSampler Tests
    // =============================================================================

    #[test]
    fn test_random_sampler_creation() {
        let sampler = RandomNegativeSampler::new(42);
        // Just check it creates without error
        assert!(true);
    }

    #[test]
    fn test_random_sampler_sample() {
        let mut sampler = RandomNegativeSampler::new(42);
        let exclude: HashSet<usize> = vec![0, 1, 2].into_iter().collect();

        let negatives = sampler.sample(100, &exclude, 5);

        assert_eq!(negatives.len(), 5);
        for &idx in &negatives {
            assert!(!exclude.contains(&idx));
            assert!(idx < 100);
        }
    }

    #[test]
    fn test_random_sampler_no_duplicates() {
        let mut sampler = RandomNegativeSampler::new(42);
        let exclude = HashSet::new();

        let negatives = sampler.sample(1000, &exclude, 50);

        let unique: HashSet<usize> = negatives.iter().copied().collect();
        assert_eq!(unique.len(), negatives.len());
    }

    #[test]
    fn test_random_sampler_deterministic() {
        let mut sampler1 = RandomNegativeSampler::new(42);
        let mut sampler2 = RandomNegativeSampler::new(42);
        let exclude = HashSet::new();

        let neg1 = sampler1.sample(1000, &exclude, 10);
        let neg2 = sampler2.sample(1000, &exclude, 10);

        assert_eq!(neg1, neg2);
    }
}
