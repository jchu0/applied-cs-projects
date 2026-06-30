//! Bi-encoder model architecture for embedding learning.

use crate::dataset::normalize;
use crate::{Error, Result, EMBEDDING_DIM};
use rand::prelude::*;

/// Pooling strategy for token embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Use CLS token embedding.
    Cls,
    /// Mean pooling over all tokens.
    Mean,
    /// Max pooling over all tokens.
    Max,
}

/// Bi-encoder for contrastive embedding learning.
pub struct BiEncoder {
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Pooling strategy.
    pub pooling: PoolingStrategy,
    /// Whether to normalize embeddings.
    pub normalize_output: bool,
    /// Projection dimension (None = no projection).
    pub projection_dim: Option<usize>,
    /// Encoder weights (simulated).
    encoder_weights: Vec<f32>,
    /// Projection weights.
    projection_weights: Option<Vec<f32>>,
}

impl BiEncoder {
    /// Create new bi-encoder.
    pub fn new(
        embedding_dim: usize,
        pooling: PoolingStrategy,
        normalize_output: bool,
        projection_dim: Option<usize>,
    ) -> Self {
        let mut rng = rand::thread_rng();

        // Initialize encoder weights (simulated)
        let encoder_weights: Vec<f32> = (0..embedding_dim * embedding_dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        // Initialize projection weights if needed
        let projection_weights = projection_dim.map(|proj_dim| {
            (0..embedding_dim * proj_dim)
                .map(|_| rng.gen_range(-0.1..0.1))
                .collect()
        });

        Self {
            embedding_dim,
            pooling,
            normalize_output,
            projection_dim,
            encoder_weights,
            projection_weights,
        }
    }

    /// Encode input tokens to embeddings.
    pub fn encode(&self, token_embeddings: &[Vec<f32>]) -> Result<Vec<f32>> {
        if token_embeddings.is_empty() {
            return Err(Error::EmptyData);
        }

        // Apply pooling strategy
        let mut embedding = match self.pooling {
            PoolingStrategy::Cls => token_embeddings[0].clone(),
            PoolingStrategy::Mean => {
                let dim = token_embeddings[0].len();
                let mut mean = vec![0.0; dim];
                for token in token_embeddings {
                    for (i, &v) in token.iter().enumerate() {
                        mean[i] += v;
                    }
                }
                let n = token_embeddings.len() as f32;
                for v in &mut mean {
                    *v /= n;
                }
                mean
            }
            PoolingStrategy::Max => {
                let dim = token_embeddings[0].len();
                let mut max = vec![f32::NEG_INFINITY; dim];
                for token in token_embeddings {
                    for (i, &v) in token.iter().enumerate() {
                        max[i] = max[i].max(v);
                    }
                }
                max
            }
        };

        // Apply projection if configured
        if let (Some(proj_dim), Some(weights)) = (self.projection_dim, &self.projection_weights) {
            embedding = self.project(&embedding, weights, proj_dim);
        }

        // Normalize for cosine similarity
        if self.normalize_output {
            normalize(&mut embedding);
        }

        Ok(embedding)
    }

    /// Project embedding to different dimension.
    fn project(&self, input: &[f32], weights: &[f32], output_dim: usize) -> Vec<f32> {
        let input_dim = input.len();
        let mut output = vec![0.0; output_dim];

        for i in 0..output_dim {
            for j in 0..input_dim {
                output[i] += input[j] * weights[i * input_dim + j];
            }
        }

        output
    }

    /// Encode batch of inputs.
    pub fn encode_batch(&self, batch: &[Vec<Vec<f32>>]) -> Result<Vec<Vec<f32>>> {
        batch.iter().map(|tokens| self.encode(tokens)).collect()
    }

    /// Get output dimension.
    pub fn output_dim(&self) -> usize {
        self.projection_dim.unwrap_or(self.embedding_dim)
    }

    /// Get output dimension (alias for output_dim).
    pub fn dim(&self) -> usize {
        self.output_dim()
    }

    /// Encode text to embeddings (generates mock token embeddings for serving).
    pub fn encode_text(&self, text: &str) -> Result<Vec<f32>> {
        // Generate mock token embeddings from text hash for reproducibility
        let mut token_embeddings = Vec::new();
        let words: Vec<&str> = text.split_whitespace().collect();
        let num_tokens = words.len().max(1);

        for (i, word) in words.iter().enumerate() {
            let mut token_emb = vec![0.0f32; self.embedding_dim];
            // Create deterministic embedding based on word hash
            let hash = word.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
            for j in 0..self.embedding_dim {
                token_emb[j] = ((hash.wrapping_add(j as u64) % 1000) as f32 / 1000.0 - 0.5) * 0.1;
            }
            token_embeddings.push(token_emb);
        }

        if token_embeddings.is_empty() {
            // Empty text - generate a default embedding
            token_embeddings.push(vec![0.0f32; self.embedding_dim]);
        }

        self.encode(&token_embeddings)
    }

    /// Get model parameters (for optimization).
    pub fn parameters(&self) -> Vec<&[f32]> {
        let mut params = vec![self.encoder_weights.as_slice()];
        if let Some(weights) = &self.projection_weights {
            params.push(weights.as_slice());
        }
        params
    }

    /// Get mutable model parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut [f32]> {
        let mut params: Vec<&mut [f32]> = vec![self.encoder_weights.as_mut_slice()];
        if let Some(weights) = &mut self.projection_weights {
            params.push(weights.as_mut_slice());
        }
        params
    }

    /// Number of parameters.
    pub fn num_parameters(&self) -> usize {
        let mut count = self.encoder_weights.len();
        if let Some(weights) = &self.projection_weights {
            count += weights.len();
        }
        count
    }
}

impl Default for BiEncoder {
    fn default() -> Self {
        Self::new(EMBEDDING_DIM, PoolingStrategy::Mean, true, None)
    }
}

/// Cross-encoder for reranking (produces scalar similarity score).
pub struct CrossEncoder {
    /// Hidden dimension.
    hidden_dim: usize,
    /// Weights for scoring.
    weights: Vec<f32>,
}

impl CrossEncoder {
    /// Create new cross-encoder.
    pub fn new(hidden_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights: Vec<f32> = (0..hidden_dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        Self { hidden_dim, weights }
    }

    /// Score query-document pair.
    pub fn score(&self, query_emb: &[f32], doc_emb: &[f32]) -> f32 {
        // Concatenate and score
        let mut score = 0.0;
        let half = self.hidden_dim / 2;

        for i in 0..half.min(query_emb.len()) {
            score += query_emb[i] * self.weights[i];
        }
        for i in 0..half.min(doc_emb.len()) {
            score += doc_emb[i] * self.weights[half + i];
        }

        // Sigmoid activation
        1.0 / (1.0 + (-score).exp())
    }

    /// Score batch of pairs.
    pub fn score_batch(&self, queries: &[Vec<f32>], docs: &[Vec<f32>]) -> Vec<f32> {
        queries
            .iter()
            .zip(docs)
            .map(|(q, d)| self.score(q, d))
            .collect()
    }
}

impl Default for CrossEncoder {
    fn default() -> Self {
        Self::new(EMBEDDING_DIM * 2)
    }
}

/// Sentence piece tokenizer simulation.
pub struct SimpleTokenizer {
    /// Vocabulary.
    vocab: std::collections::HashMap<String, usize>,
    /// Max sequence length.
    max_length: usize,
}

impl SimpleTokenizer {
    /// Create new tokenizer.
    pub fn new(max_length: usize) -> Self {
        // Simple vocabulary (would be loaded from file in production)
        let mut vocab = std::collections::HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);

        Self { vocab, max_length }
    }

    /// Tokenize text to token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![self.vocab["[CLS]"]];

        // Simple word tokenization
        for word in text.split_whitespace() {
            let token_id = self.vocab.get(word).copied().unwrap_or(1);
            tokens.push(token_id);

            if tokens.len() >= self.max_length - 1 {
                break;
            }
        }

        tokens.push(self.vocab["[SEP]"]);

        // Pad to max length
        while tokens.len() < self.max_length {
            tokens.push(0);
        }

        tokens
    }

    /// Create attention mask.
    pub fn attention_mask(&self, token_ids: &[usize]) -> Vec<f32> {
        token_ids.iter().map(|&id| if id == 0 { 0.0 } else { 1.0 }).collect()
    }
}

impl Default for SimpleTokenizer {
    fn default() -> Self {
        Self::new(512)
    }
}

/// Generate random embedding for testing.
pub fn random_embedding(dim: usize, normalize_output: bool) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut emb: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    if normalize_output {
        normalize(&mut emb);
    }

    emb
}

/// Generate batch of random embeddings.
pub fn random_embeddings(batch_size: usize, dim: usize, normalize_output: bool) -> Vec<Vec<f32>> {
    (0..batch_size)
        .map(|_| random_embedding(dim, normalize_output))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // PoolingStrategy Tests
    // =============================================================================

    #[test]
    fn test_pooling_strategy_equality() {
        assert_eq!(PoolingStrategy::Cls, PoolingStrategy::Cls);
        assert_eq!(PoolingStrategy::Mean, PoolingStrategy::Mean);
        assert_eq!(PoolingStrategy::Max, PoolingStrategy::Max);
        assert_ne!(PoolingStrategy::Cls, PoolingStrategy::Mean);
    }

    #[test]
    fn test_pooling_strategy_clone() {
        let strategy = PoolingStrategy::Mean;
        let cloned = strategy.clone();
        assert_eq!(strategy, cloned);
    }

    #[test]
    fn test_pooling_strategy_copy() {
        let strategy = PoolingStrategy::Cls;
        let copied: PoolingStrategy = strategy;
        assert_eq!(strategy, copied);
    }

    // =============================================================================
    // BiEncoder Tests
    // =============================================================================

    #[test]
    fn test_bi_encoder_creation() {
        let encoder = BiEncoder::new(64, PoolingStrategy::Mean, true, None);

        assert_eq!(encoder.embedding_dim, 64);
        assert_eq!(encoder.pooling, PoolingStrategy::Mean);
        assert!(encoder.normalize_output);
        assert!(encoder.projection_dim.is_none());
    }

    #[test]
    fn test_bi_encoder_default() {
        let encoder = BiEncoder::default();

        assert_eq!(encoder.embedding_dim, 768);
        assert_eq!(encoder.pooling, PoolingStrategy::Mean);
        assert!(encoder.normalize_output);
    }

    #[test]
    fn test_bi_encoder_encode() {
        let encoder = BiEncoder::new(64, PoolingStrategy::Mean, true, None);

        let tokens = vec![
            vec![1.0; 64],
            vec![2.0; 64],
            vec![3.0; 64],
        ];

        let embedding = encoder.encode(&tokens).unwrap();
        assert_eq!(embedding.len(), 64);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bi_encoder_encode_empty_input() {
        let encoder = BiEncoder::new(64, PoolingStrategy::Mean, true, None);

        let result = encoder.encode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bi_encoder_cls_pooling() {
        let tokens = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];

        let encoder = BiEncoder::new(2, PoolingStrategy::Cls, false, None);
        let emb = encoder.encode(&tokens).unwrap();
        assert_eq!(emb, vec![1.0, 0.0]);
    }

    #[test]
    fn test_bi_encoder_mean_pooling() {
        let tokens = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];

        let encoder = BiEncoder::new(2, PoolingStrategy::Mean, false, None);
        let emb = encoder.encode(&tokens).unwrap();
        assert_eq!(emb, vec![0.5, 0.5]);
    }

    #[test]
    fn test_bi_encoder_max_pooling() {
        let tokens = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];

        let encoder = BiEncoder::new(2, PoolingStrategy::Max, false, None);
        let emb = encoder.encode(&tokens).unwrap();
        assert_eq!(emb, vec![1.0, 1.0]);
    }

    #[test]
    fn test_bi_encoder_max_pooling_with_negatives() {
        let tokens = vec![
            vec![-1.0, -2.0],
            vec![-3.0, -1.0],
        ];

        let encoder = BiEncoder::new(2, PoolingStrategy::Max, false, None);
        let emb = encoder.encode(&tokens).unwrap();
        assert_eq!(emb, vec![-1.0, -1.0]);
    }

    #[test]
    fn test_bi_encoder_projection() {
        let encoder = BiEncoder::new(4, PoolingStrategy::Mean, false, Some(2));
        let tokens = vec![vec![1.0, 1.0, 1.0, 1.0]];

        let embedding = encoder.encode(&tokens).unwrap();
        assert_eq!(embedding.len(), 2);
    }

    #[test]
    fn test_bi_encoder_output_dim_no_projection() {
        let encoder = BiEncoder::new(64, PoolingStrategy::Mean, true, None);
        assert_eq!(encoder.output_dim(), 64);
    }

    #[test]
    fn test_bi_encoder_output_dim_with_projection() {
        let encoder = BiEncoder::new(64, PoolingStrategy::Mean, true, Some(32));
        assert_eq!(encoder.output_dim(), 32);
    }

    #[test]
    fn test_bi_encoder_encode_batch() {
        let encoder = BiEncoder::new(4, PoolingStrategy::Mean, true, None);

        let batch = vec![
            vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]],
            vec![vec![0.0, 0.0, 1.0, 0.0], vec![0.0, 0.0, 0.0, 1.0]],
        ];

        let embeddings = encoder.encode_batch(&batch).unwrap();
        assert_eq!(embeddings.len(), 2);

        for emb in embeddings {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_bi_encoder_parameters() {
        let encoder = BiEncoder::new(64, PoolingStrategy::Mean, true, None);
        let params = encoder.parameters();

        assert!(!params.is_empty());
        assert_eq!(params[0].len(), 64 * 64);
    }

    #[test]
    fn test_bi_encoder_parameters_with_projection() {
        let encoder = BiEncoder::new(64, PoolingStrategy::Mean, true, Some(32));
        let params = encoder.parameters();

        assert_eq!(params.len(), 2);
        assert_eq!(params[0].len(), 64 * 64);
        assert_eq!(params[1].len(), 64 * 32);
    }

    #[test]
    fn test_bi_encoder_num_parameters() {
        let encoder = BiEncoder::new(64, PoolingStrategy::Mean, true, None);
        assert_eq!(encoder.num_parameters(), 64 * 64);

        let encoder_proj = BiEncoder::new(64, PoolingStrategy::Mean, true, Some(32));
        assert_eq!(encoder_proj.num_parameters(), 64 * 64 + 64 * 32);
    }

    #[test]
    fn test_bi_encoder_parameters_mut() {
        let mut encoder = BiEncoder::new(4, PoolingStrategy::Mean, true, None);
        let mut params = encoder.parameters_mut();

        // Modify parameters
        params[0][0] = 99.0;

        // Check modification persisted
        assert!((encoder.parameters()[0][0] - 99.0).abs() < 0.001);
    }

    #[test]
    fn test_bi_encoder_no_normalization() {
        let encoder = BiEncoder::new(4, PoolingStrategy::Mean, false, None);
        let tokens = vec![vec![3.0, 4.0, 0.0, 0.0]];

        let embedding = encoder.encode(&tokens).unwrap();

        // Without normalization, the norm should not be 1
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Mean of [3.0, 4.0, 0.0, 0.0] = [3.0, 4.0, 0.0, 0.0], norm = 5
        assert!((norm - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_bi_encoder_single_token() {
        let encoder = BiEncoder::new(4, PoolingStrategy::Mean, true, None);
        let tokens = vec![vec![1.0, 2.0, 3.0, 4.0]];

        let embedding = encoder.encode(&tokens).unwrap();
        assert_eq!(embedding.len(), 4);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_bi_encoder_high_dimensional() {
        let encoder = BiEncoder::new(768, PoolingStrategy::Mean, true, None);
        let tokens: Vec<Vec<f32>> = (0..10)
            .map(|i| (0..768).map(|j| ((i * 768 + j) as f32).sin()).collect())
            .collect();

        let embedding = encoder.encode(&tokens).unwrap();
        assert_eq!(embedding.len(), 768);

        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    // =============================================================================
    // CrossEncoder Tests
    // =============================================================================

    #[test]
    fn test_cross_encoder_creation() {
        let encoder = CrossEncoder::new(128);
        assert!(true); // Just check it creates without error
    }

    #[test]
    fn test_cross_encoder_default() {
        let encoder = CrossEncoder::default();
        // Default hidden dim is EMBEDDING_DIM * 2 = 768 * 2 = 1536
        assert!(true);
    }

    #[test]
    fn test_cross_encoder_score() {
        let encoder = CrossEncoder::new(128);
        let query = random_embedding(64, true);
        let doc = random_embedding(64, true);

        let score = encoder.score(&query, &doc);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_cross_encoder_score_similar() {
        let encoder = CrossEncoder::new(128);
        let query = vec![1.0; 64];
        let doc = vec![1.0; 64];

        let score = encoder.score(&query, &doc);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_cross_encoder_score_batch() {
        let encoder = CrossEncoder::new(128);

        let queries: Vec<Vec<f32>> = (0..5)
            .map(|_| random_embedding(64, true))
            .collect();
        let docs: Vec<Vec<f32>> = (0..5)
            .map(|_| random_embedding(64, true))
            .collect();

        let scores = encoder.score_batch(&queries, &docs);
        assert_eq!(scores.len(), 5);

        for score in scores {
            assert!(score >= 0.0 && score <= 1.0);
        }
    }

    #[test]
    fn test_cross_encoder_sigmoid_bounds() {
        let encoder = CrossEncoder::new(128);

        // Test with extreme values
        let query_pos = vec![100.0; 64];
        let doc_pos = vec![100.0; 64];
        let score_high = encoder.score(&query_pos, &doc_pos);
        assert!(score_high >= 0.0 && score_high <= 1.0);

        let query_neg = vec![-100.0; 64];
        let doc_neg = vec![-100.0; 64];
        let score_low = encoder.score(&query_neg, &doc_neg);
        assert!(score_low >= 0.0 && score_low <= 1.0);
    }

    // =============================================================================
    // SimpleTokenizer Tests
    // =============================================================================

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = SimpleTokenizer::new(512);
        // Just check it creates without error
        assert!(true);
    }

    #[test]
    fn test_tokenizer_default() {
        let tokenizer = SimpleTokenizer::default();
        let tokens = tokenizer.tokenize("test");
        assert_eq!(tokens.len(), 512);
    }

    #[test]
    fn test_tokenizer_basic() {
        let tokenizer = SimpleTokenizer::new(10);
        let tokens = tokenizer.tokenize("hello world");

        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0], 2); // [CLS]
    }

    #[test]
    fn test_tokenizer_padding() {
        let tokenizer = SimpleTokenizer::new(20);
        let tokens = tokenizer.tokenize("hello");

        assert_eq!(tokens.len(), 20);
        assert_eq!(tokens[0], 2); // [CLS]
        // Most tokens after text should be 0 (padding)
        let padding_count = tokens.iter().filter(|&&t| t == 0).count();
        assert!(padding_count > 15);
    }

    #[test]
    fn test_tokenizer_truncation() {
        let tokenizer = SimpleTokenizer::new(5);
        let tokens = tokenizer.tokenize("one two three four five six seven eight");

        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0], 2); // [CLS]
        assert_eq!(tokens[4], 3); // [SEP]
    }

    #[test]
    fn test_tokenizer_attention_mask() {
        let tokenizer = SimpleTokenizer::new(10);
        let tokens = tokenizer.tokenize("hello world");
        let mask = tokenizer.attention_mask(&tokens);

        assert_eq!(mask.len(), 10);
        // First few should be 1.0 (non-padding)
        assert!((mask[0] - 1.0).abs() < 0.001);
        // Padding should be 0.0
        for &m in mask.iter().skip(4) {
            assert!((m - 0.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_tokenizer_empty_input() {
        let tokenizer = SimpleTokenizer::new(10);
        let tokens = tokenizer.tokenize("");

        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0], 2); // [CLS]
        assert_eq!(tokens[1], 3); // [SEP] immediately after CLS for empty input
    }

    #[test]
    fn test_tokenizer_unknown_words() {
        let tokenizer = SimpleTokenizer::new(10);
        let tokens = tokenizer.tokenize("xyz123 unknownword");

        // Unknown words should be tokenized as [UNK] (id 1)
        assert_eq!(tokens[1], 1); // First word should be UNK
        assert_eq!(tokens[2], 1); // Second word should be UNK
    }

    // =============================================================================
    // Random Embedding Tests
    // =============================================================================

    #[test]
    fn test_random_embedding() {
        let emb = random_embedding(64, false);
        assert_eq!(emb.len(), 64);
    }

    #[test]
    fn test_random_embedding_normalized() {
        let emb = random_embedding(64, true);
        assert_eq!(emb.len(), 64);

        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_random_embeddings() {
        let embs = random_embeddings(10, 64, false);
        assert_eq!(embs.len(), 10);

        for emb in &embs {
            assert_eq!(emb.len(), 64);
        }
    }

    #[test]
    fn test_random_embeddings_normalized() {
        let embs = random_embeddings(10, 64, true);
        assert_eq!(embs.len(), 10);

        for emb in &embs {
            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_random_embeddings_different() {
        let embs = random_embeddings(5, 64, true);

        // Check that embeddings are different from each other
        for i in 0..embs.len() {
            for j in (i + 1)..embs.len() {
                let diff: f32 = embs[i]
                    .iter()
                    .zip(&embs[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                assert!(diff > 0.0);
            }
        }
    }

    // =============================================================================
    // Integration Tests
    // =============================================================================

    #[test]
    fn test_bi_encoder_end_to_end() {
        let encoder = BiEncoder::new(64, PoolingStrategy::Mean, true, Some(32));

        // Simulate token embeddings from a transformer
        let tokens1: Vec<Vec<f32>> = (0..10)
            .map(|_| random_embedding(64, false))
            .collect();

        let tokens2: Vec<Vec<f32>> = (0..10)
            .map(|_| random_embedding(64, false))
            .collect();

        let emb1 = encoder.encode(&tokens1).unwrap();
        let emb2 = encoder.encode(&tokens2).unwrap();

        assert_eq!(emb1.len(), 32);
        assert_eq!(emb2.len(), 32);

        // Both should be normalized
        let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm1 - 1.0).abs() < 0.001);
        assert!((norm2 - 1.0).abs() < 0.001);

        // Compute cosine similarity
        let sim: f32 = emb1.iter().zip(&emb2).map(|(a, b)| a * b).sum();
        assert!(sim >= -1.0 && sim <= 1.0);
    }

    #[test]
    fn test_all_pooling_strategies_with_projection() {
        let strategies = vec![PoolingStrategy::Cls, PoolingStrategy::Mean, PoolingStrategy::Max];

        for strategy in strategies {
            let encoder = BiEncoder::new(8, strategy, true, Some(4));
            let tokens = vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            ];

            let emb = encoder.encode(&tokens).unwrap();
            assert_eq!(emb.len(), 4);

            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001);
        }
    }
}
