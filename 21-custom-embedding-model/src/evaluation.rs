//! Evaluation pipeline for embedding models.

use crate::dataset::cosine_similarity;
use crate::model::BiEncoder;
use crate::{Error, Result};
use std::collections::{HashMap, HashSet};

/// Retrieval evaluation metrics.
#[derive(Debug, Clone, Default)]
pub struct RetrievalMetrics {
    /// Recall at various K values.
    pub recall: HashMap<usize, f32>,
    /// Precision at various K values.
    pub precision: HashMap<usize, f32>,
    /// NDCG at various K values.
    pub ndcg: HashMap<usize, f32>,
    /// Mean Reciprocal Rank.
    pub mrr: f32,
    /// Mean Average Precision.
    pub map: f32,
}

impl RetrievalMetrics {
    /// Create new metrics with default K values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Format metrics for display.
    pub fn format(&self) -> String {
        let mut output = String::from("Retrieval Metrics:\n");
        output.push_str(&format!("  MRR: {:.4}\n", self.mrr));
        output.push_str(&format!("  MAP: {:.4}\n", self.map));

        let mut k_values: Vec<_> = self.recall.keys().collect();
        k_values.sort();

        for &k in &k_values {
            output.push_str(&format!(
                "  Recall@{}: {:.4} | Precision@{}: {:.4} | NDCG@{}: {:.4}\n",
                k, self.recall.get(&k).unwrap_or(&0.0),
                k, self.precision.get(&k).unwrap_or(&0.0),
                k, self.ndcg.get(&k).unwrap_or(&0.0)
            ));
        }

        output
    }
}

/// Embedding evaluator.
pub struct EmbeddingEvaluator {
    /// K values for metrics.
    k_values: Vec<usize>,
}

impl EmbeddingEvaluator {
    /// Create new evaluator.
    pub fn new(k_values: Vec<usize>) -> Self {
        Self { k_values }
    }

    /// Evaluate retrieval metrics.
    pub fn evaluate_retrieval(
        &self,
        query_embeddings: &[Vec<f32>],
        corpus_embeddings: &[Vec<f32>],
        relevance: &HashMap<usize, Vec<usize>>, // query_idx -> [relevant_doc_indices]
    ) -> RetrievalMetrics {
        let mut metrics = RetrievalMetrics::new();

        // Initialize metric accumulators
        let mut recall_sums: HashMap<usize, f32> = HashMap::new();
        let mut precision_sums: HashMap<usize, f32> = HashMap::new();
        let mut ndcg_sums: HashMap<usize, f32> = HashMap::new();
        let mut mrr_sum = 0.0f32;
        let mut map_sum = 0.0f32;

        for &k in &self.k_values {
            recall_sums.insert(k, 0.0);
            precision_sums.insert(k, 0.0);
            ndcg_sums.insert(k, 0.0);
        }

        let num_queries = query_embeddings.len();

        for (q_idx, query_emb) in query_embeddings.iter().enumerate() {
            // Get relevant documents for this query
            let rel_docs: HashSet<usize> = relevance
                .get(&q_idx)
                .map(|v| v.iter().copied().collect())
                .unwrap_or_default();

            if rel_docs.is_empty() {
                continue;
            }

            // Compute similarities to all corpus documents
            let mut scores: Vec<(usize, f32)> = corpus_embeddings
                .iter()
                .enumerate()
                .map(|(idx, doc_emb)| (idx, cosine_similarity(query_emb, doc_emb)))
                .collect();

            // Sort by similarity descending. `total_cmp` provides a total order
            // even for NaN scores, so malformed embeddings cannot panic here.
            scores.sort_by(|a, b| b.1.total_cmp(&a.1));

            // Get retrieved document indices
            let retrieved: Vec<usize> = scores.iter().map(|(idx, _)| *idx).collect();

            // Calculate metrics for each K
            for &k in &self.k_values {
                let top_k: HashSet<usize> = retrieved.iter().take(k).copied().collect();

                // Recall@K. `entry().or_insert()` is used instead of
                // `get_mut().unwrap()` so the accumulation is panic-free even if
                // `k_values` were somehow not pre-seeded.
                let hits = top_k.intersection(&rel_docs).count();
                let recall = hits as f32 / rel_docs.len() as f32;
                *recall_sums.entry(k).or_insert(0.0) += recall;

                // Precision@K
                let precision = hits as f32 / k as f32;
                *precision_sums.entry(k).or_insert(0.0) += precision;

                // NDCG@K
                let ndcg = self.compute_ndcg(&retrieved, &rel_docs, k);
                *ndcg_sums.entry(k).or_insert(0.0) += ndcg;
            }

            // MRR (Mean Reciprocal Rank)
            for (rank, &doc_idx) in retrieved.iter().enumerate() {
                if rel_docs.contains(&doc_idx) {
                    mrr_sum += 1.0 / (rank + 1) as f32;
                    break;
                }
            }

            // MAP (Mean Average Precision)
            let ap = self.compute_average_precision(&retrieved, &rel_docs);
            map_sum += ap;
        }

        // Average metrics
        let n = num_queries as f32;
        for &k in &self.k_values {
            metrics.recall.insert(k, recall_sums[&k] / n);
            metrics.precision.insert(k, precision_sums[&k] / n);
            metrics.ndcg.insert(k, ndcg_sums[&k] / n);
        }
        metrics.mrr = mrr_sum / n;
        metrics.map = map_sum / n;

        metrics
    }

    /// Compute NDCG@K.
    fn compute_ndcg(&self, retrieved: &[usize], relevant: &HashSet<usize>, k: usize) -> f32 {
        // DCG
        let dcg: f32 = retrieved
            .iter()
            .take(k)
            .enumerate()
            .map(|(rank, &doc_idx)| {
                if relevant.contains(&doc_idx) {
                    1.0 / (rank as f32 + 2.0).log2()
                } else {
                    0.0
                }
            })
            .sum();

        // Ideal DCG
        let ideal_k = k.min(relevant.len());
        let idcg: f32 = (0..ideal_k)
            .map(|rank| 1.0 / (rank as f32 + 2.0).log2())
            .sum();

        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
    }

    /// Compute Average Precision.
    fn compute_average_precision(&self, retrieved: &[usize], relevant: &HashSet<usize>) -> f32 {
        let mut precisions = Vec::new();
        let mut num_relevant = 0;

        for (rank, &doc_idx) in retrieved.iter().enumerate() {
            if relevant.contains(&doc_idx) {
                num_relevant += 1;
                precisions.push(num_relevant as f32 / (rank + 1) as f32);
            }
        }

        if relevant.is_empty() {
            0.0
        } else {
            precisions.iter().sum::<f32>() / relevant.len() as f32
        }
    }

    /// Evaluate clustering quality.
    pub fn evaluate_clustering(
        &self,
        embeddings: &[Vec<f32>],
        labels: &[usize],
    ) -> ClusteringMetrics {
        let n = embeddings.len();
        if n == 0 {
            return ClusteringMetrics::default();
        }

        // Compute pairwise distances
        let mut same_cluster_dists = Vec::new();
        let mut diff_cluster_dists = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let dist = 1.0 - cosine_similarity(&embeddings[i], &embeddings[j]);
                if labels[i] == labels[j] {
                    same_cluster_dists.push(dist);
                } else {
                    diff_cluster_dists.push(dist);
                }
            }
        }

        let intra_cluster_dist = if same_cluster_dists.is_empty() {
            0.0
        } else {
            same_cluster_dists.iter().sum::<f32>() / same_cluster_dists.len() as f32
        };

        let inter_cluster_dist = if diff_cluster_dists.is_empty() {
            1.0
        } else {
            diff_cluster_dists.iter().sum::<f32>() / diff_cluster_dists.len() as f32
        };

        // Silhouette-like score
        let separation = if inter_cluster_dist > 0.0 {
            (inter_cluster_dist - intra_cluster_dist) / inter_cluster_dist.max(intra_cluster_dist)
        } else {
            0.0
        };

        ClusteringMetrics {
            intra_cluster_distance: intra_cluster_dist,
            inter_cluster_distance: inter_cluster_dist,
            separation_score: separation,
        }
    }
}

impl Default for EmbeddingEvaluator {
    fn default() -> Self {
        Self::new(vec![1, 5, 10, 20, 100])
    }
}

/// Clustering evaluation metrics.
#[derive(Debug, Clone, Default)]
pub struct ClusteringMetrics {
    /// Average intra-cluster distance.
    pub intra_cluster_distance: f32,
    /// Average inter-cluster distance.
    pub inter_cluster_distance: f32,
    /// Separation score (higher is better).
    pub separation_score: f32,
}

impl ClusteringMetrics {
    /// Format metrics for display.
    pub fn format(&self) -> String {
        format!(
            "Clustering Metrics:\n  Intra-cluster dist: {:.4}\n  Inter-cluster dist: {:.4}\n  Separation: {:.4}",
            self.intra_cluster_distance,
            self.inter_cluster_distance,
            self.separation_score
        )
    }
}

/// Semantic similarity benchmark.
pub struct SemanticSimilarityBenchmark {
    /// Sentence pairs.
    sentence_pairs: Vec<(String, String)>,
    /// Gold standard scores.
    gold_scores: Vec<f32>,
}

impl SemanticSimilarityBenchmark {
    /// Create new benchmark.
    pub fn new(pairs: Vec<(String, String)>, scores: Vec<f32>) -> Self {
        Self {
            sentence_pairs: pairs,
            gold_scores: scores,
        }
    }

    /// Evaluate on benchmark.
    pub fn evaluate(&self, embeddings1: &[Vec<f32>], embeddings2: &[Vec<f32>]) -> f32 {
        if embeddings1.len() != embeddings2.len() || embeddings1.len() != self.gold_scores.len() {
            return 0.0;
        }

        // Compute predicted similarities
        let pred_scores: Vec<f32> = embeddings1
            .iter()
            .zip(embeddings2)
            .map(|(e1, e2)| cosine_similarity(e1, e2))
            .collect();

        // Compute Spearman correlation
        spearman_correlation(&pred_scores, &self.gold_scores)
    }
}

/// Compute Spearman rank correlation.
fn spearman_correlation(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len();

    // Convert to ranks
    let x_ranks = to_ranks(x);
    let y_ranks = to_ranks(y);

    // Compute Pearson correlation on ranks
    let mean_x: f32 = x_ranks.iter().sum::<f32>() / n as f32;
    let mean_y: f32 = y_ranks.iter().sum::<f32>() / n as f32;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for i in 0..n {
        let dx = x_ranks[i] - mean_x;
        let dy = y_ranks[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x == 0.0 || var_y == 0.0 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Convert values to ranks.
fn to_ranks(values: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = values.iter().cloned().enumerate().collect();
    // `total_cmp` avoids panicking on NaN values.
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut ranks = vec![0.0; values.len()];
    for (rank, (idx, _)) in indexed.into_iter().enumerate() {
        ranks[idx] = (rank + 1) as f32;
    }

    ranks
}

/// Embedding drift detector.
pub struct DriftDetector {
    /// Reference embeddings statistics.
    reference_mean: Vec<f32>,
    reference_std: Vec<f32>,
    reference_pairwise_sim: f32,
}

impl DriftDetector {
    /// Create drift detector from reference embeddings.
    pub fn new(reference_embeddings: &[Vec<f32>]) -> Self {
        let dim = if reference_embeddings.is_empty() {
            0
        } else {
            reference_embeddings[0].len()
        };

        let n = reference_embeddings.len();

        // Compute mean
        let mut mean = vec![0.0; dim];
        for emb in reference_embeddings {
            for (i, &v) in emb.iter().enumerate() {
                mean[i] += v;
            }
        }
        for v in &mut mean {
            *v /= n as f32;
        }

        // Compute std
        let mut std = vec![0.0; dim];
        for emb in reference_embeddings {
            for (i, &v) in emb.iter().enumerate() {
                std[i] += (v - mean[i]).powi(2);
            }
        }
        for v in &mut std {
            *v = (*v / n as f32).sqrt();
        }

        // Compute mean pairwise similarity
        let mut total_sim = 0.0f32;
        let mut count = 0;
        for i in 0..n.min(100) {
            for j in (i + 1)..n.min(100) {
                total_sim += cosine_similarity(&reference_embeddings[i], &reference_embeddings[j]);
                count += 1;
            }
        }
        let pairwise_sim = if count > 0 { total_sim / count as f32 } else { 0.0 };

        Self {
            reference_mean: mean,
            reference_std: std,
            reference_pairwise_sim: pairwise_sim,
        }
    }

    /// Detect drift in current embeddings.
    pub fn detect_drift(&self, current_embeddings: &[Vec<f32>]) -> DriftReport {
        let dim = self.reference_mean.len();
        let n = current_embeddings.len();

        // Compute current mean
        let mut current_mean = vec![0.0; dim];
        for emb in current_embeddings {
            for (i, &v) in emb.iter().enumerate() {
                current_mean[i] += v;
            }
        }
        for v in &mut current_mean {
            *v /= n as f32;
        }

        // Mean drift (cosine distance)
        let mean_drift = 1.0 - cosine_similarity(&self.reference_mean, &current_mean);

        // Compute current pairwise similarity
        let mut total_sim = 0.0f32;
        let mut count = 0;
        for i in 0..n.min(100) {
            for j in (i + 1)..n.min(100) {
                total_sim += cosine_similarity(&current_embeddings[i], &current_embeddings[j]);
                count += 1;
            }
        }
        let current_pairwise_sim = if count > 0 { total_sim / count as f32 } else { 0.0 };

        let pairwise_sim_drift = (current_pairwise_sim - self.reference_pairwise_sim).abs();

        let drift_detected = mean_drift > 0.1 || pairwise_sim_drift > 0.05;

        DriftReport {
            mean_drift,
            pairwise_sim_drift,
            drift_detected,
        }
    }
}

/// Drift detection report.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// Mean vector drift (cosine distance).
    pub mean_drift: f32,
    /// Pairwise similarity drift.
    pub pairwise_sim_drift: f32,
    /// Whether drift was detected.
    pub drift_detected: bool,
}

impl DriftReport {
    /// Format report for display.
    pub fn format(&self) -> String {
        format!(
            "Drift Report:\n  Mean drift: {:.4}\n  Pairwise sim drift: {:.4}\n  Drift detected: {}",
            self.mean_drift,
            self.pairwise_sim_drift,
            self.drift_detected
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::random_embeddings;

    // =============================================================================
    // RetrievalMetrics Tests
    // =============================================================================

    #[test]
    fn test_retrieval_metrics_new() {
        let metrics = RetrievalMetrics::new();

        assert!(metrics.recall.is_empty());
        assert!(metrics.precision.is_empty());
        assert!(metrics.ndcg.is_empty());
        assert!((metrics.mrr - 0.0).abs() < 0.001);
        assert!((metrics.map - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_retrieval_metrics_default() {
        let metrics = RetrievalMetrics::default();
        assert!(metrics.recall.is_empty());
    }

    #[test]
    fn test_retrieval_metrics_format() {
        let mut metrics = RetrievalMetrics::new();
        metrics.mrr = 0.75;
        metrics.map = 0.65;
        metrics.recall.insert(1, 0.5);
        metrics.recall.insert(10, 0.8);
        metrics.precision.insert(1, 0.5);
        metrics.precision.insert(10, 0.08);
        metrics.ndcg.insert(1, 0.5);
        metrics.ndcg.insert(10, 0.7);

        let formatted = metrics.format();
        assert!(formatted.contains("MRR:"));
        assert!(formatted.contains("MAP:"));
        assert!(formatted.contains("Recall@"));
        assert!(formatted.contains("Precision@"));
        assert!(formatted.contains("NDCG@"));
    }

    // =============================================================================
    // EmbeddingEvaluator Tests
    // =============================================================================

    #[test]
    fn test_evaluator_creation() {
        let evaluator = EmbeddingEvaluator::new(vec![1, 5, 10, 20]);
        assert!(true); // Just check creation succeeds
    }

    #[test]
    fn test_evaluator_default() {
        let evaluator = EmbeddingEvaluator::default();
        // Default k values are [1, 5, 10, 20, 100]
        assert!(true);
    }

    #[test]
    fn test_evaluate_retrieval_basic() {
        let evaluator = EmbeddingEvaluator::new(vec![1, 5, 10]);

        // Create query and corpus embeddings
        let query_embs = random_embeddings(10, 64, true);
        let corpus_embs = random_embeddings(100, 64, true);

        // Create relevance judgments
        let mut relevance = HashMap::new();
        for i in 0..10 {
            relevance.insert(i, vec![i % 100]);
        }

        let metrics = evaluator.evaluate_retrieval(&query_embs, &corpus_embs, &relevance);

        assert!(metrics.mrr >= 0.0 && metrics.mrr <= 1.0);
        assert!(metrics.map >= 0.0 && metrics.map <= 1.0);
        assert!(metrics.recall.contains_key(&1));
        assert!(metrics.recall.contains_key(&5));
        assert!(metrics.recall.contains_key(&10));
    }

    #[test]
    fn test_evaluate_retrieval_perfect_match() {
        let evaluator = EmbeddingEvaluator::new(vec![1, 5]);

        // Query embeddings same as first few corpus embeddings
        let query_embs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let corpus_embs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let mut relevance = HashMap::new();
        relevance.insert(0, vec![0]); // Query 0 is relevant to corpus 0
        relevance.insert(1, vec![1]); // Query 1 is relevant to corpus 1

        let metrics = evaluator.evaluate_retrieval(&query_embs, &corpus_embs, &relevance);

        // Should have perfect metrics
        assert!((metrics.mrr - 1.0).abs() < 0.001);
        assert!((metrics.map - 1.0).abs() < 0.001);
        assert!((metrics.recall[&1] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_evaluate_retrieval_no_relevance() {
        let evaluator = EmbeddingEvaluator::new(vec![1, 5]);

        let query_embs = random_embeddings(5, 64, true);
        let corpus_embs = random_embeddings(50, 64, true);

        // Empty relevance - no relevant docs
        let relevance = HashMap::new();

        let metrics = evaluator.evaluate_retrieval(&query_embs, &corpus_embs, &relevance);

        // Metrics should be 0 or NaN-safe
        assert!(metrics.mrr >= 0.0);
    }

    #[test]
    fn test_evaluate_retrieval_multiple_relevant() {
        let evaluator = EmbeddingEvaluator::new(vec![1, 5, 10]);

        let query_embs = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let corpus_embs = vec![
            vec![0.9, 0.1, 0.0, 0.0], // Similar to query
            vec![0.8, 0.2, 0.0, 0.0], // Somewhat similar
            vec![0.0, 1.0, 0.0, 0.0], // Orthogonal
            vec![0.0, 0.0, 1.0, 0.0], // Orthogonal
        ];

        let mut relevance = HashMap::new();
        relevance.insert(0, vec![0, 1]); // First two are relevant

        let metrics = evaluator.evaluate_retrieval(&query_embs, &corpus_embs, &relevance);

        assert!(metrics.mrr > 0.0);
        assert!(metrics.recall[&5] > 0.0);
    }

    // =============================================================================
    // NDCG Tests
    // =============================================================================

    #[test]
    fn test_compute_ndcg_perfect() {
        let evaluator = EmbeddingEvaluator::new(vec![5]);

        let retrieved = vec![0, 1, 2, 3, 4];
        let relevant: HashSet<usize> = vec![0, 1, 2].into_iter().collect();

        let ndcg = evaluator.compute_ndcg(&retrieved, &relevant, 5);
        // First 3 are relevant and in first 3 positions
        assert!((ndcg - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_ndcg_no_relevant() {
        let evaluator = EmbeddingEvaluator::new(vec![5]);

        let retrieved = vec![0, 1, 2, 3, 4];
        let relevant: HashSet<usize> = HashSet::new();

        let ndcg = evaluator.compute_ndcg(&retrieved, &relevant, 5);
        assert!((ndcg - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_ndcg_late_relevant() {
        let evaluator = EmbeddingEvaluator::new(vec![5]);

        let retrieved = vec![10, 11, 12, 0, 1]; // Relevant docs at positions 4,5
        let relevant: HashSet<usize> = vec![0, 1].into_iter().collect();

        let ndcg = evaluator.compute_ndcg(&retrieved, &relevant, 5);
        // NDCG should be less than 1 since relevant docs are late
        assert!(ndcg < 1.0);
        assert!(ndcg > 0.0);
    }

    // =============================================================================
    // Average Precision Tests
    // =============================================================================

    #[test]
    fn test_compute_ap_perfect() {
        let evaluator = EmbeddingEvaluator::new(vec![5]);

        let retrieved = vec![0, 1, 2, 3, 4];
        let relevant: HashSet<usize> = vec![0, 1, 2].into_iter().collect();

        let ap = evaluator.compute_average_precision(&retrieved, &relevant);
        // Precision at each relevant position: 1/1, 2/2, 3/3 = 1, 1, 1
        // AP = (1 + 1 + 1) / 3 = 1.0
        assert!((ap - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_ap_mixed() {
        let evaluator = EmbeddingEvaluator::new(vec![5]);

        let retrieved = vec![0, 10, 1, 11, 2]; // Relevant: 0, 1, 2 at positions 1, 3, 5
        let relevant: HashSet<usize> = vec![0, 1, 2].into_iter().collect();

        let ap = evaluator.compute_average_precision(&retrieved, &relevant);
        // Precision at relevant positions: 1/1=1, 2/3=0.67, 3/5=0.6
        // AP = (1 + 0.67 + 0.6) / 3 = 0.756
        assert!(ap > 0.5 && ap < 1.0);
    }

    #[test]
    fn test_compute_ap_no_relevant() {
        let evaluator = EmbeddingEvaluator::new(vec![5]);

        let retrieved = vec![0, 1, 2, 3, 4];
        let relevant: HashSet<usize> = HashSet::new();

        let ap = evaluator.compute_average_precision(&retrieved, &relevant);
        assert!((ap - 0.0).abs() < 0.001);
    }

    // =============================================================================
    // ClusteringMetrics Tests
    // =============================================================================

    #[test]
    fn test_clustering_metrics_default() {
        let metrics = ClusteringMetrics::default();

        assert!((metrics.intra_cluster_distance - 0.0).abs() < 0.001);
        assert!((metrics.inter_cluster_distance - 0.0).abs() < 0.001);
        assert!((metrics.separation_score - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_clustering_metrics_format() {
        let mut metrics = ClusteringMetrics::default();
        metrics.intra_cluster_distance = 0.1;
        metrics.inter_cluster_distance = 0.8;
        metrics.separation_score = 0.7;

        let formatted = metrics.format();
        assert!(formatted.contains("Intra-cluster"));
        assert!(formatted.contains("Inter-cluster"));
        assert!(formatted.contains("Separation"));
    }

    #[test]
    fn test_evaluate_clustering_basic() {
        let evaluator = EmbeddingEvaluator::default();

        // Create embeddings with clear clusters
        let mut embeddings = Vec::new();
        let mut labels = Vec::new();

        // Cluster 0
        for _ in 0..5 {
            let mut emb = vec![1.0, 0.0, 0.0, 0.0];
            emb[0] += rand::random::<f32>() * 0.1;
            embeddings.push(emb);
            labels.push(0);
        }

        // Cluster 1
        for _ in 0..5 {
            let mut emb = vec![0.0, 1.0, 0.0, 0.0];
            emb[1] += rand::random::<f32>() * 0.1;
            embeddings.push(emb);
            labels.push(1);
        }

        let metrics = evaluator.evaluate_clustering(&embeddings, &labels);

        // Intra-cluster distance should be small
        assert!(metrics.intra_cluster_distance < 0.2);
        // Inter-cluster distance should be large
        assert!(metrics.inter_cluster_distance > 0.5);
        // Separation should be positive
        assert!(metrics.separation_score > 0.0);
    }

    #[test]
    fn test_evaluate_clustering_empty() {
        let evaluator = EmbeddingEvaluator::default();

        let embeddings: Vec<Vec<f32>> = vec![];
        let labels: Vec<usize> = vec![];

        let metrics = evaluator.evaluate_clustering(&embeddings, &labels);

        // Should return default/zero metrics
        assert!((metrics.intra_cluster_distance - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_evaluate_clustering_single_cluster() {
        let evaluator = EmbeddingEvaluator::default();

        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.8, 0.2, 0.0],
        ];
        let labels = vec![0, 0, 0]; // All same cluster

        let metrics = evaluator.evaluate_clustering(&embeddings, &labels);

        // Only intra-cluster, no inter-cluster
        assert!(metrics.intra_cluster_distance >= 0.0);
    }

    // =============================================================================
    // Spearman Correlation Tests
    // =============================================================================

    #[test]
    fn test_spearman_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let corr = spearman_correlation(&x, &y);
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_spearman_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        let corr = spearman_correlation(&x, &y);
        assert!((corr + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_spearman_no_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 4.0, 1.0, 3.0]; // Scrambled

        let corr = spearman_correlation(&x, &y);
        // Should be between -1 and 1
        assert!(corr >= -1.0 && corr <= 1.0);
    }

    #[test]
    fn test_spearman_empty() {
        let x: Vec<f32> = vec![];
        let y: Vec<f32> = vec![];

        let corr = spearman_correlation(&x, &y);
        assert!((corr - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_spearman_mismatched_length() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];

        let corr = spearman_correlation(&x, &y);
        assert!((corr - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_spearman_single_element() {
        let x = vec![1.0];
        let y = vec![1.0];

        let corr = spearman_correlation(&x, &y);
        // Single element - undefined/zero
        assert!(!corr.is_nan());
    }

    // =============================================================================
    // To Ranks Tests
    // =============================================================================

    #[test]
    fn test_to_ranks_basic() {
        let values = vec![3.0, 1.0, 2.0];
        let ranks = to_ranks(&values);

        assert!((ranks[0] - 3.0).abs() < 0.001); // 3.0 is largest -> rank 3
        assert!((ranks[1] - 1.0).abs() < 0.001); // 1.0 is smallest -> rank 1
        assert!((ranks[2] - 2.0).abs() < 0.001); // 2.0 is middle -> rank 2
    }

    #[test]
    fn test_to_ranks_already_sorted() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ranks = to_ranks(&values);

        for (i, r) in ranks.iter().enumerate() {
            assert!((*r - (i + 1) as f32).abs() < 0.001);
        }
    }

    #[test]
    fn test_to_ranks_reverse_sorted() {
        let values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let ranks = to_ranks(&values);

        assert!((ranks[0] - 5.0).abs() < 0.001);
        assert!((ranks[4] - 1.0).abs() < 0.001);
    }

    // =============================================================================
    // SemanticSimilarityBenchmark Tests
    // =============================================================================

    #[test]
    fn test_semantic_benchmark_creation() {
        let pairs = vec![
            ("hello".to_string(), "world".to_string()),
            ("foo".to_string(), "bar".to_string()),
        ];
        let scores = vec![0.8, 0.3];

        let benchmark = SemanticSimilarityBenchmark::new(pairs, scores);
        assert!(true); // Just verify creation
    }

    #[test]
    fn test_semantic_benchmark_evaluate() {
        let pairs = vec![
            ("a".to_string(), "b".to_string()),
            ("c".to_string(), "d".to_string()),
            ("e".to_string(), "f".to_string()),
        ];
        let gold_scores = vec![1.0, 0.5, 0.0];

        let benchmark = SemanticSimilarityBenchmark::new(pairs, gold_scores);

        // Create embeddings with matching similarity pattern
        let embs1 = vec![
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
        ];
        let embs2 = vec![
            vec![1.0, 0.0, 0.0],  // cos_sim = 1.0
            vec![0.7, 0.7, 0.0],  // cos_sim ~= 0.7
            vec![0.0, 1.0, 0.0],  // cos_sim = 0.0
        ];

        let correlation = benchmark.evaluate(&embs1, &embs2);
        // Should have positive correlation
        assert!(correlation > 0.0);
    }

    #[test]
    fn test_semantic_benchmark_mismatched_sizes() {
        let pairs = vec![
            ("a".to_string(), "b".to_string()),
        ];
        let gold_scores = vec![1.0];

        let benchmark = SemanticSimilarityBenchmark::new(pairs, gold_scores);

        let embs1 = vec![vec![1.0, 0.0]];
        let embs2 = vec![vec![1.0, 0.0], vec![0.0, 1.0]]; // Wrong size

        let correlation = benchmark.evaluate(&embs1, &embs2);
        assert!((correlation - 0.0).abs() < 0.001);
    }

    // =============================================================================
    // DriftDetector Tests
    // =============================================================================

    #[test]
    fn test_drift_detector_creation() {
        let reference = random_embeddings(100, 64, true);
        let detector = DriftDetector::new(&reference);
        assert!(true); // Just verify creation
    }

    #[test]
    fn test_drift_detector_empty_reference() {
        let reference: Vec<Vec<f32>> = vec![];
        let detector = DriftDetector::new(&reference);
        assert!(true); // Should handle empty case
    }

    #[test]
    fn test_drift_detector_no_drift() {
        // Same distribution
        let reference = random_embeddings(100, 64, true);
        let detector = DriftDetector::new(&reference);

        let current = random_embeddings(100, 64, true);
        let report = detector.detect_drift(&current);

        // Mean drift should be relatively small for random normalized vectors
        assert!(report.mean_drift >= 0.0);
    }

    #[test]
    fn test_drift_detector_clear_drift() {
        let reference = random_embeddings(100, 64, true);
        let detector = DriftDetector::new(&reference);

        // Create clearly different distribution
        let drifted: Vec<Vec<f32>> = (0..100)
            .map(|_| {
                let mut emb = vec![0.0; 64];
                emb[0] = 1.0;
                emb
            })
            .collect();

        let report = detector.detect_drift(&drifted);
        assert!(report.mean_drift > 0.0);
    }

    #[test]
    fn test_drift_report_format() {
        let report = DriftReport {
            mean_drift: 0.15,
            pairwise_sim_drift: 0.08,
            drift_detected: true,
        };

        let formatted = report.format();
        assert!(formatted.contains("Mean drift:"));
        assert!(formatted.contains("Pairwise sim drift:"));
        assert!(formatted.contains("Drift detected:"));
    }

    // =============================================================================
    // Integration Tests
    // =============================================================================

    #[test]
    fn test_full_evaluation_pipeline() {
        let evaluator = EmbeddingEvaluator::new(vec![1, 5, 10, 20]);

        // Generate synthetic data
        let num_queries = 20;
        let corpus_size = 200;
        let dim = 64;

        let query_embs = random_embeddings(num_queries, dim, true);
        let corpus_embs = random_embeddings(corpus_size, dim, true);

        // Create random relevance judgments
        let mut relevance = HashMap::new();
        for i in 0..num_queries {
            relevance.insert(i, vec![i % corpus_size, (i + 1) % corpus_size]);
        }

        // Evaluate retrieval
        let retrieval_metrics = evaluator.evaluate_retrieval(
            &query_embs,
            &corpus_embs,
            &relevance,
        );

        // Verify all metrics are computed
        assert!(retrieval_metrics.recall.contains_key(&1));
        assert!(retrieval_metrics.recall.contains_key(&5));
        assert!(retrieval_metrics.recall.contains_key(&10));
        assert!(retrieval_metrics.recall.contains_key(&20));
        assert!(retrieval_metrics.mrr >= 0.0 && retrieval_metrics.mrr <= 1.0);
        assert!(retrieval_metrics.map >= 0.0 && retrieval_metrics.map <= 1.0);

        // Evaluate clustering
        let cluster_labels: Vec<usize> = (0..corpus_size).map(|i| i % 5).collect();
        let clustering_metrics = evaluator.evaluate_clustering(&corpus_embs, &cluster_labels);

        assert!(clustering_metrics.intra_cluster_distance >= 0.0);
        assert!(clustering_metrics.inter_cluster_distance >= 0.0);
    }

    #[test]
    fn test_metrics_bounds() {
        let evaluator = EmbeddingEvaluator::new(vec![1, 5, 10]);

        let query_embs = random_embeddings(10, 32, true);
        let corpus_embs = random_embeddings(50, 32, true);

        let mut relevance = HashMap::new();
        for i in 0..10 {
            relevance.insert(i, vec![i % 50]);
        }

        let metrics = evaluator.evaluate_retrieval(&query_embs, &corpus_embs, &relevance);

        // All metrics should be in valid ranges
        for (_, recall) in &metrics.recall {
            assert!(*recall >= 0.0 && *recall <= 1.0);
        }
        for (_, precision) in &metrics.precision {
            assert!(*precision >= 0.0 && *precision <= 1.0);
        }
        for (_, ndcg) in &metrics.ndcg {
            assert!(*ndcg >= 0.0 && *ndcg <= 1.0);
        }
        assert!(metrics.mrr >= 0.0 && metrics.mrr <= 1.0);
        assert!(metrics.map >= 0.0 && metrics.map <= 1.0);
    }

    // =============================================================================
    // Robustness Tests (NaN inputs must not panic)
    // =============================================================================

    #[test]
    fn test_evaluate_retrieval_tolerates_nan_scores() {
        // A zero-norm corpus embedding produces NaN cosine similarities; sorting
        // must not panic (previously partial_cmp().unwrap()).
        let evaluator = EmbeddingEvaluator::new(vec![1, 5]);
        let queries = vec![vec![1.0, 0.0, 0.0]];
        let corpus = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0], // zero norm -> NaN similarity
            vec![0.0, 1.0, 0.0],
        ];
        let mut relevance = HashMap::new();
        relevance.insert(0usize, vec![0usize]);

        let metrics = evaluator.evaluate_retrieval(&queries, &corpus, &relevance);
        // Completes without panicking and produces finite aggregate metrics.
        assert!(metrics.mrr.is_finite());
    }
}
