//! Model serving and inference service.

use crate::model::{BiEncoder, PoolingStrategy};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// Server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address.
    pub host: String,
    /// Port number.
    pub port: u16,
    /// Number of worker threads.
    pub workers: usize,
    /// Request timeout in seconds.
    pub timeout_seconds: u64,
    /// Maximum batch size.
    pub max_batch_size: usize,
    /// Enable CORS.
    pub enable_cors: bool,
    /// Model path.
    pub model_path: Option<PathBuf>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            workers: 4,
            timeout_seconds: 30,
            max_batch_size: 64,
            enable_cors: true,
            model_path: None,
        }
    }
}

/// Embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    /// Input texts to encode.
    pub texts: Vec<String>,
    /// Whether to normalize embeddings.
    pub normalize: bool,
    /// Pooling strategy.
    pub pooling: Option<String>,
}

/// Embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// Embeddings for each input text.
    pub embeddings: Vec<Vec<f32>>,
    /// Embedding dimension.
    pub dimension: usize,
    /// Number of texts encoded.
    pub count: usize,
    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
}

/// Similarity request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityRequest {
    /// Query text.
    pub query: String,
    /// Candidate texts to compare against.
    pub candidates: Vec<String>,
    /// Top-k results to return.
    pub top_k: Option<usize>,
}

/// Similarity result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResult {
    /// Candidate text.
    pub text: String,
    /// Similarity score.
    pub score: f32,
    /// Rank (1-indexed).
    pub rank: usize,
}

/// Similarity response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityResponse {
    /// Query text.
    pub query: String,
    /// Ranked results.
    pub results: Vec<SimilarityResult>,
    /// Processing time in milliseconds.
    pub processing_time_ms: u64,
}

/// Health check response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status.
    pub status: String,
    /// Model loaded.
    pub model_loaded: bool,
    /// Model dimension.
    pub model_dimension: Option<usize>,
    /// Uptime in seconds.
    pub uptime_seconds: u64,
    /// Version.
    pub version: String,
}

/// Model statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelStats {
    /// Total requests served.
    pub total_requests: u64,
    /// Total texts encoded.
    pub total_texts_encoded: u64,
    /// Average latency in milliseconds.
    pub avg_latency_ms: f64,
    /// 95th percentile latency.
    pub p95_latency_ms: f64,
    /// 99th percentile latency.
    pub p99_latency_ms: f64,
    /// Errors count.
    pub errors: u64,
}

/// Embedding service for serving model predictions.
pub struct EmbeddingService {
    /// Configuration.
    config: ServerConfig,
    /// Encoder model.
    encoder: Option<Arc<BiEncoder>>,
    /// Start time.
    start_time: std::time::Instant,
    /// Statistics.
    stats: ModelStats,
    /// Latency history for percentile calculations.
    latency_history: Vec<f64>,
}

impl EmbeddingService {
    /// Create a new embedding service.
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            encoder: None,
            start_time: std::time::Instant::now(),
            stats: ModelStats::default(),
            latency_history: Vec::new(),
        }
    }

    /// Load model from path.
    pub fn load_model(&mut self, path: &PathBuf) -> Result<()> {
        // In a real implementation, this would load model weights
        // For now, we create a default encoder
        let encoder = BiEncoder::new(768, PoolingStrategy::Mean, true, None);
        self.encoder = Some(Arc::new(encoder));
        Ok(())
    }

    /// Check if model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.encoder.is_some()
    }

    /// Get model dimension.
    pub fn model_dimension(&self) -> Option<usize> {
        self.encoder.as_ref().map(|e| e.dim())
    }

    /// Encode texts to embeddings.
    pub fn encode(&mut self, request: &EmbeddingRequest) -> Result<EmbeddingResponse> {
        let start = std::time::Instant::now();

        let encoder = self
            .encoder
            .as_ref()
            .ok_or(Error::InvalidConfig("Model not loaded".to_string()))?;

        if request.texts.is_empty() {
            return Err(Error::EmptyData);
        }

        if request.texts.len() > self.config.max_batch_size {
            return Err(Error::InvalidConfig(format!(
                "Batch size {} exceeds maximum {}",
                request.texts.len(),
                self.config.max_batch_size
            )));
        }

        let dimension = encoder.dim();
        let embeddings: Vec<Vec<f32>> = request
            .texts
            .iter()
            .filter_map(|text| {
                encoder.encode_text(text).ok().map(|emb| {
                    if request.normalize {
                        normalize_vec(&emb)
                    } else {
                        emb
                    }
                })
            })
            .collect();

        let processing_time_ms = start.elapsed().as_millis() as u64;

        // Update stats
        self.stats.total_requests += 1;
        self.stats.total_texts_encoded += request.texts.len() as u64;
        self.record_latency(processing_time_ms as f64);

        Ok(EmbeddingResponse {
            dimension,
            count: embeddings.len(),
            embeddings,
            processing_time_ms,
        })
    }

    /// Compute similarity between query and candidates.
    pub fn similarity(&mut self, request: &SimilarityRequest) -> Result<SimilarityResponse> {
        let start = std::time::Instant::now();

        let encoder = self
            .encoder
            .as_ref()
            .ok_or(Error::InvalidConfig("Model not loaded".to_string()))?;

        let query_emb = encoder.encode_text(&request.query)?;
        let query_emb = normalize_vec(&query_emb);

        let mut results: Vec<SimilarityResult> = request
            .candidates
            .iter()
            .enumerate()
            .filter_map(|(i, text)| {
                encoder.encode_text(text).ok().map(|emb| (i, text, emb))
            })
            .map(|(i, text, candidate_emb)| {
                let candidate_emb = normalize_vec(&candidate_emb);
                let score = cosine_similarity(&query_emb, &candidate_emb);

                SimilarityResult {
                    text: text.clone(),
                    score,
                    rank: i + 1, // Will be updated after sorting
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Update ranks
        for (i, result) in results.iter_mut().enumerate() {
            result.rank = i + 1;
        }

        // Apply top-k
        let top_k = request.top_k.unwrap_or(results.len());
        results.truncate(top_k);

        let processing_time_ms = start.elapsed().as_millis() as u64;
        self.stats.total_requests += 1;
        self.record_latency(processing_time_ms as f64);

        Ok(SimilarityResponse {
            query: request.query.clone(),
            results,
            processing_time_ms,
        })
    }

    /// Get health status.
    pub fn health(&self) -> HealthResponse {
        HealthResponse {
            status: "healthy".to_string(),
            model_loaded: self.is_model_loaded(),
            model_dimension: self.model_dimension(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    /// Get model statistics.
    pub fn stats(&self) -> &ModelStats {
        &self.stats
    }

    /// Record latency for percentile tracking.
    fn record_latency(&mut self, latency_ms: f64) {
        self.latency_history.push(latency_ms);

        // Keep last 1000 samples
        if self.latency_history.len() > 1000 {
            self.latency_history.remove(0);
        }

        // Update stats
        let n = self.latency_history.len() as f64;
        self.stats.avg_latency_ms =
            self.latency_history.iter().sum::<f64>() / n;

        // Calculate percentiles
        let mut sorted = self.latency_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let p95_idx = ((n * 0.95) as usize).min(sorted.len() - 1);
        let p99_idx = ((n * 0.99) as usize).min(sorted.len() - 1);

        self.stats.p95_latency_ms = sorted[p95_idx];
        self.stats.p99_latency_ms = sorted[p99_idx];
    }
}

/// Normalize a vector to unit length.
fn normalize_vec(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Batch inference configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum batch size.
    pub max_batch_size: usize,
    /// Maximum wait time in milliseconds.
    pub max_wait_ms: u64,
    /// Enable dynamic batching.
    pub dynamic_batching: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_wait_ms: 10,
            dynamic_batching: true,
        }
    }
}

/// ONNX export configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxExportConfig {
    /// Output path.
    pub output_path: PathBuf,
    /// Opset version.
    pub opset_version: u32,
    /// Enable dynamic axes.
    pub dynamic_axes: bool,
    /// Input names.
    pub input_names: Vec<String>,
    /// Output names.
    pub output_names: Vec<String>,
}

impl Default for OnnxExportConfig {
    fn default() -> Self {
        Self {
            output_path: PathBuf::from("./model.onnx"),
            opset_version: 14,
            dynamic_axes: true,
            input_names: vec!["input_ids".to_string(), "attention_mask".to_string()],
            output_names: vec!["embeddings".to_string()],
        }
    }
}

/// Model quantization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationConfig {
    /// Quantization type.
    pub quant_type: QuantizationType,
    /// Calibration samples.
    pub calibration_samples: usize,
    /// Per-channel quantization.
    pub per_channel: bool,
}

/// Quantization type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// Dynamic quantization (INT8).
    DynamicInt8,
    /// Static quantization (INT8).
    StaticInt8,
    /// Float16 quantization.
    Float16,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quant_type: QuantizationType::DynamicInt8,
            calibration_samples: 100,
            per_channel: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.workers, 4);
        assert!(config.enable_cors);
    }

    #[test]
    fn test_embedding_service_creation() {
        let config = ServerConfig::default();
        let service = EmbeddingService::new(config);
        assert!(!service.is_model_loaded());
    }

    #[test]
    fn test_embedding_service_health() {
        let config = ServerConfig::default();
        let service = EmbeddingService::new(config);
        let health = service.health();
        assert_eq!(health.status, "healthy");
        assert!(!health.model_loaded);
    }

    #[test]
    fn test_normalize_vec() {
        let v = vec![3.0, 4.0];
        let norm = normalize_vec(&v);
        let magnitude: f32 = norm.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_zero_vec() {
        let v = vec![0.0, 0.0];
        let norm = normalize_vec(&v);
        assert_eq!(norm, vec![0.0, 0.0]);
    }

    #[test]
    fn test_cosine_similarity_same() {
        let v = vec![0.6, 0.8];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_similarity_result_ordering() {
        let mut results = vec![
            SimilarityResult {
                text: "a".to_string(),
                score: 0.5,
                rank: 1,
            },
            SimilarityResult {
                text: "b".to_string(),
                score: 0.9,
                rank: 2,
            },
            SimilarityResult {
                text: "c".to_string(),
                score: 0.7,
                rank: 3,
            },
        ];

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        assert_eq!(results[0].text, "b");
        assert_eq!(results[1].text, "c");
        assert_eq!(results[2].text, "a");
    }

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 32);
        assert!(config.dynamic_batching);
    }

    #[test]
    fn test_onnx_export_config_default() {
        let config = OnnxExportConfig::default();
        assert_eq!(config.opset_version, 14);
        assert!(config.dynamic_axes);
    }

    #[test]
    fn test_quantization_config_default() {
        let config = QuantizationConfig::default();
        assert_eq!(config.quant_type, QuantizationType::DynamicInt8);
        assert_eq!(config.calibration_samples, 100);
    }

    #[test]
    fn test_model_stats_default() {
        let stats = ModelStats::default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.errors, 0);
    }

    #[test]
    fn test_embedding_request_serialize() {
        let request = EmbeddingRequest {
            texts: vec!["hello".to_string(), "world".to_string()],
            normalize: true,
            pooling: Some("mean".to_string()),
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: EmbeddingRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.texts.len(), 2);
        assert!(parsed.normalize);
    }

    #[test]
    fn test_similarity_request_serialize() {
        let request = SimilarityRequest {
            query: "test query".to_string(),
            candidates: vec!["a".to_string(), "b".to_string()],
            top_k: Some(5),
        };

        let json = serde_json::to_string(&request).unwrap();
        let parsed: SimilarityRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.query, "test query");
        assert_eq!(parsed.top_k, Some(5));
    }
}
