//! Custom Embedding Model Training System
//!
//! Production-grade embedding model training infrastructure supporting:
//! - Contrastive learning with hard negative mining
//! - Bi-encoder architecture with multiple pooling strategies
//! - Multiple loss functions (MNRL, Triplet)
//! - Comprehensive evaluation metrics (Recall@k, MRR, MAP, NDCG)
//! - MLOps integration with experiment tracking and model registry
//! - Model serving with REST API support
//! - Distributed training with DDP and FSDP
//! - ONNX export and runtime integration

pub mod dataset;
pub mod distributed;
pub mod evaluation;
pub mod loss;
pub mod mlops;
pub mod model;
pub mod onnx;
pub mod serving;
pub mod trainable;
pub mod trainer;

pub use dataset::{EmbeddingBatch, EmbeddingExample, HardNegativeMiner, MemoryBank};
pub use distributed::{
    DistributedConfig, DistributedDataParallel, DDPConfig, DDPStats,
    FullyShardedDataParallel, FSDPConfig, FSDPStats, ProcessGroup,
    DistributedSampler, ShardingStrategy,
};
pub use evaluation::{EmbeddingEvaluator, RetrievalMetrics};
pub use loss::{LossFunction, MultipleNegativesRankingLoss, TripletMarginLoss};
pub use mlops::{ExperimentConfig, ExperimentTracker, ModelRegistry, HyperparameterSearch};
pub use model::{BiEncoder, PoolingStrategy};
pub use onnx::{
    OnnxSession, SessionConfig, OnnxTensor, OnnxExporter, ExportConfig,
    OnnxQuantizer, QuantizationConfig, QuantizationMode, DynamicBatcher,
    ExecutionProvider, GraphOptimizationLevel,
};
pub use serving::{EmbeddingService, ServerConfig, EmbeddingRequest, EmbeddingResponse};
pub use trainable::{TrainableEmbedder, TrainingPair};
pub use trainer::{TrainerConfig, EmbeddingTrainer};

use thiserror::Error;

/// Result type for embedding operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for embedding system.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Evaluation error: {0}")]
    EvaluationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Empty data")]
    EmptyData,

    #[error("ONNX error: {0}")]
    OnnxError(String),

    #[error("Distributed error: {0}")]
    DistributedError(String),
}

/// Embedding dimension.
pub const EMBEDDING_DIM: usize = 768;

/// Default temperature for contrastive learning.
pub const DEFAULT_TEMPERATURE: f32 = 0.05;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(EMBEDDING_DIM > 0);
        assert!(DEFAULT_TEMPERATURE > 0.0);
    }
}
