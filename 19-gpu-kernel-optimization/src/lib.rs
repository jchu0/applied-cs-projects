//! GPU GEMM Kernel Optimization (cuBLAS-lite).
//!
//! Software implementations demonstrating GPU kernel optimization concepts
//! for General Matrix Multiplication (GEMM).

pub mod gemm;
pub mod matrix;
pub mod metrics;
pub mod autotuner;
pub mod vectorize;
pub mod memory;

use thiserror::Error;

/// Errors for GEMM operations.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for GEMM operations.
pub type Result<T> = std::result::Result<T, Error>;

// Re-exports
pub use gemm::{
    // Core GEMM functions
    naive_gemm, tiled_gemm, register_tiled_gemm, parallel_tiled_gemm,
    double_buffered_gemm, scaled_gemm,
    // Batched GEMM
    batched_gemm, batched_scaled_gemm, strided_batched_gemm,
    // Fused operations
    gemm_activation, gemm_bias_activation, gemm_fused, Activation,
    // Tensor Core / WMMA simulation
    wmma_gemm, wmma_mma_sync, WmmaConfig, WmmaFragment,
    // Roofline analysis
    RooflineModel, RooflineAnalysis, PerformanceBound,
    // Configuration
    GemmConfig, GemmKernel,
};
pub use matrix::Matrix;
pub use metrics::GemmMetrics;
pub use autotuner::Autotuner;
pub use vectorize::{Float4, Float8, VectorizedOps, SimdVectorOps, CoalescingAnalysis};
pub use memory::{
    BankConflictAnalysis, SharedMemoryConfig, OccupancyCalculator,
    KernelRequirements, OccupancyResult, MemoryAccessPattern,
};
