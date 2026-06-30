//! High-Performance Columnar Analytics Engine with SIMD Vectorization
//!
//! This engine leverages CPU-optimized techniques for maximum analytics performance:
//! - SIMD vectorization for parallel data processing
//! - Cache-conscious algorithms with blocking and prefetching
//! - Multi-core parallel execution
//! - Aligned memory allocations
//! - NUMA-aware memory management
//! - Vectorized hashing and hash-based operations
//! - Cost-based query planning

pub mod aggregate;
pub mod column;
pub mod filter;
pub mod hash;
pub mod metrics;
pub mod numa;
pub mod optimize;
pub mod planner;
pub mod scheduler;
pub mod simd;

pub use aggregate::{Aggregator, AggregateOp, AggregateValue, PartialAggregate};
pub use column::{AlignedVec, Column, ColumnType, DataChunk};
pub use filter::{FilterOp, FilterPredicate, SelectionVector, VectorizedFilter};
pub use hash::{VectorizedHash, HashTable, HashAggregatorSimd, HashJoin, BloomFilter, CountMinSketch, HyperLogLog, AggState};
pub use metrics::{PerfMetrics, QueryMetrics, BenchmarkResult, Benchmark, Timer, PerformanceModel};
pub use numa::{NumaTopology, NumaAllocator, NumaPartitionedVec, NumaExecutor, AffinitySettings};
pub use optimize::{CacheBlockConfig, CacheBlockedOps, BranchFreeOps, StreamingOps, TiledOps, AutoTuner};
pub use planner::{CostModel, CostEstimate, QueryPlanner, QueryOp, PlanNode, HardwareParams, ColumnStats};
pub use scheduler::ParallelExecutor;
pub use simd::SimdOps;

use thiserror::Error;

/// Result type for analytics operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for the analytics engine.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Type mismatch: expected {expected}, got {got}")]
    TypeMismatch { expected: String, got: String },

    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Alignment error: {0}")]
    AlignmentError(String),

    #[error("Index out of bounds: {index} >= {len}")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("Empty data")]
    EmptyData,
}

/// Cache line size for alignment.
pub const CACHE_LINE_SIZE: usize = 64;

/// Vector processing width (elements per SIMD operation).
pub const VECTOR_WIDTH: usize = 8; // AVX2: 256-bit = 8 x f32

/// Block size for cache-blocking algorithms.
pub const BLOCK_SIZE: usize = 4096;

/// Prefetch distance in cache lines.
pub const PREFETCH_DISTANCE: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert!(CACHE_LINE_SIZE.is_power_of_two());
        assert!(VECTOR_WIDTH.is_power_of_two());
    }
}
