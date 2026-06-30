//! Long-Context Attention Engine
//!
//! High-performance attention kernel library for transformers supporting
//! sequences from 32K to 1M+ tokens with memory-efficient attention variants.

pub mod attention;
pub mod auto_select;
pub mod block_sparse;
pub mod config;
pub mod flash;
pub mod kv_cache;
pub mod rope;
pub mod sliding_window;
pub mod streaming;

pub use attention::*;
pub use auto_select::*;
pub use block_sparse::*;
pub use config::*;
pub use flash::*;
pub use kv_cache::*;
pub use rope::*;
pub use sliding_window::*;
pub use streaming::*;

use thiserror::Error;

/// Attention errors.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Out of memory: required {required} bytes, available {available}")]
    OutOfMemory { required: usize, available: usize },

    #[error("KV cache error: {0}")]
    CacheError(String),

    #[error("Sequence too long: {length} exceeds maximum {max}")]
    SequenceTooLong { length: usize, max: usize },

    #[error("Invalid attention mask")]
    InvalidMask,
}

/// Result type for attention operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Default head dimension.
pub const DEFAULT_HEAD_DIM: usize = 64;

/// Default number of heads.
pub const DEFAULT_NUM_HEADS: usize = 8;

/// Default block size for block sparse attention.
pub const DEFAULT_BLOCK_SIZE: usize = 64;

/// Default window size for sliding window attention.
pub const DEFAULT_WINDOW_SIZE: usize = 4096;
