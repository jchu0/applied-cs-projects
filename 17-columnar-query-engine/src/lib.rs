//! Columnar Query Engine (DuckDB-lite).
//!
//! A high-performance in-memory analytical query engine with columnar storage
//! and vectorized execution for OLAP workloads.

pub mod types;
pub mod vector;
pub mod storage;
pub mod expression;
pub mod plan;
pub mod executor;
pub mod parser;
pub mod parallel;

use thiserror::Error;

/// Query engine errors.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Type error: {0}")]
    Type(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for query engine operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Vector size for vectorized execution.
pub const VECTOR_SIZE: usize = 2048;
