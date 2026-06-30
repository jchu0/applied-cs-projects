//! Error types for the time-series database

use std::io;
use thiserror::Error;

/// Result type for TSDB operations
pub type Result<T> = std::result::Result<T, TsdbError>;

/// Error types for the time-series database
#[derive(Error, Debug)]
pub enum TsdbError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Compression error: {0}")]
    Compression(String),

    #[error("Decompression error: {0}")]
    Decompression(String),

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(i64),

    #[error("Series not found: {0}")]
    SeriesNotFound(u64),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),

    #[error("WAL error: {0}")]
    WalError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Retention policy error: {0}")]
    RetentionError(String),

    #[error("Compaction error: {0}")]
    CompactionError(String),

    #[error("Data corruption: {0}")]
    Corruption(String),

    #[error("Buffer overflow: expected {expected}, got {actual}")]
    BufferOverflow { expected: usize, actual: usize },

    #[error("Invalid data format: {0}")]
    InvalidFormat(String),

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    #[error("Database closed")]
    DatabaseClosed,

    #[error("Lock poisoned")]
    LockPoisoned,
}

impl TsdbError {
    pub fn compression<S: Into<String>>(msg: S) -> Self {
        TsdbError::Compression(msg.into())
    }

    pub fn decompression<S: Into<String>>(msg: S) -> Self {
        TsdbError::Decompression(msg.into())
    }

    pub fn storage<S: Into<String>>(msg: S) -> Self {
        TsdbError::StorageError(msg.into())
    }

    pub fn wal<S: Into<String>>(msg: S) -> Self {
        TsdbError::WalError(msg.into())
    }

    pub fn invalid_query<S: Into<String>>(msg: S) -> Self {
        TsdbError::InvalidQuery(msg.into())
    }

    pub fn corruption<S: Into<String>>(msg: S) -> Self {
        TsdbError::Corruption(msg.into())
    }

    pub fn retention<S: Into<String>>(msg: S) -> Self {
        TsdbError::RetentionError(msg.into())
    }
}
