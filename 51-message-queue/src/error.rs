//! Error types for the message queue.

use std::io;
use thiserror::Error;

/// Result type alias for message queue operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Message queue error types.
#[derive(Error, Debug)]
pub enum Error {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Topic not found
    #[error("Topic not found: {0}")]
    TopicNotFound(String),

    /// Topic already exists
    #[error("Topic already exists: {0}")]
    TopicAlreadyExists(String),

    /// Partition not found
    #[error("Partition not found: {partition} for topic {topic}")]
    PartitionNotFound { topic: String, partition: u32 },

    /// Invalid partition count
    #[error("Invalid partition count: {0}")]
    InvalidPartitionCount(u32),

    /// Offset out of range
    #[error("Offset out of range: {offset} (valid range: {start}..{end})")]
    OffsetOutOfRange { offset: u64, start: u64, end: u64 },

    /// Consumer group not found
    #[error("Consumer group not found: {0}")]
    ConsumerGroupNotFound(String),

    /// Consumer not found
    #[error("Consumer not found: {0}")]
    ConsumerNotFound(String),

    /// Rebalance in progress
    #[error("Rebalance in progress for group: {0}")]
    RebalanceInProgress(String),

    /// Not subscribed
    #[error("Consumer not subscribed to any topics")]
    NotSubscribed,

    /// Message too large
    #[error("Message too large: {size} bytes (max: {max} bytes)")]
    MessageTooLarge { size: usize, max: usize },

    /// Compression error
    #[error("Compression error: {0}")]
    Compression(String),

    /// Decompression error
    #[error("Decompression error: {0}")]
    Decompression(String),

    /// CRC checksum mismatch
    #[error("CRC checksum mismatch: expected {expected}, got {actual}")]
    CrcMismatch { expected: u32, actual: u32 },

    /// Segment corrupted
    #[error("Segment corrupted: {0}")]
    SegmentCorrupted(String),

    /// Storage full
    #[error("Storage full")]
    StorageFull,

    /// Timeout
    #[error("Operation timed out after {0:?}")]
    Timeout(std::time::Duration),

    /// Producer closed
    #[error("Producer has been closed")]
    ProducerClosed,

    /// Consumer closed
    #[error("Consumer has been closed")]
    ConsumerClosed,

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Illegal state
    #[error("Illegal state: {0}")]
    IllegalState(String),

    /// Lock poisoned
    #[error("Lock poisoned")]
    LockPoisoned,

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),

    /// Commit failed
    #[error("Commit failed: {0}")]
    CommitFailed(String),

    /// Assignment error
    #[error("Assignment error: {0}")]
    AssignmentError(String),

    /// Channel closed
    #[error("Channel closed")]
    ChannelClosed,

    /// Unknown producer ID
    #[error("Unknown producer ID: {0}")]
    UnknownProducerId(i64),

    /// Producer fenced (epoch mismatch)
    #[error("Producer fenced: expected epoch {expected}, received {received}")]
    ProducerFenced { expected: i16, received: i16 },

    /// Duplicate sequence number (idempotent producer)
    #[error("Duplicate sequence: expected {expected}, received {received}")]
    DuplicateSequence { expected: i32, received: i32 },

    /// Out of order sequence number
    #[error("Out of order sequence: expected {expected}, received {received}")]
    OutOfOrderSequence { expected: i32, received: i32 },

    /// Invalid transaction state
    #[error("Invalid transaction state: {0}")]
    InvalidTransactionState(String),

    /// Transaction timeout
    #[error("Transaction timed out")]
    TransactionTimeout,

    /// Not leader for partition
    #[error("Not leader for partition {partition} of topic {topic}")]
    NotLeaderForPartition { topic: String, partition: u32 },

    /// Insufficient replicas
    #[error("Insufficient replicas in ISR: {available} < {required}")]
    InsufficientReplicas { available: usize, required: usize },

    /// Protocol error (malformed frame, unexpected message, oversized frame, etc.)
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Authentication failed or required
    #[error("Authentication failed: {0}")]
    Auth(String),

    /// The peer returned a typed error response
    #[error("Server error ({kind}): {message}")]
    Server {
        /// Machine-readable error kind reported by the server.
        kind: String,
        /// Human-readable detail.
        message: String,
    },
}

impl Error {
    /// Check if error is retriable
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            Error::Io(_)
                | Error::Timeout(_)
                | Error::RebalanceInProgress(_)
                | Error::ChannelClosed
        )
    }

    /// Check if error is fatal
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            Error::SegmentCorrupted(_)
                | Error::CrcMismatch { .. }
                | Error::LockPoisoned
        )
    }
}

impl From<bincode::Error> for Error {
    fn from(e: bincode::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Error::LockPoisoned
    }
}

impl<T> From<tokio::sync::mpsc::error::SendError<T>> for Error {
    fn from(_: tokio::sync::mpsc::error::SendError<T>) -> Self {
        Error::ChannelClosed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = Error::TopicNotFound("test".to_string());
        assert_eq!(err.to_string(), "Topic not found: test");
    }

    #[test]
    fn test_error_is_retriable() {
        assert!(Error::Timeout(std::time::Duration::from_secs(1)).is_retriable());
        assert!(Error::RebalanceInProgress("group1".to_string()).is_retriable());
        assert!(!Error::TopicNotFound("test".to_string()).is_retriable());
    }

    #[test]
    fn test_error_is_fatal() {
        assert!(Error::SegmentCorrupted("test".to_string()).is_fatal());
        assert!(Error::CrcMismatch { expected: 1, actual: 2 }.is_fatal());
        assert!(!Error::TopicNotFound("test".to_string()).is_fatal());
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: Error = io_err.into();
        assert!(matches!(err, Error::Io(_)));
        assert!(err.is_retriable());
    }
}
