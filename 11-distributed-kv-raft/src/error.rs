//! Error types for the Raft implementation.

use crate::NodeId;
use thiserror::Error;

/// Result type alias for Raft operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur in Raft operations.
#[derive(Error, Debug, Clone)]
pub enum Error {
    /// Not the leader; includes hint for actual leader.
    #[error("Not leader, try node {0:?}")]
    NotLeader(Option<NodeId>),

    /// Quorum could not be reached.
    #[error("Quorum not reached")]
    QuorumNotReached,

    /// Cluster is unavailable.
    #[error("Cluster unavailable")]
    ClusterUnavailable,

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Storage error.
    #[error("Storage error: {0}")]
    Storage(String),

    /// Serialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Network error.
    #[error("Network error: {0}")]
    Network(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Log entry not found.
    #[error("Log entry not found at index {0}")]
    LogEntryNotFound(u64),

    /// Invalid state transition.
    #[error("Invalid state transition: {0}")]
    InvalidStateTransition(String),

    /// Request timeout.
    #[error("Request timeout")]
    Timeout,
}
