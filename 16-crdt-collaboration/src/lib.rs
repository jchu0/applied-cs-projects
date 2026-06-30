//! CRDT-Based Real-Time Collaboration Engine.
//!
//! A Google Docs-lite collaborative editing system built on Conflict-free
//! Replicated Data Types (CRDTs). Enables real-time multi-user document
//! editing with automatic conflict resolution and eventual consistency.

pub mod api;
pub mod crdt;
pub mod document;
pub mod offline;
pub mod performance;
pub mod persistent;
pub mod presence;
pub mod protocol;
pub mod server;
pub mod storage;

use thiserror::Error;

/// Client ID type.
pub type ClientId = uuid::Uuid;

/// Document ID type.
pub type DocumentId = uuid::Uuid;

/// Collaboration errors.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Document not found: {0}")]
    DocumentNotFound(DocumentId),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Permission denied")]
    PermissionDenied,

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Result type for collaboration operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Timestamp type (Lamport clock).
pub type Timestamp = u64;
