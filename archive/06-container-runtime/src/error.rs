//! Error types for the container runtime

use std::io;
use std::path::PathBuf;

/// Result type for the container runtime
pub type Result<T> = std::result::Result<T, Error>;

/// Error type for the container runtime
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Nix error: {0}")]
    Nix(#[from] nix::Error),

    #[error("Container not found: {0}")]
    ContainerNotFound(String),

    #[error("Container already exists: {0}")]
    ContainerExists(String),

    #[error("Invalid container state: expected {expected}, got {actual}")]
    InvalidState { expected: String, actual: String },

    #[error("Invalid digest: {0}")]
    InvalidDigest(String),

    #[error("Invalid specification: {0}")]
    InvalidSpec(String),

    #[error("Namespace error: {0}")]
    Namespace(String),

    #[error("Cgroup error: {0}")]
    Cgroup(String),

    #[error("Mount error: {0}")]
    Mount(String),

    #[error("Process error: {0}")]
    Process(String),

    #[error("Fork error: {0}")]
    Fork(String),

    #[error("Runtime error: {0}")]
    Runtime(String),

    #[error("Image not found: {0}")]
    ImageNotFound(String),

    #[error("Layer not found: {0}")]
    LayerNotFound(String),

    #[error("Bundle not found: {0}")]
    BundleNotFound(PathBuf),

    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Registry error: {0}")]
    Registry(String),

    #[error("Network error: {0}")]
    Network(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}
