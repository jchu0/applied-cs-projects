//! Network stack implementation with TCP and HTTP.
//!
//! This crate provides a userspace TCP/IP stack and HTTP/1.1 parser
//! for educational purposes and embedded systems.

pub mod tcp;
pub mod http;
pub mod pool;
pub mod proxy;

use thiserror::Error;

/// Network stack errors.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Timeout")]
    Timeout,

    #[error("Connection refused")]
    ConnectionRefused,

    #[error("Connection reset")]
    ConnectionReset,

    #[error("Invalid state: {0}")]
    InvalidState(String),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result type for network operations.
pub type Result<T> = std::result::Result<T, Error>;
