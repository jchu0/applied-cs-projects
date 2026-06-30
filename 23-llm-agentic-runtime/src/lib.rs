//! LLM Agentic Runtime
//!
//! Production-grade agent runtime for LLM-powered autonomous systems
//! with tool invocation, multi-step reasoning, and state persistence.

pub mod agent;
pub mod api;
pub mod executor;
pub mod guardrails;
pub mod llm;
pub mod memory;
pub mod planner;
pub mod tools;
pub mod tracing;

pub use agent::*;
pub use executor::*;
pub use guardrails::*;
pub use llm::*;
pub use memory::*;
pub use planner::*;
pub use tools::*;
pub use tracing::*;

use thiserror::Error;

/// Agent runtime errors.
#[derive(Error, Debug)]
pub enum Error {
    #[error("Execution timeout: {0}")]
    Timeout(String),

    #[error("Maximum steps exceeded: {0}")]
    MaxStepsExceeded(usize),

    #[error("Tool not found: {0}")]
    ToolNotFound(String),

    #[error("Tool execution failed: {0}")]
    ToolExecutionFailed(String),

    #[error("Invalid action: {0}")]
    InvalidAction(String),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Planning failed: {0}")]
    PlanningFailed(String),

    #[error("Guardrail violation: {0}")]
    GuardrailViolation(String),

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type for agent operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Default maximum steps for agent execution.
pub const DEFAULT_MAX_STEPS: usize = 20;

/// Default timeout in seconds.
pub const DEFAULT_TIMEOUT_SECONDS: f64 = 300.0;
