//! Kubernetes integration for service mesh.
//!
//! Provides:
//! - Mutating admission webhook for sidecar injection
//! - CRD definitions for traffic policies
//! - Controller for watching resources

mod webhook;
mod sidecar;
mod types;

pub use webhook::WebhookServer;
pub use sidecar::SidecarInjector;
pub use types::*;
