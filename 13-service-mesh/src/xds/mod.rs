//! xDS (Discovery Service) protocol implementation.
//!
//! Implements Envoy's xDS protocol for dynamic service mesh configuration:
//! - CDS (Cluster Discovery Service): Discover available clusters/services
//! - EDS (Endpoint Discovery Service): Discover endpoints for each cluster
//! - LDS (Listener Discovery Service): Configure proxy listeners
//! - RDS (Route Discovery Service): Configure routing rules

mod server;
mod types;
mod client;

pub use server::XdsServer;
pub use types::*;
pub use client::XdsClient;
