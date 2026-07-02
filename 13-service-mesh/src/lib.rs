//! Service Mesh (Linkerd/Istio-lite)
//!
//! A service mesh providing secure, observable, and reliable service-to-service
//! communication with mTLS, service discovery, and traffic management.
//!
//! ## Features
//!
//! - **Sidecar Proxy**: Inbound/outbound traffic interception with HTTP/1.1 support
//! - **mTLS**: Mutual TLS with SPIFFE identity and certificate rotation
//! - **Service Discovery**: Service registry with health checking
//! - **Traffic Management**: Retries, circuit breakers, timeouts
//! - **xDS Protocol**: Envoy-compatible dynamic configuration
//! - **Kubernetes Integration**: Mutating webhook for sidecar injection

pub mod cert;
pub mod config;
pub mod discovery;
pub mod k8s;
pub mod metrics;
pub mod policy;
pub mod proxy;
pub mod tls;
pub mod tracing_mesh;
pub mod xds;

pub use cert::{CertificateAuthority, CertManager, IssuedCert};
pub use config::{ProxyConfig, ServiceIdentity, TlsConfig};
pub use discovery::{Endpoint, EndpointHealth, LoadBalancer, ServiceEndpoints, ServiceRegistry};
pub use k8s::{SidecarInjector, WebhookServer};
pub use metrics::ProxyMetrics;
pub use policy::{
    AuthorizationPolicy, CircuitBreaker, CircuitBreakerConfig, CircuitState, MtlsMode,
    RetryCondition, RetryPolicy, ServicePolicy, TimeoutPolicy,
};
pub use proxy::SidecarProxy;
pub use tls::{SecureConnection, TlsManager, TlsStream};
pub use tracing_mesh::{Span, SpanContext, Tracer};
pub use xds::{XdsClient, XdsServer};

/// Error types for the service mesh.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TLS error: {0}")]
    Tls(String),

    #[error("Certificate error: {0}")]
    Certificate(String),

    #[error("Unauthorized")]
    Unauthorized,

    #[error("Service not found: {0}")]
    ServiceNotFound(String),

    #[error("Circuit breaker open")]
    CircuitOpen,

    #[error("Timeout")]
    Timeout,

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serialization(e.to_string())
    }
}

/// Result type alias.
pub type Result<T> = std::result::Result<T, Error>;
