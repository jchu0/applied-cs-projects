//! Configuration for the service mesh proxy.

use crate::policy::{CircuitBreakerConfig, MtlsMode, RetryPolicy};
use std::time::{Duration, SystemTime};

/// Proxy configuration.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Service name.
    pub service_name: String,
    /// Namespace.
    pub namespace: String,
    /// Workload name.
    pub workload_name: String,

    /// Port for inbound traffic (from mesh).
    pub inbound_port: u16,
    /// Port for outbound traffic (to mesh).
    pub outbound_port: u16,
    /// Admin port for metrics and health.
    pub admin_port: u16,
    /// Application port (local service).
    pub app_port: u16,

    /// TLS configuration.
    pub tls_config: TlsConfig,
    /// mTLS mode for the inbound (server-side) data path.
    pub mtls_mode: MtlsMode,

    /// Retry policy.
    pub retry_policy: RetryPolicy,
    /// Request timeout.
    pub timeout: Duration,
    /// Circuit breaker configuration.
    pub circuit_breaker: CircuitBreakerConfig,

    /// Tracing configuration.
    pub tracing_config: TracingConfig,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            service_name: "default".to_string(),
            namespace: "default".to_string(),
            workload_name: "default".to_string(),
            inbound_port: 15006,
            outbound_port: 15001,
            admin_port: 15000,
            app_port: 8080,
            tls_config: TlsConfig::default(),
            mtls_mode: MtlsMode::Disable,
            retry_policy: RetryPolicy::default(),
            timeout: Duration::from_secs(30),
            circuit_breaker: CircuitBreakerConfig::default(),
            tracing_config: TracingConfig::default(),
        }
    }
}

/// TLS configuration.
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Certificate chain in PEM format.
    pub cert_chain: Vec<u8>,
    /// Private key in PEM format.
    pub private_key: Vec<u8>,
    /// Root CA certificate.
    pub root_ca: Vec<u8>,
    /// Certificate expiry time.
    pub cert_expiry: SystemTime,
    /// Service identity.
    pub identity: ServiceIdentity,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_chain: Vec::new(),
            private_key: Vec::new(),
            root_ca: Vec::new(),
            cert_expiry: SystemTime::now(),
            identity: ServiceIdentity::default(),
        }
    }
}

/// Service identity using SPIFFE.
#[derive(Debug, Clone, Default)]
pub struct ServiceIdentity {
    /// SPIFFE ID (e.g., spiffe://cluster.local/ns/default/sa/myservice).
    pub spiffe_id: String,
    /// Kubernetes service account.
    pub service_account: String,
    /// Namespace.
    pub namespace: String,
}

impl ServiceIdentity {
    /// Create a new service identity.
    pub fn new(namespace: &str, service_account: &str) -> Self {
        Self {
            spiffe_id: format!(
                "spiffe://cluster.local/ns/{}/sa/{}",
                namespace, service_account
            ),
            service_account: service_account.to_string(),
            namespace: namespace.to_string(),
        }
    }

    /// DNS name embedded in the workload's certificate for TLS hostname
    /// verification. Clients dial the upstream by this name so WebPKI
    /// validation succeeds while identity is still asserted via the SPIFFE URI
    /// SAN. Falls back to a placeholder when the identity is empty.
    pub fn tls_server_name(&self) -> String {
        let sa = if self.service_account.is_empty() {
            "workload"
        } else {
            &self.service_account
        };
        let ns = if self.namespace.is_empty() {
            "default"
        } else {
            &self.namespace
        };
        format!("{}.{}.mesh", sa, ns)
    }
}

/// Tracing configuration.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Enable tracing.
    pub enabled: bool,
    /// Collector endpoint.
    pub collector_endpoint: String,
    /// Sampling rate (0.0 to 1.0).
    pub sampling_rate: f64,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collector_endpoint: "localhost:14268".to_string(),
            sampling_rate: 1.0,
        }
    }
}
