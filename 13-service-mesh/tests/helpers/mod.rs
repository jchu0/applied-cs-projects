//! Test helpers and fixtures for service mesh testing

use service_mesh::discovery::{LoadBalancerType, ServiceKey};
use service_mesh::{
    CertManager, CertificateAuthority, CircuitBreaker, CircuitBreakerConfig, CircuitState,
    Endpoint, EndpointHealth, ProxyConfig, ServiceEndpoints, ServiceIdentity, ServiceRegistry,
    SidecarProxy,
};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Creates a test Certificate Authority with default settings
pub fn create_test_ca() -> CertificateAuthority {
    CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600)).expect("Failed to create test CA")
}

/// Creates a test service identity
pub fn create_test_identity(namespace: &str, service: &str) -> ServiceIdentity {
    ServiceIdentity::new(namespace, service)
}

/// Creates a configured sidecar proxy for testing
pub fn create_test_proxy(service: &str) -> SidecarProxy {
    let ca = create_test_ca();
    let identity = create_test_identity("default", service);
    let cert = ca
        .issue_certificate(&identity)
        .expect("Failed to issue test certificate");

    let cert_manager = Arc::new(CertManager::new(identity, cert, Duration::from_secs(3600)));
    let registry = Arc::new(ServiceRegistry::new());

    SidecarProxy::new(ProxyConfig::default(), registry, cert_manager)
}

/// Creates a test service registry with sample services
pub fn create_populated_registry() -> ServiceRegistry {
    let registry = ServiceRegistry::new();

    // Add frontend service
    let frontend_key = ServiceKey::new("frontend", "default", 80);
    let frontend = ServiceEndpoints {
        endpoints: vec![
            Endpoint {
                address: "10.0.1.1:80".parse().unwrap(),
                weight: 100,
                health: EndpointHealth::Healthy,
                metadata: Default::default(),
                tls_identity: ServiceIdentity::default(),
            },
            Endpoint {
                address: "10.0.1.2:80".parse().unwrap(),
                weight: 100,
                health: EndpointHealth::Healthy,
                metadata: Default::default(),
                tls_identity: ServiceIdentity::default(),
            },
        ],
        load_balancer: LoadBalancerType::RoundRobin,
        policy: Default::default(),
    };
    registry.register(frontend_key, frontend);

    // Add backend service
    let backend_key = ServiceKey::new("backend", "default", 8080);
    let backend = ServiceEndpoints {
        endpoints: vec![
            Endpoint {
                address: "10.0.2.1:8080".parse().unwrap(),
                weight: 100,
                health: EndpointHealth::Healthy,
                metadata: Default::default(),
                tls_identity: ServiceIdentity::default(),
            },
            Endpoint {
                address: "10.0.2.2:8080".parse().unwrap(),
                weight: 100,
                health: EndpointHealth::Healthy,
                metadata: Default::default(),
                tls_identity: ServiceIdentity::default(),
            },
        ],
        load_balancer: LoadBalancerType::RoundRobin,
        policy: Default::default(),
    };
    registry.register(backend_key, backend);

    registry
}

/// Mock service for testing
pub struct MockService {
    name: String,
    addr: SocketAddr,
    request_count: Arc<Mutex<usize>>,
    failure_rate: f32,
}

impl MockService {
    pub fn new(name: &str, addr: SocketAddr) -> Self {
        Self {
            name: name.to_string(),
            addr,
            request_count: Arc::new(Mutex::new(0)),
            failure_rate: 0.0,
        }
    }

    pub fn with_failure_rate(mut self, rate: f32) -> Self {
        self.failure_rate = rate;
        self
    }

    pub fn handle_request(&self) -> Result<String, String> {
        let mut count = self.request_count.lock().unwrap();
        *count += 1;

        if rand::random::<f32>() < self.failure_rate {
            Err(format!("Service {} failed", self.name))
        } else {
            Ok(format!("Response from {} (request #{})", self.name, *count))
        }
    }

    pub fn request_count(&self) -> usize {
        *self.request_count.lock().unwrap()
    }
}

/// Test harness for circuit breaker testing
pub struct CircuitBreakerTestHarness {
    breaker: CircuitBreaker,
    service: MockService,
}

impl CircuitBreakerTestHarness {
    pub fn new(service: MockService, config: CircuitBreakerConfig) -> Self {
        Self {
            breaker: CircuitBreaker::new(config),
            service,
        }
    }

    pub fn execute_request(&mut self) -> Result<String, String> {
        if self.breaker.is_open() {
            return Err("Circuit breaker open".to_string());
        }

        match self.service.handle_request() {
            Ok(response) => {
                self.breaker.record_success();
                Ok(response)
            }
            Err(error) => {
                self.breaker.record_failure();
                Err(error)
            }
        }
    }

    pub fn circuit_state(&self) -> CircuitState {
        self.breaker.state()
    }

    pub fn is_circuit_open(&self) -> bool {
        self.breaker.is_open()
    }
}

/// Metrics collector for testing
pub struct TestMetricsCollector {
    metrics: Arc<Mutex<Vec<(String, f64)>>>,
}

impl TestMetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn record(&self, metric_name: &str, value: f64) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.push((metric_name.to_string(), value));
    }

    pub fn get_metrics(&self) -> Vec<(String, f64)> {
        self.metrics.lock().unwrap().clone()
    }

    pub fn get_metric_value(&self, metric_name: &str) -> Option<f64> {
        let metrics = self.metrics.lock().unwrap();
        metrics
            .iter()
            .rev()
            .find(|(name, _)| name == metric_name)
            .map(|(_, value)| *value)
    }
}

/// Simulates network conditions for testing
pub struct NetworkSimulator {
    latency_ms: u64,
    packet_loss_rate: f32,
    bandwidth_limit: Option<usize>,
}

impl NetworkSimulator {
    pub fn new() -> Self {
        Self {
            latency_ms: 0,
            packet_loss_rate: 0.0,
            bandwidth_limit: None,
        }
    }

    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    pub fn with_packet_loss(mut self, rate: f32) -> Self {
        self.packet_loss_rate = rate;
        self
    }

    pub fn with_bandwidth_limit(mut self, bytes_per_sec: usize) -> Self {
        self.bandwidth_limit = Some(bytes_per_sec);
        self
    }

    pub fn simulate_request(&self, size_bytes: usize) -> Result<Duration, String> {
        if rand::random::<f32>() < self.packet_loss_rate {
            return Err("Packet lost".to_string());
        }

        let mut transfer_time_ms = self.latency_ms;
        if let Some(bandwidth) = self.bandwidth_limit {
            transfer_time_ms += ((size_bytes as f64 / bandwidth as f64) * 1000.0) as u64;
        }

        std::thread::sleep(Duration::from_millis(transfer_time_ms));
        Ok(Duration::from_millis(transfer_time_ms))
    }
}
