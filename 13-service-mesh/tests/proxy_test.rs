//! Unit tests for sidecar proxy functionality

use service_mesh::discovery::LoadBalancerType;
use service_mesh::{
    AuthorizationPolicy, CertManager, CertificateAuthority, CircuitBreaker, CircuitBreakerConfig,
    CircuitState, Endpoint, EndpointHealth, LoadBalancer, ProxyConfig, RetryCondition,
    RetryPolicy, ServiceIdentity, ServiceRegistry, SidecarProxy, TimeoutPolicy,
};
use std::sync::Arc;
use std::time::Duration;

fn default_identity() -> ServiceIdentity {
    ServiceIdentity::new("default", "test-service")
}

fn create_test_proxy() -> SidecarProxy {
    let ca = CertificateAuthority::new(Duration::from_secs(365 * 24 * 3600))
        .expect("Failed to create CA");

    let identity = ServiceIdentity::new("default", "test-service");
    let cert = ca
        .issue_certificate(&identity)
        .expect("Failed to issue certificate");

    let cert_manager = Arc::new(CertManager::new(identity, cert, Duration::from_secs(3600)));
    let registry = Arc::new(ServiceRegistry::new());

    SidecarProxy::new(ProxyConfig::default(), registry, cert_manager)
}

#[test]
fn test_proxy_initialization() {
    let proxy = create_test_proxy();
    let config = proxy.config();

    assert_eq!(config.inbound_port, 15006);
    assert_eq!(config.outbound_port, 15001);
    assert_eq!(config.admin_port, 15000);
}

#[test]
fn test_proxy_configuration() {
    let proxy = create_test_proxy();
    let config = proxy.config();

    assert!(config.timeout > Duration::ZERO);
    assert!(!config.service_name.is_empty());
}

#[test]
fn test_authorization_policy() {
    let policy = AuthorizationPolicy::default();

    assert!(matches!(
        policy.action,
        service_mesh::policy::AuthAction::Allow
    ));
    assert!(policy.rules.is_empty());
}

#[test]
fn test_circuit_breaker() {
    let config = CircuitBreakerConfig {
        consecutive_failures: 5,
        success_threshold: 2,
        base_ejection_time: Duration::from_millis(100),
        ..Default::default()
    };

    let circuit_breaker = CircuitBreaker::new(config);

    // Initially closed
    assert!(matches!(circuit_breaker.state(), CircuitState::Closed));
    assert!(!circuit_breaker.is_open());

    // Record failures to trip the breaker
    for _ in 0..6 {
        circuit_breaker.record_failure();
    }
    assert!(circuit_breaker.is_open());

    // Wait for half-open state
    std::thread::sleep(Duration::from_millis(150));
    assert!(!circuit_breaker.is_open()); // half-open allows requests

    // Success in half-open should close the circuit
    circuit_breaker.record_success();
    circuit_breaker.record_success();
    assert!(!circuit_breaker.is_open());
}

#[test]
fn test_retry_policy() {
    let retry_policy = RetryPolicy {
        max_retries: 3,
        retry_on: vec![RetryCondition::Status5xx, RetryCondition::StatusCode(429)],
        backoff: service_mesh::policy::BackoffConfig {
            base_interval: Duration::from_millis(100),
            max_interval: Duration::from_secs(5),
            jitter: 0.1,
        },
    };

    assert_eq!(retry_policy.max_retries, 3);
    assert_eq!(retry_policy.retry_on.len(), 2);
}

#[test]
fn test_timeout_policy() {
    let timeout_policy = TimeoutPolicy::default();

    assert!(timeout_policy.request_timeout > Duration::ZERO);
}

#[test]
fn test_proxy_metrics_snapshot() {
    let proxy = create_test_proxy();
    let snapshot = proxy.metrics.snapshot();

    assert_eq!(snapshot.requests_total, 0);
    assert_eq!(snapshot.active_connections, 0);
    assert_eq!(snapshot.connection_errors, 0);
}

#[test]
fn test_config_update() {
    let proxy = create_test_proxy();

    let mut new_config = proxy.config();
    new_config.service_name = "updated-service".to_string();
    proxy.update_config(new_config);

    assert_eq!(proxy.config().service_name, "updated-service");
}

#[test]
fn test_load_balancing() {
    let endpoints = vec![
        Endpoint {
            address: "127.0.0.1:8080".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
        Endpoint {
            address: "127.0.0.2:8080".parse().unwrap(),
            weight: 50,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
        Endpoint {
            address: "127.0.0.3:8080".parse().unwrap(),
            weight: 75,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: default_identity(),
        },
    ];

    let balancer = LoadBalancer::new(LoadBalancerType::RoundRobin);

    // Test selection works
    let e1 = balancer.select(&endpoints).unwrap();
    let e2 = balancer.select(&endpoints).unwrap();
    let e3 = balancer.select(&endpoints).unwrap();

    // All should be valid addresses
    assert!(
        e1.address == endpoints[0].address
            || e1.address == endpoints[1].address
            || e1.address == endpoints[2].address
    );
}

#[test]
fn test_backoff_calculation() {
    let proxy = create_test_proxy();
    let policy = RetryPolicy {
        max_retries: 3,
        retry_on: vec![RetryCondition::Status5xx],
        backoff: service_mesh::policy::BackoffConfig {
            base_interval: Duration::from_millis(100),
            max_interval: Duration::from_secs(5),
            jitter: 0.1,
        },
    };

    let backoff1 = proxy.calculate_backoff(1, &policy);
    let backoff2 = proxy.calculate_backoff(2, &policy);
    let backoff3 = proxy.calculate_backoff(3, &policy);

    // Exponential backoff (with jitter)
    assert!(backoff1.as_millis() >= 100);
    assert!(backoff1.as_millis() <= 150);

    assert!(backoff2.as_millis() >= 200);
    assert!(backoff2.as_millis() <= 250);

    assert!(backoff3.as_millis() >= 400);
    assert!(backoff3.as_millis() <= 500);
}

#[test]
fn test_retryable_error_detection() {
    let proxy = create_test_proxy();
    let policy = RetryPolicy {
        max_retries: 3,
        retry_on: vec![RetryCondition::Status5xx, RetryCondition::StatusCode(429)],
        backoff: service_mesh::policy::BackoffConfig::default(),
    };

    assert!(proxy.is_retryable_error(500, &policy));
    assert!(proxy.is_retryable_error(503, &policy));
    assert!(proxy.is_retryable_error(429, &policy));
    assert!(!proxy.is_retryable_error(404, &policy));
    assert!(!proxy.is_retryable_error(200, &policy));
}

#[test]
fn test_prometheus_metrics_export() {
    let proxy = create_test_proxy();
    let prometheus = proxy.metrics.to_prometheus();

    assert!(prometheus.contains("mesh_requests_total"));
    assert!(prometheus.contains("mesh_active_connections"));
}

#[test]
fn test_circuit_breaker_reset_on_success() {
    let config = CircuitBreakerConfig {
        consecutive_failures: 3,
        success_threshold: 2,
        base_ejection_time: Duration::from_millis(50),
        ..Default::default()
    };

    let circuit_breaker = CircuitBreaker::new(config);

    // Record some failures but not enough to trip
    circuit_breaker.record_failure();
    circuit_breaker.record_failure();
    assert!(!circuit_breaker.is_open());

    // Success should reset failure count
    circuit_breaker.record_success();

    // Now we need 3 more failures
    circuit_breaker.record_failure();
    circuit_breaker.record_failure();
    assert!(!circuit_breaker.is_open());
}
