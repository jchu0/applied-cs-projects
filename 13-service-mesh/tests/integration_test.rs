//! Integration tests for the service mesh

mod helpers;

use helpers::*;
use service_mesh::discovery::LoadBalancerType;
use service_mesh::{
    policy, AuthorizationPolicy, CircuitBreakerConfig, Endpoint, EndpointHealth, LoadBalancer,
    RetryCondition, RetryPolicy, ServiceIdentity, TimeoutPolicy, Tracer,
};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

#[test]
fn test_proxy_creation() {
    let proxy = create_test_proxy("test-service");
    let config = proxy.config();

    assert_eq!(config.inbound_port, 15006);
    assert_eq!(config.outbound_port, 15001);
    assert_eq!(config.admin_port, 15000);
}

#[test]
fn test_circuit_breaker_integration() {
    let service = MockService::new("flaky-service", "127.0.0.1:8080".parse().unwrap())
        .with_failure_rate(0.6);

    let config = CircuitBreakerConfig {
        consecutive_failures: 3,
        success_threshold: 2,
        base_ejection_time: Duration::from_millis(500),
        ..Default::default()
    };

    let mut harness = CircuitBreakerTestHarness::new(service, config);

    let mut failures = 0;
    let mut circuit_opened = false;

    for _ in 0..20 {
        match harness.execute_request() {
            Err(msg) if msg == "Circuit breaker open" => {
                circuit_opened = true;
                break;
            }
            Err(_) => failures += 1,
            Ok(_) => {}
        }
    }

    assert!(circuit_opened);
    assert!(failures >= 3);

    thread::sleep(Duration::from_millis(600));

    // Should be half-open now - is_open() returns false when half-open
    assert!(!harness.is_circuit_open());
}

#[test]
fn test_load_balancing_with_health_checks() {
    let endpoints = vec![
        Endpoint {
            address: "10.0.0.1:8080".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: ServiceIdentity::default(),
        },
        Endpoint {
            address: "10.0.0.2:8080".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Unhealthy,
            metadata: Default::default(),
            tls_identity: ServiceIdentity::default(),
        },
        Endpoint {
            address: "10.0.0.3:8080".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: ServiceIdentity::default(),
        },
    ];

    let balancer = LoadBalancer::new(LoadBalancerType::RoundRobin);

    for _ in 0..100 {
        let selected = balancer.select(&endpoints).unwrap();
        assert_ne!(
            selected.address,
            "10.0.0.2:8080".parse::<std::net::SocketAddr>().unwrap()
        );
    }
}

#[test]
fn test_certificate_authority() {
    let ca = create_test_ca();
    let identity = create_test_identity("default", "test-service");

    let cert = ca.issue_certificate(&identity);
    assert!(cert.is_ok());

    let issued = cert.unwrap();
    assert!(!issued.cert_chain.is_empty());
    assert!(!issued.private_key.is_empty());
}

#[test]
fn test_service_registry() {
    let registry = create_populated_registry();

    let frontend = registry.get_by_name("frontend");
    assert!(frontend.is_some());
    assert_eq!(frontend.unwrap().endpoints.len(), 2);

    let backend = registry.get_by_name("backend");
    assert!(backend.is_some());
    assert_eq!(backend.unwrap().endpoints.len(), 2);

    let unknown = registry.get_by_name("unknown-service");
    assert!(unknown.is_none());
}

#[test]
fn test_retry_policy() {
    let policy = RetryPolicy {
        max_retries: 3,
        retry_on: vec![RetryCondition::Status5xx, RetryCondition::StatusCode(429)],
        backoff: policy::BackoffConfig {
            base_interval: Duration::from_millis(100),
            max_interval: Duration::from_secs(5),
            jitter: 0.1,
        },
    };

    assert_eq!(policy.max_retries, 3);
    assert_eq!(policy.retry_on.len(), 2);
}

#[test]
fn test_concurrent_registry_access() {
    let num_threads = 10;
    let barrier = Arc::new(Barrier::new(num_threads));
    let registry = Arc::new(create_populated_registry());

    let mut handles = vec![];

    for i in 0..num_threads {
        let barrier = Arc::clone(&barrier);
        let registry = Arc::clone(&registry);

        let handle = thread::spawn(move || {
            barrier.wait();

            for _ in 0..100 {
                let _frontend = registry.get_by_name("frontend");
                let _backend = registry.get_by_name("backend");
            }

            i
        });

        handles.push(handle);
    }

    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results.len(), num_threads);
}

#[test]
fn test_metrics_collection() {
    let proxy = create_test_proxy("metrics-test");
    let snapshot = proxy.metrics.snapshot();

    assert_eq!(snapshot.requests_total, 0);
    assert_eq!(snapshot.active_connections, 0);
    assert_eq!(snapshot.connection_errors, 0);
}

#[test]
fn test_network_simulation() {
    let simulator = NetworkSimulator::new()
        .with_latency(50)
        .with_packet_loss(0.1)
        .with_bandwidth_limit(1_000_000);

    let mut successes = 0;
    let mut failures = 0;
    let mut total_latency = Duration::ZERO;

    for _ in 0..100 {
        match simulator.simulate_request(10_000) {
            Ok(latency) => {
                successes += 1;
                total_latency += latency;
            }
            Err(_) => {
                failures += 1;
            }
        }
    }

    assert!(failures >= 5 && failures <= 20);

    if successes > 0 {
        let avg_latency = total_latency / successes as u32;
        assert!(avg_latency >= Duration::from_millis(50));
    }
}

#[test]
fn test_distributed_tracing_span_creation() {
    let tracer = Tracer::new("test-service".to_string(), "localhost:14268".to_string());

    let span = tracer.start_span("test-operation", None);
    assert_eq!(span.name, "test-operation");

    let child_span = tracer.start_span("child-operation", Some(&span.context()));
    assert_eq!(child_span.name, "child-operation");

    assert_eq!(span.context().trace_id, child_span.context().trace_id);
}

#[test]
fn test_proxy_config_update() {
    let proxy = create_test_proxy("config-test");

    let mut new_config = proxy.config();
    new_config.service_name = "updated-service".to_string();
    proxy.update_config(new_config);

    assert_eq!(proxy.config().service_name, "updated-service");
}

#[test]
fn test_authorization_policy() {
    let policy = AuthorizationPolicy::default();

    assert!(matches!(policy.action, policy::AuthAction::Allow));
    assert!(policy.rules.is_empty());
}

#[test]
fn test_timeout_policy() {
    let policy = TimeoutPolicy::default();

    assert!(policy.request_timeout > Duration::ZERO);
}
