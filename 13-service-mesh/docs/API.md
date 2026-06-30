# Service Mesh API Documentation

## Overview

This document provides comprehensive API documentation for the Service Mesh implementation. The APIs are organized by component and include detailed examples for common use cases.

## Table of Contents

1. [Certificate Management](#certificate-management)
2. [Proxy Configuration](#proxy-configuration)
3. [Service Discovery](#service-discovery)
4. [Policy Management](#policy-management)
5. [Metrics and Observability](#metrics-and-observability)
6. [Error Handling](#error-handling)

## Certificate Management

### CertificateAuthority

#### Creating a Certificate Authority

```rust
use service_mesh::CertificateAuthority;
use std::time::Duration;

// Create a new CA with 10-year validity
let ca = CertificateAuthority::new(
    "my-mesh-ca",
    Duration::from_secs(10 * 365 * 24 * 3600)
)?;
```

#### Issuing Service Certificates

```rust
use service_mesh::{ServiceIdentity, IssuedCert};

// Create service identity
let identity = ServiceIdentity::new(
    "frontend",      // service name
    "production",    // namespace
    "us-west-2"      // cluster
);

// Issue certificate with 90-day validity
let cert: IssuedCert = ca.issue_certificate(
    &identity,
    Duration::from_secs(90 * 24 * 3600)
)?;

// Access certificate details
println!("Serial: {}", cert.serial_number());
println!("Subject: {}", cert.subject());
println!("Expires: {:?}", cert.not_after());
```

#### Creating Intermediate CAs

```rust
// Create intermediate CA for a specific region
let intermediate_ca = ca.create_intermediate(
    "us-west-intermediate",
    Duration::from_secs(5 * 365 * 24 * 3600)
)?;
```

### CertManager

#### Automatic Certificate Management

```rust
use service_mesh::CertManager;

let mut cert_manager = CertManager::new(ca);

// Get or issue certificate (with caching)
let cert = cert_manager.get_or_issue_certificate(
    &identity,
    Duration::from_secs(90 * 24 * 3600)
)?;

// Force certificate rotation
cert_manager.rotate_certificate(&identity)?;

// Clean up expired certificates
cert_manager.cleanup_expired();
```

### ServiceIdentity

#### SPIFFE Identity Format

```rust
let identity = ServiceIdentity::new("api-gateway", "prod", "cluster-1");

// Get SPIFFE ID
let spiffe_id = identity.spiffe_id();
// Returns: "spiffe://cluster-1/ns/prod/sa/api-gateway"

// Parse from SPIFFE ID
let parsed = ServiceIdentity::from_spiffe_id(&spiffe_id)?;
assert_eq!(parsed, identity);
```

## Proxy Configuration

### SidecarProxy

#### Basic Proxy Setup

```rust
use service_mesh::{SidecarProxy, ProxyConfig, TlsConfig};
use std::net::SocketAddr;

// Configure TLS
let tls_config = TlsConfig::new(cert, ca);

// Configure proxy
let config = ProxyConfig::new(
    "0.0.0.0:15001".parse::<SocketAddr>()?,  // listen address
    "127.0.0.1:8080".parse::<SocketAddr>()?,  // upstream service
    tls_config
);

// Create and start proxy
let proxy = SidecarProxy::new(config);
proxy.start().await?;
```

#### Advanced Configuration

```rust
use service_mesh::{ProxyConfig, ConnectionPoolConfig, ProxyMode};

let mut config = ProxyConfig::new(listen_addr, upstream_addr, tls_config);

// Configure connection pooling
config.set_connection_pool(ConnectionPoolConfig {
    max_connections: 100,
    max_idle_connections: 10,
    idle_timeout: Duration::from_secs(60),
    connection_timeout: Duration::from_secs(10),
});

// Set proxy mode
config.set_mode(ProxyMode::Transparent); // or ProxyMode::Explicit

// Enable HTTP/2
config.enable_http2(true);

// Set buffer sizes
config.set_buffer_size(64 * 1024); // 64KB
```

#### Request Forwarding

```rust
// Forward request with policies applied
let response = proxy.forward_request(
    "backend-service",
    &service_registry,
    &authorization_policy
).await?;

// With custom headers
let mut headers = HashMap::new();
headers.insert("X-Request-ID", "12345");
headers.insert("X-B3-TraceId", "abc123");

let response = proxy.forward_request_with_headers(
    "backend-service",
    &service_registry,
    &authorization_policy,
    headers
).await?;
```

## Service Discovery

### ServiceRegistry

#### Service Registration

```rust
use service_mesh::{ServiceRegistry, Endpoint};

let mut registry = ServiceRegistry::new();

// Register service with endpoints
let endpoints = vec![
    Endpoint::new("10.0.1.1:8080".parse()?, 100),  // weight 100
    Endpoint::new("10.0.1.2:8080".parse()?, 100),
    Endpoint::new("10.0.1.3:8080".parse()?, 50),   // lower weight
];

registry.register_service("api-service", endpoints);

// Register with metadata
registry.register_service_with_metadata(
    "database",
    vec![Endpoint::new("10.0.2.1:5432".parse()?, 100)],
    vec![
        ("version", "14.5"),
        ("region", "us-west-2"),
        ("replica_type", "primary")
    ]
);
```

#### Service Discovery

```rust
// Discover service endpoints
let endpoints = registry.discover("api-service")
    .ok_or("Service not found")?;

// Get service info with metadata
let info = registry.get_service_info("database")?;
println!("Version: {}", info.metadata.get("version").unwrap());

// List all services
let services = registry.list_services();
for service in services {
    println!("Service: {}", service);
}
```

#### Dynamic Updates

```rust
// Update service endpoints
let new_endpoints = vec![
    Endpoint::new("10.0.1.4:8080".parse()?, 100),
    Endpoint::new("10.0.1.5:8080".parse()?, 100),
];
registry.update_service("api-service", new_endpoints);

// Remove specific endpoint
registry.remove_endpoint("api-service", "10.0.1.4:8080".parse()?);

// Deregister entire service
registry.deregister_service("old-service");
```

### ServiceEndpoints

#### Health Management

```rust
use service_mesh::{ServiceEndpoints, EndpointHealth};

let mut endpoints = ServiceEndpoints::new("backend");

// Add endpoints
endpoints.add_endpoint(Endpoint::new(addr1, 100));
endpoints.add_endpoint(Endpoint::new(addr2, 100));

// Update health status
endpoints.update_health(addr1, EndpointHealth::Unhealthy);
endpoints.update_health(addr2, EndpointHealth::Degraded);

// Get healthy endpoints only
let healthy = endpoints.get_healthy_endpoints();

// Get endpoints by health status
let degraded = endpoints.get_by_health(EndpointHealth::Degraded);

// Check counts
println!("Total: {}", endpoints.count());
println!("Healthy: {}", endpoints.healthy_count());
println!("Unhealthy: {}", endpoints.unhealthy_count());
```

#### Endpoint Filtering

```rust
// Add endpoints with labels
endpoints.add_endpoint_with_labels(
    Endpoint::new(addr, 100),
    vec![
        ("zone", "us-west-2a"),
        ("env", "production"),
        ("canary", "false")
    ]
);

// Filter by single label
let zone_a = endpoints.filter_by_label("zone", "us-west-2a");

// Filter by multiple labels
let prod_non_canary = endpoints.filter_by_labels(vec![
    ("env", "production"),
    ("canary", "false")
]);
```

### LoadBalancer

#### Load Balancing Strategies

```rust
use service_mesh::LoadBalancer;

// Round-robin
let mut rr_balancer = LoadBalancer::round_robin(endpoints.clone());
let endpoint = rr_balancer.select()?;

// Weighted round-robin
let mut wr_balancer = LoadBalancer::weighted_round_robin(endpoints.clone());
let endpoint = wr_balancer.select()?;

// Least connections
let mut lc_balancer = LoadBalancer::least_connections(endpoints.clone());
lc_balancer.add_connection(addr);
let endpoint = lc_balancer.select()?;
lc_balancer.remove_connection(addr);

// Consistent hashing
let ch_balancer = LoadBalancer::consistent_hash(endpoints);
let endpoint = ch_balancer.select_by_key("user-123")?;
```

## Policy Management

### AuthorizationPolicy

#### Basic Authorization

```rust
use service_mesh::AuthorizationPolicy;

let mut policy = AuthorizationPolicy::default();

// Allow specific services
policy.add_allowed_service("frontend");
policy.add_allowed_service("api-gateway");

// Add namespace restrictions
policy.add_namespace_restriction("production");
policy.add_namespace_restriction("staging");

// Check authorization
let identity = ServiceIdentity::new("frontend", "production", "cluster");
assert!(policy.is_authorized(&identity));

let denied = ServiceIdentity::new("backend", "development", "cluster");
assert!(!policy.is_authorized(&denied));
```

#### Advanced Rules

```rust
// Create policy with rules
let mut policy = AuthorizationPolicy::with_rules(vec![
    Rule::allow_service("frontend"),
    Rule::allow_namespace("production"),
    Rule::deny_service("deprecated-service"),
]);

// Add custom rule
policy.add_rule(Rule::custom(|identity| {
    // Custom logic
    identity.cluster() == "trusted-cluster"
}));
```

### CircuitBreaker

#### Configuration and Usage

```rust
use service_mesh::{CircuitBreakerConfig, CircuitState};

// Configure circuit breaker
let config = CircuitBreakerConfig::new(
    5,                          // failure threshold
    0.5,                        // failure rate threshold (50%)
    Duration::from_secs(30),    // timeout before half-open
    Duration::from_secs(10),    // half-open timeout
);

// Create breaker instance
let mut breaker = config.create_breaker();

// Check if request allowed
if breaker.allow_request() {
    match make_request().await {
        Ok(_) => breaker.record_success(),
        Err(_) => breaker.record_failure(),
    }
}

// Check state
match breaker.state() {
    CircuitState::Closed => println!("Circuit is closed"),
    CircuitState::Open => println!("Circuit is open"),
    CircuitState::HalfOpen => println!("Circuit is half-open"),
}
```

### RetryPolicy

#### Retry Configuration

```rust
use service_mesh::{RetryPolicy, RetryCondition};

// Create retry policy
let retry_policy = RetryPolicy::new(
    3,                              // max attempts
    Duration::from_millis(100),     // base delay
    vec![500, 502, 503],            // retryable status codes
);

// Add custom retry condition
retry_policy.add_condition(RetryCondition::OnTimeout);
retry_policy.add_condition(RetryCondition::OnConnectionError);

// Check if should retry
let should_retry = retry_policy.should_retry(status_code, attempt);

// Get backoff duration
let backoff = retry_policy.get_backoff(attempt);

// With max backoff
let policy = RetryPolicy::with_max_backoff(
    3,
    Duration::from_millis(100),
    Duration::from_secs(10),    // max backoff
    vec![500, 502, 503],
);
```

### TimeoutPolicy

#### Timeout Management

```rust
use service_mesh::TimeoutPolicy;

// Create timeout policy
let timeout_policy = TimeoutPolicy::new(
    Duration::from_secs(5),     // request timeout
    Duration::from_secs(30),    // stream timeout
);

// Calculate deadline
let deadline = timeout_policy.calculate_deadline();

// Apply timeout to future
use tokio::time::timeout;
let result = timeout(
    timeout_policy.request_timeout(),
    make_request()
).await;

// Chain timeouts
let policy = TimeoutPolicy::default()
    .with_request_timeout(Duration::from_secs(10))
    .with_stream_timeout(Duration::from_secs(60))
    .with_connect_timeout(Duration::from_secs(5));
```

### ServicePolicy

#### Combining Policies

```rust
use service_mesh::ServicePolicy;

// Create comprehensive service policy
let service_policy = ServicePolicy::new(
    "critical-service",
    authorization_policy,
    Some(circuit_breaker_config),
    Some(retry_policy),
    Some(timeout_policy),
);

// Apply all policies
service_policy.apply(|request| async {
    // Policies are applied in order:
    // 1. Authorization
    // 2. Circuit breaker
    // 3. Timeout
    // 4. Retry on failure
    process_request(request).await
}).await?;
```

## Metrics and Observability

### ProxyMetrics

#### Collecting Metrics

```rust
use service_mesh::ProxyMetrics;

let metrics = ProxyMetrics::new();

// Record requests
metrics.record_request();
metrics.record_success(Duration::from_millis(23));
metrics.record_failure();

// Get statistics
println!("Total requests: {}", metrics.total_requests());
println!("Success rate: {:.2}%", metrics.success_rate() * 100.0);
println!("P99 latency: {:?}", metrics.p99_latency());
println!("Active connections: {}", metrics.active_connections());

// Connection tracking
metrics.increment_connections();
metrics.decrement_connections();
```

#### Histogram Metrics

```rust
// Get latency percentiles
let p50 = metrics.p50_latency();
let p95 = metrics.p95_latency();
let p99 = metrics.p99_latency();

// Get histogram data
let histogram = metrics.latency_histogram();
for bucket in histogram.buckets() {
    println!("{:?}ms: {}", bucket.upper_bound, bucket.count);
}
```

### Distributed Tracing

#### Creating Traces

```rust
use service_mesh::{Tracer, SpanContext};

let tracer = Tracer::new("my-service");

// Start root span
let root_span = tracer.start_span("handle-request");
root_span.set_tag("http.method", "GET");
root_span.set_tag("http.path", "/api/users");

// Create child span
let db_span = tracer.start_span_with_parent("database-query", &root_span);
db_span.set_tag("db.type", "postgresql");
db_span.set_tag("db.query", "SELECT * FROM users");

// Complete spans
db_span.finish();
root_span.finish();
```

#### Trace Propagation

```rust
// Extract trace context from headers
let context = SpanContext::from_headers(&request_headers)?;

// Create span with remote parent
let span = tracer.start_span_with_context("process-request", context);

// Inject context into outgoing request
let mut headers = HashMap::new();
span.inject_headers(&mut headers);
```

#### Baggage Items

```rust
// Add baggage items (propagated to all child spans)
span.set_baggage_item("user.id", "12345");
span.set_baggage_item("request.id", "abc-123");

// Access baggage in child spans
let user_id = child_span.get_baggage_item("user.id");
```

## Error Handling

### Error Types

```rust
use service_mesh::Error;

match result {
    Err(Error::Unauthorized) => {
        // Handle authorization failure
        return Response::forbidden();
    }
    Err(Error::ServiceNotFound(service)) => {
        // Handle missing service
        log::error!("Service {} not found", service);
        return Response::service_unavailable();
    }
    Err(Error::CircuitOpen) => {
        // Handle circuit breaker
        return Response::too_many_requests();
    }
    Err(Error::Timeout) => {
        // Handle timeout
        return Response::gateway_timeout();
    }
    Err(Error::Certificate(msg)) => {
        // Handle certificate issues
        log::error!("Certificate error: {}", msg);
        return Response::internal_error();
    }
    Ok(response) => response,
}
```

### Custom Error Handling

```rust
// Implement custom error handler
impl From<MyError> for service_mesh::Error {
    fn from(err: MyError) -> Self {
        Error::Custom(err.to_string())
    }
}

// Use with retry policy
let retry_policy = RetryPolicy::with_error_classifier(|error| {
    match error {
        Error::Timeout | Error::Connection(_) => true,  // Retry
        Error::Unauthorized => false,                    // Don't retry
        _ => false,
    }
});
```

## Complete Example

### Setting Up a Service with Full Mesh Features

```rust
use service_mesh::*;
use std::time::Duration;
use std::net::SocketAddr;

async fn setup_service_mesh() -> Result<(), Error> {
    // 1. Setup Certificate Authority
    let ca = CertificateAuthority::new(
        "production-mesh",
        Duration::from_secs(10 * 365 * 24 * 3600)
    )?;

    // 2. Create service identity and certificate
    let identity = ServiceIdentity::new("api-gateway", "production", "us-west-2");
    let cert = ca.issue_certificate(&identity, Duration::from_secs(90 * 24 * 3600))?;

    // 3. Configure sidecar proxy
    let tls_config = TlsConfig::new(cert, ca.clone());
    let proxy_config = ProxyConfig::new(
        "0.0.0.0:15001".parse()?,
        "127.0.0.1:8080".parse()?,
        tls_config
    );

    let proxy = SidecarProxy::new(proxy_config);

    // 4. Setup service registry
    let mut registry = ServiceRegistry::new();
    registry.register_service("backend", vec![
        Endpoint::new("10.0.1.1:8080".parse()?, 100),
        Endpoint::new("10.0.1.2:8080".parse()?, 100),
    ]);

    // 5. Configure policies
    let mut auth_policy = AuthorizationPolicy::default();
    auth_policy.add_allowed_service("api-gateway");
    auth_policy.add_namespace_restriction("production");

    let circuit_breaker = CircuitBreakerConfig::new(
        5,
        0.5,
        Duration::from_secs(30),
        Duration::from_secs(10)
    );

    let retry_policy = RetryPolicy::new(
        3,
        Duration::from_millis(100),
        vec![500, 502, 503]
    );

    let timeout_policy = TimeoutPolicy::new(
        Duration::from_secs(5),
        Duration::from_secs(30)
    );

    // 6. Create service policy
    let service_policy = ServicePolicy::new(
        "backend",
        auth_policy,
        Some(circuit_breaker),
        Some(retry_policy),
        Some(timeout_policy)
    );

    // 7. Setup tracing
    let tracer = Tracer::new("api-gateway");

    // 8. Start proxy
    proxy.start().await?;

    // 9. Handle requests
    loop {
        let span = tracer.start_span("handle-request");

        match proxy.forward_request("backend", &registry, &service_policy).await {
            Ok(response) => {
                span.set_tag("response.status", "200");
                // Process response
            }
            Err(error) => {
                span.set_tag("error", &error.to_string());
                // Handle error
            }
        }

        span.finish();
    }
}
```

## Testing

### Unit Testing Example

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_discovery() {
        let mut registry = ServiceRegistry::new();

        registry.register_service("test-service", vec![
            Endpoint::new("127.0.0.1:8080".parse().unwrap(), 100)
        ]);

        let endpoints = registry.discover("test-service");
        assert!(endpoints.is_some());
        assert_eq!(endpoints.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let config = CircuitBreakerConfig::default();
        let mut breaker = config.create_breaker();

        // Record failures
        for _ in 0..6 {
            breaker.record_failure();
        }

        // Circuit should be open
        assert!(!breaker.allow_request());
    }
}
```

## Performance Considerations

### Connection Pooling

```rust
// Optimize connection reuse
let pool_config = ConnectionPoolConfig {
    max_connections: 200,
    max_idle_connections: 50,
    idle_timeout: Duration::from_secs(300),
    connection_timeout: Duration::from_secs(10),
};
```

### Caching

```rust
// Enable discovery caching
registry.enable_caching(Duration::from_secs(60));

// Cache certificates
cert_manager.set_cache_ttl(Duration::from_secs(3600));
```

### Batch Operations

```rust
// Batch metrics reporting
metrics.enable_batching(100, Duration::from_secs(1));

// Batch trace exports
tracer.set_batch_size(50);
```