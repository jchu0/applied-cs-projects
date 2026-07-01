# Service Mesh

A service mesh data plane and control plane built from scratch in Rust. It provides
secure, observable service-to-service communication through a sidecar proxy with mTLS
(SPIFFE identity), service discovery and load balancing, traffic-management policies
(retries, circuit breaking, timeouts), Envoy-compatible xDS configuration, distributed
tracing, and a Kubernetes mutating webhook for sidecar injection.

## Features

- **Sidecar proxy** — inbound (15006) / outbound (15001) / admin (15000) TCP listeners
  that parse and forward HTTP/1.1 with retry and load balancing (`SidecarProxy`).
- **mTLS with SPIFFE identity** — an `rcgen`-based `CertificateAuthority` issues
  `spiffe://cluster.local/ns/<ns>/sa/<sa>` SAN certificates; `CertManager` tracks
  rotation at 80% of lifetime.
- **TLS transport** — `tokio-rustls` `TlsManager` builds acceptor/connector with client
  auth; `TlsStream::peer_spiffe_id()` extracts the peer SPIFFE ID via `x509-parser`.
- **Service discovery** — concurrent `ServiceRegistry` (DashMap) keyed by
  `ServiceKey{name, namespace, port}` holding `ServiceEndpoints` with health state.
- **Load balancing** — `LoadBalancer` with RoundRobin, LeastConnections, Random, and
  RingHash strategies, filtering to healthy endpoints (`LoadBalancerType`).
- **Traffic management** — `CircuitBreaker` (Closed / Open / HalfOpen), `RetryPolicy`
  with exponential backoff and jitter, and `TimeoutPolicy`.
- **xDS control plane** — `XdsServer` / `XdsClient` with CDS, EDS, LDS, and RDS handlers
  over Envoy-compatible resource types.
- **Kubernetes injection** — `SidecarInjector` generates a JSON patch (init container,
  sidecar, volumes, annotation); `WebhookServer` serves admission reviews.
- **Observability** — W3C `traceparent` context propagation (`Tracer`, `Span`) and a
  hand-written Prometheus exporter (`ProxyMetrics::to_prometheus()`).

## Architecture

```mermaid
flowchart TD
    App[Local application] -->|outbound 15001| Proxy[SidecarProxy]
    Peer[Peer sidecar] -->|inbound 15006| Proxy
    Proxy -->|admin 15000 metrics, health| Admin[Admin endpoint]
    Proxy --> Registry[ServiceRegistry]
    Proxy --> LB[LoadBalancer]
    Proxy --> CB[CircuitBreaker and RetryPolicy]
    Proxy --> TLS[TlsManager mTLS]
    Proxy --> Tracer[Tracer W3C context]
    TLS --> Cert[CertificateAuthority and CertManager]
    XdsServer[XdsServer control plane] -->|CDS EDS LDS RDS| XdsClient[XdsClient]
    XdsClient --> Registry
    Webhook[WebhookServer] --> Injector[SidecarInjector]
```

| Component | Module | Responsibility |
|-----------|--------|----------------|
| Sidecar proxy | `proxy` | Inbound/outbound/admin listeners, HTTP/1.1 forwarding, retries |
| Certificates | `cert` | rcgen CA, SPIFFE SAN cert issuance, rotation tracking |
| TLS transport | `tls` | rustls acceptor/connector, SPIFFE extraction from peer cert |
| Discovery & LB | `discovery` | Service registry, endpoint health, load-balancer strategies |
| Policies | `policy` | Circuit breaker, retry/backoff, timeout, authorization |
| Metrics | `metrics` | Counters/gauges/histograms, Prometheus text export |
| Tracing | `tracing_mesh` | Spans, W3C traceparent inject/extract |
| xDS | `xds` | Envoy-compatible CDS/EDS/LDS/RDS server and client |
| Kubernetes | `k8s` | Sidecar injection patch generation, admission webhook |

## Quick Start

### Prerequisites

- Rust 1.70+ (edition 2021) and Cargo.
- No external services are required to build or run the test suite.

### Installation

```bash
cargo build
```

### Running

The crate is a library; exercise it through the test suite or your own binary:

```bash
cargo test
```

## Usage

Issue a SPIFFE certificate, build a registry, and select an endpoint:

```rust
use service_mesh::discovery::{LoadBalancerType, ServiceKey};
use service_mesh::{
    CertificateAuthority, Endpoint, EndpointHealth, LoadBalancer, ServiceEndpoints,
    ServiceIdentity, ServiceRegistry,
};
use std::time::Duration;

// Issue an mTLS certificate for a workload identity.
let ca = CertificateAuthority::new(Duration::from_secs(24 * 3600)).unwrap();
let identity = ServiceIdentity::new("default", "frontend");
let cert = ca.issue_certificate(&identity).unwrap();
assert!(!cert.cert_chain.is_empty());
assert!(identity.spiffe_id.starts_with("spiffe://cluster.local/ns/default/sa/frontend"));

// Register a service and load-balance across its healthy endpoints.
let registry = ServiceRegistry::new();
registry.register(
    ServiceKey::new("backend", "default", 8080),
    ServiceEndpoints {
        endpoints: vec![Endpoint {
            address: "10.0.0.1:8080".parse().unwrap(),
            weight: 100,
            health: EndpointHealth::Healthy,
            metadata: Default::default(),
            tls_identity: ServiceIdentity::new("default", "backend"),
        }],
        load_balancer: LoadBalancerType::RoundRobin,
        policy: Default::default(),
    },
);

let lb = LoadBalancer::new(LoadBalancerType::RoundRobin);
let endpoints = registry.get_by_name("backend").unwrap();
let chosen = lb.select(&endpoints.endpoints).unwrap();
assert_eq!(chosen.health, EndpointHealth::Healthy);
```

Trip and recover a circuit breaker:

```rust
use service_mesh::{CircuitBreaker, CircuitBreakerConfig};
use std::time::Duration;

let cb = CircuitBreaker::new(CircuitBreakerConfig {
    consecutive_failures: 3,
    success_threshold: 2,
    base_ejection_time: Duration::from_millis(100),
    ..Default::default()
});

for _ in 0..3 {
    cb.record_failure();
}
assert!(cb.is_open());

std::thread::sleep(Duration::from_millis(150)); // ejection window elapses
assert!(!cb.is_open());                          // now half-open
cb.record_success();
cb.record_success();                             // closes the circuit
```

## What's Real vs Simulated

- **Real:** rcgen-based certificate issuance with SPIFFE SANs; mTLS transport via
  `tokio-rustls`; circuit-breaker, retry/backoff and timeout logic; load-balancer
  strategies and the concurrent service registry; HTTP/1.1 sidecar proxying over real
  TCP listeners; W3C trace-context propagation; the hand-written Prometheus exporter;
  xDS CDS/EDS/LDS/RDS resource handling; and webhook JSON-patch generation.
- **Simulated / in-process:** there is no live Kubernetes cluster connection — the
  webhook operates on `AdmissionReview` values directly. The xDS server and client share
  resources in-memory rather than over a real gRPC streaming transport (`tonic`/`hyper`
  are dependencies but the xDS exchange is in-process). No iptables interception is
  performed; the init-container/iptables redirection is described for Kubernetes but not
  run locally.

## Testing

```bash
cargo test
```

Integration tests under `tests/` cover the proxy, certificate authority, service
discovery and load balancing, circuit-breaker behaviour, retry policy, concurrent
registry access, tracing, and a network simulator. Inline `#[cfg(test)]` modules cover
each component (cert, discovery, policy, proxy, tls, tracing, xDS, k8s). No external
services are required.

## Project Structure

```
13-service-mesh/
  README.md              # This file
  Cargo.toml
  src/
    lib.rs               # Public exports and error types
    config.rs            # ProxyConfig, ServiceIdentity, TlsConfig
    cert.rs              # CertificateAuthority, CertManager, IssuedCert
    tls.rs               # TlsManager, TlsStream, SecureConnection
    discovery.rs         # ServiceRegistry, LoadBalancer, Endpoint
    policy.rs            # CircuitBreaker, RetryPolicy, TimeoutPolicy
    proxy.rs             # SidecarProxy and HTTP forwarding
    metrics.rs           # ProxyMetrics and Prometheus export
    tracing_mesh.rs      # Tracer, Span, SpanContext
    xds/                 # XdsServer, XdsClient, Envoy-compatible types
    k8s/                 # SidecarInjector, WebhookServer
  tests/                 # Integration tests
  docs/BLUEPRINT.md      # Full architecture and design
```

## License

MIT — see [LICENSE](../LICENSE)
