# Service Mesh (Linkerd/Istio-lite)

## Executive Summary

A service mesh implementation providing secure, observable, and reliable service-to-service communication. Features include sidecar proxy pattern, automatic mTLS, service discovery, traffic management (retries, timeouts, circuit breaking), and distributed tracing. Integrates with Kubernetes for automatic sidecar injection.

> **Concepts covered:** [§05 Authentication / mTLS](../../05-cross-cutting-concerns/security/authentication/authentication.md) · [§05 Observability](../../05-cross-cutting-concerns/observability/) · [§07 Kubernetes](../../07-infrastructure/kubernetes/kubernetes-guide.md) · [§01 Rust async](../../01-software-engineering/rust/05-async-rust/rust-async.md) (Tokio-based sidecar). Pairs with [Project 02 (microservice platform — what the mesh wraps)](../02-microservice-platform/) and [Project 14 (network stack)](../14-network-stack/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## System Architecture

```
                        Control Plane
    +----------------------------------------------------------+
    |                                                          |
    |  +-------------+  +------------+  +------------------+   |
    |  |   Service   |  |   Policy   |  |   Certificate    |   |
    |  |   Registry  |  |   Engine   |  |   Authority      |   |
    |  +------+------+  +-----+------+  +--------+---------+   |
    |         |               |                  |              |
    +---------|---------------|------------------|-------------+
              |               |                  |
              v               v                  v
         xDS API          xDS API           mTLS certs
              |               |                  |
    +---------+---------------+------------------+-------------+
    |                     Data Plane                           |
    |                                                          |
    |  +------------------+         +------------------+       |
    |  |    Pod A         |         |    Pod B         |       |
    |  |  +--------+      |  mTLS   |      +--------+  |       |
    |  |  | App A  |      |  <--->  |      | App B  |  |       |
    |  |  +---+----+      |         |      +----+---+  |       |
    |  |      |           |         |           |      |       |
    |  |  +---v----+      |         |      +----v---+  |       |
    |  |  |Sidecar |      |         |      |Sidecar |  |       |
    |  |  | Proxy  |<-----|---------|----->| Proxy  |  |       |
    |  |  +--------+      |         |      +--------+  |       |
    |  +------------------+         +------------------+       |
    |                                                          |
    +----------------------------------------------------------+

Traffic Flow:
  App A -> localhost:port -> Sidecar A -> [mTLS] -> Sidecar B -> App B

Sidecar Proxy Detail:
+------------------------------------------------------------------+
|                        Sidecar Proxy                              |
+------------------------------------------------------------------+
|  Inbound         |  Core              |  Outbound                 |
|  +-----------+   |  +-------------+   |  +------------------+     |
|  | TLS Term  |   |  | Connection  |   |  | Service         |     |
|  | Authz     |   |  | Pool        |   |  | Discovery       |     |
|  | Metrics   |   |  | Circuit     |   |  | Load Balancing  |     |
|  +-----------+   |  | Breaker     |   |  | Retry/Timeout   |     |
|                  |  +-------------+   |  | TLS Origination |     |
|                  |                    |  +------------------+     |
+------------------------------------------------------------------+
```

---

## Core Data Structures

### Proxy Configuration

```rust
pub struct ProxyConfig {
    // Identity
    pub service_name: String,
    pub namespace: String,
    pub workload_name: String,

    // Networking
    pub inbound_port: u16,         // Traffic from mesh
    pub outbound_port: u16,        // Traffic to mesh
    pub admin_port: u16,           // Metrics, health

    // mTLS
    pub tls_config: TlsConfig,

    // Traffic policy
    pub retry_policy: RetryPolicy,
    pub timeout: Duration,
    pub circuit_breaker: CircuitBreakerConfig,

    // Tracing
    pub tracing_config: TracingConfig,
}

pub struct TlsConfig {
    pub cert_chain: Vec<u8>,
    pub private_key: Vec<u8>,
    pub root_ca: Vec<u8>,
    pub cert_expiry: SystemTime,
    pub identity: ServiceIdentity,
}

pub struct ServiceIdentity {
    pub spiffe_id: String,         // spiffe://cluster.local/ns/default/sa/myservice
    pub service_account: String,
    pub namespace: String,
}
```

### Traffic Management

```rust
pub struct RetryPolicy {
    pub max_retries: u32,
    pub retry_on: Vec<RetryCondition>,
    pub backoff: BackoffConfig,
}

pub enum RetryCondition {
    ConnectionFailure,
    Status5xx,
    Reset,
    ConnectFailure,
    Retriable4xx,
    StatusCode(u16),
}

pub struct BackoffConfig {
    pub base_interval: Duration,
    pub max_interval: Duration,
    pub jitter: f64,
}

pub struct CircuitBreakerConfig {
    pub consecutive_failures: u32,
    pub interval: Duration,
    pub base_ejection_time: Duration,
    pub max_ejection_percent: u32,
    pub success_threshold: u32,
}

pub enum CircuitState {
    Closed,                        // Normal operation
    Open {                         // Rejecting requests
        opened_at: Instant,
        failures: u32,
    },
    HalfOpen {                     // Testing recovery
        successes: u32,
        allowed: u32,
    },
}

pub struct TimeoutPolicy {
    pub request_timeout: Duration,
    pub idle_timeout: Duration,
}
```

### Service Discovery

```rust
pub struct ServiceRegistry {
    services: HashMap<ServiceKey, ServiceEndpoints>,
    watchers: Vec<Watcher>,
}

pub struct ServiceKey {
    pub name: String,
    pub namespace: String,
    pub port: u16,
}

pub struct ServiceEndpoints {
    pub endpoints: Vec<Endpoint>,
    pub load_balancer: LoadBalancer,
    pub policy: ServicePolicy,
}

pub struct Endpoint {
    pub address: SocketAddr,
    pub weight: u32,
    pub health: EndpointHealth,
    pub metadata: HashMap<String, String>,
    pub tls_identity: ServiceIdentity,
}

pub enum EndpointHealth {
    Healthy,
    Unhealthy,
    Unknown,
}

pub enum LoadBalancer {
    RoundRobin { index: AtomicUsize },
    LeastConnections { connections: HashMap<SocketAddr, usize> },
    Random,
    RingHash { hash_ring: HashRing },
    Maglev { table: MaglevTable },
}
```

---

## Sidecar Proxy Implementation

### Proxy Core

```rust
pub struct SidecarProxy {
    config: Arc<RwLock<ProxyConfig>>,

    // Connection management
    inbound_listener: TcpListener,
    outbound_listener: TcpListener,
    connection_pool: ConnectionPool,

    // Service discovery
    service_registry: Arc<RwLock<ServiceRegistry>>,

    // TLS
    tls_acceptor: TlsAcceptor,
    tls_connector: TlsConnector,
    cert_manager: CertManager,

    // Traffic management
    circuit_breakers: HashMap<ServiceKey, CircuitBreaker>,

    // Observability
    metrics: ProxyMetrics,
    tracer: Tracer,
}

impl SidecarProxy {
    pub async fn run(&mut self) -> Result<()> {
        tokio::select! {
            _ = self.handle_inbound() => {},
            _ = self.handle_outbound() => {},
            _ = self.cert_rotation_loop() => {},
            _ = self.xds_sync_loop() => {},
            _ = self.health_check_loop() => {},
        }
        Ok(())
    }

    async fn handle_inbound(&self) -> Result<()> {
        loop {
            let (stream, peer_addr) = self.inbound_listener.accept().await?;
            let proxy = self.clone();

            tokio::spawn(async move {
                if let Err(e) = proxy.process_inbound(stream, peer_addr).await {
                    log::error!("Inbound error: {}", e);
                }
            });
        }
    }

    async fn process_inbound(&self, stream: TcpStream, peer: SocketAddr) -> Result<()> {
        let start = Instant::now();

        // Step 1: TLS termination
        let tls_stream = self.tls_acceptor.accept(stream).await?;
        let peer_identity = extract_peer_identity(&tls_stream)?;

        // Step 2: Authorization check
        if !self.authorize(&peer_identity).await? {
            self.metrics.auth_denied.inc();
            return Err(Error::Unauthorized);
        }

        // Step 3: Parse HTTP request
        let (parts, body) = parse_http_request(tls_stream).await?;

        // Step 4: Add tracing headers
        let trace_context = self.extract_or_create_trace(&parts.headers);

        // Step 5: Forward to local application
        let local_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            self.config.read().app_port,
        );

        let response = self.forward_request(local_addr, parts, body).await?;

        // Step 6: Record metrics
        let latency = start.elapsed();
        self.metrics.request_latency.observe(latency.as_secs_f64());
        self.metrics.requests_total.inc();

        Ok(())
    }

    async fn handle_outbound(&self) -> Result<()> {
        loop {
            let (stream, _) = self.outbound_listener.accept().await?;
            let proxy = self.clone();

            tokio::spawn(async move {
                if let Err(e) = proxy.process_outbound(stream).await {
                    log::error!("Outbound error: {}", e);
                }
            });
        }
    }

    async fn process_outbound(&self, stream: TcpStream) -> Result<Response> {
        let start = Instant::now();

        // Step 1: Parse request from local app
        let (parts, body) = parse_http_request(stream).await?;
        let target_service = extract_target_service(&parts)?;

        // Step 2: Service discovery
        let endpoints = self.service_registry.read()
            .get(&target_service)
            .ok_or(Error::ServiceNotFound)?
            .clone();

        // Step 3: Check circuit breaker
        let circuit_breaker = self.circuit_breakers.get(&target_service);
        if let Some(cb) = circuit_breaker {
            if cb.is_open() {
                self.metrics.circuit_open.inc();
                return Err(Error::CircuitOpen);
            }
        }

        // Step 4: Load balancing
        let endpoint = endpoints.load_balancer.select(&endpoints.endpoints)?;

        // Step 5: Establish mTLS connection
        let conn = self.get_or_create_connection(&endpoint).await?;

        // Step 6: Forward with retry policy
        let response = self.forward_with_retry(
            conn,
            parts,
            body,
            &self.config.read().retry_policy,
        ).await;

        // Step 7: Update circuit breaker
        if let Some(cb) = circuit_breaker {
            match &response {
                Ok(_) => cb.record_success(),
                Err(_) => cb.record_failure(),
            }
        }

        // Step 8: Record metrics
        let latency = start.elapsed();
        self.metrics.request_latency
            .with_label("service", &target_service.name)
            .observe(latency.as_secs_f64());

        response
    }
}
```

### Retry Logic

```rust
impl SidecarProxy {
    async fn forward_with_retry(
        &self,
        mut conn: Connection,
        parts: Parts,
        body: Bytes,
        policy: &RetryPolicy,
    ) -> Result<Response> {
        let mut attempts = 0;
        let mut last_error = None;

        loop {
            attempts += 1;

            match self.forward_once(&mut conn, &parts, &body).await {
                Ok(response) => {
                    if self.should_retry(&response, policy) && attempts <= policy.max_retries {
                        last_error = Some(Error::RetryableResponse(response.status()));
                        let backoff = self.calculate_backoff(attempts, policy);
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    return Ok(response);
                }
                Err(e) if self.is_retryable_error(&e, policy) && attempts <= policy.max_retries => {
                    last_error = Some(e);
                    let backoff = self.calculate_backoff(attempts, policy);
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }

    fn calculate_backoff(&self, attempt: u32, policy: &RetryPolicy) -> Duration {
        let base = policy.backoff.base_interval.as_millis() as f64;
        let max = policy.backoff.max_interval.as_millis() as f64;

        // Exponential backoff with jitter
        let exponential = base * 2f64.powi(attempt as i32 - 1);
        let capped = exponential.min(max);

        let jitter = rand::thread_rng().gen_range(0.0..policy.backoff.jitter);
        let with_jitter = capped * (1.0 + jitter);

        Duration::from_millis(with_jitter as u64)
    }
}
```

### Circuit Breaker

```rust
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    state: RwLock<CircuitState>,
    failures: AtomicU32,
    successes: AtomicU32,
}

impl CircuitBreaker {
    pub fn is_open(&self) -> bool {
        let state = self.state.read();
        match *state {
            CircuitState::Open { opened_at, .. } => {
                // Check if we should transition to half-open
                if opened_at.elapsed() > self.config.base_ejection_time {
                    drop(state);
                    let mut state = self.state.write();
                    *state = CircuitState::HalfOpen {
                        successes: 0,
                        allowed: 1,
                    };
                    false
                } else {
                    true
                }
            }
            CircuitState::HalfOpen { allowed, .. } => {
                // Allow limited requests through
                allowed == 0
            }
            CircuitState::Closed => false,
        }
    }

    pub fn record_success(&self) {
        let mut state = self.state.write();
        match *state {
            CircuitState::HalfOpen { successes, .. } => {
                let new_successes = successes + 1;
                if new_successes >= self.config.success_threshold {
                    *state = CircuitState::Closed;
                    self.failures.store(0, Ordering::SeqCst);
                } else {
                    *state = CircuitState::HalfOpen {
                        successes: new_successes,
                        allowed: 1,  // Allow one more
                    };
                }
            }
            CircuitState::Closed => {
                self.failures.store(0, Ordering::SeqCst);
            }
            _ => {}
        }
    }

    pub fn record_failure(&self) {
        let failures = self.failures.fetch_add(1, Ordering::SeqCst) + 1;

        if failures >= self.config.consecutive_failures {
            let mut state = self.state.write();
            *state = CircuitState::Open {
                opened_at: Instant::now(),
                failures,
            };
        }
    }
}
```

---

## Certificate Authority and mTLS

### Certificate Manager

```rust
pub struct CertificateAuthority {
    // Root CA
    root_cert: X509,
    root_key: PKey<Private>,

    // Intermediate CA for signing
    signing_cert: X509,
    signing_key: PKey<Private>,

    // Configuration
    cert_ttl: Duration,
    key_size: u32,
}

impl CertificateAuthority {
    pub fn issue_certificate(&self, identity: &ServiceIdentity) -> Result<IssuedCert> {
        // Generate key pair
        let key = PKey::generate_rsa(self.key_size)?;

        // Build certificate
        let mut builder = X509::builder()?;

        // Set serial number
        let serial = BigNum::from_u32(rand::random())?;
        builder.set_serial_number(&serial.to_asn1_integer()?)?;

        // Set validity
        let not_before = Asn1Time::days_from_now(0)?;
        let not_after = Asn1Time::from_unix(
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64
                + self.cert_ttl.as_secs() as i64
        )?;
        builder.set_not_before(&not_before)?;
        builder.set_not_after(&not_after)?;

        // Set subject (SPIFFE ID)
        let mut name = X509Name::builder()?;
        name.append_entry_by_text("O", "mesh")?;
        name.append_entry_by_text("CN", &identity.service_account)?;
        let name = name.build();
        builder.set_subject_name(&name)?;

        // Set issuer
        builder.set_issuer_name(self.signing_cert.subject_name())?;

        // Add SPIFFE SAN
        let san = SubjectAlternativeName::new()
            .uri(&identity.spiffe_id)
            .build(&builder.x509v3_context(Some(&self.signing_cert), None))?;
        builder.append_extension(san)?;

        // Set public key and sign
        builder.set_pubkey(&key)?;
        builder.sign(&self.signing_key, MessageDigest::sha256())?;

        let cert = builder.build();

        Ok(IssuedCert {
            cert_chain: vec![cert.to_pem()?, self.signing_cert.to_pem()?],
            private_key: key.private_key_to_pem_pkcs8()?,
            expiry: SystemTime::now() + self.cert_ttl,
        })
    }
}

pub struct CertManager {
    ca_client: CaClient,
    current_cert: RwLock<IssuedCert>,
    identity: ServiceIdentity,

    // Rotation
    rotation_threshold: f64,  // Rotate at 80% of lifetime
}

impl CertManager {
    pub async fn rotation_loop(&self) {
        loop {
            let cert = self.current_cert.read().clone();
            let remaining = cert.expiry.duration_since(SystemTime::now())
                .unwrap_or(Duration::ZERO);

            let total_lifetime = Duration::from_secs(3600);  // Configured
            let threshold = total_lifetime.mul_f64(self.rotation_threshold);

            if remaining < threshold {
                // Request new certificate
                match self.ca_client.request_certificate(&self.identity).await {
                    Ok(new_cert) => {
                        *self.current_cert.write() = new_cert;
                        log::info!("Certificate rotated successfully");
                    }
                    Err(e) => {
                        log::error!("Certificate rotation failed: {}", e);
                    }
                }
            }

            // Check every minute
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    }
}
```

---

## Control Plane

### Service Registry (xDS Server)

```rust
pub struct ControlPlane {
    // Service discovery
    service_registry: Arc<RwLock<ServiceRegistry>>,

    // Policy
    policy_engine: PolicyEngine,

    // Certificate authority
    ca: CertificateAuthority,

    // Connected proxies
    proxies: HashMap<ProxyId, ProxyConnection>,

    // Kubernetes integration
    k8s_client: Option<KubeClient>,
}

impl ControlPlane {
    pub async fn xds_stream(&self, proxy_id: ProxyId, mut stream: Streaming<DiscoveryRequest>) {
        // Initial configuration push
        self.send_initial_config(&proxy_id).await;

        // Process requests
        while let Some(request) = stream.message().await? {
            match request.type_url.as_str() {
                "type.googleapis.com/envoy.config.cluster.v3.Cluster" => {
                    self.handle_cds_request(&proxy_id, request).await;
                }
                "type.googleapis.com/envoy.config.endpoint.v3.ClusterLoadAssignment" => {
                    self.handle_eds_request(&proxy_id, request).await;
                }
                "type.googleapis.com/envoy.config.listener.v3.Listener" => {
                    self.handle_lds_request(&proxy_id, request).await;
                }
                "type.googleapis.com/envoy.config.route.v3.RouteConfiguration" => {
                    self.handle_rds_request(&proxy_id, request).await;
                }
                _ => {
                    log::warn!("Unknown xDS type: {}", request.type_url);
                }
            }
        }
    }

    async fn handle_eds_request(&self, proxy_id: &ProxyId, request: DiscoveryRequest) {
        let registry = self.service_registry.read();

        let mut assignments = Vec::new();
        for resource_name in &request.resource_names {
            if let Some(service) = registry.get_by_name(resource_name) {
                let assignment = ClusterLoadAssignment {
                    cluster_name: resource_name.clone(),
                    endpoints: service.endpoints.iter().map(|ep| {
                        LocalityLbEndpoints {
                            lb_endpoints: vec![LbEndpoint {
                                endpoint: Some(Endpoint {
                                    address: Some(Address {
                                        address: Some(SocketAddress {
                                            address: ep.address.ip().to_string(),
                                            port: ep.address.port() as u32,
                                        }),
                                    }),
                                }),
                                health_status: ep.health.into(),
                                load_balancing_weight: ep.weight,
                            }],
                        }
                    }).collect(),
                };
                assignments.push(assignment);
            }
        }

        self.send_discovery_response(proxy_id, assignments).await;
    }
}
```

### Policy Engine

```rust
pub struct PolicyEngine {
    policies: HashMap<ServiceKey, ServicePolicy>,
}

pub struct ServicePolicy {
    // Traffic policy
    pub retry: Option<RetryPolicy>,
    pub timeout: Option<Duration>,
    pub circuit_breaker: Option<CircuitBreakerConfig>,

    // Security policy
    pub mtls_mode: MtlsMode,
    pub authorization: AuthorizationPolicy,

    // Traffic routing
    pub routes: Vec<RouteRule>,
}

pub struct RouteRule {
    pub match_condition: RouteMatch,
    pub destination: RouteDestination,
    pub weight: u32,
}

pub struct RouteMatch {
    pub uri: Option<StringMatch>,
    pub headers: Vec<HeaderMatch>,
    pub method: Option<String>,
}

pub struct RouteDestination {
    pub service: ServiceKey,
    pub subset: Option<String>,  // For canary deployments
}

pub enum MtlsMode {
    Disable,
    Permissive,  // Accept both plaintext and mTLS
    Strict,      // mTLS only
}

pub struct AuthorizationPolicy {
    pub action: AuthAction,
    pub rules: Vec<AuthRule>,
}

pub enum AuthAction {
    Allow,
    Deny,
}

pub struct AuthRule {
    pub from: Vec<Source>,
    pub to: Vec<Operation>,
}

pub struct Source {
    pub principals: Vec<String>,     // SPIFFE IDs
    pub namespaces: Vec<String>,
    pub ip_blocks: Vec<IpNet>,
}
```

---

## Kubernetes Integration

### Mutating Webhook for Sidecar Injection

```rust
pub struct SidecarInjector {
    config: InjectorConfig,
}

impl SidecarInjector {
    pub async fn handle_admission(&self, review: AdmissionReview) -> AdmissionResponse {
        let pod: Pod = serde_json::from_value(review.request.object)?;

        // Check if injection should be skipped
        if self.should_skip(&pod) {
            return AdmissionResponse::allowed();
        }

        // Build patch
        let mut patches = Vec::new();

        // Add init container for iptables setup
        patches.push(json!({
            "op": "add",
            "path": "/spec/initContainers/-",
            "value": self.init_container()
        }));

        // Add sidecar container
        patches.push(json!({
            "op": "add",
            "path": "/spec/containers/-",
            "value": self.sidecar_container(&pod)
        }));

        // Add volumes for certificates
        patches.push(json!({
            "op": "add",
            "path": "/spec/volumes/-",
            "value": {
                "name": "mesh-certs",
                "emptyDir": { "medium": "Memory" }
            }
        }));

        AdmissionResponse::patch(patches)
    }

    fn init_container(&self) -> serde_json::Value {
        json!({
            "name": "mesh-init",
            "image": "mesh-proxy:init",
            "securityContext": {
                "capabilities": {
                    "add": ["NET_ADMIN"]
                }
            },
            "command": ["/init-iptables.sh"],
            "env": [
                {"name": "INBOUND_PORT", "value": "15006"},
                {"name": "OUTBOUND_PORT", "value": "15001"}
            ]
        })
    }

    fn sidecar_container(&self, pod: &Pod) -> serde_json::Value {
        let service_account = pod.spec.service_account_name.clone()
            .unwrap_or_else(|| "default".to_string());

        json!({
            "name": "mesh-proxy",
            "image": "mesh-proxy:latest",
            "ports": [
                {"containerPort": 15006, "name": "inbound"},
                {"containerPort": 15001, "name": "outbound"},
                {"containerPort": 15000, "name": "admin"}
            ],
            "env": [
                {"name": "SERVICE_NAME", "value": pod.metadata.name},
                {"name": "NAMESPACE", "value": pod.metadata.namespace},
                {"name": "SERVICE_ACCOUNT", "value": service_account}
            ],
            "volumeMounts": [
                {"name": "mesh-certs", "mountPath": "/etc/mesh/certs"}
            ],
            "readinessProbe": {
                "httpGet": {"path": "/ready", "port": 15000},
                "initialDelaySeconds": 2,
                "periodSeconds": 5
            }
        })
    }
}
```

### Custom Resource Definitions

```yaml
# TrafficPolicy CRD
apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: trafficpolicies.mesh.io
spec:
  group: mesh.io
  versions:
    - name: v1
      served: true
      storage: true
      schema:
        openAPIV3Schema:
          type: object
          properties:
            spec:
              type: object
              properties:
                selector:
                  type: object
                  properties:
                    matchLabels:
                      type: object
                retry:
                  type: object
                  properties:
                    maxRetries:
                      type: integer
                    retryOn:
                      type: array
                      items:
                        type: string
                timeout:
                  type: string
                circuitBreaker:
                  type: object
                  properties:
                    consecutiveFailures:
                      type: integer
                    ejectionTime:
                      type: string
  scope: Namespaced
  names:
    plural: trafficpolicies
    singular: trafficpolicy
    kind: TrafficPolicy
```

---

## Distributed Tracing

```rust
pub struct Tracer {
    service_name: String,
    collector_endpoint: String,
    sampler: Sampler,
}

impl Tracer {
    pub fn start_span(&self, name: &str, parent: Option<&SpanContext>) -> Span {
        let trace_id = parent
            .map(|p| p.trace_id)
            .unwrap_or_else(|| rand::random());

        let span_id: u64 = rand::random();
        let parent_span_id = parent.map(|p| p.span_id);

        Span {
            trace_id,
            span_id,
            parent_span_id,
            name: name.to_string(),
            service_name: self.service_name.clone(),
            start_time: SystemTime::now(),
            end_time: None,
            tags: HashMap::new(),
            logs: Vec::new(),
        }
    }

    pub fn extract_context(&self, headers: &HeaderMap) -> Option<SpanContext> {
        // W3C Trace Context format
        if let Some(traceparent) = headers.get("traceparent") {
            let parts: Vec<&str> = traceparent.to_str().ok()?.split('-').collect();
            if parts.len() >= 4 {
                return Some(SpanContext {
                    trace_id: u128::from_str_radix(parts[1], 16).ok()?,
                    span_id: u64::from_str_radix(parts[2], 16).ok()?,
                    flags: u8::from_str_radix(parts[3], 16).ok()?,
                });
            }
        }

        // Jaeger format fallback
        if let Some(uber_trace) = headers.get("uber-trace-id") {
            // Parse Jaeger format
        }

        None
    }

    pub fn inject_context(&self, context: &SpanContext, headers: &mut HeaderMap) {
        let traceparent = format!(
            "00-{:032x}-{:016x}-{:02x}",
            context.trace_id,
            context.span_id,
            context.flags
        );
        headers.insert("traceparent", traceparent.parse().unwrap());
    }
}
```

---

## Metrics and Monitoring

```rust
pub struct ProxyMetrics {
    // Request metrics
    pub requests_total: CounterVec,      // Labels: service, method, status
    pub request_latency: HistogramVec,   // Labels: service, method
    pub request_size: HistogramVec,
    pub response_size: HistogramVec,

    // Connection metrics
    pub active_connections: GaugeVec,    // Labels: direction (inbound/outbound)
    pub connection_errors: CounterVec,

    // mTLS metrics
    pub tls_handshake_latency: Histogram,
    pub cert_expiry_seconds: Gauge,
    pub auth_denied: Counter,

    // Circuit breaker
    pub circuit_open: CounterVec,        // Labels: service
    pub circuit_state: GaugeVec,         // Labels: service, state

    // Retry metrics
    pub retries_total: CounterVec,       // Labels: service
}

// Prometheus exposition
pub async fn metrics_handler() -> String {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    encoder.encode_to_string(&metric_families).unwrap()
}
```

---

## Implementation Phases

### Phase 1: Basic Proxy (Week 1-2)
- [ ] TCP proxy with connection handling
- [ ] HTTP/1.1 parsing
- [ ] Basic forwarding
- [ ] Connection pooling

### Phase 2: mTLS (Week 3-4)
- [ ] Certificate Authority implementation
- [ ] Certificate issuance
- [ ] TLS termination/origination
- [ ] Certificate rotation

### Phase 3: Service Discovery (Week 5)
- [ ] Service registry
- [ ] xDS protocol (subset)
- [ ] Load balancing algorithms
- [ ] Health checking

### Phase 4: Traffic Management (Week 6-7)
- [ ] Retry policies
- [ ] Timeouts
- [ ] Circuit breakers
- [ ] Rate limiting (stretch)

### Phase 5: Kubernetes Integration (Week 8-9)
- [ ] Mutating webhook
- [ ] Init container for iptables
- [ ] CRD for policies
- [ ] Controller for CRD sync

### Phase 6: Observability (Week 10)
- [ ] Prometheus metrics
- [ ] Distributed tracing
- [ ] Access logging
- [ ] Admin dashboard

---

## Testing Strategy

### Unit Tests
- Circuit breaker state transitions
- Load balancer selection
- Certificate validation
- Header injection

### Integration Tests
```rust
#[tokio::test]
async fn test_mtls_communication() {
    let mesh = TestMesh::new().await;

    let service_a = mesh.deploy_service("service-a").await;
    let service_b = mesh.deploy_service("service-b").await;

    // Service A calls Service B
    let response = service_a.call(service_b.address()).await.unwrap();

    // Verify mTLS was used
    assert!(response.headers().contains_key("x-mesh-tls"));
}

#[tokio::test]
async fn test_circuit_breaker() {
    let mesh = TestMesh::new().await;

    let client = mesh.deploy_service("client").await;
    let server = mesh.deploy_service("server").await;

    // Make server fail
    server.set_failure_rate(1.0);

    // Send requests until circuit opens
    for _ in 0..10 {
        let _ = client.call(server.address()).await;
    }

    // Circuit should be open
    let result = client.call(server.address()).await;
    assert!(matches!(result, Err(Error::CircuitOpen)));
}
```

---

## Stretch Goals

### Rate Limiting
- Token bucket per service
- Global rate limit service
- Header-based limits

### Multi-Cluster Mesh
- Cross-cluster service discovery
- Federated identity
- Traffic mirroring

### Advanced Traffic Routing
- Header-based routing
- Canary deployments
- Traffic mirroring/shadowing

---

## Dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
hyper = "0.14"
tonic = "0.9"                     # gRPC for xDS
openssl = "0.10"                  # TLS
kube = "0.85"                     # Kubernetes client
prometheus = "0.13"
tracing = "0.1"
```

---

## References

- [Envoy xDS Protocol](https://www.envoyproxy.io/docs/envoy/latest/api-docs/xds_protocol)
- [Linkerd Architecture](https://linkerd.io/2.14/reference/architecture/)
- [Istio Architecture](https://istio.io/latest/docs/ops/deployment/architecture/)
- [SPIFFE Specification](https://spiffe.io/docs/latest/spiffe-about/overview/)
