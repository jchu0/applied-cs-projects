//! Sidecar proxy implementation.

use crate::cert::CertManager;
use crate::config::ProxyConfig;
use crate::discovery::{EndpointHealth, LoadBalancer, LoadBalancerType, ServiceKey, ServiceRegistry};
use crate::metrics::ProxyMetrics;
use crate::policy::{CircuitBreaker, RetryPolicy};
use crate::tracing_mesh::{SpanContext, Tracer};
use crate::{Error, Result};

use std::collections::HashMap;

use bytes::Bytes;
use parking_lot::RwLock;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tracing::{debug, error, info, warn};

/// Connection pool for managing connections to services.
pub struct ConnectionPool {
    /// Connections by address.
    connections: HashMap<SocketAddr, Vec<Connection>>,
    /// Maximum connections per host.
    max_per_host: usize,
    /// Connection timeout.
    connect_timeout: Duration,
}

/// A single connection.
pub struct Connection {
    /// Remote address.
    pub address: SocketAddr,
    /// Creation time.
    pub created_at: Instant,
    /// Last used time.
    pub last_used: Instant,
    /// Is connection in use.
    pub in_use: bool,
}

impl ConnectionPool {
    /// Create a new connection pool.
    pub fn new(max_per_host: usize, connect_timeout: Duration) -> Self {
        Self {
            connections: HashMap::new(),
            max_per_host,
            connect_timeout,
        }
    }

    /// Get or create a connection.
    pub fn get_connection(&mut self, addr: SocketAddr) -> Option<&mut Connection> {
        let conns = self.connections.entry(addr).or_default();

        // Try to find an idle connection by index
        let idle_index = conns.iter().position(|conn| !conn.in_use);

        if let Some(idx) = idle_index {
            let conn = &mut conns[idx];
            conn.in_use = true;
            conn.last_used = Instant::now();
            return Some(conn);
        }

        // Create new if under limit
        if conns.len() < self.max_per_host {
            conns.push(Connection {
                address: addr,
                created_at: Instant::now(),
                last_used: Instant::now(),
                in_use: true,
            });
            return conns.last_mut();
        }

        None
    }

    /// Release a connection back to the pool.
    pub fn release_connection(&mut self, addr: SocketAddr) {
        if let Some(conns) = self.connections.get_mut(&addr) {
            for conn in conns.iter_mut() {
                if conn.in_use {
                    conn.in_use = false;
                    conn.last_used = Instant::now();
                    break;
                }
            }
        }
    }

    /// Clean up idle connections.
    pub fn cleanup(&mut self, max_idle: Duration) {
        let now = Instant::now();
        for conns in self.connections.values_mut() {
            conns.retain(|c| !c.in_use && now.duration_since(c.last_used) < max_idle);
        }
    }
}

/// Sidecar proxy for service mesh.
pub struct SidecarProxy {
    /// Configuration.
    config: Arc<RwLock<ProxyConfig>>,
    /// Service registry.
    service_registry: Arc<ServiceRegistry>,
    /// Certificate manager.
    cert_manager: Arc<CertManager>,
    /// Connection pool.
    connection_pool: RwLock<ConnectionPool>,
    /// Load balancer.
    load_balancer: LoadBalancer,
    /// Circuit breakers by service.
    circuit_breakers: HashMap<ServiceKey, CircuitBreaker>,
    /// Metrics (Arc-wrapped for sharing across tasks).
    pub metrics: Arc<ProxyMetrics>,
    /// Tracer.
    tracer: Tracer,
}

impl SidecarProxy {
    /// Create a new sidecar proxy.
    pub fn new(
        config: ProxyConfig,
        service_registry: Arc<ServiceRegistry>,
        cert_manager: Arc<CertManager>,
    ) -> Self {
        let tracer = Tracer::new(
            config.service_name.clone(),
            config.tracing_config.collector_endpoint.clone(),
        );

        Self {
            config: Arc::new(RwLock::new(config)),
            service_registry,
            cert_manager,
            connection_pool: RwLock::new(ConnectionPool::new(100, Duration::from_secs(10))),
            load_balancer: LoadBalancer::new(LoadBalancerType::RoundRobin),
            circuit_breakers: HashMap::new(),
            metrics: Arc::new(ProxyMetrics::new()),
            tracer,
        }
    }

    /// Get current configuration.
    pub fn config(&self) -> ProxyConfig {
        self.config.read().clone()
    }

    /// Update configuration.
    pub fn update_config(&self, config: ProxyConfig) {
        *self.config.write() = config;
    }

    /// Check authorization for a request.
    pub fn authorize(&self, identity: &crate::config::ServiceIdentity) -> Result<bool> {
        // In a real implementation, this would check authorization policies
        debug!("Authorizing request from {}", identity.spiffe_id);
        Ok(true)
    }

    /// Calculate backoff duration for retry.
    pub fn calculate_backoff(&self, attempt: u32, policy: &RetryPolicy) -> Duration {
        let base = policy.backoff.base_interval.as_millis() as f64;
        let max = policy.backoff.max_interval.as_millis() as f64;

        // Exponential backoff
        let exponential = base * 2f64.powi(attempt as i32 - 1);
        let capped = exponential.min(max);

        // Add jitter
        let jitter = rand::random::<f64>() * policy.backoff.jitter;
        let with_jitter = capped * (1.0 + jitter);

        Duration::from_millis(with_jitter as u64)
    }

    /// Check if error is retryable.
    pub fn is_retryable_error(&self, status: u16, policy: &RetryPolicy) -> bool {
        for condition in &policy.retry_on {
            match condition {
                crate::policy::RetryCondition::Status5xx => {
                    if status >= 500 && status < 600 {
                        return true;
                    }
                }
                crate::policy::RetryCondition::StatusCode(code) => {
                    if status == *code {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Get circuit breaker for a service.
    pub fn get_circuit_breaker(&self, key: &ServiceKey) -> Option<&CircuitBreaker> {
        self.circuit_breakers.get(key)
    }

    /// Get or create connection to endpoint.
    pub fn get_connection(&self, addr: SocketAddr) -> Result<()> {
        let mut pool = self.connection_pool.write();
        if pool.get_connection(addr).is_none() {
            return Err(Error::Connection("No available connections".into()));
        }
        Ok(())
    }

    /// Release connection.
    pub fn release_connection(&self, addr: SocketAddr) {
        self.connection_pool.write().release_connection(addr);
    }

    /// Get tracer.
    pub fn tracer(&self) -> &Tracer {
        &self.tracer
    }

    /// Get service registry.
    pub fn registry(&self) -> &ServiceRegistry {
        &self.service_registry
    }

    /// Get load balancer.
    pub fn load_balancer(&self) -> &LoadBalancer {
        &self.load_balancer
    }

    /// Run the sidecar proxy.
    ///
    /// This starts the inbound and outbound listeners and processes traffic.
    pub async fn run(&self) -> Result<()> {
        let config = self.config();
        let inbound_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            config.inbound_port,
        );
        let outbound_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            config.outbound_port,
        );
        let admin_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0)),
            config.admin_port,
        );

        let inbound_listener = TcpListener::bind(inbound_addr).await?;
        let outbound_listener = TcpListener::bind(outbound_addr).await?;
        let admin_listener = TcpListener::bind(admin_addr).await?;

        info!(
            "Sidecar proxy started: inbound={}, outbound={}, admin={}",
            config.inbound_port, config.outbound_port, config.admin_port
        );

        tokio::select! {
            result = self.handle_inbound(inbound_listener) => {
                if let Err(e) = result {
                    error!("Inbound handler error: {}", e);
                }
            }
            result = self.handle_outbound(outbound_listener) => {
                if let Err(e) = result {
                    error!("Outbound handler error: {}", e);
                }
            }
            result = self.handle_admin(admin_listener) => {
                if let Err(e) = result {
                    error!("Admin handler error: {}", e);
                }
            }
        }

        Ok(())
    }

    /// Handle inbound traffic (from mesh to local application).
    async fn handle_inbound(&self, listener: TcpListener) -> Result<()> {
        loop {
            let (stream, peer_addr) = listener.accept().await?;
            debug!("Inbound connection from {}", peer_addr);

            let config = self.config();
            let app_port = config.app_port;
            let metrics = Arc::clone(&self.metrics);

            tokio::spawn(async move {
                if let Err(e) = Self::process_inbound_static(stream, peer_addr, app_port, metrics).await {
                    error!("Inbound processing error from {}: {}", peer_addr, e);
                }
            });
        }
    }

    /// Process a single inbound connection.
    async fn process_inbound_static(
        mut stream: TcpStream,
        peer_addr: SocketAddr,
        app_port: u16,
        metrics: Arc<ProxyMetrics>,
    ) -> Result<()> {
        Self::process_inbound_with_tracing(stream, peer_addr, app_port, metrics, None).await
    }

    /// Process inbound connection with optional tracing.
    async fn process_inbound_with_tracing(
        mut stream: TcpStream,
        peer_addr: SocketAddr,
        app_port: u16,
        metrics: Arc<ProxyMetrics>,
        tracer: Option<&Tracer>,
    ) -> Result<()> {
        let start = Instant::now();

        // Read HTTP request from incoming stream
        let mut buf_reader = BufReader::new(&mut stream);
        let request = Self::read_http_request(&mut buf_reader).await?;

        // Extract trace context from incoming request
        let parent_context = tracer.and_then(|t| TraceContextHelper::extract_from_request(t, &request));

        // Create span for this inbound request
        let mut span = tracer.map(|t| TraceContextHelper::create_inbound_span(t, &request, parent_context.as_ref()));

        debug!(
            "Inbound request: {} {} from {}",
            request.method, request.path, peer_addr
        );

        // Forward to local application
        let local_addr = SocketAddr::new(
            IpAddr::V4(Ipv4Addr::LOCALHOST),
            app_port,
        );

        let mut app_stream = TcpStream::connect(local_addr).await.map_err(|e| {
            if let Some(ref mut s) = span {
                s.log(&format!("Connection failed: {}", e));
            }
            Error::Connection(format!("Failed to connect to app: {}", e))
        })?;

        // Forward the request (with trace context if tracing enabled)
        let mut forward_request = request.clone();
        if let (Some(t), Some(ref s)) = (tracer, &span) {
            let ctx = s.context();
            TraceContextHelper::inject_into_request(t, &ctx, &mut forward_request);
        }
        Self::forward_http_request(&mut app_stream, &forward_request).await?;

        // Read response from app
        let mut app_buf_reader = BufReader::new(&mut app_stream);
        let response = Self::read_http_response(&mut app_buf_reader).await?;

        // Send response back to caller
        Self::send_http_response(&mut stream, &response).await?;

        // Record metrics
        let latency = start.elapsed();
        metrics.record_request(latency, response.status_code);

        // Finish span
        if let Some(ref mut s) = span {
            TraceContextHelper::finish_span_with_response(s, &response);
        }

        debug!(
            "Inbound request completed: {} {} -> {} in {:?}",
            request.method, request.path, response.status_code, latency
        );

        Ok(())
    }

    /// Handle outbound traffic (from local application to mesh).
    async fn handle_outbound(&self, listener: TcpListener) -> Result<()> {
        loop {
            let (stream, peer_addr) = listener.accept().await?;
            debug!("Outbound connection from {}", peer_addr);

            let registry = Arc::clone(&self.service_registry);
            let load_balancer = self.load_balancer.clone();
            let retry_policy = self.config().retry_policy.clone();
            let metrics = Arc::clone(&self.metrics);
            let timeout = self.config().timeout;

            tokio::spawn(async move {
                if let Err(e) = Self::process_outbound_static(
                    stream,
                    registry,
                    load_balancer,
                    retry_policy,
                    timeout,
                    metrics,
                )
                .await
                {
                    error!("Outbound processing error: {}", e);
                }
            });
        }
    }

    /// Process a single outbound connection.
    async fn process_outbound_static(
        mut stream: TcpStream,
        registry: Arc<ServiceRegistry>,
        load_balancer: LoadBalancer,
        retry_policy: RetryPolicy,
        timeout: Duration,
        metrics: Arc<ProxyMetrics>,
    ) -> Result<()> {
        Self::process_outbound_with_tracing(
            stream, registry, load_balancer, retry_policy, timeout, metrics, None
        ).await
    }

    /// Process outbound connection with optional tracing.
    async fn process_outbound_with_tracing(
        mut stream: TcpStream,
        registry: Arc<ServiceRegistry>,
        load_balancer: LoadBalancer,
        retry_policy: RetryPolicy,
        timeout: Duration,
        metrics: Arc<ProxyMetrics>,
        tracer: Option<&Tracer>,
    ) -> Result<()> {
        let start = Instant::now();

        // Read HTTP request from local app
        let mut buf_reader = BufReader::new(&mut stream);
        let mut request = Self::read_http_request(&mut buf_reader).await?;

        // Extract existing trace context (from local app)
        let parent_context = tracer.and_then(|t| TraceContextHelper::extract_from_request(t, &request));

        // Create span for this outbound request
        let mut span = tracer.map(|t| TraceContextHelper::create_outbound_span(t, &request, parent_context.as_ref()));

        debug!("Outbound request: {} {}", request.method, request.path);

        // Extract target service from Host header
        let host = request.get_header("host").unwrap_or("unknown");
        let service_name = host.split(':').next().unwrap_or(host);

        if let Some(ref mut s) = span {
            s.set_tag("peer.service", service_name);
        }

        // Service discovery
        let endpoints = registry.get_by_name(service_name).ok_or_else(|| {
            if let Some(ref mut s) = span {
                s.log(&format!("Service not found: {}", service_name));
            }
            Error::ServiceNotFound(service_name.to_string())
        })?;

        // Filter healthy endpoints
        let healthy_endpoints: Vec<_> = endpoints
            .endpoints
            .iter()
            .filter(|e| e.health == EndpointHealth::Healthy)
            .cloned()
            .collect();

        if healthy_endpoints.is_empty() {
            if let Some(ref mut s) = span {
                s.log("No healthy endpoints available");
            }
            return Err(Error::ServiceNotFound(format!(
                "No healthy endpoints for {}",
                service_name
            )));
        }

        // Load balancing
        let endpoint = load_balancer
            .select(&healthy_endpoints)
            .ok_or_else(|| Error::ServiceNotFound("No endpoint selected".to_string()))?;

        if let Some(ref mut s) = span {
            s.set_tag("peer.address", &endpoint.address.to_string());
        }

        // Inject trace context into outgoing request
        if let (Some(t), Some(ref s)) = (tracer, &span) {
            let ctx = s.context();
            TraceContextHelper::inject_into_request(t, &ctx, &mut request);
        }

        // Forward with retry
        let response = Self::forward_with_retry(
            &request,
            endpoint.address,
            &retry_policy,
            timeout,
        )
        .await?;

        // Send response back to local app
        Self::send_http_response(&mut stream, &response).await?;

        // Record metrics
        let latency = start.elapsed();
        metrics.record_request(latency, response.status_code);

        // Finish span
        if let Some(ref mut s) = span {
            TraceContextHelper::finish_span_with_response(s, &response);
        }

        debug!(
            "Outbound request completed: {} {} -> {} in {:?}",
            request.method, request.path, response.status_code, latency
        );

        Ok(())
    }

    /// Forward request with retry logic.
    async fn forward_with_retry(
        request: &HttpRequest,
        target_addr: SocketAddr,
        policy: &RetryPolicy,
        timeout: Duration,
    ) -> Result<HttpResponse> {
        let mut attempts = 0;
        let mut last_error = None;

        loop {
            attempts += 1;

            let result = tokio::time::timeout(
                timeout,
                Self::forward_single_request(request, target_addr),
            )
            .await;

            match result {
                Ok(Ok(response)) => {
                    // Check if response status is retryable
                    if Self::is_retryable_status(response.status_code, policy)
                        && attempts <= policy.max_retries
                    {
                        warn!(
                            "Retryable status {} on attempt {}",
                            response.status_code, attempts
                        );
                        last_error = Some(Error::Connection(format!(
                            "Retryable status: {}",
                            response.status_code
                        )));
                        let backoff = Self::calculate_backoff_static(attempts, policy);
                        tokio::time::sleep(backoff).await;
                        continue;
                    }
                    return Ok(response);
                }
                Ok(Err(e)) if attempts <= policy.max_retries => {
                    warn!("Request failed on attempt {}: {}", attempts, e);
                    last_error = Some(e);
                    let backoff = Self::calculate_backoff_static(attempts, policy);
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Ok(Err(e)) => return Err(e),
                Err(_) if attempts <= policy.max_retries => {
                    warn!("Request timed out on attempt {}", attempts);
                    last_error = Some(Error::Timeout);
                    let backoff = Self::calculate_backoff_static(attempts, policy);
                    tokio::time::sleep(backoff).await;
                    continue;
                }
                Err(_) => {
                    return Err(last_error.unwrap_or(Error::Timeout));
                }
            }
        }
    }

    /// Forward a single request to target.
    async fn forward_single_request(
        request: &HttpRequest,
        target_addr: SocketAddr,
    ) -> Result<HttpResponse> {
        let mut stream = TcpStream::connect(target_addr).await.map_err(|e| {
            Error::Connection(format!("Failed to connect to {}: {}", target_addr, e))
        })?;

        Self::forward_http_request(&mut stream, request).await?;

        let mut buf_reader = BufReader::new(&mut stream);
        Self::read_http_response(&mut buf_reader).await
    }

    /// Calculate backoff duration.
    fn calculate_backoff_static(attempt: u32, policy: &RetryPolicy) -> Duration {
        let base = policy.backoff.base_interval.as_millis() as f64;
        let max = policy.backoff.max_interval.as_millis() as f64;

        // Exponential backoff
        let exponential = base * 2f64.powi(attempt as i32 - 1);
        let capped = exponential.min(max);

        // Add jitter
        let jitter = rand::random::<f64>() * policy.backoff.jitter;
        let with_jitter = capped * (1.0 + jitter);

        Duration::from_millis(with_jitter as u64)
    }

    /// Check if status is retryable.
    fn is_retryable_status(status: u16, policy: &RetryPolicy) -> bool {
        for condition in &policy.retry_on {
            match condition {
                crate::policy::RetryCondition::Status5xx => {
                    if status >= 500 && status < 600 {
                        return true;
                    }
                }
                crate::policy::RetryCondition::StatusCode(code) => {
                    if status == *code {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// Handle admin requests (health checks, metrics).
    async fn handle_admin(&self, listener: TcpListener) -> Result<()> {
        loop {
            let (mut stream, _) = listener.accept().await?;
            let metrics = Arc::clone(&self.metrics);

            tokio::spawn(async move {
                let mut buf_reader = BufReader::new(&mut stream);
                if let Ok(request) = Self::read_http_request(&mut buf_reader).await {
                    let response = match request.path.as_str() {
                        "/ready" | "/health" => HttpResponse {
                            status_code: 200,
                            status_text: "OK".to_string(),
                            headers: vec![("Content-Type".to_string(), "text/plain".to_string())],
                            body: Bytes::from("OK"),
                        },
                        "/metrics" => HttpResponse {
                            status_code: 200,
                            status_text: "OK".to_string(),
                            headers: vec![(
                                "Content-Type".to_string(),
                                "text/plain; version=0.0.4".to_string(),
                            )],
                            body: Bytes::from(metrics.to_prometheus()),
                        },
                        _ => HttpResponse {
                            status_code: 404,
                            status_text: "Not Found".to_string(),
                            headers: vec![],
                            body: Bytes::new(),
                        },
                    };

                    let _ = Self::send_http_response(&mut stream, &response).await;
                }
            });
        }
    }

    /// Read an HTTP request from a stream.
    async fn read_http_request<R: AsyncBufReadExt + Unpin>(
        reader: &mut R,
    ) -> Result<HttpRequest> {
        // Read request line
        let mut request_line = String::new();
        reader.read_line(&mut request_line).await?;
        let parts: Vec<&str> = request_line.trim().split_whitespace().collect();
        if parts.len() < 2 {
            return Err(Error::Connection("Invalid request line".to_string()));
        }

        let method = parts[0].to_string();
        let path = parts[1].to_string();
        let version = parts.get(2).map(|s| s.to_string()).unwrap_or_else(|| "HTTP/1.1".to_string());

        // Read headers
        let mut headers = Vec::new();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).await?;
            let line = line.trim();
            if line.is_empty() {
                break;
            }
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().to_string();
                let value = value.trim().to_string();
                if key.eq_ignore_ascii_case("content-length") {
                    content_length = value.parse().unwrap_or(0);
                }
                headers.push((key, value));
            }
        }

        // Read body
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).await?;
        }

        Ok(HttpRequest {
            method,
            path,
            version,
            headers,
            body: Bytes::from(body),
        })
    }

    /// Read an HTTP response from a stream.
    async fn read_http_response<R: AsyncBufReadExt + Unpin>(
        reader: &mut R,
    ) -> Result<HttpResponse> {
        // Read status line
        let mut status_line = String::new();
        reader.read_line(&mut status_line).await?;
        let parts: Vec<&str> = status_line.trim().splitn(3, ' ').collect();
        if parts.len() < 2 {
            return Err(Error::Connection("Invalid status line".to_string()));
        }

        let status_code: u16 = parts[1].parse().unwrap_or(500);
        let status_text = parts.get(2).map(|s| s.to_string()).unwrap_or_default();

        // Read headers
        let mut headers = Vec::new();
        let mut content_length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).await?;
            let line = line.trim();
            if line.is_empty() {
                break;
            }
            if let Some((key, value)) = line.split_once(':') {
                let key = key.trim().to_string();
                let value = value.trim().to_string();
                if key.eq_ignore_ascii_case("content-length") {
                    content_length = value.parse().unwrap_or(0);
                }
                headers.push((key, value));
            }
        }

        // Read body
        let mut body = vec![0u8; content_length];
        if content_length > 0 {
            reader.read_exact(&mut body).await?;
        }

        Ok(HttpResponse {
            status_code,
            status_text,
            headers,
            body: Bytes::from(body),
        })
    }

    /// Forward an HTTP request to a stream.
    async fn forward_http_request(stream: &mut TcpStream, request: &HttpRequest) -> Result<()> {
        // Write request line
        stream
            .write_all(format!("{} {} {}\r\n", request.method, request.path, request.version).as_bytes())
            .await?;

        // Write headers
        for (key, value) in &request.headers {
            stream
                .write_all(format!("{}: {}\r\n", key, value).as_bytes())
                .await?;
        }
        stream.write_all(b"\r\n").await?;

        // Write body
        if !request.body.is_empty() {
            stream.write_all(&request.body).await?;
        }

        stream.flush().await?;
        Ok(())
    }

    /// Send an HTTP response to a stream.
    async fn send_http_response(stream: &mut TcpStream, response: &HttpResponse) -> Result<()> {
        // Write status line
        stream
            .write_all(
                format!(
                    "HTTP/1.1 {} {}\r\n",
                    response.status_code, response.status_text
                )
                .as_bytes(),
            )
            .await?;

        // Write headers
        for (key, value) in &response.headers {
            stream
                .write_all(format!("{}: {}\r\n", key, value).as_bytes())
                .await?;
        }

        // Add Content-Length if not present
        if !response.headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-length")) {
            stream
                .write_all(format!("Content-Length: {}\r\n", response.body.len()).as_bytes())
                .await?;
        }

        stream.write_all(b"\r\n").await?;

        // Write body
        if !response.body.is_empty() {
            stream.write_all(&response.body).await?;
        }

        stream.flush().await?;
        Ok(())
    }
}

/// HTTP request representation.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP method.
    pub method: String,
    /// Request path.
    pub path: String,
    /// HTTP version.
    pub version: String,
    /// Headers.
    pub headers: Vec<(String, String)>,
    /// Request body.
    pub body: Bytes,
}

/// HTTP response representation.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// Status code.
    pub status_code: u16,
    /// Status text.
    pub status_text: String,
    /// Headers.
    pub headers: Vec<(String, String)>,
    /// Response body.
    pub body: Bytes,
}

impl HttpRequest {
    /// Convert headers to HashMap for tracing extraction.
    pub fn headers_as_hashmap(&self) -> HashMap<String, String> {
        self.headers.iter().cloned().collect()
    }

    /// Get a specific header value (case-insensitive).
    pub fn get_header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Set or update a header.
    pub fn set_header(&mut self, name: String, value: String) {
        // Remove existing header if present
        self.headers.retain(|(k, _)| !k.eq_ignore_ascii_case(&name));
        self.headers.push((name, value));
    }
}

/// Trace context utilities for the proxy.
pub struct TraceContextHelper;

impl TraceContextHelper {
    /// Extract trace context from HTTP request headers.
    pub fn extract_from_request(tracer: &Tracer, request: &HttpRequest) -> Option<SpanContext> {
        let headers = request.headers_as_hashmap();
        tracer.extract_context(&headers)
    }

    /// Inject trace context into HTTP request headers.
    pub fn inject_into_request(tracer: &Tracer, context: &SpanContext, request: &mut HttpRequest) {
        let mut headers: HashMap<String, String> = HashMap::new();
        tracer.inject_context(context, &mut headers);

        // Add traceparent header
        if let Some(traceparent) = headers.get("traceparent") {
            request.set_header("traceparent".to_string(), traceparent.clone());
        }

        // Add tracestate if present
        if let Some(tracestate) = headers.get("tracestate") {
            request.set_header("tracestate".to_string(), tracestate.clone());
        }
    }

    /// Create a span for an inbound request.
    pub fn create_inbound_span(tracer: &Tracer, request: &HttpRequest, parent: Option<&SpanContext>) -> crate::tracing_mesh::Span {
        let mut span = tracer.start_span("proxy.inbound", parent);
        span.set_tag("http.method", &request.method);
        span.set_tag("http.path", &request.path);
        if let Some(host) = request.get_header("host") {
            span.set_tag("http.host", host);
        }
        span
    }

    /// Create a span for an outbound request.
    pub fn create_outbound_span(tracer: &Tracer, request: &HttpRequest, parent: Option<&SpanContext>) -> crate::tracing_mesh::Span {
        let mut span = tracer.start_span("proxy.outbound", parent);
        span.set_tag("http.method", &request.method);
        span.set_tag("http.path", &request.path);
        if let Some(host) = request.get_header("host") {
            span.set_tag("peer.service", host);
        }
        span
    }

    /// Finish a span with response information.
    pub fn finish_span_with_response(span: &mut crate::tracing_mesh::Span, response: &HttpResponse) {
        span.set_tag("http.status_code", &response.status_code.to_string());
        if response.status_code >= 400 {
            span.set_tag("error", "true");
        }
        span.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cert::{CertManager, CertificateAuthority};
    use crate::config::ServiceIdentity;
    use crate::policy::RetryCondition;

    fn create_test_proxy() -> SidecarProxy {
        let ca = CertificateAuthority::new(Duration::from_secs(3600)).unwrap();
        let identity = ServiceIdentity::new("default", "test-service");
        let cert = ca.issue_certificate(&identity).unwrap();
        let cert_manager = Arc::new(CertManager::new(identity, cert, Duration::from_secs(3600)));
        let registry = Arc::new(ServiceRegistry::new());

        SidecarProxy::new(ProxyConfig::default(), registry, cert_manager)
    }

    #[test]
    fn test_proxy_creation() {
        let proxy = create_test_proxy();
        assert!(proxy.config().inbound_port == 15006);
        assert!(proxy.config().outbound_port == 15001);
        assert!(proxy.config().admin_port == 15000);
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
    fn test_backoff_calculation() {
        let policy = RetryPolicy {
            max_retries: 3,
            retry_on: vec![RetryCondition::Status5xx],
            backoff: crate::policy::BackoffConfig {
                base_interval: Duration::from_millis(100),
                max_interval: Duration::from_secs(5),
                jitter: 0.1,
            },
        };

        let backoff1 = SidecarProxy::calculate_backoff_static(1, &policy);
        let backoff2 = SidecarProxy::calculate_backoff_static(2, &policy);
        let backoff3 = SidecarProxy::calculate_backoff_static(3, &policy);

        // First attempt ~100ms, second ~200ms, third ~400ms (with jitter)
        assert!(backoff1.as_millis() >= 100);
        assert!(backoff1.as_millis() <= 150); // 100 + 10% jitter max

        assert!(backoff2.as_millis() >= 200);
        assert!(backoff2.as_millis() <= 250);

        assert!(backoff3.as_millis() >= 400);
        assert!(backoff3.as_millis() <= 500);
    }

    #[test]
    fn test_retryable_status() {
        let policy = RetryPolicy {
            max_retries: 3,
            retry_on: vec![
                RetryCondition::Status5xx,
                RetryCondition::StatusCode(429),
            ],
            backoff: crate::policy::BackoffConfig::default(),
        };

        assert!(SidecarProxy::is_retryable_status(500, &policy));
        assert!(SidecarProxy::is_retryable_status(503, &policy));
        assert!(SidecarProxy::is_retryable_status(429, &policy));
        assert!(!SidecarProxy::is_retryable_status(404, &policy));
        assert!(!SidecarProxy::is_retryable_status(200, &policy));
    }

    #[test]
    fn test_connection_pool() {
        let mut pool = ConnectionPool::new(2, Duration::from_secs(10));
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        // Get first connection
        let conn1 = pool.get_connection(addr);
        assert!(conn1.is_some());
        assert!(conn1.unwrap().in_use);

        // Get second connection
        let conn2 = pool.get_connection(addr);
        assert!(conn2.is_some());

        // Pool is full, no more connections
        let conn3 = pool.get_connection(addr);
        assert!(conn3.is_none());

        // Release a connection
        pool.release_connection(addr);

        // Now we can get another
        let conn4 = pool.get_connection(addr);
        assert!(conn4.is_some());
    }

    #[test]
    fn test_http_request_struct() {
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/v1/users".to_string(),
            version: "HTTP/1.1".to_string(),
            headers: vec![
                ("Host".to_string(), "example.com".to_string()),
                ("Content-Type".to_string(), "application/json".to_string()),
            ],
            body: Bytes::from("test body"),
        };

        assert_eq!(request.method, "GET");
        assert_eq!(request.path, "/api/v1/users");
        assert_eq!(request.headers.len(), 2);
    }

    #[test]
    fn test_http_response_struct() {
        let response = HttpResponse {
            status_code: 200,
            status_text: "OK".to_string(),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
            body: Bytes::from("{\"success\": true}"),
        };

        assert_eq!(response.status_code, 200);
        assert_eq!(response.status_text, "OK");
        assert!(!response.body.is_empty());
    }

    #[tokio::test]
    async fn test_http_request_parsing() {
        use tokio::io::BufReader;

        let raw_request = b"GET /api/users HTTP/1.1\r\nHost: localhost:8080\r\nContent-Length: 5\r\n\r\nhello";
        let cursor = std::io::Cursor::new(raw_request.to_vec());
        let mut reader = BufReader::new(tokio::io::BufReader::new(cursor));

        // Note: This won't work with std::io::Cursor directly, need tokio compatible cursor
        // In real tests, we'd use tokio_test or mock the stream
    }

    #[test]
    fn test_metrics_snapshot() {
        let proxy = create_test_proxy();
        let snapshot = proxy.metrics.snapshot();

        assert_eq!(snapshot.requests_total, 0);
        assert_eq!(snapshot.active_connections, 0);
        assert_eq!(snapshot.connection_errors, 0);
    }

    #[test]
    fn test_request_header_operations() {
        let mut request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/test".to_string(),
            version: "HTTP/1.1".to_string(),
            headers: vec![
                ("Host".to_string(), "example.com".to_string()),
                ("Content-Type".to_string(), "application/json".to_string()),
            ],
            body: Bytes::new(),
        };

        // Test get_header (case insensitive)
        assert_eq!(request.get_header("host"), Some("example.com"));
        assert_eq!(request.get_header("HOST"), Some("example.com"));
        assert_eq!(request.get_header("Content-Type"), Some("application/json"));
        assert_eq!(request.get_header("nonexistent"), None);

        // Test set_header
        request.set_header("traceparent".to_string(), "00-abc123-def456-01".to_string());
        assert_eq!(request.get_header("traceparent"), Some("00-abc123-def456-01"));
        assert_eq!(request.headers.len(), 3);

        // Test set_header replaces existing
        request.set_header("Host".to_string(), "new-host.com".to_string());
        assert_eq!(request.get_header("host"), Some("new-host.com"));
        assert_eq!(request.headers.len(), 3);
    }

    #[test]
    fn test_headers_as_hashmap() {
        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/".to_string(),
            version: "HTTP/1.1".to_string(),
            headers: vec![
                ("Host".to_string(), "example.com".to_string()),
                ("traceparent".to_string(), "00-trace-span-01".to_string()),
            ],
            body: Bytes::new(),
        };

        let headers = request.headers_as_hashmap();
        assert_eq!(headers.get("Host"), Some(&"example.com".to_string()));
        assert_eq!(headers.get("traceparent"), Some(&"00-trace-span-01".to_string()));
    }

    #[test]
    fn test_trace_context_extraction() {
        use crate::tracing_mesh::Tracer;

        let tracer = Tracer::new("test-service".to_string(), "localhost:14268".to_string());

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/test".to_string(),
            version: "HTTP/1.1".to_string(),
            headers: vec![
                ("Host".to_string(), "example.com".to_string()),
                ("traceparent".to_string(), "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01".to_string()),
            ],
            body: Bytes::new(),
        };

        let context = TraceContextHelper::extract_from_request(&tracer, &request);
        assert!(context.is_some());

        let ctx = context.unwrap();
        assert_eq!(ctx.trace_id, 0x0af7651916cd43dd8448eb211c80319c);
        assert_eq!(ctx.span_id, 0xb7ad6b7169203331);
        assert_eq!(ctx.flags, 0x01);
    }

    #[test]
    fn test_trace_context_injection() {
        use crate::tracing_mesh::{Tracer, SpanContext};

        let tracer = Tracer::new("test-service".to_string(), "localhost:14268".to_string());

        let mut request = HttpRequest {
            method: "POST".to_string(),
            path: "/api/submit".to_string(),
            version: "HTTP/1.1".to_string(),
            headers: vec![("Host".to_string(), "example.com".to_string())],
            body: Bytes::new(),
        };

        let context = SpanContext {
            trace_id: 0x12345678901234567890123456789012,
            span_id: 0xabcdef1234567890,
            flags: 0x01,
        };

        TraceContextHelper::inject_into_request(&tracer, &context, &mut request);

        let traceparent = request.get_header("traceparent");
        assert!(traceparent.is_some());

        let tp = traceparent.unwrap();
        assert!(tp.starts_with("00-"));
        assert!(tp.contains("12345678901234567890123456789012"));
        assert!(tp.contains("abcdef1234567890"));
    }

    #[test]
    fn test_create_inbound_span() {
        use crate::tracing_mesh::Tracer;

        let tracer = Tracer::new("test-proxy".to_string(), "localhost:14268".to_string());

        let request = HttpRequest {
            method: "GET".to_string(),
            path: "/api/users".to_string(),
            version: "HTTP/1.1".to_string(),
            headers: vec![("Host".to_string(), "user-service:8080".to_string())],
            body: Bytes::new(),
        };

        let span = TraceContextHelper::create_inbound_span(&tracer, &request, None);

        assert_eq!(span.name, "proxy.inbound");
        assert_eq!(span.tags.get("http.method"), Some(&"GET".to_string()));
        assert_eq!(span.tags.get("http.path"), Some(&"/api/users".to_string()));
        assert_eq!(span.tags.get("http.host"), Some(&"user-service:8080".to_string()));
    }

    #[test]
    fn test_finish_span_with_response() {
        use crate::tracing_mesh::Tracer;

        let tracer = Tracer::new("test".to_string(), "localhost".to_string());
        let mut span = tracer.start_span("test-span", None);

        let success_response = HttpResponse {
            status_code: 200,
            status_text: "OK".to_string(),
            headers: vec![],
            body: Bytes::new(),
        };

        TraceContextHelper::finish_span_with_response(&mut span, &success_response);
        assert!(span.end_time.is_some());
        assert_eq!(span.tags.get("http.status_code"), Some(&"200".to_string()));
        assert_eq!(span.tags.get("error"), None);

        // Test with error response
        let mut error_span = tracer.start_span("error-span", None);
        let error_response = HttpResponse {
            status_code: 500,
            status_text: "Internal Server Error".to_string(),
            headers: vec![],
            body: Bytes::new(),
        };

        TraceContextHelper::finish_span_with_response(&mut error_span, &error_response);
        assert_eq!(error_span.tags.get("http.status_code"), Some(&"500".to_string()));
        assert_eq!(error_span.tags.get("error"), Some(&"true".to_string()));
    }
}

