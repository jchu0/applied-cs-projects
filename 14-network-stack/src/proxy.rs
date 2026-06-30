//! Reverse proxy and load balancing implementation.
//!
//! This module provides enterprise features for building API gateways
//! and load balancers on top of the network stack.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::http::{Request, Response, Method, StatusCode, Headers};
use crate::pool::{ConnectionPool, PoolConfig, PooledConnection};

/// Load balancing algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalanceStrategy {
    /// Round-robin: cycle through backends in order
    RoundRobin,
    /// Least connections: pick backend with fewest active connections
    LeastConnections,
    /// Weighted round-robin: backends with higher weights get more traffic
    WeightedRoundRobin,
    /// Random: randomly select a backend
    Random,
    /// IP hash: consistent hashing based on client IP
    IpHash,
}

/// Backend server health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Backend is healthy and accepting requests
    Healthy,
    /// Backend is unhealthy and should not receive traffic
    Unhealthy,
    /// Backend health is unknown (not yet checked)
    Unknown,
}

/// Configuration for a backend server.
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Backend address (host:port)
    pub address: String,
    /// Weight for weighted load balancing (higher = more traffic)
    pub weight: u32,
    /// Maximum connections to this backend
    pub max_connections: usize,
    /// Health check path (e.g., "/health")
    pub health_check_path: String,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Number of failed checks before marking unhealthy
    pub unhealthy_threshold: u32,
    /// Number of successful checks before marking healthy
    pub healthy_threshold: u32,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1:8080".to_string(),
            weight: 1,
            max_connections: 100,
            health_check_path: "/health".to_string(),
            health_check_interval: Duration::from_secs(10),
            unhealthy_threshold: 3,
            healthy_threshold: 2,
        }
    }
}

/// Runtime state for a backend server.
#[derive(Debug)]
pub struct Backend {
    /// Backend configuration
    pub config: BackendConfig,
    /// Current health status
    pub status: HealthStatus,
    /// Active connection count
    pub active_connections: AtomicUsize,
    /// Total requests served
    pub total_requests: AtomicU64,
    /// Total errors
    pub total_errors: AtomicU64,
    /// Last health check time
    pub last_health_check: Option<Instant>,
    /// Consecutive health check failures
    pub consecutive_failures: u32,
    /// Consecutive health check successes
    pub consecutive_successes: u32,
    /// Average response time in milliseconds
    pub avg_response_time_ms: AtomicU64,
}

impl Backend {
    /// Create a new backend with configuration.
    pub fn new(config: BackendConfig) -> Self {
        Self {
            config,
            status: HealthStatus::Unknown,
            active_connections: AtomicUsize::new(0),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
            last_health_check: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
            avg_response_time_ms: AtomicU64::new(0),
        }
    }

    /// Check if backend can accept new connections.
    pub fn can_accept(&self) -> bool {
        self.status == HealthStatus::Healthy
            && self.active_connections.load(Ordering::Relaxed) < self.config.max_connections
    }

    /// Increment active connection count.
    pub fn acquire(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active connection count.
    pub fn release(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record a successful request.
    pub fn record_success(&self, response_time_ms: u64) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        // Simple exponential moving average
        let old = self.avg_response_time_ms.load(Ordering::Relaxed);
        let new = if old == 0 {
            response_time_ms
        } else {
            (old * 7 + response_time_ms) / 8
        };
        self.avg_response_time_ms.store(new, Ordering::Relaxed);
    }

    /// Record a failed request.
    pub fn record_error(&self) {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_errors.fetch_add(1, Ordering::Relaxed);
    }
}

/// Load balancer for selecting backends.
pub struct LoadBalancer {
    /// Available backends
    backends: Vec<Backend>,
    /// Load balancing strategy
    strategy: LoadBalanceStrategy,
    /// Round-robin counter
    rr_counter: AtomicUsize,
    /// Weighted round-robin state
    weighted_state: AtomicUsize,
}

impl LoadBalancer {
    /// Create a new load balancer.
    pub fn new(backends: Vec<BackendConfig>, strategy: LoadBalanceStrategy) -> Self {
        Self {
            backends: backends.into_iter().map(Backend::new).collect(),
            strategy,
            rr_counter: AtomicUsize::new(0),
            weighted_state: AtomicUsize::new(0),
        }
    }

    /// Get the number of backends.
    pub fn backend_count(&self) -> usize {
        self.backends.len()
    }

    /// Get healthy backends.
    pub fn healthy_backends(&self) -> Vec<&Backend> {
        self.backends
            .iter()
            .filter(|b| b.status == HealthStatus::Healthy)
            .collect()
    }

    /// Select a backend based on the load balancing strategy.
    pub fn select(&self, client_ip: Option<&str>) -> Option<&Backend> {
        let healthy: Vec<_> = self.backends
            .iter()
            .enumerate()
            .filter(|(_, b)| b.can_accept())
            .collect();

        if healthy.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalanceStrategy::RoundRobin => {
                let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed);
                healthy.get(idx % healthy.len()).map(|(_, b)| *b)
            }
            LoadBalanceStrategy::LeastConnections => {
                healthy.into_iter()
                    .min_by_key(|(_, b)| b.active_connections.load(Ordering::Relaxed))
                    .map(|(_, b)| b)
            }
            LoadBalanceStrategy::WeightedRoundRobin => {
                // Simple weighted selection
                let total_weight: u32 = healthy.iter().map(|(_, b)| b.config.weight).sum();
                if total_weight == 0 {
                    return healthy.first().map(|(_, b)| *b);
                }
                let idx = self.weighted_state.fetch_add(1, Ordering::Relaxed);
                let target = (idx as u32) % total_weight;
                let mut cumulative = 0;
                for (_, backend) in healthy {
                    cumulative += backend.config.weight;
                    if target < cumulative {
                        return Some(backend);
                    }
                }
                None
            }
            LoadBalanceStrategy::Random => {
                // Simple pseudo-random using time
                let idx = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .subsec_nanos() as usize;
                healthy.get(idx % healthy.len()).map(|(_, b)| *b)
            }
            LoadBalanceStrategy::IpHash => {
                if let Some(ip) = client_ip {
                    let hash = ip.bytes().fold(0u64, |acc, b| {
                        acc.wrapping_mul(31).wrapping_add(b as u64)
                    });
                    healthy.get(hash as usize % healthy.len()).map(|(_, b)| *b)
                } else {
                    healthy.first().map(|(_, b)| *b)
                }
            }
        }
    }

    /// Get backend by index (mutable).
    pub fn get_backend_mut(&mut self, index: usize) -> Option<&mut Backend> {
        self.backends.get_mut(index)
    }

    /// Get all backends.
    pub fn backends(&self) -> &[Backend] {
        &self.backends
    }

    /// Mark a backend as healthy.
    pub fn mark_healthy(&mut self, index: usize) {
        if let Some(backend) = self.backends.get_mut(index) {
            backend.consecutive_successes += 1;
            backend.consecutive_failures = 0;
            if backend.consecutive_successes >= backend.config.healthy_threshold {
                backend.status = HealthStatus::Healthy;
            }
            backend.last_health_check = Some(Instant::now());
        }
    }

    /// Mark a backend as unhealthy.
    pub fn mark_unhealthy(&mut self, index: usize) {
        if let Some(backend) = self.backends.get_mut(index) {
            backend.consecutive_failures += 1;
            backend.consecutive_successes = 0;
            if backend.consecutive_failures >= backend.config.unhealthy_threshold {
                backend.status = HealthStatus::Unhealthy;
            }
            backend.last_health_check = Some(Instant::now());
        }
    }
}

/// Access log entry.
#[derive(Debug, Clone)]
pub struct AccessLogEntry {
    /// Timestamp of the request
    pub timestamp: Instant,
    /// Client IP address
    pub client_ip: String,
    /// HTTP method
    pub method: Method,
    /// Request path
    pub path: String,
    /// HTTP status code
    pub status_code: u16,
    /// Response size in bytes
    pub response_size: usize,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Backend server used
    pub backend: String,
    /// User agent
    pub user_agent: Option<String>,
}

/// Access logger for recording requests.
pub struct AccessLogger {
    /// Log entries (in-memory buffer)
    entries: Vec<AccessLogEntry>,
    /// Maximum entries to keep
    max_entries: usize,
}

impl AccessLogger {
    /// Create a new access logger.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries.min(10000)),
            max_entries,
        }
    }

    /// Log a request.
    pub fn log(&mut self, entry: AccessLogEntry) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(entry);
    }

    /// Get recent entries.
    pub fn recent(&self, count: usize) -> &[AccessLogEntry] {
        let start = self.entries.len().saturating_sub(count);
        &self.entries[start..]
    }

    /// Get total entry count.
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// Proxy configuration.
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Load balancing strategy
    pub strategy: LoadBalanceStrategy,
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum request body size
    pub max_body_size: usize,
    /// Add X-Forwarded-For header
    pub add_forwarded_for: bool,
    /// Add X-Real-IP header
    pub add_real_ip: bool,
    /// Strip incoming hop-by-hop headers
    pub strip_hop_headers: bool,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalanceStrategy::RoundRobin,
            request_timeout: Duration::from_secs(30),
            max_body_size: 10 * 1024 * 1024, // 10MB
            add_forwarded_for: true,
            add_real_ip: true,
            strip_hop_headers: true,
        }
    }
}

/// Reverse proxy server.
pub struct ReverseProxy {
    /// Proxy configuration
    config: ProxyConfig,
    /// Load balancer
    load_balancer: LoadBalancer,
    /// Access logger
    logger: AccessLogger,
    /// Metrics
    total_requests: AtomicU64,
    total_errors: AtomicU64,
}

impl ReverseProxy {
    /// Create a new reverse proxy.
    pub fn new(
        config: ProxyConfig,
        backends: Vec<BackendConfig>,
    ) -> Self {
        Self {
            load_balancer: LoadBalancer::new(backends, config.strategy),
            config,
            logger: AccessLogger::new(10000),
            total_requests: AtomicU64::new(0),
            total_errors: AtomicU64::new(0),
        }
    }

    /// Get the load balancer.
    pub fn load_balancer(&self) -> &LoadBalancer {
        &self.load_balancer
    }

    /// Get the load balancer mutably.
    pub fn load_balancer_mut(&mut self) -> &mut LoadBalancer {
        &mut self.load_balancer
    }

    /// Get the access logger.
    pub fn logger(&self) -> &AccessLogger {
        &self.logger
    }

    /// Get total requests.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get total errors.
    pub fn total_errors(&self) -> u64 {
        self.total_errors.load(Ordering::Relaxed)
    }

    /// Process an incoming request (simulation).
    ///
    /// In a real implementation, this would:
    /// 1. Select a backend using the load balancer
    /// 2. Forward the request to the backend
    /// 3. Return the response to the client
    pub fn process_request(
        &mut self,
        request: &Request,
        client_ip: &str,
    ) -> Result<Response, ProxyError> {
        let start = Instant::now();
        self.total_requests.fetch_add(1, Ordering::Relaxed);

        // Select backend
        let backend = self.load_balancer.select(Some(client_ip))
            .ok_or(ProxyError::NoHealthyBackends)?;

        // In a real implementation, forward the request here
        // For now, simulate a successful response
        let response = Response::new(StatusCode::OK)
            .header("Content-Type", "text/plain")
            .header("X-Backend", &backend.config.address)
            .body(bytes::BytesMut::from(&b"OK"[..]));

        // Record metrics
        let elapsed = start.elapsed();
        backend.acquire();
        backend.record_success(elapsed.as_millis() as u64);
        backend.release();

        // Log access
        let entry = AccessLogEntry {
            timestamp: start,
            client_ip: client_ip.to_string(),
            method: request.method,
            path: request.uri.clone(),
            status_code: 200,
            response_size: 2,
            response_time_ms: elapsed.as_millis() as u64,
            backend: backend.config.address.clone(),
            user_agent: request.headers.get("user-agent").map(String::from),
        };
        self.logger.log(entry);

        Ok(response)
    }

    /// Handle a backend error.
    pub fn record_error(&self, backend_idx: usize) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
        if let Some(backend) = self.load_balancer.backends.get(backend_idx) {
            backend.record_error();
        }
    }

    /// Prepare request headers for proxying.
    pub fn prepare_headers(&self, headers: &Headers, client_ip: &str) -> Headers {
        let mut new_headers = Headers::new();

        // Copy non-hop-by-hop headers
        let hop_headers = [
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers",
            "transfer-encoding", "upgrade",
        ];

        for (key, value) in headers.iter() {
            let key_lower = key.to_lowercase();
            if !self.config.strip_hop_headers || !hop_headers.contains(&key_lower.as_str()) {
                new_headers.set(key, value);
            }
        }

        // Add forwarded headers
        if self.config.add_forwarded_for {
            if let Some(existing) = headers.get("x-forwarded-for") {
                new_headers.set("X-Forwarded-For", &format!("{}, {}", existing, client_ip));
            } else {
                new_headers.set("X-Forwarded-For", client_ip);
            }
        }

        if self.config.add_real_ip {
            new_headers.set("X-Real-IP", client_ip);
        }

        new_headers
    }
}

/// Proxy errors.
#[derive(Debug)]
pub enum ProxyError {
    /// No healthy backends available
    NoHealthyBackends,
    /// Backend connection failed
    BackendConnectionFailed(String),
    /// Request timeout
    Timeout,
    /// Request body too large
    BodyTooLarge,
    /// Invalid request
    InvalidRequest(String),
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProxyError::NoHealthyBackends => write!(f, "No healthy backends available"),
            ProxyError::BackendConnectionFailed(msg) => write!(f, "Backend connection failed: {}", msg),
            ProxyError::Timeout => write!(f, "Request timeout"),
            ProxyError::BodyTooLarge => write!(f, "Request body too large"),
            ProxyError::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
        }
    }
}

impl std::error::Error for ProxyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert_eq!(config.weight, 1);
        assert_eq!(config.max_connections, 100);
    }

    #[test]
    fn test_backend_new() {
        let config = BackendConfig {
            address: "localhost:8080".to_string(),
            ..Default::default()
        };
        let backend = Backend::new(config);
        assert_eq!(backend.status, HealthStatus::Unknown);
        assert_eq!(backend.active_connections.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_backend_acquire_release() {
        let backend = Backend::new(BackendConfig::default());
        assert_eq!(backend.active_connections.load(Ordering::Relaxed), 0);

        backend.acquire();
        assert_eq!(backend.active_connections.load(Ordering::Relaxed), 1);

        backend.acquire();
        assert_eq!(backend.active_connections.load(Ordering::Relaxed), 2);

        backend.release();
        assert_eq!(backend.active_connections.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_backend_record_success() {
        let backend = Backend::new(BackendConfig::default());
        backend.record_success(100);
        assert_eq!(backend.total_requests.load(Ordering::Relaxed), 1);
        assert_eq!(backend.avg_response_time_ms.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_backend_record_error() {
        let backend = Backend::new(BackendConfig::default());
        backend.record_error();
        assert_eq!(backend.total_requests.load(Ordering::Relaxed), 1);
        assert_eq!(backend.total_errors.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_load_balancer_new() {
        let backends = vec![
            BackendConfig { address: "backend1:80".to_string(), ..Default::default() },
            BackendConfig { address: "backend2:80".to_string(), ..Default::default() },
        ];
        let lb = LoadBalancer::new(backends, LoadBalanceStrategy::RoundRobin);
        assert_eq!(lb.backend_count(), 2);
    }

    #[test]
    fn test_load_balancer_round_robin() {
        let backends = vec![
            BackendConfig { address: "backend1:80".to_string(), ..Default::default() },
            BackendConfig { address: "backend2:80".to_string(), ..Default::default() },
        ];
        let mut lb = LoadBalancer::new(backends, LoadBalanceStrategy::RoundRobin);

        // Mark all as healthy
        lb.mark_healthy(0);
        lb.mark_healthy(0);
        lb.mark_healthy(1);
        lb.mark_healthy(1);

        let first = lb.select(None).map(|b| b.config.address.as_str());
        let second = lb.select(None).map(|b| b.config.address.as_str());

        // Round robin should alternate
        assert_ne!(first, second);
    }

    #[test]
    fn test_load_balancer_least_connections() {
        let backends = vec![
            BackendConfig { address: "backend1:80".to_string(), ..Default::default() },
            BackendConfig { address: "backend2:80".to_string(), ..Default::default() },
        ];
        let mut lb = LoadBalancer::new(backends, LoadBalanceStrategy::LeastConnections);

        lb.mark_healthy(0);
        lb.mark_healthy(0);
        lb.mark_healthy(1);
        lb.mark_healthy(1);

        // Add connections to first backend
        lb.backends[0].acquire();
        lb.backends[0].acquire();

        // Should select second backend (fewer connections)
        let selected = lb.select(None).unwrap();
        assert_eq!(selected.config.address, "backend2:80");
    }

    #[test]
    fn test_load_balancer_no_healthy() {
        let backends = vec![
            BackendConfig { address: "backend1:80".to_string(), ..Default::default() },
        ];
        let lb = LoadBalancer::new(backends, LoadBalanceStrategy::RoundRobin);

        // No backends marked healthy
        assert!(lb.select(None).is_none());
    }

    #[test]
    fn test_load_balancer_mark_healthy() {
        let backends = vec![
            BackendConfig {
                address: "backend1:80".to_string(),
                healthy_threshold: 2,
                ..Default::default()
            },
        ];
        let mut lb = LoadBalancer::new(backends, LoadBalanceStrategy::RoundRobin);

        assert_eq!(lb.backends[0].status, HealthStatus::Unknown);

        lb.mark_healthy(0);
        assert_eq!(lb.backends[0].status, HealthStatus::Unknown);

        lb.mark_healthy(0);
        assert_eq!(lb.backends[0].status, HealthStatus::Healthy);
    }

    #[test]
    fn test_load_balancer_mark_unhealthy() {
        let backends = vec![
            BackendConfig {
                address: "backend1:80".to_string(),
                unhealthy_threshold: 2,
                ..Default::default()
            },
        ];
        let mut lb = LoadBalancer::new(backends, LoadBalanceStrategy::RoundRobin);
        lb.backends[0].status = HealthStatus::Healthy;

        lb.mark_unhealthy(0);
        assert_eq!(lb.backends[0].status, HealthStatus::Healthy);

        lb.mark_unhealthy(0);
        assert_eq!(lb.backends[0].status, HealthStatus::Unhealthy);
    }

    #[test]
    fn test_access_logger_new() {
        let logger = AccessLogger::new(100);
        assert_eq!(logger.count(), 0);
    }

    #[test]
    fn test_access_logger_log() {
        let mut logger = AccessLogger::new(100);
        let entry = AccessLogEntry {
            timestamp: Instant::now(),
            client_ip: "192.168.1.1".to_string(),
            method: Method::Get,
            path: "/api/test".to_string(),
            status_code: 200,
            response_size: 1024,
            response_time_ms: 50,
            backend: "backend1:80".to_string(),
            user_agent: Some("test-agent".to_string()),
        };
        logger.log(entry);
        assert_eq!(logger.count(), 1);
    }

    #[test]
    fn test_access_logger_max_entries() {
        let mut logger = AccessLogger::new(2);

        for i in 0..5 {
            logger.log(AccessLogEntry {
                timestamp: Instant::now(),
                client_ip: format!("192.168.1.{}", i),
                method: Method::Get,
                path: "/".to_string(),
                status_code: 200,
                response_size: 0,
                response_time_ms: 0,
                backend: "backend".to_string(),
                user_agent: None,
            });
        }

        assert_eq!(logger.count(), 2);
    }

    #[test]
    fn test_proxy_config_default() {
        let config = ProxyConfig::default();
        assert_eq!(config.strategy, LoadBalanceStrategy::RoundRobin);
        assert!(config.add_forwarded_for);
    }

    #[test]
    fn test_reverse_proxy_new() {
        let backends = vec![
            BackendConfig { address: "backend1:80".to_string(), ..Default::default() },
        ];
        let proxy = ReverseProxy::new(ProxyConfig::default(), backends);
        assert_eq!(proxy.total_requests(), 0);
        assert_eq!(proxy.load_balancer().backend_count(), 1);
    }

    #[test]
    fn test_proxy_error_display() {
        let err = ProxyError::NoHealthyBackends;
        assert_eq!(err.to_string(), "No healthy backends available");

        let err = ProxyError::Timeout;
        assert_eq!(err.to_string(), "Request timeout");
    }

    #[test]
    fn test_ip_hash_consistency() {
        let backends = vec![
            BackendConfig { address: "backend1:80".to_string(), ..Default::default() },
            BackendConfig { address: "backend2:80".to_string(), ..Default::default() },
            BackendConfig { address: "backend3:80".to_string(), ..Default::default() },
        ];
        let mut lb = LoadBalancer::new(backends, LoadBalanceStrategy::IpHash);

        for i in 0..3 {
            lb.mark_healthy(i);
            lb.mark_healthy(i);
        }

        // Same IP should always select same backend
        let ip = "192.168.1.100";
        let first = lb.select(Some(ip)).map(|b| b.config.address.clone());
        let second = lb.select(Some(ip)).map(|b| b.config.address.clone());
        let third = lb.select(Some(ip)).map(|b| b.config.address.clone());

        assert_eq!(first, second);
        assert_eq!(second, third);
    }

    #[test]
    fn test_weighted_round_robin() {
        let backends = vec![
            BackendConfig { address: "heavy:80".to_string(), weight: 3, ..Default::default() },
            BackendConfig { address: "light:80".to_string(), weight: 1, ..Default::default() },
        ];
        let mut lb = LoadBalancer::new(backends, LoadBalanceStrategy::WeightedRoundRobin);

        lb.mark_healthy(0);
        lb.mark_healthy(0);
        lb.mark_healthy(1);
        lb.mark_healthy(1);

        // Over many selections, heavy should be chosen ~3x more than light
        let mut heavy_count = 0;
        let mut light_count = 0;

        for _ in 0..100 {
            if let Some(b) = lb.select(None) {
                if b.config.address == "heavy:80" {
                    heavy_count += 1;
                } else {
                    light_count += 1;
                }
            }
        }

        // Heavy should have significantly more selections
        assert!(heavy_count > light_count * 2);
    }
}
