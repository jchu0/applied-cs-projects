//! Metrics collection for the proxy.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Counter metric.
#[derive(Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Increment the counter.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Add to the counter.
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// Gauge metric.
#[derive(Default)]
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    /// Set the gauge value.
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Get current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Increment.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement.
    pub fn dec(&self) {
        self.value.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Histogram for latency tracking.
pub struct Histogram {
    /// Bucket counts.
    buckets: Vec<AtomicU64>,
    /// Bucket boundaries (in microseconds).
    boundaries: Vec<u64>,
    /// Sum of all observations.
    sum: AtomicU64,
    /// Count of observations.
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram with default buckets.
    pub fn new() -> Self {
        // Default: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 5s
        let boundaries = vec![
            1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 5000000,
        ];
        let buckets = (0..=boundaries.len())
            .map(|_| AtomicU64::new(0))
            .collect();

        Self {
            buckets,
            boundaries,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Observe a duration.
    pub fn observe(&self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        self.sum.fetch_add(micros, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        let idx = self
            .boundaries
            .iter()
            .position(|&b| micros <= b)
            .unwrap_or(self.boundaries.len());
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Get mean value.
    pub fn mean(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let sum = self.sum.load(Ordering::Relaxed);
        (sum as f64) / (count as f64)
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Proxy metrics.
#[derive(Default)]
pub struct ProxyMetrics {
    // Request metrics
    /// Total requests.
    pub requests_total: Counter,
    /// Request latency.
    pub request_latency: Histogram,
    /// Request bytes.
    pub request_bytes: Counter,
    /// Response bytes.
    pub response_bytes: Counter,

    // Connection metrics
    /// Active connections.
    pub active_connections: Gauge,
    /// Connection errors.
    pub connection_errors: Counter,

    // mTLS metrics
    /// TLS handshake latency.
    pub tls_handshake_latency: Histogram,
    /// Certificate expiry seconds.
    pub cert_expiry_seconds: Gauge,
    /// Authorization denied.
    pub auth_denied: Counter,

    // Circuit breaker
    /// Circuit open events.
    pub circuit_open: Counter,

    // Retry metrics
    /// Total retries.
    pub retries_total: Counter,
}

impl ProxyMetrics {
    /// Create new proxy metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a request with latency and status.
    pub fn record_request(&self, latency: std::time::Duration, status: u16) {
        self.requests_total.inc();
        self.request_latency.observe(latency);

        // Track 5xx errors as connection errors
        if status >= 500 {
            self.connection_errors.inc();
        }
    }

    /// Export metrics in Prometheus format.
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        // requests_total
        output.push_str("# HELP mesh_requests_total Total number of requests processed\n");
        output.push_str("# TYPE mesh_requests_total counter\n");
        output.push_str(&format!("mesh_requests_total {}\n", self.requests_total.get()));

        // request_latency
        output.push_str("# HELP mesh_request_latency_seconds Request latency in seconds\n");
        output.push_str("# TYPE mesh_request_latency_seconds histogram\n");
        let count = self.request_latency.count.load(std::sync::atomic::Ordering::Relaxed);
        let sum_micros = self.request_latency.sum.load(std::sync::atomic::Ordering::Relaxed);
        output.push_str(&format!(
            "mesh_request_latency_seconds_count {}\n",
            count
        ));
        output.push_str(&format!(
            "mesh_request_latency_seconds_sum {:.6}\n",
            sum_micros as f64 / 1_000_000.0
        ));

        // active_connections
        output.push_str("# HELP mesh_active_connections Number of active connections\n");
        output.push_str("# TYPE mesh_active_connections gauge\n");
        output.push_str(&format!(
            "mesh_active_connections {}\n",
            self.active_connections.get()
        ));

        // connection_errors
        output.push_str("# HELP mesh_connection_errors_total Total connection errors\n");
        output.push_str("# TYPE mesh_connection_errors_total counter\n");
        output.push_str(&format!(
            "mesh_connection_errors_total {}\n",
            self.connection_errors.get()
        ));

        // auth_denied
        output.push_str("# HELP mesh_auth_denied_total Total authorization denials\n");
        output.push_str("# TYPE mesh_auth_denied_total counter\n");
        output.push_str(&format!("mesh_auth_denied_total {}\n", self.auth_denied.get()));

        // circuit_open
        output.push_str("# HELP mesh_circuit_open_total Total circuit breaker openings\n");
        output.push_str("# TYPE mesh_circuit_open_total counter\n");
        output.push_str(&format!("mesh_circuit_open_total {}\n", self.circuit_open.get()));

        // retries_total
        output.push_str("# HELP mesh_retries_total Total request retries\n");
        output.push_str("# TYPE mesh_retries_total counter\n");
        output.push_str(&format!("mesh_retries_total {}\n", self.retries_total.get()));

        output
    }

    /// Get metrics snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            requests_total: self.requests_total.get(),
            request_latency_mean: self.request_latency.mean(),
            active_connections: self.active_connections.get(),
            connection_errors: self.connection_errors.get(),
            auth_denied: self.auth_denied.get(),
            circuit_open: self.circuit_open.get(),
            retries_total: self.retries_total.get(),
        }
    }
}

/// Snapshot of metrics.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub request_latency_mean: f64,
    pub active_connections: u64,
    pub connection_errors: u64,
    pub auth_denied: u64,
    pub circuit_open: u64,
    pub retries_total: u64,
}
