//! Connection pooling for HTTP connections.
//!
//! Manages persistent connections for HTTP/1.1 keep-alive.

use crate::tcp::TcpConnection;
use crate::{Error, Result};
use parking_lot::Mutex;
use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Pooled connection.
pub struct PooledConnection {
    /// TCP connection.
    pub connection: TcpConnection,
    /// Creation time.
    pub created_at: Instant,
    /// Last used time.
    pub last_used: Instant,
    /// Number of requests served.
    pub requests_served: u64,
}

impl PooledConnection {
    /// Create a new pooled connection.
    pub fn new(connection: TcpConnection) -> Self {
        let now = Instant::now();
        Self {
            connection,
            created_at: now,
            last_used: now,
            requests_served: 0,
        }
    }

    /// Check if connection is stale.
    pub fn is_stale(&self, max_idle: Duration) -> bool {
        self.last_used.elapsed() > max_idle
    }

    /// Check if connection has exceeded max lifetime.
    pub fn is_expired(&self, max_lifetime: Duration) -> bool {
        self.created_at.elapsed() > max_lifetime
    }
}

/// Connection pool configuration.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Maximum connections per host.
    pub max_per_host: usize,
    /// Maximum total connections.
    pub max_total: usize,
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Maximum idle time.
    pub max_idle_time: Duration,
    /// Maximum connection lifetime.
    pub max_lifetime: Duration,
    /// Maximum requests per connection.
    pub max_requests_per_conn: u64,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_per_host: 10,
            max_total: 100,
            connect_timeout: Duration::from_secs(10),
            max_idle_time: Duration::from_secs(90),
            max_lifetime: Duration::from_secs(3600),
            max_requests_per_conn: 1000,
        }
    }
}

/// Connection pool.
pub struct ConnectionPool {
    /// Configuration.
    config: PoolConfig,
    /// Idle connections by address.
    idle: Mutex<HashMap<SocketAddr, VecDeque<PooledConnection>>>,
    /// Count of active connections by address.
    active_counts: Mutex<HashMap<SocketAddr, usize>>,
    /// Total connection count.
    total_count: Mutex<usize>,
    /// Pool metrics.
    metrics: Arc<PoolMetrics>,
}

impl ConnectionPool {
    /// Create a new connection pool.
    pub fn new(config: PoolConfig) -> Self {
        Self {
            config,
            idle: Mutex::new(HashMap::new()),
            active_counts: Mutex::new(HashMap::new()),
            total_count: Mutex::new(0),
            metrics: Arc::new(PoolMetrics::default()),
        }
    }

    /// Get a connection from the pool.
    pub fn get(&self, addr: SocketAddr) -> Result<Option<PooledConnection>> {
        let mut idle = self.idle.lock();

        if let Some(conns) = idle.get_mut(&addr) {
            // Try to find a valid idle connection
            while let Some(mut conn) = conns.pop_front() {
                // Check if connection is still valid
                if conn.is_stale(self.config.max_idle_time) {
                    self.metrics.stale_removed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    *self.total_count.lock() -= 1;
                    continue;
                }

                if conn.is_expired(self.config.max_lifetime) {
                    self.metrics.expired_removed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    *self.total_count.lock() -= 1;
                    continue;
                }

                if conn.requests_served >= self.config.max_requests_per_conn {
                    self.metrics.maxed_out_removed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    *self.total_count.lock() -= 1;
                    continue;
                }

                // Valid connection found
                conn.last_used = Instant::now();
                conn.requests_served += 1;

                *self.active_counts.lock().entry(addr).or_insert(0) += 1;
                self.metrics.reused.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

                return Ok(Some(conn));
            }
        }

        // No idle connection available
        self.metrics.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(None)
    }

    /// Try to acquire a slot for a new connection.
    pub fn acquire_slot(&self, addr: SocketAddr) -> Result<bool> {
        let mut active = self.active_counts.lock();
        let mut total = self.total_count.lock();

        // Check per-host limit
        let host_count = active.get(&addr).copied().unwrap_or(0);
        if host_count >= self.config.max_per_host {
            return Ok(false);
        }

        // Check total limit
        if *total >= self.config.max_total {
            return Ok(false);
        }

        // Acquire slot
        *active.entry(addr).or_insert(0) += 1;
        *total += 1;

        self.metrics.created.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(true)
    }

    /// Return a connection to the pool.
    pub fn put(&self, addr: SocketAddr, conn: PooledConnection) {
        // Release active slot
        {
            let mut active = self.active_counts.lock();
            if let Some(count) = active.get_mut(&addr) {
                *count = count.saturating_sub(1);
            }
        }

        // Check if connection should be pooled
        if conn.is_expired(self.config.max_lifetime) {
            self.metrics.expired_removed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            *self.total_count.lock() -= 1;
            return;
        }

        if conn.requests_served >= self.config.max_requests_per_conn {
            self.metrics.maxed_out_removed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            *self.total_count.lock() -= 1;
            return;
        }

        // Add to idle pool
        let mut idle = self.idle.lock();
        let conns = idle.entry(addr).or_default();

        // Check if at per-host capacity
        if conns.len() >= self.config.max_per_host {
            *self.total_count.lock() -= 1;
            return;
        }

        conns.push_back(conn);
        self.metrics.returned.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Release a slot without returning a connection.
    pub fn release_slot(&self, addr: SocketAddr) {
        let mut active = self.active_counts.lock();
        if let Some(count) = active.get_mut(&addr) {
            *count = count.saturating_sub(1);
        }
        *self.total_count.lock() -= 1;
    }

    /// Clean up stale connections.
    pub fn cleanup(&self) {
        let mut idle = self.idle.lock();
        let mut total = self.total_count.lock();
        let mut removed = 0;

        for conns in idle.values_mut() {
            let before = conns.len();
            conns.retain(|c| {
                !c.is_stale(self.config.max_idle_time) && !c.is_expired(self.config.max_lifetime)
            });
            removed += before - conns.len();
        }

        *total = total.saturating_sub(removed);
        self.metrics.cleanup_removed.fetch_add(removed as u64, std::sync::atomic::Ordering::Relaxed);
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let idle = self.idle.lock();
        let active = self.active_counts.lock();

        let idle_count: usize = idle.values().map(|v| v.len()).sum();
        let active_count: usize = active.values().sum();

        PoolStats {
            idle: idle_count,
            active: active_count,
            total: *self.total_count.lock(),
            hosts: idle.len(),
        }
    }

    /// Get metrics.
    pub fn metrics(&self) -> &PoolMetrics {
        &self.metrics
    }
}

/// Pool statistics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Idle connections.
    pub idle: usize,
    /// Active connections.
    pub active: usize,
    /// Total connections.
    pub total: usize,
    /// Number of hosts.
    pub hosts: usize,
}

/// Pool metrics.
#[derive(Default)]
pub struct PoolMetrics {
    /// Connections created.
    pub created: std::sync::atomic::AtomicU64,
    /// Connections reused.
    pub reused: std::sync::atomic::AtomicU64,
    /// Pool misses.
    pub misses: std::sync::atomic::AtomicU64,
    /// Connections returned.
    pub returned: std::sync::atomic::AtomicU64,
    /// Stale connections removed.
    pub stale_removed: std::sync::atomic::AtomicU64,
    /// Expired connections removed.
    pub expired_removed: std::sync::atomic::AtomicU64,
    /// Maxed out connections removed.
    pub maxed_out_removed: std::sync::atomic::AtomicU64,
    /// Cleanup removed.
    pub cleanup_removed: std::sync::atomic::AtomicU64,
}

/// HTTP client with connection pooling.
pub struct HttpClient {
    /// Connection pool.
    pool: Arc<ConnectionPool>,
    /// Default headers.
    default_headers: crate::http::Headers,
}

impl HttpClient {
    /// Create a new HTTP client.
    pub fn new(config: PoolConfig) -> Self {
        Self {
            pool: Arc::new(ConnectionPool::new(config)),
            default_headers: crate::http::Headers::new(),
        }
    }

    /// Set default header.
    pub fn default_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.default_headers.set(name, value);
        self
    }

    /// Get connection pool.
    pub fn pool(&self) -> &ConnectionPool {
        &self.pool
    }
}

/// Connection guard for automatic return to pool.
pub struct ConnectionGuard {
    /// The connection.
    connection: Option<PooledConnection>,
    /// Pool reference.
    pool: Arc<ConnectionPool>,
    /// Address.
    addr: SocketAddr,
}

impl ConnectionGuard {
    /// Create a new guard.
    pub fn new(connection: PooledConnection, pool: Arc<ConnectionPool>, addr: SocketAddr) -> Self {
        Self {
            connection: Some(connection),
            pool,
            addr,
        }
    }

    /// Get connection reference.
    pub fn connection(&mut self) -> &mut PooledConnection {
        self.connection.as_mut().unwrap()
    }

    /// Take the connection without returning to pool.
    pub fn take(mut self) -> PooledConnection {
        self.connection.take().unwrap()
    }

    /// Mark connection as broken (don't return to pool).
    pub fn broken(mut self) {
        self.connection.take();
        self.pool.release_slot(self.addr);
    }
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        if let Some(conn) = self.connection.take() {
            self.pool.put(self.addr, conn);
        }
    }
}

/// Host connection limiter.
pub struct HostLimiter {
    /// Semaphores by host.
    limits: Mutex<HashMap<String, Arc<Semaphore>>>,
    /// Max concurrent requests per host.
    max_per_host: usize,
}

/// Simple semaphore for limiting concurrency.
pub struct Semaphore {
    /// Available permits.
    permits: Mutex<usize>,
    /// Maximum permits.
    max: usize,
}

impl Semaphore {
    /// Create a new semaphore.
    pub fn new(permits: usize) -> Self {
        Self {
            permits: Mutex::new(permits),
            max: permits,
        }
    }

    /// Try to acquire a permit.
    pub fn try_acquire(&self) -> bool {
        let mut permits = self.permits.lock();
        if *permits > 0 {
            *permits -= 1;
            true
        } else {
            false
        }
    }

    /// Release a permit.
    pub fn release(&self) {
        let mut permits = self.permits.lock();
        if *permits < self.max {
            *permits += 1;
        }
    }

    /// Get available permits.
    pub fn available(&self) -> usize {
        *self.permits.lock()
    }
}

impl HostLimiter {
    /// Create a new host limiter.
    pub fn new(max_per_host: usize) -> Self {
        Self {
            limits: Mutex::new(HashMap::new()),
            max_per_host,
        }
    }

    /// Get or create limiter for host.
    pub fn get_limiter(&self, host: &str) -> Arc<Semaphore> {
        let mut limits = self.limits.lock();
        limits
            .entry(host.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(self.max_per_host)))
            .clone()
    }

    /// Try to acquire permit for host.
    pub fn try_acquire(&self, host: &str) -> bool {
        self.get_limiter(host).try_acquire()
    }

    /// Release permit for host.
    pub fn release(&self, host: &str) {
        self.get_limiter(host).release();
    }
}

/// DNS cache for reducing lookups.
pub struct DnsCache {
    /// Cache entries.
    cache: Mutex<HashMap<String, DnsCacheEntry>>,
    /// TTL for entries.
    ttl: Duration,
}

/// DNS cache entry.
struct DnsCacheEntry {
    /// Resolved addresses.
    addresses: Vec<SocketAddr>,
    /// Expiry time.
    expires_at: Instant,
}

impl DnsCache {
    /// Create a new DNS cache.
    pub fn new(ttl: Duration) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            ttl,
        }
    }

    /// Get cached addresses.
    pub fn get(&self, host: &str) -> Option<Vec<SocketAddr>> {
        let cache = self.cache.lock();
        cache.get(host).and_then(|entry| {
            if entry.expires_at > Instant::now() {
                Some(entry.addresses.clone())
            } else {
                None
            }
        })
    }

    /// Put addresses in cache.
    pub fn put(&self, host: String, addresses: Vec<SocketAddr>) {
        let mut cache = self.cache.lock();
        cache.insert(
            host,
            DnsCacheEntry {
                addresses,
                expires_at: Instant::now() + self.ttl,
            },
        );
    }

    /// Remove expired entries.
    pub fn cleanup(&self) {
        let mut cache = self.cache.lock();
        let now = Instant::now();
        cache.retain(|_, entry| entry.expires_at > now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tcp::TcpConnection;

    fn create_test_connection(port: u16) -> TcpConnection {
        let local: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
        let remote: SocketAddr = "127.0.0.1:80".parse().unwrap();
        TcpConnection::new(local, remote)
    }

    #[test]
    fn test_pool_acquire_release() {
        let config = PoolConfig {
            max_per_host: 2,
            max_total: 10,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Acquire slots
        assert!(pool.acquire_slot(addr).unwrap());
        assert!(pool.acquire_slot(addr).unwrap());
        assert!(!pool.acquire_slot(addr).unwrap()); // At limit

        // Release slot
        pool.release_slot(addr);
        assert!(pool.acquire_slot(addr).unwrap());
    }

    #[test]
    fn test_pool_reuse() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Create and return connection
        let conn = PooledConnection::new(create_test_connection(12345));
        pool.acquire_slot(addr).unwrap();
        pool.put(addr, conn);

        // Should get back a connection
        let retrieved = pool.get(addr).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_semaphore() {
        let sem = Semaphore::new(2);

        assert!(sem.try_acquire());
        assert!(sem.try_acquire());
        assert!(!sem.try_acquire());

        sem.release();
        assert!(sem.try_acquire());
    }

    #[test]
    fn test_dns_cache() {
        let cache = DnsCache::new(Duration::from_secs(60));
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        cache.put("example.com".into(), vec![addr]);

        let result = cache.get("example.com");
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], addr);

        // Non-existent
        assert!(cache.get("unknown.com").is_none());
    }

    #[test]
    fn test_pool_stats() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        let stats = pool.stats();
        assert_eq!(stats.idle, 0);
        assert_eq!(stats.total, 0);

        pool.acquire_slot(addr).unwrap();
        let stats = pool.stats();
        assert_eq!(stats.active, 1);
        assert_eq!(stats.total, 1);
    }
}
