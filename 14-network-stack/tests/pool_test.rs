//! Comprehensive connection pool tests
//!
//! Tests for connection pooling, connection management,
//! host limiting, DNS caching, and connection guards.

use network_stack::pool::*;
use network_stack::tcp::TcpConnection;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

/// Helper to create a test TCP connection
fn create_test_connection(local_port: u16, remote_addr: &str) -> TcpConnection {
    let local: SocketAddr = format!("127.0.0.1:{}", local_port).parse().unwrap();
    let remote: SocketAddr = remote_addr.parse().unwrap();
    TcpConnection::new(local, remote)
}

// =============================================================================
// PoolConfig Tests
// =============================================================================

#[cfg(test)]
mod pool_config_tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PoolConfig::default();

        assert_eq!(config.max_per_host, 10);
        assert_eq!(config.max_total, 100);
        assert_eq!(config.connect_timeout, Duration::from_secs(10));
        assert_eq!(config.max_idle_time, Duration::from_secs(90));
        assert_eq!(config.max_lifetime, Duration::from_secs(3600));
        assert_eq!(config.max_requests_per_conn, 1000);
    }

    #[test]
    fn test_custom_config() {
        let config = PoolConfig {
            max_per_host: 5,
            max_total: 50,
            connect_timeout: Duration::from_secs(5),
            max_idle_time: Duration::from_secs(30),
            max_lifetime: Duration::from_secs(1800),
            max_requests_per_conn: 500,
        };

        assert_eq!(config.max_per_host, 5);
        assert_eq!(config.max_total, 50);
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
    }
}

// =============================================================================
// PooledConnection Tests
// =============================================================================

#[cfg(test)]
mod pooled_connection_tests {
    use super::*;

    #[test]
    fn test_new_pooled_connection() {
        let conn = create_test_connection(12345, "127.0.0.1:80");
        let pooled = PooledConnection::new(conn);

        assert_eq!(pooled.requests_served, 0);
        assert!(pooled.created_at.elapsed() < Duration::from_secs(1));
        assert!(pooled.last_used.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn test_is_stale() {
        let conn = create_test_connection(12345, "127.0.0.1:80");
        let mut pooled = PooledConnection::new(conn);

        // Freshly created should not be stale
        assert!(!pooled.is_stale(Duration::from_secs(60)));

        // Simulate old last_used time
        pooled.last_used = Instant::now() - Duration::from_secs(120);
        assert!(pooled.is_stale(Duration::from_secs(60)));
    }

    #[test]
    fn test_is_expired() {
        let conn = create_test_connection(12345, "127.0.0.1:80");
        let mut pooled = PooledConnection::new(conn);

        // Freshly created should not be expired
        assert!(!pooled.is_expired(Duration::from_secs(3600)));

        // Simulate old creation time
        pooled.created_at = Instant::now() - Duration::from_secs(7200);
        assert!(pooled.is_expired(Duration::from_secs(3600)));
    }

    #[test]
    fn test_requests_served_tracking() {
        let conn = create_test_connection(12345, "127.0.0.1:80");
        let mut pooled = PooledConnection::new(conn);

        assert_eq!(pooled.requests_served, 0);

        pooled.requests_served += 1;
        assert_eq!(pooled.requests_served, 1);

        pooled.requests_served += 10;
        assert_eq!(pooled.requests_served, 11);
    }
}

// =============================================================================
// ConnectionPool Tests
// =============================================================================

#[cfg(test)]
mod connection_pool_tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config);

        let stats = pool.stats();
        assert_eq!(stats.idle, 0);
        assert_eq!(stats.active, 0);
        assert_eq!(stats.total, 0);
        assert_eq!(stats.hosts, 0);
    }

    #[test]
    fn test_acquire_slot() {
        let config = PoolConfig {
            max_per_host: 2,
            max_total: 10,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // First slot should succeed
        assert!(pool.acquire_slot(addr).unwrap());

        // Second slot should succeed
        assert!(pool.acquire_slot(addr).unwrap());

        // Third should fail (max_per_host = 2)
        assert!(!pool.acquire_slot(addr).unwrap());
    }

    #[test]
    fn test_acquire_slot_total_limit() {
        let config = PoolConfig {
            max_per_host: 10,
            max_total: 3,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);

        let addr1: SocketAddr = "127.0.0.1:80".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:443".parse().unwrap();

        // Acquire 3 slots across different hosts
        assert!(pool.acquire_slot(addr1).unwrap());
        assert!(pool.acquire_slot(addr2).unwrap());
        assert!(pool.acquire_slot(addr1).unwrap());

        // Fourth should fail (max_total = 3)
        assert!(!pool.acquire_slot(addr2).unwrap());
    }

    #[test]
    fn test_release_slot() {
        let config = PoolConfig {
            max_per_host: 1,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Acquire slot
        assert!(pool.acquire_slot(addr).unwrap());

        // Cannot acquire another
        assert!(!pool.acquire_slot(addr).unwrap());

        // Release slot
        pool.release_slot(addr);

        // Now we can acquire again
        assert!(pool.acquire_slot(addr).unwrap());
    }

    #[test]
    fn test_put_and_get_connection() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Acquire slot and create connection
        pool.acquire_slot(addr).unwrap();
        let conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));
        pool.put(addr, conn);

        // Get should return the connection
        let retrieved = pool.get(addr).unwrap();
        assert!(retrieved.is_some());

        // Get again should return None (connection was taken)
        let empty = pool.get(addr).unwrap();
        assert!(empty.is_none());
    }

    #[test]
    fn test_get_from_empty_pool() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        let result = pool.get(addr).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_connection_reuse() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Create and put connection
        pool.acquire_slot(addr).unwrap();
        let conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));
        pool.put(addr, conn);

        // Retrieve connection
        let mut retrieved = pool.get(addr).unwrap().unwrap();

        // Verify requests_served was incremented
        assert!(retrieved.requests_served >= 1);

        // Put back for reuse
        pool.put(addr, retrieved);

        // Get again
        let reused = pool.get(addr).unwrap().unwrap();
        assert!(reused.requests_served >= 2);
    }

    #[test]
    fn test_stale_connection_removal_on_get() {
        let config = PoolConfig {
            max_idle_time: Duration::from_millis(1),
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Create and put connection
        pool.acquire_slot(addr).unwrap();
        let mut conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));

        // Make connection stale
        conn.last_used = Instant::now() - Duration::from_secs(10);
        pool.put(addr, conn);

        // Get should return None (connection was stale)
        let result = pool.get(addr).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_expired_connection_not_pooled() {
        let config = PoolConfig {
            max_lifetime: Duration::from_millis(1),
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Create expired connection
        let mut conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));
        conn.created_at = Instant::now() - Duration::from_secs(10);

        pool.acquire_slot(addr).unwrap();
        pool.put(addr, conn);

        // Connection should not have been pooled
        let result = pool.get(addr).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_max_requests_per_conn() {
        let config = PoolConfig {
            max_requests_per_conn: 5,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Create connection with max requests
        let mut conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));
        conn.requests_served = 5;

        pool.acquire_slot(addr).unwrap();
        pool.put(addr, conn);

        // Connection should not be available (max requests reached)
        let result = pool.get(addr).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_cleanup() {
        let config = PoolConfig {
            max_idle_time: Duration::from_millis(1),
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Add several connections
        for i in 0..5 {
            pool.acquire_slot(addr).unwrap();
            let mut conn = PooledConnection::new(create_test_connection(12345 + i, "127.0.0.1:80"));
            conn.last_used = Instant::now() - Duration::from_secs(10);
            pool.put(addr, conn);
        }

        // Cleanup stale connections
        pool.cleanup();

        // All should be gone
        let stats = pool.stats();
        assert_eq!(stats.idle, 0);
    }

    #[test]
    fn test_pool_stats() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config);

        let addr1: SocketAddr = "127.0.0.1:80".parse().unwrap();
        let addr2: SocketAddr = "127.0.0.1:443".parse().unwrap();

        // Add some connections
        pool.acquire_slot(addr1).unwrap();
        pool.acquire_slot(addr2).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.active, 2);
        assert_eq!(stats.total, 2);
    }

    #[test]
    fn test_pool_metrics() {
        let config = PoolConfig::default();
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Initial metrics
        let metrics = pool.metrics();
        assert_eq!(metrics.created.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.reused.load(Ordering::Relaxed), 0);

        // Create connection
        pool.acquire_slot(addr).unwrap();
        assert_eq!(metrics.created.load(Ordering::Relaxed), 1);

        // Put and get (reuse)
        let conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));
        pool.put(addr, conn);
        assert_eq!(metrics.returned.load(Ordering::Relaxed), 1);

        pool.get(addr).unwrap();
        assert_eq!(metrics.reused.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_multiple_hosts() {
        let config = PoolConfig {
            max_per_host: 2,
            max_total: 10,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);

        let hosts: Vec<SocketAddr> = vec![
            "10.0.0.1:80".parse().unwrap(),
            "10.0.0.2:80".parse().unwrap(),
            "10.0.0.3:80".parse().unwrap(),
        ];

        // Each host can have 2 connections
        for host in &hosts {
            assert!(pool.acquire_slot(*host).unwrap());
            assert!(pool.acquire_slot(*host).unwrap());
            assert!(!pool.acquire_slot(*host).unwrap()); // Third fails
        }

        let stats = pool.stats();
        assert_eq!(stats.total, 6); // 2 per host * 3 hosts
    }
}

// =============================================================================
// Semaphore Tests
// =============================================================================

#[cfg(test)]
mod semaphore_tests {
    use super::*;

    #[test]
    fn test_semaphore_creation() {
        let sem = Semaphore::new(5);
        assert_eq!(sem.available(), 5);
    }

    #[test]
    fn test_semaphore_acquire() {
        let sem = Semaphore::new(2);

        assert!(sem.try_acquire());
        assert_eq!(sem.available(), 1);

        assert!(sem.try_acquire());
        assert_eq!(sem.available(), 0);

        // Cannot acquire more
        assert!(!sem.try_acquire());
        assert_eq!(sem.available(), 0);
    }

    #[test]
    fn test_semaphore_release() {
        let sem = Semaphore::new(2);

        // Acquire all
        sem.try_acquire();
        sem.try_acquire();
        assert_eq!(sem.available(), 0);

        // Release one
        sem.release();
        assert_eq!(sem.available(), 1);

        // Can acquire again
        assert!(sem.try_acquire());
    }

    #[test]
    fn test_semaphore_over_release() {
        let sem = Semaphore::new(2);

        // Release without acquiring
        sem.release();

        // Should still be capped at max
        assert_eq!(sem.available(), 2);
    }

    #[test]
    fn test_semaphore_zero_permits() {
        let sem = Semaphore::new(0);

        assert_eq!(sem.available(), 0);
        assert!(!sem.try_acquire());
    }
}

// =============================================================================
// HostLimiter Tests
// =============================================================================

#[cfg(test)]
mod host_limiter_tests {
    use super::*;

    #[test]
    fn test_host_limiter_creation() {
        let limiter = HostLimiter::new(5);
        // Should be able to create
        let _ = limiter;
    }

    #[test]
    fn test_host_limiter_acquire() {
        let limiter = HostLimiter::new(2);

        assert!(limiter.try_acquire("example.com"));
        assert!(limiter.try_acquire("example.com"));
        assert!(!limiter.try_acquire("example.com"));
    }

    #[test]
    fn test_host_limiter_release() {
        let limiter = HostLimiter::new(1);

        assert!(limiter.try_acquire("example.com"));
        assert!(!limiter.try_acquire("example.com"));

        limiter.release("example.com");

        assert!(limiter.try_acquire("example.com"));
    }

    #[test]
    fn test_host_limiter_different_hosts() {
        let limiter = HostLimiter::new(1);

        // Each host has its own limit
        assert!(limiter.try_acquire("host1.com"));
        assert!(limiter.try_acquire("host2.com"));
        assert!(limiter.try_acquire("host3.com"));

        // But same host is limited
        assert!(!limiter.try_acquire("host1.com"));
    }

    #[test]
    fn test_get_limiter() {
        let limiter = HostLimiter::new(5);

        let sem1 = limiter.get_limiter("example.com");
        let sem2 = limiter.get_limiter("example.com");

        // Should return same semaphore for same host
        assert_eq!(sem1.available(), sem2.available());
    }
}

// =============================================================================
// DnsCache Tests
// =============================================================================

#[cfg(test)]
mod dns_cache_tests {
    use super::*;

    #[test]
    fn test_dns_cache_creation() {
        let cache = DnsCache::new(Duration::from_secs(60));
        assert!(cache.get("example.com").is_none());
    }

    #[test]
    fn test_dns_cache_put_get() {
        let cache = DnsCache::new(Duration::from_secs(60));
        let addr: SocketAddr = "93.184.216.34:80".parse().unwrap();

        cache.put("example.com".to_string(), vec![addr]);

        let result = cache.get("example.com");
        assert!(result.is_some());
        assert_eq!(result.unwrap()[0], addr);
    }

    #[test]
    fn test_dns_cache_multiple_addresses() {
        let cache = DnsCache::new(Duration::from_secs(60));
        let addrs: Vec<SocketAddr> = vec![
            "93.184.216.34:80".parse().unwrap(),
            "93.184.216.35:80".parse().unwrap(),
        ];

        cache.put("example.com".to_string(), addrs.clone());

        let result = cache.get("example.com").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], addrs[0]);
        assert_eq!(result[1], addrs[1]);
    }

    #[test]
    fn test_dns_cache_miss() {
        let cache = DnsCache::new(Duration::from_secs(60));

        assert!(cache.get("nonexistent.com").is_none());
    }

    #[test]
    fn test_dns_cache_expiry() {
        let cache = DnsCache::new(Duration::from_millis(1));
        let addr: SocketAddr = "93.184.216.34:80".parse().unwrap();

        cache.put("example.com".to_string(), vec![addr]);

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(10));

        // Should be expired
        assert!(cache.get("example.com").is_none());
    }

    #[test]
    fn test_dns_cache_cleanup() {
        let cache = DnsCache::new(Duration::from_millis(1));
        let addr: SocketAddr = "93.184.216.34:80".parse().unwrap();

        cache.put("host1.com".to_string(), vec![addr]);
        cache.put("host2.com".to_string(), vec![addr]);
        cache.put("host3.com".to_string(), vec![addr]);

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(10));

        // Cleanup
        cache.cleanup();

        // All should be gone
        assert!(cache.get("host1.com").is_none());
        assert!(cache.get("host2.com").is_none());
        assert!(cache.get("host3.com").is_none());
    }

    #[test]
    fn test_dns_cache_overwrite() {
        let cache = DnsCache::new(Duration::from_secs(60));
        let addr1: SocketAddr = "93.184.216.34:80".parse().unwrap();
        let addr2: SocketAddr = "93.184.216.35:80".parse().unwrap();

        cache.put("example.com".to_string(), vec![addr1]);
        cache.put("example.com".to_string(), vec![addr2]);

        let result = cache.get("example.com").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], addr2);
    }
}

// =============================================================================
// HttpClient Tests
// =============================================================================

#[cfg(test)]
mod http_client_tests {
    use super::*;

    #[test]
    fn test_http_client_creation() {
        let client = HttpClient::new(PoolConfig::default());
        let _ = client.pool(); // Should not panic
    }

    #[test]
    fn test_http_client_default_headers() {
        let client = HttpClient::new(PoolConfig::default())
            .default_header("User-Agent", "TestClient/1.0")
            .default_header("Accept", "application/json");

        // Client created successfully with headers
        let _ = client;
    }

    #[test]
    fn test_http_client_pool_access() {
        let config = PoolConfig {
            max_per_host: 5,
            ..Default::default()
        };
        let client = HttpClient::new(config);

        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();
        assert!(client.pool().acquire_slot(addr).unwrap());
    }
}

// =============================================================================
// ConnectionGuard Tests
// =============================================================================

#[cfg(test)]
mod connection_guard_tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_guard_connection_access() {
        let config = PoolConfig::default();
        let pool = Arc::new(ConnectionPool::new(config));
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        pool.acquire_slot(addr).unwrap();
        let conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));

        let mut guard = ConnectionGuard::new(conn, Arc::clone(&pool), addr);

        // Access connection through guard
        let conn_ref = guard.connection();
        assert_eq!(conn_ref.requests_served, 0);
    }

    #[test]
    fn test_guard_take() {
        let config = PoolConfig::default();
        let pool = Arc::new(ConnectionPool::new(config));
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        pool.acquire_slot(addr).unwrap();
        let conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));

        let guard = ConnectionGuard::new(conn, Arc::clone(&pool), addr);

        // Take connection from guard
        let taken = guard.take();
        assert_eq!(taken.requests_served, 0);
    }

    #[test]
    fn test_guard_auto_return_on_drop() {
        let config = PoolConfig::default();
        let pool = Arc::new(ConnectionPool::new(config));
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        pool.acquire_slot(addr).unwrap();
        let conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));

        {
            let _guard = ConnectionGuard::new(conn, Arc::clone(&pool), addr);
            // guard will be dropped here
        }

        // Connection should be back in pool
        let retrieved = pool.get(addr).unwrap();
        assert!(retrieved.is_some());
    }

    #[test]
    fn test_guard_broken_no_return() {
        let config = PoolConfig::default();
        let pool = Arc::new(ConnectionPool::new(config));
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        pool.acquire_slot(addr).unwrap();
        let conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));

        let guard = ConnectionGuard::new(conn, Arc::clone(&pool), addr);

        // Mark as broken
        guard.broken();

        // Connection should NOT be in pool
        let retrieved = pool.get(addr).unwrap();
        assert!(retrieved.is_none());
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod pool_integration_tests {
    use super::*;

    #[test]
    fn test_full_connection_lifecycle() {
        let config = PoolConfig {
            max_per_host: 5,
            max_total: 10,
            max_idle_time: Duration::from_secs(60),
            max_lifetime: Duration::from_secs(3600),
            max_requests_per_conn: 100,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // 1. Acquire slot
        assert!(pool.acquire_slot(addr).unwrap());

        // 2. Create connection
        let conn = PooledConnection::new(create_test_connection(12345, "127.0.0.1:80"));

        // 3. Return to pool
        pool.put(addr, conn);

        // 4. Reuse connection
        let reused = pool.get(addr).unwrap().unwrap();
        assert!(reused.requests_served >= 1);

        // 5. Return again
        pool.put(addr, reused);

        // 6. Verify stats
        let stats = pool.stats();
        assert_eq!(stats.idle, 1);
    }

    #[test]
    fn test_concurrent_access_simulation() {
        let config = PoolConfig {
            max_per_host: 3,
            max_total: 10,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);
        let addr: SocketAddr = "127.0.0.1:80".parse().unwrap();

        // Simulate concurrent access
        let mut connections = Vec::new();

        // Acquire 3 connections
        for i in 0..3 {
            assert!(pool.acquire_slot(addr).unwrap());
            let conn = PooledConnection::new(create_test_connection(12345 + i, "127.0.0.1:80"));
            connections.push(conn);
        }

        // Cannot acquire 4th
        assert!(!pool.acquire_slot(addr).unwrap());

        // Return all to pool
        for conn in connections {
            pool.put(addr, conn);
        }

        // Can now get all 3
        for _ in 0..3 {
            assert!(pool.get(addr).unwrap().is_some());
        }
    }

    #[test]
    fn test_mixed_hosts_and_limits() {
        let config = PoolConfig {
            max_per_host: 2,
            max_total: 4,
            ..Default::default()
        };
        let pool = ConnectionPool::new(config);

        let host1: SocketAddr = "10.0.0.1:80".parse().unwrap();
        let host2: SocketAddr = "10.0.0.2:80".parse().unwrap();
        let host3: SocketAddr = "10.0.0.3:80".parse().unwrap();

        // 2 for host1
        assert!(pool.acquire_slot(host1).unwrap());
        assert!(pool.acquire_slot(host1).unwrap());
        assert!(!pool.acquire_slot(host1).unwrap()); // per-host limit

        // 2 for host2
        assert!(pool.acquire_slot(host2).unwrap());
        assert!(pool.acquire_slot(host2).unwrap());

        // Already at max_total of 4, this should fail
        assert!(!pool.acquire_slot(host3).unwrap()); // total limit

        let stats = pool.stats();
        assert_eq!(stats.total, 4);
    }
}
