//! Service discovery and load balancing.

use crate::config::ServiceIdentity;
use crate::policy::ServicePolicy;
use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Service key for registry.
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ServiceKey {
    /// Service name.
    pub name: String,
    /// Namespace.
    pub namespace: String,
    /// Port.
    pub port: u16,
}

impl ServiceKey {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>, port: u16) -> Self {
        Self {
            name: name.into(),
            namespace: namespace.into(),
            port,
        }
    }
}

/// Service endpoints.
#[derive(Debug, Clone)]
pub struct ServiceEndpoints {
    /// Available endpoints.
    pub endpoints: Vec<Endpoint>,
    /// Load balancer type.
    pub load_balancer: LoadBalancerType,
    /// Service policy.
    pub policy: ServicePolicy,
}

/// A single endpoint.
#[derive(Debug, Clone)]
pub struct Endpoint {
    /// Network address.
    pub address: SocketAddr,
    /// Weight for load balancing.
    pub weight: u32,
    /// Health status.
    pub health: EndpointHealth,
    /// Metadata.
    pub metadata: HashMap<String, String>,
    /// TLS identity.
    pub tls_identity: ServiceIdentity,
}

/// Endpoint health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointHealth {
    /// Healthy and accepting traffic.
    Healthy,
    /// Unhealthy, should not receive traffic.
    Unhealthy,
    /// Status unknown.
    Unknown,
}

/// Load balancer type.
#[derive(Debug, Clone)]
pub enum LoadBalancerType {
    /// Round-robin selection.
    RoundRobin,
    /// Select endpoint with least connections.
    LeastConnections,
    /// Random selection.
    Random,
    /// Consistent hash ring.
    RingHash,
}

/// Hash a value into the 64-bit ring space.
fn hash64<T: Hash>(value: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
}

/// A consistent-hash ring with virtual nodes.
///
/// Each backend is placed at `vnodes` positions around a 64-bit hash ring.
/// A key is routed to the first backend whose position is `>=` the key's
/// hash (wrapping around the ring). This gives:
///
/// - **Stability**: the same key always maps to the same backend for a
///   given membership set.
/// - **Balance**: virtual nodes spread each backend's ownership across the
///   ring so keys distribute roughly evenly.
/// - **Minimal disruption**: adding or removing one backend only reassigns
///   the keys that fell in that backend's arcs (~`1/N` of keys), not all of
///   them as a modulo-based scheme would.
#[derive(Debug, Clone)]
pub struct ConsistentHashRing {
    /// Ring positions mapped to backend addresses.
    ring: BTreeMap<u64, SocketAddr>,
    /// Virtual nodes per backend.
    vnodes: usize,
}

impl ConsistentHashRing {
    /// Default number of virtual nodes per backend.
    pub const DEFAULT_VNODES: usize = 128;

    /// Build a ring over the given backends with `DEFAULT_VNODES` per backend.
    pub fn new(addrs: &[SocketAddr]) -> Self {
        Self::with_vnodes(addrs, Self::DEFAULT_VNODES)
    }

    /// Build a ring with an explicit number of virtual nodes per backend.
    pub fn with_vnodes(addrs: &[SocketAddr], vnodes: usize) -> Self {
        let vnodes = vnodes.max(1);
        let mut ring = BTreeMap::new();
        for addr in addrs {
            for i in 0..vnodes {
                // Hash "addr#i" so each backend occupies vnodes distinct,
                // well-scattered positions on the ring.
                let pos = hash64(&(addr, i));
                ring.insert(pos, *addr);
            }
        }
        Self { ring, vnodes }
    }

    /// Number of distinct backends on the ring.
    pub fn backend_count(&self) -> usize {
        // vnodes positions per backend; count unique addresses instead of
        // dividing, since hash collisions could (rarely) drop a position.
        self.ring
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len()
    }

    /// Look up the backend responsible for `key`.
    ///
    /// Returns the backend at the first ring position `>=` hash(key),
    /// wrapping around to the smallest position if the key hashes past the
    /// last node. `None` only when the ring is empty.
    pub fn lookup<K: Hash>(&self, key: &K) -> Option<SocketAddr> {
        if self.ring.is_empty() {
            return None;
        }
        let h = hash64(key);
        // First node clockwise from the key's position.
        self.ring
            .range(h..)
            .next()
            .map(|(_, addr)| *addr)
            // Wrap around the ring.
            .or_else(|| self.ring.values().next().copied())
    }

    /// Virtual nodes per backend.
    pub fn vnodes(&self) -> usize {
        self.vnodes
    }
}

/// Load balancer for selecting endpoints.
pub struct LoadBalancer {
    /// Load balancer type.
    lb_type: LoadBalancerType,
    /// Round-robin index.
    rr_index: AtomicUsize,
    /// Connection counts per endpoint.
    connections: DashMap<SocketAddr, usize>,
}

impl LoadBalancer {
    /// Create a new load balancer.
    pub fn new(lb_type: LoadBalancerType) -> Self {
        Self {
            lb_type,
            rr_index: AtomicUsize::new(0),
            connections: DashMap::new(),
        }
    }

    /// Select an endpoint.
    pub fn select(&self, endpoints: &[Endpoint]) -> Option<Endpoint> {
        let healthy: Vec<_> = endpoints
            .iter()
            .filter(|e| e.health == EndpointHealth::Healthy)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let selected = match &self.lb_type {
            LoadBalancerType::RoundRobin => {
                let idx = self.rr_index.fetch_add(1, Ordering::Relaxed) % healthy.len();
                healthy[idx]
            }
            LoadBalancerType::LeastConnections => {
                healthy
                    .iter()
                    .min_by_key(|e| {
                        self.connections
                            .get(&e.address)
                            .map(|c| *c)
                            .unwrap_or(0)
                    })
                    .unwrap()
            }
            LoadBalancerType::Random => {
                let idx = rand::random::<usize>() % healthy.len();
                healthy[idx]
            }
            LoadBalancerType::RingHash => {
                // Keyless entry point: RingHash needs a routing key to be
                // meaningful. Without one we fall back to round-robin so the
                // call still returns a valid healthy endpoint. Callers that
                // want consistent-hash routing should use `select_with_key`.
                let idx = self.rr_index.fetch_add(1, Ordering::Relaxed) % healthy.len();
                healthy[idx]
            }
        };

        Some(selected.clone())
    }

    /// Select an endpoint using a routing key.
    ///
    /// For [`LoadBalancerType::RingHash`] this routes `key` through a
    /// consistent-hash ring built from the healthy endpoints: the same key
    /// maps to the same backend across calls, and membership changes only
    /// reassign a bounded fraction of keys. For every other strategy the key
    /// is ignored and this behaves like [`LoadBalancer::select`].
    pub fn select_with_key<K: Hash>(&self, endpoints: &[Endpoint], key: &K) -> Option<Endpoint> {
        if !matches!(self.lb_type, LoadBalancerType::RingHash) {
            return self.select(endpoints);
        }

        let healthy: Vec<_> = endpoints
            .iter()
            .filter(|e| e.health == EndpointHealth::Healthy)
            .collect();

        if healthy.is_empty() {
            return None;
        }

        let addrs: Vec<SocketAddr> = healthy.iter().map(|e| e.address).collect();
        let ring = ConsistentHashRing::new(&addrs);
        let selected_addr = ring.lookup(key)?;

        healthy
            .iter()
            .find(|e| e.address == selected_addr)
            .map(|e| (*e).clone())
    }

    /// Track connection open.
    pub fn connection_opened(&self, addr: SocketAddr) {
        self.connections
            .entry(addr)
            .and_modify(|c| *c += 1)
            .or_insert(1);
    }

    /// Track connection closed.
    pub fn connection_closed(&self, addr: SocketAddr) {
        self.connections.entry(addr).and_modify(|c| *c = c.saturating_sub(1));
    }

    /// Clone the load balancer for use in spawned tasks.
    pub fn clone(&self) -> Self {
        Self {
            lb_type: self.lb_type.clone(),
            rr_index: AtomicUsize::new(self.rr_index.load(Ordering::Relaxed)),
            connections: DashMap::new(),
        }
    }
}

/// Service registry for discovery.
pub struct ServiceRegistry {
    /// Services by key.
    services: DashMap<ServiceKey, ServiceEndpoints>,
}

impl ServiceRegistry {
    /// Create a new service registry.
    pub fn new() -> Self {
        Self {
            services: DashMap::new(),
        }
    }

    /// Register a service.
    pub fn register(&self, key: ServiceKey, endpoints: ServiceEndpoints) {
        self.services.insert(key, endpoints);
    }

    /// Unregister a service.
    pub fn unregister(&self, key: &ServiceKey) {
        self.services.remove(key);
    }

    /// Get service endpoints.
    pub fn get(&self, key: &ServiceKey) -> Option<ServiceEndpoints> {
        self.services.get(key).map(|e| e.clone())
    }

    /// Update endpoints for a service.
    pub fn update_endpoints(&self, key: &ServiceKey, endpoints: Vec<Endpoint>) {
        if let Some(mut service) = self.services.get_mut(key) {
            service.endpoints = endpoints;
        }
    }

    /// List all services.
    pub fn list_services(&self) -> Vec<ServiceKey> {
        self.services.iter().map(|e| e.key().clone()).collect()
    }

    /// Get service by name.
    pub fn get_by_name(&self, name: &str) -> Option<ServiceEndpoints> {
        self.services
            .iter()
            .find(|e| e.key().name == name)
            .map(|e| e.value().clone())
    }
}

impl Default for ServiceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin() {
        let lb = LoadBalancer::new(LoadBalancerType::RoundRobin);

        let endpoints = vec![
            Endpoint {
                address: "127.0.0.1:8001".parse().unwrap(),
                weight: 1,
                health: EndpointHealth::Healthy,
                metadata: HashMap::new(),
                tls_identity: ServiceIdentity::default(),
            },
            Endpoint {
                address: "127.0.0.1:8002".parse().unwrap(),
                weight: 1,
                health: EndpointHealth::Healthy,
                metadata: HashMap::new(),
                tls_identity: ServiceIdentity::default(),
            },
        ];

        let e1 = lb.select(&endpoints).unwrap();
        let e2 = lb.select(&endpoints).unwrap();

        assert_ne!(e1.address, e2.address);
    }

    fn ring_endpoints(n: usize) -> Vec<Endpoint> {
        (0..n)
            .map(|i| Endpoint {
                address: format!("10.0.0.{}:8080", i + 1).parse().unwrap(),
                weight: 1,
                health: EndpointHealth::Healthy,
                metadata: HashMap::new(),
                tls_identity: ServiceIdentity::default(),
            })
            .collect()
    }

    #[test]
    fn test_ring_empty() {
        let ring = ConsistentHashRing::new(&[]);
        assert_eq!(ring.lookup(&"anything"), None);
        assert_eq!(ring.backend_count(), 0);
    }

    #[test]
    fn test_ring_single_backend_always_selected() {
        let addr: SocketAddr = "10.0.0.1:8080".parse().unwrap();
        let ring = ConsistentHashRing::new(&[addr]);
        for k in 0..100 {
            assert_eq!(ring.lookup(&format!("key-{k}")), Some(addr));
        }
    }

    #[test]
    fn test_ring_stable_mapping() {
        let addrs: Vec<SocketAddr> = ring_endpoints(4).iter().map(|e| e.address).collect();
        let ring = ConsistentHashRing::new(&addrs);
        // Same key -> same backend across repeated lookups.
        for k in 0..50 {
            let key = format!("user-{k}");
            let first = ring.lookup(&key);
            assert!(first.is_some());
            for _ in 0..5 {
                assert_eq!(ring.lookup(&key), first);
            }
        }
    }

    #[test]
    fn test_ring_distributes_keys() {
        let addrs: Vec<SocketAddr> = ring_endpoints(4).iter().map(|e| e.address).collect();
        let ring = ConsistentHashRing::new(&addrs);

        let mut counts: HashMap<SocketAddr, usize> = HashMap::new();
        for k in 0..4000 {
            let a = ring.lookup(&format!("key-{k}")).unwrap();
            *counts.entry(a).or_default() += 1;
        }

        // All backends must receive some traffic (not all-to-one).
        assert_eq!(counts.len(), 4);
        // With 128 vnodes each bucket should be reasonably balanced.
        for c in counts.values() {
            assert!(*c > 500, "bucket too small: {c}");
            assert!(*c < 1500, "bucket too large: {c}");
        }
    }

    #[test]
    fn test_ring_minimal_reassignment_on_removal() {
        let addrs: Vec<SocketAddr> = ring_endpoints(5).iter().map(|e| e.address).collect();
        let ring_before = ConsistentHashRing::new(&addrs);

        // Remove one backend.
        let removed = addrs[2];
        let remaining: Vec<SocketAddr> = addrs.iter().copied().filter(|a| *a != removed).collect();
        let ring_after = ConsistentHashRing::new(&remaining);

        let total = 5000;
        let mut moved = 0;
        for k in 0..total {
            let key = format!("key-{k}");
            let before = ring_before.lookup(&key).unwrap();
            let after = ring_after.lookup(&key).unwrap();
            if before != after {
                moved += 1;
                // Keys only move off the removed node; keys that were on a
                // surviving node must stay put.
                assert_eq!(before, removed, "key {key} moved between survivors");
            }
        }

        // Only ~1/5 of keys should move; allow generous slack.
        let frac = moved as f64 / total as f64;
        assert!(frac < 0.35, "too many keys reassigned: {frac}");
        assert!(frac > 0.05, "suspiciously few keys reassigned: {frac}");
    }

    #[test]
    fn test_load_balancer_ring_hash_stable() {
        let endpoints = ring_endpoints(4);
        let lb = LoadBalancer::new(LoadBalancerType::RingHash);

        let key = "session-abc";
        let first = lb.select_with_key(&endpoints, &key).unwrap().address;
        for _ in 0..20 {
            assert_eq!(lb.select_with_key(&endpoints, &key).unwrap().address, first);
        }
    }
}
