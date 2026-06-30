//! Service discovery and load balancing.

use crate::config::ServiceIdentity;
use crate::policy::ServicePolicy;
use dashmap::DashMap;
use std::collections::HashMap;
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
                // Simple hash for now
                let idx = rand::random::<usize>() % healthy.len();
                healthy[idx]
            }
        };

        Some(selected.clone())
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
}
