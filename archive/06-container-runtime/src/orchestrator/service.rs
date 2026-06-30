//! Service Discovery and Load Balancing
//!
//! Services provide stable network endpoints for pods.
//! Supports ClusterIP, NodePort, LoadBalancer, and ExternalName types.

use super::{ObjectMeta, ResourceId, LabelSelector, Pod};
use super::endpoints::Endpoints;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

/// Service for load balancing and service discovery
#[derive(Clone, Debug)]
pub struct Service {
    pub metadata: ObjectMeta,
    pub spec: ServiceSpec,
    pub status: ServiceStatus,
}

impl Service {
    /// Create a new service
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            spec: ServiceSpec::default(),
            status: ServiceStatus::default(),
        }
    }

    /// Create a ClusterIP service
    pub fn cluster_ip(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        let mut svc = Self::new(name, namespace);
        svc.spec.service_type = ServiceType::ClusterIP;
        svc
    }

    /// Create a NodePort service
    pub fn node_port(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        let mut svc = Self::new(name, namespace);
        svc.spec.service_type = ServiceType::NodePort;
        svc
    }

    /// Create a LoadBalancer service
    pub fn load_balancer(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        let mut svc = Self::new(name, namespace);
        svc.spec.service_type = ServiceType::LoadBalancer;
        svc
    }

    /// Add a port to the service
    pub fn with_port(mut self, port: ServicePort) -> Self {
        self.spec.ports.push(port);
        self
    }

    /// Set the selector
    pub fn with_selector(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.spec.selector.match_labels.insert(key.into(), value.into());
        self
    }

    /// Check if the service matches a pod
    pub fn matches_pod(&self, pod: &Pod) -> bool {
        self.spec.selector.matches(&pod.metadata.labels)
    }
}

/// Service specification
#[derive(Clone, Debug)]
pub struct ServiceSpec {
    /// Service type
    pub service_type: ServiceType,
    /// Pod selector
    pub selector: LabelSelector,
    /// Service ports
    pub ports: Vec<ServicePort>,
    /// Cluster IP (None for headless)
    pub cluster_ip: Option<IpAddr>,
    /// Cluster IPs (dual-stack)
    pub cluster_ips: Vec<IpAddr>,
    /// External IPs
    pub external_ips: Vec<IpAddr>,
    /// Session affinity
    pub session_affinity: SessionAffinity,
    /// Session affinity config
    pub session_affinity_config: Option<SessionAffinityConfig>,
    /// Load balancer IP
    pub load_balancer_ip: Option<IpAddr>,
    /// Load balancer source ranges
    pub load_balancer_source_ranges: Vec<String>,
    /// External traffic policy
    pub external_traffic_policy: ExternalTrafficPolicy,
    /// Internal traffic policy
    pub internal_traffic_policy: InternalTrafficPolicy,
    /// Health check node port
    pub health_check_node_port: Option<u16>,
    /// Publish not ready addresses
    pub publish_not_ready_addresses: bool,
    /// IP families
    pub ip_families: Vec<IpFamily>,
    /// IP family policy
    pub ip_family_policy: IpFamilyPolicy,
}

impl Default for ServiceSpec {
    fn default() -> Self {
        Self {
            service_type: ServiceType::ClusterIP,
            selector: LabelSelector::default(),
            ports: Vec::new(),
            cluster_ip: None,
            cluster_ips: Vec::new(),
            external_ips: Vec::new(),
            session_affinity: SessionAffinity::None,
            session_affinity_config: None,
            load_balancer_ip: None,
            load_balancer_source_ranges: Vec::new(),
            external_traffic_policy: ExternalTrafficPolicy::Cluster,
            internal_traffic_policy: InternalTrafficPolicy::Cluster,
            health_check_node_port: None,
            publish_not_ready_addresses: false,
            ip_families: vec![IpFamily::IPv4],
            ip_family_policy: IpFamilyPolicy::SingleStack,
        }
    }
}

/// Service type
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ServiceType {
    #[default]
    ClusterIP,
    NodePort,
    LoadBalancer,
    ExternalName,
}

/// Service port
#[derive(Clone, Debug)]
pub struct ServicePort {
    pub name: String,
    pub protocol: Protocol,
    pub port: u16,
    pub target_port: TargetPort,
    pub node_port: Option<u16>,
    pub app_protocol: Option<String>,
}

impl ServicePort {
    pub fn new(name: impl Into<String>, port: u16) -> Self {
        Self {
            name: name.into(),
            protocol: Protocol::TCP,
            port,
            target_port: TargetPort::Number(port),
            node_port: None,
            app_protocol: None,
        }
    }

    pub fn tcp(name: impl Into<String>, port: u16, target_port: u16) -> Self {
        Self {
            name: name.into(),
            protocol: Protocol::TCP,
            port,
            target_port: TargetPort::Number(target_port),
            node_port: None,
            app_protocol: None,
        }
    }

    pub fn with_node_port(mut self, node_port: u16) -> Self {
        self.node_port = Some(node_port);
        self
    }
}

/// Protocol
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Protocol {
    #[default]
    TCP,
    UDP,
    SCTP,
}

/// Target port
#[derive(Clone, Debug)]
pub enum TargetPort {
    Number(u16),
    Name(String),
}

impl Default for TargetPort {
    fn default() -> Self {
        TargetPort::Number(0)
    }
}

/// Session affinity
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SessionAffinity {
    #[default]
    None,
    ClientIP,
}

/// Session affinity config
#[derive(Clone, Debug)]
pub struct SessionAffinityConfig {
    pub client_ip: Option<ClientIPConfig>,
}

/// Client IP config
#[derive(Clone, Debug)]
pub struct ClientIPConfig {
    pub timeout_seconds: i32,
}

/// External traffic policy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExternalTrafficPolicy {
    #[default]
    Cluster,
    Local,
}

/// Internal traffic policy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InternalTrafficPolicy {
    #[default]
    Cluster,
    Local,
}

/// IP family
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IpFamily {
    #[default]
    IPv4,
    IPv6,
}

/// IP family policy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IpFamilyPolicy {
    #[default]
    SingleStack,
    PreferDualStack,
    RequireDualStack,
}

/// Service status
#[derive(Clone, Debug, Default)]
pub struct ServiceStatus {
    pub load_balancer: LoadBalancerStatus,
    pub conditions: Vec<ServiceCondition>,
}

/// Load balancer status
#[derive(Clone, Debug, Default)]
pub struct LoadBalancerStatus {
    pub ingress: Vec<LoadBalancerIngress>,
}

/// Load balancer ingress
#[derive(Clone, Debug)]
pub struct LoadBalancerIngress {
    pub ip: Option<IpAddr>,
    pub hostname: Option<String>,
    pub ports: Vec<PortStatus>,
}

/// Port status
#[derive(Clone, Debug)]
pub struct PortStatus {
    pub port: u16,
    pub protocol: Protocol,
    pub error: Option<String>,
}

/// Service condition
#[derive(Clone, Debug)]
pub struct ServiceCondition {
    pub condition_type: String,
    pub status: String,
    pub last_transition_time: Instant,
    pub reason: String,
    pub message: String,
}

/// Load balancer for distributing traffic
#[derive(Debug)]
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    backends: Vec<Backend>,
    round_robin_index: AtomicUsize,
    weighted_backends: Vec<(Backend, u32)>,
    sticky_sessions: HashMap<IpAddr, BackendId>,
    health_states: HashMap<BackendId, bool>,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            backends: Vec::new(),
            round_robin_index: AtomicUsize::new(0),
            weighted_backends: Vec::new(),
            sticky_sessions: HashMap::new(),
            health_states: HashMap::new(),
        }
    }

    /// Add a backend
    pub fn add_backend(&mut self, backend: Backend) {
        self.health_states.insert(backend.id.clone(), true);
        self.backends.push(backend);
    }

    /// Remove a backend
    pub fn remove_backend(&mut self, backend_id: &BackendId) {
        self.backends.retain(|b| &b.id != backend_id);
        self.health_states.remove(backend_id);
        self.sticky_sessions.retain(|_, id| id != backend_id);
    }

    /// Get healthy backends
    pub fn healthy_backends(&self) -> Vec<&Backend> {
        self.backends.iter()
            .filter(|b| *self.health_states.get(&b.id).unwrap_or(&false))
            .collect()
    }

    /// Select a backend for a request
    pub fn select(&self, client_ip: Option<IpAddr>) -> Option<&Backend> {
        let healthy = self.healthy_backends();
        if healthy.is_empty() {
            return None;
        }

        match self.strategy {
            LoadBalancingStrategy::RoundRobin => self.round_robin_select(&healthy),
            LoadBalancingStrategy::Random => self.random_select(&healthy),
            LoadBalancingStrategy::LeastConnections => self.least_connections_select(&healthy),
            LoadBalancingStrategy::IPHash => self.ip_hash_select(&healthy, client_ip),
            LoadBalancingStrategy::WeightedRoundRobin => self.weighted_round_robin_select(&healthy),
        }
    }

    fn round_robin_select<'a>(&self, backends: &[&'a Backend]) -> Option<&'a Backend> {
        if backends.is_empty() {
            return None;
        }
        let idx = self.round_robin_index.fetch_add(1, Ordering::Relaxed) % backends.len();
        Some(backends[idx])
    }

    fn random_select<'a>(&self, backends: &[&'a Backend]) -> Option<&'a Backend> {
        use std::time::{SystemTime, UNIX_EPOCH};
        if backends.is_empty() {
            return None;
        }
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as usize)
            .unwrap_or(0);
        Some(backends[seed % backends.len()])
    }

    fn least_connections_select<'a>(&self, backends: &[&'a Backend]) -> Option<&'a Backend> {
        backends.iter().min_by_key(|b| b.active_connections).copied()
    }

    fn ip_hash_select<'a>(&self, backends: &[&'a Backend], client_ip: Option<IpAddr>) -> Option<&'a Backend> {
        let ip = client_ip?;
        let hash = match ip {
            IpAddr::V4(v4) => {
                let octets = v4.octets();
                (octets[0] as usize) << 24 | (octets[1] as usize) << 16 |
                (octets[2] as usize) << 8 | (octets[3] as usize)
            }
            IpAddr::V6(v6) => {
                let segments = v6.segments();
                segments.iter().fold(0usize, |acc, &s| acc.wrapping_add(s as usize))
            }
        };
        if backends.is_empty() {
            return None;
        }
        Some(backends[hash % backends.len()])
    }

    fn weighted_round_robin_select<'a>(&self, backends: &[&'a Backend]) -> Option<&'a Backend> {
        // Simple weighted selection based on weight
        let total_weight: u32 = backends.iter().map(|b| b.weight).sum();
        if total_weight == 0 {
            return self.round_robin_select(backends);
        }

        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(0);

        let target = seed % total_weight;
        let mut cumulative = 0;

        for backend in backends {
            cumulative += backend.weight;
            if target < cumulative {
                return Some(*backend);
            }
        }

        backends.last().copied()
    }

    /// Update health state of a backend
    pub fn set_health(&mut self, backend_id: &BackendId, healthy: bool) {
        self.health_states.insert(backend_id.clone(), healthy);
    }

    /// Update active connections for a backend
    pub fn update_connections(&mut self, backend_id: &BackendId, count: u32) {
        for backend in &mut self.backends {
            if &backend.id == backend_id {
                backend.active_connections = count;
                break;
            }
        }
    }
}

/// Load balancing strategy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    #[default]
    RoundRobin,
    Random,
    LeastConnections,
    IPHash,
    WeightedRoundRobin,
}

/// Backend server
#[derive(Clone, Debug)]
pub struct Backend {
    pub id: BackendId,
    pub address: SocketAddr,
    pub weight: u32,
    pub active_connections: u32,
    pub max_connections: Option<u32>,
}

impl Backend {
    pub fn new(id: impl Into<String>, address: SocketAddr) -> Self {
        Self {
            id: BackendId(id.into()),
            address,
            weight: 1,
            active_connections: 0,
            max_connections: None,
        }
    }

    pub fn with_weight(mut self, weight: u32) -> Self {
        self.weight = weight;
        self
    }
}

/// Backend identifier
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BackendId(pub String);

/// Service discovery
#[derive(Debug)]
pub struct ServiceDiscovery {
    services: HashMap<ServiceKey, Service>,
    endpoints: HashMap<ServiceKey, Endpoints>,
    watchers: Vec<Box<dyn Fn(&ServiceEvent) + Send + Sync>>,
}

impl ServiceDiscovery {
    pub fn new() -> Self {
        Self {
            services: HashMap::new(),
            endpoints: HashMap::new(),
            watchers: Vec::new(),
        }
    }

    /// Register a service
    pub fn register(&mut self, service: Service) -> Result<()> {
        let key = ServiceKey {
            namespace: service.metadata.namespace.clone(),
            name: service.metadata.name.clone(),
        };

        let event = ServiceEvent::Added(service.clone());
        self.services.insert(key, service);
        self.notify_watchers(&event);

        Ok(())
    }

    /// Deregister a service
    pub fn deregister(&mut self, namespace: &str, name: &str) -> Result<()> {
        let key = ServiceKey {
            namespace: namespace.to_string(),
            name: name.to_string(),
        };

        if let Some(service) = self.services.remove(&key) {
            let event = ServiceEvent::Deleted(service);
            self.notify_watchers(&event);
            self.endpoints.remove(&key);
        }

        Ok(())
    }

    /// Update endpoints for a service
    pub fn update_endpoints(&mut self, namespace: &str, name: &str, endpoints: Endpoints) -> Result<()> {
        let key = ServiceKey {
            namespace: namespace.to_string(),
            name: name.to_string(),
        };

        self.endpoints.insert(key, endpoints);
        Ok(())
    }

    /// Get a service by name
    pub fn get(&self, namespace: &str, name: &str) -> Option<&Service> {
        let key = ServiceKey {
            namespace: namespace.to_string(),
            name: name.to_string(),
        };
        self.services.get(&key)
    }

    /// Get endpoints for a service
    pub fn get_endpoints(&self, namespace: &str, name: &str) -> Option<&Endpoints> {
        let key = ServiceKey {
            namespace: namespace.to_string(),
            name: name.to_string(),
        };
        self.endpoints.get(&key)
    }

    /// List all services in a namespace
    pub fn list(&self, namespace: &str) -> Vec<&Service> {
        self.services.iter()
            .filter(|(k, _)| k.namespace == namespace)
            .map(|(_, v)| v)
            .collect()
    }

    /// Resolve service to endpoints
    pub fn resolve(&self, namespace: &str, name: &str, port_name: Option<&str>) -> Vec<SocketAddr> {
        let Some(endpoints) = self.get_endpoints(namespace, name) else {
            return Vec::new();
        };

        endpoints.subsets.iter()
            .flat_map(|subset| {
                let ports: Vec<u16> = subset.ports.iter()
                    .filter(|p| port_name.map(|n| p.name == n).unwrap_or(true))
                    .map(|p| p.port)
                    .collect();

                subset.addresses.iter()
                    .flat_map(|addr| {
                        ports.iter().map(move |&port| {
                            SocketAddr::new(addr.ip, port)
                        })
                    })
            })
            .collect()
    }

    fn notify_watchers(&self, event: &ServiceEvent) {
        for watcher in &self.watchers {
            watcher(event);
        }
    }
}

impl Default for ServiceDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

/// Service key for lookups
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ServiceKey {
    pub namespace: String,
    pub name: String,
}

/// Service event
#[derive(Clone, Debug)]
pub enum ServiceEvent {
    Added(Service),
    Modified(Service),
    Deleted(Service),
}

/// DNS-based service discovery
#[derive(Debug)]
pub struct DNSServiceDiscovery {
    records: HashMap<String, Vec<DNSRecord>>,
    cluster_domain: String,
}

impl DNSServiceDiscovery {
    pub fn new(cluster_domain: impl Into<String>) -> Self {
        Self {
            records: HashMap::new(),
            cluster_domain: cluster_domain.into(),
        }
    }

    /// Add a DNS record for a service
    pub fn add_service(&mut self, service: &Service, endpoints: &Endpoints) {
        let fqdn = format!("{}.{}.svc.{}",
            service.metadata.name,
            service.metadata.namespace,
            self.cluster_domain
        );

        let mut records = Vec::new();

        // Add A records for cluster IP
        if let Some(cluster_ip) = service.spec.cluster_ip {
            records.push(DNSRecord::A {
                name: fqdn.clone(),
                ip: cluster_ip,
                ttl: 30,
            });
        }

        // Add SRV records for ports
        for port in &service.spec.ports {
            for subset in &endpoints.subsets {
                for addr in &subset.addresses {
                    records.push(DNSRecord::SRV {
                        name: format!("_{}._{}.{}", port.name, port.protocol.to_string().to_lowercase(), fqdn),
                        target: addr.hostname.clone().unwrap_or_else(|| format!("{}", addr.ip)),
                        port: port.port,
                        weight: 1,
                        priority: 0,
                        ttl: 30,
                    });
                }
            }
        }

        self.records.insert(fqdn, records);
    }

    /// Lookup DNS records
    pub fn lookup(&self, name: &str) -> Option<&Vec<DNSRecord>> {
        self.records.get(name)
    }

    /// Resolve A records
    pub fn resolve_a(&self, name: &str) -> Vec<IpAddr> {
        self.records.get(name)
            .map(|records| {
                records.iter()
                    .filter_map(|r| {
                        if let DNSRecord::A { ip, .. } = r {
                            Some(*ip)
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// DNS record types
#[derive(Clone, Debug)]
pub enum DNSRecord {
    A { name: String, ip: IpAddr, ttl: u32 },
    AAAA { name: String, ip: IpAddr, ttl: u32 },
    SRV { name: String, target: String, port: u16, weight: u16, priority: u16, ttl: u32 },
    CNAME { name: String, target: String, ttl: u32 },
    TXT { name: String, text: String, ttl: u32 },
}

impl std::fmt::Display for Protocol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Protocol::TCP => write!(f, "TCP"),
            Protocol::UDP => write!(f, "UDP"),
            Protocol::SCTP => write!(f, "SCTP"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_service_new() {
        let svc = Service::new("my-service", "default");
        assert_eq!(svc.metadata.name, "my-service");
        assert_eq!(svc.spec.service_type, ServiceType::ClusterIP);
    }

    #[test]
    fn test_service_types() {
        let cluster_ip = Service::cluster_ip("svc1", "default");
        assert_eq!(cluster_ip.spec.service_type, ServiceType::ClusterIP);

        let node_port = Service::node_port("svc2", "default");
        assert_eq!(node_port.spec.service_type, ServiceType::NodePort);

        let lb = Service::load_balancer("svc3", "default");
        assert_eq!(lb.spec.service_type, ServiceType::LoadBalancer);
    }

    #[test]
    fn test_service_builder() {
        let svc = Service::new("web", "default")
            .with_port(ServicePort::tcp("http", 80, 8080))
            .with_selector("app", "nginx");

        assert_eq!(svc.spec.ports.len(), 1);
        assert_eq!(svc.spec.ports[0].port, 80);
        assert!(svc.spec.selector.match_labels.contains_key("app"));
    }

    #[test]
    fn test_load_balancer_round_robin() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        lb.add_backend(Backend::new("b1", "10.0.0.1:8080".parse().unwrap()));
        lb.add_backend(Backend::new("b2", "10.0.0.2:8080".parse().unwrap()));
        lb.add_backend(Backend::new("b3", "10.0.0.3:8080".parse().unwrap()));

        // Should cycle through backends
        let addrs: Vec<_> = (0..6).filter_map(|_| lb.select(None).map(|b| b.address)).collect();
        assert_eq!(addrs.len(), 6);
    }

    #[test]
    fn test_load_balancer_health() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);

        let b1 = Backend::new("b1", "10.0.0.1:8080".parse().unwrap());
        let b2 = Backend::new("b2", "10.0.0.2:8080".parse().unwrap());

        lb.add_backend(b1);
        lb.add_backend(b2);

        // Both healthy
        assert_eq!(lb.healthy_backends().len(), 2);

        // Mark one unhealthy
        lb.set_health(&BackendId("b1".into()), false);
        assert_eq!(lb.healthy_backends().len(), 1);
    }

    #[test]
    fn test_load_balancer_ip_hash() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::IPHash);

        lb.add_backend(Backend::new("b1", "10.0.0.1:8080".parse().unwrap()));
        lb.add_backend(Backend::new("b2", "10.0.0.2:8080".parse().unwrap()));

        let client_ip: IpAddr = "192.168.1.100".parse().unwrap();

        // Same client should get same backend
        let b1 = lb.select(Some(client_ip)).map(|b| b.id.clone());
        let b2 = lb.select(Some(client_ip)).map(|b| b.id.clone());
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_load_balancer_least_connections() {
        let mut lb = LoadBalancer::new(LoadBalancingStrategy::LeastConnections);

        let mut b1 = Backend::new("b1", "10.0.0.1:8080".parse().unwrap());
        b1.active_connections = 10;

        let mut b2 = Backend::new("b2", "10.0.0.2:8080".parse().unwrap());
        b2.active_connections = 5;

        lb.add_backend(b1);
        lb.add_backend(b2);

        // Should select backend with fewer connections
        let selected = lb.select(None).unwrap();
        assert_eq!(selected.id.0, "b2");
    }

    #[test]
    fn test_service_discovery() {
        let mut sd = ServiceDiscovery::new();

        let svc = Service::new("my-svc", "default")
            .with_port(ServicePort::tcp("http", 80, 8080));

        sd.register(svc).unwrap();

        let found = sd.get("default", "my-svc");
        assert!(found.is_some());
        assert_eq!(found.unwrap().metadata.name, "my-svc");

        let list = sd.list("default");
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_service_discovery_resolve() {
        let mut sd = ServiceDiscovery::new();

        let svc = Service::new("web", "default");
        sd.register(svc).unwrap();

        let endpoints = Endpoints {
            metadata: ObjectMeta::new("web", "default"),
            subsets: vec![
                super::super::endpoints::EndpointSubset {
                    addresses: vec![
                        super::super::endpoints::EndpointAddress {
                            ip: "10.0.0.1".parse().unwrap(),
                            hostname: None,
                            node_name: None,
                            target_ref: None,
                        },
                        super::super::endpoints::EndpointAddress {
                            ip: "10.0.0.2".parse().unwrap(),
                            hostname: None,
                            node_name: None,
                            target_ref: None,
                        },
                    ],
                    not_ready_addresses: vec![],
                    ports: vec![
                        super::super::endpoints::EndpointPort {
                            name: "http".into(),
                            port: 8080,
                            protocol: Protocol::TCP,
                            app_protocol: None,
                        },
                    ],
                },
            ],
        };

        sd.update_endpoints("default", "web", endpoints).unwrap();

        let resolved = sd.resolve("default", "web", None);
        assert_eq!(resolved.len(), 2);
    }

    #[test]
    fn test_dns_service_discovery() {
        let mut dns = DNSServiceDiscovery::new("cluster.local");

        let mut svc = Service::new("api", "production");
        svc.spec.cluster_ip = Some(IpAddr::V4(Ipv4Addr::new(10, 96, 0, 1)));

        let endpoints = Endpoints {
            metadata: ObjectMeta::new("api", "production"),
            subsets: vec![],
        };

        dns.add_service(&svc, &endpoints);

        let resolved = dns.resolve_a("api.production.svc.cluster.local");
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0], IpAddr::V4(Ipv4Addr::new(10, 96, 0, 1)));
    }

    #[test]
    fn test_backend_weight() {
        let b = Backend::new("b1", "10.0.0.1:8080".parse().unwrap())
            .with_weight(10);
        assert_eq!(b.weight, 10);
    }

    #[test]
    fn test_service_port_node_port() {
        let port = ServicePort::tcp("http", 80, 8080)
            .with_node_port(30080);
        assert_eq!(port.node_port, Some(30080));
    }
}
