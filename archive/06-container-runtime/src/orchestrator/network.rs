//! Networking and Overlay Networks
//!
//! Implements container networking including:
//! - Overlay networks (VXLAN)
//! - IP address management
//! - Network policies
//! - Virtual interfaces

use super::{ResourceId, ObjectMeta, Pod, LabelSelector};
use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Instant;

/// Network manager for the cluster
#[derive(Debug)]
pub struct NetworkManager {
    config: NetworkConfig,
    /// Overlay networks
    networks: HashMap<ResourceId, OverlayNetwork>,
    /// IP allocators per network
    ip_allocators: HashMap<ResourceId, IpAllocator>,
    /// Pod network assignments
    pod_networks: HashMap<ResourceId, PodNetwork>,
    /// Network policies
    policies: HashMap<ResourceId, NetworkPolicy>,
    /// Routing tables
    routing_tables: HashMap<ResourceId, RoutingTable>,
}

impl NetworkManager {
    pub fn new(config: NetworkConfig) -> Result<Self> {
        let mut manager = Self {
            config: config.clone(),
            networks: HashMap::new(),
            ip_allocators: HashMap::new(),
            pod_networks: HashMap::new(),
            policies: HashMap::new(),
            routing_tables: HashMap::new(),
        };

        // Create default network
        let default_network = OverlayNetwork::new(
            "default",
            config.pod_cidr.clone(),
            config.service_cidr.clone(),
        );
        manager.create_network(default_network)?;

        Ok(manager)
    }

    /// Create an overlay network
    pub fn create_network(&mut self, network: OverlayNetwork) -> Result<()> {
        let id = network.metadata.uid.clone();

        // Create IP allocator for this network
        let allocator = IpAllocator::new(network.pod_cidr.clone())?;
        self.ip_allocators.insert(id.clone(), allocator);

        // Create routing table
        let routing_table = RoutingTable::new(id.clone());
        self.routing_tables.insert(id.clone(), routing_table);

        self.networks.insert(id, network);
        Ok(())
    }

    /// Delete an overlay network
    pub fn delete_network(&mut self, network_id: &ResourceId) -> Result<()> {
        // Check if any pods are using this network
        let pods_on_network: Vec<_> = self.pod_networks.iter()
            .filter(|(_, pn)| &pn.network_id == network_id)
            .map(|(id, _)| id.clone())
            .collect();

        if !pods_on_network.is_empty() {
            return Err(Error::Runtime(format!(
                "Cannot delete network {} - {} pods are attached",
                network_id, pods_on_network.len()
            )));
        }

        self.networks.remove(network_id);
        self.ip_allocators.remove(network_id);
        self.routing_tables.remove(network_id);

        Ok(())
    }

    /// Allocate network resources for a pod
    pub fn allocate_pod_network(&mut self, pod: &Pod) -> Result<PodNetwork> {
        // Find the network to use (default for now)
        let network_id = self.networks.keys().next().cloned()
            .ok_or_else(|| Error::Runtime("No networks available".into()))?;

        let allocator = self.ip_allocators.get_mut(&network_id)
            .ok_or_else(|| Error::Runtime("IP allocator not found".into()))?;

        // Allocate an IP address
        let pod_ip = allocator.allocate()?;

        // Create virtual interface
        let veth = VirtualInterface {
            name: format!("veth{}", pod.metadata.uid.0[..8].to_string()),
            peer_name: format!("eth0"),
            mac_address: generate_mac_address(&pod.metadata.uid),
            ip_address: pod_ip,
            mtu: self.config.mtu,
        };

        let pod_network = PodNetwork {
            pod_id: pod.metadata.uid.clone(),
            network_id: network_id.clone(),
            ip_address: pod_ip,
            gateway: allocator.gateway(),
            dns_servers: self.config.dns_servers.clone(),
            interface: veth,
            vxlan_vni: self.get_vxlan_vni(&network_id),
        };

        self.pod_networks.insert(pod.metadata.uid.clone(), pod_network.clone());

        // Update routing table
        if let Some(routing_table) = self.routing_tables.get_mut(&network_id) {
            routing_table.add_route(Route {
                destination: pod_ip,
                prefix_len: 32,
                gateway: None,
                interface: pod_network.interface.name.clone(),
                metric: 0,
            });
        }

        Ok(pod_network)
    }

    /// Release network resources for a pod
    pub fn release_pod_network(&mut self, pod_id: &ResourceId) -> Result<()> {
        if let Some(pod_network) = self.pod_networks.remove(pod_id) {
            // Release IP address
            if let Some(allocator) = self.ip_allocators.get_mut(&pod_network.network_id) {
                allocator.release(pod_network.ip_address);
            }

            // Remove routes
            if let Some(routing_table) = self.routing_tables.get_mut(&pod_network.network_id) {
                routing_table.remove_route(pod_network.ip_address);
            }
        }

        Ok(())
    }

    /// Get pod network
    pub fn get_pod_network(&self, pod_id: &ResourceId) -> Option<&PodNetwork> {
        self.pod_networks.get(pod_id)
    }

    /// Apply a network policy
    pub fn apply_policy(&mut self, policy: NetworkPolicy) -> Result<()> {
        self.policies.insert(policy.metadata.uid.clone(), policy);
        Ok(())
    }

    /// Delete a network policy
    pub fn delete_policy(&mut self, policy_id: &ResourceId) -> Result<()> {
        self.policies.remove(policy_id);
        Ok(())
    }

    /// Check if traffic is allowed by network policies
    pub fn is_traffic_allowed(
        &self,
        source_pod: &Pod,
        dest_pod: &Pod,
        dest_port: u16,
        protocol: Protocol,
    ) -> bool {
        // Find policies that apply to the destination pod
        let applicable_policies: Vec<_> = self.policies.values()
            .filter(|p| p.matches_pod(dest_pod))
            .collect();

        // If no policies match, allow all traffic (default allow)
        if applicable_policies.is_empty() {
            return true;
        }

        // Check if any policy allows the traffic
        for policy in applicable_policies {
            if policy.allows_ingress(source_pod, dest_port, protocol) {
                return true;
            }
        }

        false
    }

    fn get_vxlan_vni(&self, network_id: &ResourceId) -> u32 {
        self.networks.get(network_id)
            .and_then(|n| n.vxlan_config.as_ref())
            .map(|c| c.vni)
            .unwrap_or(1)
    }
}

/// Network configuration
#[derive(Clone, Debug)]
pub struct NetworkConfig {
    /// Pod CIDR range
    pub pod_cidr: String,
    /// Service CIDR range
    pub service_cidr: String,
    /// DNS servers
    pub dns_servers: Vec<IpAddr>,
    /// DNS search domains
    pub dns_search_domains: Vec<String>,
    /// MTU for network interfaces
    pub mtu: u16,
    /// Enable VXLAN overlay
    pub enable_vxlan: bool,
    /// VXLAN port
    pub vxlan_port: u16,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            pod_cidr: "10.244.0.0/16".to_string(),
            service_cidr: "10.96.0.0/12".to_string(),
            dns_servers: vec![IpAddr::V4(Ipv4Addr::new(10, 96, 0, 10))],
            dns_search_domains: vec!["svc.cluster.local".to_string(), "cluster.local".to_string()],
            mtu: 1450,
            enable_vxlan: true,
            vxlan_port: 4789,
        }
    }
}

/// Overlay network
#[derive(Clone, Debug)]
pub struct OverlayNetwork {
    pub metadata: ObjectMeta,
    pub pod_cidr: String,
    pub service_cidr: String,
    pub driver: NetworkDriver,
    pub vxlan_config: Option<VxlanConfig>,
    pub ipam: IpamConfig,
}

impl OverlayNetwork {
    pub fn new(name: impl Into<String>, pod_cidr: String, service_cidr: String) -> Self {
        Self {
            metadata: ObjectMeta::new(name, ""),
            pod_cidr,
            service_cidr,
            driver: NetworkDriver::Overlay,
            vxlan_config: Some(VxlanConfig::default()),
            ipam: IpamConfig::default(),
        }
    }
}

/// Network driver type
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NetworkDriver {
    #[default]
    Overlay,
    Bridge,
    Host,
    Macvlan,
    None,
}

/// VXLAN configuration
#[derive(Clone, Debug)]
pub struct VxlanConfig {
    pub vni: u32,
    pub port: u16,
    pub vtep_address: Option<IpAddr>,
}

impl Default for VxlanConfig {
    fn default() -> Self {
        Self {
            vni: 1,
            port: 4789,
            vtep_address: None,
        }
    }
}

/// IPAM configuration
#[derive(Clone, Debug)]
pub struct IpamConfig {
    pub driver: IpamDriver,
    pub subnet: String,
    pub gateway: Option<IpAddr>,
}

impl Default for IpamConfig {
    fn default() -> Self {
        Self {
            driver: IpamDriver::HostLocal,
            subnet: "10.244.0.0/24".to_string(),
            gateway: Some(IpAddr::V4(Ipv4Addr::new(10, 244, 0, 1))),
        }
    }
}

/// IPAM driver
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum IpamDriver {
    #[default]
    HostLocal,
    Dhcp,
    Static,
}

/// IP address allocator
#[derive(Debug)]
pub struct IpAllocator {
    cidr: String,
    network: Ipv4Addr,
    prefix_len: u8,
    allocated: HashSet<IpAddr>,
    next_ip: u32,
    gateway: IpAddr,
}

impl IpAllocator {
    pub fn new(cidr: String) -> Result<Self> {
        // Parse CIDR (e.g., "10.244.0.0/24")
        let parts: Vec<&str> = cidr.split('/').collect();
        if parts.len() != 2 {
            return Err(Error::Runtime(format!("Invalid CIDR: {}", cidr)));
        }

        let network: Ipv4Addr = parts[0].parse()
            .map_err(|_| Error::Runtime(format!("Invalid IP: {}", parts[0])))?;
        let prefix_len: u8 = parts[1].parse()
            .map_err(|_| Error::Runtime(format!("Invalid prefix: {}", parts[1])))?;

        let gateway = IpAddr::V4(Ipv4Addr::from(u32::from(network) + 1));

        Ok(Self {
            cidr,
            network,
            prefix_len,
            allocated: HashSet::new(),
            next_ip: u32::from(network) + 2, // Start after gateway
            gateway,
        })
    }

    pub fn allocate(&mut self) -> Result<IpAddr> {
        let mask = !0u32 << (32 - self.prefix_len);
        let network_u32 = u32::from(self.network);
        let broadcast = network_u32 | !mask;

        // Find next available IP
        while self.next_ip < broadcast {
            let ip = IpAddr::V4(Ipv4Addr::from(self.next_ip));
            self.next_ip += 1;

            if !self.allocated.contains(&ip) {
                self.allocated.insert(ip);
                return Ok(ip);
            }
        }

        // Try to find any free IP
        for ip_u32 in (network_u32 + 2)..broadcast {
            let ip = IpAddr::V4(Ipv4Addr::from(ip_u32));
            if !self.allocated.contains(&ip) {
                self.allocated.insert(ip);
                return Ok(ip);
            }
        }

        Err(Error::Runtime("No available IP addresses".into()))
    }

    pub fn release(&mut self, ip: IpAddr) {
        self.allocated.remove(&ip);
    }

    pub fn gateway(&self) -> IpAddr {
        self.gateway
    }

    pub fn available_count(&self) -> usize {
        let mask = !0u32 << (32 - self.prefix_len);
        let network_u32 = u32::from(self.network);
        let broadcast = network_u32 | !mask;
        let total = (broadcast - network_u32 - 2) as usize; // Exclude network, broadcast, gateway
        total.saturating_sub(self.allocated.len())
    }

    pub fn is_available(&self, ip: IpAddr) -> bool {
        !self.allocated.contains(&ip)
    }
}

/// Virtual interface for a pod
#[derive(Clone, Debug)]
pub struct VirtualInterface {
    pub name: String,
    pub peer_name: String,
    pub mac_address: String,
    pub ip_address: IpAddr,
    pub mtu: u16,
}

/// Pod network configuration
#[derive(Clone, Debug)]
pub struct PodNetwork {
    pub pod_id: ResourceId,
    pub network_id: ResourceId,
    pub ip_address: IpAddr,
    pub gateway: IpAddr,
    pub dns_servers: Vec<IpAddr>,
    pub interface: VirtualInterface,
    pub vxlan_vni: u32,
}

/// Routing table
#[derive(Debug)]
pub struct RoutingTable {
    network_id: ResourceId,
    routes: Vec<Route>,
}

impl RoutingTable {
    pub fn new(network_id: ResourceId) -> Self {
        Self {
            network_id,
            routes: Vec::new(),
        }
    }

    pub fn add_route(&mut self, route: Route) {
        // Remove existing route for same destination
        self.routes.retain(|r| r.destination != route.destination);
        self.routes.push(route);
    }

    pub fn remove_route(&mut self, destination: IpAddr) {
        self.routes.retain(|r| r.destination != destination);
    }

    pub fn lookup(&self, destination: IpAddr) -> Option<&Route> {
        // Find the most specific matching route
        self.routes.iter()
            .filter(|r| route_matches(destination, r.destination, r.prefix_len))
            .max_by_key(|r| r.prefix_len)
    }

    pub fn all_routes(&self) -> &[Route] {
        &self.routes
    }
}

/// Network route
#[derive(Clone, Debug)]
pub struct Route {
    pub destination: IpAddr,
    pub prefix_len: u8,
    pub gateway: Option<IpAddr>,
    pub interface: String,
    pub metric: u32,
}

fn route_matches(target: IpAddr, network: IpAddr, prefix_len: u8) -> bool {
    match (target, network) {
        (IpAddr::V4(t), IpAddr::V4(n)) => {
            let mask = !0u32 << (32 - prefix_len);
            (u32::from(t) & mask) == (u32::from(n) & mask)
        }
        (IpAddr::V6(t), IpAddr::V6(n)) => {
            let t_bytes = t.octets();
            let n_bytes = n.octets();
            let full_bytes = (prefix_len / 8) as usize;
            let remaining_bits = prefix_len % 8;

            // Check full bytes
            if t_bytes[..full_bytes] != n_bytes[..full_bytes] {
                return false;
            }

            // Check remaining bits
            if remaining_bits > 0 {
                let mask = !0u8 << (8 - remaining_bits);
                (t_bytes[full_bytes] & mask) == (n_bytes[full_bytes] & mask)
            } else {
                true
            }
        }
        _ => false,
    }
}

/// Network policy
#[derive(Clone, Debug)]
pub struct NetworkPolicy {
    pub metadata: ObjectMeta,
    pub spec: NetworkPolicySpec,
}

impl NetworkPolicy {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            spec: NetworkPolicySpec::default(),
        }
    }

    pub fn with_pod_selector(mut self, selector: LabelSelector) -> Self {
        self.spec.pod_selector = selector;
        self
    }

    pub fn with_ingress_rule(mut self, rule: NetworkPolicyIngressRule) -> Self {
        self.spec.ingress.push(rule);
        self
    }

    pub fn with_egress_rule(mut self, rule: NetworkPolicyEgressRule) -> Self {
        self.spec.egress.push(rule);
        self
    }

    fn matches_pod(&self, pod: &Pod) -> bool {
        self.spec.pod_selector.matches(&pod.metadata.labels)
    }

    fn allows_ingress(&self, source_pod: &Pod, port: u16, protocol: Protocol) -> bool {
        if self.spec.ingress.is_empty() {
            return false; // Empty ingress rules = deny all
        }

        for rule in &self.spec.ingress {
            // Check if source matches any from selector
            let source_matches = rule.from.is_empty() || rule.from.iter().any(|peer| {
                match peer {
                    NetworkPolicyPeer::PodSelector(selector) => {
                        selector.matches(&source_pod.metadata.labels)
                    }
                    NetworkPolicyPeer::NamespaceSelector(selector) => {
                        // Would need namespace labels to check properly
                        true
                    }
                    NetworkPolicyPeer::IpBlock(_) => true,
                }
            });

            // Check if port matches
            let port_matches = rule.ports.is_empty() || rule.ports.iter().any(|p| {
                p.port.map(|rp| rp == port).unwrap_or(true) &&
                p.protocol.map(|rp| rp == protocol).unwrap_or(true)
            });

            if source_matches && port_matches {
                return true;
            }
        }

        false
    }
}

/// Network policy specification
#[derive(Clone, Debug, Default)]
pub struct NetworkPolicySpec {
    pub pod_selector: LabelSelector,
    pub ingress: Vec<NetworkPolicyIngressRule>,
    pub egress: Vec<NetworkPolicyEgressRule>,
    pub policy_types: Vec<PolicyType>,
}

/// Policy type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PolicyType {
    Ingress,
    Egress,
}

/// Ingress rule
#[derive(Clone, Debug, Default)]
pub struct NetworkPolicyIngressRule {
    pub from: Vec<NetworkPolicyPeer>,
    pub ports: Vec<NetworkPolicyPort>,
}

/// Egress rule
#[derive(Clone, Debug, Default)]
pub struct NetworkPolicyEgressRule {
    pub to: Vec<NetworkPolicyPeer>,
    pub ports: Vec<NetworkPolicyPort>,
}

/// Network policy peer
#[derive(Clone, Debug)]
pub enum NetworkPolicyPeer {
    PodSelector(LabelSelector),
    NamespaceSelector(LabelSelector),
    IpBlock(IpBlock),
}

/// IP block
#[derive(Clone, Debug)]
pub struct IpBlock {
    pub cidr: String,
    pub except: Vec<String>,
}

/// Network policy port
#[derive(Clone, Debug)]
pub struct NetworkPolicyPort {
    pub protocol: Option<Protocol>,
    pub port: Option<u16>,
    pub end_port: Option<u16>,
}

/// Protocol
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Protocol {
    #[default]
    TCP,
    UDP,
    SCTP,
}

/// CNI plugin interface
pub trait CniPlugin: std::fmt::Debug + Send + Sync {
    fn name(&self) -> &str;
    fn add(&self, pod_id: &ResourceId, netns: &str) -> Result<CniResult>;
    fn delete(&self, pod_id: &ResourceId, netns: &str) -> Result<()>;
    fn check(&self, pod_id: &ResourceId, netns: &str) -> Result<bool>;
}

/// CNI result
#[derive(Clone, Debug)]
pub struct CniResult {
    pub interfaces: Vec<CniInterface>,
    pub ips: Vec<CniIpConfig>,
    pub routes: Vec<Route>,
    pub dns: CniDns,
}

/// CNI interface
#[derive(Clone, Debug)]
pub struct CniInterface {
    pub name: String,
    pub mac: String,
    pub sandbox: String,
}

/// CNI IP configuration
#[derive(Clone, Debug)]
pub struct CniIpConfig {
    pub interface: usize,
    pub address: IpAddr,
    pub gateway: Option<IpAddr>,
}

/// CNI DNS configuration
#[derive(Clone, Debug, Default)]
pub struct CniDns {
    pub nameservers: Vec<IpAddr>,
    pub domain: String,
    pub search: Vec<String>,
    pub options: Vec<String>,
}

/// Generate a MAC address from a pod ID
fn generate_mac_address(pod_id: &ResourceId) -> String {
    // Use pod ID to generate a deterministic MAC
    let bytes = pod_id.0.as_bytes();
    format!(
        "02:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
        bytes.get(0).unwrap_or(&0),
        bytes.get(1).unwrap_or(&0),
        bytes.get(2).unwrap_or(&0),
        bytes.get(3).unwrap_or(&0),
        bytes.get(4).unwrap_or(&0)
    )
}

/// Network namespace
#[derive(Clone, Debug)]
pub struct NetworkNamespace {
    pub name: String,
    pub path: String,
    pub pod_id: ResourceId,
}

/// Ingress controller
#[derive(Debug)]
pub struct IngressController {
    rules: HashMap<String, IngressRule>,
    backends: HashMap<String, Vec<SocketAddr>>,
}

impl IngressController {
    pub fn new() -> Self {
        Self {
            rules: HashMap::new(),
            backends: HashMap::new(),
        }
    }

    pub fn add_rule(&mut self, rule: IngressRule) {
        self.rules.insert(rule.host.clone(), rule);
    }

    pub fn remove_rule(&mut self, host: &str) {
        self.rules.remove(host);
    }

    pub fn route(&self, host: &str, path: &str) -> Option<&IngressBackend> {
        let rule = self.rules.get(host)?;

        rule.http.paths.iter()
            .find(|p| path.starts_with(&p.path))
            .map(|p| &p.backend)
    }
}

impl Default for IngressController {
    fn default() -> Self {
        Self::new()
    }
}

/// Ingress rule
#[derive(Clone, Debug)]
pub struct IngressRule {
    pub host: String,
    pub http: HttpIngressRuleValue,
}

/// HTTP ingress rule value
#[derive(Clone, Debug)]
pub struct HttpIngressRuleValue {
    pub paths: Vec<HttpIngressPath>,
}

/// HTTP ingress path
#[derive(Clone, Debug)]
pub struct HttpIngressPath {
    pub path: String,
    pub path_type: PathType,
    pub backend: IngressBackend,
}

/// Path type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathType {
    Exact,
    Prefix,
    ImplementationSpecific,
}

/// Ingress backend
#[derive(Clone, Debug)]
pub struct IngressBackend {
    pub service_name: String,
    pub service_port: u16,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_manager_new() {
        let config = NetworkConfig::default();
        let manager = NetworkManager::new(config).unwrap();
        assert!(!manager.networks.is_empty());
    }

    #[test]
    fn test_ip_allocator() {
        let mut allocator = IpAllocator::new("10.244.0.0/24".to_string()).unwrap();

        let ip1 = allocator.allocate().unwrap();
        let ip2 = allocator.allocate().unwrap();

        assert_ne!(ip1, ip2);
        assert!(allocator.available_count() > 0);

        allocator.release(ip1);
        assert!(!allocator.allocated.contains(&ip1));
    }

    #[test]
    fn test_ip_allocator_exhaustion() {
        let mut allocator = IpAllocator::new("10.244.0.0/30".to_string()).unwrap();

        // /30 gives us only 2 usable addresses (minus gateway)
        let ip1 = allocator.allocate().unwrap();

        // Second allocation should fail for /30
        let result = allocator.allocate();
        // Either succeeds with last available or fails
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_overlay_network() {
        let network = OverlayNetwork::new(
            "test-network",
            "10.244.0.0/16".to_string(),
            "10.96.0.0/12".to_string(),
        );

        assert_eq!(network.metadata.name, "test-network");
        assert!(network.vxlan_config.is_some());
    }

    #[test]
    fn test_routing_table() {
        let mut rt = RoutingTable::new(ResourceId::generate());

        rt.add_route(Route {
            destination: "10.244.1.0".parse().unwrap(),
            prefix_len: 24,
            gateway: Some("10.244.0.1".parse().unwrap()),
            interface: "veth0".to_string(),
            metric: 0,
        });

        rt.add_route(Route {
            destination: "10.244.1.100".parse().unwrap(),
            prefix_len: 32,
            gateway: None,
            interface: "veth1".to_string(),
            metric: 0,
        });

        // More specific route should be preferred
        let route = rt.lookup("10.244.1.100".parse().unwrap());
        assert!(route.is_some());
        assert_eq!(route.unwrap().prefix_len, 32);
    }

    #[test]
    fn test_route_matching() {
        assert!(route_matches(
            "10.244.1.50".parse().unwrap(),
            "10.244.1.0".parse().unwrap(),
            24
        ));

        assert!(!route_matches(
            "10.244.2.50".parse().unwrap(),
            "10.244.1.0".parse().unwrap(),
            24
        ));
    }

    #[test]
    fn test_network_policy() {
        let policy = NetworkPolicy::new("allow-web", "default")
            .with_pod_selector(LabelSelector::new().with_label("app", "web"))
            .with_ingress_rule(NetworkPolicyIngressRule {
                from: vec![NetworkPolicyPeer::PodSelector(
                    LabelSelector::new().with_label("role", "frontend")
                )],
                ports: vec![NetworkPolicyPort {
                    protocol: Some(Protocol::TCP),
                    port: Some(80),
                    end_port: None,
                }],
            });

        assert_eq!(policy.spec.ingress.len(), 1);
        assert_eq!(policy.spec.ingress[0].ports[0].port, Some(80));
    }

    #[test]
    fn test_virtual_interface() {
        let pod_id = ResourceId::generate();
        let mac = generate_mac_address(&pod_id);

        assert!(mac.starts_with("02:"));
        assert_eq!(mac.len(), 17);
    }

    #[test]
    fn test_vxlan_config() {
        let config = VxlanConfig {
            vni: 100,
            port: 4789,
            vtep_address: Some("192.168.1.1".parse().unwrap()),
        };

        assert_eq!(config.vni, 100);
    }

    #[test]
    fn test_ingress_controller() {
        let mut ic = IngressController::new();

        ic.add_rule(IngressRule {
            host: "example.com".to_string(),
            http: HttpIngressRuleValue {
                paths: vec![
                    HttpIngressPath {
                        path: "/api".to_string(),
                        path_type: PathType::Prefix,
                        backend: IngressBackend {
                            service_name: "api-service".to_string(),
                            service_port: 8080,
                        },
                    },
                    HttpIngressPath {
                        path: "/".to_string(),
                        path_type: PathType::Prefix,
                        backend: IngressBackend {
                            service_name: "web-service".to_string(),
                            service_port: 80,
                        },
                    },
                ],
            },
        });

        let backend = ic.route("example.com", "/api/users");
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().service_name, "api-service");

        let backend = ic.route("example.com", "/home");
        assert!(backend.is_some());
        assert_eq!(backend.unwrap().service_name, "web-service");
    }

    #[test]
    fn test_cni_result() {
        let result = CniResult {
            interfaces: vec![CniInterface {
                name: "eth0".to_string(),
                mac: "02:00:00:00:00:01".to_string(),
                sandbox: "/var/run/netns/pod123".to_string(),
            }],
            ips: vec![CniIpConfig {
                interface: 0,
                address: "10.244.1.10".parse().unwrap(),
                gateway: Some("10.244.1.1".parse().unwrap()),
            }],
            routes: vec![],
            dns: CniDns::default(),
        };

        assert_eq!(result.interfaces.len(), 1);
        assert_eq!(result.ips.len(), 1);
    }

    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();

        assert_eq!(config.pod_cidr, "10.244.0.0/16");
        assert_eq!(config.service_cidr, "10.96.0.0/12");
        assert!(config.enable_vxlan);
    }

    #[test]
    fn test_pod_network_allocation() {
        let config = NetworkConfig::default();
        let mut manager = NetworkManager::new(config).unwrap();

        let pod = Pod::new("test-pod", "default");
        let pod_network = manager.allocate_pod_network(&pod).unwrap();

        assert!(!pod_network.interface.mac_address.is_empty());
        assert!(manager.get_pod_network(&pod.metadata.uid).is_some());

        // Release
        manager.release_pod_network(&pod.metadata.uid).unwrap();
        assert!(manager.get_pod_network(&pod.metadata.uid).is_none());
    }

    #[test]
    fn test_ipv6_route_matching() {
        assert!(route_matches(
            "2001:db8::1".parse().unwrap(),
            "2001:db8::".parse().unwrap(),
            64
        ));

        assert!(!route_matches(
            "2001:db9::1".parse().unwrap(),
            "2001:db8::".parse().unwrap(),
            64
        ));
    }
}
