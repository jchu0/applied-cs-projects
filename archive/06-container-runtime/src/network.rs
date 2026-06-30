//! Container Network Setup
//!
//! Provides network namespace configuration for containers including:
//! - Virtual ethernet (veth) pair creation
//! - Bridge network setup
//! - IP address management
//! - Network namespace configuration

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};
use std::net::Ipv4Addr;
use std::path::{Path, PathBuf};
use std::process::Command;

use ipnetwork::Ipv4Network;
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::{ContainerId, Error, Result};

/// Network mode for containers
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkMode {
    /// No networking
    None,
    /// Host network namespace
    Host,
    /// Bridge network (default)
    Bridge,
    /// Container network (share with another container)
    Container(String),
    /// Custom network
    Custom(String),
}

impl Default for NetworkMode {
    fn default() -> Self {
        NetworkMode::Bridge
    }
}

/// Network configuration for a container
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network mode
    pub mode: NetworkMode,
    /// Bridge name (for bridge mode)
    pub bridge: String,
    /// Subnet for the bridge
    pub subnet: Ipv4Network,
    /// Gateway IP
    pub gateway: Ipv4Addr,
    /// DNS servers
    pub dns: Vec<Ipv4Addr>,
    /// Port mappings (host_port -> container_port)
    pub port_mappings: Vec<PortMapping>,
    /// MTU
    pub mtu: u32,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            mode: NetworkMode::Bridge,
            bridge: "docklet0".to_string(),
            subnet: "172.17.0.0/16".parse().unwrap(),
            gateway: "172.17.0.1".parse().unwrap(),
            dns: vec!["8.8.8.8".parse().unwrap(), "8.8.4.4".parse().unwrap()],
            port_mappings: Vec::new(),
            mtu: 1500,
        }
    }
}

/// Port mapping configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PortMapping {
    /// Host port
    pub host_port: u16,
    /// Container port
    pub container_port: u16,
    /// Protocol (tcp/udp)
    pub protocol: String,
    /// Host IP to bind (optional)
    pub host_ip: Option<Ipv4Addr>,
}

/// Network endpoint for a container
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkEndpoint {
    /// Container ID
    pub container_id: ContainerId,
    /// Veth pair host side name
    pub veth_host: String,
    /// Veth pair container side name
    pub veth_container: String,
    /// Assigned IP address
    pub ip_address: Ipv4Addr,
    /// MAC address
    pub mac_address: String,
    /// Bridge name
    pub bridge: String,
}

/// IP Address Manager for allocating container IPs
#[derive(Debug)]
pub struct IpAddressManager {
    /// Subnet to allocate from
    subnet: Ipv4Network,
    /// Gateway (reserved)
    gateway: Ipv4Addr,
    /// Allocated addresses
    allocated: HashMap<Ipv4Addr, ContainerId>,
    /// Next IP to try
    next_ip: u32,
}

impl IpAddressManager {
    /// Create a new IPAM
    pub fn new(subnet: Ipv4Network, gateway: Ipv4Addr) -> Self {
        let network_addr: u32 = subnet.network().into();
        Self {
            subnet,
            gateway,
            allocated: HashMap::new(),
            next_ip: network_addr + 2, // Skip network and gateway
        }
    }

    /// Allocate an IP address for a container
    pub fn allocate(&mut self, container_id: &ContainerId) -> Result<Ipv4Addr> {
        let broadcast: u32 = self.subnet.broadcast().into();

        // Try to find an available IP
        let mut attempts = 0;
        let max_attempts = self.subnet.size() as usize;

        while attempts < max_attempts {
            let ip = Ipv4Addr::from(self.next_ip);

            // Advance next_ip
            self.next_ip += 1;
            if self.next_ip >= broadcast {
                let network_addr: u32 = self.subnet.network().into();
                self.next_ip = network_addr + 2;
            }

            // Skip gateway
            if ip == self.gateway {
                attempts += 1;
                continue;
            }

            // Check if already allocated
            if self.allocated.contains_key(&ip) {
                attempts += 1;
                continue;
            }

            // Allocate
            self.allocated.insert(ip, container_id.clone());
            return Ok(ip);
        }

        Err(Error::Network("No available IP addresses".to_string()))
    }

    /// Release an IP address
    pub fn release(&mut self, ip: &Ipv4Addr) {
        self.allocated.remove(ip);
    }

    /// Check if an IP is allocated
    pub fn is_allocated(&self, ip: &Ipv4Addr) -> bool {
        self.allocated.contains_key(ip)
    }
}

/// Network manager for container networking
#[derive(Debug)]
pub struct NetworkManager {
    /// Network configuration
    config: NetworkConfig,
    /// IP address manager
    ipam: IpAddressManager,
    /// Container endpoints
    endpoints: HashMap<String, NetworkEndpoint>,
    /// Runtime directory for network state
    state_dir: PathBuf,
}

impl NetworkManager {
    /// Create a new network manager
    pub fn new(config: NetworkConfig, state_dir: impl Into<PathBuf>) -> Self {
        let ipam = IpAddressManager::new(config.subnet, config.gateway);
        Self {
            config,
            ipam,
            endpoints: HashMap::new(),
            state_dir: state_dir.into(),
        }
    }

    /// Initialize the bridge network
    pub fn init_bridge(&self) -> Result<()> {
        let bridge = &self.config.bridge;

        // Check if bridge exists
        if self.bridge_exists(bridge) {
            log::debug!("Bridge {} already exists", bridge);
            return Ok(());
        }

        log::info!("Creating bridge: {}", bridge);

        // Create bridge using ip command
        self.run_ip(&["link", "add", "name", bridge, "type", "bridge"])?;

        // Set bridge up
        self.run_ip(&["link", "set", bridge, "up"])?;

        // Assign IP to bridge
        let gateway_cidr = format!("{}/{}", self.config.gateway, self.config.subnet.prefix());
        self.run_ip(&["addr", "add", &gateway_cidr, "dev", bridge])?;

        // Enable IP forwarding
        self.enable_ip_forwarding()?;

        // Setup NAT for outbound traffic
        self.setup_nat()?;

        Ok(())
    }

    /// Check if a bridge exists
    fn bridge_exists(&self, name: &str) -> bool {
        Path::new(&format!("/sys/class/net/{}", name)).exists()
    }

    /// Run ip command
    fn run_ip(&self, args: &[&str]) -> Result<()> {
        let output = Command::new("ip")
            .args(args)
            .output()
            .map_err(|e| Error::Network(format!("Failed to run ip command: {}", e)))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Ignore "already exists" errors
            if !stderr.contains("File exists") {
                return Err(Error::Network(format!(
                    "ip {} failed: {}",
                    args.join(" "),
                    stderr
                )));
            }
        }

        Ok(())
    }

    /// Enable IP forwarding
    fn enable_ip_forwarding(&self) -> Result<()> {
        fs::write("/proc/sys/net/ipv4/ip_forward", "1")
            .map_err(|e| Error::Network(format!("Failed to enable IP forwarding: {}", e)))?;
        Ok(())
    }

    /// Setup NAT using iptables
    fn setup_nat(&self) -> Result<()> {
        let subnet = self.config.subnet.to_string();

        // Add MASQUERADE rule for outbound traffic
        let output = Command::new("iptables")
            .args([
                "-t",
                "nat",
                "-C",
                "POSTROUTING",
                "-s",
                &subnet,
                "!",
                "-o",
                &self.config.bridge,
                "-j",
                "MASQUERADE",
            ])
            .output();

        // If rule doesn't exist, add it
        if output.map(|o| !o.status.success()).unwrap_or(true) {
            Command::new("iptables")
                .args([
                    "-t",
                    "nat",
                    "-A",
                    "POSTROUTING",
                    "-s",
                    &subnet,
                    "!",
                    "-o",
                    &self.config.bridge,
                    "-j",
                    "MASQUERADE",
                ])
                .output()
                .map_err(|e| Error::Network(format!("Failed to setup NAT: {}", e)))?;
        }

        Ok(())
    }

    /// Generate a random MAC address
    fn generate_mac(&self) -> String {
        let mut rng = rand::thread_rng();
        format!(
            "02:42:{:02x}:{:02x}:{:02x}:{:02x}",
            rng.gen::<u8>(),
            rng.gen::<u8>(),
            rng.gen::<u8>(),
            rng.gen::<u8>()
        )
    }

    /// Create network endpoint for a container
    pub fn create_endpoint(&mut self, container_id: &ContainerId) -> Result<NetworkEndpoint> {
        // Allocate IP
        let ip_address = self.ipam.allocate(container_id)?;

        // Generate names
        let short_id = &container_id.0[..12.min(container_id.0.len())];
        let veth_host = format!("veth{}", &short_id[..8]);
        let veth_container = "eth0".to_string();

        // Generate MAC
        let mac_address = self.generate_mac();

        let endpoint = NetworkEndpoint {
            container_id: container_id.clone(),
            veth_host: veth_host.clone(),
            veth_container,
            ip_address,
            mac_address,
            bridge: self.config.bridge.clone(),
        };

        // Create veth pair
        self.run_ip(&[
            "link",
            "add",
            &veth_host,
            "type",
            "veth",
            "peer",
            "name",
            &format!("veth_{}", short_id),
        ])?;

        // Attach host side to bridge
        self.run_ip(&["link", "set", &veth_host, "master", &self.config.bridge])?;

        // Set host side up
        self.run_ip(&["link", "set", &veth_host, "up"])?;

        // Store endpoint
        self.endpoints.insert(container_id.0.clone(), endpoint.clone());

        // Save state
        self.save_endpoint(&endpoint)?;

        Ok(endpoint)
    }

    /// Configure network inside container namespace
    pub fn configure_container_network(
        &self,
        endpoint: &NetworkEndpoint,
        netns_path: &Path,
    ) -> Result<()> {
        let short_id = &endpoint.container_id.0[..12.min(endpoint.container_id.0.len())];
        let peer_name = format!("veth_{}", short_id);

        // Move peer into container namespace
        self.run_ip(&[
            "link",
            "set",
            &peer_name,
            "netns",
            netns_path.to_str().unwrap(),
        ])?;

        // Commands to run inside the namespace
        let ip_cidr = format!("{}/{}", endpoint.ip_address, self.config.subnet.prefix());

        // Use nsenter to configure inside namespace
        let netns_name = netns_path.file_name().unwrap().to_str().unwrap();

        // Rename interface to eth0
        Command::new("ip")
            .args(["netns", "exec", netns_name, "ip", "link", "set", &peer_name, "name", "eth0"])
            .output()
            .map_err(|e| Error::Network(format!("Failed to rename interface: {}", e)))?;

        // Set MAC address
        Command::new("ip")
            .args([
                "netns",
                "exec",
                netns_name,
                "ip",
                "link",
                "set",
                "eth0",
                "address",
                &endpoint.mac_address,
            ])
            .output()
            .map_err(|e| Error::Network(format!("Failed to set MAC: {}", e)))?;

        // Set IP address
        Command::new("ip")
            .args([
                "netns",
                "exec",
                netns_name,
                "ip",
                "addr",
                "add",
                &ip_cidr,
                "dev",
                "eth0",
            ])
            .output()
            .map_err(|e| Error::Network(format!("Failed to set IP: {}", e)))?;

        // Bring up interface
        Command::new("ip")
            .args(["netns", "exec", netns_name, "ip", "link", "set", "eth0", "up"])
            .output()
            .map_err(|e| Error::Network(format!("Failed to bring up eth0: {}", e)))?;

        // Bring up loopback
        Command::new("ip")
            .args(["netns", "exec", netns_name, "ip", "link", "set", "lo", "up"])
            .output()
            .map_err(|e| Error::Network(format!("Failed to bring up lo: {}", e)))?;

        // Add default route
        let gateway = self.config.gateway.to_string();
        Command::new("ip")
            .args([
                "netns",
                "exec",
                netns_name,
                "ip",
                "route",
                "add",
                "default",
                "via",
                &gateway,
            ])
            .output()
            .map_err(|e| Error::Network(format!("Failed to add default route: {}", e)))?;

        Ok(())
    }

    /// Remove network endpoint
    pub fn remove_endpoint(&mut self, container_id: &ContainerId) -> Result<()> {
        if let Some(endpoint) = self.endpoints.remove(&container_id.0) {
            // Delete veth pair (deleting one side removes both)
            let _ = self.run_ip(&["link", "del", &endpoint.veth_host]);

            // Release IP
            self.ipam.release(&endpoint.ip_address);

            // Remove state file
            let state_file = self.state_dir.join(format!("{}.json", container_id.0));
            let _ = fs::remove_file(state_file);
        }

        Ok(())
    }

    /// Save endpoint state to disk
    fn save_endpoint(&self, endpoint: &NetworkEndpoint) -> Result<()> {
        fs::create_dir_all(&self.state_dir)?;
        let state_file = self.state_dir.join(format!("{}.json", endpoint.container_id.0));
        let json = serde_json::to_string_pretty(endpoint)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        fs::write(state_file, json)?;
        Ok(())
    }

    /// Load endpoint state from disk
    pub fn load_endpoint(&self, container_id: &ContainerId) -> Result<NetworkEndpoint> {
        let state_file = self.state_dir.join(format!("{}.json", container_id.0));
        let json = fs::read_to_string(&state_file)?;
        let endpoint: NetworkEndpoint = serde_json::from_str(&json)
            .map_err(|e| Error::Serialization(e.to_string()))?;
        Ok(endpoint)
    }

    /// Setup port forwarding
    pub fn setup_port_forward(&self, endpoint: &NetworkEndpoint, mapping: &PortMapping) -> Result<()> {
        let container_ip = endpoint.ip_address.to_string();
        let host_port = mapping.host_port.to_string();
        let container_port = mapping.container_port.to_string();

        // DNAT rule for incoming traffic
        Command::new("iptables")
            .args([
                "-t",
                "nat",
                "-A",
                "PREROUTING",
                "-p",
                &mapping.protocol,
                "--dport",
                &host_port,
                "-j",
                "DNAT",
                "--to-destination",
                &format!("{}:{}", container_ip, container_port),
            ])
            .output()
            .map_err(|e| Error::Network(format!("Failed to setup port forward: {}", e)))?;

        // Also add rule for local traffic
        Command::new("iptables")
            .args([
                "-t",
                "nat",
                "-A",
                "OUTPUT",
                "-p",
                &mapping.protocol,
                "--dport",
                &host_port,
                "-j",
                "DNAT",
                "--to-destination",
                &format!("{}:{}", container_ip, container_port),
            ])
            .output()
            .map_err(|e| Error::Network(format!("Failed to setup local port forward: {}", e)))?;

        Ok(())
    }

    /// Generate resolv.conf content
    pub fn generate_resolv_conf(&self) -> String {
        let mut content = String::new();
        for dns in &self.config.dns {
            content.push_str(&format!("nameserver {}\n", dns));
        }
        content
    }

    /// Get network configuration
    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }
}

/// Cleanup network resources
pub fn cleanup_bridge(bridge: &str) -> Result<()> {
    // Delete bridge
    Command::new("ip")
        .args(["link", "del", bridge])
        .output()
        .map_err(|e| Error::Network(format!("Failed to delete bridge: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network_config_default() {
        let config = NetworkConfig::default();
        assert_eq!(config.bridge, "docklet0");
        assert_eq!(config.gateway, Ipv4Addr::new(172, 17, 0, 1));
        assert_eq!(config.mtu, 1500);
    }

    #[test]
    fn test_ipam_allocate() {
        let subnet = "172.17.0.0/24".parse().unwrap();
        let gateway = "172.17.0.1".parse().unwrap();
        let mut ipam = IpAddressManager::new(subnet, gateway);

        let container1 = ContainerId::from_string("container1");
        let ip1 = ipam.allocate(&container1).unwrap();
        assert_eq!(ip1, Ipv4Addr::new(172, 17, 0, 2));

        let container2 = ContainerId::from_string("container2");
        let ip2 = ipam.allocate(&container2).unwrap();
        assert_eq!(ip2, Ipv4Addr::new(172, 17, 0, 3));

        // Release and reallocate
        ipam.release(&ip1);
        let container3 = ContainerId::from_string("container3");
        let ip3 = ipam.allocate(&container3).unwrap();
        assert_eq!(ip3, Ipv4Addr::new(172, 17, 0, 4)); // Next in sequence
    }

    #[test]
    fn test_generate_mac() {
        let config = NetworkConfig::default();
        let manager = NetworkManager::new(config, "/tmp/test");
        let mac = manager.generate_mac();

        // Should start with locally administered unicast prefix
        assert!(mac.starts_with("02:42:"));
        assert_eq!(mac.len(), 17);
    }

    #[test]
    fn test_network_mode() {
        assert_eq!(NetworkMode::default(), NetworkMode::Bridge);

        let mode = NetworkMode::Container("abc123".to_string());
        if let NetworkMode::Container(id) = mode {
            assert_eq!(id, "abc123");
        } else {
            panic!("Expected Container mode");
        }
    }

    #[test]
    fn test_port_mapping() {
        let mapping = PortMapping {
            host_port: 8080,
            container_port: 80,
            protocol: "tcp".to_string(),
            host_ip: None,
        };

        assert_eq!(mapping.host_port, 8080);
        assert_eq!(mapping.container_port, 80);
    }
}
