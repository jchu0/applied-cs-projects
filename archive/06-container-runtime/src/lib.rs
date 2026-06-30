//! Docklet - A minimal container runtime and orchestrator
//!
//! This crate provides:
//! - A Docker-lite container runtime that demonstrates Linux kernel isolation
//!   primitives: namespaces, cgroups, and overlayfs.
//! - A Kubernetes-lite container orchestrator for managing containers across
//!   a cluster with scheduling, service discovery, and health checking.

pub mod container;
pub mod namespace;
pub mod cgroup;
pub mod image;
pub mod spec;
pub mod error;
pub mod runtime;
pub mod process;
pub mod overlay;
pub mod orchestrator;
pub mod registry;
pub mod network;

pub use container::{Container, ContainerConfig, ContainerState};
pub use error::{Error, Result};
pub use spec::Spec;
pub use runtime::Runtime;
pub use registry::{RegistryClient, RegistryConfig, PullResult};
pub use network::{NetworkManager, NetworkConfig, NetworkMode, NetworkEndpoint, IpAddressManager, PortMapping};
pub use orchestrator::{
    Orchestrator, OrchestratorConfig,
    Cluster, ClusterConfig, ClusterState,
    Node, NodeStatus, NodeResources,
    Pod, PodSpec, PodStatus, PodPhase,
    Service, ServiceSpec, ServiceType, LoadBalancer,
    Deployment, DeploymentSpec, DeploymentStatus,
    Scheduler, SchedulingPolicy, SchedulingResult,
    HealthChecker, HealthCheck, HealthStatus,
    NetworkManager, NetworkConfig, OverlayNetwork,
    ResourceManager, ResourceQuota,
};

use std::path::PathBuf;

/// Container ID type
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ContainerId(pub String);

impl ContainerId {
    /// Generate a new random container ID
    pub fn generate() -> Self {
        Self(uuid::Uuid::new_v4().to_string().replace("-", ""))
    }

    /// Create from string
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Get short ID (first 12 chars)
    pub fn short(&self) -> &str {
        &self.0[..12.min(self.0.len())]
    }
}

impl std::fmt::Display for ContainerId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Content-addressable digest
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Digest {
    pub algorithm: String,
    pub hash: String,
}

impl Digest {
    /// Parse digest string (e.g., "sha256:abc123...")
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split(':').collect();
        if parts.len() != 2 {
            return Err(Error::InvalidDigest(s.to_string()));
        }
        Ok(Self {
            algorithm: parts[0].to_string(),
            hash: parts[1].to_string(),
        })
    }

    /// Compute SHA256 digest of data
    pub fn sha256(data: &[u8]) -> Self {
        use sha2::{Sha256, Digest as Sha2Digest};
        let hash = Sha256::digest(data);
        Self {
            algorithm: "sha256".to_string(),
            hash: hex::encode(hash),
        }
    }
}

impl std::fmt::Display for Digest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.algorithm, self.hash)
    }
}

/// Runtime configuration
#[derive(Clone, Debug)]
pub struct RuntimeConfig {
    /// Root directory for runtime state
    pub root: PathBuf,
    /// Container state directory
    pub state_dir: PathBuf,
    /// Image storage directory
    pub image_dir: PathBuf,
    /// Cgroup mount point
    pub cgroup_root: PathBuf,
    /// Enable rootless mode
    pub rootless: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let root = PathBuf::from("/var/lib/docklet");
        Self {
            state_dir: root.join("containers"),
            image_dir: root.join("images"),
            cgroup_root: PathBuf::from("/sys/fs/cgroup"),
            root,
            rootless: false,
        }
    }
}
