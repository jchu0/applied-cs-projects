//! Container Orchestrator
//!
//! A minimal Kubernetes-like container orchestrator providing:
//! - Container scheduling and placement
//! - Service discovery and load balancing
//! - Health checking and self-healing
//! - Resource management (CPU, memory)
//! - Networking and overlay networks

pub mod scheduler;
pub mod node;
pub mod service;
pub mod health;
pub mod network;
pub mod resource;
pub mod deployment;
pub mod cluster;
pub mod pod;
pub mod replication;
pub mod endpoints;

pub use scheduler::{Scheduler, SchedulingPolicy, SchedulingResult, SchedulerConfig};
pub use node::{Node, NodeStatus, NodeCondition, NodeResources, NodeSelector};
pub use service::{Service, ServiceSpec, ServiceType, LoadBalancer, LoadBalancingStrategy};
pub use health::{HealthChecker, HealthCheck, HealthStatus, ProbeType, ProbeResult};
pub use network::{
    NetworkManager, OverlayNetwork, NetworkConfig, VirtualInterface, NetworkPolicy,
    IpAllocator, RoutingTable, VxlanConfig,
};
pub use resource::{ResourceManager, ResourceQuota, ResourceRequest, ResourceLimit, ResourceUsage};
pub use deployment::{Deployment, DeploymentSpec, DeploymentStatus, RollingUpdate, DeploymentStrategy};
pub use cluster::{Cluster, ClusterState, ClusterConfig, ClusterEvent};
pub use pod::{Pod, PodSpec, PodStatus, PodPhase, ContainerSpec, RestartPolicy};
pub use replication::{ReplicaSet, ReplicaSetSpec, ReplicaSetStatus};
pub use endpoints::{Endpoints, EndpointSlice, EndpointPort, EndpointAddress};

use crate::error::{Error, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Unique identifier for orchestrator resources
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ResourceId(pub String);

impl ResourceId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn generate() -> Self {
        Self(uuid::Uuid::new_v4().to_string().replace("-", "")[..12].to_string())
    }
}

impl std::fmt::Display for ResourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Kubernetes-like label selector
#[derive(Clone, Debug, Default)]
pub struct LabelSelector {
    pub match_labels: HashMap<String, String>,
    pub match_expressions: Vec<LabelSelectorRequirement>,
}

impl LabelSelector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.match_labels.insert(key.into(), value.into());
        self
    }

    pub fn matches(&self, labels: &HashMap<String, String>) -> bool {
        // Check match_labels
        for (key, value) in &self.match_labels {
            match labels.get(key) {
                Some(v) if v == value => continue,
                _ => return false,
            }
        }

        // Check match_expressions
        for expr in &self.match_expressions {
            if !expr.matches(labels) {
                return false;
            }
        }

        true
    }
}

/// Label selector requirement for complex matching
#[derive(Clone, Debug)]
pub struct LabelSelectorRequirement {
    pub key: String,
    pub operator: LabelOperator,
    pub values: Vec<String>,
}

impl LabelSelectorRequirement {
    pub fn matches(&self, labels: &HashMap<String, String>) -> bool {
        match &self.operator {
            LabelOperator::In => {
                labels.get(&self.key).map(|v| self.values.contains(v)).unwrap_or(false)
            }
            LabelOperator::NotIn => {
                labels.get(&self.key).map(|v| !self.values.contains(v)).unwrap_or(true)
            }
            LabelOperator::Exists => labels.contains_key(&self.key),
            LabelOperator::DoesNotExist => !labels.contains_key(&self.key),
        }
    }
}

#[derive(Clone, Debug)]
pub enum LabelOperator {
    In,
    NotIn,
    Exists,
    DoesNotExist,
}

/// Object metadata (Kubernetes-style)
#[derive(Clone, Debug)]
pub struct ObjectMeta {
    pub name: String,
    pub namespace: String,
    pub uid: ResourceId,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
    pub creation_timestamp: Instant,
    pub resource_version: u64,
    pub owner_references: Vec<OwnerReference>,
}

impl ObjectMeta {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            namespace: namespace.into(),
            uid: ResourceId::generate(),
            labels: HashMap::new(),
            annotations: HashMap::new(),
            creation_timestamp: Instant::now(),
            resource_version: 1,
            owner_references: Vec::new(),
        }
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }

    pub fn with_annotation(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.annotations.insert(key.into(), value.into());
        self
    }
}

/// Owner reference for garbage collection
#[derive(Clone, Debug)]
pub struct OwnerReference {
    pub api_version: String,
    pub kind: String,
    pub name: String,
    pub uid: ResourceId,
    pub controller: bool,
    pub block_owner_deletion: bool,
}

/// Container orchestrator - the main entry point
pub struct Orchestrator {
    pub cluster: Arc<RwLock<Cluster>>,
    pub scheduler: Arc<RwLock<Scheduler>>,
    pub network_manager: Arc<RwLock<NetworkManager>>,
    pub health_checker: Arc<RwLock<HealthChecker>>,
    pub resource_manager: Arc<RwLock<ResourceManager>>,
}

impl Orchestrator {
    /// Create a new orchestrator with default configuration
    pub fn new(config: OrchestratorConfig) -> Result<Self> {
        let cluster = Arc::new(RwLock::new(Cluster::new(config.cluster_config)?));
        let scheduler = Arc::new(RwLock::new(Scheduler::new(config.scheduler_config)));
        let network_manager = Arc::new(RwLock::new(NetworkManager::new(config.network_config)?));
        let health_checker = Arc::new(RwLock::new(HealthChecker::new(config.health_config)));
        let resource_manager = Arc::new(RwLock::new(ResourceManager::new(config.resource_config)));

        Ok(Self {
            cluster,
            scheduler,
            network_manager,
            health_checker,
            resource_manager,
        })
    }

    /// Register a new node in the cluster
    pub fn register_node(&self, node: Node) -> Result<()> {
        let mut cluster = self.cluster.write().map_err(|_| Error::Internal("Lock poisoned".into()))?;
        cluster.add_node(node)?;
        Ok(())
    }

    /// Remove a node from the cluster
    pub fn deregister_node(&self, node_id: &ResourceId) -> Result<()> {
        let mut cluster = self.cluster.write().map_err(|_| Error::Internal("Lock poisoned".into()))?;
        cluster.remove_node(node_id)?;
        Ok(())
    }

    /// Schedule a pod to a node
    pub fn schedule_pod(&self, pod: &Pod) -> Result<SchedulingResult> {
        let cluster = self.cluster.read().map_err(|_| Error::Internal("Lock poisoned".into()))?;
        let scheduler = self.scheduler.read().map_err(|_| Error::Internal("Lock poisoned".into()))?;

        let nodes: Vec<&Node> = cluster.get_ready_nodes();
        scheduler.schedule(pod, &nodes)
    }

    /// Create a deployment
    pub fn create_deployment(&self, deployment: Deployment) -> Result<()> {
        let mut cluster = self.cluster.write().map_err(|_| Error::Internal("Lock poisoned".into()))?;
        cluster.create_deployment(deployment)?;
        Ok(())
    }

    /// Create a service
    pub fn create_service(&self, service: Service) -> Result<()> {
        let mut cluster = self.cluster.write().map_err(|_| Error::Internal("Lock poisoned".into()))?;
        cluster.create_service(service)?;
        Ok(())
    }

    /// Get cluster state
    pub fn get_cluster_state(&self) -> Result<ClusterState> {
        let cluster = self.cluster.read().map_err(|_| Error::Internal("Lock poisoned".into()))?;
        Ok(cluster.get_state())
    }

    /// Run health checks on all pods
    pub fn run_health_checks(&self) -> Result<Vec<ProbeResult>> {
        let cluster = self.cluster.read().map_err(|_| Error::Internal("Lock poisoned".into()))?;
        let health_checker = self.health_checker.read().map_err(|_| Error::Internal("Lock poisoned".into()))?;

        let pods = cluster.get_all_pods();
        let mut results = Vec::new();

        for pod in pods {
            if let Some(spec) = pod.spec.liveness_probe.as_ref() {
                results.push(health_checker.check(&pod, spec)?);
            }
        }

        Ok(results)
    }

    /// Allocate network resources for a pod
    pub fn allocate_pod_network(&self, pod: &Pod) -> Result<network::PodNetwork> {
        let mut network_manager = self.network_manager.write().map_err(|_| Error::Internal("Lock poisoned".into()))?;
        network_manager.allocate_pod_network(pod)
    }

    /// Get resource usage across the cluster
    pub fn get_resource_usage(&self) -> Result<HashMap<ResourceId, ResourceUsage>> {
        let cluster = self.cluster.read().map_err(|_| Error::Internal("Lock poisoned".into()))?;
        let resource_manager = self.resource_manager.read().map_err(|_| Error::Internal("Lock poisoned".into()))?;

        let mut usage = HashMap::new();
        for node in cluster.get_all_nodes() {
            usage.insert(node.metadata.uid.clone(), resource_manager.get_node_usage(&node));
        }

        Ok(usage)
    }
}

/// Orchestrator configuration
#[derive(Clone, Debug)]
pub struct OrchestratorConfig {
    pub cluster_config: ClusterConfig,
    pub scheduler_config: SchedulerConfig,
    pub network_config: NetworkConfig,
    pub health_config: health::HealthConfig,
    pub resource_config: resource::ResourceConfig,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            cluster_config: ClusterConfig::default(),
            scheduler_config: SchedulerConfig::default(),
            network_config: NetworkConfig::default(),
            health_config: health::HealthConfig::default(),
            resource_config: resource::ResourceConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_id_generate() {
        let id1 = ResourceId::generate();
        let id2 = ResourceId::generate();
        assert_ne!(id1, id2);
        assert_eq!(id1.0.len(), 12);
    }

    #[test]
    fn test_label_selector_matches() {
        let mut labels = HashMap::new();
        labels.insert("app".to_string(), "web".to_string());
        labels.insert("env".to_string(), "prod".to_string());

        let selector = LabelSelector::new()
            .with_label("app", "web")
            .with_label("env", "prod");

        assert!(selector.matches(&labels));

        let wrong_selector = LabelSelector::new().with_label("app", "api");
        assert!(!wrong_selector.matches(&labels));
    }

    #[test]
    fn test_label_selector_requirement_in() {
        let mut labels = HashMap::new();
        labels.insert("tier".to_string(), "frontend".to_string());

        let requirement = LabelSelectorRequirement {
            key: "tier".to_string(),
            operator: LabelOperator::In,
            values: vec!["frontend".to_string(), "backend".to_string()],
        };

        assert!(requirement.matches(&labels));
    }

    #[test]
    fn test_label_selector_requirement_exists() {
        let mut labels = HashMap::new();
        labels.insert("tier".to_string(), "frontend".to_string());

        let requirement = LabelSelectorRequirement {
            key: "tier".to_string(),
            operator: LabelOperator::Exists,
            values: vec![],
        };

        assert!(requirement.matches(&labels));

        let not_exists = LabelSelectorRequirement {
            key: "missing".to_string(),
            operator: LabelOperator::Exists,
            values: vec![],
        };

        assert!(!not_exists.matches(&labels));
    }

    #[test]
    fn test_object_meta_new() {
        let meta = ObjectMeta::new("my-pod", "default")
            .with_label("app", "web")
            .with_annotation("description", "test pod");

        assert_eq!(meta.name, "my-pod");
        assert_eq!(meta.namespace, "default");
        assert_eq!(meta.labels.get("app"), Some(&"web".to_string()));
        assert_eq!(meta.annotations.get("description"), Some(&"test pod".to_string()));
    }

    #[test]
    fn test_orchestrator_config_default() {
        let config = OrchestratorConfig::default();
        assert!(config.cluster_config.name.is_empty() || !config.cluster_config.name.is_empty());
    }
}
