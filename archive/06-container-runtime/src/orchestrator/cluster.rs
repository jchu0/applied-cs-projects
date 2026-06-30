//! Cluster Management
//!
//! Manages the overall cluster state including nodes, pods, deployments, and services.

use super::{
    ObjectMeta, ResourceId, LabelSelector, Node, Pod, Service, Deployment, ReplicaSet,
    node::NodeStatus,
};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cluster state management
#[derive(Debug)]
pub struct Cluster {
    config: ClusterConfig,
    nodes: HashMap<ResourceId, Node>,
    pods: HashMap<ResourceId, Pod>,
    services: HashMap<ResourceId, Service>,
    deployments: HashMap<ResourceId, Deployment>,
    replica_sets: HashMap<ResourceId, ReplicaSet>,
    namespaces: HashMap<String, Namespace>,
    events: Vec<ClusterEvent>,
    state: ClusterState,
}

impl Cluster {
    pub fn new(config: ClusterConfig) -> Result<Self> {
        let mut cluster = Self {
            config: config.clone(),
            nodes: HashMap::new(),
            pods: HashMap::new(),
            services: HashMap::new(),
            deployments: HashMap::new(),
            replica_sets: HashMap::new(),
            namespaces: HashMap::new(),
            events: Vec::new(),
            state: ClusterState::Initializing,
        };

        // Create default namespace
        cluster.create_namespace(Namespace {
            metadata: ObjectMeta::new("default", ""),
            status: NamespaceStatus::Active,
            spec: NamespaceSpec::default(),
        })?;

        // Create kube-system namespace
        cluster.create_namespace(Namespace {
            metadata: ObjectMeta::new("kube-system", ""),
            status: NamespaceStatus::Active,
            spec: NamespaceSpec::default(),
        })?;

        Ok(cluster)
    }

    /// Get current cluster state
    pub fn get_state(&self) -> ClusterState {
        self.state.clone()
    }

    /// Add a node to the cluster
    pub fn add_node(&mut self, node: Node) -> Result<()> {
        let id = node.metadata.uid.clone();

        if self.nodes.contains_key(&id) {
            return Err(Error::Runtime("Node already exists".into()));
        }

        self.emit_event(ClusterEvent::NodeAdded {
            node_id: id.clone(),
            name: node.metadata.name.clone(),
        });

        self.nodes.insert(id, node);
        self.update_cluster_state();

        Ok(())
    }

    /// Remove a node from the cluster
    pub fn remove_node(&mut self, node_id: &ResourceId) -> Result<()> {
        if !self.nodes.contains_key(node_id) {
            return Err(Error::Runtime("Node not found".into()));
        }

        // Check if node has running pods
        let pods_on_node: Vec<_> = self.pods.values()
            .filter(|p| p.status.node_name.as_ref().map(|n| n == &node_id.0).unwrap_or(false))
            .map(|p| p.metadata.uid.clone())
            .collect();

        if !pods_on_node.is_empty() {
            return Err(Error::Runtime(format!(
                "Node has {} running pods - drain first",
                pods_on_node.len()
            )));
        }

        self.emit_event(ClusterEvent::NodeRemoved {
            node_id: node_id.clone(),
        });

        self.nodes.remove(node_id);
        self.update_cluster_state();

        Ok(())
    }

    /// Get a node by ID
    pub fn get_node(&self, node_id: &ResourceId) -> Option<&Node> {
        self.nodes.get(node_id)
    }

    /// Get a mutable reference to a node
    pub fn get_node_mut(&mut self, node_id: &ResourceId) -> Option<&mut Node> {
        self.nodes.get_mut(node_id)
    }

    /// Get all nodes
    pub fn get_all_nodes(&self) -> Vec<&Node> {
        self.nodes.values().collect()
    }

    /// Get ready nodes
    pub fn get_ready_nodes(&self) -> Vec<&Node> {
        self.nodes.values()
            .filter(|n| n.is_ready())
            .collect()
    }

    /// Create a pod
    pub fn create_pod(&mut self, pod: Pod) -> Result<()> {
        let id = pod.metadata.uid.clone();

        if self.pods.contains_key(&id) {
            return Err(Error::Runtime("Pod already exists".into()));
        }

        // Validate namespace exists
        if !self.namespaces.contains_key(&pod.metadata.namespace) {
            return Err(Error::Runtime(format!(
                "Namespace {} does not exist",
                pod.metadata.namespace
            )));
        }

        self.emit_event(ClusterEvent::PodCreated {
            pod_id: id.clone(),
            name: pod.metadata.name.clone(),
            namespace: pod.metadata.namespace.clone(),
        });

        self.pods.insert(id, pod);
        Ok(())
    }

    /// Delete a pod
    pub fn delete_pod(&mut self, pod_id: &ResourceId) -> Result<Pod> {
        let pod = self.pods.remove(pod_id)
            .ok_or_else(|| Error::Runtime("Pod not found".into()))?;

        self.emit_event(ClusterEvent::PodDeleted {
            pod_id: pod_id.clone(),
            name: pod.metadata.name.clone(),
        });

        // Remove from node if scheduled
        if let Some(ref node_name) = pod.status.node_name {
            for node in self.nodes.values_mut() {
                if node.metadata.name == *node_name {
                    node.remove_pod(pod_id);
                    break;
                }
            }
        }

        Ok(pod)
    }

    /// Get a pod by ID
    pub fn get_pod(&self, pod_id: &ResourceId) -> Option<&Pod> {
        self.pods.get(pod_id)
    }

    /// Get all pods
    pub fn get_all_pods(&self) -> Vec<&Pod> {
        self.pods.values().collect()
    }

    /// Get pods by selector
    pub fn get_pods_by_selector(&self, namespace: &str, selector: &LabelSelector) -> Vec<&Pod> {
        self.pods.values()
            .filter(|p| p.metadata.namespace == namespace && selector.matches(&p.metadata.labels))
            .collect()
    }

    /// Create a service
    pub fn create_service(&mut self, service: Service) -> Result<()> {
        let id = service.metadata.uid.clone();

        if self.services.contains_key(&id) {
            return Err(Error::Runtime("Service already exists".into()));
        }

        self.emit_event(ClusterEvent::ServiceCreated {
            service_id: id.clone(),
            name: service.metadata.name.clone(),
            namespace: service.metadata.namespace.clone(),
        });

        self.services.insert(id, service);
        Ok(())
    }

    /// Delete a service
    pub fn delete_service(&mut self, service_id: &ResourceId) -> Result<Service> {
        let service = self.services.remove(service_id)
            .ok_or_else(|| Error::Runtime("Service not found".into()))?;

        self.emit_event(ClusterEvent::ServiceDeleted {
            service_id: service_id.clone(),
            name: service.metadata.name.clone(),
        });

        Ok(service)
    }

    /// Get a service by ID
    pub fn get_service(&self, service_id: &ResourceId) -> Option<&Service> {
        self.services.get(service_id)
    }

    /// Get all services
    pub fn get_all_services(&self) -> Vec<&Service> {
        self.services.values().collect()
    }

    /// Create a deployment
    pub fn create_deployment(&mut self, deployment: Deployment) -> Result<()> {
        let id = deployment.metadata.uid.clone();

        if self.deployments.contains_key(&id) {
            return Err(Error::Runtime("Deployment already exists".into()));
        }

        self.emit_event(ClusterEvent::DeploymentCreated {
            deployment_id: id.clone(),
            name: deployment.metadata.name.clone(),
            namespace: deployment.metadata.namespace.clone(),
        });

        self.deployments.insert(id, deployment);
        Ok(())
    }

    /// Delete a deployment
    pub fn delete_deployment(&mut self, deployment_id: &ResourceId) -> Result<Deployment> {
        let deployment = self.deployments.remove(deployment_id)
            .ok_or_else(|| Error::Runtime("Deployment not found".into()))?;

        self.emit_event(ClusterEvent::DeploymentDeleted {
            deployment_id: deployment_id.clone(),
            name: deployment.metadata.name.clone(),
        });

        Ok(deployment)
    }

    /// Get a deployment by ID
    pub fn get_deployment(&self, deployment_id: &ResourceId) -> Option<&Deployment> {
        self.deployments.get(deployment_id)
    }

    /// Get all deployments
    pub fn get_all_deployments(&self) -> Vec<&Deployment> {
        self.deployments.values().collect()
    }

    /// Create a namespace
    pub fn create_namespace(&mut self, namespace: Namespace) -> Result<()> {
        let name = namespace.metadata.name.clone();

        if self.namespaces.contains_key(&name) {
            return Err(Error::Runtime("Namespace already exists".into()));
        }

        self.namespaces.insert(name, namespace);
        Ok(())
    }

    /// Delete a namespace
    pub fn delete_namespace(&mut self, name: &str) -> Result<()> {
        if name == "default" || name == "kube-system" {
            return Err(Error::Runtime("Cannot delete system namespace".into()));
        }

        // Check if namespace has resources
        let has_pods = self.pods.values().any(|p| p.metadata.namespace == name);
        let has_services = self.services.values().any(|s| s.metadata.namespace == name);
        let has_deployments = self.deployments.values().any(|d| d.metadata.namespace == name);

        if has_pods || has_services || has_deployments {
            return Err(Error::Runtime("Namespace has resources - delete them first".into()));
        }

        self.namespaces.remove(name);
        Ok(())
    }

    /// Get a namespace by name
    pub fn get_namespace(&self, name: &str) -> Option<&Namespace> {
        self.namespaces.get(name)
    }

    /// Get all namespaces
    pub fn get_all_namespaces(&self) -> Vec<&Namespace> {
        self.namespaces.values().collect()
    }

    /// Get cluster summary
    pub fn get_summary(&self) -> ClusterSummary {
        let ready_nodes = self.nodes.values().filter(|n| n.is_ready()).count();
        let running_pods = self.pods.values()
            .filter(|p| p.status.phase == super::pod::PodPhase::Running)
            .count();

        let total_cpu: u64 = self.nodes.values().map(|n| n.capacity.cpu_millis).sum();
        let used_cpu: u64 = self.nodes.values().map(|n| n.used.cpu_millis).sum();
        let total_memory: u64 = self.nodes.values().map(|n| n.capacity.memory_bytes).sum();
        let used_memory: u64 = self.nodes.values().map(|n| n.used.memory_bytes).sum();

        ClusterSummary {
            total_nodes: self.nodes.len(),
            ready_nodes,
            total_pods: self.pods.len(),
            running_pods,
            total_services: self.services.len(),
            total_deployments: self.deployments.len(),
            total_namespaces: self.namespaces.len(),
            cpu_capacity: total_cpu,
            cpu_used: used_cpu,
            memory_capacity: total_memory,
            memory_used: used_memory,
        }
    }

    /// Update node status from heartbeat
    pub fn node_heartbeat(&mut self, node_id: &ResourceId) {
        if let Some(node) = self.nodes.get_mut(node_id) {
            node.heartbeat();
        }
    }

    /// Check for stale nodes
    pub fn check_node_health(&mut self) {
        let stale_threshold = Duration::from_secs(self.config.node_heartbeat_timeout_seconds as u64);
        let now = Instant::now();

        for node in self.nodes.values_mut() {
            let last_heartbeat = node.conditions.iter()
                .find(|c| c.condition_type == super::node::NodeConditionType::Ready)
                .map(|c| c.last_heartbeat);

            if let Some(last) = last_heartbeat {
                if now.duration_since(last) > stale_threshold {
                    node.mark_not_ready("NodeNotResponding", "Node has not sent heartbeat");

                    self.emit_event(ClusterEvent::NodeNotReady {
                        node_id: node.metadata.uid.clone(),
                        reason: "Heartbeat timeout".to_string(),
                    });
                }
            }
        }

        self.update_cluster_state();
    }

    /// Emit a cluster event
    fn emit_event(&mut self, event: ClusterEvent) {
        // Keep last N events
        if self.events.len() >= self.config.max_events {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Get recent events
    pub fn get_events(&self, limit: usize) -> Vec<&ClusterEvent> {
        self.events.iter().rev().take(limit).collect()
    }

    /// Update overall cluster state
    fn update_cluster_state(&mut self) {
        let total_nodes = self.nodes.len();
        let ready_nodes = self.nodes.values().filter(|n| n.is_ready()).count();

        self.state = if total_nodes == 0 {
            ClusterState::Initializing
        } else if ready_nodes == 0 {
            ClusterState::Unhealthy
        } else if ready_nodes < total_nodes {
            ClusterState::Degraded
        } else {
            ClusterState::Healthy
        };
    }
}

/// Cluster configuration
#[derive(Clone, Debug)]
pub struct ClusterConfig {
    pub name: String,
    pub node_heartbeat_timeout_seconds: u32,
    pub pod_eviction_timeout_seconds: u32,
    pub max_events: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            node_heartbeat_timeout_seconds: 40,
            pod_eviction_timeout_seconds: 300,
            max_events: 1000,
        }
    }
}

/// Cluster state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClusterState {
    Initializing,
    Healthy,
    Degraded,
    Unhealthy,
}

/// Cluster event
#[derive(Clone, Debug)]
pub enum ClusterEvent {
    NodeAdded { node_id: ResourceId, name: String },
    NodeRemoved { node_id: ResourceId },
    NodeReady { node_id: ResourceId },
    NodeNotReady { node_id: ResourceId, reason: String },
    PodCreated { pod_id: ResourceId, name: String, namespace: String },
    PodScheduled { pod_id: ResourceId, node_id: ResourceId },
    PodDeleted { pod_id: ResourceId, name: String },
    PodFailed { pod_id: ResourceId, reason: String },
    ServiceCreated { service_id: ResourceId, name: String, namespace: String },
    ServiceDeleted { service_id: ResourceId, name: String },
    DeploymentCreated { deployment_id: ResourceId, name: String, namespace: String },
    DeploymentDeleted { deployment_id: ResourceId, name: String },
    DeploymentScaled { deployment_id: ResourceId, old_replicas: u32, new_replicas: u32 },
}

/// Namespace
#[derive(Clone, Debug)]
pub struct Namespace {
    pub metadata: ObjectMeta,
    pub status: NamespaceStatus,
    pub spec: NamespaceSpec,
}

/// Namespace status
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NamespaceStatus {
    Active,
    Terminating,
}

/// Namespace specification
#[derive(Clone, Debug, Default)]
pub struct NamespaceSpec {
    pub finalizers: Vec<String>,
}

/// Cluster summary
#[derive(Clone, Debug)]
pub struct ClusterSummary {
    pub total_nodes: usize,
    pub ready_nodes: usize,
    pub total_pods: usize,
    pub running_pods: usize,
    pub total_services: usize,
    pub total_deployments: usize,
    pub total_namespaces: usize,
    pub cpu_capacity: u64,
    pub cpu_used: u64,
    pub memory_capacity: u64,
    pub memory_used: u64,
}

impl ClusterSummary {
    pub fn cpu_utilization(&self) -> f64 {
        if self.cpu_capacity == 0 {
            return 0.0;
        }
        (self.cpu_used as f64 / self.cpu_capacity as f64) * 100.0
    }

    pub fn memory_utilization(&self) -> f64 {
        if self.memory_capacity == 0 {
            return 0.0;
        }
        (self.memory_used as f64 / self.memory_capacity as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::node::NodeResources;

    fn create_test_node(name: &str) -> Node {
        let mut node = Node::new(name, NodeResources::new(4000, 8 * 1024 * 1024 * 1024, 110, 100 * 1024 * 1024 * 1024));
        node.mark_ready();
        node
    }

    fn create_test_pod(name: &str, namespace: &str) -> Pod {
        Pod::new(name, namespace)
    }

    #[test]
    fn test_cluster_new() {
        let config = ClusterConfig::default();
        let cluster = Cluster::new(config).unwrap();

        assert!(cluster.get_namespace("default").is_some());
        assert!(cluster.get_namespace("kube-system").is_some());
        assert_eq!(cluster.get_state(), ClusterState::Initializing);
    }

    #[test]
    fn test_add_node() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();
        let node = create_test_node("node-1");

        cluster.add_node(node).unwrap();

        assert_eq!(cluster.get_all_nodes().len(), 1);
        assert_eq!(cluster.get_state(), ClusterState::Healthy);
    }

    #[test]
    fn test_add_duplicate_node() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();
        let node = create_test_node("node-1");
        let id = node.metadata.uid.clone();

        cluster.add_node(node).unwrap();

        let mut duplicate = create_test_node("node-1");
        duplicate.metadata.uid = id;

        assert!(cluster.add_node(duplicate).is_err());
    }

    #[test]
    fn test_create_pod() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();
        let pod = create_test_pod("nginx", "default");

        cluster.create_pod(pod).unwrap();

        assert_eq!(cluster.get_all_pods().len(), 1);
    }

    #[test]
    fn test_create_pod_invalid_namespace() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();
        let pod = create_test_pod("nginx", "nonexistent");

        assert!(cluster.create_pod(pod).is_err());
    }

    #[test]
    fn test_create_service() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();
        let service = Service::new("web", "default");

        cluster.create_service(service).unwrap();

        assert_eq!(cluster.get_all_services().len(), 1);
    }

    #[test]
    fn test_create_deployment() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();
        let deployment = Deployment::new("nginx", "default");

        cluster.create_deployment(deployment).unwrap();

        assert_eq!(cluster.get_all_deployments().len(), 1);
    }

    #[test]
    fn test_delete_namespace_with_resources() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();

        cluster.create_namespace(Namespace {
            metadata: ObjectMeta::new("test-ns", ""),
            status: NamespaceStatus::Active,
            spec: NamespaceSpec::default(),
        }).unwrap();

        let pod = create_test_pod("nginx", "test-ns");
        cluster.create_pod(pod).unwrap();

        assert!(cluster.delete_namespace("test-ns").is_err());
    }

    #[test]
    fn test_delete_system_namespace() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();

        assert!(cluster.delete_namespace("default").is_err());
        assert!(cluster.delete_namespace("kube-system").is_err());
    }

    #[test]
    fn test_get_pods_by_selector() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();

        let mut pod1 = create_test_pod("web-1", "default");
        pod1.metadata.labels.insert("app".to_string(), "web".to_string());

        let mut pod2 = create_test_pod("web-2", "default");
        pod2.metadata.labels.insert("app".to_string(), "web".to_string());

        let mut pod3 = create_test_pod("api", "default");
        pod3.metadata.labels.insert("app".to_string(), "api".to_string());

        cluster.create_pod(pod1).unwrap();
        cluster.create_pod(pod2).unwrap();
        cluster.create_pod(pod3).unwrap();

        let selector = LabelSelector::new().with_label("app", "web");
        let pods = cluster.get_pods_by_selector("default", &selector);

        assert_eq!(pods.len(), 2);
    }

    #[test]
    fn test_cluster_state_degraded() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();

        let node1 = create_test_node("node-1");
        let mut node2 = create_test_node("node-2");
        node2.mark_not_ready("Test", "Test failure");

        cluster.add_node(node1).unwrap();
        cluster.add_node(node2).unwrap();

        assert_eq!(cluster.get_state(), ClusterState::Degraded);
    }

    #[test]
    fn test_cluster_summary() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();

        let node = create_test_node("node-1");
        cluster.add_node(node).unwrap();

        let pod = create_test_pod("nginx", "default");
        cluster.create_pod(pod).unwrap();

        let summary = cluster.get_summary();

        assert_eq!(summary.total_nodes, 1);
        assert_eq!(summary.ready_nodes, 1);
        assert_eq!(summary.total_pods, 1);
    }

    #[test]
    fn test_get_events() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();

        let node = create_test_node("node-1");
        cluster.add_node(node).unwrap();

        let events = cluster.get_events(10);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_remove_node_with_pods() {
        let mut cluster = Cluster::new(ClusterConfig::default()).unwrap();

        let node = create_test_node("node-1");
        let node_id = node.metadata.uid.clone();
        cluster.add_node(node).unwrap();

        let mut pod = create_test_pod("nginx", "default");
        pod.status.node_name = Some(node_id.0.clone());
        cluster.create_pod(pod).unwrap();

        assert!(cluster.remove_node(&node_id).is_err());
    }
}
