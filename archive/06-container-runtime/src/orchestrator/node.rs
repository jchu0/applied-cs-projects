//! Node Management
//!
//! Represents worker nodes in the cluster and manages their state.

use super::{ObjectMeta, ResourceId, LabelSelector, Pod};
use super::scheduler::Taint;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Node in the cluster
#[derive(Clone, Debug)]
pub struct Node {
    /// Metadata
    pub metadata: ObjectMeta,
    /// Current status
    pub status: NodeStatus,
    /// Conditions
    pub conditions: Vec<NodeCondition>,
    /// Total capacity
    pub capacity: NodeResources,
    /// Allocatable resources (after system reservations)
    pub allocatable: NodeResources,
    /// Currently used resources
    pub used: NodeResources,
    /// Node taints
    pub taints: Vec<Taint>,
    /// Node addresses
    pub addresses: Vec<NodeAddress>,
    /// Pods running on this node
    pub running_pods: Vec<Pod>,
}

impl Node {
    /// Create a new node
    pub fn new(name: impl Into<String>, capacity: NodeResources) -> Self {
        let allocatable = NodeResources {
            cpu_millis: capacity.cpu_millis.saturating_sub(100),
            memory_bytes: capacity.memory_bytes.saturating_sub(100 * 1024 * 1024),
            pods: capacity.pods.saturating_sub(10),
            ephemeral_storage: capacity.ephemeral_storage.saturating_sub(1024 * 1024 * 1024),
        };

        Self {
            metadata: ObjectMeta::new(name, ""),
            status: NodeStatus::Unknown,
            conditions: vec![
                NodeCondition {
                    condition_type: NodeConditionType::Ready,
                    status: ConditionStatus::Unknown,
                    last_heartbeat: Instant::now(),
                    last_transition: Instant::now(),
                    reason: "Initializing".into(),
                    message: "Node is initializing".into(),
                },
            ],
            capacity,
            allocatable,
            used: NodeResources::default(),
            taints: vec![],
            addresses: vec![],
            running_pods: vec![],
        }
    }

    /// Check if node is ready to accept pods
    pub fn is_ready(&self) -> bool {
        self.status == NodeStatus::Ready
            && self.conditions.iter().any(|c| {
                c.condition_type == NodeConditionType::Ready
                    && c.status == ConditionStatus::True
            })
    }

    /// Check if node is schedulable
    pub fn is_schedulable(&self) -> bool {
        self.is_ready() && !self.taints.iter().any(|t| {
            t.effect == super::scheduler::TaintEffect::NoSchedule
        })
    }

    /// Get allocatable CPU
    pub fn allocatable_cpu(&self) -> u64 {
        self.allocatable.cpu_millis.saturating_sub(self.used.cpu_millis)
    }

    /// Get allocatable memory
    pub fn allocatable_memory(&self) -> u64 {
        self.allocatable.memory_bytes.saturating_sub(self.used.memory_bytes)
    }

    /// Get allocatable pods
    pub fn allocatable_pods(&self) -> u64 {
        self.allocatable.pods.saturating_sub(self.used.pods)
    }

    /// Calculate resource utilization percentage
    pub fn cpu_utilization(&self) -> f64 {
        if self.capacity.cpu_millis == 0 {
            return 0.0;
        }
        (self.used.cpu_millis as f64 / self.capacity.cpu_millis as f64) * 100.0
    }

    pub fn memory_utilization(&self) -> f64 {
        if self.capacity.memory_bytes == 0 {
            return 0.0;
        }
        (self.used.memory_bytes as f64 / self.capacity.memory_bytes as f64) * 100.0
    }

    /// Update node heartbeat
    pub fn heartbeat(&mut self) {
        for condition in &mut self.conditions {
            condition.last_heartbeat = Instant::now();
        }
    }

    /// Mark node as ready
    pub fn mark_ready(&mut self) {
        self.status = NodeStatus::Ready;
        for condition in &mut self.conditions {
            if condition.condition_type == NodeConditionType::Ready {
                condition.status = ConditionStatus::True;
                condition.last_transition = Instant::now();
                condition.reason = "KubeletReady".into();
                condition.message = "Node is ready".into();
            }
        }
    }

    /// Mark node as not ready
    pub fn mark_not_ready(&mut self, reason: &str, message: &str) {
        self.status = NodeStatus::NotReady;
        for condition in &mut self.conditions {
            if condition.condition_type == NodeConditionType::Ready {
                condition.status = ConditionStatus::False;
                condition.last_transition = Instant::now();
                condition.reason = reason.into();
                condition.message = message.into();
            }
        }
    }

    /// Add a pod to this node
    pub fn add_pod(&mut self, pod: Pod) {
        let cpu_request = pod.spec.total_cpu_request();
        let memory_request = pod.spec.total_memory_request();

        self.used.cpu_millis += cpu_request;
        self.used.memory_bytes += memory_request;
        self.used.pods += 1;

        self.running_pods.push(pod);
    }

    /// Remove a pod from this node
    pub fn remove_pod(&mut self, pod_uid: &ResourceId) -> Option<Pod> {
        if let Some(idx) = self.running_pods.iter().position(|p| &p.metadata.uid == pod_uid) {
            let pod = self.running_pods.remove(idx);

            let cpu_request = pod.spec.total_cpu_request();
            let memory_request = pod.spec.total_memory_request();

            self.used.cpu_millis = self.used.cpu_millis.saturating_sub(cpu_request);
            self.used.memory_bytes = self.used.memory_bytes.saturating_sub(memory_request);
            self.used.pods = self.used.pods.saturating_sub(1);

            return Some(pod);
        }
        None
    }

    /// Get all pods on this node
    pub fn get_pods(&self) -> &[Pod] {
        &self.running_pods
    }

    /// Apply a taint to the node
    pub fn add_taint(&mut self, taint: Taint) {
        self.taints.push(taint);
    }

    /// Remove a taint from the node
    pub fn remove_taint(&mut self, key: &str) {
        self.taints.retain(|t| t.key != key);
    }

    /// Cordon the node (make unschedulable)
    pub fn cordon(&mut self) {
        self.taints.push(Taint {
            key: "node.kubernetes.io/unschedulable".into(),
            value: "true".into(),
            effect: super::scheduler::TaintEffect::NoSchedule,
        });
    }

    /// Uncordon the node (make schedulable)
    pub fn uncordon(&mut self) {
        self.taints.retain(|t| t.key != "node.kubernetes.io/unschedulable");
    }

    /// Check if node is cordoned
    pub fn is_cordoned(&self) -> bool {
        self.taints.iter().any(|t| t.key == "node.kubernetes.io/unschedulable")
    }

    /// Drain the node (evict all pods)
    pub fn drain(&mut self) -> Vec<Pod> {
        self.cordon();
        let pods = std::mem::take(&mut self.running_pods);

        self.used = NodeResources::default();

        pods
    }
}

/// Node status
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeStatus {
    Ready,
    NotReady,
    Unknown,
}

/// Node condition
#[derive(Clone, Debug)]
pub struct NodeCondition {
    pub condition_type: NodeConditionType,
    pub status: ConditionStatus,
    pub last_heartbeat: Instant,
    pub last_transition: Instant,
    pub reason: String,
    pub message: String,
}

/// Node condition type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeConditionType {
    Ready,
    MemoryPressure,
    DiskPressure,
    PIDPressure,
    NetworkUnavailable,
}

/// Condition status
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditionStatus {
    True,
    False,
    Unknown,
}

/// Node resources
#[derive(Clone, Debug, Default)]
pub struct NodeResources {
    /// CPU in millicores
    pub cpu_millis: u64,
    /// Memory in bytes
    pub memory_bytes: u64,
    /// Maximum pods
    pub pods: u64,
    /// Ephemeral storage in bytes
    pub ephemeral_storage: u64,
}

impl NodeResources {
    pub fn new(cpu_millis: u64, memory_bytes: u64, pods: u64, ephemeral_storage: u64) -> Self {
        Self {
            cpu_millis,
            memory_bytes,
            pods,
            ephemeral_storage,
        }
    }

    /// Check if these resources can satisfy a request
    pub fn satisfies(&self, request: &NodeResources) -> bool {
        self.cpu_millis >= request.cpu_millis
            && self.memory_bytes >= request.memory_bytes
            && self.pods >= request.pods
    }

    /// Subtract resources
    pub fn subtract(&self, other: &NodeResources) -> NodeResources {
        NodeResources {
            cpu_millis: self.cpu_millis.saturating_sub(other.cpu_millis),
            memory_bytes: self.memory_bytes.saturating_sub(other.memory_bytes),
            pods: self.pods.saturating_sub(other.pods),
            ephemeral_storage: self.ephemeral_storage.saturating_sub(other.ephemeral_storage),
        }
    }

    /// Add resources
    pub fn add(&self, other: &NodeResources) -> NodeResources {
        NodeResources {
            cpu_millis: self.cpu_millis + other.cpu_millis,
            memory_bytes: self.memory_bytes + other.memory_bytes,
            pods: self.pods + other.pods,
            ephemeral_storage: self.ephemeral_storage + other.ephemeral_storage,
        }
    }
}

/// Node address
#[derive(Clone, Debug)]
pub struct NodeAddress {
    pub address_type: NodeAddressType,
    pub address: String,
}

/// Node address type
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeAddressType {
    InternalIP,
    ExternalIP,
    InternalDNS,
    ExternalDNS,
    Hostname,
}

/// Node selector for pod scheduling
#[derive(Clone, Debug)]
pub struct NodeSelector {
    pub match_labels: HashMap<String, String>,
}

impl NodeSelector {
    pub fn new() -> Self {
        Self {
            match_labels: HashMap::new(),
        }
    }

    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.match_labels.insert(key.into(), value.into());
        self
    }

    pub fn matches(&self, labels: &HashMap<String, String>) -> bool {
        self.match_labels.iter().all(|(k, v)| {
            labels.get(k).map(|lv| lv == v).unwrap_or(false)
        })
    }
}

impl Default for NodeSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Node info for display
#[derive(Clone, Debug)]
pub struct NodeInfo {
    pub name: String,
    pub status: NodeStatus,
    pub cpu_capacity: u64,
    pub cpu_used: u64,
    pub memory_capacity: u64,
    pub memory_used: u64,
    pub pod_count: usize,
    pub age: Duration,
}

impl From<&Node> for NodeInfo {
    fn from(node: &Node) -> Self {
        NodeInfo {
            name: node.metadata.name.clone(),
            status: node.status,
            cpu_capacity: node.capacity.cpu_millis,
            cpu_used: node.used.cpu_millis,
            memory_capacity: node.capacity.memory_bytes,
            memory_used: node.used.memory_bytes,
            pod_count: node.running_pods.len(),
            age: node.metadata.creation_timestamp.elapsed(),
        }
    }
}

/// Node pool for grouping nodes
#[derive(Clone, Debug)]
pub struct NodePool {
    pub name: String,
    pub labels: HashMap<String, String>,
    pub taints: Vec<Taint>,
    pub node_count: usize,
    pub min_nodes: usize,
    pub max_nodes: usize,
    pub machine_type: String,
}

impl NodePool {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            labels: HashMap::new(),
            taints: Vec::new(),
            node_count: 0,
            min_nodes: 1,
            max_nodes: 100,
            machine_type: "standard".into(),
        }
    }

    pub fn should_scale_up(&self) -> bool {
        self.node_count < self.max_nodes
    }

    pub fn should_scale_down(&self) -> bool {
        self.node_count > self.min_nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_node() -> Node {
        Node::new("test-node", NodeResources::new(4000, 8 * 1024 * 1024 * 1024, 110, 100 * 1024 * 1024 * 1024))
    }

    #[test]
    fn test_node_new() {
        let node = create_test_node();
        assert_eq!(node.metadata.name, "test-node");
        assert_eq!(node.status, NodeStatus::Unknown);
        assert_eq!(node.capacity.cpu_millis, 4000);
    }

    #[test]
    fn test_node_ready() {
        let mut node = create_test_node();
        assert!(!node.is_ready());

        node.mark_ready();
        assert!(node.is_ready());
        assert_eq!(node.status, NodeStatus::Ready);
    }

    #[test]
    fn test_node_not_ready() {
        let mut node = create_test_node();
        node.mark_ready();
        assert!(node.is_ready());

        node.mark_not_ready("OutOfDisk", "Node is out of disk");
        assert!(!node.is_ready());
        assert_eq!(node.status, NodeStatus::NotReady);
    }

    #[test]
    fn test_allocatable_resources() {
        let mut node = create_test_node();
        node.used.cpu_millis = 1000;
        node.used.memory_bytes = 2 * 1024 * 1024 * 1024;

        // Allocatable = allocatable - used
        assert!(node.allocatable_cpu() < node.allocatable.cpu_millis);
        assert!(node.allocatable_memory() < node.allocatable.memory_bytes);
    }

    #[test]
    fn test_resource_utilization() {
        let mut node = create_test_node();
        node.used.cpu_millis = 2000; // 50%
        node.used.memory_bytes = 4 * 1024 * 1024 * 1024; // 50%

        assert!((node.cpu_utilization() - 50.0).abs() < 0.01);
        assert!((node.memory_utilization() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_cordon_uncordon() {
        let mut node = create_test_node();
        assert!(!node.is_cordoned());

        node.cordon();
        assert!(node.is_cordoned());

        node.uncordon();
        assert!(!node.is_cordoned());
    }

    #[test]
    fn test_add_remove_taint() {
        let mut node = create_test_node();
        assert!(node.taints.is_empty());

        node.add_taint(Taint {
            key: "special".into(),
            value: "true".into(),
            effect: super::super::scheduler::TaintEffect::NoSchedule,
        });
        assert_eq!(node.taints.len(), 1);

        node.remove_taint("special");
        assert!(node.taints.is_empty());
    }

    #[test]
    fn test_node_selector() {
        let mut labels = HashMap::new();
        labels.insert("zone".into(), "us-west-1".into());
        labels.insert("tier".into(), "compute".into());

        let selector = NodeSelector::new()
            .with_label("zone", "us-west-1");
        assert!(selector.matches(&labels));

        let wrong_selector = NodeSelector::new()
            .with_label("zone", "us-east-1");
        assert!(!wrong_selector.matches(&labels));
    }

    #[test]
    fn test_node_resources_satisfies() {
        let available = NodeResources::new(4000, 8 * 1024 * 1024 * 1024, 100, 0);
        let request = NodeResources::new(1000, 2 * 1024 * 1024 * 1024, 1, 0);
        assert!(available.satisfies(&request));

        let big_request = NodeResources::new(5000, 2 * 1024 * 1024 * 1024, 1, 0);
        assert!(!available.satisfies(&big_request));
    }

    #[test]
    fn test_node_resources_arithmetic() {
        let a = NodeResources::new(2000, 4 * 1024 * 1024 * 1024, 50, 0);
        let b = NodeResources::new(1000, 2 * 1024 * 1024 * 1024, 25, 0);

        let sum = a.add(&b);
        assert_eq!(sum.cpu_millis, 3000);
        assert_eq!(sum.pods, 75);

        let diff = a.subtract(&b);
        assert_eq!(diff.cpu_millis, 1000);
        assert_eq!(diff.pods, 25);
    }

    #[test]
    fn test_node_pool() {
        let mut pool = NodePool::new("compute-pool");
        pool.min_nodes = 3;
        pool.max_nodes = 10;
        pool.node_count = 5;

        assert!(pool.should_scale_up());
        assert!(pool.should_scale_down());

        pool.node_count = 10;
        assert!(!pool.should_scale_up());

        pool.node_count = 3;
        assert!(!pool.should_scale_down());
    }

    #[test]
    fn test_drain() {
        let mut node = create_test_node();
        node.mark_ready();

        // Create a simple pod representation
        let pod = super::super::Pod {
            metadata: ObjectMeta::new("test-pod", "default"),
            spec: super::super::pod::PodSpec::default(),
            status: super::super::pod::PodStatus::default(),
        };

        node.add_pod(pod);
        assert_eq!(node.running_pods.len(), 1);

        let drained = node.drain();
        assert_eq!(drained.len(), 1);
        assert!(node.running_pods.is_empty());
        assert!(node.is_cordoned());
    }
}
