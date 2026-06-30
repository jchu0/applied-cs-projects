//! Container Scheduler
//!
//! Implements scheduling algorithms for placing containers on nodes.
//! Supports multiple scheduling policies:
//! - Bin Packing: Maximize resource utilization per node
//! - Spread: Distribute pods across nodes for high availability
//! - Random: Simple random placement
//! - Least Used: Place on node with most available resources
//! - Priority: Schedule based on pod priority

use super::{Node, Pod, ResourceId, LabelSelector};
use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet, BinaryHeap};
use std::cmp::Ordering;

/// Scheduler for container placement
#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    /// Cached node scores for optimization
    node_scores: HashMap<ResourceId, f64>,
    /// Nodes currently being considered for scheduling
    filter_cache: HashSet<ResourceId>,
}

impl Scheduler {
    /// Create a new scheduler with the given configuration
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            node_scores: HashMap::new(),
            filter_cache: HashSet::new(),
        }
    }

    /// Schedule a pod to an appropriate node
    pub fn schedule(&self, pod: &Pod, nodes: &[&Node]) -> Result<SchedulingResult> {
        if nodes.is_empty() {
            return Err(Error::Runtime("No nodes available for scheduling".into()));
        }

        // Phase 1: Filter - find nodes that can run the pod
        let feasible_nodes = self.filter_nodes(pod, nodes)?;

        if feasible_nodes.is_empty() {
            return Ok(SchedulingResult {
                scheduled: false,
                node_id: None,
                reason: Some("No feasible nodes found".into()),
                scores: HashMap::new(),
            });
        }

        // Phase 2: Score - rank feasible nodes
        let scored_nodes = self.score_nodes(pod, &feasible_nodes)?;

        // Phase 3: Select - choose the best node
        let (best_node, score) = scored_nodes.into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
            .ok_or_else(|| Error::Runtime("No nodes scored".into()))?;

        Ok(SchedulingResult {
            scheduled: true,
            node_id: Some(best_node.metadata.uid.clone()),
            reason: Some(format!("Scheduled with score {:.2}", score)),
            scores: HashMap::new(),
        })
    }

    /// Filter nodes that can run the pod (predicates)
    fn filter_nodes<'a>(&self, pod: &Pod, nodes: &[&'a Node]) -> Result<Vec<&'a Node>> {
        let mut feasible = Vec::new();

        for node in nodes {
            if self.check_node_feasibility(pod, node) {
                feasible.push(*node);
            }
        }

        Ok(feasible)
    }

    /// Check if a node can run a pod
    fn check_node_feasibility(&self, pod: &Pod, node: &Node) -> bool {
        // Check node readiness
        if !node.is_ready() {
            return false;
        }

        // Check node selector
        if let Some(ref selector) = pod.spec.node_selector {
            if !selector.matches(&node.metadata.labels) {
                return false;
            }
        }

        // Check resource requirements
        let required_cpu = pod.spec.total_cpu_request();
        let required_memory = pod.spec.total_memory_request();

        let available_cpu = node.allocatable_cpu();
        let available_memory = node.allocatable_memory();

        if required_cpu > available_cpu || required_memory > available_memory {
            return false;
        }

        // Check taints and tolerations
        if !self.tolerates_taints(pod, node) {
            return false;
        }

        // Check affinity rules
        if !self.check_affinity(pod, node) {
            return false;
        }

        true
    }

    /// Check if pod tolerates node taints
    fn tolerates_taints(&self, pod: &Pod, node: &Node) -> bool {
        for taint in &node.taints {
            if taint.effect == TaintEffect::NoSchedule {
                let tolerated = pod.spec.tolerations.iter().any(|t| {
                    t.key == taint.key &&
                    (t.operator == TolerationOperator::Exists || t.value == taint.value)
                });
                if !tolerated {
                    return false;
                }
            }
        }
        true
    }

    /// Check pod affinity/anti-affinity rules
    fn check_affinity(&self, pod: &Pod, node: &Node) -> bool {
        if let Some(ref affinity) = pod.spec.affinity {
            // Check node affinity
            if let Some(ref node_affinity) = affinity.node_affinity {
                if let Some(ref required) = node_affinity.required_during_scheduling {
                    let matches = required.node_selector_terms.iter().any(|term| {
                        term.match_expressions.iter().all(|expr| {
                            match &expr.operator {
                                NodeSelectorOperator::In => {
                                    node.metadata.labels.get(&expr.key)
                                        .map(|v| expr.values.contains(v))
                                        .unwrap_or(false)
                                }
                                NodeSelectorOperator::NotIn => {
                                    node.metadata.labels.get(&expr.key)
                                        .map(|v| !expr.values.contains(v))
                                        .unwrap_or(true)
                                }
                                NodeSelectorOperator::Exists => {
                                    node.metadata.labels.contains_key(&expr.key)
                                }
                                NodeSelectorOperator::DoesNotExist => {
                                    !node.metadata.labels.contains_key(&expr.key)
                                }
                                NodeSelectorOperator::Gt => {
                                    node.metadata.labels.get(&expr.key)
                                        .and_then(|v| v.parse::<i64>().ok())
                                        .map(|v| expr.values.first()
                                            .and_then(|ev| ev.parse::<i64>().ok())
                                            .map(|ev| v > ev)
                                            .unwrap_or(false))
                                        .unwrap_or(false)
                                }
                                NodeSelectorOperator::Lt => {
                                    node.metadata.labels.get(&expr.key)
                                        .and_then(|v| v.parse::<i64>().ok())
                                        .map(|v| expr.values.first()
                                            .and_then(|ev| ev.parse::<i64>().ok())
                                            .map(|ev| v < ev)
                                            .unwrap_or(false))
                                        .unwrap_or(false)
                                }
                            }
                        })
                    });
                    if !matches {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Score nodes based on the scheduling policy
    fn score_nodes<'a>(&self, pod: &Pod, nodes: &[&'a Node]) -> Result<Vec<(&'a Node, f64)>> {
        let mut scored = Vec::with_capacity(nodes.len());

        for node in nodes {
            let score = self.calculate_node_score(pod, node)?;
            scored.push((*node, score));
        }

        Ok(scored)
    }

    /// Calculate score for a single node
    fn calculate_node_score(&self, pod: &Pod, node: &Node) -> Result<f64> {
        match self.config.policy {
            SchedulingPolicy::BinPacking => self.score_bin_packing(pod, node),
            SchedulingPolicy::Spread => self.score_spread(pod, node),
            SchedulingPolicy::Random => self.score_random(),
            SchedulingPolicy::LeastUsed => self.score_least_used(pod, node),
            SchedulingPolicy::Priority => self.score_priority(pod, node),
            SchedulingPolicy::Balanced => self.score_balanced(pod, node),
        }
    }

    /// Bin packing: maximize resource utilization
    fn score_bin_packing(&self, pod: &Pod, node: &Node) -> Result<f64> {
        let cpu_request = pod.spec.total_cpu_request();
        let memory_request = pod.spec.total_memory_request();

        let cpu_capacity = node.capacity.cpu_millis;
        let memory_capacity = node.capacity.memory_bytes;

        let cpu_after = node.used.cpu_millis + cpu_request;
        let memory_after = node.used.memory_bytes + memory_request;

        // Higher score for nodes that will be more utilized after scheduling
        let cpu_utilization = cpu_after as f64 / cpu_capacity as f64;
        let memory_utilization = memory_after as f64 / memory_capacity as f64;

        // Weight CPU and memory equally
        let score = (cpu_utilization + memory_utilization) / 2.0;

        Ok(score * 100.0)
    }

    /// Spread: distribute pods across nodes
    fn score_spread(&self, pod: &Pod, node: &Node) -> Result<f64> {
        // Lower utilization = higher score
        let cpu_used_ratio = node.used.cpu_millis as f64 / node.capacity.cpu_millis as f64;
        let memory_used_ratio = node.used.memory_bytes as f64 / node.capacity.memory_bytes as f64;

        let avg_utilization = (cpu_used_ratio + memory_used_ratio) / 2.0;

        // Invert: less utilized nodes get higher scores
        Ok((1.0 - avg_utilization) * 100.0)
    }

    /// Random placement
    fn score_random(&self) -> Result<f64> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);

        // Simple pseudo-random using xorshift
        let x = seed;
        let x = x ^ (x << 13);
        let x = x ^ (x >> 7);
        let x = x ^ (x << 17);

        Ok((x % 100) as f64)
    }

    /// Least used: prefer nodes with most available resources
    fn score_least_used(&self, _pod: &Pod, node: &Node) -> Result<f64> {
        let cpu_free = node.capacity.cpu_millis.saturating_sub(node.used.cpu_millis);
        let memory_free = node.capacity.memory_bytes.saturating_sub(node.used.memory_bytes);

        let cpu_free_ratio = cpu_free as f64 / node.capacity.cpu_millis as f64;
        let memory_free_ratio = memory_free as f64 / node.capacity.memory_bytes as f64;

        Ok((cpu_free_ratio + memory_free_ratio) / 2.0 * 100.0)
    }

    /// Priority based scheduling
    fn score_priority(&self, pod: &Pod, node: &Node) -> Result<f64> {
        let base_score = self.score_least_used(pod, node)?;
        let priority_bonus = pod.spec.priority.unwrap_or(0) as f64;

        Ok(base_score + priority_bonus)
    }

    /// Balanced scoring considering multiple factors
    fn score_balanced(&self, pod: &Pod, node: &Node) -> Result<f64> {
        let bin_packing = self.score_bin_packing(pod, node)?;
        let spread = self.score_spread(pod, node)?;

        // Balance between utilization and availability
        let balanced = bin_packing * 0.4 + spread * 0.6;

        // Add affinity bonus
        let affinity_bonus = self.calculate_affinity_bonus(pod, node);

        Ok(balanced + affinity_bonus)
    }

    /// Calculate bonus score for affinity matches
    fn calculate_affinity_bonus(&self, pod: &Pod, node: &Node) -> f64 {
        let mut bonus = 0.0;

        if let Some(ref affinity) = pod.spec.affinity {
            if let Some(ref node_affinity) = affinity.node_affinity {
                if let Some(ref preferred) = node_affinity.preferred_during_scheduling {
                    for pref in preferred {
                        let matches = pref.preference.match_expressions.iter().all(|expr| {
                            match &expr.operator {
                                NodeSelectorOperator::In => {
                                    node.metadata.labels.get(&expr.key)
                                        .map(|v| expr.values.contains(v))
                                        .unwrap_or(false)
                                }
                                NodeSelectorOperator::Exists => {
                                    node.metadata.labels.contains_key(&expr.key)
                                }
                                _ => true,
                            }
                        });
                        if matches {
                            bonus += pref.weight as f64;
                        }
                    }
                }
            }
        }

        bonus
    }

    /// Preemption: find pods to evict to make room
    pub fn find_preemption_candidates(
        &self,
        pod: &Pod,
        nodes: &[&Node],
    ) -> Vec<PreemptionCandidate> {
        let mut candidates = Vec::new();

        for node in nodes {
            if let Some(candidate) = self.find_preemption_on_node(pod, node) {
                candidates.push(candidate);
            }
        }

        // Sort by number of victims (prefer fewer evictions)
        candidates.sort_by(|a, b| a.victims.len().cmp(&b.victims.len()));

        candidates
    }

    /// Find preemption candidates on a specific node
    fn find_preemption_on_node(&self, pod: &Pod, node: &Node) -> Option<PreemptionCandidate> {
        let pod_priority = pod.spec.priority.unwrap_or(0);
        let required_cpu = pod.spec.total_cpu_request();
        let required_memory = pod.spec.total_memory_request();

        // Find pods with lower priority that can be preempted
        let mut victims = Vec::new();
        let mut freed_cpu: u64 = 0;
        let mut freed_memory: u64 = 0;

        // Sort running pods by priority (lowest first)
        let mut running_pods: Vec<_> = node.running_pods.iter().collect();
        running_pods.sort_by(|a, b| {
            let a_priority = a.spec.priority.unwrap_or(0);
            let b_priority = b.spec.priority.unwrap_or(0);
            a_priority.cmp(&b_priority)
        });

        for victim in running_pods {
            let victim_priority = victim.spec.priority.unwrap_or(0);

            // Only preempt lower priority pods
            if victim_priority >= pod_priority {
                continue;
            }

            freed_cpu += victim.spec.total_cpu_request();
            freed_memory += victim.spec.total_memory_request();
            victims.push(victim.metadata.uid.clone());

            // Check if we have enough resources now
            let available_cpu = node.allocatable_cpu() + freed_cpu;
            let available_memory = node.allocatable_memory() + freed_memory;

            if required_cpu <= available_cpu && required_memory <= available_memory {
                return Some(PreemptionCandidate {
                    node_id: node.metadata.uid.clone(),
                    victims,
                    freed_cpu,
                    freed_memory,
                });
            }
        }

        None
    }
}

/// Scheduling configuration
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    pub policy: SchedulingPolicy,
    pub preemption_enabled: bool,
    pub max_scheduling_attempts: u32,
    pub scheduling_timeout_ms: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            policy: SchedulingPolicy::Balanced,
            preemption_enabled: true,
            max_scheduling_attempts: 3,
            scheduling_timeout_ms: 5000,
        }
    }
}

/// Scheduling policy
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// Pack pods densely on nodes
    BinPacking,
    /// Spread pods across nodes
    Spread,
    /// Random placement
    Random,
    /// Use least utilized node
    LeastUsed,
    /// Consider pod priority
    Priority,
    /// Balance multiple factors
    Balanced,
}

/// Result of a scheduling attempt
#[derive(Clone, Debug)]
pub struct SchedulingResult {
    pub scheduled: bool,
    pub node_id: Option<ResourceId>,
    pub reason: Option<String>,
    pub scores: HashMap<ResourceId, f64>,
}

/// Candidate for preemption
#[derive(Clone, Debug)]
pub struct PreemptionCandidate {
    pub node_id: ResourceId,
    pub victims: Vec<ResourceId>,
    pub freed_cpu: u64,
    pub freed_memory: u64,
}

/// Node taint
#[derive(Clone, Debug)]
pub struct Taint {
    pub key: String,
    pub value: String,
    pub effect: TaintEffect,
}

/// Taint effect
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TaintEffect {
    NoSchedule,
    PreferNoSchedule,
    NoExecute,
}

/// Pod toleration
#[derive(Clone, Debug)]
pub struct Toleration {
    pub key: String,
    pub operator: TolerationOperator,
    pub value: String,
    pub effect: Option<TaintEffect>,
    pub toleration_seconds: Option<i64>,
}

/// Toleration operator
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TolerationOperator {
    Equal,
    Exists,
}

/// Pod affinity configuration
#[derive(Clone, Debug, Default)]
pub struct Affinity {
    pub node_affinity: Option<NodeAffinity>,
    pub pod_affinity: Option<PodAffinity>,
    pub pod_anti_affinity: Option<PodAntiAffinity>,
}

/// Node affinity rules
#[derive(Clone, Debug)]
pub struct NodeAffinity {
    pub required_during_scheduling: Option<NodeSelector>,
    pub preferred_during_scheduling: Option<Vec<PreferredSchedulingTerm>>,
}

/// Node selector
#[derive(Clone, Debug)]
pub struct NodeSelector {
    pub node_selector_terms: Vec<NodeSelectorTerm>,
}

/// Node selector term
#[derive(Clone, Debug)]
pub struct NodeSelectorTerm {
    pub match_expressions: Vec<NodeSelectorRequirement>,
    pub match_fields: Vec<NodeSelectorRequirement>,
}

/// Node selector requirement
#[derive(Clone, Debug)]
pub struct NodeSelectorRequirement {
    pub key: String,
    pub operator: NodeSelectorOperator,
    pub values: Vec<String>,
}

/// Node selector operator
#[derive(Clone, Debug)]
pub enum NodeSelectorOperator {
    In,
    NotIn,
    Exists,
    DoesNotExist,
    Gt,
    Lt,
}

/// Preferred scheduling term with weight
#[derive(Clone, Debug)]
pub struct PreferredSchedulingTerm {
    pub weight: i32,
    pub preference: NodeSelectorTerm,
}

/// Pod affinity
#[derive(Clone, Debug)]
pub struct PodAffinity {
    pub required_during_scheduling: Vec<PodAffinityTerm>,
    pub preferred_during_scheduling: Vec<WeightedPodAffinityTerm>,
}

/// Pod anti-affinity
#[derive(Clone, Debug)]
pub struct PodAntiAffinity {
    pub required_during_scheduling: Vec<PodAffinityTerm>,
    pub preferred_during_scheduling: Vec<WeightedPodAffinityTerm>,
}

/// Pod affinity term
#[derive(Clone, Debug)]
pub struct PodAffinityTerm {
    pub label_selector: LabelSelector,
    pub topology_key: String,
    pub namespaces: Vec<String>,
}

/// Weighted pod affinity term
#[derive(Clone, Debug)]
pub struct WeightedPodAffinityTerm {
    pub weight: i32,
    pub pod_affinity_term: PodAffinityTerm,
}

/// Scheduling queue item
#[derive(Debug)]
pub struct SchedulingQueueItem {
    pub pod: Pod,
    pub attempts: u32,
    pub backoff_time: std::time::Instant,
    pub priority: i32,
}

impl PartialEq for SchedulingQueueItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for SchedulingQueueItem {}

impl PartialOrd for SchedulingQueueItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SchedulingQueueItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.cmp(&other.priority)
    }
}

/// Priority queue for scheduling
#[derive(Debug, Default)]
pub struct SchedulingQueue {
    active: BinaryHeap<SchedulingQueueItem>,
    backoff: Vec<SchedulingQueueItem>,
    unschedulable: Vec<SchedulingQueueItem>,
}

impl SchedulingQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, pod: Pod) {
        let priority = pod.spec.priority.unwrap_or(0);
        self.active.push(SchedulingQueueItem {
            pod,
            attempts: 0,
            backoff_time: std::time::Instant::now(),
            priority,
        });
    }

    pub fn pop(&mut self) -> Option<Pod> {
        // First check backoff queue for items ready to retry
        let now = std::time::Instant::now();
        let ready_idx = self.backoff.iter().position(|item| item.backoff_time <= now);

        if let Some(idx) = ready_idx {
            let item = self.backoff.remove(idx);
            return Some(item.pod);
        }

        // Then check active queue
        self.active.pop().map(|item| item.pod)
    }

    pub fn move_to_backoff(&mut self, pod: Pod, delay: std::time::Duration) {
        let priority = pod.spec.priority.unwrap_or(0);
        self.backoff.push(SchedulingQueueItem {
            pod,
            attempts: 1,
            backoff_time: std::time::Instant::now() + delay,
            priority,
        });
    }

    pub fn move_to_unschedulable(&mut self, pod: Pod) {
        let priority = pod.spec.priority.unwrap_or(0);
        self.unschedulable.push(SchedulingQueueItem {
            pod,
            attempts: 0,
            backoff_time: std::time::Instant::now(),
            priority,
        });
    }

    pub fn len(&self) -> usize {
        self.active.len() + self.backoff.len()
    }

    pub fn is_empty(&self) -> bool {
        self.active.is_empty() && self.backoff.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::{ObjectMeta, pod::PodSpec, node::NodeResources};

    fn create_test_node(name: &str, cpu_millis: u64, memory_bytes: u64) -> Node {
        Node {
            metadata: ObjectMeta::new(name, "default"),
            status: super::super::node::NodeStatus::Ready,
            conditions: vec![],
            capacity: NodeResources {
                cpu_millis,
                memory_bytes,
                pods: 110,
                ephemeral_storage: 100 * 1024 * 1024 * 1024,
            },
            allocatable: NodeResources {
                cpu_millis: cpu_millis - 100,
                memory_bytes: memory_bytes - 100 * 1024 * 1024,
                pods: 100,
                ephemeral_storage: 90 * 1024 * 1024 * 1024,
            },
            used: NodeResources {
                cpu_millis: 0,
                memory_bytes: 0,
                pods: 0,
                ephemeral_storage: 0,
            },
            taints: vec![],
            addresses: vec![],
            running_pods: vec![],
        }
    }

    fn create_test_pod(name: &str, cpu_request: u64, memory_request: u64) -> Pod {
        Pod {
            metadata: ObjectMeta::new(name, "default"),
            spec: PodSpec {
                containers: vec![super::super::pod::ContainerSpec {
                    name: "main".into(),
                    image: "nginx:latest".into(),
                    resources: Some(super::super::resource::ResourceRequirements {
                        requests: Some(super::super::resource::ResourceRequest {
                            cpu_millis: cpu_request,
                            memory_bytes: memory_request,
                        }),
                        limits: None,
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            },
            status: super::super::pod::PodStatus::default(),
        }
    }

    #[test]
    fn test_scheduler_new() {
        let config = SchedulerConfig::default();
        let scheduler = Scheduler::new(config);
        assert!(scheduler.node_scores.is_empty());
    }

    #[test]
    fn test_schedule_to_feasible_node() {
        let scheduler = Scheduler::new(SchedulerConfig::default());

        let node = create_test_node("node1", 4000, 4 * 1024 * 1024 * 1024);
        let pod = create_test_pod("pod1", 500, 256 * 1024 * 1024);

        let result = scheduler.schedule(&pod, &[&node]).unwrap();
        assert!(result.scheduled);
        assert_eq!(result.node_id.unwrap().0, node.metadata.uid.0);
    }

    #[test]
    fn test_schedule_no_feasible_nodes() {
        let scheduler = Scheduler::new(SchedulerConfig::default());

        // Node with very little resources
        let mut node = create_test_node("node1", 100, 100 * 1024 * 1024);
        node.used.cpu_millis = 90;
        node.used.memory_bytes = 90 * 1024 * 1024;

        let pod = create_test_pod("pod1", 500, 256 * 1024 * 1024);

        let result = scheduler.schedule(&pod, &[&node]).unwrap();
        assert!(!result.scheduled);
        assert!(result.reason.is_some());
    }

    #[test]
    fn test_schedule_no_nodes() {
        let scheduler = Scheduler::new(SchedulerConfig::default());
        let pod = create_test_pod("pod1", 500, 256 * 1024 * 1024);

        let result = scheduler.schedule(&pod, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_bin_packing_policy() {
        let config = SchedulerConfig {
            policy: SchedulingPolicy::BinPacking,
            ..Default::default()
        };
        let scheduler = Scheduler::new(config);

        // Two nodes with different utilization
        let mut node1 = create_test_node("node1", 4000, 4 * 1024 * 1024 * 1024);
        node1.used.cpu_millis = 2000; // 50% used

        let mut node2 = create_test_node("node2", 4000, 4 * 1024 * 1024 * 1024);
        node2.used.cpu_millis = 1000; // 25% used

        let pod = create_test_pod("pod1", 500, 256 * 1024 * 1024);

        // Bin packing should prefer more utilized node
        let result = scheduler.schedule(&pod, &[&node1, &node2]).unwrap();
        assert!(result.scheduled);
        // Should pick node1 (more utilized)
        assert_eq!(result.node_id.unwrap().0, node1.metadata.uid.0);
    }

    #[test]
    fn test_spread_policy() {
        let config = SchedulerConfig {
            policy: SchedulingPolicy::Spread,
            ..Default::default()
        };
        let scheduler = Scheduler::new(config);

        let mut node1 = create_test_node("node1", 4000, 4 * 1024 * 1024 * 1024);
        node1.used.cpu_millis = 2000; // 50% used

        let mut node2 = create_test_node("node2", 4000, 4 * 1024 * 1024 * 1024);
        node2.used.cpu_millis = 1000; // 25% used

        let pod = create_test_pod("pod1", 500, 256 * 1024 * 1024);

        // Spread should prefer less utilized node
        let result = scheduler.schedule(&pod, &[&node1, &node2]).unwrap();
        assert!(result.scheduled);
        assert_eq!(result.node_id.unwrap().0, node2.metadata.uid.0);
    }

    #[test]
    fn test_node_selector() {
        let scheduler = Scheduler::new(SchedulerConfig::default());

        let mut node1 = create_test_node("node1", 4000, 4 * 1024 * 1024 * 1024);
        node1.metadata.labels.insert("zone".into(), "us-west-1a".into());

        let mut node2 = create_test_node("node2", 4000, 4 * 1024 * 1024 * 1024);
        node2.metadata.labels.insert("zone".into(), "us-east-1a".into());

        let mut pod = create_test_pod("pod1", 500, 256 * 1024 * 1024);
        pod.spec.node_selector = Some(LabelSelector::new().with_label("zone", "us-west-1a"));

        let result = scheduler.schedule(&pod, &[&node1, &node2]).unwrap();
        assert!(result.scheduled);
        assert_eq!(result.node_id.unwrap().0, node1.metadata.uid.0);
    }

    #[test]
    fn test_scheduling_queue() {
        let mut queue = SchedulingQueue::new();

        let pod1 = create_test_pod("pod1", 100, 100 * 1024 * 1024);
        let mut pod2 = create_test_pod("pod2", 100, 100 * 1024 * 1024);
        pod2.spec.priority = Some(10);

        queue.add(pod1);
        queue.add(pod2);

        // Higher priority should come first
        let first = queue.pop().unwrap();
        assert_eq!(first.spec.priority, Some(10));
    }

    #[test]
    fn test_toleration() {
        let scheduler = Scheduler::new(SchedulerConfig::default());

        let mut node = create_test_node("node1", 4000, 4 * 1024 * 1024 * 1024);
        node.taints.push(Taint {
            key: "special".into(),
            value: "true".into(),
            effect: TaintEffect::NoSchedule,
        });

        // Pod without toleration
        let pod1 = create_test_pod("pod1", 500, 256 * 1024 * 1024);
        let result = scheduler.schedule(&pod1, &[&node]).unwrap();
        assert!(!result.scheduled);

        // Pod with toleration
        let mut pod2 = create_test_pod("pod2", 500, 256 * 1024 * 1024);
        pod2.spec.tolerations.push(Toleration {
            key: "special".into(),
            operator: TolerationOperator::Equal,
            value: "true".into(),
            effect: Some(TaintEffect::NoSchedule),
            toleration_seconds: None,
        });
        let result = scheduler.schedule(&pod2, &[&node]).unwrap();
        assert!(result.scheduled);
    }

    #[test]
    fn test_preemption() {
        let scheduler = Scheduler::new(SchedulerConfig {
            preemption_enabled: true,
            ..Default::default()
        });

        let mut node = create_test_node("node1", 2000, 2 * 1024 * 1024 * 1024);

        // Add low priority running pods
        let mut low_priority_pod = create_test_pod("low-pod", 1500, 1 * 1024 * 1024 * 1024);
        low_priority_pod.spec.priority = Some(0);
        node.running_pods.push(low_priority_pod);
        node.used.cpu_millis = 1500;
        node.used.memory_bytes = 1 * 1024 * 1024 * 1024;

        // High priority pod that needs preemption
        let mut high_priority_pod = create_test_pod("high-pod", 1000, 512 * 1024 * 1024);
        high_priority_pod.spec.priority = Some(100);

        let candidates = scheduler.find_preemption_candidates(&high_priority_pod, &[&node]);
        assert!(!candidates.is_empty());
        assert_eq!(candidates[0].node_id.0, node.metadata.uid.0);
    }
}
