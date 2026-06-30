//! Resource Management
//!
//! Manages CPU, memory, and other resources across the cluster.
//! Implements resource quotas, limits, and requests.

use super::{ResourceId, ObjectMeta, Node, Pod, LabelSelector};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Resource manager for the cluster
#[derive(Debug)]
pub struct ResourceManager {
    config: ResourceConfig,
    /// Resource quotas by namespace
    quotas: HashMap<String, ResourceQuota>,
    /// Resource usage by namespace
    usage: HashMap<String, ResourceUsage>,
    /// Limit ranges by namespace
    limit_ranges: HashMap<String, LimitRange>,
}

impl ResourceManager {
    pub fn new(config: ResourceConfig) -> Self {
        Self {
            config,
            quotas: HashMap::new(),
            usage: HashMap::new(),
            limit_ranges: HashMap::new(),
        }
    }

    /// Set a resource quota for a namespace
    pub fn set_quota(&mut self, namespace: &str, quota: ResourceQuota) {
        self.quotas.insert(namespace.to_string(), quota);
    }

    /// Get quota for a namespace
    pub fn get_quota(&self, namespace: &str) -> Option<&ResourceQuota> {
        self.quotas.get(namespace)
    }

    /// Set a limit range for a namespace
    pub fn set_limit_range(&mut self, namespace: &str, limit_range: LimitRange) {
        self.limit_ranges.insert(namespace.to_string(), limit_range);
    }

    /// Get limit range for a namespace
    pub fn get_limit_range(&self, namespace: &str) -> Option<&LimitRange> {
        self.limit_ranges.get(namespace)
    }

    /// Check if a pod's resource request is within quota
    pub fn check_quota(&self, namespace: &str, request: &ResourceRequest) -> Result<()> {
        let Some(quota) = self.quotas.get(namespace) else {
            return Ok(()); // No quota set
        };

        let current = self.usage.get(namespace).cloned().unwrap_or_default();

        // Check CPU
        if let Some(max_cpu) = quota.hard.cpu_millis {
            if current.cpu_millis + request.cpu_millis > max_cpu {
                return Err(Error::Runtime(format!(
                    "CPU quota exceeded: requested {} + current {} > limit {}",
                    request.cpu_millis, current.cpu_millis, max_cpu
                )));
            }
        }

        // Check memory
        if let Some(max_memory) = quota.hard.memory_bytes {
            if current.memory_bytes + request.memory_bytes > max_memory {
                return Err(Error::Runtime(format!(
                    "Memory quota exceeded: requested {} + current {} > limit {}",
                    request.memory_bytes, current.memory_bytes, max_memory
                )));
            }
        }

        // Check pods
        if let Some(max_pods) = quota.hard.pods {
            if current.pods + 1 > max_pods {
                return Err(Error::Runtime(format!(
                    "Pod quota exceeded: current {} >= limit {}",
                    current.pods, max_pods
                )));
            }
        }

        Ok(())
    }

    /// Check if pod resources are within limit range
    pub fn check_limit_range(&self, namespace: &str, pod: &Pod) -> Result<()> {
        let Some(limit_range) = self.limit_ranges.get(namespace) else {
            return Ok(()); // No limit range set
        };

        for container in &pod.spec.containers {
            let request = container.resources.as_ref()
                .and_then(|r| r.requests.as_ref());
            let limit = container.resources.as_ref()
                .and_then(|r| r.limits.as_ref());

            // Check container limits
            for lr in &limit_range.limits {
                if lr.limit_type != LimitType::Container {
                    continue;
                }

                // Check minimum
                if let Some(req) = request {
                    if let Some(min_cpu) = lr.min.as_ref().map(|m| m.cpu_millis) {
                        if req.cpu_millis < min_cpu {
                            return Err(Error::Runtime(format!(
                                "CPU request {} below minimum {}",
                                req.cpu_millis, min_cpu
                            )));
                        }
                    }
                    if let Some(min_mem) = lr.min.as_ref().map(|m| m.memory_bytes) {
                        if req.memory_bytes < min_mem {
                            return Err(Error::Runtime(format!(
                                "Memory request {} below minimum {}",
                                req.memory_bytes, min_mem
                            )));
                        }
                    }
                }

                // Check maximum
                if let Some(lim) = limit {
                    if let Some(max_cpu) = lr.max.as_ref().map(|m| m.cpu_millis) {
                        if lim.cpu_millis > max_cpu {
                            return Err(Error::Runtime(format!(
                                "CPU limit {} exceeds maximum {}",
                                lim.cpu_millis, max_cpu
                            )));
                        }
                    }
                    if let Some(max_mem) = lr.max.as_ref().map(|m| m.memory_bytes) {
                        if lim.memory_bytes > max_mem {
                            return Err(Error::Runtime(format!(
                                "Memory limit {} exceeds maximum {}",
                                lim.memory_bytes, max_mem
                            )));
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Add resource usage for a pod
    pub fn add_usage(&mut self, namespace: &str, request: &ResourceRequest) {
        let usage = self.usage.entry(namespace.to_string()).or_default();
        usage.cpu_millis += request.cpu_millis;
        usage.memory_bytes += request.memory_bytes;
        usage.pods += 1;
    }

    /// Remove resource usage for a pod
    pub fn remove_usage(&mut self, namespace: &str, request: &ResourceRequest) {
        let usage = self.usage.entry(namespace.to_string()).or_default();
        usage.cpu_millis = usage.cpu_millis.saturating_sub(request.cpu_millis);
        usage.memory_bytes = usage.memory_bytes.saturating_sub(request.memory_bytes);
        usage.pods = usage.pods.saturating_sub(1);
    }

    /// Get current usage for a namespace
    pub fn get_usage(&self, namespace: &str) -> ResourceUsage {
        self.usage.get(namespace).cloned().unwrap_or_default()
    }

    /// Get resource usage for a node
    pub fn get_node_usage(&self, node: &Node) -> ResourceUsage {
        ResourceUsage {
            cpu_millis: node.used.cpu_millis,
            memory_bytes: node.used.memory_bytes,
            pods: node.running_pods.len() as u64,
            ephemeral_storage: node.used.ephemeral_storage,
            ..Default::default()
        }
    }

    /// Calculate cluster-wide resource totals
    pub fn get_cluster_totals(&self, nodes: &[Node]) -> ClusterResources {
        let mut totals = ClusterResources::default();

        for node in nodes {
            totals.total_cpu_millis += node.capacity.cpu_millis;
            totals.total_memory_bytes += node.capacity.memory_bytes;
            totals.allocatable_cpu_millis += node.allocatable.cpu_millis;
            totals.allocatable_memory_bytes += node.allocatable.memory_bytes;
            totals.used_cpu_millis += node.used.cpu_millis;
            totals.used_memory_bytes += node.used.memory_bytes;
            totals.total_nodes += 1;
            totals.total_pods += node.running_pods.len() as u64;
        }

        totals
    }

    /// Apply default resources from limit range
    pub fn apply_defaults(&self, namespace: &str, pod: &mut Pod) {
        let Some(limit_range) = self.limit_ranges.get(namespace) else {
            return;
        };

        for container in &mut pod.spec.containers {
            for lr in &limit_range.limits {
                if lr.limit_type != LimitType::Container {
                    continue;
                }

                // Apply default request if not set
                if container.resources.is_none() {
                    container.resources = Some(ResourceRequirements::default());
                }

                let resources = container.resources.as_mut().unwrap();

                if resources.requests.is_none() {
                    if let Some(ref default_req) = lr.default_request {
                        resources.requests = Some(default_req.clone());
                    }
                }

                if resources.limits.is_none() {
                    if let Some(ref default_lim) = lr.default_limit {
                        resources.limits = Some(default_lim.clone());
                    }
                }
            }
        }
    }
}

/// Resource manager configuration
#[derive(Clone, Debug, Default)]
pub struct ResourceConfig {
    /// Enable resource quotas
    pub enable_quotas: bool,
    /// Enable limit ranges
    pub enable_limit_ranges: bool,
    /// Default CPU request in millicores
    pub default_cpu_request: u64,
    /// Default memory request in bytes
    pub default_memory_request: u64,
}

/// Resource quota for a namespace
#[derive(Clone, Debug)]
pub struct ResourceQuota {
    pub metadata: ObjectMeta,
    pub spec: ResourceQuotaSpec,
    pub status: ResourceQuotaStatus,
}

impl ResourceQuota {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            spec: ResourceQuotaSpec::default(),
            status: ResourceQuotaStatus::default(),
        }
    }

    pub fn with_hard_limit(mut self, resource: &str, value: u64) -> Self {
        match resource {
            "cpu" => self.spec.hard.cpu_millis = Some(value),
            "memory" => self.spec.hard.memory_bytes = Some(value),
            "pods" => self.spec.hard.pods = Some(value),
            "services" => self.spec.hard.services = Some(value),
            "secrets" => self.spec.hard.secrets = Some(value),
            "configmaps" => self.spec.hard.configmaps = Some(value),
            "persistentvolumeclaims" => self.spec.hard.persistent_volume_claims = Some(value),
            _ => {}
        }
        self
    }
}

/// Resource quota specification
#[derive(Clone, Debug, Default)]
pub struct ResourceQuotaSpec {
    /// Hard resource limits
    pub hard: ResourceList,
    /// Scope selector
    pub scope_selector: Option<ScopeSelector>,
    /// Scopes
    pub scopes: Vec<ResourceQuotaScope>,
}

/// Resource quota status
#[derive(Clone, Debug, Default)]
pub struct ResourceQuotaStatus {
    /// Currently used resources
    pub used: ResourceList,
    /// Hard limits
    pub hard: ResourceList,
}

/// Resource quota scope
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceQuotaScope {
    Terminating,
    NotTerminating,
    BestEffort,
    NotBestEffort,
    PriorityClass,
    CrossNamespacePodAffinity,
}

/// Scope selector
#[derive(Clone, Debug)]
pub struct ScopeSelector {
    pub match_expressions: Vec<ScopeSelectorRequirement>,
}

/// Scope selector requirement
#[derive(Clone, Debug)]
pub struct ScopeSelectorRequirement {
    pub scope_name: ResourceQuotaScope,
    pub operator: ScopeSelectorOperator,
    pub values: Vec<String>,
}

/// Scope selector operator
#[derive(Clone, Debug)]
pub enum ScopeSelectorOperator {
    In,
    NotIn,
    Exists,
    DoesNotExist,
}

/// Resource list
#[derive(Clone, Debug, Default)]
pub struct ResourceList {
    pub cpu_millis: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub pods: Option<u64>,
    pub services: Option<u64>,
    pub secrets: Option<u64>,
    pub configmaps: Option<u64>,
    pub persistent_volume_claims: Option<u64>,
    pub requests_cpu: Option<u64>,
    pub requests_memory: Option<u64>,
    pub limits_cpu: Option<u64>,
    pub limits_memory: Option<u64>,
}

impl ResourceList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cpu(mut self, millis: u64) -> Self {
        self.cpu_millis = Some(millis);
        self
    }

    pub fn memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    pub fn pods(mut self, count: u64) -> Self {
        self.pods = Some(count);
        self
    }
}

/// Resource request
#[derive(Clone, Debug, Default)]
pub struct ResourceRequest {
    pub cpu_millis: u64,
    pub memory_bytes: u64,
}

impl ResourceRequest {
    pub fn new(cpu_millis: u64, memory_bytes: u64) -> Self {
        Self {
            cpu_millis,
            memory_bytes,
        }
    }
}

/// Resource limit
#[derive(Clone, Debug, Default)]
pub struct ResourceLimit {
    pub cpu_millis: u64,
    pub memory_bytes: u64,
}

impl ResourceLimit {
    pub fn new(cpu_millis: u64, memory_bytes: u64) -> Self {
        Self {
            cpu_millis,
            memory_bytes,
        }
    }
}

/// Resource requirements (requests + limits)
#[derive(Clone, Debug, Default)]
pub struct ResourceRequirements {
    pub requests: Option<ResourceRequest>,
    pub limits: Option<ResourceLimit>,
}

impl ResourceRequirements {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_requests(mut self, cpu_millis: u64, memory_bytes: u64) -> Self {
        self.requests = Some(ResourceRequest::new(cpu_millis, memory_bytes));
        self
    }

    pub fn with_limits(mut self, cpu_millis: u64, memory_bytes: u64) -> Self {
        self.limits = Some(ResourceLimit::new(cpu_millis, memory_bytes));
        self
    }
}

/// Resource usage tracking
#[derive(Clone, Debug, Default)]
pub struct ResourceUsage {
    pub cpu_millis: u64,
    pub memory_bytes: u64,
    pub pods: u64,
    pub ephemeral_storage: u64,
    pub services: u64,
    pub secrets: u64,
    pub configmaps: u64,
}

/// Limit range
#[derive(Clone, Debug)]
pub struct LimitRange {
    pub metadata: ObjectMeta,
    pub spec: LimitRangeSpec,
}

impl LimitRange {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            spec: LimitRangeSpec::default(),
        }
    }

    pub fn with_container_defaults(mut self, default: ResourceLimit, default_request: ResourceRequest) -> Self {
        self.spec.limits.push(LimitRangeItem {
            limit_type: LimitType::Container,
            max: None,
            min: None,
            default_limit: Some(default),
            default_request: Some(default_request),
            max_limit_request_ratio: None,
        });
        self
    }

    pub fn with_container_limits(mut self, min: ResourceRequest, max: ResourceLimit) -> Self {
        self.spec.limits.push(LimitRangeItem {
            limit_type: LimitType::Container,
            max: Some(max),
            min: Some(min),
            default_limit: None,
            default_request: None,
            max_limit_request_ratio: None,
        });
        self
    }
}

/// Limit range specification
#[derive(Clone, Debug, Default)]
pub struct LimitRangeSpec {
    pub limits: Vec<LimitRangeItem>,
}

/// Limit range item
#[derive(Clone, Debug)]
pub struct LimitRangeItem {
    pub limit_type: LimitType,
    pub max: Option<ResourceLimit>,
    pub min: Option<ResourceRequest>,
    pub default_limit: Option<ResourceLimit>,
    pub default_request: Option<ResourceRequest>,
    pub max_limit_request_ratio: Option<f64>,
}

/// Limit type
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LimitType {
    #[default]
    Container,
    Pod,
    PersistentVolumeClaim,
}

/// Cluster-wide resource totals
#[derive(Clone, Debug, Default)]
pub struct ClusterResources {
    pub total_nodes: u64,
    pub total_pods: u64,
    pub total_cpu_millis: u64,
    pub total_memory_bytes: u64,
    pub allocatable_cpu_millis: u64,
    pub allocatable_memory_bytes: u64,
    pub used_cpu_millis: u64,
    pub used_memory_bytes: u64,
}

impl ClusterResources {
    pub fn cpu_utilization(&self) -> f64 {
        if self.allocatable_cpu_millis == 0 {
            return 0.0;
        }
        (self.used_cpu_millis as f64 / self.allocatable_cpu_millis as f64) * 100.0
    }

    pub fn memory_utilization(&self) -> f64 {
        if self.allocatable_memory_bytes == 0 {
            return 0.0;
        }
        (self.used_memory_bytes as f64 / self.allocatable_memory_bytes as f64) * 100.0
    }
}

/// Vertical Pod Autoscaler recommendation
#[derive(Clone, Debug)]
pub struct VPARecommendation {
    pub container_name: String,
    pub target: ResourceRequirements,
    pub lower_bound: ResourceRequirements,
    pub upper_bound: ResourceRequirements,
}

/// Horizontal Pod Autoscaler
#[derive(Clone, Debug)]
pub struct HorizontalPodAutoscaler {
    pub metadata: ObjectMeta,
    pub spec: HPASpec,
    pub status: HPAStatus,
}

/// HPA specification
#[derive(Clone, Debug)]
pub struct HPASpec {
    pub scale_target_ref: CrossVersionObjectReference,
    pub min_replicas: u32,
    pub max_replicas: u32,
    pub metrics: Vec<MetricSpec>,
    pub behavior: Option<HPAScalingBehavior>,
}

/// Cross version object reference
#[derive(Clone, Debug)]
pub struct CrossVersionObjectReference {
    pub kind: String,
    pub name: String,
    pub api_version: String,
}

/// Metric specification for HPA
#[derive(Clone, Debug)]
pub enum MetricSpec {
    Resource {
        name: String,
        target: MetricTarget,
    },
    Pods {
        metric: String,
        target: MetricTarget,
    },
    Object {
        described_object: CrossVersionObjectReference,
        metric: String,
        target: MetricTarget,
    },
    External {
        metric: String,
        target: MetricTarget,
    },
}

/// Metric target
#[derive(Clone, Debug)]
pub struct MetricTarget {
    pub target_type: MetricTargetType,
    pub value: Option<i64>,
    pub average_value: Option<i64>,
    pub average_utilization: Option<i32>,
}

/// Metric target type
#[derive(Clone, Debug)]
pub enum MetricTargetType {
    Utilization,
    Value,
    AverageValue,
}

/// HPA scaling behavior
#[derive(Clone, Debug)]
pub struct HPAScalingBehavior {
    pub scale_up: Option<HPAScalingRules>,
    pub scale_down: Option<HPAScalingRules>,
}

/// HPA scaling rules
#[derive(Clone, Debug)]
pub struct HPAScalingRules {
    pub stabilization_window_seconds: i32,
    pub select_policy: ScalingPolicySelect,
    pub policies: Vec<HPAScalingPolicy>,
}

/// Scaling policy selection
#[derive(Clone, Debug)]
pub enum ScalingPolicySelect {
    Max,
    Min,
    Disabled,
}

/// HPA scaling policy
#[derive(Clone, Debug)]
pub struct HPAScalingPolicy {
    pub policy_type: ScalingPolicyType,
    pub value: i32,
    pub period_seconds: i32,
}

/// Scaling policy type
#[derive(Clone, Debug)]
pub enum ScalingPolicyType {
    Pods,
    Percent,
}

/// HPA status
#[derive(Clone, Debug, Default)]
pub struct HPAStatus {
    pub observed_generation: i64,
    pub last_scale_time: Option<Instant>,
    pub current_replicas: u32,
    pub desired_replicas: u32,
    pub current_metrics: Vec<MetricStatus>,
    pub conditions: Vec<HPACondition>,
}

/// Metric status
#[derive(Clone, Debug)]
pub struct MetricStatus {
    pub metric_type: String,
    pub resource: Option<ResourceMetricStatus>,
}

/// Resource metric status
#[derive(Clone, Debug)]
pub struct ResourceMetricStatus {
    pub name: String,
    pub current: MetricValueStatus,
}

/// Metric value status
#[derive(Clone, Debug)]
pub struct MetricValueStatus {
    pub value: Option<i64>,
    pub average_value: Option<i64>,
    pub average_utilization: Option<i32>,
}

/// HPA condition
#[derive(Clone, Debug)]
pub struct HPACondition {
    pub condition_type: HPAConditionType,
    pub status: bool,
    pub last_transition_time: Instant,
    pub reason: String,
    pub message: String,
}

/// HPA condition type
#[derive(Clone, Debug)]
pub enum HPAConditionType {
    AbleToScale,
    ScalingActive,
    ScalingLimited,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_manager_new() {
        let config = ResourceConfig::default();
        let manager = ResourceManager::new(config);
        assert!(manager.quotas.is_empty());
    }

    #[test]
    fn test_resource_quota() {
        let quota = ResourceQuota::new("compute-quota", "default")
            .with_hard_limit("cpu", 8000)
            .with_hard_limit("memory", 16 * 1024 * 1024 * 1024)
            .with_hard_limit("pods", 20);

        assert_eq!(quota.spec.hard.cpu_millis, Some(8000));
        assert_eq!(quota.spec.hard.memory_bytes, Some(16 * 1024 * 1024 * 1024));
        assert_eq!(quota.spec.hard.pods, Some(20));
    }

    #[test]
    fn test_check_quota_within_limits() {
        let mut manager = ResourceManager::new(ResourceConfig::default());

        manager.set_quota("default", ResourceQuota {
            metadata: ObjectMeta::new("quota", "default"),
            spec: ResourceQuotaSpec {
                hard: ResourceList {
                    cpu_millis: Some(4000),
                    memory_bytes: Some(8 * 1024 * 1024 * 1024),
                    pods: Some(10),
                    ..Default::default()
                },
                ..Default::default()
            },
            status: ResourceQuotaStatus::default(),
        });

        let request = ResourceRequest::new(500, 512 * 1024 * 1024);
        assert!(manager.check_quota("default", &request).is_ok());
    }

    #[test]
    fn test_check_quota_exceeded() {
        let mut manager = ResourceManager::new(ResourceConfig::default());

        manager.set_quota("default", ResourceQuota {
            metadata: ObjectMeta::new("quota", "default"),
            spec: ResourceQuotaSpec {
                hard: ResourceList {
                    cpu_millis: Some(1000),
                    memory_bytes: Some(1024 * 1024 * 1024),
                    pods: Some(5),
                    ..Default::default()
                },
                ..Default::default()
            },
            status: ResourceQuotaStatus::default(),
        });

        // Add some usage
        manager.add_usage("default", &ResourceRequest::new(800, 512 * 1024 * 1024));

        // Try to add more than available
        let request = ResourceRequest::new(500, 256 * 1024 * 1024);
        assert!(manager.check_quota("default", &request).is_err());
    }

    #[test]
    fn test_add_remove_usage() {
        let mut manager = ResourceManager::new(ResourceConfig::default());

        let request = ResourceRequest::new(1000, 1024 * 1024 * 1024);
        manager.add_usage("default", &request);

        let usage = manager.get_usage("default");
        assert_eq!(usage.cpu_millis, 1000);
        assert_eq!(usage.pods, 1);

        manager.remove_usage("default", &request);
        let usage = manager.get_usage("default");
        assert_eq!(usage.cpu_millis, 0);
        assert_eq!(usage.pods, 0);
    }

    #[test]
    fn test_limit_range() {
        let limit_range = LimitRange::new("limits", "default")
            .with_container_defaults(
                ResourceLimit::new(2000, 2 * 1024 * 1024 * 1024),
                ResourceRequest::new(100, 128 * 1024 * 1024),
            )
            .with_container_limits(
                ResourceRequest::new(50, 64 * 1024 * 1024),
                ResourceLimit::new(4000, 4 * 1024 * 1024 * 1024),
            );

        assert_eq!(limit_range.spec.limits.len(), 2);
    }

    #[test]
    fn test_resource_requirements_builder() {
        let req = ResourceRequirements::new()
            .with_requests(100, 128 * 1024 * 1024)
            .with_limits(200, 256 * 1024 * 1024);

        assert!(req.requests.is_some());
        assert!(req.limits.is_some());
        assert_eq!(req.requests.unwrap().cpu_millis, 100);
        assert_eq!(req.limits.unwrap().memory_bytes, 256 * 1024 * 1024);
    }

    #[test]
    fn test_cluster_resources() {
        let mut resources = ClusterResources {
            total_nodes: 3,
            total_pods: 50,
            total_cpu_millis: 12000,
            total_memory_bytes: 48 * 1024 * 1024 * 1024,
            allocatable_cpu_millis: 10000,
            allocatable_memory_bytes: 40 * 1024 * 1024 * 1024,
            used_cpu_millis: 5000,
            used_memory_bytes: 20 * 1024 * 1024 * 1024,
        };

        assert!((resources.cpu_utilization() - 50.0).abs() < 0.01);
        assert!((resources.memory_utilization() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_resource_list_builder() {
        let list = ResourceList::new()
            .cpu(4000)
            .memory(8 * 1024 * 1024 * 1024)
            .pods(10);

        assert_eq!(list.cpu_millis, Some(4000));
        assert_eq!(list.memory_bytes, Some(8 * 1024 * 1024 * 1024));
        assert_eq!(list.pods, Some(10));
    }

    #[test]
    fn test_check_limit_range() {
        let mut manager = ResourceManager::new(ResourceConfig::default());

        manager.set_limit_range("default", LimitRange {
            metadata: ObjectMeta::new("limits", "default"),
            spec: LimitRangeSpec {
                limits: vec![LimitRangeItem {
                    limit_type: LimitType::Container,
                    max: Some(ResourceLimit::new(2000, 2 * 1024 * 1024 * 1024)),
                    min: Some(ResourceRequest::new(100, 64 * 1024 * 1024)),
                    default_limit: None,
                    default_request: None,
                    max_limit_request_ratio: None,
                }],
            },
        });

        // Create a valid pod
        let mut pod = Pod::new("test", "default");
        pod.spec.containers.push(super::super::pod::ContainerSpec {
            name: "app".into(),
            image: "nginx".into(),
            resources: Some(ResourceRequirements {
                requests: Some(ResourceRequest::new(200, 128 * 1024 * 1024)),
                limits: Some(ResourceLimit::new(1000, 1024 * 1024 * 1024)),
            }),
            ..Default::default()
        });

        assert!(manager.check_limit_range("default", &pod).is_ok());
    }

    #[test]
    fn test_check_limit_range_violation() {
        let mut manager = ResourceManager::new(ResourceConfig::default());

        manager.set_limit_range("default", LimitRange {
            metadata: ObjectMeta::new("limits", "default"),
            spec: LimitRangeSpec {
                limits: vec![LimitRangeItem {
                    limit_type: LimitType::Container,
                    max: Some(ResourceLimit::new(1000, 1024 * 1024 * 1024)),
                    min: Some(ResourceRequest::new(100, 64 * 1024 * 1024)),
                    default_limit: None,
                    default_request: None,
                    max_limit_request_ratio: None,
                }],
            },
        });

        // Create a pod that violates limits
        let mut pod = Pod::new("test", "default");
        pod.spec.containers.push(super::super::pod::ContainerSpec {
            name: "app".into(),
            image: "nginx".into(),
            resources: Some(ResourceRequirements {
                requests: Some(ResourceRequest::new(50, 32 * 1024 * 1024)), // Below min
                limits: Some(ResourceLimit::new(500, 512 * 1024 * 1024)),
            }),
            ..Default::default()
        });

        assert!(manager.check_limit_range("default", &pod).is_err());
    }

    #[test]
    fn test_hpa_spec() {
        let hpa = HorizontalPodAutoscaler {
            metadata: ObjectMeta::new("web-hpa", "default"),
            spec: HPASpec {
                scale_target_ref: CrossVersionObjectReference {
                    kind: "Deployment".into(),
                    name: "web".into(),
                    api_version: "apps/v1".into(),
                },
                min_replicas: 2,
                max_replicas: 10,
                metrics: vec![MetricSpec::Resource {
                    name: "cpu".into(),
                    target: MetricTarget {
                        target_type: MetricTargetType::Utilization,
                        value: None,
                        average_value: None,
                        average_utilization: Some(80),
                    },
                }],
                behavior: None,
            },
            status: HPAStatus::default(),
        };

        assert_eq!(hpa.spec.min_replicas, 2);
        assert_eq!(hpa.spec.max_replicas, 10);
    }
}
