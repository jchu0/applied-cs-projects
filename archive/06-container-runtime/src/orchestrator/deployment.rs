//! Deployment Management
//!
//! Manages declarative updates for Pods and ReplicaSets.
//! Supports rolling updates, rollbacks, and scaling.

use super::{ObjectMeta, ResourceId, LabelSelector, Pod, pod::PodSpec, ReplicaSet, ReplicaSetSpec};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Deployment for managing pod updates
#[derive(Clone, Debug)]
pub struct Deployment {
    pub metadata: ObjectMeta,
    pub spec: DeploymentSpec,
    pub status: DeploymentStatus,
}

impl Deployment {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            spec: DeploymentSpec::default(),
            status: DeploymentStatus::default(),
        }
    }

    pub fn with_replicas(mut self, replicas: u32) -> Self {
        self.spec.replicas = replicas;
        self
    }

    pub fn with_selector(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.spec.selector.match_labels.insert(key.into(), value.into());
        self
    }

    pub fn with_template(mut self, template: PodTemplateSpec) -> Self {
        self.spec.template = template;
        self
    }

    /// Check if deployment is progressing
    pub fn is_progressing(&self) -> bool {
        self.status.updated_replicas < self.spec.replicas
            || self.status.available_replicas < self.spec.replicas
    }

    /// Check if deployment is complete
    pub fn is_complete(&self) -> bool {
        self.status.updated_replicas == self.spec.replicas
            && self.status.available_replicas == self.spec.replicas
            && self.status.ready_replicas == self.spec.replicas
    }

    /// Check if deployment is failed
    pub fn is_failed(&self) -> bool {
        self.status.conditions.iter().any(|c| {
            c.condition_type == DeploymentConditionType::Progressing
                && !c.status
                && c.reason == "ProgressDeadlineExceeded"
        })
    }

    /// Get rollout progress as percentage
    pub fn rollout_progress(&self) -> f64 {
        if self.spec.replicas == 0 {
            return 100.0;
        }
        (self.status.updated_replicas as f64 / self.spec.replicas as f64) * 100.0
    }
}

/// Deployment specification
#[derive(Clone, Debug)]
pub struct DeploymentSpec {
    /// Number of desired replicas
    pub replicas: u32,
    /// Label selector for pods
    pub selector: LabelSelector,
    /// Pod template
    pub template: PodTemplateSpec,
    /// Deployment strategy
    pub strategy: DeploymentStrategy,
    /// Minimum seconds a pod must be ready before considered available
    pub min_ready_seconds: u32,
    /// Revision history limit
    pub revision_history_limit: u32,
    /// Progress deadline in seconds
    pub progress_deadline_seconds: u32,
    /// Paused
    pub paused: bool,
}

impl Default for DeploymentSpec {
    fn default() -> Self {
        Self {
            replicas: 1,
            selector: LabelSelector::default(),
            template: PodTemplateSpec::default(),
            strategy: DeploymentStrategy::RollingUpdate(RollingUpdate::default()),
            min_ready_seconds: 0,
            revision_history_limit: 10,
            progress_deadline_seconds: 600,
            paused: false,
        }
    }
}

/// Pod template specification
#[derive(Clone, Debug, Default)]
pub struct PodTemplateSpec {
    pub metadata: ObjectMeta,
    pub spec: PodSpec,
}

/// Deployment strategy
#[derive(Clone, Debug)]
pub enum DeploymentStrategy {
    /// Replace all pods at once
    Recreate,
    /// Rolling update with configurable parameters
    RollingUpdate(RollingUpdate),
}

impl Default for DeploymentStrategy {
    fn default() -> Self {
        DeploymentStrategy::RollingUpdate(RollingUpdate::default())
    }
}

/// Rolling update parameters
#[derive(Clone, Debug)]
pub struct RollingUpdate {
    /// Maximum number of pods that can be unavailable during update
    pub max_unavailable: IntOrString,
    /// Maximum number of pods that can be created above desired replicas
    pub max_surge: IntOrString,
}

impl Default for RollingUpdate {
    fn default() -> Self {
        Self {
            max_unavailable: IntOrString::String("25%".to_string()),
            max_surge: IntOrString::String("25%".to_string()),
        }
    }
}

impl RollingUpdate {
    /// Calculate max unavailable count
    pub fn max_unavailable_count(&self, replicas: u32) -> u32 {
        self.max_unavailable.as_count(replicas).max(1)
    }

    /// Calculate max surge count
    pub fn max_surge_count(&self, replicas: u32) -> u32 {
        self.max_surge.as_count(replicas)
    }
}

/// Integer or percentage string
#[derive(Clone, Debug)]
pub enum IntOrString {
    Int(u32),
    String(String),
}

impl IntOrString {
    pub fn as_count(&self, total: u32) -> u32 {
        match self {
            IntOrString::Int(n) => *n,
            IntOrString::String(s) => {
                if s.ends_with('%') {
                    let pct: f64 = s.trim_end_matches('%').parse().unwrap_or(0.0);
                    ((total as f64 * pct) / 100.0).ceil() as u32
                } else {
                    s.parse().unwrap_or(0)
                }
            }
        }
    }
}

/// Deployment status
#[derive(Clone, Debug, Default)]
pub struct DeploymentStatus {
    /// Total number of non-terminated pods
    pub replicas: u32,
    /// Total number of non-terminated pods with the updated template
    pub updated_replicas: u32,
    /// Total number of ready pods
    pub ready_replicas: u32,
    /// Total number of available pods
    pub available_replicas: u32,
    /// Total number of unavailable pods
    pub unavailable_replicas: u32,
    /// Observed generation
    pub observed_generation: i64,
    /// Deployment conditions
    pub conditions: Vec<DeploymentCondition>,
    /// Collision count
    pub collision_count: u32,
}

/// Deployment condition
#[derive(Clone, Debug)]
pub struct DeploymentCondition {
    pub condition_type: DeploymentConditionType,
    pub status: bool,
    pub last_update_time: Instant,
    pub last_transition_time: Instant,
    pub reason: String,
    pub message: String,
}

/// Deployment condition type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeploymentConditionType {
    Available,
    Progressing,
    ReplicaFailure,
}

/// Deployment controller for managing deployments
#[derive(Debug)]
pub struct DeploymentController {
    deployments: HashMap<ResourceId, Deployment>,
    replica_sets: HashMap<ResourceId, Vec<ReplicaSet>>,
    revision_history: HashMap<ResourceId, Vec<DeploymentRevision>>,
}

impl DeploymentController {
    pub fn new() -> Self {
        Self {
            deployments: HashMap::new(),
            replica_sets: HashMap::new(),
            revision_history: HashMap::new(),
        }
    }

    /// Create a new deployment
    pub fn create(&mut self, deployment: Deployment) -> Result<()> {
        let id = deployment.metadata.uid.clone();

        // Create initial replica set
        let rs = self.create_replica_set(&deployment, 1)?;
        self.replica_sets.insert(id.clone(), vec![rs]);

        // Store revision
        self.revision_history.insert(id.clone(), vec![DeploymentRevision {
            revision: 1,
            template: deployment.spec.template.clone(),
            timestamp: Instant::now(),
        }]);

        self.deployments.insert(id, deployment);
        Ok(())
    }

    /// Update a deployment
    pub fn update(&mut self, deployment: Deployment) -> Result<()> {
        let id = deployment.metadata.uid.clone();

        if !self.deployments.contains_key(&id) {
            return Err(Error::Runtime("Deployment not found".into()));
        }

        let old_deployment = self.deployments.get(&id).unwrap();
        let template_changed = self.template_changed(old_deployment, &deployment);

        if template_changed {
            // Create new replica set
            let revision = self.get_next_revision(&id);
            let rs = self.create_replica_set(&deployment, revision)?;

            self.replica_sets.entry(id.clone()).or_default().push(rs);

            // Store revision
            self.revision_history.entry(id.clone()).or_default().push(DeploymentRevision {
                revision,
                template: deployment.spec.template.clone(),
                timestamp: Instant::now(),
            });
        }

        self.deployments.insert(id, deployment);
        Ok(())
    }

    /// Delete a deployment
    pub fn delete(&mut self, deployment_id: &ResourceId) -> Result<()> {
        self.deployments.remove(deployment_id);
        self.replica_sets.remove(deployment_id);
        self.revision_history.remove(deployment_id);
        Ok(())
    }

    /// Scale a deployment
    pub fn scale(&mut self, deployment_id: &ResourceId, replicas: u32) -> Result<()> {
        let deployment = self.deployments.get_mut(deployment_id)
            .ok_or_else(|| Error::Runtime("Deployment not found".into()))?;

        deployment.spec.replicas = replicas;
        Ok(())
    }

    /// Pause a deployment
    pub fn pause(&mut self, deployment_id: &ResourceId) -> Result<()> {
        let deployment = self.deployments.get_mut(deployment_id)
            .ok_or_else(|| Error::Runtime("Deployment not found".into()))?;

        deployment.spec.paused = true;
        Ok(())
    }

    /// Resume a deployment
    pub fn resume(&mut self, deployment_id: &ResourceId) -> Result<()> {
        let deployment = self.deployments.get_mut(deployment_id)
            .ok_or_else(|| Error::Runtime("Deployment not found".into()))?;

        deployment.spec.paused = false;
        Ok(())
    }

    /// Rollback to a specific revision
    pub fn rollback(&mut self, deployment_id: &ResourceId, revision: u64) -> Result<()> {
        let history = self.revision_history.get(deployment_id)
            .ok_or_else(|| Error::Runtime("No revision history".into()))?;

        let target = history.iter()
            .find(|r| r.revision == revision)
            .ok_or_else(|| Error::Runtime(format!("Revision {} not found", revision)))?;

        let deployment = self.deployments.get_mut(deployment_id)
            .ok_or_else(|| Error::Runtime("Deployment not found".into()))?;

        // Update template to previous revision
        deployment.spec.template = target.template.clone();

        // Create new replica set with old template
        let new_revision = self.get_next_revision(deployment_id);
        let rs = self.create_replica_set(deployment, new_revision)?;
        self.replica_sets.entry(deployment_id.clone()).or_default().push(rs);

        Ok(())
    }

    /// Get revision history
    pub fn get_history(&self, deployment_id: &ResourceId) -> Vec<DeploymentRevision> {
        self.revision_history.get(deployment_id).cloned().unwrap_or_default()
    }

    /// Reconcile deployment state
    pub fn reconcile(&mut self, deployment_id: &ResourceId) -> Result<ReconcileResult> {
        let deployment = self.deployments.get(deployment_id)
            .ok_or_else(|| Error::Runtime("Deployment not found".into()))?;

        if deployment.spec.paused {
            return Ok(ReconcileResult::Paused);
        }

        let replica_sets = self.replica_sets.get(deployment_id)
            .ok_or_else(|| Error::Runtime("No replica sets".into()))?;

        let current_rs = replica_sets.last()
            .ok_or_else(|| Error::Runtime("No current replica set".into()))?;

        let desired = deployment.spec.replicas;
        let current = current_rs.status.ready_replicas;

        match &deployment.spec.strategy {
            DeploymentStrategy::Recreate => {
                if current < desired {
                    Ok(ReconcileResult::ScaleUp {
                        current,
                        desired,
                        to_create: desired - current,
                    })
                } else {
                    Ok(ReconcileResult::Complete)
                }
            }
            DeploymentStrategy::RollingUpdate(params) => {
                let max_unavailable = params.max_unavailable_count(desired);
                let max_surge = params.max_surge_count(desired);

                // Calculate how many pods we can create/delete
                let available = current;
                let total_allowed = desired + max_surge;
                let min_available = desired.saturating_sub(max_unavailable);

                if current < desired && available >= min_available {
                    let to_create = (desired - current).min(total_allowed - current);
                    Ok(ReconcileResult::ScaleUp {
                        current,
                        desired,
                        to_create,
                    })
                } else if current > desired {
                    let to_delete = current - desired;
                    Ok(ReconcileResult::ScaleDown {
                        current,
                        desired,
                        to_delete,
                    })
                } else {
                    Ok(ReconcileResult::Complete)
                }
            }
        }
    }

    fn create_replica_set(&self, deployment: &Deployment, revision: u64) -> Result<ReplicaSet> {
        Ok(ReplicaSet {
            metadata: ObjectMeta::new(
                format!("{}-{}", deployment.metadata.name, revision),
                deployment.metadata.namespace.clone(),
            ),
            spec: ReplicaSetSpec {
                replicas: deployment.spec.replicas,
                selector: deployment.spec.selector.clone(),
                template: deployment.spec.template.clone(),
                min_ready_seconds: deployment.spec.min_ready_seconds,
            },
            status: super::replication::ReplicaSetStatus::default(),
        })
    }

    fn template_changed(&self, old: &Deployment, new: &Deployment) -> bool {
        // Simple check - in reality would compare full template
        old.spec.template.metadata.resource_version != new.spec.template.metadata.resource_version
    }

    fn get_next_revision(&self, deployment_id: &ResourceId) -> u64 {
        self.revision_history.get(deployment_id)
            .and_then(|h| h.last())
            .map(|r| r.revision + 1)
            .unwrap_or(1)
    }
}

impl Default for DeploymentController {
    fn default() -> Self {
        Self::new()
    }
}

/// Deployment revision for rollback
#[derive(Clone, Debug)]
pub struct DeploymentRevision {
    pub revision: u64,
    pub template: PodTemplateSpec,
    pub timestamp: Instant,
}

/// Result of reconciliation
#[derive(Clone, Debug)]
pub enum ReconcileResult {
    Complete,
    Paused,
    ScaleUp { current: u32, desired: u32, to_create: u32 },
    ScaleDown { current: u32, desired: u32, to_delete: u32 },
    RollingUpdate { old_replicas: u32, new_replicas: u32 },
}

/// StatefulSet for stateful applications
#[derive(Clone, Debug)]
pub struct StatefulSet {
    pub metadata: ObjectMeta,
    pub spec: StatefulSetSpec,
    pub status: StatefulSetStatus,
}

/// StatefulSet specification
#[derive(Clone, Debug)]
pub struct StatefulSetSpec {
    pub replicas: u32,
    pub selector: LabelSelector,
    pub template: PodTemplateSpec,
    pub service_name: String,
    pub pod_management_policy: PodManagementPolicy,
    pub update_strategy: StatefulSetUpdateStrategy,
    pub volume_claim_templates: Vec<PersistentVolumeClaimTemplate>,
    pub min_ready_seconds: u32,
    pub revision_history_limit: u32,
}

/// Pod management policy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PodManagementPolicy {
    #[default]
    OrderedReady,
    Parallel,
}

/// StatefulSet update strategy
#[derive(Clone, Debug)]
pub enum StatefulSetUpdateStrategy {
    OnDelete,
    RollingUpdate { partition: u32 },
}

impl Default for StatefulSetUpdateStrategy {
    fn default() -> Self {
        StatefulSetUpdateStrategy::RollingUpdate { partition: 0 }
    }
}

/// PVC template
#[derive(Clone, Debug)]
pub struct PersistentVolumeClaimTemplate {
    pub metadata: ObjectMeta,
    pub spec: PersistentVolumeClaimSpec,
}

/// PVC specification
#[derive(Clone, Debug)]
pub struct PersistentVolumeClaimSpec {
    pub access_modes: Vec<AccessMode>,
    pub storage_class_name: Option<String>,
    pub resources: VolumeResourceRequirements,
}

/// Access mode
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessMode {
    ReadWriteOnce,
    ReadOnlyMany,
    ReadWriteMany,
    ReadWriteOncePod,
}

/// Volume resource requirements
#[derive(Clone, Debug)]
pub struct VolumeResourceRequirements {
    pub requests: HashMap<String, String>,
    pub limits: HashMap<String, String>,
}

/// StatefulSet status
#[derive(Clone, Debug, Default)]
pub struct StatefulSetStatus {
    pub observed_generation: i64,
    pub replicas: u32,
    pub ready_replicas: u32,
    pub current_replicas: u32,
    pub updated_replicas: u32,
    pub current_revision: String,
    pub update_revision: String,
    pub collision_count: u32,
    pub conditions: Vec<StatefulSetCondition>,
}

/// StatefulSet condition
#[derive(Clone, Debug)]
pub struct StatefulSetCondition {
    pub condition_type: String,
    pub status: bool,
    pub last_transition_time: Instant,
    pub reason: String,
    pub message: String,
}

/// DaemonSet for running a pod on every node
#[derive(Clone, Debug)]
pub struct DaemonSet {
    pub metadata: ObjectMeta,
    pub spec: DaemonSetSpec,
    pub status: DaemonSetStatus,
}

/// DaemonSet specification
#[derive(Clone, Debug)]
pub struct DaemonSetSpec {
    pub selector: LabelSelector,
    pub template: PodTemplateSpec,
    pub update_strategy: DaemonSetUpdateStrategy,
    pub min_ready_seconds: u32,
    pub revision_history_limit: u32,
}

/// DaemonSet update strategy
#[derive(Clone, Debug)]
pub enum DaemonSetUpdateStrategy {
    OnDelete,
    RollingUpdate { max_unavailable: IntOrString, max_surge: IntOrString },
}

impl Default for DaemonSetUpdateStrategy {
    fn default() -> Self {
        DaemonSetUpdateStrategy::RollingUpdate {
            max_unavailable: IntOrString::Int(1),
            max_surge: IntOrString::Int(0),
        }
    }
}

/// DaemonSet status
#[derive(Clone, Debug, Default)]
pub struct DaemonSetStatus {
    pub current_number_scheduled: u32,
    pub number_misscheduled: u32,
    pub desired_number_scheduled: u32,
    pub number_ready: u32,
    pub observed_generation: i64,
    pub updated_number_scheduled: u32,
    pub number_available: u32,
    pub number_unavailable: u32,
    pub collision_count: u32,
    pub conditions: Vec<DaemonSetCondition>,
}

/// DaemonSet condition
#[derive(Clone, Debug)]
pub struct DaemonSetCondition {
    pub condition_type: String,
    pub status: bool,
    pub last_transition_time: Instant,
    pub reason: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deployment_new() {
        let deployment = Deployment::new("nginx", "default")
            .with_replicas(3)
            .with_selector("app", "nginx");

        assert_eq!(deployment.metadata.name, "nginx");
        assert_eq!(deployment.spec.replicas, 3);
    }

    #[test]
    fn test_deployment_complete() {
        let mut deployment = Deployment::new("test", "default");
        deployment.spec.replicas = 3;
        deployment.status.updated_replicas = 3;
        deployment.status.available_replicas = 3;
        deployment.status.ready_replicas = 3;

        assert!(deployment.is_complete());
        assert!(!deployment.is_progressing());
    }

    #[test]
    fn test_deployment_progressing() {
        let mut deployment = Deployment::new("test", "default");
        deployment.spec.replicas = 3;
        deployment.status.updated_replicas = 2;
        deployment.status.available_replicas = 2;

        assert!(deployment.is_progressing());
        assert!(!deployment.is_complete());
    }

    #[test]
    fn test_rollout_progress() {
        let mut deployment = Deployment::new("test", "default");
        deployment.spec.replicas = 4;
        deployment.status.updated_replicas = 2;

        assert!((deployment.rollout_progress() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_int_or_string_int() {
        let ios = IntOrString::Int(5);
        assert_eq!(ios.as_count(100), 5);
    }

    #[test]
    fn test_int_or_string_percentage() {
        let ios = IntOrString::String("25%".to_string());
        assert_eq!(ios.as_count(100), 25);
        assert_eq!(ios.as_count(10), 3); // Ceil of 2.5
    }

    #[test]
    fn test_rolling_update_defaults() {
        let ru = RollingUpdate::default();
        assert_eq!(ru.max_unavailable_count(4), 1);
        assert_eq!(ru.max_surge_count(4), 1);
    }

    #[test]
    fn test_deployment_controller_create() {
        let mut controller = DeploymentController::new();

        let deployment = Deployment::new("nginx", "default").with_replicas(3);
        controller.create(deployment).unwrap();

        let history = controller.get_history(&ResourceId::generate());
        // History is empty for unknown ID
        assert!(history.is_empty());
    }

    #[test]
    fn test_deployment_controller_scale() {
        let mut controller = DeploymentController::new();

        let deployment = Deployment::new("web", "default").with_replicas(3);
        let id = deployment.metadata.uid.clone();
        controller.create(deployment).unwrap();

        controller.scale(&id, 5).unwrap();

        let scaled = controller.deployments.get(&id).unwrap();
        assert_eq!(scaled.spec.replicas, 5);
    }

    #[test]
    fn test_deployment_controller_pause_resume() {
        let mut controller = DeploymentController::new();

        let deployment = Deployment::new("api", "default");
        let id = deployment.metadata.uid.clone();
        controller.create(deployment).unwrap();

        controller.pause(&id).unwrap();
        assert!(controller.deployments.get(&id).unwrap().spec.paused);

        controller.resume(&id).unwrap();
        assert!(!controller.deployments.get(&id).unwrap().spec.paused);
    }

    #[test]
    fn test_reconcile_paused() {
        let mut controller = DeploymentController::new();

        let mut deployment = Deployment::new("test", "default");
        deployment.spec.paused = true;
        let id = deployment.metadata.uid.clone();
        controller.create(deployment).unwrap();

        controller.pause(&id).unwrap();

        let result = controller.reconcile(&id).unwrap();
        assert!(matches!(result, ReconcileResult::Paused));
    }

    #[test]
    fn test_stateful_set() {
        let ss = StatefulSet {
            metadata: ObjectMeta::new("mysql", "default"),
            spec: StatefulSetSpec {
                replicas: 3,
                selector: LabelSelector::new().with_label("app", "mysql"),
                template: PodTemplateSpec::default(),
                service_name: "mysql".to_string(),
                pod_management_policy: PodManagementPolicy::OrderedReady,
                update_strategy: StatefulSetUpdateStrategy::default(),
                volume_claim_templates: vec![],
                min_ready_seconds: 0,
                revision_history_limit: 10,
            },
            status: StatefulSetStatus::default(),
        };

        assert_eq!(ss.spec.replicas, 3);
        assert_eq!(ss.spec.pod_management_policy, PodManagementPolicy::OrderedReady);
    }

    #[test]
    fn test_daemon_set() {
        let ds = DaemonSet {
            metadata: ObjectMeta::new("fluentd", "kube-system"),
            spec: DaemonSetSpec {
                selector: LabelSelector::new().with_label("app", "fluentd"),
                template: PodTemplateSpec::default(),
                update_strategy: DaemonSetUpdateStrategy::default(),
                min_ready_seconds: 0,
                revision_history_limit: 10,
            },
            status: DaemonSetStatus::default(),
        };

        assert!(matches!(ds.spec.update_strategy, DaemonSetUpdateStrategy::RollingUpdate { .. }));
    }

    #[test]
    fn test_deployment_strategy_recreate() {
        let strategy = DeploymentStrategy::Recreate;
        assert!(matches!(strategy, DeploymentStrategy::Recreate));
    }

    #[test]
    fn test_access_modes() {
        let modes = vec![
            AccessMode::ReadWriteOnce,
            AccessMode::ReadOnlyMany,
            AccessMode::ReadWriteMany,
        ];

        assert_eq!(modes.len(), 3);
    }
}
