//! Replication Controller and ReplicaSet
//!
//! Manages pod replication to ensure desired number of pod replicas are running.

use super::{ObjectMeta, ResourceId, LabelSelector, Pod, pod::PodSpec};
use super::deployment::PodTemplateSpec;
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::time::Instant;

/// ReplicaSet ensures a specified number of pod replicas are running
#[derive(Clone, Debug)]
pub struct ReplicaSet {
    pub metadata: ObjectMeta,
    pub spec: ReplicaSetSpec,
    pub status: ReplicaSetStatus,
}

impl ReplicaSet {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            spec: ReplicaSetSpec::default(),
            status: ReplicaSetStatus::default(),
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

    /// Check if the replica set has the desired number of ready replicas
    pub fn is_ready(&self) -> bool {
        self.status.ready_replicas >= self.spec.replicas
    }

    /// Get the number of pods that need to be created
    pub fn pods_to_create(&self) -> u32 {
        self.spec.replicas.saturating_sub(self.status.replicas)
    }

    /// Get the number of pods that need to be deleted
    pub fn pods_to_delete(&self) -> u32 {
        self.status.replicas.saturating_sub(self.spec.replicas)
    }
}

/// ReplicaSet specification
#[derive(Clone, Debug, Default)]
pub struct ReplicaSetSpec {
    /// Number of desired replicas
    pub replicas: u32,
    /// Label selector for pods
    pub selector: LabelSelector,
    /// Pod template
    pub template: PodTemplateSpec,
    /// Minimum seconds a pod must be ready before considered available
    pub min_ready_seconds: u32,
}

/// ReplicaSet status
#[derive(Clone, Debug, Default)]
pub struct ReplicaSetStatus {
    /// Number of pods
    pub replicas: u32,
    /// Number of fully labeled pods
    pub fully_labeled_replicas: u32,
    /// Number of ready pods
    pub ready_replicas: u32,
    /// Number of available pods
    pub available_replicas: u32,
    /// Observed generation
    pub observed_generation: i64,
    /// Conditions
    pub conditions: Vec<ReplicaSetCondition>,
}

/// ReplicaSet condition
#[derive(Clone, Debug)]
pub struct ReplicaSetCondition {
    pub condition_type: ReplicaSetConditionType,
    pub status: bool,
    pub last_transition_time: Instant,
    pub reason: String,
    pub message: String,
}

/// ReplicaSet condition type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReplicaSetConditionType {
    ReplicaFailure,
}

/// Replication controller for managing replica sets
#[derive(Debug)]
pub struct ReplicationController {
    replica_sets: HashMap<ResourceId, ReplicaSet>,
    managed_pods: HashMap<ResourceId, Vec<ResourceId>>,
}

impl ReplicationController {
    pub fn new() -> Self {
        Self {
            replica_sets: HashMap::new(),
            managed_pods: HashMap::new(),
        }
    }

    /// Create a replica set
    pub fn create(&mut self, rs: ReplicaSet) -> Result<()> {
        let id = rs.metadata.uid.clone();

        if self.replica_sets.contains_key(&id) {
            return Err(Error::Runtime("ReplicaSet already exists".into()));
        }

        self.managed_pods.insert(id.clone(), Vec::new());
        self.replica_sets.insert(id, rs);

        Ok(())
    }

    /// Delete a replica set
    pub fn delete(&mut self, rs_id: &ResourceId) -> Result<ReplicaSet> {
        self.managed_pods.remove(rs_id);
        self.replica_sets.remove(rs_id)
            .ok_or_else(|| Error::Runtime("ReplicaSet not found".into()))
    }

    /// Update replica set
    pub fn update(&mut self, rs: ReplicaSet) -> Result<()> {
        let id = rs.metadata.uid.clone();

        if !self.replica_sets.contains_key(&id) {
            return Err(Error::Runtime("ReplicaSet not found".into()));
        }

        self.replica_sets.insert(id, rs);
        Ok(())
    }

    /// Scale replica set
    pub fn scale(&mut self, rs_id: &ResourceId, replicas: u32) -> Result<()> {
        let rs = self.replica_sets.get_mut(rs_id)
            .ok_or_else(|| Error::Runtime("ReplicaSet not found".into()))?;

        rs.spec.replicas = replicas;
        Ok(())
    }

    /// Get replica set
    pub fn get(&self, rs_id: &ResourceId) -> Option<&ReplicaSet> {
        self.replica_sets.get(rs_id)
    }

    /// List all replica sets
    pub fn list(&self) -> Vec<&ReplicaSet> {
        self.replica_sets.values().collect()
    }

    /// List replica sets in a namespace
    pub fn list_in_namespace(&self, namespace: &str) -> Vec<&ReplicaSet> {
        self.replica_sets.values()
            .filter(|rs| rs.metadata.namespace == namespace)
            .collect()
    }

    /// Reconcile a replica set - returns actions to take
    pub fn reconcile(&self, rs_id: &ResourceId, current_pods: &[Pod]) -> Result<ReconcileAction> {
        let rs = self.replica_sets.get(rs_id)
            .ok_or_else(|| Error::Runtime("ReplicaSet not found".into()))?;

        // Count matching pods
        let matching_pods: Vec<_> = current_pods.iter()
            .filter(|p| rs.spec.selector.matches(&p.metadata.labels))
            .collect();

        let current_count = matching_pods.len() as u32;
        let desired = rs.spec.replicas;

        if current_count < desired {
            // Need to create pods
            let to_create = desired - current_count;
            Ok(ReconcileAction::CreatePods {
                count: to_create,
                template: rs.spec.template.clone(),
            })
        } else if current_count > desired {
            // Need to delete pods
            let to_delete = current_count - desired;
            // Select pods to delete (prefer pending, then newest)
            let mut candidates: Vec<_> = matching_pods.into_iter()
                .map(|p| p.metadata.uid.clone())
                .collect();

            // Simple: just take the first N pods
            candidates.truncate(to_delete as usize);

            Ok(ReconcileAction::DeletePods { pod_ids: candidates })
        } else {
            Ok(ReconcileAction::None)
        }
    }

    /// Update replica set status based on current pods
    pub fn update_status(&mut self, rs_id: &ResourceId, pods: &[Pod]) -> Result<()> {
        let rs = self.replica_sets.get_mut(rs_id)
            .ok_or_else(|| Error::Runtime("ReplicaSet not found".into()))?;

        // Count matching pods
        let matching: Vec<_> = pods.iter()
            .filter(|p| rs.spec.selector.matches(&p.metadata.labels))
            .collect();

        rs.status.replicas = matching.len() as u32;

        rs.status.ready_replicas = matching.iter()
            .filter(|p| p.is_ready())
            .count() as u32;

        rs.status.available_replicas = matching.iter()
            .filter(|p| p.is_running())
            .count() as u32;

        rs.status.fully_labeled_replicas = matching.len() as u32;

        Ok(())
    }

    /// Register a pod as managed by a replica set
    pub fn register_pod(&mut self, rs_id: &ResourceId, pod_id: ResourceId) {
        if let Some(pods) = self.managed_pods.get_mut(rs_id) {
            if !pods.contains(&pod_id) {
                pods.push(pod_id);
            }
        }
    }

    /// Unregister a pod from a replica set
    pub fn unregister_pod(&mut self, rs_id: &ResourceId, pod_id: &ResourceId) {
        if let Some(pods) = self.managed_pods.get_mut(rs_id) {
            pods.retain(|id| id != pod_id);
        }
    }

    /// Get pods managed by a replica set
    pub fn get_managed_pods(&self, rs_id: &ResourceId) -> Vec<ResourceId> {
        self.managed_pods.get(rs_id).cloned().unwrap_or_default()
    }
}

impl Default for ReplicationController {
    fn default() -> Self {
        Self::new()
    }
}

/// Action to take during reconciliation
#[derive(Clone, Debug)]
pub enum ReconcileAction {
    None,
    CreatePods { count: u32, template: PodTemplateSpec },
    DeletePods { pod_ids: Vec<ResourceId> },
}

/// Job for running pods to completion
#[derive(Clone, Debug)]
pub struct Job {
    pub metadata: ObjectMeta,
    pub spec: JobSpec,
    pub status: JobStatus,
}

impl Job {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            spec: JobSpec::default(),
            status: JobStatus::default(),
        }
    }

    /// Check if job is complete
    pub fn is_complete(&self) -> bool {
        self.status.conditions.iter().any(|c| {
            c.condition_type == JobConditionType::Complete && c.status
        })
    }

    /// Check if job has failed
    pub fn is_failed(&self) -> bool {
        self.status.conditions.iter().any(|c| {
            c.condition_type == JobConditionType::Failed && c.status
        })
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> f64 {
        let desired = self.spec.completions.unwrap_or(1);
        if desired == 0 {
            return 100.0;
        }
        (self.status.succeeded as f64 / desired as f64) * 100.0
    }
}

/// Job specification
#[derive(Clone, Debug)]
pub struct JobSpec {
    /// Number of successful completions required
    pub completions: Option<u32>,
    /// Number of pods that can run in parallel
    pub parallelism: Option<u32>,
    /// Number of retries before marking failed
    pub backoff_limit: u32,
    /// Active deadline in seconds
    pub active_deadline_seconds: Option<i64>,
    /// TTL after finished in seconds
    pub ttl_seconds_after_finished: Option<u32>,
    /// Completion mode
    pub completion_mode: CompletionMode,
    /// Suspend job
    pub suspend: bool,
    /// Pod template
    pub template: PodTemplateSpec,
    /// Selector
    pub selector: Option<LabelSelector>,
    /// Manual selector
    pub manual_selector: bool,
}

impl Default for JobSpec {
    fn default() -> Self {
        Self {
            completions: Some(1),
            parallelism: Some(1),
            backoff_limit: 6,
            active_deadline_seconds: None,
            ttl_seconds_after_finished: None,
            completion_mode: CompletionMode::NonIndexed,
            suspend: false,
            template: PodTemplateSpec::default(),
            selector: None,
            manual_selector: false,
        }
    }
}

/// Completion mode
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompletionMode {
    #[default]
    NonIndexed,
    Indexed,
}

/// Job status
#[derive(Clone, Debug, Default)]
pub struct JobStatus {
    /// Conditions
    pub conditions: Vec<JobCondition>,
    /// Start time
    pub start_time: Option<Instant>,
    /// Completion time
    pub completion_time: Option<Instant>,
    /// Number of active pods
    pub active: u32,
    /// Number of succeeded pods
    pub succeeded: u32,
    /// Number of failed pods
    pub failed: u32,
    /// Completed indexes (for indexed jobs)
    pub completed_indexes: String,
    /// Uncounted terminated pods
    pub uncounted_terminated_pods: Option<UncountedTerminatedPods>,
    /// Ready pods
    pub ready: u32,
}

/// Uncounted terminated pods
#[derive(Clone, Debug, Default)]
pub struct UncountedTerminatedPods {
    pub succeeded: Vec<ResourceId>,
    pub failed: Vec<ResourceId>,
}

/// Job condition
#[derive(Clone, Debug)]
pub struct JobCondition {
    pub condition_type: JobConditionType,
    pub status: bool,
    pub last_probe_time: Option<Instant>,
    pub last_transition_time: Instant,
    pub reason: String,
    pub message: String,
}

/// Job condition type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JobConditionType {
    Complete,
    Failed,
    Suspended,
}

/// CronJob for scheduled jobs
#[derive(Clone, Debug)]
pub struct CronJob {
    pub metadata: ObjectMeta,
    pub spec: CronJobSpec,
    pub status: CronJobStatus,
}

impl CronJob {
    pub fn new(name: impl Into<String>, namespace: impl Into<String>, schedule: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            spec: CronJobSpec {
                schedule: schedule.into(),
                ..Default::default()
            },
            status: CronJobStatus::default(),
        }
    }
}

/// CronJob specification
#[derive(Clone, Debug)]
pub struct CronJobSpec {
    /// Cron schedule (e.g., "*/5 * * * *")
    pub schedule: String,
    /// Timezone
    pub time_zone: Option<String>,
    /// Starting deadline seconds
    pub starting_deadline_seconds: Option<i64>,
    /// Concurrency policy
    pub concurrency_policy: ConcurrencyPolicy,
    /// Suspend cronjob
    pub suspend: bool,
    /// Job template
    pub job_template: JobTemplateSpec,
    /// Successful jobs history limit
    pub successful_jobs_history_limit: u32,
    /// Failed jobs history limit
    pub failed_jobs_history_limit: u32,
}

impl Default for CronJobSpec {
    fn default() -> Self {
        Self {
            schedule: "* * * * *".to_string(),
            time_zone: None,
            starting_deadline_seconds: None,
            concurrency_policy: ConcurrencyPolicy::Allow,
            suspend: false,
            job_template: JobTemplateSpec::default(),
            successful_jobs_history_limit: 3,
            failed_jobs_history_limit: 1,
        }
    }
}

/// Concurrency policy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ConcurrencyPolicy {
    #[default]
    Allow,
    Forbid,
    Replace,
}

/// Job template specification
#[derive(Clone, Debug, Default)]
pub struct JobTemplateSpec {
    pub metadata: ObjectMeta,
    pub spec: JobSpec,
}

/// CronJob status
#[derive(Clone, Debug, Default)]
pub struct CronJobStatus {
    /// Active jobs
    pub active: Vec<ObjectReference>,
    /// Last schedule time
    pub last_schedule_time: Option<Instant>,
    /// Last successful time
    pub last_successful_time: Option<Instant>,
}

/// Object reference
#[derive(Clone, Debug)]
pub struct ObjectReference {
    pub kind: String,
    pub namespace: String,
    pub name: String,
    pub uid: ResourceId,
    pub api_version: String,
    pub resource_version: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replica_set_new() {
        let rs = ReplicaSet::new("nginx", "default")
            .with_replicas(3)
            .with_selector("app", "nginx");

        assert_eq!(rs.metadata.name, "nginx");
        assert_eq!(rs.spec.replicas, 3);
    }

    #[test]
    fn test_replica_set_ready() {
        let mut rs = ReplicaSet::new("test", "default")
            .with_replicas(3);

        rs.status.ready_replicas = 3;
        assert!(rs.is_ready());

        rs.status.ready_replicas = 2;
        assert!(!rs.is_ready());
    }

    #[test]
    fn test_pods_to_create() {
        let mut rs = ReplicaSet::new("test", "default")
            .with_replicas(5);

        rs.status.replicas = 3;
        assert_eq!(rs.pods_to_create(), 2);
    }

    #[test]
    fn test_pods_to_delete() {
        let mut rs = ReplicaSet::new("test", "default")
            .with_replicas(3);

        rs.status.replicas = 5;
        assert_eq!(rs.pods_to_delete(), 2);
    }

    #[test]
    fn test_replication_controller_create() {
        let mut controller = ReplicationController::new();

        let rs = ReplicaSet::new("nginx", "default").with_replicas(3);
        controller.create(rs).unwrap();

        assert_eq!(controller.list().len(), 1);
    }

    #[test]
    fn test_replication_controller_scale() {
        let mut controller = ReplicationController::new();

        let rs = ReplicaSet::new("nginx", "default").with_replicas(3);
        let id = rs.metadata.uid.clone();
        controller.create(rs).unwrap();

        controller.scale(&id, 5).unwrap();

        let scaled = controller.get(&id).unwrap();
        assert_eq!(scaled.spec.replicas, 5);
    }

    #[test]
    fn test_reconcile_create() {
        let mut controller = ReplicationController::new();

        let rs = ReplicaSet::new("nginx", "default")
            .with_replicas(3)
            .with_selector("app", "nginx");
        let id = rs.metadata.uid.clone();
        controller.create(rs).unwrap();

        // No pods exist
        let action = controller.reconcile(&id, &[]).unwrap();
        match action {
            ReconcileAction::CreatePods { count, .. } => assert_eq!(count, 3),
            _ => panic!("Expected CreatePods action"),
        }
    }

    #[test]
    fn test_reconcile_delete() {
        let mut controller = ReplicationController::new();

        let rs = ReplicaSet::new("nginx", "default")
            .with_replicas(1)
            .with_selector("app", "nginx");
        let id = rs.metadata.uid.clone();
        controller.create(rs).unwrap();

        // Create 3 matching pods
        let pods: Vec<Pod> = (0..3).map(|i| {
            let mut pod = Pod::new(format!("nginx-{}", i), "default");
            pod.metadata.labels.insert("app".to_string(), "nginx".to_string());
            pod
        }).collect();

        let action = controller.reconcile(&id, &pods).unwrap();
        match action {
            ReconcileAction::DeletePods { pod_ids } => assert_eq!(pod_ids.len(), 2),
            _ => panic!("Expected DeletePods action"),
        }
    }

    #[test]
    fn test_reconcile_none() {
        let mut controller = ReplicationController::new();

        let rs = ReplicaSet::new("nginx", "default")
            .with_replicas(2)
            .with_selector("app", "nginx");
        let id = rs.metadata.uid.clone();
        controller.create(rs).unwrap();

        // Create 2 matching pods
        let pods: Vec<Pod> = (0..2).map(|i| {
            let mut pod = Pod::new(format!("nginx-{}", i), "default");
            pod.metadata.labels.insert("app".to_string(), "nginx".to_string());
            pod
        }).collect();

        let action = controller.reconcile(&id, &pods).unwrap();
        assert!(matches!(action, ReconcileAction::None));
    }

    #[test]
    fn test_job_new() {
        let job = Job::new("backup", "default");
        assert_eq!(job.metadata.name, "backup");
        assert_eq!(job.spec.completions, Some(1));
    }

    #[test]
    fn test_job_complete() {
        let mut job = Job::new("test", "default");
        job.status.conditions.push(JobCondition {
            condition_type: JobConditionType::Complete,
            status: true,
            last_probe_time: None,
            last_transition_time: Instant::now(),
            reason: "Success".into(),
            message: "Job completed".into(),
        });

        assert!(job.is_complete());
        assert!(!job.is_failed());
    }

    #[test]
    fn test_job_failed() {
        let mut job = Job::new("test", "default");
        job.status.conditions.push(JobCondition {
            condition_type: JobConditionType::Failed,
            status: true,
            last_probe_time: None,
            last_transition_time: Instant::now(),
            reason: "BackoffLimitExceeded".into(),
            message: "Job failed".into(),
        });

        assert!(job.is_failed());
        assert!(!job.is_complete());
    }

    #[test]
    fn test_job_completion_percentage() {
        let mut job = Job::new("test", "default");
        job.spec.completions = Some(10);
        job.status.succeeded = 5;

        assert!((job.completion_percentage() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_cronjob_new() {
        let cj = CronJob::new("backup", "default", "0 2 * * *");

        assert_eq!(cj.metadata.name, "backup");
        assert_eq!(cj.spec.schedule, "0 2 * * *");
    }

    #[test]
    fn test_managed_pods() {
        let mut controller = ReplicationController::new();

        let rs = ReplicaSet::new("nginx", "default");
        let rs_id = rs.metadata.uid.clone();
        controller.create(rs).unwrap();

        let pod_id = ResourceId::generate();
        controller.register_pod(&rs_id, pod_id.clone());

        let managed = controller.get_managed_pods(&rs_id);
        assert_eq!(managed.len(), 1);

        controller.unregister_pod(&rs_id, &pod_id);
        let managed = controller.get_managed_pods(&rs_id);
        assert!(managed.is_empty());
    }
}
