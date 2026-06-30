//! Pod Management
//!
//! Pods are the smallest deployable units in the orchestrator.
//! A pod contains one or more containers that share storage and network.

use super::{ObjectMeta, ResourceId, LabelSelector};
use super::scheduler::{Affinity, Toleration};
use super::resource::ResourceRequirements;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Pod represents a group of containers
#[derive(Clone, Debug)]
pub struct Pod {
    pub metadata: ObjectMeta,
    pub spec: PodSpec,
    pub status: PodStatus,
}

impl Pod {
    /// Create a new pod
    pub fn new(name: impl Into<String>, namespace: impl Into<String>) -> Self {
        Self {
            metadata: ObjectMeta::new(name, namespace),
            spec: PodSpec::default(),
            status: PodStatus::default(),
        }
    }

    /// Add a container to the pod
    pub fn with_container(mut self, container: ContainerSpec) -> Self {
        self.spec.containers.push(container);
        self
    }

    /// Check if pod is running
    pub fn is_running(&self) -> bool {
        self.status.phase == PodPhase::Running
    }

    /// Check if pod is terminated
    pub fn is_terminated(&self) -> bool {
        matches!(self.status.phase, PodPhase::Succeeded | PodPhase::Failed)
    }

    /// Check if pod is pending
    pub fn is_pending(&self) -> bool {
        self.status.phase == PodPhase::Pending
    }

    /// Get pod conditions
    pub fn get_condition(&self, condition_type: PodConditionType) -> Option<&PodCondition> {
        self.status.conditions.iter().find(|c| c.condition_type == condition_type)
    }

    /// Check if pod is ready
    pub fn is_ready(&self) -> bool {
        self.get_condition(PodConditionType::Ready)
            .map(|c| c.status)
            .unwrap_or(false)
    }
}

/// Pod specification
#[derive(Clone, Debug, Default)]
pub struct PodSpec {
    /// Containers in the pod
    pub containers: Vec<ContainerSpec>,
    /// Init containers
    pub init_containers: Vec<ContainerSpec>,
    /// Volumes
    pub volumes: Vec<Volume>,
    /// Node selector
    pub node_selector: Option<LabelSelector>,
    /// Service account name
    pub service_account_name: String,
    /// Restart policy
    pub restart_policy: RestartPolicy,
    /// Termination grace period in seconds
    pub termination_grace_period_seconds: i64,
    /// Pod priority
    pub priority: Option<i32>,
    /// Priority class name
    pub priority_class_name: Option<String>,
    /// Affinity rules
    pub affinity: Option<Affinity>,
    /// Tolerations
    pub tolerations: Vec<Toleration>,
    /// Host network mode
    pub host_network: bool,
    /// Host PID mode
    pub host_pid: bool,
    /// Host IPC mode
    pub host_ipc: bool,
    /// DNS policy
    pub dns_policy: DNSPolicy,
    /// DNS config
    pub dns_config: Option<DNSConfig>,
    /// Security context
    pub security_context: Option<PodSecurityContext>,
    /// Hostname
    pub hostname: String,
    /// Subdomain
    pub subdomain: String,
    /// Liveness probe for the pod
    pub liveness_probe: Option<super::health::HealthCheck>,
    /// Readiness probe for the pod
    pub readiness_probe: Option<super::health::HealthCheck>,
}

impl PodSpec {
    /// Calculate total CPU request
    pub fn total_cpu_request(&self) -> u64 {
        self.containers.iter()
            .filter_map(|c| c.resources.as_ref())
            .filter_map(|r| r.requests.as_ref())
            .map(|r| r.cpu_millis)
            .sum()
    }

    /// Calculate total memory request
    pub fn total_memory_request(&self) -> u64 {
        self.containers.iter()
            .filter_map(|c| c.resources.as_ref())
            .filter_map(|r| r.requests.as_ref())
            .map(|r| r.memory_bytes)
            .sum()
    }

    /// Calculate total CPU limit
    pub fn total_cpu_limit(&self) -> u64 {
        self.containers.iter()
            .filter_map(|c| c.resources.as_ref())
            .filter_map(|r| r.limits.as_ref())
            .map(|l| l.cpu_millis)
            .sum()
    }

    /// Calculate total memory limit
    pub fn total_memory_limit(&self) -> u64 {
        self.containers.iter()
            .filter_map(|c| c.resources.as_ref())
            .filter_map(|r| r.limits.as_ref())
            .map(|l| l.memory_bytes)
            .sum()
    }
}

/// Container specification
#[derive(Clone, Debug, Default)]
pub struct ContainerSpec {
    /// Container name
    pub name: String,
    /// Container image
    pub image: String,
    /// Image pull policy
    pub image_pull_policy: ImagePullPolicy,
    /// Command to run
    pub command: Vec<String>,
    /// Arguments
    pub args: Vec<String>,
    /// Working directory
    pub working_dir: String,
    /// Environment variables
    pub env: Vec<EnvVar>,
    /// Environment from sources
    pub env_from: Vec<EnvFromSource>,
    /// Resource requirements
    pub resources: Option<ResourceRequirements>,
    /// Volume mounts
    pub volume_mounts: Vec<VolumeMount>,
    /// Ports
    pub ports: Vec<ContainerPort>,
    /// Liveness probe
    pub liveness_probe: Option<Probe>,
    /// Readiness probe
    pub readiness_probe: Option<Probe>,
    /// Startup probe
    pub startup_probe: Option<Probe>,
    /// Security context
    pub security_context: Option<SecurityContext>,
    /// Whether to allocate a TTY
    pub tty: bool,
    /// Whether stdin is open
    pub stdin: bool,
}

impl ContainerSpec {
    pub fn new(name: impl Into<String>, image: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            image: image.into(),
            image_pull_policy: ImagePullPolicy::IfNotPresent,
            ..Default::default()
        }
    }

    pub fn with_command(mut self, command: Vec<String>) -> Self {
        self.command = command;
        self
    }

    pub fn with_resources(mut self, resources: ResourceRequirements) -> Self {
        self.resources = Some(resources);
        self
    }

    pub fn with_port(mut self, port: ContainerPort) -> Self {
        self.ports.push(port);
        self
    }

    pub fn with_env(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push(EnvVar {
            name: name.into(),
            value: Some(value.into()),
            value_from: None,
        });
        self
    }
}

/// Image pull policy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImagePullPolicy {
    Always,
    #[default]
    IfNotPresent,
    Never,
}

/// Environment variable
#[derive(Clone, Debug)]
pub struct EnvVar {
    pub name: String,
    pub value: Option<String>,
    pub value_from: Option<EnvVarSource>,
}

/// Environment variable source
#[derive(Clone, Debug)]
pub enum EnvVarSource {
    FieldRef { field_path: String },
    ConfigMapKeyRef { name: String, key: String },
    SecretKeyRef { name: String, key: String },
    ResourceFieldRef { container_name: String, resource: String },
}

/// Environment from source
#[derive(Clone, Debug)]
pub struct EnvFromSource {
    pub prefix: String,
    pub config_map_ref: Option<String>,
    pub secret_ref: Option<String>,
}

/// Volume mount
#[derive(Clone, Debug)]
pub struct VolumeMount {
    pub name: String,
    pub mount_path: String,
    pub sub_path: String,
    pub read_only: bool,
}

/// Container port
#[derive(Clone, Debug)]
pub struct ContainerPort {
    pub name: String,
    pub container_port: u16,
    pub host_port: Option<u16>,
    pub protocol: Protocol,
}

impl ContainerPort {
    pub fn new(name: impl Into<String>, port: u16) -> Self {
        Self {
            name: name.into(),
            container_port: port,
            host_port: None,
            protocol: Protocol::TCP,
        }
    }
}

/// Network protocol
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Protocol {
    #[default]
    TCP,
    UDP,
    SCTP,
}

/// Container probe
#[derive(Clone, Debug)]
pub struct Probe {
    pub handler: ProbeHandler,
    pub initial_delay_seconds: i32,
    pub timeout_seconds: i32,
    pub period_seconds: i32,
    pub success_threshold: i32,
    pub failure_threshold: i32,
}

impl Default for Probe {
    fn default() -> Self {
        Self {
            handler: ProbeHandler::Exec { command: vec![] },
            initial_delay_seconds: 0,
            timeout_seconds: 1,
            period_seconds: 10,
            success_threshold: 1,
            failure_threshold: 3,
        }
    }
}

/// Probe handler
#[derive(Clone, Debug)]
pub enum ProbeHandler {
    Exec { command: Vec<String> },
    HttpGet { path: String, port: u16, scheme: String, headers: Vec<(String, String)> },
    TcpSocket { port: u16 },
    Grpc { port: u16, service: Option<String> },
}

/// Volume
#[derive(Clone, Debug)]
pub struct Volume {
    pub name: String,
    pub volume_source: VolumeSource,
}

/// Volume source
#[derive(Clone, Debug)]
pub enum VolumeSource {
    EmptyDir { medium: String, size_limit: Option<u64> },
    HostPath { path: String, host_path_type: HostPathType },
    ConfigMap { name: String, items: Vec<KeyToPath>, default_mode: u32 },
    Secret { name: String, items: Vec<KeyToPath>, default_mode: u32 },
    PersistentVolumeClaim { claim_name: String, read_only: bool },
    Projected { sources: Vec<ProjectedVolumeSource> },
}

/// Host path type
#[derive(Clone, Debug)]
pub enum HostPathType {
    Unset,
    DirectoryOrCreate,
    Directory,
    FileOrCreate,
    File,
    Socket,
    CharDevice,
    BlockDevice,
}

/// Key to path mapping
#[derive(Clone, Debug)]
pub struct KeyToPath {
    pub key: String,
    pub path: String,
    pub mode: Option<u32>,
}

/// Projected volume source
#[derive(Clone, Debug)]
pub enum ProjectedVolumeSource {
    ConfigMap { name: String, items: Vec<KeyToPath> },
    Secret { name: String, items: Vec<KeyToPath> },
    Downward { items: Vec<DownwardAPIVolumeFile> },
    ServiceAccountToken { path: String, expiration_seconds: i64, audience: String },
}

/// Downward API volume file
#[derive(Clone, Debug)]
pub struct DownwardAPIVolumeFile {
    pub path: String,
    pub field_ref: Option<String>,
    pub resource_field_ref: Option<String>,
}

/// Security context for container
#[derive(Clone, Debug, Default)]
pub struct SecurityContext {
    pub run_as_user: Option<i64>,
    pub run_as_group: Option<i64>,
    pub run_as_non_root: bool,
    pub read_only_root_filesystem: bool,
    pub privileged: bool,
    pub allow_privilege_escalation: bool,
    pub capabilities: Option<Capabilities>,
}

/// Capabilities
#[derive(Clone, Debug)]
pub struct Capabilities {
    pub add: Vec<String>,
    pub drop: Vec<String>,
}

/// Pod security context
#[derive(Clone, Debug, Default)]
pub struct PodSecurityContext {
    pub run_as_user: Option<i64>,
    pub run_as_group: Option<i64>,
    pub run_as_non_root: bool,
    pub fs_group: Option<i64>,
    pub supplemental_groups: Vec<i64>,
    pub sysctls: Vec<Sysctl>,
}

/// Sysctl
#[derive(Clone, Debug)]
pub struct Sysctl {
    pub name: String,
    pub value: String,
}

/// Restart policy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RestartPolicy {
    #[default]
    Always,
    OnFailure,
    Never,
}

/// DNS policy
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DNSPolicy {
    #[default]
    ClusterFirst,
    ClusterFirstWithHostNet,
    Default,
    None,
}

/// DNS config
#[derive(Clone, Debug)]
pub struct DNSConfig {
    pub nameservers: Vec<String>,
    pub searches: Vec<String>,
    pub options: Vec<DNSConfigOption>,
}

/// DNS config option
#[derive(Clone, Debug)]
pub struct DNSConfigOption {
    pub name: String,
    pub value: Option<String>,
}

/// Pod status
#[derive(Clone, Debug, Default)]
pub struct PodStatus {
    /// Current phase
    pub phase: PodPhase,
    /// Conditions
    pub conditions: Vec<PodCondition>,
    /// Host IP
    pub host_ip: Option<String>,
    /// Pod IP
    pub pod_ip: Option<String>,
    /// Pod IPs
    pub pod_ips: Vec<String>,
    /// Container statuses
    pub container_statuses: Vec<ContainerStatus>,
    /// Init container statuses
    pub init_container_statuses: Vec<ContainerStatus>,
    /// QoS class
    pub qos_class: QoSClass,
    /// Start time
    pub start_time: Option<Instant>,
    /// Message
    pub message: String,
    /// Reason
    pub reason: String,
    /// Node name where pod is scheduled
    pub node_name: Option<String>,
}

/// Pod phase
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PodPhase {
    #[default]
    Pending,
    Running,
    Succeeded,
    Failed,
    Unknown,
}

/// Pod condition
#[derive(Clone, Debug)]
pub struct PodCondition {
    pub condition_type: PodConditionType,
    pub status: bool,
    pub last_probe_time: Option<Instant>,
    pub last_transition_time: Instant,
    pub reason: String,
    pub message: String,
}

/// Pod condition type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PodConditionType {
    PodScheduled,
    Initialized,
    ContainersReady,
    Ready,
}

/// Container status
#[derive(Clone, Debug)]
pub struct ContainerStatus {
    pub name: String,
    pub state: ContainerState,
    pub last_state: Option<ContainerState>,
    pub ready: bool,
    pub restart_count: i32,
    pub image: String,
    pub image_id: String,
    pub container_id: Option<String>,
    pub started: bool,
}

/// Container state
#[derive(Clone, Debug)]
pub enum ContainerState {
    Waiting { reason: String, message: String },
    Running { started_at: Instant },
    Terminated { exit_code: i32, signal: Option<i32>, reason: String, message: String, started_at: Instant, finished_at: Instant },
}

/// QoS class
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QoSClass {
    Guaranteed,
    Burstable,
    #[default]
    BestEffort,
}

impl Pod {
    /// Calculate QoS class based on resource specifications
    pub fn calculate_qos_class(&self) -> QoSClass {
        let mut guaranteed = true;
        let mut burstable = false;

        for container in &self.spec.containers {
            match &container.resources {
                Some(res) => {
                    let has_requests = res.requests.is_some();
                    let has_limits = res.limits.is_some();

                    if has_requests || has_limits {
                        burstable = true;
                    }

                    if let (Some(req), Some(lim)) = (&res.requests, &res.limits) {
                        if req.cpu_millis != lim.cpu_millis || req.memory_bytes != lim.memory_bytes {
                            guaranteed = false;
                        }
                    } else {
                        guaranteed = false;
                    }
                }
                None => {
                    guaranteed = false;
                }
            }
        }

        if guaranteed && burstable {
            QoSClass::Guaranteed
        } else if burstable {
            QoSClass::Burstable
        } else {
            QoSClass::BestEffort
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::resource::{ResourceRequest, ResourceLimit};

    #[test]
    fn test_pod_new() {
        let pod = Pod::new("test-pod", "default");
        assert_eq!(pod.metadata.name, "test-pod");
        assert_eq!(pod.metadata.namespace, "default");
        assert!(pod.is_pending());
    }

    #[test]
    fn test_pod_with_container() {
        let container = ContainerSpec::new("nginx", "nginx:latest")
            .with_port(ContainerPort::new("http", 80));

        let pod = Pod::new("web", "default").with_container(container);
        assert_eq!(pod.spec.containers.len(), 1);
        assert_eq!(pod.spec.containers[0].name, "nginx");
    }

    #[test]
    fn test_container_spec_builder() {
        let container = ContainerSpec::new("app", "myapp:1.0")
            .with_command(vec!["./app".into()])
            .with_env("DEBUG", "true")
            .with_port(ContainerPort::new("http", 8080));

        assert_eq!(container.name, "app");
        assert_eq!(container.command[0], "./app");
        assert_eq!(container.env.len(), 1);
        assert_eq!(container.ports.len(), 1);
    }

    #[test]
    fn test_total_resources() {
        let container1 = ContainerSpec {
            name: "c1".into(),
            resources: Some(ResourceRequirements {
                requests: Some(ResourceRequest { cpu_millis: 100, memory_bytes: 256 * 1024 * 1024 }),
                limits: Some(ResourceLimit { cpu_millis: 200, memory_bytes: 512 * 1024 * 1024 }),
            }),
            ..Default::default()
        };

        let container2 = ContainerSpec {
            name: "c2".into(),
            resources: Some(ResourceRequirements {
                requests: Some(ResourceRequest { cpu_millis: 200, memory_bytes: 128 * 1024 * 1024 }),
                limits: Some(ResourceLimit { cpu_millis: 400, memory_bytes: 256 * 1024 * 1024 }),
            }),
            ..Default::default()
        };

        let mut pod = Pod::new("test", "default");
        pod.spec.containers = vec![container1, container2];

        assert_eq!(pod.spec.total_cpu_request(), 300);
        assert_eq!(pod.spec.total_cpu_limit(), 600);
    }

    #[test]
    fn test_qos_class_guaranteed() {
        let container = ContainerSpec {
            name: "guaranteed".into(),
            resources: Some(ResourceRequirements {
                requests: Some(ResourceRequest { cpu_millis: 100, memory_bytes: 256 * 1024 * 1024 }),
                limits: Some(ResourceLimit { cpu_millis: 100, memory_bytes: 256 * 1024 * 1024 }),
            }),
            ..Default::default()
        };

        let mut pod = Pod::new("test", "default");
        pod.spec.containers = vec![container];

        assert_eq!(pod.calculate_qos_class(), QoSClass::Guaranteed);
    }

    #[test]
    fn test_qos_class_burstable() {
        let container = ContainerSpec {
            name: "burstable".into(),
            resources: Some(ResourceRequirements {
                requests: Some(ResourceRequest { cpu_millis: 100, memory_bytes: 256 * 1024 * 1024 }),
                limits: Some(ResourceLimit { cpu_millis: 200, memory_bytes: 512 * 1024 * 1024 }),
            }),
            ..Default::default()
        };

        let mut pod = Pod::new("test", "default");
        pod.spec.containers = vec![container];

        assert_eq!(pod.calculate_qos_class(), QoSClass::Burstable);
    }

    #[test]
    fn test_qos_class_best_effort() {
        let container = ContainerSpec {
            name: "besteffort".into(),
            resources: None,
            ..Default::default()
        };

        let mut pod = Pod::new("test", "default");
        pod.spec.containers = vec![container];

        assert_eq!(pod.calculate_qos_class(), QoSClass::BestEffort);
    }

    #[test]
    fn test_pod_phases() {
        let mut pod = Pod::new("test", "default");
        assert!(pod.is_pending());
        assert!(!pod.is_running());
        assert!(!pod.is_terminated());

        pod.status.phase = PodPhase::Running;
        assert!(pod.is_running());

        pod.status.phase = PodPhase::Succeeded;
        assert!(pod.is_terminated());

        pod.status.phase = PodPhase::Failed;
        assert!(pod.is_terminated());
    }

    #[test]
    fn test_probe_default() {
        let probe = Probe::default();
        assert_eq!(probe.period_seconds, 10);
        assert_eq!(probe.failure_threshold, 3);
    }

    #[test]
    fn test_restart_policy() {
        let spec = PodSpec::default();
        assert_eq!(spec.restart_policy, RestartPolicy::Always);
    }
}
