//! Kubernetes types for admission webhook.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Kubernetes object metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObjectMeta {
    /// Object name.
    pub name: Option<String>,
    /// Object namespace.
    pub namespace: Option<String>,
    /// Unique ID.
    pub uid: Option<String>,
    /// Resource version.
    pub resource_version: Option<String>,
    /// Labels.
    #[serde(default)]
    pub labels: HashMap<String, String>,
    /// Annotations.
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

/// Pod spec (simplified).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PodSpec {
    /// Containers.
    #[serde(default)]
    pub containers: Vec<Container>,
    /// Init containers.
    #[serde(default)]
    pub init_containers: Vec<Container>,
    /// Service account name.
    pub service_account_name: Option<String>,
    /// Volumes.
    #[serde(default)]
    pub volumes: Vec<Volume>,
}

/// Container spec.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Container {
    /// Container name.
    pub name: String,
    /// Container image.
    pub image: String,
    /// Image pull policy.
    pub image_pull_policy: Option<String>,
    /// Command.
    #[serde(default)]
    pub command: Vec<String>,
    /// Arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variables.
    #[serde(default)]
    pub env: Vec<EnvVar>,
    /// Ports.
    #[serde(default)]
    pub ports: Vec<ContainerPort>,
    /// Volume mounts.
    #[serde(default)]
    pub volume_mounts: Vec<VolumeMount>,
    /// Resources.
    pub resources: Option<ResourceRequirements>,
    /// Security context.
    pub security_context: Option<SecurityContext>,
    /// Readiness probe.
    pub readiness_probe: Option<Probe>,
    /// Liveness probe.
    pub liveness_probe: Option<Probe>,
}

/// Environment variable.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvVar {
    /// Variable name.
    pub name: String,
    /// Variable value.
    pub value: Option<String>,
    /// Value from source.
    pub value_from: Option<EnvVarSource>,
}

/// Environment variable source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EnvVarSource {
    /// Field reference.
    pub field_ref: Option<ObjectFieldSelector>,
    /// Config map key reference.
    pub config_map_key_ref: Option<ConfigMapKeySelector>,
    /// Secret key reference.
    pub secret_key_ref: Option<SecretKeySelector>,
}

/// Field selector.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ObjectFieldSelector {
    pub field_path: String,
}

/// ConfigMap key selector.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigMapKeySelector {
    pub name: String,
    pub key: String,
}

/// Secret key selector.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SecretKeySelector {
    pub name: String,
    pub key: String,
}

/// Container port.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContainerPort {
    /// Port name.
    pub name: Option<String>,
    /// Container port.
    pub container_port: i32,
    /// Protocol.
    pub protocol: Option<String>,
}

/// Volume mount.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VolumeMount {
    /// Mount name.
    pub name: String,
    /// Mount path.
    pub mount_path: String,
    /// Read only.
    #[serde(default)]
    pub read_only: bool,
    /// Sub path.
    pub sub_path: Option<String>,
}

/// Volume.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Volume {
    /// Volume name.
    pub name: String,
    /// Empty dir.
    pub empty_dir: Option<EmptyDirVolumeSource>,
    /// Secret.
    pub secret: Option<SecretVolumeSource>,
    /// ConfigMap.
    pub config_map: Option<ConfigMapVolumeSource>,
}

/// EmptyDir volume source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EmptyDirVolumeSource {
    pub medium: Option<String>,
    pub size_limit: Option<String>,
}

/// Secret volume source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SecretVolumeSource {
    pub secret_name: String,
    #[serde(default)]
    pub optional: bool,
}

/// ConfigMap volume source.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConfigMapVolumeSource {
    pub name: String,
    #[serde(default)]
    pub optional: bool,
}

/// Resource requirements.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceRequirements {
    /// Limits.
    #[serde(default)]
    pub limits: HashMap<String, String>,
    /// Requests.
    #[serde(default)]
    pub requests: HashMap<String, String>,
}

/// Security context.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SecurityContext {
    /// Run as user.
    pub run_as_user: Option<i64>,
    /// Run as group.
    pub run_as_group: Option<i64>,
    /// Run as non-root.
    pub run_as_non_root: Option<bool>,
    /// Privileged.
    pub privileged: Option<bool>,
    /// Capabilities.
    pub capabilities: Option<Capabilities>,
}

/// Capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Capabilities {
    #[serde(default)]
    pub add: Vec<String>,
    #[serde(default)]
    pub drop: Vec<String>,
}

/// Probe.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Probe {
    /// HTTP get action.
    pub http_get: Option<HTTPGetAction>,
    /// TCP socket action.
    pub tcp_socket: Option<TCPSocketAction>,
    /// Exec action.
    pub exec: Option<ExecAction>,
    /// Initial delay seconds.
    pub initial_delay_seconds: Option<i32>,
    /// Period seconds.
    pub period_seconds: Option<i32>,
    /// Timeout seconds.
    pub timeout_seconds: Option<i32>,
    /// Success threshold.
    pub success_threshold: Option<i32>,
    /// Failure threshold.
    pub failure_threshold: Option<i32>,
}

/// HTTP GET action.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HTTPGetAction {
    pub path: Option<String>,
    pub port: i32,
    pub scheme: Option<String>,
}

/// TCP socket action.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TCPSocketAction {
    pub port: i32,
}

/// Exec action.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecAction {
    #[serde(default)]
    pub command: Vec<String>,
}

/// Pod (simplified).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Pod {
    /// API version.
    pub api_version: Option<String>,
    /// Kind.
    pub kind: Option<String>,
    /// Metadata.
    #[serde(default)]
    pub metadata: ObjectMeta,
    /// Spec.
    #[serde(default)]
    pub spec: PodSpec,
}

/// AdmissionReview request/response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdmissionReview {
    /// API version.
    pub api_version: String,
    /// Kind.
    pub kind: String,
    /// Request.
    pub request: Option<AdmissionRequest>,
    /// Response.
    pub response: Option<AdmissionResponse>,
}

/// Admission request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdmissionRequest {
    /// Unique ID.
    pub uid: String,
    /// Kind.
    pub kind: GroupVersionKind,
    /// Resource.
    pub resource: GroupVersionResource,
    /// Sub-resource.
    pub sub_resource: Option<String>,
    /// Request kind.
    pub request_kind: Option<GroupVersionKind>,
    /// Request resource.
    pub request_resource: Option<GroupVersionResource>,
    /// Name.
    pub name: Option<String>,
    /// Namespace.
    pub namespace: Option<String>,
    /// Operation.
    pub operation: String,
    /// User info.
    pub user_info: UserInfo,
    /// Object.
    pub object: Option<serde_json::Value>,
    /// Old object.
    pub old_object: Option<serde_json::Value>,
    /// Dry run.
    #[serde(default)]
    pub dry_run: bool,
    /// Options.
    pub options: Option<serde_json::Value>,
}

/// Group version kind.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroupVersionKind {
    pub group: String,
    pub version: String,
    pub kind: String,
}

/// Group version resource.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GroupVersionResource {
    pub group: String,
    pub version: String,
    pub resource: String,
}

/// User info.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserInfo {
    pub username: Option<String>,
    pub uid: Option<String>,
    #[serde(default)]
    pub groups: Vec<String>,
    #[serde(default)]
    pub extra: HashMap<String, Vec<String>>,
}

/// Admission response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AdmissionResponse {
    /// Request UID.
    pub uid: String,
    /// Allowed.
    pub allowed: bool,
    /// Status.
    pub status: Option<Status>,
    /// Patch.
    pub patch: Option<String>,
    /// Patch type.
    pub patch_type: Option<String>,
    /// Audit annotations.
    #[serde(default)]
    pub audit_annotations: HashMap<String, String>,
    /// Warnings.
    #[serde(default)]
    pub warnings: Vec<String>,
}

/// Status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Status {
    pub code: Option<i32>,
    pub message: Option<String>,
    pub reason: Option<String>,
}

/// JSON Patch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonPatchOp {
    /// Operation type.
    pub op: String,
    /// Path.
    pub path: String,
    /// Value.
    pub value: Option<serde_json::Value>,
}

impl AdmissionReview {
    /// Create a new admission review response.
    pub fn response(uid: String, allowed: bool) -> Self {
        Self {
            api_version: "admission.k8s.io/v1".to_string(),
            kind: "AdmissionReview".to_string(),
            request: None,
            response: Some(AdmissionResponse {
                uid,
                allowed,
                status: None,
                patch: None,
                patch_type: None,
                audit_annotations: HashMap::new(),
                warnings: vec![],
            }),
        }
    }

    /// Create a response with a patch.
    pub fn response_with_patch(uid: String, patch: Vec<JsonPatchOp>) -> Self {
        let patch_json = serde_json::to_string(&patch).unwrap_or_default();
        let patch_base64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            patch_json.as_bytes(),
        );

        Self {
            api_version: "admission.k8s.io/v1".to_string(),
            kind: "AdmissionReview".to_string(),
            request: None,
            response: Some(AdmissionResponse {
                uid,
                allowed: true,
                status: None,
                patch: Some(patch_base64),
                patch_type: Some("JSONPatch".to_string()),
                audit_annotations: HashMap::new(),
                warnings: vec![],
            }),
        }
    }

    /// Create a rejection response.
    pub fn reject(uid: String, reason: &str) -> Self {
        Self {
            api_version: "admission.k8s.io/v1".to_string(),
            kind: "AdmissionReview".to_string(),
            request: None,
            response: Some(AdmissionResponse {
                uid,
                allowed: false,
                status: Some(Status {
                    code: Some(403),
                    message: Some(reason.to_string()),
                    reason: Some("Forbidden".to_string()),
                }),
                patch: None,
                patch_type: None,
                audit_annotations: HashMap::new(),
                warnings: vec![],
            }),
        }
    }
}

/// TrafficPolicy CRD (Custom Resource Definition).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrafficPolicy {
    /// API version.
    pub api_version: String,
    /// Kind.
    pub kind: String,
    /// Metadata.
    pub metadata: ObjectMeta,
    /// Spec.
    pub spec: TrafficPolicySpec,
    /// Status.
    pub status: Option<TrafficPolicyStatus>,
}

/// TrafficPolicy spec.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrafficPolicySpec {
    /// Target service.
    pub target_ref: PolicyTargetReference,
    /// Timeout settings.
    pub timeout: Option<TimeoutConfig>,
    /// Retry settings.
    pub retry: Option<RetryConfig>,
    /// Circuit breaker settings.
    pub circuit_breaker: Option<CircuitBreakerConfig>,
    /// Rate limit settings.
    pub rate_limit: Option<RateLimitConfig>,
}

/// Policy target reference.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PolicyTargetReference {
    /// Group.
    pub group: String,
    /// Kind.
    pub kind: String,
    /// Name.
    pub name: String,
    /// Namespace.
    pub namespace: Option<String>,
}

/// Timeout configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeoutConfig {
    /// Request timeout.
    pub request: String,
    /// Idle timeout.
    pub idle: Option<String>,
}

/// Retry configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetryConfig {
    /// Number of retries.
    pub attempts: i32,
    /// Per-retry timeout.
    pub per_try_timeout: String,
    /// Retry on conditions.
    #[serde(default)]
    pub retry_on: Vec<String>,
}

/// Circuit breaker configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CircuitBreakerConfig {
    /// Maximum connections.
    pub max_connections: Option<i32>,
    /// Maximum pending requests.
    pub max_pending_requests: Option<i32>,
    /// Maximum requests.
    pub max_requests: Option<i32>,
    /// Maximum retries.
    pub max_retries: Option<i32>,
}

/// Rate limit configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RateLimitConfig {
    /// Requests per second.
    pub requests_per_second: i32,
    /// Burst size.
    pub burst_size: Option<i32>,
}

/// TrafficPolicy status.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrafficPolicyStatus {
    /// Observed generation.
    pub observed_generation: i64,
    /// Conditions.
    #[serde(default)]
    pub conditions: Vec<Condition>,
}

/// Status condition.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Condition {
    /// Type.
    #[serde(rename = "type")]
    pub condition_type: String,
    /// Status.
    pub status: String,
    /// Last transition time.
    pub last_transition_time: Option<String>,
    /// Reason.
    pub reason: Option<String>,
    /// Message.
    pub message: Option<String>,
}
