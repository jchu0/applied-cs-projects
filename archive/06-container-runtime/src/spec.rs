//! OCI Runtime Specification types
//!
//! Implements the OCI Runtime Specification v1.0.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// OCI Runtime Specification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Spec {
    /// OCI version
    #[serde(rename = "ociVersion")]
    pub oci_version: String,
    /// Root filesystem
    pub root: Root,
    /// Mounts
    #[serde(default)]
    pub mounts: Vec<Mount>,
    /// Process to run
    pub process: Process,
    /// Hostname
    #[serde(default)]
    pub hostname: String,
    /// Linux-specific configuration
    #[serde(default)]
    pub linux: Option<Linux>,
    /// Annotations
    #[serde(default)]
    pub annotations: HashMap<String, String>,
}

impl Default for Spec {
    fn default() -> Self {
        Self {
            oci_version: "1.0.0".to_string(),
            root: Root::default(),
            mounts: vec![
                // Default mounts
                Mount {
                    destination: "/proc".into(),
                    mount_type: "proc".to_string(),
                    source: "proc".into(),
                    options: vec!["nosuid".to_string(), "noexec".to_string(), "nodev".to_string()],
                },
                Mount {
                    destination: "/dev".into(),
                    mount_type: "tmpfs".to_string(),
                    source: "tmpfs".into(),
                    options: vec!["nosuid".to_string(), "strictatime".to_string(), "mode=755".to_string(), "size=65536k".to_string()],
                },
                Mount {
                    destination: "/sys".into(),
                    mount_type: "sysfs".to_string(),
                    source: "sysfs".into(),
                    options: vec!["nosuid".to_string(), "noexec".to_string(), "nodev".to_string(), "ro".to_string()],
                },
            ],
            process: Process::default(),
            hostname: "container".to_string(),
            linux: Some(Linux::default()),
            annotations: HashMap::new(),
        }
    }
}

/// Root filesystem
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Root {
    /// Path to rootfs
    pub path: PathBuf,
    /// Mount as read-only
    #[serde(default)]
    pub readonly: bool,
}

impl Default for Root {
    fn default() -> Self {
        Self {
            path: "rootfs".into(),
            readonly: false,
        }
    }
}

/// Mount point
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Mount {
    /// Destination path in container
    pub destination: PathBuf,
    /// Type of mount
    #[serde(rename = "type", default)]
    pub mount_type: String,
    /// Source path
    #[serde(default)]
    pub source: PathBuf,
    /// Mount options
    #[serde(default)]
    pub options: Vec<String>,
}

/// Process to run
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Process {
    /// Allocate terminal
    #[serde(default)]
    pub terminal: bool,
    /// User to run as
    pub user: User,
    /// Arguments (first is executable)
    pub args: Vec<String>,
    /// Environment variables
    #[serde(default)]
    pub env: Vec<String>,
    /// Working directory
    pub cwd: PathBuf,
    /// Capabilities
    #[serde(default)]
    pub capabilities: Option<Capabilities>,
    /// Resource limits
    #[serde(default)]
    pub rlimits: Vec<Rlimit>,
    /// No new privileges
    #[serde(rename = "noNewPrivileges", default)]
    pub no_new_privileges: bool,
}

impl Default for Process {
    fn default() -> Self {
        Self {
            terminal: false,
            user: User::default(),
            args: vec!["/bin/sh".to_string()],
            env: vec![
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
                "TERM=xterm".to_string(),
            ],
            cwd: "/".into(),
            capabilities: None,
            rlimits: vec![],
            no_new_privileges: true,
        }
    }
}

/// User specification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct User {
    /// User ID
    pub uid: u32,
    /// Group ID
    pub gid: u32,
    /// Additional group IDs
    #[serde(rename = "additionalGids", default)]
    pub additional_gids: Vec<u32>,
}

impl Default for User {
    fn default() -> Self {
        Self {
            uid: 0,
            gid: 0,
            additional_gids: vec![],
        }
    }
}

/// Linux capabilities
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capabilities {
    /// Bounding set
    #[serde(default)]
    pub bounding: Vec<String>,
    /// Effective set
    #[serde(default)]
    pub effective: Vec<String>,
    /// Inheritable set
    #[serde(default)]
    pub inheritable: Vec<String>,
    /// Permitted set
    #[serde(default)]
    pub permitted: Vec<String>,
    /// Ambient set
    #[serde(default)]
    pub ambient: Vec<String>,
}

/// Resource limit
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rlimit {
    /// Type of limit (e.g., "RLIMIT_NOFILE")
    #[serde(rename = "type")]
    pub limit_type: String,
    /// Hard limit
    pub hard: u64,
    /// Soft limit
    pub soft: u64,
}

/// Linux-specific configuration
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Linux {
    /// Namespaces to create
    #[serde(default)]
    pub namespaces: Vec<Namespace>,
    /// UID mappings
    #[serde(rename = "uidMappings", default)]
    pub uid_mappings: Vec<IdMapping>,
    /// GID mappings
    #[serde(rename = "gidMappings", default)]
    pub gid_mappings: Vec<IdMapping>,
    /// Devices
    #[serde(default)]
    pub devices: Vec<Device>,
    /// Cgroups path
    #[serde(rename = "cgroupsPath", default)]
    pub cgroups_path: String,
    /// Resources
    #[serde(default)]
    pub resources: Option<Resources>,
    /// Masked paths
    #[serde(rename = "maskedPaths", default)]
    pub masked_paths: Vec<String>,
    /// Read-only paths
    #[serde(rename = "readonlyPaths", default)]
    pub readonly_paths: Vec<String>,
}

/// Namespace configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Namespace {
    /// Namespace type
    #[serde(rename = "type")]
    pub ns_type: String,
    /// Path to existing namespace to join
    #[serde(default)]
    pub path: Option<PathBuf>,
}

/// ID mapping for user/group namespaces
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IdMapping {
    /// Container ID
    #[serde(rename = "containerID")]
    pub container_id: u32,
    /// Host ID
    #[serde(rename = "hostID")]
    pub host_id: u32,
    /// Size of mapping
    pub size: u32,
}

/// Device configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Device {
    /// Device type
    #[serde(rename = "type")]
    pub device_type: String,
    /// Path in container
    pub path: PathBuf,
    /// Major number
    pub major: i64,
    /// Minor number
    pub minor: i64,
    /// File mode
    #[serde(rename = "fileMode")]
    pub file_mode: Option<u32>,
    /// Owner UID
    pub uid: Option<u32>,
    /// Owner GID
    pub gid: Option<u32>,
}

/// Resource configuration
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct Resources {
    /// Memory configuration
    #[serde(default)]
    pub memory: Option<MemoryResources>,
    /// CPU configuration
    #[serde(default)]
    pub cpu: Option<CpuResources>,
    /// PIDs configuration
    #[serde(default)]
    pub pids: Option<PidsResources>,
}

/// Memory resources
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryResources {
    /// Memory limit in bytes
    #[serde(default)]
    pub limit: Option<i64>,
    /// Memory reservation
    #[serde(default)]
    pub reservation: Option<i64>,
    /// Swap limit
    #[serde(default)]
    pub swap: Option<i64>,
}

/// CPU resources
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CpuResources {
    /// CPU shares
    #[serde(default)]
    pub shares: Option<u64>,
    /// CPU quota
    #[serde(default)]
    pub quota: Option<i64>,
    /// CPU period
    #[serde(default)]
    pub period: Option<u64>,
    /// CPU affinity
    #[serde(default)]
    pub cpus: Option<String>,
    /// NUMA memory nodes
    #[serde(default)]
    pub mems: Option<String>,
}

/// PIDs resources
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PidsResources {
    /// Max number of PIDs
    pub limit: i64,
}

impl Spec {
    /// Load spec from a JSON file
    pub fn load(path: impl AsRef<std::path::Path>) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let spec: Spec = serde_json::from_str(&content)?;
        Ok(spec)
    }

    /// Save spec to a JSON file
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}
