//! Container management
//!
//! Provides container lifecycle management.

use std::path::PathBuf;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::namespace::NamespaceSet;
use crate::spec::Spec;
use crate::{ContainerId, Result};

/// Container state
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ContainerState {
    /// Container is being created
    Creating,
    /// Container has been created but not started
    Created,
    /// Container is running
    Running,
    /// Container has stopped
    Stopped,
    /// Container is paused
    Paused,
}

impl std::fmt::Display for ContainerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContainerState::Creating => write!(f, "creating"),
            ContainerState::Created => write!(f, "created"),
            ContainerState::Running => write!(f, "running"),
            ContainerState::Stopped => write!(f, "stopped"),
            ContainerState::Paused => write!(f, "paused"),
        }
    }
}

/// Container runtime state (OCI state)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    /// OCI version
    #[serde(rename = "ociVersion")]
    pub oci_version: String,
    /// Container ID
    pub id: String,
    /// Container status
    pub status: ContainerState,
    /// PID of the container init process
    pub pid: Option<i32>,
    /// Bundle path
    pub bundle: PathBuf,
    /// Annotations
    #[serde(default)]
    pub annotations: std::collections::HashMap<String, String>,
}

/// Container configuration
#[derive(Clone, Debug)]
pub struct ContainerConfig {
    /// OCI runtime spec
    pub spec: Spec,
    /// Rootfs path
    pub rootfs: PathBuf,
    /// Use terminal
    pub terminal: bool,
    /// Console socket path
    pub console_socket: Option<PathBuf>,
}

/// Container instance
#[derive(Debug)]
pub struct Container {
    /// Container ID
    pub id: ContainerId,
    /// Current state
    pub state: ContainerState,
    /// Bundle path
    pub bundle: PathBuf,
    /// Rootfs path
    pub rootfs: PathBuf,
    /// PID of init process
    pub pid: Option<i32>,
    /// Creation time
    pub created: SystemTime,
    /// Namespaces
    pub namespaces: NamespaceSet,
    /// Cgroup path
    pub cgroup_path: PathBuf,
}

impl Container {
    /// Create a new container
    pub fn new(
        id: ContainerId,
        bundle: PathBuf,
        rootfs: PathBuf,
        cgroup_path: PathBuf,
    ) -> Self {
        Self {
            id,
            state: ContainerState::Creating,
            bundle,
            rootfs,
            pid: None,
            created: SystemTime::now(),
            namespaces: NamespaceSet::default(),
            cgroup_path,
        }
    }

    /// Get OCI state
    pub fn oci_state(&self) -> State {
        State {
            oci_version: "1.0.0".to_string(),
            id: self.id.0.clone(),
            status: self.state.clone(),
            pid: self.pid,
            bundle: self.bundle.clone(),
            annotations: std::collections::HashMap::new(),
        }
    }

    /// Transition to created state
    pub fn mark_created(&mut self) -> Result<()> {
        if self.state != ContainerState::Creating {
            return Err(crate::Error::InvalidState {
                expected: "creating".to_string(),
                actual: self.state.to_string(),
            });
        }
        self.state = ContainerState::Created;
        Ok(())
    }

    /// Transition to running state
    pub fn mark_running(&mut self, pid: i32) -> Result<()> {
        if self.state != ContainerState::Created {
            return Err(crate::Error::InvalidState {
                expected: "created".to_string(),
                actual: self.state.to_string(),
            });
        }
        self.state = ContainerState::Running;
        self.pid = Some(pid);
        Ok(())
    }

    /// Transition to stopped state
    pub fn mark_stopped(&mut self) -> Result<()> {
        if self.state != ContainerState::Running && self.state != ContainerState::Paused {
            return Err(crate::Error::InvalidState {
                expected: "running or paused".to_string(),
                actual: self.state.to_string(),
            });
        }
        self.state = ContainerState::Stopped;
        self.pid = None;
        Ok(())
    }

    /// Check if container is running
    pub fn is_running(&self) -> bool {
        self.state == ContainerState::Running
    }
}

/// Options for creating a container
#[derive(Clone, Debug, Default)]
pub struct CreateOptions {
    /// Console socket for PTY
    pub console_socket: Option<PathBuf>,
    /// PID file path
    pub pid_file: Option<PathBuf>,
    /// Don't pivot root
    pub no_pivot: bool,
    /// Enable rootless mode
    pub rootless: bool,
}

/// Options for starting a container
#[derive(Clone, Debug, Default)]
pub struct StartOptions {
    /// Detach from container
    pub detach: bool,
}

/// Options for exec in container
#[derive(Clone, Debug, Default)]
pub struct ExecOptions {
    /// Allocate a TTY
    pub tty: bool,
    /// Detach from process
    pub detach: bool,
    /// User to run as
    pub user: Option<String>,
    /// Environment variables
    pub env: Vec<String>,
    /// Working directory
    pub cwd: Option<PathBuf>,
}
