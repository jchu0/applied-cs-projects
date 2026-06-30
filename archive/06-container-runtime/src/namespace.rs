//! Linux namespace management
//!
//! Provides namespace creation and management for process isolation.

use std::fs;
use std::os::fd::{OwnedFd, FromRawFd, AsRawFd};
use std::path::Path;

use nix::sched::{self, CloneFlags};
use nix::unistd::Pid;

use crate::{Error, Result};

/// Types of Linux namespaces
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NamespaceType {
    /// User namespace - user and group ID isolation
    User,
    /// Mount namespace - filesystem mount isolation
    Mount,
    /// PID namespace - process ID isolation
    Pid,
    /// Network namespace - network stack isolation
    Network,
    /// IPC namespace - System V IPC isolation
    Ipc,
    /// UTS namespace - hostname and domain name isolation
    Uts,
    /// Cgroup namespace - cgroup hierarchy isolation
    Cgroup,
}

impl NamespaceType {
    /// Get the clone flag for this namespace type
    pub fn clone_flag(&self) -> CloneFlags {
        match self {
            NamespaceType::User => CloneFlags::CLONE_NEWUSER,
            NamespaceType::Mount => CloneFlags::CLONE_NEWNS,
            NamespaceType::Pid => CloneFlags::CLONE_NEWPID,
            NamespaceType::Network => CloneFlags::CLONE_NEWNET,
            NamespaceType::Ipc => CloneFlags::CLONE_NEWIPC,
            NamespaceType::Uts => CloneFlags::CLONE_NEWUTS,
            NamespaceType::Cgroup => CloneFlags::CLONE_NEWCGROUP,
        }
    }

    /// Get the proc path name for this namespace type
    pub fn proc_name(&self) -> &'static str {
        match self {
            NamespaceType::User => "user",
            NamespaceType::Mount => "mnt",
            NamespaceType::Pid => "pid",
            NamespaceType::Network => "net",
            NamespaceType::Ipc => "ipc",
            NamespaceType::Uts => "uts",
            NamespaceType::Cgroup => "cgroup",
        }
    }
}

/// Handle to a namespace file descriptor
#[derive(Debug)]
pub struct NamespaceHandle {
    /// Namespace type
    pub ns_type: NamespaceType,
    /// File descriptor
    fd: OwnedFd,
}

impl NamespaceHandle {
    /// Open a namespace from a path
    pub fn open(ns_type: NamespaceType, path: &Path) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC)
            .open(path)?;

        let fd = unsafe { OwnedFd::from_raw_fd(file.as_raw_fd()) };

        // Prevent the file from being closed (OwnedFd now owns it)
        std::mem::forget(file);

        Ok(Self { ns_type, fd })
    }

    /// Enter this namespace
    pub fn enter(&self) -> Result<()> {
        sched::setns(&self.fd, self.ns_type.clone_flag())?;
        Ok(())
    }
}

// OwnedFd automatically closes on drop, no manual Drop needed

/// Set of namespace handles
#[derive(Debug, Default)]
pub struct NamespaceSet {
    pub user: Option<NamespaceHandle>,
    pub mount: Option<NamespaceHandle>,
    pub pid: Option<NamespaceHandle>,
    pub network: Option<NamespaceHandle>,
    pub ipc: Option<NamespaceHandle>,
    pub uts: Option<NamespaceHandle>,
    pub cgroup: Option<NamespaceHandle>,
}

impl NamespaceSet {
    /// Get clone flags for all namespaces that should be created
    pub fn clone_flags(&self, types: &[NamespaceType]) -> CloneFlags {
        let mut flags = CloneFlags::empty();
        for ns_type in types {
            flags |= ns_type.clone_flag();
        }
        flags
    }

    /// Open namespaces from a process
    pub fn from_pid(pid: Pid, types: &[NamespaceType]) -> Result<Self> {
        let mut set = Self::default();
        let proc_path = format!("/proc/{}/ns", pid);

        for ns_type in types {
            let path = format!("{}/{}", proc_path, ns_type.proc_name());
            let handle = NamespaceHandle::open(*ns_type, Path::new(&path))?;

            match ns_type {
                NamespaceType::User => set.user = Some(handle),
                NamespaceType::Mount => set.mount = Some(handle),
                NamespaceType::Pid => set.pid = Some(handle),
                NamespaceType::Network => set.network = Some(handle),
                NamespaceType::Ipc => set.ipc = Some(handle),
                NamespaceType::Uts => set.uts = Some(handle),
                NamespaceType::Cgroup => set.cgroup = Some(handle),
            }
        }

        Ok(set)
    }
}

/// UID/GID mapping for user namespaces
#[derive(Clone, Debug)]
pub struct IdMapping {
    /// ID inside the container
    pub container_id: u32,
    /// ID outside the container (host)
    pub host_id: u32,
    /// Number of IDs to map
    pub size: u32,
}

/// Write UID mappings to proc
pub fn write_uid_mappings(pid: Pid, mappings: &[IdMapping]) -> Result<()> {
    let path = format!("/proc/{}/uid_map", pid);
    let content: String = mappings
        .iter()
        .map(|m| format!("{} {} {}", m.container_id, m.host_id, m.size))
        .collect::<Vec<_>>()
        .join("\n");

    fs::write(&path, content).map_err(|e| {
        Error::Namespace(format!("Failed to write uid_map: {}", e))
    })
}

/// Write GID mappings to proc
pub fn write_gid_mappings(pid: Pid, mappings: &[IdMapping]) -> Result<()> {
    // Must disable setgroups first for unprivileged users
    let setgroups_path = format!("/proc/{}/setgroups", pid);
    let _ = fs::write(&setgroups_path, "deny");

    let path = format!("/proc/{}/gid_map", pid);
    let content: String = mappings
        .iter()
        .map(|m| format!("{} {} {}", m.container_id, m.host_id, m.size))
        .collect::<Vec<_>>()
        .join("\n");

    fs::write(&path, content).map_err(|e| {
        Error::Namespace(format!("Failed to write gid_map: {}", e))
    })
}

/// Create a new namespace by unsharing
pub fn unshare_namespaces(flags: CloneFlags) -> Result<()> {
    sched::unshare(flags)?;
    Ok(())
}

/// Set hostname (requires UTS namespace)
pub fn set_hostname(hostname: &str) -> Result<()> {
    nix::unistd::sethostname(hostname)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_type_proc_name() {
        assert_eq!(NamespaceType::Pid.proc_name(), "pid");
        assert_eq!(NamespaceType::Network.proc_name(), "net");
        assert_eq!(NamespaceType::Mount.proc_name(), "mnt");
    }

    #[test]
    fn test_clone_flags() {
        let set = NamespaceSet::default();
        let flags = set.clone_flags(&[NamespaceType::Pid, NamespaceType::Mount]);
        assert!(flags.contains(CloneFlags::CLONE_NEWPID));
        assert!(flags.contains(CloneFlags::CLONE_NEWNS));
    }
}
