//! Container runtime operations
//!
//! Handles container creation, start, stop, and deletion.

use std::fs;
use std::path::{Path, PathBuf};
use std::os::unix::fs::PermissionsExt;

use nix::mount::{mount, umount2, MntFlags, MsFlags};
use nix::sched::CloneFlags;
use nix::sys::signal::Signal;
use nix::sys::wait::waitpid;
use nix::unistd::{chdir, fork, pivot_root, ForkResult, Pid};

use crate::cgroup::CgroupManager;
use crate::container::{ContainerState, State};
use crate::namespace::{set_hostname, unshare_namespaces, NamespaceType};
use crate::spec::Spec;
use crate::{Error, Result, RuntimeConfig};

/// Container runtime for managing container lifecycle
pub struct Runtime {
    config: RuntimeConfig,
}

impl Runtime {
    /// Create a new runtime
    pub fn new(config: RuntimeConfig) -> Self {
        Self { config }
    }

    /// Create a new container
    pub fn create(
        &self,
        container_id: &str,
        bundle: &Path,
        _console_socket: Option<PathBuf>,
        pid_file: Option<PathBuf>,
    ) -> Result<()> {
        // Validate container doesn't exist
        let state_path = self.config.state_dir.join(container_id);
        if state_path.exists() {
            return Err(Error::ContainerExists(container_id.to_string()));
        }

        // Load spec
        let spec_path = bundle.join("config.json");
        let spec = Spec::load(&spec_path)?;

        // Create state directory
        fs::create_dir_all(&state_path)?;
        fs::set_permissions(&state_path, fs::Permissions::from_mode(0o700))?;

        // Determine root path
        let rootfs = if spec.root.path.is_absolute() {
            spec.root.path.clone()
        } else {
            bundle.join(&spec.root.path)
        };

        // Fork to create container init process
        match unsafe { fork() } {
            Ok(ForkResult::Parent { child }) => {
                // Parent: save state
                let state = State {
                    oci_version: "1.0.0".to_string(),
                    id: container_id.to_string(),
                    status: ContainerState::Created,
                    pid: Some(child.as_raw()),
                    bundle: bundle.to_path_buf(),
                    annotations: spec.annotations.clone(),
                };

                // Save state to file
                let state_json = serde_json::to_string_pretty(&state)?;
                fs::write(state_path.join("state.json"), state_json)?;

                // Write PID file if requested
                if let Some(pid_path) = pid_file {
                    fs::write(pid_path, child.to_string())?;
                }

                log::info!("Container {} created with PID {}", container_id, child);
            }
            Ok(ForkResult::Child) => {
                // Child: set up container environment
                if let Err(e) = self.setup_container(&spec, &rootfs, container_id) {
                    log::error!("Container setup failed: {}", e);
                    std::process::exit(1);
                }

                // Wait for start signal (in real implementation, use a sync primitive)
                // For now, just pause
                loop {
                    std::thread::sleep(std::time::Duration::from_secs(1));
                }
            }
            Err(e) => {
                return Err(Error::Fork(e.to_string()));
            }
        }

        Ok(())
    }

    /// Set up container namespaces and filesystem
    fn setup_container(&self, spec: &Spec, rootfs: &Path, container_id: &str) -> Result<()> {
        // Get Linux-specific config
        let linux = spec.linux.as_ref().ok_or_else(|| {
            Error::Runtime("Linux config required".to_string())
        })?;

        // Determine which namespaces to create
        let mut clone_flags = CloneFlags::empty();
        for ns in &linux.namespaces {
            let ns_type = match ns.ns_type.as_str() {
                "pid" => NamespaceType::Pid,
                "network" | "net" => NamespaceType::Network,
                "mount" | "mnt" => NamespaceType::Mount,
                "ipc" => NamespaceType::Ipc,
                "uts" => NamespaceType::Uts,
                "user" => NamespaceType::User,
                "cgroup" => NamespaceType::Cgroup,
                _ => continue,
            };
            clone_flags |= ns_type.clone_flag();
        }

        // Create namespaces
        unshare_namespaces(clone_flags)?;

        // Set hostname if UTS namespace is created
        if clone_flags.contains(CloneFlags::CLONE_NEWUTS) && !spec.hostname.is_empty() {
            set_hostname(&spec.hostname)?;
        }

        // Set up cgroups
        if let Some(resources) = &linux.resources {
            let cgroup_path = PathBuf::from(format!("/sys/fs/cgroup/docklet/{}", container_id));
            let cgroup = CgroupManager::new(cgroup_path);

            let config = crate::cgroup::CgroupConfig::from_linux_resources(resources);
            cgroup.create(&config)?;
            cgroup.attach(Pid::this())?;
        }

        // Set up mount namespace
        if clone_flags.contains(CloneFlags::CLONE_NEWNS) {
            self.setup_mounts(spec, rootfs)?;
        }

        Ok(())
    }

    /// Set up container mounts and pivot_root
    fn setup_mounts(&self, spec: &Spec, rootfs: &Path) -> Result<()> {
        // Make mount namespace private
        mount::<str, str, str, str>(
            None,
            "/",
            None,
            MsFlags::MS_REC | MsFlags::MS_PRIVATE,
            None,
        )?;

        // Bind mount rootfs to itself
        mount::<Path, Path, str, str>(
            Some(rootfs),
            rootfs,
            None,
            MsFlags::MS_BIND | MsFlags::MS_REC,
            None,
        )?;

        // Create put_old directory for pivot_root
        let put_old = rootfs.join("put_old");
        fs::create_dir_all(&put_old)?;

        // Pivot root
        pivot_root(rootfs, &put_old)?;
        chdir("/")?;

        // Unmount put_old
        umount2("/put_old", MntFlags::MNT_DETACH)?;
        fs::remove_dir("/put_old")?;

        // Mount proc and other mounts
        for mount_spec in &spec.mounts {
            self.do_mount(mount_spec)?;
        }

        Ok(())
    }

    /// Perform a single mount
    fn do_mount(&self, mount_spec: &crate::spec::Mount) -> Result<()> {
        // Create mount point
        if !mount_spec.destination.exists() {
            fs::create_dir_all(&mount_spec.destination)?;
        }

        let mut flags = MsFlags::empty();
        let mut data = Vec::new();

        for opt in &mount_spec.options {
            match opt.as_str() {
                "bind" => flags |= MsFlags::MS_BIND,
                "rbind" => flags |= MsFlags::MS_BIND | MsFlags::MS_REC,
                "ro" => flags |= MsFlags::MS_RDONLY,
                "rw" => {}
                "nosuid" => flags |= MsFlags::MS_NOSUID,
                "noexec" => flags |= MsFlags::MS_NOEXEC,
                "nodev" => flags |= MsFlags::MS_NODEV,
                "relatime" => flags |= MsFlags::MS_RELATIME,
                "strictatime" => flags |= MsFlags::MS_STRICTATIME,
                _ => data.push(opt.as_str()),
            }
        }

        let data_str = if data.is_empty() {
            None
        } else {
            Some(data.join(","))
        };

        let source: Option<&Path> = if mount_spec.source.as_os_str().is_empty() {
            None
        } else {
            Some(&mount_spec.source)
        };

        mount(
            source,
            &mount_spec.destination,
            Some(mount_spec.mount_type.as_str()),
            flags,
            data_str.as_deref(),
        )?;

        Ok(())
    }

    /// Start a created container
    pub fn start(&self, container_id: &str) -> Result<()> {
        let state_path = self.config.state_dir.join(container_id);
        if !state_path.exists() {
            return Err(Error::ContainerNotFound(container_id.to_string()));
        }

        // Load state
        let state_json = fs::read_to_string(state_path.join("state.json"))?;
        let mut state: State = serde_json::from_str(&state_json)?;

        if state.status != ContainerState::Created {
            return Err(Error::Runtime(format!(
                "Container {} is not in created state",
                container_id
            )));
        }

        // Update state
        state.status = ContainerState::Running;
        let state_json = serde_json::to_string_pretty(&state)?;
        fs::write(state_path.join("state.json"), state_json)?;

        // In a real implementation, we would signal the container to start
        log::info!("Container {} started", container_id);

        Ok(())
    }

    /// Kill a container
    pub fn kill(&self, container_id: &str, signal: Signal) -> Result<()> {
        let state_path = self.config.state_dir.join(container_id);
        if !state_path.exists() {
            return Err(Error::ContainerNotFound(container_id.to_string()));
        }

        // Load state
        let state_json = fs::read_to_string(state_path.join("state.json"))?;
        let state: State = serde_json::from_str(&state_json)?;

        if let Some(pid) = state.pid {
            let pid = Pid::from_raw(pid as i32);
            nix::sys::signal::kill(pid, signal)?;
            log::info!("Sent {} to container {}", signal, container_id);
        }

        Ok(())
    }

    /// Delete a container
    pub fn delete(&self, container_id: &str, force: bool) -> Result<()> {
        let state_path = self.config.state_dir.join(container_id);
        if !state_path.exists() {
            return Err(Error::ContainerNotFound(container_id.to_string()));
        }

        // Load state
        let state_json = fs::read_to_string(state_path.join("state.json"))?;
        let state: State = serde_json::from_str(&state_json)?;

        // Check if container is stopped
        if state.status != ContainerState::Stopped && !force {
            return Err(Error::Runtime(format!(
                "Container {} is not stopped (use --force to delete)",
                container_id
            )));
        }

        // Kill if force and running
        if force && state.pid.is_some() {
            let pid = Pid::from_raw(state.pid.unwrap() as i32);
            let _ = nix::sys::signal::kill(pid, Signal::SIGKILL);
            let _ = waitpid(pid, None);
        }

        // Clean up cgroup
        let cgroup_path = PathBuf::from(format!("/sys/fs/cgroup/docklet/{}", container_id));
        if cgroup_path.exists() {
            let cgroup = CgroupManager::new(cgroup_path);
            let _ = cgroup.remove();
        }

        // Remove state directory
        fs::remove_dir_all(&state_path)?;

        log::info!("Container {} deleted", container_id);
        Ok(())
    }

    /// Get container state
    pub fn state(&self, container_id: &str) -> Result<State> {
        let state_path = self.config.state_dir.join(container_id);
        if !state_path.exists() {
            return Err(Error::ContainerNotFound(container_id.to_string()));
        }

        let state_json = fs::read_to_string(state_path.join("state.json"))?;
        let mut state: State = serde_json::from_str(&state_json)?;

        // Check if process is still running
        if let Some(pid) = state.pid {
            let pid = Pid::from_raw(pid as i32);
            match nix::sys::signal::kill(pid, None) {
                Ok(_) => {} // Process still running
                Err(_) => {
                    // Process has exited
                    state.status = ContainerState::Stopped;
                    state.pid = None;
                }
            }
        }

        Ok(state)
    }

    /// List containers
    pub fn list(&self) -> Result<Vec<State>> {
        let mut containers = Vec::new();

        if !self.config.state_dir.exists() {
            return Ok(containers);
        }

        for entry in fs::read_dir(&self.config.state_dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let state_file = entry.path().join("state.json");
                if state_file.exists() {
                    let state_json = fs::read_to_string(state_file)?;
                    let state: State = serde_json::from_str(&state_json)?;
                    containers.push(state);
                }
            }
        }

        Ok(containers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_runtime_new() {
        let temp = TempDir::new().unwrap();
        let config = RuntimeConfig {
            root: temp.path().to_path_buf(),
            state_dir: temp.path().join("containers"),
            image_dir: temp.path().join("images"),
            cgroup_root: PathBuf::from("/sys/fs/cgroup"),
            rootless: false,
        };
        let _runtime = Runtime::new(config);
    }
}
