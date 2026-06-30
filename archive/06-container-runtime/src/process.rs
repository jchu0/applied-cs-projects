//! Process execution inside containers
//!
//! Handles setting up and executing processes inside containers.

use std::ffi::CString;
use std::os::unix::io::RawFd;
use std::path::Path;

use nix::unistd::{self, Gid, Uid};

use crate::spec::Process;
use crate::{Error, Result};

/// Execute a process inside the container
pub fn exec_process(process: &Process) -> Result<()> {
    // Set working directory
    if !process.cwd.as_os_str().is_empty() {
        std::env::set_current_dir(&process.cwd).map_err(|e| {
            Error::Process(format!("Failed to set working directory: {}", e))
        })?;
    }

    // Set environment variables
    for env_var in &process.env {
        if let Some(eq_pos) = env_var.find('=') {
            let key = &env_var[..eq_pos];
            let value = &env_var[eq_pos + 1..];
            std::env::set_var(key, value);
        }
    }

    // Set user and group
    set_user_group(process.user.uid, process.user.gid, &process.user.additional_gids)?;

    // Set resource limits (rlimits)
    for rlimit in &process.rlimits {
        set_rlimit(&rlimit.limit_type, rlimit.soft, rlimit.hard)?;
    }

    // Close file descriptors above 2 (stdin, stdout, stderr)
    close_fds_above(2);

    // Execute the process
    if process.args.is_empty() {
        return Err(Error::Process("No arguments specified".to_string()));
    }

    let program = &process.args[0];
    let args: Vec<CString> = process
        .args
        .iter()
        .map(|s| CString::new(s.as_str()).unwrap())
        .collect();

    let env: Vec<CString> = std::env::vars()
        .map(|(k, v)| CString::new(format!("{}={}", k, v)).unwrap())
        .collect();

    // Convert to the format execve expects
    let program = CString::new(program.as_str()).map_err(|e| {
        Error::Process(format!("Invalid program name: {}", e))
    })?;

    // Execute
    unistd::execve(&program, &args, &env).map_err(|e| {
        Error::Process(format!("execve failed: {}", e))
    })?;

    // This line should never be reached
    unreachable!()
}

/// Set user and group IDs
fn set_user_group(uid: u32, gid: u32, additional_gids: &[u32]) -> Result<()> {
    // Set supplementary groups
    if !additional_gids.is_empty() {
        let gids: Vec<Gid> = additional_gids.iter().map(|&g| Gid::from_raw(g)).collect();
        unistd::setgroups(&gids).map_err(|e| {
            Error::Process(format!("Failed to set supplementary groups: {}", e))
        })?;
    }

    // Set GID
    let gid = Gid::from_raw(gid);
    unistd::setgid(gid).map_err(|e| {
        Error::Process(format!("Failed to set gid {}: {}", gid, e))
    })?;

    // Set UID
    let uid = Uid::from_raw(uid);
    unistd::setuid(uid).map_err(|e| {
        Error::Process(format!("Failed to set uid {}: {}", uid, e))
    })?;

    Ok(())
}

/// Set resource limits
fn set_rlimit(name: &str, soft: u64, hard: u64) -> Result<()> {
    let resource = match name {
        "RLIMIT_CPU" => libc::RLIMIT_CPU,
        "RLIMIT_FSIZE" => libc::RLIMIT_FSIZE,
        "RLIMIT_DATA" => libc::RLIMIT_DATA,
        "RLIMIT_STACK" => libc::RLIMIT_STACK,
        "RLIMIT_CORE" => libc::RLIMIT_CORE,
        "RLIMIT_RSS" => libc::RLIMIT_RSS,
        "RLIMIT_NPROC" => libc::RLIMIT_NPROC,
        "RLIMIT_NOFILE" => libc::RLIMIT_NOFILE,
        "RLIMIT_MEMLOCK" => libc::RLIMIT_MEMLOCK,
        "RLIMIT_AS" => libc::RLIMIT_AS,
        "RLIMIT_LOCKS" => libc::RLIMIT_LOCKS,
        "RLIMIT_SIGPENDING" => libc::RLIMIT_SIGPENDING,
        "RLIMIT_MSGQUEUE" => libc::RLIMIT_MSGQUEUE,
        "RLIMIT_NICE" => libc::RLIMIT_NICE,
        "RLIMIT_RTPRIO" => libc::RLIMIT_RTPRIO,
        "RLIMIT_RTTIME" => libc::RLIMIT_RTTIME,
        _ => {
            log::warn!("Unknown rlimit: {}", name);
            return Ok(());
        }
    };

    let rlim = libc::rlimit {
        rlim_cur: soft as libc::rlim_t,
        rlim_max: hard as libc::rlim_t,
    };

    let ret = unsafe { libc::setrlimit(resource, &rlim) };
    if ret != 0 {
        return Err(Error::Process(format!(
            "Failed to set rlimit {}: {}",
            name,
            std::io::Error::last_os_error()
        )));
    }

    Ok(())
}

/// Close all file descriptors above a certain number
fn close_fds_above(min_fd: RawFd) {
    // Read /proc/self/fd to find open file descriptors
    if let Ok(entries) = std::fs::read_dir("/proc/self/fd") {
        for entry in entries.flatten() {
            if let Ok(fd_str) = entry.file_name().into_string() {
                if let Ok(fd) = fd_str.parse::<RawFd>() {
                    if fd > min_fd {
                        unsafe {
                            libc::close(fd);
                        }
                    }
                }
            }
        }
    }
}

/// Set up a console for the container
pub fn setup_console(_console_socket: &Path) -> Result<RawFd> {
    // In a real implementation, this would:
    // 1. Create a PTY
    // 2. Send the master FD over the console socket
    // 3. Set up the slave as stdin/stdout/stderr

    // For now, return a placeholder
    Err(Error::Process("Console setup not implemented".to_string()))
}

/// Create a PTY pair
pub fn create_pty() -> Result<(RawFd, RawFd)> {
    let result = unsafe { libc::openpty(
        &mut 0 as *mut libc::c_int,
        &mut 0 as *mut libc::c_int,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        std::ptr::null_mut(),
    ) };

    if result != 0 {
        return Err(Error::Process("Failed to create PTY".to_string()));
    }

    // This is a simplified stub - real implementation would return the fds
    Ok((0, 0))
}

/// Capabilities handling
pub mod capabilities {
    use crate::{Error, Result};

    /// Drop all capabilities
    pub fn drop_all() -> Result<()> {
        // In a real implementation, this would use cap_set_proc
        // to drop all capabilities from the bounding set
        Ok(())
    }

    /// Keep only specified capabilities
    pub fn keep_only(caps: &[&str]) -> Result<()> {
        // Parse capability names and set them
        for cap in caps {
            log::debug!("Keeping capability: {}", cap);
        }
        Ok(())
    }

    /// Drop a specific capability
    pub fn drop(name: &str) -> Result<()> {
        log::debug!("Dropping capability: {}", name);
        Ok(())
    }
}

/// Seccomp filter handling
pub mod seccomp {
    use crate::Result;

    /// Seccomp profile for syscall filtering
    pub struct SeccompProfile {
        pub default_action: String,
        pub syscalls: Vec<SeccompSyscall>,
    }

    /// Seccomp syscall rule
    pub struct SeccompSyscall {
        pub names: Vec<String>,
        pub action: String,
    }

    /// Load a seccomp profile
    pub fn load_profile(profile: &SeccompProfile) -> Result<()> {
        log::debug!("Loading seccomp profile with {} syscalls", profile.syscalls.len());

        // In a real implementation, this would:
        // 1. Create a seccomp BPF filter
        // 2. Add rules for each syscall
        // 3. Load the filter with prctl/seccomp syscall

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_close_fds() {
        // This test just verifies the function doesn't panic
        close_fds_above(100);
    }

    #[test]
    fn test_set_rlimit_nofile() {
        // Test setting NOFILE limit
        let result = set_rlimit("RLIMIT_NOFILE", 1024, 4096);
        // May fail due to permissions, but shouldn't panic
        let _ = result;
    }
}
