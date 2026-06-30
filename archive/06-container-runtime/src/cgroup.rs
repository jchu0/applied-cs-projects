//! Cgroup management
//!
//! Provides cgroup v2 management for resource constraints.

use std::fs;
use std::path::{Path, PathBuf};

use nix::unistd::Pid;

use crate::{Error, Result};

/// Cgroup manager for cgroup v2
#[derive(Debug)]
pub struct CgroupManager {
    /// Root path of cgroup hierarchy
    root: PathBuf,
}

impl CgroupManager {
    /// Create a new cgroup manager for a specific path
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { root: path.into() }
    }

    /// Create the cgroup directory and apply config
    pub fn create(&self, config: &CgroupConfig) -> Result<()> {
        fs::create_dir_all(&self.root)?;

        // Apply resource limits
        self.apply_config(&self.root, config)?;

        Ok(())
    }

    /// Remove the cgroup
    pub fn remove(&self) -> Result<()> {
        // Kill all processes first
        let procs_path = self.root.join("cgroup.procs");
        if let Ok(pids) = fs::read_to_string(&procs_path) {
            for pid_str in pids.lines() {
                if let Ok(pid) = pid_str.parse::<i32>() {
                    let _ = nix::sys::signal::kill(
                        Pid::from_raw(pid),
                        nix::sys::signal::Signal::SIGKILL,
                    );
                }
            }
        }

        // Remove cgroup directory
        fs::remove_dir(&self.root).map_err(|e| {
            Error::Cgroup(format!("Failed to remove cgroup: {}", e))
        })
    }

    /// Attach a process to the cgroup
    pub fn attach(&self, pid: Pid) -> Result<()> {
        let procs_path = self.root.join("cgroup.procs");
        fs::write(&procs_path, pid.to_string()).map_err(|e| {
            Error::Cgroup(format!("Failed to attach process {}: {}", pid, e))
        })
    }

    /// Apply cgroup configuration
    fn apply_config(&self, cgroup_path: &Path, config: &CgroupConfig) -> Result<()> {
        // Memory limits
        if let Some(mem) = &config.memory {
            if mem.limit > 0 {
                self.write_cgroup_file(cgroup_path, "memory.max", &mem.limit.to_string())?;
            }
            if mem.reservation > 0 {
                self.write_cgroup_file(cgroup_path, "memory.low", &mem.reservation.to_string())?;
            }
            if mem.swap > 0 {
                self.write_cgroup_file(cgroup_path, "memory.swap.max", &mem.swap.to_string())?;
            }
        }

        // CPU limits
        if let Some(cpu) = &config.cpu {
            if cpu.shares > 0 {
                // Convert shares to weight (1-10000)
                let weight = ((cpu.shares * 10000) / 1024).max(1).min(10000);
                self.write_cgroup_file(cgroup_path, "cpu.weight", &weight.to_string())?;
            }
            if cpu.quota > 0 && cpu.period > 0 {
                let max = format!("{} {}", cpu.quota, cpu.period);
                self.write_cgroup_file(cgroup_path, "cpu.max", &max)?;
            }
            if !cpu.cpus.is_empty() {
                self.write_cgroup_file(cgroup_path, "cpuset.cpus", &cpu.cpus)?;
            }
            if !cpu.mems.is_empty() {
                self.write_cgroup_file(cgroup_path, "cpuset.mems", &cpu.mems)?;
            }
        }

        // PID limits
        if let Some(pids) = &config.pids {
            if pids.max > 0 {
                self.write_cgroup_file(cgroup_path, "pids.max", &pids.max.to_string())?;
            }
        }

        // I/O limits
        if let Some(io) = &config.io {
            if io.weight > 0 {
                self.write_cgroup_file(cgroup_path, "io.weight", &io.weight.to_string())?;
            }
        }

        Ok(())
    }

    /// Write to a cgroup control file
    fn write_cgroup_file(&self, cgroup_path: &Path, filename: &str, value: &str) -> Result<()> {
        let path = cgroup_path.join(filename);
        fs::write(&path, value).map_err(|e| {
            Error::Cgroup(format!("Failed to write {}: {}", path.display(), e))
        })
    }

    /// Read from a cgroup control file
    fn read_cgroup_file(&self, cgroup_path: &Path, filename: &str) -> Result<String> {
        let path = cgroup_path.join(filename);
        fs::read_to_string(&path).map_err(|e| {
            Error::Cgroup(format!("Failed to read {}: {}", path.display(), e))
        })
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> Result<u64> {
        let content = self.read_cgroup_file(&self.root, "memory.current")?;
        content.trim().parse().map_err(|_| {
            Error::Cgroup("Invalid memory.current value".to_string())
        })
    }

    /// Get current CPU usage
    pub fn cpu_usage(&self) -> Result<CpuStats> {
        let content = self.read_cgroup_file(&self.root, "cpu.stat")?;
        let mut stats = CpuStats::default();

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                continue;
            }

            match parts[0] {
                "usage_usec" => {
                    stats.usage_usec = parts[1].parse().unwrap_or(0);
                }
                "user_usec" => {
                    stats.user_usec = parts[1].parse().unwrap_or(0);
                }
                "system_usec" => {
                    stats.system_usec = parts[1].parse().unwrap_or(0);
                }
                _ => {}
            }
        }

        Ok(stats)
    }

    /// Get number of processes in cgroup
    pub fn process_count(&self) -> Result<usize> {
        let content = self.read_cgroup_file(&self.root, "pids.current")?;
        content.trim().parse().map_err(|_| {
            Error::Cgroup("Invalid pids.current value".to_string())
        })
    }

    /// Freeze all processes in cgroup
    pub fn freeze(&self) -> Result<()> {
        self.write_cgroup_file(&self.root, "cgroup.freeze", "1")
    }

    /// Thaw (unfreeze) all processes in cgroup
    pub fn thaw(&self) -> Result<()> {
        self.write_cgroup_file(&self.root, "cgroup.freeze", "0")
    }
}

/// Cgroup configuration
#[derive(Clone, Debug, Default)]
pub struct CgroupConfig {
    pub memory: Option<MemoryConfig>,
    pub cpu: Option<CpuConfig>,
    pub pids: Option<PidsConfig>,
    pub io: Option<IoConfig>,
}

impl CgroupConfig {
    /// Create CgroupConfig from OCI Resources
    pub fn from_linux_resources(resources: &crate::spec::Resources) -> Self {
        let mut config = CgroupConfig::default();

        // Memory
        if let Some(mem) = &resources.memory {
            config.memory = Some(MemoryConfig {
                limit: mem.limit.unwrap_or(0) as u64,
                reservation: mem.reservation.unwrap_or(0) as u64,
                swap: mem.swap.unwrap_or(0) as u64,
                oom_kill_disable: false,
            });
        }

        // CPU
        if let Some(cpu) = &resources.cpu {
            config.cpu = Some(CpuConfig {
                shares: cpu.shares.unwrap_or(0),
                quota: cpu.quota.unwrap_or(0),
                period: cpu.period.unwrap_or(100000),
                cpus: cpu.cpus.clone().unwrap_or_default(),
                mems: cpu.mems.clone().unwrap_or_default(),
            });
        }

        // PIDs
        if let Some(pids) = &resources.pids {
            config.pids = Some(PidsConfig {
                max: pids.limit as u64,
            });
        }

        config
    }
}

/// Memory cgroup configuration
#[derive(Clone, Debug, Default)]
pub struct MemoryConfig {
    /// Hard memory limit in bytes
    pub limit: u64,
    /// Memory reservation (soft limit) in bytes
    pub reservation: u64,
    /// Swap limit in bytes
    pub swap: u64,
    /// Disable OOM killer
    pub oom_kill_disable: bool,
}

/// CPU cgroup configuration
#[derive(Clone, Debug, Default)]
pub struct CpuConfig {
    /// CPU shares (relative weight)
    pub shares: u64,
    /// CPU quota in microseconds per period
    pub quota: i64,
    /// CPU period in microseconds
    pub period: u64,
    /// CPU affinity (e.g., "0-3" or "0,2")
    pub cpus: String,
    /// NUMA memory nodes
    pub mems: String,
}

/// PIDs cgroup configuration
#[derive(Clone, Debug, Default)]
pub struct PidsConfig {
    /// Maximum number of processes
    pub max: u64,
}

/// I/O cgroup configuration
#[derive(Clone, Debug, Default)]
pub struct IoConfig {
    /// I/O weight (1-10000)
    pub weight: u64,
}

/// CPU statistics
#[derive(Clone, Debug, Default)]
pub struct CpuStats {
    /// Total CPU time in microseconds
    pub usage_usec: u64,
    /// User CPU time in microseconds
    pub user_usec: u64,
    /// System CPU time in microseconds
    pub system_usec: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cgroup_config_default() {
        let config = CgroupConfig::default();
        assert!(config.memory.is_none());
        assert!(config.cpu.is_none());
        assert!(config.pids.is_none());
    }

    #[test]
    fn test_memory_config() {
        let mem = MemoryConfig {
            limit: 1024 * 1024 * 512, // 512MB
            reservation: 1024 * 1024 * 256,
            swap: 0,
            oom_kill_disable: false,
        };
        assert_eq!(mem.limit, 536870912);
    }
}
