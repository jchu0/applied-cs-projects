//! Replication module for master-replica synchronization
//!
//! Implements Redis-compatible replication with:
//! - Full synchronization (SYNC)
//! - Partial resynchronization (PSYNC)
//! - Replication backlog for partial resync
//! - Replica state tracking

mod master;
mod replica;
mod backlog;

pub use master::MasterState;
pub use replica::ReplicaState;
pub use backlog::ReplicationBacklog;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Replication role
#[derive(Debug, Clone, PartialEq)]
pub enum ReplicationRole {
    Master,
    Replica,
}

/// Replica information
#[derive(Debug, Clone)]
pub struct ReplicaInfo {
    /// Replica ID
    pub id: String,
    /// Replica address
    pub addr: String,
    /// Last acknowledged offset
    pub ack_offset: u64,
    /// Last ACK time
    pub last_ack: Instant,
    /// Replica state
    pub state: ReplicaConnectionState,
}

/// Replica connection state
#[derive(Debug, Clone, PartialEq)]
pub enum ReplicaConnectionState {
    /// Initial connection
    Connect,
    /// Waiting for SYNC
    Connecting,
    /// Performing full sync
    Sync,
    /// Connected and streaming
    Connected,
}

/// Replication configuration
#[derive(Debug, Clone)]
pub struct ReplicationConfig {
    /// Replication ID (40 hex chars)
    pub repl_id: String,
    /// Secondary replication ID for partial resync after failover
    pub repl_id2: String,
    /// Current master offset
    pub master_offset: u64,
    /// Second replication ID offset
    pub second_offset: i64,
    /// Backlog buffer size
    pub backlog_size: usize,
    /// Replica timeout in seconds
    pub repl_timeout: u64,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            repl_id: generate_repl_id(),
            repl_id2: String::new(),
            master_offset: 0,
            second_offset: -1,
            backlog_size: 1024 * 1024, // 1MB default
            repl_timeout: 60,
        }
    }
}

/// Generate a random replication ID
fn generate_repl_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:040x}", timestamp)
}

/// Replication manager
pub struct ReplicationManager {
    /// Current role
    role: ReplicationRole,
    /// Configuration
    config: ReplicationConfig,
    /// Connected replicas (when master)
    replicas: HashMap<String, ReplicaInfo>,
    /// Replication backlog
    backlog: ReplicationBacklog,
    /// Master host (when replica)
    master_host: Option<String>,
    /// Master port (when replica)
    master_port: Option<u16>,
}

impl ReplicationManager {
    /// Create a new replication manager as master
    pub fn new_master() -> Self {
        Self {
            role: ReplicationRole::Master,
            config: ReplicationConfig::default(),
            replicas: HashMap::new(),
            backlog: ReplicationBacklog::new(1024 * 1024),
            master_host: None,
            master_port: None,
        }
    }

    /// Create a new replication manager as replica
    pub fn new_replica(master_host: String, master_port: u16) -> Self {
        Self {
            role: ReplicationRole::Replica,
            config: ReplicationConfig::default(),
            replicas: HashMap::new(),
            backlog: ReplicationBacklog::new(1024 * 1024),
            master_host: Some(master_host),
            master_port: Some(master_port),
        }
    }

    /// Get current role
    pub fn role(&self) -> &ReplicationRole {
        &self.role
    }

    /// Get replication ID
    pub fn repl_id(&self) -> &str {
        &self.config.repl_id
    }

    /// Get master offset
    pub fn master_offset(&self) -> u64 {
        self.config.master_offset
    }

    /// Add data to replication backlog
    pub fn feed_backlog(&mut self, data: &[u8]) {
        self.backlog.append(data);
        self.config.master_offset += data.len() as u64;
    }

    /// Register a new replica
    pub fn add_replica(&mut self, id: String, addr: String) -> &ReplicaInfo {
        let info = ReplicaInfo {
            id: id.clone(),
            addr,
            ack_offset: 0,
            last_ack: Instant::now(),
            state: ReplicaConnectionState::Connect,
        };
        self.replicas.insert(id.clone(), info);
        self.replicas.get(&id).unwrap()
    }

    /// Update replica ACK
    pub fn update_replica_ack(&mut self, id: &str, offset: u64) {
        if let Some(replica) = self.replicas.get_mut(id) {
            replica.ack_offset = offset;
            replica.last_ack = Instant::now();
        }
    }

    /// Get replica info
    pub fn get_replica(&self, id: &str) -> Option<&ReplicaInfo> {
        self.replicas.get(id)
    }

    /// Remove replica
    pub fn remove_replica(&mut self, id: &str) -> Option<ReplicaInfo> {
        self.replicas.remove(id)
    }

    /// Get number of connected replicas
    pub fn replica_count(&self) -> usize {
        self.replicas
            .values()
            .filter(|r| r.state == ReplicaConnectionState::Connected)
            .count()
    }

    /// Check if partial resync is possible
    pub fn can_partial_resync(&self, repl_id: &str, offset: u64) -> bool {
        if repl_id != self.config.repl_id && repl_id != self.config.repl_id2 {
            return false;
        }
        self.backlog.contains_offset(offset)
    }

    /// Get data for partial resync
    pub fn get_partial_data(&self, offset: u64) -> Option<Vec<u8>> {
        self.backlog.get_from_offset(offset)
    }

    /// Get all replicas
    pub fn replicas(&self) -> impl Iterator<Item = &ReplicaInfo> {
        self.replicas.values()
    }

    /// Promote replica to master
    pub fn promote_to_master(&mut self) {
        self.role = ReplicationRole::Master;
        self.master_host = None;
        self.master_port = None;
        // Keep current repl_id for partial resync
        self.config.repl_id2 = self.config.repl_id.clone();
        self.config.repl_id = generate_repl_id();
        self.config.second_offset = self.config.master_offset as i64;
    }

    /// Get replication info for INFO command
    pub fn info(&self) -> String {
        let mut info = String::new();

        info.push_str(&format!("role:{}\n", match self.role {
            ReplicationRole::Master => "master",
            ReplicationRole::Replica => "slave",
        }));

        match self.role {
            ReplicationRole::Master => {
                info.push_str(&format!("connected_slaves:{}\n", self.replica_count()));
                for (i, replica) in self.replicas.values().enumerate() {
                    info.push_str(&format!(
                        "slave{}:ip={},offset={},lag={}\n",
                        i,
                        replica.addr,
                        replica.ack_offset,
                        replica.last_ack.elapsed().as_secs()
                    ));
                }
            }
            ReplicationRole::Replica => {
                if let (Some(host), Some(port)) = (&self.master_host, self.master_port) {
                    info.push_str(&format!("master_host:{}\n", host));
                    info.push_str(&format!("master_port:{}\n", port));
                }
            }
        }

        info.push_str(&format!("master_replid:{}\n", self.config.repl_id));
        info.push_str(&format!("master_repl_offset:{}\n", self.config.master_offset));

        info
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replication_manager() {
        let mut manager = ReplicationManager::new_master();
        assert_eq!(*manager.role(), ReplicationRole::Master);
        assert_eq!(manager.replica_count(), 0);

        // Add replica
        manager.add_replica("replica1".to_string(), "127.0.0.1:6380".to_string());
        assert_eq!(manager.replicas.len(), 1);
    }

    #[test]
    fn test_backlog_feed() {
        let mut manager = ReplicationManager::new_master();
        assert_eq!(manager.master_offset(), 0);

        manager.feed_backlog(b"*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n");
        assert!(manager.master_offset() > 0);
    }

    #[test]
    fn test_promote_to_master() {
        let mut manager = ReplicationManager::new_replica("localhost".to_string(), 6379);
        assert_eq!(*manager.role(), ReplicationRole::Replica);

        manager.promote_to_master();
        assert_eq!(*manager.role(), ReplicationRole::Master);
    }
}
