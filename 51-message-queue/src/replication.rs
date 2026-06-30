//! Multi-broker replication for high availability.

use crate::error::{Error, Result};
use crate::message::Message;
use crate::partition::PartitionId;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Broker identifier.
pub type BrokerId = u32;

/// Replica state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplicaState {
    /// Replica is syncing with leader.
    Syncing,
    /// Replica is in-sync with leader.
    InSync,
    /// Replica is lagging behind.
    Lagging,
    /// Replica is offline.
    Offline,
}

/// Replica information.
#[derive(Debug, Clone)]
pub struct ReplicaInfo {
    /// Broker ID hosting this replica.
    pub broker_id: BrokerId,
    /// Current log end offset.
    pub log_end_offset: u64,
    /// High watermark (committed offset).
    pub high_watermark: u64,
    /// Replica state.
    pub state: ReplicaState,
    /// Last fetch time.
    pub last_fetch_time: Instant,
    /// Last caught up time (when replica became in-sync).
    pub last_caught_up_time: Option<Instant>,
}

impl ReplicaInfo {
    /// Create new replica info.
    pub fn new(broker_id: BrokerId) -> Self {
        Self {
            broker_id,
            log_end_offset: 0,
            high_watermark: 0,
            state: ReplicaState::Syncing,
            last_fetch_time: Instant::now(),
            last_caught_up_time: None,
        }
    }

    /// Update replica progress.
    pub fn update_progress(&mut self, offset: u64, leader_offset: u64, max_lag: u64) {
        self.log_end_offset = offset;
        self.last_fetch_time = Instant::now();

        let lag = leader_offset.saturating_sub(offset);
        if lag <= max_lag {
            if self.state != ReplicaState::InSync {
                self.last_caught_up_time = Some(Instant::now());
            }
            self.state = ReplicaState::InSync;
        } else {
            self.state = ReplicaState::Lagging;
        }
    }

    /// Check if replica is in-sync.
    pub fn is_in_sync(&self) -> bool {
        self.state == ReplicaState::InSync
    }

    /// Get lag behind leader.
    pub fn lag(&self, leader_offset: u64) -> u64 {
        leader_offset.saturating_sub(self.log_end_offset)
    }
}

/// Replication configuration.
#[derive(Debug, Clone)]
pub struct ReplicationConfig {
    /// Replication factor.
    pub replication_factor: u32,
    /// Minimum in-sync replicas required for acks=all.
    pub min_isr: u32,
    /// Maximum lag for a replica to be considered in-sync.
    pub max_lag_messages: u64,
    /// Maximum time for a replica to be considered in-sync.
    pub replica_lag_time_max: Duration,
    /// Fetch wait time before returning empty response.
    pub fetch_wait_max: Duration,
    /// Fetch minimum bytes.
    pub fetch_min_bytes: usize,
    /// Fetch maximum bytes.
    pub fetch_max_bytes: usize,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            replication_factor: 3,
            min_isr: 2,
            max_lag_messages: 4000,
            replica_lag_time_max: Duration::from_secs(30),
            fetch_wait_max: Duration::from_millis(500),
            fetch_min_bytes: 1,
            fetch_max_bytes: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// Acknowledgment level for produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Acks {
    /// No acknowledgment required (fire-and-forget).
    None,
    /// Leader acknowledgment only.
    Leader,
    /// All in-sync replicas acknowledgment.
    All,
}

impl Default for Acks {
    fn default() -> Self {
        Acks::Leader
    }
}

/// Partition replica set.
pub struct ReplicaSet {
    /// Topic name.
    topic: String,
    /// Partition ID.
    partition: PartitionId,
    /// Leader broker ID.
    leader: AtomicU32,
    /// Leader epoch (incremented on leader change).
    leader_epoch: AtomicU32,
    /// Assigned replicas.
    replicas: Vec<BrokerId>,
    /// Replica information.
    replica_info: RwLock<HashMap<BrokerId, ReplicaInfo>>,
    /// In-sync replica set.
    isr: RwLock<HashSet<BrokerId>>,
    /// High watermark (minimum of ISR offsets).
    high_watermark: AtomicU64,
    /// Configuration.
    config: ReplicationConfig,
}

impl ReplicaSet {
    /// Create a new replica set.
    pub fn new(
        topic: String,
        partition: PartitionId,
        replicas: Vec<BrokerId>,
        leader: BrokerId,
        config: ReplicationConfig,
    ) -> Self {
        let mut replica_info = HashMap::new();
        let mut isr = HashSet::new();

        for &broker_id in &replicas {
            replica_info.insert(broker_id, ReplicaInfo::new(broker_id));
            isr.insert(broker_id);
        }

        Self {
            topic,
            partition,
            leader: AtomicU32::new(leader),
            leader_epoch: AtomicU32::new(0),
            replicas,
            replica_info: RwLock::new(replica_info),
            isr: RwLock::new(isr),
            high_watermark: AtomicU64::new(0),
            config,
        }
    }

    /// Get topic name.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Get partition ID.
    pub fn partition(&self) -> PartitionId {
        self.partition
    }

    /// Get current leader.
    pub fn leader(&self) -> BrokerId {
        self.leader.load(Ordering::Acquire)
    }

    /// Get leader epoch.
    pub fn leader_epoch(&self) -> u32 {
        self.leader_epoch.load(Ordering::Acquire)
    }

    /// Get assigned replicas.
    pub fn replicas(&self) -> &[BrokerId] {
        &self.replicas
    }

    /// Get in-sync replicas.
    pub fn isr(&self) -> Vec<BrokerId> {
        self.isr.read().iter().copied().collect()
    }

    /// Get high watermark.
    pub fn high_watermark(&self) -> u64 {
        self.high_watermark.load(Ordering::Acquire)
    }

    /// Check if broker is leader.
    pub fn is_leader(&self, broker_id: BrokerId) -> bool {
        self.leader() == broker_id
    }

    /// Check if broker is in ISR.
    pub fn is_in_isr(&self, broker_id: BrokerId) -> bool {
        self.isr.read().contains(&broker_id)
    }

    /// Update replica fetch progress.
    pub fn update_replica_fetch(
        &self,
        broker_id: BrokerId,
        fetch_offset: u64,
        leader_log_end_offset: u64,
    ) {
        let mut replica_info = self.replica_info.write();

        if let Some(info) = replica_info.get_mut(&broker_id) {
            info.update_progress(
                fetch_offset,
                leader_log_end_offset,
                self.config.max_lag_messages,
            );

            // Update ISR
            let mut isr = self.isr.write();
            if info.is_in_sync() {
                isr.insert(broker_id);
            } else {
                isr.remove(&broker_id);
            }

            drop(isr);
        }

        // Update high watermark
        self.update_high_watermark(&replica_info);
    }

    /// Update high watermark based on ISR offsets.
    fn update_high_watermark(&self, replica_info: &HashMap<BrokerId, ReplicaInfo>) {
        let isr = self.isr.read();

        let min_offset = isr
            .iter()
            .filter_map(|broker_id| replica_info.get(broker_id))
            .map(|info| info.log_end_offset)
            .min()
            .unwrap_or(0);

        self.high_watermark.store(min_offset, Ordering::Release);
    }

    /// Elect new leader from ISR.
    pub fn elect_leader(&self, preferred: Option<BrokerId>) -> Option<BrokerId> {
        let isr = self.isr.read();

        // Try preferred leader if in ISR
        if let Some(preferred_id) = preferred {
            if isr.contains(&preferred_id) {
                self.leader.store(preferred_id, Ordering::Release);
                self.leader_epoch.fetch_add(1, Ordering::AcqRel);
                return Some(preferred_id);
            }
        }

        // Otherwise, pick first available ISR member
        if let Some(&new_leader) = isr.iter().next() {
            self.leader.store(new_leader, Ordering::Release);
            self.leader_epoch.fetch_add(1, Ordering::AcqRel);
            return Some(new_leader);
        }

        None
    }

    /// Check if we have enough ISR members.
    pub fn has_min_isr(&self) -> bool {
        self.isr.read().len() >= self.config.min_isr as usize
    }

    /// Shrink ISR by removing slow replicas.
    pub fn maybe_shrink_isr(&self, leader_offset: u64) {
        let mut replica_info = self.replica_info.write();
        let mut isr = self.isr.write();

        let now = Instant::now();
        let mut to_remove = Vec::new();

        for broker_id in isr.iter() {
            if let Some(info) = replica_info.get_mut(broker_id) {
                let time_since_fetch = now.duration_since(info.last_fetch_time);

                if time_since_fetch > self.config.replica_lag_time_max
                    || info.lag(leader_offset) > self.config.max_lag_messages
                {
                    info.state = ReplicaState::Lagging;
                    to_remove.push(*broker_id);
                }
            }
        }

        for broker_id in to_remove {
            isr.remove(&broker_id);
            tracing::warn!(
                topic = %self.topic,
                partition = %self.partition,
                broker = %broker_id,
                "Replica removed from ISR"
            );
        }
    }

    /// Expand ISR by adding caught-up replicas.
    pub fn maybe_expand_isr(&self, leader_offset: u64) {
        let replica_info = self.replica_info.read();
        let mut isr = self.isr.write();

        for (&broker_id, info) in replica_info.iter() {
            if !isr.contains(&broker_id)
                && info.is_in_sync()
                && info.lag(leader_offset) <= self.config.max_lag_messages
            {
                isr.insert(broker_id);
                tracing::info!(
                    topic = %self.topic,
                    partition = %self.partition,
                    broker = %broker_id,
                    "Replica added to ISR"
                );
            }
        }
    }

    /// Get replica state summary.
    pub fn replica_states(&self) -> Vec<(BrokerId, ReplicaState, u64)> {
        let replica_info = self.replica_info.read();
        replica_info
            .iter()
            .map(|(&id, info)| (id, info.state, info.log_end_offset))
            .collect()
    }
}

/// Fetch request for replication.
#[derive(Debug, Clone)]
pub struct FetchRequest {
    /// Replica ID (broker making the request).
    pub replica_id: BrokerId,
    /// Max wait time.
    pub max_wait: Duration,
    /// Min bytes to return.
    pub min_bytes: usize,
    /// Max bytes to return.
    pub max_bytes: usize,
    /// Partitions to fetch.
    pub partitions: Vec<FetchPartition>,
}

/// Partition in a fetch request.
#[derive(Debug, Clone)]
pub struct FetchPartition {
    /// Topic name.
    pub topic: String,
    /// Partition ID.
    pub partition: PartitionId,
    /// Fetch offset.
    pub fetch_offset: u64,
    /// Max bytes for this partition.
    pub max_bytes: usize,
}

/// Fetch response.
#[derive(Debug)]
pub struct FetchResponse {
    /// Partitions with data.
    pub partitions: Vec<FetchPartitionResponse>,
    /// Throttle time if rate limited.
    pub throttle_time: Duration,
}

/// Partition in a fetch response.
#[derive(Debug)]
pub struct FetchPartitionResponse {
    /// Topic name.
    pub topic: String,
    /// Partition ID.
    pub partition: PartitionId,
    /// Error if any.
    pub error: Option<Error>,
    /// High watermark.
    pub high_watermark: u64,
    /// Last stable offset.
    pub last_stable_offset: u64,
    /// Log start offset.
    pub log_start_offset: u64,
    /// Fetched messages.
    pub messages: Vec<Message>,
}

/// Replication manager.
pub struct ReplicationManager {
    /// This broker's ID.
    broker_id: BrokerId,
    /// Replica sets by topic/partition.
    replica_sets: RwLock<HashMap<(String, PartitionId), Arc<ReplicaSet>>>,
    /// Configuration.
    config: ReplicationConfig,
}

impl ReplicationManager {
    /// Create a new replication manager.
    pub fn new(broker_id: BrokerId, config: ReplicationConfig) -> Self {
        Self {
            broker_id,
            replica_sets: RwLock::new(HashMap::new()),
            config,
        }
    }

    /// Get broker ID.
    pub fn broker_id(&self) -> BrokerId {
        self.broker_id
    }

    /// Get configuration.
    pub fn config(&self) -> &ReplicationConfig {
        &self.config
    }

    /// Add a replica set.
    pub fn add_replica_set(
        &self,
        topic: String,
        partition: PartitionId,
        replicas: Vec<BrokerId>,
        leader: BrokerId,
    ) {
        let replica_set = Arc::new(ReplicaSet::new(
            topic.clone(),
            partition,
            replicas,
            leader,
            self.config.clone(),
        ));

        self.replica_sets
            .write()
            .insert((topic, partition), replica_set);
    }

    /// Get replica set for a partition.
    pub fn get_replica_set(&self, topic: &str, partition: PartitionId) -> Option<Arc<ReplicaSet>> {
        self.replica_sets
            .read()
            .get(&(topic.to_string(), partition))
            .cloned()
    }

    /// Check if this broker is leader for a partition.
    pub fn is_leader(&self, topic: &str, partition: PartitionId) -> bool {
        self.get_replica_set(topic, partition)
            .map(|rs| rs.is_leader(self.broker_id))
            .unwrap_or(false)
    }

    /// Get leader for a partition.
    pub fn get_leader(&self, topic: &str, partition: PartitionId) -> Option<BrokerId> {
        self.get_replica_set(topic, partition).map(|rs| rs.leader())
    }

    /// Get all partitions this broker is leader for.
    pub fn leader_partitions(&self) -> Vec<(String, PartitionId)> {
        self.replica_sets
            .read()
            .iter()
            .filter(|(_, rs)| rs.is_leader(self.broker_id))
            .map(|((topic, partition), _)| (topic.clone(), *partition))
            .collect()
    }

    /// Get all partitions this broker is a follower for.
    pub fn follower_partitions(&self) -> Vec<(String, PartitionId)> {
        self.replica_sets
            .read()
            .iter()
            .filter(|(_, rs)| {
                rs.replicas().contains(&self.broker_id) && !rs.is_leader(self.broker_id)
            })
            .map(|((topic, partition), _)| (topic.clone(), *partition))
            .collect()
    }

    /// Process replica fetch and update progress.
    pub fn process_fetch(
        &self,
        topic: &str,
        partition: PartitionId,
        replica_id: BrokerId,
        fetch_offset: u64,
        leader_log_end_offset: u64,
    ) {
        if let Some(replica_set) = self.get_replica_set(topic, partition) {
            replica_set.update_replica_fetch(replica_id, fetch_offset, leader_log_end_offset);
        }
    }

    /// Check if partition has minimum ISR.
    pub fn has_min_isr(&self, topic: &str, partition: PartitionId) -> bool {
        self.get_replica_set(topic, partition)
            .map(|rs| rs.has_min_isr())
            .unwrap_or(false)
    }

    /// Get high watermark for a partition.
    pub fn high_watermark(&self, topic: &str, partition: PartitionId) -> Option<u64> {
        self.get_replica_set(topic, partition)
            .map(|rs| rs.high_watermark())
    }

    /// Trigger leader election for a partition.
    pub fn elect_leader(
        &self,
        topic: &str,
        partition: PartitionId,
        preferred: Option<BrokerId>,
    ) -> Option<BrokerId> {
        self.get_replica_set(topic, partition)
            .and_then(|rs| rs.elect_leader(preferred))
    }

    /// Perform ISR maintenance for all partitions.
    pub fn maintain_isr(&self, get_log_end_offset: impl Fn(&str, PartitionId) -> u64) {
        for ((topic, partition), replica_set) in self.replica_sets.read().iter() {
            if replica_set.is_leader(self.broker_id) {
                let leader_offset = get_log_end_offset(topic, *partition);
                replica_set.maybe_shrink_isr(leader_offset);
                replica_set.maybe_expand_isr(leader_offset);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replica_info_new() {
        let info = ReplicaInfo::new(1);
        assert_eq!(info.broker_id, 1);
        assert_eq!(info.log_end_offset, 0);
        assert_eq!(info.state, ReplicaState::Syncing);
    }

    #[test]
    fn test_replica_info_update_progress() {
        let mut info = ReplicaInfo::new(1);
        info.update_progress(100, 100, 4000);
        assert!(info.is_in_sync());
        assert_eq!(info.log_end_offset, 100);
    }

    #[test]
    fn test_replica_info_lagging() {
        let mut info = ReplicaInfo::new(1);
        info.update_progress(0, 10000, 4000);
        assert_eq!(info.state, ReplicaState::Lagging);
        assert!(!info.is_in_sync());
    }

    #[test]
    fn test_replica_set_new() {
        let rs = ReplicaSet::new(
            "test".to_string(),
            0,
            vec![1, 2, 3],
            1,
            ReplicationConfig::default(),
        );

        assert_eq!(rs.topic(), "test");
        assert_eq!(rs.partition(), 0);
        assert_eq!(rs.leader(), 1);
        assert_eq!(rs.replicas(), &[1, 2, 3]);
        assert!(rs.is_leader(1));
        assert!(!rs.is_leader(2));
    }

    #[test]
    fn test_replica_set_isr() {
        let rs = ReplicaSet::new(
            "test".to_string(),
            0,
            vec![1, 2, 3],
            1,
            ReplicationConfig::default(),
        );

        let isr = rs.isr();
        assert_eq!(isr.len(), 3);
        assert!(rs.is_in_isr(1));
        assert!(rs.is_in_isr(2));
        assert!(rs.is_in_isr(3));
    }

    #[test]
    fn test_replica_set_update_fetch() {
        let rs = ReplicaSet::new(
            "test".to_string(),
            0,
            vec![1, 2, 3],
            1,
            ReplicationConfig::default(),
        );

        rs.update_replica_fetch(2, 100, 100);

        let states = rs.replica_states();
        let follower = states.iter().find(|(id, _, _)| *id == 2).unwrap();
        assert_eq!(follower.2, 100);
    }

    #[test]
    fn test_replica_set_high_watermark() {
        let rs = ReplicaSet::new(
            "test".to_string(),
            0,
            vec![1, 2, 3],
            1,
            ReplicationConfig::default(),
        );

        rs.update_replica_fetch(1, 100, 100);
        rs.update_replica_fetch(2, 80, 100);
        rs.update_replica_fetch(3, 90, 100);

        assert_eq!(rs.high_watermark(), 80);
    }

    #[test]
    fn test_replica_set_elect_leader() {
        let rs = ReplicaSet::new(
            "test".to_string(),
            0,
            vec![1, 2, 3],
            1,
            ReplicationConfig::default(),
        );

        let new_leader = rs.elect_leader(Some(2));
        assert_eq!(new_leader, Some(2));
        assert_eq!(rs.leader(), 2);
        assert_eq!(rs.leader_epoch(), 1);
    }

    #[test]
    fn test_replica_set_has_min_isr() {
        let config = ReplicationConfig {
            min_isr: 2,
            ..Default::default()
        };

        let rs = ReplicaSet::new("test".to_string(), 0, vec![1, 2, 3], 1, config);
        assert!(rs.has_min_isr());
    }

    #[test]
    fn test_replication_manager_new() {
        let manager = ReplicationManager::new(1, ReplicationConfig::default());
        assert_eq!(manager.broker_id(), 1);
    }

    #[test]
    fn test_replication_manager_add_get() {
        let manager = ReplicationManager::new(1, ReplicationConfig::default());

        manager.add_replica_set("test".to_string(), 0, vec![1, 2, 3], 1);

        let rs = manager.get_replica_set("test", 0);
        assert!(rs.is_some());
        assert_eq!(rs.unwrap().leader(), 1);
    }

    #[test]
    fn test_replication_manager_is_leader() {
        let manager = ReplicationManager::new(1, ReplicationConfig::default());

        manager.add_replica_set("test".to_string(), 0, vec![1, 2, 3], 1);

        assert!(manager.is_leader("test", 0));
    }

    #[test]
    fn test_replication_manager_leader_partitions() {
        let manager = ReplicationManager::new(1, ReplicationConfig::default());

        manager.add_replica_set("topic1".to_string(), 0, vec![1, 2, 3], 1);
        manager.add_replica_set("topic1".to_string(), 1, vec![1, 2, 3], 2);
        manager.add_replica_set("topic2".to_string(), 0, vec![1, 2, 3], 1);

        let leader_partitions = manager.leader_partitions();
        assert_eq!(leader_partitions.len(), 2);
    }

    #[test]
    fn test_replication_manager_follower_partitions() {
        let manager = ReplicationManager::new(2, ReplicationConfig::default());

        manager.add_replica_set("topic1".to_string(), 0, vec![1, 2, 3], 1);
        manager.add_replica_set("topic1".to_string(), 1, vec![1, 2, 3], 2);

        let follower_partitions = manager.follower_partitions();
        assert_eq!(follower_partitions.len(), 1);
    }

    #[test]
    fn test_replication_manager_process_fetch() {
        let manager = ReplicationManager::new(1, ReplicationConfig::default());

        manager.add_replica_set("test".to_string(), 0, vec![1, 2, 3], 1);
        manager.process_fetch("test", 0, 2, 100, 100);

        let rs = manager.get_replica_set("test", 0).unwrap();
        let states = rs.replica_states();
        let follower = states.iter().find(|(id, _, _)| *id == 2).unwrap();
        assert_eq!(follower.2, 100);
    }

    #[test]
    fn test_replication_manager_high_watermark() {
        let manager = ReplicationManager::new(1, ReplicationConfig::default());

        manager.add_replica_set("test".to_string(), 0, vec![1, 2, 3], 1);

        let hw = manager.high_watermark("test", 0);
        assert!(hw.is_some());
    }

    #[test]
    fn test_acks_default() {
        let acks = Acks::default();
        assert_eq!(acks, Acks::Leader);
    }

    #[test]
    fn test_replication_config_default() {
        let config = ReplicationConfig::default();
        assert_eq!(config.replication_factor, 3);
        assert_eq!(config.min_isr, 2);
        assert_eq!(config.max_lag_messages, 4000);
    }
}
