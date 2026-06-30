//! Replication: ReplicaManager and ReplicaFetcher for leader-follower replication.

use crate::broker::{BrokerId, PartitionAssignment};
use crate::log::{Partition, RecordBatch, SegmentConfig, TopicPartition};
use crate::protocol::{FetchPartition, FetchRequest, FetchResponse};
use crate::{Error, Offset, Result};

use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Replication configuration.
#[derive(Debug, Clone)]
pub struct ReplicationConfig {
    /// Replica fetch interval in milliseconds.
    pub fetch_interval_ms: u64,
    /// Maximum wait time for fetch requests.
    pub fetch_max_wait_ms: u64,
    /// Minimum bytes for fetch requests.
    pub fetch_min_bytes: u32,
    /// Maximum bytes per partition for fetch.
    pub fetch_max_bytes: u32,
    /// ISR lag time threshold in milliseconds.
    pub replica_lag_time_max_ms: u64,
    /// ISR lag offset threshold.
    pub replica_lag_max_messages: u64,
    /// Number of fetcher threads.
    pub num_fetcher_threads: u32,
    /// Fetch backoff on error.
    pub fetch_backoff_ms: u64,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            fetch_interval_ms: 100,
            fetch_max_wait_ms: 500,
            fetch_min_bytes: 1,
            fetch_max_bytes: 1024 * 1024,
            replica_lag_time_max_ms: 10000,
            replica_lag_max_messages: 4000,
            num_fetcher_threads: 1,
            fetch_backoff_ms: 100,
        }
    }
}

/// State of a replica on a follower.
#[derive(Debug, Clone)]
pub struct ReplicaState {
    /// The leader broker ID.
    pub leader_id: BrokerId,
    /// Current leader epoch.
    pub leader_epoch: u32,
    /// Next offset to fetch from leader.
    pub fetch_offset: Offset,
    /// Local log end offset.
    pub log_end_offset: Offset,
    /// Local high watermark.
    pub high_watermark: Offset,
    /// Last successful fetch time.
    pub last_fetch_time: Instant,
    /// Whether currently fetching.
    pub is_fetching: bool,
    /// Consecutive fetch failures.
    pub fetch_failures: u32,
}

impl ReplicaState {
    pub fn new(leader_id: BrokerId, leader_epoch: u32) -> Self {
        Self {
            leader_id,
            leader_epoch,
            fetch_offset: 0,
            log_end_offset: 0,
            high_watermark: 0,
            last_fetch_time: Instant::now(),
            is_fetching: false,
            fetch_failures: 0,
        }
    }
}

/// Tracks state of remote replicas (from leader's perspective).
#[derive(Debug, Clone)]
pub struct RemoteReplicaState {
    /// Broker ID of the replica.
    pub broker_id: BrokerId,
    /// Last known fetch offset (log end offset on replica).
    pub log_end_offset: Offset,
    /// Last fetch time.
    pub last_fetch_time: Instant,
    /// Whether replica is in sync.
    pub in_sync: bool,
}

impl RemoteReplicaState {
    pub fn new(broker_id: BrokerId) -> Self {
        Self {
            broker_id,
            log_end_offset: 0,
            last_fetch_time: Instant::now(),
            in_sync: true,
        }
    }
}

/// Replica manager handles all replication for partitions on this broker.
pub struct ReplicaManager {
    /// This broker's ID.
    broker_id: BrokerId,
    /// Replication configuration.
    config: ReplicationConfig,
    /// Data directory.
    data_dir: PathBuf,
    /// Segment configuration.
    segment_config: SegmentConfig,
    /// Partitions this broker hosts.
    partitions: DashMap<TopicPartition, RwLock<Partition>>,
    /// Partition assignments.
    assignments: DashMap<TopicPartition, PartitionAssignment>,
    /// Leader partitions (partitions where this broker is leader).
    leader_partitions: DashMap<TopicPartition, ()>,
    /// Follower partitions with their state.
    follower_partitions: DashMap<TopicPartition, RwLock<ReplicaState>>,
    /// Remote replica states (for partitions where we're leader).
    remote_replicas: DashMap<(TopicPartition, BrokerId), RwLock<RemoteReplicaState>>,
    /// Whether the manager is running.
    running: AtomicBool,
}

impl ReplicaManager {
    /// Create a new replica manager.
    pub fn new(
        broker_id: BrokerId,
        config: ReplicationConfig,
        data_dir: PathBuf,
        segment_config: SegmentConfig,
    ) -> Self {
        Self {
            broker_id,
            config,
            data_dir,
            segment_config,
            partitions: DashMap::new(),
            assignments: DashMap::new(),
            leader_partitions: DashMap::new(),
            follower_partitions: DashMap::new(),
            remote_replicas: DashMap::new(),
            running: AtomicBool::new(false),
        }
    }

    /// Start the replica manager.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        info!("Replica manager started for broker {}", self.broker_id);
    }

    /// Stop the replica manager.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("Replica manager stopped for broker {}", self.broker_id);
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Become leader for a partition.
    pub fn become_leader(
        &self,
        tp: &TopicPartition,
        assignment: PartitionAssignment,
    ) -> Result<()> {
        info!(
            "Broker {} becoming leader for {:?} with epoch {}",
            self.broker_id, tp, assignment.leader_epoch
        );

        // Ensure we have the partition
        if !self.partitions.contains_key(tp) {
            let partition = Partition::new(
                &self.data_dir,
                &tp.topic,
                tp.partition,
                self.segment_config.clone(),
            )?;
            self.partitions.insert(tp.clone(), RwLock::new(partition));
        }

        // Update assignment
        self.assignments.insert(tp.clone(), assignment.clone());

        // Track as leader
        self.leader_partitions.insert(tp.clone(), ());
        self.follower_partitions.remove(tp);

        // Initialize remote replica tracking
        for replica_id in &assignment.replicas {
            if *replica_id != self.broker_id {
                let key = (tp.clone(), *replica_id);
                self.remote_replicas
                    .insert(key, RwLock::new(RemoteReplicaState::new(*replica_id)));
            }
        }

        Ok(())
    }

    /// Become follower for a partition.
    pub fn become_follower(
        &self,
        tp: &TopicPartition,
        assignment: PartitionAssignment,
    ) -> Result<()> {
        info!(
            "Broker {} becoming follower for {:?} (leader: {}, epoch: {})",
            self.broker_id, tp, assignment.leader, assignment.leader_epoch
        );

        // Ensure we have the partition
        if !self.partitions.contains_key(tp) {
            let partition = Partition::new(
                &self.data_dir,
                &tp.topic,
                tp.partition,
                self.segment_config.clone(),
            )?;
            self.partitions.insert(tp.clone(), RwLock::new(partition));
        }

        // Get current log end offset
        let log_end_offset = {
            let partition = self.partitions.get(tp).unwrap();
            let p = partition.read();
            p.log_end_offset
        };

        // Update assignment
        self.assignments.insert(tp.clone(), assignment.clone());

        // Track as follower
        self.leader_partitions.remove(tp);
        let mut state = ReplicaState::new(assignment.leader, assignment.leader_epoch);
        state.fetch_offset = log_end_offset;
        state.log_end_offset = log_end_offset;
        self.follower_partitions
            .insert(tp.clone(), RwLock::new(state));

        // Clean up remote replica tracking
        for key in self.remote_replicas.iter() {
            if key.key().0 == *tp {
                self.remote_replicas.remove(key.key());
            }
        }

        Ok(())
    }

    /// Stop replica (partition removed from this broker).
    pub fn stop_replica(&self, tp: &TopicPartition) -> Result<()> {
        info!("Stopping replica for {:?} on broker {}", tp, self.broker_id);

        self.partitions.remove(tp);
        self.assignments.remove(tp);
        self.leader_partitions.remove(tp);
        self.follower_partitions.remove(tp);

        // Clean up remote replica tracking
        let keys_to_remove: Vec<_> = self
            .remote_replicas
            .iter()
            .filter(|e| e.key().0 == *tp)
            .map(|e| e.key().clone())
            .collect();
        for key in keys_to_remove {
            self.remote_replicas.remove(&key);
        }

        Ok(())
    }

    /// Check if this broker is leader for partition.
    pub fn is_leader(&self, tp: &TopicPartition) -> bool {
        self.leader_partitions.contains_key(tp)
    }

    /// Check if this broker is follower for partition.
    pub fn is_follower(&self, tp: &TopicPartition) -> bool {
        self.follower_partitions.contains_key(tp)
    }

    /// Append records to leader partition.
    pub fn append_to_leader(&self, tp: &TopicPartition, batch: RecordBatch) -> Result<Offset> {
        if !self.is_leader(tp) {
            return Err(Error::NotLeader);
        }

        let partition = self
            .partitions
            .get(tp)
            .ok_or(Error::PartitionNotFound(tp.partition))?;
        let mut partition = partition.write();

        let offset = partition.append(batch)?;
        Ok(offset)
    }

    /// Append records as follower (from replication).
    pub fn append_as_follower(
        &self,
        tp: &TopicPartition,
        batches: Vec<RecordBatch>,
    ) -> Result<()> {
        if !self.is_follower(tp) {
            return Err(Error::Internal("Not a follower".into()));
        }

        let partition = self
            .partitions
            .get(tp)
            .ok_or(Error::PartitionNotFound(tp.partition))?;
        let mut partition = partition.write();

        for batch in batches {
            partition.append(batch)?;
        }

        // Update follower state
        if let Some(state) = self.follower_partitions.get(tp) {
            let mut state = state.write();
            state.log_end_offset = partition.log_end_offset;
            state.fetch_offset = partition.log_end_offset;
            state.last_fetch_time = Instant::now();
        }

        Ok(())
    }

    /// Read from partition (for fetch requests).
    pub fn read(
        &self,
        tp: &TopicPartition,
        offset: Offset,
        max_bytes: u32,
    ) -> Result<Vec<RecordBatch>> {
        let partition = self
            .partitions
            .get(tp)
            .ok_or(Error::PartitionNotFound(tp.partition))?;
        let mut partition = partition.write();

        partition.read(offset, max_bytes)
    }

    /// Update high watermark for a partition.
    pub fn update_high_watermark(&self, tp: &TopicPartition, hw: Offset) -> Result<()> {
        let partition = self
            .partitions
            .get(tp)
            .ok_or(Error::PartitionNotFound(tp.partition))?;
        let mut partition = partition.write();

        partition.update_high_watermark(hw);

        // Update follower state if applicable
        if let Some(state) = self.follower_partitions.get(tp) {
            let mut state = state.write();
            state.high_watermark = hw;
        }

        Ok(())
    }

    /// Handle fetch request from a replica.
    pub fn handle_replica_fetch(
        &self,
        replica_id: BrokerId,
        tp: &TopicPartition,
        fetch_offset: Offset,
    ) -> Result<(Vec<RecordBatch>, Offset)> {
        if !self.is_leader(tp) {
            return Err(Error::NotLeader);
        }

        // Update remote replica state
        let key = (tp.clone(), replica_id);
        if let Some(state) = self.remote_replicas.get(&key) {
            let mut state = state.write();
            state.log_end_offset = fetch_offset;
            state.last_fetch_time = Instant::now();
        }

        // Read records
        let batches = self.read(tp, fetch_offset, self.config.fetch_max_bytes)?;

        // Get high watermark
        let hw = self.get_high_watermark(tp)?;

        // Maybe update ISR and high watermark
        self.maybe_update_isr_and_hw(tp)?;

        Ok((batches, hw))
    }

    /// Get high watermark for a partition.
    pub fn get_high_watermark(&self, tp: &TopicPartition) -> Result<Offset> {
        let partition = self
            .partitions
            .get(tp)
            .ok_or(Error::PartitionNotFound(tp.partition))?;
        let partition = partition.read();

        Ok(partition.high_watermark)
    }

    /// Get log end offset for a partition.
    pub fn get_log_end_offset(&self, tp: &TopicPartition) -> Result<Offset> {
        let partition = self
            .partitions
            .get(tp)
            .ok_or(Error::PartitionNotFound(tp.partition))?;
        let partition = partition.read();

        Ok(partition.log_end_offset)
    }

    /// Update ISR and high watermark for a leader partition.
    pub fn maybe_update_isr_and_hw(&self, tp: &TopicPartition) -> Result<bool> {
        if !self.is_leader(tp) {
            return Ok(false);
        }

        let assignment = self
            .assignments
            .get(tp)
            .ok_or(Error::PartitionNotFound(tp.partition))?;

        let partition = self
            .partitions
            .get(tp)
            .ok_or(Error::PartitionNotFound(tp.partition))?;
        let partition = partition.read();

        let log_end_offset = partition.log_end_offset;
        let now = Instant::now();
        let max_lag_time = Duration::from_millis(self.config.replica_lag_time_max_ms);

        let mut new_isr = HashSet::new();
        new_isr.insert(self.broker_id); // Leader always in ISR

        // Check each replica
        for replica_id in &assignment.replicas {
            if *replica_id == self.broker_id {
                continue;
            }

            let key = (tp.clone(), *replica_id);
            if let Some(state) = self.remote_replicas.get(&key) {
                let state = state.read();
                let lag_time = now.duration_since(state.last_fetch_time);
                let lag_offset = log_end_offset.saturating_sub(state.log_end_offset);

                if lag_time < max_lag_time && lag_offset < self.config.replica_lag_max_messages {
                    new_isr.insert(*replica_id);
                }
            }
        }

        // Calculate new high watermark (min offset among ISR)
        let mut min_offset = log_end_offset;
        for replica_id in &new_isr {
            if *replica_id == self.broker_id {
                continue;
            }

            let key = (tp.clone(), *replica_id);
            if let Some(state) = self.remote_replicas.get(&key) {
                let state = state.read();
                min_offset = min_offset.min(state.log_end_offset);
            }
        }

        drop(partition);

        // Update high watermark
        let partition = self.partitions.get(tp).unwrap();
        let mut partition = partition.write();
        let old_hw = partition.high_watermark;
        partition.high_watermark = min_offset;

        // Check if ISR changed
        let current_isr: HashSet<BrokerId> = assignment.isr.iter().copied().collect();
        let isr_changed = new_isr != current_isr;

        if isr_changed || min_offset > old_hw {
            debug!(
                "Partition {:?}: ISR changed: {}, HW: {} -> {}",
                tp, isr_changed, old_hw, min_offset
            );
        }

        Ok(isr_changed)
    }

    /// Get follower partitions that need fetching.
    pub fn get_follower_partitions_to_fetch(&self) -> Vec<(TopicPartition, BrokerId, Offset)> {
        let mut result = Vec::new();

        for entry in self.follower_partitions.iter() {
            let tp = entry.key();
            let state = entry.value().read();

            if !state.is_fetching {
                result.push((tp.clone(), state.leader_id, state.fetch_offset));
            }
        }

        result
    }

    /// Mark partition as fetching.
    pub fn mark_fetching(&self, tp: &TopicPartition, fetching: bool) {
        if let Some(state) = self.follower_partitions.get(tp) {
            let mut state = state.write();
            state.is_fetching = fetching;
        }
    }

    /// Record fetch failure.
    pub fn record_fetch_failure(&self, tp: &TopicPartition) {
        if let Some(state) = self.follower_partitions.get(tp) {
            let mut state = state.write();
            state.fetch_failures += 1;
            state.is_fetching = false;
        }
    }

    /// Get partition info.
    pub fn get_partition_info(&self, tp: &TopicPartition) -> Option<PartitionInfo> {
        let partition = self.partitions.get(tp)?;
        let partition = partition.read();
        let assignment = self.assignments.get(tp)?;

        Some(PartitionInfo {
            topic: tp.topic.clone(),
            partition: tp.partition,
            leader: assignment.leader,
            replicas: assignment.replicas.clone(),
            isr: assignment.isr.clone(),
            high_watermark: partition.high_watermark,
            log_end_offset: partition.log_end_offset,
            is_leader: self.is_leader(tp),
        })
    }

    /// Get all leader partitions.
    pub fn leader_partitions(&self) -> Vec<TopicPartition> {
        self.leader_partitions
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    /// Get all follower partitions.
    pub fn follower_partitions(&self) -> Vec<TopicPartition> {
        self.follower_partitions
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    /// Get all partitions.
    pub fn all_partitions(&self) -> Vec<TopicPartition> {
        self.partitions.iter().map(|e| e.key().clone()).collect()
    }

    /// Flush all partitions to disk.
    pub fn flush_all(&self) -> Result<()> {
        for entry in self.partitions.iter() {
            let mut partition = entry.value().write();
            partition.flush()?;
        }
        Ok(())
    }
}

/// Partition information.
#[derive(Debug, Clone)]
pub struct PartitionInfo {
    pub topic: String,
    pub partition: u32,
    pub leader: BrokerId,
    pub replicas: Vec<BrokerId>,
    pub isr: Vec<BrokerId>,
    pub high_watermark: Offset,
    pub log_end_offset: Offset,
    pub is_leader: bool,
}

/// Replica fetcher that fetches data from leaders.
pub struct ReplicaFetcher {
    /// Fetcher ID.
    id: u32,
    /// Broker ID.
    broker_id: BrokerId,
    /// Configuration.
    config: ReplicationConfig,
    /// Replica manager reference.
    replica_manager: Arc<ReplicaManager>,
    /// Whether the fetcher is running.
    running: AtomicBool,
    /// Pending fetch results.
    pending_results: DashMap<TopicPartition, FetchResult>,
}

/// Result of a fetch operation.
#[derive(Debug, Clone)]
pub struct FetchResult {
    /// Topic-partition.
    pub tp: TopicPartition,
    /// Fetched batches.
    pub batches: Vec<RecordBatch>,
    /// Leader's high watermark.
    pub high_watermark: Offset,
    /// Error if any.
    pub error: Option<String>,
}

impl ReplicaFetcher {
    /// Create a new replica fetcher.
    pub fn new(
        id: u32,
        broker_id: BrokerId,
        config: ReplicationConfig,
        replica_manager: Arc<ReplicaManager>,
    ) -> Self {
        Self {
            id,
            broker_id,
            config,
            replica_manager,
            running: AtomicBool::new(false),
            pending_results: DashMap::new(),
        }
    }

    /// Start the fetcher.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        info!("Replica fetcher {} started for broker {}", self.id, self.broker_id);
    }

    /// Stop the fetcher.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("Replica fetcher {} stopped", self.id);
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Build fetch request for all follower partitions.
    pub fn build_fetch_requests(&self) -> HashMap<BrokerId, FetchRequest> {
        let partitions = self.replica_manager.get_follower_partitions_to_fetch();
        let mut requests: HashMap<BrokerId, Vec<FetchPartition>> = HashMap::new();

        for (tp, leader_id, fetch_offset) in partitions {
            requests.entry(leader_id).or_default().push(FetchPartition {
                topic: tp.topic.clone(),
                partition: tp.partition,
                fetch_offset,
                max_bytes: self.config.fetch_max_bytes,
            });

            // Mark as fetching
            self.replica_manager.mark_fetching(&tp, true);
        }

        requests
            .into_iter()
            .map(|(leader_id, partitions)| {
                (
                    leader_id,
                    FetchRequest {
                        replica_id: self.broker_id as i32,
                        max_wait_ms: self.config.fetch_max_wait_ms as u32,
                        min_bytes: self.config.fetch_min_bytes,
                        max_bytes: self.config.fetch_max_bytes * partitions.len() as u32,
                        isolation_level: 0,
                        session_id: 0,
                        session_epoch: 0,
                        partitions,
                    },
                )
            })
            .collect()
    }

    /// Process fetch response from a leader.
    pub fn process_fetch_response(&self, response: FetchResponse) -> Result<()> {
        for partition_response in response.responses {
            let tp = TopicPartition::new(
                // Topic is not in FetchPartitionResponse, need to track separately
                // For now, we'll use a placeholder approach
                "",
                partition_response.partition,
            );

            if partition_response.error_code != 0 {
                self.replica_manager.record_fetch_failure(&tp);
                continue;
            }

            // Append fetched records
            if !partition_response.record_batches.is_empty() {
                if let Err(e) = self
                    .replica_manager
                    .append_as_follower(&tp, partition_response.record_batches)
                {
                    error!("Failed to append as follower for {:?}: {}", tp, e);
                    self.replica_manager.record_fetch_failure(&tp);
                    continue;
                }
            }

            // Update high watermark
            self.replica_manager
                .update_high_watermark(&tp, partition_response.high_watermark)?;

            // Mark fetch complete
            self.replica_manager.mark_fetching(&tp, false);
        }

        Ok(())
    }

    /// Process fetch responses with topic information.
    pub fn process_fetch_response_with_topics(
        &self,
        responses: Vec<(TopicPartition, Vec<RecordBatch>, Offset)>,
    ) -> Result<()> {
        for (tp, batches, high_watermark) in responses {
            // Append fetched records
            if !batches.is_empty() {
                if let Err(e) = self.replica_manager.append_as_follower(&tp, batches) {
                    error!("Failed to append as follower for {:?}: {}", tp, e);
                    self.replica_manager.record_fetch_failure(&tp);
                    continue;
                }
            }

            // Update high watermark
            self.replica_manager.update_high_watermark(&tp, high_watermark)?;

            // Mark fetch complete
            self.replica_manager.mark_fetching(&tp, false);
        }

        Ok(())
    }

    /// Get pending results.
    pub fn get_pending_results(&self) -> Vec<FetchResult> {
        let mut results = Vec::new();
        for entry in self.pending_results.iter() {
            results.push(entry.value().clone());
        }
        self.pending_results.clear();
        results
    }
}

/// Delayed produce operation waiting for replication.
#[derive(Debug)]
pub struct DelayedProduce {
    /// Topic-partition.
    pub tp: TopicPartition,
    /// Required offset to be replicated.
    pub required_offset: Offset,
    /// Required acknowledgment level.
    pub required_acks: i16,
    /// Deadline for the operation.
    pub deadline: Instant,
    /// Whether completed.
    pub completed: bool,
}

impl DelayedProduce {
    /// Create a new delayed produce.
    pub fn new(tp: TopicPartition, required_offset: Offset, required_acks: i16, timeout: Duration) -> Self {
        Self {
            tp,
            required_offset,
            required_acks,
            deadline: Instant::now() + timeout,
            completed: false,
        }
    }

    /// Check if this operation can be completed.
    pub fn try_complete(&mut self, high_watermark: Offset, isr_size: usize) -> bool {
        if self.completed {
            return true;
        }

        let can_complete = match self.required_acks {
            0 => true,                              // acks=0: fire and forget
            1 => true,                              // acks=1: leader ack (already done)
            -1 => high_watermark >= self.required_offset, // acks=all: wait for ISR
            _ => false,
        };

        if can_complete {
            self.completed = true;
        }

        can_complete
    }

    /// Check if expired.
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.deadline
    }
}

/// Purgatory for delayed operations.
pub struct DelayedOperationPurgatory<T> {
    /// Pending operations by topic-partition.
    operations: DashMap<TopicPartition, VecDeque<T>>,
}

impl<T> DelayedOperationPurgatory<T> {
    /// Create a new purgatory.
    pub fn new() -> Self {
        Self {
            operations: DashMap::new(),
        }
    }

    /// Add an operation.
    pub fn add(&self, tp: TopicPartition, operation: T) {
        self.operations.entry(tp).or_default().push_back(operation);
    }

    /// Get operations for a partition.
    pub fn get_operations(&self, tp: &TopicPartition) -> Option<VecDeque<T>> {
        self.operations.remove(tp).map(|(_, v)| v)
    }
}

impl<T> Default for DelayedOperationPurgatory<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_replica_manager() -> ReplicaManager {
        let dir = tempdir().unwrap();
        ReplicaManager::new(
            0,
            ReplicationConfig::default(),
            dir.into_path(),
            SegmentConfig::default(),
        )
    }

    #[test]
    fn test_become_leader() {
        let rm = create_test_replica_manager();
        let tp = TopicPartition::new("test", 0);
        let assignment = PartitionAssignment {
            leader: 0,
            replicas: vec![0, 1, 2],
            isr: vec![0, 1, 2],
            leader_epoch: 1,
        };

        rm.become_leader(&tp, assignment).unwrap();

        assert!(rm.is_leader(&tp));
        assert!(!rm.is_follower(&tp));
    }

    #[test]
    fn test_become_follower() {
        let rm = create_test_replica_manager();
        let tp = TopicPartition::new("test", 0);
        let assignment = PartitionAssignment {
            leader: 1,
            replicas: vec![0, 1, 2],
            isr: vec![0, 1, 2],
            leader_epoch: 1,
        };

        rm.become_follower(&tp, assignment).unwrap();

        assert!(!rm.is_leader(&tp));
        assert!(rm.is_follower(&tp));
    }

    #[test]
    fn test_append_to_leader() {
        let rm = create_test_replica_manager();
        let tp = TopicPartition::new("test", 0);
        let assignment = PartitionAssignment {
            leader: 0,
            replicas: vec![0],
            isr: vec![0],
            leader_epoch: 1,
        };

        rm.become_leader(&tp, assignment).unwrap();

        let batch = RecordBatch::new(
            0,
            vec![crate::log::Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(b"key".to_vec()),
                value: Some(b"value".to_vec()),
                headers: vec![],
            }],
        );

        let offset = rm.append_to_leader(&tp, batch).unwrap();
        assert_eq!(offset, 0);

        let leo = rm.get_log_end_offset(&tp).unwrap();
        assert_eq!(leo, 1);
    }

    #[test]
    fn test_read_from_partition() {
        let rm = create_test_replica_manager();
        let tp = TopicPartition::new("test", 0);
        let assignment = PartitionAssignment {
            leader: 0,
            replicas: vec![0],
            isr: vec![0],
            leader_epoch: 1,
        };

        rm.become_leader(&tp, assignment).unwrap();

        let batch = RecordBatch::new(
            0,
            vec![crate::log::Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(b"key".to_vec()),
                value: Some(b"value".to_vec()),
                headers: vec![],
            }],
        );

        rm.append_to_leader(&tp, batch).unwrap();

        let batches = rm.read(&tp, 0, 1024).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].records[0].key, Some(b"key".to_vec()));
    }

    #[test]
    fn test_delayed_produce() {
        let tp = TopicPartition::new("test", 0);
        let mut delayed = DelayedProduce::new(tp, 10, -1, Duration::from_secs(30));

        assert!(!delayed.try_complete(5, 3));
        assert!(!delayed.completed);

        assert!(delayed.try_complete(10, 3));
        assert!(delayed.completed);
    }
}
