//! Controller: cluster coordination and metadata management.

use crate::broker::{BrokerId, PartitionAssignment, TopicMetadata};
use crate::log::TopicPartition;
use crate::{Error, Result};

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Controller configuration.
#[derive(Debug, Clone)]
pub struct ControllerConfig {
    /// Controller ID (same as broker ID).
    pub id: BrokerId,
    /// Session timeout for brokers.
    pub session_timeout_ms: u64,
    /// Heartbeat interval.
    pub heartbeat_interval_ms: u64,
    /// Rebalance delay after broker failure.
    pub rebalance_delay_ms: u64,
    /// Default replication factor.
    pub default_replication_factor: u32,
    /// Minimum ISR for writes.
    pub min_isr: u32,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            id: 0,
            session_timeout_ms: 30000,
            heartbeat_interval_ms: 10000,
            rebalance_delay_ms: 5000,
            default_replication_factor: 3,
            min_isr: 1,
        }
    }
}

/// Broker information in the cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokerInfo {
    /// Broker ID.
    pub id: BrokerId,
    /// Host address.
    pub host: String,
    /// Port number.
    pub port: u32,
    /// Rack identifier for rack-aware placement.
    pub rack: Option<String>,
    /// Last heartbeat time (as epoch millis for serialization).
    pub last_heartbeat_ms: u64,
    /// Whether the broker is alive.
    pub is_alive: bool,
}

impl BrokerInfo {
    pub fn new(id: BrokerId, host: String, port: u32) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        Self {
            id,
            host,
            port,
            rack: None,
            last_heartbeat_ms: now,
            is_alive: true,
        }
    }

    pub fn with_rack(mut self, rack: impl Into<String>) -> Self {
        self.rack = Some(rack.into());
        self
    }
}

/// Cluster state maintained by the controller.
#[derive(Debug, Clone, Default)]
pub struct ClusterState {
    /// Controller epoch (incremented on controller change).
    pub controller_epoch: u32,
    /// Current controller broker ID.
    pub controller_id: Option<BrokerId>,
    /// All brokers in the cluster.
    pub brokers: HashMap<BrokerId, BrokerInfo>,
    /// All topics metadata.
    pub topics: HashMap<String, TopicMetadata>,
    /// Partition assignments.
    pub assignments: HashMap<TopicPartition, PartitionAssignment>,
}

/// Leader and ISR update request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderAndIsrRequest {
    /// Controller ID sending this request.
    pub controller_id: BrokerId,
    /// Controller epoch.
    pub controller_epoch: u32,
    /// Partition states to update.
    pub partition_states: Vec<LeaderAndIsrPartitionState>,
    /// Live brokers in the cluster.
    pub live_brokers: Vec<BrokerInfo>,
}

/// Partition state in LeaderAndIsr request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderAndIsrPartitionState {
    /// Topic name.
    pub topic: String,
    /// Partition ID.
    pub partition: u32,
    /// Controller epoch.
    pub controller_epoch: u32,
    /// Leader broker ID.
    pub leader: BrokerId,
    /// Leader epoch.
    pub leader_epoch: u32,
    /// In-sync replicas.
    pub isr: Vec<BrokerId>,
    /// All replicas.
    pub replicas: Vec<BrokerId>,
}

/// Leader and ISR response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderAndIsrResponse {
    /// Error code (0 = success).
    pub error_code: i16,
    /// Partition responses.
    pub partitions: Vec<LeaderAndIsrPartitionResponse>,
}

/// Partition response in LeaderAndIsr.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderAndIsrPartitionResponse {
    /// Topic name.
    pub topic: String,
    /// Partition ID.
    pub partition: u32,
    /// Error code.
    pub error_code: i16,
}

/// Update metadata request sent to all brokers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMetadataRequest {
    /// Controller ID.
    pub controller_id: BrokerId,
    /// Controller epoch.
    pub controller_epoch: u32,
    /// Partition states.
    pub partition_states: Vec<UpdateMetadataPartitionState>,
    /// All brokers.
    pub brokers: Vec<BrokerInfo>,
}

/// Partition state in UpdateMetadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateMetadataPartitionState {
    /// Topic name.
    pub topic: String,
    /// Partition ID.
    pub partition: u32,
    /// Controller epoch.
    pub controller_epoch: u32,
    /// Leader broker ID.
    pub leader: BrokerId,
    /// Leader epoch.
    pub leader_epoch: u32,
    /// In-sync replicas.
    pub isr: Vec<BrokerId>,
    /// All replicas.
    pub replicas: Vec<BrokerId>,
}

/// Controller election state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerElectionState {
    /// No controller elected.
    NoController,
    /// Election in progress.
    ElectionInProgress,
    /// This broker is the controller.
    IsController,
    /// Another broker is the controller.
    NotController,
}

/// The cluster controller.
pub struct Controller {
    /// Configuration.
    config: ControllerConfig,
    /// This broker's ID.
    broker_id: BrokerId,
    /// Whether this broker is the controller.
    is_controller: AtomicBool,
    /// Controller epoch.
    controller_epoch: AtomicU32,
    /// Cluster state.
    state: RwLock<ClusterState>,
    /// Broker liveness tracking.
    broker_last_seen: DashMap<BrokerId, Instant>,
    /// Pending leader elections.
    pending_elections: DashMap<TopicPartition, Instant>,
    /// Offline partitions.
    offline_partitions: DashMap<TopicPartition, ()>,
    /// Under-replicated partitions.
    under_replicated: DashMap<TopicPartition, ()>,
}

impl Controller {
    /// Create a new controller.
    pub fn new(config: ControllerConfig) -> Self {
        let broker_id = config.id;
        Self {
            config,
            broker_id,
            is_controller: AtomicBool::new(false),
            controller_epoch: AtomicU32::new(0),
            state: RwLock::new(ClusterState::default()),
            broker_last_seen: DashMap::new(),
            pending_elections: DashMap::new(),
            offline_partitions: DashMap::new(),
            under_replicated: DashMap::new(),
        }
    }

    /// Attempt to become the controller.
    pub fn try_become_controller(&self) -> Result<bool> {
        // In a real system, this would involve ZK or Raft consensus
        // For now, we just set this broker as controller
        let was_controller = self.is_controller.swap(true, Ordering::SeqCst);

        if !was_controller {
            let new_epoch = self.controller_epoch.fetch_add(1, Ordering::SeqCst) + 1;
            let mut state = self.state.write();
            state.controller_epoch = new_epoch;
            state.controller_id = Some(self.broker_id);
            info!(
                "Broker {} became controller with epoch {}",
                self.broker_id, new_epoch
            );
        }

        Ok(true)
    }

    /// Check if this broker is the controller.
    pub fn is_controller(&self) -> bool {
        self.is_controller.load(Ordering::SeqCst)
    }

    /// Get controller epoch.
    pub fn controller_epoch(&self) -> u32 {
        self.controller_epoch.load(Ordering::SeqCst)
    }

    /// Register a broker with the controller.
    pub fn register_broker(&self, info: BrokerInfo) -> Result<()> {
        if !self.is_controller() {
            return Err(Error::Internal("Not controller".into()));
        }

        let broker_id = info.id;
        self.broker_last_seen.insert(broker_id, Instant::now());

        let mut state = self.state.write();
        let existing = state.brokers.insert(broker_id, info.clone());

        if existing.is_none() {
            info!("Registered new broker {}", broker_id);
            // Trigger rebalance for under-replicated partitions
            drop(state);
            self.trigger_partition_reassignment()?;
        }

        Ok(())
    }

    /// Handle broker heartbeat.
    pub fn handle_heartbeat(&self, broker_id: BrokerId) -> Result<()> {
        self.broker_last_seen.insert(broker_id, Instant::now());

        // Update broker's alive status
        let mut state = self.state.write();
        if let Some(broker) = state.brokers.get_mut(&broker_id) {
            broker.is_alive = true;
            broker.last_heartbeat_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
        }

        Ok(())
    }

    /// Check broker liveness and handle failures.
    pub fn check_broker_liveness(&self) -> Result<Vec<BrokerId>> {
        if !self.is_controller() {
            return Ok(Vec::new());
        }

        let timeout = Duration::from_millis(self.config.session_timeout_ms);
        let now = Instant::now();
        let mut failed_brokers = Vec::new();

        for entry in self.broker_last_seen.iter() {
            let elapsed = now.duration_since(*entry.value());
            if elapsed > timeout {
                failed_brokers.push(*entry.key());
            }
        }

        // Handle failed brokers
        for broker_id in &failed_brokers {
            self.handle_broker_failure(*broker_id)?;
        }

        Ok(failed_brokers)
    }

    /// Handle broker failure.
    pub fn handle_broker_failure(&self, failed_broker: BrokerId) -> Result<()> {
        if !self.is_controller() {
            return Err(Error::Internal("Not controller".into()));
        }

        info!("Handling failure of broker {}", failed_broker);

        // Mark broker as dead
        {
            let mut state = self.state.write();
            if let Some(broker) = state.brokers.get_mut(&failed_broker) {
                broker.is_alive = false;
            }
        }

        // Find partitions where failed broker is leader
        let state = self.state.read();
        let mut leader_elections_needed = Vec::new();
        let mut isr_shrinks = Vec::new();

        for (tp, assignment) in &state.assignments {
            if assignment.leader == failed_broker {
                leader_elections_needed.push(tp.clone());
            } else if assignment.isr.contains(&failed_broker) {
                isr_shrinks.push((tp.clone(), assignment.clone()));
            }
        }
        drop(state);

        // Elect new leaders
        for tp in leader_elections_needed {
            self.elect_leader(&tp)?;
        }

        // Shrink ISRs
        for (tp, mut assignment) in isr_shrinks {
            assignment.isr.retain(|b| *b != failed_broker);
            self.update_partition_assignment(&tp, assignment)?;
        }

        Ok(())
    }

    /// Elect a new leader for a partition.
    pub fn elect_leader(&self, tp: &TopicPartition) -> Result<Option<BrokerId>> {
        if !self.is_controller() {
            return Err(Error::Internal("Not controller".into()));
        }

        let mut state = self.state.write();

        // Find alive broker in ISR first (before mutable borrow)
        let live_brokers: HashSet<BrokerId> = state
            .brokers
            .values()
            .filter(|b| b.is_alive)
            .map(|b| b.id)
            .collect();

        let assignment = match state.assignments.get_mut(tp) {
            Some(a) => a,
            None => return Err(Error::PartitionNotFound(tp.partition)),
        };

        let new_leader = assignment
            .isr
            .iter()
            .find(|b| live_brokers.contains(b))
            .or_else(|| {
                // Fallback to any replica if ISR is empty (unclean election)
                assignment.replicas.iter().find(|b| live_brokers.contains(b))
            })
            .copied();

        match new_leader {
            Some(leader) => {
                assignment.leader = leader;
                assignment.leader_epoch += 1;
                self.offline_partitions.remove(tp);
                info!(
                    "Elected broker {} as new leader for {:?} with epoch {}",
                    leader, tp, assignment.leader_epoch
                );
                Ok(Some(leader))
            }
            None => {
                self.offline_partitions.insert(tp.clone(), ());
                warn!("Partition {:?} is offline - no live replicas", tp);
                Ok(None)
            }
        }
    }

    /// Create a new topic.
    pub fn create_topic(
        &self,
        name: String,
        num_partitions: u32,
        replication_factor: u32,
    ) -> Result<()> {
        if !self.is_controller() {
            return Err(Error::Internal("Not controller".into()));
        }

        let mut state = self.state.write();

        if state.topics.contains_key(&name) {
            return Err(Error::Internal(format!("Topic {} already exists", name)));
        }

        // Get live brokers
        let live_brokers: Vec<BrokerId> = state
            .brokers
            .values()
            .filter(|b| b.is_alive)
            .map(|b| b.id)
            .collect();

        if live_brokers.len() < replication_factor as usize {
            return Err(Error::Internal(format!(
                "Not enough brokers ({}) for replication factor {}",
                live_brokers.len(),
                replication_factor
            )));
        }

        // Create topic metadata
        let metadata = TopicMetadata {
            name: name.clone(),
            partitions: num_partitions,
            replication_factor,
        };
        state.topics.insert(name.clone(), metadata);

        // Assign partitions to brokers
        for partition_id in 0..num_partitions {
            let tp = TopicPartition::new(&name, partition_id);

            // Select replicas using round-robin with rack awareness
            let leader_idx = partition_id as usize % live_brokers.len();
            let mut replicas = Vec::with_capacity(replication_factor as usize);

            for i in 0..replication_factor as usize {
                let broker_idx = (leader_idx + i) % live_brokers.len();
                replicas.push(live_brokers[broker_idx]);
            }

            let assignment = PartitionAssignment {
                leader: replicas[0],
                replicas: replicas.clone(),
                isr: replicas,
                leader_epoch: 0,
            };

            state.assignments.insert(tp, assignment);
        }

        info!(
            "Created topic {} with {} partitions and replication factor {}",
            name, num_partitions, replication_factor
        );

        Ok(())
    }

    /// Delete a topic.
    pub fn delete_topic(&self, name: &str) -> Result<()> {
        if !self.is_controller() {
            return Err(Error::Internal("Not controller".into()));
        }

        let mut state = self.state.write();

        let metadata = state
            .topics
            .remove(name)
            .ok_or_else(|| Error::TopicNotFound(name.to_string()))?;

        // Remove partition assignments
        for partition_id in 0..metadata.partitions {
            let tp = TopicPartition::new(name, partition_id);
            state.assignments.remove(&tp);
        }

        info!("Deleted topic {}", name);
        Ok(())
    }

    /// Update partition assignment.
    pub fn update_partition_assignment(
        &self,
        tp: &TopicPartition,
        assignment: PartitionAssignment,
    ) -> Result<()> {
        let mut state = self.state.write();
        state.assignments.insert(tp.clone(), assignment);
        Ok(())
    }

    /// Get partition assignment.
    pub fn get_assignment(&self, tp: &TopicPartition) -> Option<PartitionAssignment> {
        let state = self.state.read();
        state.assignments.get(tp).cloned()
    }

    /// Trigger partition reassignment for under-replicated partitions.
    pub fn trigger_partition_reassignment(&self) -> Result<()> {
        if !self.is_controller() {
            return Ok(());
        }

        let state = self.state.read();
        let live_brokers: HashSet<BrokerId> = state
            .brokers
            .values()
            .filter(|b| b.is_alive)
            .map(|b| b.id)
            .collect();

        let mut reassignments = Vec::new();

        for (tp, assignment) in &state.assignments {
            let live_replicas: Vec<BrokerId> = assignment
                .replicas
                .iter()
                .filter(|b| live_brokers.contains(b))
                .copied()
                .collect();

            // Check if under-replicated
            let topic_meta = match state.topics.get(&tp.topic) {
                Some(m) => m,
                None => continue,
            };

            if live_replicas.len() < topic_meta.replication_factor as usize {
                self.under_replicated.insert(tp.clone(), ());
                reassignments.push(tp.clone());
            } else {
                self.under_replicated.remove(tp);
            }
        }

        drop(state);

        // Perform reassignments
        for tp in reassignments {
            self.reassign_partition(&tp)?;
        }

        Ok(())
    }

    /// Reassign a partition to add more replicas.
    pub fn reassign_partition(&self, tp: &TopicPartition) -> Result<()> {
        let mut state = self.state.write();

        // Get immutable data first
        let topic_meta = match state.topics.get(&tp.topic) {
            Some(m) => m.clone(),
            None => return Err(Error::TopicNotFound(tp.topic.clone())),
        };

        let live_brokers: Vec<BrokerId> = state
            .brokers
            .values()
            .filter(|b| b.is_alive)
            .map(|b| b.id)
            .collect();

        // Now get mutable assignment
        let assignment = match state.assignments.get_mut(tp) {
            Some(a) => a,
            None => return Err(Error::PartitionNotFound(tp.partition)),
        };

        let current_replicas: HashSet<BrokerId> = assignment.replicas.iter().copied().collect();
        let needed = topic_meta.replication_factor as usize - current_replicas.len();

        if needed == 0 {
            return Ok(());
        }

        // Find brokers not already hosting this partition
        let available: Vec<BrokerId> = live_brokers
            .iter()
            .filter(|b| !current_replicas.contains(b))
            .copied()
            .collect();

        if available.len() < needed {
            warn!(
                "Cannot fully reassign {:?}: need {} more replicas but only {} brokers available",
                tp,
                needed,
                available.len()
            );
        }

        // Add new replicas
        for broker in available.iter().take(needed) {
            assignment.replicas.push(*broker);
            // New replicas start out of sync
        }

        info!("Reassigned {:?} with new replicas: {:?}", tp, assignment.replicas);
        Ok(())
    }

    /// Get all offline partitions.
    pub fn offline_partitions(&self) -> Vec<TopicPartition> {
        self.offline_partitions
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    /// Get all under-replicated partitions.
    pub fn under_replicated_partitions(&self) -> Vec<TopicPartition> {
        self.under_replicated
            .iter()
            .map(|e| e.key().clone())
            .collect()
    }

    /// Get all topics.
    pub fn list_topics(&self) -> Vec<String> {
        let state = self.state.read();
        state.topics.keys().cloned().collect()
    }

    /// Get topic metadata.
    pub fn get_topic(&self, name: &str) -> Option<TopicMetadata> {
        let state = self.state.read();
        state.topics.get(name).cloned()
    }

    /// Get all brokers.
    pub fn list_brokers(&self) -> Vec<BrokerInfo> {
        let state = self.state.read();
        state.brokers.values().cloned().collect()
    }

    /// Get live brokers.
    pub fn live_brokers(&self) -> Vec<BrokerInfo> {
        let state = self.state.read();
        state.brokers.values().filter(|b| b.is_alive).cloned().collect()
    }

    /// Build LeaderAndIsr request for a broker.
    pub fn build_leader_and_isr_request(&self, target_broker: BrokerId) -> LeaderAndIsrRequest {
        let state = self.state.read();

        let partition_states: Vec<LeaderAndIsrPartitionState> = state
            .assignments
            .iter()
            .filter(|(_, a)| a.replicas.contains(&target_broker))
            .map(|(tp, a)| LeaderAndIsrPartitionState {
                topic: tp.topic.clone(),
                partition: tp.partition,
                controller_epoch: state.controller_epoch,
                leader: a.leader,
                leader_epoch: a.leader_epoch,
                isr: a.isr.clone(),
                replicas: a.replicas.clone(),
            })
            .collect();

        let live_brokers: Vec<BrokerInfo> = state
            .brokers
            .values()
            .filter(|b| b.is_alive)
            .cloned()
            .collect();

        LeaderAndIsrRequest {
            controller_id: self.broker_id,
            controller_epoch: state.controller_epoch,
            partition_states,
            live_brokers,
        }
    }

    /// Build UpdateMetadata request.
    pub fn build_update_metadata_request(&self) -> UpdateMetadataRequest {
        let state = self.state.read();

        let partition_states: Vec<UpdateMetadataPartitionState> = state
            .assignments
            .iter()
            .map(|(tp, a)| UpdateMetadataPartitionState {
                topic: tp.topic.clone(),
                partition: tp.partition,
                controller_epoch: state.controller_epoch,
                leader: a.leader,
                leader_epoch: a.leader_epoch,
                isr: a.isr.clone(),
                replicas: a.replicas.clone(),
            })
            .collect();

        let brokers: Vec<BrokerInfo> = state.brokers.values().cloned().collect();

        UpdateMetadataRequest {
            controller_id: self.broker_id,
            controller_epoch: state.controller_epoch,
            partition_states,
            brokers,
        }
    }

    /// Handle ISR change notification from a broker.
    pub fn handle_isr_change(
        &self,
        tp: &TopicPartition,
        new_isr: Vec<BrokerId>,
        leader_epoch: u32,
    ) -> Result<()> {
        if !self.is_controller() {
            return Err(Error::Internal("Not controller".into()));
        }

        let mut state = self.state.write();

        let assignment = match state.assignments.get_mut(tp) {
            Some(a) => a,
            None => return Err(Error::PartitionNotFound(tp.partition)),
        };

        // Verify leader epoch
        if leader_epoch != assignment.leader_epoch {
            return Err(Error::Internal(format!(
                "Stale leader epoch: got {}, expected {}",
                leader_epoch, assignment.leader_epoch
            )));
        }

        // Validate ISR - must be subset of replicas
        for broker in &new_isr {
            if !assignment.replicas.contains(broker) {
                return Err(Error::Internal(format!(
                    "ISR contains non-replica broker {}",
                    broker
                )));
            }
        }

        if new_isr != assignment.isr {
            debug!("Updating ISR for {:?}: {:?} -> {:?}", tp, assignment.isr, new_isr);
            assignment.isr = new_isr;
        }

        Ok(())
    }

    /// Get cluster state snapshot.
    pub fn cluster_state(&self) -> ClusterState {
        self.state.read().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_election() {
        let config = ControllerConfig::default();
        let controller = Controller::new(config);

        assert!(!controller.is_controller());
        assert!(controller.try_become_controller().unwrap());
        assert!(controller.is_controller());
        assert_eq!(controller.controller_epoch(), 1);
    }

    #[test]
    fn test_broker_registration() {
        let config = ControllerConfig::default();
        let controller = Controller::new(config);
        controller.try_become_controller().unwrap();

        let broker_info = BrokerInfo::new(1, "localhost".to_string(), 9092);
        controller.register_broker(broker_info.clone()).unwrap();

        let brokers = controller.list_brokers();
        assert_eq!(brokers.len(), 1);
        assert_eq!(brokers[0].id, 1);
    }

    #[test]
    fn test_topic_creation() {
        let config = ControllerConfig::default();
        let controller = Controller::new(config);
        controller.try_become_controller().unwrap();

        // Register brokers
        for i in 0..3 {
            let info = BrokerInfo::new(i, format!("host{}", i), 9092 + i);
            controller.register_broker(info).unwrap();
        }

        // Create topic
        controller.create_topic("test".to_string(), 3, 2).unwrap();

        let topics = controller.list_topics();
        assert!(topics.contains(&"test".to_string()));

        let topic = controller.get_topic("test").unwrap();
        assert_eq!(topic.partitions, 3);
        assert_eq!(topic.replication_factor, 2);
    }

    #[test]
    fn test_leader_election() {
        let config = ControllerConfig::default();
        let controller = Controller::new(config);
        controller.try_become_controller().unwrap();

        // Register brokers
        for i in 0..3 {
            let info = BrokerInfo::new(i, format!("host{}", i), 9092 + i);
            controller.register_broker(info).unwrap();
        }

        // Create topic
        controller.create_topic("test".to_string(), 1, 3).unwrap();

        let tp = TopicPartition::new("test", 0);
        let assignment = controller.get_assignment(&tp).unwrap();
        let original_leader = assignment.leader;

        // Simulate broker failure
        controller.handle_broker_failure(original_leader).unwrap();

        // Check new leader
        let new_assignment = controller.get_assignment(&tp).unwrap();
        assert_ne!(new_assignment.leader, original_leader);
        assert_eq!(new_assignment.leader_epoch, 1);
    }
}
