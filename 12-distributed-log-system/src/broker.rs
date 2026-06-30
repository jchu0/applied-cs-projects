//! Broker: partition management, replication, and coordination.

use crate::log::{Partition, RecordBatch, SegmentConfig, TopicPartition};
use crate::protocol::{FetchRequest, FetchResponse, ProduceRequest, ProduceResponse};
use crate::{Error, Offset, Result};

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Broker identifier.
pub type BrokerId = u32;

/// Broker configuration.
#[derive(Debug, Clone)]
pub struct BrokerConfig {
    /// Broker ID.
    pub id: BrokerId,
    /// Listen address.
    pub listen_addr: String,
    /// Data directory.
    pub data_dir: PathBuf,
    /// Default replication factor.
    pub default_replication_factor: u32,
    /// Default number of partitions.
    pub default_num_partitions: u32,
    /// Segment configuration.
    pub segment_config: SegmentConfig,
    /// ISR lag time threshold.
    pub isr_lag_time_ms: u64,
    /// ISR lag offset threshold.
    pub isr_lag_offset: u64,
}

impl Default for BrokerConfig {
    fn default() -> Self {
        Self {
            id: 0,
            listen_addr: "127.0.0.1:9092".to_string(),
            data_dir: PathBuf::from("/tmp/kafka-lite"),
            default_replication_factor: 3,
            default_num_partitions: 3,
            segment_config: SegmentConfig::default(),
            isr_lag_time_ms: 10000,
            isr_lag_offset: 1000,
        }
    }
}

/// Topic metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicMetadata {
    pub name: String,
    pub partitions: u32,
    pub replication_factor: u32,
}

/// Partition assignment information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionAssignment {
    pub leader: BrokerId,
    pub replicas: Vec<BrokerId>,
    pub isr: Vec<BrokerId>,
    pub leader_epoch: u32,
}

/// Replica state on a follower.
#[derive(Debug, Clone)]
pub struct ReplicaState {
    pub fetch_offset: Offset,
    pub last_fetch_time: Instant,
}

/// A broker in the cluster.
pub struct Broker {
    /// Broker ID.
    pub id: BrokerId,
    /// Configuration.
    config: BrokerConfig,
    /// Partitions this broker manages.
    partitions: DashMap<TopicPartition, RwLock<Partition>>,
    /// Topic metadata.
    topics: DashMap<String, TopicMetadata>,
    /// Partition assignments.
    assignments: DashMap<TopicPartition, PartitionAssignment>,
    /// Replica manager state.
    replica_states: DashMap<(TopicPartition, BrokerId), ReplicaState>,
    /// Is this broker the controller?
    is_controller: bool,
    /// Metrics.
    pub metrics: BrokerMetrics,
}

impl Broker {
    /// Create a new broker.
    pub fn new(config: BrokerConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.data_dir)?;

        Ok(Self {
            id: config.id,
            config,
            partitions: DashMap::new(),
            topics: DashMap::new(),
            assignments: DashMap::new(),
            replica_states: DashMap::new(),
            is_controller: false,
            metrics: BrokerMetrics::new(),
        })
    }

    /// Create a new topic.
    pub fn create_topic(
        &self,
        name: String,
        partitions: u32,
        replication_factor: u32,
    ) -> Result<()> {
        if self.topics.contains_key(&name) {
            return Err(Error::Internal(format!("Topic {} already exists", name)));
        }

        // Store metadata
        let metadata = TopicMetadata {
            name: name.clone(),
            partitions,
            replication_factor,
        };
        self.topics.insert(name.clone(), metadata);

        // Create partitions
        for partition_id in 0..partitions {
            let tp = TopicPartition::new(&name, partition_id);

            let partition = Partition::new(
                &self.config.data_dir,
                &name,
                partition_id,
                self.config.segment_config.clone(),
            )?;

            self.partitions.insert(tp.clone(), RwLock::new(partition));

            // For simplicity, set this broker as leader
            let assignment = PartitionAssignment {
                leader: self.id,
                replicas: vec![self.id],
                isr: vec![self.id],
                leader_epoch: 0,
            };
            self.assignments.insert(tp, assignment);
        }

        info!("Created topic {} with {} partitions", name, partitions);
        Ok(())
    }

    /// Handle produce request.
    pub fn handle_produce(&self, request: ProduceRequest) -> Result<ProduceResponse> {
        let mut partition_responses = Vec::new();

        for topic_data in request.topic_data {
            for partition_data in topic_data.partition_data {
                let tp = TopicPartition::new(&topic_data.topic, partition_data.partition);

                // Check if we're the leader
                let assignment = self.assignments.get(&tp).ok_or(Error::NotLeader)?;
                if assignment.leader != self.id {
                    return Err(Error::NotLeader);
                }

                // Append to partition
                let partition = self.partitions.get(&tp).ok_or(Error::NotLeader)?;
                let mut partition = partition.write();

                let batch_length = partition_data.records.batch_length;
                let offset = partition.append(partition_data.records)?;

                // Update high watermark for single broker
                let log_end = partition.log_end_offset;
                partition.update_high_watermark(log_end);

                self.metrics.messages_in.fetch_add(1, Ordering::Relaxed);
                self.metrics
                    .bytes_in
                    .fetch_add(batch_length as u64, Ordering::Relaxed);

                partition_responses.push(crate::protocol::PartitionProduceResponse {
                    partition: partition_data.partition,
                    error_code: 0,
                    base_offset: offset,
                    log_append_time: 0,
                });
            }
        }

        Ok(ProduceResponse {
            responses: partition_responses,
            throttle_time_ms: 0,
        })
    }

    /// Handle fetch request.
    pub fn handle_fetch(&self, request: FetchRequest) -> Result<FetchResponse> {
        let mut topic_responses = Vec::new();

        for partition_req in request.partitions {
            let tp = TopicPartition::new(&partition_req.topic, partition_req.partition);

            let partition = self
                .partitions
                .get(&tp)
                .ok_or(Error::PartitionNotFound(partition_req.partition))?;
            let mut partition = partition.write();

            // Read from partition
            let batches = partition.read(partition_req.fetch_offset, partition_req.max_bytes)?;

            self.metrics.messages_out.fetch_add(
                batches.iter().map(|b| b.records.len()).sum::<usize>() as u64,
                Ordering::Relaxed,
            );

            topic_responses.push(crate::protocol::FetchPartitionResponse {
                partition: partition_req.partition,
                error_code: 0,
                high_watermark: partition.high_watermark,
                last_stable_offset: partition.high_watermark,
                log_start_offset: 0,
                record_batches: batches,
            });

            // Update replica state if this is a follower fetch
            if request.replica_id >= 0 {
                let key = (tp.clone(), request.replica_id as BrokerId);
                self.replica_states.insert(
                    key,
                    ReplicaState {
                        fetch_offset: partition_req.fetch_offset,
                        last_fetch_time: Instant::now(),
                    },
                );
            }
        }

        Ok(FetchResponse {
            throttle_time_ms: 0,
            error_code: 0,
            session_id: 0,
            responses: topic_responses,
        })
    }

    /// Check and update ISR for a partition.
    pub fn check_isr(&self, tp: &TopicPartition) {
        let assignment = match self.assignments.get(tp) {
            Some(a) => a.clone(),
            None => return,
        };

        if assignment.leader != self.id {
            return; // Only leader manages ISR
        }

        let partition = match self.partitions.get(tp) {
            Some(p) => p,
            None => return,
        };
        let partition = partition.read();

        let mut new_isr = vec![self.id]; // Leader always in ISR
        let now = Instant::now();

        for &replica_id in &assignment.replicas {
            if replica_id == self.id {
                continue;
            }

            let key = (tp.clone(), replica_id);
            if let Some(state) = self.replica_states.get(&key) {
                let lag_time = now.duration_since(state.last_fetch_time);
                let lag_offset = partition.log_end_offset.saturating_sub(state.fetch_offset);

                if lag_time.as_millis() < self.config.isr_lag_time_ms as u128
                    && lag_offset < self.config.isr_lag_offset
                {
                    new_isr.push(replica_id);
                }
            }
        }

        // Update ISR if changed
        if new_isr != assignment.isr {
            let mut updated = assignment.clone();
            updated.isr = new_isr;
            self.assignments.insert(tp.clone(), updated);
        }
    }

    /// Get leader for a partition.
    pub fn leader_for(&self, tp: &TopicPartition) -> Option<BrokerId> {
        self.assignments.get(tp).map(|a| a.leader)
    }

    /// Check if this broker is leader for partition.
    pub fn is_leader_for(&self, tp: &TopicPartition) -> bool {
        self.assignments
            .get(tp)
            .map(|a| a.leader == self.id)
            .unwrap_or(false)
    }

    /// Get partition metadata.
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
        })
    }

    /// List all topics.
    pub fn list_topics(&self) -> Vec<String> {
        self.topics.iter().map(|e| e.key().clone()).collect()
    }

    /// Get topic metadata.
    pub fn get_topic(&self, name: &str) -> Option<TopicMetadata> {
        self.topics.get(name).map(|e| e.clone())
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
}

/// Broker metrics.
pub struct BrokerMetrics {
    /// Bytes received.
    pub bytes_in: AtomicU64,
    /// Bytes sent.
    pub bytes_out: AtomicU64,
    /// Messages received.
    pub messages_in: AtomicU64,
    /// Messages sent.
    pub messages_out: AtomicU64,
    /// Under-replicated partitions.
    pub under_replicated_partitions: AtomicU64,
    /// Active connections.
    pub active_connections: AtomicU64,
}

impl BrokerMetrics {
    pub fn new() -> Self {
        Self {
            bytes_in: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
            messages_in: AtomicU64::new(0),
            messages_out: AtomicU64::new(0),
            under_replicated_partitions: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
        }
    }

    /// Get metrics snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            bytes_in: self.bytes_in.load(Ordering::Relaxed),
            bytes_out: self.bytes_out.load(Ordering::Relaxed),
            messages_in: self.messages_in.load(Ordering::Relaxed),
            messages_out: self.messages_out.load(Ordering::Relaxed),
            under_replicated_partitions: self.under_replicated_partitions.load(Ordering::Relaxed),
            active_connections: self.active_connections.load(Ordering::Relaxed),
        }
    }
}

impl Default for BrokerMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics snapshot.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub bytes_in: u64,
    pub bytes_out: u64,
    pub messages_in: u64,
    pub messages_out: u64,
    pub under_replicated_partitions: u64,
    pub active_connections: u64,
}

/// Group coordinator for consumer groups.
pub struct GroupCoordinator {
    /// Consumer groups.
    groups: DashMap<String, ConsumerGroupState>,
}

/// Consumer group state.
#[derive(Debug, Clone)]
pub struct ConsumerGroupState {
    pub group_id: String,
    pub state: GroupState,
    pub generation_id: u32,
    pub leader: Option<String>,
    pub members: HashMap<String, MemberMetadata>,
    pub offsets: HashMap<TopicPartition, OffsetAndMetadata>,
}

/// Group state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupState {
    Empty,
    PreparingRebalance,
    CompletingRebalance,
    Stable,
    Dead,
}

/// Member metadata.
#[derive(Debug, Clone)]
pub struct MemberMetadata {
    pub member_id: String,
    pub client_id: String,
    pub client_host: String,
    pub session_timeout_ms: u32,
    pub subscriptions: Vec<String>,
}

/// Offset with metadata.
#[derive(Debug, Clone)]
pub struct OffsetAndMetadata {
    pub offset: Offset,
    pub metadata: String,
    pub commit_timestamp: i64,
}

impl GroupCoordinator {
    /// Create a new group coordinator.
    pub fn new() -> Self {
        Self {
            groups: DashMap::new(),
        }
    }

    /// Get or create a consumer group.
    pub fn get_or_create_group(&self, group_id: &str) -> ConsumerGroupState {
        self.groups
            .entry(group_id.to_string())
            .or_insert_with(|| ConsumerGroupState {
                group_id: group_id.to_string(),
                state: GroupState::Empty,
                generation_id: 0,
                leader: None,
                members: HashMap::new(),
                offsets: HashMap::new(),
            })
            .clone()
    }

    /// Commit offsets for a group.
    pub fn commit_offsets(
        &self,
        group_id: &str,
        offsets: Vec<(TopicPartition, OffsetAndMetadata)>,
    ) -> Result<()> {
        if let Some(mut group) = self.groups.get_mut(group_id) {
            for (tp, offset) in offsets {
                group.offsets.insert(tp, offset);
            }
        }
        Ok(())
    }

    /// Fetch committed offsets.
    pub fn fetch_offsets(
        &self,
        group_id: &str,
        partitions: &[TopicPartition],
    ) -> HashMap<TopicPartition, Offset> {
        let mut result = HashMap::new();

        if let Some(group) = self.groups.get(group_id) {
            for tp in partitions {
                if let Some(offset) = group.offsets.get(tp) {
                    result.insert(tp.clone(), offset.offset);
                }
            }
        }

        result
    }
}

impl Default for GroupCoordinator {
    fn default() -> Self {
        Self::new()
    }
}
