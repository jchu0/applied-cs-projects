//! Consumer API for reading records from the log.

use crate::broker::BrokerId;
use crate::log::{Header, TopicPartition};
use crate::protocol::{FetchPartition, FetchRequest};
use crate::{Error, Offset, Result, Timestamp};

use std::collections::HashMap;
use std::time::Duration;

/// Consumer configuration.
#[derive(Debug, Clone)]
pub struct ConsumerConfig {
    /// Consumer group ID.
    pub group_id: String,
    /// Client ID.
    pub client_id: String,
    /// Auto-commit offsets.
    pub enable_auto_commit: bool,
    /// Auto-commit interval.
    pub auto_commit_interval_ms: u64,
    /// Session timeout.
    pub session_timeout_ms: u32,
    /// Heartbeat interval.
    pub heartbeat_interval_ms: u32,
    /// Maximum poll records.
    pub max_poll_records: u32,
    /// Fetch minimum bytes.
    pub fetch_min_bytes: u32,
    /// Fetch maximum bytes.
    pub fetch_max_bytes: u32,
    /// Fetch maximum wait.
    pub fetch_max_wait_ms: u32,
    /// Auto offset reset.
    pub auto_offset_reset: AutoOffsetReset,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            group_id: "".to_string(),
            client_id: "kafka-lite-consumer".to_string(),
            enable_auto_commit: true,
            auto_commit_interval_ms: 5000,
            session_timeout_ms: 30000,
            heartbeat_interval_ms: 3000,
            max_poll_records: 500,
            fetch_min_bytes: 1,
            fetch_max_bytes: 52428800,
            fetch_max_wait_ms: 500,
            auto_offset_reset: AutoOffsetReset::Latest,
        }
    }
}

/// Auto offset reset policy.
#[derive(Debug, Clone, Copy)]
pub enum AutoOffsetReset {
    /// Reset to earliest offset.
    Earliest,
    /// Reset to latest offset.
    Latest,
    /// Throw error if no offset found.
    None,
}

/// A record received by consumer.
#[derive(Debug, Clone)]
pub struct ConsumerRecord {
    /// Topic name.
    pub topic: String,
    /// Partition.
    pub partition: u32,
    /// Offset.
    pub offset: Offset,
    /// Timestamp.
    pub timestamp: Timestamp,
    /// Key.
    pub key: Option<Vec<u8>>,
    /// Value.
    pub value: Option<Vec<u8>>,
    /// Headers.
    pub headers: Vec<Header>,
}

/// Consumer group.
pub struct ConsumerGroup {
    /// Group ID.
    pub group_id: String,
    /// Generation ID.
    pub generation_id: u32,
    /// Member ID.
    pub member_id: String,
    /// Leader member ID.
    pub leader: String,
    /// Group members.
    pub members: Vec<String>,
}

/// Kafka-lite consumer.
pub struct Consumer {
    /// Configuration.
    config: ConsumerConfig,
    /// Subscribed topics.
    subscriptions: Vec<String>,
    /// Assigned partitions.
    assignment: Vec<TopicPartition>,
    /// Current positions.
    positions: HashMap<TopicPartition, Offset>,
    /// Committed offsets.
    committed: HashMap<TopicPartition, Offset>,
    /// Group coordinator.
    coordinator: Option<BrokerId>,
    /// Member ID.
    member_id: String,
    /// Generation ID.
    generation_id: u32,
}

impl Consumer {
    /// Create a new consumer.
    pub fn new(config: ConsumerConfig) -> Self {
        Self {
            config,
            subscriptions: Vec::new(),
            assignment: Vec::new(),
            positions: HashMap::new(),
            committed: HashMap::new(),
            coordinator: None,
            member_id: String::new(),
            generation_id: 0,
        }
    }

    /// Subscribe to topics.
    pub fn subscribe(&mut self, topics: Vec<String>) -> Result<()> {
        self.subscriptions = topics;
        Ok(())
    }

    /// Assign specific partitions.
    pub fn assign(&mut self, partitions: Vec<TopicPartition>) -> Result<()> {
        self.assignment = partitions;
        Ok(())
    }

    /// Build fetch request for assigned partitions.
    pub fn build_fetch_request(&self) -> FetchRequest {
        let partitions: Vec<FetchPartition> = self
            .assignment
            .iter()
            .map(|tp| {
                let offset = self.positions.get(tp).copied().unwrap_or(0);
                FetchPartition {
                    topic: tp.topic.clone(),
                    partition: tp.partition,
                    fetch_offset: offset,
                    max_bytes: self.config.fetch_max_bytes,
                }
            })
            .collect();

        FetchRequest {
            replica_id: -1, // Consumer
            max_wait_ms: self.config.fetch_max_wait_ms,
            min_bytes: self.config.fetch_min_bytes,
            max_bytes: self.config.fetch_max_bytes,
            isolation_level: 0,
            session_id: 0,
            session_epoch: 0,
            partitions,
        }
    }

    /// Update position after consuming records.
    pub fn update_position(&mut self, tp: &TopicPartition, offset: Offset) {
        self.positions.insert(tp.clone(), offset);
    }

    /// Commit current positions.
    pub fn commit(&mut self) -> Result<HashMap<TopicPartition, Offset>> {
        let to_commit = self.positions.clone();
        self.committed = to_commit.clone();
        Ok(to_commit)
    }

    /// Commit specific offsets.
    pub fn commit_sync(
        &mut self,
        offsets: HashMap<TopicPartition, Offset>,
    ) -> Result<()> {
        for (tp, offset) in offsets {
            self.committed.insert(tp, offset);
        }
        Ok(())
    }

    /// Get current position for partition.
    pub fn position(&self, tp: &TopicPartition) -> Option<Offset> {
        self.positions.get(tp).copied()
    }

    /// Get committed offset for partition.
    pub fn committed(&self, tp: &TopicPartition) -> Option<Offset> {
        self.committed.get(tp).copied()
    }

    /// Seek to specific offset.
    pub fn seek(&mut self, tp: TopicPartition, offset: Offset) {
        self.positions.insert(tp, offset);
    }

    /// Seek to beginning of partitions.
    pub fn seek_to_beginning(&mut self, partitions: &[TopicPartition]) {
        for tp in partitions {
            self.positions.insert(tp.clone(), 0);
        }
    }

    /// Seek to end of partitions.
    pub fn seek_to_end(&mut self, partitions: &[TopicPartition]) {
        for tp in partitions {
            // Would query broker for log end offset
            self.positions.insert(tp.clone(), u64::MAX);
        }
    }

    /// Get assignment.
    pub fn assignment(&self) -> &[TopicPartition] {
        &self.assignment
    }

    /// Get subscriptions.
    pub fn subscription(&self) -> &[String] {
        &self.subscriptions
    }

    /// Pause consumption from partitions.
    pub fn pause(&mut self, _partitions: &[TopicPartition]) {
        // Would mark partitions as paused
    }

    /// Resume consumption from partitions.
    pub fn resume(&mut self, _partitions: &[TopicPartition]) {
        // Would mark partitions as resumed
    }

    /// Get paused partitions.
    pub fn paused(&self) -> Vec<TopicPartition> {
        Vec::new()
    }

    /// Close the consumer.
    pub fn close(&mut self) {
        self.assignment.clear();
        self.subscriptions.clear();
    }
}

/// Partition assignor strategy.
pub trait PartitionAssignor {
    /// Assign partitions to members.
    fn assign(
        &self,
        members: &[String],
        partitions: &[TopicPartition],
    ) -> HashMap<String, Vec<TopicPartition>>;
}

/// Range partition assignor.
pub struct RangeAssignor;

impl PartitionAssignor for RangeAssignor {
    fn assign(
        &self,
        members: &[String],
        partitions: &[TopicPartition],
    ) -> HashMap<String, Vec<TopicPartition>> {
        let mut assignment: HashMap<String, Vec<TopicPartition>> = HashMap::new();

        // Group partitions by topic
        let mut by_topic: HashMap<String, Vec<TopicPartition>> = HashMap::new();
        for tp in partitions {
            by_topic.entry(tp.topic.clone()).or_default().push(tp.clone());
        }

        // Assign each topic's partitions
        for (_topic, mut topic_partitions) in by_topic {
            topic_partitions.sort_by_key(|tp| tp.partition);

            let n_members = members.len();
            let n_partitions = topic_partitions.len();
            let per_member = n_partitions / n_members;
            let extra = n_partitions % n_members;

            let mut idx = 0;
            for (i, member) in members.iter().enumerate() {
                let count = per_member + if i < extra { 1 } else { 0 };
                let member_partitions: Vec<_> = topic_partitions[idx..idx + count].to_vec();
                assignment
                    .entry(member.clone())
                    .or_default()
                    .extend(member_partitions);
                idx += count;
            }
        }

        assignment
    }
}

/// Round-robin partition assignor.
pub struct RoundRobinAssignor;

impl PartitionAssignor for RoundRobinAssignor {
    fn assign(
        &self,
        members: &[String],
        partitions: &[TopicPartition],
    ) -> HashMap<String, Vec<TopicPartition>> {
        let mut assignment: HashMap<String, Vec<TopicPartition>> = HashMap::new();

        for (i, tp) in partitions.iter().enumerate() {
            let member = &members[i % members.len()];
            assignment
                .entry(member.clone())
                .or_default()
                .push(tp.clone());
        }

        assignment
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_range_assignor() {
        let assignor = RangeAssignor;
        let members = vec!["m1".to_string(), "m2".to_string()];
        let partitions = vec![
            TopicPartition::new("t1", 0),
            TopicPartition::new("t1", 1),
            TopicPartition::new("t1", 2),
        ];

        let assignment = assignor.assign(&members, &partitions);

        assert_eq!(assignment.get("m1").unwrap().len(), 2);
        assert_eq!(assignment.get("m2").unwrap().len(), 1);
    }

    #[test]
    fn test_round_robin_assignor() {
        let assignor = RoundRobinAssignor;
        let members = vec!["m1".to_string(), "m2".to_string()];
        let partitions = vec![
            TopicPartition::new("t1", 0),
            TopicPartition::new("t1", 1),
            TopicPartition::new("t1", 2),
            TopicPartition::new("t1", 3),
        ];

        let assignment = assignor.assign(&members, &partitions);

        assert_eq!(assignment.get("m1").unwrap().len(), 2);
        assert_eq!(assignment.get("m2").unwrap().len(), 2);
    }
}
