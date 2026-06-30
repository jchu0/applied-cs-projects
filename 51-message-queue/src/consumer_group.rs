//! Consumer group management.

use crate::config::AssignmentStrategy;
use crate::error::{Error, Result};
use crate::offset::TopicPartition;
use crate::partition::PartitionId;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Consumer group state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupState {
    /// Group is empty (no members).
    Empty,
    /// Preparing for rebalance.
    PreparingRebalance,
    /// Completing rebalance.
    CompletingRebalance,
    /// Group is stable.
    Stable,
    /// Group is dead.
    Dead,
}

impl Default for GroupState {
    fn default() -> Self {
        GroupState::Empty
    }
}

impl std::fmt::Display for GroupState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GroupState::Empty => write!(f, "Empty"),
            GroupState::PreparingRebalance => write!(f, "PreparingRebalance"),
            GroupState::CompletingRebalance => write!(f, "CompletingRebalance"),
            GroupState::Stable => write!(f, "Stable"),
            GroupState::Dead => write!(f, "Dead"),
        }
    }
}

/// Consumer group member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMember {
    /// Member ID.
    pub id: String,
    /// Client ID.
    pub client_id: String,
    /// Client host.
    pub client_host: String,
    /// Session timeout.
    pub session_timeout: Duration,
    /// Rebalance timeout.
    pub rebalance_timeout: Duration,
    /// Last heartbeat.
    #[serde(skip)]
    pub last_heartbeat: Option<Instant>,
    /// Assigned partitions.
    pub assignment: Vec<TopicPartition>,
    /// Subscribed topics.
    pub subscriptions: Vec<String>,
    /// Member metadata.
    pub metadata: Vec<u8>,
}

impl GroupMember {
    /// Create a new group member.
    pub fn new(
        id: impl Into<String>,
        client_id: impl Into<String>,
        session_timeout: Duration,
    ) -> Self {
        Self {
            id: id.into(),
            client_id: client_id.into(),
            client_host: String::new(),
            session_timeout,
            rebalance_timeout: session_timeout * 2,
            last_heartbeat: Some(Instant::now()),
            assignment: Vec::new(),
            subscriptions: Vec::new(),
            metadata: Vec::new(),
        }
    }

    /// Update heartbeat.
    pub fn heartbeat(&mut self) {
        self.last_heartbeat = Some(Instant::now());
    }

    /// Check if session has expired.
    pub fn is_expired(&self) -> bool {
        self.last_heartbeat
            .map(|t| t.elapsed() > self.session_timeout)
            .unwrap_or(true)
    }

    /// Set subscriptions.
    pub fn set_subscriptions(&mut self, topics: Vec<String>) {
        self.subscriptions = topics;
    }

    /// Set assignment.
    pub fn set_assignment(&mut self, partitions: Vec<TopicPartition>) {
        self.assignment = partitions;
    }
}

/// Consumer group configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupConfig {
    /// Session timeout.
    pub session_timeout: Duration,
    /// Heartbeat interval.
    pub heartbeat_interval: Duration,
    /// Rebalance timeout.
    pub rebalance_timeout: Duration,
    /// Assignment strategy.
    pub assignment_strategy: AssignmentStrategy,
    /// Maximum poll interval.
    pub max_poll_interval: Duration,
}

impl Default for GroupConfig {
    fn default() -> Self {
        Self {
            session_timeout: Duration::from_secs(10),
            heartbeat_interval: Duration::from_secs(3),
            rebalance_timeout: Duration::from_secs(60),
            assignment_strategy: AssignmentStrategy::Range,
            max_poll_interval: Duration::from_secs(300),
        }
    }
}

/// Consumer group.
pub struct ConsumerGroup {
    /// Group ID.
    id: String,
    /// Configuration.
    config: GroupConfig,
    /// Current state.
    state: RwLock<GroupState>,
    /// Generation ID.
    generation: AtomicU32,
    /// Members.
    members: RwLock<HashMap<String, GroupMember>>,
    /// Leader ID.
    leader_id: RwLock<Option<String>>,
    /// Subscribed topics.
    subscribed_topics: RwLock<HashSet<String>>,
    /// Protocol type.
    protocol_type: String,
    /// Protocol name.
    protocol_name: RwLock<Option<String>>,
}

impl ConsumerGroup {
    /// Create a new consumer group.
    pub fn new(id: impl Into<String>, config: GroupConfig) -> Self {
        Self {
            id: id.into(),
            config,
            state: RwLock::new(GroupState::Empty),
            generation: AtomicU32::new(0),
            members: RwLock::new(HashMap::new()),
            leader_id: RwLock::new(None),
            subscribed_topics: RwLock::new(HashSet::new()),
            protocol_type: "consumer".to_string(),
            protocol_name: RwLock::new(None),
        }
    }

    /// Get group ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get current state.
    pub fn state(&self) -> GroupState {
        *self.state.read()
    }

    /// Get generation ID.
    pub fn generation(&self) -> u32 {
        self.generation.load(Ordering::Acquire)
    }

    /// Get leader ID.
    pub fn leader_id(&self) -> Option<String> {
        self.leader_id.read().clone()
    }

    /// Get member count.
    pub fn member_count(&self) -> usize {
        self.members.read().len()
    }

    /// Get all members.
    pub fn members(&self) -> Vec<GroupMember> {
        self.members.read().values().cloned().collect()
    }

    /// Get a specific member.
    pub fn get_member(&self, member_id: &str) -> Option<GroupMember> {
        self.members.read().get(member_id).cloned()
    }

    /// Join the group.
    pub fn join(
        &self,
        member_id: Option<String>,
        client_id: String,
        subscriptions: Vec<String>,
    ) -> Result<JoinGroupResponse> {
        let mut members = self.members.write();
        let mut state = self.state.write();

        // Generate member ID if not provided
        let member_id = member_id.unwrap_or_else(|| {
            format!("{}-{}", client_id, uuid::Uuid::new_v4())
        });

        // Create or update member
        let member = members.entry(member_id.clone()).or_insert_with(|| {
            GroupMember::new(&member_id, &client_id, self.config.session_timeout)
        });

        member.set_subscriptions(subscriptions.clone());
        member.heartbeat();

        // Update subscribed topics
        {
            let mut topics = self.subscribed_topics.write();
            for topic in &subscriptions {
                topics.insert(topic.clone());
            }
        }

        // Set leader if first member
        let is_leader = {
            let mut leader = self.leader_id.write();
            if leader.is_none() {
                *leader = Some(member_id.clone());
                true
            } else {
                leader.as_ref() == Some(&member_id)
            }
        };

        // Trigger rebalance if new member
        if *state == GroupState::Stable {
            *state = GroupState::PreparingRebalance;
        }

        Ok(JoinGroupResponse {
            member_id,
            generation_id: self.generation(),
            leader_id: self.leader_id().unwrap_or_default(),
            is_leader,
            members: if is_leader {
                members.values().cloned().collect()
            } else {
                Vec::new()
            },
        })
    }

    /// Sync group (receive assignment).
    pub fn sync(
        &self,
        member_id: &str,
        generation_id: u32,
        assignments: Option<HashMap<String, Vec<TopicPartition>>>,
    ) -> Result<SyncGroupResponse> {
        // Verify generation
        if generation_id != self.generation() {
            return Err(Error::RebalanceInProgress(self.id.clone()));
        }

        let mut members = self.members.write();

        // Verify member exists
        if !members.contains_key(member_id) {
            return Err(Error::ConsumerNotFound(member_id.to_string()));
        }

        // Apply assignments if leader
        if let Some(assignments) = assignments {
            for (mid, partitions) in assignments {
                if let Some(member) = members.get_mut(&mid) {
                    member.set_assignment(partitions);
                }
            }

            // Transition to stable
            *self.state.write() = GroupState::Stable;
        }

        // Get this member's assignment
        let assignment = members
            .get(member_id)
            .map(|m| m.assignment.clone())
            .unwrap_or_default();

        Ok(SyncGroupResponse {
            generation_id: self.generation(),
            assignment,
        })
    }

    /// Process heartbeat.
    pub fn heartbeat(&self, member_id: &str, generation_id: u32) -> Result<HeartbeatResponse> {
        // Verify generation
        if generation_id != self.generation() {
            return Err(Error::RebalanceInProgress(self.id.clone()));
        }

        let mut members = self.members.write();

        let member = members
            .get_mut(member_id)
            .ok_or_else(|| Error::ConsumerNotFound(member_id.to_string()))?;

        member.heartbeat();

        Ok(HeartbeatResponse {
            rebalance_needed: *self.state.read() != GroupState::Stable,
        })
    }

    /// Leave the group.
    pub fn leave(&self, member_id: &str) -> Result<()> {
        let mut members = self.members.write();

        if members.remove(member_id).is_none() {
            return Err(Error::ConsumerNotFound(member_id.to_string()));
        }

        // Update leader if needed
        {
            let mut leader = self.leader_id.write();
            if leader.as_ref() == Some(&member_id.to_string()) {
                *leader = members.keys().next().cloned();
            }
        }

        // Trigger rebalance
        if !members.is_empty() {
            *self.state.write() = GroupState::PreparingRebalance;
        } else {
            *self.state.write() = GroupState::Empty;
        }

        Ok(())
    }

    /// Perform assignment using configured strategy.
    pub fn assign(
        &self,
        topic_partitions: &HashMap<String, Vec<PartitionId>>,
    ) -> HashMap<String, Vec<TopicPartition>> {
        let members = self.members.read();
        let mut assignments: HashMap<String, Vec<TopicPartition>> = HashMap::new();

        // Initialize empty assignments for all members
        for member_id in members.keys() {
            assignments.insert(member_id.clone(), Vec::new());
        }

        // Get members sorted by ID for consistent assignment
        let mut member_ids: Vec<_> = members.keys().cloned().collect();
        member_ids.sort();

        if member_ids.is_empty() {
            return assignments;
        }

        match self.config.assignment_strategy {
            AssignmentStrategy::Range => {
                self.assign_range(&member_ids, topic_partitions, &mut assignments)
            }
            AssignmentStrategy::RoundRobin => {
                self.assign_round_robin(&member_ids, topic_partitions, &mut assignments)
            }
            _ => {
                self.assign_round_robin(&member_ids, topic_partitions, &mut assignments)
            }
        }

        assignments
    }

    /// Range assignment strategy.
    fn assign_range(
        &self,
        member_ids: &[String],
        topic_partitions: &HashMap<String, Vec<PartitionId>>,
        assignments: &mut HashMap<String, Vec<TopicPartition>>,
    ) {
        for (topic, partitions) in topic_partitions {
            let num_members = member_ids.len();
            let num_partitions = partitions.len();

            let partitions_per_member = num_partitions / num_members;
            let extra = num_partitions % num_members;

            let mut partition_idx = 0;

            for (member_idx, member_id) in member_ids.iter().enumerate() {
                let count = partitions_per_member + if member_idx < extra { 1 } else { 0 };

                let member_assignments = assignments.entry(member_id.clone()).or_default();

                for _ in 0..count {
                    if partition_idx < num_partitions {
                        member_assignments.push(TopicPartition::new(
                            topic.clone(),
                            partitions[partition_idx],
                        ));
                        partition_idx += 1;
                    }
                }
            }
        }
    }

    /// Round-robin assignment strategy.
    fn assign_round_robin(
        &self,
        member_ids: &[String],
        topic_partitions: &HashMap<String, Vec<PartitionId>>,
        assignments: &mut HashMap<String, Vec<TopicPartition>>,
    ) {
        // Collect all partitions
        let mut all_partitions: Vec<TopicPartition> = Vec::new();
        for (topic, partitions) in topic_partitions {
            for &partition in partitions {
                all_partitions.push(TopicPartition::new(topic.clone(), partition));
            }
        }

        // Sort for consistency
        all_partitions.sort_by(|a, b| {
            a.topic.cmp(&b.topic).then(a.partition.cmp(&b.partition))
        });

        // Round-robin assignment
        for (i, tp) in all_partitions.into_iter().enumerate() {
            let member_id = &member_ids[i % member_ids.len()];
            assignments.entry(member_id.clone()).or_default().push(tp);
        }
    }

    /// Increment generation.
    pub fn increment_generation(&self) -> u32 {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Check for expired members.
    pub fn check_expired_members(&self) -> Vec<String> {
        let members = self.members.read();
        members
            .iter()
            .filter(|(_, m)| m.is_expired())
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Remove expired members.
    pub fn remove_expired_members(&self) -> Vec<String> {
        let expired = self.check_expired_members();

        if !expired.is_empty() {
            let mut members = self.members.write();
            for id in &expired {
                members.remove(id);
            }

            if !members.is_empty() {
                *self.state.write() = GroupState::PreparingRebalance;
            } else {
                *self.state.write() = GroupState::Empty;
            }
        }

        expired
    }

    /// Get group description.
    pub fn describe(&self) -> GroupDescription {
        GroupDescription {
            group_id: self.id.clone(),
            state: self.state(),
            protocol_type: self.protocol_type.clone(),
            protocol: self.protocol_name.read().clone(),
            members: self.members(),
        }
    }
}

/// Join group response.
#[derive(Debug, Clone)]
pub struct JoinGroupResponse {
    /// Member ID.
    pub member_id: String,
    /// Generation ID.
    pub generation_id: u32,
    /// Leader ID.
    pub leader_id: String,
    /// Is this member the leader.
    pub is_leader: bool,
    /// Other members (only for leader).
    pub members: Vec<GroupMember>,
}

/// Sync group response.
#[derive(Debug, Clone)]
pub struct SyncGroupResponse {
    /// Generation ID.
    pub generation_id: u32,
    /// Assigned partitions.
    pub assignment: Vec<TopicPartition>,
}

/// Heartbeat response.
#[derive(Debug, Clone)]
pub struct HeartbeatResponse {
    /// Whether rebalance is needed.
    pub rebalance_needed: bool,
}

/// Group description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupDescription {
    /// Group ID.
    pub group_id: String,
    /// Current state.
    pub state: GroupState,
    /// Protocol type.
    pub protocol_type: String,
    /// Protocol name.
    pub protocol: Option<String>,
    /// Members.
    pub members: Vec<GroupMember>,
}

/// Consumer group manager.
pub struct GroupManager {
    /// Groups by ID.
    groups: RwLock<HashMap<String, std::sync::Arc<ConsumerGroup>>>,
    /// Default configuration.
    default_config: GroupConfig,
}

impl GroupManager {
    /// Create a new group manager.
    pub fn new(config: GroupConfig) -> Self {
        Self {
            groups: RwLock::new(HashMap::new()),
            default_config: config,
        }
    }

    /// Get or create a group.
    pub fn get_or_create(&self, group_id: &str) -> std::sync::Arc<ConsumerGroup> {
        {
            let groups = self.groups.read();
            if let Some(group) = groups.get(group_id) {
                return std::sync::Arc::clone(group);
            }
        }

        let mut groups = self.groups.write();
        let group = groups
            .entry(group_id.to_string())
            .or_insert_with(|| std::sync::Arc::new(ConsumerGroup::new(group_id, self.default_config.clone())));

        std::sync::Arc::clone(group)
    }

    /// List all groups.
    pub fn list_groups(&self) -> Vec<String> {
        self.groups.read().keys().cloned().collect()
    }

    /// Delete a group.
    pub fn delete_group(&self, group_id: &str) -> Result<()> {
        let mut groups = self.groups.write();

        if let Some(group) = groups.get(group_id) {
            if group.state() != GroupState::Empty && group.state() != GroupState::Dead {
                return Err(Error::IllegalState(
                    "Cannot delete non-empty group".to_string(),
                ));
            }
        }

        groups.remove(group_id);
        Ok(())
    }

    /// Describe all groups.
    pub fn describe_all(&self) -> Vec<GroupDescription> {
        self.groups.read().values().map(|g| g.describe()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_group() -> ConsumerGroup {
        ConsumerGroup::new("test-group", GroupConfig::default())
    }

    #[test]
    fn test_group_create() {
        let group = create_group();

        assert_eq!(group.id(), "test-group");
        assert_eq!(group.state(), GroupState::Empty);
        assert_eq!(group.generation(), 0);
        assert_eq!(group.member_count(), 0);
    }

    #[test]
    fn test_group_join() {
        let group = create_group();

        let response = group
            .join(None, "client-1".to_string(), vec!["topic1".to_string()])
            .unwrap();

        assert!(!response.member_id.is_empty());
        assert!(response.is_leader);
        assert_eq!(group.member_count(), 1);
    }

    #[test]
    fn test_group_multiple_members() {
        let group = create_group();

        let resp1 = group
            .join(None, "client-1".to_string(), vec!["topic1".to_string()])
            .unwrap();

        let resp2 = group
            .join(None, "client-2".to_string(), vec!["topic1".to_string()])
            .unwrap();

        assert!(resp1.is_leader);
        assert!(!resp2.is_leader);
        assert_eq!(group.member_count(), 2);
    }

    #[test]
    fn test_group_heartbeat() {
        let group = create_group();

        let join_resp = group
            .join(None, "client-1".to_string(), vec!["topic1".to_string()])
            .unwrap();

        let heartbeat_resp = group
            .heartbeat(&join_resp.member_id, join_resp.generation_id)
            .unwrap();

        // New member triggers rebalance
        assert!(heartbeat_resp.rebalance_needed);
    }

    #[test]
    fn test_group_leave() {
        let group = create_group();

        let join_resp = group
            .join(Some("member-1".to_string()), "client-1".to_string(), vec!["topic1".to_string()])
            .unwrap();

        group.leave(&join_resp.member_id).unwrap();

        assert_eq!(group.member_count(), 0);
        assert_eq!(group.state(), GroupState::Empty);
    }

    #[test]
    fn test_group_assignment_range() {
        let group = create_group();

        group
            .join(Some("member-1".to_string()), "client-1".to_string(), vec!["topic1".to_string()])
            .unwrap();

        group
            .join(Some("member-2".to_string()), "client-2".to_string(), vec!["topic1".to_string()])
            .unwrap();

        let mut topic_partitions = HashMap::new();
        topic_partitions.insert("topic1".to_string(), vec![0, 1, 2, 3]);

        let assignments = group.assign(&topic_partitions);

        assert_eq!(assignments.len(), 2);

        let total_partitions: usize = assignments.values().map(|v| v.len()).sum();
        assert_eq!(total_partitions, 4);
    }

    #[test]
    fn test_group_assignment_round_robin() {
        let config = GroupConfig {
            assignment_strategy: AssignmentStrategy::RoundRobin,
            ..Default::default()
        };

        let group = ConsumerGroup::new("test-group", config);

        group
            .join(Some("member-1".to_string()), "client-1".to_string(), vec!["topic1".to_string()])
            .unwrap();

        group
            .join(Some("member-2".to_string()), "client-2".to_string(), vec!["topic1".to_string()])
            .unwrap();

        let mut topic_partitions = HashMap::new();
        topic_partitions.insert("topic1".to_string(), vec![0, 1, 2, 3, 4, 5]);

        let assignments = group.assign(&topic_partitions);

        // Each member should get 3 partitions
        for partitions in assignments.values() {
            assert_eq!(partitions.len(), 3);
        }
    }

    #[test]
    fn test_group_sync() {
        let group = create_group();

        let join_resp = group
            .join(Some("member-1".to_string()), "client-1".to_string(), vec!["topic1".to_string()])
            .unwrap();

        let mut assignments = HashMap::new();
        assignments.insert(
            "member-1".to_string(),
            vec![TopicPartition::new("topic1", 0)],
        );

        let sync_resp = group
            .sync(&join_resp.member_id, join_resp.generation_id, Some(assignments))
            .unwrap();

        assert_eq!(sync_resp.assignment.len(), 1);
        assert_eq!(group.state(), GroupState::Stable);
    }

    #[test]
    fn test_group_describe() {
        let group = create_group();

        group
            .join(None, "client-1".to_string(), vec!["topic1".to_string()])
            .unwrap();

        let desc = group.describe();

        assert_eq!(desc.group_id, "test-group");
        assert_eq!(desc.members.len(), 1);
    }

    #[test]
    fn test_group_manager() {
        let manager = GroupManager::new(GroupConfig::default());

        let _ = manager.get_or_create("group-1");
        let _ = manager.get_or_create("group-2");

        let groups = manager.list_groups();
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_member_expiration() {
        let config = GroupConfig {
            session_timeout: Duration::from_millis(10),
            ..Default::default()
        };

        let group = ConsumerGroup::new("test-group", config);

        group
            .join(Some("member-1".to_string()), "client-1".to_string(), vec!["topic1".to_string()])
            .unwrap();

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(20));

        let expired = group.check_expired_members();
        assert_eq!(expired.len(), 1);
    }

    #[test]
    fn test_generation_increment() {
        let group = create_group();

        assert_eq!(group.generation(), 0);

        group.increment_generation();
        assert_eq!(group.generation(), 1);

        group.increment_generation();
        assert_eq!(group.generation(), 2);
    }

    #[test]
    fn test_group_state_display() {
        assert_eq!(format!("{}", GroupState::Empty), "Empty");
        assert_eq!(format!("{}", GroupState::Stable), "Stable");
        assert_eq!(format!("{}", GroupState::PreparingRebalance), "PreparingRebalance");
    }
}
