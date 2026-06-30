//! Consumer group protocol: JoinGroup, SyncGroup, Heartbeat, and Leave.

use crate::broker::{BrokerId, OffsetAndMetadata};
use crate::consumer::{PartitionAssignor, RangeAssignor, RoundRobinAssignor};
use crate::log::TopicPartition;
use crate::protocol::{
    error_codes, HeartbeatRequest, HeartbeatResponse, JoinGroupMember, JoinGroupProtocol,
    JoinGroupRequest, JoinGroupResponse, LeaveGroupRequest, LeaveGroupResponse,
    SyncGroupAssignment, SyncGroupRequest, SyncGroupResponse,
};
use crate::{Error, Offset, Result};

use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Group coordinator configuration.
#[derive(Debug, Clone)]
pub struct GroupCoordinatorConfig {
    /// Session timeout in milliseconds.
    pub session_timeout_ms: u64,
    /// Maximum session timeout.
    pub max_session_timeout_ms: u64,
    /// Minimum session timeout.
    pub min_session_timeout_ms: u64,
    /// Rebalance timeout in milliseconds.
    pub rebalance_timeout_ms: u64,
    /// Heartbeat interval.
    pub heartbeat_interval_ms: u64,
    /// Initial rebalance delay.
    pub initial_rebalance_delay_ms: u64,
}

impl Default for GroupCoordinatorConfig {
    fn default() -> Self {
        Self {
            session_timeout_ms: 30000,
            max_session_timeout_ms: 300000,
            min_session_timeout_ms: 6000,
            rebalance_timeout_ms: 60000,
            heartbeat_interval_ms: 3000,
            initial_rebalance_delay_ms: 3000,
        }
    }
}

/// Consumer group state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupState {
    /// Group has no members.
    Empty,
    /// Group is waiting for members to join.
    PreparingRebalance,
    /// Group is waiting for leader to send assignments.
    CompletingRebalance,
    /// Group is stable with all members synchronized.
    Stable,
    /// Group is dead and should be removed.
    Dead,
}

impl GroupState {
    pub fn as_str(&self) -> &'static str {
        match self {
            GroupState::Empty => "Empty",
            GroupState::PreparingRebalance => "PreparingRebalance",
            GroupState::CompletingRebalance => "CompletingRebalance",
            GroupState::Stable => "Stable",
            GroupState::Dead => "Dead",
        }
    }
}

/// Member metadata.
#[derive(Debug, Clone)]
pub struct MemberMetadata {
    /// Member ID.
    pub member_id: String,
    /// Client ID.
    pub client_id: String,
    /// Client host.
    pub client_host: String,
    /// Session timeout.
    pub session_timeout_ms: u64,
    /// Rebalance timeout.
    pub rebalance_timeout_ms: u64,
    /// Protocol type.
    pub protocol_type: String,
    /// Supported protocols.
    pub protocols: Vec<JoinGroupProtocol>,
    /// Assigned partitions (serialized).
    pub assignment: Vec<u8>,
    /// Last heartbeat time.
    pub last_heartbeat: Instant,
    /// Whether member is awaiting join response.
    pub awaiting_join: bool,
    /// Whether member is awaiting sync response.
    pub awaiting_sync: bool,
}

impl MemberMetadata {
    pub fn new(
        member_id: String,
        client_id: String,
        client_host: String,
        session_timeout_ms: u64,
        rebalance_timeout_ms: u64,
        protocol_type: String,
        protocols: Vec<JoinGroupProtocol>,
    ) -> Self {
        Self {
            member_id,
            client_id,
            client_host,
            session_timeout_ms,
            rebalance_timeout_ms,
            protocol_type,
            protocols,
            assignment: Vec::new(),
            last_heartbeat: Instant::now(),
            awaiting_join: false,
            awaiting_sync: false,
        }
    }

    pub fn update_heartbeat(&mut self) {
        self.last_heartbeat = Instant::now();
    }

    pub fn is_expired(&self, timeout: Duration) -> bool {
        self.last_heartbeat.elapsed() > timeout
    }
}

/// A consumer group.
pub struct ConsumerGroup {
    /// Group ID.
    group_id: String,
    /// Current state.
    state: RwLock<GroupState>,
    /// Generation ID.
    generation_id: AtomicU32,
    /// Current leader member ID.
    leader: RwLock<Option<String>>,
    /// Protocol type.
    protocol_type: RwLock<Option<String>>,
    /// Selected protocol.
    protocol: RwLock<Option<String>>,
    /// Group members.
    members: DashMap<String, MemberMetadata>,
    /// Pending members (awaiting join).
    pending_members: DashMap<String, MemberMetadata>,
    /// Committed offsets.
    offsets: DashMap<TopicPartition, OffsetAndMetadata>,
    /// Subscribed topics.
    subscriptions: RwLock<HashSet<String>>,
    /// Last state change time.
    last_state_change: RwLock<Instant>,
    /// Member ID counter.
    member_id_counter: AtomicU32,
}

impl ConsumerGroup {
    /// Create a new consumer group.
    pub fn new(group_id: String) -> Self {
        Self {
            group_id,
            state: RwLock::new(GroupState::Empty),
            generation_id: AtomicU32::new(0),
            leader: RwLock::new(None),
            protocol_type: RwLock::new(None),
            protocol: RwLock::new(None),
            members: DashMap::new(),
            pending_members: DashMap::new(),
            offsets: DashMap::new(),
            subscriptions: RwLock::new(HashSet::new()),
            last_state_change: RwLock::new(Instant::now()),
            member_id_counter: AtomicU32::new(0),
        }
    }

    /// Get group ID.
    pub fn group_id(&self) -> &str {
        &self.group_id
    }

    /// Get current state.
    pub fn state(&self) -> GroupState {
        *self.state.read()
    }

    /// Set state.
    fn set_state(&self, new_state: GroupState) {
        let mut state = self.state.write();
        if *state != new_state {
            debug!(
                "Group {} state change: {} -> {}",
                self.group_id,
                state.as_str(),
                new_state.as_str()
            );
            *state = new_state;
            *self.last_state_change.write() = Instant::now();
        }
    }

    /// Get generation ID.
    pub fn generation_id(&self) -> u32 {
        self.generation_id.load(Ordering::SeqCst)
    }

    /// Increment generation.
    fn next_generation(&self) -> u32 {
        self.generation_id.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Get leader.
    pub fn leader(&self) -> Option<String> {
        self.leader.read().clone()
    }

    /// Generate a new member ID.
    pub fn generate_member_id(&self, client_id: &str) -> String {
        let counter = self.member_id_counter.fetch_add(1, Ordering::SeqCst);
        format!("{}-{}", client_id, counter)
    }

    /// Check if group has members.
    pub fn has_members(&self) -> bool {
        !self.members.is_empty()
    }

    /// Get member count.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Get all member IDs.
    pub fn member_ids(&self) -> Vec<String> {
        self.members.iter().map(|e| e.key().clone()).collect()
    }

    /// Get member.
    pub fn get_member(&self, member_id: &str) -> Option<MemberMetadata> {
        self.members.get(member_id).map(|e| e.value().clone())
    }

    /// Add or update member.
    pub fn add_member(&self, member: MemberMetadata) {
        self.members.insert(member.member_id.clone(), member);
    }

    /// Remove member.
    pub fn remove_member(&self, member_id: &str) -> Option<MemberMetadata> {
        self.members.remove(member_id).map(|(_, m)| m)
    }

    /// Check if all members have joined.
    pub fn all_members_joined(&self) -> bool {
        self.members.iter().all(|e| !e.value().awaiting_join)
    }

    /// Check if all members have synced.
    pub fn all_members_synced(&self) -> bool {
        self.members.iter().all(|e| !e.value().awaiting_sync)
    }

    /// Select protocol.
    pub fn select_protocol(&self) -> Option<String> {
        // Find protocol supported by all members
        let mut protocol_counts: HashMap<String, usize> = HashMap::new();
        let member_count = self.members.len();

        for member in self.members.iter() {
            for protocol in &member.protocols {
                *protocol_counts.entry(protocol.name.clone()).or_insert(0) += 1;
            }
        }

        // Select first protocol supported by all
        for member in self.members.iter() {
            for protocol in &member.protocols {
                if protocol_counts.get(&protocol.name) == Some(&member_count) {
                    return Some(protocol.name.clone());
                }
            }
        }

        None
    }

    /// Commit offsets.
    pub fn commit_offsets(&self, offsets: Vec<(TopicPartition, OffsetAndMetadata)>) {
        for (tp, offset) in offsets {
            self.offsets.insert(tp, offset);
        }
    }

    /// Get committed offset.
    pub fn get_committed_offset(&self, tp: &TopicPartition) -> Option<Offset> {
        self.offsets.get(tp).map(|e| e.offset)
    }

    /// Get all committed offsets.
    pub fn get_all_offsets(&self) -> HashMap<TopicPartition, OffsetAndMetadata> {
        self.offsets
            .iter()
            .map(|e| (e.key().clone(), e.value().clone()))
            .collect()
    }
}

/// Delayed join operation.
#[derive(Debug)]
pub struct DelayedJoin {
    /// Group ID.
    pub group_id: String,
    /// Member ID.
    pub member_id: String,
    /// Deadline.
    pub deadline: Instant,
}

/// Group coordinator manages all consumer groups.
pub struct GroupCoordinatorService {
    /// Configuration.
    config: GroupCoordinatorConfig,
    /// Consumer groups by group ID.
    groups: DashMap<String, Arc<ConsumerGroup>>,
    /// Pending join operations.
    pending_joins: DashMap<String, Vec<DelayedJoin>>,
    /// Running flag.
    running: AtomicBool,
}

impl GroupCoordinatorService {
    /// Create a new group coordinator.
    pub fn new(config: GroupCoordinatorConfig) -> Self {
        Self {
            config,
            groups: DashMap::new(),
            pending_joins: DashMap::new(),
            running: AtomicBool::new(false),
        }
    }

    /// Start the coordinator.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        info!("Group coordinator started");
    }

    /// Stop the coordinator.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("Group coordinator stopped");
    }

    /// Get or create a group.
    pub fn get_or_create_group(&self, group_id: &str) -> Arc<ConsumerGroup> {
        self.groups
            .entry(group_id.to_string())
            .or_insert_with(|| Arc::new(ConsumerGroup::new(group_id.to_string())))
            .clone()
    }

    /// Get a group if it exists.
    pub fn get_group(&self, group_id: &str) -> Option<Arc<ConsumerGroup>> {
        self.groups.get(group_id).map(|e| e.clone())
    }

    /// Handle JoinGroup request.
    pub fn handle_join_group(&self, request: JoinGroupRequest) -> JoinGroupResponse {
        debug!(
            "JoinGroup request for group {} from member {}",
            request.group_id, request.member_id
        );

        // Validate session timeout
        if request.session_timeout_ms < self.config.min_session_timeout_ms as u32
            || request.session_timeout_ms > self.config.max_session_timeout_ms as u32
        {
            return JoinGroupResponse {
                throttle_time_ms: 0,
                error_code: error_codes::INVALID_GROUP_ID,
                generation_id: -1,
                protocol_name: String::new(),
                leader: String::new(),
                member_id: request.member_id,
                members: vec![],
            };
        }

        let group = self.get_or_create_group(&request.group_id);

        // Generate member ID if empty
        let member_id = if request.member_id.is_empty() {
            group.generate_member_id(&request.protocol_type)
        } else {
            request.member_id.clone()
        };

        // Handle based on state
        let current_state = group.state();

        match current_state {
            GroupState::Empty | GroupState::Stable => {
                // Add member and start rebalance
                let member = MemberMetadata::new(
                    member_id.clone(),
                    request.protocol_type.clone(),
                    String::new(), // Would get from connection
                    request.session_timeout_ms as u64,
                    request.rebalance_timeout_ms as u64,
                    request.protocol_type.clone(),
                    request.protocols.clone(),
                );
                group.add_member(member);

                // Transition to PreparingRebalance
                group.set_state(GroupState::PreparingRebalance);

                // Schedule delayed response
                let delay = Duration::from_millis(self.config.initial_rebalance_delay_ms);
                let deadline = Instant::now() + delay;

                self.pending_joins
                    .entry(request.group_id.clone())
                    .or_default()
                    .push(DelayedJoin {
                        group_id: request.group_id.clone(),
                        member_id: member_id.clone(),
                        deadline,
                    });

                // Check if we should complete immediately
                if self.should_complete_join(&group) {
                    return self.complete_join(&group, &member_id);
                }

                // Return pending response
                JoinGroupResponse {
                    throttle_time_ms: 0,
                    error_code: 0,
                    generation_id: -1,
                    protocol_name: String::new(),
                    leader: String::new(),
                    member_id,
                    members: vec![],
                }
            }

            GroupState::PreparingRebalance => {
                // Add to pending members
                let member = MemberMetadata::new(
                    member_id.clone(),
                    request.protocol_type.clone(),
                    String::new(),
                    request.session_timeout_ms as u64,
                    request.rebalance_timeout_ms as u64,
                    request.protocol_type.clone(),
                    request.protocols.clone(),
                );
                group.add_member(member);

                self.pending_joins
                    .entry(request.group_id.clone())
                    .or_default()
                    .push(DelayedJoin {
                        group_id: request.group_id.clone(),
                        member_id: member_id.clone(),
                        deadline: Instant::now()
                            + Duration::from_millis(self.config.rebalance_timeout_ms),
                    });

                if self.should_complete_join(&group) {
                    return self.complete_join(&group, &member_id);
                }

                JoinGroupResponse {
                    throttle_time_ms: 0,
                    error_code: 0,
                    generation_id: -1,
                    protocol_name: String::new(),
                    leader: String::new(),
                    member_id,
                    members: vec![],
                }
            }

            GroupState::CompletingRebalance => {
                // Must rejoin - return REBALANCE_IN_PROGRESS
                JoinGroupResponse {
                    throttle_time_ms: 0,
                    error_code: error_codes::REBALANCE_IN_PROGRESS,
                    generation_id: -1,
                    protocol_name: String::new(),
                    leader: String::new(),
                    member_id,
                    members: vec![],
                }
            }

            GroupState::Dead => {
                JoinGroupResponse {
                    throttle_time_ms: 0,
                    error_code: error_codes::UNKNOWN_MEMBER_ID,
                    generation_id: -1,
                    protocol_name: String::new(),
                    leader: String::new(),
                    member_id,
                    members: vec![],
                }
            }
        }
    }

    /// Check if join should complete.
    fn should_complete_join(&self, group: &ConsumerGroup) -> bool {
        // In real impl, would wait for all known members or timeout
        // For simplicity, complete after first member joins
        group.member_count() >= 1
    }

    /// Complete the join phase.
    fn complete_join(&self, group: &ConsumerGroup, for_member: &str) -> JoinGroupResponse {
        // Increment generation
        let generation = group.next_generation();

        // Select protocol
        let protocol = group.select_protocol().unwrap_or_default();
        *group.protocol.write() = Some(protocol.clone());

        // Select leader (first member)
        let member_ids = group.member_ids();
        let leader = member_ids.first().cloned().unwrap_or_default();
        *group.leader.write() = Some(leader.clone());

        // Build member list (only for leader)
        let members = if for_member == leader {
            member_ids
                .iter()
                .filter_map(|id| {
                    group.get_member(id).map(|m| {
                        // Get protocol metadata for selected protocol
                        let metadata = m
                            .protocols
                            .iter()
                            .find(|p| p.name == protocol)
                            .map(|p| p.metadata.clone())
                            .unwrap_or_default();

                        JoinGroupMember {
                            member_id: id.clone(),
                            metadata,
                        }
                    })
                })
                .collect()
        } else {
            vec![]
        };

        // Transition to CompletingRebalance
        group.set_state(GroupState::CompletingRebalance);

        info!(
            "Group {} completed join: generation={}, leader={}, members={}",
            group.group_id(),
            generation,
            leader,
            member_ids.len()
        );

        JoinGroupResponse {
            throttle_time_ms: 0,
            error_code: 0,
            generation_id: generation as i32,
            protocol_name: protocol,
            leader,
            member_id: for_member.to_string(),
            members,
        }
    }

    /// Handle SyncGroup request.
    pub fn handle_sync_group(&self, request: SyncGroupRequest) -> SyncGroupResponse {
        debug!(
            "SyncGroup request for group {} from member {}",
            request.group_id, request.member_id
        );

        let group = match self.get_group(&request.group_id) {
            Some(g) => g,
            None => {
                return SyncGroupResponse {
                    throttle_time_ms: 0,
                    error_code: error_codes::INVALID_GROUP_ID,
                    assignment: vec![],
                }
            }
        };

        // Validate member
        if group.get_member(&request.member_id).is_none() {
            return SyncGroupResponse {
                throttle_time_ms: 0,
                error_code: error_codes::UNKNOWN_MEMBER_ID,
                assignment: vec![],
            };
        }

        // Validate generation
        if request.generation_id != group.generation_id() as i32 {
            return SyncGroupResponse {
                throttle_time_ms: 0,
                error_code: error_codes::ILLEGAL_GENERATION,
                assignment: vec![],
            };
        }

        let current_state = group.state();

        match current_state {
            GroupState::CompletingRebalance => {
                // If leader, store assignments
                let leader = group.leader();
                if leader.as_ref() == Some(&request.member_id) {
                    for assignment in &request.assignments {
                        if let Some(mut member) = group.members.get_mut(&assignment.member_id) {
                            member.assignment = assignment.assignment.clone();
                            member.awaiting_sync = false;
                        }
                    }
                }

                // Get this member's assignment
                let assignment = group
                    .get_member(&request.member_id)
                    .map(|m| m.assignment)
                    .unwrap_or_default();

                // Check if all synced
                if group.all_members_synced() {
                    group.set_state(GroupState::Stable);
                    info!(
                        "Group {} is now stable with {} members",
                        group.group_id(),
                        group.member_count()
                    );
                }

                SyncGroupResponse {
                    throttle_time_ms: 0,
                    error_code: 0,
                    assignment,
                }
            }

            GroupState::Stable => {
                // Return current assignment
                let assignment = group
                    .get_member(&request.member_id)
                    .map(|m| m.assignment)
                    .unwrap_or_default();

                SyncGroupResponse {
                    throttle_time_ms: 0,
                    error_code: 0,
                    assignment,
                }
            }

            _ => SyncGroupResponse {
                throttle_time_ms: 0,
                error_code: error_codes::REBALANCE_IN_PROGRESS,
                assignment: vec![],
            },
        }
    }

    /// Handle Heartbeat request.
    pub fn handle_heartbeat(&self, request: HeartbeatRequest) -> HeartbeatResponse {
        let group = match self.get_group(&request.group_id) {
            Some(g) => g,
            None => {
                return HeartbeatResponse {
                    throttle_time_ms: 0,
                    error_code: error_codes::INVALID_GROUP_ID,
                }
            }
        };

        // Validate member
        let mut member = match group.members.get_mut(&request.member_id) {
            Some(m) => m,
            None => {
                return HeartbeatResponse {
                    throttle_time_ms: 0,
                    error_code: error_codes::UNKNOWN_MEMBER_ID,
                }
            }
        };

        // Validate generation
        if request.generation_id != group.generation_id() as i32 {
            return HeartbeatResponse {
                throttle_time_ms: 0,
                error_code: error_codes::ILLEGAL_GENERATION,
            };
        }

        // Update heartbeat
        member.value_mut().update_heartbeat();

        // Check state
        match group.state() {
            GroupState::Stable => HeartbeatResponse {
                throttle_time_ms: 0,
                error_code: 0,
            },
            GroupState::PreparingRebalance | GroupState::CompletingRebalance => HeartbeatResponse {
                throttle_time_ms: 0,
                error_code: error_codes::REBALANCE_IN_PROGRESS,
            },
            _ => HeartbeatResponse {
                throttle_time_ms: 0,
                error_code: error_codes::UNKNOWN_MEMBER_ID,
            },
        }
    }

    /// Handle LeaveGroup request.
    pub fn handle_leave_group(&self, request: LeaveGroupRequest) -> LeaveGroupResponse {
        debug!(
            "LeaveGroup request for group {} from member {}",
            request.group_id, request.member_id
        );

        let group = match self.get_group(&request.group_id) {
            Some(g) => g,
            None => {
                return LeaveGroupResponse {
                    throttle_time_ms: 0,
                    error_code: error_codes::INVALID_GROUP_ID,
                }
            }
        };

        // Remove member
        if group.remove_member(&request.member_id).is_none() {
            return LeaveGroupResponse {
                throttle_time_ms: 0,
                error_code: error_codes::UNKNOWN_MEMBER_ID,
            };
        }

        info!(
            "Member {} left group {}",
            request.member_id, request.group_id
        );

        // Trigger rebalance if group is stable and has remaining members
        if group.state() == GroupState::Stable && group.has_members() {
            group.set_state(GroupState::PreparingRebalance);
        } else if !group.has_members() {
            group.set_state(GroupState::Empty);
        }

        LeaveGroupResponse {
            throttle_time_ms: 0,
            error_code: 0,
        }
    }

    /// Check for expired members across all groups.
    pub fn check_expired_members(&self) -> Vec<(String, String)> {
        let mut expired = Vec::new();

        for group_entry in self.groups.iter() {
            let group = group_entry.value();
            let timeout = Duration::from_millis(self.config.session_timeout_ms);

            let expired_members: Vec<String> = group
                .members
                .iter()
                .filter(|e| e.value().is_expired(timeout))
                .map(|e| e.key().clone())
                .collect();

            for member_id in expired_members {
                if group.remove_member(&member_id).is_some() {
                    warn!(
                        "Removed expired member {} from group {}",
                        member_id,
                        group.group_id()
                    );
                    expired.push((group.group_id().to_string(), member_id));
                }
            }

            // Update group state if needed
            if !group.has_members() {
                group.set_state(GroupState::Empty);
            } else if group.state() == GroupState::Stable {
                group.set_state(GroupState::PreparingRebalance);
            }
        }

        expired
    }

    /// List all groups.
    pub fn list_groups(&self) -> Vec<GroupInfo> {
        self.groups
            .iter()
            .map(|e| GroupInfo {
                group_id: e.key().clone(),
                state: e.value().state(),
                member_count: e.value().member_count(),
                generation_id: e.value().generation_id(),
            })
            .collect()
    }

    /// Get group details.
    pub fn describe_group(&self, group_id: &str) -> Option<GroupDetails> {
        self.get_group(group_id).map(|g| GroupDetails {
            group_id: g.group_id().to_string(),
            state: g.state(),
            protocol_type: g.protocol_type.read().clone(),
            protocol: g.protocol.read().clone(),
            generation_id: g.generation_id(),
            leader: g.leader(),
            members: g
                .members
                .iter()
                .map(|e| MemberInfo {
                    member_id: e.key().clone(),
                    client_id: e.value().client_id.clone(),
                    client_host: e.value().client_host.clone(),
                    assignment_size: e.value().assignment.len(),
                })
                .collect(),
        })
    }
}

/// Group info for listing.
#[derive(Debug, Clone)]
pub struct GroupInfo {
    pub group_id: String,
    pub state: GroupState,
    pub member_count: usize,
    pub generation_id: u32,
}

/// Detailed group information.
#[derive(Debug, Clone)]
pub struct GroupDetails {
    pub group_id: String,
    pub state: GroupState,
    pub protocol_type: Option<String>,
    pub protocol: Option<String>,
    pub generation_id: u32,
    pub leader: Option<String>,
    pub members: Vec<MemberInfo>,
}

/// Member info in group details.
#[derive(Debug, Clone)]
pub struct MemberInfo {
    pub member_id: String,
    pub client_id: String,
    pub client_host: String,
    pub assignment_size: usize,
}

/// Assignment serialization/deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemberAssignment {
    /// Assigned partitions.
    pub partitions: Vec<TopicPartition>,
    /// User data.
    pub user_data: Vec<u8>,
}

impl MemberAssignment {
    pub fn new(partitions: Vec<TopicPartition>) -> Self {
        Self {
            partitions,
            user_data: Vec::new(),
        }
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| Error::Serialization(e.to_string()))
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| Error::Serialization(e.to_string()))
    }
}

/// Subscription metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionMetadata {
    /// Subscribed topics.
    pub topics: Vec<String>,
    /// User data.
    pub user_data: Vec<u8>,
}

impl SubscriptionMetadata {
    pub fn new(topics: Vec<String>) -> Self {
        Self {
            topics,
            user_data: Vec::new(),
        }
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| Error::Serialization(e.to_string()))
    }

    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| Error::Serialization(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_consumer_group_creation() {
        let group = ConsumerGroup::new("test-group".to_string());
        assert_eq!(group.group_id(), "test-group");
        assert_eq!(group.state(), GroupState::Empty);
        assert_eq!(group.generation_id(), 0);
        assert!(!group.has_members());
    }

    #[test]
    fn test_member_id_generation() {
        let group = ConsumerGroup::new("test-group".to_string());
        let id1 = group.generate_member_id("client");
        let id2 = group.generate_member_id("client");
        assert_ne!(id1, id2);
        assert!(id1.starts_with("client-"));
    }

    #[test]
    fn test_add_remove_member() {
        let group = ConsumerGroup::new("test-group".to_string());

        let member = MemberMetadata::new(
            "member-1".to_string(),
            "client-1".to_string(),
            "localhost".to_string(),
            30000,
            60000,
            "consumer".to_string(),
            vec![],
        );

        group.add_member(member);
        assert!(group.has_members());
        assert_eq!(group.member_count(), 1);

        let removed = group.remove_member("member-1");
        assert!(removed.is_some());
        assert!(!group.has_members());
    }

    #[test]
    fn test_group_state_transitions() {
        let group = ConsumerGroup::new("test-group".to_string());

        assert_eq!(group.state(), GroupState::Empty);

        group.set_state(GroupState::PreparingRebalance);
        assert_eq!(group.state(), GroupState::PreparingRebalance);

        group.set_state(GroupState::CompletingRebalance);
        assert_eq!(group.state(), GroupState::CompletingRebalance);

        group.set_state(GroupState::Stable);
        assert_eq!(group.state(), GroupState::Stable);
    }

    #[test]
    fn test_generation_increment() {
        let group = ConsumerGroup::new("test-group".to_string());

        assert_eq!(group.generation_id(), 0);

        let gen1 = group.next_generation();
        assert_eq!(gen1, 1);
        assert_eq!(group.generation_id(), 1);

        let gen2 = group.next_generation();
        assert_eq!(gen2, 2);
    }

    #[test]
    fn test_offset_commit() {
        let group = ConsumerGroup::new("test-group".to_string());
        let tp = TopicPartition::new("topic", 0);

        let offset = OffsetAndMetadata {
            offset: 100,
            metadata: "".to_string(),
            commit_timestamp: 0,
        };

        group.commit_offsets(vec![(tp.clone(), offset)]);

        let committed = group.get_committed_offset(&tp);
        assert_eq!(committed, Some(100));
    }

    #[test]
    fn test_group_coordinator_service() {
        let config = GroupCoordinatorConfig::default();
        let coordinator = GroupCoordinatorService::new(config);

        coordinator.start();

        let group = coordinator.get_or_create_group("test-group");
        assert_eq!(group.group_id(), "test-group");

        let groups = coordinator.list_groups();
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].group_id, "test-group");

        coordinator.stop();
    }

    #[test]
    fn test_join_group_flow() {
        let config = GroupCoordinatorConfig::default();
        let coordinator = GroupCoordinatorService::new(config);

        // First join
        let request = JoinGroupRequest {
            group_id: "test-group".to_string(),
            session_timeout_ms: 30000,
            rebalance_timeout_ms: 60000,
            member_id: "".to_string(),
            protocol_type: "consumer".to_string(),
            protocols: vec![JoinGroupProtocol {
                name: "range".to_string(),
                metadata: vec![],
            }],
        };

        let response = coordinator.handle_join_group(request);
        assert_eq!(response.error_code, 0);
        assert!(!response.member_id.is_empty());
    }

    #[test]
    fn test_heartbeat() {
        let config = GroupCoordinatorConfig::default();
        let coordinator = GroupCoordinatorService::new(config);

        // Join first
        let join_request = JoinGroupRequest {
            group_id: "test-group".to_string(),
            session_timeout_ms: 30000,
            rebalance_timeout_ms: 60000,
            member_id: "".to_string(),
            protocol_type: "consumer".to_string(),
            protocols: vec![JoinGroupProtocol {
                name: "range".to_string(),
                metadata: vec![],
            }],
        };

        let join_response = coordinator.handle_join_group(join_request);
        let member_id = join_response.member_id;
        let generation = join_response.generation_id;

        // Heartbeat
        let hb_request = HeartbeatRequest {
            group_id: "test-group".to_string(),
            generation_id: generation,
            member_id: member_id.clone(),
        };

        let hb_response = coordinator.handle_heartbeat(hb_request);
        // May return REBALANCE_IN_PROGRESS since we're still in CompletingRebalance
        assert!(hb_response.error_code == 0 || hb_response.error_code == error_codes::REBALANCE_IN_PROGRESS);
    }

    #[test]
    fn test_leave_group() {
        let config = GroupCoordinatorConfig::default();
        let coordinator = GroupCoordinatorService::new(config);

        // Join first
        let join_request = JoinGroupRequest {
            group_id: "test-group".to_string(),
            session_timeout_ms: 30000,
            rebalance_timeout_ms: 60000,
            member_id: "".to_string(),
            protocol_type: "consumer".to_string(),
            protocols: vec![JoinGroupProtocol {
                name: "range".to_string(),
                metadata: vec![],
            }],
        };

        let join_response = coordinator.handle_join_group(join_request);
        let member_id = join_response.member_id;

        // Leave
        let leave_request = LeaveGroupRequest {
            group_id: "test-group".to_string(),
            member_id: member_id.clone(),
        };

        let leave_response = coordinator.handle_leave_group(leave_request);
        assert_eq!(leave_response.error_code, 0);

        // Verify member removed
        let group = coordinator.get_group("test-group").unwrap();
        assert_eq!(group.member_count(), 0);
    }

    #[test]
    fn test_member_assignment_serialization() {
        let partitions = vec![
            TopicPartition::new("topic1", 0),
            TopicPartition::new("topic1", 1),
        ];

        let assignment = MemberAssignment::new(partitions.clone());
        let data = assignment.serialize().unwrap();
        let deserialized = MemberAssignment::deserialize(&data).unwrap();

        assert_eq!(deserialized.partitions.len(), 2);
        assert_eq!(deserialized.partitions[0].topic, "topic1");
    }

    #[test]
    fn test_subscription_metadata() {
        let topics = vec!["topic1".to_string(), "topic2".to_string()];
        let subscription = SubscriptionMetadata::new(topics.clone());

        let data = subscription.serialize().unwrap();
        let deserialized = SubscriptionMetadata::deserialize(&data).unwrap();

        assert_eq!(deserialized.topics, topics);
    }

    #[test]
    fn test_member_expiration() {
        let mut member = MemberMetadata::new(
            "member-1".to_string(),
            "client-1".to_string(),
            "localhost".to_string(),
            100, // Very short timeout
            60000,
            "consumer".to_string(),
            vec![],
        );

        assert!(!member.is_expired(Duration::from_millis(100)));

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(150));
        assert!(member.is_expired(Duration::from_millis(100)));
    }
}
