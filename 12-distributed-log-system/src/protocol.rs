//! Protocol: request and response message types.

use crate::log::{RecordBatch, TopicPartition};
use crate::Offset;
use serde::{Deserialize, Serialize};

/// API request types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    Produce(ProduceRequest),
    Fetch(FetchRequest),
    ListOffsets(ListOffsetsRequest),
    Metadata(MetadataRequest),
    OffsetCommit(OffsetCommitRequest),
    OffsetFetch(OffsetFetchRequest),
    JoinGroup(JoinGroupRequest),
    SyncGroup(SyncGroupRequest),
    Heartbeat(HeartbeatRequest),
    LeaveGroup(LeaveGroupRequest),
}

/// API response types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    Produce(ProduceResponse),
    Fetch(FetchResponse),
    ListOffsets(ListOffsetsResponse),
    Metadata(MetadataResponse),
    OffsetCommit(OffsetCommitResponse),
    OffsetFetch(OffsetFetchResponse),
    JoinGroup(JoinGroupResponse),
    SyncGroup(SyncGroupResponse),
    Heartbeat(HeartbeatResponse),
    LeaveGroup(LeaveGroupResponse),
}

/// Produce request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProduceRequest {
    pub transactional_id: Option<String>,
    pub acks: i16,
    pub timeout_ms: u32,
    pub topic_data: Vec<TopicProduceData>,
}

/// Produce data for a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicProduceData {
    pub topic: String,
    pub partition_data: Vec<PartitionProduceData>,
}

/// Produce data for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionProduceData {
    pub partition: u32,
    pub records: RecordBatch,
}

/// Produce response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProduceResponse {
    pub responses: Vec<PartitionProduceResponse>,
    pub throttle_time_ms: u32,
}

/// Produce response for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionProduceResponse {
    pub partition: u32,
    pub error_code: i16,
    pub base_offset: Offset,
    pub log_append_time: i64,
}

/// Fetch request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchRequest {
    pub replica_id: i32,
    pub max_wait_ms: u32,
    pub min_bytes: u32,
    pub max_bytes: u32,
    pub isolation_level: i8,
    pub session_id: i32,
    pub session_epoch: i32,
    pub partitions: Vec<FetchPartition>,
}

/// Fetch request for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchPartition {
    pub topic: String,
    pub partition: u32,
    pub fetch_offset: Offset,
    pub max_bytes: u32,
}

/// Fetch response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchResponse {
    pub throttle_time_ms: u32,
    pub error_code: i16,
    pub session_id: i32,
    pub responses: Vec<FetchPartitionResponse>,
}

/// Fetch response for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchPartitionResponse {
    pub partition: u32,
    pub error_code: i16,
    pub high_watermark: Offset,
    pub last_stable_offset: Offset,
    pub log_start_offset: Offset,
    pub record_batches: Vec<RecordBatch>,
}

/// List offsets request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListOffsetsRequest {
    pub replica_id: i32,
    pub isolation_level: i8,
    pub topics: Vec<ListOffsetsTopicRequest>,
}

/// List offsets for a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListOffsetsTopicRequest {
    pub topic: String,
    pub partitions: Vec<ListOffsetsPartitionRequest>,
}

/// List offsets for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListOffsetsPartitionRequest {
    pub partition: u32,
    pub timestamp: i64, // -1 for latest, -2 for earliest
}

/// List offsets response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListOffsetsResponse {
    pub throttle_time_ms: u32,
    pub topics: Vec<ListOffsetsTopicResponse>,
}

/// List offsets response for a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListOffsetsTopicResponse {
    pub topic: String,
    pub partitions: Vec<ListOffsetsPartitionResponse>,
}

/// List offsets response for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListOffsetsPartitionResponse {
    pub partition: u32,
    pub error_code: i16,
    pub timestamp: i64,
    pub offset: Offset,
}

/// Metadata request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataRequest {
    pub topics: Option<Vec<String>>,
    pub allow_auto_topic_creation: bool,
}

/// Metadata response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataResponse {
    pub throttle_time_ms: u32,
    pub brokers: Vec<BrokerMetadata>,
    pub cluster_id: Option<String>,
    pub controller_id: i32,
    pub topics: Vec<TopicMetadata>,
}

/// Broker metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokerMetadata {
    pub node_id: i32,
    pub host: String,
    pub port: u32,
    pub rack: Option<String>,
}

/// Topic metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicMetadata {
    pub error_code: i16,
    pub name: String,
    pub is_internal: bool,
    pub partitions: Vec<PartitionMetadata>,
}

/// Partition metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionMetadata {
    pub error_code: i16,
    pub partition: u32,
    pub leader: i32,
    pub leader_epoch: i32,
    pub replicas: Vec<i32>,
    pub isr: Vec<i32>,
}

/// Offset commit request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetCommitRequest {
    pub group_id: String,
    pub generation_id: i32,
    pub member_id: String,
    pub topics: Vec<OffsetCommitTopic>,
}

/// Offset commit for a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetCommitTopic {
    pub topic: String,
    pub partitions: Vec<OffsetCommitPartition>,
}

/// Offset commit for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetCommitPartition {
    pub partition: u32,
    pub offset: Offset,
    pub metadata: String,
}

/// Offset commit response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetCommitResponse {
    pub throttle_time_ms: u32,
    pub topics: Vec<OffsetCommitTopicResponse>,
}

/// Offset commit response for a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetCommitTopicResponse {
    pub topic: String,
    pub partitions: Vec<OffsetCommitPartitionResponse>,
}

/// Offset commit response for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetCommitPartitionResponse {
    pub partition: u32,
    pub error_code: i16,
}

/// Offset fetch request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetFetchRequest {
    pub group_id: String,
    pub topics: Vec<OffsetFetchTopic>,
}

/// Offset fetch for a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetFetchTopic {
    pub topic: String,
    pub partitions: Vec<u32>,
}

/// Offset fetch response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetFetchResponse {
    pub throttle_time_ms: u32,
    pub topics: Vec<OffsetFetchTopicResponse>,
    pub error_code: i16,
}

/// Offset fetch response for a topic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetFetchTopicResponse {
    pub topic: String,
    pub partitions: Vec<OffsetFetchPartitionResponse>,
}

/// Offset fetch response for a partition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetFetchPartitionResponse {
    pub partition: u32,
    pub offset: Offset,
    pub metadata: String,
    pub error_code: i16,
}

/// Join group request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinGroupRequest {
    pub group_id: String,
    pub session_timeout_ms: u32,
    pub rebalance_timeout_ms: u32,
    pub member_id: String,
    pub protocol_type: String,
    pub protocols: Vec<JoinGroupProtocol>,
}

/// Join group protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinGroupProtocol {
    pub name: String,
    pub metadata: Vec<u8>,
}

/// Join group response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinGroupResponse {
    pub throttle_time_ms: u32,
    pub error_code: i16,
    pub generation_id: i32,
    pub protocol_name: String,
    pub leader: String,
    pub member_id: String,
    pub members: Vec<JoinGroupMember>,
}

/// Join group member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinGroupMember {
    pub member_id: String,
    pub metadata: Vec<u8>,
}

/// Sync group request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncGroupRequest {
    pub group_id: String,
    pub generation_id: i32,
    pub member_id: String,
    pub assignments: Vec<SyncGroupAssignment>,
}

/// Sync group assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncGroupAssignment {
    pub member_id: String,
    pub assignment: Vec<u8>,
}

/// Sync group response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncGroupResponse {
    pub throttle_time_ms: u32,
    pub error_code: i16,
    pub assignment: Vec<u8>,
}

/// Heartbeat request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatRequest {
    pub group_id: String,
    pub generation_id: i32,
    pub member_id: String,
}

/// Heartbeat response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatResponse {
    pub throttle_time_ms: u32,
    pub error_code: i16,
}

/// Leave group request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaveGroupRequest {
    pub group_id: String,
    pub member_id: String,
}

/// Leave group response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaveGroupResponse {
    pub throttle_time_ms: u32,
    pub error_code: i16,
}

/// Error codes.
pub mod error_codes {
    pub const NONE: i16 = 0;
    pub const UNKNOWN: i16 = -1;
    pub const OFFSET_OUT_OF_RANGE: i16 = 1;
    pub const CORRUPT_MESSAGE: i16 = 2;
    pub const UNKNOWN_TOPIC_OR_PARTITION: i16 = 3;
    pub const LEADER_NOT_AVAILABLE: i16 = 5;
    pub const NOT_LEADER_FOR_PARTITION: i16 = 6;
    pub const REQUEST_TIMED_OUT: i16 = 7;
    pub const REPLICA_NOT_AVAILABLE: i16 = 9;
    pub const RECORD_LIST_TOO_LARGE: i16 = 18;
    pub const NOT_ENOUGH_REPLICAS: i16 = 19;
    pub const NOT_ENOUGH_REPLICAS_AFTER_APPEND: i16 = 20;
    pub const INVALID_REQUIRED_ACKS: i16 = 21;
    pub const ILLEGAL_GENERATION: i16 = 22;
    pub const REBALANCE_IN_PROGRESS: i16 = 27;
    pub const INVALID_GROUP_ID: i16 = 24;
    pub const UNKNOWN_MEMBER_ID: i16 = 25;
    pub const NOT_COORDINATOR: i16 = 16;
}
