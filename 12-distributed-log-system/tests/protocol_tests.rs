//! Comprehensive tests for protocol message types.

use distributed_log_system::protocol::*;
use distributed_log_system::log::{Record, RecordBatch, TopicPartition};

// =============================================================================
// Request/Response Enum Tests
// =============================================================================

#[test]
fn test_request_produce_variant() {
    let request = Request::Produce(ProduceRequest {
        transactional_id: None,
        acks: 1,
        timeout_ms: 30000,
        topic_data: vec![],
    });

    assert!(matches!(request, Request::Produce(_)));
}

#[test]
fn test_request_fetch_variant() {
    let request = Request::Fetch(FetchRequest {
        replica_id: -1,
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 1048576,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![],
    });

    assert!(matches!(request, Request::Fetch(_)));
}

#[test]
fn test_request_metadata_variant() {
    let request = Request::Metadata(MetadataRequest {
        topics: Some(vec!["test".to_string()]),
        allow_auto_topic_creation: false,
    });

    assert!(matches!(request, Request::Metadata(_)));
}

#[test]
fn test_response_produce_variant() {
    let response = Response::Produce(ProduceResponse {
        responses: vec![],
        throttle_time_ms: 0,
    });

    assert!(matches!(response, Response::Produce(_)));
}

// =============================================================================
// ProduceRequest Tests
// =============================================================================

#[test]
fn test_produce_request_basic() {
    let request = ProduceRequest {
        transactional_id: None,
        acks: 1,
        timeout_ms: 30000,
        topic_data: vec![],
    };

    assert!(request.transactional_id.is_none());
    assert_eq!(request.acks, 1);
    assert_eq!(request.timeout_ms, 30000);
}

#[test]
fn test_produce_request_with_transaction() {
    let request = ProduceRequest {
        transactional_id: Some("tx-123".to_string()),
        acks: -1,
        timeout_ms: 60000,
        topic_data: vec![],
    };

    assert_eq!(request.transactional_id, Some("tx-123".to_string()));
    assert_eq!(request.acks, -1);
}

#[test]
fn test_produce_request_with_data() {
    let batch = RecordBatch::new(
        0,
        vec![Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 0,
            key: Some(b"key".to_vec()),
            value: Some(b"value".to_vec()),
            headers: vec![],
        }],
    );

    let request = ProduceRequest {
        transactional_id: None,
        acks: 1,
        timeout_ms: 30000,
        topic_data: vec![TopicProduceData {
            topic: "test".to_string(),
            partition_data: vec![PartitionProduceData {
                partition: 0,
                records: batch,
            }],
        }],
    };

    assert_eq!(request.topic_data.len(), 1);
    assert_eq!(request.topic_data[0].topic, "test");
    assert_eq!(request.topic_data[0].partition_data.len(), 1);
}

// =============================================================================
// ProduceResponse Tests
// =============================================================================

#[test]
fn test_produce_response_success() {
    let response = ProduceResponse {
        responses: vec![PartitionProduceResponse {
            partition: 0,
            error_code: 0,
            base_offset: 100,
            log_append_time: 1234567890,
        }],
        throttle_time_ms: 0,
    };

    assert_eq!(response.responses[0].error_code, 0);
    assert_eq!(response.responses[0].base_offset, 100);
}

#[test]
fn test_produce_response_error() {
    let response = ProduceResponse {
        responses: vec![PartitionProduceResponse {
            partition: 0,
            error_code: error_codes::NOT_LEADER_FOR_PARTITION,
            base_offset: 0, // Error case: offset not assigned
            log_append_time: -1,
        }],
        throttle_time_ms: 0,
    };

    assert_eq!(response.responses[0].error_code, error_codes::NOT_LEADER_FOR_PARTITION);
}

#[test]
fn test_produce_response_multiple_partitions() {
    let response = ProduceResponse {
        responses: vec![
            PartitionProduceResponse {
                partition: 0,
                error_code: 0,
                base_offset: 100,
                log_append_time: 0,
            },
            PartitionProduceResponse {
                partition: 1,
                error_code: 0,
                base_offset: 50,
                log_append_time: 0,
            },
            PartitionProduceResponse {
                partition: 2,
                error_code: 0,
                base_offset: 200,
                log_append_time: 0,
            },
        ],
        throttle_time_ms: 100,
    };

    assert_eq!(response.responses.len(), 3);
    assert_eq!(response.throttle_time_ms, 100);
}

// =============================================================================
// FetchRequest Tests
// =============================================================================

#[test]
fn test_fetch_request_consumer() {
    let request = FetchRequest {
        replica_id: -1, // Consumer
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 1048576,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![],
    };

    assert_eq!(request.replica_id, -1);
}

#[test]
fn test_fetch_request_replica() {
    let request = FetchRequest {
        replica_id: 2, // Follower broker
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 1048576,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![],
    };

    assert_eq!(request.replica_id, 2);
}

#[test]
fn test_fetch_request_with_partitions() {
    let request = FetchRequest {
        replica_id: -1,
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 1048576,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![
            FetchPartition {
                topic: "topic1".to_string(),
                partition: 0,
                fetch_offset: 100,
                max_bytes: 1048576,
            },
            FetchPartition {
                topic: "topic1".to_string(),
                partition: 1,
                fetch_offset: 200,
                max_bytes: 1048576,
            },
        ],
    };

    assert_eq!(request.partitions.len(), 2);
    assert_eq!(request.partitions[0].fetch_offset, 100);
    assert_eq!(request.partitions[1].fetch_offset, 200);
}

// =============================================================================
// FetchResponse Tests
// =============================================================================

#[test]
fn test_fetch_response_success() {
    let response = FetchResponse {
        throttle_time_ms: 0,
        error_code: 0,
        session_id: 0,
        responses: vec![FetchPartitionResponse {
            topic: "topic".to_string(),
            partition: 0,
            error_code: 0,
            high_watermark: 1000,
            last_stable_offset: 1000,
            log_start_offset: 0,
            record_batches: vec![],
        }],
    };

    assert_eq!(response.error_code, 0);
    assert_eq!(response.responses[0].high_watermark, 1000);
}

#[test]
fn test_fetch_response_with_records() {
    let batch = RecordBatch::new(
        100,
        vec![Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 0,
            key: Some(b"key".to_vec()),
            value: Some(b"value".to_vec()),
            headers: vec![],
        }],
    );

    let response = FetchResponse {
        throttle_time_ms: 0,
        error_code: 0,
        session_id: 0,
        responses: vec![FetchPartitionResponse {
            topic: "topic".to_string(),
            partition: 0,
            error_code: 0,
            high_watermark: 101,
            last_stable_offset: 101,
            log_start_offset: 0,
            record_batches: vec![batch],
        }],
    };

    assert_eq!(response.responses[0].record_batches.len(), 1);
}

// =============================================================================
// ListOffsetsRequest Tests
// =============================================================================

#[test]
fn test_list_offsets_request_latest() {
    let request = ListOffsetsRequest {
        replica_id: -1,
        isolation_level: 0,
        topics: vec![ListOffsetsTopicRequest {
            topic: "test".to_string(),
            partitions: vec![ListOffsetsPartitionRequest {
                partition: 0,
                timestamp: -1, // Latest
            }],
        }],
    };

    assert_eq!(request.topics[0].partitions[0].timestamp, -1);
}

#[test]
fn test_list_offsets_request_earliest() {
    let request = ListOffsetsRequest {
        replica_id: -1,
        isolation_level: 0,
        topics: vec![ListOffsetsTopicRequest {
            topic: "test".to_string(),
            partitions: vec![ListOffsetsPartitionRequest {
                partition: 0,
                timestamp: -2, // Earliest
            }],
        }],
    };

    assert_eq!(request.topics[0].partitions[0].timestamp, -2);
}

// =============================================================================
// MetadataRequest Tests
// =============================================================================

#[test]
fn test_metadata_request_all_topics() {
    let request = MetadataRequest {
        topics: None,
        allow_auto_topic_creation: false,
    };

    assert!(request.topics.is_none());
}

#[test]
fn test_metadata_request_specific_topics() {
    let request = MetadataRequest {
        topics: Some(vec!["topic1".to_string(), "topic2".to_string()]),
        allow_auto_topic_creation: true,
    };

    assert_eq!(request.topics.as_ref().unwrap().len(), 2);
    assert!(request.allow_auto_topic_creation);
}

// =============================================================================
// MetadataResponse Tests
// =============================================================================

#[test]
fn test_metadata_response() {
    let response = MetadataResponse {
        throttle_time_ms: 0,
        brokers: vec![
            BrokerMetadata {
                node_id: 0,
                host: "broker1".to_string(),
                port: 9092,
                rack: Some("rack1".to_string()),
            },
            BrokerMetadata {
                node_id: 1,
                host: "broker2".to_string(),
                port: 9092,
                rack: Some("rack2".to_string()),
            },
        ],
        cluster_id: Some("cluster-123".to_string()),
        controller_id: 0,
        topics: vec![],
    };

    assert_eq!(response.brokers.len(), 2);
    assert_eq!(response.controller_id, 0);
}

#[test]
fn test_metadata_response_with_topics() {
    let response = MetadataResponse {
        throttle_time_ms: 0,
        brokers: vec![],
        cluster_id: None,
        controller_id: 0,
        topics: vec![TopicMetadata {
            error_code: 0,
            name: "test".to_string(),
            is_internal: false,
            partitions: vec![
                PartitionMetadata {
                    error_code: 0,
                    partition: 0,
                    leader: 0,
                    leader_epoch: 1,
                    replicas: vec![0, 1, 2],
                    isr: vec![0, 1, 2],
                },
            ],
        }],
    };

    assert_eq!(response.topics.len(), 1);
    assert_eq!(response.topics[0].partitions[0].replicas.len(), 3);
}

// =============================================================================
// OffsetCommit Tests
// =============================================================================

#[test]
fn test_offset_commit_request() {
    let request = OffsetCommitRequest {
        group_id: "group1".to_string(),
        generation_id: 1,
        member_id: "member1".to_string(),
        topics: vec![OffsetCommitTopic {
            topic: "test".to_string(),
            partitions: vec![OffsetCommitPartition {
                partition: 0,
                offset: 100,
                metadata: "".to_string(),
            }],
        }],
    };

    assert_eq!(request.group_id, "group1");
    assert_eq!(request.generation_id, 1);
}

#[test]
fn test_offset_commit_response() {
    let response = OffsetCommitResponse {
        throttle_time_ms: 0,
        topics: vec![OffsetCommitTopicResponse {
            topic: "test".to_string(),
            partitions: vec![OffsetCommitPartitionResponse {
                partition: 0,
                error_code: 0,
            }],
        }],
    };

    assert_eq!(response.topics[0].partitions[0].error_code, 0);
}

// =============================================================================
// OffsetFetch Tests
// =============================================================================

#[test]
fn test_offset_fetch_request() {
    let request = OffsetFetchRequest {
        group_id: "group1".to_string(),
        topics: vec![OffsetFetchTopic {
            topic: "test".to_string(),
            partitions: vec![0, 1, 2],
        }],
    };

    assert_eq!(request.topics[0].partitions.len(), 3);
}

#[test]
fn test_offset_fetch_response() {
    let response = OffsetFetchResponse {
        throttle_time_ms: 0,
        topics: vec![OffsetFetchTopicResponse {
            topic: "test".to_string(),
            partitions: vec![OffsetFetchPartitionResponse {
                partition: 0,
                offset: 100,
                metadata: "".to_string(),
                error_code: 0,
            }],
        }],
        error_code: 0,
    };

    assert_eq!(response.topics[0].partitions[0].offset, 100);
}

// =============================================================================
// JoinGroup Tests
// =============================================================================

#[test]
fn test_join_group_request() {
    let request = JoinGroupRequest {
        group_id: "group1".to_string(),
        session_timeout_ms: 30000,
        rebalance_timeout_ms: 60000,
        member_id: "".to_string(),
        protocol_type: "consumer".to_string(),
        protocols: vec![JoinGroupProtocol {
            name: "range".to_string(),
            metadata: vec![],
        }],
    };

    assert!(request.member_id.is_empty());
    assert_eq!(request.protocol_type, "consumer");
}

#[test]
fn test_join_group_response() {
    let response = JoinGroupResponse {
        throttle_time_ms: 0,
        error_code: 0,
        generation_id: 1,
        protocol_name: "range".to_string(),
        leader: "member1".to_string(),
        member_id: "member1".to_string(),
        members: vec![
            JoinGroupMember {
                member_id: "member1".to_string(),
                metadata: vec![],
            },
            JoinGroupMember {
                member_id: "member2".to_string(),
                metadata: vec![],
            },
        ],
    };

    assert_eq!(response.generation_id, 1);
    assert_eq!(response.members.len(), 2);
}

// =============================================================================
// SyncGroup Tests
// =============================================================================

#[test]
fn test_sync_group_request() {
    let request = SyncGroupRequest {
        group_id: "group1".to_string(),
        generation_id: 1,
        member_id: "member1".to_string(),
        assignments: vec![
            SyncGroupAssignment {
                member_id: "member1".to_string(),
                assignment: vec![1, 2, 3],
            },
            SyncGroupAssignment {
                member_id: "member2".to_string(),
                assignment: vec![4, 5, 6],
            },
        ],
    };

    assert_eq!(request.assignments.len(), 2);
}

#[test]
fn test_sync_group_response() {
    let response = SyncGroupResponse {
        throttle_time_ms: 0,
        error_code: 0,
        assignment: vec![1, 2, 3, 4, 5],
    };

    assert_eq!(response.assignment.len(), 5);
}

// =============================================================================
// Heartbeat Tests
// =============================================================================

#[test]
fn test_heartbeat_request() {
    let request = HeartbeatRequest {
        group_id: "group1".to_string(),
        generation_id: 1,
        member_id: "member1".to_string(),
    };

    assert_eq!(request.generation_id, 1);
}

#[test]
fn test_heartbeat_response_success() {
    let response = HeartbeatResponse {
        throttle_time_ms: 0,
        error_code: 0,
    };

    assert_eq!(response.error_code, 0);
}

#[test]
fn test_heartbeat_response_rebalance() {
    let response = HeartbeatResponse {
        throttle_time_ms: 0,
        error_code: error_codes::REBALANCE_IN_PROGRESS,
    };

    assert_eq!(response.error_code, error_codes::REBALANCE_IN_PROGRESS);
}

// =============================================================================
// LeaveGroup Tests
// =============================================================================

#[test]
fn test_leave_group_request() {
    let request = LeaveGroupRequest {
        group_id: "group1".to_string(),
        member_id: "member1".to_string(),
    };

    assert_eq!(request.group_id, "group1");
}

#[test]
fn test_leave_group_response() {
    let response = LeaveGroupResponse {
        throttle_time_ms: 0,
        error_code: 0,
    };

    assert_eq!(response.error_code, 0);
}

// =============================================================================
// Error Codes Tests
// =============================================================================

#[test]
fn test_error_codes() {
    assert_eq!(error_codes::NONE, 0);
    assert_eq!(error_codes::UNKNOWN, -1);
    assert_eq!(error_codes::OFFSET_OUT_OF_RANGE, 1);
    assert_eq!(error_codes::CORRUPT_MESSAGE, 2);
    assert_eq!(error_codes::UNKNOWN_TOPIC_OR_PARTITION, 3);
    assert_eq!(error_codes::LEADER_NOT_AVAILABLE, 5);
    assert_eq!(error_codes::NOT_LEADER_FOR_PARTITION, 6);
    assert_eq!(error_codes::REQUEST_TIMED_OUT, 7);
}

#[test]
fn test_error_codes_replication() {
    assert_eq!(error_codes::REPLICA_NOT_AVAILABLE, 9);
    assert_eq!(error_codes::NOT_ENOUGH_REPLICAS, 19);
    assert_eq!(error_codes::NOT_ENOUGH_REPLICAS_AFTER_APPEND, 20);
}

#[test]
fn test_error_codes_group() {
    assert_eq!(error_codes::ILLEGAL_GENERATION, 22);
    assert_eq!(error_codes::REBALANCE_IN_PROGRESS, 27);
    assert_eq!(error_codes::INVALID_GROUP_ID, 24);
    assert_eq!(error_codes::UNKNOWN_MEMBER_ID, 25);
    assert_eq!(error_codes::NOT_COORDINATOR, 16);
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_produce_request_serialization() {
    let request = ProduceRequest {
        transactional_id: None,
        acks: 1,
        timeout_ms: 30000,
        topic_data: vec![],
    };

    let serialized = bincode::serialize(&request).unwrap();
    let deserialized: ProduceRequest = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.acks, request.acks);
    assert_eq!(deserialized.timeout_ms, request.timeout_ms);
}

#[test]
fn test_fetch_response_serialization() {
    let response = FetchResponse {
        throttle_time_ms: 100,
        error_code: 0,
        session_id: 123,
        responses: vec![],
    };

    let serialized = bincode::serialize(&response).unwrap();
    let deserialized: FetchResponse = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.throttle_time_ms, response.throttle_time_ms);
    assert_eq!(deserialized.session_id, response.session_id);
}

#[test]
fn test_join_group_request_serialization() {
    let request = JoinGroupRequest {
        group_id: "test-group".to_string(),
        session_timeout_ms: 30000,
        rebalance_timeout_ms: 60000,
        member_id: "member-1".to_string(),
        protocol_type: "consumer".to_string(),
        protocols: vec![JoinGroupProtocol {
            name: "range".to_string(),
            metadata: vec![1, 2, 3],
        }],
    };

    let serialized = bincode::serialize(&request).unwrap();
    let deserialized: JoinGroupRequest = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.group_id, request.group_id);
    assert_eq!(deserialized.protocols.len(), 1);
}
