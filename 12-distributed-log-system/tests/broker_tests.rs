//! Comprehensive tests for broker components.

use distributed_log_system::broker::{
    Broker, BrokerConfig, BrokerId, BrokerMetrics, ConsumerGroupState, GroupCoordinator, GroupState,
    MemberMetadata, OffsetAndMetadata, PartitionAssignment, PartitionInfo, ReplicaState,
    TopicMetadata,
};
use distributed_log_system::log::{Record, RecordBatch, SegmentConfig, TopicPartition};
use distributed_log_system::protocol::{
    FetchPartition, FetchRequest, PartitionProduceData, ProduceRequest, TopicProduceData,
};
use std::time::Instant;
use tempfile::tempdir;

// =============================================================================
// BrokerConfig Tests
// =============================================================================

#[test]
fn test_broker_config_default() {
    let config = BrokerConfig::default();

    assert_eq!(config.id, 0);
    assert_eq!(config.listen_addr, "127.0.0.1:9092");
    assert_eq!(config.default_replication_factor, 3);
    assert_eq!(config.default_num_partitions, 3);
}

#[test]
fn test_broker_config_custom() {
    let config = BrokerConfig {
        id: 5,
        listen_addr: "0.0.0.0:9093".to_string(),
        data_dir: "/tmp/test".into(),
        default_replication_factor: 2,
        default_num_partitions: 6,
        segment_config: SegmentConfig::default(),
        isr_lag_time_ms: 5000,
        isr_lag_offset: 500,
    };

    assert_eq!(config.id, 5);
    assert_eq!(config.listen_addr, "0.0.0.0:9093");
    assert_eq!(config.default_replication_factor, 2);
}

// =============================================================================
// Broker Tests
// =============================================================================

#[test]
fn test_broker_creation() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    assert_eq!(broker.id, 0);
}

#[test]
fn test_broker_create_topic() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("test-topic".to_string(), 3, 1).unwrap();

    let topics = broker.list_topics();
    assert!(topics.contains(&"test-topic".to_string()));

    let topic = broker.get_topic("test-topic").unwrap();
    assert_eq!(topic.partitions, 3);
    assert_eq!(topic.replication_factor, 1);
}

#[test]
fn test_broker_create_topic_duplicate() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("test-topic".to_string(), 3, 1).unwrap();

    let result = broker.create_topic("test-topic".to_string(), 3, 1);
    assert!(result.is_err());
}

#[test]
fn test_broker_produce() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("test-topic".to_string(), 1, 1).unwrap();

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
            topic: "test-topic".to_string(),
            partition_data: vec![PartitionProduceData {
                partition: 0,
                records: batch,
            }],
        }],
    };

    let response = broker.handle_produce(request).unwrap();
    assert!(!response.responses.is_empty());
    assert_eq!(response.responses[0].error_code, 0);
    assert_eq!(response.responses[0].base_offset, 0);
}

#[test]
fn test_broker_fetch() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("test-topic".to_string(), 1, 1).unwrap();

    // Produce some data
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

    let produce_request = ProduceRequest {
        transactional_id: None,
        acks: 1,
        timeout_ms: 30000,
        topic_data: vec![TopicProduceData {
            topic: "test-topic".to_string(),
            partition_data: vec![PartitionProduceData {
                partition: 0,
                records: batch,
            }],
        }],
    };

    broker.handle_produce(produce_request).unwrap();

    // Fetch data
    let fetch_request = FetchRequest {
        replica_id: -1,
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 1048576,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![FetchPartition {
            topic: "test-topic".to_string(),
            partition: 0,
            fetch_offset: 0,
            max_bytes: 1048576,
        }],
    };

    let response = broker.handle_fetch(fetch_request).unwrap();
    assert!(!response.responses.is_empty());
    assert_eq!(response.responses[0].error_code, 0);
    assert!(!response.responses[0].record_batches.is_empty());
}

#[test]
fn test_broker_leader_check() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("test-topic".to_string(), 1, 1).unwrap();

    let tp = TopicPartition::new("test-topic", 0);
    assert!(broker.is_leader_for(&tp));

    let leader = broker.leader_for(&tp);
    assert_eq!(leader, Some(0));
}

#[test]
fn test_broker_partition_info() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("test-topic".to_string(), 1, 1).unwrap();

    let tp = TopicPartition::new("test-topic", 0);
    let info = broker.get_partition_info(&tp).unwrap();

    assert_eq!(info.topic, "test-topic");
    assert_eq!(info.partition, 0);
    assert_eq!(info.leader, 0);
}

#[test]
fn test_broker_list_topics() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("topic1".to_string(), 1, 1).unwrap();
    broker.create_topic("topic2".to_string(), 2, 1).unwrap();
    broker.create_topic("topic3".to_string(), 3, 1).unwrap();

    let topics = broker.list_topics();
    assert_eq!(topics.len(), 3);
    assert!(topics.contains(&"topic1".to_string()));
    assert!(topics.contains(&"topic2".to_string()));
    assert!(topics.contains(&"topic3".to_string()));
}

// =============================================================================
// BrokerMetrics Tests
// =============================================================================

#[test]
fn test_broker_metrics_new() {
    let metrics = BrokerMetrics::new();
    let snapshot = metrics.snapshot();

    assert_eq!(snapshot.bytes_in, 0);
    assert_eq!(snapshot.bytes_out, 0);
    assert_eq!(snapshot.messages_in, 0);
    assert_eq!(snapshot.messages_out, 0);
}

#[test]
fn test_broker_metrics_update() {
    use std::sync::atomic::Ordering;

    let metrics = BrokerMetrics::new();

    metrics.bytes_in.fetch_add(100, Ordering::Relaxed);
    metrics.messages_in.fetch_add(5, Ordering::Relaxed);

    let snapshot = metrics.snapshot();
    assert_eq!(snapshot.bytes_in, 100);
    assert_eq!(snapshot.messages_in, 5);
}

// =============================================================================
// TopicMetadata Tests
// =============================================================================

#[test]
fn test_topic_metadata() {
    let metadata = TopicMetadata {
        name: "test-topic".to_string(),
        partitions: 6,
        replication_factor: 3,
    };

    assert_eq!(metadata.name, "test-topic");
    assert_eq!(metadata.partitions, 6);
    assert_eq!(metadata.replication_factor, 3);
}

// =============================================================================
// PartitionAssignment Tests
// =============================================================================

#[test]
fn test_partition_assignment() {
    let assignment = PartitionAssignment {
        leader: 1,
        replicas: vec![1, 2, 3],
        isr: vec![1, 2, 3],
        leader_epoch: 0,
    };

    assert_eq!(assignment.leader, 1);
    assert_eq!(assignment.replicas.len(), 3);
    assert_eq!(assignment.isr.len(), 3);
    assert_eq!(assignment.leader_epoch, 0);
}

#[test]
fn test_partition_assignment_with_reduced_isr() {
    let assignment = PartitionAssignment {
        leader: 1,
        replicas: vec![1, 2, 3],
        isr: vec![1, 2], // Broker 3 fell behind
        leader_epoch: 1,
    };

    assert_eq!(assignment.isr.len(), 2);
    assert!(!assignment.isr.contains(&3));
}

// =============================================================================
// ReplicaState Tests
// =============================================================================

#[test]
fn test_replica_state() {
    let state = ReplicaState {
        fetch_offset: 100,
        last_fetch_time: Instant::now(),
    };

    assert_eq!(state.fetch_offset, 100);
}

// =============================================================================
// GroupCoordinator Tests
// =============================================================================

#[test]
fn test_group_coordinator_creation() {
    let coordinator = GroupCoordinator::new();
    assert!(coordinator.fetch_offsets("test-group", &[]).is_empty());
}

#[test]
fn test_group_coordinator_get_or_create() {
    let coordinator = GroupCoordinator::new();

    let group1 = coordinator.get_or_create_group("group1");
    assert_eq!(group1.group_id, "group1");
    assert_eq!(group1.state, GroupState::Empty);

    // Get same group again
    let group2 = coordinator.get_or_create_group("group1");
    assert_eq!(group2.group_id, "group1");
}

#[test]
fn test_group_coordinator_commit_offsets() {
    let coordinator = GroupCoordinator::new();

    // Create group first
    coordinator.get_or_create_group("test-group");

    let tp = TopicPartition::new("topic", 0);
    let offset = OffsetAndMetadata {
        offset: 100,
        metadata: "".to_string(),
        commit_timestamp: 1234567890,
    };

    coordinator
        .commit_offsets("test-group", vec![(tp.clone(), offset)])
        .unwrap();

    let fetched = coordinator.fetch_offsets("test-group", &[tp.clone()]);
    assert_eq!(fetched.get(&tp), Some(&100));
}

#[test]
fn test_group_coordinator_commit_multiple_partitions() {
    let coordinator = GroupCoordinator::new();
    coordinator.get_or_create_group("test-group");

    let tp0 = TopicPartition::new("topic", 0);
    let tp1 = TopicPartition::new("topic", 1);
    let tp2 = TopicPartition::new("topic", 2);

    let offsets = vec![
        (
            tp0.clone(),
            OffsetAndMetadata {
                offset: 100,
                metadata: "".to_string(),
                commit_timestamp: 0,
            },
        ),
        (
            tp1.clone(),
            OffsetAndMetadata {
                offset: 200,
                metadata: "".to_string(),
                commit_timestamp: 0,
            },
        ),
        (
            tp2.clone(),
            OffsetAndMetadata {
                offset: 300,
                metadata: "".to_string(),
                commit_timestamp: 0,
            },
        ),
    ];

    coordinator.commit_offsets("test-group", offsets).unwrap();

    let fetched = coordinator.fetch_offsets("test-group", &[tp0.clone(), tp1.clone(), tp2.clone()]);

    assert_eq!(fetched.get(&tp0), Some(&100));
    assert_eq!(fetched.get(&tp1), Some(&200));
    assert_eq!(fetched.get(&tp2), Some(&300));
}

#[test]
fn test_group_coordinator_fetch_unknown_group() {
    let coordinator = GroupCoordinator::new();

    let tp = TopicPartition::new("topic", 0);
    let fetched = coordinator.fetch_offsets("unknown-group", &[tp]);

    assert!(fetched.is_empty());
}

// =============================================================================
// ConsumerGroupState Tests
// =============================================================================

#[test]
fn test_consumer_group_state_creation() {
    let state = ConsumerGroupState {
        group_id: "test-group".to_string(),
        state: GroupState::Empty,
        generation_id: 0,
        leader: None,
        members: std::collections::HashMap::new(),
        offsets: std::collections::HashMap::new(),
    };

    assert_eq!(state.group_id, "test-group");
    assert_eq!(state.state, GroupState::Empty);
    assert_eq!(state.generation_id, 0);
    assert!(state.leader.is_none());
}

// =============================================================================
// GroupState Tests
// =============================================================================

#[test]
fn test_group_state_variants() {
    assert_eq!(GroupState::Empty, GroupState::Empty);
    assert_ne!(GroupState::Empty, GroupState::Stable);
    assert_ne!(GroupState::PreparingRebalance, GroupState::CompletingRebalance);
}

// =============================================================================
// MemberMetadata Tests
// =============================================================================

#[test]
fn test_member_metadata() {
    let metadata = MemberMetadata {
        member_id: "member-1".to_string(),
        client_id: "client-1".to_string(),
        client_host: "localhost".to_string(),
        session_timeout_ms: 30000,
        subscriptions: vec!["topic1".to_string(), "topic2".to_string()],
    };

    assert_eq!(metadata.member_id, "member-1");
    assert_eq!(metadata.subscriptions.len(), 2);
}

// =============================================================================
// PartitionInfo Tests
// =============================================================================

#[test]
fn test_partition_info() {
    let info = PartitionInfo {
        topic: "test-topic".to_string(),
        partition: 0,
        leader: 1,
        replicas: vec![1, 2, 3],
        isr: vec![1, 2, 3],
        high_watermark: 1000,
        log_end_offset: 1050,
    };

    assert_eq!(info.topic, "test-topic");
    assert_eq!(info.partition, 0);
    assert_eq!(info.high_watermark, 1000);
    assert_eq!(info.log_end_offset, 1050);
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_broker_produce_consume_flow() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("test-topic".to_string(), 3, 1).unwrap();

    // Produce to multiple partitions
    for partition in 0..3 {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key-{}", partition).into_bytes()),
                value: Some(format!("value-{}", partition).into_bytes()),
                headers: vec![],
            }],
        );

        let request = ProduceRequest {
            transactional_id: None,
            acks: 1,
            timeout_ms: 30000,
            topic_data: vec![TopicProduceData {
                topic: "test-topic".to_string(),
                partition_data: vec![PartitionProduceData {
                    partition,
                    records: batch,
                }],
            }],
        };

        let response = broker.handle_produce(request).unwrap();
        assert_eq!(response.responses[0].error_code, 0);
    }

    // Fetch from all partitions
    for partition in 0..3 {
        let fetch_request = FetchRequest {
            replica_id: -1,
            max_wait_ms: 500,
            min_bytes: 1,
            max_bytes: 1048576,
            isolation_level: 0,
            session_id: 0,
            session_epoch: 0,
            partitions: vec![FetchPartition {
                topic: "test-topic".to_string(),
                partition,
                fetch_offset: 0,
                max_bytes: 1048576,
            }],
        };

        let response = broker.handle_fetch(fetch_request).unwrap();
        assert!(!response.responses.is_empty());
        assert!(!response.responses[0].record_batches.is_empty());
    }
}

#[test]
fn test_broker_high_volume() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("high-volume".to_string(), 1, 1).unwrap();

    // Produce many messages
    for i in 0..100 {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key-{}", i).into_bytes()),
                value: Some(format!("value-{}", i).into_bytes()),
                headers: vec![],
            }],
        );

        let request = ProduceRequest {
            transactional_id: None,
            acks: 1,
            timeout_ms: 30000,
            topic_data: vec![TopicProduceData {
                topic: "high-volume".to_string(),
                partition_data: vec![PartitionProduceData {
                    partition: 0,
                    records: batch,
                }],
            }],
        };

        broker.handle_produce(request).unwrap();
    }

    // Verify all messages
    let fetch_request = FetchRequest {
        replica_id: -1,
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 10485760,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![FetchPartition {
            topic: "high-volume".to_string(),
            partition: 0,
            fetch_offset: 0,
            max_bytes: 10485760,
        }],
    };

    let response = broker.handle_fetch(fetch_request).unwrap();
    let total_records: usize = response.responses[0]
        .record_batches
        .iter()
        .map(|b| b.records.len())
        .sum();

    assert_eq!(total_records, 100);
}
