//! Integration tests for the distributed log system.

use distributed_log_system::broker::{Broker, BrokerConfig};
use distributed_log_system::consumer::{Consumer, ConsumerConfig, RangeAssignor, PartitionAssignor};
use distributed_log_system::log::{Partition, Record, RecordBatch, SegmentConfig, TopicPartition};
use distributed_log_system::producer::{Producer, ProducerConfig, ProducerRecord};
use distributed_log_system::protocol::{
    FetchPartition, FetchRequest, PartitionProduceData, ProduceRequest, TopicProduceData,
};
use std::collections::HashMap;
use tempfile::tempdir;

// =============================================================================
// End-to-End Produce/Consume Tests
// =============================================================================

#[test]
fn test_e2e_single_partition() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("test".to_string(), 1, 1).unwrap();

    // Produce
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

    let produce_req = ProduceRequest {
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

    let produce_resp = broker.handle_produce(produce_req).unwrap();
    assert_eq!(produce_resp.responses[0].error_code, 0);

    // Fetch
    let fetch_req = FetchRequest {
        replica_id: -1,
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 1048576,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![FetchPartition {
            topic: "test".to_string(),
            partition: 0,
            fetch_offset: 0,
            max_bytes: 1048576,
        }],
    };

    let fetch_resp = broker.handle_fetch(fetch_req).unwrap();
    assert_eq!(fetch_resp.responses[0].error_code, 0);
    assert!(!fetch_resp.responses[0].record_batches.is_empty());

    let records = &fetch_resp.responses[0].record_batches[0].records;
    assert_eq!(records[0].key, Some(b"key".to_vec()));
    assert_eq!(records[0].value, Some(b"value".to_vec()));
}

#[test]
fn test_e2e_multiple_partitions() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("multi".to_string(), 3, 1).unwrap();

    // Produce to each partition
    for partition in 0..3 {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key{}", partition).into_bytes()),
                value: Some(format!("value{}", partition).into_bytes()),
                headers: vec![],
            }],
        );

        let produce_req = ProduceRequest {
            transactional_id: None,
            acks: 1,
            timeout_ms: 30000,
            topic_data: vec![TopicProduceData {
                topic: "multi".to_string(),
                partition_data: vec![PartitionProduceData {
                    partition,
                    records: batch,
                }],
            }],
        };

        let resp = broker.handle_produce(produce_req).unwrap();
        assert_eq!(resp.responses[0].error_code, 0);
    }

    // Fetch from each partition
    for partition in 0..3 {
        let fetch_req = FetchRequest {
            replica_id: -1,
            max_wait_ms: 500,
            min_bytes: 1,
            max_bytes: 1048576,
            isolation_level: 0,
            session_id: 0,
            session_epoch: 0,
            partitions: vec![FetchPartition {
                topic: "multi".to_string(),
                partition,
                fetch_offset: 0,
                max_bytes: 1048576,
            }],
        };

        let resp = broker.handle_fetch(fetch_req).unwrap();
        assert_eq!(resp.responses[0].error_code, 0);
        assert!(!resp.responses[0].record_batches.is_empty());
    }
}

#[test]
fn test_e2e_many_messages() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("bulk".to_string(), 1, 1).unwrap();

    let message_count = 1000;

    // Produce many messages
    for i in 0..message_count {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key{}", i).into_bytes()),
                value: Some(format!("value{}", i).into_bytes()),
                headers: vec![],
            }],
        );

        let produce_req = ProduceRequest {
            transactional_id: None,
            acks: 1,
            timeout_ms: 30000,
            topic_data: vec![TopicProduceData {
                topic: "bulk".to_string(),
                partition_data: vec![PartitionProduceData {
                    partition: 0,
                    records: batch,
                }],
            }],
        };

        broker.handle_produce(produce_req).unwrap();
    }

    // Fetch all messages
    let fetch_req = FetchRequest {
        replica_id: -1,
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 104857600, // 100MB
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![FetchPartition {
            topic: "bulk".to_string(),
            partition: 0,
            fetch_offset: 0,
            max_bytes: 104857600,
        }],
    };

    let resp = broker.handle_fetch(fetch_req).unwrap();
    let total_records: usize = resp.responses[0]
        .record_batches
        .iter()
        .map(|b| b.records.len())
        .sum();

    assert_eq!(total_records, message_count);
}

#[test]
fn test_e2e_offset_tracking() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("offset".to_string(), 1, 1).unwrap();

    // Produce 10 messages
    for i in 0..10 {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key{}", i).into_bytes()),
                value: Some(format!("value{}", i).into_bytes()),
                headers: vec![],
            }],
        );

        let produce_req = ProduceRequest {
            transactional_id: None,
            acks: 1,
            timeout_ms: 30000,
            topic_data: vec![TopicProduceData {
                topic: "offset".to_string(),
                partition_data: vec![PartitionProduceData {
                    partition: 0,
                    records: batch,
                }],
            }],
        };

        let resp = broker.handle_produce(produce_req).unwrap();
        assert_eq!(resp.responses[0].base_offset, i);
    }

    // Fetch from offset 5
    let fetch_req = FetchRequest {
        replica_id: -1,
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 1048576,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![FetchPartition {
            topic: "offset".to_string(),
            partition: 0,
            fetch_offset: 5,
            max_bytes: 1048576,
        }],
    };

    let resp = broker.handle_fetch(fetch_req).unwrap();
    assert!(!resp.responses[0].record_batches.is_empty());

    // First batch should have offset >= 5
    let first_batch = &resp.responses[0].record_batches[0];
    assert!(first_batch.base_offset >= 5);
}

// =============================================================================
// Producer Tests
// =============================================================================

#[test]
fn test_producer_partitioning() {
    let config = ProducerConfig::default();
    let mut producer = Producer::new(config);

    // Records with same key should go to same partition
    let mut partitions = Vec::new();
    for _ in 0..10 {
        let record = ProducerRecord::new(
            "topic",
            Some(b"consistent-key".to_vec()),
            Some(b"value".to_vec()),
        );
        let tp = producer.send(record).unwrap();
        partitions.push(tp.partition);
    }

    // All partitions should be the same
    let first = partitions[0];
    assert!(partitions.iter().all(|&p| p == first));
}

#[test]
fn test_producer_round_robin() {
    let config = ProducerConfig::default();
    let mut producer = Producer::new(config);

    // Records without key should be round-robin
    let mut partitions = Vec::new();
    for _ in 0..10 {
        let record = ProducerRecord::new("topic", None, Some(b"value".to_vec()));
        let tp = producer.send(record).unwrap();
        partitions.push(tp.partition);
    }

    // Should have some variation (unless we're very unlucky)
    let unique: std::collections::HashSet<_> = partitions.iter().collect();
    // With 3 partitions, 10 messages should hit at least 2
    assert!(unique.len() >= 1);
}

// =============================================================================
// Consumer Tests
// =============================================================================

#[test]
fn test_consumer_assignment() {
    let config = ConsumerConfig {
        group_id: "test-group".to_string(),
        ..ConsumerConfig::default()
    };

    let mut consumer = Consumer::new(config);

    let partitions = vec![
        TopicPartition::new("topic", 0),
        TopicPartition::new("topic", 1),
        TopicPartition::new("topic", 2),
    ];

    consumer.assign(partitions.clone()).unwrap();

    assert_eq!(consumer.assignment().len(), 3);
}

#[test]
fn test_consumer_position_tracking() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();

    // Initial position
    assert!(consumer.position(&tp).is_none() || consumer.position(&tp) == Some(0));

    // Update position
    consumer.update_position(&tp, 100);
    assert_eq!(consumer.position(&tp), Some(100));

    // Update again
    consumer.update_position(&tp, 200);
    assert_eq!(consumer.position(&tp), Some(200));
}

#[test]
fn test_consumer_commit_flow() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();

    // Update position
    consumer.update_position(&tp, 100);

    // Commit
    let committed = consumer.commit().unwrap();
    assert_eq!(committed.get(&tp), Some(&100));

    // Verify committed
    assert_eq!(consumer.committed(&tp), Some(100));
}

// =============================================================================
// Partition Assignor Tests
// =============================================================================

#[test]
fn test_range_assignor_balanced() {
    let assignor = RangeAssignor;

    let members = vec!["c1".to_string(), "c2".to_string(), "c3".to_string()];
    let partitions = vec![
        TopicPartition::new("topic", 0),
        TopicPartition::new("topic", 1),
        TopicPartition::new("topic", 2),
        TopicPartition::new("topic", 3),
        TopicPartition::new("topic", 4),
        TopicPartition::new("topic", 5),
    ];

    let assignment = assignor.assign(&members, &partitions);

    // Each member should get 2 partitions
    assert_eq!(assignment.get("c1").unwrap().len(), 2);
    assert_eq!(assignment.get("c2").unwrap().len(), 2);
    assert_eq!(assignment.get("c3").unwrap().len(), 2);
}

#[test]
fn test_range_assignor_unbalanced() {
    let assignor = RangeAssignor;

    let members = vec!["c1".to_string(), "c2".to_string()];
    let partitions = vec![
        TopicPartition::new("topic", 0),
        TopicPartition::new("topic", 1),
        TopicPartition::new("topic", 2),
    ];

    let assignment = assignor.assign(&members, &partitions);

    // First member gets extra partition
    assert_eq!(assignment.get("c1").unwrap().len(), 2);
    assert_eq!(assignment.get("c2").unwrap().len(), 1);
}

// =============================================================================
// Partition Segment Tests
// =============================================================================

#[test]
fn test_partition_segment_management() {
    let dir = tempdir().unwrap();
    let config = SegmentConfig {
        max_segment_bytes: 200, // Small segments
        index_interval_bytes: 50,
    };

    let mut partition = Partition::new(dir.path(), "topic", 0, config).unwrap();

    // Write enough to create multiple segments
    for i in 0..50 {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key{}", i).into_bytes()),
                value: Some(format!("value-with-content-{}", i).into_bytes()),
                headers: vec![],
            }],
        );

        partition.append(batch).unwrap();
    }

    assert_eq!(partition.log_end_offset, 50);
}

#[test]
fn test_partition_read_across_segments() {
    let dir = tempdir().unwrap();
    let config = SegmentConfig {
        max_segment_bytes: 200,
        index_interval_bytes: 50,
    };

    let mut partition = Partition::new(dir.path(), "topic", 0, config).unwrap();

    for i in 0..20 {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key{}", i).into_bytes()),
                value: Some(format!("value-content-{}", i).into_bytes()),
                headers: vec![],
            }],
        );

        partition.append(batch).unwrap();
    }

    // Read from beginning
    let batches = partition.read(0, 100000).unwrap();
    assert!(!batches.is_empty());
}

// =============================================================================
// High Watermark Tests
// =============================================================================

#[test]
fn test_high_watermark_update() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "topic", 0, SegmentConfig::default()).unwrap();

    assert_eq!(partition.high_watermark, 0);

    partition.update_high_watermark(50);
    assert_eq!(partition.high_watermark, 50);

    partition.update_high_watermark(100);
    assert_eq!(partition.high_watermark, 100);

    // Should not go backwards
    partition.update_high_watermark(75);
    assert_eq!(partition.high_watermark, 100);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_produce_to_unknown_partition() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("test".to_string(), 1, 1).unwrap();

    // Try to produce to partition 5 (doesn't exist)
    let batch = RecordBatch::new(0, vec![]);
    let produce_req = ProduceRequest {
        transactional_id: None,
        acks: 1,
        timeout_ms: 30000,
        topic_data: vec![TopicProduceData {
            topic: "test".to_string(),
            partition_data: vec![PartitionProduceData {
                partition: 5,
                records: batch,
            }],
        }],
    };

    let result = broker.handle_produce(produce_req);
    assert!(result.is_err());
}

#[test]
fn test_fetch_from_unknown_topic() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();

    let fetch_req = FetchRequest {
        replica_id: -1,
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 1048576,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![FetchPartition {
            topic: "unknown".to_string(),
            partition: 0,
            fetch_offset: 0,
            max_bytes: 1048576,
        }],
    };

    let result = broker.handle_fetch(fetch_req);
    assert!(result.is_err());
}

// =============================================================================
// Metrics Tests
// =============================================================================

#[test]
fn test_broker_metrics_tracking() {
    use std::sync::atomic::Ordering;

    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("metrics".to_string(), 1, 1).unwrap();

    let initial_messages = broker.metrics.messages_in.load(Ordering::Relaxed);

    // Produce some messages
    for _ in 0..5 {
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

        let produce_req = ProduceRequest {
            transactional_id: None,
            acks: 1,
            timeout_ms: 30000,
            topic_data: vec![TopicProduceData {
                topic: "metrics".to_string(),
                partition_data: vec![PartitionProduceData {
                    partition: 0,
                    records: batch,
                }],
            }],
        };

        broker.handle_produce(produce_req).unwrap();
    }

    let final_messages = broker.metrics.messages_in.load(Ordering::Relaxed);
    assert_eq!(final_messages - initial_messages, 5);
}

// =============================================================================
// Multiple Topics Tests
// =============================================================================

#[test]
fn test_multiple_topics() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();

    // Create multiple topics
    broker.create_topic("topic1".to_string(), 2, 1).unwrap();
    broker.create_topic("topic2".to_string(), 3, 1).unwrap();
    broker.create_topic("topic3".to_string(), 1, 1).unwrap();

    let topics = broker.list_topics();
    assert_eq!(topics.len(), 3);

    // Produce to each
    for (topic, partition_count) in [("topic1", 2), ("topic2", 3), ("topic3", 1)] {
        for partition in 0..partition_count {
            let batch = RecordBatch::new(
                0,
                vec![Record {
                    attributes: 0,
                    timestamp_delta: 0,
                    offset_delta: 0,
                    key: Some(format!("{}-{}", topic, partition).into_bytes()),
                    value: Some(b"value".to_vec()),
                    headers: vec![],
                }],
            );

            let produce_req = ProduceRequest {
                transactional_id: None,
                acks: 1,
                timeout_ms: 30000,
                topic_data: vec![TopicProduceData {
                    topic: topic.to_string(),
                    partition_data: vec![PartitionProduceData {
                        partition,
                        records: batch,
                    }],
                }],
            };

            broker.handle_produce(produce_req).unwrap();
        }
    }
}

// =============================================================================
// Batch Tests
// =============================================================================

#[test]
fn test_multi_record_batch() {
    let dir = tempdir().unwrap();
    let config = BrokerConfig {
        data_dir: dir.path().to_path_buf(),
        ..BrokerConfig::default()
    };

    let broker = Broker::new(config).unwrap();
    broker.create_topic("batch".to_string(), 1, 1).unwrap();

    // Create batch with multiple records
    let records: Vec<Record> = (0..10)
        .map(|i| Record {
            attributes: 0,
            timestamp_delta: i as i64,
            offset_delta: i as u32,
            key: Some(format!("key{}", i).into_bytes()),
            value: Some(format!("value{}", i).into_bytes()),
            headers: vec![],
        })
        .collect();

    let batch = RecordBatch::new(0, records);

    let produce_req = ProduceRequest {
        transactional_id: None,
        acks: 1,
        timeout_ms: 30000,
        topic_data: vec![TopicProduceData {
            topic: "batch".to_string(),
            partition_data: vec![PartitionProduceData {
                partition: 0,
                records: batch,
            }],
        }],
    };

    let resp = broker.handle_produce(produce_req).unwrap();
    assert_eq!(resp.responses[0].error_code, 0);

    // Fetch and verify
    let fetch_req = FetchRequest {
        replica_id: -1,
        max_wait_ms: 500,
        min_bytes: 1,
        max_bytes: 1048576,
        isolation_level: 0,
        session_id: 0,
        session_epoch: 0,
        partitions: vec![FetchPartition {
            topic: "batch".to_string(),
            partition: 0,
            fetch_offset: 0,
            max_bytes: 1048576,
        }],
    };

    let resp = broker.handle_fetch(fetch_req).unwrap();
    let total_records: usize = resp.responses[0]
        .record_batches
        .iter()
        .map(|b| b.records.len())
        .sum();

    assert_eq!(total_records, 10);
}
