//! Comprehensive tests for producer and consumer components.

use distributed_log_system::consumer::{
    AutoOffsetReset, Consumer, ConsumerConfig, ConsumerGroup, ConsumerRecord, PartitionAssignor,
    RangeAssignor, RoundRobinAssignor,
};
use distributed_log_system::log::TopicPartition;
use distributed_log_system::producer::{
    Acks, Producer, ProducerConfig, ProducerRecord, RecordAccumulator, RecordMetadata,
};

// =============================================================================
// Acks Tests
// =============================================================================

#[test]
fn test_acks_none() {
    let acks = Acks::None;
    assert_eq!(acks.to_i16(), 0);
}

#[test]
fn test_acks_leader() {
    let acks = Acks::Leader;
    assert_eq!(acks.to_i16(), 1);
}

#[test]
fn test_acks_all() {
    let acks = Acks::All;
    assert_eq!(acks.to_i16(), -1);
}

// =============================================================================
// ProducerConfig Tests
// =============================================================================

#[test]
fn test_producer_config_default() {
    let config = ProducerConfig::default();

    assert_eq!(config.client_id, "kafka-lite-producer");
    assert_eq!(config.timeout_ms, 30000);
    assert_eq!(config.batch_size, 16384);
    assert!(!config.enable_idempotence);
}

#[test]
fn test_producer_config_custom() {
    let config = ProducerConfig {
        client_id: "my-producer".to_string(),
        acks: Acks::All,
        timeout_ms: 60000,
        batch_size: 32768,
        linger_ms: 5,
        max_request_size: 2097152,
        enable_idempotence: true,
        retries: 5,
    };

    assert_eq!(config.client_id, "my-producer");
    assert_eq!(config.batch_size, 32768);
    assert!(config.enable_idempotence);
}

// =============================================================================
// ProducerRecord Tests
// =============================================================================

#[test]
fn test_producer_record_creation() {
    let record = ProducerRecord::new("topic", Some(b"key".to_vec()), Some(b"value".to_vec()));

    assert_eq!(record.topic, "topic");
    assert_eq!(record.key, Some(b"key".to_vec()));
    assert_eq!(record.value, Some(b"value".to_vec()));
    assert!(record.partition.is_none());
}

#[test]
fn test_producer_record_with_partition() {
    let record =
        ProducerRecord::new("topic", Some(b"key".to_vec()), Some(b"value".to_vec())).with_partition(5);

    assert_eq!(record.partition, Some(5));
}

#[test]
fn test_producer_record_with_header() {
    let record = ProducerRecord::new("topic", Some(b"key".to_vec()), Some(b"value".to_vec()))
        .with_header("content-type", b"application/json".to_vec())
        .with_header("trace-id", b"abc123".to_vec());

    assert_eq!(record.headers.len(), 2);
    assert_eq!(record.headers[0].0, "content-type");
}

#[test]
fn test_producer_record_no_key() {
    let record = ProducerRecord::new("topic", None, Some(b"value".to_vec()));

    assert!(record.key.is_none());
    assert!(record.value.is_some());
}

#[test]
fn test_producer_record_no_value() {
    let record = ProducerRecord::new("topic", Some(b"key".to_vec()), None);

    assert!(record.key.is_some());
    assert!(record.value.is_none());
}

// =============================================================================
// RecordAccumulator Tests
// =============================================================================

#[test]
fn test_record_accumulator_creation() {
    let config = ProducerConfig::default();
    let accumulator = RecordAccumulator::new(&config);

    let tp = TopicPartition::new("topic", 0);
    assert!(!accumulator.is_ready(&tp));
}

#[test]
fn test_record_accumulator_append() {
    let config = ProducerConfig {
        batch_size: 100,
        ..ProducerConfig::default()
    };
    let mut accumulator = RecordAccumulator::new(&config);

    let tp = TopicPartition::new("topic", 0);
    let record = distributed_log_system::log::Record {
        attributes: 0,
        timestamp_delta: 0,
        offset_delta: 0,
        key: Some(b"key".to_vec()),
        value: Some(b"value".to_vec()),
        headers: vec![],
    };

    let ready = accumulator.append(tp.clone(), record);
    assert!(!ready); // Small record, not ready yet
}

#[test]
fn test_record_accumulator_ready() {
    let config = ProducerConfig {
        batch_size: 20, // Very small batch size
        ..ProducerConfig::default()
    };
    let mut accumulator = RecordAccumulator::new(&config);

    let tp = TopicPartition::new("topic", 0);

    // Append enough to fill batch
    for _ in 0..5 {
        let record = distributed_log_system::log::Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 0,
            key: Some(b"key".to_vec()),
            value: Some(b"value".to_vec()),
            headers: vec![],
        };
        accumulator.append(tp.clone(), record);
    }

    assert!(accumulator.is_ready(&tp));
}

#[test]
fn test_record_accumulator_drain() {
    let config = ProducerConfig::default();
    let mut accumulator = RecordAccumulator::new(&config);

    let tp = TopicPartition::new("topic", 0);

    for i in 0..3 {
        let record = distributed_log_system::log::Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: i as u32,
            key: Some(format!("key{}", i).into_bytes()),
            value: Some(format!("value{}", i).into_bytes()),
            headers: vec![],
        };
        accumulator.append(tp.clone(), record);
    }

    let drained = accumulator.drain(&tp);
    assert_eq!(drained.len(), 3);
}

#[test]
fn test_record_accumulator_ready_partitions() {
    let config = ProducerConfig {
        batch_size: 20,
        ..ProducerConfig::default()
    };
    let mut accumulator = RecordAccumulator::new(&config);

    let tp0 = TopicPartition::new("topic", 0);
    let tp1 = TopicPartition::new("topic", 1);

    // Fill tp0
    for _ in 0..5 {
        let record = distributed_log_system::log::Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 0,
            key: Some(b"key".to_vec()),
            value: Some(b"value".to_vec()),
            headers: vec![],
        };
        accumulator.append(tp0.clone(), record);
    }

    // Add one to tp1 (not full)
    let record = distributed_log_system::log::Record {
        attributes: 0,
        timestamp_delta: 0,
        offset_delta: 0,
        key: Some(b"k".to_vec()),
        value: Some(b"v".to_vec()),
        headers: vec![],
    };
    accumulator.append(tp1.clone(), record);

    let ready = accumulator.ready_partitions();
    assert!(ready.contains(&tp0));
    assert!(!ready.contains(&tp1));
}

// =============================================================================
// Producer Tests
// =============================================================================

#[test]
fn test_producer_creation() {
    let config = ProducerConfig::default();
    let mut producer = Producer::new(config);

    assert!(producer.flush().is_empty());
}

#[test]
fn test_producer_send() {
    let config = ProducerConfig::default();
    let mut producer = Producer::new(config);

    let record = ProducerRecord::new("topic", Some(b"key".to_vec()), Some(b"value".to_vec()));

    let tp = producer.send(record).unwrap();
    assert_eq!(tp.topic, "topic");
}

#[test]
fn test_producer_send_with_partition() {
    let config = ProducerConfig::default();
    let mut producer = Producer::new(config);

    let record = ProducerRecord::new("topic", Some(b"key".to_vec()), Some(b"value".to_vec()))
        .with_partition(2);

    let tp = producer.send(record).unwrap();
    assert_eq!(tp.partition, 2);
}

#[test]
fn test_producer_build_request() {
    let config = ProducerConfig {
        batch_size: 20,
        ..ProducerConfig::default()
    };
    let mut producer = Producer::new(config);

    // Send enough records to trigger a ready batch
    for _ in 0..5 {
        let record = ProducerRecord::new("topic", Some(b"key".to_vec()), Some(b"value".to_vec()))
            .with_partition(0);
        producer.send(record).unwrap();
    }

    let request = producer.build_produce_request();
    assert!(request.is_some());
}

#[test]
fn test_producer_partitioning_by_key() {
    let config = ProducerConfig::default();
    let mut producer = Producer::new(config);

    // Same key should go to same partition
    let record1 = ProducerRecord::new("topic", Some(b"same-key".to_vec()), Some(b"v1".to_vec()));
    let record2 = ProducerRecord::new("topic", Some(b"same-key".to_vec()), Some(b"v2".to_vec()));

    let tp1 = producer.send(record1).unwrap();
    let tp2 = producer.send(record2).unwrap();

    assert_eq!(tp1.partition, tp2.partition);
}

// =============================================================================
// ConsumerConfig Tests
// =============================================================================

#[test]
fn test_consumer_config_default() {
    let config = ConsumerConfig::default();

    assert_eq!(config.client_id, "kafka-lite-consumer");
    assert!(config.enable_auto_commit);
    assert_eq!(config.session_timeout_ms, 30000);
    assert_eq!(config.max_poll_records, 500);
}

#[test]
fn test_consumer_config_custom() {
    let config = ConsumerConfig {
        group_id: "my-group".to_string(),
        client_id: "my-consumer".to_string(),
        enable_auto_commit: false,
        auto_commit_interval_ms: 10000,
        session_timeout_ms: 60000,
        heartbeat_interval_ms: 5000,
        max_poll_records: 1000,
        fetch_min_bytes: 1024,
        fetch_max_bytes: 104857600,
        fetch_max_wait_ms: 1000,
        auto_offset_reset: AutoOffsetReset::Earliest,
    };

    assert_eq!(config.group_id, "my-group");
    assert!(!config.enable_auto_commit);
}

// =============================================================================
// AutoOffsetReset Tests
// =============================================================================

#[test]
fn test_auto_offset_reset_earliest() {
    let reset = AutoOffsetReset::Earliest;
    assert!(matches!(reset, AutoOffsetReset::Earliest));
}

#[test]
fn test_auto_offset_reset_latest() {
    let reset = AutoOffsetReset::Latest;
    assert!(matches!(reset, AutoOffsetReset::Latest));
}

#[test]
fn test_auto_offset_reset_none() {
    let reset = AutoOffsetReset::None;
    assert!(matches!(reset, AutoOffsetReset::None));
}

// =============================================================================
// Consumer Tests
// =============================================================================

#[test]
fn test_consumer_creation() {
    let config = ConsumerConfig::default();
    let consumer = Consumer::new(config);

    assert!(consumer.subscription().is_empty());
    assert!(consumer.assignment().is_empty());
}

#[test]
fn test_consumer_subscribe() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    consumer
        .subscribe(vec!["topic1".to_string(), "topic2".to_string()])
        .unwrap();

    let subs = consumer.subscription();
    assert_eq!(subs.len(), 2);
    assert!(subs.contains(&"topic1".to_string()));
    assert!(subs.contains(&"topic2".to_string()));
}

#[test]
fn test_consumer_assign() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let partitions = vec![
        TopicPartition::new("topic", 0),
        TopicPartition::new("topic", 1),
    ];

    consumer.assign(partitions.clone()).unwrap();

    let assignment = consumer.assignment();
    assert_eq!(assignment.len(), 2);
}

#[test]
fn test_consumer_seek() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();

    consumer.seek(tp.clone(), 100);

    assert_eq!(consumer.position(&tp), Some(100));
}

#[test]
fn test_consumer_seek_to_beginning() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();

    consumer.seek(tp.clone(), 100);
    consumer.seek_to_beginning(&[tp.clone()]);

    assert_eq!(consumer.position(&tp), Some(0));
}

#[test]
fn test_consumer_seek_to_end() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();

    consumer.seek_to_end(&[tp.clone()]);

    assert_eq!(consumer.position(&tp), Some(u64::MAX));
}

#[test]
fn test_consumer_commit() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();
    consumer.seek(tp.clone(), 100);

    let committed = consumer.commit().unwrap();

    assert_eq!(committed.get(&tp), Some(&100));
    assert_eq!(consumer.committed(&tp), Some(100));
}

#[test]
fn test_consumer_commit_sync() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();

    let mut offsets = std::collections::HashMap::new();
    offsets.insert(tp.clone(), 500);

    consumer.commit_sync(offsets).unwrap();

    assert_eq!(consumer.committed(&tp), Some(500));
}

#[test]
fn test_consumer_pause_resume() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();

    consumer.pause(&[tp.clone()]);
    // Paused partitions would be tracked internally

    consumer.resume(&[tp.clone()]);

    let paused = consumer.paused();
    assert!(paused.is_empty());
}

#[test]
fn test_consumer_close() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    consumer
        .subscribe(vec!["topic".to_string()])
        .unwrap();
    consumer.assign(vec![TopicPartition::new("topic", 0)]).unwrap();

    consumer.close();

    assert!(consumer.subscription().is_empty());
    assert!(consumer.assignment().is_empty());
}

#[test]
fn test_consumer_build_fetch_request() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();
    consumer.seek(tp.clone(), 50);

    let request = consumer.build_fetch_request();

    assert_eq!(request.partitions.len(), 1);
    assert_eq!(request.partitions[0].fetch_offset, 50);
}

#[test]
fn test_consumer_update_position() {
    let config = ConsumerConfig::default();
    let mut consumer = Consumer::new(config);

    let tp = TopicPartition::new("topic", 0);
    consumer.assign(vec![tp.clone()]).unwrap();

    consumer.update_position(&tp, 200);

    assert_eq!(consumer.position(&tp), Some(200));
}

// =============================================================================
// ConsumerGroup Tests
// =============================================================================

#[test]
fn test_consumer_group() {
    let group = ConsumerGroup {
        group_id: "test-group".to_string(),
        generation_id: 1,
        member_id: "member-1".to_string(),
        leader: "member-1".to_string(),
        members: vec!["member-1".to_string(), "member-2".to_string()],
    };

    assert_eq!(group.group_id, "test-group");
    assert_eq!(group.generation_id, 1);
    assert_eq!(group.members.len(), 2);
}

// =============================================================================
// PartitionAssignor Tests
// =============================================================================

#[test]
fn test_range_assignor_single_member() {
    let assignor = RangeAssignor;
    let members = vec!["m1".to_string()];
    let partitions = vec![
        TopicPartition::new("t1", 0),
        TopicPartition::new("t1", 1),
        TopicPartition::new("t1", 2),
    ];

    let assignment = assignor.assign(&members, &partitions);

    assert_eq!(assignment.get("m1").unwrap().len(), 3);
}

#[test]
fn test_range_assignor_multiple_members() {
    let assignor = RangeAssignor;
    let members = vec!["m1".to_string(), "m2".to_string()];
    let partitions = vec![
        TopicPartition::new("t1", 0),
        TopicPartition::new("t1", 1),
        TopicPartition::new("t1", 2),
    ];

    let assignment = assignor.assign(&members, &partitions);

    // 3 partitions / 2 members = 2 for first, 1 for second
    assert_eq!(assignment.get("m1").unwrap().len(), 2);
    assert_eq!(assignment.get("m2").unwrap().len(), 1);
}

#[test]
fn test_range_assignor_even_distribution() {
    let assignor = RangeAssignor;
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

#[test]
fn test_round_robin_assignor_single_member() {
    let assignor = RoundRobinAssignor;
    let members = vec!["m1".to_string()];
    let partitions = vec![
        TopicPartition::new("t1", 0),
        TopicPartition::new("t1", 1),
        TopicPartition::new("t1", 2),
    ];

    let assignment = assignor.assign(&members, &partitions);

    assert_eq!(assignment.get("m1").unwrap().len(), 3);
}

#[test]
fn test_round_robin_assignor_multiple_members() {
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

#[test]
fn test_round_robin_assignor_uneven() {
    let assignor = RoundRobinAssignor;
    let members = vec!["m1".to_string(), "m2".to_string(), "m3".to_string()];
    let partitions = vec![
        TopicPartition::new("t1", 0),
        TopicPartition::new("t1", 1),
        TopicPartition::new("t1", 2),
        TopicPartition::new("t1", 3),
        TopicPartition::new("t1", 4),
    ];

    let assignment = assignor.assign(&members, &partitions);

    let total: usize = assignment.values().map(|v| v.len()).sum();
    assert_eq!(total, 5);
}

#[test]
fn test_round_robin_assignor_many_partitions() {
    let assignor = RoundRobinAssignor;
    let members = vec!["m1".to_string(), "m2".to_string()];
    let partitions: Vec<TopicPartition> = (0..100)
        .map(|i| TopicPartition::new("t1", i))
        .collect();

    let assignment = assignor.assign(&members, &partitions);

    assert_eq!(assignment.get("m1").unwrap().len(), 50);
    assert_eq!(assignment.get("m2").unwrap().len(), 50);
}

#[test]
fn test_range_assignor_multiple_topics() {
    let assignor = RangeAssignor;
    let members = vec!["m1".to_string(), "m2".to_string()];
    let partitions = vec![
        TopicPartition::new("t1", 0),
        TopicPartition::new("t1", 1),
        TopicPartition::new("t2", 0),
        TopicPartition::new("t2", 1),
    ];

    let assignment = assignor.assign(&members, &partitions);

    // Each topic's partitions are assigned separately
    let m1_partitions = assignment.get("m1").unwrap();
    let m2_partitions = assignment.get("m2").unwrap();

    assert_eq!(m1_partitions.len() + m2_partitions.len(), 4);
}

// =============================================================================
// ConsumerRecord Tests
// =============================================================================

#[test]
fn test_consumer_record() {
    let record = ConsumerRecord {
        topic: "test-topic".to_string(),
        partition: 0,
        offset: 100,
        timestamp: 1234567890,
        key: Some(b"key".to_vec()),
        value: Some(b"value".to_vec()),
        headers: vec![],
    };

    assert_eq!(record.topic, "test-topic");
    assert_eq!(record.partition, 0);
    assert_eq!(record.offset, 100);
}

#[test]
fn test_consumer_record_with_headers() {
    let record = ConsumerRecord {
        topic: "test-topic".to_string(),
        partition: 0,
        offset: 100,
        timestamp: 1234567890,
        key: Some(b"key".to_vec()),
        value: Some(b"value".to_vec()),
        headers: vec![
            distributed_log_system::log::Header {
                key: "header1".to_string(),
                value: b"value1".to_vec(),
            },
        ],
    };

    assert_eq!(record.headers.len(), 1);
}
