//! Comprehensive integration tests for the message queue.

use message_queue::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

// Helper functions

fn create_test_dir() -> TempDir {
    TempDir::new().unwrap()
}

fn create_test_message(content: &str) -> Message {
    Message::new(content.as_bytes().to_vec())
}

fn create_keyed_message(key: &str, value: &str) -> Message {
    Message::with_key(key.as_bytes().to_vec(), value.as_bytes().to_vec())
}

// ============================================================================
// Message Tests
// ============================================================================

#[test]
fn test_message_id_uniqueness() {
    let ids: Vec<_> = (0..1000).map(|_| MessageId::new()).collect();
    let unique: std::collections::HashSet<_> = ids.iter().collect();
    assert_eq!(ids.len(), unique.len());
}

#[test]
fn test_message_id_roundtrip() {
    let id = MessageId::new();
    let bytes = id.as_bytes();
    let restored = MessageId::from_bytes(bytes).unwrap();
    assert_eq!(id, restored);
}

#[test]
fn test_message_id_string_parsing() {
    let id = MessageId::new();
    let s = id.to_string();
    let parsed: MessageId = s.parse().unwrap();
    assert_eq!(id, parsed);
}

#[test]
fn test_headers_operations() {
    let mut headers = Headers::new();
    headers.insert("key1", vec![1, 2, 3]);
    headers.insert_string("key2", "value");

    assert_eq!(headers.get("key1"), Some([1, 2, 3].as_slice()));
    assert_eq!(headers.get_string("key2"), Some("value"));
    assert!(headers.contains("key1"));
    assert!(!headers.contains("nonexistent"));

    headers.remove("key1");
    assert!(!headers.contains("key1"));
}

#[test]
fn test_message_serialization_roundtrip() {
    let msg = MessageBuilder::new("test payload")
        .key("test-key")
        .header("h1", vec![1, 2, 3])
        .string_header("h2", "header-value")
        .timestamp(12345678)
        .build();

    let serialized = msg.serialize().unwrap();
    let deserialized = Message::deserialize(&serialized).unwrap();

    assert_eq!(msg.id, deserialized.id);
    assert_eq!(msg.key, deserialized.key);
    assert_eq!(msg.payload, deserialized.payload);
    assert_eq!(msg.timestamp, deserialized.timestamp);
    assert_eq!(msg.headers.len(), deserialized.headers.len());
}

#[test]
fn test_message_crc_validation() {
    let msg = create_test_message("Hello");
    let mut serialized = msg.serialize().unwrap().to_vec();

    // Corrupt the data
    serialized[20] ^= 0xFF;

    let result = Message::deserialize(&serialized);
    assert!(result.is_err());
}

#[test]
fn test_message_batch_operations() {
    let mut batch = MessageBatch::new();
    assert!(batch.is_empty());

    batch.push(create_test_message("msg1"));
    batch.push(create_test_message("msg2"));
    batch.push(create_test_message("msg3"));

    assert_eq!(batch.len(), 3);
    assert!(!batch.is_empty());
    assert!(batch.size() > 0);
}

#[test]
fn test_message_batch_serialization() {
    let batch = MessageBatch::with_messages(vec![
        create_test_message("msg1"),
        create_test_message("msg2"),
    ])
    .with_compression(Compression::None)
    .with_base_offset(100)
    .with_partition(2);

    let serialized = batch.serialize().unwrap();
    let deserialized = MessageBatch::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.len(), 2);
    assert_eq!(deserialized.base_offset, 100);
    assert_eq!(deserialized.partition, 2);
}

#[test]
fn test_message_batch_with_lz4_compression() {
    let batch = MessageBatch::with_messages(vec![
        create_test_message("This is a longer message for compression testing"),
        create_test_message("Another message with similar content"),
    ])
    .with_compression(Compression::Lz4);

    let serialized = batch.serialize().unwrap();
    let deserialized = MessageBatch::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.len(), 2);
    assert_eq!(deserialized.compression, Compression::Lz4);
}

#[test]
fn test_message_batch_with_gzip_compression() {
    let batch = MessageBatch::with_messages(vec![
        create_test_message("Message 1"),
        create_test_message("Message 2"),
    ])
    .with_compression(Compression::Gzip);

    let serialized = batch.serialize().unwrap();
    let deserialized = MessageBatch::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.len(), 2);
}

#[test]
fn test_message_batch_with_snappy_compression() {
    let batch = MessageBatch::with_messages(vec![
        create_test_message("Snappy compressed message"),
    ])
    .with_compression(Compression::Snappy);

    let serialized = batch.serialize().unwrap();
    let deserialized = MessageBatch::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.len(), 1);
}

// ============================================================================
// Compression Tests
// ============================================================================

#[test]
fn test_compression_types() {
    let data = b"Hello, World! This is a test.";

    for compression in [
        Compression::None,
        Compression::Gzip,
        Compression::Lz4,
        Compression::Snappy,
    ] {
        let compressed = compression.compress(data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_ref(), data);
    }
}

#[test]
fn test_compression_large_data() {
    let data: Vec<u8> = (0..100000).map(|i| (i % 256) as u8).collect();

    for compression in [Compression::Gzip, Compression::Lz4, Compression::Snappy] {
        let compressed = compression.compress(&data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_ref(), data.as_slice());
    }
}

#[test]
fn test_compression_empty_data() {
    let data = b"";

    for compression in [
        Compression::None,
        Compression::Gzip,
        Compression::Lz4,
        Compression::Snappy,
    ] {
        let compressed = compression.compress(data).unwrap();
        let decompressed = compression.decompress(&compressed).unwrap();
        assert_eq!(decompressed.as_ref(), data);
    }
}

#[test]
fn test_compression_from_str() {
    assert_eq!("none".parse::<Compression>().unwrap(), Compression::None);
    assert_eq!("gzip".parse::<Compression>().unwrap(), Compression::Gzip);
    assert_eq!("lz4".parse::<Compression>().unwrap(), Compression::Lz4);
    assert_eq!("snappy".parse::<Compression>().unwrap(), Compression::Snappy);
    assert_eq!("GZIP".parse::<Compression>().unwrap(), Compression::Gzip);
}

#[test]
fn test_compression_stats() {
    let mut stats = compression::CompressionStats::new();
    stats.record_compression(1000, 500);
    stats.record_compression(2000, 1000);

    assert_eq!(stats.uncompressed_bytes, 3000);
    assert_eq!(stats.compressed_bytes, 1500);
    assert!((stats.compression_ratio() - 0.5).abs() < 0.01);
    assert!((stats.space_savings() - 50.0).abs() < 1.0);
}

// ============================================================================
// Storage Tests
// ============================================================================

#[test]
fn test_storage_create_and_append() {
    let dir = create_test_dir();
    let config = StorageConfig::new().with_base_dir(dir.path());
    let storage = Storage::new(config).unwrap();

    let msg = create_test_message("Hello, World!");
    let offset = storage.append(&msg).unwrap();

    assert_eq!(offset, 0);
    assert_eq!(storage.end_offset(), 1);
}

#[test]
fn test_storage_read_after_write() {
    let dir = create_test_dir();
    let config = StorageConfig::new().with_base_dir(dir.path());
    let storage = Storage::new(config).unwrap();

    let msg = create_test_message("Test message");
    let offset = storage.append(&msg).unwrap();

    let read_msg = storage.read(offset).unwrap();
    assert_eq!(read_msg.payload, msg.payload);
}

#[test]
fn test_storage_batch_append() {
    let dir = create_test_dir();
    let config = StorageConfig::new().with_base_dir(dir.path());
    let storage = Storage::new(config).unwrap();

    let messages: Vec<_> = (0..10)
        .map(|i| create_test_message(&format!("Message {}", i)))
        .collect();

    let offsets = storage.append_batch(&messages).unwrap();

    assert_eq!(offsets.len(), 10);
    assert_eq!(offsets, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
}

#[test]
fn test_storage_read_range() {
    let dir = create_test_dir();
    let config = StorageConfig::new().with_base_dir(dir.path());
    let storage = Storage::new(config).unwrap();

    for i in 0..20 {
        storage.append(&create_test_message(&format!("Msg {}", i))).unwrap();
    }

    let messages = storage.read_range(5, 10).unwrap();
    assert_eq!(messages.len(), 10);

    for (i, msg) in messages.iter().enumerate() {
        let expected = format!("Msg {}", i + 5);
        assert_eq!(msg.payload.as_ref(), expected.as_bytes());
    }
}

#[test]
fn test_storage_persistence() {
    let dir = create_test_dir();

    // Write data
    {
        let config = StorageConfig::new().with_base_dir(dir.path());
        let storage = Storage::new(config).unwrap();
        storage.append(&create_test_message("Persistent")).unwrap();
        storage.flush().unwrap();
    }

    // Read back
    {
        let config = StorageConfig::new().with_base_dir(dir.path());
        let storage = Storage::open(config).unwrap();
        let msg = storage.read(0).unwrap();
        assert_eq!(msg.payload.as_ref(), b"Persistent");
    }
}

#[test]
fn test_storage_stats() {
    let dir = create_test_dir();
    let config = StorageConfig::new().with_base_dir(dir.path());
    let storage = Storage::new(config).unwrap();

    storage.append(&create_test_message("Test")).unwrap();
    storage.read(0).unwrap();

    let stats = storage.stats();
    assert!(stats.bytes_written > 0);
    assert!(stats.messages_written == 1);
    assert!(stats.bytes_read > 0);
    assert!(stats.messages_read == 1);
}

// ============================================================================
// Topic Tests
// ============================================================================

#[test]
fn test_topic_creation() {
    let dir = create_test_dir();
    let config = TopicConfig::new().with_partitions(4);
    let topic = Topic::new("test-topic", config, dir.path()).unwrap();

    assert_eq!(topic.name(), "test-topic");
    assert_eq!(topic.partition_count(), 4);
}

#[test]
fn test_topic_append_with_partitioning() {
    let dir = create_test_dir();
    let config = TopicConfig::new().with_partitions(4);
    let topic = Topic::new("test-topic", config, dir.path()).unwrap();

    // Messages with same key should go to same partition
    let msg1 = create_keyed_message("key1", "value1");
    let msg2 = create_keyed_message("key1", "value2");

    let (p1, _) = topic.append(msg1).unwrap();
    let (p2, _) = topic.append(msg2).unwrap();

    assert_eq!(p1, p2);
}

#[test]
fn test_topic_fetch() {
    let dir = create_test_dir();
    let config = TopicConfig::new().with_partitions(1);
    let topic = Topic::new("test-topic", config, dir.path()).unwrap();

    for i in 0..10 {
        topic.append_to_partition(0, create_test_message(&format!("Msg {}", i))).unwrap();
    }

    let result = topic.fetch(0, 0, 5, 0).unwrap();
    assert_eq!(result.len(), 5);
}

#[test]
fn test_topic_log_end_offsets() {
    let dir = create_test_dir();
    let config = TopicConfig::new().with_partitions(2);
    let topic = Topic::new("test-topic", config, dir.path()).unwrap();

    topic.append_to_partition(0, create_test_message("P0-1")).unwrap();
    topic.append_to_partition(0, create_test_message("P0-2")).unwrap();
    topic.append_to_partition(1, create_test_message("P1-1")).unwrap();

    let offsets = topic.log_end_offsets();
    assert_eq!(offsets.get(&0), Some(&2));
    assert_eq!(offsets.get(&1), Some(&1));
}

#[test]
fn test_topic_persistence() {
    let dir = create_test_dir();
    let config = TopicConfig::new().with_partitions(2);

    {
        let topic = Topic::new("test-topic", config.clone(), dir.path()).unwrap();
        topic.append_to_partition(0, create_test_message("Test")).unwrap();
        topic.flush().unwrap();
    }

    {
        let topic = Topic::open("test-topic", dir.path(), config).unwrap();
        let msg = topic.read(0, 0).unwrap();
        assert_eq!(msg.payload.as_ref(), b"Test");
    }
}

// ============================================================================
// Topic Manager Tests
// ============================================================================

#[test]
fn test_topic_manager_create_topic() {
    let dir = create_test_dir();
    let manager = TopicManager::new(dir.path(), TopicConfig::default()).unwrap();

    let topic = manager.create_topic("topic1", None).unwrap();
    assert_eq!(topic.name(), "topic1");
    assert!(manager.topic_exists("topic1"));
}

#[test]
fn test_topic_manager_delete_topic() {
    let dir = create_test_dir();
    let manager = TopicManager::new(dir.path(), TopicConfig::default()).unwrap();

    manager.create_topic("topic1", None).unwrap();
    manager.delete_topic("topic1").unwrap();
    assert!(!manager.topic_exists("topic1"));
}

#[test]
fn test_topic_manager_list_topics() {
    let dir = create_test_dir();
    let manager = TopicManager::new(dir.path(), TopicConfig::default()).unwrap();

    manager.create_topic("topic1", None).unwrap();
    manager.create_topic("topic2", None).unwrap();
    manager.create_topic("topic3", None).unwrap();

    let topics = manager.list_topics();
    assert_eq!(topics.len(), 3);
}

#[test]
fn test_topic_manager_get_or_create() {
    let dir = create_test_dir();
    let manager = TopicManager::new(dir.path(), TopicConfig::default()).unwrap();

    let t1 = manager.get_or_create_topic("topic1").unwrap();
    let t2 = manager.get_or_create_topic("topic1").unwrap();

    assert!(Arc::ptr_eq(&t1, &t2));
}

#[test]
fn test_topic_manager_topic_already_exists() {
    let dir = create_test_dir();
    let manager = TopicManager::new(dir.path(), TopicConfig::default()).unwrap();

    manager.create_topic("topic1", None).unwrap();
    let result = manager.create_topic("topic1", None);

    assert!(result.is_err());
}

// ============================================================================
// Producer Tests
// ============================================================================

#[test]
fn test_producer_send() {
    let dir = create_test_dir();
    let topic_manager = Arc::new(
        TopicManager::new(dir.path(), TopicConfig::default()).unwrap()
    );
    let producer = Producer::new(ProducerConfig::default(), topic_manager);

    let result = producer.send("test-topic", create_test_message("Hello")).unwrap();
    assert_eq!(result.topic, "test-topic");
}

#[test]
fn test_producer_send_with_key() {
    let dir = create_test_dir();
    let topic_manager = Arc::new(
        TopicManager::new(dir.path(), TopicConfig::default()).unwrap()
    );
    let producer = Producer::new(ProducerConfig::default(), topic_manager);

    let msg1 = create_keyed_message("key1", "value1");
    let msg2 = create_keyed_message("key1", "value2");

    let r1 = producer.send("test-topic", msg1).unwrap();
    let r2 = producer.send("test-topic", msg2).unwrap();

    assert_eq!(r1.partition, r2.partition);
}

#[test]
fn test_producer_send_batch() {
    let dir = create_test_dir();
    let topic_manager = Arc::new(
        TopicManager::new(dir.path(), TopicConfig::default()).unwrap()
    );
    let producer = Producer::new(ProducerConfig::default(), topic_manager);

    let messages: Vec<_> = (0..10)
        .map(|i| create_test_message(&format!("Msg {}", i)))
        .collect();

    let results = producer.send_batch("test-topic", messages).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn test_producer_metrics() {
    let dir = create_test_dir();
    let topic_manager = Arc::new(
        TopicManager::new(dir.path(), TopicConfig::default()).unwrap()
    );
    let producer = Producer::new(ProducerConfig::default(), topic_manager);

    producer.send("test-topic", create_test_message("Test")).unwrap();
    producer.send("test-topic", create_test_message("Test")).unwrap();

    let metrics = producer.metrics();
    assert_eq!(metrics.records_sent(), 2);
}

#[test]
fn test_producer_close() {
    let dir = create_test_dir();
    let topic_manager = Arc::new(
        TopicManager::new(dir.path(), TopicConfig::default()).unwrap()
    );
    let producer = Producer::new(ProducerConfig::default(), topic_manager);

    producer.close().unwrap();

    let result = producer.send("test-topic", create_test_message("Test"));
    assert!(result.is_err());
}

// ============================================================================
// Consumer Tests
// ============================================================================

fn setup_consumer_test() -> (TempDir, Arc<TopicManager>) {
    let dir = create_test_dir();
    let manager = Arc::new(
        TopicManager::new(dir.path(), TopicConfig::default()).unwrap()
    );

    // Create topic and add messages
    let topic = manager.create_topic("test-topic", None).unwrap();
    for i in 0..20 {
        topic.append_to_partition(0, create_test_message(&format!("Msg {}", i))).unwrap();
    }

    (dir, manager)
}

#[test]
fn test_consumer_subscribe() {
    let (_dir, manager) = setup_consumer_test();

    let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();
    consumer.subscribe(&["test-topic"]).unwrap();

    assert!(consumer.subscription().contains(&"test-topic".to_string()));
    assert!(!consumer.assignment().is_empty());
}

#[test]
fn test_consumer_poll() {
    let (_dir, manager) = setup_consumer_test();

    let config = ConsumerConfig::new()
        .with_offset_reset(config::OffsetReset::Earliest)
        .with_max_poll_records(10);
    let consumer = Consumer::new(config, manager).unwrap();
    consumer.subscribe(&["test-topic"]).unwrap();

    let messages = consumer.poll(Duration::from_secs(1)).unwrap();
    assert!(!messages.is_empty());
    assert!(messages.len() <= 10);
}

#[test]
fn test_consumer_seek() {
    let (_dir, manager) = setup_consumer_test();

    let config = ConsumerConfig::new().with_offset_reset(config::OffsetReset::Earliest);
    let consumer = Consumer::new(config, manager).unwrap();
    consumer.subscribe(&["test-topic"]).unwrap();

    let tp = offset::TopicPartition::new("test-topic", 0);
    consumer.seek(&tp, 10).unwrap();

    assert_eq!(consumer.position(&tp), Some(10));
}

#[test]
fn test_consumer_commit() {
    let dir = create_test_dir();
    let manager = Arc::new(
        TopicManager::new(dir.path(), TopicConfig::default()).unwrap()
    );

    let topic = manager.create_topic("test-topic", None).unwrap();
    topic.append_to_partition(0, create_test_message("Test")).unwrap();

    let config = ConsumerConfig::new()
        .with_group_id("test-group")
        .with_offset_reset(config::OffsetReset::Earliest);
    let consumer = Consumer::new(config, manager).unwrap();
    consumer.subscribe(&["test-topic"]).unwrap();
    consumer.poll(Duration::from_secs(1)).unwrap();
    consumer.commit().unwrap();

    let tp = offset::TopicPartition::new("test-topic", 0);
    assert!(consumer.committed(&tp).is_some());
}

#[test]
fn test_consumer_pause_resume() {
    let (_dir, manager) = setup_consumer_test();

    let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();
    consumer.subscribe(&["test-topic"]).unwrap();

    let tp = offset::TopicPartition::new("test-topic", 0);

    consumer.pause(&[tp.clone()]);
    assert!(consumer.paused().contains(&tp));

    consumer.resume(&[tp.clone()]);
    assert!(!consumer.paused().contains(&tp));
}

#[test]
fn test_consumer_unsubscribe() {
    let (_dir, manager) = setup_consumer_test();

    let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();
    consumer.subscribe(&["test-topic"]).unwrap();
    consumer.unsubscribe();

    assert!(consumer.subscription().is_empty());
    assert!(consumer.assignment().is_empty());
}

#[test]
fn test_consumer_close() {
    let (_dir, manager) = setup_consumer_test();

    let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();
    consumer.subscribe(&["test-topic"]).unwrap();
    consumer.close().unwrap();

    let result = consumer.poll(Duration::from_millis(100));
    assert!(result.is_err());
}

// ============================================================================
// Consumer Group Tests
// ============================================================================

#[test]
fn test_consumer_group_join() {
    let group = consumer_group::ConsumerGroup::new(
        "test-group",
        consumer_group::GroupConfig::default(),
    );

    let response = group
        .join(None, "client-1".to_string(), vec!["topic1".to_string()])
        .unwrap();

    assert!(!response.member_id.is_empty());
    assert!(response.is_leader);
    assert_eq!(group.member_count(), 1);
}

#[test]
fn test_consumer_group_multiple_members() {
    let group = consumer_group::ConsumerGroup::new(
        "test-group",
        consumer_group::GroupConfig::default(),
    );

    let r1 = group
        .join(None, "client-1".to_string(), vec!["topic1".to_string()])
        .unwrap();

    let r2 = group
        .join(None, "client-2".to_string(), vec!["topic1".to_string()])
        .unwrap();

    assert!(r1.is_leader);
    assert!(!r2.is_leader);
    assert_eq!(group.member_count(), 2);
}

#[test]
fn test_consumer_group_leave() {
    let group = consumer_group::ConsumerGroup::new(
        "test-group",
        consumer_group::GroupConfig::default(),
    );

    let response = group
        .join(Some("member-1".to_string()), "client-1".to_string(), vec!["topic1".to_string()])
        .unwrap();

    group.leave(&response.member_id).unwrap();
    assert_eq!(group.member_count(), 0);
}

#[test]
fn test_consumer_group_assignment() {
    let group = consumer_group::ConsumerGroup::new(
        "test-group",
        consumer_group::GroupConfig::default(),
    );

    group.join(Some("m1".to_string()), "c1".to_string(), vec!["t1".to_string()]).unwrap();
    group.join(Some("m2".to_string()), "c2".to_string(), vec!["t1".to_string()]).unwrap();

    let mut topic_partitions = HashMap::new();
    topic_partitions.insert("t1".to_string(), vec![0, 1, 2, 3]);

    let assignments = group.assign(&topic_partitions);

    let total: usize = assignments.values().map(|v| v.len()).sum();
    assert_eq!(total, 4);
}

// ============================================================================
// Broker Tests
// ============================================================================

#[test]
fn test_broker_create() {
    let dir = create_test_dir();
    let config = BrokerConfig::default()
        .with_data_dir(dir.path().join("data"))
        .with_log_dir(dir.path().join("logs"));

    let broker = Broker::new(config).unwrap();
    assert!(!broker.is_running());
}

#[test]
fn test_broker_start_stop() {
    let dir = create_test_dir();
    let config = BrokerConfig::default()
        .with_data_dir(dir.path().join("data"))
        .with_log_dir(dir.path().join("logs"));

    let broker = Broker::new(config).unwrap();
    broker.start().unwrap();
    assert!(broker.is_running());

    broker.stop().unwrap();
    assert!(!broker.is_running());
}

#[test]
fn test_broker_produce_consume() {
    let dir = create_test_dir();
    let config = BrokerConfig::default()
        .with_data_dir(dir.path().join("data"))
        .with_log_dir(dir.path().join("logs"));

    let broker = Broker::new(config).unwrap();
    broker.start().unwrap();

    let msg = create_test_message("Hello, Broker!");
    let (partition, offset) = broker.produce("test-topic", msg).unwrap();

    let messages = broker.fetch("test-topic", partition, offset, 10, 0).unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].payload.as_ref(), b"Hello, Broker!");
}

#[test]
fn test_broker_topic_operations() {
    let dir = create_test_dir();
    let config = BrokerConfig::default()
        .with_data_dir(dir.path().join("data"))
        .with_log_dir(dir.path().join("logs"));

    let broker = Broker::new(config).unwrap();
    broker.start().unwrap();

    broker.create_topic("topic1", None).unwrap();
    broker.create_topic("topic2", None).unwrap();

    assert_eq!(broker.list_topics().len(), 2);

    broker.delete_topic("topic1").unwrap();
    assert_eq!(broker.list_topics().len(), 1);
}

#[test]
fn test_broker_metrics() {
    let dir = create_test_dir();
    let config = BrokerConfig::default()
        .with_data_dir(dir.path().join("data"))
        .with_log_dir(dir.path().join("logs"));

    let broker = Broker::new(config).unwrap();
    broker.start().unwrap();

    broker.produce("test-topic", create_test_message("Test")).unwrap();

    let metrics = broker.metrics();
    assert_eq!(metrics.messages_produced(), 1);
    assert_eq!(metrics.topics_created(), 1);
}

// ============================================================================
// Offset Tests
// ============================================================================

#[test]
fn test_offset_store() {
    let dir = create_test_dir();
    let store = offset::OffsetStore::new("test-group", dir.path()).unwrap();

    let tp = offset::TopicPartition::new("topic1", 0);

    store.commit(tp.clone(), 100).unwrap();
    assert_eq!(store.get_offset(&tp), Some(100));

    store.commit(tp.clone(), 200).unwrap();
    assert_eq!(store.get_offset(&tp), Some(200));
}

#[test]
fn test_offset_store_persistence() {
    let dir = create_test_dir();
    let tp = offset::TopicPartition::new("topic1", 0);

    {
        let store = offset::OffsetStore::new("test-group", dir.path()).unwrap();
        store.commit(tp.clone(), 100).unwrap();
        store.flush().unwrap();
    }

    {
        let store = offset::OffsetStore::new("test-group", dir.path()).unwrap();
        assert_eq!(store.get_offset(&tp), Some(100));
    }
}

#[test]
fn test_offset_tracker() {
    let tracker = offset::OffsetTracker::new();
    let tp = offset::TopicPartition::new("topic1", 0);

    tracker.set_position(tp.clone(), 100);
    tracker.update_high_watermark(tp.clone(), 200);

    assert_eq!(tracker.position(&tp), Some(100));
    assert_eq!(tracker.high_watermark(&tp), Some(200));
    assert_eq!(tracker.lag(&tp), Some(100));
}

// ============================================================================
// Index Tests
// ============================================================================

#[test]
fn test_memory_index() {
    let index = index::MemoryIndex::new();

    index.insert(0, 0);
    index.insert(100, 1000);
    index.insert(200, 2000);

    assert_eq!(index.get(100), Some(1000));
    assert_eq!(index.lookup(150), Some(1000));
    assert_eq!(index.lookup(250), Some(2000));
}

#[test]
fn test_memory_index_truncate() {
    let index = index::MemoryIndex::new();

    index.insert(0, 0);
    index.insert(100, 1000);
    index.insert(200, 2000);

    index.truncate_to(100);
    assert!(index.get(200).is_none());
    assert_eq!(index.get(100), Some(1000));
}

// ============================================================================
// Segment Tests
// ============================================================================

#[test]
fn test_segment_create_append_read() {
    let dir = create_test_dir();
    let segment = segment::Segment::new(dir.path(), 0, 1024 * 1024).unwrap();

    let msg = create_test_message("Test");
    let offset = segment.append(&msg).unwrap();

    assert_eq!(offset, 0);

    let read_msg = segment.read(0).unwrap();
    assert_eq!(read_msg.payload, msg.payload);
}

#[test]
fn test_segment_read_range() {
    let dir = create_test_dir();
    let segment = segment::Segment::new(dir.path(), 0, 1024 * 1024).unwrap();

    for i in 0..10 {
        segment.append(&create_test_message(&format!("Msg {}", i))).unwrap();
    }

    let messages = segment.read_range(3, 5).unwrap();
    assert_eq!(messages.len(), 5);
}

#[test]
fn test_segment_manager() {
    let dir = create_test_dir();
    let manager = segment::SegmentManager::new(dir.path(), 1024 * 1024).unwrap();

    for i in 0..10 {
        manager.append(&create_test_message(&format!("Msg {}", i))).unwrap();
    }

    assert_eq!(manager.end_offset(), 10);

    let messages = manager.read_range(0, 10).unwrap();
    assert_eq!(messages.len(), 10);
}

// ============================================================================
// Partition Tests
// ============================================================================

#[test]
fn test_partition_create() {
    let dir = create_test_dir();
    let config = TopicConfig::default();
    let partition = partition::Partition::new("test-topic", 0, dir.path(), &config).unwrap();

    assert_eq!(partition.topic(), "test-topic");
    assert_eq!(partition.id(), 0);
    assert_eq!(partition.log_end_offset(), 0);
}

#[test]
fn test_partition_append_read() {
    let dir = create_test_dir();
    let config = TopicConfig::default();
    let partition = partition::Partition::new("test-topic", 0, dir.path(), &config).unwrap();

    let msg = create_test_message("Test");
    let offset = partition.append(msg).unwrap();

    let read_msg = partition.read(offset).unwrap();
    assert_eq!(read_msg.payload.as_ref(), b"Test");
}

#[test]
fn test_partitioner_round_robin() {
    let partitioner = partition::Partitioner::new(
        partition::PartitionStrategy::RoundRobin,
        4,
    );

    let msg = create_test_message("test");

    let p1 = partitioner.partition(&msg);
    let p2 = partitioner.partition(&msg);
    let p3 = partitioner.partition(&msg);
    let p4 = partitioner.partition(&msg);
    let p5 = partitioner.partition(&msg);

    assert_eq!(vec![p1, p2, p3, p4, p5], vec![0, 1, 2, 3, 0]);
}

#[test]
fn test_partitioner_key_hash() {
    let partitioner = partition::Partitioner::new(
        partition::PartitionStrategy::KeyHash,
        4,
    );

    let msg1 = create_keyed_message("key1", "value1");
    let msg2 = create_keyed_message("key1", "value2");

    let p1 = partitioner.partition(&msg1);
    let p2 = partitioner.partition(&msg2);

    assert_eq!(p1, p2);
}

// ============================================================================
// Config Tests
// ============================================================================

#[test]
fn test_broker_config() {
    let config = BrokerConfig::default()
        .with_broker_id(5)
        .with_port(9093)
        .with_default_partitions(8);

    assert_eq!(config.broker_id, 5);
    assert_eq!(config.port, 9093);
    assert_eq!(config.default_partitions, 8);
}

#[test]
fn test_topic_config() {
    let config = TopicConfig::new()
        .with_partitions(16)
        .with_replication(3)
        .with_compression(Compression::Lz4);

    assert_eq!(config.partition_count, 16);
    assert_eq!(config.replication_factor, 3);
    assert_eq!(config.compression, Compression::Lz4);
}

#[test]
fn test_producer_config() {
    let config = ProducerConfig::new()
        .with_client_id("test-producer")
        .with_acks(config::Acks::All)
        .with_compression(Compression::Gzip);

    assert_eq!(config.client_id, "test-producer");
    assert_eq!(config.acks, config::Acks::All);
    assert_eq!(config.compression, Compression::Gzip);
}

#[test]
fn test_consumer_config() {
    let config = ConsumerConfig::new()
        .with_client_id("test-consumer")
        .with_group_id("test-group")
        .with_auto_commit(false);

    assert_eq!(config.client_id, "test-consumer");
    assert_eq!(config.group_id, Some("test-group".to_string()));
    assert!(!config.enable_auto_commit);
}

// ============================================================================
// Error Tests
// ============================================================================

#[test]
fn test_error_display() {
    let err = Error::TopicNotFound("test".to_string());
    assert_eq!(err.to_string(), "Topic not found: test");

    let err = Error::PartitionNotFound {
        topic: "test".to_string(),
        partition: 5,
    };
    assert!(err.to_string().contains("test"));
    assert!(err.to_string().contains("5"));
}

#[test]
fn test_error_retriable() {
    assert!(Error::Timeout(Duration::from_secs(1)).is_retriable());
    assert!(Error::RebalanceInProgress("group".to_string()).is_retriable());
    assert!(!Error::TopicNotFound("test".to_string()).is_retriable());
}

#[test]
fn test_error_fatal() {
    assert!(Error::SegmentCorrupted("test".to_string()).is_fatal());
    assert!(Error::CrcMismatch { expected: 1, actual: 2 }.is_fatal());
    assert!(!Error::TopicNotFound("test".to_string()).is_fatal());
}

// ============================================================================
// End-to-End Tests
// ============================================================================

#[test]
fn test_end_to_end_produce_consume() {
    let dir = create_test_dir();
    let topic_manager = Arc::new(
        TopicManager::new(dir.path(), TopicConfig::default()).unwrap()
    );

    // Create producer
    let producer = Producer::new(ProducerConfig::default(), Arc::clone(&topic_manager));

    // Produce messages
    for i in 0..100 {
        producer.send("e2e-topic", create_test_message(&format!("Message {}", i))).unwrap();
    }

    // Create consumer
    let config = ConsumerConfig::new()
        .with_offset_reset(config::OffsetReset::Earliest)
        .with_max_poll_records(50);
    let consumer = Consumer::new(config, topic_manager).unwrap();
    consumer.subscribe(&["e2e-topic"]).unwrap();

    // Consume messages
    let mut total = 0;
    for _ in 0..3 {
        let messages = consumer.poll(Duration::from_secs(1)).unwrap();
        total += messages.len();
        if total >= 100 {
            break;
        }
    }

    assert_eq!(total, 100);
}

#[test]
fn test_end_to_end_with_multiple_partitions() {
    let dir = create_test_dir();
    let topic_config = TopicConfig::new().with_partitions(4);
    let topic_manager = Arc::new(
        TopicManager::new(dir.path(), topic_config).unwrap()
    );

    // Create producer
    let producer = Producer::new(ProducerConfig::default(), Arc::clone(&topic_manager));

    // Produce keyed messages
    for i in 0..100 {
        let key = format!("key{}", i % 10);
        let msg = create_keyed_message(&key, &format!("value{}", i));
        producer.send("multi-partition", msg).unwrap();
    }

    // Verify distribution across partitions
    let topic = topic_manager.get_topic("multi-partition").unwrap();
    let offsets = topic.log_end_offsets();

    let total: u64 = offsets.values().sum();
    assert_eq!(total, 100);
}

#[test]
fn test_end_to_end_with_broker() {
    let dir = create_test_dir();
    let config = BrokerConfig::default()
        .with_data_dir(dir.path().join("data"))
        .with_log_dir(dir.path().join("logs"));

    let broker = Broker::new(config).unwrap();
    broker.start().unwrap();

    // Produce
    for i in 0..50 {
        broker.produce("broker-topic", create_test_message(&format!("Msg {}", i))).unwrap();
    }

    // Get offsets
    let offsets = broker.get_offsets("broker-topic").unwrap();
    let total: u64 = offsets.values().map(|(_, end)| *end).sum();
    assert_eq!(total, 50);

    // Consume from each partition
    let mut consumed = 0;
    for (partition, (start, end)) in &offsets {
        let messages = broker.fetch("broker-topic", *partition, *start, (*end - *start) as usize, 0).unwrap();
        consumed += messages.len();
    }
    assert_eq!(consumed, 50);

    broker.stop().unwrap();
}
