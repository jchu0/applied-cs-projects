//! Performance benchmarks for the Distributed Log System.
//!
//! Run benchmarks with: `cargo bench`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use distributed_log_system::{
    Partition, Producer, ProducerConfig, ProducerRecord, RecordBatch, SegmentConfig, TopicPartition,
};
use std::time::Duration;
use tempfile::TempDir;

/// Benchmark record batch serialization.
fn bench_record_batch_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("record_batch_serialization");

    // Small batch (10 records, 100 bytes each)
    let small_records: Vec<_> = (0..10)
        .map(|i| distributed_log_system::log::Record {
            attributes: 0,
            timestamp_delta: i,
            offset_delta: i as i32,
            key: Some(format!("key{}", i).into_bytes()),
            value: Some(vec![0u8; 100]),
            headers: vec![],
        })
        .collect();
    let small_batch = RecordBatch::new(0, small_records);

    // Medium batch (100 records, 1KB each)
    let medium_records: Vec<_> = (0..100)
        .map(|i| distributed_log_system::log::Record {
            attributes: 0,
            timestamp_delta: i,
            offset_delta: i as i32,
            key: Some(format!("key{}", i).into_bytes()),
            value: Some(vec![0u8; 1024]),
            headers: vec![],
        })
        .collect();
    let medium_batch = RecordBatch::new(0, medium_records);

    // Large batch (1000 records, 1KB each)
    let large_records: Vec<_> = (0..1000)
        .map(|i| distributed_log_system::log::Record {
            attributes: 0,
            timestamp_delta: i,
            offset_delta: i as i32,
            key: Some(format!("key{}", i).into_bytes()),
            value: Some(vec![0u8; 1024]),
            headers: vec![],
        })
        .collect();
    let large_batch = RecordBatch::new(0, large_records);

    group.throughput(Throughput::Elements(10));
    group.bench_function("serialize_small_10", |b| {
        b.iter(|| bincode::serialize(black_box(&small_batch)).unwrap())
    });

    group.throughput(Throughput::Elements(100));
    group.bench_function("serialize_medium_100", |b| {
        b.iter(|| bincode::serialize(black_box(&medium_batch)).unwrap())
    });

    group.throughput(Throughput::Elements(1000));
    group.bench_function("serialize_large_1000", |b| {
        b.iter(|| bincode::serialize(black_box(&large_batch)).unwrap())
    });

    // Deserialize benchmarks
    let small_bytes = bincode::serialize(&small_batch).unwrap();
    let medium_bytes = bincode::serialize(&medium_batch).unwrap();

    group.throughput(Throughput::Elements(10));
    group.bench_function("deserialize_small_10", |b| {
        b.iter(|| bincode::deserialize::<RecordBatch>(black_box(&small_bytes)).unwrap())
    });

    group.throughput(Throughput::Elements(100));
    group.bench_function("deserialize_medium_100", |b| {
        b.iter(|| bincode::deserialize::<RecordBatch>(black_box(&medium_bytes)).unwrap())
    });

    group.finish();
}

/// Benchmark producer record creation and batching.
fn bench_producer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("producer_operations");

    group.bench_function("create_producer_record", |b| {
        b.iter(|| {
            ProducerRecord::new(
                black_box("test-topic"),
                black_box(Some(b"key".to_vec())),
                black_box(Some(b"value".to_vec())),
            )
        })
    });

    group.bench_function("create_producer_with_headers", |b| {
        b.iter(|| {
            ProducerRecord::new(
                black_box("test-topic"),
                black_box(Some(b"key".to_vec())),
                black_box(Some(b"value".to_vec())),
            )
            .with_header("header1", b"value1".to_vec())
            .with_header("header2", b"value2".to_vec())
        })
    });

    group.bench_function("producer_send", |b| {
        let config = ProducerConfig::default();
        let mut producer = Producer::new(config);

        b.iter(|| {
            let record = ProducerRecord::new(
                "test-topic",
                Some(b"key".to_vec()),
                Some(vec![0u8; 100]),
            );
            producer.send(black_box(record))
        })
    });

    group.finish();
}

/// Benchmark producer throughput with different batch sizes.
fn bench_producer_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("producer_throughput");

    for batch_size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_records", batch_size),
            batch_size,
            |b, &size| {
                let config = ProducerConfig {
                    batch_size: size * 100, // Large enough to hold all records
                    ..Default::default()
                };
                let mut producer = Producer::new(config);

                b.iter(|| {
                    for i in 0..size {
                        let record = ProducerRecord::new(
                            "test-topic",
                            Some(format!("key{}", i).into_bytes()),
                            Some(vec![0u8; 100]),
                        );
                        producer.send(record).unwrap();
                    }
                })
            },
        );
    }

    group.finish();
}

/// Benchmark topic partition operations.
fn bench_topic_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("topic_partition");

    group.bench_function("create_topic_partition", |b| {
        b.iter(|| TopicPartition::new(black_box("test-topic"), black_box(0)))
    });

    group.bench_function("topic_partition_hash", |b| {
        let tp = TopicPartition::new("test-topic", 0);
        b.iter(|| {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            black_box(&tp).hash(&mut hasher);
            hasher.finish()
        })
    });

    group.bench_function("topic_partition_clone", |b| {
        let tp = TopicPartition::new("test-topic-with-longer-name", 42);
        b.iter(|| black_box(&tp).clone())
    });

    group.finish();
}

/// Benchmark CRC32 computation (used for batch integrity).
fn bench_crc32(c: &mut Criterion) {
    let mut group = c.benchmark_group("crc32");

    let small_data = vec![0u8; 100];
    let medium_data = vec![0u8; 1024];
    let large_data = vec![0u8; 1024 * 1024];

    group.throughput(Throughput::Bytes(100));
    group.bench_function("crc32_100b", |b| {
        b.iter(|| crc32fast::hash(black_box(&small_data)))
    });

    group.throughput(Throughput::Bytes(1024));
    group.bench_function("crc32_1kb", |b| {
        b.iter(|| crc32fast::hash(black_box(&medium_data)))
    });

    group.throughput(Throughput::Bytes(1024 * 1024));
    group.bench_function("crc32_1mb", |b| {
        b.iter(|| crc32fast::hash(black_box(&large_data)))
    });

    group.finish();
}

/// Benchmark idempotent producer sequence tracking.
fn bench_idempotent(c: &mut Criterion) {
    use distributed_log_system::idempotent::{IdempotentConfig, SequenceTracker, ProducerStateManager};

    let mut group = c.benchmark_group("idempotent");

    group.bench_function("sequence_check_valid", |b| {
        let config = IdempotentConfig::default();
        let mut tracker = SequenceTracker::new(config);
        let mut seq = 0i32;

        b.iter(|| {
            let result = tracker.check_and_update(1, 0, seq, seq as u64);
            seq += 1;
            black_box(result)
        })
    });

    group.bench_function("sequence_check_multiple_producers", |b| {
        let config = IdempotentConfig::default();
        let mut tracker = SequenceTracker::new(config);

        // Pre-populate with producers
        for i in 0..100 {
            tracker.check_and_update(i, 0, 0, i as u64);
        }

        let mut producer = 0i64;
        let mut seq = 1i32;

        b.iter(|| {
            producer = (producer + 1) % 100;
            let result = tracker.check_and_update(producer, 0, seq, seq as u64);
            if producer == 0 {
                seq += 1;
            }
            black_box(result)
        })
    });

    group.bench_function("producer_state_manager", |b| {
        let config = IdempotentConfig::default();
        let manager = ProducerStateManager::new(config);
        let tp = TopicPartition::new("test", 0);
        let mut seq = 0i32;

        b.iter(|| {
            let result = manager.check_and_update(&tp, 1, 0, seq, seq as u64);
            seq += 1;
            black_box(result)
        })
    });

    group.finish();
}

/// Benchmark protocol message creation.
fn bench_protocol_messages(c: &mut Criterion) {
    use distributed_log_system::protocol::*;

    let mut group = c.benchmark_group("protocol_messages");

    group.bench_function("create_produce_request", |b| {
        let records: Vec<_> = (0..10)
            .map(|i| distributed_log_system::log::Record {
                attributes: 0,
                timestamp_delta: i,
                offset_delta: i as i32,
                key: Some(format!("key{}", i).into_bytes()),
                value: Some(vec![0u8; 100]),
                headers: vec![],
            })
            .collect();
        let batch = RecordBatch::new(0, records);

        b.iter(|| {
            ProduceRequest {
                transactional_id: None,
                acks: -1,
                timeout_ms: 30000,
                topic_data: vec![TopicProduceData {
                    topic: "test-topic".to_string(),
                    partition_data: vec![PartitionProduceData {
                        partition: 0,
                        records: black_box(batch.clone()),
                    }],
                }],
            }
        })
    });

    group.bench_function("create_fetch_request", |b| {
        b.iter(|| {
            FetchRequest {
                replica_id: -1,
                max_wait_ms: 500,
                min_bytes: 1,
                max_bytes: 1024 * 1024,
                isolation_level: 0,
                session_id: 0,
                session_epoch: -1,
                partitions: vec![
                    FetchPartition {
                        topic: "test-topic".to_string(),
                        partition: 0,
                        fetch_offset: 0,
                        max_bytes: 1024 * 1024,
                    },
                    FetchPartition {
                        topic: "test-topic".to_string(),
                        partition: 1,
                        fetch_offset: 100,
                        max_bytes: 1024 * 1024,
                    },
                ],
            }
        })
    });

    group.bench_function("serialize_produce_request", |b| {
        let records: Vec<_> = (0..10)
            .map(|i| distributed_log_system::log::Record {
                attributes: 0,
                timestamp_delta: i,
                offset_delta: i as i32,
                key: Some(format!("key{}", i).into_bytes()),
                value: Some(vec![0u8; 100]),
                headers: vec![],
            })
            .collect();
        let batch = RecordBatch::new(0, records);

        let request = ProduceRequest {
            transactional_id: None,
            acks: -1,
            timeout_ms: 30000,
            topic_data: vec![TopicProduceData {
                topic: "test-topic".to_string(),
                partition_data: vec![PartitionProduceData {
                    partition: 0,
                    records: batch,
                }],
            }],
        };

        b.iter(|| bincode::serialize(black_box(&request)).unwrap())
    });

    group.finish();
}

/// Benchmark segment config and partition creation.
fn bench_partition_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition_creation");

    group.bench_function("create_segment_config", |b| {
        b.iter(|| {
            SegmentConfig {
                max_segment_bytes: black_box(1024 * 1024 * 1024),
                max_index_entries: black_box(10_000_000),
                index_interval_bytes: black_box(4096),
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_record_batch_serialization,
    bench_producer_operations,
    bench_producer_throughput,
    bench_topic_partition,
    bench_crc32,
    bench_idempotent,
    bench_protocol_messages,
    bench_partition_creation,
);
criterion_main!(benches);
