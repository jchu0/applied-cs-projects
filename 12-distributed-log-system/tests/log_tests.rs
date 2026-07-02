//! Comprehensive tests for log storage components.

use distributed_log_system::cleaner::{CleanerConfig, LogCompactor, SegmentInfo};
use distributed_log_system::log::{
    Header, IndexEntry, LogSegment, Partition, Record, RecordBatch, SegmentConfig,
    TimeIndexEntry, TopicPartition,
};
use std::time::SystemTime;
use tempfile::tempdir;

// =============================================================================
// TopicPartition Tests
// =============================================================================

#[test]
fn test_topic_partition_creation() {
    let tp = TopicPartition::new("test-topic", 0);
    assert_eq!(tp.topic, "test-topic");
    assert_eq!(tp.partition, 0);
}

#[test]
fn test_topic_partition_equality() {
    let tp1 = TopicPartition::new("topic", 0);
    let tp2 = TopicPartition::new("topic", 0);
    let tp3 = TopicPartition::new("topic", 1);
    let tp4 = TopicPartition::new("other", 0);

    assert_eq!(tp1, tp2);
    assert_ne!(tp1, tp3);
    assert_ne!(tp1, tp4);
}

#[test]
fn test_topic_partition_hash() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(TopicPartition::new("topic", 0));
    set.insert(TopicPartition::new("topic", 1));
    set.insert(TopicPartition::new("topic", 0)); // Duplicate

    assert_eq!(set.len(), 2);
}

#[test]
fn test_topic_partition_clone() {
    let tp1 = TopicPartition::new("topic", 5);
    let tp2 = tp1.clone();

    assert_eq!(tp1, tp2);
}

// =============================================================================
// Record Tests
// =============================================================================

#[test]
fn test_record_with_key_and_value() {
    let record = Record {
        attributes: 0,
        timestamp_delta: 100,
        offset_delta: 0,
        key: Some(b"key".to_vec()),
        value: Some(b"value".to_vec()),
        headers: vec![],
    };

    assert_eq!(record.key, Some(b"key".to_vec()));
    assert_eq!(record.value, Some(b"value".to_vec()));
}

#[test]
fn test_record_without_key() {
    let record = Record {
        attributes: 0,
        timestamp_delta: 0,
        offset_delta: 0,
        key: None,
        value: Some(b"value".to_vec()),
        headers: vec![],
    };

    assert!(record.key.is_none());
    assert!(record.value.is_some());
}

#[test]
fn test_record_tombstone() {
    let record = Record {
        attributes: 0,
        timestamp_delta: 0,
        offset_delta: 0,
        key: Some(b"key".to_vec()),
        value: None, // Tombstone
        headers: vec![],
    };

    assert!(record.key.is_some());
    assert!(record.value.is_none());
}

#[test]
fn test_record_with_headers() {
    let record = Record {
        attributes: 0,
        timestamp_delta: 0,
        offset_delta: 0,
        key: Some(b"key".to_vec()),
        value: Some(b"value".to_vec()),
        headers: vec![
            Header {
                key: "header1".to_string(),
                value: b"value1".to_vec(),
            },
            Header {
                key: "header2".to_string(),
                value: b"value2".to_vec(),
            },
        ],
    };

    assert_eq!(record.headers.len(), 2);
    assert_eq!(record.headers[0].key, "header1");
}

#[test]
fn test_record_empty_key_and_value() {
    let record = Record {
        attributes: 0,
        timestamp_delta: 0,
        offset_delta: 0,
        key: Some(vec![]),
        value: Some(vec![]),
        headers: vec![],
    };

    assert_eq!(record.key, Some(vec![]));
    assert_eq!(record.value, Some(vec![]));
}

// =============================================================================
// Header Tests
// =============================================================================

#[test]
fn test_header_creation() {
    let header = Header {
        key: "content-type".to_string(),
        value: b"application/json".to_vec(),
    };

    assert_eq!(header.key, "content-type");
    assert_eq!(header.value, b"application/json".to_vec());
}

#[test]
fn test_header_empty_value() {
    let header = Header {
        key: "empty".to_string(),
        value: vec![],
    };

    assert!(header.value.is_empty());
}

// =============================================================================
// RecordBatch Tests
// =============================================================================

#[test]
fn test_record_batch_creation() {
    let records = vec![
        Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 0,
            key: Some(b"key1".to_vec()),
            value: Some(b"value1".to_vec()),
            headers: vec![],
        },
        Record {
            attributes: 0,
            timestamp_delta: 10,
            offset_delta: 1,
            key: Some(b"key2".to_vec()),
            value: Some(b"value2".to_vec()),
            headers: vec![],
        },
    ];

    let batch = RecordBatch::new(100, records);

    assert_eq!(batch.base_offset, 100);
    assert_eq!(batch.record_count(), 2);
    assert_eq!(batch.last_offset_delta, 1);
    assert_eq!(batch.magic, 2);
}

#[test]
fn test_record_batch_empty() {
    let batch = RecordBatch::new(0, vec![]);

    assert_eq!(batch.record_count(), 0);
    assert_eq!(batch.last_offset_delta, 0);
}

#[test]
fn test_record_batch_last_offset() {
    let records = vec![
        Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 0,
            key: None,
            value: Some(b"v1".to_vec()),
            headers: vec![],
        },
        Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 1,
            key: None,
            value: Some(b"v2".to_vec()),
            headers: vec![],
        },
        Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 2,
            key: None,
            value: Some(b"v3".to_vec()),
            headers: vec![],
        },
    ];

    let batch = RecordBatch::new(10, records);

    assert_eq!(batch.last_offset(), 12);
}

#[test]
fn test_record_batch_serialization() {
    let records = vec![Record {
        attributes: 0,
        timestamp_delta: 0,
        offset_delta: 0,
        key: Some(b"key".to_vec()),
        value: Some(b"value".to_vec()),
        headers: vec![],
    }];

    let batch = RecordBatch::new(0, records);
    let data = batch.serialize().unwrap();
    let deserialized = RecordBatch::deserialize(&data).unwrap();

    assert_eq!(deserialized.base_offset, batch.base_offset);
    assert_eq!(deserialized.record_count(), batch.record_count());
    assert_eq!(deserialized.records[0].key, batch.records[0].key);
}

#[test]
fn test_record_batch_serialization_large() {
    let mut records = Vec::new();
    for i in 0..100 {
        records.push(Record {
            attributes: 0,
            timestamp_delta: i as i64,
            offset_delta: i as u32,
            key: Some(format!("key{}", i).into_bytes()),
            value: Some(format!("value{}", i).into_bytes()),
            headers: vec![],
        });
    }

    let batch = RecordBatch::new(1000, records);
    let data = batch.serialize().unwrap();
    let deserialized = RecordBatch::deserialize(&data).unwrap();

    assert_eq!(deserialized.record_count(), 100);
}

#[test]
fn test_record_batch_with_producer_info() {
    let batch = RecordBatch {
        base_offset: 0,
        batch_length: 0,
        partition_leader_epoch: 5,
        magic: 2,
        crc: 0,
        attributes: 0,
        last_offset_delta: 0,
        base_timestamp: 1234567890,
        max_timestamp: 1234567890,
        producer_id: 123,
        producer_epoch: 1,
        base_sequence: 10,
        records: vec![],
    };

    assert_eq!(batch.producer_id, 123);
    assert_eq!(batch.producer_epoch, 1);
    assert_eq!(batch.base_sequence, 10);
}

// =============================================================================
// SegmentConfig Tests
// =============================================================================

#[test]
fn test_segment_config_default() {
    let config = SegmentConfig::default();

    assert_eq!(config.max_segment_bytes, 1024 * 1024 * 1024);
    assert_eq!(config.index_interval_bytes, 4096);
}

#[test]
fn test_segment_config_custom() {
    let config = SegmentConfig {
        max_segment_bytes: 1024 * 1024,
        index_interval_bytes: 1024,
    };

    assert_eq!(config.max_segment_bytes, 1024 * 1024);
    assert_eq!(config.index_interval_bytes, 1024);
}

// =============================================================================
// IndexEntry Tests
// =============================================================================

#[test]
fn test_index_entry() {
    let entry = IndexEntry {
        relative_offset: 100,
        position: 4096,
    };

    assert_eq!(entry.relative_offset, 100);
    assert_eq!(entry.position, 4096);
}

#[test]
fn test_index_entry_zero() {
    let entry = IndexEntry {
        relative_offset: 0,
        position: 0,
    };

    assert_eq!(entry.relative_offset, 0);
    assert_eq!(entry.position, 0);
}

// =============================================================================
// TimeIndexEntry Tests
// =============================================================================

#[test]
fn test_time_index_entry() {
    let entry = TimeIndexEntry {
        timestamp: 1234567890,
        relative_offset: 500,
    };

    assert_eq!(entry.timestamp, 1234567890);
    assert_eq!(entry.relative_offset, 500);
}

// =============================================================================
// LogSegment Tests
// =============================================================================

#[test]
fn test_log_segment_creation() {
    let dir = tempdir().unwrap();
    let segment = LogSegment::new(dir.path(), 0, SegmentConfig::default()).unwrap();

    assert_eq!(segment.base_offset, 0);
    assert_eq!(segment.size(), 0);
    assert!(!segment.is_full());
}

#[test]
fn test_log_segment_append() {
    let dir = tempdir().unwrap();
    let mut segment = LogSegment::new(dir.path(), 0, SegmentConfig::default()).unwrap();

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

    segment.append(&batch).unwrap();

    assert!(segment.size() > 0);
}

#[test]
fn test_log_segment_read() {
    let dir = tempdir().unwrap();
    let mut segment = LogSegment::new(dir.path(), 0, SegmentConfig::default()).unwrap();

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

    segment.append(&batch).unwrap();

    let batches = segment.read(0, 10000).unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].records[0].key, Some(b"key".to_vec()));
}

#[test]
fn test_log_segment_multiple_batches() {
    let dir = tempdir().unwrap();
    let mut segment = LogSegment::new(dir.path(), 0, SegmentConfig::default()).unwrap();

    for i in 0..10 {
        let batch = RecordBatch::new(
            i,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key{}", i).into_bytes()),
                value: Some(format!("value{}", i).into_bytes()),
                headers: vec![],
            }],
        );
        segment.append(&batch).unwrap();
    }

    let batches = segment.read(0, 100000).unwrap();
    assert_eq!(batches.len(), 10);
}

#[test]
fn test_log_segment_read_from_offset() {
    let dir = tempdir().unwrap();
    let mut segment = LogSegment::new(dir.path(), 0, SegmentConfig::default()).unwrap();

    for i in 0..5 {
        let batch = RecordBatch::new(
            i,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key{}", i).into_bytes()),
                value: Some(format!("value{}", i).into_bytes()),
                headers: vec![],
            }],
        );
        segment.append(&batch).unwrap();
    }

    // Read from offset 3
    let batches = segment.read(3, 100000).unwrap();
    assert!(batches.len() >= 2);
    assert!(batches[0].base_offset >= 3);
}

#[test]
fn test_log_segment_flush() {
    let dir = tempdir().unwrap();
    let mut segment = LogSegment::new(dir.path(), 0, SegmentConfig::default()).unwrap();

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

    segment.append(&batch).unwrap();
    segment.flush().unwrap();

    // Verify files exist
    assert!(dir.path().join("00000000000000000000.log").exists());
}

#[test]
fn test_log_segment_is_full() {
    let dir = tempdir().unwrap();
    let config = SegmentConfig {
        max_segment_bytes: 100, // Very small
        index_interval_bytes: 50,
    };
    let mut segment = LogSegment::new(dir.path(), 0, config).unwrap();

    // Append enough data to fill segment
    for i in 0..10 {
        let batch = RecordBatch::new(
            i,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(b"key".to_vec()),
                value: Some(b"large value that takes up space".to_vec()),
                headers: vec![],
            }],
        );
        segment.append(&batch).unwrap();
    }

    assert!(segment.is_full());
}

// =============================================================================
// Partition Tests
// =============================================================================

#[test]
fn test_partition_creation() {
    let dir = tempdir().unwrap();
    let partition = Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

    assert_eq!(partition.topic, "test-topic");
    assert_eq!(partition.partition_id, 0);
    assert_eq!(partition.log_end_offset, 0);
    assert_eq!(partition.high_watermark, 0);
}

#[test]
fn test_partition_append() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

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

    let offset = partition.append(batch).unwrap();

    assert_eq!(offset, 0);
    assert_eq!(partition.log_end_offset, 1);
}

#[test]
fn test_partition_append_multiple() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

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

        let offset = partition.append(batch).unwrap();
        assert_eq!(offset, i);
    }

    assert_eq!(partition.log_end_offset, 10);
}

#[test]
fn test_partition_read() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

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

    partition.append(batch).unwrap();

    let batches = partition.read(0, 10000).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].records[0].key, Some(b"key".to_vec()));
}

#[test]
fn test_partition_high_watermark() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

    assert_eq!(partition.high_watermark, 0);

    partition.update_high_watermark(100);
    assert_eq!(partition.high_watermark, 100);

    // Should not decrease
    partition.update_high_watermark(50);
    assert_eq!(partition.high_watermark, 100);
}

#[test]
fn test_partition_topic_partition() {
    let dir = tempdir().unwrap();
    let partition =
        Partition::new(dir.path(), "my-topic", 5, SegmentConfig::default()).unwrap();

    let tp = partition.topic_partition();
    assert_eq!(tp.topic, "my-topic");
    assert_eq!(tp.partition, 5);
}

#[test]
fn test_partition_flush() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

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

    partition.append(batch).unwrap();
    partition.flush().unwrap();
}

#[test]
fn test_partition_leader_epoch() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

    assert_eq!(partition.leader_epoch, 0);

    partition.leader_epoch = 5;
    assert_eq!(partition.leader_epoch, 5);
}

#[test]
fn test_partition_isr() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

    assert!(partition.isr.is_empty());

    partition.isr.insert(1);
    partition.isr.insert(2);
    partition.isr.insert(3);

    assert_eq!(partition.isr.len(), 3);
    assert!(partition.isr.contains(&1));
}

#[test]
fn test_partition_leader() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

    assert!(partition.leader.is_none());

    partition.leader = Some(1);
    assert_eq!(partition.leader, Some(1));
}

// =============================================================================
// Segment Rotation Tests
// =============================================================================

#[test]
fn test_partition_segment_rotation() {
    let dir = tempdir().unwrap();
    let config = SegmentConfig {
        max_segment_bytes: 200, // Very small to force rotation
        index_interval_bytes: 50,
    };
    let mut partition = Partition::new(dir.path(), "test-topic", 0, config).unwrap();

    // Append many records to force segment rotation
    for i in 0..100 {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("key{}", i).into_bytes()),
                value: Some(format!("value-with-extra-content-{}", i).into_bytes()),
                headers: vec![],
            }],
        );

        partition.append(batch).unwrap();
    }

    assert_eq!(partition.log_end_offset, 100);
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_log_segment_read_invalid_offset() {
    let dir = tempdir().unwrap();
    let mut segment = LogSegment::new(dir.path(), 100, SegmentConfig::default()).unwrap();

    // Try to read offset before segment base
    let result = segment.read(50, 1000);
    assert!(result.is_err());
}

// =============================================================================
// CRC Tests
// =============================================================================

#[test]
fn test_record_batch_crc() {
    let dir = tempdir().unwrap();
    let mut segment = LogSegment::new(dir.path(), 0, SegmentConfig::default()).unwrap();

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

    segment.append(&batch).unwrap();

    // Read back and verify CRC was validated
    let batches = segment.read(0, 10000).unwrap();
    assert_eq!(batches.len(), 1);
}

// =============================================================================
// Large Data Tests
// =============================================================================

#[test]
fn test_large_value() {
    let dir = tempdir().unwrap();
    let mut segment = LogSegment::new(dir.path(), 0, SegmentConfig::default()).unwrap();

    let large_value = vec![0u8; 1024 * 1024]; // 1MB

    let batch = RecordBatch::new(
        0,
        vec![Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 0,
            key: Some(b"key".to_vec()),
            value: Some(large_value.clone()),
            headers: vec![],
        }],
    );

    segment.append(&batch).unwrap();

    let batches = segment.read(0, 2 * 1024 * 1024).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].records[0].value.as_ref().unwrap().len(), 1024 * 1024);
}

#[test]
fn test_many_small_records() {
    let dir = tempdir().unwrap();
    let mut partition =
        Partition::new(dir.path(), "test-topic", 0, SegmentConfig::default()).unwrap();

    for i in 0..1000 {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("k{}", i).into_bytes()),
                value: Some(format!("v{}", i).into_bytes()),
                headers: vec![],
            }],
        );

        partition.append(batch).unwrap();
    }

    assert_eq!(partition.log_end_offset, 1000);
}

// =============================================================================
// Compaction and retention/deletion tests
// =============================================================================

/// Helper: write one single-record batch per offset into a segment dir.
fn write_kv_segment(dir: &std::path::Path, entries: &[(u64, &str, Option<&str>)]) {
    let mut seg = LogSegment::new(dir, 0, SegmentConfig::default()).unwrap();
    for (offset, key, value) in entries {
        let batch = RecordBatch::new(
            *offset,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(key.as_bytes().to_vec()),
                value: value.map(|v| v.as_bytes().to_vec()),
                headers: vec![],
            }],
        );
        seg.append(&batch).unwrap();
    }
    seg.flush().unwrap();
}

#[test]
fn test_compaction_reduces_to_latest_value_per_key() {
    // A compacted topic with repeated keys must be reduced to the latest value
    // per key. Candidates must actually be found and compacted (regression for
    // the empty `get_segment_metadata` / no-op compaction).
    let data_dir = tempdir().unwrap();
    let work_dir = tempdir().unwrap();

    // key1 written 3x, key2 written 2x, key3 once -> 6 records, 3 unique keys.
    write_kv_segment(
        data_dir.path(),
        &[
            (0, "key1", Some("v1a")),
            (1, "key2", Some("v2a")),
            (2, "key1", Some("v1b")),
            (3, "key3", Some("v3")),
            (4, "key2", Some("v2b")),
            (5, "key1", Some("v1c")),
        ],
    );

    let compactor = LogCompactor::new(
        CleanerConfig::default(),
        work_dir.path().to_path_buf(),
        SegmentConfig::default(),
    );

    let segments = vec![SegmentInfo {
        base_offset: 0,
        size: 1000,
        last_modified: SystemTime::now(),
    }];

    let result = compactor.compact(data_dir.path(), &segments).unwrap();

    // Compaction actually ran over the segment.
    assert_eq!(result.segments_read, 1);
    // Exactly one surviving record per unique key.
    assert_eq!(result.records_kept, 3, "expected 3 latest-value records kept");
    // The three superseded duplicates were removed.
    assert_eq!(result.records_removed, 3, "expected 3 duplicates removed");
}

#[test]
fn test_retention_deletion_reclaims_disk() {
    // After deleting a segment, its underlying files must no longer exist on
    // disk (regression for delete_segment not unlinking files).
    let dir = tempdir().unwrap();

    // Use a tiny max segment size so appends roll into multiple segments.
    let small = SegmentConfig {
        max_segment_bytes: 1,
        index_interval_bytes: 4096,
    };
    let mut partition = Partition::new(dir.path(), "retention-topic-2", 0, small).unwrap();

    for i in 0..5 {
        let batch = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(format!("k{}", i).into_bytes()),
                value: Some(vec![0u8; 64]),
                headers: vec![],
            }],
        );
        partition.append(batch).unwrap();
    }
    partition.flush().unwrap();

    // There must be more than one segment (offset 0 is not active).
    assert!(partition.segment_count() > 1, "expected multiple segments");

    let meta = partition.segment_metadata();
    let base = meta[0].base_offset;

    // Files for the oldest segment must exist before deletion.
    let dir_for_partition = dir.path().join("retention-topic-2").join("0");
    let log_path = dir_for_partition.join(format!("{:020}.log", base));
    let index_path = dir_for_partition.join(format!("{:020}.index", base));
    let time_path = dir_for_partition.join(format!("{:020}.timeindex", base));
    assert!(log_path.exists(), "log file should exist before deletion");

    // Delete the segment; files must be unlinked from disk.
    partition.delete_segment(base).unwrap();

    assert!(!log_path.exists(), "log file should be removed from disk");
    assert!(!index_path.exists(), "index file should be removed from disk");
    assert!(!time_path.exists(), "timeindex file should be removed from disk");
    assert_eq!(partition.segment_count(), meta.len() - 1);
}
