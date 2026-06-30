//! Log storage: segments, partitions, and record batches.

use crate::{Error, Offset, Result, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use tracing::{debug, warn};

/// Topic-partition identifier.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct TopicPartition {
    pub topic: String,
    pub partition: u32,
}

impl TopicPartition {
    pub fn new(topic: impl Into<String>, partition: u32) -> Self {
        Self {
            topic: topic.into(),
            partition,
        }
    }
}

/// A batch of records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordBatch {
    /// First offset in this batch.
    pub base_offset: Offset,
    /// Total byte length of batch.
    pub batch_length: u32,
    /// Leader epoch when batch was written.
    pub partition_leader_epoch: u32,
    /// Format version.
    pub magic: u8,
    /// CRC of the batch data.
    pub crc: u32,
    /// Batch attributes (compression, timestamp type).
    pub attributes: u16,
    /// Delta from base_offset to last offset.
    pub last_offset_delta: u32,
    /// Timestamp of first record.
    pub base_timestamp: Timestamp,
    /// Maximum timestamp in batch.
    pub max_timestamp: Timestamp,
    /// Producer ID for idempotence.
    pub producer_id: i64,
    /// Producer epoch.
    pub producer_epoch: i16,
    /// Base sequence number.
    pub base_sequence: u32,
    /// The actual records.
    pub records: Vec<Record>,
}

impl RecordBatch {
    /// Create a new record batch.
    pub fn new(base_offset: Offset, records: Vec<Record>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as Timestamp;

        let last_offset_delta = if records.is_empty() {
            0
        } else {
            records.len() as u32 - 1
        };

        Self {
            base_offset,
            batch_length: 0,
            partition_leader_epoch: 0,
            magic: 2,
            crc: 0,
            attributes: 0,
            last_offset_delta,
            base_timestamp: now,
            max_timestamp: now,
            producer_id: -1,
            producer_epoch: -1,
            base_sequence: 0,
            records,
        }
    }

    /// Get number of records.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Get last offset in batch.
    pub fn last_offset(&self) -> Offset {
        self.base_offset + self.last_offset_delta as u64
    }

    /// Serialize batch to bytes.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| Error::Serialization(e.to_string()))
    }

    /// Deserialize batch from bytes.
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| Error::Serialization(e.to_string()))
    }
}

/// A single record in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    /// Record attributes.
    pub attributes: u8,
    /// Timestamp delta from batch base.
    pub timestamp_delta: i64,
    /// Offset delta from batch base.
    pub offset_delta: u32,
    /// Record key.
    pub key: Option<Vec<u8>>,
    /// Record value.
    pub value: Option<Vec<u8>>,
    /// Record headers.
    pub headers: Vec<Header>,
}

/// Record header.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Header {
    pub key: String,
    pub value: Vec<u8>,
}

/// Configuration for log segments.
#[derive(Debug, Clone)]
pub struct SegmentConfig {
    /// Maximum segment size in bytes.
    pub max_segment_bytes: u64,
    /// Index interval in bytes.
    pub index_interval_bytes: u64,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            max_segment_bytes: 1024 * 1024 * 1024, // 1GB
            index_interval_bytes: 4096,
        }
    }
}

/// Sparse index entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct IndexEntry {
    /// Offset relative to segment base.
    pub relative_offset: u32,
    /// Position in log file.
    pub position: u32,
}

/// Time index entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct TimeIndexEntry {
    /// Timestamp.
    pub timestamp: Timestamp,
    /// Offset relative to segment base.
    pub relative_offset: u32,
}

/// A log segment (portion of a partition).
pub struct LogSegment {
    /// Base offset of this segment.
    pub base_offset: Offset,
    /// Segment directory.
    dir: PathBuf,
    /// Log file (actual data).
    log_file: File,
    /// Index file (sparse offset index).
    index_file: File,
    /// Time index file.
    time_index_file: File,
    /// Current write position.
    position: u64,
    /// Maximum timestamp in segment.
    max_timestamp: Timestamp,
    /// Configuration.
    config: SegmentConfig,
    /// Bytes since last index entry.
    bytes_since_last_index: u64,
    /// In-memory index entries.
    index_entries: Vec<IndexEntry>,
}

impl LogSegment {
    /// Create a new log segment.
    pub fn new(dir: impl AsRef<Path>, base_offset: Offset, config: SegmentConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;

        let log_path = dir.join(format!("{:020}.log", base_offset));
        let index_path = dir.join(format!("{:020}.index", base_offset));
        let time_index_path = dir.join(format!("{:020}.timeindex", base_offset));

        let log_file = OpenOptions::new()
            .create(true)
            .read(true)
            .append(true)
            .open(&log_path)?;

        let index_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&index_path)?;

        let time_index_file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .open(&time_index_path)?;

        Ok(Self {
            base_offset,
            dir,
            log_file,
            index_file,
            time_index_file,
            position: 0,
            max_timestamp: 0,
            config,
            bytes_since_last_index: 0,
            index_entries: Vec::new(),
        })
    }

    /// Append a record batch to the segment.
    pub fn append(&mut self, batch: &RecordBatch) -> Result<()> {
        let data = batch.serialize()?;
        let length = data.len() as u32;
        let crc = crc32fast::hash(&data);

        // Write: [length: u32][crc: u32][data]
        self.log_file.write_all(&length.to_le_bytes())?;
        self.log_file.write_all(&crc.to_le_bytes())?;
        self.log_file.write_all(&data)?;

        let entry_size = 8 + data.len() as u64;

        // Update index if needed
        self.bytes_since_last_index += entry_size;
        if self.bytes_since_last_index >= self.config.index_interval_bytes {
            let entry = IndexEntry {
                relative_offset: (batch.base_offset - self.base_offset) as u32,
                position: self.position as u32,
            };
            self.index_entries.push(entry);
            self.write_index_entry(&entry)?;
            self.bytes_since_last_index = 0;
        }

        // Update time index
        if batch.max_timestamp > self.max_timestamp {
            self.max_timestamp = batch.max_timestamp;
            let time_entry = TimeIndexEntry {
                timestamp: batch.max_timestamp,
                relative_offset: (batch.base_offset - self.base_offset) as u32,
            };
            self.write_time_index_entry(&time_entry)?;
        }

        self.position += entry_size;

        Ok(())
    }

    /// Write index entry to file.
    fn write_index_entry(&mut self, entry: &IndexEntry) -> Result<()> {
        self.index_file
            .write_all(&entry.relative_offset.to_le_bytes())?;
        self.index_file.write_all(&entry.position.to_le_bytes())?;
        Ok(())
    }

    /// Write time index entry to file.
    fn write_time_index_entry(&mut self, entry: &TimeIndexEntry) -> Result<()> {
        self.time_index_file
            .write_all(&entry.timestamp.to_le_bytes())?;
        self.time_index_file
            .write_all(&entry.relative_offset.to_le_bytes())?;
        Ok(())
    }

    /// Read records starting from offset.
    pub fn read(&mut self, offset: Offset, max_bytes: u32) -> Result<Vec<RecordBatch>> {
        let position = self.find_position(offset)?;

        self.log_file.seek(SeekFrom::Start(position))?;
        let mut reader = BufReader::new(&self.log_file);

        let mut batches = Vec::new();
        let mut bytes_read = 0u32;

        while bytes_read < max_bytes {
            // Read length
            let mut len_buf = [0u8; 4];
            if reader.read_exact(&mut len_buf).is_err() {
                break;
            }
            let length = u32::from_le_bytes(len_buf);

            // Read CRC
            let mut crc_buf = [0u8; 4];
            reader.read_exact(&mut crc_buf)?;
            let stored_crc = u32::from_le_bytes(crc_buf);

            // Read data
            let mut data = vec![0u8; length as usize];
            reader.read_exact(&mut data)?;

            // Verify CRC
            let computed_crc = crc32fast::hash(&data);
            if computed_crc != stored_crc {
                warn!("CRC mismatch at position {}", position + bytes_read as u64);
                break;
            }

            let batch = RecordBatch::deserialize(&data)?;
            if batch.base_offset >= offset {
                batches.push(batch);
            }

            bytes_read += 8 + length;
        }

        Ok(batches)
    }

    /// Find file position for offset using index.
    fn find_position(&self, offset: Offset) -> Result<u64> {
        if offset < self.base_offset {
            return Err(Error::InvalidOffset(offset));
        }

        let relative_offset = (offset - self.base_offset) as u32;

        // Binary search in index
        let idx = self
            .index_entries
            .partition_point(|e| e.relative_offset <= relative_offset);

        if idx > 0 {
            Ok(self.index_entries[idx - 1].position as u64)
        } else {
            Ok(0)
        }
    }

    /// Check if segment is full.
    pub fn is_full(&self) -> bool {
        self.position >= self.config.max_segment_bytes
    }

    /// Get segment size.
    pub fn size(&self) -> u64 {
        self.position
    }

    /// Flush to disk.
    pub fn flush(&mut self) -> Result<()> {
        self.log_file.sync_all()?;
        self.index_file.sync_all()?;
        self.time_index_file.sync_all()?;
        Ok(())
    }
}

/// A partition (ordered log for a topic).
pub struct Partition {
    /// Topic name.
    pub topic: String,
    /// Partition ID.
    pub partition_id: u32,
    /// Partition directory.
    dir: PathBuf,
    /// Log segments by base offset.
    segments: BTreeMap<Offset, LogSegment>,
    /// Active segment for writes.
    active_segment_offset: Offset,
    /// Configuration.
    config: SegmentConfig,

    // Replication state
    /// Leader epoch.
    pub leader_epoch: u32,
    /// High watermark (committed offset).
    pub high_watermark: Offset,
    /// Log end offset (last appended).
    pub log_end_offset: Offset,

    // ISR management
    /// In-sync replicas.
    pub isr: HashSet<crate::BrokerId>,
    /// Current leader.
    pub leader: Option<crate::BrokerId>,
}

impl Partition {
    /// Create a new partition.
    pub fn new(
        dir: impl AsRef<Path>,
        topic: impl Into<String>,
        partition_id: u32,
        config: SegmentConfig,
    ) -> Result<Self> {
        let topic = topic.into();
        let dir = dir.as_ref().join(&topic).join(partition_id.to_string());
        std::fs::create_dir_all(&dir)?;

        let mut partition = Self {
            topic,
            partition_id,
            dir: dir.clone(),
            segments: BTreeMap::new(),
            active_segment_offset: 0,
            config: config.clone(),
            leader_epoch: 0,
            high_watermark: 0,
            log_end_offset: 0,
            isr: HashSet::new(),
            leader: None,
        };

        // Create initial segment
        let segment = LogSegment::new(&dir, 0, config)?;
        partition.segments.insert(0, segment);

        Ok(partition)
    }

    /// Append a record batch.
    pub fn append(&mut self, batch: RecordBatch) -> Result<Offset> {
        // Get active segment
        let segment = self
            .segments
            .get_mut(&self.active_segment_offset)
            .ok_or(Error::Internal("No active segment".into()))?;

        // Roll segment if full
        if segment.is_full() {
            self.roll_segment()?;
        }

        let segment = self.segments.get_mut(&self.active_segment_offset).unwrap();

        let offset = self.log_end_offset;
        let mut batch = batch;
        batch.base_offset = offset;

        segment.append(&batch)?;
        self.log_end_offset = batch.last_offset() + 1;

        Ok(offset)
    }

    /// Roll to a new segment.
    fn roll_segment(&mut self) -> Result<()> {
        let new_base = self.log_end_offset;
        let segment = LogSegment::new(&self.dir, new_base, self.config.clone())?;
        self.segments.insert(new_base, segment);
        self.active_segment_offset = new_base;
        Ok(())
    }

    /// Read records from offset.
    pub fn read(&mut self, offset: Offset, max_bytes: u32) -> Result<Vec<RecordBatch>> {
        // Find segment containing offset
        let segment_offset = self
            .segments
            .range(..=offset)
            .next_back()
            .map(|(k, _)| *k)
            .unwrap_or(0);

        let segment = self
            .segments
            .get_mut(&segment_offset)
            .ok_or(Error::InvalidOffset(offset))?;

        segment.read(offset, max_bytes)
    }

    /// Update high watermark.
    pub fn update_high_watermark(&mut self, offset: Offset) {
        if offset > self.high_watermark {
            self.high_watermark = offset;
        }
    }

    /// Get topic-partition identifier.
    pub fn topic_partition(&self) -> TopicPartition {
        TopicPartition::new(&self.topic, self.partition_id)
    }

    /// Flush all segments to disk.
    pub fn flush(&mut self) -> Result<()> {
        for segment in self.segments.values_mut() {
            segment.flush()?;
        }
        Ok(())
    }

    /// Delete old segments for retention.
    pub fn delete_segment(&mut self, base_offset: Offset) -> Result<()> {
        self.segments.remove(&base_offset);
        // Would also delete files in real implementation
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

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

        assert_eq!(deserialized.records.len(), 1);
        assert_eq!(deserialized.records[0].key, Some(b"key".to_vec()));
    }

    #[test]
    fn test_partition_append_read() {
        let dir = tempdir().unwrap();
        let mut partition =
            Partition::new(dir.path(), "test", 0, SegmentConfig::default()).unwrap();

        let records = vec![Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 0,
            key: Some(b"key".to_vec()),
            value: Some(b"value".to_vec()),
            headers: vec![],
        }];

        let batch = RecordBatch::new(0, records);
        let offset = partition.append(batch).unwrap();

        assert_eq!(offset, 0);

        let batches = partition.read(0, 1000).unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].records[0].key, Some(b"key".to_vec()));
    }
}
