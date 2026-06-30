//! Log cleaner: retention and compaction.

use crate::log::{IndexEntry, LogSegment, Partition, Record, RecordBatch, SegmentConfig, TimeIndexEntry};
use crate::{Error, Offset, Result, Timestamp};

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tracing::{debug, error, info, warn};

/// Cleaner configuration.
#[derive(Debug, Clone)]
pub struct CleanerConfig {
    /// Retention time in milliseconds (0 = unlimited).
    pub retention_ms: u64,
    /// Retention size in bytes (0 = unlimited).
    pub retention_bytes: u64,
    /// Minimum segment age before compaction (milliseconds).
    pub min_compaction_lag_ms: u64,
    /// Maximum segment size for compaction.
    pub segment_size: u64,
    /// Minimum cleanable ratio (0.0 to 1.0).
    pub min_cleanable_ratio: f64,
    /// Delete retention time for tombstones (milliseconds).
    pub delete_retention_ms: u64,
    /// Maximum number of segments to clean at once.
    pub max_clean_segments: usize,
    /// Clean interval in milliseconds.
    pub clean_interval_ms: u64,
    /// Enable log compaction.
    pub enable_compaction: bool,
}

impl Default for CleanerConfig {
    fn default() -> Self {
        Self {
            retention_ms: 7 * 24 * 60 * 60 * 1000, // 7 days
            retention_bytes: 0,                     // unlimited
            min_compaction_lag_ms: 60 * 1000,       // 1 minute
            segment_size: 1024 * 1024 * 1024,       // 1 GB
            min_cleanable_ratio: 0.5,
            delete_retention_ms: 24 * 60 * 60 * 1000, // 24 hours
            max_clean_segments: 10,
            clean_interval_ms: 30 * 1000, // 30 seconds
            enable_compaction: false,
        }
    }
}

/// Cleanup policy for a topic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupPolicy {
    /// Delete old segments based on time or size.
    Delete,
    /// Compact the log by key.
    Compact,
    /// Both delete and compact.
    CompactDelete,
}

impl Default for CleanupPolicy {
    fn default() -> Self {
        CleanupPolicy::Delete
    }
}

/// Segment metadata for cleaning decisions.
#[derive(Debug, Clone)]
pub struct SegmentMetadata {
    /// Base offset of the segment.
    pub base_offset: Offset,
    /// Segment size in bytes.
    pub size: u64,
    /// Last modified time (epoch ms).
    pub last_modified_ms: u64,
    /// Maximum timestamp in segment.
    pub max_timestamp: Timestamp,
    /// Number of records.
    pub record_count: u64,
    /// Estimated cleanable bytes.
    pub cleanable_bytes: u64,
}

/// Log cleaner for retention and compaction.
pub struct LogCleaner {
    /// Configuration.
    config: CleanerConfig,
    /// Whether the cleaner is running.
    running: AtomicBool,
    /// Total bytes cleaned.
    bytes_cleaned: AtomicU64,
    /// Total segments deleted.
    segments_deleted: AtomicU64,
    /// Total segments compacted.
    segments_compacted: AtomicU64,
}

impl LogCleaner {
    /// Create a new log cleaner.
    pub fn new(config: CleanerConfig) -> Self {
        Self {
            config,
            running: AtomicBool::new(false),
            bytes_cleaned: AtomicU64::new(0),
            segments_deleted: AtomicU64::new(0),
            segments_compacted: AtomicU64::new(0),
        }
    }

    /// Start the cleaner.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        info!("Log cleaner started");
    }

    /// Stop the cleaner.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("Log cleaner stopped");
    }

    /// Check if running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Clean a partition based on retention policy.
    pub fn clean_retention(&self, partition: &mut Partition) -> Result<CleanResult> {
        let mut result = CleanResult::default();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Get all segment metadata
        let segments = self.get_segment_metadata(partition)?;
        if segments.is_empty() {
            return Ok(result);
        }

        // Calculate total size
        let total_size: u64 = segments.iter().map(|s| s.size).sum();

        // Find segments to delete
        let mut segments_to_delete = Vec::new();
        let mut size_freed: u64 = 0;
        let mut remaining_size = total_size;

        for segment in &segments {
            // Skip active segment (last one)
            if segment.base_offset == partition.log_end_offset.saturating_sub(1) {
                continue;
            }

            let mut should_delete = false;

            // Time-based retention
            if self.config.retention_ms > 0 {
                let age = now.saturating_sub(segment.last_modified_ms);
                if age > self.config.retention_ms {
                    should_delete = true;
                    debug!(
                        "Segment {} expired: age {}ms > retention {}ms",
                        segment.base_offset, age, self.config.retention_ms
                    );
                }
            }

            // Size-based retention (delete oldest first)
            if self.config.retention_bytes > 0 && remaining_size > self.config.retention_bytes {
                should_delete = true;
                debug!(
                    "Segment {} exceeds size: remaining {}B > retention {}B",
                    segment.base_offset, remaining_size, self.config.retention_bytes
                );
            }

            if should_delete {
                segments_to_delete.push(segment.base_offset);
                size_freed += segment.size;
                remaining_size = remaining_size.saturating_sub(segment.size);
            }
        }

        // Delete segments
        for base_offset in segments_to_delete {
            if let Err(e) = partition.delete_segment(base_offset) {
                error!("Failed to delete segment {}: {}", base_offset, e);
            } else {
                result.segments_deleted += 1;
                info!("Deleted segment with base offset {}", base_offset);
            }
        }

        result.bytes_freed = size_freed;
        self.bytes_cleaned.fetch_add(size_freed, Ordering::Relaxed);
        self.segments_deleted
            .fetch_add(result.segments_deleted as u64, Ordering::Relaxed);

        Ok(result)
    }

    /// Get metadata for all segments in a partition.
    fn get_segment_metadata(&self, partition: &Partition) -> Result<Vec<SegmentMetadata>> {
        // This would query the partition for its segments
        // For now, return an empty list as a placeholder
        // In a real implementation, this would iterate over partition.segments
        Ok(Vec::new())
    }

    /// Calculate cleanable ratio for a partition.
    pub fn cleanable_ratio(&self, partition: &Partition) -> Result<f64> {
        let segments = self.get_segment_metadata(partition)?;
        if segments.is_empty() {
            return Ok(0.0);
        }

        let total_bytes: u64 = segments.iter().map(|s| s.size).sum();
        let cleanable_bytes: u64 = segments.iter().map(|s| s.cleanable_bytes).sum();

        if total_bytes == 0 {
            Ok(0.0)
        } else {
            Ok(cleanable_bytes as f64 / total_bytes as f64)
        }
    }

    /// Get cleaner statistics.
    pub fn stats(&self) -> CleanerStats {
        CleanerStats {
            bytes_cleaned: self.bytes_cleaned.load(Ordering::Relaxed),
            segments_deleted: self.segments_deleted.load(Ordering::Relaxed),
            segments_compacted: self.segments_compacted.load(Ordering::Relaxed),
        }
    }
}

/// Result of a clean operation.
#[derive(Debug, Clone, Default)]
pub struct CleanResult {
    /// Segments deleted.
    pub segments_deleted: usize,
    /// Segments compacted.
    pub segments_compacted: usize,
    /// Bytes freed.
    pub bytes_freed: u64,
    /// Records removed.
    pub records_removed: u64,
}

/// Cleaner statistics.
#[derive(Debug, Clone, Default)]
pub struct CleanerStats {
    /// Total bytes cleaned.
    pub bytes_cleaned: u64,
    /// Total segments deleted.
    pub segments_deleted: u64,
    /// Total segments compacted.
    pub segments_compacted: u64,
}

/// Log compactor for key-based compaction.
pub struct LogCompactor {
    /// Configuration.
    config: CleanerConfig,
    /// Working directory for temp files.
    work_dir: PathBuf,
    /// Segment configuration.
    segment_config: SegmentConfig,
}

impl LogCompactor {
    /// Create a new log compactor.
    pub fn new(config: CleanerConfig, work_dir: PathBuf, segment_config: SegmentConfig) -> Self {
        Self {
            config,
            work_dir,
            segment_config,
        }
    }

    /// Compact segments for a partition.
    pub fn compact(&self, dir: &Path, segments: &[SegmentInfo]) -> Result<CompactionResult> {
        if segments.is_empty() {
            return Ok(CompactionResult::default());
        }

        info!("Starting compaction for {} segments", segments.len());

        // Phase 1: Build offset map (key -> latest offset with non-null value)
        let offset_map = self.build_offset_map(dir, segments)?;

        info!("Built offset map with {} unique keys", offset_map.len());

        // Phase 2: Rewrite segments keeping only latest values
        let result = self.rewrite_segments(dir, segments, &offset_map)?;

        info!(
            "Compaction complete: {} bytes freed, {} records removed",
            result.bytes_freed, result.records_removed
        );

        Ok(result)
    }

    /// Build map of key -> latest offset.
    fn build_offset_map(&self, dir: &Path, segments: &[SegmentInfo]) -> Result<HashMap<Vec<u8>, OffsetInfo>> {
        let mut offset_map: HashMap<Vec<u8>, OffsetInfo> = HashMap::new();

        for segment in segments {
            let batches = self.read_segment(dir, segment.base_offset)?;

            for batch in batches {
                for (i, record) in batch.records.iter().enumerate() {
                    let offset = batch.base_offset + i as u64;

                    if let Some(key) = &record.key {
                        let info = OffsetInfo {
                            offset,
                            is_tombstone: record.value.is_none(),
                            timestamp: batch.base_timestamp + record.timestamp_delta,
                        };

                        // Always keep latest offset for each key
                        offset_map.insert(key.clone(), info);
                    }
                }
            }
        }

        Ok(offset_map)
    }

    /// Read all batches from a segment.
    fn read_segment(&self, dir: &Path, base_offset: Offset) -> Result<Vec<RecordBatch>> {
        let log_path = dir.join(format!("{:020}.log", base_offset));
        let file = File::open(&log_path)?;
        let mut reader = BufReader::new(file);

        let mut batches = Vec::new();

        loop {
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
                warn!("CRC mismatch in segment {}", base_offset);
                break;
            }

            let batch = RecordBatch::deserialize(&data)?;
            batches.push(batch);
        }

        Ok(batches)
    }

    /// Rewrite segments keeping only latest values per key.
    fn rewrite_segments(
        &self,
        dir: &Path,
        segments: &[SegmentInfo],
        offset_map: &HashMap<Vec<u8>, OffsetInfo>,
    ) -> Result<CompactionResult> {
        let mut result = CompactionResult::default();
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let delete_horizon = now.saturating_sub(self.config.delete_retention_ms);

        // Create temp directory for new segments
        let temp_dir = self.work_dir.join("compaction_temp");
        fs::create_dir_all(&temp_dir)?;

        let mut new_segments: Vec<(Offset, PathBuf)> = Vec::new();
        let mut current_writer: Option<SegmentWriter> = None;
        let mut current_base: Offset = 0;

        for segment in segments {
            let batches = self.read_segment(dir, segment.base_offset)?;
            result.segments_read += 1;

            for batch in batches {
                let mut kept_records = Vec::new();

                for (i, record) in batch.records.into_iter().enumerate() {
                    let offset = batch.base_offset + i as u64;

                    let should_keep = match &record.key {
                        Some(key) => {
                            // Keep if this is the latest offset for this key
                            match offset_map.get(key) {
                                Some(info) if info.offset == offset => {
                                    // Check tombstone retention
                                    if info.is_tombstone && (info.timestamp as u64) < delete_horizon {
                                        result.tombstones_removed += 1;
                                        false
                                    } else {
                                        true
                                    }
                                }
                                _ => {
                                    result.records_removed += 1;
                                    false
                                }
                            }
                        }
                        None => {
                            // Records without keys are always kept
                            true
                        }
                    };

                    if should_keep {
                        kept_records.push(record);
                        result.records_kept += 1;
                    } else {
                        result.records_removed += 1;
                    }
                }

                if !kept_records.is_empty() {
                    // Write kept records
                    if current_writer.is_none() {
                        current_base = batch.base_offset;
                        current_writer = Some(SegmentWriter::new(
                            &temp_dir,
                            current_base,
                            self.segment_config.clone(),
                        )?);
                    }

                    let writer = current_writer.as_mut().unwrap();
                    let new_batch = RecordBatch {
                        base_offset: batch.base_offset,
                        batch_length: 0,
                        partition_leader_epoch: batch.partition_leader_epoch,
                        magic: batch.magic,
                        crc: 0,
                        attributes: batch.attributes,
                        last_offset_delta: (kept_records.len() as u32).saturating_sub(1),
                        base_timestamp: batch.base_timestamp,
                        max_timestamp: batch.max_timestamp,
                        producer_id: batch.producer_id,
                        producer_epoch: batch.producer_epoch,
                        base_sequence: batch.base_sequence,
                        records: kept_records,
                    };

                    writer.append(&new_batch)?;

                    // Roll segment if full
                    if writer.size() >= self.config.segment_size {
                        // Take ownership to call finish()
                        let owned_writer = current_writer.take().unwrap();
                        let path = owned_writer.finish()?;
                        new_segments.push((current_base, path));
                    }
                }
            }
        }

        // Finish last segment
        if let Some(writer) = current_writer {
            let path = writer.finish()?;
            new_segments.push((current_base, path));
        }

        result.segments_written = new_segments.len();

        // Calculate bytes freed
        let old_size: u64 = segments.iter().map(|s| s.size).sum();
        let new_size: u64 = new_segments
            .iter()
            .map(|(_, p)| fs::metadata(p).map(|m| m.len()).unwrap_or(0))
            .sum();
        result.bytes_freed = old_size.saturating_sub(new_size);

        // Swap old segments with new ones
        // In a real implementation, this would:
        // 1. Rename new segments to final names
        // 2. Delete old segment files
        // 3. Update partition metadata

        // Cleanup temp directory
        let _ = fs::remove_dir_all(&temp_dir);

        Ok(result)
    }
}

/// Information about a segment for compaction.
#[derive(Debug, Clone)]
pub struct SegmentInfo {
    /// Base offset.
    pub base_offset: Offset,
    /// Size in bytes.
    pub size: u64,
    /// Last modified time.
    pub last_modified: SystemTime,
}

/// Offset information for compaction.
#[derive(Debug, Clone)]
struct OffsetInfo {
    /// The offset.
    offset: Offset,
    /// Whether this is a tombstone (null value).
    is_tombstone: bool,
    /// Timestamp of the record.
    timestamp: Timestamp,
}

/// Result of compaction.
#[derive(Debug, Clone, Default)]
pub struct CompactionResult {
    /// Segments read.
    pub segments_read: usize,
    /// Segments written.
    pub segments_written: usize,
    /// Bytes freed.
    pub bytes_freed: u64,
    /// Records kept.
    pub records_kept: u64,
    /// Records removed (duplicates).
    pub records_removed: u64,
    /// Tombstones removed (expired).
    pub tombstones_removed: u64,
}

/// Segment writer for compaction output.
struct SegmentWriter {
    /// Base offset.
    base_offset: Offset,
    /// Output directory.
    dir: PathBuf,
    /// Log file.
    log_file: File,
    /// Index file.
    index_file: File,
    /// Time index file.
    time_index_file: File,
    /// Current size.
    size: u64,
    /// Bytes since last index entry.
    bytes_since_index: u64,
    /// Configuration.
    config: SegmentConfig,
}

impl SegmentWriter {
    /// Create a new segment writer.
    fn new(dir: &Path, base_offset: Offset, config: SegmentConfig) -> Result<Self> {
        let log_path = dir.join(format!("{:020}.log", base_offset));
        let index_path = dir.join(format!("{:020}.index", base_offset));
        let time_index_path = dir.join(format!("{:020}.timeindex", base_offset));

        let log_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&log_path)?;

        let index_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&index_path)?;

        let time_index_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&time_index_path)?;

        Ok(Self {
            base_offset,
            dir: dir.to_path_buf(),
            log_file,
            index_file,
            time_index_file,
            size: 0,
            bytes_since_index: 0,
            config,
        })
    }

    /// Append a batch to the segment.
    fn append(&mut self, batch: &RecordBatch) -> Result<()> {
        let data = batch.serialize()?;
        let length = data.len() as u32;
        let crc = crc32fast::hash(&data);

        // Write: [length: u32][crc: u32][data]
        self.log_file.write_all(&length.to_le_bytes())?;
        self.log_file.write_all(&crc.to_le_bytes())?;
        self.log_file.write_all(&data)?;

        let entry_size = 8 + data.len() as u64;

        // Update index if needed
        self.bytes_since_index += entry_size;
        if self.bytes_since_index >= self.config.index_interval_bytes {
            let entry = IndexEntry {
                relative_offset: (batch.base_offset - self.base_offset) as u32,
                position: self.size as u32,
            };
            self.index_file
                .write_all(&entry.relative_offset.to_le_bytes())?;
            self.index_file.write_all(&entry.position.to_le_bytes())?;
            self.bytes_since_index = 0;
        }

        // Write time index
        let time_entry = TimeIndexEntry {
            timestamp: batch.max_timestamp,
            relative_offset: (batch.base_offset - self.base_offset) as u32,
        };
        self.time_index_file
            .write_all(&time_entry.timestamp.to_le_bytes())?;
        self.time_index_file
            .write_all(&time_entry.relative_offset.to_le_bytes())?;

        self.size += entry_size;

        Ok(())
    }

    /// Get current size.
    fn size(&self) -> u64 {
        self.size
    }

    /// Finish writing and return the log file path.
    fn finish(mut self) -> Result<PathBuf> {
        self.log_file.sync_all()?;
        self.index_file.sync_all()?;
        self.time_index_file.sync_all()?;

        Ok(self.dir.join(format!("{:020}.log", self.base_offset)))
    }
}

/// Retention manager for scheduled cleanup.
pub struct RetentionManager {
    /// Cleaner.
    cleaner: LogCleaner,
    /// Compactor.
    compactor: Option<LogCompactor>,
    /// Running flag.
    running: AtomicBool,
    /// Last clean time.
    last_clean: std::sync::Mutex<Instant>,
}

impl RetentionManager {
    /// Create a new retention manager.
    pub fn new(config: CleanerConfig, work_dir: PathBuf, segment_config: SegmentConfig) -> Self {
        let compactor = if config.enable_compaction {
            Some(LogCompactor::new(
                config.clone(),
                work_dir,
                segment_config,
            ))
        } else {
            None
        };

        Self {
            cleaner: LogCleaner::new(config),
            compactor,
            running: AtomicBool::new(false),
            last_clean: std::sync::Mutex::new(Instant::now()),
        }
    }

    /// Start the retention manager.
    pub fn start(&self) {
        self.running.store(true, Ordering::SeqCst);
        self.cleaner.start();
        info!("Retention manager started");
    }

    /// Stop the retention manager.
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.cleaner.stop();
        info!("Retention manager stopped");
    }

    /// Check if it's time to clean.
    pub fn should_clean(&self) -> bool {
        let last = self.last_clean.lock().unwrap();
        let elapsed = last.elapsed().as_millis() as u64;
        elapsed >= self.cleaner.config.clean_interval_ms
    }

    /// Clean a partition.
    pub fn clean_partition(&self, partition: &mut Partition) -> Result<CleanResult> {
        *self.last_clean.lock().unwrap() = Instant::now();
        self.cleaner.clean_retention(partition)
    }

    /// Compact a partition.
    pub fn compact_partition(
        &self,
        dir: &Path,
        segments: &[SegmentInfo],
    ) -> Result<CompactionResult> {
        match &self.compactor {
            Some(compactor) => compactor.compact(dir, segments),
            None => Ok(CompactionResult::default()),
        }
    }

    /// Get cleaner stats.
    pub fn stats(&self) -> CleanerStats {
        self.cleaner.stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cleaner_config() {
        let config = CleanerConfig::default();
        assert_eq!(config.retention_ms, 7 * 24 * 60 * 60 * 1000);
        assert!(!config.enable_compaction);
    }

    #[test]
    fn test_log_cleaner_creation() {
        let config = CleanerConfig::default();
        let cleaner = LogCleaner::new(config);

        assert!(!cleaner.is_running());
        cleaner.start();
        assert!(cleaner.is_running());
        cleaner.stop();
        assert!(!cleaner.is_running());
    }

    #[test]
    fn test_cleaner_stats() {
        let config = CleanerConfig::default();
        let cleaner = LogCleaner::new(config);

        let stats = cleaner.stats();
        assert_eq!(stats.bytes_cleaned, 0);
        assert_eq!(stats.segments_deleted, 0);
    }

    #[test]
    fn test_segment_writer() {
        let dir = tempdir().unwrap();
        let config = SegmentConfig::default();
        let mut writer = SegmentWriter::new(dir.path(), 0, config).unwrap();

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

        writer.append(&batch).unwrap();
        assert!(writer.size() > 0);

        let path = writer.finish().unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_offset_map_building() {
        let dir = tempdir().unwrap();
        let config = CleanerConfig::default();
        let compactor = LogCompactor::new(
            config,
            dir.path().to_path_buf(),
            SegmentConfig::default(),
        );

        // Write a test segment
        let mut segment_writer = SegmentWriter::new(
            dir.path(),
            0,
            SegmentConfig::default(),
        ).unwrap();

        // First batch with key1
        let batch1 = RecordBatch::new(
            0,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(b"key1".to_vec()),
                value: Some(b"value1".to_vec()),
                headers: vec![],
            }],
        );
        segment_writer.append(&batch1).unwrap();

        // Second batch with same key (should replace)
        let batch2 = RecordBatch::new(
            1,
            vec![Record {
                attributes: 0,
                timestamp_delta: 0,
                offset_delta: 0,
                key: Some(b"key1".to_vec()),
                value: Some(b"value2".to_vec()),
                headers: vec![],
            }],
        );
        segment_writer.append(&batch2).unwrap();

        segment_writer.finish().unwrap();

        // Read and build offset map
        let segments = vec![SegmentInfo {
            base_offset: 0,
            size: 100,
            last_modified: SystemTime::now(),
        }];

        let offset_map = compactor.build_offset_map(dir.path(), &segments).unwrap();

        // Should have only one entry for key1, pointing to offset 1
        assert_eq!(offset_map.len(), 1);
        let info = offset_map.get(&b"key1".to_vec()).unwrap();
        assert_eq!(info.offset, 1);
    }

    #[test]
    fn test_retention_manager() {
        let dir = tempdir().unwrap();
        let config = CleanerConfig {
            clean_interval_ms: 100,
            ..CleanerConfig::default()
        };
        let manager = RetentionManager::new(
            config,
            dir.path().to_path_buf(),
            SegmentConfig::default(),
        );

        manager.start();
        assert!(manager.cleaner.is_running());

        // Initially should not need cleaning
        std::thread::sleep(Duration::from_millis(150));
        assert!(manager.should_clean());

        manager.stop();
    }

    #[test]
    fn test_cleanup_policy() {
        assert_eq!(CleanupPolicy::default(), CleanupPolicy::Delete);
    }

    #[test]
    fn test_compaction_result_default() {
        let result = CompactionResult::default();
        assert_eq!(result.segments_read, 0);
        assert_eq!(result.records_kept, 0);
        assert_eq!(result.bytes_freed, 0);
    }
}
