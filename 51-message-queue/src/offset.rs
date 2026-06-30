//! Offset management for consumers and partitions.

use crate::error::{Error, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;

/// Special offset values.
pub mod offsets {
    /// Start from the beginning.
    pub const EARLIEST: i64 = -2;
    /// Start from the end.
    pub const LATEST: i64 = -1;
    /// Invalid offset.
    pub const INVALID: i64 = -1000;
}

/// Offset metadata for a consumer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffsetMetadata {
    /// The committed offset.
    pub offset: u64,
    /// Optional metadata string.
    pub metadata: Option<String>,
    /// Timestamp when offset was committed.
    pub timestamp: u64,
    /// Leader epoch at commit time.
    pub leader_epoch: Option<u32>,
}

impl OffsetMetadata {
    /// Create new offset metadata.
    pub fn new(offset: u64) -> Self {
        Self {
            offset,
            metadata: None,
            timestamp: crate::message::current_timestamp(),
            leader_epoch: None,
        }
    }

    /// Create with metadata.
    pub fn with_metadata(offset: u64, metadata: impl Into<String>) -> Self {
        Self {
            offset,
            metadata: Some(metadata.into()),
            timestamp: crate::message::current_timestamp(),
            leader_epoch: None,
        }
    }
}

/// Topic-partition identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TopicPartition {
    /// Topic name.
    pub topic: String,
    /// Partition ID.
    pub partition: u32,
}

impl TopicPartition {
    /// Create a new topic-partition.
    pub fn new(topic: impl Into<String>, partition: u32) -> Self {
        Self {
            topic: topic.into(),
            partition,
        }
    }
}

impl std::fmt::Display for TopicPartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.topic, self.partition)
    }
}

/// Offset storage for a consumer group.
#[derive(Debug)]
pub struct OffsetStore {
    /// Group ID.
    group_id: String,
    /// Storage directory.
    dir: PathBuf,
    /// In-memory offsets.
    offsets: RwLock<HashMap<TopicPartition, OffsetMetadata>>,
    /// Dirty flag.
    dirty: RwLock<bool>,
}

impl OffsetStore {
    /// Create a new offset store.
    pub fn new(group_id: impl Into<String>, dir: impl Into<PathBuf>) -> Result<Self> {
        let group_id = group_id.into();
        let dir = dir.into();

        std::fs::create_dir_all(&dir)?;

        let store = Self {
            group_id,
            dir,
            offsets: RwLock::new(HashMap::new()),
            dirty: RwLock::new(false),
        };

        store.load()?;

        Ok(store)
    }

    /// Get offset for a topic-partition.
    pub fn get(&self, tp: &TopicPartition) -> Option<OffsetMetadata> {
        self.offsets.read().get(tp).cloned()
    }

    /// Get committed offset for a topic-partition.
    pub fn get_offset(&self, tp: &TopicPartition) -> Option<u64> {
        self.offsets.read().get(tp).map(|m| m.offset)
    }

    /// Commit an offset.
    pub fn commit(&self, tp: TopicPartition, offset: u64) -> Result<()> {
        self.commit_with_metadata(tp, OffsetMetadata::new(offset))
    }

    /// Commit an offset with metadata.
    pub fn commit_with_metadata(&self, tp: TopicPartition, metadata: OffsetMetadata) -> Result<()> {
        {
            let mut offsets = self.offsets.write();
            offsets.insert(tp, metadata);
            *self.dirty.write() = true;
        }
        Ok(())
    }

    /// Remove offset for a topic-partition.
    pub fn remove(&self, tp: &TopicPartition) -> Option<OffsetMetadata> {
        let mut offsets = self.offsets.write();
        let removed = offsets.remove(tp);
        if removed.is_some() {
            *self.dirty.write() = true;
        }
        removed
    }

    /// Get all topic-partitions.
    pub fn topic_partitions(&self) -> Vec<TopicPartition> {
        self.offsets.read().keys().cloned().collect()
    }

    /// Get all offsets.
    pub fn all_offsets(&self) -> HashMap<TopicPartition, OffsetMetadata> {
        self.offsets.read().clone()
    }

    /// Flush offsets to disk.
    pub fn flush(&self) -> Result<()> {
        if !*self.dirty.read() {
            return Ok(());
        }

        let path = self.dir.join(format!("{}.offsets", self.group_id));
        let temp_path = self.dir.join(format!("{}.offsets.tmp", self.group_id));

        let offsets = self.offsets.read();

        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)?;

        for (tp, metadata) in offsets.iter() {
            let line = serde_json::to_string(&(tp, metadata))
                .map_err(|e| Error::Serialization(e.to_string()))?;
            writeln!(file, "{}", line)?;
        }

        file.sync_all()?;
        drop(file);

        std::fs::rename(&temp_path, &path)?;

        *self.dirty.write() = false;

        Ok(())
    }

    /// Load offsets from disk.
    fn load(&self) -> Result<()> {
        let path = self.dir.join(format!("{}.offsets", self.group_id));

        if !path.exists() {
            return Ok(());
        }

        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        let mut offsets = self.offsets.write();

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }

            let (tp, metadata): (TopicPartition, OffsetMetadata) =
                serde_json::from_str(&line)
                    .map_err(|e| Error::Serialization(e.to_string()))?;
            offsets.insert(tp, metadata);
        }

        Ok(())
    }

    /// Clear all offsets.
    pub fn clear(&self) {
        let mut offsets = self.offsets.write();
        offsets.clear();
        *self.dirty.write() = true;
    }

    /// Get number of tracked topic-partitions.
    pub fn len(&self) -> usize {
        self.offsets.read().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.offsets.read().is_empty()
    }
}

impl Drop for OffsetStore {
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

/// Offset tracker for a consumer.
#[derive(Debug)]
pub struct OffsetTracker {
    /// Current positions (next offset to fetch).
    positions: RwLock<HashMap<TopicPartition, u64>>,
    /// Committed offsets.
    committed: RwLock<HashMap<TopicPartition, u64>>,
    /// High watermarks.
    high_watermarks: RwLock<HashMap<TopicPartition, u64>>,
    /// Log end offsets.
    log_end_offsets: RwLock<HashMap<TopicPartition, u64>>,
}

impl OffsetTracker {
    /// Create a new offset tracker.
    pub fn new() -> Self {
        Self {
            positions: RwLock::new(HashMap::new()),
            committed: RwLock::new(HashMap::new()),
            high_watermarks: RwLock::new(HashMap::new()),
            log_end_offsets: RwLock::new(HashMap::new()),
        }
    }

    /// Get current position.
    pub fn position(&self, tp: &TopicPartition) -> Option<u64> {
        self.positions.read().get(tp).copied()
    }

    /// Set position.
    pub fn set_position(&self, tp: TopicPartition, offset: u64) {
        self.positions.write().insert(tp, offset);
    }

    /// Update position after fetch.
    pub fn update_position(&self, tp: &TopicPartition, next_offset: u64) {
        self.positions.write().insert(tp.clone(), next_offset);
    }

    /// Get committed offset.
    pub fn committed(&self, tp: &TopicPartition) -> Option<u64> {
        self.committed.read().get(tp).copied()
    }

    /// Set committed offset.
    pub fn set_committed(&self, tp: TopicPartition, offset: u64) {
        self.committed.write().insert(tp, offset);
    }

    /// Get high watermark.
    pub fn high_watermark(&self, tp: &TopicPartition) -> Option<u64> {
        self.high_watermarks.read().get(tp).copied()
    }

    /// Update high watermark.
    pub fn update_high_watermark(&self, tp: TopicPartition, offset: u64) {
        self.high_watermarks.write().insert(tp, offset);
    }

    /// Get log end offset.
    pub fn log_end_offset(&self, tp: &TopicPartition) -> Option<u64> {
        self.log_end_offsets.read().get(tp).copied()
    }

    /// Update log end offset.
    pub fn update_log_end_offset(&self, tp: TopicPartition, offset: u64) {
        self.log_end_offsets.write().insert(tp, offset);
    }

    /// Get lag for a partition.
    pub fn lag(&self, tp: &TopicPartition) -> Option<u64> {
        let position = self.position(tp)?;
        let high_watermark = self.high_watermark(tp)?;
        Some(high_watermark.saturating_sub(position))
    }

    /// Get total lag across all partitions.
    pub fn total_lag(&self) -> u64 {
        let positions = self.positions.read();
        let high_watermarks = self.high_watermarks.read();

        positions
            .iter()
            .map(|(tp, pos)| {
                high_watermarks
                    .get(tp)
                    .map(|hw| hw.saturating_sub(*pos))
                    .unwrap_or(0)
            })
            .sum()
    }

    /// Get all positions.
    pub fn all_positions(&self) -> HashMap<TopicPartition, u64> {
        self.positions.read().clone()
    }

    /// Get all committed offsets.
    pub fn all_committed(&self) -> HashMap<TopicPartition, u64> {
        self.committed.read().clone()
    }

    /// Reset tracker for a partition.
    pub fn reset(&self, tp: &TopicPartition) {
        self.positions.write().remove(tp);
        self.committed.write().remove(tp);
        self.high_watermarks.write().remove(tp);
        self.log_end_offsets.write().remove(tp);
    }

    /// Clear all tracking data.
    pub fn clear(&self) {
        self.positions.write().clear();
        self.committed.write().clear();
        self.high_watermarks.write().clear();
        self.log_end_offsets.write().clear();
    }

    /// Seek to a specific offset.
    pub fn seek(&self, tp: TopicPartition, offset: u64) {
        self.positions.write().insert(tp, offset);
    }

    /// Seek to beginning.
    pub fn seek_to_beginning(&self, tp: &TopicPartition) {
        if let Some(&leo) = self.log_end_offsets.read().get(tp) {
            // Assuming log start is 0 for simplicity
            self.positions.write().insert(tp.clone(), 0);
        }
    }

    /// Seek to end.
    pub fn seek_to_end(&self, tp: &TopicPartition) {
        if let Some(&leo) = self.log_end_offsets.read().get(tp) {
            self.positions.write().insert(tp.clone(), leo);
        }
    }
}

impl Default for OffsetTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Commit synchronization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitMode {
    /// Synchronous commit (wait for confirmation).
    Sync,
    /// Asynchronous commit (fire and forget).
    Async,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_topic_partition() {
        let tp = TopicPartition::new("test-topic", 0);
        assert_eq!(tp.topic, "test-topic");
        assert_eq!(tp.partition, 0);
        assert_eq!(tp.to_string(), "test-topic-0");
    }

    #[test]
    fn test_offset_metadata() {
        let meta = OffsetMetadata::new(100);
        assert_eq!(meta.offset, 100);
        assert!(meta.metadata.is_none());
        assert!(meta.timestamp > 0);

        let meta = OffsetMetadata::with_metadata(200, "some-metadata");
        assert_eq!(meta.offset, 200);
        assert_eq!(meta.metadata, Some("some-metadata".to_string()));
    }

    #[test]
    fn test_offset_store() {
        let dir = TempDir::new().unwrap();
        let store = OffsetStore::new("test-group", dir.path()).unwrap();

        let tp = TopicPartition::new("test-topic", 0);

        // Initially empty
        assert!(store.get(&tp).is_none());
        assert!(store.is_empty());

        // Commit offset
        store.commit(tp.clone(), 100).unwrap();

        // Get offset
        let meta = store.get(&tp).unwrap();
        assert_eq!(meta.offset, 100);
        assert_eq!(store.len(), 1);

        // Update offset
        store.commit(tp.clone(), 200).unwrap();
        assert_eq!(store.get_offset(&tp), Some(200));

        // Flush
        store.flush().unwrap();

        // Remove
        let removed = store.remove(&tp);
        assert!(removed.is_some());
        assert!(store.get(&tp).is_none());
    }

    #[test]
    fn test_offset_store_persistence() {
        let dir = TempDir::new().unwrap();
        let tp = TopicPartition::new("test-topic", 0);

        // Create store and commit
        {
            let store = OffsetStore::new("test-group", dir.path()).unwrap();
            store.commit(tp.clone(), 100).unwrap();
            store.flush().unwrap();
        }

        // Reload and verify
        {
            let store = OffsetStore::new("test-group", dir.path()).unwrap();
            assert_eq!(store.get_offset(&tp), Some(100));
        }
    }

    #[test]
    fn test_offset_tracker() {
        let tracker = OffsetTracker::new();
        let tp = TopicPartition::new("test-topic", 0);

        // Set position
        tracker.set_position(tp.clone(), 100);
        assert_eq!(tracker.position(&tp), Some(100));

        // Set committed
        tracker.set_committed(tp.clone(), 50);
        assert_eq!(tracker.committed(&tp), Some(50));

        // Set high watermark
        tracker.update_high_watermark(tp.clone(), 200);
        assert_eq!(tracker.high_watermark(&tp), Some(200));

        // Calculate lag
        assert_eq!(tracker.lag(&tp), Some(100)); // 200 - 100

        // Seek
        tracker.seek(tp.clone(), 150);
        assert_eq!(tracker.position(&tp), Some(150));
        assert_eq!(tracker.lag(&tp), Some(50)); // 200 - 150

        // Reset
        tracker.reset(&tp);
        assert!(tracker.position(&tp).is_none());
    }

    #[test]
    fn test_offset_tracker_total_lag() {
        let tracker = OffsetTracker::new();

        let tp1 = TopicPartition::new("topic", 0);
        let tp2 = TopicPartition::new("topic", 1);

        tracker.set_position(tp1.clone(), 100);
        tracker.set_position(tp2.clone(), 200);

        tracker.update_high_watermark(tp1.clone(), 150);
        tracker.update_high_watermark(tp2.clone(), 300);

        // Lag: (150-100) + (300-200) = 50 + 100 = 150
        assert_eq!(tracker.total_lag(), 150);
    }

    #[test]
    fn test_offset_store_multiple_partitions() {
        let dir = TempDir::new().unwrap();
        let store = OffsetStore::new("test-group", dir.path()).unwrap();

        let tp1 = TopicPartition::new("topic", 0);
        let tp2 = TopicPartition::new("topic", 1);
        let tp3 = TopicPartition::new("topic", 2);

        store.commit(tp1.clone(), 100).unwrap();
        store.commit(tp2.clone(), 200).unwrap();
        store.commit(tp3.clone(), 300).unwrap();

        assert_eq!(store.len(), 3);

        let all = store.all_offsets();
        assert_eq!(all.len(), 3);
        assert_eq!(all.get(&tp1).unwrap().offset, 100);
        assert_eq!(all.get(&tp2).unwrap().offset, 200);
        assert_eq!(all.get(&tp3).unwrap().offset, 300);
    }

    #[test]
    fn test_offset_tracker_seek_operations() {
        let tracker = OffsetTracker::new();
        let tp = TopicPartition::new("topic", 0);

        // Set log end offset
        tracker.update_log_end_offset(tp.clone(), 1000);

        // Seek to end
        tracker.seek_to_end(&tp);
        assert_eq!(tracker.position(&tp), Some(1000));

        // Seek to beginning
        tracker.seek_to_beginning(&tp);
        assert_eq!(tracker.position(&tp), Some(0));

        // Manual seek
        tracker.seek(tp.clone(), 500);
        assert_eq!(tracker.position(&tp), Some(500));
    }

    #[test]
    fn test_offset_store_clear() {
        let dir = TempDir::new().unwrap();
        let store = OffsetStore::new("test-group", dir.path()).unwrap();

        store.commit(TopicPartition::new("topic", 0), 100).unwrap();
        store.commit(TopicPartition::new("topic", 1), 200).unwrap();

        assert_eq!(store.len(), 2);

        store.clear();
        assert!(store.is_empty());
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
}
