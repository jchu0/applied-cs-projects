//! Storage layer for message persistence.

use crate::error::{Error, Result};
use crate::message::{Message, MessageBatch};
use crate::segment::SegmentManager;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Storage configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Base directory for storage.
    pub base_dir: PathBuf,
    /// Maximum segment size in bytes.
    pub segment_bytes: u64,
    /// Retention time in milliseconds.
    pub retention_ms: u64,
    /// Retention size in bytes.
    pub retention_bytes: u64,
    /// Flush interval in milliseconds.
    pub flush_interval_ms: u64,
    /// Index interval in bytes.
    pub index_interval_bytes: u32,
    /// Enable fsync on append.
    pub sync_on_append: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("/tmp/mq/data"),
            segment_bytes: 1024 * 1024 * 1024, // 1GB
            retention_ms: 7 * 24 * 60 * 60 * 1000, // 7 days
            retention_bytes: u64::MAX,
            flush_interval_ms: 1000,
            index_interval_bytes: 4096,
            sync_on_append: false,
        }
    }
}

impl StorageConfig {
    /// Create a new storage config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set base directory.
    pub fn with_base_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.base_dir = dir.into();
        self
    }

    /// Set segment size.
    pub fn with_segment_bytes(mut self, bytes: u64) -> Self {
        self.segment_bytes = bytes;
        self
    }

    /// Set retention time.
    pub fn with_retention_ms(mut self, ms: u64) -> Self {
        self.retention_ms = ms;
        self
    }

    /// Set retention size.
    pub fn with_retention_bytes(mut self, bytes: u64) -> Self {
        self.retention_bytes = bytes;
        self
    }
}

/// Storage statistics.
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total bytes written.
    pub bytes_written: u64,
    /// Total bytes read.
    pub bytes_read: u64,
    /// Total messages written.
    pub messages_written: u64,
    /// Total messages read.
    pub messages_read: u64,
    /// Number of flushes.
    pub flush_count: u64,
    /// Total flush time (nanoseconds).
    pub flush_time_ns: u64,
}

impl StorageStats {
    /// Record a write operation.
    pub fn record_write(&mut self, bytes: u64, messages: u64) {
        self.bytes_written += bytes;
        self.messages_written += messages;
    }

    /// Record a read operation.
    pub fn record_read(&mut self, bytes: u64, messages: u64) {
        self.bytes_read += bytes;
        self.messages_read += messages;
    }

    /// Record a flush operation.
    pub fn record_flush(&mut self, duration: Duration) {
        self.flush_count += 1;
        self.flush_time_ns += duration.as_nanos() as u64;
    }

    /// Get average flush time.
    pub fn avg_flush_time(&self) -> Duration {
        if self.flush_count == 0 {
            Duration::ZERO
        } else {
            Duration::from_nanos(self.flush_time_ns / self.flush_count)
        }
    }
}

/// Storage engine for a single partition.
pub struct Storage {
    /// Configuration.
    config: StorageConfig,
    /// Segment manager.
    segment_manager: SegmentManager,
    /// Statistics.
    stats: RwLock<StorageStats>,
    /// Last flush time.
    last_flush: RwLock<Instant>,
    /// Pending bytes since last flush.
    pending_bytes: AtomicU64,
}

impl Storage {
    /// Create a new storage engine.
    pub fn new(config: StorageConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.base_dir)?;

        let segment_manager = SegmentManager::new(&config.base_dir, config.segment_bytes)?;

        Ok(Self {
            config,
            segment_manager,
            stats: RwLock::new(StorageStats::default()),
            last_flush: RwLock::new(Instant::now()),
            pending_bytes: AtomicU64::new(0),
        })
    }

    /// Open existing storage.
    pub fn open(config: StorageConfig) -> Result<Self> {
        if !config.base_dir.exists() {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Storage directory not found",
            )));
        }

        let segment_manager = SegmentManager::new(&config.base_dir, config.segment_bytes)?;

        Ok(Self {
            config,
            segment_manager,
            stats: RwLock::new(StorageStats::default()),
            last_flush: RwLock::new(Instant::now()),
            pending_bytes: AtomicU64::new(0),
        })
    }

    /// Append a message.
    pub fn append(&self, message: &Message) -> Result<u64> {
        let size = message.size() as u64;
        let offset = self.segment_manager.append(message)?;

        self.pending_bytes.fetch_add(size, Ordering::Relaxed);
        self.stats.write().record_write(size, 1);

        // Check if we need to flush
        if self.config.sync_on_append || self.should_flush() {
            self.flush()?;
        }

        Ok(offset)
    }

    /// Append a batch of messages.
    pub fn append_batch(&self, messages: &[Message]) -> Result<Vec<u64>> {
        let mut offsets = Vec::with_capacity(messages.len());
        let mut total_size = 0u64;

        for message in messages {
            let size = message.size() as u64;
            let offset = self.segment_manager.append(message)?;
            offsets.push(offset);
            total_size += size;
        }

        self.pending_bytes.fetch_add(total_size, Ordering::Relaxed);
        self.stats.write().record_write(total_size, messages.len() as u64);

        if self.should_flush() {
            self.flush()?;
        }

        Ok(offsets)
    }

    /// Read a message by offset.
    pub fn read(&self, offset: u64) -> Result<Message> {
        let message = self.segment_manager.read(offset)?;
        self.stats.write().record_read(message.size() as u64, 1);
        Ok(message)
    }

    /// Read messages in a range.
    pub fn read_range(&self, start_offset: u64, max_messages: usize) -> Result<Vec<Message>> {
        let messages = self.segment_manager.read_range(start_offset, max_messages)?;
        let total_size: u64 = messages.iter().map(|m| m.size() as u64).sum();
        self.stats.write().record_read(total_size, messages.len() as u64);
        Ok(messages)
    }

    /// Read messages up to a byte limit.
    pub fn read_bytes(&self, start_offset: u64, max_bytes: usize) -> Result<Vec<Message>> {
        let mut messages = Vec::new();
        let mut total_bytes = 0;
        let mut current_offset = start_offset;

        while total_bytes < max_bytes {
            match self.segment_manager.read(current_offset) {
                Ok(msg) => {
                    let msg_size = msg.size();
                    if total_bytes + msg_size > max_bytes && !messages.is_empty() {
                        break;
                    }
                    total_bytes += msg_size;
                    messages.push(msg);
                    current_offset += 1;
                }
                Err(Error::OffsetOutOfRange { .. }) => break,
                Err(e) => return Err(e),
            }
        }

        self.stats.write().record_read(total_bytes as u64, messages.len() as u64);
        Ok(messages)
    }

    /// Check if flush is needed.
    fn should_flush(&self) -> bool {
        let last_flush = *self.last_flush.read();
        let elapsed = last_flush.elapsed();
        elapsed.as_millis() >= self.config.flush_interval_ms as u128
    }

    /// Flush to disk.
    pub fn flush(&self) -> Result<()> {
        let start = Instant::now();
        self.segment_manager.flush()?;
        let duration = start.elapsed();

        self.pending_bytes.store(0, Ordering::Relaxed);
        *self.last_flush.write() = Instant::now();
        self.stats.write().record_flush(duration);

        Ok(())
    }

    /// Get start offset.
    pub fn start_offset(&self) -> u64 {
        self.segment_manager.start_offset()
    }

    /// Get end offset (next to be written).
    pub fn end_offset(&self) -> u64 {
        self.segment_manager.end_offset()
    }

    /// Get high watermark.
    pub fn high_watermark(&self) -> u64 {
        self.segment_manager.high_watermark()
    }

    /// Get total size.
    pub fn total_size(&self) -> u64 {
        self.segment_manager.total_size()
    }

    /// Get segment count.
    pub fn segment_count(&self) -> usize {
        self.segment_manager.segment_count()
    }

    /// Get statistics.
    pub fn stats(&self) -> StorageStats {
        self.stats.read().clone()
    }

    /// Apply retention policy.
    pub fn apply_retention(&self) -> Result<RetentionResult> {
        let mut result = RetentionResult::default();

        // Time-based retention
        let cutoff_time = crate::message::current_timestamp()
            .saturating_sub(self.config.retention_ms);

        // Find oldest offset to keep
        let start = self.start_offset();
        let end = self.end_offset();

        let mut delete_before = start;
        for offset in start..end {
            match self.read(offset) {
                Ok(msg) => {
                    if msg.timestamp >= cutoff_time {
                        break;
                    }
                    delete_before = offset + 1;
                }
                Err(_) => break,
            }
        }

        // Size-based retention
        let mut total_size = self.total_size();
        while total_size > self.config.retention_bytes && delete_before < end {
            if let Ok(msg) = self.read(delete_before) {
                total_size -= msg.size() as u64;
                delete_before += 1;
            } else {
                break;
            }
        }

        // Delete segments
        if delete_before > start {
            result.segments_deleted = self.segment_manager.delete_before(delete_before)?;
            result.offsets_deleted = delete_before - start;
        }

        Ok(result)
    }

    /// Get configuration.
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }
}

/// Result of retention policy application.
#[derive(Debug, Clone, Default)]
pub struct RetentionResult {
    /// Number of segments deleted.
    pub segments_deleted: usize,
    /// Number of offsets deleted.
    pub offsets_deleted: u64,
}

/// Storage manager for multiple partitions.
pub struct StorageManager {
    /// Base directory.
    base_dir: PathBuf,
    /// Default configuration.
    default_config: StorageConfig,
    /// Storage instances by topic-partition.
    storages: RwLock<HashMap<String, Arc<Storage>>>,
}

impl StorageManager {
    /// Create a new storage manager.
    pub fn new(base_dir: impl Into<PathBuf>, config: StorageConfig) -> Result<Self> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)?;

        Ok(Self {
            base_dir,
            default_config: config,
            storages: RwLock::new(HashMap::new()),
        })
    }

    /// Get or create storage for a topic-partition.
    pub fn get_or_create(&self, topic: &str, partition: u32) -> Result<Arc<Storage>> {
        let key = format!("{}-{}", topic, partition);

        // Check if exists
        {
            let storages = self.storages.read();
            if let Some(storage) = storages.get(&key) {
                return Ok(Arc::clone(storage));
            }
        }

        // Create new storage
        let partition_dir = self.base_dir.join(topic).join(partition.to_string());
        let config = StorageConfig {
            base_dir: partition_dir,
            ..self.default_config.clone()
        };

        let storage = Arc::new(Storage::new(config)?);

        let mut storages = self.storages.write();
        storages.insert(key.clone(), Arc::clone(&storage));

        Ok(storage)
    }

    /// Get existing storage.
    pub fn get(&self, topic: &str, partition: u32) -> Option<Arc<Storage>> {
        let key = format!("{}-{}", topic, partition);
        self.storages.read().get(&key).cloned()
    }

    /// Remove storage.
    pub fn remove(&self, topic: &str, partition: u32) -> Option<Arc<Storage>> {
        let key = format!("{}-{}", topic, partition);
        self.storages.write().remove(&key)
    }

    /// Flush all storages.
    pub fn flush_all(&self) -> Result<()> {
        let storages = self.storages.read();
        for storage in storages.values() {
            storage.flush()?;
        }
        Ok(())
    }

    /// Apply retention to all storages.
    pub fn apply_retention_all(&self) -> Result<Vec<RetentionResult>> {
        let storages = self.storages.read();
        let mut results = Vec::new();
        for storage in storages.values() {
            results.push(storage.apply_retention()?);
        }
        Ok(results)
    }

    /// Get total size across all storages.
    pub fn total_size(&self) -> u64 {
        let storages = self.storages.read();
        storages.values().map(|s| s.total_size()).sum()
    }

    /// Get storage count.
    pub fn storage_count(&self) -> usize {
        self.storages.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_message(content: &str) -> Message {
        Message::new(content.as_bytes().to_vec())
    }

    #[test]
    fn test_storage_config() {
        let config = StorageConfig::new()
            .with_segment_bytes(1024 * 1024)
            .with_retention_ms(3600000);

        assert_eq!(config.segment_bytes, 1024 * 1024);
        assert_eq!(config.retention_ms, 3600000);
    }

    #[test]
    fn test_storage_create() {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::new().with_base_dir(dir.path());
        let storage = Storage::new(config).unwrap();

        assert_eq!(storage.start_offset(), 0);
        assert_eq!(storage.end_offset(), 0);
    }

    #[test]
    fn test_storage_append_read() {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::new().with_base_dir(dir.path());
        let storage = Storage::new(config).unwrap();

        let msg1 = create_test_message("Hello");
        let msg2 = create_test_message("World");

        let offset1 = storage.append(&msg1).unwrap();
        let offset2 = storage.append(&msg2).unwrap();

        assert_eq!(offset1, 0);
        assert_eq!(offset2, 1);

        let read1 = storage.read(0).unwrap();
        let read2 = storage.read(1).unwrap();

        assert_eq!(read1.payload, msg1.payload);
        assert_eq!(read2.payload, msg2.payload);
    }

    #[test]
    fn test_storage_append_batch() {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::new().with_base_dir(dir.path());
        let storage = Storage::new(config).unwrap();

        let messages: Vec<Message> = (0..5)
            .map(|i| create_test_message(&format!("Message {}", i)))
            .collect();

        let offsets = storage.append_batch(&messages).unwrap();

        assert_eq!(offsets, vec![0, 1, 2, 3, 4]);
        assert_eq!(storage.end_offset(), 5);
    }

    #[test]
    fn test_storage_read_range() {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::new().with_base_dir(dir.path());
        let storage = Storage::new(config).unwrap();

        for i in 0..10 {
            storage.append(&create_test_message(&format!("Message {}", i))).unwrap();
        }

        let messages = storage.read_range(3, 5).unwrap();
        assert_eq!(messages.len(), 5);

        for (i, msg) in messages.iter().enumerate() {
            let expected = format!("Message {}", i + 3);
            assert_eq!(msg.payload.as_ref(), expected.as_bytes());
        }
    }

    #[test]
    fn test_storage_read_bytes() {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::new().with_base_dir(dir.path());
        let storage = Storage::new(config).unwrap();

        for i in 0..10 {
            storage.append(&create_test_message(&format!("Message {}", i))).unwrap();
        }

        // Read with byte limit
        let messages = storage.read_bytes(0, 200).unwrap();
        assert!(!messages.is_empty());
    }

    #[test]
    fn test_storage_stats() {
        let dir = TempDir::new().unwrap();
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

    #[test]
    fn test_storage_flush() {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::new()
            .with_base_dir(dir.path())
            .with_segment_bytes(1024 * 1024);
        let storage = Storage::new(config).unwrap();

        storage.append(&create_test_message("Test")).unwrap();
        storage.flush().unwrap();

        let stats = storage.stats();
        assert_eq!(stats.flush_count, 1);
    }

    #[test]
    fn test_storage_manager() {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::new();
        let manager = StorageManager::new(dir.path(), config).unwrap();

        let storage1 = manager.get_or_create("topic1", 0).unwrap();
        let storage2 = manager.get_or_create("topic1", 1).unwrap();
        let storage3 = manager.get_or_create("topic1", 0).unwrap();

        // Same partition should return same storage
        assert!(Arc::ptr_eq(&storage1, &storage3));
        assert!(!Arc::ptr_eq(&storage1, &storage2));

        assert_eq!(manager.storage_count(), 2);
    }

    #[test]
    fn test_storage_manager_operations() {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::new();
        let manager = StorageManager::new(dir.path(), config).unwrap();

        let storage = manager.get_or_create("test", 0).unwrap();
        storage.append(&create_test_message("Hello")).unwrap();

        manager.flush_all().unwrap();

        assert!(manager.total_size() > 0);
    }

    #[test]
    fn test_storage_high_watermark() {
        let dir = TempDir::new().unwrap();
        let config = StorageConfig::new().with_base_dir(dir.path());
        let storage = Storage::new(config).unwrap();

        assert_eq!(storage.high_watermark(), 0);

        storage.append(&create_test_message("Test1")).unwrap();
        assert_eq!(storage.high_watermark(), 0);

        storage.append(&create_test_message("Test2")).unwrap();
        assert_eq!(storage.high_watermark(), 1);
    }

    #[test]
    fn test_storage_reopen() {
        let dir = TempDir::new().unwrap();

        // Create and write
        {
            let config = StorageConfig::new().with_base_dir(dir.path());
            let storage = Storage::new(config).unwrap();
            storage.append(&create_test_message("Hello")).unwrap();
            storage.append(&create_test_message("World")).unwrap();
            storage.flush().unwrap();
        }

        // Reopen and verify
        {
            let config = StorageConfig::new().with_base_dir(dir.path());
            let storage = Storage::open(config).unwrap();
            assert_eq!(storage.end_offset(), 2);

            let msg = storage.read(0).unwrap();
            assert_eq!(msg.payload.as_ref(), b"Hello");
        }
    }

    #[test]
    fn test_retention_result() {
        let result = RetentionResult {
            segments_deleted: 5,
            offsets_deleted: 1000,
        };

        assert_eq!(result.segments_deleted, 5);
        assert_eq!(result.offsets_deleted, 1000);
    }

    #[test]
    fn test_storage_stats_operations() {
        let mut stats = StorageStats::default();

        stats.record_write(100, 1);
        stats.record_write(200, 2);
        assert_eq!(stats.bytes_written, 300);
        assert_eq!(stats.messages_written, 3);

        stats.record_read(50, 1);
        assert_eq!(stats.bytes_read, 50);
        assert_eq!(stats.messages_read, 1);

        stats.record_flush(Duration::from_millis(10));
        stats.record_flush(Duration::from_millis(20));
        assert_eq!(stats.flush_count, 2);
        assert_eq!(stats.avg_flush_time(), Duration::from_millis(15));
    }
}
