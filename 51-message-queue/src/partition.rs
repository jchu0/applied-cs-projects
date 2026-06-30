//! Partition management.

use crate::config::TopicConfig;
use crate::error::{Error, Result};
use crate::message::Message;
use crate::storage::{Storage, StorageConfig};
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use xxhash_rust::xxh3::xxh3_64;

/// Partition identifier.
pub type PartitionId = u32;

/// A partition within a topic.
pub struct Partition {
    /// Topic name.
    topic: String,
    /// Partition ID.
    id: PartitionId,
    /// Storage engine.
    storage: Storage,
    /// High watermark (last committed offset).
    high_watermark: AtomicU64,
    /// Log end offset.
    log_end_offset: AtomicU64,
    /// Leader epoch.
    leader_epoch: AtomicU32,
    /// Is this partition the leader.
    is_leader: bool,
}

impl Partition {
    /// Create a new partition.
    pub fn new(
        topic: impl Into<String>,
        id: PartitionId,
        base_dir: impl Into<PathBuf>,
        config: &TopicConfig,
    ) -> Result<Self> {
        let topic = topic.into();
        let base_dir = base_dir.into();
        let partition_dir = base_dir.join(&topic).join(id.to_string());

        let storage_config = StorageConfig {
            base_dir: partition_dir,
            segment_bytes: config.segment_bytes,
            retention_ms: config.retention_ms,
            retention_bytes: config.retention_bytes,
            ..StorageConfig::default()
        };

        let storage = Storage::new(storage_config)?;
        let log_end = storage.end_offset();

        Ok(Self {
            topic,
            id,
            storage,
            high_watermark: AtomicU64::new(log_end.saturating_sub(1)),
            log_end_offset: AtomicU64::new(log_end),
            leader_epoch: AtomicU32::new(0),
            is_leader: true,
        })
    }

    /// Open an existing partition.
    pub fn open(
        topic: impl Into<String>,
        id: PartitionId,
        base_dir: impl Into<PathBuf>,
        config: &TopicConfig,
    ) -> Result<Self> {
        let topic = topic.into();
        let base_dir = base_dir.into();
        let partition_dir = base_dir.join(&topic).join(id.to_string());

        let storage_config = StorageConfig {
            base_dir: partition_dir,
            segment_bytes: config.segment_bytes,
            retention_ms: config.retention_ms,
            retention_bytes: config.retention_bytes,
            ..StorageConfig::default()
        };

        let storage = Storage::open(storage_config)?;
        let log_end = storage.end_offset();

        Ok(Self {
            topic,
            id,
            storage,
            high_watermark: AtomicU64::new(log_end.saturating_sub(1)),
            log_end_offset: AtomicU64::new(log_end),
            leader_epoch: AtomicU32::new(0),
            is_leader: true,
        })
    }

    /// Get topic name.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Get partition ID.
    pub fn id(&self) -> PartitionId {
        self.id
    }

    /// Append a message.
    pub fn append(&self, mut message: Message) -> Result<u64> {
        if !self.is_leader {
            return Err(Error::IllegalState("Not the leader".to_string()));
        }

        message.partition = self.id;
        let offset = self.storage.append(&message)?;

        self.log_end_offset.store(offset + 1, Ordering::Release);
        self.high_watermark.store(offset, Ordering::Release);

        Ok(offset)
    }

    /// Append multiple messages.
    pub fn append_batch(&self, messages: &mut [Message]) -> Result<Vec<u64>> {
        if !self.is_leader {
            return Err(Error::IllegalState("Not the leader".to_string()));
        }

        for msg in messages.iter_mut() {
            msg.partition = self.id;
        }

        let offsets = self.storage.append_batch(messages)?;

        if let Some(&last) = offsets.last() {
            self.log_end_offset.store(last + 1, Ordering::Release);
            self.high_watermark.store(last, Ordering::Release);
        }

        Ok(offsets)
    }

    /// Read a message at a specific offset.
    pub fn read(&self, offset: u64) -> Result<Message> {
        self.storage.read(offset)
    }

    /// Read messages in a range.
    pub fn read_range(&self, start_offset: u64, max_messages: usize) -> Result<Vec<Message>> {
        self.storage.read_range(start_offset, max_messages)
    }

    /// Fetch messages for a consumer.
    pub fn fetch(
        &self,
        offset: u64,
        max_messages: usize,
        max_bytes: usize,
    ) -> Result<FetchResult> {
        let mut messages = if max_bytes > 0 {
            self.storage.read_bytes(offset, max_bytes)?
        } else {
            self.storage.read_range(offset, max_messages)?
        };

        // Enforce max_messages limit
        if max_messages > 0 && messages.len() > max_messages {
            messages.truncate(max_messages);
        }

        let next_offset = if messages.is_empty() {
            offset
        } else {
            messages.last().map(|m| m.offset + 1).unwrap_or(offset)
        };

        Ok(FetchResult {
            messages,
            next_offset,
            high_watermark: self.high_watermark(),
            log_end_offset: self.log_end_offset(),
        })
    }

    /// Get start offset.
    pub fn start_offset(&self) -> u64 {
        self.storage.start_offset()
    }

    /// Get log end offset (next offset to be written).
    pub fn log_end_offset(&self) -> u64 {
        self.log_end_offset.load(Ordering::Acquire)
    }

    /// Get high watermark.
    pub fn high_watermark(&self) -> u64 {
        self.high_watermark.load(Ordering::Acquire)
    }

    /// Update high watermark.
    pub fn update_high_watermark(&self, offset: u64) {
        loop {
            let current = self.high_watermark.load(Ordering::Acquire);
            if offset <= current {
                break;
            }
            if self
                .high_watermark
                .compare_exchange_weak(current, offset, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                break;
            }
        }
    }

    /// Get leader epoch.
    pub fn leader_epoch(&self) -> u32 {
        self.leader_epoch.load(Ordering::Acquire)
    }

    /// Increment leader epoch.
    pub fn increment_epoch(&self) -> u32 {
        self.leader_epoch.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Flush to disk.
    pub fn flush(&self) -> Result<()> {
        self.storage.flush()
    }

    /// Get total size.
    pub fn size(&self) -> u64 {
        self.storage.total_size()
    }

    /// Apply retention policy.
    pub fn apply_retention(&self) -> Result<()> {
        self.storage.apply_retention()?;
        Ok(())
    }

    /// Get message count.
    pub fn message_count(&self) -> u64 {
        self.log_end_offset().saturating_sub(self.start_offset())
    }

    /// Check if partition is empty.
    pub fn is_empty(&self) -> bool {
        self.start_offset() == self.log_end_offset()
    }
}

/// Result of a fetch operation.
#[derive(Debug)]
pub struct FetchResult {
    /// Fetched messages.
    pub messages: Vec<Message>,
    /// Next offset to fetch.
    pub next_offset: u64,
    /// Current high watermark.
    pub high_watermark: u64,
    /// Log end offset.
    pub log_end_offset: u64,
}

impl FetchResult {
    /// Check if there are more messages available.
    pub fn has_more(&self) -> bool {
        self.next_offset < self.high_watermark
    }

    /// Get number of messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// Partitioning strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionStrategy {
    /// Round-robin assignment.
    RoundRobin,
    /// Hash-based on message key.
    KeyHash,
    /// Sticky (consistent) partitioning.
    Sticky,
    /// Random partition.
    Random,
    /// Manual partition assignment.
    Manual,
}

impl Default for PartitionStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// Partitioner for assigning messages to partitions.
pub struct Partitioner {
    /// Partitioning strategy.
    strategy: PartitionStrategy,
    /// Round-robin counter.
    round_robin_counter: AtomicU32,
    /// Number of partitions.
    partition_count: u32,
    /// Sticky partition (for sticky strategy).
    sticky_partition: AtomicU32,
}

impl Partitioner {
    /// Create a new partitioner.
    pub fn new(strategy: PartitionStrategy, partition_count: u32) -> Self {
        Self {
            strategy,
            round_robin_counter: AtomicU32::new(0),
            partition_count,
            sticky_partition: AtomicU32::new(0),
        }
    }

    /// Get partition for a message.
    pub fn partition(&self, message: &Message) -> PartitionId {
        match self.strategy {
            PartitionStrategy::RoundRobin => self.round_robin(),
            PartitionStrategy::KeyHash => self.key_hash(message),
            PartitionStrategy::Sticky => self.sticky(),
            PartitionStrategy::Random => self.random(),
            PartitionStrategy::Manual => message.partition,
        }
    }

    /// Round-robin partitioning.
    fn round_robin(&self) -> PartitionId {
        let counter = self.round_robin_counter.fetch_add(1, Ordering::Relaxed);
        counter % self.partition_count
    }

    /// Hash-based partitioning.
    fn key_hash(&self, message: &Message) -> PartitionId {
        if let Some(ref key) = message.key {
            let hash = xxh3_64(key);
            (hash % self.partition_count as u64) as PartitionId
        } else {
            self.round_robin()
        }
    }

    /// Sticky partitioning.
    fn sticky(&self) -> PartitionId {
        self.sticky_partition.load(Ordering::Acquire)
    }

    /// Set sticky partition.
    pub fn set_sticky_partition(&self, partition: PartitionId) {
        self.sticky_partition.store(partition, Ordering::Release);
    }

    /// Random partitioning.
    fn random(&self) -> PartitionId {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        nanos % self.partition_count
    }

    /// Get partition count.
    pub fn partition_count(&self) -> u32 {
        self.partition_count
    }

    /// Update partition count.
    pub fn set_partition_count(&mut self, count: u32) {
        self.partition_count = count;
    }
}

/// Murmur2 hash (Kafka-compatible).
pub fn murmur2(data: &[u8]) -> u32 {
    const M: u32 = 0x5bd1e995;
    const R: u32 = 24;
    const SEED: u32 = 0x9747b28c;

    let mut h: u32 = SEED ^ (data.len() as u32);
    let mut i = 0;

    while i + 4 <= data.len() {
        let mut k = u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
        k = k.wrapping_mul(M);
        k ^= k >> R;
        k = k.wrapping_mul(M);
        h = h.wrapping_mul(M);
        h ^= k;
        i += 4;
    }

    match data.len() - i {
        3 => {
            h ^= (data[i + 2] as u32) << 16;
            h ^= (data[i + 1] as u32) << 8;
            h ^= data[i] as u32;
            h = h.wrapping_mul(M);
        }
        2 => {
            h ^= (data[i + 1] as u32) << 8;
            h ^= data[i] as u32;
            h = h.wrapping_mul(M);
        }
        1 => {
            h ^= data[i] as u32;
            h = h.wrapping_mul(M);
        }
        _ => {}
    }

    h ^= h >> 13;
    h = h.wrapping_mul(M);
    h ^= h >> 15;

    h
}

/// Calculate partition using Kafka-compatible algorithm.
pub fn kafka_partition(key: &[u8], partition_count: u32) -> PartitionId {
    let hash = murmur2(key);
    (hash & 0x7fffffff) % partition_count
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_message(content: &str) -> Message {
        Message::new(content.as_bytes().to_vec())
    }

    #[test]
    fn test_partition_create() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let partition = Partition::new("test-topic", 0, dir.path(), &config).unwrap();

        assert_eq!(partition.topic(), "test-topic");
        assert_eq!(partition.id(), 0);
        assert_eq!(partition.log_end_offset(), 0);
        assert!(partition.is_empty());
    }

    #[test]
    fn test_partition_append_read() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let partition = Partition::new("test-topic", 0, dir.path(), &config).unwrap();

        let msg1 = create_test_message("Hello");
        let msg2 = create_test_message("World");

        let offset1 = partition.append(msg1).unwrap();
        let offset2 = partition.append(msg2).unwrap();

        assert_eq!(offset1, 0);
        assert_eq!(offset2, 1);
        assert_eq!(partition.log_end_offset(), 2);
        assert_eq!(partition.high_watermark(), 1);

        let read1 = partition.read(0).unwrap();
        let read2 = partition.read(1).unwrap();

        assert_eq!(read1.payload.as_ref(), b"Hello");
        assert_eq!(read2.payload.as_ref(), b"World");
        assert_eq!(read1.partition, 0);
    }

    #[test]
    fn test_partition_fetch() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let partition = Partition::new("test-topic", 0, dir.path(), &config).unwrap();

        for i in 0..10 {
            partition.append(create_test_message(&format!("Msg {}", i))).unwrap();
        }

        let result = partition.fetch(3, 5, 0).unwrap();

        assert_eq!(result.len(), 5);
        assert_eq!(result.next_offset, 8);
        assert!(result.has_more());
    }

    #[test]
    fn test_partition_message_count() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let partition = Partition::new("test-topic", 0, dir.path(), &config).unwrap();

        assert_eq!(partition.message_count(), 0);

        for _ in 0..5 {
            partition.append(create_test_message("Test")).unwrap();
        }

        assert_eq!(partition.message_count(), 5);
    }

    #[test]
    fn test_partitioner_round_robin() {
        let partitioner = Partitioner::new(PartitionStrategy::RoundRobin, 4);

        let msg = create_test_message("test");

        let p1 = partitioner.partition(&msg);
        let p2 = partitioner.partition(&msg);
        let p3 = partitioner.partition(&msg);
        let p4 = partitioner.partition(&msg);
        let p5 = partitioner.partition(&msg);

        assert_eq!(p1, 0);
        assert_eq!(p2, 1);
        assert_eq!(p3, 2);
        assert_eq!(p4, 3);
        assert_eq!(p5, 0); // Wraps around
    }

    #[test]
    fn test_partitioner_key_hash() {
        let partitioner = Partitioner::new(PartitionStrategy::KeyHash, 4);

        let msg1 = Message::with_key("key1", "payload");
        let msg2 = Message::with_key("key2", "payload");
        let msg3 = Message::with_key("key1", "different payload");

        let p1 = partitioner.partition(&msg1);
        let p3 = partitioner.partition(&msg3);

        // Same key should go to same partition
        assert_eq!(p1, p3);

        // Different keys may go to different partitions
        let p2 = partitioner.partition(&msg2);
        assert!(p2 < 4);
    }

    #[test]
    fn test_partitioner_sticky() {
        let partitioner = Partitioner::new(PartitionStrategy::Sticky, 4);

        partitioner.set_sticky_partition(2);

        let msg = create_test_message("test");

        assert_eq!(partitioner.partition(&msg), 2);
        assert_eq!(partitioner.partition(&msg), 2);
        assert_eq!(partitioner.partition(&msg), 2);
    }

    #[test]
    fn test_murmur2() {
        // Test known values
        let hash1 = murmur2(b"key1");
        let hash2 = murmur2(b"key2");
        let hash3 = murmur2(b"key1");

        assert_eq!(hash1, hash3); // Same input, same output
        assert_ne!(hash1, hash2); // Different input, different output
    }

    #[test]
    fn test_kafka_partition() {
        let p1 = kafka_partition(b"key1", 10);
        let p2 = kafka_partition(b"key1", 10);

        assert_eq!(p1, p2);
        assert!(p1 < 10);
    }

    #[test]
    fn test_partition_reopen() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();

        // Create and write
        {
            let partition = Partition::new("test-topic", 0, dir.path(), &config).unwrap();
            partition.append(create_test_message("Hello")).unwrap();
            partition.append(create_test_message("World")).unwrap();
            partition.flush().unwrap();
        }

        // Reopen and verify
        {
            let partition = Partition::open("test-topic", 0, dir.path(), &config).unwrap();
            assert_eq!(partition.log_end_offset(), 2);

            let msg = partition.read(0).unwrap();
            assert_eq!(msg.payload.as_ref(), b"Hello");
        }
    }

    #[test]
    fn test_fetch_result() {
        let result = FetchResult {
            messages: vec![create_test_message("test")],
            next_offset: 1,
            high_watermark: 10,
            log_end_offset: 11,
        };

        assert!(!result.is_empty());
        assert_eq!(result.len(), 1);
        assert!(result.has_more());
    }

    #[test]
    fn test_partition_leader_epoch() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let partition = Partition::new("test-topic", 0, dir.path(), &config).unwrap();

        assert_eq!(partition.leader_epoch(), 0);

        let new_epoch = partition.increment_epoch();
        assert_eq!(new_epoch, 1);
        assert_eq!(partition.leader_epoch(), 1);
    }

    #[test]
    fn test_partition_batch_append() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let partition = Partition::new("test-topic", 0, dir.path(), &config).unwrap();

        let mut messages: Vec<Message> = (0..5)
            .map(|i| create_test_message(&format!("Msg {}", i)))
            .collect();

        let offsets = partition.append_batch(&mut messages).unwrap();

        assert_eq!(offsets, vec![0, 1, 2, 3, 4]);
        assert_eq!(partition.log_end_offset(), 5);
        assert_eq!(partition.high_watermark(), 4);
    }
}
