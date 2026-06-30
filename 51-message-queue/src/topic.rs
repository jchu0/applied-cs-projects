//! Topic management.

use crate::config::TopicConfig;
use crate::error::{Error, Result};
use crate::message::Message;
use crate::partition::{FetchResult, Partition, PartitionId, Partitioner, PartitionStrategy};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// A topic containing multiple partitions.
pub struct Topic {
    /// Topic name.
    name: String,
    /// Topic configuration.
    config: TopicConfig,
    /// Partitions.
    partitions: RwLock<Vec<Arc<Partition>>>,
    /// Base directory.
    base_dir: PathBuf,
    /// Partitioner.
    partitioner: Partitioner,
}

impl Topic {
    /// Create a new topic.
    pub fn new(
        name: impl Into<String>,
        config: TopicConfig,
        base_dir: impl Into<PathBuf>,
    ) -> Result<Self> {
        let name = name.into();
        let base_dir = base_dir.into();
        let topic_dir = base_dir.join(&name);

        std::fs::create_dir_all(&topic_dir)?;

        let mut partitions = Vec::with_capacity(config.partition_count as usize);

        for i in 0..config.partition_count {
            let partition = Partition::new(&name, i, &base_dir, &config)?;
            partitions.push(Arc::new(partition));
        }

        let partitioner = Partitioner::new(PartitionStrategy::KeyHash, config.partition_count);

        Ok(Self {
            name,
            config,
            partitions: RwLock::new(partitions),
            base_dir,
            partitioner,
        })
    }

    /// Open an existing topic.
    pub fn open(
        name: impl Into<String>,
        base_dir: impl Into<PathBuf>,
        config: TopicConfig,
    ) -> Result<Self> {
        let name = name.into();
        let base_dir = base_dir.into();
        let topic_dir = base_dir.join(&name);

        if !topic_dir.exists() {
            return Err(Error::TopicNotFound(name));
        }

        // Discover partitions
        let mut partition_ids: Vec<u32> = Vec::new();

        for entry in std::fs::read_dir(&topic_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if let Ok(id) = name.parse::<u32>() {
                        partition_ids.push(id);
                    }
                }
            }
        }

        partition_ids.sort();

        let partition_count = partition_ids.len().max(config.partition_count as usize) as u32;

        let mut partitions = Vec::with_capacity(partition_count as usize);

        for id in 0..partition_count {
            if partition_ids.contains(&id) {
                let partition = Partition::open(&name, id, &base_dir, &config)?;
                partitions.push(Arc::new(partition));
            } else {
                let partition = Partition::new(&name, id, &base_dir, &config)?;
                partitions.push(Arc::new(partition));
            }
        }

        let partitioner = Partitioner::new(PartitionStrategy::KeyHash, partition_count);

        Ok(Self {
            name,
            config,
            partitions: RwLock::new(partitions),
            base_dir,
            partitioner,
        })
    }

    /// Get topic name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get topic configuration.
    pub fn config(&self) -> &TopicConfig {
        &self.config
    }

    /// Get number of partitions.
    pub fn partition_count(&self) -> u32 {
        self.partitions.read().len() as u32
    }

    /// Get a specific partition.
    pub fn partition(&self, id: PartitionId) -> Option<Arc<Partition>> {
        self.partitions.read().get(id as usize).cloned()
    }

    /// Get all partitions.
    pub fn partitions(&self) -> Vec<Arc<Partition>> {
        self.partitions.read().clone()
    }

    /// Append a message to the topic.
    pub fn append(&self, message: Message) -> Result<(PartitionId, u64)> {
        let partition_id = self.partitioner.partition(&message);
        let partition = self.partition(partition_id).ok_or(Error::PartitionNotFound {
            topic: self.name.clone(),
            partition: partition_id,
        })?;

        let offset = partition.append(message)?;
        Ok((partition_id, offset))
    }

    /// Append a message to a specific partition.
    pub fn append_to_partition(&self, partition_id: PartitionId, message: Message) -> Result<u64> {
        let partition = self.partition(partition_id).ok_or(Error::PartitionNotFound {
            topic: self.name.clone(),
            partition: partition_id,
        })?;

        partition.append(message)
    }

    /// Read a message from a specific partition.
    pub fn read(&self, partition_id: PartitionId, offset: u64) -> Result<Message> {
        let partition = self.partition(partition_id).ok_or(Error::PartitionNotFound {
            topic: self.name.clone(),
            partition: partition_id,
        })?;

        partition.read(offset)
    }

    /// Fetch messages from a partition.
    pub fn fetch(
        &self,
        partition_id: PartitionId,
        offset: u64,
        max_messages: usize,
        max_bytes: usize,
    ) -> Result<FetchResult> {
        let partition = self.partition(partition_id).ok_or(Error::PartitionNotFound {
            topic: self.name.clone(),
            partition: partition_id,
        })?;

        partition.fetch(offset, max_messages, max_bytes)
    }

    /// Get log end offset for all partitions.
    pub fn log_end_offsets(&self) -> HashMap<PartitionId, u64> {
        self.partitions
            .read()
            .iter()
            .enumerate()
            .map(|(i, p)| (i as PartitionId, p.log_end_offset()))
            .collect()
    }

    /// Get high watermarks for all partitions.
    pub fn high_watermarks(&self) -> HashMap<PartitionId, u64> {
        self.partitions
            .read()
            .iter()
            .enumerate()
            .map(|(i, p)| (i as PartitionId, p.high_watermark()))
            .collect()
    }

    /// Flush all partitions.
    pub fn flush(&self) -> Result<()> {
        for partition in self.partitions.read().iter() {
            partition.flush()?;
        }
        Ok(())
    }

    /// Apply retention policy to all partitions.
    pub fn apply_retention(&self) -> Result<()> {
        for partition in self.partitions.read().iter() {
            partition.apply_retention()?;
        }
        Ok(())
    }

    /// Get total size across all partitions.
    pub fn total_size(&self) -> u64 {
        self.partitions.read().iter().map(|p| p.size()).sum()
    }

    /// Get total message count.
    pub fn message_count(&self) -> u64 {
        self.partitions.read().iter().map(|p| p.message_count()).sum()
    }

    /// Describe the topic.
    pub fn describe(&self) -> TopicDescription {
        let partitions = self.partitions.read();
        let partition_info: Vec<PartitionInfo> = partitions
            .iter()
            .map(|p| PartitionInfo {
                id: p.id(),
                leader_epoch: p.leader_epoch(),
                log_start_offset: p.start_offset(),
                log_end_offset: p.log_end_offset(),
                high_watermark: p.high_watermark(),
                size: p.size(),
            })
            .collect();

        TopicDescription {
            name: self.name.clone(),
            partition_count: partitions.len() as u32,
            replication_factor: self.config.replication_factor,
            config: self.config.clone(),
            partitions: partition_info,
        }
    }
}

/// Topic description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicDescription {
    /// Topic name.
    pub name: String,
    /// Number of partitions.
    pub partition_count: u32,
    /// Replication factor.
    pub replication_factor: u32,
    /// Configuration.
    pub config: TopicConfig,
    /// Partition information.
    pub partitions: Vec<PartitionInfo>,
}

/// Partition information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionInfo {
    /// Partition ID.
    pub id: PartitionId,
    /// Leader epoch.
    pub leader_epoch: u32,
    /// Log start offset.
    pub log_start_offset: u64,
    /// Log end offset.
    pub log_end_offset: u64,
    /// High watermark.
    pub high_watermark: u64,
    /// Size in bytes.
    pub size: u64,
}

/// Topic manager for handling multiple topics.
pub struct TopicManager {
    /// Base directory.
    base_dir: PathBuf,
    /// Topics by name.
    topics: RwLock<HashMap<String, Arc<Topic>>>,
    /// Default topic configuration.
    default_config: TopicConfig,
}

impl TopicManager {
    /// Create a new topic manager.
    pub fn new(base_dir: impl Into<PathBuf>, default_config: TopicConfig) -> Result<Self> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir)?;

        Ok(Self {
            base_dir,
            topics: RwLock::new(HashMap::new()),
            default_config,
        })
    }

    /// Load existing topics from disk.
    pub fn load(&self) -> Result<()> {
        let mut topics = self.topics.write();

        for entry in std::fs::read_dir(&self.base_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    // Skip hidden directories and metadata
                    if name.starts_with('.') || name == "__consumer_offsets" {
                        continue;
                    }

                    let topic = Topic::open(name, &self.base_dir, self.default_config.clone())?;
                    topics.insert(name.to_string(), Arc::new(topic));
                }
            }
        }

        Ok(())
    }

    /// Create a new topic.
    pub fn create_topic(&self, name: impl Into<String>, config: Option<TopicConfig>) -> Result<Arc<Topic>> {
        let name = name.into();
        let config = config.unwrap_or_else(|| self.default_config.clone());

        {
            let topics = self.topics.read();
            if topics.contains_key(&name) {
                return Err(Error::TopicAlreadyExists(name));
            }
        }

        let topic = Topic::new(&name, config, &self.base_dir)?;
        let topic = Arc::new(topic);

        self.topics.write().insert(name, Arc::clone(&topic));

        Ok(topic)
    }

    /// Get a topic.
    pub fn get_topic(&self, name: &str) -> Option<Arc<Topic>> {
        self.topics.read().get(name).cloned()
    }

    /// Get or create a topic.
    pub fn get_or_create_topic(&self, name: &str) -> Result<Arc<Topic>> {
        // Check if exists
        if let Some(topic) = self.get_topic(name) {
            return Ok(topic);
        }

        // Create new topic
        self.create_topic(name, None)
    }

    /// Delete a topic.
    pub fn delete_topic(&self, name: &str) -> Result<()> {
        let topic = {
            let mut topics = self.topics.write();
            topics.remove(name)
        };

        if topic.is_none() {
            return Err(Error::TopicNotFound(name.to_string()));
        }

        // Delete topic directory
        let topic_dir = self.base_dir.join(name);
        if topic_dir.exists() {
            std::fs::remove_dir_all(topic_dir)?;
        }

        Ok(())
    }

    /// List all topics.
    pub fn list_topics(&self) -> Vec<String> {
        self.topics.read().keys().cloned().collect()
    }

    /// Check if topic exists.
    pub fn topic_exists(&self, name: &str) -> bool {
        self.topics.read().contains_key(name)
    }

    /// Get topic count.
    pub fn topic_count(&self) -> usize {
        self.topics.read().len()
    }

    /// Describe a topic.
    pub fn describe_topic(&self, name: &str) -> Result<TopicDescription> {
        let topic = self.get_topic(name).ok_or_else(|| Error::TopicNotFound(name.to_string()))?;
        Ok(topic.describe())
    }

    /// Flush all topics.
    pub fn flush_all(&self) -> Result<()> {
        for topic in self.topics.read().values() {
            topic.flush()?;
        }
        Ok(())
    }

    /// Apply retention to all topics.
    pub fn apply_retention_all(&self) -> Result<()> {
        for topic in self.topics.read().values() {
            topic.apply_retention()?;
        }
        Ok(())
    }

    /// Get total size across all topics.
    pub fn total_size(&self) -> u64 {
        self.topics.read().values().map(|t| t.total_size()).sum()
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
    fn test_topic_create() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::new().with_partitions(4);
        let topic = Topic::new("test-topic", config, dir.path()).unwrap();

        assert_eq!(topic.name(), "test-topic");
        assert_eq!(topic.partition_count(), 4);
    }

    #[test]
    fn test_topic_append_read() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::new().with_partitions(4);
        let topic = Topic::new("test-topic", config, dir.path()).unwrap();

        let msg = Message::with_key("key1", "Hello");
        let (partition, offset) = topic.append(msg).unwrap();

        assert!(partition < 4);
        assert_eq!(offset, 0);

        let read = topic.read(partition, offset).unwrap();
        assert_eq!(read.payload.as_ref(), b"Hello");
    }

    #[test]
    fn test_topic_append_to_partition() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::new().with_partitions(4);
        let topic = Topic::new("test-topic", config, dir.path()).unwrap();

        let offset = topic.append_to_partition(2, create_test_message("Hello")).unwrap();
        assert_eq!(offset, 0);

        let msg = topic.read(2, 0).unwrap();
        assert_eq!(msg.payload.as_ref(), b"Hello");
    }

    #[test]
    fn test_topic_fetch() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::new().with_partitions(1);
        let topic = Topic::new("test-topic", config, dir.path()).unwrap();

        for i in 0..10 {
            topic.append_to_partition(0, create_test_message(&format!("Msg {}", i))).unwrap();
        }

        let result = topic.fetch(0, 3, 5, 0).unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.next_offset, 8);
    }

    #[test]
    fn test_topic_persistence() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::new().with_partitions(2);

        // Create and write
        {
            let topic = Topic::new("test-topic", config.clone(), dir.path()).unwrap();
            topic.append_to_partition(0, create_test_message("Hello")).unwrap();
            topic.append_to_partition(1, create_test_message("World")).unwrap();
            topic.flush().unwrap();
        }

        // Reopen and verify
        {
            let topic = Topic::open("test-topic", dir.path(), config).unwrap();
            assert_eq!(topic.partition_count(), 2);

            let msg0 = topic.read(0, 0).unwrap();
            let msg1 = topic.read(1, 0).unwrap();

            assert_eq!(msg0.payload.as_ref(), b"Hello");
            assert_eq!(msg1.payload.as_ref(), b"World");
        }
    }

    #[test]
    fn test_topic_describe() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::new().with_partitions(2);
        let topic = Topic::new("test-topic", config, dir.path()).unwrap();

        topic.append_to_partition(0, create_test_message("Test")).unwrap();

        let desc = topic.describe();
        assert_eq!(desc.name, "test-topic");
        assert_eq!(desc.partition_count, 2);
        assert_eq!(desc.partitions.len(), 2);
    }

    #[test]
    fn test_topic_manager_create() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let manager = TopicManager::new(dir.path(), config).unwrap();

        let topic = manager.create_topic("topic1", None).unwrap();
        assert_eq!(topic.name(), "topic1");

        assert!(manager.topic_exists("topic1"));
        assert!(!manager.topic_exists("topic2"));
    }

    #[test]
    fn test_topic_manager_get_or_create() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let manager = TopicManager::new(dir.path(), config).unwrap();

        let topic1 = manager.get_or_create_topic("topic1").unwrap();
        let topic2 = manager.get_or_create_topic("topic1").unwrap();

        assert!(Arc::ptr_eq(&topic1, &topic2));
    }

    #[test]
    fn test_topic_manager_delete() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let manager = TopicManager::new(dir.path(), config).unwrap();

        manager.create_topic("topic1", None).unwrap();
        assert!(manager.topic_exists("topic1"));

        manager.delete_topic("topic1").unwrap();
        assert!(!manager.topic_exists("topic1"));
    }

    #[test]
    fn test_topic_manager_list() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let manager = TopicManager::new(dir.path(), config).unwrap();

        manager.create_topic("topic1", None).unwrap();
        manager.create_topic("topic2", None).unwrap();
        manager.create_topic("topic3", None).unwrap();

        let topics = manager.list_topics();
        assert_eq!(topics.len(), 3);
        assert!(topics.contains(&"topic1".to_string()));
        assert!(topics.contains(&"topic2".to_string()));
        assert!(topics.contains(&"topic3".to_string()));
    }

    #[test]
    fn test_topic_manager_describe() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::new().with_partitions(8);
        let manager = TopicManager::new(dir.path(), config).unwrap();

        manager.create_topic("topic1", None).unwrap();

        let desc = manager.describe_topic("topic1").unwrap();
        assert_eq!(desc.name, "topic1");
        assert_eq!(desc.partition_count, 8);
    }

    #[test]
    fn test_topic_log_end_offsets() {
        let dir = TempDir::new().unwrap();
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
    fn test_topic_message_count() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::new().with_partitions(2);
        let topic = Topic::new("test-topic", config, dir.path()).unwrap();

        assert_eq!(topic.message_count(), 0);

        topic.append_to_partition(0, create_test_message("1")).unwrap();
        topic.append_to_partition(0, create_test_message("2")).unwrap();
        topic.append_to_partition(1, create_test_message("3")).unwrap();

        assert_eq!(topic.message_count(), 3);
    }

    #[test]
    fn test_topic_already_exists() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let manager = TopicManager::new(dir.path(), config).unwrap();

        manager.create_topic("topic1", None).unwrap();
        let result = manager.create_topic("topic1", None);

        assert!(matches!(result, Err(Error::TopicAlreadyExists(_))));
    }

    #[test]
    fn test_topic_not_found() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let manager = TopicManager::new(dir.path(), config).unwrap();

        let result = manager.describe_topic("nonexistent");
        assert!(matches!(result, Err(Error::TopicNotFound(_))));
    }

    #[test]
    fn test_topic_manager_load() {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();

        // Create topics
        {
            let manager = TopicManager::new(dir.path(), config.clone()).unwrap();
            manager.create_topic("topic1", None).unwrap();
            manager.create_topic("topic2", None).unwrap();
            manager.flush_all().unwrap();
        }

        // Reload
        {
            let manager = TopicManager::new(dir.path(), config).unwrap();
            manager.load().unwrap();

            assert!(manager.topic_exists("topic1"));
            assert!(manager.topic_exists("topic2"));
            assert_eq!(manager.topic_count(), 2);
        }
    }
}
