//! Broker for managing topics, producers, and consumers.

use crate::config::{BrokerConfig, TopicConfig};
use crate::consumer_group::{ConsumerGroup, GroupConfig, GroupManager};
use crate::error::{Error, Result};
use crate::message::Message;
use crate::offset::TopicPartition;
use crate::partition::PartitionId;
use crate::topic::{Topic, TopicDescription, TopicManager};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Message broker.
pub struct Broker {
    /// Broker configuration.
    config: BrokerConfig,
    /// Topic manager.
    topic_manager: Arc<TopicManager>,
    /// Group manager.
    group_manager: GroupManager,
    /// Is broker running.
    running: AtomicBool,
    /// Broker metrics.
    metrics: BrokerMetrics,
    /// Start time.
    start_time: Instant,
}

impl Broker {
    /// Create a new broker.
    pub fn new(config: BrokerConfig) -> Result<Self> {
        config.validate().map_err(Error::InvalidConfig)?;

        std::fs::create_dir_all(&config.data_dir)?;
        std::fs::create_dir_all(&config.log_dir)?;

        let topic_config = TopicConfig {
            partition_count: config.default_partitions,
            replication_factor: config.default_replication_factor,
            retention_ms: config.retention_ms,
            retention_bytes: config.retention_bytes,
            segment_bytes: config.segment_bytes,
            compression: config.compression,
            ..TopicConfig::default()
        };

        let topic_manager = Arc::new(TopicManager::new(&config.data_dir, topic_config)?);

        let group_config = GroupConfig::default();
        let group_manager = GroupManager::new(group_config);

        Ok(Self {
            config,
            topic_manager,
            group_manager,
            running: AtomicBool::new(false),
            metrics: BrokerMetrics::new(),
            start_time: Instant::now(),
        })
    }

    /// Start the broker.
    pub fn start(&self) -> Result<()> {
        if self.running.swap(true, Ordering::AcqRel) {
            return Err(Error::IllegalState("Broker already running".to_string()));
        }

        // Load existing topics
        self.topic_manager.load()?;

        tracing::info!(
            "Broker {} started on {}",
            self.config.broker_id,
            self.config.socket_addr()
        );

        Ok(())
    }

    /// Stop the broker.
    pub fn stop(&self) -> Result<()> {
        if !self.running.swap(false, Ordering::AcqRel) {
            return Err(Error::IllegalState("Broker not running".to_string()));
        }

        // Flush all topics
        self.topic_manager.flush_all()?;

        tracing::info!("Broker {} stopped", self.config.broker_id);

        Ok(())
    }

    /// Check if broker is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// Get broker ID.
    pub fn broker_id(&self) -> u32 {
        self.config.broker_id
    }

    /// Get configuration.
    pub fn config(&self) -> &BrokerConfig {
        &self.config
    }

    /// Get topic manager.
    pub fn topic_manager(&self) -> &Arc<TopicManager> {
        &self.topic_manager
    }

    // Topic operations

    /// Create a topic.
    pub fn create_topic(&self, name: &str, config: Option<TopicConfig>) -> Result<Arc<Topic>> {
        let topic = self.topic_manager.create_topic(name, config)?;
        self.metrics.record_topic_created();
        Ok(topic)
    }

    /// Delete a topic.
    pub fn delete_topic(&self, name: &str) -> Result<()> {
        self.topic_manager.delete_topic(name)?;
        self.metrics.record_topic_deleted();
        Ok(())
    }

    /// Get a topic.
    pub fn get_topic(&self, name: &str) -> Option<Arc<Topic>> {
        self.topic_manager.get_topic(name)
    }

    /// Get or create a topic.
    pub fn get_or_create_topic(&self, name: &str) -> Result<Arc<Topic>> {
        if self.config.auto_create_topics {
            // Check if topic exists before creating
            let existed = self.topic_manager.get_topic(name).is_some();
            let topic = self.topic_manager.get_or_create_topic(name)?;
            // Record metric if a new topic was created
            if !existed {
                self.metrics.record_topic_created();
            }
            Ok(topic)
        } else {
            self.topic_manager
                .get_topic(name)
                .ok_or_else(|| Error::TopicNotFound(name.to_string()))
        }
    }

    /// List topics.
    pub fn list_topics(&self) -> Vec<String> {
        self.topic_manager.list_topics()
    }

    /// Describe a topic.
    pub fn describe_topic(&self, name: &str) -> Result<TopicDescription> {
        self.topic_manager.describe_topic(name)
    }

    // Message operations

    /// Produce a message to a topic.
    pub fn produce(&self, topic: &str, message: Message) -> Result<(PartitionId, u64)> {
        let topic_arc = self.get_or_create_topic(topic)?;
        let result = topic_arc.append(message)?;
        self.metrics.record_produce();
        Ok(result)
    }

    /// Produce a message to a specific partition.
    pub fn produce_to_partition(
        &self,
        topic: &str,
        partition: PartitionId,
        message: Message,
    ) -> Result<u64> {
        let topic_arc = self.get_or_create_topic(topic)?;
        let offset = topic_arc.append_to_partition(partition, message)?;
        self.metrics.record_produce();
        Ok(offset)
    }

    /// Fetch messages from a partition.
    pub fn fetch(
        &self,
        topic: &str,
        partition: PartitionId,
        offset: u64,
        max_messages: usize,
        max_bytes: usize,
    ) -> Result<Vec<Message>> {
        let topic_arc = self.get_topic(topic).ok_or_else(|| {
            Error::TopicNotFound(topic.to_string())
        })?;

        let result = topic_arc.fetch(partition, offset, max_messages, max_bytes)?;
        self.metrics.record_fetch(result.messages.len());
        Ok(result.messages)
    }

    // Offset operations

    /// Get offsets for a topic.
    pub fn get_offsets(&self, topic: &str) -> Result<HashMap<PartitionId, (u64, u64)>> {
        let topic_arc = self.get_topic(topic).ok_or_else(|| {
            Error::TopicNotFound(topic.to_string())
        })?;

        let mut offsets = HashMap::new();
        for partition in topic_arc.partitions() {
            offsets.insert(partition.id(), (partition.start_offset(), partition.log_end_offset()));
        }

        Ok(offsets)
    }

    // Consumer group operations

    /// Get or create a consumer group.
    pub fn get_or_create_group(&self, group_id: &str) -> Arc<ConsumerGroup> {
        self.group_manager.get_or_create(group_id)
    }

    /// List consumer groups.
    pub fn list_groups(&self) -> Vec<String> {
        self.group_manager.list_groups()
    }

    // Metrics

    /// Get broker metrics.
    pub fn metrics(&self) -> &BrokerMetrics {
        &self.metrics
    }

    /// Get uptime.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    // Maintenance

    /// Apply retention to all topics.
    pub fn apply_retention(&self) -> Result<()> {
        self.topic_manager.apply_retention_all()
    }

    /// Flush all data to disk.
    pub fn flush(&self) -> Result<()> {
        self.topic_manager.flush_all()
    }
}

/// Broker metrics.
pub struct BrokerMetrics {
    /// Messages produced.
    messages_produced: AtomicU64,
    /// Messages consumed.
    messages_consumed: AtomicU64,
    /// Topics created.
    topics_created: AtomicU64,
    /// Topics deleted.
    topics_deleted: AtomicU64,
    /// Bytes received.
    bytes_received: AtomicU64,
    /// Bytes sent.
    bytes_sent: AtomicU64,
    /// Active connections.
    active_connections: AtomicU64,
    /// Total requests.
    total_requests: AtomicU64,
    /// Failed requests.
    failed_requests: AtomicU64,
}

impl BrokerMetrics {
    /// Create new metrics.
    fn new() -> Self {
        Self {
            messages_produced: AtomicU64::new(0),
            messages_consumed: AtomicU64::new(0),
            topics_created: AtomicU64::new(0),
            topics_deleted: AtomicU64::new(0),
            bytes_received: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            active_connections: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            failed_requests: AtomicU64::new(0),
        }
    }

    /// Record a produce operation.
    pub fn record_produce(&self) {
        self.messages_produced.fetch_add(1, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a fetch operation.
    pub fn record_fetch(&self, count: usize) {
        self.messages_consumed.fetch_add(count as u64, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Record topic creation.
    pub fn record_topic_created(&self) {
        self.topics_created.fetch_add(1, Ordering::Relaxed);
    }

    /// Record topic deletion.
    pub fn record_topic_deleted(&self) {
        self.topics_deleted.fetch_add(1, Ordering::Relaxed);
    }

    /// Record bytes received.
    pub fn record_bytes_received(&self, bytes: u64) {
        self.bytes_received.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes sent.
    pub fn record_bytes_sent(&self, bytes: u64) {
        self.bytes_sent.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record connection opened.
    pub fn connection_opened(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Record connection closed.
    pub fn connection_closed(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Record failed request.
    pub fn record_failure(&self) {
        self.failed_requests.fetch_add(1, Ordering::Relaxed);
    }

    /// Get messages produced.
    pub fn messages_produced(&self) -> u64 {
        self.messages_produced.load(Ordering::Relaxed)
    }

    /// Get messages consumed.
    pub fn messages_consumed(&self) -> u64 {
        self.messages_consumed.load(Ordering::Relaxed)
    }

    /// Get topics created.
    pub fn topics_created(&self) -> u64 {
        self.topics_created.load(Ordering::Relaxed)
    }

    /// Get topics deleted.
    pub fn topics_deleted(&self) -> u64 {
        self.topics_deleted.load(Ordering::Relaxed)
    }

    /// Get bytes received.
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received.load(Ordering::Relaxed)
    }

    /// Get bytes sent.
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent.load(Ordering::Relaxed)
    }

    /// Get active connections.
    pub fn active_connections(&self) -> u64 {
        self.active_connections.load(Ordering::Relaxed)
    }

    /// Get total requests.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get failed requests.
    pub fn failed_requests(&self) -> u64 {
        self.failed_requests.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_broker() -> (TempDir, Broker) {
        let dir = TempDir::new().unwrap();
        let config = BrokerConfig::default()
            .with_data_dir(dir.path().join("data"))
            .with_log_dir(dir.path().join("logs"));

        let broker = Broker::new(config).unwrap();
        (dir, broker)
    }

    #[test]
    fn test_broker_create() {
        let (_dir, broker) = create_broker();

        assert!(!broker.is_running());
        assert_eq!(broker.broker_id(), 1);
    }

    #[test]
    fn test_broker_start_stop() {
        let (_dir, broker) = create_broker();

        broker.start().unwrap();
        assert!(broker.is_running());

        broker.stop().unwrap();
        assert!(!broker.is_running());
    }

    #[test]
    fn test_broker_create_topic() {
        let (_dir, broker) = create_broker();
        broker.start().unwrap();

        let topic = broker.create_topic("test-topic", None).unwrap();
        assert_eq!(topic.name(), "test-topic");

        assert!(broker.get_topic("test-topic").is_some());
        assert!(broker.list_topics().contains(&"test-topic".to_string()));
    }

    #[test]
    fn test_broker_delete_topic() {
        let (_dir, broker) = create_broker();
        broker.start().unwrap();

        broker.create_topic("test-topic", None).unwrap();
        broker.delete_topic("test-topic").unwrap();

        assert!(broker.get_topic("test-topic").is_none());
    }

    #[test]
    fn test_broker_produce_fetch() {
        let (_dir, broker) = create_broker();
        broker.start().unwrap();

        let msg = Message::new("Hello, World!");
        let (partition, offset) = broker.produce("test-topic", msg).unwrap();

        assert_eq!(offset, 0);

        let messages = broker.fetch("test-topic", partition, 0, 10, 0).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].payload.as_ref(), b"Hello, World!");
    }

    #[test]
    fn test_broker_produce_to_partition() {
        let (_dir, broker) = create_broker();
        broker.start().unwrap();

        broker.create_topic("test-topic", None).unwrap();

        let msg = Message::new("Test");
        let offset = broker.produce_to_partition("test-topic", 0, msg).unwrap();

        assert_eq!(offset, 0);
    }

    #[test]
    fn test_broker_get_offsets() {
        let (_dir, broker) = create_broker();
        broker.start().unwrap();

        broker.create_topic("test-topic", None).unwrap();
        broker.produce("test-topic", Message::new("msg1")).unwrap();
        broker.produce("test-topic", Message::new("msg2")).unwrap();

        let offsets = broker.get_offsets("test-topic").unwrap();
        assert!(!offsets.is_empty());
    }

    #[test]
    fn test_broker_metrics() {
        let (_dir, broker) = create_broker();
        broker.start().unwrap();

        broker.produce("test-topic", Message::new("Test")).unwrap();
        broker.fetch("test-topic", 0, 0, 10, 0).unwrap();

        let metrics = broker.metrics();
        assert_eq!(metrics.messages_produced(), 1);
        assert!(metrics.messages_consumed() > 0);
        assert_eq!(metrics.topics_created(), 1);
    }

    #[test]
    fn test_broker_uptime() {
        let (_dir, broker) = create_broker();

        std::thread::sleep(Duration::from_millis(10));

        assert!(broker.uptime().as_millis() >= 10);
    }

    #[test]
    fn test_broker_auto_create_disabled() {
        let dir = TempDir::new().unwrap();
        let config = BrokerConfig::default()
            .with_data_dir(dir.path().join("data"))
            .with_log_dir(dir.path().join("logs"));

        // Modify config to disable auto-create
        let mut config = config;
        config.auto_create_topics = false;

        let broker = Broker::new(config).unwrap();
        broker.start().unwrap();

        let result = broker.get_or_create_topic("nonexistent");
        assert!(matches!(result, Err(Error::TopicNotFound(_))));
    }

    #[test]
    fn test_broker_list_groups() {
        let (_dir, broker) = create_broker();
        broker.start().unwrap();

        broker.get_or_create_group("group1");
        broker.get_or_create_group("group2");

        let groups = broker.list_groups();
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn test_broker_describe_topic() {
        let (_dir, broker) = create_broker();
        broker.start().unwrap();

        broker.create_topic("test-topic", None).unwrap();

        let desc = broker.describe_topic("test-topic").unwrap();
        assert_eq!(desc.name, "test-topic");
        assert_eq!(desc.partition_count, 4); // Default
    }

    #[test]
    fn test_broker_flush() {
        let (_dir, broker) = create_broker();
        broker.start().unwrap();

        broker.produce("test-topic", Message::new("Test")).unwrap();
        broker.flush().unwrap();
    }

    #[test]
    fn test_broker_double_start() {
        let (_dir, broker) = create_broker();

        broker.start().unwrap();
        let result = broker.start();

        assert!(matches!(result, Err(Error::IllegalState(_))));
    }

    #[test]
    fn test_broker_stop_not_running() {
        let (_dir, broker) = create_broker();

        let result = broker.stop();
        assert!(matches!(result, Err(Error::IllegalState(_))));
    }
}
