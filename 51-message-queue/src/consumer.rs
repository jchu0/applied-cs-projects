//! Consumer API for consuming messages.

use crate::config::{ConsumerConfig, OffsetReset};
use crate::error::{Error, Result};
use crate::message::Message;
use crate::offset::{OffsetStore, OffsetTracker, TopicPartition};
use crate::partition::{FetchResult, PartitionId};
use crate::topic::TopicManager;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Consumer for consuming messages from topics.
pub struct Consumer {
    /// Configuration.
    config: ConsumerConfig,
    /// Topic manager.
    topic_manager: Arc<TopicManager>,
    /// Subscribed topics.
    subscriptions: RwLock<HashSet<String>>,
    /// Assigned partitions.
    assignments: RwLock<Vec<TopicPartition>>,
    /// Offset tracker.
    offset_tracker: OffsetTracker,
    /// Offset store (for commits).
    offset_store: Option<OffsetStore>,
    /// Is consumer closed.
    closed: AtomicBool,
    /// Last poll time.
    last_poll: Mutex<Instant>,
    /// Metrics.
    metrics: ConsumerMetrics,
    /// Paused partitions.
    paused: RwLock<HashSet<TopicPartition>>,
}

impl Consumer {
    /// Create a new consumer.
    pub fn new(config: ConsumerConfig, topic_manager: Arc<TopicManager>) -> Result<Self> {
        let offset_store = if let Some(ref group_id) = config.group_id {
            let store_path = topic_manager
                .get_topic("__consumer_offsets")
                .map(|_| std::path::PathBuf::from("/tmp/mq/offsets"))
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp/mq/offsets"));

            Some(OffsetStore::new(group_id, store_path)?)
        } else {
            None
        };

        Ok(Self {
            config,
            topic_manager,
            subscriptions: RwLock::new(HashSet::new()),
            assignments: RwLock::new(Vec::new()),
            offset_tracker: OffsetTracker::new(),
            offset_store,
            closed: AtomicBool::new(false),
            last_poll: Mutex::new(Instant::now()),
            metrics: ConsumerMetrics::new(),
            paused: RwLock::new(HashSet::new()),
        })
    }

    /// Subscribe to topics.
    pub fn subscribe(&self, topics: &[&str]) -> Result<()> {
        self.check_closed()?;

        let mut subscriptions = self.subscriptions.write();
        let mut assignments = self.assignments.write();

        subscriptions.clear();
        assignments.clear();

        for &topic in topics {
            subscriptions.insert(topic.to_string());

            // Auto-assign all partitions of the topic
            if let Some(topic_arc) = self.topic_manager.get_topic(topic) {
                for partition_id in 0..topic_arc.partition_count() {
                    let tp = TopicPartition::new(topic, partition_id);
                    assignments.push(tp.clone());

                    // Initialize offset
                    self.initialize_offset(&tp)?;
                }
            }
        }

        Ok(())
    }

    /// Assign specific partitions.
    pub fn assign(&self, partitions: &[TopicPartition]) -> Result<()> {
        self.check_closed()?;

        let mut assignments = self.assignments.write();
        assignments.clear();

        for tp in partitions {
            assignments.push(tp.clone());
            self.initialize_offset(tp)?;
        }

        Ok(())
    }

    /// Initialize offset for a partition.
    fn initialize_offset(&self, tp: &TopicPartition) -> Result<()> {
        // Check if we have a committed offset
        if let Some(ref store) = self.offset_store {
            if let Some(offset) = store.get_offset(tp) {
                self.offset_tracker.set_position(tp.clone(), offset);
                return Ok(());
            }
        }

        // Use auto offset reset policy
        let topic = self.topic_manager.get_topic(&tp.topic).ok_or_else(|| {
            Error::TopicNotFound(tp.topic.clone())
        })?;

        let partition = topic.partition(tp.partition).ok_or(Error::PartitionNotFound {
            topic: tp.topic.clone(),
            partition: tp.partition,
        })?;

        let offset = match self.config.auto_offset_reset {
            OffsetReset::Earliest => partition.start_offset(),
            OffsetReset::Latest => partition.log_end_offset(),
            OffsetReset::None => {
                return Err(Error::OffsetOutOfRange {
                    offset: 0,
                    start: partition.start_offset(),
                    end: partition.log_end_offset(),
                });
            }
        };

        self.offset_tracker.set_position(tp.clone(), offset);

        Ok(())
    }

    /// Poll for messages.
    pub fn poll(&self, timeout: Duration) -> Result<Vec<Message>> {
        self.check_closed()?;

        *self.last_poll.lock() = Instant::now();

        let assignments = self.assignments.read().clone();
        let paused = self.paused.read();

        if assignments.is_empty() {
            return Err(Error::NotSubscribed);
        }

        let mut all_messages = Vec::new();
        let start = Instant::now();
        let max_records = self.config.max_poll_records;

        for tp in &assignments {
            if paused.contains(tp) {
                continue;
            }

            if all_messages.len() >= max_records {
                break;
            }

            let remaining = max_records - all_messages.len();

            if let Some(position) = self.offset_tracker.position(tp) {
                match self.fetch_partition(tp, position, remaining) {
                    Ok(result) => {
                        if !result.messages.is_empty() {
                            self.offset_tracker.update_position(tp, result.next_offset);
                            self.offset_tracker.update_high_watermark(tp.clone(), result.high_watermark);

                            self.metrics.record_fetch(result.messages.len());

                            all_messages.extend(result.messages);
                        }
                    }
                    Err(e) => {
                        self.metrics.record_error();
                        // Log error but continue with other partitions
                        tracing::error!("Error fetching from {}: {}", tp, e);
                    }
                }
            }

            // Check timeout
            if start.elapsed() >= timeout {
                break;
            }
        }

        Ok(all_messages)
    }

    /// Fetch messages from a partition.
    fn fetch_partition(
        &self,
        tp: &TopicPartition,
        offset: u64,
        max_messages: usize,
    ) -> Result<FetchResult> {
        let topic = self.topic_manager.get_topic(&tp.topic).ok_or_else(|| {
            Error::TopicNotFound(tp.topic.clone())
        })?;

        topic.fetch(
            tp.partition,
            offset,
            max_messages,
            self.config.fetch_max_bytes,
        )
    }

    /// Commit current offsets.
    pub fn commit(&self) -> Result<()> {
        self.check_closed()?;

        if let Some(ref store) = self.offset_store {
            let positions = self.offset_tracker.all_positions();

            for (tp, offset) in positions {
                store.commit(tp, offset)?;
            }

            store.flush()?;
            self.metrics.record_commit();
        }

        Ok(())
    }

    /// Commit synchronously and wait.
    pub fn commit_sync(&self) -> Result<()> {
        self.commit()
    }

    /// Commit specific offsets.
    pub fn commit_offsets(&self, offsets: &HashMap<TopicPartition, u64>) -> Result<()> {
        self.check_closed()?;

        if let Some(ref store) = self.offset_store {
            for (tp, offset) in offsets {
                store.commit(tp.clone(), *offset)?;
            }
            store.flush()?;
        }

        Ok(())
    }

    /// Seek to a specific offset.
    pub fn seek(&self, tp: &TopicPartition, offset: u64) -> Result<()> {
        self.check_closed()?;

        // Validate offset
        let topic = self.topic_manager.get_topic(&tp.topic).ok_or_else(|| {
            Error::TopicNotFound(tp.topic.clone())
        })?;

        let partition = topic.partition(tp.partition).ok_or(Error::PartitionNotFound {
            topic: tp.topic.clone(),
            partition: tp.partition,
        })?;

        let start = partition.start_offset();
        let end = partition.log_end_offset();

        if offset < start || offset > end {
            return Err(Error::OffsetOutOfRange {
                offset,
                start,
                end,
            });
        }

        self.offset_tracker.seek(tp.clone(), offset);

        Ok(())
    }

    /// Seek to beginning.
    pub fn seek_to_beginning(&self, partitions: &[TopicPartition]) -> Result<()> {
        self.check_closed()?;

        for tp in partitions {
            let topic = self.topic_manager.get_topic(&tp.topic).ok_or_else(|| {
                Error::TopicNotFound(tp.topic.clone())
            })?;

            let partition = topic.partition(tp.partition).ok_or(Error::PartitionNotFound {
                topic: tp.topic.clone(),
                partition: tp.partition,
            })?;

            let start = partition.start_offset();
            self.offset_tracker.seek(tp.clone(), start);
        }

        Ok(())
    }

    /// Seek to end.
    pub fn seek_to_end(&self, partitions: &[TopicPartition]) -> Result<()> {
        self.check_closed()?;

        for tp in partitions {
            let topic = self.topic_manager.get_topic(&tp.topic).ok_or_else(|| {
                Error::TopicNotFound(tp.topic.clone())
            })?;

            let partition = topic.partition(tp.partition).ok_or(Error::PartitionNotFound {
                topic: tp.topic.clone(),
                partition: tp.partition,
            })?;

            let end = partition.log_end_offset();
            self.offset_tracker.seek(tp.clone(), end);
        }

        Ok(())
    }

    /// Get current position for a partition.
    pub fn position(&self, tp: &TopicPartition) -> Option<u64> {
        self.offset_tracker.position(tp)
    }

    /// Get committed offset for a partition.
    pub fn committed(&self, tp: &TopicPartition) -> Option<u64> {
        self.offset_store.as_ref()?.get_offset(tp)
    }

    /// Pause consumption from partitions.
    pub fn pause(&self, partitions: &[TopicPartition]) {
        let mut paused = self.paused.write();
        for tp in partitions {
            paused.insert(tp.clone());
        }
    }

    /// Resume consumption from partitions.
    pub fn resume(&self, partitions: &[TopicPartition]) {
        let mut paused = self.paused.write();
        for tp in partitions {
            paused.remove(tp);
        }
    }

    /// Get paused partitions.
    pub fn paused(&self) -> Vec<TopicPartition> {
        self.paused.read().iter().cloned().collect()
    }

    /// Get assigned partitions.
    pub fn assignment(&self) -> Vec<TopicPartition> {
        self.assignments.read().clone()
    }

    /// Get subscribed topics.
    pub fn subscription(&self) -> Vec<String> {
        self.subscriptions.read().iter().cloned().collect()
    }

    /// Unsubscribe from all topics.
    pub fn unsubscribe(&self) {
        self.subscriptions.write().clear();
        self.assignments.write().clear();
        self.offset_tracker.clear();
    }

    /// Close the consumer.
    pub fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::Release);

        // Commit offsets before closing
        if self.config.enable_auto_commit {
            let _ = self.commit();
        }

        Ok(())
    }

    /// Check if consumer is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(Error::ConsumerClosed)
        } else {
            Ok(())
        }
    }

    /// Get consumer metrics.
    pub fn metrics(&self) -> &ConsumerMetrics {
        &self.metrics
    }

    /// Get configuration.
    pub fn config(&self) -> &ConsumerConfig {
        &self.config
    }

    /// Get lag for all assigned partitions.
    pub fn lag(&self) -> HashMap<TopicPartition, u64> {
        let assignments = self.assignments.read();
        let mut lags = HashMap::new();

        for tp in assignments.iter() {
            if let Some(lag) = self.offset_tracker.lag(tp) {
                lags.insert(tp.clone(), lag);
            }
        }

        lags
    }

    /// Get total lag.
    pub fn total_lag(&self) -> u64 {
        self.offset_tracker.total_lag()
    }
}

/// Consumer metrics.
pub struct ConsumerMetrics {
    /// Total records consumed.
    records_consumed: AtomicU64,
    /// Total bytes consumed.
    bytes_consumed: AtomicU64,
    /// Total fetch operations.
    fetch_count: AtomicU64,
    /// Total commits.
    commit_count: AtomicU64,
    /// Total errors.
    errors: AtomicU64,
    /// Records per second (approximate).
    consume_rate: AtomicU64,
    /// Last rate calculation.
    last_rate_time: Mutex<Instant>,
    /// Records since last rate calculation.
    records_since_rate: AtomicU64,
}

impl ConsumerMetrics {
    /// Create new metrics.
    fn new() -> Self {
        Self {
            records_consumed: AtomicU64::new(0),
            bytes_consumed: AtomicU64::new(0),
            fetch_count: AtomicU64::new(0),
            commit_count: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            consume_rate: AtomicU64::new(0),
            last_rate_time: Mutex::new(Instant::now()),
            records_since_rate: AtomicU64::new(0),
        }
    }

    /// Record a fetch operation.
    fn record_fetch(&self, count: usize) {
        self.records_consumed.fetch_add(count as u64, Ordering::Relaxed);
        self.fetch_count.fetch_add(1, Ordering::Relaxed);
        self.records_since_rate.fetch_add(count as u64, Ordering::Relaxed);

        // Update rate
        let mut last_time = self.last_rate_time.lock();
        if last_time.elapsed() >= Duration::from_secs(1) {
            let records = self.records_since_rate.swap(0, Ordering::Relaxed);
            let elapsed = last_time.elapsed().as_secs_f64();
            let rate = (records as f64 / elapsed) as u64;
            self.consume_rate.store(rate, Ordering::Relaxed);
            *last_time = Instant::now();
        }
    }

    /// Record a commit.
    fn record_commit(&self) {
        self.commit_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an error.
    fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total records consumed.
    pub fn records_consumed(&self) -> u64 {
        self.records_consumed.load(Ordering::Relaxed)
    }

    /// Get total bytes consumed.
    pub fn bytes_consumed(&self) -> u64 {
        self.bytes_consumed.load(Ordering::Relaxed)
    }

    /// Get fetch count.
    pub fn fetch_count(&self) -> u64 {
        self.fetch_count.load(Ordering::Relaxed)
    }

    /// Get commit count.
    pub fn commit_count(&self) -> u64 {
        self.commit_count.load(Ordering::Relaxed)
    }

    /// Get error count.
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Get consume rate.
    pub fn consume_rate(&self) -> u64 {
        self.consume_rate.load(Ordering::Relaxed)
    }
}

/// Consumer builder.
pub struct ConsumerBuilder {
    config: ConsumerConfig,
}

impl ConsumerBuilder {
    /// Create a new consumer builder.
    pub fn new() -> Self {
        Self {
            config: ConsumerConfig::default(),
        }
    }

    /// Set client ID.
    pub fn client_id(mut self, id: impl Into<String>) -> Self {
        self.config.client_id = id.into();
        self
    }

    /// Set group ID.
    pub fn group_id(mut self, id: impl Into<String>) -> Self {
        self.config.group_id = Some(id.into());
        self
    }

    /// Set auto commit.
    pub fn auto_commit(mut self, enabled: bool) -> Self {
        self.config.enable_auto_commit = enabled;
        self
    }

    /// Set offset reset policy.
    pub fn offset_reset(mut self, reset: OffsetReset) -> Self {
        self.config.auto_offset_reset = reset;
        self
    }

    /// Set max poll records.
    pub fn max_poll_records(mut self, records: usize) -> Self {
        self.config.max_poll_records = records;
        self
    }

    /// Set fetch max bytes.
    pub fn fetch_max_bytes(mut self, bytes: usize) -> Self {
        self.config.fetch_max_bytes = bytes;
        self
    }

    /// Build the consumer.
    pub fn build(self, topic_manager: Arc<TopicManager>) -> Result<Consumer> {
        Consumer::new(self.config, topic_manager)
    }
}

impl Default for ConsumerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-export FetchResult for convenience.
pub use crate::partition::FetchResult as FetchResultType;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TopicConfig;
    use tempfile::TempDir;

    fn setup() -> (TempDir, Arc<TopicManager>) {
        let dir = TempDir::new().unwrap();
        let config = TopicConfig::default();
        let manager = Arc::new(TopicManager::new(dir.path(), config).unwrap());
        (dir, manager)
    }

    fn setup_with_messages(topic: &str, count: usize) -> (TempDir, Arc<TopicManager>) {
        let (dir, manager) = setup();

        let topic_arc = manager.get_or_create_topic(topic).unwrap();
        for i in 0..count {
            let msg = Message::new(format!("Message {}", i));
            topic_arc.append_to_partition(0, msg).unwrap();
        }

        (dir, manager)
    }

    #[test]
    fn test_consumer_create() {
        let (_dir, manager) = setup();
        let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();

        assert!(consumer.subscription().is_empty());
        assert!(consumer.assignment().is_empty());
    }

    #[test]
    fn test_consumer_subscribe() {
        let (_dir, manager) = setup();
        manager.create_topic("test-topic", None).unwrap();

        let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();
        consumer.subscribe(&["test-topic"]).unwrap();

        assert!(consumer.subscription().contains(&"test-topic".to_string()));
        assert!(!consumer.assignment().is_empty());
    }

    #[test]
    fn test_consumer_poll() {
        let (_dir, manager) = setup_with_messages("test-topic", 10);

        let config = ConsumerConfig::new().with_offset_reset(OffsetReset::Earliest);
        let consumer = Consumer::new(config, manager).unwrap();
        consumer.subscribe(&["test-topic"]).unwrap();

        let messages = consumer.poll(Duration::from_secs(1)).unwrap();
        assert!(!messages.is_empty());
    }

    #[test]
    fn test_consumer_seek() {
        let (_dir, manager) = setup_with_messages("test-topic", 10);

        let config = ConsumerConfig::new().with_offset_reset(OffsetReset::Earliest);
        let consumer = Consumer::new(config, manager).unwrap();
        consumer.subscribe(&["test-topic"]).unwrap();

        let tp = TopicPartition::new("test-topic", 0);
        consumer.seek(&tp, 5).unwrap();

        assert_eq!(consumer.position(&tp), Some(5));
    }

    #[test]
    fn test_consumer_commit() {
        let (dir, manager) = setup_with_messages("test-topic", 10);

        let config = ConsumerConfig::new()
            .with_group_id("test-group")
            .with_offset_reset(OffsetReset::Earliest);
        let consumer = Consumer::new(config, manager).unwrap();
        consumer.subscribe(&["test-topic"]).unwrap();

        // Poll some messages
        consumer.poll(Duration::from_secs(1)).unwrap();

        // Commit
        consumer.commit().unwrap();

        let tp = TopicPartition::new("test-topic", 0);
        assert!(consumer.committed(&tp).is_some());
    }

    #[test]
    fn test_consumer_pause_resume() {
        let (_dir, manager) = setup_with_messages("test-topic", 10);

        let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();
        consumer.subscribe(&["test-topic"]).unwrap();

        let tp = TopicPartition::new("test-topic", 0);

        consumer.pause(&[tp.clone()]);
        assert!(consumer.paused().contains(&tp));

        consumer.resume(&[tp.clone()]);
        assert!(!consumer.paused().contains(&tp));
    }

    #[test]
    fn test_consumer_close() {
        let (_dir, manager) = setup();
        let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();

        consumer.close().unwrap();

        let result = consumer.poll(Duration::from_millis(100));
        assert!(matches!(result, Err(Error::ConsumerClosed)));
    }

    #[test]
    fn test_consumer_assign() {
        let (_dir, manager) = setup();
        manager.create_topic("test-topic", None).unwrap();

        let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();

        let partitions = vec![
            TopicPartition::new("test-topic", 0),
            TopicPartition::new("test-topic", 1),
        ];

        consumer.assign(&partitions).unwrap();

        let assignment = consumer.assignment();
        assert_eq!(assignment.len(), 2);
    }

    #[test]
    fn test_consumer_unsubscribe() {
        let (_dir, manager) = setup();
        manager.create_topic("test-topic", None).unwrap();

        let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();
        consumer.subscribe(&["test-topic"]).unwrap();

        assert!(!consumer.assignment().is_empty());

        consumer.unsubscribe();

        assert!(consumer.subscription().is_empty());
        assert!(consumer.assignment().is_empty());
    }

    #[test]
    fn test_consumer_metrics() {
        let (_dir, manager) = setup_with_messages("test-topic", 10);

        let config = ConsumerConfig::new().with_offset_reset(OffsetReset::Earliest);
        let consumer = Consumer::new(config, manager).unwrap();
        consumer.subscribe(&["test-topic"]).unwrap();

        consumer.poll(Duration::from_secs(1)).unwrap();

        let metrics = consumer.metrics();
        assert!(metrics.records_consumed() > 0);
        assert!(metrics.fetch_count() > 0);
    }

    #[test]
    fn test_consumer_builder() {
        let (_dir, manager) = setup();

        let consumer = ConsumerBuilder::new()
            .client_id("my-consumer")
            .group_id("my-group")
            .auto_commit(false)
            .offset_reset(OffsetReset::Earliest)
            .max_poll_records(100)
            .build(manager)
            .unwrap();

        assert_eq!(consumer.config().client_id, "my-consumer");
        assert_eq!(consumer.config().group_id, Some("my-group".to_string()));
        assert!(!consumer.config().enable_auto_commit);
    }

    #[test]
    fn test_consumer_seek_to_beginning_end() {
        let (_dir, manager) = setup_with_messages("test-topic", 10);

        let config = ConsumerConfig::new().with_offset_reset(OffsetReset::Latest);
        let consumer = Consumer::new(config, manager).unwrap();
        consumer.subscribe(&["test-topic"]).unwrap();

        let tp = TopicPartition::new("test-topic", 0);

        // Seek to beginning
        consumer.seek_to_beginning(&[tp.clone()]).unwrap();
        assert_eq!(consumer.position(&tp), Some(0));

        // Seek to end
        consumer.seek_to_end(&[tp.clone()]).unwrap();
        assert_eq!(consumer.position(&tp), Some(10));
    }

    #[test]
    fn test_consumer_not_subscribed() {
        let (_dir, manager) = setup();
        let consumer = Consumer::new(ConsumerConfig::default(), manager).unwrap();

        let result = consumer.poll(Duration::from_millis(100));
        assert!(matches!(result, Err(Error::NotSubscribed)));
    }
}
