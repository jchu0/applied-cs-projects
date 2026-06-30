//! Producer API for publishing messages.

use crate::compression::Compression;
use crate::config::{Acks, ProducerConfig};
use crate::error::{Error, Result};
use crate::message::{Message, MessageBatch, MessageBuilder};
use crate::partition::{PartitionId, Partitioner, PartitionStrategy};
use crate::topic::TopicManager;
use parking_lot::{Mutex, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Record metadata returned after sending a message.
#[derive(Debug, Clone)]
pub struct RecordMetadata {
    /// Topic name.
    pub topic: String,
    /// Partition ID.
    pub partition: PartitionId,
    /// Offset within partition.
    pub offset: u64,
    /// Timestamp.
    pub timestamp: u64,
    /// Serialized size.
    pub serialized_size: usize,
}

/// Producer for publishing messages to topics.
pub struct Producer {
    /// Configuration.
    config: ProducerConfig,
    /// Topic manager.
    topic_manager: Arc<TopicManager>,
    /// Partitioners by topic.
    partitioners: RwLock<HashMap<String, Partitioner>>,
    /// Pending batches by topic-partition.
    batches: Mutex<HashMap<String, PendingBatch>>,
    /// Is producer closed.
    closed: AtomicBool,
    /// Metrics.
    metrics: ProducerMetrics,
}

impl Producer {
    /// Create a new producer.
    pub fn new(config: ProducerConfig, topic_manager: Arc<TopicManager>) -> Self {
        Self {
            config,
            topic_manager,
            partitioners: RwLock::new(HashMap::new()),
            batches: Mutex::new(HashMap::new()),
            closed: AtomicBool::new(false),
            metrics: ProducerMetrics::new(),
        }
    }

    /// Send a message to a topic.
    pub fn send(&self, topic: &str, message: Message) -> Result<RecordMetadata> {
        self.check_closed()?;

        // Get or create topic
        let topic_arc = self.topic_manager.get_or_create_topic(topic)?;

        // Get or create partitioner
        let partition = {
            let mut partitioners = self.partitioners.write();
            let partitioner = partitioners.entry(topic.to_string()).or_insert_with(|| {
                Partitioner::new(PartitionStrategy::KeyHash, topic_arc.partition_count())
            });
            partitioner.partition(&message)
        };

        let serialized_size = message.size();

        // If batching is disabled, send immediately
        if self.config.linger_ms == 0 {
            let offset = topic_arc.append_to_partition(partition, message)?;

            self.metrics.record_sent(serialized_size);

            return Ok(RecordMetadata {
                topic: topic.to_string(),
                partition,
                offset,
                timestamp: crate::message::current_timestamp(),
                serialized_size,
            });
        }

        // Add to batch
        self.add_to_batch(topic, partition, message)?;

        // Check if we should flush
        self.maybe_flush_batch(topic, partition)?;

        // For batched sends, we return a placeholder (in a real implementation, this would be async)
        Ok(RecordMetadata {
            topic: topic.to_string(),
            partition,
            offset: 0, // Will be set when batch is actually sent
            timestamp: crate::message::current_timestamp(),
            serialized_size,
        })
    }

    /// Send a message to a specific partition.
    pub fn send_to_partition(
        &self,
        topic: &str,
        partition: PartitionId,
        message: Message,
    ) -> Result<RecordMetadata> {
        self.check_closed()?;

        let topic_arc = self.topic_manager.get_or_create_topic(topic)?;
        let serialized_size = message.size();

        let offset = topic_arc.append_to_partition(partition, message)?;

        self.metrics.record_sent(serialized_size);

        Ok(RecordMetadata {
            topic: topic.to_string(),
            partition,
            offset,
            timestamp: crate::message::current_timestamp(),
            serialized_size,
        })
    }

    /// Send a batch of messages.
    pub fn send_batch(&self, topic: &str, messages: Vec<Message>) -> Result<Vec<RecordMetadata>> {
        self.check_closed()?;

        let topic_arc = self.topic_manager.get_or_create_topic(topic)?;
        let mut results = Vec::with_capacity(messages.len());

        // Group messages by partition
        let mut by_partition: HashMap<PartitionId, Vec<Message>> = HashMap::new();

        {
            let mut partitioners = self.partitioners.write();
            let partitioner = partitioners.entry(topic.to_string()).or_insert_with(|| {
                Partitioner::new(PartitionStrategy::KeyHash, topic_arc.partition_count())
            });

            for msg in messages {
                let partition = partitioner.partition(&msg);
                by_partition.entry(partition).or_default().push(msg);
            }
        }

        // Send to each partition
        for (partition, mut msgs) in by_partition {
            let offsets = topic_arc
                .partition(partition)
                .ok_or(Error::PartitionNotFound {
                    topic: topic.to_string(),
                    partition,
                })?
                .append_batch(&mut msgs)?;

            for (msg, offset) in msgs.into_iter().zip(offsets) {
                let serialized_size = msg.size();
                self.metrics.record_sent(serialized_size);

                results.push(RecordMetadata {
                    topic: topic.to_string(),
                    partition,
                    offset,
                    timestamp: msg.timestamp,
                    serialized_size,
                });
            }
        }

        Ok(results)
    }

    /// Add message to pending batch.
    fn add_to_batch(&self, topic: &str, partition: PartitionId, message: Message) -> Result<()> {
        let key = format!("{}-{}", topic, partition);
        let mut batches = self.batches.lock();

        let batch = batches.entry(key).or_insert_with(|| PendingBatch {
            topic: topic.to_string(),
            partition,
            messages: Vec::new(),
            size: 0,
            created: Instant::now(),
        });

        batch.size += message.size();
        batch.messages.push(message);

        Ok(())
    }

    /// Check if batch should be flushed and flush if needed.
    fn maybe_flush_batch(&self, topic: &str, partition: PartitionId) -> Result<()> {
        let key = format!("{}-{}", topic, partition);

        let should_flush = {
            let batches = self.batches.lock();
            if let Some(batch) = batches.get(&key) {
                batch.size >= self.config.batch_size
                    || batch.created.elapsed().as_millis() >= self.config.linger_ms as u128
            } else {
                false
            }
        };

        if should_flush {
            self.flush_batch(&key)?;
        }

        Ok(())
    }

    /// Flush a specific batch.
    fn flush_batch(&self, key: &str) -> Result<()> {
        let batch = {
            let mut batches = self.batches.lock();
            batches.remove(key)
        };

        if let Some(batch) = batch {
            let topic = self.topic_manager.get_topic(&batch.topic).ok_or_else(|| {
                Error::TopicNotFound(batch.topic.clone())
            })?;

            topic
                .partition(batch.partition)
                .ok_or(Error::PartitionNotFound {
                    topic: batch.topic.clone(),
                    partition: batch.partition,
                })?
                .append_batch(&mut batch.messages.into_iter().collect::<Vec<_>>())?;
        }

        Ok(())
    }

    /// Flush all pending batches.
    pub fn flush(&self) -> Result<()> {
        let keys: Vec<String> = {
            let batches = self.batches.lock();
            batches.keys().cloned().collect()
        };

        for key in keys {
            self.flush_batch(&key)?;
        }

        Ok(())
    }

    /// Close the producer.
    pub fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::Release);
        self.flush()
    }

    /// Check if producer is closed.
    fn check_closed(&self) -> Result<()> {
        if self.closed.load(Ordering::Acquire) {
            Err(Error::ProducerClosed)
        } else {
            Ok(())
        }
    }

    /// Get producer metrics.
    pub fn metrics(&self) -> &ProducerMetrics {
        &self.metrics
    }

    /// Get configuration.
    pub fn config(&self) -> &ProducerConfig {
        &self.config
    }
}

/// Pending batch of messages.
struct PendingBatch {
    topic: String,
    partition: PartitionId,
    messages: Vec<Message>,
    size: usize,
    created: Instant,
}

/// Producer metrics.
pub struct ProducerMetrics {
    /// Total records sent.
    records_sent: AtomicU64,
    /// Total bytes sent.
    bytes_sent: AtomicU64,
    /// Total batches sent.
    batches_sent: AtomicU64,
    /// Total errors.
    errors: AtomicU64,
    /// Records per second (approximate).
    send_rate: AtomicU64,
    /// Last rate calculation time.
    last_rate_time: Mutex<Instant>,
    /// Records since last rate calculation.
    records_since_rate: AtomicU64,
}

impl ProducerMetrics {
    /// Create new metrics.
    fn new() -> Self {
        Self {
            records_sent: AtomicU64::new(0),
            bytes_sent: AtomicU64::new(0),
            batches_sent: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            send_rate: AtomicU64::new(0),
            last_rate_time: Mutex::new(Instant::now()),
            records_since_rate: AtomicU64::new(0),
        }
    }

    /// Record a successful send.
    fn record_sent(&self, bytes: usize) {
        self.records_sent.fetch_add(1, Ordering::Relaxed);
        self.bytes_sent.fetch_add(bytes as u64, Ordering::Relaxed);
        self.records_since_rate.fetch_add(1, Ordering::Relaxed);

        // Update rate every second
        let mut last_time = self.last_rate_time.lock();
        if last_time.elapsed() >= Duration::from_secs(1) {
            let records = self.records_since_rate.swap(0, Ordering::Relaxed);
            let elapsed = last_time.elapsed().as_secs_f64();
            let rate = (records as f64 / elapsed) as u64;
            self.send_rate.store(rate, Ordering::Relaxed);
            *last_time = Instant::now();
        }
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a batch sent.
    pub fn record_batch(&self) {
        self.batches_sent.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total records sent.
    pub fn records_sent(&self) -> u64 {
        self.records_sent.load(Ordering::Relaxed)
    }

    /// Get total bytes sent.
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent.load(Ordering::Relaxed)
    }

    /// Get total batches sent.
    pub fn batches_sent(&self) -> u64 {
        self.batches_sent.load(Ordering::Relaxed)
    }

    /// Get error count.
    pub fn errors(&self) -> u64 {
        self.errors.load(Ordering::Relaxed)
    }

    /// Get approximate send rate (records/second).
    pub fn send_rate(&self) -> u64 {
        self.send_rate.load(Ordering::Relaxed)
    }
}

/// Producer builder for fluent configuration.
pub struct ProducerBuilder {
    config: ProducerConfig,
}

impl ProducerBuilder {
    /// Create a new producer builder.
    pub fn new() -> Self {
        Self {
            config: ProducerConfig::default(),
        }
    }

    /// Set client ID.
    pub fn client_id(mut self, id: impl Into<String>) -> Self {
        self.config.client_id = id.into();
        self
    }

    /// Set acknowledgment mode.
    pub fn acks(mut self, acks: Acks) -> Self {
        self.config.acks = acks;
        self
    }

    /// Set compression.
    pub fn compression(mut self, compression: Compression) -> Self {
        self.config.compression = compression;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    /// Set linger time.
    pub fn linger_ms(mut self, ms: u64) -> Self {
        self.config.linger_ms = ms;
        self
    }

    /// Enable idempotence.
    pub fn idempotence(mut self, enabled: bool) -> Self {
        self.config.enable_idempotence = enabled;
        self
    }

    /// Set retries.
    pub fn retries(mut self, retries: u32) -> Self {
        self.config.retries = retries;
        self
    }

    /// Build the producer.
    pub fn build(self, topic_manager: Arc<TopicManager>) -> Producer {
        Producer::new(self.config, topic_manager)
    }
}

impl Default for ProducerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

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

    #[test]
    fn test_producer_send() {
        let (_dir, manager) = setup();
        let producer = Producer::new(ProducerConfig::default(), manager);

        let msg = Message::new("Hello, World!");
        let result = producer.send("test-topic", msg).unwrap();

        assert_eq!(result.topic, "test-topic");
        assert!(result.partition < 4);
    }

    #[test]
    fn test_producer_send_with_key() {
        let (_dir, manager) = setup();
        let producer = Producer::new(ProducerConfig::default(), manager);

        let msg = Message::with_key("key1", "Hello");
        let result1 = producer.send("test-topic", msg).unwrap();

        let msg = Message::with_key("key1", "World");
        let result2 = producer.send("test-topic", msg).unwrap();

        // Same key should go to same partition
        assert_eq!(result1.partition, result2.partition);
    }

    #[test]
    fn test_producer_send_to_partition() {
        let (_dir, manager) = setup();
        let producer = Producer::new(ProducerConfig::default(), manager);

        let msg = Message::new("Hello");
        let result = producer.send_to_partition("test-topic", 2, msg).unwrap();

        assert_eq!(result.partition, 2);
        assert_eq!(result.offset, 0);
    }

    #[test]
    fn test_producer_send_batch() {
        let (_dir, manager) = setup();
        let producer = Producer::new(ProducerConfig::default(), manager);

        let messages: Vec<Message> = (0..5)
            .map(|i| Message::with_key(format!("key{}", i % 2), format!("Message {}", i)))
            .collect();

        let results = producer.send_batch("test-topic", messages).unwrap();

        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_producer_flush() {
        let (_dir, manager) = setup();
        let mut config = ProducerConfig::default();
        config.linger_ms = 1000; // Enable batching

        let producer = Producer::new(config, manager);

        let msg = Message::new("Hello");
        producer.send("test-topic", msg).unwrap();

        producer.flush().unwrap();
    }

    #[test]
    fn test_producer_close() {
        let (_dir, manager) = setup();
        let producer = Producer::new(ProducerConfig::default(), manager);

        producer.close().unwrap();

        let result = producer.send("test-topic", Message::new("Test"));
        assert!(matches!(result, Err(Error::ProducerClosed)));
    }

    #[test]
    fn test_producer_metrics() {
        let (_dir, manager) = setup();
        let producer = Producer::new(ProducerConfig::default(), manager);

        producer.send("test-topic", Message::new("Hello")).unwrap();
        producer.send("test-topic", Message::new("World")).unwrap();

        let metrics = producer.metrics();
        assert_eq!(metrics.records_sent(), 2);
        assert!(metrics.bytes_sent() > 0);
    }

    #[test]
    fn test_producer_builder() {
        let (_dir, manager) = setup();

        let producer = ProducerBuilder::new()
            .client_id("my-producer")
            .acks(Acks::All)
            .compression(Compression::Lz4)
            .batch_size(32768)
            .build(manager);

        assert_eq!(producer.config().client_id, "my-producer");
        assert_eq!(producer.config().acks, Acks::All);
        assert_eq!(producer.config().compression, Compression::Lz4);
    }

    #[test]
    fn test_record_metadata() {
        let metadata = RecordMetadata {
            topic: "test".to_string(),
            partition: 0,
            offset: 100,
            timestamp: 12345,
            serialized_size: 50,
        };

        assert_eq!(metadata.topic, "test");
        assert_eq!(metadata.partition, 0);
        assert_eq!(metadata.offset, 100);
    }

    #[test]
    fn test_producer_metrics_rate() {
        let metrics = ProducerMetrics::new();

        for _ in 0..100 {
            metrics.record_sent(100);
        }

        assert_eq!(metrics.records_sent(), 100);
        assert_eq!(metrics.bytes_sent(), 10000);
    }

    #[test]
    fn test_producer_multiple_topics() {
        let (_dir, manager) = setup();
        let producer = Producer::new(ProducerConfig::default(), manager);

        producer.send("topic1", Message::new("msg1")).unwrap();
        producer.send("topic2", Message::new("msg2")).unwrap();
        producer.send("topic3", Message::new("msg3")).unwrap();

        assert_eq!(producer.metrics().records_sent(), 3);
    }
}
