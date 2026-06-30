//! Producer API for sending records to the log.

use crate::broker::BrokerId;
use crate::log::{Record, RecordBatch, TopicPartition};
use crate::protocol::{PartitionProduceData, ProduceRequest, TopicProduceData};
use crate::{Error, Offset, Result, Timestamp};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;
use tracing::debug;

/// Acknowledgment level for produce requests.
#[derive(Debug, Clone, Copy)]
pub enum Acks {
    /// Don't wait for acknowledgment (fire and forget).
    None,
    /// Wait for leader acknowledgment.
    Leader,
    /// Wait for all in-sync replicas.
    All,
}

impl Acks {
    pub fn to_i16(&self) -> i16 {
        match self {
            Acks::None => 0,
            Acks::Leader => 1,
            Acks::All => -1,
        }
    }
}

/// Producer configuration.
#[derive(Debug, Clone)]
pub struct ProducerConfig {
    /// Client identifier.
    pub client_id: String,
    /// Acknowledgment level.
    pub acks: Acks,
    /// Request timeout.
    pub timeout_ms: u32,
    /// Maximum batch size.
    pub batch_size: usize,
    /// Linger time for batching.
    pub linger_ms: u64,
    /// Maximum request size.
    pub max_request_size: usize,
    /// Enable idempotence.
    pub enable_idempotence: bool,
    /// Maximum retries.
    pub retries: u32,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            client_id: "kafka-lite-producer".to_string(),
            acks: Acks::Leader,
            timeout_ms: 30000,
            batch_size: 16384,
            linger_ms: 0,
            max_request_size: 1048576,
            enable_idempotence: false,
            retries: 3,
        }
    }
}

/// A record to produce.
#[derive(Debug, Clone)]
pub struct ProducerRecord {
    /// Topic name.
    pub topic: String,
    /// Partition (optional, will be determined by partitioner if not set).
    pub partition: Option<u32>,
    /// Record key.
    pub key: Option<Vec<u8>>,
    /// Record value.
    pub value: Option<Vec<u8>>,
    /// Record timestamp.
    pub timestamp: Option<Timestamp>,
    /// Record headers.
    pub headers: Vec<(String, Vec<u8>)>,
}

impl ProducerRecord {
    /// Create a new producer record.
    pub fn new(topic: impl Into<String>, key: Option<Vec<u8>>, value: Option<Vec<u8>>) -> Self {
        Self {
            topic: topic.into(),
            partition: None,
            key,
            value,
            timestamp: None,
            headers: Vec::new(),
        }
    }

    /// Set partition.
    pub fn with_partition(mut self, partition: u32) -> Self {
        self.partition = Some(partition);
        self
    }

    /// Add header.
    pub fn with_header(mut self, key: impl Into<String>, value: Vec<u8>) -> Self {
        self.headers.push((key.into(), value));
        self
    }
}

/// Metadata returned after successful produce.
#[derive(Debug, Clone)]
pub struct RecordMetadata {
    /// Topic name.
    pub topic: String,
    /// Partition.
    pub partition: u32,
    /// Offset of the record.
    pub offset: Offset,
    /// Timestamp of the record.
    pub timestamp: Timestamp,
    /// Serialized size in bytes.
    pub serialized_size: usize,
}

/// Record accumulator for batching.
pub struct RecordAccumulator {
    /// Batches by partition.
    batches: HashMap<TopicPartition, Vec<Record>>,
    /// Batch size.
    batch_size: usize,
    /// Current sizes by partition.
    sizes: HashMap<TopicPartition, usize>,
}

impl RecordAccumulator {
    /// Create a new accumulator.
    pub fn new(config: &ProducerConfig) -> Self {
        Self {
            batches: HashMap::new(),
            batch_size: config.batch_size,
            sizes: HashMap::new(),
        }
    }

    /// Append a record to the accumulator.
    pub fn append(&mut self, tp: TopicPartition, record: Record) -> bool {
        let size = record.key.as_ref().map(|k| k.len()).unwrap_or(0)
            + record.value.as_ref().map(|v| v.len()).unwrap_or(0);

        let current_size = self.sizes.entry(tp.clone()).or_insert(0);
        *current_size += size;

        self.batches.entry(tp.clone()).or_default().push(record);

        *current_size >= self.batch_size
    }

    /// Check if partition has a ready batch.
    pub fn is_ready(&self, tp: &TopicPartition) -> bool {
        self.sizes.get(tp).copied().unwrap_or(0) >= self.batch_size
    }

    /// Drain batch for partition.
    pub fn drain(&mut self, tp: &TopicPartition) -> Vec<Record> {
        self.sizes.remove(tp);
        self.batches.remove(tp).unwrap_or_default()
    }

    /// Get all ready partitions.
    pub fn ready_partitions(&self) -> Vec<TopicPartition> {
        self.sizes
            .iter()
            .filter(|(_, size)| **size >= self.batch_size)
            .map(|(tp, _)| tp.clone())
            .collect()
    }
}

/// Kafka-lite producer.
pub struct Producer {
    /// Configuration.
    config: ProducerConfig,
    /// Record accumulator.
    accumulator: RecordAccumulator,
    /// Partition counter for round-robin.
    partition_counter: AtomicU32,
    /// Producer ID for idempotence.
    producer_id: i64,
    /// Producer epoch.
    producer_epoch: i16,
    /// Sequence numbers by partition.
    sequences: HashMap<TopicPartition, u32>,
}

impl Producer {
    /// Create a new producer.
    pub fn new(config: ProducerConfig) -> Self {
        Self {
            accumulator: RecordAccumulator::new(&config),
            config,
            partition_counter: AtomicU32::new(0),
            producer_id: -1,
            producer_epoch: -1,
            sequences: HashMap::new(),
        }
    }

    /// Send a record.
    pub fn send(&mut self, record: ProducerRecord) -> Result<TopicPartition> {
        // Determine partition
        let partition = match record.partition {
            Some(p) => p,
            None => self.partition_for(&record),
        };

        let tp = TopicPartition::new(&record.topic, partition);

        // Convert to internal record
        let internal_record = Record {
            attributes: 0,
            timestamp_delta: 0,
            offset_delta: 0,
            key: record.key,
            value: record.value,
            headers: record
                .headers
                .into_iter()
                .map(|(k, v)| crate::log::Header { key: k, value: v })
                .collect(),
        };

        // Add to accumulator
        let _ready = self.accumulator.append(tp.clone(), internal_record);

        Ok(tp)
    }

    /// Determine partition for a record.
    fn partition_for(&self, record: &ProducerRecord) -> u32 {
        if let Some(key) = &record.key {
            // Hash key to determine partition (simplified)
            let hash = crc32fast::hash(key);
            hash % 3 // Assume 3 partitions
        } else {
            // Round-robin
            self.partition_counter.fetch_add(1, Ordering::Relaxed) % 3
        }
    }

    /// Build produce request for ready batches.
    pub fn build_produce_request(&mut self) -> Option<ProduceRequest> {
        let ready = self.accumulator.ready_partitions();
        if ready.is_empty() {
            return None;
        }

        let mut topic_data: HashMap<String, Vec<PartitionProduceData>> = HashMap::new();

        for tp in ready {
            let records = self.accumulator.drain(&tp);
            if records.is_empty() {
                continue;
            }

            let batch = RecordBatch::new(0, records);

            let data = PartitionProduceData {
                partition: tp.partition,
                records: batch,
            };

            topic_data
                .entry(tp.topic.clone())
                .or_default()
                .push(data);
        }

        let topic_data: Vec<TopicProduceData> = topic_data
            .into_iter()
            .map(|(topic, partition_data)| TopicProduceData {
                topic,
                partition_data,
            })
            .collect();

        Some(ProduceRequest {
            transactional_id: None,
            acks: self.config.acks.to_i16(),
            timeout_ms: self.config.timeout_ms,
            topic_data,
        })
    }

    /// Flush all pending records.
    pub fn flush(&mut self) -> Vec<TopicPartition> {
        let partitions: Vec<_> = self.accumulator.batches.keys().cloned().collect();
        partitions
    }
}
