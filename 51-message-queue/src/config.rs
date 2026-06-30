//! Configuration types for the message queue.

use crate::compression::Compression;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Broker configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokerConfig {
    /// Broker ID.
    pub broker_id: u32,
    /// Data directory.
    pub data_dir: PathBuf,
    /// Log directory.
    pub log_dir: PathBuf,
    /// Host to bind to.
    pub host: String,
    /// Port to listen on.
    pub port: u16,
    /// Default number of partitions for new topics.
    pub default_partitions: u32,
    /// Default replication factor.
    pub default_replication_factor: u32,
    /// Default retention in milliseconds.
    pub retention_ms: u64,
    /// Default retention in bytes.
    pub retention_bytes: u64,
    /// Default segment size.
    pub segment_bytes: u64,
    /// Default compression.
    pub compression: Compression,
    /// Maximum message size.
    pub max_message_bytes: usize,
    /// Maximum batch size.
    pub max_batch_size: usize,
    /// Auto-create topics.
    pub auto_create_topics: bool,
    /// Minimum in-sync replicas.
    pub min_insync_replicas: u32,
    /// Flush interval (messages).
    pub flush_messages: u64,
    /// Flush interval (milliseconds).
    pub flush_ms: u64,
    /// Number of I/O threads.
    pub io_threads: usize,
    /// Number of network threads.
    pub network_threads: usize,
    /// Socket receive buffer size.
    pub socket_receive_buffer_bytes: usize,
    /// Socket send buffer size.
    pub socket_send_buffer_bytes: usize,
    /// Maximum connections.
    pub max_connections: usize,
    /// Connection timeout.
    pub connection_timeout_ms: u64,
    /// Request timeout.
    pub request_timeout_ms: u64,
}

impl Default for BrokerConfig {
    fn default() -> Self {
        Self {
            broker_id: 1,
            data_dir: PathBuf::from("/tmp/mq/data"),
            log_dir: PathBuf::from("/tmp/mq/logs"),
            host: "127.0.0.1".to_string(),
            port: 9092,
            default_partitions: 4,
            default_replication_factor: 1,
            retention_ms: 7 * 24 * 60 * 60 * 1000, // 7 days
            retention_bytes: u64::MAX,
            segment_bytes: 1024 * 1024 * 1024, // 1GB
            compression: Compression::None,
            max_message_bytes: 1024 * 1024, // 1MB
            max_batch_size: 16384,          // 16KB
            auto_create_topics: true,
            min_insync_replicas: 1,
            flush_messages: 10000,
            flush_ms: 1000,
            io_threads: 4,
            network_threads: 4,
            socket_receive_buffer_bytes: 102400,
            socket_send_buffer_bytes: 102400,
            max_connections: 10000,
            connection_timeout_ms: 30000,
            request_timeout_ms: 30000,
        }
    }
}

impl BrokerConfig {
    /// Create a new broker config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set broker ID.
    pub fn with_broker_id(mut self, id: u32) -> Self {
        self.broker_id = id;
        self
    }

    /// Set data directory.
    pub fn with_data_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.data_dir = dir.into();
        self
    }

    /// Set log directory.
    pub fn with_log_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.log_dir = dir.into();
        self
    }

    /// Set host.
    pub fn with_host(mut self, host: impl Into<String>) -> Self {
        self.host = host.into();
        self
    }

    /// Set port.
    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    /// Set default partitions.
    pub fn with_default_partitions(mut self, partitions: u32) -> Self {
        self.default_partitions = partitions;
        self
    }

    /// Set retention in milliseconds.
    pub fn with_retention_ms(mut self, ms: u64) -> Self {
        self.retention_ms = ms;
        self
    }

    /// Set segment size.
    pub fn with_segment_bytes(mut self, bytes: u64) -> Self {
        self.segment_bytes = bytes;
        self
    }

    /// Set compression.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Get socket address.
    pub fn socket_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.default_partitions == 0 {
            return Err("default_partitions must be > 0".to_string());
        }
        if self.segment_bytes < 1024 {
            return Err("segment_bytes must be >= 1024".to_string());
        }
        if self.max_message_bytes == 0 {
            return Err("max_message_bytes must be > 0".to_string());
        }
        if self.max_message_bytes > self.segment_bytes as usize {
            return Err("max_message_bytes must be <= segment_bytes".to_string());
        }
        Ok(())
    }
}

/// Topic configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicConfig {
    /// Number of partitions.
    pub partition_count: u32,
    /// Replication factor.
    pub replication_factor: u32,
    /// Retention in bytes.
    pub retention_bytes: u64,
    /// Retention in milliseconds.
    pub retention_ms: u64,
    /// Segment size in bytes.
    pub segment_bytes: u64,
    /// Compression type.
    pub compression: Compression,
    /// Minimum in-sync replicas.
    pub min_insync_replicas: u32,
    /// Cleanup policy.
    pub cleanup_policy: CleanupPolicy,
    /// Maximum message size.
    pub max_message_bytes: usize,
    /// Message timestamp type.
    pub message_timestamp_type: TimestampType,
    /// Flush interval (messages).
    pub flush_messages: u64,
    /// Flush interval (milliseconds).
    pub flush_ms: u64,
}

impl Default for TopicConfig {
    fn default() -> Self {
        Self {
            partition_count: 4,
            replication_factor: 1,
            retention_bytes: u64::MAX,
            retention_ms: 7 * 24 * 60 * 60 * 1000, // 7 days
            segment_bytes: 1024 * 1024 * 1024,     // 1GB
            compression: Compression::None,
            min_insync_replicas: 1,
            cleanup_policy: CleanupPolicy::Delete,
            max_message_bytes: 1024 * 1024, // 1MB
            message_timestamp_type: TimestampType::CreateTime,
            flush_messages: 10000,
            flush_ms: 1000,
        }
    }
}

impl TopicConfig {
    /// Create a new topic config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set partition count.
    pub fn with_partitions(mut self, count: u32) -> Self {
        self.partition_count = count;
        self
    }

    /// Set replication factor.
    pub fn with_replication(mut self, factor: u32) -> Self {
        self.replication_factor = factor;
        self
    }

    /// Set retention in milliseconds.
    pub fn with_retention_ms(mut self, ms: u64) -> Self {
        self.retention_ms = ms;
        self
    }

    /// Set retention in bytes.
    pub fn with_retention_bytes(mut self, bytes: u64) -> Self {
        self.retention_bytes = bytes;
        self
    }

    /// Set segment size.
    pub fn with_segment_bytes(mut self, bytes: u64) -> Self {
        self.segment_bytes = bytes;
        self
    }

    /// Set compression.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set cleanup policy.
    pub fn with_cleanup_policy(mut self, policy: CleanupPolicy) -> Self {
        self.cleanup_policy = policy;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.partition_count == 0 {
            return Err("partition_count must be > 0".to_string());
        }
        if self.replication_factor == 0 {
            return Err("replication_factor must be > 0".to_string());
        }
        if self.segment_bytes < 1024 {
            return Err("segment_bytes must be >= 1024".to_string());
        }
        if self.min_insync_replicas > self.replication_factor {
            return Err("min_insync_replicas must be <= replication_factor".to_string());
        }
        Ok(())
    }
}

/// Cleanup policy for topics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CleanupPolicy {
    /// Delete old segments.
    #[default]
    Delete,
    /// Compact messages by key.
    Compact,
    /// Both delete and compact.
    CompactDelete,
}

impl std::fmt::Display for CleanupPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CleanupPolicy::Delete => write!(f, "delete"),
            CleanupPolicy::Compact => write!(f, "compact"),
            CleanupPolicy::CompactDelete => write!(f, "compact,delete"),
        }
    }
}

/// Timestamp type for messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TimestampType {
    /// Use creation timestamp from producer.
    #[default]
    CreateTime,
    /// Use log append timestamp from broker.
    LogAppendTime,
}

/// Producer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProducerConfig {
    /// Bootstrap servers.
    pub bootstrap_servers: Vec<String>,
    /// Client ID.
    pub client_id: String,
    /// Acknowledgments required.
    pub acks: Acks,
    /// Number of retries.
    pub retries: u32,
    /// Retry backoff.
    pub retry_backoff_ms: u64,
    /// Batch size.
    pub batch_size: usize,
    /// Linger time.
    pub linger_ms: u64,
    /// Buffer memory.
    pub buffer_memory: usize,
    /// Maximum request size.
    pub max_request_size: usize,
    /// Compression type.
    pub compression: Compression,
    /// Request timeout.
    pub request_timeout_ms: u64,
    /// Delivery timeout.
    pub delivery_timeout_ms: u64,
    /// Enable idempotent producer.
    pub enable_idempotence: bool,
    /// Maximum inflight requests.
    pub max_inflight_requests: u32,
    /// Transaction ID.
    pub transactional_id: Option<String>,
    /// Transaction timeout.
    pub transaction_timeout_ms: u64,
}

impl Default for ProducerConfig {
    fn default() -> Self {
        Self {
            bootstrap_servers: vec!["localhost:9092".to_string()],
            client_id: "producer-1".to_string(),
            acks: Acks::All,
            retries: 3,
            retry_backoff_ms: 100,
            batch_size: 16384,
            linger_ms: 0,
            buffer_memory: 32 * 1024 * 1024,
            max_request_size: 1024 * 1024,
            compression: Compression::None,
            request_timeout_ms: 30000,
            delivery_timeout_ms: 120000,
            enable_idempotence: false,
            max_inflight_requests: 5,
            transactional_id: None,
            transaction_timeout_ms: 60000,
        }
    }
}

impl ProducerConfig {
    /// Create a new producer config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set client ID.
    pub fn with_client_id(mut self, id: impl Into<String>) -> Self {
        self.client_id = id.into();
        self
    }

    /// Set acknowledgments.
    pub fn with_acks(mut self, acks: Acks) -> Self {
        self.acks = acks;
        self
    }

    /// Set compression.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Enable idempotence.
    pub fn with_idempotence(mut self, enabled: bool) -> Self {
        self.enable_idempotence = enabled;
        self
    }

    /// Set batch size.
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set linger time.
    pub fn with_linger_ms(mut self, ms: u64) -> Self {
        self.linger_ms = ms;
        self
    }
}

/// Acknowledgment mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Acks {
    /// No acknowledgment required.
    None,
    /// Leader acknowledgment.
    Leader,
    /// All replicas acknowledgment.
    #[default]
    All,
}

impl Acks {
    /// Get as i16 value.
    pub fn as_i16(&self) -> i16 {
        match self {
            Acks::None => 0,
            Acks::Leader => 1,
            Acks::All => -1,
        }
    }
}

/// Consumer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerConfig {
    /// Bootstrap servers.
    pub bootstrap_servers: Vec<String>,
    /// Client ID.
    pub client_id: String,
    /// Consumer group ID.
    pub group_id: Option<String>,
    /// Auto commit.
    pub enable_auto_commit: bool,
    /// Auto commit interval.
    pub auto_commit_interval_ms: u64,
    /// Auto offset reset.
    pub auto_offset_reset: OffsetReset,
    /// Fetch minimum bytes.
    pub fetch_min_bytes: usize,
    /// Fetch maximum bytes.
    pub fetch_max_bytes: usize,
    /// Fetch maximum wait.
    pub fetch_max_wait_ms: u64,
    /// Maximum poll records.
    pub max_poll_records: usize,
    /// Maximum poll interval.
    pub max_poll_interval_ms: u64,
    /// Session timeout.
    pub session_timeout_ms: u64,
    /// Heartbeat interval.
    pub heartbeat_interval_ms: u64,
    /// Partition assignment strategy.
    pub partition_assignment_strategy: AssignmentStrategy,
    /// Isolation level.
    pub isolation_level: IsolationLevel,
}

impl Default for ConsumerConfig {
    fn default() -> Self {
        Self {
            bootstrap_servers: vec!["localhost:9092".to_string()],
            client_id: "consumer-1".to_string(),
            group_id: None,
            enable_auto_commit: true,
            auto_commit_interval_ms: 5000,
            auto_offset_reset: OffsetReset::Latest,
            fetch_min_bytes: 1,
            fetch_max_bytes: 52428800,
            fetch_max_wait_ms: 500,
            max_poll_records: 500,
            max_poll_interval_ms: 300000,
            session_timeout_ms: 10000,
            heartbeat_interval_ms: 3000,
            partition_assignment_strategy: AssignmentStrategy::Range,
            isolation_level: IsolationLevel::ReadUncommitted,
        }
    }
}

impl ConsumerConfig {
    /// Create a new consumer config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set client ID.
    pub fn with_client_id(mut self, id: impl Into<String>) -> Self {
        self.client_id = id.into();
        self
    }

    /// Set group ID.
    pub fn with_group_id(mut self, id: impl Into<String>) -> Self {
        self.group_id = Some(id.into());
        self
    }

    /// Set auto commit.
    pub fn with_auto_commit(mut self, enabled: bool) -> Self {
        self.enable_auto_commit = enabled;
        self
    }

    /// Set offset reset policy.
    pub fn with_offset_reset(mut self, reset: OffsetReset) -> Self {
        self.auto_offset_reset = reset;
        self
    }

    /// Set max poll records.
    pub fn with_max_poll_records(mut self, records: usize) -> Self {
        self.max_poll_records = records;
        self
    }
}

/// Offset reset policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OffsetReset {
    /// Start from earliest available offset.
    Earliest,
    /// Start from latest offset.
    #[default]
    Latest,
    /// Fail if no offset found.
    None,
}

/// Partition assignment strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AssignmentStrategy {
    /// Range assignment.
    #[default]
    Range,
    /// Round-robin assignment.
    RoundRobin,
    /// Sticky assignment.
    Sticky,
    /// Cooperative sticky assignment.
    CooperativeSticky,
}

/// Transaction isolation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IsolationLevel {
    /// Read uncommitted transactions.
    #[default]
    ReadUncommitted,
    /// Read only committed transactions.
    ReadCommitted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broker_config_default() {
        let config = BrokerConfig::default();
        assert_eq!(config.broker_id, 1);
        assert_eq!(config.port, 9092);
        assert_eq!(config.default_partitions, 4);
    }

    #[test]
    fn test_broker_config_builder() {
        let config = BrokerConfig::new()
            .with_broker_id(2)
            .with_port(9093)
            .with_default_partitions(8);

        assert_eq!(config.broker_id, 2);
        assert_eq!(config.port, 9093);
        assert_eq!(config.default_partitions, 8);
    }

    #[test]
    fn test_broker_config_validate() {
        let config = BrokerConfig::new();
        assert!(config.validate().is_ok());

        let invalid = BrokerConfig::new().with_default_partitions(0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_topic_config_default() {
        let config = TopicConfig::default();
        assert_eq!(config.partition_count, 4);
        assert_eq!(config.replication_factor, 1);
    }

    #[test]
    fn test_topic_config_builder() {
        let config = TopicConfig::new()
            .with_partitions(16)
            .with_replication(3)
            .with_compression(Compression::Lz4);

        assert_eq!(config.partition_count, 16);
        assert_eq!(config.replication_factor, 3);
        assert_eq!(config.compression, Compression::Lz4);
    }

    #[test]
    fn test_topic_config_validate() {
        let config = TopicConfig::new();
        assert!(config.validate().is_ok());

        let invalid = TopicConfig::new().with_partitions(0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_producer_config() {
        let config = ProducerConfig::new()
            .with_client_id("my-producer")
            .with_acks(Acks::Leader)
            .with_compression(Compression::Snappy);

        assert_eq!(config.client_id, "my-producer");
        assert_eq!(config.acks, Acks::Leader);
        assert_eq!(config.compression, Compression::Snappy);
    }

    #[test]
    fn test_consumer_config() {
        let config = ConsumerConfig::new()
            .with_client_id("my-consumer")
            .with_group_id("my-group")
            .with_auto_commit(false);

        assert_eq!(config.client_id, "my-consumer");
        assert_eq!(config.group_id, Some("my-group".to_string()));
        assert!(!config.enable_auto_commit);
    }

    #[test]
    fn test_acks() {
        assert_eq!(Acks::None.as_i16(), 0);
        assert_eq!(Acks::Leader.as_i16(), 1);
        assert_eq!(Acks::All.as_i16(), -1);
    }

    #[test]
    fn test_socket_addr() {
        let config = BrokerConfig::new().with_host("0.0.0.0").with_port(9092);
        assert_eq!(config.socket_addr(), "0.0.0.0:9092");
    }
}
