//! High-Performance Message Queue
//!
//! A production-grade message queue system with:
//! - Persistent message storage with Write-Ahead Logging
//! - Topic-based pub/sub with partitioning
//! - Consumer groups with rebalancing
//! - At-least-once and exactly-once delivery semantics
//! - Message ordering guarantees within partitions

pub mod message;
pub mod storage;
pub mod topic;
pub mod partition;
pub mod producer;
pub mod consumer;
pub mod consumer_group;
pub mod broker;
pub mod config;
pub mod error;
pub mod compression;
pub mod index;
pub mod segment;
pub mod offset;
pub mod replication;
pub mod transaction;

pub use message::{Message, MessageBatch, MessageId, Headers, MessageBuilder};
pub use storage::{Storage, StorageConfig};
pub use topic::{Topic, TopicManager};
pub use config::TopicConfig;
pub use partition::{Partition, FetchResult};
pub use producer::{Producer, RecordMetadata};
pub use config::ProducerConfig;
pub use consumer::Consumer;
pub use config::ConsumerConfig;
pub use consumer_group::{ConsumerGroup, GroupConfig, GroupState};
pub use broker::Broker;
pub use config::BrokerConfig;
pub use error::{Error, Result};
pub use compression::Compression;
pub use replication::{ReplicationManager, ReplicaSet, ReplicationConfig, Acks};
pub use transaction::{TransactionCoordinator, ProducerState, TransactionState};

/// Message queue version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default segment size (1GB)
pub const DEFAULT_SEGMENT_SIZE: u64 = 1024 * 1024 * 1024;

/// Default retention period (7 days in milliseconds)
pub const DEFAULT_RETENTION_MS: u64 = 7 * 24 * 60 * 60 * 1000;

/// Default batch size
pub const DEFAULT_BATCH_SIZE: usize = 16384;

/// Default fetch max bytes
pub const DEFAULT_FETCH_MAX_BYTES: usize = 52428800; // 50MB
