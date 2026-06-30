//! Distributed Log System (Kafka-lite)
//!
//! A high-throughput distributed commit log system providing durable, ordered,
//! append-only log storage with partitioning and replication.

pub mod broker;
pub mod cleaner;
pub mod consumer;
pub mod controller;
pub mod group;
pub mod idempotent;
pub mod log;
pub mod producer;
pub mod protocol;
pub mod replication;
pub mod transport;

pub use broker::{Broker, BrokerConfig, BrokerId, BrokerMetrics};
pub use cleaner::{CleanerConfig, CleanupPolicy, LogCleaner, LogCompactor, RetentionManager};
pub use consumer::{Consumer, ConsumerConfig, ConsumerGroup, ConsumerRecord};
pub use controller::{Controller, ControllerConfig, BrokerInfo, ClusterState};
pub use group::{ConsumerGroup as GroupState, GroupCoordinatorService, GroupCoordinatorConfig};
pub use log::{
    IndexEntry, LogSegment, Partition, RecordBatch, SegmentConfig, TimeIndexEntry, TopicPartition,
};
pub use producer::{Acks, Producer, ProducerConfig, ProducerRecord, RecordMetadata};
pub use protocol::{
    FetchRequest, FetchResponse, ProduceRequest, ProduceResponse, Request, Response,
};
pub use replication::{ReplicaManager, ReplicaFetcher, ReplicationConfig};
pub use transport::{BrokerClient, BrokerHandler, GroupCoordinatorClient, KafkaServerBuilder, TransportConfig};
pub use idempotent::{IdempotentProducer, ProducerStateManager, SequenceTracker};

/// Offset type (position in the log).
pub type Offset = u64;

/// Timestamp type (milliseconds since epoch).
pub type Timestamp = i64;

/// Error types for the log system.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid offset: {0}")]
    InvalidOffset(Offset),

    #[error("Topic not found: {0}")]
    TopicNotFound(String),

    #[error("Partition not found: {0}")]
    PartitionNotFound(u32),

    #[error("Not leader for partition")]
    NotLeader,

    #[error("Broker not available: {0}")]
    BrokerNotAvailable(BrokerId),

    #[error("Consumer group error: {0}")]
    ConsumerGroup(String),

    #[error("Replication error: {0}")]
    Replication(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Result type alias.
pub type Result<T> = std::result::Result<T, Error>;
