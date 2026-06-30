# High-Performance Message Queue

## Executive Summary

A production-grade, high-performance message queue system implementing:
- Persistent message storage with WAL (Write-Ahead Logging)
- Topic-based pub/sub with partitioning
- Consumer groups with rebalancing
- At-least-once and exactly-once delivery semantics
- Message ordering guarantees within partitions
- Batched producer/consumer APIs for throughput
- Acknowledgment-based consumption with configurable timeouts

> **Concepts covered:** [§01 Rust async](../../01-software-engineering/rust/05-async-rust/rust-async.md) (Tokio-based I/O) · [§02 Streaming — Kafka](../../02-data-engineering/04-streaming/kafka/) · [§02 Real-time analytics](../../02-data-engineering/04-streaming/real-time-analytics/). Pairs with [Project 12 (distributed log)](../12-distributed-log-system/) and underpins [Project 08 (streaming platform)](../08-streaming-platform/) and [Project 36 (streaming analytics)](../36-distributed-streaming-analytics/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## System Architecture

```
                         ┌─────────────────────────────────────────┐
                         │            Message Queue                │
                         │                                         │
    ┌──────────┐        ┌─────────────────────────────────────────┤
    │ Producer │───────▶│  Producer API                           │
    │   API    │        │  - Batch sending                        │
    └──────────┘        │  - Partitioning                         │
                        │  - Compression                          │
                        └────────────────┬────────────────────────┘
                                         │
                                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                     Topic Manager                            │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │                    Topic: orders                        ││
    │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  ││
    │  │  │Partition │ │Partition │ │Partition │ │Partition │  ││
    │  │  │    0     │ │    1     │ │    2     │ │    3     │  ││
    │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  ││
    │  └─────────────────────────────────────────────────────────┘│
    └──────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    Storage Layer                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
    │  │ Write-Ahead │  │   Segment   │  │   Index Manager     │  │
    │  │    Log      │  │   Files     │  │   (offset → pos)    │  │
    │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
    └──────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                  Consumer API                                │
    │  ┌─────────────────────────────────────────────────────────┐│
    │  │              Consumer Group: order-processors           ││
    │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐               ││
    │  │  │Consumer 1│ │Consumer 2│ │Consumer 3│               ││
    │  │  │ P0, P1   │ │   P2     │ │   P3     │               ││
    │  │  └──────────┘ └──────────┘ └──────────┘               ││
    │  └─────────────────────────────────────────────────────────┘│
    └──────────────────────────────────────────────────────────────┘
```

---

## Core Data Structures

### Message

```rust
pub struct Message {
    /// Unique message ID
    pub id: MessageId,
    /// Message key for partitioning
    pub key: Option<Vec<u8>>,
    /// Message payload
    pub payload: Vec<u8>,
    /// Headers/metadata
    pub headers: HashMap<String, Vec<u8>>,
    /// Timestamp
    pub timestamp: u64,
    /// Offset within partition
    pub offset: u64,
}

pub struct MessageBatch {
    pub messages: Vec<Message>,
    pub compression: Compression,
}

pub enum Compression {
    None,
    Gzip,
    Lz4,
    Snappy,
}
```

### Topic and Partition

```rust
pub struct Topic {
    pub name: String,
    pub partitions: Vec<Partition>,
    pub config: TopicConfig,
}

pub struct TopicConfig {
    pub partition_count: u32,
    pub replication_factor: u32,
    pub retention_bytes: u64,
    pub retention_ms: u64,
    pub segment_bytes: u64,
    pub compression: Compression,
}

pub struct Partition {
    pub id: u32,
    pub topic: String,
    pub log: Log,
    pub high_watermark: u64,
    pub leader_epoch: u32,
}
```

### Storage

```rust
pub struct Log {
    pub dir: PathBuf,
    pub segments: Vec<Segment>,
    pub active_segment: Segment,
    pub index: Index,
}

pub struct Segment {
    pub base_offset: u64,
    pub file: File,
    pub size: u64,
    pub max_size: u64,
}

pub struct Index {
    pub entries: BTreeMap<u64, u64>,  // offset -> file position
}
```

### Consumer Group

```rust
pub struct ConsumerGroup {
    pub id: String,
    pub members: HashMap<String, GroupMember>,
    pub assignments: HashMap<String, Vec<PartitionAssignment>>,
    pub generation: u32,
    pub state: GroupState,
}

pub struct GroupMember {
    pub id: String,
    pub client_id: String,
    pub session_timeout: Duration,
    pub last_heartbeat: Instant,
}

pub struct PartitionAssignment {
    pub topic: String,
    pub partition: u32,
    pub offset: u64,
}

pub enum GroupState {
    Empty,
    PreparingRebalance,
    CompletingRebalance,
    Stable,
    Dead,
}
```

---

## Implementation Phases

### Phase 1: Core Message Types (Week 1)
- [ ] Message struct with serialization
- [ ] MessageId generation (UUID or snowflake)
- [ ] Headers support
- [ ] Timestamp handling
- [ ] Compression support

### Phase 2: Storage Layer (Week 2)
- [ ] Segment file management
- [ ] Write-ahead logging
- [ ] Offset index
- [ ] File rotation
- [ ] Retention policy enforcement

### Phase 3: Topic Management (Week 3)
- [ ] Topic creation/deletion
- [ ] Partition management
- [ ] Configuration handling
- [ ] Topic metadata persistence

### Phase 4: Producer API (Week 4)
- [ ] Single message send
- [ ] Batch send
- [ ] Partitioning strategies
- [ ] Acknowledgment handling
- [ ] Retries and timeouts

### Phase 5: Consumer API (Week 5)
- [ ] Offset management
- [ ] Fetch with batching
- [ ] Commit handling
- [ ] Seek operations

### Phase 6: Consumer Groups (Week 6)
- [ ] Group membership
- [ ] Partition assignment
- [ ] Rebalancing protocol
- [ ] Heartbeat mechanism

### Phase 7: Delivery Guarantees (Week 7)
- [ ] At-least-once delivery
- [ ] Exactly-once semantics
- [ ] Idempotent producers
- [ ] Transaction support

### Phase 8: Performance Optimization (Week 8)
- [ ] Zero-copy reads
- [ ] Memory-mapped files
- [ ] Batch compression
- [ ] Connection pooling

---

## API Design

### Producer API

```rust
pub trait Producer {
    fn send(&self, topic: &str, message: Message) -> Result<RecordMetadata>;
    fn send_batch(&self, topic: &str, batch: MessageBatch) -> Result<Vec<RecordMetadata>>;
    fn flush(&self) -> Result<()>;
}

pub struct RecordMetadata {
    pub topic: String,
    pub partition: u32,
    pub offset: u64,
    pub timestamp: u64,
}
```

### Consumer API

```rust
pub trait Consumer {
    fn subscribe(&mut self, topics: &[&str]) -> Result<()>;
    fn poll(&mut self, timeout: Duration) -> Result<Vec<Message>>;
    fn commit(&mut self) -> Result<()>;
    fn commit_sync(&mut self) -> Result<()>;
    fn seek(&mut self, partition: u32, offset: u64) -> Result<()>;
    fn position(&self, partition: u32) -> Result<u64>;
}
```

### Admin API

```rust
pub trait Admin {
    fn create_topic(&self, topic: &str, config: TopicConfig) -> Result<()>;
    fn delete_topic(&self, topic: &str) -> Result<()>;
    fn list_topics(&self) -> Result<Vec<String>>;
    fn describe_topic(&self, topic: &str) -> Result<TopicDescription>;
    fn alter_topic(&self, topic: &str, config: TopicConfig) -> Result<()>;
}
```

---

## Testing Strategy

### Unit Tests
- Message serialization/deserialization
- Segment file operations
- Index operations
- Partitioner logic
- Consumer group state machine

### Integration Tests
- End-to-end message flow
- Multiple consumers
- Partition rebalancing
- Retention enforcement
- Recovery after crash

### Performance Tests
- Throughput benchmarks
- Latency measurements
- Memory usage
- Disk I/O patterns

---

## Dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
bytes = "1"
crc32fast = "1"
thiserror = "1"
serde = { version = "1", features = ["derive"] }
bincode = "1"
uuid = { version = "1", features = ["v4"] }
parking_lot = "0.12"
crossbeam = "0.8"
memmap2 = "0.9"
lz4 = "1"
flate2 = "1"
tracing = "0.1"

[dev-dependencies]
tempfile = "3"
criterion = "0.5"
proptest = "1"
tokio-test = "0.4"
```

---

## References

- [Apache Kafka Protocol](https://kafka.apache.org/protocol)
- [Redpanda Architecture](https://redpanda.com/blog/redpanda-architecture)
- [LogDevice Design](https://logdevice.io/docs/concepts.html)
- [RocketMQ Design](https://rocketmq.apache.org/docs/domainModel/01main/)
