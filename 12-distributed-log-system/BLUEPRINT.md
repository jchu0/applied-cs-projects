# Distributed Log System (Kafka-lite)

## Executive Summary

A high-throughput distributed commit log system inspired by Apache Kafka. This system provides durable, ordered, append-only log storage with horizontal scaling through partitioning and fault tolerance through replication. Supports consumer groups for parallel consumption and exactly-once semantics.

> **Concepts covered:** [§02 Kafka streaming](../../02-data-engineering/04-streaming/kafka/kafka-streaming.md) (this project *implements* the broker side: WAL, partitions, replication, consumer groups) · [§02 Real-time analytics](../../02-data-engineering/04-streaming/real-time-analytics/real-time-analytics.md). Compare to [Project 51 (Tokio-based message queue)](../51-message-queue/); used by [Project 08 (streaming platform)](../08-streaming-platform/) and [Project 36 (streaming analytics)](../36-distributed-streaming-analytics/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

---

## System Architecture

```
                         Producers
                    /       |       \
                   v        v        v
            +------+  +------+  +------+
            |Broker|  |Broker|  |Broker|   (Broker Cluster)
            |  1   |  |  2   |  |  3   |
            +--+---+  +--+---+  +--+---+
               |         |         |
        +------+---------+---------+------+
        |      Metadata & Coordination     |
        |          (Controller)            |
        +---------------------------------+
               |         |         |
            +--+---+  +--+---+  +--+---+
            |Topic |  |Topic |  |Topic |   (Partitions)
            |Part 0|  |Part 1|  |Part 2|
            +------+  +------+  +------+
                 \       |       /
                  v      v      v
                    Consumers
              (Consumer Groups)

Partition Detail:
+-----------------------------------------------------------------+
|                        Partition 0                               |
+-----------------------------------------------------------------+
| Segment 0       | Segment 1       | Segment 2      | Active Seg  |
| (0-999)         | (1000-1999)     | (2000-2999)    | (3000-...)  |
+-----------------+-----------------+----------------+-------------+
| .log  .index    | .log  .index    | .log  .index   | .log .index |
+-----------------+-----------------+----------------+-------------+

Replication:
  Partition 0: Leader=Broker1, Followers=[Broker2, Broker3]
  Partition 1: Leader=Broker2, Followers=[Broker1, Broker3]
  Partition 2: Leader=Broker3, Followers=[Broker1, Broker2]
```

---

## Core Data Structures

### Message and Record Batch

```rust
pub struct RecordBatch {
    base_offset: u64,
    batch_length: u32,
    partition_leader_epoch: u32,
    magic: u8,                    // Version
    crc: u32,
    attributes: u16,              // Compression, timestamp type
    last_offset_delta: u32,
    base_timestamp: i64,
    max_timestamp: i64,
    producer_id: i64,             // For idempotent/transactional
    producer_epoch: i16,
    base_sequence: u32,
    records: Vec<Record>,
}

pub struct Record {
    length: usize,                // Varint
    attributes: u8,
    timestamp_delta: i64,         // Varint, relative to batch
    offset_delta: u32,            // Varint, relative to batch
    key: Option<Vec<u8>>,
    value: Option<Vec<u8>>,
    headers: Vec<Header>,
}

pub struct Header {
    key: String,
    value: Vec<u8>,
}
```

### Partition and Log Segment

```rust
pub struct Partition {
    topic: String,
    partition_id: u32,

    // Segments
    segments: BTreeMap<u64, LogSegment>,  // base_offset -> segment
    active_segment: LogSegment,

    // Replication state
    leader_epoch: u32,
    high_watermark: u64,          // Last replicated to all ISR
    log_end_offset: u64,          // Last appended offset

    // ISR management
    isr: HashSet<BrokerId>,
    leader: Option<BrokerId>,
}

pub struct LogSegment {
    base_offset: u64,

    // Files
    log_file: File,               // Actual data
    index_file: File,             // Sparse offset index
    time_index_file: File,        // Timestamp index

    // In-memory
    position: u64,                // Current write position
    max_timestamp: i64,

    // Config
    max_segment_bytes: u64,
    index_interval_bytes: u64,
}

// Sparse index entry (every N bytes)
pub struct IndexEntry {
    relative_offset: u32,         // Offset relative to base
    position: u32,                // File position
}

pub struct TimeIndexEntry {
    timestamp: i64,
    relative_offset: u32,
}
```

### Broker State

```rust
pub struct Broker {
    id: BrokerId,
    config: BrokerConfig,

    // Topic/partition management
    partitions: HashMap<TopicPartition, Partition>,

    // Replication
    replica_manager: ReplicaManager,
    replica_fetcher: ReplicaFetcher,

    // Consumer groups
    group_coordinator: GroupCoordinator,

    // Controller communication
    controller_channel: ControllerChannel,

    // Metrics
    metrics: BrokerMetrics,
}

pub struct ReplicaManager {
    // Leader partitions this broker owns
    leader_partitions: HashSet<TopicPartition>,

    // Follower state
    follower_state: HashMap<TopicPartition, FollowerState>,
}

pub struct FollowerState {
    leader_id: BrokerId,
    leader_epoch: u32,
    fetch_offset: u64,            // Next offset to fetch
}
```

---

## Log Storage Format

### Segment File Layout

```
.log file (append-only binary):
+------------------+------------------+------------------+
|   RecordBatch 1  |   RecordBatch 2  |   RecordBatch 3  |
+------------------+------------------+------------------+
|  offset 0-49     |  offset 50-99    |  offset 100-149  |
+------------------+------------------+------------------+

.index file (sparse index, memory-mapped):
+------------------------+------------------------+
| offset=0, position=0   | offset=50, position=4096|
+------------------------+------------------------+
| offset=100, pos=8192   | offset=150, pos=12288  |
+------------------------+------------------------+

.timeindex file:
+---------------------------+---------------------------+
| timestamp=1000, offset=0  | timestamp=1050, offset=50 |
+---------------------------+---------------------------+
```

### Reading by Offset

```rust
impl LogSegment {
    pub fn read(&self, offset: u64, max_bytes: u32) -> Result<Vec<RecordBatch>> {
        // Step 1: Binary search in sparse index
        let position = self.find_position(offset)?;

        // Step 2: Scan forward from position to find exact offset
        let mut reader = BufReader::new(&self.log_file);
        reader.seek(SeekFrom::Start(position))?;

        let mut batches = Vec::new();
        let mut bytes_read = 0;

        while bytes_read < max_bytes {
            let batch = RecordBatch::deserialize(&mut reader)?;

            if batch.base_offset >= offset {
                batches.push(batch);
                bytes_read += batch.batch_length;
            }

            if batch.base_offset + batch.last_offset_delta as u64 >= offset + 1000 {
                break;  // Enough records
            }
        }

        Ok(batches)
    }

    fn find_position(&self, offset: u64) -> Result<u64> {
        // Memory-map the index file
        let mmap = unsafe { Mmap::map(&self.index_file)? };
        let entries: &[IndexEntry] = bytemuck::cast_slice(&mmap);

        // Binary search for largest offset <= target
        let relative_offset = (offset - self.base_offset) as u32;
        let idx = entries.partition_point(|e| e.relative_offset <= relative_offset);

        if idx > 0 {
            Ok(entries[idx - 1].position as u64)
        } else {
            Ok(0)
        }
    }
}
```

---

## Replication Protocol

### Leader-Follower Replication

```rust
impl ReplicaFetcher {
    pub async fn fetch_loop(&mut self) {
        loop {
            for (tp, state) in &mut self.follower_state {
                let request = FetchRequest {
                    replica_id: self.broker_id,
                    max_wait_ms: 500,
                    min_bytes: 1,
                    partitions: vec![FetchPartition {
                        topic: tp.topic.clone(),
                        partition: tp.partition,
                        fetch_offset: state.fetch_offset,
                        max_bytes: 1_000_000,
                    }],
                };

                let response = self.send_fetch(state.leader_id, request).await?;

                for partition_data in response.partitions {
                    self.handle_fetch_response(tp, partition_data, state).await?;
                }
            }

            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn handle_fetch_response(
        &mut self,
        tp: &TopicPartition,
        data: FetchPartitionResponse,
        state: &mut FollowerState,
    ) -> Result<()> {
        // Append records to local log
        let partition = self.partitions.get_mut(tp).unwrap();

        for batch in data.record_batches {
            partition.append_as_follower(batch)?;
        }

        // Update high watermark
        partition.high_watermark = data.high_watermark;
        state.fetch_offset = partition.log_end_offset;

        Ok(())
    }
}
```

### ISR Management

```rust
impl Partition {
    pub fn check_isr(&mut self, replica_states: &HashMap<BrokerId, ReplicaState>) {
        let max_lag = Duration::from_secs(10);
        let now = Instant::now();

        let mut new_isr = HashSet::new();
        new_isr.insert(self.leader.unwrap());  // Leader always in ISR

        for (broker_id, state) in replica_states {
            if *broker_id == self.leader.unwrap() {
                continue;
            }

            let lag_time = now.duration_since(state.last_fetch_time);
            let lag_offset = self.log_end_offset - state.fetch_offset;

            // In ISR if caught up within time and offset limits
            if lag_time < max_lag && lag_offset < 1000 {
                new_isr.insert(*broker_id);
            }
        }

        if new_isr != self.isr {
            // ISR changed, notify controller
            self.isr = new_isr;
            self.notify_isr_change();
        }
    }

    pub fn update_high_watermark(&mut self, replica_offsets: &HashMap<BrokerId, u64>) {
        // High watermark = min offset among all ISR members
        let hw = self.isr.iter()
            .filter_map(|id| replica_offsets.get(id))
            .min()
            .copied()
            .unwrap_or(self.high_watermark);

        self.high_watermark = hw;
    }
}
```

---

## Producer API

### Produce Request Flow

```rust
pub struct Producer {
    client_id: String,
    config: ProducerConfig,
    metadata: ClusterMetadata,

    // Batching
    accumulators: HashMap<TopicPartition, RecordAccumulator>,

    // Idempotence
    producer_id: i64,
    producer_epoch: i16,
    sequence_numbers: HashMap<TopicPartition, u32>,
}

impl Producer {
    pub async fn send(&mut self, record: ProducerRecord) -> Result<RecordMetadata> {
        // Step 1: Get partition (by key hash or round-robin)
        let partition = self.partition_for(&record)?;
        let tp = TopicPartition::new(&record.topic, partition);

        // Step 2: Add to accumulator (batching)
        let accumulator = self.accumulators.entry(tp.clone())
            .or_insert_with(|| RecordAccumulator::new(&self.config));

        let future = accumulator.append(record)?;

        // Step 3: Check if batch is ready to send
        if accumulator.ready() {
            self.send_batch(&tp).await?;
        }

        // Step 4: Wait for acknowledgment
        future.await
    }

    async fn send_batch(&mut self, tp: &TopicPartition) -> Result<()> {
        let accumulator = self.accumulators.get_mut(tp).unwrap();
        let batch = accumulator.drain();

        // Add idempotence info
        let seq = self.sequence_numbers.entry(tp.clone()).or_insert(0);
        let batch = batch.with_producer_id(self.producer_id, self.producer_epoch, *seq);
        *seq += batch.record_count();

        // Find leader
        let leader = self.metadata.leader_for(tp)?;

        // Send produce request
        let request = ProduceRequest {
            acks: self.config.acks,
            timeout_ms: self.config.timeout_ms,
            topic_data: vec![TopicProduceData {
                topic: tp.topic.clone(),
                partition_data: vec![PartitionProduceData {
                    partition: tp.partition,
                    records: batch,
                }],
            }],
        };

        let response = self.send_to_broker(leader, request).await?;

        // Handle response
        self.handle_produce_response(tp, response)?;

        Ok(())
    }
}

pub enum Acks {
    None,       // Don't wait (acks=0)
    Leader,     // Wait for leader (acks=1)
    All,        // Wait for all ISR (acks=-1)
}
```

---

## Consumer API

### Consumer Group Protocol

```rust
pub struct Consumer {
    group_id: String,
    member_id: String,
    generation_id: u32,

    // Assigned partitions
    assignment: Vec<TopicPartition>,

    // Offset tracking
    positions: HashMap<TopicPartition, u64>,
    committed: HashMap<TopicPartition, u64>,

    coordinator: BrokerId,
}

impl Consumer {
    pub async fn subscribe(&mut self, topics: Vec<String>) -> Result<()> {
        // Step 1: Find group coordinator
        self.coordinator = self.find_coordinator().await?;

        // Step 2: Join group
        let join_response = self.join_group(&topics).await?;
        self.generation_id = join_response.generation_id;
        self.member_id = join_response.member_id;

        // Step 3: If leader, assign partitions
        if join_response.leader == self.member_id {
            let assignment = self.assign_partitions(
                join_response.members,
                join_response.subscriptions,
            )?;
            self.sync_group(assignment).await?;
        } else {
            self.sync_group(HashMap::new()).await?;
        }

        Ok(())
    }

    fn assign_partitions(
        &self,
        members: Vec<String>,
        subscriptions: HashMap<String, Vec<String>>,
    ) -> Result<HashMap<String, Vec<TopicPartition>>> {
        // Range assignor (simple strategy)
        let mut assignment: HashMap<String, Vec<TopicPartition>> = HashMap::new();

        for topic in self.get_all_topics(&subscriptions) {
            let partitions = self.metadata.partitions_for(&topic)?;
            let subscribed_members: Vec<_> = members.iter()
                .filter(|m| subscriptions[*m].contains(&topic))
                .collect();

            let partitions_per_member = partitions.len() / subscribed_members.len();
            let extra = partitions.len() % subscribed_members.len();

            let mut partition_idx = 0;
            for (i, member) in subscribed_members.iter().enumerate() {
                let count = partitions_per_member + if i < extra { 1 } else { 0 };
                let member_assignment = assignment.entry((*member).clone())
                    .or_insert_with(Vec::new);

                for _ in 0..count {
                    member_assignment.push(TopicPartition::new(&topic, partition_idx));
                    partition_idx += 1;
                }
            }
        }

        Ok(assignment)
    }

    pub async fn poll(&mut self, timeout: Duration) -> Result<Vec<ConsumerRecord>> {
        let mut records = Vec::new();

        for tp in &self.assignment {
            let offset = self.positions.get(tp).copied().unwrap_or(0);
            let leader = self.metadata.leader_for(tp)?;

            let response = self.fetch(leader, tp, offset).await?;

            for batch in response.record_batches {
                for record in batch.records {
                    let consumer_record = ConsumerRecord {
                        topic: tp.topic.clone(),
                        partition: tp.partition,
                        offset: batch.base_offset + record.offset_delta as u64,
                        timestamp: batch.base_timestamp + record.timestamp_delta,
                        key: record.key,
                        value: record.value,
                        headers: record.headers,
                    };
                    records.push(consumer_record);
                }
            }

            // Update position
            if let Some(last) = records.last() {
                self.positions.insert(tp.clone(), last.offset + 1);
            }
        }

        Ok(records)
    }

    pub async fn commit(&mut self) -> Result<()> {
        let offsets: Vec<_> = self.positions.iter()
            .map(|(tp, offset)| OffsetCommit {
                topic: tp.topic.clone(),
                partition: tp.partition,
                offset: *offset,
                metadata: String::new(),
            })
            .collect();

        self.commit_offsets(offsets).await?;
        self.committed = self.positions.clone();

        Ok(())
    }
}
```

### Group Coordinator

```rust
pub struct GroupCoordinator {
    groups: HashMap<String, ConsumerGroup>,
}

pub struct ConsumerGroup {
    group_id: String,
    state: GroupState,
    generation_id: u32,
    leader: Option<String>,
    members: HashMap<String, MemberMetadata>,

    // Committed offsets
    offsets: HashMap<TopicPartition, OffsetAndMetadata>,

    // Delayed operations
    pending_joins: Vec<DelayedJoin>,
}

pub enum GroupState {
    Empty,
    PreparingRebalance,
    CompletingRebalance,
    Stable,
    Dead,
}

impl GroupCoordinator {
    pub fn handle_join_group(&mut self, request: JoinGroupRequest) -> JoinGroupResponse {
        let group = self.groups.entry(request.group_id.clone())
            .or_insert_with(|| ConsumerGroup::new(&request.group_id));

        match group.state {
            GroupState::Empty | GroupState::Stable => {
                // Add member and trigger rebalance
                group.add_member(request.member_id.clone(), request.protocols.clone());
                group.state = GroupState::PreparingRebalance;

                // Wait for all members or timeout
                group.pending_joins.push(DelayedJoin {
                    member_id: request.member_id,
                    deadline: Instant::now() + Duration::from_secs(5),
                });

                // Check if all members have joined
                if group.all_members_joined() {
                    self.complete_join(group)
                } else {
                    JoinGroupResponse::pending()
                }
            }
            GroupState::PreparingRebalance => {
                // Join in progress, add to pending
                group.pending_joins.push(DelayedJoin {
                    member_id: request.member_id,
                    deadline: Instant::now() + Duration::from_secs(5),
                });
                JoinGroupResponse::pending()
            }
            _ => JoinGroupResponse::error(ErrorCode::IllegalGeneration),
        }
    }
}
```

---

## Log Retention and Compaction

### Time/Size-Based Retention

```rust
pub struct LogCleaner {
    config: CleanerConfig,
}

impl LogCleaner {
    pub fn clean(&self, partition: &mut Partition) -> Result<()> {
        let now = SystemTime::now();
        let retention_ms = self.config.retention_ms;
        let retention_bytes = self.config.retention_bytes;

        let mut segments_to_delete = Vec::new();
        let mut total_size = 0u64;

        // Calculate total size
        for segment in partition.segments.values() {
            total_size += segment.size();
        }

        // Find segments to delete
        for (base_offset, segment) in &partition.segments {
            // Time-based retention
            let segment_age = now.duration_since(segment.last_modified())?;
            if segment_age > Duration::from_millis(retention_ms) {
                segments_to_delete.push(*base_offset);
                continue;
            }

            // Size-based retention (delete oldest first)
            if total_size > retention_bytes {
                segments_to_delete.push(*base_offset);
                total_size -= segment.size();
            }
        }

        // Delete segments
        for base_offset in segments_to_delete {
            partition.delete_segment(base_offset)?;
        }

        Ok(())
    }
}
```

### Log Compaction

```rust
pub struct LogCompactor {
    config: CompactionConfig,
}

impl LogCompactor {
    pub fn compact(&self, partition: &mut Partition) -> Result<()> {
        // Step 1: Build offset map (key -> latest offset)
        let mut offset_map: HashMap<Vec<u8>, u64> = HashMap::new();

        for segment in partition.segments.values() {
            for batch in segment.read_all()? {
                for record in batch.records {
                    if let Some(key) = &record.key {
                        let offset = batch.base_offset + record.offset_delta as u64;
                        offset_map.insert(key.clone(), offset);
                    }
                }
            }
        }

        // Step 2: Rewrite segments keeping only latest values
        let mut new_segments = BTreeMap::new();
        let mut writer = SegmentWriter::new(partition.dir(), 0)?;

        for segment in partition.segments.values() {
            for batch in segment.read_all()? {
                let mut kept_records = Vec::new();

                for record in batch.records {
                    let offset = batch.base_offset + record.offset_delta as u64;

                    if let Some(key) = &record.key {
                        // Keep if this is the latest offset for this key
                        if offset_map.get(key) == Some(&offset) {
                            // Also check for tombstone (null value)
                            if record.value.is_some() {
                                kept_records.push(record);
                            }
                        }
                    } else {
                        // No key, keep all
                        kept_records.push(record);
                    }
                }

                if !kept_records.is_empty() {
                    writer.append_records(kept_records)?;
                }
            }
        }

        // Step 3: Swap old segments with new
        writer.close()?;
        partition.swap_segments(new_segments)?;

        Ok(())
    }
}
```

---

## Controller

```rust
pub struct Controller {
    broker_id: BrokerId,
    cluster_state: ClusterState,

    // Metadata
    topics: HashMap<String, TopicMetadata>,
    brokers: HashMap<BrokerId, BrokerInfo>,

    // Partition assignments
    partition_assignments: HashMap<TopicPartition, PartitionAssignment>,
}

pub struct PartitionAssignment {
    leader: BrokerId,
    replicas: Vec<BrokerId>,
    isr: Vec<BrokerId>,
    leader_epoch: u32,
}

impl Controller {
    pub fn handle_broker_failure(&mut self, failed_broker: BrokerId) {
        // Find all partitions where failed broker is leader
        for (tp, assignment) in &mut self.partition_assignments {
            if assignment.leader == failed_broker {
                // Elect new leader from ISR
                let new_leader = assignment.isr.iter()
                    .find(|b| **b != failed_broker)
                    .copied();

                if let Some(leader) = new_leader {
                    assignment.leader = leader;
                    assignment.leader_epoch += 1;
                    assignment.isr.retain(|b| *b != failed_broker);

                    // Notify new leader
                    self.send_leader_and_isr(tp, assignment);
                } else {
                    // No ISR available, partition offline
                    log::error!("Partition {:?} is offline", tp);
                }
            } else {
                // Remove from ISR
                assignment.isr.retain(|b| *b != failed_broker);
            }
        }
    }

    pub fn create_topic(&mut self, name: String, partitions: u32, replication: u32) -> Result<()> {
        let brokers: Vec<_> = self.brokers.keys().copied().collect();

        if replication as usize > brokers.len() {
            return Err(Error::NotEnoughBrokers);
        }

        let mut assignments = Vec::new();

        for partition_id in 0..partitions {
            // Round-robin leader assignment
            let leader_idx = partition_id as usize % brokers.len();
            let leader = brokers[leader_idx];

            // Select replicas (rack-aware in production)
            let mut replicas = vec![leader];
            for i in 1..replication as usize {
                let replica_idx = (leader_idx + i) % brokers.len();
                replicas.push(brokers[replica_idx]);
            }

            assignments.push(PartitionAssignment {
                leader,
                replicas: replicas.clone(),
                isr: replicas,
                leader_epoch: 0,
            });
        }

        // Store and propagate
        self.topics.insert(name.clone(), TopicMetadata {
            name: name.clone(),
            partitions,
            replication_factor: replication,
        });

        for (i, assignment) in assignments.into_iter().enumerate() {
            let tp = TopicPartition::new(&name, i as u32);
            self.partition_assignments.insert(tp, assignment);
        }

        // Notify all brokers
        self.broadcast_metadata_update();

        Ok(())
    }
}
```

---

## Metrics and Monitoring

```rust
pub struct BrokerMetrics {
    // Throughput
    pub bytes_in_per_sec: Counter,
    pub bytes_out_per_sec: Counter,
    pub messages_in_per_sec: Counter,
    pub requests_per_sec: HashMap<ApiKey, Counter>,

    // Latency
    pub produce_latency: Histogram,
    pub fetch_latency: Histogram,
    pub request_queue_time: Histogram,

    // Replication
    pub under_replicated_partitions: Gauge,
    pub isr_shrinks_per_sec: Counter,
    pub isr_expands_per_sec: Counter,

    // Consumer groups
    pub active_group_count: Gauge,
    pub consumer_lag: HashMap<TopicPartition, Gauge>,
}

pub struct ConsumerMetrics {
    pub records_consumed: Counter,
    pub bytes_consumed: Counter,
    pub fetch_latency: Histogram,
    pub commit_latency: Histogram,
    pub rebalance_count: Counter,
    pub lag: HashMap<TopicPartition, Gauge>,
}
```

---

## Implementation Phases

### Phase 1: Single Broker Log (Week 1-2)
- [ ] Log segment file format
- [ ] Append and read operations
- [ ] Sparse index for offset lookup
- [ ] Segment rotation

### Phase 2: Produce/Fetch API (Week 3)
- [ ] Produce request handling
- [ ] Fetch request handling
- [ ] Batch accumulator for producers
- [ ] gRPC or custom protocol

### Phase 3: Replication (Week 4-5)
- [ ] Leader-follower replication
- [ ] ISR management
- [ ] High watermark tracking
- [ ] Leader election

### Phase 4: Controller (Week 6)
- [ ] Cluster metadata management
- [ ] Broker liveness
- [ ] Topic creation/deletion
- [ ] Partition assignment

### Phase 5: Consumer Groups (Week 7)
- [ ] Group coordinator
- [ ] Join/Sync protocol
- [ ] Partition assignment strategies
- [ ] Offset commit storage

### Phase 6: Retention & Compaction (Week 8)
- [ ] Time-based retention
- [ ] Size-based retention
- [ ] Log compaction
- [ ] Tombstone handling

### Phase 7: Production Hardening (Week 9-10)
- [ ] Idempotent producers
- [ ] Transactions (stretch)
- [ ] Quotas
- [ ] Comprehensive metrics

---

## Testing Strategy

### Unit Tests
- Segment read/write
- Index lookup
- Batch serialization
- Offset calculation

### Integration Tests
```rust
#[tokio::test]
async fn test_produce_consume() {
    let cluster = TestCluster::new(3).await;
    cluster.create_topic("test", 3, 2).await.unwrap();

    let producer = cluster.producer();
    let consumer = cluster.consumer("group1");

    // Produce messages
    for i in 0..100 {
        producer.send("test", format!("key-{}", i), format!("value-{}", i)).await.unwrap();
    }

    // Consume and verify
    consumer.subscribe(vec!["test".to_string()]).await.unwrap();
    let records = consumer.poll(Duration::from_secs(10)).await.unwrap();

    assert_eq!(records.len(), 100);
}

#[tokio::test]
async fn test_leader_failover() {
    let cluster = TestCluster::new(3).await;
    cluster.create_topic("test", 1, 3).await.unwrap();

    // Produce some messages
    let producer = cluster.producer();
    producer.send("test", "key", "value").await.unwrap();

    // Kill leader
    let leader = cluster.leader_for("test", 0);
    cluster.kill_broker(leader).await;

    // Should still be able to produce after failover
    producer.send("test", "key2", "value2").await.unwrap();
}
```

### Performance Benchmarks
- Producer throughput: target 100k+ msg/s
- Consumer throughput: target 200k+ msg/s
- p99 latency under load
- Replication lag under load

---

## Stretch Goals

### Exactly-Once Semantics
- Idempotent producers with sequence numbers
- Transactional producers
- Read-committed isolation

### Tiered Storage
- Move old segments to object storage (S3)
- Fetch from remote on read
- Cost optimization

### Schema Registry
- Schema validation
- Schema evolution
- Avro/Protobuf support

---

## Dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
bytes = "1"
memmap2 = "0.5"              # Memory-mapped index files
crc32fast = "1"
lz4 = "1"                     # Compression
zstd = "0.12"
dashmap = "5"                 # Concurrent maps
crossbeam = "0.8"
```

---

## References

- [Kafka Protocol Guide](https://kafka.apache.org/protocol)
- [Kafka Design](https://kafka.apache.org/documentation/#design)
- [Kafka Internals](https://developer.confluent.io/learn-kafka/architecture/get-started/)
- [Redpanda](https://github.com/redpanda-data/redpanda) - Modern Kafka alternative
