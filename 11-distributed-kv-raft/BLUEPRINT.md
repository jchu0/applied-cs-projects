# Distributed Key-Value Store with Raft Consensus

> **Concepts covered:** §01 software-engineering — `rust/*`; §02 data-engineering — `04-streaming` (replication intuition)

## Executive Summary

A strongly consistent distributed key-value store built on the Raft consensus algorithm. This system provides linearizable read/write operations across a replicated cluster, supporting automatic leader election, log compaction via snapshots, and dynamic cluster membership changes.

---

## System Architecture

```
                              Client Requests
                                    |
                    +---------------+---------------+
                    |               |               |
                    v               v               v
            +-------------+ +-------------+ +-------------+
            |   Node 1    | |   Node 2    | |   Node 3    |
            |   LEADER    | |  FOLLOWER   | |  FOLLOWER   |
            +------+------+ +------+------+ +------+------+
                   |               |               |
        +----------+----------+    |    +----------+
        |          |          |    |    |          |
   +----v----+ +---v----+ +---v----v----v---+ +----v----+
   |   WAL   | |  FSM   | |   Raft Layer    | |Snapshot |
   | (Log)   | | (KV)   | | (Consensus)     | | Store   |
   +---------+ +--------+ +-----------------+ +---------+

Raft Cluster Communication:
+--------+     AppendEntries/RequestVote RPC     +--------+
| Node A | <----------------------------------> | Node B |
+--------+                                       +--------+
     ^                                               ^
     |                                               |
     +----------------+        +---------------------+
                      |        |
                      v        v
                   +-------------+
                   |   Node C    |
                   +-------------+
```

---

## Raft Protocol Deep Dive

### State Machine

```
                    times out,          receives votes from
                   starts election       majority of servers
                  +---------------+     +------------------+
                  |               |     |                  |
                  |               v     v                  |
              +-------+       +-----------+         +--------+
   starts --> |Follower| ---> | Candidate | ------> | Leader |
              +-------+       +-----------+         +--------+
                  ^                |                    |
                  |                |                    |
                  |   discovers    |    discovers       |
                  |   current      |    higher term     |
                  |   leader or    +--------------------+
                  |   higher term
                  +----------------------------------------+
```

### Term and Log Structure

```
Log Index:    1     2     3     4     5     6     7     8
           +-----+-----+-----+-----+-----+-----+-----+-----+
Term:      |  1  |  1  |  1  |  2  |  2  |  3  |  3  |  3  |
           +-----+-----+-----+-----+-----+-----+-----+-----+
Command:   | x=1 | y=2 | z=3 | x=5 | a=1 | b=2 | c=3 | d=4 |
           +-----+-----+-----+-----+-----+-----+-----+-----+
                              ^                       ^
                          commitIndex              lastApplied

Commit Flow:
  Leader receives entry -> Appends to local log -> Replicates to followers
  -> Majority acknowledges -> Leader commits -> Notifies followers
  -> Followers commit -> Apply to FSM
```

---

## Core Data Structures

### Raft Node State

```rust
pub struct RaftNode {
    // Persistent state (survives restarts)
    current_term: u64,
    voted_for: Option<NodeId>,
    log: Vec<LogEntry>,

    // Volatile state on all servers
    commit_index: u64,
    last_applied: u64,
    state: RaftState,

    // Volatile state on leaders
    next_index: HashMap<NodeId, u64>,   // For each peer: next log index to send
    match_index: HashMap<NodeId, u64>,  // For each peer: highest replicated index

    // Cluster configuration
    id: NodeId,
    peers: Vec<NodeId>,

    // Components
    wal: WriteAheadLog,
    fsm: KeyValueFSM,
    snapshot_store: SnapshotStore,
    transport: RpcTransport,

    // Timing
    election_timeout: Duration,
    heartbeat_interval: Duration,
    last_heartbeat: Instant,
}

pub struct LogEntry {
    term: u64,
    index: u64,
    command: Command,
    entry_type: EntryType,
}

pub enum EntryType {
    Command,
    Configuration,  // For cluster membership changes
    NoOp,          // Leader's first entry after election
}

pub enum Command {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    Get { key: Vec<u8> },  // For linearizable reads via log
}

pub enum RaftState {
    Follower { leader_id: Option<NodeId> },
    Candidate { votes_received: HashSet<NodeId> },
    Leader {
        lease_expiry: Option<Instant>,
        pending_reads: Vec<PendingRead>,
    },
}
```

### Key-Value FSM

```rust
pub struct KeyValueFSM {
    data: HashMap<Vec<u8>, Vec<u8>>,
    last_applied_index: u64,
    last_applied_term: u64,
}

impl StateMachine for KeyValueFSM {
    fn apply(&mut self, entry: &LogEntry) -> Result<ApplyResult> {
        match &entry.command {
            Command::Put { key, value } => {
                self.data.insert(key.clone(), value.clone());
                Ok(ApplyResult::Success)
            }
            Command::Delete { key } => {
                self.data.remove(key);
                Ok(ApplyResult::Success)
            }
            Command::Get { key } => {
                let value = self.data.get(key).cloned();
                Ok(ApplyResult::Value(value))
            }
        }
    }

    fn snapshot(&self) -> Snapshot {
        Snapshot {
            last_included_index: self.last_applied_index,
            last_included_term: self.last_applied_term,
            data: bincode::serialize(&self.data).unwrap(),
        }
    }

    fn restore(&mut self, snapshot: &Snapshot) {
        self.data = bincode::deserialize(&snapshot.data).unwrap();
        self.last_applied_index = snapshot.last_included_index;
        self.last_applied_term = snapshot.last_included_term;
    }
}
```

### Write-Ahead Log

```rust
pub struct WriteAheadLog {
    dir: PathBuf,
    current_segment: Segment,
    segments: Vec<SegmentMeta>,

    // In-memory index for fast lookups
    index: BTreeMap<u64, LogPosition>,  // log_index -> file position
}

pub struct Segment {
    file: File,
    base_index: u64,
    byte_offset: u64,
}

pub struct LogPosition {
    segment_id: u64,
    offset: u64,
    length: u32,
}

impl WriteAheadLog {
    pub fn append(&mut self, entries: &[LogEntry]) -> Result<()> {
        for entry in entries {
            let data = entry.serialize()?;

            // Write: [length: u32][crc: u32][data]
            let crc = crc32fast::hash(&data);
            self.current_segment.file.write_all(&(data.len() as u32).to_le_bytes())?;
            self.current_segment.file.write_all(&crc.to_le_bytes())?;
            self.current_segment.file.write_all(&data)?;

            // Update index
            self.index.insert(entry.index, LogPosition {
                segment_id: self.current_segment.base_index,
                offset: self.current_segment.byte_offset,
                length: data.len() as u32,
            });

            self.current_segment.byte_offset += 8 + data.len() as u64;
        }

        self.current_segment.file.sync_all()?;  // Durability!
        Ok(())
    }

    pub fn truncate_suffix(&mut self, from_index: u64) -> Result<()> {
        // Called when follower's log conflicts with leader
        if let Some(pos) = self.index.get(&from_index) {
            self.current_segment.file.set_len(pos.offset)?;
            self.index = self.index.split_off(&from_index).0;
        }
        Ok(())
    }
}
```

---

## RPC Protocol

### AppendEntries RPC

```rust
pub struct AppendEntriesRequest {
    term: u64,
    leader_id: NodeId,
    prev_log_index: u64,
    prev_log_term: u64,
    entries: Vec<LogEntry>,     // Empty for heartbeat
    leader_commit: u64,
}

pub struct AppendEntriesResponse {
    term: u64,
    success: bool,

    // Optimization: help leader find matching index faster
    conflict_index: Option<u64>,
    conflict_term: Option<u64>,
}

impl RaftNode {
    pub fn handle_append_entries(&mut self, req: AppendEntriesRequest) -> AppendEntriesResponse {
        // Rule 1: Reply false if term < currentTerm
        if req.term < self.current_term {
            return AppendEntriesResponse {
                term: self.current_term,
                success: false,
                conflict_index: None,
                conflict_term: None,
            };
        }

        // Update term if necessary
        if req.term > self.current_term {
            self.current_term = req.term;
            self.voted_for = None;
            self.transition_to_follower(Some(req.leader_id));
        }

        self.last_heartbeat = Instant::now();

        // Rule 2: Reply false if log doesn't contain entry at prevLogIndex
        // with matching prevLogTerm
        if req.prev_log_index > 0 {
            match self.log.get(req.prev_log_index as usize - 1) {
                None => {
                    return AppendEntriesResponse {
                        term: self.current_term,
                        success: false,
                        conflict_index: Some(self.log.len() as u64 + 1),
                        conflict_term: None,
                    };
                }
                Some(entry) if entry.term != req.prev_log_term => {
                    // Find first index of conflicting term
                    let conflict_term = entry.term;
                    let conflict_index = self.log.iter()
                        .position(|e| e.term == conflict_term)
                        .map(|i| i as u64 + 1)
                        .unwrap_or(1);

                    return AppendEntriesResponse {
                        term: self.current_term,
                        success: false,
                        conflict_index: Some(conflict_index),
                        conflict_term: Some(conflict_term),
                    };
                }
                _ => {}
            }
        }

        // Rule 3: Delete conflicting entries and append new ones
        for (i, entry) in req.entries.iter().enumerate() {
            let idx = req.prev_log_index as usize + i + 1;
            if idx <= self.log.len() {
                if self.log[idx - 1].term != entry.term {
                    self.log.truncate(idx - 1);
                    self.wal.truncate_suffix(idx as u64).unwrap();
                    self.log.push(entry.clone());
                }
            } else {
                self.log.push(entry.clone());
            }
        }

        // Persist to WAL
        if !req.entries.is_empty() {
            self.wal.append(&req.entries).unwrap();
        }

        // Rule 5: Update commit index
        if req.leader_commit > self.commit_index {
            self.commit_index = std::cmp::min(
                req.leader_commit,
                self.log.last().map(|e| e.index).unwrap_or(0)
            );
            self.apply_committed_entries();
        }

        AppendEntriesResponse {
            term: self.current_term,
            success: true,
            conflict_index: None,
            conflict_term: None,
        }
    }
}
```

### RequestVote RPC

```rust
pub struct RequestVoteRequest {
    term: u64,
    candidate_id: NodeId,
    last_log_index: u64,
    last_log_term: u64,
}

pub struct RequestVoteResponse {
    term: u64,
    vote_granted: bool,
}

impl RaftNode {
    pub fn handle_request_vote(&mut self, req: RequestVoteRequest) -> RequestVoteResponse {
        if req.term < self.current_term {
            return RequestVoteResponse {
                term: self.current_term,
                vote_granted: false,
            };
        }

        if req.term > self.current_term {
            self.current_term = req.term;
            self.voted_for = None;
            self.transition_to_follower(None);
        }

        // Check if we can vote for this candidate
        let can_vote = self.voted_for.is_none() || self.voted_for == Some(req.candidate_id);

        // Check if candidate's log is at least as up-to-date
        let last_log_term = self.log.last().map(|e| e.term).unwrap_or(0);
        let last_log_index = self.log.len() as u64;

        let log_ok = req.last_log_term > last_log_term ||
            (req.last_log_term == last_log_term && req.last_log_index >= last_log_index);

        if can_vote && log_ok {
            self.voted_for = Some(req.candidate_id);
            self.persist_state();
            self.last_heartbeat = Instant::now();  // Reset election timeout

            RequestVoteResponse {
                term: self.current_term,
                vote_granted: true,
            }
        } else {
            RequestVoteResponse {
                term: self.current_term,
                vote_granted: false,
            }
        }
    }
}
```

---

## Linearizable Reads

### Read Index Protocol

```rust
impl RaftNode {
    pub async fn linearizable_read(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Only leader can serve linearizable reads
        if !self.is_leader() {
            return Err(Error::NotLeader(self.get_leader_id()));
        }

        // Step 1: Save current commit index as read_index
        let read_index = self.commit_index;

        // Step 2: Exchange heartbeats with quorum to confirm leadership
        let quorum_size = (self.peers.len() + 1) / 2 + 1;
        let mut confirmations = 1;  // Self

        for peer in &self.peers {
            let resp = self.send_heartbeat(peer).await?;
            if resp.term == self.current_term && resp.success {
                confirmations += 1;
            }
            if confirmations >= quorum_size {
                break;
            }
        }

        if confirmations < quorum_size {
            return Err(Error::QuorumNotReached);
        }

        // Step 3: Wait until applied index >= read_index
        while self.last_applied < read_index {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }

        // Step 4: Read from FSM
        Ok(self.fsm.get(key))
    }
}
```

### Leader Lease Optimization

```rust
pub struct LeaderLease {
    start: Instant,
    duration: Duration,  // Typically election_timeout / 2
}

impl RaftNode {
    pub fn lease_read(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if !self.is_leader() {
            return Err(Error::NotLeader(self.get_leader_id()));
        }

        // Check if lease is still valid
        if let RaftState::Leader { lease_expiry, .. } = &self.state {
            if let Some(expiry) = lease_expiry {
                if Instant::now() < *expiry {
                    // Lease valid, can read locally without round-trip
                    return Ok(self.fsm.get(key));
                }
            }
        }

        // Lease expired, fall back to read-index protocol
        self.linearizable_read(key).await
    }

    pub fn extend_lease(&mut self) {
        // Called when AppendEntries succeeds to quorum
        let lease_duration = self.election_timeout / 2;
        if let RaftState::Leader { lease_expiry, .. } = &mut self.state {
            *lease_expiry = Some(Instant::now() + lease_duration);
        }
    }
}
```

---

## Snapshot and Log Compaction

```rust
pub struct Snapshot {
    last_included_index: u64,
    last_included_term: u64,
    data: Vec<u8>,

    // For cluster membership
    configuration: ClusterConfig,
}

pub struct InstallSnapshotRequest {
    term: u64,
    leader_id: NodeId,
    last_included_index: u64,
    last_included_term: u64,
    offset: u64,
    data: Vec<u8>,
    done: bool,
}

impl RaftNode {
    pub fn create_snapshot(&mut self) -> Result<()> {
        // Only snapshot if we have enough entries
        if self.last_applied < self.snapshot_threshold {
            return Ok(());
        }

        let snapshot = self.fsm.snapshot();
        self.snapshot_store.save(&snapshot)?;

        // Compact log: remove entries covered by snapshot
        let compact_index = snapshot.last_included_index as usize;
        if compact_index > 0 && compact_index <= self.log.len() {
            self.log = self.log[compact_index..].to_vec();
            self.wal.compact(snapshot.last_included_index)?;
        }

        Ok(())
    }

    pub fn handle_install_snapshot(&mut self, req: InstallSnapshotRequest) -> Result<u64> {
        if req.term < self.current_term {
            return Ok(self.current_term);
        }

        if req.term > self.current_term {
            self.current_term = req.term;
            self.transition_to_follower(Some(req.leader_id));
        }

        // Write snapshot chunk to temp file
        self.snapshot_store.write_chunk(req.offset, &req.data)?;

        if req.done {
            // Snapshot complete, install it
            let snapshot = self.snapshot_store.finalize()?;

            // If snapshot contains newer data than our log
            if req.last_included_index > self.log.last().map(|e| e.index).unwrap_or(0) {
                self.log.clear();
            } else {
                // Keep log entries after snapshot
                let idx = self.log.iter()
                    .position(|e| e.index > req.last_included_index)
                    .unwrap_or(self.log.len());
                self.log = self.log[idx..].to_vec();
            }

            // Restore FSM from snapshot
            self.fsm.restore(&snapshot);
            self.last_applied = req.last_included_index;
            self.commit_index = req.last_included_index;
        }

        Ok(self.current_term)
    }
}
```

---

## Client API

```rust
pub struct KVClient {
    cluster: Vec<NodeAddress>,
    leader_id: Option<NodeId>,
    timeout: Duration,
}

impl KVClient {
    pub async fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        self.execute_with_retry(Command::Put { key, value }).await
    }

    pub async fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // For linearizable read
        match self.execute_with_retry(Command::Get { key: key.to_vec() }).await {
            Ok(ApplyResult::Value(v)) => Ok(v),
            _ => Err(Error::Internal),
        }
    }

    pub async fn delete(&mut self, key: &[u8]) -> Result<()> {
        self.execute_with_retry(Command::Delete { key: key.to_vec() }).await
    }

    async fn execute_with_retry(&mut self, cmd: Command) -> Result<ApplyResult> {
        let mut retries = 3;

        loop {
            let target = self.leader_id
                .map(|id| self.cluster[id as usize].clone())
                .unwrap_or_else(|| self.cluster[0].clone());

            match self.send_request(&target, cmd.clone()).await {
                Ok(resp) => return Ok(resp),
                Err(Error::NotLeader(leader_hint)) => {
                    self.leader_id = leader_hint;
                    retries -= 1;
                }
                Err(e) => {
                    // Try next node
                    self.leader_id = None;
                    retries -= 1;
                }
            }

            if retries == 0 {
                return Err(Error::ClusterUnavailable);
            }
        }
    }
}
```

---

## Cluster Membership Changes

### Joint Consensus

```rust
pub struct ClusterConfig {
    old: HashSet<NodeId>,
    new: Option<HashSet<NodeId>>,  // None = not in joint config
}

impl RaftNode {
    pub fn add_node(&mut self, node_id: NodeId) -> Result<()> {
        if !self.is_leader() {
            return Err(Error::NotLeader(self.get_leader_id()));
        }

        // Create joint configuration
        let mut new_config = self.config.old.clone();
        new_config.insert(node_id);

        let joint_config = ClusterConfig {
            old: self.config.old.clone(),
            new: Some(new_config.clone()),
        };

        // Append C_old,new to log
        self.append_config_entry(joint_config)?;

        // Wait for C_old,new to commit (requires majority of both old and new)
        // Then append C_new
        // ...

        Ok(())
    }

    fn quorum_size(&self) -> usize {
        match &self.config.new {
            None => self.config.old.len() / 2 + 1,
            Some(new) => {
                // Joint consensus: need majority from both configs
                let old_quorum = self.config.old.len() / 2 + 1;
                let new_quorum = new.len() / 2 + 1;
                std::cmp::max(old_quorum, new_quorum)
            }
        }
    }
}
```

---

## Metrics and Monitoring

```rust
pub struct RaftMetrics {
    // Election metrics
    pub elections_started: Counter,
    pub elections_won: Counter,
    pub leader_changes: Counter,

    // Replication metrics
    pub entries_appended: Counter,
    pub entries_committed: Counter,
    pub entries_applied: Counter,

    // Latency metrics
    pub commit_latency: Histogram,
    pub apply_latency: Histogram,
    pub rpc_latency: Histogram,

    // Health metrics
    pub peers_connected: Gauge,
    pub log_size: Gauge,
    pub snapshot_size: Gauge,

    // Leader-specific
    pub replication_lag: HashMap<NodeId, Gauge>,  // Per-follower lag
}

pub struct HealthCheck {
    pub is_leader: bool,
    pub term: u64,
    pub commit_index: u64,
    pub applied_index: u64,
    pub cluster_size: usize,
    pub healthy_peers: usize,
    pub state: String,
}
```

---

## Implementation Phases

### Phase 1: Raft Core (Week 1-2)
- [ ] Implement RaftNode struct with state management
- [ ] Leader election with RequestVote RPC
- [ ] Log replication with AppendEntries RPC
- [ ] Basic in-memory log storage
- [ ] gRPC transport layer
- [ ] Single-node operation

### Phase 2: Persistence (Week 3)
- [ ] Write-ahead log with checksums
- [ ] Persistent state (term, votedFor)
- [ ] Recovery from WAL on startup
- [ ] Log segment rotation

### Phase 3: Key-Value FSM (Week 4)
- [ ] In-memory hash map FSM
- [ ] Apply committed entries
- [ ] Client request handling
- [ ] Request deduplication (client ID + sequence number)

### Phase 4: Snapshots (Week 5)
- [ ] FSM snapshot serialization
- [ ] Log compaction
- [ ] InstallSnapshot RPC
- [ ] Snapshot transfer to slow followers

### Phase 5: Linearizable Reads (Week 6)
- [ ] Read-index protocol
- [ ] Leader lease optimization
- [ ] Follower reads with lease verification

### Phase 6: Cluster Management (Week 7)
- [ ] Joint consensus for membership changes
- [ ] Add/remove node operations
- [ ] Leader transfer

### Phase 7: Production Hardening (Week 8)
- [ ] Pre-vote extension (prevent disruption from partitioned nodes)
- [ ] Check quorum (leader step-down if partitioned)
- [ ] Batching and pipelining
- [ ] Comprehensive metrics
- [ ] Chaos testing

---

## Testing Strategy

### Unit Tests
- State machine transitions
- Log truncation and appending
- Vote counting and quorum logic
- WAL corruption handling

### Integration Tests
- 3-node cluster operations
- Leader election timeout
- Follower catchup
- Snapshot installation

### Chaos Tests
```rust
#[test]
fn test_leader_failure() {
    let cluster = TestCluster::new(5);
    cluster.wait_for_leader();

    let old_leader = cluster.leader();
    cluster.partition_node(old_leader);

    // Cluster should elect new leader
    cluster.wait_for_leader_change(Duration::from_secs(5));

    // Writes should succeed
    cluster.put("key", "value").unwrap();

    // Heal partition
    cluster.heal_partition(old_leader);

    // Old leader should become follower
    cluster.wait_for_node_state(old_leader, RaftState::Follower, Duration::from_secs(5));
}

#[test]
fn test_network_partition() {
    let cluster = TestCluster::new(5);

    // Partition into [0,1] and [2,3,4]
    cluster.create_partition(&[0, 1], &[2, 3, 4]);

    // Minority partition should not elect leader
    // Majority partition should have working leader

    cluster.heal_all_partitions();

    // All nodes should converge
    cluster.wait_for_convergence(Duration::from_secs(10));
}
```

### Performance Tests
- Throughput: ops/sec at various cluster sizes
- Latency: p50/p99 commit latency
- Recovery time: time to elect new leader
- Snapshot: time to transfer large snapshots

---

## Stretch Goals

### Multi-Raft Shards
- Key-range based sharding
- Independent Raft groups per shard
- Cross-shard transactions (2PC)
- Shard splitting/merging

### Watch API
```rust
pub async fn watch(&mut self, key_prefix: &[u8]) -> WatchStream {
    // Server-side streaming of changes
    // Useful for service discovery, config changes
}
```

### Dynamic Cluster Resize
- Learner nodes (non-voting)
- Automatic rebalancing
- Graceful decommissioning

---

## Dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
tonic = "0.9"                    # gRPC
prost = "0.11"                   # Protocol Buffers
serde = { version = "1", features = ["derive"] }
bincode = "1"                    # Serialization
crc32fast = "1"                  # Checksums
bytes = "1"
parking_lot = "0.12"             # Fast mutexes
tracing = "0.1"
rand = "0.8"
```

---

## References

- [Raft Paper](https://raft.github.io/raft.pdf)
- [Raft Visualization](https://raft.github.io/)
- [etcd Raft Implementation](https://github.com/etcd-io/raft)
- [TiKV Raft](https://github.com/tikv/raft-rs)
- [Students' Guide to Raft](https://thesquareplanet.com/blog/students-guide-to-raft/)
