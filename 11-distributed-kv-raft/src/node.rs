//! Core Raft node implementation.

use crate::config::{ClusterConfig, RaftConfig};
use crate::error::{Error, Result};
use crate::rpc::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    RequestVoteRequest, RequestVoteResponse,
};
use crate::storage::{KeyValueFSM, Snapshot, SnapshotStore, Storage, WriteAheadLog};
use crate::{LogIndex, NodeId, Term};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Command types for the state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Command {
    /// Put a key-value pair.
    Put { key: Vec<u8>, value: Vec<u8> },
    /// Delete a key.
    Delete { key: Vec<u8> },
    /// Get a value (for linearizable reads via log).
    Get { key: Vec<u8> },
    /// No-op command (used after leader election).
    NoOp,
}

/// Type of log entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EntryType {
    /// Normal command entry.
    Command,
    /// Configuration change entry.
    Configuration,
    /// No-op entry (leader's first entry after election).
    NoOp,
}

/// A single entry in the Raft log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Term when entry was received by leader.
    pub term: Term,
    /// Position in log.
    pub index: LogIndex,
    /// The command to apply.
    pub command: Command,
    /// Type of this entry.
    pub entry_type: EntryType,
}

/// State of a Raft node.
#[derive(Debug, Clone)]
pub enum RaftState {
    /// Follower state.
    Follower {
        /// Current known leader, if any.
        leader_id: Option<NodeId>,
    },
    /// Candidate state (during election).
    Candidate {
        /// Nodes that voted for us.
        votes_received: HashSet<NodeId>,
    },
    /// Leader state.
    Leader {
        /// Lease expiry time for fast reads.
        lease_expiry: Option<Instant>,
        /// Next index to send to each peer.
        next_index: HashMap<NodeId, LogIndex>,
        /// Highest index known to be replicated on each peer.
        match_index: HashMap<NodeId, LogIndex>,
    },
}

/// Result of applying a command to the state machine.
#[derive(Debug, Clone)]
pub enum ApplyResult {
    /// Operation succeeded.
    Success,
    /// Operation returned a value.
    Value(Option<Vec<u8>>),
    /// Operation failed.
    Failed(String),
}

/// Pending read request waiting for commit.
#[derive(Debug)]
pub struct PendingRead {
    /// Read index to wait for.
    pub read_index: LogIndex,
    /// Key to read.
    pub key: Vec<u8>,
    /// Channel to send result.
    pub response_tx: tokio::sync::oneshot::Sender<Result<Option<Vec<u8>>>>,
}

/// Configuration change entry for joint consensus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigurationChange {
    /// Old configuration nodes.
    pub old_nodes: HashSet<NodeId>,
    /// New configuration nodes.
    pub new_nodes: HashSet<NodeId>,
    /// Whether this is the final (new-only) configuration.
    pub is_final: bool,
}

/// Pending client write request.
#[derive(Debug)]
pub struct PendingWrite {
    /// Log index for this write.
    pub index: LogIndex,
    /// Channel to send result.
    pub response_tx: tokio::sync::oneshot::Sender<Result<()>>,
}

/// Pre-vote state for preventing disruption from partitioned nodes.
#[derive(Debug, Clone)]
pub struct PreVoteState {
    /// Term we're pre-voting for.
    pub term: Term,
    /// Votes received.
    pub votes_received: HashSet<NodeId>,
}

/// Core Raft node.
pub struct RaftNode {
    // Configuration
    pub config: RaftConfig,

    // Persistent state (must survive restarts)
    pub current_term: Term,
    pub voted_for: Option<NodeId>,
    pub log: Vec<LogEntry>,

    // Volatile state on all servers
    pub commit_index: LogIndex,
    pub last_applied: LogIndex,
    pub state: RaftState,

    // Cluster configuration
    pub cluster_config: ClusterConfig,

    // Components
    pub fsm: KeyValueFSM,
    pub storage: Option<Box<dyn Storage + Send + Sync>>,
    pub snapshot_store: Option<SnapshotStore>,

    // Timing
    pub last_heartbeat: Instant,

    // Pending operations
    pub pending_reads: Vec<PendingRead>,
    pub pending_writes: Vec<PendingWrite>,

    // Pre-vote state (for preventing disruption)
    pub pre_vote_state: Option<PreVoteState>,
    pub enable_pre_vote: bool,

    // Check quorum (leader steps down if can't reach majority)
    pub enable_check_quorum: bool,
    pub last_quorum_check: Instant,
    pub peer_last_contact: HashMap<NodeId, Instant>,

    // Snapshot metadata
    pub snapshot_last_index: LogIndex,
    pub snapshot_last_term: Term,

    // Running state
    pub is_running: bool,
    pub is_partitioned: bool,
}

impl RaftNode {
    /// Create a new Raft node.
    pub fn new(config: RaftConfig) -> Self {
        let cluster_nodes = config.cluster_nodes();
        let cluster_config = ClusterConfig::new(cluster_nodes);
        let peer_last_contact: HashMap<NodeId, Instant> = config
            .peers
            .iter()
            .map(|p| (p.id, Instant::now()))
            .collect();

        Self {
            config,
            current_term: 0,
            voted_for: None,
            log: Vec::new(),
            commit_index: 0,
            last_applied: 0,
            state: RaftState::Follower { leader_id: None },
            cluster_config,
            fsm: KeyValueFSM::new(),
            storage: None,
            snapshot_store: None,
            last_heartbeat: Instant::now(),
            pending_reads: Vec::new(),
            pending_writes: Vec::new(),
            pre_vote_state: None,
            enable_pre_vote: true,
            enable_check_quorum: true,
            last_quorum_check: Instant::now(),
            peer_last_contact,
            snapshot_last_index: 0,
            snapshot_last_term: 0,
            is_running: false,
            is_partitioned: false,
        }
    }

    /// Create a new Raft node with storage.
    pub fn with_storage(config: RaftConfig, storage: Box<dyn Storage + Send + Sync>) -> Self {
        let mut node = Self::new(config);
        node.storage = Some(storage);
        node
    }

    /// Set the snapshot store.
    pub fn set_snapshot_store(&mut self, store: SnapshotStore) {
        self.snapshot_store = Some(store);
    }

    /// Enable or disable pre-vote.
    pub fn set_pre_vote(&mut self, enabled: bool) {
        self.enable_pre_vote = enabled;
    }

    /// Enable or disable check quorum.
    pub fn set_check_quorum(&mut self, enabled: bool) {
        self.enable_check_quorum = enabled;
    }

    /// Set partitioned state (for testing).
    pub fn set_partitioned(&mut self, partitioned: bool) {
        self.is_partitioned = partitioned;
    }

    /// Check if node is partitioned.
    pub fn is_partitioned(&self) -> bool {
        self.is_partitioned
    }

    /// Start the node.
    pub fn start(&mut self) -> Result<()> {
        self.is_running = true;
        self.last_heartbeat = Instant::now();
        info!("Node {} started", self.config.id);
        Ok(())
    }

    /// Stop the node.
    pub fn stop(&mut self) -> Result<()> {
        self.is_running = false;
        self.transition_to_follower(None);
        info!("Node {} stopped", self.config.id);
        Ok(())
    }

    /// Check if node is running.
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Get node ID.
    pub fn id(&self) -> NodeId {
        self.config.id
    }

    /// Get current term.
    pub fn term(&self) -> Term {
        self.current_term
    }

    /// Check if this node is the leader.
    pub fn is_leader(&self) -> bool {
        matches!(self.state, RaftState::Leader { .. })
    }

    /// Get the current leader ID.
    pub fn leader_id(&self) -> Option<NodeId> {
        match &self.state {
            RaftState::Leader { .. } => Some(self.config.id),
            RaftState::Follower { leader_id } => *leader_id,
            RaftState::Candidate { .. } => None,
        }
    }

    /// Get last log index.
    pub fn last_log_index(&self) -> LogIndex {
        self.log.last().map(|e| e.index).unwrap_or(0)
    }

    /// Get last log term.
    pub fn last_log_term(&self) -> Term {
        self.log.last().map(|e| e.term).unwrap_or(0)
    }

    /// Transition to follower state.
    pub fn transition_to_follower(&mut self, leader_id: Option<NodeId>) {
        info!(
            "Node {} transitioning to follower (term {})",
            self.config.id, self.current_term
        );
        self.state = RaftState::Follower { leader_id };
    }

    /// Transition to candidate state and start election.
    pub fn transition_to_candidate(&mut self) {
        self.current_term += 1;
        self.voted_for = Some(self.config.id);

        let mut votes = HashSet::new();
        votes.insert(self.config.id); // Vote for self

        info!(
            "Node {} transitioning to candidate (term {})",
            self.config.id, self.current_term
        );

        self.state = RaftState::Candidate {
            votes_received: votes,
        };
        self.last_heartbeat = Instant::now();
    }

    /// Transition to leader state.
    pub fn transition_to_leader(&mut self) {
        info!(
            "Node {} transitioning to leader (term {})",
            self.config.id, self.current_term
        );

        let last_index = self.last_log_index();
        let mut next_index = HashMap::new();
        let mut match_index = HashMap::new();

        for peer in &self.config.peers {
            next_index.insert(peer.id, last_index + 1);
            match_index.insert(peer.id, 0);
        }

        self.state = RaftState::Leader {
            lease_expiry: None,
            next_index,
            match_index,
        };

        // Append no-op entry to commit entries from previous terms
        let noop = LogEntry {
            term: self.current_term,
            index: self.last_log_index() + 1,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        };
        self.log.push(noop);
    }

    /// Check if election timeout has elapsed.
    pub fn election_timeout_elapsed(&self) -> bool {
        let timeout = self.config.random_election_timeout();
        self.last_heartbeat.elapsed() > timeout
    }

    /// Reset election timeout.
    pub fn reset_election_timeout(&mut self) {
        self.last_heartbeat = Instant::now();
    }

    /// Handle RequestVote RPC.
    pub fn handle_request_vote(&mut self, req: RequestVoteRequest) -> RequestVoteResponse {
        // Rule 1: Reply false if term < currentTerm
        if req.term < self.current_term {
            return RequestVoteResponse {
                term: self.current_term,
                vote_granted: false,
            };
        }

        // Update term if necessary
        if req.term > self.current_term {
            self.current_term = req.term;
            self.voted_for = None;
            self.transition_to_follower(None);
        }

        // Check if we can vote for this candidate
        let can_vote = self.voted_for.is_none() || self.voted_for == Some(req.candidate_id);

        // Check if candidate's log is at least as up-to-date
        let last_log_term = self.last_log_term();
        let last_log_index = self.last_log_index();

        let log_ok = req.last_log_term > last_log_term
            || (req.last_log_term == last_log_term && req.last_log_index >= last_log_index);

        if can_vote && log_ok {
            self.voted_for = Some(req.candidate_id);
            self.reset_election_timeout();

            debug!(
                "Node {} voted for {} in term {}",
                self.config.id, req.candidate_id, self.current_term
            );

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

    /// Handle RequestVote response.
    pub fn handle_request_vote_response(&mut self, from: NodeId, resp: RequestVoteResponse) {
        if resp.term > self.current_term {
            self.current_term = resp.term;
            self.voted_for = None;
            self.transition_to_follower(None);
            return;
        }

        if let RaftState::Candidate { votes_received } = &mut self.state {
            if resp.vote_granted {
                votes_received.insert(from);

                // Check if we have quorum
                let quorum = self.config.quorum_size();
                if votes_received.len() >= quorum {
                    self.transition_to_leader();
                }
            }
        }
    }

    /// Handle AppendEntries RPC.
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
        }

        // Always transition to follower when receiving AppendEntries
        self.transition_to_follower(Some(req.leader_id));
        self.reset_election_timeout();

        // Rule 2: Check if log contains entry at prevLogIndex with matching term
        if req.prev_log_index > 0 {
            match self.get_log_entry(req.prev_log_index) {
                None => {
                    return AppendEntriesResponse {
                        term: self.current_term,
                        success: false,
                        conflict_index: Some(self.last_log_index() + 1),
                        conflict_term: None,
                    };
                }
                Some(entry) if entry.term != req.prev_log_term => {
                    let conflict_term = entry.term;
                    let conflict_index = self
                        .log
                        .iter()
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
        for entry in &req.entries {
            let idx = entry.index as usize;
            if idx <= self.log.len() {
                if let Some(existing) = self.log.get(idx - 1) {
                    if existing.term != entry.term {
                        self.log.truncate(idx - 1);
                        self.log.push(entry.clone());
                    }
                }
            } else {
                self.log.push(entry.clone());
            }
        }

        // Rule 5: Update commit index
        if req.leader_commit > self.commit_index {
            self.commit_index = std::cmp::min(req.leader_commit, self.last_log_index());
            self.apply_committed_entries();
        }

        AppendEntriesResponse {
            term: self.current_term,
            success: true,
            conflict_index: None,
            conflict_term: None,
        }
    }

    /// Handle AppendEntries response from follower.
    pub fn handle_append_entries_response(
        &mut self,
        from: NodeId,
        resp: AppendEntriesResponse,
        entries_sent: usize,
    ) {
        if resp.term > self.current_term {
            self.current_term = resp.term;
            self.voted_for = None;
            self.transition_to_follower(None);
            return;
        }

        // Compute values before borrowing state mutably
        let last_idx = self.last_log_index();
        let lease_duration = self.config.election_timeout_min / 2;
        let enable_lease = self.config.enable_leader_lease;

        let should_advance_commit = if let RaftState::Leader {
            next_index,
            match_index,
            lease_expiry,
        } = &mut self.state
        {
            if resp.success {
                // Update next_index and match_index for follower
                if let Some(next) = next_index.get_mut(&from) {
                    *next += entries_sent as u64;
                }
                if let Some(matched) = match_index.get_mut(&from) {
                    *matched = last_idx;
                }

                // Extend lease
                if enable_lease {
                    *lease_expiry = Some(Instant::now() + lease_duration);
                }
                true
            } else {
                // Decrement next_index and retry
                if let Some(next) = next_index.get_mut(&from) {
                    if let Some(conflict_index) = resp.conflict_index {
                        *next = conflict_index;
                    } else if *next > 1 {
                        *next -= 1;
                    }
                }
                false
            }
        } else {
            false
        };

        // Advance commit index after releasing mutable borrow
        if should_advance_commit {
            self.try_advance_commit_index();
        }
    }

    /// Try to advance commit index based on replication.
    fn try_advance_commit_index(&mut self) {
        if let RaftState::Leader { match_index, .. } = &self.state {
            // Find median match index (quorum)
            let mut indices: Vec<LogIndex> = match_index.values().copied().collect();
            indices.push(self.last_log_index()); // Include self
            indices.sort_unstable();

            let quorum_idx = indices.len() - self.config.quorum_size();
            let new_commit = indices[quorum_idx];

            // Only commit entries from current term
            if new_commit > self.commit_index {
                if let Some(entry) = self.get_log_entry(new_commit) {
                    if entry.term == self.current_term {
                        self.commit_index = new_commit;
                        self.apply_committed_entries();
                    }
                }
            }
        }
    }

    /// Apply committed entries to state machine.
    fn apply_committed_entries(&mut self) {
        while self.last_applied < self.commit_index {
            self.last_applied += 1;
            // Clone entry to release borrow before applying
            if let Some(entry) = self.get_log_entry(self.last_applied).cloned() {
                let result = self.fsm.apply(&entry);
                debug!(
                    "Node {} applied entry {} (term {}): {:?}",
                    self.config.id, entry.index, entry.term, result
                );
            }
        }
    }

    /// Get log entry at given index.
    fn get_log_entry(&self, index: LogIndex) -> Option<&LogEntry> {
        if index == 0 || index as usize > self.log.len() {
            None
        } else {
            Some(&self.log[index as usize - 1])
        }
    }

    /// Propose a command (leader only).
    pub fn propose(&mut self, command: Command) -> Result<LogIndex> {
        if !self.is_leader() {
            return Err(Error::NotLeader(self.leader_id()));
        }

        let entry = LogEntry {
            term: self.current_term,
            index: self.last_log_index() + 1,
            command,
            entry_type: EntryType::Command,
        };

        let index = entry.index;
        self.log.push(entry);

        Ok(index)
    }

    /// Get entries to send to a follower.
    pub fn get_entries_for_peer(&self, peer_id: NodeId) -> (LogIndex, Term, Vec<LogEntry>) {
        if let RaftState::Leader { next_index, .. } = &self.state {
            let next = next_index.get(&peer_id).copied().unwrap_or(1);
            let prev_index = next - 1;
            let prev_term = if prev_index == 0 {
                0
            } else {
                self.get_log_entry(prev_index)
                    .map(|e| e.term)
                    .unwrap_or(0)
            };

            let entries: Vec<LogEntry> = self
                .log
                .iter()
                .filter(|e| e.index >= next)
                .take(self.config.max_entries_per_rpc)
                .cloned()
                .collect();

            (prev_index, prev_term, entries)
        } else {
            (0, 0, Vec::new())
        }
    }

    /// Build AppendEntries request for a peer.
    pub fn build_append_entries_request(&self, peer_id: NodeId) -> AppendEntriesRequest {
        let (prev_log_index, prev_log_term, entries) = self.get_entries_for_peer(peer_id);

        AppendEntriesRequest {
            term: self.current_term,
            leader_id: self.config.id,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit: self.commit_index,
        }
    }

    /// Build RequestVote request.
    pub fn build_request_vote_request(&self) -> RequestVoteRequest {
        RequestVoteRequest {
            term: self.current_term,
            candidate_id: self.config.id,
            last_log_index: self.last_log_index(),
            last_log_term: self.last_log_term(),
        }
    }

    /// Get value from FSM (for local reads).
    pub fn get(&self, key: &[u8]) -> Option<Vec<u8>> {
        self.fsm.get(key)
    }

    /// Check if leader lease is valid (for fast reads).
    pub fn lease_valid(&self) -> bool {
        if let RaftState::Leader { lease_expiry, .. } = &self.state {
            if let Some(expiry) = lease_expiry {
                return Instant::now() < *expiry;
            }
        }
        false
    }

    /// Get cluster configuration.
    pub fn cluster_config(&self) -> &ClusterConfig {
        &self.cluster_config
    }

    /// Get commit index.
    pub fn commit_index(&self) -> LogIndex {
        self.commit_index
    }

    /// Get last applied index.
    pub fn last_applied(&self) -> LogIndex {
        self.last_applied
    }

    /// Get peer IDs.
    pub fn peer_ids(&self) -> Vec<NodeId> {
        self.config.peers.iter().map(|p| p.id).collect()
    }

    /// Get the current state.
    pub fn state(&self) -> &RaftState {
        &self.state
    }

    /// Get voted for.
    pub fn voted_for(&self) -> Option<NodeId> {
        self.voted_for
    }

    /// Get log size.
    pub fn log_size(&self) -> usize {
        self.log.len()
    }

    /// Get the config.
    pub fn config(&self) -> &RaftConfig {
        &self.config
    }

    // ========================================================================
    // InstallSnapshot RPC handling
    // ========================================================================

    /// Handle InstallSnapshot RPC.
    pub fn handle_install_snapshot(
        &mut self,
        req: InstallSnapshotRequest,
    ) -> InstallSnapshotResponse {
        // Reply false if term < currentTerm
        if req.term < self.current_term {
            return InstallSnapshotResponse {
                term: self.current_term,
            };
        }

        // Update term if necessary
        if req.term > self.current_term {
            self.current_term = req.term;
            self.voted_for = None;
            self.transition_to_follower(Some(req.leader_id));
        }

        self.reset_election_timeout();

        // Write snapshot chunk
        if let Some(ref mut store) = self.snapshot_store {
            if let Err(e) = store.write_chunk(req.offset, &req.data) {
                warn!("Failed to write snapshot chunk: {}", e);
            }
        }

        // If done, finalize and install snapshot
        if req.done {
            self.install_snapshot_data(req.last_included_index, req.last_included_term);
        }

        InstallSnapshotResponse {
            term: self.current_term,
        }
    }

    /// Install snapshot data after receiving all chunks.
    fn install_snapshot_data(&mut self, last_included_index: LogIndex, last_included_term: Term) {
        // Finalize the snapshot
        if let Some(ref mut store) = self.snapshot_store {
            match store.finalize() {
                Ok(snapshot) => {
                    // Restore FSM from snapshot
                    if let Err(e) = self.fsm.restore(&snapshot) {
                        warn!("Failed to restore FSM from snapshot: {}", e);
                        return;
                    }

                    // Discard entire log if snapshot contains newer data
                    if last_included_index > self.last_log_index() {
                        self.log.clear();
                    } else {
                        // Keep log entries after snapshot
                        self.log
                            .retain(|e| e.index > last_included_index);
                    }

                    // Update state
                    self.snapshot_last_index = last_included_index;
                    self.snapshot_last_term = last_included_term;
                    self.commit_index = std::cmp::max(self.commit_index, last_included_index);
                    self.last_applied = last_included_index;

                    info!(
                        "Node {} installed snapshot up to index {}",
                        self.config.id, last_included_index
                    );
                }
                Err(e) => {
                    warn!("Failed to finalize snapshot: {}", e);
                }
            }
        }
    }

    /// Build InstallSnapshot request for a peer.
    pub fn build_install_snapshot_request(
        &self,
        offset: u64,
        chunk_size: usize,
    ) -> Option<InstallSnapshotRequest> {
        let snapshot = self.fsm.snapshot();
        let data_len = snapshot.data.len() as u64;

        if offset >= data_len {
            return None;
        }

        let end = std::cmp::min(offset + chunk_size as u64, data_len);
        let chunk = snapshot.data[offset as usize..end as usize].to_vec();
        let done = end >= data_len;

        Some(InstallSnapshotRequest {
            term: self.current_term,
            leader_id: self.config.id,
            last_included_index: snapshot.last_included_index,
            last_included_term: snapshot.last_included_term,
            offset,
            data: chunk,
            done,
        })
    }

    // ========================================================================
    // Snapshot and log compaction
    // ========================================================================

    /// Create a snapshot of the current state.
    pub fn create_snapshot(&mut self) -> Result<()> {
        let snapshot = self.fsm.snapshot();

        if let Some(ref mut store) = self.snapshot_store {
            store.save(&snapshot)?;
        }

        // Update snapshot metadata
        self.snapshot_last_index = snapshot.last_included_index;
        self.snapshot_last_term = snapshot.last_included_term;

        // Compact log
        self.compact_log(snapshot.last_included_index)?;

        info!(
            "Node {} created snapshot at index {}",
            self.config.id, snapshot.last_included_index
        );

        Ok(())
    }

    /// Maybe create a snapshot if threshold exceeded.
    pub fn maybe_create_snapshot(&mut self) -> Result<bool> {
        let entries_since_snapshot = self.last_applied - self.snapshot_last_index;
        if entries_since_snapshot >= self.config.snapshot_threshold {
            self.create_snapshot()?;
            return Ok(true);
        }
        Ok(false)
    }

    /// Compact log up to given index.
    pub fn compact_log(&mut self, up_to: LogIndex) -> Result<()> {
        // Remove entries up to and including the given index
        self.log.retain(|e| e.index > up_to);

        // Compact storage if available
        if let Some(ref mut storage) = self.storage {
            storage.compact(up_to)?;
        }

        debug!(
            "Node {} compacted log up to index {}",
            self.config.id, up_to
        );

        Ok(())
    }

    // ========================================================================
    // Linearizable reads
    // ========================================================================

    /// Perform a linearizable read (read-index protocol).
    pub fn linearizable_read(&mut self, key: Vec<u8>) -> Result<Option<Vec<u8>>> {
        // Only leader can serve linearizable reads
        if !self.is_leader() {
            return Err(Error::NotLeader(self.leader_id()));
        }

        // If leader lease is valid, we can read locally
        if self.config.enable_leader_lease && self.lease_valid() {
            return Ok(self.fsm.get(&key));
        }

        // Otherwise, we need to verify leadership with quorum
        // For simplicity in single-threaded context, we use committed entries
        // In async context, this would require heartbeat exchange

        // Wait for any pending entries to be applied
        if self.last_applied >= self.commit_index {
            return Ok(self.fsm.get(&key));
        }

        // Need to wait for entries to be applied
        Err(Error::Internal("Read index not yet applied".to_string()))
    }

    /// Perform a lease-based read (fast path).
    pub fn lease_read(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if !self.is_leader() {
            return Err(Error::NotLeader(self.leader_id()));
        }

        if !self.lease_valid() {
            return Err(Error::Internal("Leader lease expired".to_string()));
        }

        Ok(self.fsm.get(key))
    }

    /// Update leader lease after successful quorum acknowledgment.
    pub fn update_leader_lease(&mut self) {
        if !self.config.enable_leader_lease {
            return;
        }

        if let RaftState::Leader { lease_expiry, .. } = &mut self.state {
            let lease_duration = self.config.election_timeout_min / 2;
            *lease_expiry = Some(Instant::now() + lease_duration);
        }
    }

    // ========================================================================
    // Cluster membership changes (joint consensus)
    // ========================================================================

    /// Start a membership change (add/remove node).
    pub fn propose_membership_change(&mut self, new_nodes: HashSet<NodeId>) -> Result<LogIndex> {
        if !self.is_leader() {
            return Err(Error::NotLeader(self.leader_id()));
        }

        // Check if already in joint consensus
        if self.cluster_config.is_joint() {
            return Err(Error::InvalidStateTransition(
                "Already in joint consensus".to_string(),
            ));
        }

        // Create joint configuration entry
        let config_change = ConfigurationChange {
            old_nodes: self.cluster_config.current.clone(),
            new_nodes: new_nodes.clone(),
            is_final: false,
        };

        let entry = LogEntry {
            term: self.current_term,
            index: self.last_log_index() + 1,
            command: Command::NoOp, // Configuration is carried in entry_type
            entry_type: EntryType::Configuration,
        };

        let index = entry.index;
        self.log.push(entry);

        // Update cluster config to joint
        self.cluster_config.next = Some(new_nodes);

        info!(
            "Node {} proposed membership change at index {}",
            self.config.id, index
        );

        Ok(index)
    }

    /// Finalize membership change (after joint config is committed).
    pub fn finalize_membership_change(&mut self) -> Result<LogIndex> {
        if !self.is_leader() {
            return Err(Error::NotLeader(self.leader_id()));
        }

        // Check if in joint consensus
        let new_nodes = match &self.cluster_config.next {
            Some(nodes) => nodes.clone(),
            None => {
                return Err(Error::InvalidStateTransition(
                    "Not in joint consensus".to_string(),
                ))
            }
        };

        // Create final configuration entry
        let entry = LogEntry {
            term: self.current_term,
            index: self.last_log_index() + 1,
            command: Command::NoOp,
            entry_type: EntryType::Configuration,
        };

        let index = entry.index;
        self.log.push(entry);

        // Update cluster config to new
        self.cluster_config.current = new_nodes;
        self.cluster_config.next = None;

        info!(
            "Node {} finalized membership change at index {}",
            self.config.id, index
        );

        Ok(index)
    }

    /// Add a node to the cluster.
    pub fn add_node(&mut self, node_id: NodeId) -> Result<LogIndex> {
        let mut new_nodes = self.cluster_config.current.clone();
        new_nodes.insert(node_id);
        self.propose_membership_change(new_nodes)
    }

    /// Remove a node from the cluster.
    pub fn remove_node(&mut self, node_id: NodeId) -> Result<LogIndex> {
        let mut new_nodes = self.cluster_config.current.clone();
        new_nodes.remove(&node_id);
        self.propose_membership_change(new_nodes)
    }

    /// Transfer leadership to another node.
    pub fn transfer_leadership(&mut self, target: NodeId) -> Result<()> {
        if !self.is_leader() {
            return Err(Error::NotLeader(self.leader_id()));
        }

        if !self.cluster_config.current.contains(&target) {
            return Err(Error::Config(format!(
                "Target node {} not in cluster",
                target
            )));
        }

        // Step down and let target node time out and start election
        info!(
            "Node {} transferring leadership to {}",
            self.config.id, target
        );

        // For now, just step down
        self.transition_to_follower(Some(target));

        Ok(())
    }

    // ========================================================================
    // Pre-vote extension (prevents disruption from partitioned nodes)
    // ========================================================================

    /// Start pre-vote phase.
    pub fn start_pre_vote(&mut self) -> bool {
        if !self.enable_pre_vote {
            return false;
        }

        // Don't start pre-vote if we're the leader
        if self.is_leader() {
            return false;
        }

        // Start pre-vote with term+1
        let pre_vote_term = self.current_term + 1;
        let mut votes = HashSet::new();
        votes.insert(self.config.id); // Vote for self

        self.pre_vote_state = Some(PreVoteState {
            term: pre_vote_term,
            votes_received: votes,
        });

        info!(
            "Node {} starting pre-vote for term {}",
            self.config.id, pre_vote_term
        );

        true
    }

    /// Handle pre-vote request.
    pub fn handle_pre_vote_request(&self, req: &RequestVoteRequest) -> RequestVoteResponse {
        // Pre-vote doesn't increment term, just checks if vote would be granted
        let would_grant = self.would_grant_vote(req);

        RequestVoteResponse {
            term: self.current_term,
            vote_granted: would_grant,
        }
    }

    /// Handle pre-vote response.
    pub fn handle_pre_vote_response(&mut self, from: NodeId, resp: RequestVoteResponse) -> bool {
        if let Some(ref mut pre_vote) = self.pre_vote_state {
            if resp.vote_granted {
                pre_vote.votes_received.insert(from);

                // Check if we have pre-vote quorum
                let quorum = self.cluster_config.quorum_size();
                if pre_vote.votes_received.len() >= quorum {
                    // Pre-vote successful, now start real election
                    info!(
                        "Node {} won pre-vote, starting election",
                        self.config.id
                    );
                    self.pre_vote_state = None;
                    self.transition_to_candidate();
                    return true;
                }
            }
        }
        false
    }

    /// Check if we would grant a vote (without actually granting).
    fn would_grant_vote(&self, req: &RequestVoteRequest) -> bool {
        // Check if candidate's log is at least as up-to-date
        let last_log_term = self.last_log_term();
        let last_log_index = self.last_log_index();

        let log_ok = req.last_log_term > last_log_term
            || (req.last_log_term == last_log_term && req.last_log_index >= last_log_index);

        // Check if we haven't heard from leader recently
        let leader_active = !self.election_timeout_elapsed();

        // Would grant if: log is ok AND (we haven't voted OR would vote for this candidate)
        // AND we haven't heard from leader recently
        log_ok && !leader_active
    }

    // ========================================================================
    // Check quorum (leader steps down if partitioned)
    // ========================================================================

    /// Check if leader can still reach quorum.
    pub fn check_quorum(&mut self) -> bool {
        if !self.is_leader() {
            return true; // Only applies to leaders
        }

        if !self.enable_check_quorum {
            return true;
        }

        let now = Instant::now();
        let check_interval = self.config.election_timeout_min;

        // Only check periodically
        if now.duration_since(self.last_quorum_check) < check_interval {
            return true;
        }

        self.last_quorum_check = now;

        // Count reachable peers
        let mut reachable = 1; // Self
        let election_timeout = self.config.election_timeout_max;

        for (_peer_id, last_contact) in &self.peer_last_contact {
            if now.duration_since(*last_contact) < election_timeout {
                reachable += 1;
            }
        }

        let quorum = self.cluster_config.quorum_size();
        if reachable < quorum {
            warn!(
                "Node {} stepping down: can only reach {} of {} quorum",
                self.config.id, reachable, quorum
            );
            self.transition_to_follower(None);
            return false;
        }

        true
    }

    /// Record contact with a peer.
    pub fn record_peer_contact(&mut self, peer_id: NodeId) {
        self.peer_last_contact.insert(peer_id, Instant::now());
    }

    // ========================================================================
    // Batching and pipelining
    // ========================================================================

    /// Batch multiple commands into a single log append.
    pub fn batch_propose(&mut self, commands: Vec<Command>) -> Result<Vec<LogIndex>> {
        if !self.is_leader() {
            return Err(Error::NotLeader(self.leader_id()));
        }

        let mut indices = Vec::with_capacity(commands.len());
        let start_index = self.last_log_index() + 1;

        for (i, command) in commands.into_iter().enumerate() {
            let entry = LogEntry {
                term: self.current_term,
                index: start_index + i as u64,
                command,
                entry_type: EntryType::Command,
            };
            self.log.push(entry);
            indices.push(start_index + i as u64);
        }

        Ok(indices)
    }

    // ========================================================================
    // Write and read operations (for client API)
    // ========================================================================

    /// Write a key-value pair.
    pub fn write(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        self.propose(Command::Put { key, value })?;
        Ok(())
    }

    /// Read a value by key.
    pub fn read(&self, key: Vec<u8>) -> Result<Option<Vec<u8>>> {
        Ok(self.fsm.get(&key))
    }

    /// Delete a key.
    pub fn delete(&mut self, key: Vec<u8>) -> Result<()> {
        self.propose(Command::Delete { key })?;
        Ok(())
    }

    // ========================================================================
    // Update match index (for leader tracking)
    // ========================================================================

    /// Update match index for a peer.
    pub fn update_match_index(&mut self, peer_id: NodeId, match_idx: LogIndex) {
        if let RaftState::Leader { match_index, .. } = &mut self.state {
            match_index.insert(peer_id, match_idx);
            self.try_advance_commit_index();
        }
    }

    /// Get match index for a peer.
    pub fn get_match_index(&self, peer_id: NodeId) -> Option<LogIndex> {
        if let RaftState::Leader { match_index, .. } = &self.state {
            return match_index.get(&peer_id).copied();
        }
        None
    }

    /// Get next index for a peer.
    pub fn get_next_index(&self, peer_id: NodeId) -> Option<LogIndex> {
        if let RaftState::Leader { next_index, .. } = &self.state {
            return next_index.get(&peer_id).copied();
        }
        None
    }

    // ========================================================================
    // Additional accessors
    // ========================================================================

    /// Get FSM.
    pub fn fsm(&self) -> &KeyValueFSM {
        &self.fsm
    }

    /// Get mutable FSM.
    pub fn fsm_mut(&mut self) -> &mut KeyValueFSM {
        &mut self.fsm
    }

    /// Get all log entries.
    pub fn log_entries(&self) -> &[LogEntry] {
        &self.log
    }

    /// Get entries in range.
    pub fn get_entries(&self, from: LogIndex, to: LogIndex) -> Vec<LogEntry> {
        self.log
            .iter()
            .filter(|e| e.index >= from && e.index <= to)
            .cloned()
            .collect()
    }
}

/// Thread-safe wrapper for RaftNode.
pub type SharedRaftNode = Arc<RwLock<RaftNode>>;
