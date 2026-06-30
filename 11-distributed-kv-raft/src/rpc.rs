//! RPC message types for Raft communication.

use crate::node::LogEntry;
use crate::{LogIndex, NodeId, Term};
use serde::{Deserialize, Serialize};

/// AppendEntries RPC request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesRequest {
    /// Leader's term.
    pub term: Term,
    /// Leader's ID.
    pub leader_id: NodeId,
    /// Index of log entry immediately preceding new ones.
    pub prev_log_index: LogIndex,
    /// Term of prev_log_index entry.
    pub prev_log_term: Term,
    /// Log entries to store (empty for heartbeat).
    pub entries: Vec<LogEntry>,
    /// Leader's commit index.
    pub leader_commit: LogIndex,
}

/// AppendEntries RPC response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppendEntriesResponse {
    /// Current term for leader to update itself.
    pub term: Term,
    /// True if follower contained entry matching prev_log_index and prev_log_term.
    pub success: bool,
    /// Optimization: index of conflicting entry (for faster log catchup).
    pub conflict_index: Option<LogIndex>,
    /// Optimization: term of conflicting entry.
    pub conflict_term: Option<Term>,
}

/// RequestVote RPC request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteRequest {
    /// Candidate's term.
    pub term: Term,
    /// Candidate requesting vote.
    pub candidate_id: NodeId,
    /// Index of candidate's last log entry.
    pub last_log_index: LogIndex,
    /// Term of candidate's last log entry.
    pub last_log_term: Term,
}

/// RequestVote RPC response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestVoteResponse {
    /// Current term for candidate to update itself.
    pub term: Term,
    /// True if candidate received vote.
    pub vote_granted: bool,
}

/// InstallSnapshot RPC request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotRequest {
    /// Leader's term.
    pub term: Term,
    /// Leader's ID.
    pub leader_id: NodeId,
    /// Snapshot replaces all entries up through and including this index.
    pub last_included_index: LogIndex,
    /// Term of last_included_index.
    pub last_included_term: Term,
    /// Byte offset where chunk is positioned in the snapshot file.
    pub offset: u64,
    /// Raw bytes of the snapshot chunk.
    pub data: Vec<u8>,
    /// True if this is the last chunk.
    pub done: bool,
}

/// InstallSnapshot RPC response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallSnapshotResponse {
    /// Current term for leader to update itself.
    pub term: Term,
}

/// Client request to the Raft cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientRequest {
    /// Get a value.
    Get { key: Vec<u8> },
    /// Put a key-value pair.
    Put { key: Vec<u8>, value: Vec<u8> },
    /// Delete a key.
    Delete { key: Vec<u8> },
}

/// Client response from the Raft cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientResponse {
    /// Success with optional value.
    Success { value: Option<Vec<u8>> },
    /// Not the leader, try this node instead.
    NotLeader { leader_hint: Option<NodeId> },
    /// Error occurred.
    Error { message: String },
}
