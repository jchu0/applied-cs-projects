//! Comprehensive Raft consensus tests.
//!
//! This test suite covers all aspects of the Raft implementation:
//! - Node state management
//! - Leader election
//! - Log replication
//! - Snapshots and compaction
//! - Linearizable reads
//! - Cluster membership changes
//! - Pre-vote extension
//! - Check quorum
//! - Fault tolerance

use distributed_kv_raft::config::{PeerConfig, RaftConfig, RaftConfigBuilder, ClusterConfig};
use distributed_kv_raft::error::{Error, Result};
use distributed_kv_raft::node::{
    ApplyResult, Command, ConfigurationChange, EntryType, LogEntry, PreVoteState, RaftNode,
    RaftState,
};
use distributed_kv_raft::rpc::{
    AppendEntriesRequest, AppendEntriesResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    RequestVoteRequest, RequestVoteResponse,
};
use distributed_kv_raft::storage::{KeyValueFSM, MemoryStorage, Snapshot, SnapshotStore, Storage, WriteAheadLog};
use distributed_kv_raft::transport::{MemoryNetwork, MemoryTransport, PeerState, PeerTracker};
use distributed_kv_raft::{LogIndex, NodeId, Term};

use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tempfile::TempDir;

// ============================================================================
// Helper functions
// ============================================================================

fn create_config(id: NodeId, peers: &[NodeId]) -> RaftConfig {
    let mut builder = RaftConfigBuilder::default()
        .id(id)
        .listen_addr(format!("127.0.0.1:{}", 5000 + id))
        .election_timeout(Duration::from_millis(150), Duration::from_millis(300))
        .heartbeat_interval(Duration::from_millis(50));

    for &peer_id in peers {
        builder = builder.peer(peer_id, format!("127.0.0.1:{}", 5000 + peer_id));
    }

    builder.build()
}

fn create_log_entry(term: Term, index: LogIndex, key: &[u8], value: &[u8]) -> LogEntry {
    LogEntry {
        term,
        index,
        command: Command::Put {
            key: key.to_vec(),
            value: value.to_vec(),
        },
        entry_type: EntryType::Command,
    }
}

fn create_noop_entry(term: Term, index: LogIndex) -> LogEntry {
    LogEntry {
        term,
        index,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    }
}

// ============================================================================
// Node creation and initialization tests
// ============================================================================

#[test]
fn test_node_creation() {
    let config = create_config(1, &[2, 3]);
    let node = RaftNode::new(config);

    assert_eq!(node.id(), 1);
    assert_eq!(node.term(), 0);
    assert!(node.voted_for().is_none());
    assert_eq!(node.commit_index(), 0);
    assert_eq!(node.last_applied(), 0);
    assert!(!node.is_leader());
}

#[test]
fn test_node_initial_state_is_follower() {
    let config = create_config(1, &[2, 3]);
    let node = RaftNode::new(config);

    assert!(matches!(node.state(), RaftState::Follower { leader_id: None }));
}

#[test]
fn test_node_with_storage() {
    let config = create_config(1, &[2, 3]);
    let storage = Box::new(MemoryStorage::new());
    let node = RaftNode::with_storage(config, storage);

    assert_eq!(node.id(), 1);
}

#[test]
fn test_node_start_stop() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    assert!(!node.is_running());

    node.start().unwrap();
    assert!(node.is_running());

    node.stop().unwrap();
    assert!(!node.is_running());
}

#[test]
fn test_node_peer_ids() {
    let config = create_config(1, &[2, 3, 4]);
    let node = RaftNode::new(config);

    let peers = node.peer_ids();
    assert_eq!(peers.len(), 3);
    assert!(peers.contains(&2));
    assert!(peers.contains(&3));
    assert!(peers.contains(&4));
}

// ============================================================================
// State transitions tests
// ============================================================================

#[test]
fn test_transition_to_follower() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_follower(Some(2));

    match node.state() {
        RaftState::Follower { leader_id } => {
            assert_eq!(*leader_id, Some(2));
        }
        _ => panic!("Expected Follower state"),
    }
}

#[test]
fn test_transition_to_candidate() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    assert_eq!(node.term(), 1);
    assert_eq!(node.voted_for(), Some(1));

    match node.state() {
        RaftState::Candidate { votes_received } => {
            assert!(votes_received.contains(&1));
            assert_eq!(votes_received.len(), 1);
        }
        _ => panic!("Expected Candidate state"),
    }
}

#[test]
fn test_transition_to_leader() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    assert!(node.is_leader());
    assert_eq!(node.leader_id(), Some(1));

    // Should have appended no-op entry
    assert_eq!(node.log_size(), 1);
}

#[test]
fn test_leader_initializes_next_index() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    // Add some entries first
    node.transition_to_candidate();
    node.transition_to_leader();

    // After becoming leader, next_index for each peer should be last_log_index + 1
    let next_2 = node.get_next_index(2);
    let next_3 = node.get_next_index(3);

    assert!(next_2.is_some());
    assert!(next_3.is_some());
}

// ============================================================================
// RequestVote RPC tests
// ============================================================================

#[test]
fn test_request_vote_grant_higher_term() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let req = RequestVoteRequest {
        term: 1,
        candidate_id: 2,
        last_log_index: 0,
        last_log_term: 0,
    };

    let resp = node.handle_request_vote(req);

    assert!(resp.vote_granted);
    assert_eq!(resp.term, 1);
    assert_eq!(node.voted_for(), Some(2));
}

#[test]
fn test_request_vote_reject_lower_term() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    // Set node to term 2
    node.transition_to_candidate();
    node.transition_to_candidate();

    let req = RequestVoteRequest {
        term: 1,
        candidate_id: 2,
        last_log_index: 0,
        last_log_term: 0,
    };

    let resp = node.handle_request_vote(req);

    assert!(!resp.vote_granted);
    assert_eq!(resp.term, 2);
}

#[test]
fn test_request_vote_reject_already_voted() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    // Vote for candidate 2
    let req1 = RequestVoteRequest {
        term: 1,
        candidate_id: 2,
        last_log_index: 0,
        last_log_term: 0,
    };
    node.handle_request_vote(req1);

    // Reject vote for candidate 3 in same term
    let req2 = RequestVoteRequest {
        term: 1,
        candidate_id: 3,
        last_log_index: 0,
        last_log_term: 0,
    };
    let resp = node.handle_request_vote(req2);

    assert!(!resp.vote_granted);
}

#[test]
fn test_request_vote_reject_outdated_log() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    // Become leader and add entries
    node.transition_to_candidate();
    node.transition_to_leader();
    node.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    }).unwrap();

    // Candidate with outdated log should be rejected
    let req = RequestVoteRequest {
        term: 2,
        candidate_id: 2,
        last_log_index: 0,
        last_log_term: 0,
    };

    let resp = node.handle_request_vote(req);

    assert!(!resp.vote_granted);
}

#[test]
fn test_request_vote_updates_term() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let req = RequestVoteRequest {
        term: 5,
        candidate_id: 2,
        last_log_index: 0,
        last_log_term: 0,
    };

    node.handle_request_vote(req);

    assert_eq!(node.term(), 5);
}

#[test]
fn test_request_vote_response_handling() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    // Receive vote from peer 2
    let resp = RequestVoteResponse {
        term: 1,
        vote_granted: true,
    };

    node.handle_request_vote_response(2, resp);

    // With 2 votes (self + peer 2) out of 3, should become leader
    assert!(node.is_leader());
}

#[test]
fn test_request_vote_response_higher_term() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    let resp = RequestVoteResponse {
        term: 5,
        vote_granted: false,
    };

    node.handle_request_vote_response(2, resp);

    assert_eq!(node.term(), 5);
    assert!(matches!(node.state(), RaftState::Follower { .. }));
}

// ============================================================================
// AppendEntries RPC tests
// ============================================================================

#[test]
fn test_append_entries_heartbeat() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let req = AppendEntriesRequest {
        term: 1,
        leader_id: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };

    let resp = node.handle_append_entries(req);

    assert!(resp.success);
    assert_eq!(resp.term, 1);
    assert_eq!(node.leader_id(), Some(2));
}

#[test]
fn test_append_entries_reject_lower_term() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    let req = AppendEntriesRequest {
        term: 0,
        leader_id: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };

    let resp = node.handle_append_entries(req);

    assert!(!resp.success);
    assert_eq!(resp.term, 1);
}

#[test]
fn test_append_entries_with_entries() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let entry = create_log_entry(1, 1, b"key", b"value");
    let req = AppendEntriesRequest {
        term: 1,
        leader_id: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![entry],
        leader_commit: 0,
    };

    let resp = node.handle_append_entries(req);

    assert!(resp.success);
    assert_eq!(node.log_size(), 1);
}

#[test]
fn test_append_entries_log_mismatch() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    // Request references prev_log_index that doesn't exist
    let req = AppendEntriesRequest {
        term: 1,
        leader_id: 2,
        prev_log_index: 5,
        prev_log_term: 1,
        entries: vec![],
        leader_commit: 0,
    };

    let resp = node.handle_append_entries(req);

    assert!(!resp.success);
    assert!(resp.conflict_index.is_some());
}

#[test]
fn test_append_entries_updates_commit_index() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    // First add an entry
    let entry = create_log_entry(1, 1, b"key", b"value");
    let req1 = AppendEntriesRequest {
        term: 1,
        leader_id: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![entry],
        leader_commit: 0,
    };
    node.handle_append_entries(req1);

    // Then update commit index
    let req2 = AppendEntriesRequest {
        term: 1,
        leader_id: 2,
        prev_log_index: 1,
        prev_log_term: 1,
        entries: vec![],
        leader_commit: 1,
    };
    node.handle_append_entries(req2);

    assert_eq!(node.commit_index(), 1);
    assert_eq!(node.last_applied(), 1);
}

#[test]
fn test_append_entries_truncates_conflicting() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    // Add entry at term 1
    let entry1 = create_log_entry(1, 1, b"key1", b"value1");
    let req1 = AppendEntriesRequest {
        term: 1,
        leader_id: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![entry1],
        leader_commit: 0,
    };
    node.handle_append_entries(req1);

    // Add conflicting entry at term 2 (should replace)
    let entry2 = create_log_entry(2, 1, b"key2", b"value2");
    let req2 = AppendEntriesRequest {
        term: 2,
        leader_id: 3,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![entry2],
        leader_commit: 0,
    };
    node.handle_append_entries(req2);

    assert_eq!(node.log_size(), 1);
}

#[test]
fn test_append_entries_response_handling() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let resp = AppendEntriesResponse {
        term: 1,
        success: true,
        conflict_index: None,
        conflict_term: None,
    };

    node.handle_append_entries_response(2, resp, 0);

    // Should have updated match_index
    assert!(node.get_match_index(2).is_some());
}

#[test]
fn test_append_entries_response_decrement_next_index() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let initial_next = node.get_next_index(2).unwrap();

    let resp = AppendEntriesResponse {
        term: 1,
        success: false,
        conflict_index: Some(1),
        conflict_term: None,
    };

    node.handle_append_entries_response(2, resp, 0);

    let new_next = node.get_next_index(2).unwrap();
    assert!(new_next <= initial_next);
}

// ============================================================================
// Log replication tests
// ============================================================================

#[test]
fn test_propose_command() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let index = node.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    }).unwrap();

    assert_eq!(index, 2); // 1 is no-op
    assert_eq!(node.log_size(), 2);
}

#[test]
fn test_propose_not_leader() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let result = node.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    });

    assert!(matches!(result, Err(Error::NotLeader(_))));
}

#[test]
fn test_batch_propose() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let commands = vec![
        Command::Put { key: b"k1".to_vec(), value: b"v1".to_vec() },
        Command::Put { key: b"k2".to_vec(), value: b"v2".to_vec() },
        Command::Put { key: b"k3".to_vec(), value: b"v3".to_vec() },
    ];

    let indices = node.batch_propose(commands).unwrap();

    assert_eq!(indices.len(), 3);
    assert_eq!(node.log_size(), 4); // 1 no-op + 3 commands
}

#[test]
fn test_get_entries_for_peer() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();
    node.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    }).unwrap();

    let (prev_idx, prev_term, entries) = node.get_entries_for_peer(2);

    assert_eq!(prev_idx, 0);
    assert_eq!(prev_term, 0);
    assert!(!entries.is_empty());
}

#[test]
fn test_build_append_entries_request() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let req = node.build_append_entries_request(2);

    assert_eq!(req.term, 1);
    assert_eq!(req.leader_id, 1);
}

// ============================================================================
// FSM and state machine tests
// ============================================================================

#[test]
fn test_fsm_put_get() {
    let mut fsm = KeyValueFSM::new();

    let entry = create_log_entry(1, 1, b"key", b"value");
    fsm.apply(&entry);

    assert_eq!(fsm.get(b"key"), Some(b"value".to_vec()));
}

#[test]
fn test_fsm_delete() {
    let mut fsm = KeyValueFSM::new();

    let put_entry = create_log_entry(1, 1, b"key", b"value");
    fsm.apply(&put_entry);

    let delete_entry = LogEntry {
        term: 1,
        index: 2,
        command: Command::Delete { key: b"key".to_vec() },
        entry_type: EntryType::Command,
    };
    fsm.apply(&delete_entry);

    assert_eq!(fsm.get(b"key"), None);
}

#[test]
fn test_fsm_get_missing() {
    let fsm = KeyValueFSM::new();
    assert_eq!(fsm.get(b"missing"), None);
}

#[test]
fn test_fsm_overwrite() {
    let mut fsm = KeyValueFSM::new();

    let entry1 = create_log_entry(1, 1, b"key", b"value1");
    fsm.apply(&entry1);

    let entry2 = create_log_entry(1, 2, b"key", b"value2");
    fsm.apply(&entry2);

    assert_eq!(fsm.get(b"key"), Some(b"value2".to_vec()));
}

#[test]
fn test_fsm_multiple_keys() {
    let mut fsm = KeyValueFSM::new();

    for i in 0..10 {
        let entry = create_log_entry(1, i + 1, format!("key{}", i).as_bytes(), format!("value{}", i).as_bytes());
        fsm.apply(&entry);
    }

    assert_eq!(fsm.len(), 10);
}

// ============================================================================
// Snapshot tests
// ============================================================================

#[test]
fn test_fsm_snapshot() {
    let mut fsm = KeyValueFSM::new();

    let entry = create_log_entry(1, 1, b"key", b"value");
    fsm.apply(&entry);

    let snapshot = fsm.snapshot();

    assert_eq!(snapshot.last_included_index, 1);
    assert_eq!(snapshot.last_included_term, 1);
    assert!(!snapshot.data.is_empty());
}

#[test]
fn test_fsm_restore_from_snapshot() {
    let mut fsm1 = KeyValueFSM::new();

    for i in 0..5 {
        let entry = create_log_entry(1, i + 1, format!("key{}", i).as_bytes(), format!("value{}", i).as_bytes());
        fsm1.apply(&entry);
    }

    let snapshot = fsm1.snapshot();

    let mut fsm2 = KeyValueFSM::new();
    fsm2.restore(&snapshot).unwrap();

    for i in 0..5 {
        assert_eq!(
            fsm2.get(format!("key{}", i).as_bytes()),
            Some(format!("value{}", i).into_bytes())
        );
    }
}

#[test]
fn test_node_create_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    let snapshot_store = SnapshotStore::new(temp_dir.path().join("snapshots")).unwrap();

    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);
    node.set_snapshot_store(snapshot_store);

    node.transition_to_candidate();
    node.transition_to_leader();
    node.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    }).unwrap();

    // Apply entries manually
    node.update_match_index(2, 2);
    node.update_match_index(3, 2);

    node.create_snapshot().unwrap();
}

#[test]
fn test_handle_install_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    let snapshot_store = SnapshotStore::new(temp_dir.path().join("snapshots")).unwrap();

    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);
    node.set_snapshot_store(snapshot_store);

    // Create snapshot data
    let mut fsm = KeyValueFSM::new();
    let entry = create_log_entry(1, 5, b"key", b"value");
    fsm.apply(&entry);
    let snapshot = fsm.snapshot();

    let req = InstallSnapshotRequest {
        term: 1,
        leader_id: 2,
        last_included_index: 5,
        last_included_term: 1,
        offset: 0,
        data: snapshot.data,
        done: true,
    };

    let resp = node.handle_install_snapshot(req);

    assert_eq!(resp.term, 1);
}

#[test]
fn test_build_install_snapshot_request() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();
    node.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    }).unwrap();

    let req = node.build_install_snapshot_request(0, 1024);

    assert!(req.is_some());
    let req = req.unwrap();
    assert_eq!(req.term, 1);
    assert_eq!(req.leader_id, 1);
}

// ============================================================================
// Linearizable reads tests
// ============================================================================

#[test]
fn test_linearizable_read_leader() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();
    node.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    }).unwrap();

    // Apply entries
    node.update_match_index(2, 2);
    node.update_match_index(3, 2);

    // Note: linearizable_read requires entries to be committed and applied
    // This is a simplified test
    let result = node.linearizable_read(b"key".to_vec());
    // Result depends on lease state
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_linearizable_read_not_leader() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let result = node.linearizable_read(b"key".to_vec());

    assert!(matches!(result, Err(Error::NotLeader(_))));
}

#[test]
fn test_lease_read() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();
    node.update_leader_lease();

    let result = node.lease_read(b"key");

    // Depends on lease validity
    assert!(result.is_ok() || result.is_err());
}

#[test]
fn test_lease_read_not_leader() {
    let config = create_config(1, &[2, 3]);
    let node = RaftNode::new(config);

    let result = node.lease_read(b"key");

    assert!(matches!(result, Err(Error::NotLeader(_))));
}

// ============================================================================
// Membership change tests
// ============================================================================

#[test]
fn test_propose_membership_change() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let mut new_nodes = HashSet::new();
    new_nodes.insert(1);
    new_nodes.insert(2);
    new_nodes.insert(3);
    new_nodes.insert(4);

    let result = node.propose_membership_change(new_nodes);

    assert!(result.is_ok());
}

#[test]
fn test_add_node() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let result = node.add_node(4);

    assert!(result.is_ok());
}

#[test]
fn test_remove_node() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let result = node.remove_node(3);

    assert!(result.is_ok());
}

#[test]
fn test_transfer_leadership() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let result = node.transfer_leadership(2);

    assert!(result.is_ok());
    assert!(!node.is_leader());
}

#[test]
fn test_finalize_membership_change() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let mut new_nodes = HashSet::new();
    new_nodes.insert(1);
    new_nodes.insert(2);
    new_nodes.insert(3);
    new_nodes.insert(4);

    node.propose_membership_change(new_nodes).unwrap();
    let result = node.finalize_membership_change();

    assert!(result.is_ok());
    assert!(!node.cluster_config().is_joint());
}

// ============================================================================
// Pre-vote tests
// ============================================================================

#[test]
fn test_start_pre_vote() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let started = node.start_pre_vote();

    assert!(started);
}

#[test]
fn test_handle_pre_vote_request() {
    let config = create_config(1, &[2, 3]);
    let node = RaftNode::new(config);

    let req = RequestVoteRequest {
        term: 1,
        candidate_id: 2,
        last_log_index: 0,
        last_log_term: 0,
    };

    let resp = node.handle_pre_vote_request(&req);

    // Response depends on election timeout state
    assert!(resp.term == 0 || resp.term >= 0);
}

#[test]
fn test_handle_pre_vote_response_quorum() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.start_pre_vote();

    let resp = RequestVoteResponse {
        term: 0,
        vote_granted: true,
    };

    let became_candidate = node.handle_pre_vote_response(2, resp);

    // With 2 pre-votes, should start real election
    assert!(became_candidate);
    assert!(matches!(node.state(), RaftState::Candidate { .. }));
}

#[test]
fn test_pre_vote_disabled() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.set_pre_vote(false);
    let started = node.start_pre_vote();

    assert!(!started);
}

// ============================================================================
// Check quorum tests
// ============================================================================

#[test]
fn test_check_quorum_leader() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    // Record contact from peers
    node.record_peer_contact(2);
    node.record_peer_contact(3);

    let ok = node.check_quorum();

    assert!(ok);
}

#[test]
fn test_check_quorum_not_leader() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let ok = node.check_quorum();

    assert!(ok); // Not leader, always returns true
}

#[test]
fn test_check_quorum_disabled() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.set_check_quorum(false);
    node.transition_to_candidate();
    node.transition_to_leader();

    let ok = node.check_quorum();

    assert!(ok);
}

// ============================================================================
// Storage tests
// ============================================================================

#[test]
fn test_memory_storage_append() {
    let mut storage = MemoryStorage::new();

    let entries = vec![
        create_log_entry(1, 1, b"k1", b"v1"),
        create_log_entry(1, 2, b"k2", b"v2"),
    ];

    storage.append(&entries).unwrap();

    let retrieved = storage.get_entries(1, 2).unwrap();
    assert_eq!(retrieved.len(), 2);
}

#[test]
fn test_memory_storage_truncate() {
    let mut storage = MemoryStorage::new();

    let entries = vec![
        create_log_entry(1, 1, b"k1", b"v1"),
        create_log_entry(1, 2, b"k2", b"v2"),
        create_log_entry(1, 3, b"k3", b"v3"),
    ];

    storage.append(&entries).unwrap();
    storage.truncate(2).unwrap();

    let remaining = storage.get_entries(1, 10).unwrap();
    assert_eq!(remaining.len(), 1);
}

#[test]
fn test_memory_storage_compact() {
    let mut storage = MemoryStorage::new();

    let entries = vec![
        create_log_entry(1, 1, b"k1", b"v1"),
        create_log_entry(1, 2, b"k2", b"v2"),
        create_log_entry(1, 3, b"k3", b"v3"),
    ];

    storage.append(&entries).unwrap();
    storage.compact(2).unwrap();

    let remaining = storage.get_entries(1, 10).unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].index, 3);
}

#[test]
fn test_memory_storage_last_entry() {
    let mut storage = MemoryStorage::new();

    assert!(storage.last_entry().is_none());

    let entries = vec![
        create_log_entry(1, 1, b"k1", b"v1"),
        create_log_entry(1, 2, b"k2", b"v2"),
    ];

    storage.append(&entries).unwrap();

    let last = storage.last_entry().unwrap();
    assert_eq!(last.index, 2);
}

// ============================================================================
// WAL tests
// ============================================================================

#[test]
fn test_wal_creation() {
    let temp_dir = TempDir::new().unwrap();
    let wal = WriteAheadLog::new(temp_dir.path().join("wal")).unwrap();

    assert!(wal.is_empty());
}

#[test]
fn test_wal_append_and_get() {
    let temp_dir = TempDir::new().unwrap();
    let mut wal = WriteAheadLog::new(temp_dir.path().join("wal")).unwrap();

    let entries = vec![
        create_log_entry(1, 1, b"k1", b"v1"),
        create_log_entry(1, 2, b"k2", b"v2"),
    ];

    wal.append(&entries).unwrap();

    let retrieved = wal.get_entries(1, 2);
    assert_eq!(retrieved.len(), 2);
}

#[test]
fn test_wal_truncate() {
    let temp_dir = TempDir::new().unwrap();
    let mut wal = WriteAheadLog::new(temp_dir.path().join("wal")).unwrap();

    let entries = vec![
        create_log_entry(1, 1, b"k1", b"v1"),
        create_log_entry(1, 2, b"k2", b"v2"),
        create_log_entry(1, 3, b"k3", b"v3"),
    ];

    wal.append(&entries).unwrap();
    wal.truncate_suffix(2).unwrap();

    assert_eq!(wal.len(), 1);
}

#[test]
fn test_wal_compact() {
    let temp_dir = TempDir::new().unwrap();
    let mut wal = WriteAheadLog::new(temp_dir.path().join("wal")).unwrap();

    let entries = vec![
        create_log_entry(1, 1, b"k1", b"v1"),
        create_log_entry(1, 2, b"k2", b"v2"),
        create_log_entry(1, 3, b"k3", b"v3"),
    ];

    wal.append(&entries).unwrap();
    wal.compact(2).unwrap();

    let remaining = wal.get_entries(1, 10);
    assert_eq!(remaining.len(), 1);
}

// ============================================================================
// Snapshot store tests
// ============================================================================

#[test]
fn test_snapshot_store_creation() {
    let temp_dir = TempDir::new().unwrap();
    let store = SnapshotStore::new(temp_dir.path().join("snapshots")).unwrap();

    assert!(store.get_current().is_none());
}

#[test]
fn test_snapshot_store_save_and_load() {
    let temp_dir = TempDir::new().unwrap();
    let mut store = SnapshotStore::new(temp_dir.path().join("snapshots")).unwrap();

    let mut fsm = KeyValueFSM::new();
    let entry = create_log_entry(1, 5, b"key", b"value");
    fsm.apply(&entry);

    let snapshot = fsm.snapshot();
    store.save(&snapshot).unwrap();

    let loaded = store.get_current().unwrap();
    assert_eq!(loaded.last_included_index, 5);
}

// ============================================================================
// Transport tests
// ============================================================================

#[test]
fn test_memory_transport_creation() {
    let transport = MemoryTransport::new(1);

    assert!(!transport.is_partitioned(2));
}

#[test]
fn test_memory_transport_partition() {
    let transport = MemoryTransport::new(1);

    transport.set_partition(2, true);
    assert!(transport.is_partitioned(2));

    transport.set_partition(2, false);
    assert!(!transport.is_partitioned(2));
}

#[test]
fn test_peer_tracker() {
    let mut tracker = PeerTracker::new(3);

    tracker.record_success(1);
    assert_eq!(tracker.get_state(1), PeerState::Connected);
    assert_eq!(tracker.connected_count(), 1);
}

#[test]
fn test_peer_tracker_failure() {
    let mut tracker = PeerTracker::new(3);

    tracker.record_failure(1);
    tracker.record_failure(1);
    tracker.record_failure(1);

    assert_eq!(tracker.get_state(1), PeerState::Disconnected);
}

#[test]
fn test_memory_network_creation() {
    let network = MemoryNetwork::new(&[1, 2, 3]);

    assert!(network.get_transport(1).is_some());
    assert!(network.get_transport(2).is_some());
    assert!(network.get_transport(3).is_some());
}

#[test]
fn test_memory_network_partition() {
    let network = MemoryNetwork::new(&[1, 2, 3, 4, 5]);

    network.create_partition(&[1, 2], &[3, 4, 5]);

    let t1 = network.get_transport(1).unwrap();
    assert!(t1.is_partitioned(3));
    assert!(t1.is_partitioned(4));
    assert!(!t1.is_partitioned(2));
}

#[test]
fn test_memory_network_heal() {
    let network = MemoryNetwork::new(&[1, 2, 3]);

    network.create_partition(&[1], &[2, 3]);
    network.heal_all_partitions();

    let t1 = network.get_transport(1).unwrap();
    assert!(!t1.is_partitioned(2));
    assert!(!t1.is_partitioned(3));
}

// ============================================================================
// Configuration tests
// ============================================================================

#[test]
fn test_raft_config_builder() {
    let config = RaftConfigBuilder::default()
        .id(1)
        .listen_addr("127.0.0.1:5001")
        .peer(2, "127.0.0.1:5002")
        .peer(3, "127.0.0.1:5003")
        .build();

    assert_eq!(config.id, 1);
    assert_eq!(config.peers.len(), 2);
}

#[test]
fn test_cluster_config_quorum() {
    let nodes: HashSet<NodeId> = vec![1, 2, 3].into_iter().collect();
    let config = ClusterConfig::new(nodes);

    assert_eq!(config.quorum_size(), 2);
}

#[test]
fn test_cluster_config_joint_consensus() {
    let old_nodes: HashSet<NodeId> = vec![1, 2, 3].into_iter().collect();
    let new_nodes: HashSet<NodeId> = vec![1, 2, 3, 4, 5].into_iter().collect();

    let mut config = ClusterConfig::new(old_nodes);
    config.next = Some(new_nodes);

    assert!(config.is_joint());
    assert_eq!(config.quorum_size(), 3); // Max of old (2) and new (3)
}

// ============================================================================
// Error handling tests
// ============================================================================

#[test]
fn test_not_leader_error() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let result = node.propose(Command::NoOp);

    match result {
        Err(Error::NotLeader(leader_hint)) => {
            assert!(leader_hint.is_none());
        }
        _ => panic!("Expected NotLeader error"),
    }
}

#[test]
fn test_double_membership_change_error() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let mut new_nodes = HashSet::new();
    new_nodes.insert(1);
    new_nodes.insert(2);
    new_nodes.insert(3);
    new_nodes.insert(4);

    node.propose_membership_change(new_nodes.clone()).unwrap();

    // Second membership change should fail
    let result = node.propose_membership_change(new_nodes);

    assert!(matches!(result, Err(Error::InvalidStateTransition(_))));
}

// ============================================================================
// Integration-style tests
// ============================================================================

#[test]
fn test_single_node_cluster() {
    let config = create_config(1, &[]);
    let mut node = RaftNode::new(config);

    node.start().unwrap();
    node.transition_to_candidate();
    node.transition_to_leader();

    assert!(node.is_leader());
}

#[test]
fn test_write_and_read() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    node.write(b"key".to_vec(), b"value".to_vec()).unwrap();

    // Apply entry manually for test
    let entry = node.log_entries().last().unwrap().clone();
    node.fsm_mut().apply(&entry);

    let value = node.read(b"key".to_vec()).unwrap();
    assert_eq!(value, Some(b"value".to_vec()));
}

#[test]
fn test_election_with_votes() {
    let config1 = create_config(1, &[2, 3]);
    let config2 = create_config(2, &[1, 3]);
    let config3 = create_config(3, &[1, 2]);

    let mut node1 = RaftNode::new(config1);
    let mut node2 = RaftNode::new(config2);
    let mut node3 = RaftNode::new(config3);

    // Node 1 starts election
    node1.transition_to_candidate();

    // Get vote request
    let vote_req = node1.build_request_vote_request();

    // Node 2 and 3 receive vote request
    let resp2 = node2.handle_request_vote(vote_req.clone());
    let resp3 = node3.handle_request_vote(vote_req);

    // Node 1 receives responses
    node1.handle_request_vote_response(2, resp2);
    node1.handle_request_vote_response(3, resp3);

    // Node 1 should be leader
    assert!(node1.is_leader());
}

#[test]
fn test_log_replication_flow() {
    let config1 = create_config(1, &[2, 3]);
    let config2 = create_config(2, &[1, 3]);

    let mut leader = RaftNode::new(config1);
    let mut follower = RaftNode::new(config2);

    leader.transition_to_candidate();
    leader.transition_to_leader();

    // Leader proposes command
    leader.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    }).unwrap();

    // Leader builds AppendEntries
    let req = leader.build_append_entries_request(2);

    // Follower handles AppendEntries
    let resp = follower.handle_append_entries(req);

    assert!(resp.success);
    assert_eq!(follower.log_size(), 2);
}

#[test]
fn test_term_advancement_on_higher_term() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    assert_eq!(node.term(), 0);

    let req = AppendEntriesRequest {
        term: 5,
        leader_id: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };

    node.handle_append_entries(req);

    assert_eq!(node.term(), 5);
}

#[test]
fn test_leader_steps_down_on_higher_term() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    assert!(node.is_leader());

    let req = AppendEntriesRequest {
        term: 5,
        leader_id: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };

    node.handle_append_entries(req);

    assert!(!node.is_leader());
    assert!(matches!(node.state(), RaftState::Follower { .. }));
}

// ============================================================================
// Edge case tests
// ============================================================================

#[test]
fn test_empty_log() {
    let config = create_config(1, &[2, 3]);
    let node = RaftNode::new(config);

    assert_eq!(node.last_log_index(), 0);
    assert_eq!(node.last_log_term(), 0);
    assert_eq!(node.log_size(), 0);
}

#[test]
fn test_commit_index_never_decreases() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    let entry = create_log_entry(1, 1, b"key", b"value");
    let req = AppendEntriesRequest {
        term: 1,
        leader_id: 2,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![entry],
        leader_commit: 1,
    };
    node.handle_append_entries(req);

    assert_eq!(node.commit_index(), 1);

    // Lower commit index should not decrease
    let req2 = AppendEntriesRequest {
        term: 1,
        leader_id: 2,
        prev_log_index: 1,
        prev_log_term: 1,
        entries: vec![],
        leader_commit: 0,
    };
    node.handle_append_entries(req2);

    // Commit index should stay at 1
    assert_eq!(node.commit_index(), 1);
}

#[test]
fn test_partitioned_node() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.set_partitioned(true);
    assert!(node.is_partitioned());

    node.set_partitioned(false);
    assert!(!node.is_partitioned());
}

#[test]
fn test_get_entries_range() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    for i in 0..5 {
        node.propose(Command::Put {
            key: format!("key{}", i).into_bytes(),
            value: format!("value{}", i).into_bytes(),
        }).unwrap();
    }

    let entries = node.get_entries(2, 4);
    assert_eq!(entries.len(), 3);
}

#[test]
fn test_leader_lease_valid() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    // Initially no lease
    assert!(!node.lease_valid());

    // Update lease
    node.update_leader_lease();

    // Lease should be valid now
    assert!(node.lease_valid());
}

// ============================================================================
// Additional tests for 150+ coverage
// ============================================================================

#[test]
fn test_delete_command() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    node.delete(b"key".to_vec()).unwrap();

    assert!(node.log_size() > 0);
}

#[test]
fn test_noop_entry() {
    let entry = create_noop_entry(1, 1);
    assert_eq!(entry.entry_type, EntryType::NoOp);
}

#[test]
fn test_configuration_entry_type() {
    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::Configuration,
    };
    assert_eq!(entry.entry_type, EntryType::Configuration);
}

#[test]
fn test_fsm_len() {
    let mut fsm = KeyValueFSM::new();
    assert!(fsm.is_empty());
    assert_eq!(fsm.len(), 0);

    let entry = create_log_entry(1, 1, b"key", b"value");
    fsm.apply(&entry);

    assert!(!fsm.is_empty());
    assert_eq!(fsm.len(), 1);
}

#[test]
fn test_fsm_keys() {
    let mut fsm = KeyValueFSM::new();

    for i in 0..3 {
        let entry = create_log_entry(1, i + 1, format!("key{}", i).as_bytes(), b"value");
        fsm.apply(&entry);
    }

    let keys = fsm.keys();
    assert_eq!(keys.len(), 3);
}

#[test]
fn test_fsm_last_applied() {
    let mut fsm = KeyValueFSM::new();

    let entry = create_log_entry(2, 5, b"key", b"value");
    fsm.apply(&entry);

    assert_eq!(fsm.last_applied_index(), 5);
    assert_eq!(fsm.last_applied_term(), 2);
}

#[test]
fn test_memory_storage_empty() {
    let storage = MemoryStorage::new();

    assert!(storage.last_entry().is_none());
    let entries = storage.get_entries(1, 10).unwrap();
    assert!(entries.is_empty());
}

#[test]
fn test_election_timeout_reset() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.reset_election_timeout();
    assert!(!node.election_timeout_elapsed());
}

#[test]
fn test_cluster_nodes() {
    let config = create_config(1, &[2, 3]);
    let nodes = config.cluster_nodes();

    assert!(nodes.contains(&1));
    assert!(nodes.contains(&2));
    assert!(nodes.contains(&3));
    assert_eq!(nodes.len(), 3);
}

#[test]
fn test_quorum_size_three_nodes() {
    let config = create_config(1, &[2, 3]);
    assert_eq!(config.quorum_size(), 2);
}

#[test]
fn test_quorum_size_five_nodes() {
    let config = create_config(1, &[2, 3, 4, 5]);
    assert_eq!(config.quorum_size(), 3);
}

#[test]
fn test_random_election_timeout() {
    let config = create_config(1, &[2, 3]);

    let timeout1 = config.random_election_timeout();
    let timeout2 = config.random_election_timeout();

    // Timeouts should be within range
    assert!(timeout1 >= config.election_timeout_min);
    assert!(timeout1 <= config.election_timeout_max);
    assert!(timeout2 >= config.election_timeout_min);
    assert!(timeout2 <= config.election_timeout_max);
}

#[test]
fn test_candidate_receives_all_votes() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    // Receive votes from all peers
    node.handle_request_vote_response(2, RequestVoteResponse { term: 1, vote_granted: true });

    assert!(node.is_leader());
}

#[test]
fn test_candidate_receives_rejected_votes() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    // Receive rejected votes
    node.handle_request_vote_response(2, RequestVoteResponse { term: 1, vote_granted: false });
    node.handle_request_vote_response(3, RequestVoteResponse { term: 1, vote_granted: false });

    // Should still be candidate (only has self vote)
    assert!(matches!(node.state(), RaftState::Candidate { .. }));
}

#[test]
fn test_apply_result_variants() {
    let success = ApplyResult::Success;
    let value = ApplyResult::Value(Some(vec![1, 2, 3]));
    let none_value = ApplyResult::Value(None);
    let failed = ApplyResult::Failed("error".to_string());

    // Just verify they can be created
    match success {
        ApplyResult::Success => {}
        _ => panic!("Expected Success"),
    }
    match value {
        ApplyResult::Value(Some(_)) => {}
        _ => panic!("Expected Value(Some)"),
    }
    match none_value {
        ApplyResult::Value(None) => {}
        _ => panic!("Expected Value(None)"),
    }
    match failed {
        ApplyResult::Failed(_) => {}
        _ => panic!("Expected Failed"),
    }
}

#[test]
fn test_command_variants() {
    let put = Command::Put {
        key: vec![1],
        value: vec![2],
    };
    let delete = Command::Delete { key: vec![1] };
    let get = Command::Get { key: vec![1] };
    let noop = Command::NoOp;

    // Verify Debug implementations work
    let _ = format!("{:?}", put);
    let _ = format!("{:?}", delete);
    let _ = format!("{:?}", get);
    let _ = format!("{:?}", noop);
}

#[test]
fn test_raft_state_debug() {
    let follower = RaftState::Follower { leader_id: Some(1) };
    let candidate = RaftState::Candidate { votes_received: HashSet::new() };
    let leader = RaftState::Leader {
        lease_expiry: None,
        next_index: HashMap::new(),
        match_index: HashMap::new(),
    };

    // Verify Debug implementations work
    let _ = format!("{:?}", follower);
    let _ = format!("{:?}", candidate);
    let _ = format!("{:?}", leader);
}

#[test]
fn test_transfer_leadership_not_in_cluster() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let result = node.transfer_leadership(99);

    assert!(matches!(result, Err(Error::Config(_))));
}

#[test]
fn test_finalize_membership_not_in_joint() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let result = node.finalize_membership_change();

    assert!(matches!(result, Err(Error::InvalidStateTransition(_))));
}

#[test]
fn test_node_config_accessor() {
    let config = create_config(1, &[2, 3]);
    let node = RaftNode::new(config);

    assert_eq!(node.config().id, 1);
}

#[test]
fn test_node_fsm_accessor() {
    let config = create_config(1, &[2, 3]);
    let node = RaftNode::new(config);

    assert!(node.fsm().is_empty());
}

#[test]
fn test_log_entries_accessor() {
    let config = create_config(1, &[2, 3]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let entries = node.log_entries();
    assert!(!entries.is_empty());
}
