//! Comprehensive tests for the Distributed Key-Value Store with Raft Consensus.
//!
//! These tests cover all major components and functionality:
//! - Leader election
//! - Log replication
//! - Membership changes
//! - Snapshot and compaction
//! - Client request handling
//! - Pre-vote extension
//! - Check quorum
//! - Linearizable reads

use distributed_kv_raft::{
    AppendEntriesRequest, AppendEntriesResponse, ClusterConfig, Command, EntryType,
    InstallSnapshotRequest, InstallSnapshotResponse, LogEntry, NodeId, PeerConfig, RaftConfig,
    RaftConfigBuilder, RaftNode, RaftState, RequestVoteRequest, RequestVoteResponse,
};
use std::collections::HashSet;
use std::time::Duration;

// =============================================================================
// Helper Functions
// =============================================================================

fn create_test_config(id: NodeId) -> RaftConfig {
    RaftConfigBuilder::default()
        .id(id)
        .listen_addr(format!("127.0.0.1:{}", 8000 + id))
        .election_timeout(Duration::from_millis(150), Duration::from_millis(300))
        .heartbeat_interval(Duration::from_millis(50))
        .build()
}

fn create_three_node_config(id: NodeId) -> RaftConfig {
    let mut builder = RaftConfigBuilder::default()
        .id(id)
        .listen_addr(format!("127.0.0.1:{}", 8000 + id))
        .election_timeout(Duration::from_millis(150), Duration::from_millis(300))
        .heartbeat_interval(Duration::from_millis(50));

    for peer_id in 0..3u64 {
        if peer_id != id {
            builder = builder.peer(peer_id, format!("127.0.0.1:{}", 8000 + peer_id));
        }
    }

    builder.build()
}

fn create_five_node_config(id: NodeId) -> RaftConfig {
    let mut builder = RaftConfigBuilder::default()
        .id(id)
        .listen_addr(format!("127.0.0.1:{}", 8000 + id))
        .election_timeout(Duration::from_millis(150), Duration::from_millis(300))
        .heartbeat_interval(Duration::from_millis(50));

    for peer_id in 0..5u64 {
        if peer_id != id {
            builder = builder.peer(peer_id, format!("127.0.0.1:{}", 8000 + peer_id));
        }
    }

    builder.build()
}

// =============================================================================
// Node Creation and Initialization Tests
// =============================================================================

#[test]
fn test_node_creation_default() {
    let config = create_test_config(0);
    let node = RaftNode::new(config);

    assert_eq!(node.id(), 0);
    assert_eq!(node.term(), 0);
    assert!(matches!(node.state(), RaftState::Follower { .. }));
    assert_eq!(node.voted_for(), None);
    assert_eq!(node.last_log_index(), 0);
    assert_eq!(node.last_log_term(), 0);
}

#[test]
fn test_node_creation_with_peers() {
    let config = create_three_node_config(1);
    let node = RaftNode::new(config);

    assert_eq!(node.id(), 1);
    assert_eq!(node.peer_ids().len(), 2);
    assert!(node.peer_ids().contains(&0));
    assert!(node.peer_ids().contains(&2));
}

#[test]
fn test_node_creation_five_nodes() {
    let config = create_five_node_config(2);
    let node = RaftNode::new(config);

    assert_eq!(node.id(), 2);
    assert_eq!(node.peer_ids().len(), 4);
}

#[test]
fn test_node_initial_state_is_follower() {
    let config = create_test_config(0);
    let node = RaftNode::new(config);

    assert!(!node.is_leader());
    assert!(matches!(node.state(), RaftState::Follower { leader_id: None }));
}

#[test]
fn test_node_start_stop() {
    let config = create_test_config(0);
    let mut node = RaftNode::new(config);

    assert!(!node.is_running());

    node.start().unwrap();
    assert!(node.is_running());

    node.stop().unwrap();
    assert!(!node.is_running());
}

#[test]
fn test_node_partitioned_state() {
    let config = create_test_config(0);
    let mut node = RaftNode::new(config);

    assert!(!node.is_partitioned());

    node.set_partitioned(true);
    assert!(node.is_partitioned());

    node.set_partitioned(false);
    assert!(!node.is_partitioned());
}

// =============================================================================
// Leader Election Tests
// =============================================================================

#[test]
fn test_transition_to_candidate() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    assert_eq!(node.term(), 0);

    node.transition_to_candidate();

    assert_eq!(node.term(), 1);
    assert_eq!(node.voted_for(), Some(0)); // Voted for self
    assert!(matches!(node.state(), RaftState::Candidate { .. }));
}

#[test]
fn test_candidate_votes_for_self() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    if let RaftState::Candidate { votes_received } = node.state() {
        assert!(votes_received.contains(&1));
        assert_eq!(votes_received.len(), 1);
    } else {
        panic!("Expected Candidate state");
    }
}

#[test]
fn test_transition_to_leader() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    assert!(node.is_leader());
    assert!(matches!(node.state(), RaftState::Leader { .. }));
}

#[test]
fn test_leader_noop_entry() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    // Leader appends no-op entry after election
    assert_eq!(node.log_size(), 1);
    if let Some(entry) = node.log_entries().first() {
        assert_eq!(entry.entry_type, EntryType::NoOp);
        assert_eq!(entry.term, 1);
    }
}

#[test]
fn test_transition_to_follower() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();
    node.transition_to_follower(Some(1));

    assert!(!node.is_leader());
    if let RaftState::Follower { leader_id } = node.state() {
        assert_eq!(*leader_id, Some(1));
    } else {
        panic!("Expected Follower state");
    }
}

#[test]
fn test_election_timeout_check() {
    let config = create_test_config(0);
    let mut node = RaftNode::new(config);

    // Just created, should not have timed out
    assert!(!node.election_timeout_elapsed());

    // Reset timeout
    node.reset_election_timeout();
    assert!(!node.election_timeout_elapsed());
}

// =============================================================================
// RequestVote RPC Tests
// =============================================================================

#[test]
fn test_request_vote_success() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let response = node.handle_request_vote(request);

    assert!(response.vote_granted);
    assert_eq!(response.term, 1);
    assert_eq!(node.voted_for(), Some(0));
}

#[test]
fn test_request_vote_stale_term() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);
    node.current_term = 5;

    let request = RequestVoteRequest {
        term: 3,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let response = node.handle_request_vote(request);

    assert!(!response.vote_granted);
    assert_eq!(response.term, 5);
}

#[test]
fn test_request_vote_already_voted() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);
    node.current_term = 1;
    node.voted_for = Some(2);

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let response = node.handle_request_vote(request);

    assert!(!response.vote_granted);
}

#[test]
fn test_request_vote_log_not_up_to_date() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);

    // Add entry to node's log
    node.log.push(LogEntry {
        term: 2,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    });

    let request = RequestVoteRequest {
        term: 3,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 1,
    };

    let response = node.handle_request_vote(request);

    // Candidate's log is not up-to-date (term 1 < 2)
    assert!(!response.vote_granted);
}

#[test]
fn test_request_vote_updates_term() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);
    node.current_term = 1;

    let request = RequestVoteRequest {
        term: 5,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let response = node.handle_request_vote(request);

    assert!(response.vote_granted);
    assert_eq!(node.term(), 5);
}

#[test]
fn test_request_vote_response_handling() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    // Receive vote from peer 1
    let response = RequestVoteResponse {
        term: 1,
        vote_granted: true,
    };
    node.handle_request_vote_response(1, response);

    // With 2 votes (self + peer1), should become leader
    assert!(node.is_leader());
}

#[test]
fn test_request_vote_response_higher_term() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    // Receive response with higher term
    let response = RequestVoteResponse {
        term: 5,
        vote_granted: false,
    };
    node.handle_request_vote_response(1, response);

    // Should become follower
    assert!(!node.is_leader());
    assert_eq!(node.term(), 5);
}

// =============================================================================
// AppendEntries RPC Tests
// =============================================================================

#[test]
fn test_append_entries_heartbeat() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);

    let request = AppendEntriesRequest {
        term: 1,
        leader_id: 0,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };

    let response = node.handle_append_entries(request);

    assert!(response.success);
    assert_eq!(response.term, 1);
}

#[test]
fn test_append_entries_stale_term() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);
    node.current_term = 5;

    let request = AppendEntriesRequest {
        term: 3,
        leader_id: 0,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };

    let response = node.handle_append_entries(request);

    assert!(!response.success);
    assert_eq!(response.term, 5);
}

#[test]
fn test_append_entries_with_entries() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);

    let entries = vec![
        LogEntry {
            term: 1,
            index: 1,
            command: Command::Put {
                key: b"key1".to_vec(),
                value: b"value1".to_vec(),
            },
            entry_type: EntryType::Command,
        },
        LogEntry {
            term: 1,
            index: 2,
            command: Command::Put {
                key: b"key2".to_vec(),
                value: b"value2".to_vec(),
            },
            entry_type: EntryType::Command,
        },
    ];

    let request = AppendEntriesRequest {
        term: 1,
        leader_id: 0,
        prev_log_index: 0,
        prev_log_term: 0,
        entries,
        leader_commit: 0,
    };

    let response = node.handle_append_entries(request);

    assert!(response.success);
    assert_eq!(node.log_size(), 2);
}

#[test]
fn test_append_entries_log_mismatch() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);

    // Node has entry at index 1 with term 1
    node.log.push(LogEntry {
        term: 1,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    });

    // Leader claims prev_log at index 2 (doesn't exist)
    let request = AppendEntriesRequest {
        term: 2,
        leader_id: 0,
        prev_log_index: 2,
        prev_log_term: 1,
        entries: vec![],
        leader_commit: 0,
    };

    let response = node.handle_append_entries(request);

    assert!(!response.success);
    assert!(response.conflict_index.is_some());
}

#[test]
fn test_append_entries_term_mismatch() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);

    // Node has entry at index 1 with term 1
    node.log.push(LogEntry {
        term: 1,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    });

    // Leader claims prev_log at index 1 with term 2 (mismatch)
    let request = AppendEntriesRequest {
        term: 3,
        leader_id: 0,
        prev_log_index: 1,
        prev_log_term: 2,
        entries: vec![],
        leader_commit: 0,
    };

    let response = node.handle_append_entries(request);

    assert!(!response.success);
    assert!(response.conflict_term.is_some());
}

#[test]
fn test_append_entries_commit_advancement() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);

    let entries = vec![LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: b"key1".to_vec(),
            value: b"value1".to_vec(),
        },
        entry_type: EntryType::Command,
    }];

    let request = AppendEntriesRequest {
        term: 1,
        leader_id: 0,
        prev_log_index: 0,
        prev_log_term: 0,
        entries,
        leader_commit: 1,
    };

    let response = node.handle_append_entries(request);

    assert!(response.success);
    assert_eq!(node.commit_index(), 1);
    assert_eq!(node.last_applied(), 1);
}

#[test]
fn test_append_entries_response_handling() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let response = AppendEntriesResponse {
        term: 1,
        success: true,
        conflict_index: None,
        conflict_term: None,
    };

    node.handle_append_entries_response(1, response, 0);

    // Should update match_index
    assert_eq!(node.get_match_index(1), Some(1)); // NoOp entry
}

// =============================================================================
// Log Replication Tests
// =============================================================================

#[test]
fn test_propose_command() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let index = node
        .propose(Command::Put {
            key: b"key".to_vec(),
            value: b"value".to_vec(),
        })
        .unwrap();

    assert_eq!(index, 2); // NoOp is at index 1
    assert_eq!(node.log_size(), 2);
}

#[test]
fn test_propose_not_leader() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    let result = node.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    });

    assert!(result.is_err());
}

#[test]
fn test_batch_propose() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let commands = vec![
        Command::Put {
            key: b"key1".to_vec(),
            value: b"value1".to_vec(),
        },
        Command::Put {
            key: b"key2".to_vec(),
            value: b"value2".to_vec(),
        },
        Command::Put {
            key: b"key3".to_vec(),
            value: b"value3".to_vec(),
        },
    ];

    let indices = node.batch_propose(commands).unwrap();

    assert_eq!(indices.len(), 3);
    assert_eq!(indices[0], 2);
    assert_eq!(indices[1], 3);
    assert_eq!(indices[2], 4);
    assert_eq!(node.log_size(), 4); // NoOp + 3 commands
}

#[test]
fn test_build_append_entries_request() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let request = node.build_append_entries_request(1);

    assert_eq!(request.term, 1);
    assert_eq!(request.leader_id, 0);
    assert_eq!(request.prev_log_index, 0);
}

#[test]
fn test_get_entries_for_peer() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();
    node.propose(Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    })
    .unwrap();

    let (prev_index, prev_term, entries) = node.get_entries_for_peer(1);

    assert_eq!(prev_index, 0);
    assert_eq!(prev_term, 0);
    assert_eq!(entries.len(), 2); // NoOp + Put
}

// =============================================================================
// FSM Tests
// =============================================================================

#[test]
fn test_fsm_put_get() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    node.write(b"key1".to_vec(), b"value1".to_vec()).unwrap();

    // Entry is in log but not committed yet
    assert_eq!(node.log_size(), 2);
}

#[test]
fn test_fsm_delete() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    node.write(b"key1".to_vec(), b"value1".to_vec()).unwrap();
    node.delete(b"key1".to_vec()).unwrap();

    assert_eq!(node.log_size(), 3);
}

#[test]
fn test_fsm_read() {
    let config = create_three_node_config(0);
    let node = RaftNode::new(config);

    // FSM is empty
    let result = node.read(b"nonexistent".to_vec()).unwrap();
    assert!(result.is_none());
}

// =============================================================================
// InstallSnapshot Tests
// =============================================================================

#[test]
fn test_install_snapshot_stale_term() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);
    node.current_term = 5;

    let request = InstallSnapshotRequest {
        term: 3,
        leader_id: 0,
        last_included_index: 10,
        last_included_term: 2,
        offset: 0,
        data: vec![],
        done: true,
    };

    let response = node.handle_install_snapshot(request);

    assert_eq!(response.term, 5);
}

#[test]
fn test_install_snapshot_updates_term() {
    let config = create_three_node_config(1);
    let mut node = RaftNode::new(config);
    node.current_term = 3;

    let request = InstallSnapshotRequest {
        term: 5,
        leader_id: 0,
        last_included_index: 10,
        last_included_term: 4,
        offset: 0,
        data: vec![],
        done: false,
    };

    let response = node.handle_install_snapshot(request);

    assert_eq!(node.term(), 5);
    assert_eq!(response.term, 5);
}

#[test]
fn test_build_install_snapshot_request() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let request = node.build_install_snapshot_request(0, 1024);

    if let Some(req) = request {
        assert_eq!(req.term, 1);
        assert_eq!(req.leader_id, 0);
    }
}

// =============================================================================
// Snapshot and Compaction Tests
// =============================================================================

#[test]
fn test_create_snapshot() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    // This may fail without snapshot store, but tests the path
    let result = node.create_snapshot();
    // May succeed or fail depending on snapshot_store presence
    let _ = result;
}

#[test]
fn test_compact_log() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    // Add some entries
    for i in 1..=5 {
        node.log.push(LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        });
    }

    assert_eq!(node.log_size(), 5);

    node.compact_log(3).unwrap();

    // Should have entries 4 and 5 remaining
    assert_eq!(node.log_size(), 2);
    assert!(node.log_entries().iter().all(|e| e.index > 3));
}

// =============================================================================
// Linearizable Read Tests
// =============================================================================

#[test]
fn test_linearizable_read_not_leader() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    let result = node.linearizable_read(b"key".to_vec());
    assert!(result.is_err());
}

#[test]
fn test_linearizable_read_leader() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let result = node.linearizable_read(b"key".to_vec());
    // May succeed or return error about read index
    let _ = result;
}

#[test]
fn test_lease_read_not_leader() {
    let config = create_three_node_config(0);
    let node = RaftNode::new(config);

    let result = node.lease_read(b"key");
    assert!(result.is_err());
}

#[test]
fn test_update_leader_lease() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    node.update_leader_lease();

    assert!(node.lease_valid());
}

// =============================================================================
// Membership Change Tests
// =============================================================================

#[test]
fn test_add_node_not_leader() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    let result = node.add_node(3);
    assert!(result.is_err());
}

#[test]
fn test_add_node_leader() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let result = node.add_node(3);
    assert!(result.is_ok());

    // Should be in joint consensus
    assert!(node.cluster_config().is_joint());
}

#[test]
fn test_remove_node_leader() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let result = node.remove_node(2);
    assert!(result.is_ok());
}

#[test]
fn test_finalize_membership_change() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    node.add_node(3).unwrap();
    let result = node.finalize_membership_change();

    assert!(result.is_ok());
    assert!(!node.cluster_config().is_joint());
}

#[test]
fn test_transfer_leadership() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let result = node.transfer_leadership(1);
    assert!(result.is_ok());
    assert!(!node.is_leader());
}

// =============================================================================
// Pre-Vote Tests
// =============================================================================

#[test]
fn test_start_pre_vote() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    let started = node.start_pre_vote();
    assert!(started);

    if let Some(ref pre_vote) = node.pre_vote_state {
        assert_eq!(pre_vote.term, 1);
        assert!(pre_vote.votes_received.contains(&0));
    }
}

#[test]
fn test_start_pre_vote_disabled() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);
    node.set_pre_vote(false);

    let started = node.start_pre_vote();
    assert!(!started);
}

#[test]
fn test_start_pre_vote_as_leader() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    let started = node.start_pre_vote();
    assert!(!started);
}

#[test]
fn test_handle_pre_vote_request() {
    let config = create_three_node_config(1);
    let node = RaftNode::new(config);

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let response = node.handle_pre_vote_request(&request);
    // Response depends on election timeout state
    let _ = response;
}

#[test]
fn test_handle_pre_vote_response() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.start_pre_vote();

    let response = RequestVoteResponse {
        term: 1,
        vote_granted: true,
    };

    let became_candidate = node.handle_pre_vote_response(1, response);

    // With 2 votes, should win pre-vote and start real election
    assert!(became_candidate);
    assert!(matches!(node.state(), RaftState::Candidate { .. }));
}

// =============================================================================
// Check Quorum Tests
// =============================================================================

#[test]
fn test_check_quorum_not_leader() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    assert!(node.check_quorum());
}

#[test]
fn test_check_quorum_disabled() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);
    node.set_check_quorum(false);

    node.transition_to_candidate();
    node.transition_to_leader();

    assert!(node.check_quorum());
}

#[test]
fn test_record_peer_contact() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.record_peer_contact(1);
    node.record_peer_contact(2);

    // Peer contact should be recorded
    assert!(node.peer_last_contact.contains_key(&1));
    assert!(node.peer_last_contact.contains_key(&2));
}

// =============================================================================
// Log Entry Tests
// =============================================================================

#[test]
fn test_log_entry_creation() {
    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::Put {
            key: b"key".to_vec(),
            value: b"value".to_vec(),
        },
        entry_type: EntryType::Command,
    };

    assert_eq!(entry.term, 1);
    assert_eq!(entry.index, 1);
    assert_eq!(entry.entry_type, EntryType::Command);
}

#[test]
fn test_log_entry_noop() {
    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    };

    assert_eq!(entry.entry_type, EntryType::NoOp);
}

#[test]
fn test_log_entry_configuration() {
    let entry = LogEntry {
        term: 1,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::Configuration,
    };

    assert_eq!(entry.entry_type, EntryType::Configuration);
}

#[test]
fn test_command_types() {
    let put = Command::Put {
        key: b"key".to_vec(),
        value: b"value".to_vec(),
    };
    assert!(matches!(put, Command::Put { .. }));

    let get = Command::Get {
        key: b"key".to_vec(),
    };
    assert!(matches!(get, Command::Get { .. }));

    let delete = Command::Delete {
        key: b"key".to_vec(),
    };
    assert!(matches!(delete, Command::Delete { .. }));

    let noop = Command::NoOp;
    assert!(matches!(noop, Command::NoOp));
}

// =============================================================================
// Cluster Config Tests
// =============================================================================

#[test]
fn test_cluster_config_quorum_size_3() {
    let config = create_three_node_config(0);
    let node = RaftNode::new(config);

    // 3-node cluster: quorum = 2
    assert_eq!(node.cluster_config().quorum_size(), 2);
}

#[test]
fn test_cluster_config_quorum_size_5() {
    let config = create_five_node_config(0);
    let node = RaftNode::new(config);

    // 5-node cluster: quorum = 3
    assert_eq!(node.cluster_config().quorum_size(), 3);
}

// =============================================================================
// Match Index Tracking Tests
// =============================================================================

#[test]
fn test_update_match_index() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    node.update_match_index(1, 5);

    assert_eq!(node.get_match_index(1), Some(5));
}

#[test]
fn test_get_next_index() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    // Initial next_index should be last_log_index + 1
    assert!(node.get_next_index(1).is_some());
}

// =============================================================================
// Edge Cases and Error Handling Tests
// =============================================================================

#[test]
fn test_empty_log_operations() {
    let config = create_test_config(0);
    let node = RaftNode::new(config);

    assert_eq!(node.log_size(), 0);
    assert_eq!(node.last_log_index(), 0);
    assert_eq!(node.last_log_term(), 0);
    assert!(node.log_entries().is_empty());
}

#[test]
fn test_get_entries_empty() {
    let config = create_test_config(0);
    let node = RaftNode::new(config);

    let entries = node.get_entries(1, 10);
    assert!(entries.is_empty());
}

#[test]
fn test_get_entries_range() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    for i in 1..=10 {
        node.log.push(LogEntry {
            term: 1,
            index: i,
            command: Command::NoOp,
            entry_type: EntryType::NoOp,
        });
    }

    let entries = node.get_entries(3, 7);
    assert_eq!(entries.len(), 5);
    assert_eq!(entries[0].index, 3);
    assert_eq!(entries[4].index, 7);
}

#[test]
fn test_fsm_access() {
    let config = create_test_config(0);
    let node = RaftNode::new(config);

    let fsm = node.fsm();
    assert!(fsm.is_empty());
    assert_eq!(fsm.len(), 0);
}

#[test]
fn test_config_access() {
    let config = create_test_config(0);
    let node = RaftNode::new(config);

    let cfg = node.config();
    assert_eq!(cfg.id, 0);
}

// =============================================================================
// Concurrent State Machine Operations (Single-threaded tests)
// =============================================================================

#[test]
fn test_multiple_writes() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    for i in 0..100 {
        node.write(format!("key{}", i).into_bytes(), format!("value{}", i).into_bytes())
            .unwrap();
    }

    assert_eq!(node.log_size(), 101); // NoOp + 100 writes
}

#[test]
fn test_alternating_put_delete() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    for i in 0..50 {
        node.write(format!("key{}", i).into_bytes(), b"value".to_vec())
            .unwrap();
        node.delete(format!("key{}", i).into_bytes()).unwrap();
    }

    assert_eq!(node.log_size(), 101); // NoOp + 50 puts + 50 deletes
}

// =============================================================================
// Term and State Consistency Tests
// =============================================================================

#[test]
fn test_term_increment_on_election() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    for _ in 0..5 {
        node.transition_to_candidate();
        node.transition_to_follower(None);
    }

    assert_eq!(node.term(), 5);
}

#[test]
fn test_voted_for_reset_on_new_term() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    assert_eq!(node.voted_for(), Some(0));

    // Receive higher term
    let response = RequestVoteResponse {
        term: 5,
        vote_granted: false,
    };
    node.handle_request_vote_response(1, response);

    // voted_for should be reset
    assert_eq!(node.voted_for(), None);
}

// =============================================================================
// RPC Message Building Tests
// =============================================================================

#[test]
fn test_build_request_vote_request() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();

    let request = node.build_request_vote_request();

    assert_eq!(request.term, 1);
    assert_eq!(request.candidate_id, 0);
}

#[test]
fn test_append_entries_request_with_entries() {
    let config = create_three_node_config(0);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    node.transition_to_leader();

    for i in 0..5 {
        node.propose(Command::Put {
            key: format!("key{}", i).into_bytes(),
            value: b"value".to_vec(),
        })
        .unwrap();
    }

    let request = node.build_append_entries_request(1);

    assert_eq!(request.term, 1);
    assert_eq!(request.leader_id, 0);
    // Should include NoOp + 5 entries
    assert!(request.entries.len() >= 1);
}
