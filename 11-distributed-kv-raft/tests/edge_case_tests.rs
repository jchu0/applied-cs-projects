//! Edge case and integration tests for the Raft implementation.
//!
//! This file contains additional tests for boundary conditions, error handling,
//! and complex scenarios that may not be covered by the main test suites.

use distributed_kv_raft::{
    config::{ClusterConfig, PeerConfig, RaftConfig, RaftConfigBuilder},
    error::Error,
    metrics::{Counter, Gauge, Histogram, RaftMetrics},
    node::{ApplyResult, Command, EntryType, LogEntry, RaftNode, RaftState},
    rpc::{
        AppendEntriesRequest, AppendEntriesResponse, ClientRequest, ClientResponse,
        InstallSnapshotRequest, InstallSnapshotResponse, RequestVoteRequest, RequestVoteResponse,
    },
    storage::{KeyValueFSM, MemoryStorage, Snapshot, SnapshotStore, Storage, WriteAheadLog},
    transport::{MemoryNetwork, MemoryTransport, PeerState, PeerTracker},
    LogIndex, NodeId, Term,
};

use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tempfile::TempDir;

// =============================================================================
// Helper Functions
// =============================================================================

fn create_config(id: NodeId, peers: &[NodeId]) -> RaftConfig {
    let mut builder = RaftConfigBuilder::default()
        .id(id)
        .listen_addr(format!("127.0.0.1:{}", 7000 + id))
        .election_timeout(Duration::from_millis(150), Duration::from_millis(300))
        .heartbeat_interval(Duration::from_millis(50));

    for &peer_id in peers {
        builder = builder.peer(peer_id, format!("127.0.0.1:{}", 7000 + peer_id));
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

// =============================================================================
// Boundary Condition Tests
// =============================================================================

#[test]
fn test_empty_log_last_index() {
    let config = create_config(0, &[]);
    let node = RaftNode::new(config);

    assert_eq!(node.last_log_index(), 0);
    assert_eq!(node.last_log_term(), 0);
}

#[test]
fn test_single_entry_log() {
    let config = create_config(0, &[1]);
    let mut node = RaftNode::new(config);
    node.current_term = 1;
    node.transition_to_leader();

    // NoOp entry is added
    assert_eq!(node.last_log_index(), 1);
    assert_eq!(node.last_log_term(), 1);
}

#[test]
fn test_zero_term_handling() {
    let config = create_config(0, &[1]);
    let node = RaftNode::new(config);

    assert_eq!(node.term(), 0);
}

#[test]
fn test_maximum_term_value() {
    let config = create_config(0, &[1]);
    let mut node = RaftNode::new(config);

    // Test with very large term values
    node.current_term = u64::MAX - 1;

    let request = AppendEntriesRequest {
        term: u64::MAX,
        leader_id: 1,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };

    let response = node.handle_append_entries(request);
    assert!(response.success);
    assert_eq!(node.term(), u64::MAX);
}

#[test]
fn test_single_node_cluster_quorum() {
    let config = create_config(0, &[]);
    assert_eq!(config.quorum_size(), 1);
}

#[test]
fn test_even_cluster_size_quorum() {
    // 4 node cluster
    let config = create_config(0, &[1, 2, 3]);
    assert_eq!(config.quorum_size(), 3); // (4/2 + 1)
}

#[test]
fn test_large_cluster_quorum() {
    // 9 node cluster
    let peers: Vec<NodeId> = (1..9).collect();
    let config = create_config(0, &peers);
    assert_eq!(config.quorum_size(), 5); // (9/2 + 1)
}

// =============================================================================
// Log Consistency Tests
// =============================================================================

#[test]
fn test_log_term_consistency() {
    let config = create_config(0, &[1]);
    let mut node = RaftNode::new(config);
    node.current_term = 5;
    node.transition_to_leader();

    // All new entries should have the current term
    node.propose(Command::Put {
        key: b"k1".to_vec(),
        value: b"v1".to_vec(),
    }).unwrap();

    assert!(node.log.iter().all(|e| e.term <= 5));
    assert_eq!(node.log.last().unwrap().term, 5);
}

#[test]
fn test_log_index_continuity() {
    let config = create_config(0, &[1]);
    let mut node = RaftNode::new(config);
    node.current_term = 1;
    node.transition_to_leader();

    for _ in 0..10 {
        node.propose(Command::Put {
            key: b"k".to_vec(),
            value: b"v".to_vec(),
        }).unwrap();
    }

    // Verify indices are continuous
    for (i, entry) in node.log.iter().enumerate() {
        assert_eq!(entry.index, (i + 1) as LogIndex);
    }
}

#[test]
fn test_append_entries_overwrites_conflicting() {
    let config = create_config(0, &[1]);
    let mut node = RaftNode::new(config);
    node.current_term = 3;

    // Add some entries with term 2
    node.log.push(LogEntry {
        term: 2,
        index: 1,
        command: Command::Put { key: b"k1".to_vec(), value: b"old".to_vec() },
        entry_type: EntryType::Command,
    });
    node.log.push(LogEntry {
        term: 2,
        index: 2,
        command: Command::Put { key: b"k2".to_vec(), value: b"old".to_vec() },
        entry_type: EntryType::Command,
    });

    // New entry with different term should replace from that index
    let new_entry = LogEntry {
        term: 3,
        index: 2,
        command: Command::Put { key: b"k2".to_vec(), value: b"new".to_vec() },
        entry_type: EntryType::Command,
    };

    let request = AppendEntriesRequest {
        term: 3,
        leader_id: 1,
        prev_log_index: 1,
        prev_log_term: 2,
        entries: vec![new_entry],
        leader_commit: 0,
    };

    let response = node.handle_append_entries(request);
    assert!(response.success);
    assert_eq!(node.log.len(), 2);
    assert_eq!(node.log[1].term, 3);
}

// =============================================================================
// State Machine Tests
// =============================================================================

#[test]
fn test_fsm_empty_key() {
    let mut fsm = KeyValueFSM::new();

    fsm.apply(&LogEntry {
        term: 1,
        index: 1,
        command: Command::Put { key: vec![], value: b"value".to_vec() },
        entry_type: EntryType::Command,
    });

    assert_eq!(fsm.get(&[]), Some(b"value".to_vec()));
}

#[test]
fn test_fsm_empty_value() {
    let mut fsm = KeyValueFSM::new();

    fsm.apply(&LogEntry {
        term: 1,
        index: 1,
        command: Command::Put { key: b"key".to_vec(), value: vec![] },
        entry_type: EntryType::Command,
    });

    assert_eq!(fsm.get(b"key"), Some(vec![]));
}

#[test]
fn test_fsm_unicode_keys() {
    let mut fsm = KeyValueFSM::new();
    let key = "こんにちは".as_bytes().to_vec();
    let value = "世界".as_bytes().to_vec();

    fsm.apply(&LogEntry {
        term: 1,
        index: 1,
        command: Command::Put { key: key.clone(), value: value.clone() },
        entry_type: EntryType::Command,
    });

    assert_eq!(fsm.get(&key), Some(value));
}

#[test]
fn test_fsm_null_bytes_in_key() {
    let mut fsm = KeyValueFSM::new();
    let key = vec![0, 1, 0, 2, 0];
    let value = b"value".to_vec();

    fsm.apply(&LogEntry {
        term: 1,
        index: 1,
        command: Command::Put { key: key.clone(), value: value.clone() },
        entry_type: EntryType::Command,
    });

    assert_eq!(fsm.get(&key), Some(value));
}

#[test]
fn test_fsm_snapshot_empty() {
    let fsm = KeyValueFSM::new();
    let snapshot = fsm.snapshot();

    assert_eq!(snapshot.last_included_index, 0);
    assert_eq!(snapshot.last_included_term, 0);
}

#[test]
fn test_fsm_restore_empty_snapshot() {
    let mut fsm = KeyValueFSM::new();

    // Add some data
    fsm.apply(&create_log_entry(1, 1, b"key", b"value"));
    assert_eq!(fsm.len(), 1);

    // Create a valid empty snapshot (from a new FSM)
    let empty_fsm = KeyValueFSM::new();
    let empty_snapshot = empty_fsm.snapshot();

    fsm.restore(&empty_snapshot).unwrap();
    assert_eq!(fsm.len(), 0);
}

// =============================================================================
// Election Edge Cases
// =============================================================================

#[test]
fn test_election_tie_breaking() {
    // When two candidates have the same term, the one with longer log wins
    let config1 = create_config(0, &[1, 2]);
    let config2 = create_config(1, &[0, 2]);

    let mut node1 = RaftNode::new(config1);
    let mut node2 = RaftNode::new(config2);

    // Node 1 has longer log
    node1.log.push(create_log_entry(1, 1, b"k1", b"v1"));
    node1.log.push(create_log_entry(1, 2, b"k2", b"v2"));

    // Node 2 has shorter log
    node2.log.push(create_log_entry(1, 1, b"k1", b"v1"));

    // Node 2 becomes candidate
    node2.transition_to_candidate();

    // Node 1 receives vote request - should reject (node1 has longer log)
    let request = node2.build_request_vote_request();
    let response = node1.handle_request_vote(request);

    assert!(!response.vote_granted);
}

#[test]
fn test_election_higher_term_log_wins() {
    let config = create_config(0, &[1]);
    let mut node = RaftNode::new(config);
    node.current_term = 3;

    // Node has entry with term 2
    node.log.push(LogEntry {
        term: 2,
        index: 1,
        command: Command::NoOp,
        entry_type: EntryType::NoOp,
    });

    // Candidate has entry with term 3 - should win
    let request = RequestVoteRequest {
        term: 4,
        candidate_id: 1,
        last_log_index: 1,
        last_log_term: 3,
    };

    let response = node.handle_request_vote(request);
    assert!(response.vote_granted);
}

#[test]
fn test_candidate_loses_to_leader() {
    let config = create_config(0, &[1]);
    let mut node = RaftNode::new(config);

    node.transition_to_candidate();
    assert!(matches!(node.state, RaftState::Candidate { .. }));

    // Receive AppendEntries from a leader in same term
    let request = AppendEntriesRequest {
        term: 1,
        leader_id: 1,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };

    node.handle_append_entries(request);

    // Should step down to follower
    assert!(matches!(node.state, RaftState::Follower { leader_id: Some(1) }));
}

// =============================================================================
// Network Partition Simulation
// =============================================================================

#[test]
fn test_partitioned_node_increments_term() {
    let config = create_config(0, &[1, 2]);
    let mut node = RaftNode::new(config);

    // Simulate election timeouts without hearing from leader
    for _ in 0..5 {
        node.transition_to_candidate();
        // No votes received, election times out
    }

    // Term should have increased with each failed election
    assert!(node.term() >= 5);
}

#[test]
fn test_stale_leader_detection() {
    let config = create_config(0, &[1, 2]);
    let mut node = RaftNode::new(config);

    node.current_term = 5;
    node.transition_to_leader();

    // Receive response with higher term
    let response = AppendEntriesResponse {
        term: 10,
        success: false,
        conflict_index: None,
        conflict_term: None,
    };

    node.handle_append_entries_response(1, response, 0);

    // Should step down
    assert!(!node.is_leader());
    assert_eq!(node.term(), 10);
}

// =============================================================================
// Storage Edge Cases
// =============================================================================

#[test]
fn test_memory_storage_empty_operations() {
    let mut storage = MemoryStorage::new();

    // Operations on empty storage should not panic
    assert!(storage.last_entry().is_none());
    assert_eq!(storage.get_entries(1, 10).unwrap().len(), 0);
}

#[test]
fn test_memory_storage_truncate_at_zero() {
    let mut storage = MemoryStorage::new();
    storage.append(&[create_log_entry(1, 1, b"k", b"v")]).unwrap();

    // Truncate at index 0 should remove everything starting from 1
    storage.truncate(1).unwrap();
    assert!(storage.last_entry().is_none());
}

#[test]
fn test_memory_storage_compact_beyond_log() {
    let mut storage = MemoryStorage::new();
    storage.append(&[
        create_log_entry(1, 1, b"k1", b"v1"),
        create_log_entry(1, 2, b"k2", b"v2"),
    ]).unwrap();

    // Compact beyond current log length
    storage.compact(100).unwrap();

    // All entries should be removed
    assert_eq!(storage.get_entries(1, 100).unwrap().len(), 0);
}

#[test]
fn test_write_ahead_log_empty() {
    let temp_dir = TempDir::new().unwrap();
    let wal = WriteAheadLog::new(temp_dir.path()).unwrap();

    assert!(wal.is_empty());
    assert_eq!(wal.len(), 0);
}

#[test]
fn test_snapshot_store_no_snapshot() {
    let temp_dir = TempDir::new().unwrap();
    let store = SnapshotStore::new(temp_dir.path()).unwrap();

    assert!(store.get_current().is_none());
}

#[test]
fn test_snapshot_store_multiple_saves() {
    let temp_dir = TempDir::new().unwrap();
    let mut store = SnapshotStore::new(temp_dir.path()).unwrap();

    // Save multiple snapshots
    for i in 1..=5 {
        let snapshot = Snapshot {
            last_included_index: i * 100,
            last_included_term: i as Term,
            data: format!("data_{}", i).into_bytes(),
        };
        store.save(&snapshot).unwrap();
    }

    // Should have the latest
    let current = store.get_current().unwrap();
    assert_eq!(current.last_included_index, 500);
}

// =============================================================================
// Metrics Tests
// =============================================================================

#[test]
fn test_counter_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let counter = Arc::new(Counter::default());
    let mut handles = vec![];

    for _ in 0..10 {
        let c = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                c.inc();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(counter.get(), 1000);
}

#[test]
fn test_gauge_thread_safety() {
    use std::sync::Arc;
    use std::thread;

    let gauge = Arc::new(Gauge::default());
    gauge.set(1000);

    let mut handles = vec![];

    for _ in 0..10 {
        let g = Arc::clone(&gauge);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                g.inc();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    assert_eq!(gauge.get(), 2000);
}

#[test]
fn test_histogram_with_zero_duration() {
    let histogram = Histogram::new();

    histogram.observe(Duration::from_nanos(0));

    assert_eq!(histogram.mean(), 0.0);
}

#[test]
fn test_histogram_with_very_long_duration() {
    let histogram = Histogram::new();

    histogram.observe(Duration::from_secs(100));

    assert!(histogram.percentile(0.99) > 0);
}

// =============================================================================
// Transport Tests
// =============================================================================

#[test]
fn test_peer_tracker_multiple_peers() {
    let mut tracker = PeerTracker::new(3);

    // Track multiple peers
    for peer_id in 0..5 {
        tracker.record_success(peer_id);
    }

    assert_eq!(tracker.connected_count(), 5);
}

#[test]
fn test_peer_tracker_mixed_states() {
    let mut tracker = PeerTracker::new(3);

    tracker.record_success(0);
    tracker.record_success(1);
    tracker.record_failure(2);
    tracker.record_failure(2);
    tracker.record_failure(2); // Disconnected

    assert_eq!(tracker.connected_count(), 2);
    assert_eq!(tracker.get_state(0), PeerState::Connected);
    assert_eq!(tracker.get_state(1), PeerState::Connected);
    assert_eq!(tracker.get_state(2), PeerState::Disconnected);
}

#[test]
fn test_memory_network_isolation() {
    let network = MemoryNetwork::new(&[0, 1, 2, 3, 4]);

    // Create partition: [0, 1] vs [2, 3, 4]
    network.create_partition(&[0, 1], &[2, 3, 4]);

    let t0 = network.get_transport(0).unwrap();
    let t2 = network.get_transport(2).unwrap();

    // Cross-partition
    assert!(t0.is_partitioned(2));
    assert!(t0.is_partitioned(3));
    assert!(t2.is_partitioned(0));
    assert!(t2.is_partitioned(1));

    // Same partition
    assert!(!t0.is_partitioned(1));
    assert!(!t2.is_partitioned(3));
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_error_display() {
    let errors = vec![
        Error::NotLeader(Some(1)),
        Error::NotLeader(None),
        Error::QuorumNotReached,
        Error::ClusterUnavailable,
        Error::Timeout,
        Error::Storage("test".to_string()),
        Error::Serialization("test".to_string()),
        Error::Network("test".to_string()),
        Error::Config("test".to_string()),
        Error::Internal("test".to_string()),
    ];

    for error in errors {
        let msg = format!("{}", error);
        assert!(!msg.is_empty());
    }
}

#[test]
fn test_error_debug() {
    let error = Error::NotLeader(Some(1));
    let debug_str = format!("{:?}", error);
    assert!(debug_str.contains("NotLeader"));
}

// =============================================================================
// Configuration Validation Tests
// =============================================================================

#[test]
fn test_config_cluster_nodes_includes_self() {
    let config = create_config(0, &[1, 2]);
    let nodes = config.cluster_nodes();

    assert!(nodes.contains(&0));
    assert!(nodes.contains(&1));
    assert!(nodes.contains(&2));
}

#[test]
fn test_config_random_timeout_in_range() {
    let config = RaftConfigBuilder::default()
        .election_timeout(Duration::from_millis(100), Duration::from_millis(200))
        .build();

    for _ in 0..100 {
        let timeout = config.random_election_timeout();
        assert!(timeout >= Duration::from_millis(100));
        assert!(timeout <= Duration::from_millis(200));
    }
}

// =============================================================================
// Cluster Configuration Tests
// =============================================================================

#[test]
fn test_cluster_config_joint_consensus() {
    let old: HashSet<NodeId> = vec![0, 1, 2].into_iter().collect();
    let new: HashSet<NodeId> = vec![0, 1, 2, 3, 4].into_iter().collect();

    let config = ClusterConfig {
        current: old,
        next: Some(new),
    };

    assert!(config.is_joint());
    // Need majority from both: max(2, 3) = 3
    assert_eq!(config.quorum_size(), 3);
}

#[test]
fn test_cluster_config_shrinking() {
    let old: HashSet<NodeId> = vec![0, 1, 2, 3, 4].into_iter().collect();
    let new: HashSet<NodeId> = vec![0, 1, 2].into_iter().collect();

    let config = ClusterConfig {
        current: old,
        next: Some(new),
    };

    // Need majority from both: max(3, 2) = 3
    assert_eq!(config.quorum_size(), 3);
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_log_entry_serialization() {
    let entry = create_log_entry(5, 100, b"key", b"value");

    let serialized = bincode::serialize(&entry).unwrap();
    let deserialized: LogEntry = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.term, entry.term);
    assert_eq!(deserialized.index, entry.index);
}

#[test]
fn test_snapshot_serialization() {
    let snapshot = Snapshot {
        last_included_index: 1000,
        last_included_term: 50,
        data: b"test snapshot data".to_vec(),
    };

    let serialized = bincode::serialize(&snapshot).unwrap();
    let deserialized: Snapshot = bincode::deserialize(&serialized).unwrap();

    assert_eq!(deserialized.last_included_index, snapshot.last_included_index);
    assert_eq!(deserialized.last_included_term, snapshot.last_included_term);
    assert_eq!(deserialized.data, snapshot.data);
}

#[test]
fn test_command_serialization() {
    let commands = vec![
        Command::Put { key: b"k".to_vec(), value: b"v".to_vec() },
        Command::Delete { key: b"k".to_vec() },
        Command::Get { key: b"k".to_vec() },
        Command::NoOp,
    ];

    for cmd in commands {
        let serialized = bincode::serialize(&cmd).unwrap();
        let deserialized: Command = bincode::deserialize(&serialized).unwrap();
        // Just verify it doesn't panic
        let _ = format!("{:?}", deserialized);
    }
}

// =============================================================================
// Concurrent Access Tests
// =============================================================================

#[test]
fn test_multiple_proposals() {
    let config = create_config(0, &[1]);
    let mut node = RaftNode::new(config);
    node.current_term = 1;
    node.transition_to_leader();

    // Rapid proposals
    for i in 0..100 {
        node.propose(Command::Put {
            key: format!("key{}", i).into_bytes(),
            value: format!("value{}", i).into_bytes(),
        }).unwrap();
    }

    // All should be in log (NoOp + 100 entries)
    assert_eq!(node.log.len(), 101);
}

#[test]
fn test_fsm_many_operations() {
    let mut fsm = KeyValueFSM::new();

    // Many puts
    for i in 0..1000 {
        fsm.apply(&create_log_entry(1, i + 1, format!("key{}", i).as_bytes(), format!("value{}", i).as_bytes()));
    }

    assert_eq!(fsm.len(), 1000);

    // Many deletes
    for i in 0..500 {
        fsm.apply(&LogEntry {
            term: 1,
            index: 1001 + i,
            command: Command::Delete { key: format!("key{}", i).into_bytes() },
            entry_type: EntryType::Command,
        });
    }

    assert_eq!(fsm.len(), 500);
}

// =============================================================================
// Recovery Tests
// =============================================================================

#[test]
fn test_wal_persistence() {
    let temp_dir = TempDir::new().unwrap();

    // Write entries
    {
        let mut wal = WriteAheadLog::new(temp_dir.path()).unwrap();
        for i in 1..=100 {
            wal.append(&[create_log_entry(1, i, format!("k{}", i).as_bytes(), b"v")]).unwrap();
        }
    }

    // Read back
    {
        let wal = WriteAheadLog::new(temp_dir.path()).unwrap();
        assert_eq!(wal.len(), 100);
    }
}

#[test]
fn test_snapshot_persistence() {
    let temp_dir = TempDir::new().unwrap();

    // Save snapshot
    {
        let mut store = SnapshotStore::new(temp_dir.path()).unwrap();
        let snapshot = Snapshot {
            last_included_index: 1000,
            last_included_term: 10,
            data: b"persistent data".to_vec(),
        };
        store.save(&snapshot).unwrap();
    }

    // Load back
    {
        let store = SnapshotStore::new(temp_dir.path()).unwrap();
        let snapshot = store.get_current().unwrap();
        assert_eq!(snapshot.last_included_index, 1000);
        assert_eq!(snapshot.data, b"persistent data".to_vec());
    }
}
