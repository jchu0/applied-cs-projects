//! Comprehensive transport layer tests.

use distributed_kv_raft::{
    AppendEntriesRequest, LogEntry, MemoryNetwork, MemoryTransport, PeerState, PeerTracker,
    RaftMessage, RequestVoteRequest, Transport,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

// =============================================================================
// MemoryTransport Tests
// =============================================================================

#[test]
fn test_memory_transport_new() {
    let transport = MemoryTransport::new(0);
    assert!(!transport.is_partitioned(1));
}

#[test]
fn test_memory_transport_register_peer() {
    let transport = MemoryTransport::new(0);
    let (tx, _rx) = mpsc::channel(10);

    transport.register_peer(1, tx);

    let peers = transport.peers();
    assert!(peers.read().contains_key(&1));
}

#[test]
fn test_memory_transport_unregister_peer() {
    let transport = MemoryTransport::new(0);
    let (tx, _rx) = mpsc::channel(10);

    transport.register_peer(1, tx);
    transport.unregister_peer(1);

    let peers = transport.peers();
    assert!(!peers.read().contains_key(&1));
}

#[test]
fn test_memory_transport_partition() {
    let transport = MemoryTransport::new(0);

    assert!(!transport.is_partitioned(1));

    transport.set_partition(1, true);
    assert!(transport.is_partitioned(1));

    transport.set_partition(1, false);
    assert!(!transport.is_partitioned(1));
}

#[test]
fn test_memory_transport_delay() {
    let transport = MemoryTransport::new(0);

    transport.set_delay(Some(Duration::from_millis(100)));
    // Delay is set but we can't easily verify without async

    transport.set_delay(None);
}

// =============================================================================
// MemoryNetwork Tests
// =============================================================================

#[test]
fn test_memory_network_creation() {
    let network = MemoryNetwork::new(&[0, 1, 2]);

    assert!(network.get_transport(0).is_some());
    assert!(network.get_transport(1).is_some());
    assert!(network.get_transport(2).is_some());
    assert!(network.get_transport(3).is_none());
}

#[test]
fn test_memory_network_large_cluster() {
    let node_ids: Vec<u64> = (0..10).collect();
    let network = MemoryNetwork::new(&node_ids);

    for id in 0..10 {
        assert!(network.get_transport(id).is_some());
    }
}

#[test]
fn test_memory_network_take_receiver() {
    let mut network = MemoryNetwork::new(&[0, 1, 2]);

    let receiver = network.take_receiver(0);
    assert!(receiver.is_some());

    // Taking again should return None
    let receiver2 = network.take_receiver(0);
    assert!(receiver2.is_none());
}

#[test]
fn test_memory_network_partition() {
    let network = MemoryNetwork::new(&[0, 1, 2, 3, 4]);

    network.create_partition(&[0, 1], &[2, 3, 4]);

    let transport0 = network.get_transport(0).unwrap();
    assert!(transport0.is_partitioned(2));
    assert!(transport0.is_partitioned(3));
    assert!(transport0.is_partitioned(4));
    assert!(!transport0.is_partitioned(1));

    let transport2 = network.get_transport(2).unwrap();
    assert!(transport2.is_partitioned(0));
    assert!(transport2.is_partitioned(1));
    assert!(!transport2.is_partitioned(3));
    assert!(!transport2.is_partitioned(4));
}

#[test]
fn test_memory_network_heal_partition() {
    let network = MemoryNetwork::new(&[0, 1, 2]);

    network.create_partition(&[0], &[1, 2]);
    network.heal_all_partitions();

    let transport0 = network.get_transport(0).unwrap();
    assert!(!transport0.is_partitioned(1));
    assert!(!transport0.is_partitioned(2));
}

#[test]
fn test_memory_network_set_delay() {
    let network = MemoryNetwork::new(&[0, 1, 2]);

    network.set_delay(Some(Duration::from_millis(50)));
    // Delay is applied to all transports

    network.set_delay(None);
}

// =============================================================================
// PeerTracker Tests
// =============================================================================

#[test]
fn test_peer_tracker_new() {
    let tracker = PeerTracker::new(3);
    assert_eq!(tracker.connected_count(), 0);
}

#[test]
fn test_peer_tracker_success() {
    let mut tracker = PeerTracker::new(3);

    tracker.record_success(1);

    assert_eq!(tracker.get_state(1), PeerState::Connected);
    assert_eq!(tracker.connected_count(), 1);
}

#[test]
fn test_peer_tracker_multiple_success() {
    let mut tracker = PeerTracker::new(3);

    tracker.record_success(1);
    tracker.record_success(2);
    tracker.record_success(3);

    assert_eq!(tracker.connected_count(), 3);
}

#[test]
fn test_peer_tracker_failure() {
    let mut tracker = PeerTracker::new(3);

    tracker.record_success(1);
    tracker.record_failure(1);

    assert_eq!(tracker.get_state(1), PeerState::Connected);
}

#[test]
fn test_peer_tracker_failure_threshold() {
    let mut tracker = PeerTracker::new(3);

    tracker.record_success(1);
    tracker.record_failure(1);
    tracker.record_failure(1);
    tracker.record_failure(1); // 3rd failure

    assert_eq!(tracker.get_state(1), PeerState::Disconnected);
}

#[test]
fn test_peer_tracker_recovery() {
    let mut tracker = PeerTracker::new(3);

    // Fail the peer
    tracker.record_failure(1);
    tracker.record_failure(1);
    tracker.record_failure(1);
    assert_eq!(tracker.get_state(1), PeerState::Disconnected);

    // Recover
    tracker.record_success(1);
    assert_eq!(tracker.get_state(1), PeerState::Connected);
}

#[test]
fn test_peer_tracker_unknown_peer() {
    let tracker = PeerTracker::new(3);

    assert_eq!(tracker.get_state(999), PeerState::Connecting);
}

#[test]
fn test_peer_tracker_all_states() {
    let mut tracker = PeerTracker::new(3);

    tracker.record_success(1);
    tracker.record_failure(2);
    tracker.record_failure(2);
    tracker.record_failure(2);

    let states = tracker.all_states();
    assert_eq!(states.get(&1), Some(&PeerState::Connected));
    assert_eq!(states.get(&2), Some(&PeerState::Disconnected));
}

// =============================================================================
// Async Transport Tests
// =============================================================================

#[tokio::test]
async fn test_memory_transport_send_request_vote() {
    let transport1 = Arc::new(MemoryTransport::new(0));
    let (tx, mut rx) = mpsc::channel(10);

    transport1.register_peer(1, tx);

    // Spawn handler for peer 1
    let handle = tokio::spawn(async move {
        if let Some(RaftMessage::RequestVote {
            request,
            response_tx,
        }) = rx.recv().await
        {
            let response = distributed_kv_raft::RequestVoteResponse {
                term: request.term,
                vote_granted: true,
            };
            let _ = response_tx.send(response);
        }
    });

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let response = transport1.send_request_vote(1, request).await.unwrap();

    assert!(response.vote_granted);
    assert_eq!(response.term, 1);

    handle.await.unwrap();
}

#[tokio::test]
async fn test_memory_transport_send_append_entries() {
    let transport1 = Arc::new(MemoryTransport::new(0));
    let (tx, mut rx) = mpsc::channel(10);

    transport1.register_peer(1, tx);

    let handle = tokio::spawn(async move {
        if let Some(RaftMessage::AppendEntries {
            request,
            response_tx,
        }) = rx.recv().await
        {
            let response = distributed_kv_raft::AppendEntriesResponse {
                term: request.term,
                success: true,
                conflict_index: None,
                conflict_term: None,
            };
            let _ = response_tx.send(response);
        }
    });

    let request = AppendEntriesRequest {
        term: 1,
        leader_id: 0,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![],
        leader_commit: 0,
    };

    let response = transport1.send_append_entries(1, request).await.unwrap();

    assert!(response.success);
    assert_eq!(response.term, 1);

    handle.await.unwrap();
}

#[tokio::test]
async fn test_memory_transport_partitioned_request() {
    let transport1 = Arc::new(MemoryTransport::new(0));
    let (tx, _rx) = mpsc::channel(10);

    transport1.register_peer(1, tx);
    transport1.set_partition(1, true);

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let result = transport1.send_request_vote(1, request).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_memory_transport_peer_not_found() {
    let transport1 = Arc::new(MemoryTransport::new(0));

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    // Peer 1 not registered
    let result = transport1.send_request_vote(1, request).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_memory_network_full_communication() {
    let mut network = MemoryNetwork::new(&[0, 1, 2]);

    let transport0 = network.get_transport(0).unwrap();
    let mut rx1 = network.take_receiver(1).unwrap();

    // Spawn handler for node 1
    let handle = tokio::spawn(async move {
        if let Some(RaftMessage::RequestVote {
            request,
            response_tx,
        }) = rx1.recv().await
        {
            let response = distributed_kv_raft::RequestVoteResponse {
                term: request.term,
                vote_granted: true,
            };
            let _ = response_tx.send(response);
        }
    });

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let response = transport0.send_request_vote(1, request).await.unwrap();

    assert!(response.vote_granted);

    handle.await.unwrap();
}

#[tokio::test]
async fn test_memory_network_partition_prevents_communication() {
    let mut network = MemoryNetwork::new(&[0, 1, 2]);

    network.create_partition(&[0], &[1, 2]);

    let transport0 = network.get_transport(0).unwrap();

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let result = transport0.send_request_vote(1, request).await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_memory_network_heal_restores_communication() {
    let mut network = MemoryNetwork::new(&[0, 1]);

    network.create_partition(&[0], &[1]);
    network.heal_all_partitions();

    let transport0 = network.get_transport(0).unwrap();
    let mut rx1 = network.take_receiver(1).unwrap();

    let handle = tokio::spawn(async move {
        if let Some(RaftMessage::RequestVote {
            request,
            response_tx,
        }) = rx1.recv().await
        {
            let response = distributed_kv_raft::RequestVoteResponse {
                term: request.term,
                vote_granted: true,
            };
            let _ = response_tx.send(response);
        }
    });

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let response = transport0.send_request_vote(1, request).await.unwrap();

    assert!(response.vote_granted);

    handle.await.unwrap();
}

// =============================================================================
// Timeout Tests
// =============================================================================

#[tokio::test]
async fn test_memory_transport_timeout() {
    let transport1 = Arc::new(MemoryTransport::new(0));
    let (tx, _rx) = mpsc::channel(10);

    transport1.register_peer(1, tx);
    // Don't handle the message, causing timeout

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let result = tokio::time::timeout(
        Duration::from_millis(600),
        transport1.send_request_vote(1, request),
    )
    .await;

    // Should timeout since receiver is not handling
    assert!(result.is_err() || result.unwrap().is_err());
}

// =============================================================================
// Delay Tests
// =============================================================================

#[tokio::test]
async fn test_memory_transport_with_delay() {
    let transport1 = Arc::new(MemoryTransport::new(0));
    let (tx, mut rx) = mpsc::channel(10);

    transport1.register_peer(1, tx);
    transport1.set_delay(Some(Duration::from_millis(50)));

    let handle = tokio::spawn(async move {
        if let Some(RaftMessage::RequestVote {
            request,
            response_tx,
        }) = rx.recv().await
        {
            let response = distributed_kv_raft::RequestVoteResponse {
                term: request.term,
                vote_granted: true,
            };
            let _ = response_tx.send(response);
        }
    });

    let start = std::time::Instant::now();

    let request = RequestVoteRequest {
        term: 1,
        candidate_id: 0,
        last_log_index: 0,
        last_log_term: 0,
    };

    let _response = transport1.send_request_vote(1, request).await.unwrap();

    let elapsed = start.elapsed();
    assert!(elapsed >= Duration::from_millis(50));

    handle.await.unwrap();
}
