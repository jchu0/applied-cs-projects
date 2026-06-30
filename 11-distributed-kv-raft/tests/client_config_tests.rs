//! Comprehensive tests for client and config modules.

use distributed_kv_raft::{
    ClusterConfig, KVClient, NodeAddress, PeerConfig, RaftConfig, RaftConfigBuilder,
    RequestTracker,
};
use std::collections::HashSet;
use std::time::Duration;

// =============================================================================
// RaftConfig Tests
// =============================================================================

#[test]
fn test_raft_config_default() {
    let config = RaftConfig::default();

    assert_eq!(config.id, 0);
    assert!(config.peers.is_empty());
    assert_eq!(config.heartbeat_interval, Duration::from_millis(50));
}

#[test]
fn test_raft_config_builder_basic() {
    let config = RaftConfigBuilder::default()
        .id(1)
        .listen_addr("127.0.0.1:8001")
        .build();

    assert_eq!(config.id, 1);
    assert_eq!(config.listen_addr, "127.0.0.1:8001");
}

#[test]
fn test_raft_config_builder_with_peers() {
    let config = RaftConfigBuilder::default()
        .id(0)
        .peer(1, "127.0.0.1:8001")
        .peer(2, "127.0.0.1:8002")
        .build();

    assert_eq!(config.peers.len(), 2);
    assert_eq!(config.peers[0].id, 1);
    assert_eq!(config.peers[1].id, 2);
}

#[test]
fn test_raft_config_builder_election_timeout() {
    let config = RaftConfigBuilder::default()
        .id(0)
        .election_timeout(Duration::from_millis(200), Duration::from_millis(400))
        .build();

    assert_eq!(config.election_timeout_min, Duration::from_millis(200));
    assert_eq!(config.election_timeout_max, Duration::from_millis(400));
}

#[test]
fn test_raft_config_builder_heartbeat_interval() {
    let config = RaftConfigBuilder::default()
        .id(0)
        .heartbeat_interval(Duration::from_millis(100))
        .build();

    assert_eq!(config.heartbeat_interval, Duration::from_millis(100));
}

#[test]
fn test_raft_config_builder_data_dir() {
    let config = RaftConfigBuilder::default()
        .id(0)
        .data_dir("/var/lib/raft")
        .build();

    assert_eq!(config.data_dir, "/var/lib/raft");
}

#[test]
fn test_raft_config_cluster_nodes() {
    let config = RaftConfigBuilder::default()
        .id(0)
        .peer(1, "127.0.0.1:8001")
        .peer(2, "127.0.0.1:8002")
        .build();

    let nodes = config.cluster_nodes();

    assert_eq!(nodes.len(), 3);
    assert!(nodes.contains(&0));
    assert!(nodes.contains(&1));
    assert!(nodes.contains(&2));
}

#[test]
fn test_raft_config_quorum_size_3() {
    let config = RaftConfigBuilder::default()
        .id(0)
        .peer(1, "127.0.0.1:8001")
        .peer(2, "127.0.0.1:8002")
        .build();

    assert_eq!(config.quorum_size(), 2);
}

#[test]
fn test_raft_config_quorum_size_5() {
    let config = RaftConfigBuilder::default()
        .id(0)
        .peer(1, "127.0.0.1:8001")
        .peer(2, "127.0.0.1:8002")
        .peer(3, "127.0.0.1:8003")
        .peer(4, "127.0.0.1:8004")
        .build();

    assert_eq!(config.quorum_size(), 3);
}

#[test]
fn test_raft_config_quorum_size_1() {
    let config = RaftConfigBuilder::default().id(0).build();

    assert_eq!(config.quorum_size(), 1);
}

#[test]
fn test_raft_config_random_election_timeout() {
    let config = RaftConfigBuilder::default()
        .id(0)
        .election_timeout(Duration::from_millis(100), Duration::from_millis(200))
        .build();

    for _ in 0..100 {
        let timeout = config.random_election_timeout();
        assert!(timeout >= Duration::from_millis(100));
        assert!(timeout <= Duration::from_millis(200));
    }
}

// =============================================================================
// PeerConfig Tests
// =============================================================================

#[test]
fn test_peer_config_creation() {
    let peer = PeerConfig {
        id: 1,
        addr: "127.0.0.1:8001".to_string(),
    };

    assert_eq!(peer.id, 1);
    assert_eq!(peer.addr, "127.0.0.1:8001");
}

#[test]
fn test_peer_config_clone() {
    let peer = PeerConfig {
        id: 2,
        addr: "localhost:5000".to_string(),
    };

    let cloned = peer.clone();

    assert_eq!(cloned.id, 2);
    assert_eq!(cloned.addr, "localhost:5000");
}

// =============================================================================
// ClusterConfig Tests
// =============================================================================

#[test]
fn test_cluster_config_new() {
    let mut nodes = HashSet::new();
    nodes.insert(0);
    nodes.insert(1);
    nodes.insert(2);

    let config = ClusterConfig::new(nodes.clone());

    assert_eq!(config.current, nodes);
    assert!(config.next.is_none());
}

#[test]
fn test_cluster_config_is_joint_false() {
    let mut nodes = HashSet::new();
    nodes.insert(0);
    nodes.insert(1);

    let config = ClusterConfig::new(nodes);

    assert!(!config.is_joint());
}

#[test]
fn test_cluster_config_is_joint_true() {
    let mut nodes = HashSet::new();
    nodes.insert(0);
    nodes.insert(1);

    let mut config = ClusterConfig::new(nodes);

    let mut new_nodes = HashSet::new();
    new_nodes.insert(0);
    new_nodes.insert(1);
    new_nodes.insert(2);
    config.next = Some(new_nodes);

    assert!(config.is_joint());
}

#[test]
fn test_cluster_config_nodes() {
    let mut nodes = HashSet::new();
    nodes.insert(0);
    nodes.insert(1);
    nodes.insert(2);

    let config = ClusterConfig::new(nodes.clone());

    assert_eq!(config.nodes(), &nodes);
}

#[test]
fn test_cluster_config_quorum_size_normal() {
    let mut nodes = HashSet::new();
    nodes.insert(0);
    nodes.insert(1);
    nodes.insert(2);

    let config = ClusterConfig::new(nodes);

    assert_eq!(config.quorum_size(), 2);
}

#[test]
fn test_cluster_config_quorum_size_joint() {
    let mut old_nodes = HashSet::new();
    old_nodes.insert(0);
    old_nodes.insert(1);
    old_nodes.insert(2);

    let mut config = ClusterConfig::new(old_nodes);

    let mut new_nodes = HashSet::new();
    new_nodes.insert(0);
    new_nodes.insert(1);
    new_nodes.insert(2);
    new_nodes.insert(3);
    new_nodes.insert(4);
    config.next = Some(new_nodes);

    // Joint consensus: need majority from both configs
    // Old: 3 nodes -> quorum = 2
    // New: 5 nodes -> quorum = 3
    // Max = 3
    assert_eq!(config.quorum_size(), 3);
}

// =============================================================================
// NodeAddress Tests
// =============================================================================

#[test]
fn test_node_address_creation() {
    let addr = NodeAddress {
        id: 1,
        addr: "127.0.0.1:8001".to_string(),
    };

    assert_eq!(addr.id, 1);
    assert_eq!(addr.addr, "127.0.0.1:8001");
}

#[test]
fn test_node_address_clone() {
    let addr = NodeAddress {
        id: 2,
        addr: "localhost:5000".to_string(),
    };

    let cloned = addr.clone();

    assert_eq!(cloned.id, 2);
    assert_eq!(cloned.addr, "localhost:5000");
}

// =============================================================================
// KVClient Tests
// =============================================================================

#[test]
fn test_kv_client_creation() {
    let cluster = vec![
        NodeAddress {
            id: 0,
            addr: "127.0.0.1:8000".to_string(),
        },
        NodeAddress {
            id: 1,
            addr: "127.0.0.1:8001".to_string(),
        },
        NodeAddress {
            id: 2,
            addr: "127.0.0.1:8002".to_string(),
        },
    ];

    let client = KVClient::new(cluster);

    // Client is created
    let _ = client;
}

#[test]
fn test_kv_client_with_timeout() {
    let cluster = vec![NodeAddress {
        id: 0,
        addr: "127.0.0.1:8000".to_string(),
    }];

    let client = KVClient::new(cluster).with_timeout(Duration::from_secs(10));

    let _ = client;
}

#[test]
fn test_kv_client_with_retries() {
    let cluster = vec![NodeAddress {
        id: 0,
        addr: "127.0.0.1:8000".to_string(),
    }];

    let client = KVClient::new(cluster).with_retries(5);

    let _ = client;
}

#[test]
fn test_kv_client_chained_config() {
    let cluster = vec![NodeAddress {
        id: 0,
        addr: "127.0.0.1:8000".to_string(),
    }];

    let client = KVClient::new(cluster)
        .with_timeout(Duration::from_secs(5))
        .with_retries(3);

    let _ = client;
}

// =============================================================================
// RequestTracker Tests
// =============================================================================

#[test]
fn test_request_tracker_new() {
    let tracker = RequestTracker::new(100);

    // No completed requests initially
    assert!(tracker.get_completed(1, 1).is_none());
}

#[test]
fn test_request_tracker_record_and_get() {
    let mut tracker = RequestTracker::new(100);

    let response = distributed_kv_raft::ClientResponse::Success {
        value: Some(b"test".to_vec()),
    };

    tracker.record_completed(1, 1, response.clone());

    let retrieved = tracker.get_completed(1, 1);
    assert!(retrieved.is_some());
}

#[test]
fn test_request_tracker_wrong_sequence() {
    let mut tracker = RequestTracker::new(100);

    let response = distributed_kv_raft::ClientResponse::Success { value: None };

    tracker.record_completed(1, 1, response);

    // Wrong sequence number
    let retrieved = tracker.get_completed(1, 2);
    assert!(retrieved.is_none());
}

#[test]
fn test_request_tracker_wrong_client() {
    let mut tracker = RequestTracker::new(100);

    let response = distributed_kv_raft::ClientResponse::Success { value: None };

    tracker.record_completed(1, 1, response);

    // Wrong client ID
    let retrieved = tracker.get_completed(2, 1);
    assert!(retrieved.is_none());
}

#[test]
fn test_request_tracker_overwrite() {
    let mut tracker = RequestTracker::new(100);

    let response1 = distributed_kv_raft::ClientResponse::Success {
        value: Some(b"first".to_vec()),
    };
    let response2 = distributed_kv_raft::ClientResponse::Success {
        value: Some(b"second".to_vec()),
    };

    tracker.record_completed(1, 1, response1);
    tracker.record_completed(1, 2, response2);

    // Should have newer response
    let retrieved = tracker.get_completed(1, 2);
    assert!(retrieved.is_some());

    // Old sequence not available
    let old = tracker.get_completed(1, 1);
    assert!(old.is_none());
}

#[test]
fn test_request_tracker_multiple_clients() {
    let mut tracker = RequestTracker::new(100);

    let response1 = distributed_kv_raft::ClientResponse::Success {
        value: Some(b"client1".to_vec()),
    };
    let response2 = distributed_kv_raft::ClientResponse::Success {
        value: Some(b"client2".to_vec()),
    };

    tracker.record_completed(1, 1, response1);
    tracker.record_completed(2, 1, response2);

    assert!(tracker.get_completed(1, 1).is_some());
    assert!(tracker.get_completed(2, 1).is_some());
}

// =============================================================================
// Config Edge Cases
// =============================================================================

#[test]
fn test_config_single_node() {
    let config = RaftConfigBuilder::default().id(0).build();

    assert_eq!(config.cluster_nodes().len(), 1);
    assert_eq!(config.quorum_size(), 1);
}

#[test]
fn test_config_even_cluster() {
    let config = RaftConfigBuilder::default()
        .id(0)
        .peer(1, "127.0.0.1:8001")
        .peer(2, "127.0.0.1:8002")
        .peer(3, "127.0.0.1:8003")
        .build();

    // 4 nodes: quorum = 3
    assert_eq!(config.quorum_size(), 3);
}

#[test]
fn test_cluster_config_single_node() {
    let mut nodes = HashSet::new();
    nodes.insert(0);

    let config = ClusterConfig::new(nodes);

    assert_eq!(config.quorum_size(), 1);
}

#[test]
fn test_cluster_config_two_nodes() {
    let mut nodes = HashSet::new();
    nodes.insert(0);
    nodes.insert(1);

    let config = ClusterConfig::new(nodes);

    assert_eq!(config.quorum_size(), 2);
}

// =============================================================================
// Debug Trait Tests
// =============================================================================

#[test]
fn test_raft_config_debug() {
    let config = RaftConfigBuilder::default().id(1).build();

    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("id"));
}

#[test]
fn test_peer_config_debug() {
    let peer = PeerConfig {
        id: 1,
        addr: "127.0.0.1:8001".to_string(),
    };

    let debug_str = format!("{:?}", peer);
    assert!(debug_str.contains("127.0.0.1"));
}

#[test]
fn test_cluster_config_debug() {
    let mut nodes = HashSet::new();
    nodes.insert(0);

    let config = ClusterConfig::new(nodes);

    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("current"));
}

#[test]
fn test_node_address_debug() {
    let addr = NodeAddress {
        id: 1,
        addr: "127.0.0.1:8001".to_string(),
    };

    let debug_str = format!("{:?}", addr);
    assert!(debug_str.contains("127.0.0.1"));
}
