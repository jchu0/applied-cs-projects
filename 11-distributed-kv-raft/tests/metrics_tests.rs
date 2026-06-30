//! Comprehensive tests for the metrics module.

use distributed_kv_raft::{
    ClusterMetrics, Counter, Gauge, HealthCheck, Histogram, MetricsSnapshot, PeerMetrics,
    RaftMetrics,
};
use std::time::Duration;

// =============================================================================
// Counter Tests
// =============================================================================

#[test]
fn test_counter_new() {
    let counter = Counter::default();
    assert_eq!(counter.get(), 0);
}

#[test]
fn test_counter_inc() {
    let counter = Counter::default();
    counter.inc();
    assert_eq!(counter.get(), 1);
}

#[test]
fn test_counter_inc_multiple() {
    let counter = Counter::default();
    for _ in 0..100 {
        counter.inc();
    }
    assert_eq!(counter.get(), 100);
}

#[test]
fn test_counter_add() {
    let counter = Counter::default();
    counter.add(50);
    assert_eq!(counter.get(), 50);
}

#[test]
fn test_counter_add_multiple() {
    let counter = Counter::default();
    counter.add(10);
    counter.add(20);
    counter.add(30);
    assert_eq!(counter.get(), 60);
}

// =============================================================================
// Gauge Tests
// =============================================================================

#[test]
fn test_gauge_new() {
    let gauge = Gauge::default();
    assert_eq!(gauge.get(), 0);
}

#[test]
fn test_gauge_set() {
    let gauge = Gauge::default();
    gauge.set(42);
    assert_eq!(gauge.get(), 42);
}

#[test]
fn test_gauge_inc() {
    let gauge = Gauge::default();
    gauge.set(10);
    gauge.inc();
    assert_eq!(gauge.get(), 11);
}

#[test]
fn test_gauge_dec() {
    let gauge = Gauge::default();
    gauge.set(10);
    gauge.dec();
    assert_eq!(gauge.get(), 9);
}

#[test]
fn test_gauge_overwrite() {
    let gauge = Gauge::default();
    gauge.set(100);
    gauge.set(50);
    assert_eq!(gauge.get(), 50);
}

// =============================================================================
// Histogram Tests
// =============================================================================

#[test]
fn test_histogram_new() {
    let histogram = Histogram::new();
    assert_eq!(histogram.mean(), 0.0);
}

#[test]
fn test_histogram_default() {
    let histogram = Histogram::default();
    assert_eq!(histogram.mean(), 0.0);
}

#[test]
fn test_histogram_observe() {
    let histogram = Histogram::new();
    histogram.observe(Duration::from_millis(10));
    assert!(histogram.mean() > 0.0);
}

#[test]
fn test_histogram_observe_multiple() {
    let histogram = Histogram::new();
    for i in 1..=10 {
        histogram.observe(Duration::from_millis(i * 10));
    }
    // Mean should be around 55ms (average of 10, 20, ..., 100)
    let mean = histogram.mean();
    assert!(mean > 0.0);
}

#[test]
fn test_histogram_percentile_50() {
    let histogram = Histogram::new();
    for _ in 0..100 {
        histogram.observe(Duration::from_millis(50));
    }
    let p50 = histogram.percentile(0.5);
    // Should be in the bucket containing 50ms
    assert!(p50 > 0);
}

#[test]
fn test_histogram_percentile_99() {
    let histogram = Histogram::new();
    for _ in 0..100 {
        histogram.observe(Duration::from_millis(10));
    }
    let p99 = histogram.percentile(0.99);
    // Should be in or near the first bucket
    assert!(p99 > 0);
}

#[test]
fn test_histogram_empty_percentile() {
    let histogram = Histogram::new();
    assert_eq!(histogram.percentile(0.5), 0);
}

// =============================================================================
// RaftMetrics Tests
// =============================================================================

#[test]
fn test_raft_metrics_new() {
    let metrics = RaftMetrics::new();
    assert_eq!(metrics.elections_started.get(), 0);
    assert_eq!(metrics.elections_won.get(), 0);
}

#[test]
fn test_raft_metrics_default() {
    let metrics = RaftMetrics::default();
    assert_eq!(metrics.elections_started.get(), 0);
}

#[test]
fn test_raft_metrics_record_election_start() {
    let metrics = RaftMetrics::new();
    metrics.record_election_start();
    assert_eq!(metrics.elections_started.get(), 1);
}

#[test]
fn test_raft_metrics_record_election_win() {
    let metrics = RaftMetrics::new();
    metrics.record_election_win();
    assert_eq!(metrics.elections_won.get(), 1);
    assert_eq!(metrics.leader_changes.get(), 1);
}

#[test]
fn test_raft_metrics_record_entries_appended() {
    let metrics = RaftMetrics::new();
    metrics.record_entries_appended(10);
    assert_eq!(metrics.entries_appended.get(), 10);
    assert_eq!(metrics.log_size.get(), 10);
}

#[test]
fn test_raft_metrics_record_commit() {
    let metrics = RaftMetrics::new();
    metrics.record_commit(Duration::from_millis(50));
    assert_eq!(metrics.entries_committed.get(), 1);
}

#[test]
fn test_raft_metrics_record_apply() {
    let metrics = RaftMetrics::new();
    metrics.record_apply(Duration::from_millis(10));
    assert_eq!(metrics.entries_applied.get(), 1);
}

#[test]
fn test_raft_metrics_record_rpc() {
    let metrics = RaftMetrics::new();
    metrics.record_rpc(Duration::from_millis(5));
    // RPC latency should be recorded
    assert!(metrics.rpc_latency.mean() > 0.0);
}

#[test]
fn test_raft_metrics_update_state() {
    let metrics = RaftMetrics::new();
    metrics.update_state(5, 100, 90);
    assert_eq!(metrics.current_term.get(), 5);
    assert_eq!(metrics.commit_index.get(), 100);
    assert_eq!(metrics.last_applied.get(), 90);
}

#[test]
fn test_raft_metrics_snapshot() {
    let metrics = RaftMetrics::new();

    metrics.record_election_start();
    metrics.record_election_start();
    metrics.record_election_win();
    metrics.record_entries_appended(50);
    metrics.update_state(3, 40, 35);

    let snapshot = metrics.snapshot();

    assert_eq!(snapshot.elections_started, 2);
    assert_eq!(snapshot.elections_won, 1);
    assert_eq!(snapshot.entries_appended, 50);
    assert_eq!(snapshot.current_term, 3);
    assert_eq!(snapshot.commit_index, 40);
    assert_eq!(snapshot.last_applied, 35);
}

// =============================================================================
// PeerMetrics Tests
// =============================================================================

#[test]
fn test_peer_metrics_default() {
    let metrics = PeerMetrics::default();
    assert_eq!(metrics.next_index, 0);
    assert_eq!(metrics.match_index, 0);
    assert_eq!(metrics.rpc_failures, 0);
    assert!(!metrics.reachable);
}

#[test]
fn test_peer_metrics_fields() {
    let mut metrics = PeerMetrics::default();
    metrics.next_index = 10;
    metrics.match_index = 5;
    metrics.reachable = true;

    assert_eq!(metrics.next_index, 10);
    assert_eq!(metrics.match_index, 5);
    assert!(metrics.reachable);
}

// =============================================================================
// ClusterMetrics Tests
// =============================================================================

#[test]
fn test_cluster_metrics_new() {
    let metrics = ClusterMetrics::new();
    assert!(metrics.nodes.is_empty());
}

#[test]
fn test_cluster_metrics_default() {
    let metrics = ClusterMetrics::default();
    assert!(metrics.nodes.is_empty());
}

#[test]
fn test_cluster_metrics_add_node() {
    let mut metrics = ClusterMetrics::new();
    metrics.add_node(0);
    metrics.add_node(1);
    metrics.add_node(2);

    assert!(metrics.get(0).is_some());
    assert!(metrics.get(1).is_some());
    assert!(metrics.get(2).is_some());
    assert!(metrics.get(3).is_none());
}

#[test]
fn test_cluster_metrics_get_mut() {
    let mut metrics = ClusterMetrics::new();
    metrics.add_node(0);

    if let Some(node_metrics) = metrics.get_mut(0) {
        node_metrics.record_election_start();
    }

    assert_eq!(metrics.get(0).unwrap().elections_started.get(), 1);
}

// =============================================================================
// HealthCheck Tests
// =============================================================================

#[test]
fn test_health_check_healthy() {
    let check = HealthCheck {
        is_leader: true,
        term: 5,
        commit_index: 100,
        applied_index: 100,
        cluster_size: 3,
        healthy_peers: 2,
        state: "Leader".to_string(),
        leader_id: Some(0),
    };

    assert!(check.is_healthy());
}

#[test]
fn test_health_check_no_leader() {
    let check = HealthCheck {
        is_leader: false,
        term: 1,
        commit_index: 0,
        applied_index: 0,
        cluster_size: 3,
        healthy_peers: 2,
        state: "Follower".to_string(),
        leader_id: None,
    };

    assert!(!check.is_healthy());
}

#[test]
fn test_health_check_insufficient_peers() {
    let check = HealthCheck {
        is_leader: true,
        term: 5,
        commit_index: 100,
        applied_index: 100,
        cluster_size: 5,
        healthy_peers: 1,
        state: "Leader".to_string(),
        leader_id: Some(0),
    };

    assert!(!check.is_healthy());
}

#[test]
fn test_health_check_fields() {
    let check = HealthCheck {
        is_leader: false,
        term: 10,
        commit_index: 500,
        applied_index: 495,
        cluster_size: 5,
        healthy_peers: 4,
        state: "Follower".to_string(),
        leader_id: Some(2),
    };

    assert!(!check.is_leader);
    assert_eq!(check.term, 10);
    assert_eq!(check.commit_index, 500);
    assert_eq!(check.applied_index, 495);
    assert_eq!(check.cluster_size, 5);
    assert_eq!(check.healthy_peers, 4);
    assert_eq!(check.state, "Follower");
    assert_eq!(check.leader_id, Some(2));
}

// =============================================================================
// MetricsSnapshot Tests
// =============================================================================

#[test]
fn test_metrics_snapshot_clone() {
    let snapshot = MetricsSnapshot {
        elections_started: 5,
        elections_won: 3,
        leader_changes: 3,
        entries_appended: 1000,
        entries_committed: 950,
        entries_applied: 940,
        current_term: 10,
        commit_index: 950,
        last_applied: 940,
        commit_latency_p50: 5000,
        commit_latency_p99: 20000,
        rpc_latency_mean: 2500.0,
    };

    let cloned = snapshot.clone();

    assert_eq!(cloned.elections_started, 5);
    assert_eq!(cloned.entries_appended, 1000);
}

#[test]
fn test_metrics_snapshot_debug() {
    let snapshot = MetricsSnapshot {
        elections_started: 1,
        elections_won: 1,
        leader_changes: 1,
        entries_appended: 10,
        entries_committed: 10,
        entries_applied: 10,
        current_term: 2,
        commit_index: 10,
        last_applied: 10,
        commit_latency_p50: 1000,
        commit_latency_p99: 5000,
        rpc_latency_mean: 500.0,
    };

    let debug_str = format!("{:?}", snapshot);
    assert!(debug_str.contains("elections_started"));
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_metrics_complete_flow() {
    let metrics = RaftMetrics::new();

    // Simulate a complete election and operation flow
    metrics.record_election_start();
    metrics.record_election_win();
    metrics.update_state(1, 0, 0);

    for i in 1..=10 {
        metrics.record_entries_appended(1);
        metrics.record_commit(Duration::from_millis(10));
        metrics.record_apply(Duration::from_millis(5));
        metrics.update_state(1, i, i);
    }

    let snapshot = metrics.snapshot();

    assert_eq!(snapshot.elections_started, 1);
    assert_eq!(snapshot.elections_won, 1);
    assert_eq!(snapshot.entries_appended, 10);
    assert_eq!(snapshot.entries_committed, 10);
    assert_eq!(snapshot.entries_applied, 10);
    assert_eq!(snapshot.current_term, 1);
    assert_eq!(snapshot.commit_index, 10);
    assert_eq!(snapshot.last_applied, 10);
}

#[test]
fn test_cluster_metrics_full_cluster() {
    let mut metrics = ClusterMetrics::new();

    for id in 0..5 {
        metrics.add_node(id);
    }

    // Record operations on each node
    for id in 0..5 {
        let node = metrics.get_mut(id).unwrap();
        node.record_election_start();
        if id == 0 {
            node.record_election_win();
        }
        node.record_entries_appended(10);
    }

    // Verify
    for id in 0..5 {
        let node = metrics.get(id).unwrap();
        assert_eq!(node.elections_started.get(), 1);
        assert_eq!(node.entries_appended.get(), 10);
    }

    assert_eq!(metrics.get(0).unwrap().elections_won.get(), 1);
}
