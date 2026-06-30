//! Metrics and monitoring for the Raft cluster.

use crate::{LogIndex, NodeId, Term};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Counter metric.
#[derive(Default)]
pub struct Counter {
    value: AtomicU64,
}

impl Counter {
    /// Increment the counter.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Add to the counter.
    pub fn add(&self, n: u64) {
        self.value.fetch_add(n, Ordering::Relaxed);
    }

    /// Get current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }
}

/// Gauge metric (can go up or down).
#[derive(Default)]
pub struct Gauge {
    value: AtomicU64,
}

impl Gauge {
    /// Set the gauge value.
    pub fn set(&self, value: u64) {
        self.value.store(value, Ordering::Relaxed);
    }

    /// Get current value.
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::Relaxed)
    }

    /// Increment the gauge.
    pub fn inc(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the gauge.
    pub fn dec(&self) {
        self.value.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Histogram for tracking distributions.
pub struct Histogram {
    /// Buckets for distribution.
    buckets: Vec<AtomicU64>,
    /// Bucket boundaries (in microseconds).
    boundaries: Vec<u64>,
    /// Sum of all observations.
    sum: AtomicU64,
    /// Count of observations.
    count: AtomicU64,
}

impl Histogram {
    /// Create a new histogram with default buckets.
    pub fn new() -> Self {
        // Default buckets: 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s
        let boundaries = vec![1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000];
        let buckets = (0..=boundaries.len())
            .map(|_| AtomicU64::new(0))
            .collect();

        Self {
            buckets,
            boundaries,
            sum: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Observe a duration.
    pub fn observe(&self, duration: Duration) {
        let micros = duration.as_micros() as u64;
        self.sum.fetch_add(micros, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);

        // Find bucket
        let idx = self
            .boundaries
            .iter()
            .position(|&b| micros <= b)
            .unwrap_or(self.boundaries.len());
        self.buckets[idx].fetch_add(1, Ordering::Relaxed);
    }

    /// Get mean value.
    pub fn mean(&self) -> f64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0.0;
        }
        let sum = self.sum.load(Ordering::Relaxed);
        (sum as f64) / (count as f64)
    }

    /// Get percentile (approximate).
    pub fn percentile(&self, p: f64) -> u64 {
        let count = self.count.load(Ordering::Relaxed);
        if count == 0 {
            return 0;
        }

        let target = (count as f64 * p) as u64;
        let mut accumulated = 0u64;

        for (i, bucket) in self.buckets.iter().enumerate() {
            accumulated += bucket.load(Ordering::Relaxed);
            if accumulated >= target {
                return if i < self.boundaries.len() {
                    self.boundaries[i]
                } else {
                    self.boundaries.last().copied().unwrap_or(0) * 2
                };
            }
        }

        self.boundaries.last().copied().unwrap_or(0)
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for a Raft node.
#[derive(Default)]
pub struct RaftMetrics {
    // Election metrics
    /// Number of elections started.
    pub elections_started: Counter,
    /// Number of elections won.
    pub elections_won: Counter,
    /// Number of leader changes observed.
    pub leader_changes: Counter,

    // Replication metrics
    /// Log entries appended.
    pub entries_appended: Counter,
    /// Log entries committed.
    pub entries_committed: Counter,
    /// Log entries applied to FSM.
    pub entries_applied: Counter,

    // RPC metrics
    /// AppendEntries RPCs sent.
    pub append_entries_sent: Counter,
    /// AppendEntries RPCs received.
    pub append_entries_received: Counter,
    /// RequestVote RPCs sent.
    pub request_vote_sent: Counter,
    /// RequestVote RPCs received.
    pub request_vote_received: Counter,

    // Latency metrics
    /// Commit latency histogram.
    pub commit_latency: Histogram,
    /// Apply latency histogram.
    pub apply_latency: Histogram,
    /// RPC latency histogram.
    pub rpc_latency: Histogram,

    // Health metrics
    /// Number of connected peers.
    pub peers_connected: Gauge,
    /// Current log size.
    pub log_size: Gauge,
    /// Latest snapshot size.
    pub snapshot_size: Gauge,

    // State metrics
    /// Current term.
    pub current_term: Gauge,
    /// Commit index.
    pub commit_index: Gauge,
    /// Last applied index.
    pub last_applied: Gauge,
}

impl RaftMetrics {
    /// Create new metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record election start.
    pub fn record_election_start(&self) {
        self.elections_started.inc();
    }

    /// Record election win.
    pub fn record_election_win(&self) {
        self.elections_won.inc();
        self.leader_changes.inc();
    }

    /// Record entry append.
    pub fn record_entries_appended(&self, count: u64) {
        self.entries_appended.add(count);
        self.log_size.set(self.log_size.get() + count);
    }

    /// Record commit.
    pub fn record_commit(&self, latency: Duration) {
        self.entries_committed.inc();
        self.commit_latency.observe(latency);
    }

    /// Record apply.
    pub fn record_apply(&self, latency: Duration) {
        self.entries_applied.inc();
        self.apply_latency.observe(latency);
    }

    /// Record RPC.
    pub fn record_rpc(&self, latency: Duration) {
        self.rpc_latency.observe(latency);
    }

    /// Update state metrics.
    pub fn update_state(&self, term: Term, commit: LogIndex, applied: LogIndex) {
        self.current_term.set(term);
        self.commit_index.set(commit);
        self.last_applied.set(applied);
    }

    /// Get metrics snapshot.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            elections_started: self.elections_started.get(),
            elections_won: self.elections_won.get(),
            leader_changes: self.leader_changes.get(),
            entries_appended: self.entries_appended.get(),
            entries_committed: self.entries_committed.get(),
            entries_applied: self.entries_applied.get(),
            current_term: self.current_term.get(),
            commit_index: self.commit_index.get(),
            last_applied: self.last_applied.get(),
            commit_latency_p50: self.commit_latency.percentile(0.5),
            commit_latency_p99: self.commit_latency.percentile(0.99),
            rpc_latency_mean: self.rpc_latency.mean(),
        }
    }
}

/// Snapshot of metrics at a point in time.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub elections_started: u64,
    pub elections_won: u64,
    pub leader_changes: u64,
    pub entries_appended: u64,
    pub entries_committed: u64,
    pub entries_applied: u64,
    pub current_term: u64,
    pub commit_index: u64,
    pub last_applied: u64,
    pub commit_latency_p50: u64,
    pub commit_latency_p99: u64,
    pub rpc_latency_mean: f64,
}

/// Health check result.
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Whether this node is the leader.
    pub is_leader: bool,
    /// Current term.
    pub term: Term,
    /// Commit index.
    pub commit_index: LogIndex,
    /// Applied index.
    pub applied_index: LogIndex,
    /// Total cluster size.
    pub cluster_size: usize,
    /// Number of healthy peers.
    pub healthy_peers: usize,
    /// Current state name.
    pub state: String,
    /// Leader ID if known.
    pub leader_id: Option<NodeId>,
}

impl HealthCheck {
    /// Check if node is healthy.
    pub fn is_healthy(&self) -> bool {
        // Healthy if we know the leader and have majority connectivity
        self.leader_id.is_some() && self.healthy_peers >= self.cluster_size / 2
    }
}

/// Per-peer replication metrics.
#[derive(Debug, Clone, Default)]
pub struct PeerMetrics {
    /// Next index to send.
    pub next_index: LogIndex,
    /// Highest replicated index.
    pub match_index: LogIndex,
    /// Last successful contact.
    pub last_contact: Option<Instant>,
    /// Number of failed RPCs.
    pub rpc_failures: u64,
    /// Whether peer is currently reachable.
    pub reachable: bool,
}

/// Metrics aggregator for the cluster.
pub struct ClusterMetrics {
    /// Per-node metrics.
    pub nodes: HashMap<NodeId, RaftMetrics>,
}

impl ClusterMetrics {
    /// Create new cluster metrics.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
        }
    }

    /// Add metrics for a node.
    pub fn add_node(&mut self, id: NodeId) {
        self.nodes.insert(id, RaftMetrics::new());
    }

    /// Get metrics for a node.
    pub fn get(&self, id: NodeId) -> Option<&RaftMetrics> {
        self.nodes.get(&id)
    }

    /// Get mutable metrics for a node.
    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut RaftMetrics> {
        self.nodes.get_mut(&id)
    }
}

impl Default for ClusterMetrics {
    fn default() -> Self {
        Self::new()
    }
}
