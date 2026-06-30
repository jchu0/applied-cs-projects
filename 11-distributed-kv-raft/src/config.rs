//! Configuration for Raft nodes.

use crate::NodeId;
use std::collections::HashSet;
use std::time::Duration;

/// Configuration for a Raft node.
#[derive(Debug, Clone)]
pub struct RaftConfig {
    /// This node's unique identifier.
    pub id: NodeId,

    /// Address this node listens on.
    pub listen_addr: String,

    /// Peer node addresses.
    pub peers: Vec<PeerConfig>,

    /// Election timeout range (randomized within this range).
    pub election_timeout_min: Duration,
    pub election_timeout_max: Duration,

    /// Heartbeat interval (must be < election_timeout_min).
    pub heartbeat_interval: Duration,

    /// Maximum entries per AppendEntries RPC.
    pub max_entries_per_rpc: usize,

    /// Snapshot threshold (number of entries before compaction).
    pub snapshot_threshold: u64,

    /// Data directory for persistence.
    pub data_dir: String,

    /// Enable leader lease for faster reads.
    pub enable_leader_lease: bool,
}

/// Configuration for a peer node.
#[derive(Debug, Clone)]
pub struct PeerConfig {
    /// Peer's node ID.
    pub id: NodeId,

    /// Peer's network address.
    pub addr: String,
}

impl Default for RaftConfig {
    fn default() -> Self {
        Self {
            id: 0,
            listen_addr: "127.0.0.1:8080".to_string(),
            peers: Vec::new(),
            election_timeout_min: Duration::from_millis(150),
            election_timeout_max: Duration::from_millis(300),
            heartbeat_interval: Duration::from_millis(50),
            max_entries_per_rpc: 100,
            snapshot_threshold: 10000,
            data_dir: "/tmp/raft".to_string(),
            enable_leader_lease: true,
        }
    }
}

impl RaftConfig {
    /// Create a new configuration builder.
    pub fn builder() -> RaftConfigBuilder {
        RaftConfigBuilder::default()
    }

    /// Get all node IDs in the cluster (including self).
    pub fn cluster_nodes(&self) -> HashSet<NodeId> {
        let mut nodes = HashSet::new();
        nodes.insert(self.id);
        for peer in &self.peers {
            nodes.insert(peer.id);
        }
        nodes
    }

    /// Get quorum size for the cluster.
    pub fn quorum_size(&self) -> usize {
        let cluster_size = self.peers.len() + 1;
        cluster_size / 2 + 1
    }

    /// Generate random election timeout.
    pub fn random_election_timeout(&self) -> Duration {
        use rand::Rng;
        let min = self.election_timeout_min.as_millis() as u64;
        let max = self.election_timeout_max.as_millis() as u64;
        let timeout = rand::thread_rng().gen_range(min..=max);
        Duration::from_millis(timeout)
    }
}

/// Builder for RaftConfig.
#[derive(Default)]
pub struct RaftConfigBuilder {
    config: RaftConfig,
}

impl RaftConfigBuilder {
    /// Set node ID.
    pub fn id(mut self, id: NodeId) -> Self {
        self.config.id = id;
        self
    }

    /// Set listen address.
    pub fn listen_addr(mut self, addr: impl Into<String>) -> Self {
        self.config.listen_addr = addr.into();
        self
    }

    /// Add a peer.
    pub fn peer(mut self, id: NodeId, addr: impl Into<String>) -> Self {
        self.config.peers.push(PeerConfig {
            id,
            addr: addr.into(),
        });
        self
    }

    /// Set election timeout range.
    pub fn election_timeout(mut self, min: Duration, max: Duration) -> Self {
        self.config.election_timeout_min = min;
        self.config.election_timeout_max = max;
        self
    }

    /// Set heartbeat interval.
    pub fn heartbeat_interval(mut self, interval: Duration) -> Self {
        self.config.heartbeat_interval = interval;
        self
    }

    /// Set data directory.
    pub fn data_dir(mut self, dir: impl Into<String>) -> Self {
        self.config.data_dir = dir.into();
        self
    }

    /// Build the configuration.
    pub fn build(self) -> RaftConfig {
        self.config
    }
}

/// Cluster configuration for membership changes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ClusterConfig {
    /// Current (old) configuration.
    pub current: HashSet<NodeId>,

    /// New configuration (for joint consensus).
    pub next: Option<HashSet<NodeId>>,
}

impl ClusterConfig {
    /// Create a new cluster configuration.
    pub fn new(nodes: HashSet<NodeId>) -> Self {
        Self {
            current: nodes,
            next: None,
        }
    }

    /// Check if in joint consensus mode.
    pub fn is_joint(&self) -> bool {
        self.next.is_some()
    }

    /// Get all nodes in current configuration.
    pub fn nodes(&self) -> &HashSet<NodeId> {
        &self.current
    }

    /// Calculate quorum size.
    pub fn quorum_size(&self) -> usize {
        match &self.next {
            None => self.current.len() / 2 + 1,
            Some(next) => {
                // Need majority from both configurations
                let old_quorum = self.current.len() / 2 + 1;
                let new_quorum = next.len() / 2 + 1;
                std::cmp::max(old_quorum, new_quorum)
            }
        }
    }
}
