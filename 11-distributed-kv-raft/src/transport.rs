//! Transport layer for Raft node communication.
//!
//! This module provides the network transport for RPC communication between Raft nodes.

use crate::error::{Error, Result};
use crate::rpc::{
    AppendEntriesRequest, AppendEntriesResponse, ClientRequest, ClientResponse,
    InstallSnapshotRequest, InstallSnapshotResponse, RequestVoteRequest, RequestVoteResponse,
};
use crate::NodeId;

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

/// Message types for internal transport.
#[derive(Debug)]
pub enum RaftMessage {
    /// RequestVote RPC.
    RequestVote {
        request: RequestVoteRequest,
        response_tx: oneshot::Sender<RequestVoteResponse>,
    },
    /// AppendEntries RPC.
    AppendEntries {
        request: AppendEntriesRequest,
        response_tx: oneshot::Sender<AppendEntriesResponse>,
    },
    /// InstallSnapshot RPC.
    InstallSnapshot {
        request: InstallSnapshotRequest,
        response_tx: oneshot::Sender<InstallSnapshotResponse>,
    },
    /// Client request.
    ClientRequest {
        request: ClientRequest,
        response_tx: oneshot::Sender<ClientResponse>,
    },
}

/// Trait for transport implementations.
#[async_trait]
pub trait Transport: Send + Sync {
    /// Send RequestVote RPC to a peer.
    async fn send_request_vote(
        &self,
        target: NodeId,
        request: RequestVoteRequest,
    ) -> Result<RequestVoteResponse>;

    /// Send AppendEntries RPC to a peer.
    async fn send_append_entries(
        &self,
        target: NodeId,
        request: AppendEntriesRequest,
    ) -> Result<AppendEntriesResponse>;

    /// Send InstallSnapshot RPC to a peer.
    async fn send_install_snapshot(
        &self,
        target: NodeId,
        request: InstallSnapshotRequest,
    ) -> Result<InstallSnapshotResponse>;

    /// Send a client request to a node.
    ///
    /// This is the client-facing command path: a `Put`/`Get`/`Delete` is
    /// delivered to the target node's event loop, which proposes it through
    /// Raft (leader), replicates to a quorum, applies it, and returns the
    /// applied result. Followers reply with [`ClientResponse::NotLeader`] and a
    /// leader hint so the caller can retry against the current leader.
    ///
    /// The default implementation returns a network error; transports that do
    /// not (yet) carry the client path — e.g. `GrpcTransport`, which is
    /// structurally complete but not wired for client requests — inherit it.
    async fn send_client_request(
        &self,
        target: NodeId,
        request: ClientRequest,
    ) -> Result<ClientResponse> {
        let _ = (target, request);
        Err(Error::Network(
            "send_client_request not supported by this transport".to_string(),
        ))
    }
}

/// In-memory transport for testing.
pub struct MemoryTransport {
    /// Node ID of this transport.
    node_id: NodeId,
    /// Map of node ID to message sender.
    peers: Arc<RwLock<HashMap<NodeId, mpsc::Sender<RaftMessage>>>>,
    /// RPC timeout.
    timeout: Duration,
    /// Simulated network partitions.
    partitions: Arc<RwLock<HashMap<NodeId, bool>>>,
    /// Simulated message delay.
    delay: Arc<RwLock<Option<Duration>>>,
}

impl MemoryTransport {
    /// Create a new memory transport.
    pub fn new(node_id: NodeId) -> Self {
        Self {
            node_id,
            peers: Arc::new(RwLock::new(HashMap::new())),
            timeout: Duration::from_millis(500),
            partitions: Arc::new(RwLock::new(HashMap::new())),
            delay: Arc::new(RwLock::new(None)),
        }
    }

    /// Register a peer's message channel.
    pub fn register_peer(&self, peer_id: NodeId, sender: mpsc::Sender<RaftMessage>) {
        self.peers.write().insert(peer_id, sender);
    }

    /// Unregister a peer.
    pub fn unregister_peer(&self, peer_id: NodeId) {
        self.peers.write().remove(&peer_id);
    }

    /// Set a network partition with a peer.
    pub fn set_partition(&self, peer_id: NodeId, partitioned: bool) {
        self.partitions.write().insert(peer_id, partitioned);
    }

    /// Check if partitioned from a peer.
    pub fn is_partitioned(&self, peer_id: NodeId) -> bool {
        *self.partitions.read().get(&peer_id).unwrap_or(&false)
    }

    /// Set simulated network delay.
    pub fn set_delay(&self, delay: Option<Duration>) {
        *self.delay.write() = delay;
    }

    /// Get the peers map (for testing).
    pub fn peers(&self) -> Arc<RwLock<HashMap<NodeId, mpsc::Sender<RaftMessage>>>> {
        Arc::clone(&self.peers)
    }

    /// Helper to send a message and wait for response.
    async fn send_with_timeout<T>(
        &self,
        target: NodeId,
        create_message: impl FnOnce(oneshot::Sender<T>) -> RaftMessage,
    ) -> Result<T> {
        // Check for partition.
        if self.is_partitioned(target) {
            return Err(Error::Network(format!("Partitioned from node {}", target)));
        }

        // Apply simulated delay.
        let delay_value = *self.delay.read();
        if let Some(delay) = delay_value {
            tokio::time::sleep(delay).await;
        }

        let sender = {
            let peers = self.peers.read();
            peers.get(&target).cloned()
        };

        let sender = sender.ok_or_else(|| {
            Error::Network(format!("Peer {} not connected", target))
        })?;

        let (response_tx, response_rx) = oneshot::channel();
        let message = create_message(response_tx);

        sender.send(message).await.map_err(|_| {
            Error::Network(format!("Failed to send message to {}", target))
        })?;

        tokio::time::timeout(self.timeout, response_rx)
            .await
            .map_err(|_| Error::Timeout)?
            .map_err(|_| Error::Network("Response channel closed".to_string()))
    }
}

#[async_trait]
impl Transport for MemoryTransport {
    async fn send_request_vote(
        &self,
        target: NodeId,
        request: RequestVoteRequest,
    ) -> Result<RequestVoteResponse> {
        debug!(
            "Node {} sending RequestVote to {} (term {})",
            self.node_id, target, request.term
        );
        self.send_with_timeout(target, |response_tx| RaftMessage::RequestVote {
            request,
            response_tx,
        })
        .await
    }

    async fn send_append_entries(
        &self,
        target: NodeId,
        request: AppendEntriesRequest,
    ) -> Result<AppendEntriesResponse> {
        debug!(
            "Node {} sending AppendEntries to {} (term {}, {} entries)",
            self.node_id,
            target,
            request.term,
            request.entries.len()
        );
        self.send_with_timeout(target, |response_tx| RaftMessage::AppendEntries {
            request,
            response_tx,
        })
        .await
    }

    async fn send_install_snapshot(
        &self,
        target: NodeId,
        request: InstallSnapshotRequest,
    ) -> Result<InstallSnapshotResponse> {
        debug!(
            "Node {} sending InstallSnapshot to {} (last_included_index {})",
            self.node_id, target, request.last_included_index
        );
        self.send_with_timeout(target, |response_tx| RaftMessage::InstallSnapshot {
            request,
            response_tx,
        })
        .await
    }

    async fn send_client_request(
        &self,
        target: NodeId,
        request: ClientRequest,
    ) -> Result<ClientResponse> {
        debug!(
            "Node {} forwarding client request {:?} to {}",
            self.node_id, request, target
        );
        self.send_with_timeout(target, |response_tx| RaftMessage::ClientRequest {
            request,
            response_tx,
        })
        .await
    }
}

/// Network of memory transports for testing.
pub struct MemoryNetwork {
    /// Transports by node ID.
    transports: HashMap<NodeId, Arc<MemoryTransport>>,
    /// Message receivers by node ID.
    receivers: HashMap<NodeId, mpsc::Receiver<RaftMessage>>,
    /// Message senders by node ID (retained so external clients can be wired
    /// to every node's event loop).
    senders: HashMap<NodeId, mpsc::Sender<RaftMessage>>,
}

impl MemoryNetwork {
    /// Create a new memory network with the given node IDs.
    pub fn new(node_ids: &[NodeId]) -> Self {
        let mut transports = HashMap::new();
        let mut receivers = HashMap::new();
        let mut senders: HashMap<NodeId, mpsc::Sender<RaftMessage>> = HashMap::new();

        // Create channels for each node.
        for &id in node_ids {
            let (tx, rx) = mpsc::channel(1000);
            senders.insert(id, tx);
            receivers.insert(id, rx);
            transports.insert(id, Arc::new(MemoryTransport::new(id)));
        }

        // Register peers for each transport.
        for (&id, transport) in &transports {
            for (&peer_id, sender) in &senders {
                if id != peer_id {
                    transport.register_peer(peer_id, sender.clone());
                }
            }
        }

        Self {
            transports,
            receivers,
            senders,
        }
    }

    /// Get transport for a node.
    pub fn get_transport(&self, node_id: NodeId) -> Option<Arc<MemoryTransport>> {
        self.transports.get(&node_id).cloned()
    }

    /// Build a transport for an external client.
    ///
    /// Unlike a node's transport (which is registered with peers only), the
    /// returned transport is registered with *every* node's event loop, so a
    /// client can deliver [`ClientRequest`]s to any node — including a
    /// follower, which will respond with a leader hint. The synthetic node ID
    /// [`u64::MAX`] identifies the client in transport logs.
    pub fn client_transport(&self) -> Arc<MemoryTransport> {
        let transport = MemoryTransport::new(u64::MAX);
        for (&id, sender) in &self.senders {
            transport.register_peer(id, sender.clone());
        }
        Arc::new(transport)
    }

    /// Take the receiver for a node.
    pub fn take_receiver(&mut self, node_id: NodeId) -> Option<mpsc::Receiver<RaftMessage>> {
        self.receivers.remove(&node_id)
    }

    /// Create a partition between two groups of nodes.
    pub fn create_partition(&self, group1: &[NodeId], group2: &[NodeId]) {
        for &id1 in group1 {
            if let Some(transport) = self.transports.get(&id1) {
                for &id2 in group2 {
                    transport.set_partition(id2, true);
                }
            }
        }
        for &id2 in group2 {
            if let Some(transport) = self.transports.get(&id2) {
                for &id1 in group1 {
                    transport.set_partition(id1, true);
                }
            }
        }
    }

    /// Heal all partitions.
    pub fn heal_all_partitions(&self) {
        for transport in self.transports.values() {
            transport.partitions.write().clear();
        }
    }

    /// Set delay for all transports.
    pub fn set_delay(&self, delay: Option<Duration>) {
        for transport in self.transports.values() {
            transport.set_delay(delay);
        }
    }
}

/// Peer connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerState {
    /// Peer is connected and responsive.
    Connected,
    /// Peer is unreachable.
    Disconnected,
    /// Peer connection is being established.
    Connecting,
}

/// Peer connection tracking.
pub struct PeerTracker {
    /// State of each peer.
    states: HashMap<NodeId, PeerState>,
    /// Number of consecutive failures per peer.
    failures: HashMap<NodeId, usize>,
    /// Maximum failures before marking disconnected.
    max_failures: usize,
}

impl PeerTracker {
    /// Create a new peer tracker.
    pub fn new(max_failures: usize) -> Self {
        Self {
            states: HashMap::new(),
            failures: HashMap::new(),
            max_failures,
        }
    }

    /// Record a successful RPC to a peer.
    pub fn record_success(&mut self, peer_id: NodeId) {
        self.states.insert(peer_id, PeerState::Connected);
        self.failures.insert(peer_id, 0);
    }

    /// Record a failed RPC to a peer.
    pub fn record_failure(&mut self, peer_id: NodeId) {
        let failures = self.failures.entry(peer_id).or_insert(0);
        *failures += 1;

        if *failures >= self.max_failures {
            self.states.insert(peer_id, PeerState::Disconnected);
        }
    }

    /// Get the state of a peer.
    pub fn get_state(&self, peer_id: NodeId) -> PeerState {
        *self.states.get(&peer_id).unwrap_or(&PeerState::Connecting)
    }

    /// Get connected peer count.
    pub fn connected_count(&self) -> usize {
        self.states
            .values()
            .filter(|&&s| s == PeerState::Connected)
            .count()
    }

    /// Get all peer states.
    pub fn all_states(&self) -> &HashMap<NodeId, PeerState> {
        &self.states
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peer_tracker_success() {
        let mut tracker = PeerTracker::new(3);

        tracker.record_success(1);
        assert_eq!(tracker.get_state(1), PeerState::Connected);
        assert_eq!(tracker.connected_count(), 1);
    }

    #[test]
    fn test_peer_tracker_failure() {
        let mut tracker = PeerTracker::new(3);

        tracker.record_success(1);
        tracker.record_failure(1);
        tracker.record_failure(1);
        assert_eq!(tracker.get_state(1), PeerState::Connected);

        tracker.record_failure(1);
        assert_eq!(tracker.get_state(1), PeerState::Disconnected);
    }

    #[test]
    fn test_peer_tracker_recovery() {
        let mut tracker = PeerTracker::new(3);

        // Fail peer.
        for _ in 0..3 {
            tracker.record_failure(1);
        }
        assert_eq!(tracker.get_state(1), PeerState::Disconnected);

        // Recover peer.
        tracker.record_success(1);
        assert_eq!(tracker.get_state(1), PeerState::Connected);
    }

    #[tokio::test]
    async fn test_memory_network_creation() {
        let network = MemoryNetwork::new(&[0, 1, 2]);

        assert!(network.get_transport(0).is_some());
        assert!(network.get_transport(1).is_some());
        assert!(network.get_transport(2).is_some());
        assert!(network.get_transport(3).is_none());
    }

    #[tokio::test]
    async fn test_memory_network_partition() {
        let network = MemoryNetwork::new(&[0, 1, 2, 3, 4]);

        network.create_partition(&[0, 1], &[2, 3, 4]);

        let transport0 = network.get_transport(0).unwrap();
        assert!(transport0.is_partitioned(2));
        assert!(transport0.is_partitioned(3));
        assert!(!transport0.is_partitioned(1));

        let transport2 = network.get_transport(2).unwrap();
        assert!(transport2.is_partitioned(0));
        assert!(transport2.is_partitioned(1));
        assert!(!transport2.is_partitioned(3));
    }

    #[tokio::test]
    async fn test_memory_network_heal_partition() {
        let network = MemoryNetwork::new(&[0, 1, 2]);

        network.create_partition(&[0], &[1, 2]);
        network.heal_all_partitions();

        let transport0 = network.get_transport(0).unwrap();
        assert!(!transport0.is_partitioned(1));
        assert!(!transport0.is_partitioned(2));
    }
}
