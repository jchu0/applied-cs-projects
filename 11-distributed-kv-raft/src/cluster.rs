//! Cluster coordination and management.
//!
//! This module provides the RaftCluster type that coordinates multiple nodes
//! and handles the event loop for processing Raft messages.

use crate::config::RaftConfig;
use crate::error::{Error, Result};
use crate::node::{Command, LogEntry, RaftNode, RaftState};
use crate::rpc::{
    AppendEntriesRequest, AppendEntriesResponse, ClientRequest, ClientResponse,
    InstallSnapshotRequest, InstallSnapshotResponse, RequestVoteRequest, RequestVoteResponse,
};
use crate::storage::{KeyValueFSM, Snapshot, SnapshotStore};
use crate::transport::{MemoryTransport, RaftMessage, Transport};
use crate::{LogIndex, NodeId, Term};

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, info, warn};

/// A running Raft node with its event loop.
pub struct RaftClusterNode {
    /// The Raft node.
    node: Arc<RwLock<RaftNode>>,
    /// Transport for network communication.
    transport: Arc<MemoryTransport>,
    /// Message receiver.
    receiver: mpsc::Receiver<RaftMessage>,
    /// Whether the node is running.
    running: Arc<RwLock<bool>>,
    /// Snapshot store.
    snapshot_store: Option<SnapshotStore>,
    /// Last snapshot index.
    last_snapshot_index: LogIndex,
    /// Snapshot threshold.
    snapshot_threshold: u64,
}

impl RaftClusterNode {
    /// Create a new cluster node.
    pub fn new(
        node: RaftNode,
        transport: Arc<MemoryTransport>,
        receiver: mpsc::Receiver<RaftMessage>,
    ) -> Self {
        let config = node.config.clone();
        Self {
            node: Arc::new(RwLock::new(node)),
            transport,
            receiver,
            running: Arc::new(RwLock::new(false)),
            snapshot_store: None,
            last_snapshot_index: 0,
            snapshot_threshold: config.snapshot_threshold,
        }
    }

    /// Get a reference to the node.
    pub fn node(&self) -> Arc<RwLock<RaftNode>> {
        Arc::clone(&self.node)
    }

    /// Get the node ID.
    pub fn id(&self) -> NodeId {
        self.node.read().id()
    }

    /// Check if this node is the leader.
    pub fn is_leader(&self) -> bool {
        self.node.read().is_leader()
    }

    /// Get the current term.
    pub fn term(&self) -> Term {
        self.node.read().term()
    }

    /// Get the leader ID.
    pub fn leader_id(&self) -> Option<NodeId> {
        self.node.read().leader_id()
    }

    /// Run the event loop.
    pub async fn run(&mut self) {
        *self.running.write() = true;
        info!("Node {} starting event loop", self.id());

        let heartbeat_interval = {
            let node = self.node.read();
            node.config.heartbeat_interval
        };

        let mut heartbeat_timer = interval(heartbeat_interval);
        let mut election_check_timer = interval(Duration::from_millis(50));

        while *self.running.read() {
            tokio::select! {
                // Handle incoming messages.
                Some(msg) = self.receiver.recv() => {
                    self.handle_message(msg).await;
                }

                // Heartbeat timer for leaders.
                _ = heartbeat_timer.tick() => {
                    self.maybe_send_heartbeats().await;
                }

                // Election timeout check.
                _ = election_check_timer.tick() => {
                    self.maybe_start_election().await;
                }
            }

            // Check for snapshot.
            self.maybe_create_snapshot().await;
        }

        info!("Node {} stopped event loop", self.id());
    }

    /// Stop the event loop.
    pub fn stop(&self) {
        *self.running.write() = false;
    }

    /// Handle an incoming message.
    async fn handle_message(&self, msg: RaftMessage) {
        match msg {
            RaftMessage::RequestVote {
                request,
                response_tx,
            } => {
                let response = self.node.write().handle_request_vote(request);
                let _ = response_tx.send(response);
            }
            RaftMessage::AppendEntries {
                request,
                response_tx,
            } => {
                let response = self.node.write().handle_append_entries(request);
                let _ = response_tx.send(response);
            }
            RaftMessage::InstallSnapshot {
                request,
                response_tx,
            } => {
                let response = self.handle_install_snapshot(request).await;
                let _ = response_tx.send(response);
            }
            RaftMessage::ClientRequest {
                request,
                response_tx,
            } => {
                let response = self.handle_client_request(request).await;
                let _ = response_tx.send(response);
            }
        }
    }

    /// Handle InstallSnapshot RPC.
    async fn handle_install_snapshot(
        &self,
        request: InstallSnapshotRequest,
    ) -> InstallSnapshotResponse {
        let mut node = self.node.write();

        // Check term.
        if request.term < node.term() {
            return InstallSnapshotResponse { term: node.term() };
        }

        if request.term > node.term() {
            node.current_term = request.term;
            node.voted_for = None;
            node.transition_to_follower(Some(request.leader_id));
        }

        node.reset_election_timeout();

        // For simplicity, we assume the entire snapshot is sent in one chunk.
        if request.done {
            let snapshot = Snapshot {
                last_included_index: request.last_included_index,
                last_included_term: request.last_included_term,
                data: request.data,
            };

            // Restore FSM from snapshot.
            if let Err(e) = node.fsm.restore(&snapshot) {
                warn!("Failed to restore snapshot: {}", e);
            }

            // Update state.
            node.commit_index = request.last_included_index;
            node.last_applied = request.last_included_index;

            // Clear log up to snapshot.
            node.log.retain(|e| e.index > request.last_included_index);
        }

        InstallSnapshotResponse { term: node.term() }
    }

    /// Handle a client request.
    async fn handle_client_request(&self, request: ClientRequest) -> ClientResponse {
        let is_leader = self.node.read().is_leader();
        let leader_id = self.node.read().leader_id();

        if !is_leader {
            return ClientResponse::NotLeader {
                leader_hint: leader_id,
            };
        }

        match request {
            ClientRequest::Get { key } => {
                // For linearizable reads, we need to verify leadership.
                match self.linearizable_read(&key).await {
                    Ok(value) => ClientResponse::Success { value },
                    Err(e) => ClientResponse::Error {
                        message: e.to_string(),
                    },
                }
            }
            ClientRequest::Put { key, value } => {
                let command = Command::Put { key, value };
                match self.propose(command).await {
                    Ok(_) => ClientResponse::Success { value: None },
                    Err(e) => ClientResponse::Error {
                        message: e.to_string(),
                    },
                }
            }
            ClientRequest::Delete { key } => {
                let command = Command::Delete { key };
                match self.propose(command).await {
                    Ok(_) => ClientResponse::Success { value: None },
                    Err(e) => ClientResponse::Error {
                        message: e.to_string(),
                    },
                }
            }
        }
    }

    /// Propose a command to the cluster.
    pub async fn propose(&self, command: Command) -> Result<LogIndex> {
        // Append to log.
        let index = self.node.write().propose(command)?;

        // Replicate to peers.
        self.replicate_to_peers().await;

        Ok(index)
    }

    /// Perform a linearizable read.
    pub async fn linearizable_read(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Extract values from guard in a separate scope to ensure it's dropped before await.
        let (is_leader, leader_id, lease_valid, cached_value) = {
            let node = self.node.read();
            (
                node.is_leader(),
                node.leader_id(),
                node.lease_valid(),
                node.get(key),
            )
        };

        if !is_leader {
            return Err(Error::NotLeader(leader_id));
        }

        // Check if we have a valid lease.
        if lease_valid {
            return Ok(cached_value);
        }

        // Verify leadership by sending heartbeats.
        let confirmed = self.confirm_leadership().await?;
        if !confirmed {
            let leader_id = self.node.read().leader_id();
            return Err(Error::NotLeader(leader_id));
        }

        // Wait for commit index to be applied.
        let read_index = self.node.read().commit_index();
        self.wait_for_apply(read_index).await?;

        // Read from FSM.
        Ok(self.node.read().get(key))
    }

    /// Confirm leadership by getting acknowledgment from quorum.
    async fn confirm_leadership(&self) -> Result<bool> {
        let peer_ids = self.node.read().peer_ids();
        let quorum_size = (peer_ids.len() + 1) / 2 + 1;
        let mut confirmations = 1; // Self.

        for peer_id in peer_ids {
            let request = self.node.read().build_append_entries_request(peer_id);
            match self.transport.send_append_entries(peer_id, request).await {
                Ok(resp) if resp.success => {
                    confirmations += 1;
                    if confirmations >= quorum_size {
                        return Ok(true);
                    }
                }
                _ => {}
            }
        }

        Ok(confirmations >= quorum_size)
    }

    /// Wait for a log index to be applied.
    async fn wait_for_apply(&self, index: LogIndex) -> Result<()> {
        let timeout = Duration::from_secs(5);
        let start = Instant::now();

        while start.elapsed() < timeout {
            if self.node.read().last_applied() >= index {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        Err(Error::Timeout)
    }

    /// Maybe start an election if timeout elapsed.
    async fn maybe_start_election(&self) {
        let should_start = {
            let node = self.node.read();
            !node.is_leader() && node.election_timeout_elapsed()
        };

        if should_start {
            self.start_election().await;
        }
    }

    /// Start an election.
    async fn start_election(&self) {
        {
            let mut node = self.node.write();
            node.transition_to_candidate();
        }

        let (request, peer_ids) = {
            let node = self.node.read();
            (node.build_request_vote_request(), node.peer_ids())
        };

        // Request votes from peers.
        for peer_id in peer_ids {
            let transport = Arc::clone(&self.transport);
            let node = Arc::clone(&self.node);
            let req = request.clone();

            tokio::spawn(async move {
                match transport.send_request_vote(peer_id, req).await {
                    Ok(response) => {
                        node.write().handle_request_vote_response(peer_id, response);
                    }
                    Err(e) => {
                        debug!("Failed to get vote from {}: {}", peer_id, e);
                    }
                }
            });
        }
    }

    /// Maybe send heartbeats if leader.
    async fn maybe_send_heartbeats(&self) {
        let is_leader = self.node.read().is_leader();
        if is_leader {
            self.replicate_to_peers().await;
        }
    }

    /// Replicate log entries to peers.
    async fn replicate_to_peers(&self) {
        let peer_ids = self.node.read().peer_ids();

        for peer_id in peer_ids {
            let transport = Arc::clone(&self.transport);
            let node = Arc::clone(&self.node);
            let request = node.read().build_append_entries_request(peer_id);
            let entries_count = request.entries.len();

            tokio::spawn(async move {
                match transport.send_append_entries(peer_id, request).await {
                    Ok(response) => {
                        node.write()
                            .handle_append_entries_response(peer_id, response, entries_count);
                    }
                    Err(e) => {
                        debug!("Failed to replicate to {}: {}", peer_id, e);
                    }
                }
            });
        }
    }

    /// Maybe create a snapshot.
    async fn maybe_create_snapshot(&mut self) {
        let node = self.node.read();
        let should_snapshot =
            node.last_applied() - self.last_snapshot_index > self.snapshot_threshold;

        if should_snapshot {
            let snapshot = node.fsm.snapshot();
            self.last_snapshot_index = snapshot.last_included_index;

            // In a real implementation, we would persist this.
            debug!(
                "Created snapshot at index {}",
                snapshot.last_included_index
            );
        }
    }
}

/// A cluster of Raft nodes for testing.
pub struct TestCluster {
    /// Nodes in the cluster.
    nodes: Vec<RaftClusterNode>,
    /// Handles for running node tasks.
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl TestCluster {
    /// Create a new test cluster with the given node configurations.
    pub fn new(configs: Vec<RaftConfig>, network: &mut crate::transport::MemoryNetwork) -> Self {
        let mut nodes = Vec::new();

        for config in configs {
            let node_id = config.id;
            let node = RaftNode::new(config);
            let transport = network.get_transport(node_id).expect("Transport not found");
            let receiver = network.take_receiver(node_id).expect("Receiver not found");

            nodes.push(RaftClusterNode::new(node, transport, receiver));
        }

        Self {
            nodes,
            handles: Vec::new(),
        }
    }

    /// Start all nodes.
    pub async fn start(&mut self) {
        for node in &mut self.nodes {
            let node_arc = node.node();
            let transport = Arc::clone(&node.transport);
            let receiver = std::mem::replace(
                &mut node.receiver,
                mpsc::channel(1).1, // Dummy receiver.
            );
            let running = Arc::clone(&node.running);

            *running.write() = true;

            let handle = tokio::spawn(async move {
                let mut cluster_node = RaftClusterNode {
                    node: node_arc,
                    transport,
                    receiver,
                    running,
                    snapshot_store: None,
                    last_snapshot_index: 0,
                    snapshot_threshold: 10000,
                };
                cluster_node.run().await;
            });

            self.handles.push(handle);
        }
    }

    /// Stop all nodes.
    pub async fn stop(&mut self) {
        for node in &self.nodes {
            node.stop();
        }

        for handle in self.handles.drain(..) {
            let _ = handle.await;
        }
    }

    /// Get a node by index.
    pub fn node(&self, index: usize) -> &RaftClusterNode {
        &self.nodes[index]
    }

    /// Get the leader node, if any.
    pub fn leader(&self) -> Option<&RaftClusterNode> {
        self.nodes.iter().find(|n| n.is_leader())
    }

    /// Wait for a leader to be elected.
    pub async fn wait_for_leader(&self, timeout: Duration) -> Result<NodeId> {
        let start = Instant::now();

        while start.elapsed() < timeout {
            if let Some(leader) = self.leader() {
                return Ok(leader.id());
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }

        Err(Error::Timeout)
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[RaftClusterNode] {
        &self.nodes
    }

    /// Get node count.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if cluster is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RaftConfigBuilder;
    use crate::transport::MemoryNetwork;

    fn create_test_configs(count: usize) -> Vec<RaftConfig> {
        (0..count as u64)
            .map(|id| {
                let mut builder = RaftConfigBuilder::default()
                    .id(id)
                    .listen_addr(format!("127.0.0.1:{}", 5000 + id))
                    .election_timeout(Duration::from_millis(150), Duration::from_millis(300))
                    .heartbeat_interval(Duration::from_millis(50));

                for peer_id in 0..count as u64 {
                    if peer_id != id {
                        builder = builder.peer(peer_id, format!("127.0.0.1:{}", 5000 + peer_id));
                    }
                }

                builder.build()
            })
            .collect()
    }

    #[tokio::test]
    async fn test_cluster_creation() {
        let configs = create_test_configs(3);
        let node_ids: Vec<_> = configs.iter().map(|c| c.id).collect();
        let mut network = MemoryNetwork::new(&node_ids);

        let cluster = TestCluster::new(configs, &mut network);
        assert_eq!(cluster.len(), 3);
    }
}
