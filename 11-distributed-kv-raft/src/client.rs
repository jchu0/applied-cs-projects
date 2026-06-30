//! Client API for the Raft KV store.

use crate::error::{Error, Result};
use crate::node::Command;
use crate::rpc::{ClientRequest, ClientResponse};
use crate::NodeId;

use std::collections::HashMap;
use std::time::Duration;
use tracing::{debug, warn};

/// Address of a cluster node.
#[derive(Debug, Clone)]
pub struct NodeAddress {
    /// Node ID.
    pub id: NodeId,
    /// Network address.
    pub addr: String,
}

/// Client for the distributed KV store.
pub struct KVClient {
    /// Cluster node addresses.
    cluster: Vec<NodeAddress>,
    /// Current known leader.
    leader_id: Option<NodeId>,
    /// Request timeout.
    timeout: Duration,
    /// Maximum retries.
    max_retries: usize,
    /// Client ID for request deduplication.
    client_id: u64,
    /// Sequence number for requests.
    sequence: u64,
}

impl KVClient {
    /// Create a new client.
    pub fn new(cluster: Vec<NodeAddress>) -> Self {
        Self {
            cluster,
            leader_id: None,
            timeout: Duration::from_secs(5),
            max_retries: 3,
            client_id: rand::random(),
            sequence: 0,
        }
    }

    /// Set request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set maximum retries.
    pub fn with_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Put a key-value pair.
    pub async fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let request = ClientRequest::Put { key, value };
        match self.execute_with_retry(request).await? {
            ClientResponse::Success { .. } => Ok(()),
            ClientResponse::Error { message } => Err(Error::Internal(message)),
            ClientResponse::NotLeader { .. } => Err(Error::ClusterUnavailable),
        }
    }

    /// Get a value by key.
    pub async fn get(&mut self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let request = ClientRequest::Get { key: key.to_vec() };
        match self.execute_with_retry(request).await? {
            ClientResponse::Success { value } => Ok(value),
            ClientResponse::Error { message } => Err(Error::Internal(message)),
            ClientResponse::NotLeader { .. } => Err(Error::ClusterUnavailable),
        }
    }

    /// Delete a key.
    pub async fn delete(&mut self, key: &[u8]) -> Result<()> {
        let request = ClientRequest::Delete { key: key.to_vec() };
        match self.execute_with_retry(request).await? {
            ClientResponse::Success { .. } => Ok(()),
            ClientResponse::Error { message } => Err(Error::Internal(message)),
            ClientResponse::NotLeader { .. } => Err(Error::ClusterUnavailable),
        }
    }

    /// Execute a request with retry logic.
    async fn execute_with_retry(&mut self, request: ClientRequest) -> Result<ClientResponse> {
        let mut retries = self.max_retries;

        loop {
            let target = self.get_target();
            self.sequence += 1;

            match self.send_request(&target, request.clone()).await {
                Ok(resp) => {
                    match &resp {
                        ClientResponse::NotLeader { leader_hint } => {
                            self.leader_id = *leader_hint;
                            retries -= 1;
                        }
                        _ => return Ok(resp),
                    }
                }
                Err(e) => {
                    warn!("Request to {} failed: {}", target.addr, e);
                    self.leader_id = None;
                    retries -= 1;
                }
            }

            if retries == 0 {
                return Err(Error::ClusterUnavailable);
            }

            // Exponential backoff
            tokio::time::sleep(Duration::from_millis(100 * (self.max_retries - retries) as u64))
                .await;
        }
    }

    /// Get target node for request.
    fn get_target(&self) -> NodeAddress {
        if let Some(leader_id) = self.leader_id {
            if let Some(addr) = self.cluster.iter().find(|n| n.id == leader_id) {
                return addr.clone();
            }
        }

        // Round-robin if no known leader
        let idx = (self.sequence as usize) % self.cluster.len();
        self.cluster[idx].clone()
    }

    /// Send request to a node (stub - would use gRPC in real implementation).
    async fn send_request(
        &self,
        target: &NodeAddress,
        request: ClientRequest,
    ) -> Result<ClientResponse> {
        // In real implementation, this would use gRPC
        debug!("Sending {:?} to {}", request, target.addr);

        // Stub response
        Ok(ClientResponse::Success { value: None })
    }
}

/// Request tracker for deduplication.
pub struct RequestTracker {
    /// Completed requests: client_id -> (sequence, response).
    completed: HashMap<u64, (u64, ClientResponse)>,
    /// Maximum tracked sequences per client.
    max_per_client: usize,
}

impl RequestTracker {
    /// Create a new request tracker.
    pub fn new(max_per_client: usize) -> Self {
        Self {
            completed: HashMap::new(),
            max_per_client,
        }
    }

    /// Check if request was already completed.
    pub fn get_completed(&self, client_id: u64, sequence: u64) -> Option<&ClientResponse> {
        self.completed
            .get(&client_id)
            .filter(|(seq, _)| *seq == sequence)
            .map(|(_, resp)| resp)
    }

    /// Record a completed request.
    pub fn record_completed(&mut self, client_id: u64, sequence: u64, response: ClientResponse) {
        self.completed.insert(client_id, (sequence, response));
    }
}
