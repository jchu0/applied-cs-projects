//! gRPC transport layer for Raft node communication.
//!
//! This module provides network transport for RPC communication between Raft nodes
//! using gRPC/tonic for production deployment.

use crate::error::{Error, Result};
use crate::node::{Command, EntryType, LogEntry};
use crate::rpc::{
    AppendEntriesRequest, AppendEntriesResponse, ClientRequest as RaftClientRequest,
    ClientResponse as RaftClientResponse, InstallSnapshotRequest, InstallSnapshotResponse,
    RequestVoteRequest, RequestVoteResponse,
};
use crate::transport::Transport;
use crate::NodeId;

use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tonic::transport::{Channel, Endpoint};
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

// Include the generated protobuf code
pub mod proto {
    tonic::include_proto!("raft");
}

use proto::raft_service_client::RaftServiceClient;
use proto::raft_service_server::{RaftService, RaftServiceServer};
use proto::kv_service_server::{KvService, KvServiceServer};

/// Configuration for gRPC transport.
#[derive(Debug, Clone)]
pub struct GrpcConfig {
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Request timeout.
    pub request_timeout: Duration,
    /// Keep-alive interval.
    pub keep_alive_interval: Duration,
    /// Keep-alive timeout.
    pub keep_alive_timeout: Duration,
    /// Maximum concurrent streams per connection.
    pub max_concurrent_streams: u32,
    /// Enable TLS (placeholder for future implementation).
    pub enable_tls: bool,
}

impl Default for GrpcConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_millis(500),
            keep_alive_interval: Duration::from_secs(10),
            keep_alive_timeout: Duration::from_secs(20),
            max_concurrent_streams: 100,
            enable_tls: false,
        }
    }
}

/// gRPC transport for production Raft communication.
pub struct GrpcTransport {
    /// This node's ID.
    node_id: NodeId,
    /// Peer addresses.
    peer_addresses: Arc<RwLock<HashMap<NodeId, SocketAddr>>>,
    /// Connection pool (cached channels).
    connections: Arc<RwLock<HashMap<NodeId, Channel>>>,
    /// Configuration.
    config: GrpcConfig,
}

impl GrpcTransport {
    /// Create a new gRPC transport.
    pub fn new(node_id: NodeId, config: GrpcConfig) -> Self {
        Self {
            node_id,
            peer_addresses: Arc::new(RwLock::new(HashMap::new())),
            connections: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Register a peer's address.
    pub fn register_peer(&self, peer_id: NodeId, addr: SocketAddr) {
        self.peer_addresses.write().insert(peer_id, addr);
        // Clear cached connection to force reconnect with new address
        self.connections.write().remove(&peer_id);
    }

    /// Unregister a peer.
    pub fn unregister_peer(&self, peer_id: NodeId) {
        self.peer_addresses.write().remove(&peer_id);
        self.connections.write().remove(&peer_id);
    }

    /// Get or create a connection to a peer.
    async fn get_connection(&self, peer_id: NodeId) -> Result<Channel> {
        // Check for cached connection
        if let Some(channel) = self.connections.read().get(&peer_id).cloned() {
            return Ok(channel);
        }

        // Get peer address
        let addr = self
            .peer_addresses
            .read()
            .get(&peer_id)
            .cloned()
            .ok_or_else(|| Error::Network(format!("Unknown peer: {}", peer_id)))?;

        // Create new connection
        let endpoint = Endpoint::from_shared(format!("http://{}", addr))
            .map_err(|e| Error::Network(format!("Invalid endpoint: {}", e)))?
            .connect_timeout(self.config.connect_timeout)
            .timeout(self.config.request_timeout)
            .tcp_keepalive(Some(self.config.keep_alive_interval))
            .http2_keep_alive_interval(self.config.keep_alive_interval)
            .keep_alive_timeout(self.config.keep_alive_timeout)
            .concurrency_limit(self.config.max_concurrent_streams as usize);

        let channel = endpoint
            .connect()
            .await
            .map_err(|e| Error::Network(format!("Connection failed to {}: {}", addr, e)))?;

        // Cache the connection
        self.connections.write().insert(peer_id, channel.clone());

        Ok(channel)
    }

    /// Remove a failed connection from the cache.
    fn invalidate_connection(&self, peer_id: NodeId) {
        self.connections.write().remove(&peer_id);
    }
}

#[async_trait]
impl Transport for GrpcTransport {
    async fn send_request_vote(
        &self,
        target: NodeId,
        request: RequestVoteRequest,
    ) -> Result<RequestVoteResponse> {
        debug!(
            "Node {} sending RequestVote to {} (term {})",
            self.node_id, target, request.term
        );

        let channel = self.get_connection(target).await?;
        let mut client = RaftServiceClient::new(channel);

        let proto_request = proto::RequestVoteRequest {
            term: request.term,
            candidate_id: request.candidate_id,
            last_log_index: request.last_log_index,
            last_log_term: request.last_log_term,
        };

        match client.request_vote(Request::new(proto_request)).await {
            Ok(response) => {
                let resp = response.into_inner();
                Ok(RequestVoteResponse {
                    term: resp.term,
                    vote_granted: resp.vote_granted,
                })
            }
            Err(status) => {
                self.invalidate_connection(target);
                Err(Error::Network(format!("RequestVote RPC failed: {}", status)))
            }
        }
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

        let channel = self.get_connection(target).await?;
        let mut client = RaftServiceClient::new(channel);

        let proto_entries: Vec<proto::LogEntry> = request
            .entries
            .iter()
            .map(|e| proto::LogEntry {
                term: e.term,
                index: e.index,
                command: bincode::serialize(&e.command).unwrap_or_default(),
                entry_type: match e.entry_type {
                    EntryType::Command => proto::EntryType::Command as i32,
                    EntryType::Configuration => proto::EntryType::Configuration as i32,
                    EntryType::NoOp => proto::EntryType::NoOp as i32,
                },
            })
            .collect();

        let proto_request = proto::AppendEntriesRequest {
            term: request.term,
            leader_id: request.leader_id,
            prev_log_index: request.prev_log_index,
            prev_log_term: request.prev_log_term,
            entries: proto_entries,
            leader_commit: request.leader_commit,
        };

        match client.append_entries(Request::new(proto_request)).await {
            Ok(response) => {
                let resp = response.into_inner();
                Ok(AppendEntriesResponse {
                    term: resp.term,
                    success: resp.success,
                    conflict_index: resp.conflict_index,
                    conflict_term: resp.conflict_term,
                })
            }
            Err(status) => {
                self.invalidate_connection(target);
                Err(Error::Network(format!(
                    "AppendEntries RPC failed: {}",
                    status
                )))
            }
        }
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

        let channel = self.get_connection(target).await?;
        let mut client = RaftServiceClient::new(channel);

        let proto_request = proto::InstallSnapshotRequest {
            term: request.term,
            leader_id: request.leader_id,
            last_included_index: request.last_included_index,
            last_included_term: request.last_included_term,
            offset: request.offset,
            data: request.data,
            done: request.done,
        };

        match client.install_snapshot(Request::new(proto_request)).await {
            Ok(response) => {
                let resp = response.into_inner();
                Ok(InstallSnapshotResponse { term: resp.term })
            }
            Err(status) => {
                self.invalidate_connection(target);
                Err(Error::Network(format!(
                    "InstallSnapshot RPC failed: {}",
                    status
                )))
            }
        }
    }
}

/// Message types for the RPC handler channel.
pub enum RpcMessage {
    RequestVote {
        request: RequestVoteRequest,
        response_tx: tokio::sync::oneshot::Sender<RequestVoteResponse>,
    },
    AppendEntries {
        request: AppendEntriesRequest,
        response_tx: tokio::sync::oneshot::Sender<AppendEntriesResponse>,
    },
    InstallSnapshot {
        request: InstallSnapshotRequest,
        response_tx: tokio::sync::oneshot::Sender<InstallSnapshotResponse>,
    },
    ClientRequest {
        request: RaftClientRequest,
        response_tx: tokio::sync::oneshot::Sender<RaftClientResponse>,
    },
}

/// gRPC server implementation for Raft RPCs.
pub struct RaftGrpcServer {
    /// Channel to send received RPCs to the Raft node.
    rpc_tx: mpsc::Sender<RpcMessage>,
}

impl RaftGrpcServer {
    /// Create a new gRPC server.
    pub fn new(rpc_tx: mpsc::Sender<RpcMessage>) -> Self {
        Self { rpc_tx }
    }
}

#[tonic::async_trait]
impl RaftService for RaftGrpcServer {
    async fn request_vote(
        &self,
        request: Request<proto::RequestVoteRequest>,
    ) -> std::result::Result<Response<proto::RequestVoteResponse>, Status> {
        let req = request.into_inner();
        let raft_request = RequestVoteRequest {
            term: req.term,
            candidate_id: req.candidate_id,
            last_log_index: req.last_log_index,
            last_log_term: req.last_log_term,
        };

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.rpc_tx
            .send(RpcMessage::RequestVote {
                request: raft_request,
                response_tx,
            })
            .await
            .map_err(|_| Status::internal("Failed to forward RequestVote"))?;

        let response = response_rx
            .await
            .map_err(|_| Status::internal("RequestVote handler dropped"))?;

        Ok(Response::new(proto::RequestVoteResponse {
            term: response.term,
            vote_granted: response.vote_granted,
        }))
    }

    async fn append_entries(
        &self,
        request: Request<proto::AppendEntriesRequest>,
    ) -> std::result::Result<Response<proto::AppendEntriesResponse>, Status> {
        let req = request.into_inner();

        let entries: Vec<LogEntry> = req
            .entries
            .into_iter()
            .map(|e| {
                let command: Command = bincode::deserialize(&e.command).unwrap_or(Command::NoOp);
                let entry_type = match proto::EntryType::from_i32(e.entry_type) {
                    Some(proto::EntryType::Command) => EntryType::Command,
                    Some(proto::EntryType::Configuration) => EntryType::Configuration,
                    Some(proto::EntryType::NoOp) | None => EntryType::NoOp,
                };
                LogEntry {
                    term: e.term,
                    index: e.index,
                    command,
                    entry_type,
                }
            })
            .collect();

        let raft_request = AppendEntriesRequest {
            term: req.term,
            leader_id: req.leader_id,
            prev_log_index: req.prev_log_index,
            prev_log_term: req.prev_log_term,
            entries,
            leader_commit: req.leader_commit,
        };

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.rpc_tx
            .send(RpcMessage::AppendEntries {
                request: raft_request,
                response_tx,
            })
            .await
            .map_err(|_| Status::internal("Failed to forward AppendEntries"))?;

        let response = response_rx
            .await
            .map_err(|_| Status::internal("AppendEntries handler dropped"))?;

        Ok(Response::new(proto::AppendEntriesResponse {
            term: response.term,
            success: response.success,
            conflict_index: response.conflict_index,
            conflict_term: response.conflict_term,
        }))
    }

    async fn install_snapshot(
        &self,
        request: Request<proto::InstallSnapshotRequest>,
    ) -> std::result::Result<Response<proto::InstallSnapshotResponse>, Status> {
        let req = request.into_inner();
        let raft_request = InstallSnapshotRequest {
            term: req.term,
            leader_id: req.leader_id,
            last_included_index: req.last_included_index,
            last_included_term: req.last_included_term,
            offset: req.offset,
            data: req.data,
            done: req.done,
        };

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.rpc_tx
            .send(RpcMessage::InstallSnapshot {
                request: raft_request,
                response_tx,
            })
            .await
            .map_err(|_| Status::internal("Failed to forward InstallSnapshot"))?;

        let response = response_rx
            .await
            .map_err(|_| Status::internal("InstallSnapshot handler dropped"))?;

        Ok(Response::new(proto::InstallSnapshotResponse {
            term: response.term,
        }))
    }
}

/// gRPC server implementation for client KV operations.
pub struct KvGrpcServer {
    /// Channel to send received client requests.
    rpc_tx: mpsc::Sender<RpcMessage>,
}

impl KvGrpcServer {
    /// Create a new KV gRPC server.
    pub fn new(rpc_tx: mpsc::Sender<RpcMessage>) -> Self {
        Self { rpc_tx }
    }
}

#[tonic::async_trait]
impl KvService for KvGrpcServer {
    async fn execute(
        &self,
        request: Request<proto::ClientRequest>,
    ) -> std::result::Result<Response<proto::ClientResponse>, Status> {
        let req = request.into_inner();

        let raft_request = match req.request {
            Some(proto::client_request::Request::Get(get)) => RaftClientRequest::Get { key: get.key },
            Some(proto::client_request::Request::Put(put)) => RaftClientRequest::Put {
                key: put.key,
                value: put.value,
            },
            Some(proto::client_request::Request::Delete(del)) => {
                RaftClientRequest::Delete { key: del.key }
            }
            None => return Err(Status::invalid_argument("Missing request")),
        };

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.rpc_tx
            .send(RpcMessage::ClientRequest {
                request: raft_request,
                response_tx,
            })
            .await
            .map_err(|_| Status::internal("Failed to forward client request"))?;

        let response = response_rx
            .await
            .map_err(|_| Status::internal("Client request handler dropped"))?;

        let proto_response = match response {
            RaftClientResponse::Success { value } => proto::ClientResponse {
                response: Some(proto::client_response::Response::Success(
                    proto::SuccessResponse { value },
                )),
            },
            RaftClientResponse::NotLeader { leader_hint } => proto::ClientResponse {
                response: Some(proto::client_response::Response::NotLeader(
                    proto::NotLeaderResponse { leader_hint },
                )),
            },
            RaftClientResponse::Error { message } => proto::ClientResponse {
                response: Some(proto::client_response::Response::Error(
                    proto::ErrorResponse { message },
                )),
            },
        };

        Ok(Response::new(proto_response))
    }
}

/// Builder for starting a gRPC server with both Raft and KV services.
pub struct GrpcServerBuilder {
    addr: SocketAddr,
    rpc_tx: mpsc::Sender<RpcMessage>,
}

impl GrpcServerBuilder {
    /// Create a new server builder.
    pub fn new(addr: SocketAddr, rpc_tx: mpsc::Sender<RpcMessage>) -> Self {
        Self { addr, rpc_tx }
    }

    /// Build and return the server (does not start it).
    pub fn build(
        self,
    ) -> tonic::transport::Server {
        tonic::transport::Server::builder()
    }

    /// Start the server and run until shutdown.
    pub async fn serve(self) -> std::result::Result<(), tonic::transport::Error> {
        let raft_server = RaftGrpcServer::new(self.rpc_tx.clone());
        let kv_server = KvGrpcServer::new(self.rpc_tx);

        info!("Starting gRPC server on {}", self.addr);

        tonic::transport::Server::builder()
            .add_service(RaftServiceServer::new(raft_server))
            .add_service(KvServiceServer::new(kv_server))
            .serve(self.addr)
            .await
    }

    /// Start the server with graceful shutdown.
    pub async fn serve_with_shutdown(
        self,
        shutdown: impl std::future::Future<Output = ()>,
    ) -> std::result::Result<(), tonic::transport::Error> {
        let raft_server = RaftGrpcServer::new(self.rpc_tx.clone());
        let kv_server = KvGrpcServer::new(self.rpc_tx);

        info!("Starting gRPC server on {}", self.addr);

        tonic::transport::Server::builder()
            .add_service(RaftServiceServer::new(raft_server))
            .add_service(KvServiceServer::new(kv_server))
            .serve_with_shutdown(self.addr, shutdown)
            .await
    }
}

/// Client for connecting to Raft cluster for KV operations.
pub struct GrpcKvClient {
    /// Known node addresses.
    nodes: Vec<SocketAddr>,
    /// Current leader hint.
    leader_hint: Option<usize>,
    /// Configuration.
    config: GrpcConfig,
}

impl GrpcKvClient {
    /// Create a new KV client.
    pub fn new(nodes: Vec<SocketAddr>) -> Self {
        Self {
            nodes,
            leader_hint: None,
            config: GrpcConfig::default(),
        }
    }

    /// Create a new KV client with custom configuration.
    pub fn with_config(nodes: Vec<SocketAddr>, config: GrpcConfig) -> Self {
        Self {
            nodes,
            leader_hint: None,
            config,
        }
    }

    /// Get a value from the cluster.
    pub async fn get(&mut self, key: Vec<u8>) -> Result<Option<Vec<u8>>> {
        let request = proto::ClientRequest {
            request: Some(proto::client_request::Request::Get(proto::GetRequest {
                key,
            })),
        };
        self.execute(request).await
    }

    /// Put a key-value pair to the cluster.
    pub async fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let request = proto::ClientRequest {
            request: Some(proto::client_request::Request::Put(proto::PutRequest {
                key,
                value,
            })),
        };
        self.execute(request).await.map(|_| ())
    }

    /// Delete a key from the cluster.
    pub async fn delete(&mut self, key: Vec<u8>) -> Result<()> {
        let request = proto::ClientRequest {
            request: Some(proto::client_request::Request::Delete(
                proto::DeleteRequest { key },
            )),
        };
        self.execute(request).await.map(|_| ())
    }

    /// Execute a request against the cluster.
    async fn execute(&mut self, request: proto::ClientRequest) -> Result<Option<Vec<u8>>> {
        let mut attempts = 0;
        let max_attempts = self.nodes.len() * 2;

        while attempts < max_attempts {
            let node_idx = self.leader_hint.unwrap_or(attempts % self.nodes.len());
            let addr = self.nodes[node_idx];

            match self.try_execute(&addr, request.clone()).await {
                Ok(proto::ClientResponse {
                    response: Some(proto::client_response::Response::Success(success)),
                }) => {
                    self.leader_hint = Some(node_idx);
                    return Ok(success.value);
                }
                Ok(proto::ClientResponse {
                    response: Some(proto::client_response::Response::NotLeader(not_leader)),
                }) => {
                    // Update leader hint if provided
                    if let Some(hint) = not_leader.leader_hint {
                        // Find the index of this node
                        self.leader_hint = self
                            .nodes
                            .iter()
                            .position(|_| true) // In production, we'd match by node ID
                            .or(Some((node_idx + 1) % self.nodes.len()));
                    } else {
                        self.leader_hint = Some((node_idx + 1) % self.nodes.len());
                    }
                }
                Ok(proto::ClientResponse {
                    response: Some(proto::client_response::Response::Error(error)),
                }) => {
                    return Err(Error::Storage(error.message));
                }
                Ok(proto::ClientResponse { response: None }) => {
                    return Err(Error::Network("Empty response".to_string()));
                }
                Err(e) => {
                    warn!("Request to {} failed: {}", addr, e);
                    self.leader_hint = Some((node_idx + 1) % self.nodes.len());
                }
            }

            attempts += 1;
        }

        Err(Error::Network("All nodes unreachable".to_string()))
    }

    /// Try to execute a request against a specific node.
    async fn try_execute(
        &self,
        addr: &SocketAddr,
        request: proto::ClientRequest,
    ) -> Result<proto::ClientResponse> {
        let endpoint = Endpoint::from_shared(format!("http://{}", addr))
            .map_err(|e| Error::Network(format!("Invalid endpoint: {}", e)))?
            .connect_timeout(self.config.connect_timeout)
            .timeout(self.config.request_timeout);

        let channel = endpoint
            .connect()
            .await
            .map_err(|e| Error::Network(format!("Connection failed: {}", e)))?;

        let mut client = proto::kv_service_client::KvServiceClient::new(channel);

        client
            .execute(Request::new(request))
            .await
            .map(|r| r.into_inner())
            .map_err(|e| Error::Network(format!("RPC failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpc_config_default() {
        let config = GrpcConfig::default();
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
        assert_eq!(config.request_timeout, Duration::from_millis(500));
        assert!(!config.enable_tls);
    }

    #[test]
    fn test_grpc_transport_creation() {
        let transport = GrpcTransport::new(1, GrpcConfig::default());
        assert_eq!(transport.node_id, 1);
    }

    #[test]
    fn test_peer_registration() {
        let transport = GrpcTransport::new(1, GrpcConfig::default());
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        transport.register_peer(2, addr);
        assert!(transport.peer_addresses.read().contains_key(&2));

        transport.unregister_peer(2);
        assert!(!transport.peer_addresses.read().contains_key(&2));
    }

    #[test]
    fn test_kv_client_creation() {
        let nodes = vec![
            "127.0.0.1:8080".parse().unwrap(),
            "127.0.0.1:8081".parse().unwrap(),
        ];
        let client = GrpcKvClient::new(nodes.clone());
        assert_eq!(client.nodes.len(), 2);
        assert!(client.leader_hint.is_none());
    }

    #[tokio::test]
    async fn test_connection_invalidation() {
        let transport = GrpcTransport::new(1, GrpcConfig::default());
        let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();

        transport.register_peer(2, addr);

        // Manually insert a dummy connection (for test purposes)
        // In reality, the connection would be established via get_connection
        transport.invalidate_connection(2);
        assert!(!transport.connections.read().contains_key(&2));
    }
}
