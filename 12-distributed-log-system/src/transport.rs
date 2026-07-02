//! gRPC transport layer for distributed broker communication.
//!
//! This module provides network transport for client-broker and broker-broker
//! communication using gRPC/tonic.

use crate::log::{Header, Record, RecordBatch, TopicPartition};
use crate::protocol::{self, error_codes};
use crate::{Error, Offset, Result};

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
    tonic::include_proto!("kafka");
}

use proto::broker_service_client::BrokerServiceClient;
use proto::broker_service_server::{BrokerService, BrokerServiceServer};
use proto::group_coordinator_service_client::GroupCoordinatorServiceClient;
use proto::group_coordinator_service_server::{GroupCoordinatorService, GroupCoordinatorServiceServer};
use proto::replication_service_client::ReplicationServiceClient;
use proto::replication_service_server::{ReplicationService, ReplicationServiceServer};

/// Configuration for gRPC transport.
#[derive(Debug, Clone)]
pub struct TransportConfig {
    /// Connection timeout.
    pub connect_timeout: Duration,
    /// Request timeout.
    pub request_timeout: Duration,
    /// Keep-alive interval.
    pub keep_alive_interval: Duration,
    /// Maximum message size (default 10MB).
    pub max_message_size: usize,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            keep_alive_interval: Duration::from_secs(10),
            max_message_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// Broker client for communicating with remote brokers.
pub struct BrokerClient {
    /// Broker address.
    addr: SocketAddr,
    /// Configuration.
    config: TransportConfig,
    /// Cached channel.
    channel: RwLock<Option<Channel>>,
}

impl BrokerClient {
    /// Create a new broker client.
    pub fn new(addr: SocketAddr, config: TransportConfig) -> Self {
        Self {
            addr,
            config,
            channel: RwLock::new(None),
        }
    }

    /// Get or create a connection.
    async fn get_channel(&self) -> Result<Channel> {
        // Check for cached channel
        if let Some(channel) = self.channel.read().clone() {
            return Ok(channel);
        }

        // Create new connection
        let endpoint = Endpoint::from_shared(format!("http://{}", self.addr))
            .map_err(|e| Error::Internal(format!("Invalid endpoint: {}", e)))?
            .connect_timeout(self.config.connect_timeout)
            .timeout(self.config.request_timeout)
            .tcp_keepalive(Some(self.config.keep_alive_interval));

        let channel = endpoint
            .connect()
            .await
            .map_err(|e| Error::Internal(format!("Connection failed: {}", e)))?;

        // Cache the channel
        *self.channel.write() = Some(channel.clone());

        Ok(channel)
    }

    /// Send a produce request.
    pub async fn produce(
        &self,
        request: protocol::ProduceRequest,
    ) -> Result<protocol::ProduceResponse> {
        let channel = self.get_channel().await?;
        let mut client = BrokerServiceClient::new(channel);

        let proto_request = convert_produce_request(request);

        match client.produce(Request::new(proto_request)).await {
            Ok(response) => Ok(convert_produce_response(response.into_inner())),
            Err(status) => {
                *self.channel.write() = None; // Invalidate on error
                Err(Error::Internal(format!("Produce RPC failed: {}", status)))
            }
        }
    }

    /// Send a fetch request.
    pub async fn fetch(&self, request: protocol::FetchRequest) -> Result<protocol::FetchResponse> {
        let channel = self.get_channel().await?;
        let mut client = BrokerServiceClient::new(channel);

        let proto_request = convert_fetch_request(request);

        match client.fetch(Request::new(proto_request)).await {
            Ok(response) => Ok(convert_fetch_response(response.into_inner())),
            Err(status) => {
                *self.channel.write() = None;
                Err(Error::Internal(format!("Fetch RPC failed: {}", status)))
            }
        }
    }

    /// Get metadata.
    pub async fn metadata(
        &self,
        request: protocol::MetadataRequest,
    ) -> Result<protocol::MetadataResponse> {
        let channel = self.get_channel().await?;
        let mut client = BrokerServiceClient::new(channel);

        let proto_request = proto::MetadataRequest {
            topics: request.topics.unwrap_or_default(),
            allow_auto_topic_creation: request.allow_auto_topic_creation,
        };

        match client.metadata(Request::new(proto_request)).await {
            Ok(response) => Ok(convert_metadata_response(response.into_inner())),
            Err(status) => {
                *self.channel.write() = None;
                Err(Error::Internal(format!("Metadata RPC failed: {}", status)))
            }
        }
    }
}

/// Group coordinator client.
pub struct GroupCoordinatorClient {
    /// Coordinator address.
    addr: SocketAddr,
    /// Configuration.
    config: TransportConfig,
    /// Cached channel.
    channel: RwLock<Option<Channel>>,
}

impl GroupCoordinatorClient {
    /// Create a new group coordinator client.
    pub fn new(addr: SocketAddr, config: TransportConfig) -> Self {
        Self {
            addr,
            config,
            channel: RwLock::new(None),
        }
    }

    /// Get or create a connection.
    async fn get_channel(&self) -> Result<Channel> {
        if let Some(channel) = self.channel.read().clone() {
            return Ok(channel);
        }

        let endpoint = Endpoint::from_shared(format!("http://{}", self.addr))
            .map_err(|e| Error::Internal(format!("Invalid endpoint: {}", e)))?
            .connect_timeout(self.config.connect_timeout)
            .timeout(self.config.request_timeout);

        let channel = endpoint
            .connect()
            .await
            .map_err(|e| Error::Internal(format!("Connection failed: {}", e)))?;

        *self.channel.write() = Some(channel.clone());
        Ok(channel)
    }

    /// Join a consumer group.
    pub async fn join_group(
        &self,
        request: protocol::JoinGroupRequest,
    ) -> Result<protocol::JoinGroupResponse> {
        let channel = self.get_channel().await?;
        let mut client = GroupCoordinatorServiceClient::new(channel);

        let proto_request = proto::JoinGroupRequest {
            group_id: request.group_id,
            session_timeout_ms: request.session_timeout_ms,
            rebalance_timeout_ms: request.rebalance_timeout_ms,
            member_id: request.member_id,
            protocol_type: request.protocol_type,
            protocols: request
                .protocols
                .into_iter()
                .map(|p| proto::JoinGroupProtocol {
                    name: p.name,
                    metadata: p.metadata,
                })
                .collect(),
        };

        match client.join_group(Request::new(proto_request)).await {
            Ok(response) => {
                let resp = response.into_inner();
                Ok(protocol::JoinGroupResponse {
                    throttle_time_ms: resp.throttle_time_ms,
                    error_code: resp.error_code as i16,
                    generation_id: resp.generation_id,
                    protocol_name: resp.protocol_name,
                    leader: resp.leader,
                    member_id: resp.member_id,
                    members: resp
                        .members
                        .into_iter()
                        .map(|m| protocol::JoinGroupMember {
                            member_id: m.member_id,
                            metadata: m.metadata,
                        })
                        .collect(),
                })
            }
            Err(status) => {
                *self.channel.write() = None;
                Err(Error::Internal(format!("JoinGroup RPC failed: {}", status)))
            }
        }
    }

    /// Sync group.
    pub async fn sync_group(
        &self,
        request: protocol::SyncGroupRequest,
    ) -> Result<protocol::SyncGroupResponse> {
        let channel = self.get_channel().await?;
        let mut client = GroupCoordinatorServiceClient::new(channel);

        let proto_request = proto::SyncGroupRequest {
            group_id: request.group_id,
            generation_id: request.generation_id,
            member_id: request.member_id,
            assignments: request
                .assignments
                .into_iter()
                .map(|a| proto::SyncGroupAssignment {
                    member_id: a.member_id,
                    assignment: a.assignment,
                })
                .collect(),
        };

        match client.sync_group(Request::new(proto_request)).await {
            Ok(response) => {
                let resp = response.into_inner();
                Ok(protocol::SyncGroupResponse {
                    throttle_time_ms: resp.throttle_time_ms,
                    error_code: resp.error_code as i16,
                    assignment: resp.assignment,
                })
            }
            Err(status) => {
                *self.channel.write() = None;
                Err(Error::Internal(format!("SyncGroup RPC failed: {}", status)))
            }
        }
    }

    /// Send heartbeat.
    pub async fn heartbeat(
        &self,
        request: protocol::HeartbeatRequest,
    ) -> Result<protocol::HeartbeatResponse> {
        let channel = self.get_channel().await?;
        let mut client = GroupCoordinatorServiceClient::new(channel);

        let proto_request = proto::HeartbeatRequest {
            group_id: request.group_id,
            generation_id: request.generation_id,
            member_id: request.member_id,
        };

        match client.heartbeat(Request::new(proto_request)).await {
            Ok(response) => {
                let resp = response.into_inner();
                Ok(protocol::HeartbeatResponse {
                    throttle_time_ms: resp.throttle_time_ms,
                    error_code: resp.error_code as i16,
                })
            }
            Err(status) => {
                *self.channel.write() = None;
                Err(Error::Internal(format!("Heartbeat RPC failed: {}", status)))
            }
        }
    }

    /// Commit offsets.
    pub async fn offset_commit(
        &self,
        request: protocol::OffsetCommitRequest,
    ) -> Result<protocol::OffsetCommitResponse> {
        let channel = self.get_channel().await?;
        let mut client = GroupCoordinatorServiceClient::new(channel);

        let proto_request = proto::OffsetCommitRequest {
            group_id: request.group_id,
            generation_id: request.generation_id,
            member_id: request.member_id,
            topics: request
                .topics
                .into_iter()
                .map(|t| proto::OffsetCommitTopic {
                    topic: t.topic,
                    partitions: t
                        .partitions
                        .into_iter()
                        .map(|p| proto::OffsetCommitPartition {
                            partition: p.partition,
                            offset: p.offset,
                            metadata: p.metadata,
                        })
                        .collect(),
                })
                .collect(),
        };

        match client.offset_commit(Request::new(proto_request)).await {
            Ok(response) => {
                let resp = response.into_inner();
                Ok(protocol::OffsetCommitResponse {
                    throttle_time_ms: resp.throttle_time_ms,
                    topics: resp
                        .topics
                        .into_iter()
                        .map(|t| protocol::OffsetCommitTopicResponse {
                            topic: t.topic,
                            partitions: t
                                .partitions
                                .into_iter()
                                .map(|p| protocol::OffsetCommitPartitionResponse {
                                    partition: p.partition,
                                    error_code: p.error_code as i16,
                                })
                                .collect(),
                        })
                        .collect(),
                })
            }
            Err(status) => {
                *self.channel.write() = None;
                Err(Error::Internal(format!(
                    "OffsetCommit RPC failed: {}",
                    status
                )))
            }
        }
    }
}

/// Message handler trait for broker operations.
#[async_trait]
pub trait BrokerHandler: Send + Sync {
    /// Handle produce request.
    async fn handle_produce(
        &self,
        request: protocol::ProduceRequest,
    ) -> Result<protocol::ProduceResponse>;

    /// Handle fetch request.
    async fn handle_fetch(
        &self,
        request: protocol::FetchRequest,
    ) -> Result<protocol::FetchResponse>;

    /// Handle metadata request.
    async fn handle_metadata(
        &self,
        request: protocol::MetadataRequest,
    ) -> Result<protocol::MetadataResponse>;

    /// Handle list offsets request.
    async fn handle_list_offsets(
        &self,
        request: protocol::ListOffsetsRequest,
    ) -> Result<protocol::ListOffsetsResponse>;
}

/// gRPC server implementation for broker service.
pub struct BrokerGrpcServer<H: BrokerHandler> {
    handler: Arc<H>,
}

impl<H: BrokerHandler + 'static> BrokerGrpcServer<H> {
    /// Create a new broker gRPC server.
    pub fn new(handler: Arc<H>) -> Self {
        Self { handler }
    }
}

#[tonic::async_trait]
impl<H: BrokerHandler + 'static> BrokerService for BrokerGrpcServer<H> {
    async fn produce(
        &self,
        request: Request<proto::ProduceRequest>,
    ) -> std::result::Result<Response<proto::ProduceResponse>, Status> {
        let req = convert_proto_produce_request(request.into_inner());

        match self.handler.handle_produce(req).await {
            Ok(response) => Ok(Response::new(convert_produce_request_response(response))),
            Err(e) => Err(Status::internal(format!("Produce failed: {}", e))),
        }
    }

    async fn fetch(
        &self,
        request: Request<proto::FetchRequest>,
    ) -> std::result::Result<Response<proto::FetchResponse>, Status> {
        let req = convert_proto_fetch_request(request.into_inner());

        match self.handler.handle_fetch(req).await {
            Ok(response) => Ok(Response::new(convert_fetch_request_response(response))),
            Err(e) => Err(Status::internal(format!("Fetch failed: {}", e))),
        }
    }

    async fn metadata(
        &self,
        request: Request<proto::MetadataRequest>,
    ) -> std::result::Result<Response<proto::MetadataResponse>, Status> {
        let inner = request.into_inner();
        let req = protocol::MetadataRequest {
            topics: if inner.topics.is_empty() {
                None
            } else {
                Some(inner.topics)
            },
            allow_auto_topic_creation: inner.allow_auto_topic_creation,
        };

        match self.handler.handle_metadata(req).await {
            Ok(response) => Ok(Response::new(convert_metadata_request_response(response))),
            Err(e) => Err(Status::internal(format!("Metadata failed: {}", e))),
        }
    }

    async fn list_offsets(
        &self,
        request: Request<proto::ListOffsetsRequest>,
    ) -> std::result::Result<Response<proto::ListOffsetsResponse>, Status> {
        let proto_req = request.into_inner();
        let req = protocol::ListOffsetsRequest {
            replica_id: proto_req.replica_id,
            isolation_level: proto_req.isolation_level as i8,
            topics: proto_req
                .topics
                .into_iter()
                .map(|t| protocol::ListOffsetsTopicRequest {
                    topic: t.topic,
                    partitions: t
                        .partitions
                        .into_iter()
                        .map(|p| protocol::ListOffsetsPartitionRequest {
                            partition: p.partition,
                            timestamp: p.timestamp,
                        })
                        .collect(),
                })
                .collect(),
        };

        match self.handler.handle_list_offsets(req).await {
            Ok(response) => {
                let proto_resp = proto::ListOffsetsResponse {
                    throttle_time_ms: response.throttle_time_ms,
                    topics: response
                        .topics
                        .into_iter()
                        .map(|t| proto::ListOffsetsTopicResponse {
                            topic: t.topic,
                            partitions: t
                                .partitions
                                .into_iter()
                                .map(|p| proto::ListOffsetsPartitionResponse {
                                    partition: p.partition,
                                    error_code: p.error_code as i32,
                                    timestamp: p.timestamp,
                                    offset: p.offset,
                                })
                                .collect(),
                        })
                        .collect(),
                };
                Ok(Response::new(proto_resp))
            }
            Err(e) => Err(Status::internal(format!("ListOffsets failed: {}", e))),
        }
    }
}

/// Server builder for starting gRPC services.
pub struct KafkaServerBuilder {
    addr: SocketAddr,
}

impl KafkaServerBuilder {
    /// Create a new server builder.
    pub fn new(addr: SocketAddr) -> Self {
        Self { addr }
    }

    /// Start the server with a broker handler.
    pub async fn serve<H: BrokerHandler + 'static>(
        self,
        handler: Arc<H>,
    ) -> std::result::Result<(), tonic::transport::Error> {
        let broker_server = BrokerGrpcServer::new(handler);

        info!("Starting Kafka gRPC server on {}", self.addr);

        tonic::transport::Server::builder()
            .add_service(BrokerServiceServer::new(broker_server))
            .serve(self.addr)
            .await
    }

    /// Start with graceful shutdown.
    pub async fn serve_with_shutdown<H: BrokerHandler + 'static>(
        self,
        handler: Arc<H>,
        shutdown: impl std::future::Future<Output = ()>,
    ) -> std::result::Result<(), tonic::transport::Error> {
        let broker_server = BrokerGrpcServer::new(handler);

        info!("Starting Kafka gRPC server on {}", self.addr);

        tonic::transport::Server::builder()
            .add_service(BrokerServiceServer::new(broker_server))
            .serve_with_shutdown(self.addr, shutdown)
            .await
    }
}

// ============================================================================
// Conversion Functions
// ============================================================================

fn convert_produce_request(req: protocol::ProduceRequest) -> proto::ProduceRequest {
    proto::ProduceRequest {
        transactional_id: req.transactional_id,
        acks: req.acks as i32,
        timeout_ms: req.timeout_ms,
        topic_data: req
            .topic_data
            .into_iter()
            .map(|t| proto::TopicProduceData {
                topic: t.topic,
                partition_data: t
                    .partition_data
                    .into_iter()
                    .map(|p| proto::PartitionProduceData {
                        partition: p.partition,
                        records: Some(convert_record_batch(&p.records)),
                        first_sequence: 0,
                    })
                    .collect(),
            })
            .collect(),
        producer_id: -1,
        producer_epoch: -1,
    }
}

fn convert_proto_produce_request(req: proto::ProduceRequest) -> protocol::ProduceRequest {
    protocol::ProduceRequest {
        transactional_id: req.transactional_id,
        acks: req.acks as i16,
        timeout_ms: req.timeout_ms,
        topic_data: req
            .topic_data
            .into_iter()
            .map(|t| protocol::TopicProduceData {
                topic: t.topic,
                partition_data: t
                    .partition_data
                    .into_iter()
                    .map(|p| protocol::PartitionProduceData {
                        partition: p.partition,
                        records: convert_proto_record_batch(p.records),
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn convert_produce_response(resp: proto::ProduceResponse) -> protocol::ProduceResponse {
    protocol::ProduceResponse {
        responses: resp
            .responses
            .into_iter()
            .flat_map(|t| {
                t.partition_responses
                    .into_iter()
                    .map(|p| protocol::PartitionProduceResponse {
                        partition: p.partition,
                        error_code: p.error_code as i16,
                        base_offset: p.base_offset,
                        log_append_time: p.log_append_time,
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
        throttle_time_ms: resp.throttle_time_ms,
    }
}

fn convert_produce_request_response(resp: protocol::ProduceResponse) -> proto::ProduceResponse {
    proto::ProduceResponse {
        responses: vec![proto::TopicProduceResponse {
            topic: String::new(),
            partition_responses: resp
                .responses
                .into_iter()
                .map(|p| proto::PartitionProduceResponse {
                    partition: p.partition,
                    error_code: p.error_code as i32,
                    base_offset: p.base_offset,
                    log_append_time: p.log_append_time,
                })
                .collect(),
        }],
        throttle_time_ms: resp.throttle_time_ms,
    }
}

fn convert_fetch_request(req: protocol::FetchRequest) -> proto::FetchRequest {
    proto::FetchRequest {
        replica_id: req.replica_id,
        max_wait_ms: req.max_wait_ms,
        min_bytes: req.min_bytes,
        max_bytes: req.max_bytes,
        isolation_level: req.isolation_level as i32,
        session_id: req.session_id,
        session_epoch: req.session_epoch,
        topics: req
            .partitions
            .into_iter()
            .fold(HashMap::new(), |mut acc, p| {
                acc.entry(p.topic.clone())
                    .or_insert_with(Vec::new)
                    .push(proto::FetchPartition {
                        partition: p.partition,
                        fetch_offset: p.fetch_offset,
                        max_bytes: p.max_bytes,
                    });
                acc
            })
            .into_iter()
            .map(|(topic, partitions)| proto::FetchTopic { topic, partitions })
            .collect(),
    }
}

fn convert_proto_fetch_request(req: proto::FetchRequest) -> protocol::FetchRequest {
    protocol::FetchRequest {
        replica_id: req.replica_id,
        max_wait_ms: req.max_wait_ms,
        min_bytes: req.min_bytes,
        max_bytes: req.max_bytes,
        isolation_level: req.isolation_level as i8,
        session_id: req.session_id,
        session_epoch: req.session_epoch,
        partitions: req
            .topics
            .into_iter()
            .flat_map(|t| {
                t.partitions
                    .into_iter()
                    .map(move |p| protocol::FetchPartition {
                        topic: t.topic.clone(),
                        partition: p.partition,
                        fetch_offset: p.fetch_offset,
                        max_bytes: p.max_bytes,
                    })
            })
            .collect(),
    }
}

fn convert_fetch_response(resp: proto::FetchResponse) -> protocol::FetchResponse {
    protocol::FetchResponse {
        throttle_time_ms: resp.throttle_time_ms,
        error_code: resp.error_code as i16,
        session_id: resp.session_id,
        responses: resp
            .responses
            .into_iter()
            .flat_map(|t| {
                let topic = t.topic.clone();
                t.partitions
                    .into_iter()
                    .map(move |p| protocol::FetchPartitionResponse {
                        topic: topic.clone(),
                        partition: p.partition,
                        error_code: p.error_code as i16,
                        high_watermark: p.high_watermark,
                        last_stable_offset: p.last_stable_offset,
                        log_start_offset: p.log_start_offset,
                        record_batches: p
                            .record_batches
                            .into_iter()
                            .map(|b| convert_proto_record_batch(Some(b)))
                            .collect(),
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    }
}

fn convert_fetch_request_response(resp: protocol::FetchResponse) -> proto::FetchResponse {
    // Group partition responses back under their topic so a multi-topic fetch
    // round-trips correctly (preserving insertion order of topics).
    let mut topic_order: Vec<String> = Vec::new();
    let mut by_topic: std::collections::HashMap<String, Vec<proto::FetchPartitionResponse>> =
        std::collections::HashMap::new();

    for p in resp.responses {
        let converted = proto::FetchPartitionResponse {
            partition: p.partition,
            error_code: p.error_code as i32,
            high_watermark: p.high_watermark,
            last_stable_offset: p.last_stable_offset,
            log_start_offset: p.log_start_offset,
            record_batches: p
                .record_batches
                .into_iter()
                .map(|b| convert_record_batch(&b))
                .collect(),
        };
        if !by_topic.contains_key(&p.topic) {
            topic_order.push(p.topic.clone());
        }
        by_topic.entry(p.topic).or_default().push(converted);
    }

    proto::FetchResponse {
        throttle_time_ms: resp.throttle_time_ms,
        error_code: resp.error_code as i32,
        session_id: resp.session_id,
        responses: topic_order
            .into_iter()
            .map(|topic| proto::FetchTopicResponse {
                partitions: by_topic.remove(&topic).unwrap_or_default(),
                topic,
            })
            .collect(),
    }
}

fn convert_metadata_response(resp: proto::MetadataResponse) -> protocol::MetadataResponse {
    protocol::MetadataResponse {
        throttle_time_ms: resp.throttle_time_ms,
        brokers: resp
            .brokers
            .into_iter()
            .map(|b| protocol::BrokerMetadata {
                node_id: b.node_id,
                host: b.host,
                port: b.port,
                rack: b.rack,
            })
            .collect(),
        cluster_id: resp.cluster_id,
        controller_id: resp.controller_id,
        topics: resp
            .topics
            .into_iter()
            .map(|t| protocol::TopicMetadata {
                error_code: t.error_code as i16,
                name: t.name,
                is_internal: t.is_internal,
                partitions: t
                    .partitions
                    .into_iter()
                    .map(|p| protocol::PartitionMetadata {
                        error_code: p.error_code as i16,
                        partition: p.partition,
                        leader: p.leader,
                        leader_epoch: p.leader_epoch,
                        replicas: p.replicas,
                        isr: p.isr,
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn convert_metadata_request_response(resp: protocol::MetadataResponse) -> proto::MetadataResponse {
    proto::MetadataResponse {
        throttle_time_ms: resp.throttle_time_ms,
        brokers: resp
            .brokers
            .into_iter()
            .map(|b| proto::BrokerMetadata {
                node_id: b.node_id,
                host: b.host,
                port: b.port,
                rack: b.rack,
            })
            .collect(),
        cluster_id: resp.cluster_id,
        controller_id: resp.controller_id,
        topics: resp
            .topics
            .into_iter()
            .map(|t| proto::TopicMetadata {
                error_code: t.error_code as i32,
                name: t.name,
                is_internal: t.is_internal,
                partitions: t
                    .partitions
                    .into_iter()
                    .map(|p| proto::PartitionMetadata {
                        error_code: p.error_code as i32,
                        partition: p.partition,
                        leader: p.leader,
                        leader_epoch: p.leader_epoch,
                        replicas: p.replicas,
                        isr: p.isr,
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn convert_record_batch(batch: &RecordBatch) -> proto::RecordBatch {
    proto::RecordBatch {
        base_offset: batch.base_offset as i64,
        partition_leader_epoch: batch.partition_leader_epoch as i32,
        first_timestamp: batch.base_timestamp,
        max_timestamp: batch.max_timestamp,
        producer_id: batch.producer_id,
        producer_epoch: batch.producer_epoch as i32,
        first_sequence: batch.base_sequence as i32,
        records: batch
            .records
            .iter()
            .map(|r| proto::Record {
                attributes: r.attributes as i32,
                timestamp_delta: r.timestamp_delta,
                offset_delta: r.offset_delta as i32,
                key: r.key.clone(),
                value: r.value.clone(),
                headers: r
                    .headers
                    .iter()
                    .map(|h| proto::Header {
                        key: h.key.clone(),
                        value: h.value.clone(),
                    })
                    .collect(),
            })
            .collect(),
    }
}

fn convert_proto_record_batch(batch: Option<proto::RecordBatch>) -> RecordBatch {
    match batch {
        Some(b) => RecordBatch {
            base_offset: b.base_offset as u64,
            batch_length: 0,
            partition_leader_epoch: b.partition_leader_epoch as u32,
            magic: 2,
            crc: 0,
            attributes: 0,
            last_offset_delta: (b.records.len() as i32).saturating_sub(1) as u32,
            base_timestamp: b.first_timestamp,
            max_timestamp: b.max_timestamp,
            producer_id: b.producer_id,
            producer_epoch: b.producer_epoch as i16,
            base_sequence: b.first_sequence as u32,
            records: b
                .records
                .into_iter()
                .map(|r| Record {
                    attributes: r.attributes as u8,
                    timestamp_delta: r.timestamp_delta,
                    offset_delta: r.offset_delta as u32,
                    key: r.key,
                    value: r.value,
                    headers: r
                        .headers
                        .into_iter()
                        .map(|h| Header {
                            key: h.key,
                            value: h.value,
                        })
                        .collect(),
                })
                .collect(),
        },
        None => RecordBatch::new(0, Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transport_config_default() {
        let config = TransportConfig::default();
        assert_eq!(config.connect_timeout, Duration::from_secs(5));
        assert_eq!(config.request_timeout, Duration::from_secs(30));
        assert_eq!(config.max_message_size, 10 * 1024 * 1024);
    }

    #[test]
    fn test_broker_client_creation() {
        let addr: SocketAddr = "127.0.0.1:9092".parse().unwrap();
        let client = BrokerClient::new(addr, TransportConfig::default());
        assert_eq!(client.addr, addr);
    }

    #[test]
    fn test_group_coordinator_client_creation() {
        let addr: SocketAddr = "127.0.0.1:9092".parse().unwrap();
        let client = GroupCoordinatorClient::new(addr, TransportConfig::default());
        assert_eq!(client.addr, addr);
    }
}
