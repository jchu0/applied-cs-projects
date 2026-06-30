//! Distributed Key-Value Store with Raft Consensus
//!
//! A strongly consistent distributed key-value store built on the Raft consensus algorithm.
//! Provides linearizable read/write operations across a replicated cluster.

pub mod client;
pub mod cluster;
pub mod config;
pub mod error;
pub mod grpc;
pub mod metrics;
pub mod node;
pub mod rpc;
pub mod storage;
pub mod transport;

pub use client::{KVClient, NodeAddress, RequestTracker};
pub use config::{ClusterConfig, PeerConfig, RaftConfig, RaftConfigBuilder};
pub use error::{Error, Result};
pub use metrics::{
    ClusterMetrics, Counter, Gauge, HealthCheck, Histogram, MetricsSnapshot, PeerMetrics,
    RaftMetrics,
};
pub use node::{
    ApplyResult, Command, ConfigurationChange, EntryType, LogEntry, PendingRead, PendingWrite,
    PreVoteState, RaftNode, RaftState, SharedRaftNode,
};
pub use rpc::{
    AppendEntriesRequest, AppendEntriesResponse, ClientRequest, ClientResponse,
    InstallSnapshotRequest, InstallSnapshotResponse, RequestVoteRequest, RequestVoteResponse,
};
pub use storage::{KeyValueFSM, MemoryStorage, Snapshot, SnapshotStore, Storage, WriteAheadLog};
pub use transport::{MemoryNetwork, MemoryTransport, PeerState, PeerTracker, RaftMessage, Transport};
pub use cluster::{RaftClusterNode, TestCluster};
pub use grpc::{GrpcConfig, GrpcKvClient, GrpcServerBuilder, GrpcTransport, RpcMessage};

/// Node identifier type.
pub type NodeId = u64;

/// Term number type.
pub type Term = u64;

/// Log index type.
pub type LogIndex = u64;
