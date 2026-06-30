# HDFS-Like Distributed File System Architecture

## Overview

This project implements a distributed file system inspired by the Hadoop Distributed File System (HDFS). It provides reliable, scalable storage for large files across a cluster of commodity hardware.

## System Architecture

### High-Level Design

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│   NameNode  │────▶│  DataNodes  │
│  (HDFS API) │     │  (Metadata) │     │   (Storage) │
└─────────────┘     └─────────────┘     └─────────────┘
       │                    │                    │
       │                    │                    │
       └────────────────────┴────────────────────┘
              TCP/IP Communication
```

### Core Components

#### 1. NameNode (Master)

The NameNode is the centerpiece of the HDFS architecture:

- **Namespace Management**: Maintains the filesystem tree and metadata for all files and directories
- **Block Management**: Maps files to blocks and blocks to DataNodes
- **Replication Management**: Ensures blocks meet their target replication factor
- **DataNode Coordination**: Handles registration, heartbeats, and block reports from DataNodes

**Key Data Structures:**
- `_files`: Dictionary mapping file paths to FileInfo objects
- `_directories`: Dictionary mapping directory paths to DirectoryInfo objects
- `_blocks`: Dictionary mapping block IDs to Block objects
- `_block_to_nodes`: Dictionary mapping block IDs to sets of DataNode IDs
- `_datanodes`: Dictionary of registered DataNode information

**Critical Operations:**
- File creation/deletion
- Block allocation
- Replication monitoring
- Safe mode management
- Checkpointing

#### 2. DataNode (Worker)

DataNodes provide actual storage:

- **Block Storage**: Stores and retrieves data blocks on local filesystem
- **Block Reporting**: Periodically sends list of blocks to NameNode
- **Heartbeat**: Sends regular heartbeats to indicate aliveness
- **Pipeline Replication**: Participates in write pipelines for replication

**Key Features:**
- Local filesystem-based storage
- Block corruption detection
- Background block scanning
- Bandwidth throttling support

#### 3. Client

The client library provides the user-facing API:

- **File Operations**: Create, read, write, delete files
- **Directory Operations**: Create, list, delete directories
- **Stream Processing**: Supports streaming reads/writes for large files
- **Retry Logic**: Automatic retry on DataNode failures

**Key Features:**
- Transparent block management
- Location-aware reads
- Parallel block operations
- Client-side caching (optional)

## Data Flow

### Write Operation

1. **Client** requests file creation from **NameNode**
2. **NameNode** creates file metadata and allocates blocks
3. **NameNode** returns block IDs and DataNode locations
4. **Client** writes data directly to **DataNodes**
5. **DataNodes** replicate blocks in pipeline fashion
6. **DataNodes** report block storage to **NameNode**

```
Client ──1──▶ NameNode
       ◀──2── (block allocation)
       ──3──▶ DataNode1 ──▶ DataNode2 ──▶ DataNode3
       ◀──4── (acknowledgments)
```

### Read Operation

1. **Client** requests file info from **NameNode**
2. **NameNode** returns block locations
3. **Client** reads blocks directly from **DataNodes**
4. **Client** assembles blocks into complete file

```
Client ──1──▶ NameNode
       ◀──2── (block locations)
       ──3──▶ DataNode (parallel reads)
       ◀──4── (block data)
```

## Fault Tolerance

### NameNode High Availability

- **Checkpointing**: Periodic snapshots of namespace and block mappings
- **Edit Log**: Write-ahead log for all metadata changes
- **Secondary NameNode**: (Future) Standby node for failover

### DataNode Failure Handling

- **Heartbeat Monitoring**: Detect failed nodes via missing heartbeats
- **Block Re-replication**: Automatically create new replicas when nodes fail
- **Rack Awareness**: (Future) Place replicas across different racks

### Data Integrity

- **Block Checksums**: (Optional) Verify data integrity on read
- **Block Scanner**: Background process to detect corruption
- **Corruption Reporting**: Report corrupted blocks to NameNode

## Replication Strategy

### Default Policy

- **Replication Factor**: Default is 3 replicas per block
- **Placement Strategy**:
  1. First replica on local node (if applicable)
  2. Second replica on different node
  3. Third replica on another different node

### Under-Replication Handling

1. NameNode continuously monitors replication levels
2. Detects under-replicated blocks during block reports
3. Schedules re-replication on available DataNodes
4. Prioritizes based on:
   - Current replication level
   - Block age
   - File priority

## Network Protocol

### Message Types

All communication uses a custom protocol with these message types:

- **Control Messages**:
  - `REGISTER_DATANODE`: DataNode registration
  - `HEARTBEAT`: Liveness check
  - `BLOCK_REPORT`: List of stored blocks

- **File Operations**:
  - `CREATE_FILE`: Create new file
  - `DELETE_FILE`: Remove file
  - `GET_FILE_INFO`: Retrieve metadata

- **Block Operations**:
  - `ALLOCATE_BLOCKS`: Request new blocks
  - `GET_BLOCK_LOCATIONS`: Find block replicas
  - `REPORT_BAD_BLOCKS`: Report corruption

### Serialization

- JSON-based message serialization
- Length-prefixed messages (4-byte header)
- TCP sockets for all communication

## Performance Optimizations

### Client-Side

- **Parallel Block Operations**: Read/write multiple blocks concurrently
- **Location-Aware Reads**: Prefer local replicas when available
- **Client Caching**: Optional metadata caching to reduce NameNode load
- **Streaming API**: Process large files without loading into memory

### NameNode

- **In-Memory Metadata**: All namespace kept in RAM for fast access
- **Batch Processing**: Group operations where possible
- **Lazy Deletion**: Mark for deletion, clean up asynchronously

### DataNode

- **Direct I/O**: Bypass system cache for large transfers
- **Pipeline Writes**: Overlap network and disk I/O
- **Background Tasks**: Non-blocking corruption scanning

## Scalability Considerations

### Horizontal Scaling

- **DataNodes**: Add nodes to increase storage capacity
- **Blocks**: Fixed-size blocks enable easy distribution
- **Parallel Operations**: Multiple clients can operate simultaneously

### Limitations

- **Single NameNode**: Current bottleneck for metadata operations
- **Memory Constraints**: NameNode memory limits total files/blocks
- **Network Bandwidth**: Replication can consume significant bandwidth

### Future Enhancements

1. **NameNode Federation**: Multiple NameNodes for namespace partitioning
2. **Erasure Coding**: Reduce storage overhead while maintaining reliability
3. **Tiered Storage**: Support for SSD/HDD/Archive tiers
4. **Small File Optimization**: Pack small files into containers

## Security Model

### Current Implementation

- **Basic Authentication**: Node ID-based identification
- **Network Isolation**: Assumes trusted network environment

### Future Security Features

1. **Kerberos Integration**: Strong authentication
2. **Wire Encryption**: TLS for all communication
3. **Access Control Lists**: File/directory permissions
4. **Audit Logging**: Track all operations

## Configuration Parameters

### NameNode Configuration

```python
default_replication = 3          # Target replication factor
default_block_size = 128MB       # Default block size
heartbeat_interval = 3s          # Expected heartbeat frequency
checkpoint_interval = 3600s      # Checkpoint frequency
safe_mode_threshold = 0.999      # Block report threshold
```

### DataNode Configuration

```python
data_dir = "/var/hdfs/data"     # Local storage directory
heartbeat_interval = 3s         # Heartbeat frequency
block_report_interval = 3600s   # Full block report frequency
max_bandwidth = None            # Bandwidth limit (bytes/sec)
```

### Client Configuration

```python
block_size = 128MB              # Block size for new files
replication = 3                 # Replication factor
retry_count = 3                 # Number of retry attempts
cache_ttl = 60s                # Metadata cache timeout
```

## Monitoring and Metrics

### Key Metrics

- **System Health**:
  - Number of live/dead DataNodes
  - Total/used/remaining capacity
  - Under-replicated blocks

- **Performance**:
  - Read/write throughput
  - Operation latency
  - Queue depths

- **Operations**:
  - Files/directories created/deleted
  - Blocks allocated/deleted
  - Replication operations

### Health Checks

1. **NameNode Health**: Check responsiveness and memory usage
2. **DataNode Health**: Monitor disk space and corruption rate
3. **Network Health**: Track connection failures and latency

## Deployment Architecture

### Recommended Setup

```
┌─────────────────────────────────────┐
│          Load Balancer              │
└─────────────────────────────────────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
┌─────────┐ ┌─────────┐ ┌─────────┐
│ Client1 │ │ Client2 │ │ Client3 │
└─────────┘ └─────────┘ └─────────┘
     │            │            │
     └────────────┼────────────┘
                  │
          ┌───────────┐
          │ NameNode  │
          └───────────┘
                  │
     ┌────────────┼────────────┐
     │            │            │
┌──────────┐ ┌──────────┐ ┌──────────┐
│DataNode1 │ │DataNode2 │ │DataNode3 │
└──────────┘ └──────────┘ └──────────┘
```

### Hardware Requirements

**NameNode:**
- CPU: 4+ cores
- RAM: 8GB minimum (1GB per million files)
- Disk: SSD preferred for edit logs
- Network: 1Gbps minimum

**DataNode:**
- CPU: 2+ cores
- RAM: 4GB minimum
- Disk: Large capacity HDDs
- Network: 1Gbps minimum

## Testing Architecture

### Unit Testing
- Component isolation with mocks
- Data structure validation
- Protocol testing

### Integration Testing
- Multi-node cluster setup
- End-to-end scenarios
- Failure injection

### Performance Testing
- Throughput benchmarks
- Scalability tests
- Stress testing