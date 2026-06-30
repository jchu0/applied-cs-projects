# HDFS API Documentation

## Client API

The HDFS client provides a high-level interface for interacting with the distributed file system.

### Initialization

```python
from hdfs.client import HDFSClient

# Create a client instance
client = HDFSClient(
    namenode_host="localhost",      # NameNode hostname
    namenode_port=9000,             # NameNode port
    block_size=128 * 1024 * 1024,   # Default block size (128MB)
    replication=3                    # Default replication factor
)
```

### File Operations

#### Create File

Create a new file in HDFS.

```python
async def create(
    path: str,
    replication: Optional[int] = None,
    block_size: Optional[int] = None,
    overwrite: bool = False
) -> None
```

**Parameters:**
- `path`: Absolute path for the new file
- `replication`: Replication factor (uses default if not specified)
- `block_size`: Block size in bytes (uses default if not specified)
- `overwrite`: Whether to overwrite existing file

**Example:**
```python
# Create a new file
await client.create("/data/myfile.txt", replication=2, block_size=64*1024*1024)

# Overwrite existing file
await client.create("/data/myfile.txt", overwrite=True)
```

#### Write File

Write data to a file.

```python
async def write(
    path: str,
    data: bytes,
    append: bool = False
) -> None
```

**Parameters:**
- `path`: File path
- `data`: Binary data to write
- `append`: Whether to append to existing file

**Example:**
```python
# Write new file
data = b"Hello, HDFS!"
await client.write("/data/hello.txt", data)

# Write large file
with open("local_file.bin", "rb") as f:
    large_data = f.read()
await client.write("/data/large_file.bin", large_data)
```

#### Read File

Read complete file from HDFS.

```python
async def read(path: str) -> bytes
```

**Parameters:**
- `path`: File path to read

**Returns:**
- Complete file contents as bytes

**Example:**
```python
# Read file
data = await client.read("/data/myfile.txt")
print(data.decode('utf-8'))

# Read binary file
binary_data = await client.read("/data/image.jpg")
```

#### Stream Read

Read file in chunks (for large files).

```python
async def stream_read(
    path: str,
    chunk_size: int = 1024 * 1024
) -> AsyncIterator[bytes]
```

**Parameters:**
- `path`: File path to read
- `chunk_size`: Size of each chunk in bytes

**Example:**
```python
# Stream large file
async for chunk in client.stream_read("/data/huge_file.bin", chunk_size=10*1024*1024):
    process_chunk(chunk)  # Process 10MB at a time
```

#### Delete File

Delete a file from HDFS.

```python
async def delete(path: str) -> None
```

**Parameters:**
- `path`: File path to delete

**Example:**
```python
# Delete file
await client.delete("/data/old_file.txt")
```

#### Append to File

Append data to an existing file.

```python
async def append(path: str, data: bytes) -> None
```

**Parameters:**
- `path`: File path
- `data`: Data to append

**Example:**
```python
# Append to log file
log_entry = b"[2024-01-01] New log entry\n"
await client.append("/logs/app.log", log_entry)
```

### Directory Operations

#### Create Directory

Create a new directory.

```python
async def mkdir(path: str) -> None
```

**Parameters:**
- `path`: Directory path

**Example:**
```python
# Create directory
await client.mkdir("/data/processed")

# Create nested directories
await client.mkdir("/data/2024/january")
```

#### List Directory

List contents of a directory.

```python
async def listdir(path: str) -> List[str]
```

**Parameters:**
- `path`: Directory path

**Returns:**
- List of file and directory names

**Example:**
```python
# List root directory
contents = await client.listdir("/")
for item in contents:
    print(item)

# List specific directory
files = await client.listdir("/data")
```

#### Remove Directory

Remove an empty directory.

```python
async def rmdir(path: str) -> None
```

**Parameters:**
- `path`: Directory path (must be empty)

**Example:**
```python
# Remove empty directory
await client.rmdir("/data/temp")
```

### Metadata Operations

#### Get File Status

Get detailed information about a file.

```python
async def get_file_status(path: str) -> Dict[str, Any]
```

**Parameters:**
- `path`: File path

**Returns:**
- Dictionary containing file metadata

**Example:**
```python
# Get file info
status = await client.get_file_status("/data/myfile.txt")
print(f"Size: {status['size']} bytes")
print(f"Replication: {status['replication']}")
print(f"Block Size: {status['block_size']}")
print(f"Modification Time: {status['modification_time']}")
```

#### Rename File/Directory

Rename or move a file or directory.

```python
async def rename(src: str, dst: str) -> None
```

**Parameters:**
- `src`: Source path
- `dst`: Destination path

**Example:**
```python
# Rename file
await client.rename("/data/old_name.txt", "/data/new_name.txt")

# Move file to different directory
await client.rename("/temp/file.txt", "/data/file.txt")
```

#### Check Existence

Check if a file or directory exists.

```python
async def exists(path: str) -> bool
```

**Parameters:**
- `path`: Path to check

**Example:**
```python
# Check if file exists
if await client.exists("/data/important.txt"):
    data = await client.read("/data/important.txt")
```

## NameNode API

The NameNode provides administrative APIs for cluster management.

### File System Operations

```python
from hdfs.namenode import NameNode

namenode = NameNode(
    default_replication=3,
    default_block_size=128 * 1024 * 1024,
    heartbeat_interval=3.0
)
```

#### Create File

```python
def create_file(
    path: str,
    replication: Optional[int] = None,
    block_size: Optional[int] = None,
    overwrite: bool = False
) -> FileInfo
```

#### Get File Information

```python
def get_file_info(path: str) -> FileInfo
```

#### Delete File

```python
def delete_file(path: str) -> None
```

#### Create Directory

```python
def create_directory(path: str) -> None
```

#### List Directory

```python
def list_directory(path: str) -> List[str]
```

### Block Management

#### Allocate Blocks

```python
def allocate_blocks(
    path: str,
    num_blocks: int,
    block_size: int
) -> List[Block]
```

#### Get Block Locations

```python
def get_block_locations(block_id: BlockID) -> List[BlockLocation]
```

### DataNode Management

#### Register DataNode

```python
def register_datanode(
    node_id: str,
    host: str,
    port: int,
    capacity: Optional[int] = None
) -> None
```

#### Handle Heartbeat

```python
def handle_heartbeat(
    node_id: str,
    capacity: int,
    used: int,
    remaining: int
) -> HeartbeatResponse
```

#### Process Block Report

```python
def handle_block_report(
    node_id: str,
    blocks: List[BlockID]
) -> None
```

### Administrative Operations

#### Get Statistics

```python
def get_statistics() -> Dict[str, Any]
```

**Returns:**
```python
{
    'total_files': 1234,
    'total_directories': 56,
    'total_blocks': 5678,
    'total_capacity': 1099511627776,  # bytes
    'total_used': 549755813888,       # bytes
    'total_remaining': 549755813888,   # bytes
    'total_datanodes': 10,
    'live_datanodes': 9,
    'dead_datanodes': 1,
    'under_replicated_blocks': 5,
    'corrupt_blocks': 0,
    'missing_blocks': 0
}
```

#### Safe Mode Operations

```python
# Check safe mode status
def is_in_safe_mode() -> bool

# Manually exit safe mode
def exit_safe_mode() -> None

# Enter safe mode
def enter_safe_mode() -> None
```

#### Checkpointing

```python
# Save namespace checkpoint
def save_checkpoint(path: str) -> None

# Load namespace from checkpoint
def load_checkpoint(path: str) -> None
```

## DataNode API

The DataNode provides storage operations.

### Storage Operations

```python
from hdfs.datanode import DataNode

datanode = DataNode(
    node_id="datanode-1",
    data_dir="/var/hdfs/data",
    namenode_host="localhost",
    namenode_port=9000
)
```

#### Store Block

```python
def store_block(block_id: BlockID, data: bytes) -> None
```

#### Retrieve Block

```python
def retrieve_block(block_id: BlockID) -> bytes
```

#### Delete Block

```python
def delete_block(block_id: BlockID) -> None
```

#### Get Block Report

```python
def get_block_report() -> List[BlockID]
```

### Administrative Operations

#### Get Storage Info

```python
def get_storage_info() -> Dict[str, int]
```

**Returns:**
```python
{
    'capacity': 1099511627776,  # Total capacity in bytes
    'used': 549755813888,       # Used space in bytes
    'remaining': 549755813888   # Available space in bytes
}
```

#### Send Heartbeat

```python
async def send_heartbeat() -> None
```

#### Send Block Report

```python
async def send_block_report() -> None
```

## Error Handling

### Exception Types

```python
from hdfs.common.protocol import (
    HDFSError,           # Base exception
    FileNotFoundError,   # File/directory doesn't exist
    FileExistsError,     # File/directory already exists
    DirectoryNotEmptyError,  # Directory contains items
    NoDataNodeError,     # No DataNodes available
    QuotaExceededException,  # Quota limit exceeded
    SafeModeException    # Operation blocked in safe mode
)
```

### Error Handling Examples

```python
# Handle specific errors
try:
    data = await client.read("/nonexistent.txt")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except HDFSError as e:
    print(f"HDFS error: {e}")

# Retry on failure
max_retries = 3
for attempt in range(max_retries):
    try:
        await client.write("/data/file.txt", data)
        break
    except NoDataNodeError:
        if attempt == max_retries - 1:
            raise
        await asyncio.sleep(2 ** attempt)  # Exponential backoff
```

## Advanced Features

### Pipeline Writes

Enable pipeline replication for writes:

```python
# DataNode pipeline write
def store_block_pipeline(
    block_id: BlockID,
    data: bytes,
    downstream_nodes: List[DataNode]
) -> None
```

### Client-Side Caching

Enable metadata caching:

```python
client = HDFSClient(
    namenode_host="localhost",
    enable_cache=True,
    cache_ttl=60  # Cache timeout in seconds
)
```

### Bandwidth Throttling

Limit transfer bandwidth:

```python
datanode = DataNode(
    node_id="datanode-1",
    data_dir="/var/hdfs/data",
    max_bandwidth=10 * 1024 * 1024  # 10MB/s limit
)
```

### Custom Replication Policy

Implement custom block placement:

```python
class CustomReplicationPolicy(ReplicationPolicy):
    def choose_targets(
        self,
        num_replicas: int,
        datanodes: List[DataNodeInfo],
        excluded: List[NodeID] = None
    ) -> List[DataNodeInfo]:
        # Custom logic for replica placement
        pass
```

## Performance Tips

### Large File Handling

```python
# Use streaming for large files
async for chunk in client.stream_read("/huge_file.bin"):
    process(chunk)  # Process incrementally

# Write in chunks
chunk_size = 64 * 1024 * 1024  # 64MB chunks
for i in range(0, len(data), chunk_size):
    chunk = data[i:i + chunk_size]
    await client.append("/output.bin", chunk)
```

### Parallel Operations

```python
# Parallel file writes
files = [f"/data/file{i}.txt" for i in range(10)]
tasks = [client.write(f, data) for f in files]
await asyncio.gather(*tasks)

# Parallel reads
tasks = [client.read(f) for f in files]
results = await asyncio.gather(*tasks)
```

### Connection Pooling

```python
# Reuse client connections
client = HDFSClient(
    namenode_host="localhost",
    connection_pool_size=10,
    connection_timeout=30
)
```

## Monitoring and Metrics

### Client Metrics

```python
# Get client statistics
stats = client.get_stats()
print(f"Total bytes read: {stats['bytes_read']}")
print(f"Total bytes written: {stats['bytes_written']}")
print(f"Read operations: {stats['read_ops']}")
print(f"Write operations: {stats['write_ops']}")
```

### System Metrics

```python
# Get system-wide metrics
metrics = namenode.get_metrics()
print(f"Total capacity: {metrics['capacity_total']}")
print(f"Used capacity: {metrics['capacity_used']}")
print(f"Remaining: {metrics['capacity_remaining']}")
print(f"Live nodes: {metrics['nodes_live']}")
```