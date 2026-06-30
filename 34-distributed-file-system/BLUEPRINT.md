# Distributed File System (HDFS-lite) — Technical Blueprint

> **Concepts covered:** §02 data-engineering — `06-infrastructure`, `04-streaming`

## Executive Summary

This project is an HDFS-style distributed file system implemented in Python with
`asyncio`. It follows the classic Hadoop architecture: a single **NameNode**
that owns all filesystem metadata and block-to-node mapping, plus a fleet of
**DataNodes** that own block storage and serve bulk data transfer. Clients talk
to the NameNode to look up which DataNodes hold each block, then talk to
DataNodes directly to move bytes.

The system targets the same workload HDFS targets — large, mostly-append files
read sequentially — and reuses HDFS's coarse-grained design choices: 128 MiB
default block size, configurable per-file replication factor (default 3),
rack-aware replica placement, periodic heartbeats and block reports, a startup
**safe mode**, and JSON checkpoint snapshots of the namespace for persistence.

It is deliberately *not* a Ceph clone: there is no CRUSH algorithm, no
placement groups, no decentralized object placement, and no monitor quorum.
Metadata is centralized in the NameNode, just like HDFS, and the placement
algorithm is a simple rack-aware load-balanced score (40% remaining space, 60%
inverse block count, with a first-pass pick-one-per-rack rule).

This document describes the system **as built** in `src/hdfs/`. Where the code
takes a shortcut compared to production HDFS, the shortcut is called out
explicitly rather than papered over.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Data Types](#core-data-types)
3. [Wire Protocol](#wire-protocol)
4. [NameNode](#namenode)
5. [DataNode](#datanode)
6. [Client](#client)
7. [Data Flow](#data-flow)
8. [Replica Placement Algorithm](#replica-placement-algorithm)
9. [Fault Tolerance](#fault-tolerance)
10. [Persistence and Safe Mode](#persistence-and-safe-mode)
11. [Implementation Phases](#implementation-phases)
12. [Known Gaps and Simplifications](#known-gaps-and-simplifications)
13. [File Layout](#file-layout)
14. [References](#references)

---

## Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HDFS-lite Cluster                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                          ┌────────────────────┐                             │
│                          │     HDFSClient     │                             │
│                          │  create / read /   │                             │
│                          │  mkdir / listdir / │                             │
│                          │  delete / rename   │                             │
│                          └─────────┬──────────┘                             │
│                                    │                                        │
│              ┌─────── metadata RPC ┼──── block transfer ─────┐              │
│              │                     │                         │              │
│              ▼                     ▼                         ▼              │
│      ┌────────────────┐    ┌────────────────┐       ┌────────────────┐     │
│      │   NameNode     │    │   DataNode 1   │       │   DataNode N   │     │
│      │  (singleton)   │    │  rack=/r1      │       │  rack=/r2      │     │
│      │                │    │                │       │                │     │
│      │ • namespace    │◀──▶│ • blk_xxx      │  ...  │ • blk_yyy      │     │
│      │ • blocks       │ ht │ • blk_xxx.crc  │  hb   │ • blk_yyy.crc  │     │
│      │ • datanode     │ bt │ • heartbeat    │       │ • heartbeat    │     │
│      │   registry     │    │ • block report │       │ • block report │     │
│      │ • placement    │    │                │       │                │     │
│      └────────────────┘    └────────────────┘       └────────────────┘     │
│             ▲                                                               │
│             │ checkpoint.json (manual save/load)                            │
│             ▼                                                               │
│      ┌────────────────┐                                                     │
│      │   on-disk      │                                                     │
│      │   namespace    │                                                     │
│      │   snapshot     │                                                     │
│      └────────────────┘                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

hb = heartbeat (DN → NN every ~3s, with piggy-backed commands NN → DN)
bt = block transfer (client ↔ DN, length-prefixed JSON over TCP)
```

### Roles

| Component   | Process     | State                                        | Talks to        |
|-------------|-------------|----------------------------------------------|-----------------|
| NameNode    | `NameNodeServer` (`asyncio`)  | Namespace, block map, DN registry            | Clients, DNs    |
| DataNode    | `DataNodeServer` (`asyncio`)  | Local block files + checksums                | NameNode, Clients |
| Client      | `HDFSClient` library         | Optional per-path metadata cache (TTL)       | NameNode, DNs   |

The NameNode is **a single process**. There is no Secondary NameNode, no
NameNode HA, no Federation. Recovery is by reloading a JSON checkpoint.

---

## Core Data Types

Defined in `src/hdfs/common/types.py`.

```python
BlockID = str          # "blk_" + 16-char hex, e.g. "blk_3a7f9c1d8e2b4a90"
NodeID  = str          # "node_" + 12-char hex, e.g. "node_7c3e9a1b8d2f"

class ReplicationPolicy(Enum):
    DEFAULT         = "default"   # standard 3x replication (only one really wired)
    ERASURE_CODING  = "ec"        # placeholder, not implemented
    SINGLE          = "single"    # no replication

@dataclass
class Block:
    block_id: BlockID
    size: int = 0
    generation_stamp: int = 0    # ms-since-epoch when allocated

@dataclass
class BlockLocation:
    block_id: BlockID
    node_id: NodeID
    host: str
    port: int
    rack: str = "/default-rack"

@dataclass
class DataNodeInfo:
    node_id: NodeID
    host: str
    port: int
    rack: str = "/default-rack"
    capacity: int = 0          # total bytes
    used: int = 0
    remaining: int = 0
    last_heartbeat: float = 0.0
    blocks: Set[BlockID] = set()

    @property
    def is_alive(self) -> bool:
        return time.time() - self.last_heartbeat < 30.0

@dataclass
class FileInfo:
    path: str
    size: int = 0
    block_size: int = 128 * 1024 * 1024   # 128 MiB
    replication: int = 3
    blocks: List[BlockID] = []
    modification_time: float
    access_time: float
    owner: str = "hdfs"
    group: str = "supergroup"
    permission: int = 0o644

@dataclass
class DirectoryInfo:
    path: str
    children: Set[str] = set()     # base names of immediate children only
    modification_time: float
    owner: str = "hdfs"
    group: str = "supergroup"
    permission: int = 0o755
```

A few things worth noting about how these are used:

- **Block IDs are strings** that start with `"blk_"`. The NameNode actually
  pattern-matches on this prefix in `get_block_locations()` to disambiguate
  "did the caller pass me a path or a block ID?" — a deliberate shortcut for
  protocol simplicity.
- A `DirectoryInfo` stores only immediate child *names*, not paths. Full
  paths are reconstructed by concatenation, and rename re-walks the subtree
  to update `_files`/`_directories` keys.
- There is **no inode number**. The flat dict `_files: Dict[path, FileInfo]`
  is the source of truth; `_directories: Dict[path, DirectoryInfo]` is the
  parallel structure for dirs. This means `rename` has to update every
  descendant entry. It is O(N) in the subtree size.

---

## Wire Protocol

Defined in `src/hdfs/common/protocol.py`.

All RPC — client ↔ NameNode, client ↔ DataNode, DataNode ↔ NameNode — uses
the same framing:

```
┌────────────┬──────────────────────────────────────┐
│ 4 bytes BE │ JSON-encoded Message payload         │
│  length    │  {"type": ..., "payload": ..., ...}  │
└────────────┴──────────────────────────────────────┘
```

The transport is plain TCP via `asyncio.open_connection` /
`asyncio.start_server`. There is no TLS, no auth, no compression. Binary block
data is transported as **hex-encoded strings inside the JSON payload**, which
is the single most expensive shortcut in the system (2× the wire size for
block bodies). The system is therefore appropriate for correctness testing and
small workloads but not for production throughput.

### Message Types

```python
class MessageType(Enum):
    # Namespace ops (client → NameNode)
    CREATE_FILE          = "create_file"
    OPEN_FILE            = "open_file"
    DELETE_FILE          = "delete_file"
    RENAME_FILE          = "rename_file"
    MKDIR                = "mkdir"
    DELETE_DIR           = "delete_dir"
    LIST_DIR             = "list_dir"
    GET_FILE_INFO        = "get_file_info"

    # Block lifecycle (client → NameNode)
    ADD_BLOCK            = "add_block"
    COMPLETE_FILE        = "complete_file"
    GET_BLOCK_LOCATIONS  = "get_block_locations"
    REPORT_BAD_BLOCKS    = "report_bad_blocks"

    # DataNode registration & health (DataNode → NameNode)
    REGISTER_DATANODE    = "register_datanode"
    HEARTBEAT            = "heartbeat"
    BLOCK_REPORT         = "block_report"
    BLOCK_RECEIVED       = "block_received"

    # Data transfer (client ↔ DataNode)
    READ_BLOCK           = "read_block"
    WRITE_BLOCK          = "write_block"
    COPY_BLOCK           = "copy_block"
    DELETE_BLOCK         = "delete_block"

    # Generic responses
    SUCCESS              = "success"
    ERROR                = "error"
```

Every reply is itself a `Message` (`SUCCESS` with payload, or `ERROR` with an
`"error"` string). Errors raised by the NameNode core that are typed —
`FileNotFoundError`, `FileExistsError`, `DirectoryNotEmptyError`,
`NoDataNodeError`, `BlockNotFoundError`, `ReplicationError`, `HDFSError` — are
caught at the dispatcher boundary in `NameNodeServer._process_message` and
flattened into `ERROR` payloads.

---

## NameNode

Implemented in `src/hdfs/namenode/namenode.py`. Two classes:

- `NameNode`: the in-memory metadata service. Pure Python, no asyncio.
- `NameNodeServer`: the asyncio TCP front-end that adapts `MessageType`
  requests to `NameNode` method calls.

### In-Memory State

```python
class NameNode:
    # tunables
    default_replication:    int                 # 3
    default_block_size:     int                 # 128 MiB
    heartbeat_interval:     float               # 3.0 s
    checkpoint_interval:    float               # 3600 s (not actively used)

    # namespace
    _files:        Dict[path, FileInfo]                  # canonical file map
    _directories:  Dict[path, DirectoryInfo]             # incl. "/" pre-created

    # block map (forward + reverse)
    _blocks:          Dict[BlockID, Block]
    _block_to_nodes:  Dict[BlockID, Set[NodeID]]         # reverse index

    # datanode registry
    _datanodes: Dict[NodeID, DataNodeInfo]

    # work queues to dispatch via heartbeat responses
    _pending_replications: List[(BlockID, [NodeID])]     # (block, target list)
    _pending_deletions:    Dict[NodeID, List[BlockID]]   # per-DN trash bin

    # startup gate
    _safe_mode:           bool
    _safe_mode_threshold: float                          # 0.999
```

All state is in memory. Persistence is via explicit `save_checkpoint(path)` /
`load_checkpoint(path)` calls that dump the dicts above to a JSON file. There
is no edit log / write-ahead log.

### Operations

#### Namespace operations

`create_file`, `delete_file`, `rename`, `mkdir`, `delete_directory`,
`list_directory`, `list_directory_detailed`, `get_file_info`,
`get_file_info_dict` — all of these manipulate `_files` / `_directories`
under no locks (single-threaded asyncio model). Parents must exist unless
`mkdir(..., create_parents=True)` is used.

`rename(src, dst)` does the obvious move-and-update-children walk; for a
directory rename it calls `_update_subtree_paths(old_prefix, new_prefix)`
which rewrites every descendant key in both `_files` and `_directories`.

#### Block allocation (`add_block`)

```python
def add_block(self, path) -> (Block, List[BlockLocation]):
    if self._safe_mode:
        raise HDFSError("NameNode is in safe mode. Cannot allocate blocks.")
    if path not in self._files:
        raise FileNotFoundError(...)

    targets = self._select_datanodes_for_block(replication)   # see placement
    if not targets:
        raise NoDataNodeError(...)

    block = Block(block_id=generate_block_id())
    self._blocks[block.block_id] = block
    self._files[path].blocks.append(block.block_id)
    return block, [BlockLocation(... for each target ...)]
```

The NameNode **does not yet record** which DataNodes were *intended* to hold
the new block — `_block_to_nodes[block_id]` stays empty until DataNodes
explicitly call `block_received` (via `BLOCK_RECEIVED`) or include the block
in a `BLOCK_REPORT`. This is HDFS-faithful: the NameNode trusts only what
DataNodes report.

#### `complete_file(path, size)`

Sets `FileInfo.size`, runs `_check_quota_for_write` to enforce any inherited
space quota on ancestor directories (each ancestor `DirectoryInfo` can carry
an optional `space_quota` attribute set via `set_quota`), and stamps
`modification_time`.

#### Block locations

```python
def get_block_locations(self, path_or_block_id):
    if path_or_block_id.startswith("blk_"):
        return self.get_block_locations_by_id(...)      # List[BlockLocation]
    return self.get_block_locations_for_file(...)       # List[List[BlockLocation]]
```

Per-file lookup returns a list-of-lists in block order, which is what the
client iterates when reading the file end-to-end.
`get_block_locations_by_id` filters out DataNodes whose `is_alive` is false
(no heartbeat in the last 30 s).

### Heartbeats and Block Reports

```python
def heartbeat(node_id, used, remaining) -> HeartbeatResponse:
    if node_id not in self._datanodes:
        return HeartbeatResponse(commands=[{"type": "re-register"}])

    self._datanodes[node_id].last_heartbeat = time.time()
    # update usage
    ...

    # 1. age out any DataNode that hasn't checked in for >30 s
    self.check_and_remove_dead_nodes()

    # 2. drain pending work for this node
    commands = []
    if node_id in self._pending_deletions:
        commands.append({"type": "delete", "block_ids": [...]})
    for block, targets in pending_replications:
        if node_id holds block:
            commands.append({"type": "replicate", "block_id": ..., "targets": [...]})

    return HeartbeatResponse(commands=commands)
```

Every heartbeat is also when liveness is **enforced**: the NameNode walks its
DataNode registry and removes any node whose `last_heartbeat` is more than
30 s old, scheduling re-replication of its blocks in the process.

`block_report(BlockReport)` is processed by clearing this node's membership
from every entry in `_block_to_nodes` and then re-adding it for the blocks
the DN reports. Unknown blocks (no `Block` object in `_blocks`) are ignored.

`block_received(node_id, block_id, size)` is the fast path called when a
client write finishes a single block; it updates the size on `Block`, adds
the node to the reverse index, and adds the block to the DN's recorded set.

### Statistics, Quotas, Checkpointing

- `get_statistics()` returns file count, dir count, block count, DN count,
  total/used/remaining capacity, and safe-mode status.
- `set_quota(path, namespace_quota, space_quota)` attaches optional
  attributes to a `DirectoryInfo`. `_check_quota_for_write(path, size)`
  walks ancestors and rejects writes that would push consumed space over a
  space quota.
- `save_checkpoint(path)` serializes `_files`, `_directories`, `_blocks`,
  `_datanodes`, and `_block_to_nodes` to a JSON file.
  `load_checkpoint(path)` reads it back. DN `last_heartbeat` is reset to
  `now()` on load to avoid immediate eviction, which means a checkpoint
  reload doesn't preserve "this DN is actually dead" — every DN starts as
  "alive" until 30 s pass without a real heartbeat.

---

## DataNode

Implemented in `src/hdfs/datanode/datanode.py`. Two classes:

- `DataNode`: the storage engine (sync, file-backed).
- `DataNodeServer`: the asyncio server for client block transfer.

### On-Disk Layout

```
<data_dir>/
  └── blocks/
        ├── blk_3a7f9c1d8e2b4a90
        ├── blk_3a7f9c1d8e2b4a90.crc      (logical — checksum is in memory)
        ├── blk_8c1e0d2f7b3a9c45
        └── ...
```

Each block file's name **is** the block ID. On startup, `_scan_blocks()`
walks `blocks/` and for each `blk_*` file records:

- `_blocks: Dict[BlockID, int]` — block id → on-disk size
- `_block_checksums: Dict[BlockID, str]` — block id → MD5 of bytes (computed
  on scan)

`used_space` is `sum(self._blocks.values())`; `remaining_space` is `capacity
- used_space`.

### Block Operations

```python
def write_block(block_id, data) -> int:
    open(blocks/block_id, 'wb').write(data)
    _blocks[block_id] = len(data)
    _block_checksums[block_id] = md5(data).hexdigest()
    return len(data)

def read_block(block_id, offset=0, length=-1) -> bytes:
    if block_id not in _blocks: raise BlockNotFoundError
    f = open(blocks/block_id, 'rb'); f.seek(offset); return f.read(length)

def delete_block(block_id) -> bool: ...
def verify_block(block_id) -> bool:        # recompute md5, compare
def scan_blocks() -> List[BlockID]:        # return all corrupted ids
```

Pipeline replication exists on the DataNode in stub form:

```python
def store_block_pipeline(block_id, data, downstream_nodes=None) -> int:
    size = self.write_block(block_id, data)
    if downstream_nodes:
        for node in downstream_nodes:
            if hasattr(node, 'store_block'):
                node.store_block(block_id, data)
    return size
```

This forwards by *calling Python methods on a passed-in DataNode object*,
which works in unit tests but is not actually used by the live client (see
[Known Gaps](#known-gaps-and-simplifications)).

### NameNode Communication

The DataNode talks to the NameNode over the same `Message` protocol:

- `register_with_namenode()` — `REGISTER_DATANODE` on startup, sending
  `(node_id, host, port, capacity)`. The default rack `/default-rack` is used
  unless the NameNode is told otherwise on the NN-side `register_datanode`
  call.
- `send_heartbeat()` — `HEARTBEAT` every 3 s with `used` and `remaining`.
  Returns any commands the NameNode has queued for us.
- `send_block_report()` — `BLOCK_REPORT` with the full list of block IDs.
  Sent at startup and on a long interval (3600 s default).
- `report_block_received(block_id, size)` — `BLOCK_RECEIVED` after a
  successful write.

### Command Execution

When the heartbeat reply carries commands:

```python
async def execute_command(self, command):
    if cmd_type == "delete":
        for block_id in command["block_ids"]:
            self.delete_block(block_id)
    elif cmd_type == "replicate":
        # logs the intent only — does not actually copy the block
        logger.info(f"Replicating {block_id} to {targets}")
    elif cmd_type == "re-register":
        await self.register_with_namenode()
```

The `replicate` command is a documentation-only stub (see Known Gaps).

### Background Tasks

`DataNodeServer.start()` kicks off three things:
1. The asyncio TCP server on `(self.host, self.port)`.
2. A `heartbeat_loop()` task (3 s cadence).
3. A `block_report_loop()` task (3600 s cadence).

The data-transfer server handles `READ_BLOCK`, `WRITE_BLOCK`, and
`DELETE_BLOCK`. `WRITE_BLOCK` writes locally and then fires
`report_block_received` back to the NameNode.

---

## Client

Implemented in `src/hdfs/client/client.py`. Three classes:

- `HDFSClient` — the main API.
- `HDFSOutputStream` — buffered streaming writer.
- `HDFSInputStream` — block-at-a-time streaming reader.

### High-Level API

```python
client = HDFSClient(namenode_host="localhost", namenode_port=9000,
                    block_size=128*1024*1024, replication=3)

# Namespace
await client.mkdir("/data", create_parents=True)
await client.listdir("/data")
await client.exists("/data/file.bin")
await client.get_file_info("/data/file.bin")
await client.rename("/a", "/b")
await client.delete("/data/file.bin")
await client.rmdir("/data", recursive=False)

# Bulk I/O
await client.create("/data/file.bin", payload_bytes, replication=3)
await client.read("/data/file.bin")
await client.append("/data/file.bin", more_bytes)
await client.put("local.txt", "/data/file.bin")
await client.get("/data/file.bin", "local.txt")

# Streaming
async with await client.open_for_write("/big.bin") as f: await f.write(...)
async with await client.open_for_read("/big.bin") as f: chunk = await f.read(N)
async for chunk in client.stream_read("/big.bin", chunk_size=1<<20): ...
```

There is an optional in-memory metadata cache (`enable_cache=True`,
`cache_ttl=60s`) and a `verify_checksum` flag. The cache is keyed by path and
holds the raw response payload.

### Write Path

`HDFSClient.create(path, data, replication, block_size, overwrite)`:

```
1. CREATE_FILE  → NN  (sets up FileInfo, registers under parent dir)
2. for each block-sized slice of `data`:
       ADD_BLOCK         → NN  → returns (block_id, locations[3])
       for loc in locations:
           WRITE_BLOCK   → DataNode(loc.host, loc.port)
3. COMPLETE_FILE → NN  (records final size)
```

Key choice: the client **fans the WRITE_BLOCK out to every replica itself**.
This is a star topology, not the pipelined fan-out used by real HDFS. It is
simpler and shorter-latency for small blocks; for full-sized 128 MiB blocks
it costs the client a 3× upstream bandwidth. The DataNode's
`store_block_pipeline()` stub exists for unit tests that pass live DataNode
objects but is not invoked from the wire path.

### Read Path

`HDFSClient.read(path)`:

```
1. GET_BLOCK_LOCATIONS → NN → returns [[loc, loc, loc], [loc, loc, loc], ...]
2. for each block's locations:
       try locations in order until one returns SUCCESS:
           READ_BLOCK → DataNode
3. concatenate.
```

The client iterates the replica list in the order the NameNode returned it
(which itself is iteration order of a `set`). There is no preference for
"closest" replica.

### Streaming

`HDFSOutputStream` buffers writes in a `BytesIO` and flushes a new block
each time the buffer hits `block_size`. On `close()` it flushes whatever is
left and issues `COMPLETE_FILE`.

`HDFSInputStream` keeps a current-block buffer and an offset and loads the
next block on demand. `stream_read()` is a convenience generator over the
same machinery, yielding bytes in roughly `chunk_size` chunks (rounded up
to whole blocks).

---

## Data Flow

### File Write (small file, fits in one block)

```
Client                 NameNode           DataNode1   DataNode2   DataNode3
  │  CREATE_FILE         │                    │           │           │
  │ ───────────────────▶ │                    │           │           │
  │  ◀──────── SUCCESS   │                    │           │           │
  │                      │                    │           │           │
  │  ADD_BLOCK           │                    │           │           │
  │ ───────────────────▶ │                    │           │           │
  │                      │ select 3 DNs       │           │           │
  │  ◀── (blk_id, locs)  │ (rack-aware,       │           │           │
  │                      │  load-balanced)    │           │           │
  │                      │                    │           │           │
  │  WRITE_BLOCK(blk, data) ───────────────▶  │           │           │
  │  ◀────────────────────── SUCCESS          │ ─ BLOCK_RECEIVED ─▶ NN│
  │  WRITE_BLOCK(blk, data) ───────────────────────────▶  │           │
  │  ◀───────────────────────────────── SUCCESS           │ ─→ NN     │
  │  WRITE_BLOCK(blk, data) ───────────────────────────────────────▶  │
  │  ◀───────────────────────────────────────────── SUCCESS           │
  │                                                                   │
  │  COMPLETE_FILE       │                                            │
  │ ───────────────────▶ │  (size, mtime, quota check)                │
  │  ◀──────── SUCCESS   │                                            │
```

### File Read

```
Client                 NameNode             DataNode (first alive replica)
  │  GET_BLOCK_LOCATIONS │                       │
  │ ──────────────────▶  │                       │
  │  ◀── [[locs], [locs] │ (filtered by is_alive)│
  │                      │                       │
  │  for each block:                             │
  │     READ_BLOCK(blk_id) ────────────────────▶ │
  │  ◀──────────────────────── data (hex-encoded)│
  │     (concatenate locally)                    │
```

### DataNode Liveness Loop

```
DataNode                                    NameNode
   │                                            │
   │  every 3 s: HEARTBEAT(used, remaining) ──▶ │
   │                                            │ update last_heartbeat
   │                                            │ run check_and_remove_dead_nodes
   │                                            │ drain pending_deletions[me]
   │                                            │ drain pending_replications I host
   │  ◀────── HeartbeatResponse(commands)       │
   │                                            │
   │  for cmd in commands:                      │
   │      execute_command(cmd)                  │
   │       (delete → local delete_block)        │
   │       (replicate → log-only stub)          │
   │       (re-register → call REGISTER_DATANODE)│
   │                                            │
   │  every 3600 s: BLOCK_REPORT(all blk ids) ▶ │
   │                                            │ reset _block_to_nodes for me
   │                                            │ rebuild from report
   │                                            │ _check_safe_mode()
   │  ◀────── SUCCESS                           │
```

---

## Replica Placement Algorithm

Implemented in `NameNode._select_datanodes_for_block(replication)`.

```python
def _select_datanodes_for_block(self, replication):
    # 1. Filter: alive DNs with at least one block's worth of space.
    available = [(nid, dn) for nid, dn in self._datanodes.items()
                 if dn.is_alive and dn.remaining > self.default_block_size]
    if len(available) < replication:
        return [nid for nid, _ in available]

    # 2. Score each candidate.
    #    block_count(nid) = how many blocks in _block_to_nodes contain nid
    block_counts = {nid: count_blocks_held(nid) for nid, _ in available}
    max_rem      = max(dn.remaining for _, dn in available) or 1
    max_blocks   = max(block_counts.values()) + 1

    def load_score(nid, dn):
        return 0.6 * block_counts[nid] / max_blocks \
             + 0.4 * (1 - dn.remaining / max_rem)

    available.sort(key=lambda x: load_score(*x))   # lowest score first

    # 3. Two-pass rack-aware fill.
    selected, racks_used = [], set()
    for nid, dn in available:                       # pass 1: one per rack
        if len(selected) >= replication: break
        if dn.rack not in racks_used:
            selected.append(nid); racks_used.add(dn.rack)
    for nid, dn in available:                       # pass 2: fill the rest
        if len(selected) >= replication: break
        if nid not in selected:
            selected.append(nid)
    return selected
```

Properties:

- **Rack diversity is best-effort.** With three replicas and three or more
  distinct racks among the lowest-load candidates, all three picks land on
  different racks. With fewer racks, the second pass fills remaining slots
  from any rack.
- **Load is a weighted sum, not a hard constraint.** 60% goes to "how many
  blocks does this DN already hold", 40% to "how full is it (as fraction of
  the fullest peer)". Both are normalized only against the *currently
  alive* set.
- **No graph / pseudo-random hashing.** This is not CRUSH; the placement is
  recomputed from scratch on every `add_block`. Locality is therefore *not*
  stable as the cluster scales or as DNs come and go — but on the flip side,
  there are no placement-group invariants to keep consistent.

This is a deliberate simplification: real HDFS picks one local replica, one
on a remote rack, one on the same remote rack as the second, in that fixed
order. The score-based version is easier to reason about and tests well; it
loses the "first replica is local to the writer" optimization (the client
has no concept of locality in this codebase anyway).

---

## Fault Tolerance

### DataNode Failure

Detected during *any* heartbeat (and on every block report). The NameNode
runs `check_and_remove_dead_nodes(timeout=30.0)` which scans the registry,
removes nodes whose `last_heartbeat` is older than 30 s, and calls
`_remove_datanode(node_id)`:

```python
def _remove_datanode(self, node_id):
    # Drop the node from every block's location set
    for block_id in list(self._block_to_nodes.keys()):
        self._block_to_nodes[block_id].discard(node_id)
    del self._datanodes[node_id]
    self._schedule_replication_for_lost_node(node_id)
```

`_schedule_replication_for_lost_node` walks under-replicated blocks (file's
target `replication` > current replica count), picks one alive DN that
doesn't already hold the block, and appends `(block_id, [target])` to
`_pending_replications`. The next heartbeat from any DN that *holds* the
block will pick up the work via:

```python
for block_id, targets in list(self._pending_replications):
    if node_id in self._block_to_nodes.get(block_id, set()):
        commands.append({"type": "replicate", ...})
        self._pending_replications.remove((block_id, targets))
```

### Block Loss / Corruption

The DataNode's `verify_block(block_id)` recomputes MD5 and compares against
the cached checksum. `scan_blocks()` returns all corrupted block IDs. The
client never sends checksums over the wire to the NameNode, so block
corruption is detected only by:

1. The DataNode's own background verification (not currently scheduled).
2. A read that fails on one replica — the client falls through to the next
   replica in the location list.

### NameNode Failure

There is no failover. Recovery is operational:

1. Restart the NameNode process.
2. `load_checkpoint(path)` to recover namespace state.
3. Wait for DataNodes to re-register (heartbeats will tell them to via the
   `re-register` command) and send block reports.
4. `_safe_mode` stays True until ≥ 99.9% of known blocks have at least one
   reporting replica.

Any namespace mutations between the last checkpoint and the crash are lost.
This is the single biggest gap relative to real HDFS, which uses an
edit log that is replayed atop the latest fsimage.

### Client Retry

The client retries on the *read* side by trying each replica in turn. It
does not retry RPCs on transient errors; an `HDFSError` propagates to the
caller. Higher-level apps are expected to implement their own retry.

---

## Persistence and Safe Mode

### Safe Mode

On startup, `_safe_mode = True`. While in safe mode:

- `add_block` raises `HDFSError("NameNode is in safe mode...")`.
- All other reads and namespace operations succeed.

The mode is checked after every `block_report` via `_check_safe_mode()`:

```python
def _check_safe_mode(self):
    if not self._safe_mode: return
    total = len(self._blocks)
    if total == 0:
        self._safe_mode = False
        return
    reported = sum(1 for bid in self._blocks
                     if self._block_to_nodes.get(bid))
    if reported / total >= 0.999:
        self._safe_mode = False
```

The empty-cluster case (no checkpoint, no blocks yet) exits safe mode
immediately, which is what we want for a fresh start. A reloaded checkpoint
with a known block set will hold the cluster in safe mode until enough
DataNodes have reported.

### Checkpointing

`save_checkpoint(path)` writes a single JSON file containing the full
namespace plus block map. The expected operational pattern is:

- Run `save_checkpoint` periodically (e.g. from cron or via the
  `checkpoint_interval` knob — currently the knob is stored but no
  background task actually invokes the save; this is a TODO).
- After a NameNode crash, restart with `load_checkpoint`.

There is no incremental log between checkpoints, so namespace operations
performed *after* the most recent checkpoint are not durable.

---

## Implementation Phases

The system was built in four phases. All four are complete; the test suite
exercises every phase.

### Phase 1 — Common Types & Wire Protocol

- [x] `BlockID`, `NodeID` generators
- [x] `Block`, `BlockLocation`, `DataNodeInfo`
- [x] `FileInfo`, `DirectoryInfo`
- [x] `ReplicationPolicy` enum (only `DEFAULT` and `SINGLE` are wired)
- [x] `Message`, `MessageType`, serialize/deserialize
- [x] Length-prefixed JSON framing over asyncio TCP
- [x] Typed `HDFSError` hierarchy

### Phase 2 — NameNode

- [x] Namespace ops (`create_file`, `mkdir`, `delete_*`, `rename`, `listdir`)
- [x] Path resolution by direct dict lookup (no inode tree walk)
- [x] Block allocation (`add_block`, `complete_file`)
- [x] Rack-aware load-balanced placement
- [x] DataNode registration, heartbeat, block report
- [x] Pending replication / deletion queues
- [x] Dead-node detection (30 s timeout)
- [x] Safe mode with 99.9% threshold
- [x] JSON `save_checkpoint` / `load_checkpoint`
- [x] Per-directory namespace and space quotas
- [x] Asyncio server (`NameNodeServer`) wiring all of this to the protocol

### Phase 3 — DataNode

- [x] Block storage as plain files under `<data_dir>/blocks/blk_*`
- [x] MD5 checksum on write; in-memory checksum map
- [x] `write_block` / `read_block` / `delete_block` / `verify_block`
- [x] Startup `_scan_blocks` and `recover_blocks`
- [x] `register_with_namenode`, heartbeat loop, block report loop
- [x] Command execution: `delete`, `re-register`
- [x] `store_block_pipeline` (in-process pipeline for tests)
- [x] `store_block_throttled` (bandwidth simulation for tests)
- [x] Asyncio server (`DataNodeServer`) for `READ_BLOCK` / `WRITE_BLOCK` /
      `DELETE_BLOCK`

### Phase 4 — Client

- [x] Synchronous-feeling async API: `create`, `read`, `delete`, `rename`,
      `mkdir`, `listdir`, `rmdir`, `exists`, `get_file_info`, `append`,
      `put`, `get`
- [x] Streaming: `HDFSOutputStream`, `HDFSInputStream`, `stream_read`
- [x] Multi-replica write (star fan-out from client)
- [x] Multi-replica read with fallthrough on per-replica failure
- [x] Optional metadata cache with TTL
- [x] Quasi-checksum verification path (`_read_block_with_checksum`,
      partially wired)

---

## Known Gaps and Simplifications

The implementation is honest about being HDFS-*lite*. The following are
known shortcuts that would need work to claim full HDFS parity:

1. **No edit log / WAL.** Persistence is JSON snapshot only. Namespace
   changes between checkpoints are lost on a NameNode crash.

2. **No NameNode HA.** Single process, single point of failure. There is no
   Secondary NameNode that rolls the edit log forward, no JournalNodes,
   no Active/Standby pair.

3. **No pipelined client writes.** The client writes to all replicas itself
   in sequence:
   ```python
   for loc in locations:
       await self._send_to_datanode(loc["host"], loc["port"], WRITE_BLOCK(...))
   ```
   Real HDFS uses a pipeline (client → DN1 → DN2 → DN3) with acks streamed
   back. The DataNode does have a `store_block_pipeline` method, but it
   forwards by calling Python methods on a passed-in object reference, so
   it's only usable from in-process unit tests.

4. **`replicate` command is a stub.** When the NameNode tells a DataNode to
   replicate a block elsewhere, the DataNode logs the intent but does not
   open a connection to the target and copy the bytes. Re-replication after
   DN loss therefore detects the under-replication and *schedules* the work,
   but the work doesn't actually run. This is the most consequential gap for
   self-healing claims.

5. **Block bodies are hex-encoded inside JSON.** A 128 MiB block becomes
   256 MiB on the wire. Acceptable for testing, unacceptable for any real
   throughput target. A length-prefixed binary frame would be the
   straightforward fix.

6. **No authentication, no encryption, no quotas-on-the-wire.** It is a
   single-tenant testbed. The `owner`/`group`/`permission` fields exist
   but aren't enforced.

7. **No locality.** The client picks the first replica returned by the
   NameNode, and the NameNode returns replicas in `set` iteration order.
   There is no rack-local preference for reads.

8. **No append at the block level.** `HDFSClient.append(path, data)` reads
   the entire existing file, concatenates, deletes, and rewrites. Fine
   for tests; not a real append.

9. **Erasure coding is enum-only.** `ReplicationPolicy.ERASURE_CODING`
   exists; nothing in the codebase produces or decodes EC stripes.

10. **`checkpoint_interval` is configured but not driven.** Nothing
    actually triggers periodic checkpoints; an operator must call
    `save_checkpoint` explicitly.

11. **Safe-mode block reporting only counts blocks the NameNode already
    knows about.** A fresh NameNode with no `_blocks` exits safe mode
    immediately, which is correct for cold start but means a NameNode that
    loses its checkpoint will *not* enter a useful safe mode just from
    block reports.

12. **`rename` of a large subtree is O(N).** Every descendant key in
    `_files` and `_directories` is rewritten. Acceptable at the scales the
    tests cover; would matter for millions of files.

13. **Pending-replication consumption is best-effort.** A
    `_pending_replications` entry is dispatched only when a DN that holds
    the block heartbeats. If no surviving holder heartbeats before the
    NameNode restarts, the work is lost.

---

## File Layout

```
34-distributed-file-system/
├── BLUEPRINT.md                       ← this document
├── PROGRESS.md
├── README.md
├── pyproject.toml
├── docker-compose.yml                 (NameNode + 3 DataNode services)
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   └── CONTRIBUTING.md
├── src/
│   └── hdfs/
│       ├── __init__.py                (re-exports the public surface)
│       ├── common/
│       │   ├── types.py               (~150 LOC: BlockID, FileInfo, etc.)
│       │   └── protocol.py            (~135 LOC: Message, MessageType, errors)
│       ├── namenode/
│       │   └── namenode.py            (~1080 LOC: NameNode + NameNodeServer)
│       ├── datanode/
│       │   └── datanode.py            (~540 LOC: DataNode + DataNodeServer)
│       └── client/
│           └── client.py              (~770 LOC: HDFSClient + streams)
└── tests/
    ├── conftest.py / fixtures.py
    ├── test_namenode.py
    ├── test_datanode.py
    ├── test_client.py
    ├── test_integration.py
    └── test_replication.py            (40 dedicated replication tests)
```

Approximate breakdown: ~2680 LOC of production code, ~2600 LOC of tests.

---

## Testing Strategy

The test suite is structured around the four phases plus an integration
layer.

| Test file              | What it exercises                                                                                       |
|------------------------|---------------------------------------------------------------------------------------------------------|
| `test_namenode.py`     | Namespace ops, block allocation, DN registry, heartbeats, safe mode, checkpoint round-trip, quotas      |
| `test_datanode.py`     | Block read/write/delete, checksum verify, block scan/recover, pipeline (in-process), throttling        |
| `test_client.py`       | Client API surface, error mapping, metadata cache TTL, streaming reads/writes                          |
| `test_integration.py`  | End-to-end create / read / list / rename / delete against an in-process cluster                         |
| `test_replication.py`  | 40 tests covering rack-aware placement, replication factor enforcement, failure recovery, under/over- replication detection, data consistency, safe-mode behavior, and async heartbeats |

A representative shape of an integration test:

```python
async def test_create_read_round_trip():
    nn = NameNode()
    dns = [DataNode(node_id=f"node_{i}", data_dir=f"/tmp/dn{i}",
                    host="localhost", port=50010+i, rack=f"/rack{i%2}")
           for i in range(3)]
    for dn in dns:
        nn.register_datanode(dn.node_id, dn.host, dn.port,
                             capacity=10*1024**3, rack=dn.rack)
        nn.handle_block_report(dn.node_id, [])     # exit safe mode

    # ... create file, push blocks through write_block, complete_file ...
    # ... read back via get_block_locations + read_block ...
    assert read_data == written_data
```

The replication tests do most of the heavy lifting: they cover the placement
algorithm, the dead-node detection path, the under-replication detection,
and the safe-mode threshold.

---

## Performance Notes

The system has not been formally benchmarked. The known performance ceilings
are imposed by the design choices above, in roughly this order:

1. **JSON-encoded block transfer.** Hex-doubling plus JSON parsing puts a
   hard cap somewhere south of ~50 MiB/s per stream on modern hardware,
   well below what real HDFS achieves.
2. **Single-threaded NameNode.** asyncio + Python GIL means all metadata
   ops are serialized. Adequate for thousands of files; would saturate at
   well below HDFS's millions.
3. **Star-topology writes from client.** With replication factor 3, the
   client's upstream bandwidth is the bottleneck for writes.
4. **No batching of heartbeats / block reports.** Each is a fresh TCP
   connection.

Improving any of these would not require a redesign — they are localized
shortcuts — but doing so was out of scope for this project.

---

## References

- *The Hadoop Distributed File System*, Shvachko, Kuang, Radia, Chansler
  (MSST 2010) — the architectural blueprint this project most closely
  follows.
- *The Google File System*, Ghemawat, Gobioff, Leung (SOSP 2003) — the
  prior art for primary/secondary chunk replication. The system here has
  no leases or primary chunk server because the client writes directly to
  every replica.
- Apache Hadoop HDFS source tree — referenced for the heartbeat / block
  report cadences and the safe-mode threshold.
