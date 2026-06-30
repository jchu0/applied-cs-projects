# CRDT-Based Real-Time Collaboration Engine

> **Concepts covered:** §01 software-engineering — `typescript/*`; §02 data-engineering — `04-streaming` (distributed state)

## Project Overview

A Google Docs-lite collaborative editing system built on Conflict-free Replicated Data Types (CRDTs). This engine enables real-time multi-user document editing with automatic conflict resolution, offline support, and eventual consistency guarantees without centralized locking or consensus protocols.

## Architecture

### High-Level Design

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client A      │     │   Client B      │     │   Client C      │
│  ┌───────────┐  │     │  ┌───────────┐  │     │  ┌───────────┐  │
│  │ Editor UI │  │     │  │ Editor UI │  │     │  │ Editor UI │  │
│  │(ProseMirror)│ │     │  │(ProseMirror)│ │     │  │(CodeMirror)│ │
│  └─────┬─────┘  │     │  └─────┬─────┘  │     │  └─────┬─────┘  │
│  ┌─────┴─────┐  │     │  ┌─────┴─────┐  │     │  ┌─────┴─────┐  │
│  │CRDT Engine│  │     │  │CRDT Engine│  │     │  │CRDT Engine│  │
│  └─────┬─────┘  │     │  └─────┬─────┘  │     │  └─────┬─────┘  │
└────────┼────────┘     └────────┼────────┘     └────────┼────────┘
         │                       │                       │
         └───────────┬───────────┴───────────┬───────────┘
                     │    WebSocket          │
              ┌──────┴──────────────────────┴──────┐
              │       Collaboration Server         │
              │  ┌────────────┐ ┌──────────────┐   │
              │  │  Document  │ │   Presence   │   │
              │  │  Manager   │ │   Tracker    │   │
              │  └─────┬──────┘ └──────────────┘   │
              │  ┌─────┴──────┐ ┌──────────────┐   │
              │  │ Op Router  │ │  ACL Engine  │   │
              │  └─────┬──────┘ └──────────────┘   │
              └────────┼──────────────────────────┘
                       │
              ┌────────┴────────┐
              │  Storage Layer  │
              │ ┌─────┐ ┌─────┐ │
              │ │State│ │OpLog│ │
              │ └─────┘ └─────┘ │
              └─────────────────┘
```

### Core Components

#### 1. Client Editor Layer
- **ProseMirror Integration**: Rich text editing with schema-defined document structure
- **CodeMirror Integration**: Code editing with syntax highlighting
- **CRDT Adapter**: Translates editor operations to CRDT operations
- **Local State**: Maintains local CRDT replica for immediate responsiveness

#### 2. CRDT Engine
- **Sequence CRDT**: RGA/LSEQ/Yjs-style for text sequences
- **Map CRDT**: LWW-Register or MV-Register for key-value attributes
- **Counter CRDT**: G-Counter/PN-Counter for numeric values
- **Tombstone Management**: Tracks deleted elements for consistency

#### 3. Collaboration Server
- **Document Manager**: Handles document lifecycle, versioning
- **Operation Router**: Broadcasts operations to connected clients
- **Presence Tracker**: Manages cursor positions, selections, user status
- **ACL Engine**: Per-document access control and permissions

#### 4. Storage Layer
- **State Store**: Compacted CRDT state snapshots
- **Operation Log**: Append-only log for operation history
- **Checkpoint Manager**: Periodic state snapshots for fast recovery

## CRDT Internals

### Sequence CRDT Design (RGA-based)

```rust
// Position identifier for unique, ordered elements
struct PositionID {
    lamport: u64,           // Lamport timestamp
    client_id: ClientID,    // Unique client identifier
    seq: u32,               // Sequence number within same lamport
}

// Operation types
enum Operation {
    Insert {
        id: PositionID,
        after: PositionID,  // Position to insert after
        value: char,        // Character value (or node for rich text)
        attributes: HashMap<String, AttributeValue>,
    },
    Delete {
        id: PositionID,
        deleted_by: PositionID,
    },
    Format {
        start: PositionID,
        end: PositionID,
        attribute: String,
        value: AttributeValue,
    },
}

// CRDT state representation
struct Document {
    elements: BTreeMap<PositionID, Element>,
    tombstones: HashSet<PositionID>,
    vector_clock: VectorClock,
}

struct Element {
    id: PositionID,
    value: char,
    left: Option<PositionID>,
    right: Option<PositionID>,
    attributes: HashMap<String, (u64, AttributeValue)>, // (timestamp, value)
}
```

### Vector Clocks and Causality

```rust
struct VectorClock {
    clocks: HashMap<ClientID, u64>,
}

impl VectorClock {
    fn increment(&mut self, client_id: ClientID) {
        *self.clocks.entry(client_id).or_insert(0) += 1;
    }

    fn merge(&mut self, other: &VectorClock) {
        for (client, &time) in &other.clocks {
            let entry = self.clocks.entry(*client).or_insert(0);
            *entry = (*entry).max(time);
        }
    }

    fn happens_before(&self, other: &VectorClock) -> bool {
        let mut dominated = false;
        for (client, &time) in &self.clocks {
            let other_time = other.clocks.get(client).copied().unwrap_or(0);
            if time > other_time {
                return false;
            }
            if time < other_time {
                dominated = true;
            }
        }
        dominated || other.clocks.len() > self.clocks.len()
    }
}
```

### Alternative: LSEQ Position Generation

```rust
struct LSEQPosition {
    path: Vec<(u16, BoundaryStrategy)>,
}

enum BoundaryStrategy {
    Left,   // Allocate from left boundary
    Right,  // Allocate from right boundary
}

impl LSEQPosition {
    fn between(left: &LSEQPosition, right: &LSEQPosition,
               level: usize, client_id: ClientID) -> LSEQPosition {
        // Generates position between two positions
        // Uses adaptive boundary strategy per level
        // Ensures unique, ordered positions
    }
}
```

## WebSocket Protocol

### Message Format

```typescript
interface WSMessage {
    type: MessageType;
    doc_id: string;
    client_id: string;
    timestamp: number;
    payload: MessagePayload;
}

enum MessageType {
    // Document operations
    OPERATION = 'operation',
    OPERATION_ACK = 'operation_ack',
    SYNC_REQUEST = 'sync_request',
    SYNC_RESPONSE = 'sync_response',

    // Presence
    CURSOR_UPDATE = 'cursor_update',
    SELECTION_UPDATE = 'selection_update',
    USER_JOIN = 'user_join',
    USER_LEAVE = 'user_leave',

    // Control
    HEARTBEAT = 'heartbeat',
    ERROR = 'error',
}

interface OperationMessage {
    type: 'operation';
    doc_id: string;
    client_id: string;
    vector_clock: Record<string, number>;
    operations: CRDTOperation[];
}

interface SyncResponse {
    type: 'sync_response';
    doc_id: string;
    state: CompactedState;
    pending_ops: CRDTOperation[];
    vector_clock: Record<string, number>;
}
```

### Connection Lifecycle

```
Client                          Server
  │                                │
  ├─── CONNECT ──────────────────►│
  │                                │
  │◄── AUTH_CHALLENGE ─────────────┤
  │                                │
  ├─── AUTH_RESPONSE ────────────►│
  │                                │
  │◄── AUTH_SUCCESS ───────────────┤
  │                                │
  ├─── JOIN_DOCUMENT ────────────►│
  │                                │
  │◄── SYNC_RESPONSE ──────────────┤
  │◄── PRESENCE_UPDATE ────────────┤
  │                                │
  ├─── OPERATION ────────────────►│
  │◄── OPERATION_ACK ──────────────┤
  │◄── OPERATION (from others) ────┤
  │                                │
  ├─── CURSOR_UPDATE ────────────►│
  │◄── CURSOR_UPDATE (broadcast) ──┤
  │                                │
```

## API Design

### Server API

```rust
// Document management
POST   /api/documents              // Create document
GET    /api/documents/:id          // Get document metadata
DELETE /api/documents/:id          // Archive document
GET    /api/documents/:id/history  // Get operation history
POST   /api/documents/:id/snapshot // Create snapshot

// Access control
GET    /api/documents/:id/acl      // Get ACL
PUT    /api/documents/:id/acl      // Update ACL
POST   /api/documents/:id/share    // Generate share link

// WebSocket
WS     /ws/collaborate             // Real-time collaboration
```

### Client SDK API

```typescript
class CollaborationClient {
    // Connection
    connect(url: string, options: ConnectionOptions): Promise<void>;
    disconnect(): void;

    // Document operations
    openDocument(docId: string): Promise<CollaborativeDocument>;
    closeDocument(docId: string): void;

    // Awareness/Presence
    setUserInfo(info: UserInfo): void;
    setCursor(position: Position): void;
    setSelection(range: Range): void;

    // Events
    on(event: 'operation', handler: (op: Operation) => void): void;
    on(event: 'presence', handler: (presence: PresenceUpdate) => void): void;
    on(event: 'disconnect', handler: () => void): void;
}

class CollaborativeDocument {
    // Local operations
    insert(position: number, text: string): Operation[];
    delete(start: number, end: number): Operation[];
    format(start: number, end: number, attributes: Attributes): Operation[];

    // State
    getText(): string;
    getState(): DocumentState;
    getVersion(): VectorClock;

    // Undo/Redo
    undo(): void;
    redo(): void;

    // Events
    on(event: 'update', handler: (update: Update) => void): void;
}
```

## Enterprise Features

### 1. Presence and Awareness

```typescript
interface PresenceState {
    user_id: string;
    name: string;
    color: string;
    cursor: {
        position: number;
        anchor: number;  // For selections
    };
    selection: {
        start: number;
        end: number;
    };
    last_active: number;
    status: 'active' | 'idle' | 'away';
}

// Server broadcasts presence updates
// Client renders cursors with user colors
// Idle detection after 30s, away after 5min
```

### 2. Offline Editing

```typescript
class OfflineManager {
    // Persistence
    saveLocalState(docId: string, state: DocumentState): void;
    loadLocalState(docId: string): DocumentState | null;

    // Queue management
    queueOperation(op: Operation): void;
    getPendingOperations(): Operation[];

    // Sync
    async syncOnReconnect(): Promise<SyncResult>;
    resolveConflicts(local: State, remote: State): State;
}

// IndexedDB storage for local state
// Operation queue persisted for durability
// Automatic sync with exponential backoff on reconnect
```

### 3. Access Control Lists (ACLs)

```rust
enum Permission {
    Read,
    Write,
    Comment,
    Admin,
}

struct ACLEntry {
    principal: Principal,  // User, Group, or Role
    permissions: Vec<Permission>,
    granted_by: UserId,
    granted_at: Timestamp,
    expires_at: Option<Timestamp>,
}

struct DocumentACL {
    doc_id: DocumentId,
    owner: UserId,
    entries: Vec<ACLEntry>,
    public_access: Option<Permission>,
    link_sharing: Option<LinkShare>,
}

// Permission checks on every operation
// Hierarchical inheritance (org > team > document)
// Time-limited access for external sharing
```

### 4. Version History and Snapshots

```rust
struct Snapshot {
    id: SnapshotId,
    doc_id: DocumentId,
    timestamp: Timestamp,
    vector_clock: VectorClock,
    state: CompactedState,
    created_by: Option<UserId>,  // None for auto-snapshots
}

// Auto-snapshot every N operations or T time
// Named snapshots for user-created versions
// Efficient diff between snapshots
// Restore to any snapshot
```

### 5. Audit Logging

```rust
struct AuditEvent {
    event_id: Uuid,
    timestamp: Timestamp,
    user_id: UserId,
    doc_id: DocumentId,
    action: AuditAction,
    details: serde_json::Value,
    ip_address: IpAddr,
    user_agent: String,
}

enum AuditAction {
    DocumentCreated,
    DocumentOpened,
    DocumentEdited,
    DocumentShared,
    PermissionChanged,
    DocumentDeleted,
    DocumentRestored,
}
```

## Performance Considerations

### Client-Side Optimizations

1. **Operation Batching**
   - Batch rapid keystrokes into single operations
   - Configurable batch window (default: 50ms)
   - Flush on pause or explicit save

2. **Lazy Rendering**
   - Only render visible portion of document
   - Virtual scrolling for large documents
   - Incremental DOM updates

3. **State Compaction**
   - Periodically compact local CRDT state
   - Garbage collect old tombstones
   - Merge adjacent text nodes

### Server-Side Optimizations

1. **Document Sharding**
   - Shard documents by ID across server instances
   - Consistent hashing for routing
   - Hot-standby replicas for failover

2. **Operation Coalescing**
   - Coalesce operations before broadcast
   - Reduces network overhead
   - Maintains causal ordering

3. **Checkpoint Strategy**
   - Snapshot every 1000 operations
   - Async snapshot to avoid blocking
   - Prune old operation log entries

### Memory Management

```rust
// Tombstone garbage collection
impl Document {
    fn gc_tombstones(&mut self, min_vector_clock: &VectorClock) {
        // Only GC tombstones seen by all clients
        self.tombstones.retain(|id| {
            let op_vc = self.get_operation_clock(id);
            !min_vector_clock.dominates(&op_vc)
        });
    }
}

// String interning for repeated content
// Rope data structure for large documents
// Memory-mapped operation log
```

### Benchmarks

| Operation | Target Latency | Throughput |
|-----------|---------------|------------|
| Local insert | < 1ms | 10K ops/s |
| Remote op apply | < 5ms | 5K ops/s |
| Document sync | < 100ms | 100 docs/s |
| Snapshot create | < 50ms | 50/s |
| Presence broadcast | < 20ms | 1K clients |

## Implementation Phases

### Phase 1: Core CRDT Engine (Weeks 1-3)
- [ ] Implement RGA sequence CRDT
- [ ] Position ID generation and comparison
- [ ] Insert/Delete operations
- [ ] Vector clock tracking
- [ ] Basic merge algorithm
- [ ] Unit tests for CRDT properties

### Phase 2: WebSocket Server (Weeks 4-5)
- [ ] WebSocket connection handling
- [ ] Document session management
- [ ] Operation broadcasting
- [ ] Client connection tracking
- [ ] Basic authentication
- [ ] Message serialization

### Phase 3: Client Integration (Weeks 6-7)
- [ ] ProseMirror plugin for CRDT
- [ ] Editor operation translation
- [ ] Local state management
- [ ] Optimistic updates
- [ ] Connection management

### Phase 4: Storage Layer (Weeks 8-9)
- [ ] Operation log persistence
- [ ] State snapshots
- [ ] Document metadata storage
- [ ] Efficient state loading
- [ ] Log compaction

### Phase 5: Presence System (Week 10)
- [ ] Cursor position tracking
- [ ] Selection broadcasting
- [ ] User status management
- [ ] Presence rendering in editor
- [ ] Idle/away detection

### Phase 6: Offline Support (Weeks 11-12)
- [ ] IndexedDB state persistence
- [ ] Operation queue
- [ ] Reconnection sync
- [ ] Conflict resolution UI
- [ ] Optimistic offline editing

### Phase 7: Enterprise Features (Weeks 13-14)
- [ ] ACL implementation
- [ ] Audit logging
- [ ] Version history
- [ ] Share links
- [ ] Admin dashboard

### Phase 8: Performance & Polish (Weeks 15-16)
- [ ] Operation batching
- [ ] State compaction
- [ ] Memory optimization
- [ ] Load testing
- [ ] Documentation

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_concurrent_inserts() {
    let mut doc1 = Document::new("client1");
    let mut doc2 = Document::new("client2");

    // Concurrent inserts at same position
    let op1 = doc1.insert(0, 'A');
    let op2 = doc2.insert(0, 'B');

    // Apply operations in different orders
    doc1.apply(&op2);
    doc2.apply(&op1);

    // Must converge to same state
    assert_eq!(doc1.text(), doc2.text());
}

#[test]
fn test_crdt_properties() {
    // Commutativity: apply(op1, op2) == apply(op2, op1)
    // Associativity: apply(apply(s, op1), op2) == apply(s, merge(op1, op2))
    // Idempotency: apply(s, op) == apply(apply(s, op), op)
}
```

### Integration Tests

```typescript
describe('Collaboration Flow', () => {
    it('should sync operations between clients', async () => {
        const client1 = await createClient('user1');
        const client2 = await createClient('user2');

        const doc1 = await client1.openDocument('test-doc');
        const doc2 = await client2.openDocument('test-doc');

        doc1.insert(0, 'Hello');
        await waitForSync();

        expect(doc2.getText()).toBe('Hello');
    });

    it('should handle offline editing', async () => {
        const client = await createClient('user');
        const doc = await client.openDocument('test-doc');

        client.goOffline();
        doc.insert(0, 'Offline edit');

        await client.goOnline();
        await waitForSync();

        // Verify sync with server
    });
});
```

### Property-Based Tests

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_convergence(
        ops1 in vec(operation_strategy(), 0..100),
        ops2 in vec(operation_strategy(), 0..100)
    ) {
        let mut doc1 = Document::new("client1");
        let mut doc2 = Document::new("client2");

        for op in &ops1 { doc1.apply(op); }
        for op in &ops2 { doc2.apply(op); }

        // Cross-apply
        for op in &ops2 { doc1.apply(op); }
        for op in &ops1 { doc2.apply(op); }

        assert_eq!(doc1.state(), doc2.state());
    }
}
```

### Load Testing

```bash
# Simulate 1000 concurrent users
k6 run --vus 1000 --duration 5m collaboration-load-test.js

# Metrics to track:
# - Operation latency p50, p95, p99
# - Sync lag between clients
# - Server memory usage
# - WebSocket connection stability
```

## Stretch Goals

### Multi-Document System
- Document linking and embedding
- Cross-document references
- Workspace organization
- Bulk operations across documents

### CRDT for Rich Content
- Comment threads as CRDT
- Annotations with positions
- Drawing/shapes layer
- Tables and structured data

### Advanced Collaboration
- Suggested edits mode
- Branch/merge workflows
- Real-time voice/video integration
- AI-powered conflict resolution

### Performance Enhancements
- WASM CRDT engine for client
- Binary protocol (Protocol Buffers)
- Delta compression for sync
- P2P direct connections between clients

## Technology Stack

### Server
- **Runtime**: Rust with Tokio async runtime
- **WebSocket**: tokio-tungstenite
- **Storage**: PostgreSQL + Redis
- **Message Queue**: NATS or Redis Streams

### Client
- **Editor**: ProseMirror / CodeMirror 6
- **CRDT**: Yjs or custom implementation
- **State**: IndexedDB for persistence
- **Transport**: Native WebSocket

### Infrastructure
- **Load Balancer**: Nginx with sticky sessions
- **Container**: Docker + Kubernetes
- **Monitoring**: Prometheus + Grafana
- **Tracing**: Jaeger

## References

- [CRDT.tech](https://crdt.tech/) - CRDT resources and papers
- [Yjs](https://github.com/yjs/yjs) - Reference CRDT implementation
- [Automerge](https://automerge.org/) - JSON CRDT library
- [ProseMirror](https://prosemirror.net/) - Rich text editor
- [Martin Kleppmann's CRDT papers](https://martin.kleppmann.com/)
