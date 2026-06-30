# CRDT Collaboration API Documentation

## Table of Contents

1. [WebSocket API](#websocket-api)
2. [Client Library API](#client-library-api)
3. [CRDT Operations API](#crdt-operations-api)
4. [Server Management API](#server-management-api)
5. [Storage API](#storage-api)

## WebSocket API

### Connection

#### Endpoint
```
ws://[server]:[port]/ws
```

#### Connection Example
```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
    console.log('Connected to collaboration server');
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleMessage(message);
};

ws.onerror = (error) => {
    console.error('WebSocket error:', error);
};
```

### Message Types

#### Join Session
```json
{
    "type": "Join",
    "session_id": "collab-session-123",
    "user_id": "user-456",
    "document_id": "doc-789"
}
```

**Response:**
```json
{
    "type": "Acknowledge",
    "operation_id": "op-uuid",
    "timestamp": 1234567890
}
```

#### Leave Session
```json
{
    "type": "Leave",
    "session_id": "collab-session-123",
    "user_id": "user-456"
}
```

#### Send Operation
```json
{
    "type": "Operation",
    "operation": {
        "id": "op-uuid",
        "op_type": {
            "Insert": {
                "position": 10,
                "value": "Hello"
            }
        },
        "timestamp": {
            "replica_id": "user-456",
            "counter": 42
        },
        "dependencies": ["op-uuid-1", "op-uuid-2"]
    }
}
```

#### Update Cursor Position
```json
{
    "type": "CursorPosition",
    "user_id": "user-456",
    "position": 25
}
```

#### Update Selection
```json
{
    "type": "Selection",
    "user_id": "user-456",
    "start": 10,
    "end": 20
}
```

#### Request Sync
```json
{
    "type": "SyncRequest",
    "document_id": "doc-789",
    "from_version": 100
}
```

**Response:**
```json
{
    "type": "SyncResponse",
    "operations": [...],
    "version": 150
}
```

#### Error Message
```json
{
    "type": "Error",
    "code": "INVALID_OPERATION",
    "message": "Operation position out of bounds"
}
```

## Client Library API

### Document Operations

#### Create Document
```rust
use crdt_collaboration::document::CollaborativeDocument;

let doc = CollaborativeDocument::new(
    "doc-id".to_string(),
    "replica-id".to_string()
);
```

#### Insert Text
```rust
// Insert at position
doc.insert(position: usize, character: char) -> Result<OperationId>

// Insert string
doc.insert_text(position: usize, text: &str) -> Result<Vec<OperationId>>

// Example
doc.insert_text(0, "Hello World").await?;
```

#### Delete Text
```rust
// Delete single character
doc.delete(position: usize) -> Result<OperationId>

// Delete range
doc.delete_range(start: usize, end: usize) -> Result<Vec<OperationId>>

// Example
doc.delete_range(5, 10).await?;
```

#### Get Content
```rust
let content: String = doc.get_content().await;
println!("Document: {}", content);
```

#### Undo/Redo
```rust
// Undo last operation
doc.undo().await?;

// Redo previously undone operation
doc.redo().await?;
```

#### Metadata
```rust
// Set metadata
doc.set_metadata("author", "Alice").await;

// Get metadata
let author = doc.get_metadata("author").await;
```

### CRDT Operations API

#### LWW Register
```rust
use crdt_collaboration::crdt::LWWRegister;

let mut reg = LWWRegister::new("replica-1".to_string());

// Set value
reg.set("new value");

// Get value
let value = reg.value();

// Merge with another register
let merged = reg.merge(&other_register);
```

#### G-Counter
```rust
use crdt_collaboration::crdt::GCounter;

let mut counter = GCounter::new("replica-1".to_string());

// Increment
counter.increment();

// Get value
let value = counter.value();

// Merge
let merged = counter.merge(&other_counter);
```

#### PN-Counter
```rust
use crdt_collaboration::crdt::PNCounter;

let mut counter = PNCounter::new("replica-1".to_string());

// Operations
counter.increment();
counter.decrement();

// Get value
let value = counter.value(); // Can be negative
```

#### OR-Set
```rust
use crdt_collaboration::crdt::ORSet;

let mut set = ORSet::new("replica-1".to_string());

// Add/remove items
set.add("item1");
set.remove("item1");

// Check membership
if set.contains("item1") {
    println!("Item exists");
}

// Get all items
let items: Vec<String> = set.to_vec();
```

#### RGA List
```rust
use crdt_collaboration::crdt::RGAList;

let mut list = RGAList::new("replica-1".to_string());

// Insert at position
list.insert(0, "first");
list.insert(1, "second");

// Remove at position
list.remove(0);

// Get element
let elem = list.get(0);

// Get length
let len = list.len();
```

### Presence API

#### Create Presence Manager
```rust
use crdt_collaboration::presence::PresenceManager;

let mut presence = PresenceManager::new("user-id".to_string());
```

#### Update Presence
```rust
// Update cursor
presence.update_cursor(position: usize);

// Update selection
presence.update_selection(Some((start, end)));

// Update status
presence.update_status(PresenceStatus::Active);
```

#### Track Peers
```rust
// Update peer presence
presence.update_peer_presence("peer-id", peer_info);

// Get all peers
let peers = presence.get_all_peers();

// Get specific peer
let peer = presence.get_peer("peer-id");
```

## Server Management API

### Start Server
```rust
use crdt_collaboration::server::CollaborationServer;

let mut server = CollaborationServer::new("127.0.0.1:8080");
server.start().await?;
```

### Session Management
```rust
// Create session
server.create_session(session_id: String) -> SessionHandle

// Get active sessions
let sessions = server.get_active_sessions();

// Get session info
let info = server.get_session_info(session_id);

// Close session
server.close_session(session_id).await;
```

### Client Management
```rust
// Get connected clients
let clients = server.get_connected_clients();

// Disconnect client
server.disconnect_client(client_id).await;

// Get client info
let info = server.get_client_info(client_id);
```

## Storage API

### Document Storage
```rust
use crdt_collaboration::storage::DocumentStorage;

let storage = DocumentStorage::new("./data").await?;

// Save document
storage.save_document(&document).await?;

// Load document
let doc = storage.load_document("doc-id").await?;

// Delete document
storage.delete_document("doc-id").await?;

// List documents
let docs = storage.list_documents().await?;
```

### Operation Log
```rust
// Append operation
storage.append_operation("doc-id", operation).await?;

// Get operations since version
let ops = storage.get_operations_since("doc-id", version).await?;

// Compact log
storage.compact_log("doc-id", target_version).await?;
```

### Snapshots
```rust
// Create snapshot
let snapshot = storage.create_snapshot(&document).await?;

// Load snapshot
let doc = storage.load_from_snapshot(snapshot_id).await?;

// List snapshots
let snapshots = storage.list_snapshots("doc-id").await?;

// Prune old snapshots
storage.prune_snapshots("doc-id", keep_last: 5).await?;
```

## Error Handling

### Error Types
```rust
pub enum CollaborationError {
    InvalidOperation(String),
    PositionOutOfBounds { position: usize, length: usize },
    DocumentNotFound(String),
    SessionNotFound(String),
    NetworkError(String),
    StorageError(String),
    SerializationError(String),
}
```

### Error Handling Example
```rust
match doc.insert(position, 'x').await {
    Ok(op_id) => {
        println!("Operation {} successful", op_id);
    }
    Err(CollaborationError::PositionOutOfBounds { position, length }) => {
        eprintln!("Invalid position {} for document length {}", position, length);
    }
    Err(e) => {
        eprintln!("Operation failed: {}", e);
    }
}
```

## Rate Limiting

### Client Rate Limits
- Operations: 100/second per client
- Sync requests: 10/minute per client
- Join/leave: 5/minute per client

### Example Rate Limit Response
```json
{
    "type": "Error",
    "code": "RATE_LIMITED",
    "message": "Too many requests",
    "retry_after": 5000
}
```

## Authentication

### Token-based Auth
```javascript
const ws = new WebSocket('ws://localhost:8080/ws', {
    headers: {
        'Authorization': 'Bearer YOUR_AUTH_TOKEN'
    }
});
```

### Session Tokens
```json
{
    "type": "Authenticate",
    "token": "session-token-xyz",
    "user_id": "user-456"
}
```

## Examples

### Complete Collaboration Session
```rust
use crdt_collaboration::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create document
    let mut doc = document::CollaborativeDocument::new(
        "doc-1".to_string(),
        "user-1".to_string()
    );

    // Connect to server
    let client = client::CollaborationClient::connect("ws://localhost:8080").await?;

    // Join session
    client.join_session("session-1", "user-1", "doc-1").await?;

    // Make edits
    doc.insert_text(0, "Hello collaborative world!").await?;

    // Send operations
    let ops = doc.get_operations_since(0).await;
    for op in ops {
        client.send_operation(op).await?;
    }

    // Listen for remote operations
    while let Some(msg) = client.receive().await {
        match msg {
            Message::Operation { operation } => {
                doc.apply_operation(operation).await?;
            }
            Message::CursorPosition { user_id, position } => {
                println!("{} cursor at {}", user_id, position);
            }
            _ => {}
        }
    }

    Ok(())
}
```

### Conflict Resolution Example
```rust
// Two users edit same position concurrently
let mut doc1 = CollaborativeDocument::new("doc", "user1");
let mut doc2 = CollaborativeDocument::new("doc", "user2");

// Both insert at position 0
doc1.insert_text(0, "Alice: ").await?;
doc2.insert_text(0, "Bob: ").await?;

// Exchange and apply operations
let ops1 = doc1.get_operations_since(0).await;
let ops2 = doc2.get_operations_since(0).await;

for op in ops2 {
    doc1.apply_operation(op).await?;
}

for op in ops1 {
    doc2.apply_operation(op).await?;
}

// Both documents converge to same state
assert_eq!(doc1.get_content().await, doc2.get_content().await);
```