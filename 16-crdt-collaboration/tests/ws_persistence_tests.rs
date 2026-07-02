//! End-to-end tests for the runnable server: WebSocket collaboration
//! convergence, handshake auth, and SQLite persistence across restarts.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use crdt_collaboration::api::{create_router, ApiState};
use crdt_collaboration::crdt::{Operation, PositionId, VectorClock};
use crdt_collaboration::document::{Document, DocumentMetadata};
use crdt_collaboration::persistent::SqliteBackend;
use crdt_collaboration::protocol::{Message, OperationMessage};
use crdt_collaboration::server::{CollaborationServer, ServerConfig};
use crdt_collaboration::storage::StorageManager;
use crdt_collaboration::{ClientId, DocumentId};

use futures_util::{SinkExt, StreamExt};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Serializes tests that read/mutate the process-global `API_KEYS` env var so
/// they do not race (the router snapshots it at construction time).
static ENV_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Boot the server on an ephemeral port and return its address plus the shared
/// storage handle (so tests can inspect persisted state).
async fn spawn_server(storage: Arc<StorageManager>) -> SocketAddr {
    let server = Arc::new(CollaborationServer::new(
        ServerConfig::default(),
        storage.clone(),
    ));
    let state = Arc::new(ApiState::new(server, storage));
    let app = create_router(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });

    // Give the listener a moment to become ready.
    tokio::time::sleep(Duration::from_millis(50)).await;
    addr
}

/// Build a single-character insert operation at the document root.
fn insert_op(client: ClientId, value: char, lamport: u64) -> Operation {
    Operation::Insert {
        id: PositionId::new(lamport, client, 1),
        after: PositionId::root(),
        value,
        attributes: HashMap::new(),
    }
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn test_two_ws_clients_converge() {
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::remove_var("API_KEYS");
    let storage = Arc::new(StorageManager::new());
    let addr = spawn_server(storage.clone()).await;

    let doc_id = DocumentId::new_v4();
    let url = format!("ws://{}/ws/{}", addr, doc_id);

    // Connect two independent clients to the same document.
    let (mut ws_a, _) = tokio_tungstenite::connect_async(&url).await.unwrap();
    let (mut ws_b, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

    // Both receive an initial SyncResponse on connect. Drain client B's.
    // (Client A's is drained implicitly when we look for its ack below.)
    let _ = ws_b.next().await;

    // Client A applies an insert of 'H'.
    let client_a = ClientId::new_v4();
    let op = insert_op(client_a, 'H', 1);
    let msg = Message::Operation(OperationMessage {
        doc_id,
        client_id: client_a,
        vector_clock: VectorClock::new(),
        operations: vec![op.clone()],
        seq: 0,
    });
    ws_a
        .send(WsMessage::Text(serde_json::to_string(&msg).unwrap()))
        .await
        .unwrap();

    // Client B should receive the broadcast operation and apply it locally.
    let mut doc_b = Document::new(doc_id);
    let mut got_op = false;
    for _ in 0..10 {
        match tokio::time::timeout(Duration::from_secs(2), ws_b.next()).await {
            Ok(Some(Ok(WsMessage::Text(text)))) => {
                if let Ok(Message::Operation(op_msg)) = serde_json::from_str::<Message>(&text) {
                    for o in &op_msg.operations {
                        doc_b.apply(o).unwrap();
                    }
                    got_op = true;
                    break;
                }
            }
            Ok(Some(Ok(_))) => continue,
            _ => break,
        }
    }
    assert!(got_op, "client B never received the broadcast operation");

    // The server-side document should now read "H", and client B converges.
    let server_snapshot = storage.load_snapshot(&doc_id).await.unwrap();
    // The op was applied in-session; fetch via a fresh sync from client B too.
    assert_eq!(doc_b.text(), "H", "client B did not converge to server state");

    // Server persisted a snapshot on the operation; confirm it matches.
    if let Some(snap) = server_snapshot {
        let server_doc = Document::from_snapshot(snap);
        assert_eq!(server_doc.text(), "H");
    }
}

#[tokio::test]
#[allow(clippy::await_holding_lock)]
async fn test_ws_auth_rejects_without_key() {
    let _guard = ENV_GUARD.lock().unwrap();
    std::env::set_var("API_KEYS", "ws-secret");
    let storage = Arc::new(StorageManager::new());
    let addr = spawn_server(storage).await;
    let doc_id = DocumentId::new_v4();

    // No key: handshake should fail (HTTP 401 before upgrade).
    let url = format!("ws://{}/ws/{}", addr, doc_id);
    let result = tokio_tungstenite::connect_async(&url).await;
    assert!(result.is_err(), "handshake should be rejected without a key");

    // With a valid key via query param: handshake succeeds.
    let url_ok = format!("ws://{}/ws/{}?api_key=ws-secret", addr, doc_id);
    let ok = tokio_tungstenite::connect_async(&url_ok).await;
    assert!(ok.is_ok(), "handshake with valid key should succeed");

    std::env::remove_var("API_KEYS");
}

#[tokio::test]
async fn test_sqlite_persistence_survives_restart() {
    let tmp = std::env::temp_dir().join(format!("crdt_persist_{}.db", uuid::Uuid::new_v4()));
    let path = tmp.to_str().unwrap().to_string();

    let doc_id = DocumentId::new_v4();
    let owner = ClientId::new_v4();

    // --- First "process": write a document through the store. ---
    {
        let backend = Arc::new(SqliteBackend::new(&path).unwrap());
        let storage = StorageManager::with_backend(backend);
        assert!(storage.is_persistent());

        storage
            .save_metadata(DocumentMetadata {
                id: doc_id,
                title: "Persisted".to_string(),
                owner,
                created_at: 1,
                updated_at: 1,
                version: 1,
                archived: false,
            })
            .await
            .unwrap();

        // Build a document with real CRDT content and persist its snapshot.
        let mut doc = Document::new(doc_id);
        let client = ClientId::new_v4();
        doc.insert(client, PositionId::root(), 'H', HashMap::new())
            .unwrap();
        doc.insert(client, doc.position_at(1).unwrap(), 'i', HashMap::new())
            .unwrap();
        assert_eq!(doc.text(), "Hi");

        storage.save_snapshot(&doc.snapshot()).await.unwrap();
    }
    // Store dropped here, closing the SQLite connection.

    // --- Second "process": recreate over the same path and recover. ---
    {
        let backend = Arc::new(SqliteBackend::new(&path).unwrap());
        let storage = StorageManager::with_backend(backend);

        let meta = storage.load_metadata(&doc_id).await.unwrap();
        assert!(meta.is_some(), "metadata must survive restart");
        assert_eq!(meta.unwrap().title, "Persisted");

        let snap = storage
            .load_snapshot(&doc_id)
            .await
            .unwrap()
            .expect("snapshot must survive restart");
        let doc = Document::from_snapshot(snap);
        assert_eq!(
            doc.text(),
            "Hi",
            "recovered document must reconstruct its CRDT state"
        );
    }

    let _ = std::fs::remove_file(&path);
}
