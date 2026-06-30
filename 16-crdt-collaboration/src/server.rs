//! Collaboration server implementation.

use crate::crdt::Operation;
use crate::document::Document;
use crate::presence::{PresenceManager, PresenceState};
use crate::protocol::{Message, OperationAck, OperationMessage, SyncResponse, UserJoin, UserLeave};
use crate::storage::StorageManager;
use crate::{ClientId, DocumentId, Result};
use dashmap::DashMap;
use futures_util::{SinkExt, StreamExt};
use parking_lot::RwLock;
use std::sync::Arc;
use tokio::sync::broadcast;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message as WsMessage;

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Maximum connections per document.
    pub max_connections_per_doc: usize,
    /// Operation batch window in milliseconds.
    pub batch_window_ms: u64,
    /// Heartbeat interval in seconds.
    pub heartbeat_interval_secs: u64,
    /// Snapshot interval (number of operations).
    pub snapshot_interval: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_connections_per_doc: 100,
            batch_window_ms: 50,
            heartbeat_interval_secs: 30,
            snapshot_interval: 1000,
        }
    }
}

/// Document session.
pub struct DocumentSession {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Document state.
    pub document: RwLock<Document>,
    /// Connected clients.
    pub clients: DashMap<ClientId, ClientConnection>,
    /// Broadcast channel for operations.
    pub broadcast: broadcast::Sender<Message>,
    /// Operation counter.
    pub op_counter: std::sync::atomic::AtomicU64,
}

impl DocumentSession {
    /// Create new document session.
    pub fn new(doc_id: DocumentId, document: Document) -> Self {
        let (broadcast, _) = broadcast::channel(1000);
        Self {
            doc_id,
            document: RwLock::new(document),
            clients: DashMap::new(),
            broadcast,
            op_counter: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Add a client.
    pub fn add_client(&self, client_id: ClientId, connection: ClientConnection) {
        self.clients.insert(client_id, connection);
    }

    /// Remove a client.
    pub fn remove_client(&self, client_id: &ClientId) {
        self.clients.remove(client_id);
    }

    /// Get client count.
    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// Apply operations and broadcast.
    pub fn apply_operations(
        &self,
        client_id: ClientId,
        operations: Vec<Operation>,
    ) -> Result<u64> {
        let mut doc = self.document.write();

        for op in &operations {
            doc.apply(op)?;
        }

        let seq = self.op_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Broadcast to other clients
        let msg = Message::Operation(OperationMessage {
            doc_id: self.doc_id,
            client_id,
            vector_clock: doc.vector_clock.clone(),
            operations,
            seq,
        });

        let _ = self.broadcast.send(msg);

        Ok(seq)
    }

    /// Get sync response for new client.
    pub fn get_sync_response(&self) -> SyncResponse {
        let doc = self.document.read();
        let snapshot = doc.snapshot();

        SyncResponse {
            doc_id: self.doc_id,
            state: serde_json::to_vec(&snapshot).unwrap_or_default(),
            pending_ops: vec![],
            vector_clock: doc.vector_clock.clone(),
        }
    }
}

/// Client connection info.
#[derive(Debug)]
pub struct ClientConnection {
    /// Client ID.
    pub client_id: ClientId,
    /// Message sender.
    pub sender: mpsc::Sender<Message>,
    /// User name.
    pub name: String,
    /// User color.
    pub color: String,
}

/// Collaboration server.
pub struct CollaborationServer {
    /// Server configuration.
    pub config: ServerConfig,
    /// Document sessions.
    pub sessions: DashMap<DocumentId, Arc<DocumentSession>>,
    /// Presence manager.
    pub presence: PresenceManager,
    /// Storage manager.
    pub storage: Arc<StorageManager>,
}

impl CollaborationServer {
    /// Create new server.
    pub fn new(config: ServerConfig, storage: Arc<StorageManager>) -> Self {
        Self {
            config,
            sessions: DashMap::new(),
            presence: PresenceManager::new(),
            storage,
        }
    }

    /// Get or create document session.
    pub async fn get_or_create_session(
        &self,
        doc_id: DocumentId,
    ) -> Result<Arc<DocumentSession>> {
        if let Some(session) = self.sessions.get(&doc_id) {
            return Ok(session.clone());
        }

        // Load from storage or create new
        let document = if let Some(snapshot) = self.storage.load_snapshot(&doc_id).await? {
            Document::from_snapshot(snapshot)
        } else {
            Document::new(doc_id)
        };

        let session = Arc::new(DocumentSession::new(doc_id, document));
        self.sessions.insert(doc_id, session.clone());

        Ok(session)
    }

    /// Handle client connection.
    pub async fn handle_client(
        self: Arc<Self>,
        client_id: ClientId,
        doc_id: DocumentId,
        name: String,
        color: String,
        mut ws_receiver: impl StreamExt<Item = std::result::Result<WsMessage, tokio_tungstenite::tungstenite::Error>> + Unpin,
        mut ws_sender: impl SinkExt<WsMessage> + Unpin,
    ) -> Result<()> {
        let session = self.get_or_create_session(doc_id).await?;

        // Check connection limit
        if session.client_count() >= self.config.max_connections_per_doc {
            let error = Message::Error(crate::protocol::ErrorMessage {
                code: 429,
                message: "Too many connections".into(),
            });
            let _ = ws_sender.send(WsMessage::Text(
                serde_json::to_string(&error).unwrap()
            )).await;
            return Ok(());
        }

        // Create message channel
        let (tx, mut rx) = mpsc::channel::<Message>(100);

        // Add client to session
        let connection = ClientConnection {
            client_id,
            sender: tx.clone(),
            name: name.clone(),
            color: color.clone(),
        };
        session.add_client(client_id, connection);

        // Add to presence
        let presence = self.presence.get_or_create(doc_id);
        let state = PresenceState::new(client_id, name.clone(), color.clone());
        presence.add_user(state.clone());

        // Subscribe to broadcast
        let mut broadcast_rx = session.broadcast.subscribe();

        // Send sync response
        let sync_response = Message::SyncResponse(session.get_sync_response());
        let _ = ws_sender.send(WsMessage::Text(
            serde_json::to_string(&sync_response).unwrap()
        )).await;

        // Broadcast user join
        let join_msg = Message::UserJoin(UserJoin {
            doc_id,
            user: crate::protocol::UserInfo {
                client_id,
                name: name.clone(),
                color: color.clone(),
                avatar_url: None,
            },
        });
        let _ = session.broadcast.send(join_msg);

        // Main loop
        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                Some(msg) = ws_receiver.next() => {
                    match msg {
                        Ok(WsMessage::Text(text)) => {
                            if let Ok(message) = serde_json::from_str::<Message>(&text) {
                                self.handle_message(&session, client_id, message).await?;
                            }
                        }
                        Ok(WsMessage::Close(_)) => break,
                        Err(_) => break,
                        _ => {}
                    }
                }

                // Handle outgoing messages from channel
                Some(msg) = rx.recv() => {
                    let text = serde_json::to_string(&msg).unwrap();
                    if ws_sender.send(WsMessage::Text(text)).await.is_err() {
                        break;
                    }
                }

                // Handle broadcast messages
                Ok(msg) = broadcast_rx.recv() => {
                    // Don't send back to originator
                    if let Message::Operation(ref op) = msg {
                        if op.client_id == client_id {
                            continue;
                        }
                    }

                    let text = serde_json::to_string(&msg).unwrap();
                    if ws_sender.send(WsMessage::Text(text)).await.is_err() {
                        break;
                    }
                }
            }
        }

        // Cleanup
        session.remove_client(&client_id);
        presence.remove_user(&client_id);

        // Broadcast user leave
        let leave_msg = Message::UserLeave(UserLeave {
            doc_id,
            client_id,
        });
        let _ = session.broadcast.send(leave_msg);

        // Cleanup empty sessions
        if session.client_count() == 0 {
            // Save snapshot before removing
            let doc = session.document.read();
            let _ = self.storage.save_snapshot(&doc.snapshot()).await;
            self.sessions.remove(&doc_id);
        }

        Ok(())
    }

    /// Handle a message from a client.
    async fn handle_message(
        &self,
        session: &DocumentSession,
        client_id: ClientId,
        message: Message,
    ) -> Result<()> {
        match message {
            Message::Operation(op_msg) => {
                let seq = session.apply_operations(client_id, op_msg.operations)?;

                // Send ack back to sender
                if let Some(client) = session.clients.get(&client_id) {
                    let ack = Message::OperationAck(OperationAck {
                        doc_id: session.doc_id,
                        seq,
                        server_timestamp: current_timestamp(),
                    });
                    let _ = client.sender.send(ack).await;
                }

                // Check if we need to snapshot
                let op_count = session.op_counter.load(std::sync::atomic::Ordering::SeqCst);
                if op_count % self.config.snapshot_interval == 0 {
                    let doc = session.document.read();
                    let _ = self.storage.save_snapshot(&doc.snapshot()).await;
                }
            }

            Message::CursorUpdate(cursor) => {
                let presence = self.presence.get_or_create(session.doc_id);
                presence.update_cursor(&client_id, cursor.position, cursor.position);

                // Broadcast to others
                let _ = session.broadcast.send(Message::CursorUpdate(cursor));
            }

            Message::SelectionUpdate(selection) => {
                let presence = self.presence.get_or_create(session.doc_id);
                presence.update_selection(&client_id, selection.start, selection.end);

                // Broadcast to others
                let _ = session.broadcast.send(Message::SelectionUpdate(selection));
            }

            Message::SyncRequest(_) => {
                if let Some(client) = session.clients.get(&client_id) {
                    let sync_response = Message::SyncResponse(session.get_sync_response());
                    let _ = client.sender.send(sync_response).await;
                }
            }

            Message::Heartbeat(_) => {
                // Update last active time
                let presence = self.presence.get_or_create(session.doc_id);
                if let Some(mut state) = presence.users.get_mut(&client_id) {
                    state.last_active = current_timestamp();
                    state.status = crate::presence::UserStatus::Active;
                }
                drop(presence);
            }

            _ => {}
        }

        Ok(())
    }
}

/// Get current timestamp in milliseconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
