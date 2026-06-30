//! WebSocket protocol messages.

use crate::crdt::{Operation, VectorClock};
use crate::presence::PresenceState;
use crate::{ClientId, DocumentId};
use serde::{Deserialize, Serialize};

/// WebSocket message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum Message {
    // Document operations
    #[serde(rename = "operation")]
    Operation(OperationMessage),

    #[serde(rename = "operation_ack")]
    OperationAck(OperationAck),

    #[serde(rename = "sync_request")]
    SyncRequest(SyncRequest),

    #[serde(rename = "sync_response")]
    SyncResponse(SyncResponse),

    // Presence
    #[serde(rename = "cursor_update")]
    CursorUpdate(CursorUpdate),

    #[serde(rename = "selection_update")]
    SelectionUpdate(SelectionUpdate),

    #[serde(rename = "user_join")]
    UserJoin(UserJoin),

    #[serde(rename = "user_leave")]
    UserLeave(UserLeave),

    #[serde(rename = "presence_sync")]
    PresenceSync(PresenceSync),

    // Control
    #[serde(rename = "heartbeat")]
    Heartbeat(Heartbeat),

    #[serde(rename = "error")]
    Error(ErrorMessage),

    // Authentication
    #[serde(rename = "auth")]
    Auth(AuthMessage),

    #[serde(rename = "auth_response")]
    AuthResponse(AuthResponse),
}

/// Operation message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationMessage {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Client ID.
    pub client_id: ClientId,
    /// Vector clock at time of operation.
    pub vector_clock: VectorClock,
    /// Operations to apply.
    pub operations: Vec<Operation>,
    /// Message sequence number.
    pub seq: u64,
}

/// Operation acknowledgment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationAck {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Acknowledged sequence number.
    pub seq: u64,
    /// Server timestamp.
    pub server_timestamp: u64,
}

/// Sync request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncRequest {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Client's current vector clock.
    pub vector_clock: VectorClock,
}

/// Sync response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncResponse {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Compacted state.
    pub state: Vec<u8>,
    /// Pending operations since snapshot.
    pub pending_ops: Vec<Operation>,
    /// Current vector clock.
    pub vector_clock: VectorClock,
}

/// Cursor update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CursorUpdate {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Client ID.
    pub client_id: ClientId,
    /// Cursor position.
    pub position: usize,
}

/// Selection update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionUpdate {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Client ID.
    pub client_id: ClientId,
    /// Selection start.
    pub start: usize,
    /// Selection end.
    pub end: usize,
}

/// User join notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserJoin {
    /// Document ID.
    pub doc_id: DocumentId,
    /// User info.
    pub user: UserInfo,
}

/// User leave notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserLeave {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Client ID.
    pub client_id: ClientId,
}

/// Presence sync message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresenceSync {
    /// Document ID.
    pub doc_id: DocumentId,
    /// All users' presence states.
    pub users: Vec<PresenceState>,
}

/// Heartbeat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Heartbeat {
    /// Client timestamp.
    pub timestamp: u64,
}

/// Error message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMessage {
    /// Error code.
    pub code: u32,
    /// Error message.
    pub message: String,
}

/// Authentication message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthMessage {
    /// Authentication token.
    pub token: String,
    /// Client ID.
    pub client_id: ClientId,
}

/// Authentication response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthResponse {
    /// Success flag.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
    /// Assigned client ID.
    pub client_id: Option<ClientId>,
}

/// User info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserInfo {
    /// Client ID.
    pub client_id: ClientId,
    /// User name.
    pub name: String,
    /// User color.
    pub color: String,
    /// Avatar URL.
    pub avatar_url: Option<String>,
}

/// Join document request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinDocument {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Client's current vector clock (empty for new clients).
    pub vector_clock: Option<VectorClock>,
}

/// Connection options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionOptions {
    /// Reconnection strategy.
    pub reconnect: bool,
    /// Max reconnection attempts.
    pub max_reconnect_attempts: u32,
    /// Reconnection delay in milliseconds.
    pub reconnect_delay_ms: u64,
}

impl Default for ConnectionOptions {
    fn default() -> Self {
        Self {
            reconnect: true,
            max_reconnect_attempts: 10,
            reconnect_delay_ms: 1000,
        }
    }
}
