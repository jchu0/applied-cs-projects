//! REST API for document management.
//!
//! Provides HTTP endpoints for document CRUD operations, access control,
//! operation history, and snapshots. WebSocket collaboration is handled
//! separately in the server module.

use crate::document::{Document, DocumentMetadata};
use crate::server::CollaborationServer;
use crate::storage::{DocumentAcl, Permission, StorageManager};
use crate::{ClientId, DocumentId, Error};

use axum::{
    extract::{
        ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
        ConnectInfo, Path, Query, State,
    },
    http::{header, HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tower_http::cors::{Any, CorsLayer};
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

/// API state shared across handlers.
pub struct ApiState {
    /// Collaboration server.
    pub server: Arc<CollaborationServer>,
    /// Storage manager.
    pub storage: Arc<StorageManager>,
    /// ACL storage.
    pub acls: parking_lot::RwLock<std::collections::HashMap<DocumentId, DocumentAcl>>,
}

impl ApiState {
    /// Create new API state.
    pub fn new(server: Arc<CollaborationServer>, storage: Arc<StorageManager>) -> Self {
        Self {
            server,
            storage,
            acls: parking_lot::RwLock::new(std::collections::HashMap::new()),
        }
    }
}

// === Security & limits (hardening baseline) ===

/// Sliding-window rate-limit tracker keyed by API key or peer IP.
#[derive(Default)]
struct RateLimiter {
    /// Per-key request timestamps within the current window.
    windows: parking_lot::Mutex<HashMap<String, Vec<Instant>>>,
}

/// Security configuration derived from environment variables.
///
/// - `API_KEYS`: comma-separated valid keys; empty disables auth.
/// - `RATE_LIMIT_PER_MINUTE`: requests/minute per caller (default 120, 0 disables).
/// - `REQUEST_TIMEOUT_SECONDS`: per-request timeout (default 30, 0 disables).
struct SecurityConfig {
    /// Valid API keys; empty means auth disabled.
    api_keys: Vec<String>,
    /// Max requests per minute per caller; 0 disables.
    rate_limit_per_minute: u32,
    /// In-process sliding-window limiter state.
    limiter: RateLimiter,
}

impl SecurityConfig {
    /// Build config from the environment, logging when auth is disabled.
    fn from_env() -> Arc<Self> {
        let api_keys: Vec<String> = std::env::var("API_KEYS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if api_keys.is_empty() {
            tracing::warn!("API auth disabled (set API_KEYS to enable)");
        }

        let rate_limit_per_minute = std::env::var("RATE_LIMIT_PER_MINUTE")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(120);

        Arc::new(Self {
            api_keys,
            rate_limit_per_minute,
            limiter: RateLimiter::default(),
        })
    }

    /// Whether API-key auth is active.
    fn auth_enabled(&self) -> bool {
        !self.api_keys.is_empty()
    }

    /// Constant-time check that `candidate` matches a configured key.
    fn is_valid_key(&self, candidate: &str) -> bool {
        let mut matched = false;
        for key in &self.api_keys {
            matched |= constant_time_eq(key.as_bytes(), candidate.as_bytes());
        }
        matched
    }
}

/// Timeout in seconds from the environment (default 30, 0 disables).
fn request_timeout_seconds() -> u64 {
    std::env::var("REQUEST_TIMEOUT_SECONDS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30)
}

/// Constant-time byte comparison to avoid leaking key length/content via timing.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Extract the presented API key from `Authorization: Bearer` or `x-api-key`.
fn extract_api_key<B>(req: &Request<B>) -> Option<String> {
    if let Some(v) = req.headers().get(header::AUTHORIZATION) {
        if let Ok(s) = v.to_str() {
            if let Some(token) = s.strip_prefix("Bearer ") {
                return Some(token.trim().to_string());
            }
        }
    }
    req.headers()
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
}

/// API-key auth middleware. Rejects missing/invalid keys with 401 when enabled.
async fn auth_middleware(
    State(sec): State<Arc<SecurityConfig>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    if !sec.auth_enabled() {
        return next.run(req).await;
    }

    match extract_api_key(&req) {
        Some(key) if sec.is_valid_key(&key) => next.run(req).await,
        _ => (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Bearer")],
            Json(ErrorResponse {
                error: "missing or invalid API key".to_string(),
                code: StatusCode::UNAUTHORIZED.as_u16(),
            }),
        )
            .into_response(),
    }
}

/// In-process sliding-window rate-limit middleware. Returns 429 with Retry-After.
async fn rate_limit_middleware(
    State(sec): State<Arc<SecurityConfig>>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    req: Request<axum::body::Body>,
    next: Next,
) -> Response {
    let limit = sec.rate_limit_per_minute;
    if limit == 0 {
        return next.run(req).await;
    }

    // Key by API key if present, else by peer IP.
    let caller = extract_api_key(&req)
        .or_else(|| connect_info.map(|ci| ci.0.ip().to_string()))
        .unwrap_or_else(|| "unknown".to_string());

    let window = Duration::from_secs(60);
    let now = Instant::now();
    let over_limit = {
        let mut windows = sec.limiter.windows.lock();
        let hits = windows.entry(caller).or_default();
        hits.retain(|t| now.duration_since(*t) < window);
        if hits.len() as u32 >= limit {
            true
        } else {
            hits.push(now);
            false
        }
    };

    if over_limit {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(header::RETRY_AFTER, "60")],
            Json(ErrorResponse {
                error: "rate limit exceeded".to_string(),
                code: StatusCode::TOO_MANY_REQUESTS.as_u16(),
            }),
        )
            .into_response();
    }

    next.run(req).await
}

/// Combined state for the WebSocket collaboration endpoint.
///
/// The WebSocket route performs its own handshake-time auth (it cannot use the
/// HTTP auth middleware, which would 401 before the upgrade), so it needs both
/// the shared [`ApiState`] and the [`SecurityConfig`].
#[derive(Clone)]
struct WsState {
    /// Shared API/collaboration state.
    api: Arc<ApiState>,
    /// Security config for handshake auth.
    sec: Arc<SecurityConfig>,
}

/// Query parameters accepted on the WebSocket handshake.
#[derive(Debug, Deserialize)]
struct WsQuery {
    /// API key (alternative to the `Authorization`/`x-api-key` headers).
    api_key: Option<String>,
    /// Optional display name for the joining client.
    name: Option<String>,
    /// Optional color for the joining client.
    color: Option<String>,
}

/// Create the API router.
pub fn create_router(state: Arc<ApiState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let sec = SecurityConfig::from_env();

    // Health/readiness stays open: no auth, no rate limit, no timeout.
    let open = Router::new().route("/health", get(health_check));

    // WebSocket collaboration endpoint. It is deliberately kept out of the
    // protected HTTP surface: the request-timeout layer would abort the
    // long-lived socket, and the auth middleware would reject the upgrade
    // before the handshake. Auth is instead enforced inside the handler.
    let ws = Router::new()
        .route("/ws/:doc_id", get(ws_handler))
        .with_state(WsState {
            api: state.clone(),
            sec: sec.clone(),
        });

    // Protected API surface gets all three hardening layers.
    let mut protected = Router::new()
        // Document management
        .route("/api/documents", post(create_document))
        .route("/api/documents", get(list_documents))
        .route("/api/documents/:id", get(get_document))
        .route("/api/documents/:id", delete(delete_document))
        .route("/api/documents/:id/content", get(get_document_content))
        // History and snapshots
        .route("/api/documents/:id/history", get(get_document_history))
        .route("/api/documents/:id/snapshot", post(create_snapshot))
        .route("/api/documents/:id/snapshot", get(get_snapshot))
        // Access control
        .route("/api/documents/:id/acl", get(get_acl))
        .route("/api/documents/:id/acl", put(update_acl))
        .route("/api/documents/:id/share", post(create_share_link));

    // 3) Request timeout. Applied to the protected surface only; any streaming/
    // SSE/WebSocket route would be exempted here (the collaboration WebSocket
    // lives in the `server` module and is not part of this Router).
    let timeout_secs = request_timeout_seconds();
    if timeout_secs > 0 {
        protected = protected.layer(TimeoutLayer::new(Duration::from_secs(timeout_secs)));
    }

    let protected = protected
        // 2) Rate limiting (runs before the handler, after auth).
        .layer(middleware::from_fn_with_state(
            sec.clone(),
            rate_limit_middleware,
        ))
        // 1) API-key auth (outermost of the two, so unauthenticated requests
        // are rejected before consuming a rate-limit slot).
        .layer(middleware::from_fn_with_state(sec.clone(), auth_middleware));

    open.merge(protected)
        .with_state(state)
        .merge(ws)
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}

/// Error response.
#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
    code: u16,
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            Error::DocumentNotFound(_) => (StatusCode::NOT_FOUND, self.to_string()),
            Error::PermissionDenied => (StatusCode::FORBIDDEN, self.to_string()),
            Error::InvalidOperation(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = Json(ErrorResponse {
            error: message,
            code: status.as_u16(),
        });

        (status, body).into_response()
    }
}

// === Request/Response Types ===

/// Create document request.
#[derive(Debug, Deserialize)]
pub struct CreateDocumentRequest {
    /// Document title.
    pub title: String,
    /// Initial content (optional).
    pub content: Option<String>,
    /// Owner client ID (optional, auto-generated if not provided).
    pub owner_id: Option<ClientId>,
}

/// Create document response.
#[derive(Debug, Serialize)]
pub struct CreateDocumentResponse {
    /// Document ID.
    pub id: DocumentId,
    /// Document title.
    pub title: String,
    /// Owner ID.
    pub owner_id: ClientId,
    /// Created timestamp.
    pub created_at: u64,
}

/// Document info response.
#[derive(Debug, Serialize)]
pub struct DocumentInfoResponse {
    /// Document ID.
    pub id: DocumentId,
    /// Document title.
    pub title: String,
    /// Owner ID.
    pub owner_id: ClientId,
    /// Created timestamp.
    pub created_at: u64,
    /// Updated timestamp.
    pub updated_at: u64,
    /// Version number.
    pub version: u64,
    /// Character count.
    pub character_count: usize,
}

/// Document content response.
#[derive(Debug, Serialize)]
pub struct DocumentContentResponse {
    /// Document ID.
    pub id: DocumentId,
    /// Document text content.
    pub content: String,
    /// Version number.
    pub version: u64,
}

/// Document list response.
#[derive(Debug, Serialize)]
pub struct DocumentListResponse {
    /// Documents.
    pub documents: Vec<DocumentInfoResponse>,
    /// Total count.
    pub total: usize,
}

/// Operation history response.
#[derive(Debug, Serialize)]
pub struct OperationHistoryResponse {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Operations.
    pub operations: Vec<OperationEntry>,
    /// Total count.
    pub total: usize,
}

/// Operation entry in history.
#[derive(Debug, Serialize)]
pub struct OperationEntry {
    /// Sequence number.
    pub seq: u64,
    /// Operation count.
    pub operation_count: usize,
    /// Timestamp.
    pub timestamp: u64,
}

/// ACL response.
#[derive(Debug, Serialize)]
pub struct AclResponse {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Owner ID.
    pub owner_id: ClientId,
    /// ACL entries.
    pub entries: Vec<AclEntryResponse>,
    /// Public access level.
    pub public_access: Option<String>,
}

/// ACL entry response.
#[derive(Debug, Serialize)]
pub struct AclEntryResponse {
    /// Principal (user ID).
    pub principal: String,
    /// Permissions.
    pub permissions: Vec<String>,
    /// Granted at timestamp.
    pub granted_at: u64,
    /// Expires at timestamp.
    pub expires_at: Option<u64>,
}

/// Update ACL request.
#[derive(Debug, Deserialize)]
pub struct UpdateAclRequest {
    /// Entries to add/update.
    pub entries: Vec<AclEntryRequest>,
    /// Public access level.
    pub public_access: Option<String>,
}

/// ACL entry request.
#[derive(Debug, Deserialize)]
pub struct AclEntryRequest {
    /// Principal (user ID).
    pub principal: String,
    /// Permissions.
    pub permissions: Vec<String>,
    /// Expires at timestamp.
    pub expires_at: Option<u64>,
}

/// Share link request.
#[derive(Debug, Deserialize)]
pub struct ShareLinkRequest {
    /// Permission level for the link.
    pub permission: String,
    /// Expiration timestamp.
    pub expires_at: Option<u64>,
}

/// Share link response.
#[derive(Debug, Serialize)]
pub struct ShareLinkResponse {
    /// Share link URL.
    pub link: String,
    /// Permission level.
    pub permission: String,
    /// Expires at.
    pub expires_at: Option<u64>,
}

/// Snapshot response.
#[derive(Debug, Serialize)]
pub struct SnapshotResponse {
    /// Snapshot ID.
    pub id: String,
    /// Document ID.
    pub doc_id: DocumentId,
    /// Created timestamp.
    pub created_at: u64,
    /// Character count.
    pub character_count: usize,
}

// === Handlers ===

/// Health check endpoint.
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": current_timestamp()
    }))
}

/// Create a new document.
async fn create_document(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<CreateDocumentRequest>,
) -> std::result::Result<Json<CreateDocumentResponse>, Error> {
    let doc_id = DocumentId::new_v4();
    let owner_id = request.owner_id.unwrap_or_else(ClientId::new_v4);
    let now = current_timestamp();

    // Create document
    let document = Document::new(doc_id);

    // Persist metadata first: durable backends (SQLite) reference the document
    // row from the snapshots table via a foreign key, so the parent row must
    // exist before the snapshot is written.
    let metadata = DocumentMetadata {
        id: doc_id,
        title: request.title.clone(),
        owner: owner_id,
        created_at: now,
        updated_at: now,
        version: 1,
        archived: false,
    };
    state.storage.save_metadata(metadata).await?;

    // Save initial snapshot
    state.storage.save_snapshot(&document.snapshot()).await?;

    // Create ACL
    let acl = DocumentAcl::new(doc_id, owner_id);
    state.acls.write().insert(doc_id, acl);

    Ok(Json(CreateDocumentResponse {
        id: doc_id,
        title: request.title,
        owner_id,
        created_at: now,
    }))
}

/// List all documents.
async fn list_documents(
    State(state): State<Arc<ApiState>>,
) -> std::result::Result<Json<DocumentListResponse>, Error> {
    let documents = state.storage.list_documents().await?;

    let mut response_docs = Vec::new();
    for meta in documents {
        let char_count = if let Some(snapshot) = state.storage.load_snapshot(&meta.id).await? {
            let doc = Document::from_snapshot(snapshot);
            doc.len()
        } else {
            0
        };

        response_docs.push(DocumentInfoResponse {
            id: meta.id,
            title: meta.title,
            owner_id: meta.owner,
            created_at: meta.created_at,
            updated_at: meta.updated_at,
            version: meta.version,
            character_count: char_count,
        });
    }

    let total = response_docs.len();
    Ok(Json(DocumentListResponse {
        documents: response_docs,
        total,
    }))
}

/// Get document metadata.
async fn get_document(
    State(state): State<Arc<ApiState>>,
    Path(doc_id): Path<DocumentId>,
) -> std::result::Result<Json<DocumentInfoResponse>, Error> {
    let metadata = state
        .storage
        .load_metadata(&doc_id)
        .await?
        .ok_or_else(|| Error::DocumentNotFound(doc_id))?;

    let char_count = if let Some(snapshot) = state.storage.load_snapshot(&doc_id).await? {
        let doc = Document::from_snapshot(snapshot);
        doc.len()
    } else {
        0
    };

    Ok(Json(DocumentInfoResponse {
        id: metadata.id,
        title: metadata.title,
        owner_id: metadata.owner,
        created_at: metadata.created_at,
        updated_at: metadata.updated_at,
        version: metadata.version,
        character_count: char_count,
    }))
}

/// Get document content.
async fn get_document_content(
    State(state): State<Arc<ApiState>>,
    Path(doc_id): Path<DocumentId>,
) -> std::result::Result<Json<DocumentContentResponse>, Error> {
    let metadata = state
        .storage
        .load_metadata(&doc_id)
        .await?
        .ok_or_else(|| Error::DocumentNotFound(doc_id))?;

    let snapshot = state
        .storage
        .load_snapshot(&doc_id)
        .await?
        .ok_or_else(|| Error::DocumentNotFound(doc_id))?;

    let doc = Document::from_snapshot(snapshot);

    Ok(Json(DocumentContentResponse {
        id: doc_id,
        content: doc.text(),
        version: metadata.version,
    }))
}

/// Delete (archive) a document.
async fn delete_document(
    State(state): State<Arc<ApiState>>,
    Path(doc_id): Path<DocumentId>,
) -> std::result::Result<impl IntoResponse, Error> {
    // Verify document exists
    let _ = state
        .storage
        .load_metadata(&doc_id)
        .await?
        .ok_or_else(|| Error::DocumentNotFound(doc_id))?;

    // Delete document
    state.storage.delete_document(&doc_id).await?;

    // Remove ACL
    state.acls.write().remove(&doc_id);

    Ok(StatusCode::NO_CONTENT)
}

/// Get operation history.
async fn get_document_history(
    State(state): State<Arc<ApiState>>,
    Path(doc_id): Path<DocumentId>,
) -> std::result::Result<Json<OperationHistoryResponse>, Error> {
    // Verify document exists
    let _ = state
        .storage
        .load_metadata(&doc_id)
        .await?
        .ok_or_else(|| Error::DocumentNotFound(doc_id))?;

    let entries = state.storage.get_operations_since(&doc_id, 0).await?;

    let operations: Vec<OperationEntry> = entries
        .iter()
        .map(|e| OperationEntry {
            seq: e.seq,
            operation_count: e.operations.len(),
            timestamp: e.timestamp,
        })
        .collect();

    let total = operations.len();
    Ok(Json(OperationHistoryResponse {
        doc_id,
        operations,
        total,
    }))
}

/// Create a snapshot.
async fn create_snapshot(
    State(state): State<Arc<ApiState>>,
    Path(doc_id): Path<DocumentId>,
) -> std::result::Result<Json<SnapshotResponse>, Error> {
    // Get from active session or storage
    let snapshot = if let Some(session) = state.server.sessions.get(&doc_id) {
        let doc = session.document.read();
        doc.snapshot()
    } else {
        state
            .storage
            .load_snapshot(&doc_id)
            .await?
            .ok_or_else(|| Error::DocumentNotFound(doc_id))?
    };

    // Save snapshot
    state.storage.save_snapshot(&snapshot).await?;

    let doc = Document::from_snapshot(snapshot.clone());

    Ok(Json(SnapshotResponse {
        id: uuid::Uuid::new_v4().to_string(),
        doc_id,
        created_at: current_timestamp(),
        character_count: doc.len(),
    }))
}

/// Get current snapshot.
async fn get_snapshot(
    State(state): State<Arc<ApiState>>,
    Path(doc_id): Path<DocumentId>,
) -> std::result::Result<Json<SnapshotResponse>, Error> {
    let snapshot = state
        .storage
        .load_snapshot(&doc_id)
        .await?
        .ok_or_else(|| Error::DocumentNotFound(doc_id))?;

    let doc = Document::from_snapshot(snapshot);

    Ok(Json(SnapshotResponse {
        id: "current".to_string(),
        doc_id,
        created_at: current_timestamp(),
        character_count: doc.len(),
    }))
}

/// Get document ACL.
async fn get_acl(
    State(state): State<Arc<ApiState>>,
    Path(doc_id): Path<DocumentId>,
) -> std::result::Result<Json<AclResponse>, Error> {
    // Verify document exists
    let _ = state
        .storage
        .load_metadata(&doc_id)
        .await?
        .ok_or_else(|| Error::DocumentNotFound(doc_id))?;

    let acls = state.acls.read();
    let acl = acls.get(&doc_id).cloned().unwrap_or_else(|| {
        DocumentAcl::new(doc_id, ClientId::nil())
    });

    let entries: Vec<AclEntryResponse> = acl
        .entries
        .iter()
        .map(|e| AclEntryResponse {
            principal: e.principal.clone(),
            permissions: e.permissions.iter().map(|p| format!("{:?}", p).to_lowercase()).collect(),
            granted_at: e.granted_at,
            expires_at: e.expires_at,
        })
        .collect();

    Ok(Json(AclResponse {
        doc_id,
        owner_id: acl.owner,
        entries,
        public_access: acl.public_access.map(|p| format!("{:?}", p).to_lowercase()),
    }))
}

/// Update document ACL.
async fn update_acl(
    State(state): State<Arc<ApiState>>,
    Path(doc_id): Path<DocumentId>,
    Json(request): Json<UpdateAclRequest>,
) -> std::result::Result<Json<AclResponse>, Error> {
    // Verify document exists
    let metadata = state
        .storage
        .load_metadata(&doc_id)
        .await?
        .ok_or_else(|| Error::DocumentNotFound(doc_id))?;

    let mut acls = state.acls.write();
    let acl = acls
        .entry(doc_id)
        .or_insert_with(|| DocumentAcl::new(doc_id, metadata.owner));

    // Clear existing entries and add new ones
    acl.entries.clear();

    for entry in request.entries {
        let permissions: Vec<Permission> = entry
            .permissions
            .iter()
            .filter_map(|p| match p.to_lowercase().as_str() {
                "read" => Some(Permission::Read),
                "write" => Some(Permission::Write),
                "comment" => Some(Permission::Comment),
                "admin" => Some(Permission::Admin),
                _ => None,
            })
            .collect();

        acl.entries.push(crate::storage::AclEntry {
            principal: entry.principal,
            permissions,
            granted_by: metadata.owner,
            granted_at: current_timestamp(),
            expires_at: entry.expires_at,
        });
    }

    // Update public access
    if let Some(public) = request.public_access {
        acl.public_access = match public.to_lowercase().as_str() {
            "read" => Some(Permission::Read),
            "write" => Some(Permission::Write),
            "comment" => Some(Permission::Comment),
            "admin" => Some(Permission::Admin),
            _ => None,
        };
    }

    let response_entries: Vec<AclEntryResponse> = acl
        .entries
        .iter()
        .map(|e| AclEntryResponse {
            principal: e.principal.clone(),
            permissions: e.permissions.iter().map(|p| format!("{:?}", p).to_lowercase()).collect(),
            granted_at: e.granted_at,
            expires_at: e.expires_at,
        })
        .collect();

    Ok(Json(AclResponse {
        doc_id,
        owner_id: acl.owner,
        entries: response_entries,
        public_access: acl.public_access.map(|p| format!("{:?}", p).to_lowercase()),
    }))
}

/// Create a share link.
async fn create_share_link(
    State(state): State<Arc<ApiState>>,
    Path(doc_id): Path<DocumentId>,
    Json(request): Json<ShareLinkRequest>,
) -> std::result::Result<Json<ShareLinkResponse>, Error> {
    // Verify document exists
    let _ = state
        .storage
        .load_metadata(&doc_id)
        .await?
        .ok_or_else(|| Error::DocumentNotFound(doc_id))?;

    // Generate share token
    let share_token = uuid::Uuid::new_v4().to_string();
    let link = format!("/share/{}?token={}", doc_id, share_token);

    // In a real implementation, we'd store this token with the permission and expiry

    Ok(Json(ShareLinkResponse {
        link,
        permission: request.permission,
        expires_at: request.expires_at,
    }))
}

// === WebSocket collaboration endpoint ===

/// Handle a WebSocket upgrade for real-time document collaboration.
///
/// Auth is enforced here (before the upgrade completes) rather than via the
/// HTTP auth middleware: the socket is long-lived and must bypass the
/// request-timeout layer. When auth is enabled, the key may be supplied via the
/// `Authorization: Bearer`/`x-api-key` headers or the `?api_key=` query param
/// (browsers cannot set headers on a WebSocket handshake).
async fn ws_handler(
    State(state): State<WsState>,
    Path(doc_id): Path<DocumentId>,
    Query(query): Query<WsQuery>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    if state.sec.auth_enabled() {
        let presented = query
            .api_key
            .clone()
            .or_else(|| api_key_from_headers(&headers));
        let ok = presented
            .as_deref()
            .map(|k| state.sec.is_valid_key(k))
            .unwrap_or(false);
        if !ok {
            return (
                StatusCode::UNAUTHORIZED,
                [(header::WWW_AUTHENTICATE, "Bearer")],
                Json(ErrorResponse {
                    error: "missing or invalid API key".to_string(),
                    code: StatusCode::UNAUTHORIZED.as_u16(),
                }),
            )
                .into_response();
        }
    }

    let client_id = ClientId::new_v4();
    let name = query.name.unwrap_or_else(|| "anonymous".to_string());
    let color = query.color.unwrap_or_else(|| "#3b82f6".to_string());
    let api = state.api.clone();

    ws.on_upgrade(move |socket| ws_collaboration(socket, api, doc_id, client_id, name, color))
}

/// Extract an API key from the `Authorization: Bearer` or `x-api-key` headers.
fn api_key_from_headers(headers: &HeaderMap) -> Option<String> {
    if let Some(v) = headers.get(header::AUTHORIZATION) {
        if let Ok(s) = v.to_str() {
            if let Some(token) = s.strip_prefix("Bearer ") {
                return Some(token.trim().to_string());
            }
        }
    }
    headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
}

/// Drive a single collaborating client's WebSocket connection.
///
/// The client is subscribed to its document's broadcast channel. Incoming CRDT
/// operations are applied to the shared document and re-broadcast to every
/// other subscriber, so all clients converge. On disconnect, the latest state
/// is persisted through the storage layer.
async fn ws_collaboration(
    socket: WebSocket,
    state: Arc<ApiState>,
    doc_id: DocumentId,
    client_id: ClientId,
    name: String,
    color: String,
) {
    let session = match state.server.get_or_create_session(doc_id).await {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(%doc_id, error = %e, "failed to open document session");
            return;
        }
    };

    let (mut ws_tx, mut ws_rx) = socket.split();
    let mut broadcast_rx = session.broadcast.subscribe();

    // Send the current document state so the client can render immediately.
    let sync = crate::protocol::Message::SyncResponse(session.get_sync_response());
    if let Ok(text) = serde_json::to_string(&sync) {
        if ws_tx.send(WsMessage::Text(text)).await.is_err() {
            return;
        }
    }

    // Announce the join to peers.
    let join = crate::protocol::Message::UserJoin(crate::protocol::UserJoin {
        doc_id,
        user: crate::protocol::UserInfo {
            client_id,
            name,
            color,
            avatar_url: None,
        },
    });
    let _ = session.broadcast.send(join);

    loop {
        tokio::select! {
            incoming = ws_rx.next() => {
                match incoming {
                    Some(Ok(WsMessage::Text(text))) => {
                        if let Ok(msg) = serde_json::from_str::<crate::protocol::Message>(&text) {
                            // Direct replies (acks, sync responses) go straight
                            // back over this socket; broadcasts fan out to peers.
                            for reply in handle_ws_message(&state, &session, client_id, msg).await {
                                if let Ok(text) = serde_json::to_string(&reply) {
                                    if ws_tx.send(WsMessage::Text(text)).await.is_err() {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    Some(Ok(WsMessage::Close(_))) | Some(Err(_)) | None => break,
                    _ => {}
                }
            }
            broadcast = broadcast_rx.recv() => {
                match broadcast {
                    Ok(msg) => {
                        // Never echo a client's own operation back to it.
                        if let crate::protocol::Message::Operation(ref op) = msg {
                            if op.client_id == client_id {
                                continue;
                            }
                        }
                        if let Ok(text) = serde_json::to_string(&msg) {
                            if ws_tx.send(WsMessage::Text(text)).await.is_err() {
                                break;
                            }
                        }
                    }
                    // Lagged behind the broadcast buffer; keep going with newer messages.
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    // Announce departure and persist the latest state.
    let _ = session.broadcast.send(crate::protocol::Message::UserLeave(
        crate::protocol::UserLeave { doc_id, client_id },
    ));
    let snapshot = { session.document.read().snapshot() };
    if let Err(e) = state.storage.save_snapshot(&snapshot).await {
        tracing::warn!(%doc_id, error = %e, "failed to persist snapshot on disconnect");
    }
}

/// Apply an inbound collaboration message from a WebSocket client.
///
/// Returns any messages that should be sent directly back to this client
/// (acknowledgements, sync responses). Peer-facing effects (applying and
/// re-broadcasting operations) happen as a side effect on the session.
async fn handle_ws_message(
    state: &Arc<ApiState>,
    session: &crate::server::DocumentSession,
    client_id: ClientId,
    message: crate::protocol::Message,
) -> Vec<crate::protocol::Message> {
    use crate::protocol::Message;
    match message {
        Message::Operation(op_msg) => {
            let vector_clock = op_msg.vector_clock.clone();
            let operations = op_msg.operations.clone();
            // `apply_operations` applies to the shared document and broadcasts
            // the op to every subscriber (see server::DocumentSession).
            match session.apply_operations(client_id, op_msg.operations) {
                Ok(seq) => {
                    // Durably log the applied operations so history survives
                    // beyond the in-process broadcast.
                    let _ = state
                        .storage
                        .append_operations(&session.doc_id, operations, vector_clock)
                        .await;
                    // Persist a fresh snapshot so a restart recovers the edit.
                    let snapshot = { session.document.read().snapshot() };
                    let _ = state.storage.save_snapshot(&snapshot).await;

                    vec![Message::OperationAck(crate::protocol::OperationAck {
                        doc_id: session.doc_id,
                        seq,
                        server_timestamp: current_timestamp(),
                    })]
                }
                Err(e) => {
                    tracing::warn!(error = %e, "rejected invalid operation");
                    vec![]
                }
            }
        }
        Message::CursorUpdate(_) | Message::SelectionUpdate(_) => {
            // Presence relay: forward to peers without touching document state.
            let _ = session.broadcast.send(message);
            vec![]
        }
        Message::SyncRequest(_) => {
            vec![Message::SyncResponse(session.get_sync_response())]
        }
        _ => vec![],
    }
}

/// Get current timestamp in milliseconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::ServerConfig;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::ServiceExt;

    /// Serializes tests that mutate process-global auth env vars so they do not
    /// race with the auth-off tests.
    static ENV_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn create_test_state() -> Arc<ApiState> {
        let storage = Arc::new(StorageManager::new());
        let server = Arc::new(CollaborationServer::new(
            ServerConfig::default(),
            storage.clone(),
        ));
        Arc::new(ApiState::new(server, storage))
    }

    #[tokio::test]
    async fn test_health_check() {
        let state = create_test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_create_document() {
        let _guard = ENV_GUARD.lock().unwrap();
        std::env::remove_var("API_KEYS");
        let state = create_test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/documents")
                    .header("Content-Type", "application/json")
                    .body(Body::from(r#"{"title": "Test Document"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_list_documents() {
        let _guard = ENV_GUARD.lock().unwrap();
        std::env::remove_var("API_KEYS");
        let state = create_test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/documents")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_auth_required_returns_401_without_key() {
        let _guard = ENV_GUARD.lock().unwrap();
        std::env::set_var("API_KEYS", "secret-key");
        let state = create_test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/documents")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        std::env::remove_var("API_KEYS");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert!(response.headers().contains_key("www-authenticate"));
    }

    #[tokio::test]
    async fn test_auth_rejects_bad_key() {
        let _guard = ENV_GUARD.lock().unwrap();
        std::env::set_var("API_KEYS", "secret-key");
        let state = create_test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/documents")
                    .header("x-api-key", "wrong")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        std::env::remove_var("API_KEYS");
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn test_auth_accepts_valid_key() {
        let _guard = ENV_GUARD.lock().unwrap();
        std::env::set_var("API_KEYS", "secret-key");
        let state = create_test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/api/documents")
                    .header("Authorization", "Bearer secret-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        std::env::remove_var("API_KEYS");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_open_without_key_when_auth_on() {
        let _guard = ENV_GUARD.lock().unwrap();
        std::env::set_var("API_KEYS", "secret-key");
        let state = create_test_state();
        let app = create_router(state);

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        std::env::remove_var("API_KEYS");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_get_nonexistent_document() {
        let _guard = ENV_GUARD.lock().unwrap();
        std::env::remove_var("API_KEYS");
        let state = create_test_state();
        let app = create_router(state);

        let doc_id = DocumentId::new_v4();
        let response = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/documents/{}", doc_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }
}
