//! Storage layer for documents and operations.
//!
//! [`StorageManager`] is the interface consumed by the HTTP API and the
//! collaboration server. By default it keeps everything in process memory,
//! which is convenient for tests and ephemeral deployments. When constructed
//! with [`StorageManager::with_backend`] it instead delegates every call to a
//! durable [`StorageBackend`](crate::persistent::StorageBackend) (e.g. the
//! bundled SQLite backend), so document state survives process restarts.

use crate::crdt::{Operation, VectorClock};
use crate::document::{DocumentMetadata, DocumentSnapshot};
use crate::persistent::StorageBackend;
use crate::{ClientId, DocumentId, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// In-memory storage state, used when no durable backend is configured.
#[derive(Default)]
struct InMemoryStore {
    /// Document snapshots.
    snapshots: RwLock<HashMap<DocumentId, DocumentSnapshot>>,
    /// Operation logs.
    op_logs: RwLock<HashMap<DocumentId, OperationLog>>,
    /// Document metadata.
    metadata: RwLock<HashMap<DocumentId, DocumentMetadata>>,
}

/// Storage manager.
///
/// Either backed by process memory (default) or a durable
/// [`StorageBackend`](crate::persistent::StorageBackend).
pub struct StorageManager {
    /// In-memory state (used only when `backend` is `None`).
    mem: InMemoryStore,
    /// Optional durable backend. When set, all operations delegate to it.
    backend: Option<Arc<dyn StorageBackend>>,
}

impl StorageManager {
    /// Create a new in-memory storage manager.
    pub fn new() -> Self {
        Self {
            mem: InMemoryStore::default(),
            backend: None,
        }
    }

    /// Create a storage manager backed by a durable [`StorageBackend`].
    ///
    /// All reads and writes are delegated to `backend`, so document state
    /// persists across process restarts (e.g. with the SQLite backend).
    pub fn with_backend(backend: Arc<dyn StorageBackend>) -> Self {
        Self {
            mem: InMemoryStore::default(),
            backend: Some(backend),
        }
    }

    /// Whether this manager is backed by durable storage.
    pub fn is_persistent(&self) -> bool {
        self.backend.is_some()
    }

    /// Save a document snapshot.
    pub async fn save_snapshot(&self, snapshot: &DocumentSnapshot) -> Result<()> {
        if let Some(backend) = &self.backend {
            return backend.save_snapshot(snapshot).await;
        }
        self.mem.snapshots.write().insert(snapshot.id, snapshot.clone());
        Ok(())
    }

    /// Load a document snapshot.
    pub async fn load_snapshot(&self, doc_id: &DocumentId) -> Result<Option<DocumentSnapshot>> {
        if let Some(backend) = &self.backend {
            return backend.load_snapshot(doc_id).await;
        }
        Ok(self.mem.snapshots.read().get(doc_id).cloned())
    }

    /// Append operations to log.
    pub async fn append_operations(
        &self,
        doc_id: &DocumentId,
        operations: Vec<Operation>,
        vector_clock: VectorClock,
    ) -> Result<u64> {
        if let Some(backend) = &self.backend {
            return backend.append_operations(doc_id, operations, vector_clock).await;
        }
        let mut logs = self.mem.op_logs.write();
        let log = logs.entry(*doc_id).or_insert_with(|| OperationLog::new(*doc_id));

        let seq = log.append(operations, vector_clock);
        Ok(seq)
    }

    /// Get operations since a sequence number.
    pub async fn get_operations_since(
        &self,
        doc_id: &DocumentId,
        since_seq: u64,
    ) -> Result<Vec<OperationEntry>> {
        if let Some(backend) = &self.backend {
            return backend.get_operations_since(doc_id, since_seq).await;
        }
        let logs = self.mem.op_logs.read();
        if let Some(log) = logs.get(doc_id) {
            Ok(log.get_since(since_seq))
        } else {
            Ok(vec![])
        }
    }

    /// Get operations since a vector clock.
    ///
    /// Only supported by the in-memory store; durable backends filter by
    /// sequence number instead. Returns an empty vector when backed durably.
    pub async fn get_operations_after_clock(
        &self,
        doc_id: &DocumentId,
        vector_clock: &VectorClock,
    ) -> Result<Vec<Operation>> {
        if self.backend.is_some() {
            return Ok(vec![]);
        }
        let logs = self.mem.op_logs.read();
        if let Some(log) = logs.get(doc_id) {
            Ok(log.get_after_clock(vector_clock))
        } else {
            Ok(vec![])
        }
    }

    /// Save document metadata.
    pub async fn save_metadata(&self, metadata: DocumentMetadata) -> Result<()> {
        if let Some(backend) = &self.backend {
            return backend.save_metadata(metadata).await;
        }
        self.mem.metadata.write().insert(metadata.id, metadata);
        Ok(())
    }

    /// Load document metadata.
    pub async fn load_metadata(&self, doc_id: &DocumentId) -> Result<Option<DocumentMetadata>> {
        if let Some(backend) = &self.backend {
            return backend.load_metadata(doc_id).await;
        }
        Ok(self.mem.metadata.read().get(doc_id).cloned())
    }

    /// List all documents.
    pub async fn list_documents(&self) -> Result<Vec<DocumentMetadata>> {
        if let Some(backend) = &self.backend {
            return backend.list_documents().await;
        }
        Ok(self.mem.metadata.read().values().cloned().collect())
    }

    /// Delete a document.
    pub async fn delete_document(&self, doc_id: &DocumentId) -> Result<()> {
        if let Some(backend) = &self.backend {
            return backend.delete_document(doc_id).await;
        }
        self.mem.snapshots.write().remove(doc_id);
        self.mem.op_logs.write().remove(doc_id);
        self.mem.metadata.write().remove(doc_id);
        Ok(())
    }

    /// Compact operation log by pruning old entries.
    ///
    /// Keeps only the most recent `keep_entries` log entries.
    /// Returns the number of entries removed.
    pub async fn compact(&self, doc_id: &DocumentId) -> Result<()> {
        let keep_entries = 100;
        if let Some(backend) = &self.backend {
            // For durable backends, keep the most recent `keep_entries` by seq.
            let entries = backend.get_operations_since(doc_id, 0).await?;
            if entries.len() > keep_entries {
                if let Some(cutoff) = entries.get(entries.len() - keep_entries) {
                    backend.compact(doc_id, cutoff.seq).await?;
                }
            }
            return Ok(());
        }
        let mut logs = self.mem.op_logs.write();
        if let Some(log) = logs.get_mut(doc_id) {
            log.truncate_before(keep_entries);
        }
        Ok(())
    }
}

impl Default for StorageManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Operation log for a document.
pub struct OperationLog {
    /// Document ID.
    doc_id: DocumentId,
    /// Entries in order.
    entries: Vec<OperationEntry>,
    /// Next sequence number.
    next_seq: u64,
}

impl OperationLog {
    /// Create new operation log.
    pub fn new(doc_id: DocumentId) -> Self {
        Self {
            doc_id,
            entries: Vec::new(),
            next_seq: 1,
        }
    }

    /// Append operations.
    pub fn append(&mut self, operations: Vec<Operation>, vector_clock: VectorClock) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;

        let entry = OperationEntry {
            seq,
            operations,
            vector_clock,
            timestamp: current_timestamp(),
        };

        self.entries.push(entry);
        seq
    }

    /// Get entries since a sequence number.
    pub fn get_since(&self, since_seq: u64) -> Vec<OperationEntry> {
        self.entries
            .iter()
            .filter(|e| e.seq > since_seq)
            .cloned()
            .collect()
    }

    /// Get operations after a vector clock.
    pub fn get_after_clock(&self, vector_clock: &VectorClock) -> Vec<Operation> {
        self.entries
            .iter()
            .filter(|e| !vector_clock.dominates(&e.vector_clock))
            .flat_map(|e| e.operations.clone())
            .collect()
    }

    /// Get total entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Truncate old entries, keeping only the most recent `keep` entries.
    pub fn truncate_before(&mut self, keep: usize) {
        if self.entries.len() > keep {
            let drain_count = self.entries.len() - keep;
            self.entries.drain(..drain_count);
        }
    }
}

/// Operation log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationEntry {
    /// Sequence number.
    pub seq: u64,
    /// Operations.
    pub operations: Vec<Operation>,
    /// Vector clock at time of operation.
    pub vector_clock: VectorClock,
    /// Server timestamp.
    pub timestamp: u64,
}

/// Checkpoint for fast recovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Checkpoint ID.
    pub id: uuid::Uuid,
    /// Document ID.
    pub doc_id: DocumentId,
    /// Timestamp.
    pub timestamp: u64,
    /// Vector clock at checkpoint.
    pub vector_clock: VectorClock,
    /// Sequence number.
    pub seq: u64,
    /// Snapshot data.
    pub snapshot: DocumentSnapshot,
}

/// Audit event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Event ID.
    pub event_id: uuid::Uuid,
    /// Timestamp.
    pub timestamp: u64,
    /// User ID.
    pub user_id: ClientId,
    /// Document ID.
    pub doc_id: DocumentId,
    /// Action.
    pub action: AuditAction,
    /// Additional details.
    pub details: serde_json::Value,
}

/// Audit action types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditAction {
    /// Document created.
    DocumentCreated,
    /// Document opened.
    DocumentOpened,
    /// Document edited.
    DocumentEdited,
    /// Document shared.
    DocumentShared,
    /// Permission changed.
    PermissionChanged,
    /// Document deleted.
    DocumentDeleted,
    /// Document restored.
    DocumentRestored,
}

/// Audit log.
pub struct AuditLog {
    /// Events.
    events: RwLock<Vec<AuditEvent>>,
}

impl AuditLog {
    /// Create new audit log.
    pub fn new() -> Self {
        Self {
            events: RwLock::new(Vec::new()),
        }
    }

    /// Log an event.
    pub fn log(&self, event: AuditEvent) {
        self.events.write().push(event);
    }

    /// Get events for a document.
    pub fn get_for_document(&self, doc_id: &DocumentId) -> Vec<AuditEvent> {
        self.events
            .read()
            .iter()
            .filter(|e| e.doc_id == *doc_id)
            .cloned()
            .collect()
    }

    /// Get events for a user.
    pub fn get_for_user(&self, user_id: &ClientId) -> Vec<AuditEvent> {
        self.events
            .read()
            .iter()
            .filter(|e| e.user_id == *user_id)
            .cloned()
            .collect()
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Access control entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AclEntry {
    /// Principal (user or group).
    pub principal: String,
    /// Permissions.
    pub permissions: Vec<Permission>,
    /// Granted by.
    pub granted_by: ClientId,
    /// Granted at.
    pub granted_at: u64,
    /// Expires at.
    pub expires_at: Option<u64>,
}

/// Permission types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Permission {
    /// Read access.
    Read,
    /// Write access.
    Write,
    /// Comment access.
    Comment,
    /// Admin access.
    Admin,
}

/// Document access control list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentAcl {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Owner.
    pub owner: ClientId,
    /// ACL entries.
    pub entries: Vec<AclEntry>,
    /// Public access level.
    pub public_access: Option<Permission>,
}

impl DocumentAcl {
    /// Create new ACL.
    pub fn new(doc_id: DocumentId, owner: ClientId) -> Self {
        Self {
            doc_id,
            owner,
            entries: Vec::new(),
            public_access: None,
        }
    }

    /// Check if user has permission.
    pub fn has_permission(&self, user_id: &ClientId, permission: Permission) -> bool {
        // Owner has all permissions
        if user_id == &self.owner {
            return true;
        }

        // Check public access
        if let Some(public) = self.public_access {
            if permission_level(public) >= permission_level(permission) {
                return true;
            }
        }

        // Check ACL entries
        let now = current_timestamp();
        for entry in &self.entries {
            if entry.principal == user_id.to_string() {
                // Check expiration
                if let Some(expires) = entry.expires_at {
                    if now > expires {
                        continue;
                    }
                }

                // Check permission
                if entry.permissions.iter().any(|p| permission_level(*p) >= permission_level(permission)) {
                    return true;
                }
            }
        }

        false
    }

    /// Grant permission.
    pub fn grant(&mut self, user_id: &str, permission: Permission, granted_by: ClientId) {
        self.entries.push(AclEntry {
            principal: user_id.to_string(),
            permissions: vec![permission],
            granted_by,
            granted_at: current_timestamp(),
            expires_at: None,
        });
    }

    /// Revoke permission.
    pub fn revoke(&mut self, user_id: &str) {
        self.entries.retain(|e| e.principal != user_id);
    }
}

/// Get permission level for comparison.
fn permission_level(permission: Permission) -> u8 {
    match permission {
        Permission::Read => 1,
        Permission::Comment => 2,
        Permission::Write => 3,
        Permission::Admin => 4,
    }
}

/// Get current timestamp in milliseconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
