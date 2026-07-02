//! Persistent storage backends for CRDT collaboration.
//!
//! This module provides durable storage options using SQLite for
//! single-server deployments and PostgreSQL for distributed setups.

use crate::crdt::{Operation, VectorClock};
use crate::document::{DocumentMetadata, DocumentSnapshot};
use crate::storage::{AclEntry, AuditAction, AuditEvent, DocumentAcl, OperationEntry, Permission};
use crate::{DocumentId, Error, Result};
use async_trait::async_trait;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Storage backend trait for pluggable persistence.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Save a document snapshot.
    async fn save_snapshot(&self, snapshot: &DocumentSnapshot) -> Result<()>;

    /// Load a document snapshot.
    async fn load_snapshot(&self, doc_id: &DocumentId) -> Result<Option<DocumentSnapshot>>;

    /// Append operations to the log.
    async fn append_operations(
        &self,
        doc_id: &DocumentId,
        operations: Vec<Operation>,
        vector_clock: VectorClock,
    ) -> Result<u64>;

    /// Get operations since a sequence number.
    async fn get_operations_since(
        &self,
        doc_id: &DocumentId,
        since_seq: u64,
    ) -> Result<Vec<OperationEntry>>;

    /// Save document metadata.
    async fn save_metadata(&self, metadata: DocumentMetadata) -> Result<()>;

    /// Load document metadata.
    async fn load_metadata(&self, doc_id: &DocumentId) -> Result<Option<DocumentMetadata>>;

    /// List all documents.
    async fn list_documents(&self) -> Result<Vec<DocumentMetadata>>;

    /// Delete a document.
    async fn delete_document(&self, doc_id: &DocumentId) -> Result<()>;

    /// Compact operation log by removing entries before a checkpoint.
    async fn compact(&self, doc_id: &DocumentId, before_seq: u64) -> Result<u64>;

    /// Save ACL.
    async fn save_acl(&self, acl: &DocumentAcl) -> Result<()>;

    /// Load ACL.
    async fn load_acl(&self, doc_id: &DocumentId) -> Result<Option<DocumentAcl>>;

    /// Log an audit event.
    async fn log_audit(&self, event: AuditEvent) -> Result<()>;

    /// Get audit events for a document.
    async fn get_audit_for_document(&self, doc_id: &DocumentId) -> Result<Vec<AuditEvent>>;
}

/// SQLite-based persistent storage backend.
pub struct SqliteBackend {
    /// Database connection (protected by mutex for thread-safety).
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl SqliteBackend {
    /// Create a new SQLite backend.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = rusqlite::Connection::open(path)
            .map_err(|e| Error::Storage(format!("Failed to open SQLite: {}", e)))?;

        Self::init_schema_sync(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Create an in-memory SQLite backend for testing.
    pub fn in_memory() -> Result<Self> {
        let conn = rusqlite::Connection::open_in_memory()
            .map_err(|e| Error::Storage(format!("Failed to open in-memory SQLite: {}", e)))?;

        Self::init_schema_sync(&conn)?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn init_schema_sync(conn: &rusqlite::Connection) -> Result<()> {
        conn.execute_batch(
            r#"
            -- Document metadata
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                owner TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                archived INTEGER NOT NULL DEFAULT 0
            );

            -- Document snapshots.
            -- No foreign key to documents: a live document session (e.g. one
            -- opened directly over the WebSocket route) can persist its state
            -- before, or without, an API-created metadata row.
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                content TEXT NOT NULL,
                elements TEXT NOT NULL DEFAULT '{}',
                vector_clock TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                UNIQUE(doc_id)
            );

            -- Operation log (see note above re: no documents foreign key).
            CREATE TABLE IF NOT EXISTS operations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                operations TEXT NOT NULL,
                vector_clock TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                UNIQUE(doc_id, seq)
            );
            CREATE INDEX IF NOT EXISTS idx_operations_doc_seq ON operations(doc_id, seq);

            -- ACL entries
            CREATE TABLE IF NOT EXISTS acl (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL REFERENCES documents(id),
                owner TEXT NOT NULL,
                public_access TEXT,
                UNIQUE(doc_id)
            );

            CREATE TABLE IF NOT EXISTS acl_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL REFERENCES documents(id),
                principal TEXT NOT NULL,
                permissions TEXT NOT NULL,
                granted_by TEXT NOT NULL,
                granted_at INTEGER NOT NULL,
                expires_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_acl_entries_doc ON acl_entries(doc_id);

            -- Audit log
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_audit_doc ON audit_log(doc_id);
            CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_log(user_id);

            -- Checkpoints for compaction
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(id),
                seq INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                snapshot_id TEXT REFERENCES snapshots(id)
            );
            CREATE INDEX IF NOT EXISTS idx_checkpoints_doc ON checkpoints(doc_id);
            "#,
        )
        .map_err(|e| Error::Storage(format!("Failed to initialize schema: {}", e)))?;

        Ok(())
    }
}

#[async_trait]
impl StorageBackend for SqliteBackend {
    async fn save_snapshot(&self, snapshot: &DocumentSnapshot) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        // A snapshot's `id` is the document id, so the primary key and the
        // `doc_id` column are the same value (one snapshot row per document).
        let doc_id = snapshot.id.to_string();
        let content = serde_json::to_string(&snapshot.content)
            .map_err(|e| Error::Storage(format!("Serialization error: {}", e)))?;
        // The full CRDT element map is what actually reconstructs the document;
        // `content` is only a convenience cache of the rendered text. Elements
        // are stored as a JSON array (each `Element` carries its own position
        // id) because JSON object keys must be strings, and `PositionId` is a
        // struct key.
        let element_values: Vec<&crate::crdt::Element> = snapshot.elements.values().collect();
        let elements = serde_json::to_string(&element_values)
            .map_err(|e| Error::Storage(format!("Serialization error: {}", e)))?;
        let vector_clock = serde_json::to_string(&snapshot.vector_clock)
            .map_err(|e| Error::Storage(format!("Serialization error: {}", e)))?;

        conn.execute(
            "INSERT OR REPLACE INTO snapshots (id, doc_id, content, elements, vector_clock, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                doc_id,
                doc_id,
                content,
                elements,
                vector_clock,
                snapshot.timestamp
            ],
        )
        .map_err(|e| Error::Storage(format!("Failed to save snapshot: {}", e)))?;

        Ok(())
    }

    async fn load_snapshot(&self, doc_id: &DocumentId) -> Result<Option<DocumentSnapshot>> {
        let conn = self.conn.lock().unwrap();
        let id = doc_id.to_string();

        let result = conn.query_row(
            "SELECT content, elements, vector_clock, timestamp FROM snapshots WHERE doc_id = ?1",
            [&id],
            |row| {
                let content_str: String = row.get(0)?;
                let elements_str: String = row.get(1)?;
                let vc_str: String = row.get(2)?;
                let timestamp: u64 = row.get(3)?;
                Ok((content_str, elements_str, vc_str, timestamp))
            },
        );

        match result {
            Ok((content_str, elements_str, vc_str, timestamp)) => {
                let content = serde_json::from_str(&content_str)
                    .map_err(|e| Error::Storage(format!("Deserialization error: {}", e)))?;
                // Elements are stored as a JSON array; rebuild the position-keyed map.
                let element_values: Vec<crate::crdt::Element> = serde_json::from_str(&elements_str)
                    .map_err(|e| Error::Storage(format!("Deserialization error: {}", e)))?;
                let elements = element_values
                    .into_iter()
                    .map(|e| (e.id.clone(), e))
                    .collect();
                let vector_clock = serde_json::from_str(&vc_str)
                    .map_err(|e| Error::Storage(format!("Deserialization error: {}", e)))?;

                Ok(Some(DocumentSnapshot {
                    id: *doc_id,
                    elements,
                    content,
                    vector_clock,
                    timestamp,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(Error::Storage(format!("Query error: {}", e))),
        }
    }

    async fn append_operations(
        &self,
        doc_id: &DocumentId,
        operations: Vec<Operation>,
        vector_clock: VectorClock,
    ) -> Result<u64> {
        let conn = self.conn.lock().unwrap();
        let id = doc_id.to_string();

        // Get next sequence number
        let seq: u64 = conn
            .query_row(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM operations WHERE doc_id = ?1",
                [&id],
                |row| row.get(0),
            )
            .map_err(|e| Error::Storage(format!("Query error: {}", e)))?;

        let ops_json = serde_json::to_string(&operations)
            .map_err(|e| Error::Storage(format!("Serialization error: {}", e)))?;
        let vc_json = serde_json::to_string(&vector_clock)
            .map_err(|e| Error::Storage(format!("Serialization error: {}", e)))?;
        let timestamp = current_timestamp();

        conn.execute(
            "INSERT INTO operations (doc_id, seq, operations, vector_clock, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![id, seq, ops_json, vc_json, timestamp],
        )
        .map_err(|e| Error::Storage(format!("Failed to append operations: {}", e)))?;

        Ok(seq)
    }

    async fn get_operations_since(
        &self,
        doc_id: &DocumentId,
        since_seq: u64,
    ) -> Result<Vec<OperationEntry>> {
        let conn = self.conn.lock().unwrap();
        let id = doc_id.to_string();

        let mut stmt = conn
            .prepare(
                "SELECT seq, operations, vector_clock, timestamp
                 FROM operations WHERE doc_id = ?1 AND seq > ?2 ORDER BY seq",
            )
            .map_err(|e| Error::Storage(format!("Prepare error: {}", e)))?;

        let rows = stmt
            .query_map(rusqlite::params![id, since_seq], |row| {
                let seq: u64 = row.get(0)?;
                let ops_json: String = row.get(1)?;
                let vc_json: String = row.get(2)?;
                let timestamp: u64 = row.get(3)?;
                Ok((seq, ops_json, vc_json, timestamp))
            })
            .map_err(|e| Error::Storage(format!("Query error: {}", e)))?;

        let mut entries = Vec::new();
        for row in rows {
            let (seq, ops_json, vc_json, timestamp) =
                row.map_err(|e| Error::Storage(format!("Row error: {}", e)))?;

            let operations: Vec<Operation> = serde_json::from_str(&ops_json)
                .map_err(|e| Error::Storage(format!("Deserialization error: {}", e)))?;
            let vector_clock: VectorClock = serde_json::from_str(&vc_json)
                .map_err(|e| Error::Storage(format!("Deserialization error: {}", e)))?;

            entries.push(OperationEntry {
                seq,
                operations,
                vector_clock,
                timestamp,
            });
        }

        Ok(entries)
    }

    async fn save_metadata(&self, metadata: DocumentMetadata) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let id = metadata.id.to_string();

        conn.execute(
            "INSERT OR REPLACE INTO documents (id, title, owner, created_at, updated_at, version, archived)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                id,
                metadata.title,
                metadata.owner.to_string(),
                metadata.created_at,
                metadata.updated_at,
                metadata.version,
                if metadata.archived { 1 } else { 0 }
            ],
        )
        .map_err(|e| Error::Storage(format!("Failed to save metadata: {}", e)))?;

        Ok(())
    }

    async fn load_metadata(&self, doc_id: &DocumentId) -> Result<Option<DocumentMetadata>> {
        let conn = self.conn.lock().unwrap();
        let id = doc_id.to_string();

        let result = conn.query_row(
            "SELECT title, owner, created_at, updated_at, version, archived
             FROM documents WHERE id = ?1",
            [&id],
            |row| {
                let title: String = row.get(0)?;
                let owner_str: String = row.get(1)?;
                let created_at: u64 = row.get(2)?;
                let updated_at: u64 = row.get(3)?;
                let version: u64 = row.get(4)?;
                let archived: i32 = row.get(5)?;
                Ok((title, owner_str, created_at, updated_at, version, archived != 0))
            },
        );

        match result {
            Ok((title, owner_str, created_at, updated_at, version, archived)) => {
                let owner = uuid::Uuid::parse_str(&owner_str)
                    .map_err(|e| Error::Storage(format!("Invalid UUID: {}", e)))?;

                Ok(Some(DocumentMetadata {
                    id: *doc_id,
                    title,
                    owner,
                    created_at,
                    updated_at,
                    version,
                    archived,
                }))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(Error::Storage(format!("Query error: {}", e))),
        }
    }

    async fn list_documents(&self) -> Result<Vec<DocumentMetadata>> {
        let conn = self.conn.lock().unwrap();

        let mut stmt = conn
            .prepare(
                "SELECT id, title, owner, created_at, updated_at, version, archived
                 FROM documents WHERE archived = 0 ORDER BY updated_at DESC",
            )
            .map_err(|e| Error::Storage(format!("Prepare error: {}", e)))?;

        let rows = stmt
            .query_map([], |row| {
                let id_str: String = row.get(0)?;
                let title: String = row.get(1)?;
                let owner_str: String = row.get(2)?;
                let created_at: u64 = row.get(3)?;
                let updated_at: u64 = row.get(4)?;
                let version: u64 = row.get(5)?;
                let archived: i32 = row.get(6)?;
                Ok((id_str, title, owner_str, created_at, updated_at, version, archived != 0))
            })
            .map_err(|e| Error::Storage(format!("Query error: {}", e)))?;

        let mut documents = Vec::new();
        for row in rows {
            let (id_str, title, owner_str, created_at, updated_at, version, archived) =
                row.map_err(|e| Error::Storage(format!("Row error: {}", e)))?;

            let id = uuid::Uuid::parse_str(&id_str)
                .map_err(|e| Error::Storage(format!("Invalid UUID: {}", e)))?;
            let owner = uuid::Uuid::parse_str(&owner_str)
                .map_err(|e| Error::Storage(format!("Invalid UUID: {}", e)))?;

            documents.push(DocumentMetadata {
                id,
                title,
                owner,
                created_at,
                updated_at,
                version,
                archived,
            });
        }

        Ok(documents)
    }

    async fn delete_document(&self, doc_id: &DocumentId) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let id = doc_id.to_string();

        // Soft delete by setting archived flag
        conn.execute(
            "UPDATE documents SET archived = 1, updated_at = ?1 WHERE id = ?2",
            rusqlite::params![current_timestamp(), id],
        )
        .map_err(|e| Error::Storage(format!("Failed to delete document: {}", e)))?;

        Ok(())
    }

    async fn compact(&self, doc_id: &DocumentId, before_seq: u64) -> Result<u64> {
        let conn = self.conn.lock().unwrap();
        let id = doc_id.to_string();

        // Delete operations before the given sequence number
        let deleted = conn
            .execute(
                "DELETE FROM operations WHERE doc_id = ?1 AND seq < ?2",
                rusqlite::params![id, before_seq],
            )
            .map_err(|e| Error::Storage(format!("Failed to compact: {}", e)))?;

        Ok(deleted as u64)
    }

    async fn save_acl(&self, acl: &DocumentAcl) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let id = acl.doc_id.to_string();

        // Save main ACL record
        let public_access = acl.public_access.map(|p| format!("{:?}", p));
        conn.execute(
            "INSERT OR REPLACE INTO acl (doc_id, owner, public_access)
             VALUES (?1, ?2, ?3)",
            rusqlite::params![id, acl.owner.to_string(), public_access],
        )
        .map_err(|e| Error::Storage(format!("Failed to save ACL: {}", e)))?;

        // Delete existing entries and re-insert
        conn.execute("DELETE FROM acl_entries WHERE doc_id = ?1", [&id])
            .map_err(|e| Error::Storage(format!("Failed to clear ACL entries: {}", e)))?;

        for entry in &acl.entries {
            let perms_json = serde_json::to_string(&entry.permissions)
                .map_err(|e| Error::Storage(format!("Serialization error: {}", e)))?;

            conn.execute(
                "INSERT INTO acl_entries (doc_id, principal, permissions, granted_by, granted_at, expires_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    id,
                    entry.principal,
                    perms_json,
                    entry.granted_by.to_string(),
                    entry.granted_at,
                    entry.expires_at
                ],
            )
            .map_err(|e| Error::Storage(format!("Failed to save ACL entry: {}", e)))?;
        }

        Ok(())
    }

    async fn load_acl(&self, doc_id: &DocumentId) -> Result<Option<DocumentAcl>> {
        let conn = self.conn.lock().unwrap();
        let id = doc_id.to_string();

        // Load main ACL record
        let result = conn.query_row(
            "SELECT owner, public_access FROM acl WHERE doc_id = ?1",
            [&id],
            |row| {
                let owner_str: String = row.get(0)?;
                let public_access: Option<String> = row.get(1)?;
                Ok((owner_str, public_access))
            },
        );

        let (owner_str, public_access_str) = match result {
            Ok(r) => r,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(Error::Storage(format!("Query error: {}", e))),
        };

        let owner = uuid::Uuid::parse_str(&owner_str)
            .map_err(|e| Error::Storage(format!("Invalid UUID: {}", e)))?;

        let public_access = public_access_str.map(|s| match s.as_str() {
            "Read" => Permission::Read,
            "Write" => Permission::Write,
            "Comment" => Permission::Comment,
            "Admin" => Permission::Admin,
            _ => Permission::Read,
        });

        // Load ACL entries
        let mut stmt = conn
            .prepare(
                "SELECT principal, permissions, granted_by, granted_at, expires_at
                 FROM acl_entries WHERE doc_id = ?1",
            )
            .map_err(|e| Error::Storage(format!("Prepare error: {}", e)))?;

        let rows = stmt
            .query_map([&id], |row| {
                let principal: String = row.get(0)?;
                let perms_json: String = row.get(1)?;
                let granted_by_str: String = row.get(2)?;
                let granted_at: u64 = row.get(3)?;
                let expires_at: Option<u64> = row.get(4)?;
                Ok((principal, perms_json, granted_by_str, granted_at, expires_at))
            })
            .map_err(|e| Error::Storage(format!("Query error: {}", e)))?;

        let mut entries = Vec::new();
        for row in rows {
            let (principal, perms_json, granted_by_str, granted_at, expires_at) =
                row.map_err(|e| Error::Storage(format!("Row error: {}", e)))?;

            let permissions: Vec<Permission> = serde_json::from_str(&perms_json)
                .map_err(|e| Error::Storage(format!("Deserialization error: {}", e)))?;
            let granted_by = uuid::Uuid::parse_str(&granted_by_str)
                .map_err(|e| Error::Storage(format!("Invalid UUID: {}", e)))?;

            entries.push(AclEntry {
                principal,
                permissions,
                granted_by,
                granted_at,
                expires_at,
            });
        }

        Ok(Some(DocumentAcl {
            doc_id: *doc_id,
            owner,
            entries,
            public_access,
        }))
    }

    async fn log_audit(&self, event: AuditEvent) -> Result<()> {
        let conn = self.conn.lock().unwrap();

        let action = format!("{:?}", event.action);
        let details = event.details.to_string();

        conn.execute(
            "INSERT INTO audit_log (event_id, timestamp, user_id, doc_id, action, details)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                event.event_id.to_string(),
                event.timestamp,
                event.user_id.to_string(),
                event.doc_id.to_string(),
                action,
                details
            ],
        )
        .map_err(|e| Error::Storage(format!("Failed to log audit event: {}", e)))?;

        Ok(())
    }

    async fn get_audit_for_document(&self, doc_id: &DocumentId) -> Result<Vec<AuditEvent>> {
        let conn = self.conn.lock().unwrap();
        let id = doc_id.to_string();

        let mut stmt = conn
            .prepare(
                "SELECT event_id, timestamp, user_id, action, details
                 FROM audit_log WHERE doc_id = ?1 ORDER BY timestamp DESC",
            )
            .map_err(|e| Error::Storage(format!("Prepare error: {}", e)))?;

        let rows = stmt
            .query_map([&id], |row| {
                let event_id_str: String = row.get(0)?;
                let timestamp: u64 = row.get(1)?;
                let user_id_str: String = row.get(2)?;
                let action_str: String = row.get(3)?;
                let details_str: String = row.get(4)?;
                Ok((event_id_str, timestamp, user_id_str, action_str, details_str))
            })
            .map_err(|e| Error::Storage(format!("Query error: {}", e)))?;

        let mut events = Vec::new();
        for row in rows {
            let (event_id_str, timestamp, user_id_str, action_str, details_str) =
                row.map_err(|e| Error::Storage(format!("Row error: {}", e)))?;

            let event_id = uuid::Uuid::parse_str(&event_id_str)
                .map_err(|e| Error::Storage(format!("Invalid UUID: {}", e)))?;
            let user_id = uuid::Uuid::parse_str(&user_id_str)
                .map_err(|e| Error::Storage(format!("Invalid UUID: {}", e)))?;

            let action = match action_str.as_str() {
                "DocumentCreated" => AuditAction::DocumentCreated,
                "DocumentOpened" => AuditAction::DocumentOpened,
                "DocumentEdited" => AuditAction::DocumentEdited,
                "DocumentShared" => AuditAction::DocumentShared,
                "PermissionChanged" => AuditAction::PermissionChanged,
                "DocumentDeleted" => AuditAction::DocumentDeleted,
                "DocumentRestored" => AuditAction::DocumentRestored,
                _ => AuditAction::DocumentOpened,
            };

            let details: serde_json::Value = serde_json::from_str(&details_str)
                .unwrap_or(serde_json::Value::Null);

            events.push(AuditEvent {
                event_id,
                timestamp,
                user_id,
                doc_id: *doc_id,
                action,
                details,
            });
        }

        Ok(events)
    }
}

/// Get current timestamp in milliseconds.
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Compaction manager for automatic log cleanup.
pub struct CompactionManager {
    /// Storage backend.
    backend: Arc<dyn StorageBackend>,
    /// Compaction threshold (number of operations before compaction).
    threshold: u64,
    /// Keep recent operations count.
    keep_recent: u64,
}

impl CompactionManager {
    /// Create a new compaction manager.
    pub fn new(backend: Arc<dyn StorageBackend>, threshold: u64, keep_recent: u64) -> Self {
        Self {
            backend,
            threshold,
            keep_recent,
        }
    }

    /// Check if compaction is needed and perform it.
    pub async fn maybe_compact(&self, doc_id: &DocumentId) -> Result<bool> {
        // Get current operations count
        let ops = self.backend.get_operations_since(doc_id, 0).await?;
        let count = ops.len() as u64;

        if count < self.threshold {
            return Ok(false);
        }

        // Compact - keep only recent operations
        let compact_before = count.saturating_sub(self.keep_recent);
        if compact_before > 0 {
            self.backend.compact(doc_id, compact_before).await?;
            return Ok(true);
        }

        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_backend_metadata() {
        let backend = SqliteBackend::in_memory().unwrap();

        let doc_id = uuid::Uuid::new_v4();
        let owner = uuid::Uuid::new_v4();

        let metadata = DocumentMetadata {
            id: doc_id,
            title: "Test Document".to_string(),
            owner,
            created_at: 1000,
            updated_at: 1000,
            version: 1,
            archived: false,
        };

        backend.save_metadata(metadata.clone()).await.unwrap();

        let loaded = backend.load_metadata(&doc_id).await.unwrap().unwrap();
        assert_eq!(loaded.title, "Test Document");
        assert_eq!(loaded.owner, owner);
    }

    #[tokio::test]
    async fn test_sqlite_backend_operations() {
        let backend = SqliteBackend::in_memory().unwrap();

        let doc_id = uuid::Uuid::new_v4();
        let owner = uuid::Uuid::new_v4();

        // First save metadata (foreign key)
        let metadata = DocumentMetadata {
            id: doc_id,
            title: "Test".to_string(),
            owner,
            created_at: 1000,
            updated_at: 1000,
            version: 1,
            archived: false,
        };
        backend.save_metadata(metadata).await.unwrap();

        // Append operations
        let ops = vec![];
        let vc = VectorClock::new();

        let seq1 = backend.append_operations(&doc_id, ops.clone(), vc.clone()).await.unwrap();
        assert_eq!(seq1, 1);

        let seq2 = backend.append_operations(&doc_id, ops.clone(), vc.clone()).await.unwrap();
        assert_eq!(seq2, 2);

        // Get operations since
        let entries = backend.get_operations_since(&doc_id, 0).await.unwrap();
        assert_eq!(entries.len(), 2);

        let entries = backend.get_operations_since(&doc_id, 1).await.unwrap();
        assert_eq!(entries.len(), 1);
    }

    #[tokio::test]
    async fn test_sqlite_backend_compaction() {
        let backend = SqliteBackend::in_memory().unwrap();

        let doc_id = uuid::Uuid::new_v4();
        let owner = uuid::Uuid::new_v4();

        // First save metadata
        let metadata = DocumentMetadata {
            id: doc_id,
            title: "Test".to_string(),
            owner,
            created_at: 1000,
            updated_at: 1000,
            version: 1,
            archived: false,
        };
        backend.save_metadata(metadata).await.unwrap();

        // Add multiple operations
        let ops = vec![];
        let vc = VectorClock::new();

        for _ in 0..5 {
            backend.append_operations(&doc_id, ops.clone(), vc.clone()).await.unwrap();
        }

        // Compact
        let deleted = backend.compact(&doc_id, 3).await.unwrap();
        assert_eq!(deleted, 2);

        // Verify remaining operations
        let entries = backend.get_operations_since(&doc_id, 0).await.unwrap();
        assert_eq!(entries.len(), 3);
    }

    #[tokio::test]
    async fn test_sqlite_backend_acl() {
        let backend = SqliteBackend::in_memory().unwrap();

        let doc_id = uuid::Uuid::new_v4();
        let owner = uuid::Uuid::new_v4();

        // Create document first (required for foreign key constraint)
        let metadata = DocumentMetadata {
            id: doc_id,
            title: "Test".to_string(),
            owner,
            created_at: 1000,
            updated_at: 1000,
            version: 1,
            archived: false,
        };
        backend.save_metadata(metadata).await.unwrap();

        let mut acl = DocumentAcl::new(doc_id, owner);
        acl.grant("user1", Permission::Read, owner);
        acl.grant("user2", Permission::Write, owner);

        backend.save_acl(&acl).await.unwrap();

        let loaded = backend.load_acl(&doc_id).await.unwrap().unwrap();
        assert_eq!(loaded.owner, owner);
        assert_eq!(loaded.entries.len(), 2);
    }

    #[tokio::test]
    async fn test_sqlite_backend_audit() {
        let backend = SqliteBackend::in_memory().unwrap();

        let doc_id = uuid::Uuid::new_v4();
        let user_id = uuid::Uuid::new_v4();

        let event = AuditEvent {
            event_id: uuid::Uuid::new_v4(),
            timestamp: 1000,
            user_id,
            doc_id,
            action: AuditAction::DocumentCreated,
            details: serde_json::json!({"test": "value"}),
        };

        backend.log_audit(event).await.unwrap();

        let events = backend.get_audit_for_document(&doc_id).await.unwrap();
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].action, AuditAction::DocumentCreated));
    }
}
