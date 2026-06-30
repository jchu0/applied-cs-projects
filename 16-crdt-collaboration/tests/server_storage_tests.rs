//! Tests for server, storage, performance, and API components.

use crdt_collaboration::crdt::*;
use crdt_collaboration::document::{Document, DocumentMetadata};
use crdt_collaboration::performance::*;
use crdt_collaboration::server::*;
use crdt_collaboration::storage::*;
use crdt_collaboration::{ClientId, DocumentId};
use std::collections::HashMap;
use std::sync::Arc;

/// Helper to create a DocumentMetadata for testing.
fn make_metadata(did: DocumentId) -> DocumentMetadata {
    DocumentMetadata {
        id: did,
        title: "Test Document".to_string(),
        owner: client_id(1),
        created_at: 1000,
        updated_at: 1000,
        version: 1,
        archived: false,
    }
}

/// Helper to create deterministic client IDs for testing.
fn client_id(n: u8) -> ClientId {
    ClientId::from_bytes([n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
}

/// Helper to create deterministic document IDs for testing.
fn doc_id(n: u8) -> DocumentId {
    DocumentId::from_bytes([n, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
}

// =============================================================================
// Server Tests
// =============================================================================

#[cfg(test)]
mod server_tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.max_connections_per_doc, 100);
        assert_eq!(config.batch_window_ms, 50);
        assert_eq!(config.heartbeat_interval_secs, 30);
        assert_eq!(config.snapshot_interval, 1000);
    }

    #[test]
    fn test_document_session_creation() {
        let did = doc_id(1);
        let doc = Document::new(did);
        let session = DocumentSession::new(did, doc);

        assert_eq!(session.doc_id, did);
        assert_eq!(session.client_count(), 0);
    }

    #[test]
    fn test_document_session_add_remove_client() {
        let did = doc_id(1);
        let doc = Document::new(did);
        let session = DocumentSession::new(did, doc);
        let cid = client_id(1);

        let (tx, _rx) = tokio::sync::mpsc::channel(10);
        let conn = ClientConnection {
            client_id: cid,
            sender: tx,
            name: "Alice".to_string(),
            color: "#ff0000".to_string(),
        };

        session.add_client(cid, conn);
        assert_eq!(session.client_count(), 1);

        session.remove_client(&cid);
        assert_eq!(session.client_count(), 0);
    }

    #[test]
    fn test_document_session_apply_operations() {
        let did = doc_id(1);
        let doc = Document::new(did);
        let session = DocumentSession::new(did, doc);
        let cid = client_id(1);

        let op = Operation::Insert {
            id: PositionId::new(1, cid, 0),
            after: PositionId::root(),
            value: 'H',
            attributes: HashMap::new(),
        };

        let seq = session.apply_operations(cid, vec![op]).unwrap();
        assert_eq!(seq, 0);

        // Verify the document was updated
        let doc = session.document.read();
        assert_eq!(doc.text(), "H");
    }

    #[test]
    fn test_document_session_get_sync_response() {
        let did = doc_id(1);
        let doc = Document::new(did);
        let session = DocumentSession::new(did, doc);

        let resp = session.get_sync_response();
        assert_eq!(resp.doc_id, did);
        assert!(resp.pending_ops.is_empty());
    }

    #[test]
    fn test_document_session_multiple_operations() {
        let did = doc_id(1);
        let doc = Document::new(did);
        let session = DocumentSession::new(did, doc);
        let cid = client_id(1);

        let op1 = Operation::Insert {
            id: PositionId::new(1, cid, 0),
            after: PositionId::root(),
            value: 'A',
            attributes: HashMap::new(),
        };
        let op2 = Operation::Insert {
            id: PositionId::new(2, cid, 0),
            after: PositionId::new(1, cid, 0),
            value: 'B',
            attributes: HashMap::new(),
        };

        let seq1 = session.apply_operations(cid, vec![op1]).unwrap();
        let seq2 = session.apply_operations(cid, vec![op2]).unwrap();
        assert_eq!(seq2, seq1 + 1);

        let doc = session.document.read();
        assert_eq!(doc.text(), "AB");
    }

    #[tokio::test]
    async fn test_collaboration_server_creation() {
        let config = ServerConfig::default();
        let storage = Arc::new(StorageManager::new());
        let server = CollaborationServer::new(config, storage);

        assert!(server.sessions.is_empty());
    }

    #[tokio::test]
    async fn test_collaboration_server_get_or_create_session() {
        let config = ServerConfig::default();
        let storage = Arc::new(StorageManager::new());
        let server = CollaborationServer::new(config, storage);

        let did = doc_id(1);
        let session = server.get_or_create_session(did).await.unwrap();
        assert_eq!(session.doc_id, did);

        // Getting same doc should return same session
        let session2 = server.get_or_create_session(did).await.unwrap();
        assert_eq!(session.doc_id, session2.doc_id);
        assert_eq!(server.sessions.len(), 1);
    }

    #[tokio::test]
    async fn test_collaboration_server_multiple_sessions() {
        let config = ServerConfig::default();
        let storage = Arc::new(StorageManager::new());
        let server = CollaborationServer::new(config, storage);

        server.get_or_create_session(doc_id(1)).await.unwrap();
        server.get_or_create_session(doc_id(2)).await.unwrap();
        server.get_or_create_session(doc_id(3)).await.unwrap();

        assert_eq!(server.sessions.len(), 3);
    }
}

// =============================================================================
// Storage Tests
// =============================================================================

#[cfg(test)]
mod storage_tests {
    use super::*;

    #[tokio::test]
    async fn test_storage_save_load_snapshot() {
        let storage = StorageManager::new();
        let did = doc_id(1);
        let doc = Document::new(did);
        let snapshot = doc.snapshot();

        storage.save_snapshot(&snapshot).await.unwrap();
        let loaded = storage.load_snapshot(&did).await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().id, did);
    }

    #[tokio::test]
    async fn test_storage_load_nonexistent_snapshot() {
        let storage = StorageManager::new();
        let result = storage.load_snapshot(&doc_id(99)).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_storage_append_and_get_operations() {
        let storage = StorageManager::new();
        let did = doc_id(1);
        let cid = client_id(1);

        let ops = vec![Operation::Insert {
            id: PositionId::new(1, cid, 0),
            after: PositionId::root(),
            value: 'A',
            attributes: HashMap::new(),
        }];

        let seq = storage.append_operations(&did, ops, VectorClock::new()).await.unwrap();
        assert_eq!(seq, 1);

        let entries = storage.get_operations_since(&did, 0).await.unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].seq, 1);
    }

    #[tokio::test]
    async fn test_storage_get_operations_since_filters() {
        let storage = StorageManager::new();
        let did = doc_id(1);
        let cid = client_id(1);

        for i in 1..=5 {
            let ops = vec![Operation::Insert {
                id: PositionId::new(i, cid, 0),
                after: PositionId::root(),
                value: 'A',
                attributes: HashMap::new(),
            }];
            storage.append_operations(&did, ops, VectorClock::new()).await.unwrap();
        }

        // Get since seq 3 (should return seq 4 and 5)
        let entries = storage.get_operations_since(&did, 3).await.unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].seq, 4);
        assert_eq!(entries[1].seq, 5);
    }

    #[tokio::test]
    async fn test_storage_metadata_save_load() {
        let storage = StorageManager::new();
        let did = doc_id(1);
        let metadata = make_metadata(did);

        storage.save_metadata(metadata).await.unwrap();
        let loaded = storage.load_metadata(&did).await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().id, did);
    }

    #[tokio::test]
    async fn test_storage_list_documents() {
        let storage = StorageManager::new();

        for i in 1..=3 {
            storage.save_metadata(make_metadata(doc_id(i))).await.unwrap();
        }

        let docs = storage.list_documents().await.unwrap();
        assert_eq!(docs.len(), 3);
    }

    #[tokio::test]
    async fn test_storage_delete_document() {
        let storage = StorageManager::new();
        let did = doc_id(1);
        let doc = Document::new(did);

        storage.save_snapshot(&doc.snapshot()).await.unwrap();
        storage.save_metadata(make_metadata(did)).await.unwrap();

        storage.delete_document(&did).await.unwrap();
        assert!(storage.load_snapshot(&did).await.unwrap().is_none());
        assert!(storage.load_metadata(&did).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_storage_compact() {
        let storage = StorageManager::new();
        let did = doc_id(1);
        let cid = client_id(1);

        // Add 150 entries
        for i in 1..=150 {
            let ops = vec![Operation::Insert {
                id: PositionId::new(i, cid, 0),
                after: PositionId::root(),
                value: 'A',
                attributes: HashMap::new(),
            }];
            storage.append_operations(&did, ops, VectorClock::new()).await.unwrap();
        }

        // Compact
        storage.compact(&did).await.unwrap();

        // Should have at most 100 entries now
        let entries = storage.get_operations_since(&did, 0).await.unwrap();
        assert!(entries.len() <= 100);
    }

    #[test]
    fn test_operation_log_len_and_empty() {
        let log = OperationLog::new(doc_id(1));
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn test_operation_log_append() {
        let mut log = OperationLog::new(doc_id(1));
        let cid = client_id(1);

        let ops = vec![Operation::Insert {
            id: PositionId::new(1, cid, 0),
            after: PositionId::root(),
            value: 'A',
            attributes: HashMap::new(),
        }];

        let seq = log.append(ops, VectorClock::new());
        assert_eq!(seq, 1);
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_operation_log_truncate_before() {
        let mut log = OperationLog::new(doc_id(1));
        let cid = client_id(1);

        for i in 1..=10 {
            let ops = vec![Operation::Insert {
                id: PositionId::new(i, cid, 0),
                after: PositionId::root(),
                value: 'A',
                attributes: HashMap::new(),
            }];
            log.append(ops, VectorClock::new());
        }

        assert_eq!(log.len(), 10);
        log.truncate_before(3);
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_operation_log_truncate_noop_when_small() {
        let mut log = OperationLog::new(doc_id(1));
        let cid = client_id(1);

        let ops = vec![Operation::Insert {
            id: PositionId::new(1, cid, 0),
            after: PositionId::root(),
            value: 'A',
            attributes: HashMap::new(),
        }];
        log.append(ops, VectorClock::new());

        log.truncate_before(10);
        assert_eq!(log.len(), 1);
    }
}

// =============================================================================
// ACL Tests
// =============================================================================

#[cfg(test)]
mod acl_tests {
    use super::*;

    #[test]
    fn test_acl_owner_has_all_permissions() {
        let owner = client_id(1);
        let acl = DocumentAcl::new(doc_id(1), owner);

        assert!(acl.has_permission(&owner, Permission::Read));
        assert!(acl.has_permission(&owner, Permission::Write));
        assert!(acl.has_permission(&owner, Permission::Admin));
    }

    #[test]
    fn test_acl_grant_and_check() {
        let owner = client_id(1);
        let user = client_id(2);
        let mut acl = DocumentAcl::new(doc_id(1), owner);

        // User should not have access initially
        assert!(!acl.has_permission(&user, Permission::Read));

        // Grant write access
        acl.grant(&user.to_string(), Permission::Write, owner);
        assert!(acl.has_permission(&user, Permission::Read));
        assert!(acl.has_permission(&user, Permission::Write));
    }

    #[test]
    fn test_acl_revoke() {
        let owner = client_id(1);
        let user = client_id(2);
        let mut acl = DocumentAcl::new(doc_id(1), owner);

        acl.grant(&user.to_string(), Permission::Write, owner);
        assert!(acl.has_permission(&user, Permission::Write));

        acl.revoke(&user.to_string());
        assert!(!acl.has_permission(&user, Permission::Write));
    }

    #[test]
    fn test_acl_public_access() {
        let owner = client_id(1);
        let user = client_id(2);
        let mut acl = DocumentAcl::new(doc_id(1), owner);

        acl.public_access = Some(Permission::Read);
        assert!(acl.has_permission(&user, Permission::Read));
        assert!(!acl.has_permission(&user, Permission::Write));
    }
}

// =============================================================================
// Audit Log Tests
// =============================================================================

#[cfg(test)]
mod audit_tests {
    use super::*;

    #[test]
    fn test_audit_log_creation() {
        let log = AuditLog::new();
        let events = log.get_for_document(&doc_id(1));
        assert!(events.is_empty());
    }

    #[test]
    fn test_audit_log_event() {
        let log = AuditLog::new();
        let did = doc_id(1);
        let uid = client_id(1);

        log.log(AuditEvent {
            event_id: uuid::Uuid::new_v4(),
            timestamp: 1000,
            user_id: uid,
            doc_id: did,
            action: AuditAction::DocumentCreated,
            details: serde_json::json!({}),
        });

        let events = log.get_for_document(&did);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_audit_log_get_for_user() {
        let log = AuditLog::new();
        let uid1 = client_id(1);
        let uid2 = client_id(2);

        log.log(AuditEvent {
            event_id: uuid::Uuid::new_v4(),
            timestamp: 1000,
            user_id: uid1,
            doc_id: doc_id(1),
            action: AuditAction::DocumentCreated,
            details: serde_json::json!({}),
        });

        log.log(AuditEvent {
            event_id: uuid::Uuid::new_v4(),
            timestamp: 2000,
            user_id: uid2,
            doc_id: doc_id(1),
            action: AuditAction::DocumentEdited,
            details: serde_json::json!({}),
        });

        let user1_events = log.get_for_user(&uid1);
        assert_eq!(user1_events.len(), 1);

        let user2_events = log.get_for_user(&uid2);
        assert_eq!(user2_events.len(), 1);
    }

    #[test]
    fn test_audit_log_multiple_events() {
        let log = AuditLog::new();
        let did = doc_id(1);
        let uid = client_id(1);

        for action in [AuditAction::DocumentCreated, AuditAction::DocumentEdited, AuditAction::DocumentShared] {
            log.log(AuditEvent {
                event_id: uuid::Uuid::new_v4(),
                timestamp: 1000,
                user_id: uid,
                doc_id: did,
                action,
                details: serde_json::json!({}),
            });
        }

        let events = log.get_for_document(&did);
        assert_eq!(events.len(), 3);
    }
}

// =============================================================================
// Performance Module Tests
// =============================================================================

#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_batcher_flush_expired() {
        let config = BatchConfig {
            max_batch_size: 100,
            max_batch_delay_ms: 1, // 1ms expiry
            adaptive: false,
        };
        let batcher = OperationBatcher::new(config);
        let did = doc_id(1);
        let cid = client_id(1);

        batcher.add_operation(
            did,
            Operation::Insert {
                id: PositionId::new(1, cid, 0),
                after: PositionId::root(),
                value: 'a',
                attributes: HashMap::new(),
            },
            VectorClock::new(),
        );

        // Wait for expiry
        std::thread::sleep(std::time::Duration::from_millis(10));
        let flushed = batcher.flush_expired();
        assert_eq!(flushed.len(), 1);
        assert_eq!(flushed[0].0, did);
    }

    #[test]
    fn test_batcher_stats() {
        let config = BatchConfig {
            max_batch_size: 2,
            max_batch_delay_ms: 1000,
            adaptive: false,
        };
        let batcher = OperationBatcher::new(config);
        let did = doc_id(1);
        let cid = client_id(1);

        // Add 2 operations to trigger a flush
        batcher.add_operation(
            did,
            Operation::Insert {
                id: PositionId::new(1, cid, 0),
                after: PositionId::root(),
                value: 'a',
                attributes: HashMap::new(),
            },
            VectorClock::new(),
        );
        batcher.add_operation(
            did,
            Operation::Insert {
                id: PositionId::new(2, cid, 0),
                after: PositionId::root(),
                value: 'b',
                attributes: HashMap::new(),
            },
            VectorClock::new(),
        );

        let (ops, batches, avg) = batcher.get_stats();
        assert_eq!(ops, 2);
        assert_eq!(batches, 1);
        assert_eq!(avg, 2.0);
    }

    #[test]
    fn test_tombstone_gc_record_and_should_run() {
        let gc = TombstoneGC::new(GCConfig {
            min_operations_between_gc: 5,
            max_tombstone_ratio: 0.3,
            auto_gc: true,
            ..Default::default()
        });
        let did = doc_id(1);

        // Not enough operations yet
        for _ in 0..3 {
            gc.record_operation(did);
        }
        assert!(!gc.should_run_gc(&did, 40, 100));

        // Enough operations, high tombstone ratio
        for _ in 0..5 {
            gc.record_operation(did);
        }
        assert!(gc.should_run_gc(&did, 40, 100));
    }

    #[test]
    fn test_tombstone_gc_auto_disabled() {
        let gc = TombstoneGC::new(GCConfig {
            auto_gc: false,
            min_operations_between_gc: 0,
            max_tombstone_ratio: 0.0,
            ..Default::default()
        });
        let did = doc_id(1);
        gc.record_operation(did);
        assert!(!gc.should_run_gc(&did, 100, 100));
    }

    #[test]
    fn test_tombstone_gc_stats() {
        let gc = TombstoneGC::new(GCConfig::default());
        let (collected, runs, freed) = gc.get_stats();
        assert_eq!(collected, 0);
        assert_eq!(runs, 0);
        assert_eq!(freed, 0);
    }

    #[test]
    fn test_version_history_get_latest() {
        let history = VersionHistory::new(100, 10);
        let did = doc_id(1);
        let cid = client_id(1);

        assert!(history.get_latest_version(&did).is_none());

        history.create_version(did, Some("v1".to_string()), cid, VectorClock::new(), 1, None);
        history.create_version(did, Some("v2".to_string()), cid, VectorClock::new(), 10, None);

        let latest = history.get_latest_version(&did).unwrap();
        assert_eq!(latest.version_number, 2);
        assert_eq!(latest.label, Some("v2".to_string()));
    }

    #[test]
    fn test_version_history_get_by_id() {
        let history = VersionHistory::new(100, 10);
        let did = doc_id(1);
        let cid = client_id(1);

        let v = history.create_version(did, Some("test".to_string()), cid, VectorClock::new(), 1, None);
        let fetched = history.get_version_by_id(&did, v.id).unwrap();
        assert_eq!(fetched.label, Some("test".to_string()));
    }

    #[test]
    fn test_version_history_delete() {
        let history = VersionHistory::new(100, 10);
        let did = doc_id(1);
        let cid = client_id(1);

        history.create_version(did, None, cid, VectorClock::new(), 1, None);
        history.create_version(did, None, cid, VectorClock::new(), 2, None);

        assert!(history.delete_version(&did, 1));
        assert_eq!(history.get_versions(&did).len(), 1);

        // Deleting nonexistent version
        assert!(!history.delete_version(&did, 99));
    }

    #[test]
    fn test_version_history_record_operation() {
        let history = VersionHistory::new(3, 10);
        let did = doc_id(1);

        assert!(!history.record_operation(did));
        assert!(!history.record_operation(did));
        // Third operation should trigger auto-version
        assert!(history.record_operation(did));
    }

    #[test]
    fn test_version_history_max_versions_pruning() {
        let history = VersionHistory::new(100, 3);
        let did = doc_id(1);
        let cid = client_id(1);

        for i in 1..=5 {
            history.create_version(did, None, cid, VectorClock::new(), i, None);
        }

        // Should be pruned to max 3
        let versions = history.get_versions(&did);
        assert!(versions.len() <= 3);
    }

    #[test]
    fn test_log_compactor_compact() {
        let compactor = LogCompactor::new(100, 20);
        let did = doc_id(1);
        let doc = Document::new(did);
        let snapshot = doc.snapshot();
        let cid = client_id(1);

        // Create entries
        let mut entries = Vec::new();
        for i in 1..=50 {
            entries.push(OperationEntry {
                seq: i,
                operations: vec![Operation::Insert {
                    id: PositionId::new(i, cid, 0),
                    after: PositionId::root(),
                    value: 'A',
                    attributes: HashMap::new(),
                }],
                vector_clock: VectorClock::new(),
                timestamp: i * 1000,
            });
        }

        let (checkpoint, kept) = compactor.compact(did, snapshot, &entries);
        assert_eq!(kept.len(), 20);
        assert_eq!(checkpoint.doc_id, did);

        let (compactions, removed, checkpoints) = compactor.get_stats();
        assert_eq!(compactions, 1);
        assert_eq!(removed, 30);
        assert_eq!(checkpoints, 1);
    }

    #[test]
    fn test_memory_monitor_estimate_document_memory() {
        let monitor = MemoryMonitor::new(1024 * 1024);
        let est = monitor.estimate_document_memory(1000, 100, 4);
        assert!(est > 0);
    }

    #[test]
    fn test_memory_monitor_estimate_log_memory() {
        let monitor = MemoryMonitor::new(1024 * 1024);
        let est = monitor.estimate_log_memory(500, 3);
        assert!(est > 0);
    }

    #[test]
    fn test_memory_monitor_total_across_documents() {
        let monitor = MemoryMonitor::new(1024 * 1024);

        monitor.update_stats(doc_id(1), MemoryStats {
            document_bytes: 1000,
            operation_log_bytes: 500,
            snapshot_bytes: 200,
            tombstone_count: 10,
            active_element_count: 100,
        });
        monitor.update_stats(doc_id(2), MemoryStats {
            document_bytes: 2000,
            operation_log_bytes: 1000,
            snapshot_bytes: 400,
            tombstone_count: 20,
            active_element_count: 200,
        });

        let total = monitor.total_memory();
        assert_eq!(total, 1000 + 500 + 200 + 2000 + 1000 + 400);
    }
}

// =============================================================================
// Compute Minimum Clock Tests
// =============================================================================

#[cfg(test)]
mod minimum_clock_tests {
    use super::*;

    #[test]
    fn test_vector_clock_clients() {
        let mut vc = VectorClock::new();
        let c1 = client_id(1);
        let c2 = client_id(2);

        vc.increment(c1);
        vc.increment(c2);

        let clients = vc.clients();
        assert_eq!(clients.len(), 2);
        assert!(clients.contains(&c1));
        assert!(clients.contains(&c2));
    }

    #[test]
    fn test_vector_clock_dominates() {
        let mut vc1 = VectorClock::new();
        let mut vc2 = VectorClock::new();
        let c1 = client_id(1);
        let c2 = client_id(2);

        vc1.set(c1, 5);
        vc1.set(c2, 3);

        vc2.set(c1, 3);
        vc2.set(c2, 2);

        assert!(vc1.dominates(&vc2));
        assert!(!vc2.dominates(&vc1));
    }
}

// =============================================================================
// API State Tests
// =============================================================================

#[cfg(test)]
mod api_state_tests {
    use super::*;
    use crdt_collaboration::api::ApiState;

    #[test]
    fn test_api_state_creation() {
        let config = ServerConfig::default();
        let storage = Arc::new(StorageManager::new());
        let server = Arc::new(CollaborationServer::new(config, storage.clone()));
        let state = ApiState::new(server, storage);

        assert!(state.acls.read().is_empty());
    }
}
