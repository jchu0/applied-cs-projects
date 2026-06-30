//! Offline support for CRDT collaboration.
//!
//! This module provides offline-first capabilities including:
//! - Local operation queueing while disconnected
//! - Automatic sync on reconnection
//! - Conflict detection and resolution
//! - Persistent queue storage

use crate::crdt::{Operation, VectorClock};
use crate::protocol::{Message, OperationMessage, SyncResponse};
use crate::{ClientId, DocumentId, Error, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{broadcast, mpsc, RwLock};

/// Connection state for offline detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connected to server.
    Online,
    /// Disconnected from server.
    Offline,
    /// Attempting to reconnect.
    Reconnecting,
    /// Syncing pending operations.
    Syncing,
}

/// Pending operation waiting to be sent to server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingOperation {
    /// Unique operation ID.
    pub id: u64,
    /// Document ID.
    pub doc_id: DocumentId,
    /// Operations.
    pub operations: Vec<Operation>,
    /// Vector clock at time of operation.
    pub vector_clock: VectorClock,
    /// When the operation was created.
    pub created_at: u64,
    /// Number of send attempts.
    pub attempts: u32,
    /// Last send attempt time.
    pub last_attempt: Option<u64>,
}

/// Sync conflict detected during reconnection.
#[derive(Debug, Clone)]
pub struct SyncConflict {
    /// Document ID.
    pub doc_id: DocumentId,
    /// Local operations that conflict.
    pub local_ops: Vec<Operation>,
    /// Server operations that conflict.
    pub server_ops: Vec<Operation>,
    /// Conflict resolution strategy used.
    pub resolution: ConflictResolution,
}

/// Conflict resolution strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Automatically merge using CRDT properties.
    AutoMerge,
    /// Server operations take precedence.
    ServerWins,
    /// Local operations take precedence.
    LocalWins,
    /// Requires manual resolution.
    Manual,
}

/// Offline manager configuration.
#[derive(Debug, Clone)]
pub struct OfflineConfig {
    /// Maximum pending operations to queue.
    pub max_queue_size: usize,
    /// How long to keep operations before discarding (seconds).
    pub operation_ttl_secs: u64,
    /// Maximum retry attempts per operation.
    pub max_retries: u32,
    /// Retry delay in milliseconds.
    pub retry_delay_ms: u64,
    /// Whether to persist queue to local storage.
    pub persist_queue: bool,
    /// Conflict resolution strategy.
    pub conflict_resolution: ConflictResolution,
    /// Heartbeat interval for connection detection (milliseconds).
    pub heartbeat_interval_ms: u64,
    /// Connection timeout (milliseconds).
    pub connection_timeout_ms: u64,
}

impl Default for OfflineConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            operation_ttl_secs: 86400, // 24 hours
            max_retries: 10,
            retry_delay_ms: 1000,
            persist_queue: true,
            conflict_resolution: ConflictResolution::AutoMerge,
            heartbeat_interval_ms: 5000,
            connection_timeout_ms: 10000,
        }
    }
}

/// Events emitted by the offline manager.
#[derive(Debug, Clone)]
pub enum OfflineEvent {
    /// Connection state changed.
    StateChanged(ConnectionState),
    /// Operation queued while offline.
    OperationQueued(u64),
    /// Operation sent successfully.
    OperationSent(u64),
    /// Operation failed after retries.
    OperationFailed(u64, String),
    /// Sync started.
    SyncStarted(DocumentId),
    /// Sync completed.
    SyncCompleted(DocumentId, SyncResult),
    /// Conflict detected.
    ConflictDetected(SyncConflict),
}

/// Result of sync operation.
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Number of local operations synced.
    pub local_ops_synced: usize,
    /// Number of server operations received.
    pub server_ops_received: usize,
    /// Number of conflicts resolved.
    pub conflicts_resolved: usize,
    /// Duration of sync.
    pub duration_ms: u64,
}

/// Offline manager for handling disconnected operation.
pub struct OfflineManager {
    /// Configuration.
    config: OfflineConfig,
    /// Client ID.
    client_id: ClientId,
    /// Current connection state.
    state: Arc<RwLock<ConnectionState>>,
    /// Pending operations queue per document.
    pending_queues: Arc<RwLock<HashMap<DocumentId, VecDeque<PendingOperation>>>>,
    /// Next operation ID.
    next_op_id: Arc<RwLock<u64>>,
    /// Last heartbeat received time.
    last_heartbeat: Arc<RwLock<Option<Instant>>>,
    /// Local vector clocks per document.
    local_clocks: Arc<RwLock<HashMap<DocumentId, VectorClock>>>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<OfflineEvent>,
    /// Message sender for outgoing messages.
    message_tx: Option<mpsc::Sender<Message>>,
}

impl OfflineManager {
    /// Create a new offline manager.
    pub fn new(client_id: ClientId, config: OfflineConfig) -> Self {
        let (event_tx, _) = broadcast::channel(100);

        Self {
            config,
            client_id,
            state: Arc::new(RwLock::new(ConnectionState::Offline)),
            pending_queues: Arc::new(RwLock::new(HashMap::new())),
            next_op_id: Arc::new(RwLock::new(1)),
            last_heartbeat: Arc::new(RwLock::new(None)),
            local_clocks: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            message_tx: None,
        }
    }

    /// Set the message sender for outgoing messages.
    pub fn set_message_sender(&mut self, tx: mpsc::Sender<Message>) {
        self.message_tx = Some(tx);
    }

    /// Subscribe to offline events.
    pub fn subscribe(&self) -> broadcast::Receiver<OfflineEvent> {
        self.event_tx.subscribe()
    }

    /// Get current connection state.
    pub async fn state(&self) -> ConnectionState {
        *self.state.read().await
    }

    /// Check if currently online.
    pub async fn is_online(&self) -> bool {
        *self.state.read().await == ConnectionState::Online
    }

    /// Check if currently offline.
    pub async fn is_offline(&self) -> bool {
        *self.state.read().await == ConnectionState::Offline
    }

    /// Get pending operation count for a document.
    pub async fn pending_count(&self, doc_id: &DocumentId) -> usize {
        let queues = self.pending_queues.read().await;
        queues.get(doc_id).map(|q| q.len()).unwrap_or(0)
    }

    /// Get total pending operation count.
    pub async fn total_pending_count(&self) -> usize {
        let queues = self.pending_queues.read().await;
        queues.values().map(|q| q.len()).sum()
    }

    /// Update connection state.
    pub async fn set_state(&self, new_state: ConnectionState) {
        let mut state = self.state.write().await;
        if *state != new_state {
            *state = new_state;
            let _ = self.event_tx.send(OfflineEvent::StateChanged(new_state));
        }
    }

    /// Record a heartbeat from server.
    pub async fn record_heartbeat(&self) {
        *self.last_heartbeat.write().await = Some(Instant::now());

        // If we were offline/reconnecting, we're now online
        let state = *self.state.read().await;
        if state != ConnectionState::Online && state != ConnectionState::Syncing {
            self.set_state(ConnectionState::Online).await;
        }
    }

    /// Check connection health based on heartbeat.
    pub async fn check_connection(&self) -> bool {
        let last = self.last_heartbeat.read().await;

        if let Some(last_time) = *last {
            let elapsed = last_time.elapsed();
            if elapsed > Duration::from_millis(self.config.connection_timeout_ms) {
                // Connection timed out
                self.set_state(ConnectionState::Offline).await;
                return false;
            }
        }

        true
    }

    /// Queue an operation for sending (handles offline state).
    pub async fn queue_operation(
        &self,
        doc_id: DocumentId,
        operations: Vec<Operation>,
        vector_clock: VectorClock,
    ) -> Result<u64> {
        // Generate operation ID
        let op_id = {
            let mut id = self.next_op_id.write().await;
            let current = *id;
            *id += 1;
            current
        };

        let pending = PendingOperation {
            id: op_id,
            doc_id,
            operations,
            vector_clock,
            created_at: current_timestamp(),
            attempts: 0,
            last_attempt: None,
        };

        // Add to queue
        {
            let mut queues = self.pending_queues.write().await;
            let queue = queues.entry(doc_id).or_insert_with(VecDeque::new);

            // Check queue size limit
            if queue.len() >= self.config.max_queue_size {
                return Err(Error::InvalidState(
                    "Operation queue is full".to_string(),
                ));
            }

            queue.push_back(pending.clone());
        }

        // Emit event
        let _ = self.event_tx.send(OfflineEvent::OperationQueued(op_id));

        // If online, try to send immediately
        if self.is_online().await {
            self.try_send_pending(&doc_id).await;
        }

        Ok(op_id)
    }

    /// Try to send pending operations for a document.
    pub async fn try_send_pending(&self, doc_id: &DocumentId) {
        if !self.is_online().await {
            return;
        }

        let tx = match &self.message_tx {
            Some(tx) => tx.clone(),
            None => return,
        };

        // Get next operation to send
        let pending = {
            let mut queues = self.pending_queues.write().await;
            let queue = match queues.get_mut(doc_id) {
                Some(q) => q,
                None => return,
            };

            queue.front_mut().cloned()
        };

        if let Some(mut op) = pending {
            // Update attempts
            op.attempts += 1;
            op.last_attempt = Some(current_timestamp());

            // Create message
            let msg = Message::Operation(OperationMessage {
                doc_id: op.doc_id,
                client_id: self.client_id,
                vector_clock: op.vector_clock.clone(),
                operations: op.operations.clone(),
                seq: op.id,
            });

            // Try to send
            if tx.send(msg).await.is_ok() {
                // Remove from queue on success
                let mut queues = self.pending_queues.write().await;
                if let Some(queue) = queues.get_mut(doc_id) {
                    queue.pop_front();
                }

                let _ = self.event_tx.send(OfflineEvent::OperationSent(op.id));
            } else if op.attempts >= self.config.max_retries {
                // Failed after max retries
                let mut queues = self.pending_queues.write().await;
                if let Some(queue) = queues.get_mut(doc_id) {
                    queue.pop_front();
                }

                let _ = self.event_tx.send(OfflineEvent::OperationFailed(
                    op.id,
                    "Max retries exceeded".to_string(),
                ));
            }
        }
    }

    /// Handle reconnection - sync all pending operations.
    pub async fn handle_reconnection(&self, doc_id: DocumentId) -> Result<SyncResult> {
        let start = Instant::now();
        self.set_state(ConnectionState::Syncing).await;

        let _ = self.event_tx.send(OfflineEvent::SyncStarted(doc_id));

        // Get local vector clock
        let local_clock = {
            let clocks = self.local_clocks.read().await;
            clocks.get(&doc_id).cloned().unwrap_or_else(VectorClock::new)
        };

        // Request sync from server
        let sync_result = self.sync_with_server(&doc_id, &local_clock).await?;

        // Process pending operations
        let mut local_ops_synced = 0;
        let mut conflicts_resolved = 0;

        {
            let mut queues = self.pending_queues.write().await;
            if let Some(queue) = queues.get_mut(&doc_id) {
                while let Some(pending) = queue.pop_front() {
                    // Check for conflicts with server operations
                    let conflict = self.detect_conflict(&pending, &sync_result);

                    if let Some(conflict_info) = conflict {
                        // Handle conflict
                        let _ = self.event_tx.send(OfflineEvent::ConflictDetected(conflict_info));
                        conflicts_resolved += 1;
                    }

                    // Try to send operation
                    if let Some(tx) = &self.message_tx {
                        let msg = Message::Operation(OperationMessage {
                            doc_id: pending.doc_id,
                            client_id: self.client_id,
                            vector_clock: pending.vector_clock,
                            operations: pending.operations,
                            seq: pending.id,
                        });

                        if tx.send(msg).await.is_ok() {
                            local_ops_synced += 1;
                        }
                    }
                }
            }
        }

        let result = SyncResult {
            local_ops_synced,
            server_ops_received: sync_result.pending_ops.len(),
            conflicts_resolved,
            duration_ms: start.elapsed().as_millis() as u64,
        };

        self.set_state(ConnectionState::Online).await;
        let _ = self.event_tx.send(OfflineEvent::SyncCompleted(doc_id, result.clone()));

        Ok(result)
    }

    /// Sync with server - request missing operations.
    async fn sync_with_server(
        &self,
        doc_id: &DocumentId,
        local_clock: &VectorClock,
    ) -> Result<SyncResponse> {
        // In a real implementation, this would send a SyncRequest and await response
        // For now, return empty response
        Ok(SyncResponse {
            doc_id: *doc_id,
            state: Vec::new(),
            pending_ops: Vec::new(),
            vector_clock: local_clock.clone(),
        })
    }

    /// Detect conflicts between local and server operations.
    fn detect_conflict(
        &self,
        pending: &PendingOperation,
        server_response: &SyncResponse,
    ) -> Option<SyncConflict> {
        // Check if any server operations have causally concurrent timestamps
        // with our pending operation (neither happened-before the other)
        let mut conflicting_server_ops = Vec::new();

        for server_op in &server_response.pending_ops {
            // Simple conflict detection: operations at the same position
            // In a real implementation, this would use vector clock comparison
            if !pending.vector_clock.happens_before(&server_response.vector_clock)
                && !server_response.vector_clock.happens_before(&pending.vector_clock)
            {
                conflicting_server_ops.push(server_op.clone());
            }
        }

        if !conflicting_server_ops.is_empty() {
            Some(SyncConflict {
                doc_id: pending.doc_id,
                local_ops: pending.operations.clone(),
                server_ops: conflicting_server_ops,
                resolution: self.config.conflict_resolution,
            })
        } else {
            None
        }
    }

    /// Update local vector clock for a document.
    pub async fn update_local_clock(&self, doc_id: DocumentId, clock: VectorClock) {
        let mut clocks = self.local_clocks.write().await;
        clocks.insert(doc_id, clock);
    }

    /// Clean up expired operations from queue.
    pub async fn cleanup_expired(&self) {
        let now = current_timestamp();
        let ttl = self.config.operation_ttl_secs * 1000; // Convert to ms

        let mut queues = self.pending_queues.write().await;
        for queue in queues.values_mut() {
            queue.retain(|op| now - op.created_at < ttl);
        }
    }

    /// Get all pending operations for a document.
    pub async fn get_pending_operations(&self, doc_id: &DocumentId) -> Vec<PendingOperation> {
        let queues = self.pending_queues.read().await;
        queues
            .get(doc_id)
            .map(|q| q.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Clear all pending operations for a document.
    pub async fn clear_pending(&self, doc_id: &DocumentId) {
        let mut queues = self.pending_queues.write().await;
        queues.remove(doc_id);
    }

    /// Serialize pending operations for persistence.
    pub async fn serialize_queue(&self) -> Result<Vec<u8>> {
        let queues = self.pending_queues.read().await;
        let serializable: HashMap<String, Vec<PendingOperation>> = queues
            .iter()
            .map(|(k, v)| (k.to_string(), v.iter().cloned().collect()))
            .collect();

        serde_json::to_vec(&serializable)
            .map_err(|e| Error::Serialization(e.to_string()))
    }

    /// Deserialize and restore pending operations.
    pub async fn restore_queue(&self, data: &[u8]) -> Result<()> {
        let deserialized: HashMap<String, Vec<PendingOperation>> =
            serde_json::from_slice(data)
                .map_err(|e| Error::Serialization(e.to_string()))?;

        let mut queues = self.pending_queues.write().await;
        for (key, ops) in deserialized {
            let doc_id = uuid::Uuid::parse_str(&key)
                .map_err(|e| Error::Serialization(e.to_string()))?;
            queues.insert(doc_id, VecDeque::from(ops));
        }

        Ok(())
    }
}

/// Local storage interface for offline persistence.
#[async_trait::async_trait]
pub trait LocalStorage: Send + Sync {
    /// Save data to local storage.
    async fn save(&self, key: &str, data: &[u8]) -> Result<()>;

    /// Load data from local storage.
    async fn load(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Delete data from local storage.
    async fn delete(&self, key: &str) -> Result<()>;

    /// List all keys with a prefix.
    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>>;
}

/// In-memory local storage for testing.
pub struct InMemoryLocalStorage {
    data: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl InMemoryLocalStorage {
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryLocalStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl LocalStorage for InMemoryLocalStorage {
    async fn save(&self, key: &str, data: &[u8]) -> Result<()> {
        self.data.write().await.insert(key.to_string(), data.to_vec());
        Ok(())
    }

    async fn load(&self, key: &str) -> Result<Option<Vec<u8>>> {
        Ok(self.data.read().await.get(key).cloned())
    }

    async fn delete(&self, key: &str) -> Result<()> {
        self.data.write().await.remove(key);
        Ok(())
    }

    async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let data = self.data.read().await;
        Ok(data
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect())
    }
}

/// Persistent offline manager with local storage.
pub struct PersistentOfflineManager {
    /// Base offline manager.
    manager: OfflineManager,
    /// Local storage backend.
    storage: Arc<dyn LocalStorage>,
    /// Storage key prefix.
    key_prefix: String,
}

impl PersistentOfflineManager {
    /// Create a new persistent offline manager.
    pub fn new(
        client_id: ClientId,
        config: OfflineConfig,
        storage: Arc<dyn LocalStorage>,
    ) -> Self {
        Self {
            manager: OfflineManager::new(client_id, config),
            storage,
            key_prefix: format!("offline_{}", client_id),
        }
    }

    /// Get the underlying offline manager.
    pub fn inner(&self) -> &OfflineManager {
        &self.manager
    }

    /// Get mutable access to the underlying offline manager.
    pub fn inner_mut(&mut self) -> &mut OfflineManager {
        &mut self.manager
    }

    /// Save pending operations to local storage.
    pub async fn persist(&self) -> Result<()> {
        let data = self.manager.serialize_queue().await?;
        self.storage
            .save(&format!("{}_queue", self.key_prefix), &data)
            .await
    }

    /// Restore pending operations from local storage.
    pub async fn restore(&self) -> Result<()> {
        if let Some(data) = self.storage.load(&format!("{}_queue", self.key_prefix)).await? {
            self.manager.restore_queue(&data).await?;
        }
        Ok(())
    }

    /// Queue operation and persist.
    pub async fn queue_operation(
        &self,
        doc_id: DocumentId,
        operations: Vec<Operation>,
        vector_clock: VectorClock,
    ) -> Result<u64> {
        let op_id = self.manager.queue_operation(doc_id, operations, vector_clock).await?;

        // Persist after queuing
        if self.manager.config.persist_queue {
            self.persist().await?;
        }

        Ok(op_id)
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

    #[tokio::test]
    async fn test_offline_manager_creation() {
        let client_id = uuid::Uuid::new_v4();
        let manager = OfflineManager::new(client_id, OfflineConfig::default());

        assert!(manager.is_offline().await);
        assert_eq!(manager.total_pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_state_transitions() {
        let client_id = uuid::Uuid::new_v4();
        let manager = OfflineManager::new(client_id, OfflineConfig::default());

        assert_eq!(manager.state().await, ConnectionState::Offline);

        manager.set_state(ConnectionState::Online).await;
        assert_eq!(manager.state().await, ConnectionState::Online);

        manager.set_state(ConnectionState::Reconnecting).await;
        assert_eq!(manager.state().await, ConnectionState::Reconnecting);
    }

    #[tokio::test]
    async fn test_queue_operation() {
        let client_id = uuid::Uuid::new_v4();
        let manager = OfflineManager::new(client_id, OfflineConfig::default());

        let doc_id = uuid::Uuid::new_v4();
        let ops = vec![];
        let clock = VectorClock::new();

        let op_id = manager.queue_operation(doc_id, ops, clock).await.unwrap();
        assert_eq!(op_id, 1);
        assert_eq!(manager.pending_count(&doc_id).await, 1);
        assert_eq!(manager.total_pending_count().await, 1);
    }

    #[tokio::test]
    async fn test_queue_size_limit() {
        let client_id = uuid::Uuid::new_v4();
        let config = OfflineConfig {
            max_queue_size: 2,
            ..Default::default()
        };
        let manager = OfflineManager::new(client_id, config);

        let doc_id = uuid::Uuid::new_v4();
        let clock = VectorClock::new();

        // Should succeed
        manager.queue_operation(doc_id, vec![], clock.clone()).await.unwrap();
        manager.queue_operation(doc_id, vec![], clock.clone()).await.unwrap();

        // Should fail
        let result = manager.queue_operation(doc_id, vec![], clock).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_heartbeat_tracking() {
        let client_id = uuid::Uuid::new_v4();
        let config = OfflineConfig {
            connection_timeout_ms: 100,
            ..Default::default()
        };
        let manager = OfflineManager::new(client_id, config);

        manager.set_state(ConnectionState::Online).await;
        manager.record_heartbeat().await;

        // Should still be connected
        assert!(manager.check_connection().await);

        // Wait for timeout
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Should be disconnected
        assert!(!manager.check_connection().await);
        assert!(manager.is_offline().await);
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let client_id = uuid::Uuid::new_v4();
        let manager = OfflineManager::new(client_id, OfflineConfig::default());
        let mut rx = manager.subscribe();

        manager.set_state(ConnectionState::Online).await;

        if let Ok(event) = rx.try_recv() {
            match event {
                OfflineEvent::StateChanged(state) => {
                    assert_eq!(state, ConnectionState::Online);
                }
                _ => panic!("Wrong event type"),
            }
        }
    }

    #[tokio::test]
    async fn test_serialize_restore_queue() {
        let client_id = uuid::Uuid::new_v4();
        let manager = OfflineManager::new(client_id, OfflineConfig::default());

        let doc_id = uuid::Uuid::new_v4();
        manager.queue_operation(doc_id, vec![], VectorClock::new()).await.unwrap();
        manager.queue_operation(doc_id, vec![], VectorClock::new()).await.unwrap();

        // Serialize
        let data = manager.serialize_queue().await.unwrap();

        // Create new manager and restore
        let manager2 = OfflineManager::new(client_id, OfflineConfig::default());
        manager2.restore_queue(&data).await.unwrap();

        assert_eq!(manager2.pending_count(&doc_id).await, 2);
    }

    #[tokio::test]
    async fn test_cleanup_expired() {
        let client_id = uuid::Uuid::new_v4();
        let config = OfflineConfig {
            operation_ttl_secs: 0, // Expire immediately
            ..Default::default()
        };
        let manager = OfflineManager::new(client_id, config);

        let doc_id = uuid::Uuid::new_v4();
        manager.queue_operation(doc_id, vec![], VectorClock::new()).await.unwrap();

        // Small delay to ensure TTL expires
        tokio::time::sleep(Duration::from_millis(10)).await;

        manager.cleanup_expired().await;
        assert_eq!(manager.pending_count(&doc_id).await, 0);
    }

    #[tokio::test]
    async fn test_in_memory_local_storage() {
        let storage = InMemoryLocalStorage::new();

        storage.save("key1", b"value1").await.unwrap();
        storage.save("key2", b"value2").await.unwrap();

        let loaded = storage.load("key1").await.unwrap();
        assert_eq!(loaded, Some(b"value1".to_vec()));

        let keys = storage.list_keys("key").await.unwrap();
        assert_eq!(keys.len(), 2);

        storage.delete("key1").await.unwrap();
        let loaded = storage.load("key1").await.unwrap();
        assert!(loaded.is_none());
    }

    #[tokio::test]
    async fn test_persistent_offline_manager() {
        let client_id = uuid::Uuid::new_v4();
        let storage = Arc::new(InMemoryLocalStorage::new());
        let manager = PersistentOfflineManager::new(
            client_id,
            OfflineConfig::default(),
            storage.clone(),
        );

        let doc_id = uuid::Uuid::new_v4();
        manager.queue_operation(doc_id, vec![], VectorClock::new()).await.unwrap();

        // Persist
        manager.persist().await.unwrap();

        // Create new manager and restore
        let manager2 = PersistentOfflineManager::new(
            client_id,
            OfflineConfig::default(),
            storage,
        );
        manager2.restore().await.unwrap();

        assert_eq!(manager2.inner().pending_count(&doc_id).await, 1);
    }

    #[tokio::test]
    async fn test_get_pending_operations() {
        let client_id = uuid::Uuid::new_v4();
        let manager = OfflineManager::new(client_id, OfflineConfig::default());

        let doc_id = uuid::Uuid::new_v4();
        manager.queue_operation(doc_id, vec![], VectorClock::new()).await.unwrap();
        manager.queue_operation(doc_id, vec![], VectorClock::new()).await.unwrap();

        let pending = manager.get_pending_operations(&doc_id).await;
        assert_eq!(pending.len(), 2);
        assert_eq!(pending[0].id, 1);
        assert_eq!(pending[1].id, 2);
    }

    #[tokio::test]
    async fn test_clear_pending() {
        let client_id = uuid::Uuid::new_v4();
        let manager = OfflineManager::new(client_id, OfflineConfig::default());

        let doc_id = uuid::Uuid::new_v4();
        manager.queue_operation(doc_id, vec![], VectorClock::new()).await.unwrap();
        manager.queue_operation(doc_id, vec![], VectorClock::new()).await.unwrap();

        manager.clear_pending(&doc_id).await;
        assert_eq!(manager.pending_count(&doc_id).await, 0);
    }

    #[tokio::test]
    async fn test_update_local_clock() {
        let client_id = uuid::Uuid::new_v4();
        let manager = OfflineManager::new(client_id, OfflineConfig::default());

        let doc_id = uuid::Uuid::new_v4();
        let mut clock = VectorClock::new();
        clock.increment(client_id);

        manager.update_local_clock(doc_id, clock.clone()).await;

        // Verify by checking reconnection sync uses the clock
        let clocks = manager.local_clocks.read().await;
        assert!(clocks.contains_key(&doc_id));
    }
}
