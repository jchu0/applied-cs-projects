//! Performance optimization module.
//!
//! Provides operation batching, memory optimization, tombstone garbage collection,
//! and version history management.

use crate::crdt::{Operation, PositionId, VectorClock};
use crate::document::DocumentSnapshot;
use crate::storage::{Checkpoint, OperationEntry};
use crate::{ClientId, DocumentId};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// =============================================================================
// Operation Batching
// =============================================================================

/// Configuration for operation batching.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size before flushing.
    pub max_batch_size: usize,
    /// Maximum time to hold operations before flushing.
    pub max_batch_delay_ms: u64,
    /// Enable adaptive batching based on load.
    pub adaptive: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 50,
            max_batch_delay_ms: 100,
            adaptive: true,
        }
    }
}

/// Operation batcher for reducing network and storage overhead.
pub struct OperationBatcher {
    /// Configuration.
    config: BatchConfig,
    /// Pending operations per document.
    pending: RwLock<HashMap<DocumentId, PendingBatch>>,
    /// Statistics.
    stats: BatcherStats,
}

/// Pending batch of operations.
struct PendingBatch {
    /// Operations waiting to be flushed.
    operations: Vec<Operation>,
    /// First operation timestamp.
    first_op_time: Instant,
    /// Vector clock.
    vector_clock: VectorClock,
}

/// Batching statistics.
#[derive(Default)]
pub struct BatcherStats {
    /// Total operations batched.
    operations_batched: AtomicU64,
    /// Total batches flushed.
    batches_flushed: AtomicU64,
    /// Average batch size.
    avg_batch_size: AtomicU64,
}

impl OperationBatcher {
    /// Create new batcher.
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            pending: RwLock::new(HashMap::new()),
            stats: BatcherStats::default(),
        }
    }

    /// Add operation to batch.
    /// Returns Some(batch) if batch should be flushed.
    pub fn add_operation(
        &self,
        doc_id: DocumentId,
        operation: Operation,
        vector_clock: VectorClock,
    ) -> Option<Vec<Operation>> {
        let mut pending = self.pending.write();

        let batch = pending.entry(doc_id).or_insert_with(|| PendingBatch {
            operations: Vec::new(),
            first_op_time: Instant::now(),
            vector_clock: vector_clock.clone(),
        });

        batch.operations.push(operation);
        batch.vector_clock = vector_clock;

        self.stats.operations_batched.fetch_add(1, Ordering::Relaxed);

        // Check if should flush
        if self.should_flush(batch) {
            let ops = std::mem::take(&mut batch.operations);
            pending.remove(&doc_id);
            self.stats.batches_flushed.fetch_add(1, Ordering::Relaxed);
            return Some(ops);
        }

        None
    }

    /// Force flush all pending operations for a document.
    pub fn flush(&self, doc_id: &DocumentId) -> Option<Vec<Operation>> {
        let mut pending = self.pending.write();

        if let Some(batch) = pending.remove(doc_id) {
            if !batch.operations.is_empty() {
                self.stats.batches_flushed.fetch_add(1, Ordering::Relaxed);
                return Some(batch.operations);
            }
        }

        None
    }

    /// Flush all expired batches.
    pub fn flush_expired(&self) -> Vec<(DocumentId, Vec<Operation>)> {
        let mut pending = self.pending.write();
        let max_delay = Duration::from_millis(self.config.max_batch_delay_ms);
        let mut flushed = Vec::new();

        let mut to_remove = Vec::new();
        for (doc_id, batch) in pending.iter() {
            if batch.first_op_time.elapsed() >= max_delay && !batch.operations.is_empty() {
                to_remove.push(*doc_id);
            }
        }

        for doc_id in to_remove {
            if let Some(batch) = pending.remove(&doc_id) {
                flushed.push((doc_id, batch.operations));
                self.stats.batches_flushed.fetch_add(1, Ordering::Relaxed);
            }
        }

        flushed
    }

    /// Check if batch should be flushed.
    fn should_flush(&self, batch: &PendingBatch) -> bool {
        // Size threshold
        if batch.operations.len() >= self.config.max_batch_size {
            return true;
        }

        // Time threshold
        if batch.first_op_time.elapsed() >= Duration::from_millis(self.config.max_batch_delay_ms) {
            return true;
        }

        false
    }

    /// Get statistics.
    pub fn get_stats(&self) -> (u64, u64, f64) {
        let ops = self.stats.operations_batched.load(Ordering::Relaxed);
        let batches = self.stats.batches_flushed.load(Ordering::Relaxed);
        let avg = if batches > 0 {
            ops as f64 / batches as f64
        } else {
            0.0
        };
        (ops, batches, avg)
    }
}

// =============================================================================
// Tombstone Garbage Collection
// =============================================================================

/// Configuration for garbage collection.
#[derive(Debug, Clone)]
pub struct GCConfig {
    /// Minimum age of tombstones before collection (milliseconds).
    pub min_tombstone_age_ms: u64,
    /// Maximum tombstone ratio before triggering GC.
    pub max_tombstone_ratio: f64,
    /// Minimum operations between GC runs.
    pub min_operations_between_gc: usize,
    /// Enable automatic GC.
    pub auto_gc: bool,
}

impl Default for GCConfig {
    fn default() -> Self {
        Self {
            min_tombstone_age_ms: 60 * 60 * 1000, // 1 hour
            max_tombstone_ratio: 0.5,              // 50% tombstones
            min_operations_between_gc: 1000,
            auto_gc: true,
        }
    }
}

/// Tombstone garbage collector.
pub struct TombstoneGC {
    /// Configuration.
    config: GCConfig,
    /// Operations since last GC per document.
    ops_since_gc: RwLock<HashMap<DocumentId, usize>>,
    /// Statistics.
    stats: GCStats,
}

/// GC statistics.
#[derive(Default)]
pub struct GCStats {
    /// Total tombstones collected.
    tombstones_collected: AtomicU64,
    /// Total GC runs.
    gc_runs: AtomicU64,
    /// Bytes freed.
    bytes_freed: AtomicU64,
}

impl TombstoneGC {
    /// Create new garbage collector.
    pub fn new(config: GCConfig) -> Self {
        Self {
            config,
            ops_since_gc: RwLock::new(HashMap::new()),
            stats: GCStats::default(),
        }
    }

    /// Record operation for tracking.
    pub fn record_operation(&self, doc_id: DocumentId) {
        let mut ops = self.ops_since_gc.write();
        *ops.entry(doc_id).or_insert(0) += 1;
    }

    /// Check if GC should run for a document.
    pub fn should_run_gc(&self, doc_id: &DocumentId, tombstone_count: usize, total_count: usize) -> bool {
        if !self.config.auto_gc {
            return false;
        }

        let ops = self.ops_since_gc.read();
        let ops_since = ops.get(doc_id).copied().unwrap_or(0);

        // Check operation threshold
        if ops_since < self.config.min_operations_between_gc {
            return false;
        }

        // Check tombstone ratio
        if total_count > 0 {
            let ratio = tombstone_count as f64 / total_count as f64;
            if ratio >= self.config.max_tombstone_ratio {
                return true;
            }
        }

        false
    }

    /// Collect tombstones from a document.
    /// Returns list of position IDs that can be removed.
    ///
    /// Parameters:
    /// - doc_id: Document identifier
    /// - tombstones: List of (position_id, timestamp) for deleted elements
    /// - doc_clock: Document's vector clock
    /// - all_clocks: Vector clocks from all connected clients
    pub fn collect_tombstones(
        &self,
        doc_id: &DocumentId,
        tombstones: &[(PositionId, u64)],
        doc_clock: &VectorClock,
        all_clocks: &[VectorClock],
    ) -> Vec<PositionId> {
        let mut to_remove = Vec::new();
        let now = current_timestamp();

        // Compute minimum vector clock (intersection of all clocks)
        let min_clock = compute_minimum_clock(all_clocks);

        // Find tombstones that are:
        // 1. Old enough
        // 2. Seen by all clients (causally stable)
        for (pos_id, timestamp) in tombstones {
            // Check if tombstone is old enough
            let age = now.saturating_sub(*timestamp);
            if age < self.config.min_tombstone_age_ms {
                continue;
            }

            // Check if causally stable (all clients have seen this)
            if min_clock.dominates(doc_clock) {
                to_remove.push(pos_id.clone());
            }
        }

        if !to_remove.is_empty() {
            self.stats.tombstones_collected.fetch_add(to_remove.len() as u64, Ordering::Relaxed);
            self.stats.gc_runs.fetch_add(1, Ordering::Relaxed);

            // Reset operation counter
            self.ops_since_gc.write().insert(*doc_id, 0);
        }

        to_remove
    }

    /// Get statistics.
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.stats.tombstones_collected.load(Ordering::Relaxed),
            self.stats.gc_runs.load(Ordering::Relaxed),
            self.stats.bytes_freed.load(Ordering::Relaxed),
        )
    }
}

/// Compute minimum vector clock (pointwise min across all clocks).
fn compute_minimum_clock(clocks: &[VectorClock]) -> VectorClock {
    if clocks.is_empty() {
        return VectorClock::new();
    }

    let mut result = clocks[0].clone();
    for clock in &clocks[1..] {
        let mut min_clock = VectorClock::new();
        // For each client in result, take the min with the other clock.
        // Clients not in the other clock have implicit timestamp 0,
        // so they drop out of the minimum.
        for client_id in result.clients() {
            let t1 = result.get(&client_id);
            let t2 = clock.get(&client_id);
            let min_t = t1.min(t2);
            if min_t > 0 {
                min_clock.set(client_id, min_t);
            }
        }
        result = min_clock;
    }
    result
}

// =============================================================================
// Version History
// =============================================================================

/// Version information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Version {
    /// Version ID.
    pub id: uuid::Uuid,
    /// Document ID.
    pub doc_id: DocumentId,
    /// Version number.
    pub version_number: u64,
    /// Created at timestamp.
    pub created_at: u64,
    /// Created by user.
    pub created_by: ClientId,
    /// Label/name for the version.
    pub label: Option<String>,
    /// Vector clock at this version.
    pub vector_clock: VectorClock,
    /// Sequence number in operation log.
    pub sequence_number: u64,
    /// Document snapshot at this version.
    pub snapshot: Option<DocumentSnapshot>,
}

/// Version history manager.
pub struct VersionHistory {
    /// Versions per document.
    versions: RwLock<HashMap<DocumentId, Vec<Version>>>,
    /// Auto-version interval (operations).
    auto_version_interval: usize,
    /// Operations since last auto-version.
    ops_since_version: RwLock<HashMap<DocumentId, usize>>,
    /// Maximum versions to retain.
    max_versions: usize,
}

impl VersionHistory {
    /// Create new version history manager.
    pub fn new(auto_version_interval: usize, max_versions: usize) -> Self {
        Self {
            versions: RwLock::new(HashMap::new()),
            auto_version_interval,
            ops_since_version: RwLock::new(HashMap::new()),
            max_versions,
        }
    }

    /// Create a named version.
    pub fn create_version(
        &self,
        doc_id: DocumentId,
        label: Option<String>,
        created_by: ClientId,
        vector_clock: VectorClock,
        sequence_number: u64,
        snapshot: Option<DocumentSnapshot>,
    ) -> Version {
        let mut versions = self.versions.write();
        let doc_versions = versions.entry(doc_id).or_insert_with(Vec::new);

        let version_number = doc_versions.len() as u64 + 1;

        let version = Version {
            id: uuid::Uuid::new_v4(),
            doc_id,
            version_number,
            created_at: current_timestamp(),
            created_by,
            label,
            vector_clock,
            sequence_number,
            snapshot,
        };

        doc_versions.push(version.clone());

        // Prune old versions if needed
        while doc_versions.len() > self.max_versions {
            // Keep first and last, remove middle
            if doc_versions.len() > 2 {
                doc_versions.remove(1);
            } else {
                break;
            }
        }

        // Reset operation counter
        self.ops_since_version.write().insert(doc_id, 0);

        version
    }

    /// Record operation for auto-versioning.
    pub fn record_operation(&self, doc_id: DocumentId) -> bool {
        let mut ops = self.ops_since_version.write();
        let count = ops.entry(doc_id).or_insert(0);
        *count += 1;

        if *count >= self.auto_version_interval {
            return true; // Should create auto-version
        }

        false
    }

    /// Get all versions for a document.
    pub fn get_versions(&self, doc_id: &DocumentId) -> Vec<Version> {
        self.versions
            .read()
            .get(doc_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get a specific version.
    pub fn get_version(&self, doc_id: &DocumentId, version_number: u64) -> Option<Version> {
        self.versions
            .read()
            .get(doc_id)?
            .iter()
            .find(|v| v.version_number == version_number)
            .cloned()
    }

    /// Get version by ID.
    pub fn get_version_by_id(&self, doc_id: &DocumentId, version_id: uuid::Uuid) -> Option<Version> {
        self.versions
            .read()
            .get(doc_id)?
            .iter()
            .find(|v| v.id == version_id)
            .cloned()
    }

    /// Get latest version.
    pub fn get_latest_version(&self, doc_id: &DocumentId) -> Option<Version> {
        self.versions
            .read()
            .get(doc_id)?
            .last()
            .cloned()
    }

    /// Compare two versions.
    pub fn compare_versions(
        &self,
        doc_id: &DocumentId,
        version1: u64,
        version2: u64,
    ) -> Option<VersionComparison> {
        let versions = self.versions.read();
        let doc_versions = versions.get(doc_id)?;

        let v1 = doc_versions.iter().find(|v| v.version_number == version1)?;
        let v2 = doc_versions.iter().find(|v| v.version_number == version2)?;

        Some(VersionComparison {
            from_version: version1,
            to_version: version2,
            from_timestamp: v1.created_at,
            to_timestamp: v2.created_at,
            from_seq: v1.sequence_number,
            to_seq: v2.sequence_number,
            operations_between: v2.sequence_number.saturating_sub(v1.sequence_number),
        })
    }

    /// Delete a version.
    pub fn delete_version(&self, doc_id: &DocumentId, version_number: u64) -> bool {
        let mut versions = self.versions.write();
        if let Some(doc_versions) = versions.get_mut(doc_id) {
            let original_len = doc_versions.len();
            doc_versions.retain(|v| v.version_number != version_number);
            return doc_versions.len() < original_len;
        }
        false
    }
}

/// Comparison between two versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionComparison {
    /// Source version number.
    pub from_version: u64,
    /// Target version number.
    pub to_version: u64,
    /// Source timestamp.
    pub from_timestamp: u64,
    /// Target timestamp.
    pub to_timestamp: u64,
    /// Source sequence number.
    pub from_seq: u64,
    /// Target sequence number.
    pub to_seq: u64,
    /// Operations between versions.
    pub operations_between: u64,
}

// =============================================================================
// Log Compaction
// =============================================================================

/// Log compaction manager.
pub struct LogCompactor {
    /// Minimum log entries before compaction.
    min_entries: usize,
    /// Maximum log entries to keep after compaction.
    max_entries_after: usize,
    /// Compaction statistics.
    stats: CompactionStats,
}

/// Compaction statistics.
#[derive(Default)]
pub struct CompactionStats {
    /// Total compactions performed.
    compactions: AtomicU64,
    /// Entries removed.
    entries_removed: AtomicU64,
    /// Checkpoints created.
    checkpoints_created: AtomicU64,
}

impl LogCompactor {
    /// Create new compactor.
    pub fn new(min_entries: usize, max_entries_after: usize) -> Self {
        Self {
            min_entries,
            max_entries_after,
            stats: CompactionStats::default(),
        }
    }

    /// Check if compaction is needed.
    pub fn needs_compaction(&self, log_size: usize) -> bool {
        log_size >= self.min_entries
    }

    /// Perform compaction.
    /// Returns (checkpoint, entries_to_keep).
    pub fn compact(
        &self,
        doc_id: DocumentId,
        snapshot: DocumentSnapshot,
        entries: &[OperationEntry],
    ) -> (Checkpoint, Vec<OperationEntry>) {
        // Find cutoff point
        let cutoff = entries.len().saturating_sub(self.max_entries_after);

        // Create checkpoint at cutoff
        let checkpoint_entry = &entries[cutoff];
        let checkpoint = Checkpoint {
            id: uuid::Uuid::new_v4(),
            doc_id,
            timestamp: current_timestamp(),
            vector_clock: checkpoint_entry.vector_clock.clone(),
            seq: checkpoint_entry.seq,
            snapshot: snapshot.clone(),
        };

        // Keep entries after cutoff
        let entries_to_keep: Vec<OperationEntry> = entries[cutoff..].to_vec();

        // Update stats
        let removed = cutoff;
        self.stats.compactions.fetch_add(1, Ordering::Relaxed);
        self.stats.entries_removed.fetch_add(removed as u64, Ordering::Relaxed);
        self.stats.checkpoints_created.fetch_add(1, Ordering::Relaxed);

        (checkpoint, entries_to_keep)
    }

    /// Get statistics.
    pub fn get_stats(&self) -> (u64, u64, u64) {
        (
            self.stats.compactions.load(Ordering::Relaxed),
            self.stats.entries_removed.load(Ordering::Relaxed),
            self.stats.checkpoints_created.load(Ordering::Relaxed),
        )
    }
}

// =============================================================================
// Memory Monitor
// =============================================================================

/// Memory usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Estimated document memory (bytes).
    pub document_bytes: u64,
    /// Estimated operation log memory (bytes).
    pub operation_log_bytes: u64,
    /// Estimated snapshot memory (bytes).
    pub snapshot_bytes: u64,
    /// Total tombstones.
    pub tombstone_count: u64,
    /// Active elements.
    pub active_element_count: u64,
}

/// Memory monitor for tracking memory usage.
pub struct MemoryMonitor {
    /// Stats per document.
    stats: RwLock<HashMap<DocumentId, MemoryStats>>,
    /// Memory threshold for alerts.
    alert_threshold_bytes: u64,
}

impl MemoryMonitor {
    /// Create new monitor.
    pub fn new(alert_threshold_bytes: u64) -> Self {
        Self {
            stats: RwLock::new(HashMap::new()),
            alert_threshold_bytes,
        }
    }

    /// Update stats for a document.
    pub fn update_stats(&self, doc_id: DocumentId, stats: MemoryStats) {
        self.stats.write().insert(doc_id, stats);
    }

    /// Get stats for a document.
    pub fn get_stats(&self, doc_id: &DocumentId) -> Option<MemoryStats> {
        self.stats.read().get(doc_id).cloned()
    }

    /// Get total memory usage.
    pub fn total_memory(&self) -> u64 {
        self.stats
            .read()
            .values()
            .map(|s| s.document_bytes + s.operation_log_bytes + s.snapshot_bytes)
            .sum()
    }

    /// Check if any document exceeds threshold.
    pub fn check_alerts(&self) -> Vec<DocumentId> {
        self.stats
            .read()
            .iter()
            .filter(|(_, s)| {
                s.document_bytes + s.operation_log_bytes > self.alert_threshold_bytes
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Estimate memory for document.
    pub fn estimate_document_memory(
        &self,
        element_count: usize,
        tombstone_count: usize,
        avg_element_size: usize,
    ) -> u64 {
        // Estimate: position_id (48 bytes) + element (variable) + overhead
        let per_element = 48 + avg_element_size + 16; // overhead
        ((element_count + tombstone_count) * per_element) as u64
    }

    /// Estimate operation log memory.
    pub fn estimate_log_memory(&self, entry_count: usize, avg_ops_per_entry: usize) -> u64 {
        // Estimate: entry overhead (64 bytes) + operation (~32 bytes each)
        let per_entry = 64 + (avg_ops_per_entry * 32);
        (entry_count * per_entry) as u64
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

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

    #[test]
    fn test_batch_config_default() {
        let config = BatchConfig::default();
        assert_eq!(config.max_batch_size, 50);
        assert_eq!(config.max_batch_delay_ms, 100);
        assert!(config.adaptive);
    }

    #[test]
    fn test_operation_batcher_add() {
        let config = BatchConfig {
            max_batch_size: 3,
            max_batch_delay_ms: 1000,
            adaptive: false,
        };

        let batcher = OperationBatcher::new(config);
        let doc_id = uuid::Uuid::new_v4();

        // Add operations
        let result1 = batcher.add_operation(
            doc_id,
            Operation::Insert {
                id: PositionId::new(1, uuid::Uuid::new_v4(), 0),
                after: PositionId::root(),
                value: 'a',
                attributes: std::collections::HashMap::new(),
            },
            VectorClock::new(),
        );
        assert!(result1.is_none());

        let result2 = batcher.add_operation(
            doc_id,
            Operation::Insert {
                id: PositionId::new(2, uuid::Uuid::new_v4(), 0),
                after: PositionId::root(),
                value: 'b',
                attributes: std::collections::HashMap::new(),
            },
            VectorClock::new(),
        );
        assert!(result2.is_none());

        // Third should trigger flush
        let result3 = batcher.add_operation(
            doc_id,
            Operation::Insert {
                id: PositionId::new(3, uuid::Uuid::new_v4(), 0),
                after: PositionId::root(),
                value: 'c',
                attributes: std::collections::HashMap::new(),
            },
            VectorClock::new(),
        );
        assert!(result3.is_some());
        assert_eq!(result3.unwrap().len(), 3);
    }

    #[test]
    fn test_operation_batcher_flush() {
        let batcher = OperationBatcher::new(BatchConfig::default());
        let doc_id = uuid::Uuid::new_v4();

        // Add operation
        batcher.add_operation(
            doc_id,
            Operation::Insert {
                id: PositionId::new(1, uuid::Uuid::new_v4(), 0),
                after: PositionId::root(),
                value: 'a',
                attributes: std::collections::HashMap::new(),
            },
            VectorClock::new(),
        );

        // Force flush
        let result = batcher.flush(&doc_id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);

        // Second flush should be empty
        let result2 = batcher.flush(&doc_id);
        assert!(result2.is_none());
    }

    #[test]
    fn test_version_history_create() {
        let history = VersionHistory::new(100, 10);
        let doc_id = uuid::Uuid::new_v4();
        let client_id = uuid::Uuid::new_v4();

        let version = history.create_version(
            doc_id,
            Some("Initial".to_string()),
            client_id,
            VectorClock::new(),
            1,
            None,
        );

        assert_eq!(version.version_number, 1);
        assert_eq!(version.label, Some("Initial".to_string()));

        // Create another
        let version2 = history.create_version(
            doc_id,
            Some("Second".to_string()),
            client_id,
            VectorClock::new(),
            10,
            None,
        );

        assert_eq!(version2.version_number, 2);
    }

    #[test]
    fn test_version_history_get_versions() {
        let history = VersionHistory::new(100, 10);
        let doc_id = uuid::Uuid::new_v4();
        let client_id = uuid::Uuid::new_v4();

        // Create versions
        history.create_version(doc_id, Some("v1".to_string()), client_id, VectorClock::new(), 1, None);
        history.create_version(doc_id, Some("v2".to_string()), client_id, VectorClock::new(), 2, None);
        history.create_version(doc_id, Some("v3".to_string()), client_id, VectorClock::new(), 3, None);

        let versions = history.get_versions(&doc_id);
        assert_eq!(versions.len(), 3);
    }

    #[test]
    fn test_version_history_compare() {
        let history = VersionHistory::new(100, 10);
        let doc_id = uuid::Uuid::new_v4();
        let client_id = uuid::Uuid::new_v4();

        history.create_version(doc_id, None, client_id, VectorClock::new(), 1, None);
        history.create_version(doc_id, None, client_id, VectorClock::new(), 50, None);

        let comparison = history.compare_versions(&doc_id, 1, 2);
        assert!(comparison.is_some());
        let comp = comparison.unwrap();
        assert_eq!(comp.from_version, 1);
        assert_eq!(comp.to_version, 2);
        assert_eq!(comp.operations_between, 49);
    }

    #[test]
    fn test_gc_config_default() {
        let config = GCConfig::default();
        assert_eq!(config.min_tombstone_age_ms, 60 * 60 * 1000);
        assert_eq!(config.max_tombstone_ratio, 0.5);
        assert!(config.auto_gc);
    }

    #[test]
    fn test_tombstone_gc_should_run() {
        let gc = TombstoneGC::new(GCConfig {
            min_operations_between_gc: 10,
            max_tombstone_ratio: 0.5,
            ..Default::default()
        });

        let doc_id = uuid::Uuid::new_v4();

        // Record operations
        for _ in 0..15 {
            gc.record_operation(doc_id);
        }

        // Should run with high tombstone ratio
        assert!(gc.should_run_gc(&doc_id, 60, 100));

        // Should not run with low tombstone ratio
        assert!(!gc.should_run_gc(&doc_id, 10, 100));
    }

    #[test]
    fn test_log_compactor_needs_compaction() {
        let compactor = LogCompactor::new(100, 20);

        assert!(!compactor.needs_compaction(50));
        assert!(compactor.needs_compaction(100));
        assert!(compactor.needs_compaction(200));
    }

    #[test]
    fn test_memory_monitor() {
        let monitor = MemoryMonitor::new(1024 * 1024); // 1MB threshold
        let doc_id = uuid::Uuid::new_v4();

        monitor.update_stats(doc_id, MemoryStats {
            document_bytes: 100000,
            operation_log_bytes: 50000,
            snapshot_bytes: 25000,
            tombstone_count: 100,
            active_element_count: 1000,
        });

        let stats = monitor.get_stats(&doc_id);
        assert!(stats.is_some());

        let total = monitor.total_memory();
        assert_eq!(total, 175000);

        // Should not have alerts (below threshold)
        let alerts = monitor.check_alerts();
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_memory_monitor_alerts() {
        let monitor = MemoryMonitor::new(100000); // Low threshold
        let doc_id = uuid::Uuid::new_v4();

        monitor.update_stats(doc_id, MemoryStats {
            document_bytes: 80000,
            operation_log_bytes: 50000,
            snapshot_bytes: 10000,
            tombstone_count: 100,
            active_element_count: 1000,
        });

        // Should have alert (above threshold)
        let alerts = monitor.check_alerts();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0], doc_id);
    }
}
