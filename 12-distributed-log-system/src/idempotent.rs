//! Idempotent producer support for exactly-once semantics.
//!
//! This module provides producer state management and sequence number tracking
//! to enable exactly-once delivery of messages. The broker maintains state for
//! each producer to detect and reject duplicate messages.

use crate::log::{RecordBatch, TopicPartition};
use crate::protocol::error_codes;
use crate::{Error, Offset, Result};

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Producer ID type.
pub type ProducerId = i64;

/// Producer epoch type.
pub type ProducerEpoch = i16;

/// Sequence number type.
pub type SequenceNumber = i32;

/// Configuration for idempotent producer tracking.
#[derive(Debug, Clone)]
pub struct IdempotentConfig {
    /// Maximum number of producers to track per partition.
    pub max_producers_per_partition: usize,
    /// Time after which idle producer state is expired.
    pub producer_state_expiry: Duration,
    /// Number of batches to track for duplicate detection.
    pub max_batches_per_producer: usize,
    /// Enable strict sequence checking.
    pub enable_strict_sequence_check: bool,
}

impl Default for IdempotentConfig {
    fn default() -> Self {
        Self {
            max_producers_per_partition: 1000,
            producer_state_expiry: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            max_batches_per_producer: 5,
            enable_strict_sequence_check: true,
        }
    }
}

/// State for a single producer on a partition.
#[derive(Debug, Clone)]
pub struct ProducerState {
    /// Producer ID.
    pub producer_id: ProducerId,
    /// Current epoch (fencing).
    pub producer_epoch: ProducerEpoch,
    /// Last sequence number seen.
    pub last_sequence: SequenceNumber,
    /// Offset of the last batch.
    pub last_offset: Offset,
    /// Timestamp of last activity.
    pub last_timestamp: Instant,
    /// Recent batch metadata for duplicate detection.
    pub recent_batches: Vec<BatchMetadata>,
}

impl ProducerState {
    /// Create new producer state.
    pub fn new(producer_id: ProducerId, producer_epoch: ProducerEpoch) -> Self {
        Self {
            producer_id,
            producer_epoch,
            last_sequence: -1,
            last_offset: 0,
            last_timestamp: Instant::now(),
            recent_batches: Vec::new(),
        }
    }

    /// Check if this state is newer than another.
    pub fn is_newer_than(&self, other: &ProducerState) -> bool {
        self.producer_epoch > other.producer_epoch
            || (self.producer_epoch == other.producer_epoch
                && self.last_sequence > other.last_sequence)
    }

    /// Update state with a new batch.
    pub fn update(&mut self, first_sequence: SequenceNumber, last_offset: Offset, max_batches: usize) {
        let batch_size = 1; // Simplified
        self.last_sequence = first_sequence + batch_size - 1;
        self.last_offset = last_offset;
        self.last_timestamp = Instant::now();

        // Track recent batch for duplicate detection
        self.recent_batches.push(BatchMetadata {
            first_sequence,
            last_offset,
            timestamp: Instant::now(),
        });

        // Keep only recent batches
        if self.recent_batches.len() > max_batches {
            self.recent_batches.remove(0);
        }
    }

    /// Check if state has expired.
    pub fn is_expired(&self, expiry: Duration) -> bool {
        self.last_timestamp.elapsed() > expiry
    }
}

/// Metadata for a batch (used for duplicate detection).
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    /// First sequence in the batch.
    pub first_sequence: SequenceNumber,
    /// Last offset in the log.
    pub last_offset: Offset,
    /// When this batch was received.
    pub timestamp: Instant,
}

/// Result of sequence validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceCheckResult {
    /// Sequence is valid and new.
    Valid,
    /// This is a duplicate batch (exact match).
    Duplicate { existing_offset: Offset },
    /// Sequence is out of order.
    OutOfOrder { expected: SequenceNumber },
    /// Producer has been fenced (newer epoch exists).
    Fenced { current_epoch: ProducerEpoch },
}

/// Sequence tracker for a single partition.
pub struct SequenceTracker {
    /// Configuration.
    config: IdempotentConfig,
    /// Producer states by producer ID.
    producers: HashMap<ProducerId, ProducerState>,
    /// Last cleanup timestamp.
    last_cleanup: Instant,
}

impl SequenceTracker {
    /// Create a new sequence tracker.
    pub fn new(config: IdempotentConfig) -> Self {
        Self {
            config,
            producers: HashMap::new(),
            last_cleanup: Instant::now(),
        }
    }

    /// Check and update sequence for a batch.
    pub fn check_and_update(
        &mut self,
        producer_id: ProducerId,
        producer_epoch: ProducerEpoch,
        first_sequence: SequenceNumber,
        assigned_offset: Offset,
    ) -> SequenceCheckResult {
        // Periodic cleanup
        self.maybe_cleanup();

        // Get or create producer state
        let state = self.producers.entry(producer_id).or_insert_with(|| {
            ProducerState::new(producer_id, producer_epoch)
        });

        // Check epoch (fencing)
        if producer_epoch < state.producer_epoch {
            return SequenceCheckResult::Fenced {
                current_epoch: state.producer_epoch,
            };
        }

        // New epoch - reset state
        if producer_epoch > state.producer_epoch {
            debug!(
                "Producer {} epoch change: {} -> {}",
                producer_id, state.producer_epoch, producer_epoch
            );
            *state = ProducerState::new(producer_id, producer_epoch);
        }

        // Check for duplicate
        for batch in &state.recent_batches {
            if batch.first_sequence == first_sequence {
                return SequenceCheckResult::Duplicate {
                    existing_offset: batch.last_offset,
                };
            }
        }

        // Check sequence order
        let expected_sequence = state.last_sequence + 1;
        if self.config.enable_strict_sequence_check && first_sequence != expected_sequence {
            // Allow sequence 0 for new producers
            if !(state.last_sequence == -1 && first_sequence == 0) {
                return SequenceCheckResult::OutOfOrder {
                    expected: expected_sequence,
                };
            }
        }

        // Update state
        state.update(first_sequence, assigned_offset, self.config.max_batches_per_producer);

        SequenceCheckResult::Valid
    }

    /// Check sequence without updating (for validation only).
    pub fn check_sequence(
        &self,
        producer_id: ProducerId,
        producer_epoch: ProducerEpoch,
        first_sequence: SequenceNumber,
    ) -> SequenceCheckResult {
        let state = match self.producers.get(&producer_id) {
            Some(s) => s,
            None => {
                // New producer, valid if starting at 0
                if first_sequence == 0 {
                    return SequenceCheckResult::Valid;
                } else {
                    return SequenceCheckResult::OutOfOrder { expected: 0 };
                }
            }
        };

        // Check epoch
        if producer_epoch < state.producer_epoch {
            return SequenceCheckResult::Fenced {
                current_epoch: state.producer_epoch,
            };
        }

        // Check for duplicate
        for batch in &state.recent_batches {
            if batch.first_sequence == first_sequence {
                return SequenceCheckResult::Duplicate {
                    existing_offset: batch.last_offset,
                };
            }
        }

        // Check sequence order
        if producer_epoch > state.producer_epoch {
            // New epoch, expect 0
            if first_sequence != 0 {
                return SequenceCheckResult::OutOfOrder { expected: 0 };
            }
        } else {
            let expected = state.last_sequence + 1;
            if first_sequence != expected {
                return SequenceCheckResult::OutOfOrder { expected };
            }
        }

        SequenceCheckResult::Valid
    }

    /// Get producer state.
    pub fn get_state(&self, producer_id: ProducerId) -> Option<&ProducerState> {
        self.producers.get(&producer_id)
    }

    /// Remove expired producer states.
    fn maybe_cleanup(&mut self) {
        // Cleanup every minute
        if self.last_cleanup.elapsed() < Duration::from_secs(60) {
            return;
        }

        let expiry = self.config.producer_state_expiry;
        let before = self.producers.len();

        self.producers.retain(|_, state| !state.is_expired(expiry));

        let removed = before - self.producers.len();
        if removed > 0 {
            debug!("Cleaned up {} expired producer states", removed);
        }

        self.last_cleanup = Instant::now();
    }

    /// Get number of tracked producers.
    pub fn producer_count(&self) -> usize {
        self.producers.len()
    }
}

/// Producer state manager for the entire broker.
pub struct ProducerStateManager {
    /// Configuration.
    config: IdempotentConfig,
    /// Trackers by partition.
    trackers: RwLock<HashMap<TopicPartition, SequenceTracker>>,
    /// Next producer ID to allocate.
    next_producer_id: AtomicI64,
}

impl ProducerStateManager {
    /// Create a new producer state manager.
    pub fn new(config: IdempotentConfig) -> Self {
        Self {
            config,
            trackers: RwLock::new(HashMap::new()),
            next_producer_id: AtomicI64::new(1),
        }
    }

    /// Allocate a new producer ID.
    pub fn allocate_producer_id(&self) -> (ProducerId, ProducerEpoch) {
        let id = self.next_producer_id.fetch_add(1, Ordering::SeqCst);
        (id, 0)
    }

    /// Check and update sequence for a batch.
    pub fn check_and_update(
        &self,
        tp: &TopicPartition,
        producer_id: ProducerId,
        producer_epoch: ProducerEpoch,
        first_sequence: SequenceNumber,
        assigned_offset: Offset,
    ) -> SequenceCheckResult {
        let mut trackers = self.trackers.write();
        let tracker = trackers
            .entry(tp.clone())
            .or_insert_with(|| SequenceTracker::new(self.config.clone()));

        tracker.check_and_update(producer_id, producer_epoch, first_sequence, assigned_offset)
    }

    /// Check sequence without updating.
    pub fn check_sequence(
        &self,
        tp: &TopicPartition,
        producer_id: ProducerId,
        producer_epoch: ProducerEpoch,
        first_sequence: SequenceNumber,
    ) -> SequenceCheckResult {
        let trackers = self.trackers.read();
        match trackers.get(tp) {
            Some(tracker) => tracker.check_sequence(producer_id, producer_epoch, first_sequence),
            None => {
                // New partition, valid if starting at 0
                if first_sequence == 0 {
                    SequenceCheckResult::Valid
                } else {
                    SequenceCheckResult::OutOfOrder { expected: 0 }
                }
            }
        }
    }

    /// Get producer state for a partition.
    pub fn get_producer_state(
        &self,
        tp: &TopicPartition,
        producer_id: ProducerId,
    ) -> Option<ProducerState> {
        let trackers = self.trackers.read();
        trackers
            .get(tp)
            .and_then(|t| t.get_state(producer_id))
            .cloned()
    }

    /// Get total tracked producer count.
    pub fn total_producer_count(&self) -> usize {
        self.trackers.read().values().map(|t| t.producer_count()).sum()
    }
}

/// Idempotent producer with sequence tracking.
pub struct IdempotentProducer {
    /// Producer ID.
    producer_id: ProducerId,
    /// Producer epoch.
    producer_epoch: ProducerEpoch,
    /// Sequence numbers by partition.
    sequences: RwLock<HashMap<TopicPartition, SequenceNumber>>,
    /// Enable idempotence.
    enabled: bool,
}

impl IdempotentProducer {
    /// Create a new idempotent producer.
    pub fn new(producer_id: ProducerId, producer_epoch: ProducerEpoch, enabled: bool) -> Self {
        Self {
            producer_id,
            producer_epoch,
            sequences: RwLock::new(HashMap::new()),
            enabled,
        }
    }

    /// Create from producer state manager allocation.
    pub fn allocate(manager: &ProducerStateManager, enabled: bool) -> Self {
        let (id, epoch) = if enabled {
            manager.allocate_producer_id()
        } else {
            (-1, -1)
        };
        Self::new(id, epoch, enabled)
    }

    /// Get producer ID.
    pub fn producer_id(&self) -> ProducerId {
        self.producer_id
    }

    /// Get producer epoch.
    pub fn producer_epoch(&self) -> ProducerEpoch {
        self.producer_epoch
    }

    /// Check if idempotence is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get and increment sequence for a partition.
    pub fn next_sequence(&self, tp: &TopicPartition) -> SequenceNumber {
        if !self.enabled {
            return -1;
        }

        let mut sequences = self.sequences.write();
        let seq = sequences.entry(tp.clone()).or_insert(-1);
        *seq += 1;
        *seq
    }

    /// Get current sequence for a partition.
    pub fn current_sequence(&self, tp: &TopicPartition) -> SequenceNumber {
        self.sequences.read().get(tp).copied().unwrap_or(-1)
    }

    /// Reset sequence on error (for retry).
    pub fn reset_sequence(&self, tp: &TopicPartition, sequence: SequenceNumber) {
        if self.enabled {
            self.sequences.write().insert(tp.clone(), sequence - 1);
        }
    }

    /// Prepare a record batch with idempotent fields.
    pub fn prepare_batch(&self, tp: &TopicPartition, batch: &mut RecordBatch) {
        if self.enabled {
            batch.producer_id = self.producer_id;
            batch.producer_epoch = self.producer_epoch;
            batch.base_sequence = self.next_sequence(tp) as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_tracker_valid_sequence() {
        let config = IdempotentConfig::default();
        let mut tracker = SequenceTracker::new(config);

        // First batch at sequence 0
        let result = tracker.check_and_update(1, 0, 0, 100);
        assert_eq!(result, SequenceCheckResult::Valid);

        // Next batch at sequence 1
        let result = tracker.check_and_update(1, 0, 1, 101);
        assert_eq!(result, SequenceCheckResult::Valid);
    }

    #[test]
    fn test_sequence_tracker_duplicate() {
        let config = IdempotentConfig::default();
        let mut tracker = SequenceTracker::new(config);

        // First batch
        let result = tracker.check_and_update(1, 0, 0, 100);
        assert_eq!(result, SequenceCheckResult::Valid);

        // Duplicate batch
        let result = tracker.check_and_update(1, 0, 0, 100);
        assert!(matches!(result, SequenceCheckResult::Duplicate { .. }));
    }

    #[test]
    fn test_sequence_tracker_out_of_order() {
        let config = IdempotentConfig::default();
        let mut tracker = SequenceTracker::new(config);

        // First batch
        tracker.check_and_update(1, 0, 0, 100);

        // Skip sequence 1, try sequence 2
        let result = tracker.check_and_update(1, 0, 2, 102);
        assert!(matches!(result, SequenceCheckResult::OutOfOrder { .. }));
    }

    #[test]
    fn test_sequence_tracker_fencing() {
        let config = IdempotentConfig::default();
        let mut tracker = SequenceTracker::new(config);

        // First batch with epoch 1
        tracker.check_and_update(1, 1, 0, 100);

        // Try with older epoch
        let result = tracker.check_and_update(1, 0, 0, 100);
        assert!(matches!(result, SequenceCheckResult::Fenced { .. }));
    }

    #[test]
    fn test_sequence_tracker_epoch_bump() {
        let config = IdempotentConfig::default();
        let mut tracker = SequenceTracker::new(config);

        // First batch with epoch 0
        tracker.check_and_update(1, 0, 0, 100);
        tracker.check_and_update(1, 0, 1, 101);

        // New epoch resets sequence
        let result = tracker.check_and_update(1, 1, 0, 200);
        assert_eq!(result, SequenceCheckResult::Valid);
    }

    #[test]
    fn test_producer_state_manager() {
        let config = IdempotentConfig::default();
        let manager = ProducerStateManager::new(config);

        let tp = TopicPartition::new("test", 0);

        // Allocate producer ID
        let (id, epoch) = manager.allocate_producer_id();
        assert!(id > 0);
        assert_eq!(epoch, 0);

        // Check and update
        let result = manager.check_and_update(&tp, id, epoch, 0, 100);
        assert_eq!(result, SequenceCheckResult::Valid);
    }

    #[test]
    fn test_idempotent_producer() {
        let config = IdempotentConfig::default();
        let manager = ProducerStateManager::new(config);

        let producer = IdempotentProducer::allocate(&manager, true);
        assert!(producer.is_enabled());
        assert!(producer.producer_id() > 0);

        let tp = TopicPartition::new("test", 0);

        // Get sequences
        assert_eq!(producer.next_sequence(&tp), 0);
        assert_eq!(producer.next_sequence(&tp), 1);
        assert_eq!(producer.current_sequence(&tp), 1);
    }

    #[test]
    fn test_idempotent_producer_disabled() {
        let config = IdempotentConfig::default();
        let manager = ProducerStateManager::new(config);

        let producer = IdempotentProducer::allocate(&manager, false);
        assert!(!producer.is_enabled());
        assert_eq!(producer.producer_id(), -1);

        let tp = TopicPartition::new("test", 0);
        assert_eq!(producer.next_sequence(&tp), -1);
    }

    #[test]
    fn test_multiple_producers() {
        let config = IdempotentConfig::default();
        let mut tracker = SequenceTracker::new(config);

        // Producer 1
        let result = tracker.check_and_update(1, 0, 0, 100);
        assert_eq!(result, SequenceCheckResult::Valid);

        // Producer 2
        let result = tracker.check_and_update(2, 0, 0, 101);
        assert_eq!(result, SequenceCheckResult::Valid);

        // Producer 1 continues
        let result = tracker.check_and_update(1, 0, 1, 102);
        assert_eq!(result, SequenceCheckResult::Valid);

        assert_eq!(tracker.producer_count(), 2);
    }

    #[test]
    fn test_producer_state_expiry() {
        let mut config = IdempotentConfig::default();
        config.producer_state_expiry = Duration::from_millis(1);

        let mut tracker = SequenceTracker::new(config);

        // Add producer
        tracker.check_and_update(1, 0, 0, 100);
        assert_eq!(tracker.producer_count(), 1);

        // Wait for expiry
        std::thread::sleep(Duration::from_millis(10));

        // Force cleanup by updating last_cleanup
        tracker.last_cleanup = Instant::now() - Duration::from_secs(61);
        tracker.maybe_cleanup();

        assert_eq!(tracker.producer_count(), 0);
    }
}
