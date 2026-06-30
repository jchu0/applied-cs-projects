//! Exactly-once semantics with idempotent producers and transactions.

use crate::error::{Error, Result};
use crate::offset::TopicPartition;
use crate::partition::PartitionId;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Producer ID for idempotency.
pub type ProducerId = i64;

/// Producer epoch for fencing.
pub type ProducerEpoch = i16;

/// Sequence number for deduplication.
pub type SequenceNumber = i32;

/// Transaction ID.
pub type TransactionalId = String;

/// Transaction state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionState {
    /// Transaction is empty (no writes yet).
    Empty,
    /// Transaction is ongoing.
    Ongoing,
    /// Transaction is preparing to commit.
    PrepareCommit,
    /// Transaction is preparing to abort.
    PrepareAbort,
    /// Transaction is complete (committed).
    CompleteCommit,
    /// Transaction is complete (aborted).
    CompleteAbort,
    /// Transaction is dead (timed out or fenced).
    Dead,
}

impl TransactionState {
    /// Check if transaction can accept new operations.
    pub fn is_active(&self) -> bool {
        matches!(self, TransactionState::Empty | TransactionState::Ongoing)
    }

    /// Check if transaction is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            TransactionState::CompleteCommit
                | TransactionState::CompleteAbort
                | TransactionState::Dead
        )
    }
}

/// Producer state for idempotency.
#[derive(Debug)]
pub struct ProducerState {
    /// Producer ID.
    producer_id: ProducerId,
    /// Current epoch (for fencing).
    epoch: AtomicU64,
    /// Last sequence number per partition.
    sequences: RwLock<HashMap<TopicPartition, SequenceNumber>>,
    /// Pending batch sequences (not yet committed).
    pending_sequences: RwLock<HashMap<TopicPartition, Vec<SequenceNumber>>>,
    /// Current transaction state.
    transaction_state: RwLock<TransactionState>,
    /// Last update time.
    last_update: RwLock<Instant>,
    /// Transaction timeout.
    transaction_timeout: Duration,
}

impl ProducerState {
    /// Create a new producer state.
    pub fn new(producer_id: ProducerId, epoch: ProducerEpoch, transaction_timeout: Duration) -> Self {
        Self {
            producer_id,
            epoch: AtomicU64::new(epoch as u64),
            sequences: RwLock::new(HashMap::new()),
            pending_sequences: RwLock::new(HashMap::new()),
            transaction_state: RwLock::new(TransactionState::Empty),
            last_update: RwLock::new(Instant::now()),
            transaction_timeout,
        }
    }

    /// Get producer ID.
    pub fn producer_id(&self) -> ProducerId {
        self.producer_id
    }

    /// Get current epoch.
    pub fn epoch(&self) -> ProducerEpoch {
        self.epoch.load(Ordering::Acquire) as ProducerEpoch
    }

    /// Bump epoch (for fencing old producers).
    pub fn bump_epoch(&self) -> ProducerEpoch {
        self.epoch.fetch_add(1, Ordering::AcqRel) as ProducerEpoch + 1
    }

    /// Get last sequence for a partition.
    pub fn last_sequence(&self, tp: &TopicPartition) -> Option<SequenceNumber> {
        self.sequences.read().get(tp).copied()
    }

    /// Check and update sequence number for a batch.
    pub fn check_sequence(
        &self,
        tp: &TopicPartition,
        base_sequence: SequenceNumber,
        batch_size: i32,
    ) -> Result<()> {
        let mut sequences = self.sequences.write();
        let expected = sequences.get(tp).map(|s| s + 1).unwrap_or(0);

        if base_sequence < expected {
            // Duplicate batch
            return Err(Error::DuplicateSequence {
                expected,
                received: base_sequence,
            });
        }

        if base_sequence > expected {
            // Out of order batch
            return Err(Error::OutOfOrderSequence {
                expected,
                received: base_sequence,
            });
        }

        // Update last sequence
        let last_sequence = base_sequence + batch_size - 1;
        sequences.insert(tp.clone(), last_sequence);

        *self.last_update.write() = Instant::now();

        Ok(())
    }

    /// Get transaction state.
    pub fn transaction_state(&self) -> TransactionState {
        *self.transaction_state.read()
    }

    /// Begin a transaction.
    pub fn begin_transaction(&self) -> Result<()> {
        let mut state = self.transaction_state.write();

        if !state.is_active() && !state.is_terminal() {
            return Err(Error::InvalidTransactionState(format!(
                "Cannot begin transaction in state {:?}",
                *state
            )));
        }

        *state = TransactionState::Ongoing;
        *self.last_update.write() = Instant::now();

        Ok(())
    }

    /// Check if transaction has timed out.
    pub fn is_expired(&self) -> bool {
        let last = *self.last_update.read();
        last.elapsed() > self.transaction_timeout
    }

    /// Add partition to transaction.
    pub fn add_partition(&self, tp: TopicPartition) -> Result<()> {
        let state = self.transaction_state.read();

        if !state.is_active() {
            return Err(Error::InvalidTransactionState(format!(
                "Cannot add partition in state {:?}",
                *state
            )));
        }

        self.pending_sequences.write().entry(tp).or_default();
        *self.last_update.write() = Instant::now();

        Ok(())
    }

    /// Prepare to commit.
    pub fn prepare_commit(&self) -> Result<()> {
        let mut state = self.transaction_state.write();

        if *state != TransactionState::Ongoing {
            return Err(Error::InvalidTransactionState(format!(
                "Cannot prepare commit in state {:?}",
                *state
            )));
        }

        *state = TransactionState::PrepareCommit;
        *self.last_update.write() = Instant::now();

        Ok(())
    }

    /// Prepare to abort.
    pub fn prepare_abort(&self) -> Result<()> {
        let mut state = self.transaction_state.write();

        if *state != TransactionState::Ongoing && *state != TransactionState::PrepareCommit {
            return Err(Error::InvalidTransactionState(format!(
                "Cannot prepare abort in state {:?}",
                *state
            )));
        }

        *state = TransactionState::PrepareAbort;
        *self.last_update.write() = Instant::now();

        Ok(())
    }

    /// Complete commit.
    pub fn complete_commit(&self) -> Result<()> {
        let mut state = self.transaction_state.write();

        if *state != TransactionState::PrepareCommit {
            return Err(Error::InvalidTransactionState(format!(
                "Cannot complete commit in state {:?}",
                *state
            )));
        }

        // Clear pending sequences
        self.pending_sequences.write().clear();

        *state = TransactionState::CompleteCommit;
        *self.last_update.write() = Instant::now();

        Ok(())
    }

    /// Complete abort.
    pub fn complete_abort(&self) -> Result<()> {
        let mut state = self.transaction_state.write();

        if *state != TransactionState::PrepareAbort {
            return Err(Error::InvalidTransactionState(format!(
                "Cannot complete abort in state {:?}",
                *state
            )));
        }

        // Rollback sequences
        self.pending_sequences.write().clear();

        *state = TransactionState::CompleteAbort;
        *self.last_update.write() = Instant::now();

        Ok(())
    }

    /// Mark as dead (fenced or expired).
    pub fn mark_dead(&self) {
        *self.transaction_state.write() = TransactionState::Dead;
    }

    /// Reset for new transaction.
    pub fn reset(&self) {
        *self.transaction_state.write() = TransactionState::Empty;
        self.pending_sequences.write().clear();
        *self.last_update.write() = Instant::now();
    }
}

/// Transaction metadata.
#[derive(Debug, Clone)]
pub struct TransactionMetadata {
    /// Transactional ID.
    pub transactional_id: TransactionalId,
    /// Producer ID.
    pub producer_id: ProducerId,
    /// Producer epoch.
    pub producer_epoch: ProducerEpoch,
    /// Transaction timeout.
    pub timeout: Duration,
    /// Current state.
    pub state: TransactionState,
    /// Partitions in this transaction.
    pub partitions: HashSet<TopicPartition>,
    /// Transaction start time.
    pub start_time: Instant,
    /// Last update time.
    pub last_update_time: Instant,
}

impl TransactionMetadata {
    /// Create new transaction metadata.
    pub fn new(
        transactional_id: TransactionalId,
        producer_id: ProducerId,
        producer_epoch: ProducerEpoch,
        timeout: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            transactional_id,
            producer_id,
            producer_epoch,
            timeout,
            state: TransactionState::Empty,
            partitions: HashSet::new(),
            start_time: now,
            last_update_time: now,
        }
    }

    /// Check if transaction has expired.
    pub fn is_expired(&self) -> bool {
        self.last_update_time.elapsed() > self.timeout
    }
}

/// Producer ID manager.
pub struct ProducerIdManager {
    /// Next producer ID to allocate.
    next_id: AtomicI64,
    /// Block size for ID allocation.
    block_size: i64,
    /// Current block end.
    block_end: AtomicI64,
}

impl ProducerIdManager {
    /// Create a new producer ID manager.
    pub fn new(start_id: ProducerId, block_size: i64) -> Self {
        Self {
            next_id: AtomicI64::new(start_id),
            block_size,
            block_end: AtomicI64::new(start_id + block_size),
        }
    }

    /// Allocate a new producer ID.
    pub fn allocate(&self) -> Result<ProducerId> {
        let id = self.next_id.fetch_add(1, Ordering::AcqRel);

        // Check if we need a new block
        if id >= self.block_end.load(Ordering::Acquire) {
            // In a real implementation, would request new block from controller
            self.block_end
                .fetch_add(self.block_size, Ordering::AcqRel);
        }

        Ok(id)
    }

    /// Get next ID (without allocating).
    pub fn peek_next(&self) -> ProducerId {
        self.next_id.load(Ordering::Acquire)
    }
}

/// Transaction coordinator.
pub struct TransactionCoordinator {
    /// Producer states by ID.
    producer_states: RwLock<HashMap<ProducerId, Arc<ProducerState>>>,
    /// Transactional ID to producer ID mapping.
    transactional_ids: RwLock<HashMap<TransactionalId, ProducerId>>,
    /// Producer ID manager.
    id_manager: ProducerIdManager,
    /// Default transaction timeout.
    default_timeout: Duration,
    /// Transaction timeout check interval.
    timeout_check_interval: Duration,
}

impl TransactionCoordinator {
    /// Create a new transaction coordinator.
    pub fn new(start_id: ProducerId, default_timeout: Duration) -> Self {
        Self {
            producer_states: RwLock::new(HashMap::new()),
            transactional_ids: RwLock::new(HashMap::new()),
            id_manager: ProducerIdManager::new(start_id, 1000),
            default_timeout,
            timeout_check_interval: Duration::from_secs(10),
        }
    }

    /// Initialize producer ID (for idempotent or transactional producer).
    pub fn init_producer_id(
        &self,
        transactional_id: Option<&str>,
        transaction_timeout: Option<Duration>,
    ) -> Result<(ProducerId, ProducerEpoch)> {
        let timeout = transaction_timeout.unwrap_or(self.default_timeout);

        if let Some(tid) = transactional_id {
            // Transactional producer
            let mut tid_map = self.transactional_ids.write();
            let mut states = self.producer_states.write();

            if let Some(&existing_pid) = tid_map.get(tid) {
                // Existing transactional ID - bump epoch
                if let Some(state) = states.get(&existing_pid) {
                    let new_epoch = state.bump_epoch();
                    state.reset();
                    return Ok((existing_pid, new_epoch));
                }
            }

            // New transactional ID
            let producer_id = self.id_manager.allocate()?;
            let epoch: ProducerEpoch = 0;

            tid_map.insert(tid.to_string(), producer_id);
            states.insert(
                producer_id,
                Arc::new(ProducerState::new(producer_id, epoch, timeout)),
            );

            Ok((producer_id, epoch))
        } else {
            // Idempotent-only producer
            let producer_id = self.id_manager.allocate()?;
            let epoch: ProducerEpoch = 0;

            self.producer_states.write().insert(
                producer_id,
                Arc::new(ProducerState::new(producer_id, epoch, timeout)),
            );

            Ok((producer_id, epoch))
        }
    }

    /// Get producer state.
    pub fn get_producer_state(&self, producer_id: ProducerId) -> Option<Arc<ProducerState>> {
        self.producer_states.read().get(&producer_id).cloned()
    }

    /// Validate producer epoch (fence old producers).
    pub fn validate_epoch(
        &self,
        producer_id: ProducerId,
        epoch: ProducerEpoch,
    ) -> Result<Arc<ProducerState>> {
        let state = self
            .get_producer_state(producer_id)
            .ok_or(Error::UnknownProducerId(producer_id))?;

        if state.epoch() != epoch {
            return Err(Error::ProducerFenced {
                expected: state.epoch(),
                received: epoch,
            });
        }

        Ok(state)
    }

    /// Begin a transaction.
    pub fn begin_transaction(
        &self,
        producer_id: ProducerId,
        epoch: ProducerEpoch,
    ) -> Result<()> {
        let state = self.validate_epoch(producer_id, epoch)?;
        state.begin_transaction()
    }

    /// Add partition to transaction.
    pub fn add_partition_to_transaction(
        &self,
        producer_id: ProducerId,
        epoch: ProducerEpoch,
        topic: &str,
        partition: PartitionId,
    ) -> Result<()> {
        let state = self.validate_epoch(producer_id, epoch)?;
        let tp = TopicPartition::new(topic.to_string(), partition);
        state.add_partition(tp)
    }

    /// Commit a transaction.
    pub fn commit_transaction(
        &self,
        producer_id: ProducerId,
        epoch: ProducerEpoch,
    ) -> Result<()> {
        let state = self.validate_epoch(producer_id, epoch)?;
        state.prepare_commit()?;
        // In a real implementation, would write commit markers to partitions
        state.complete_commit()
    }

    /// Abort a transaction.
    pub fn abort_transaction(
        &self,
        producer_id: ProducerId,
        epoch: ProducerEpoch,
    ) -> Result<()> {
        let state = self.validate_epoch(producer_id, epoch)?;
        state.prepare_abort()?;
        // In a real implementation, would write abort markers to partitions
        state.complete_abort()
    }

    /// Check and deduplicate a batch.
    pub fn check_sequence(
        &self,
        producer_id: ProducerId,
        epoch: ProducerEpoch,
        topic: &str,
        partition: PartitionId,
        base_sequence: SequenceNumber,
        batch_size: i32,
    ) -> Result<()> {
        let state = self.validate_epoch(producer_id, epoch)?;
        let tp = TopicPartition::new(topic.to_string(), partition);
        state.check_sequence(&tp, base_sequence, batch_size)
    }

    /// Expire old transactions.
    pub fn expire_transactions(&self) {
        let states = self.producer_states.read();

        for state in states.values() {
            if state.is_expired() {
                let txn_state = state.transaction_state();
                if !txn_state.is_terminal() {
                    // Abort expired transaction
                    let _ = state.prepare_abort();
                    let _ = state.complete_abort();
                    tracing::warn!(
                        producer_id = %state.producer_id(),
                        "Expired transaction aborted"
                    );
                }
            }
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> TransactionCoordinatorStats {
        let states = self.producer_states.read();

        let mut active_transactions = 0;
        let mut total_producers = 0;

        for state in states.values() {
            total_producers += 1;
            if state.transaction_state().is_active()
                && state.transaction_state() == TransactionState::Ongoing
            {
                active_transactions += 1;
            }
        }

        TransactionCoordinatorStats {
            total_producers,
            active_transactions,
            transactional_ids: self.transactional_ids.read().len(),
        }
    }
}

/// Transaction coordinator statistics.
#[derive(Debug, Clone)]
pub struct TransactionCoordinatorStats {
    /// Total number of producers.
    pub total_producers: usize,
    /// Active transactions.
    pub active_transactions: usize,
    /// Registered transactional IDs.
    pub transactional_ids: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_producer_state_new() {
        let state = ProducerState::new(1, 0, Duration::from_secs(60));
        assert_eq!(state.producer_id(), 1);
        assert_eq!(state.epoch(), 0);
        assert_eq!(state.transaction_state(), TransactionState::Empty);
    }

    #[test]
    fn test_producer_state_bump_epoch() {
        let state = ProducerState::new(1, 0, Duration::from_secs(60));
        let new_epoch = state.bump_epoch();
        assert_eq!(new_epoch, 1);
        assert_eq!(state.epoch(), 1);
    }

    #[test]
    fn test_producer_state_sequence() {
        let state = ProducerState::new(1, 0, Duration::from_secs(60));
        let tp = TopicPartition::new("test".to_string(), 0);

        // First batch
        state.check_sequence(&tp, 0, 5).unwrap();
        assert_eq!(state.last_sequence(&tp), Some(4));

        // Second batch
        state.check_sequence(&tp, 5, 3).unwrap();
        assert_eq!(state.last_sequence(&tp), Some(7));
    }

    #[test]
    fn test_producer_state_duplicate_sequence() {
        let state = ProducerState::new(1, 0, Duration::from_secs(60));
        let tp = TopicPartition::new("test".to_string(), 0);

        state.check_sequence(&tp, 0, 5).unwrap();
        let result = state.check_sequence(&tp, 0, 5);
        assert!(matches!(result, Err(Error::DuplicateSequence { .. })));
    }

    #[test]
    fn test_producer_state_out_of_order() {
        let state = ProducerState::new(1, 0, Duration::from_secs(60));
        let tp = TopicPartition::new("test".to_string(), 0);

        state.check_sequence(&tp, 0, 5).unwrap();
        let result = state.check_sequence(&tp, 10, 5);
        assert!(matches!(result, Err(Error::OutOfOrderSequence { .. })));
    }

    #[test]
    fn test_producer_state_transaction() {
        let state = ProducerState::new(1, 0, Duration::from_secs(60));

        state.begin_transaction().unwrap();
        assert_eq!(state.transaction_state(), TransactionState::Ongoing);

        let tp = TopicPartition::new("test".to_string(), 0);
        state.add_partition(tp).unwrap();

        state.prepare_commit().unwrap();
        assert_eq!(state.transaction_state(), TransactionState::PrepareCommit);

        state.complete_commit().unwrap();
        assert_eq!(state.transaction_state(), TransactionState::CompleteCommit);
    }

    #[test]
    fn test_producer_state_abort() {
        let state = ProducerState::new(1, 0, Duration::from_secs(60));

        state.begin_transaction().unwrap();
        state.prepare_abort().unwrap();
        assert_eq!(state.transaction_state(), TransactionState::PrepareAbort);

        state.complete_abort().unwrap();
        assert_eq!(state.transaction_state(), TransactionState::CompleteAbort);
    }

    #[test]
    fn test_producer_id_manager() {
        let manager = ProducerIdManager::new(0, 100);

        let id1 = manager.allocate().unwrap();
        let id2 = manager.allocate().unwrap();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_transaction_coordinator_init() {
        let coordinator = TransactionCoordinator::new(0, Duration::from_secs(60));

        let (pid, epoch) = coordinator.init_producer_id(None, None).unwrap();
        assert_eq!(pid, 0);
        assert_eq!(epoch, 0);

        let (pid2, epoch2) = coordinator.init_producer_id(None, None).unwrap();
        assert_eq!(pid2, 1);
        assert_eq!(epoch2, 0);
    }

    #[test]
    fn test_transaction_coordinator_transactional() {
        let coordinator = TransactionCoordinator::new(0, Duration::from_secs(60));

        let (pid1, epoch1) = coordinator
            .init_producer_id(Some("txn-1"), None)
            .unwrap();
        assert_eq!(epoch1, 0);

        // Same transactional ID should bump epoch
        let (pid2, epoch2) = coordinator
            .init_producer_id(Some("txn-1"), None)
            .unwrap();
        assert_eq!(pid1, pid2);
        assert_eq!(epoch2, 1);
    }

    #[test]
    fn test_transaction_coordinator_fence() {
        let coordinator = TransactionCoordinator::new(0, Duration::from_secs(60));

        let (pid, _) = coordinator.init_producer_id(Some("txn-1"), None).unwrap();

        // Bump epoch
        let (_, new_epoch) = coordinator
            .init_producer_id(Some("txn-1"), None)
            .unwrap();

        // Old epoch should be fenced
        let result = coordinator.validate_epoch(pid, 0);
        assert!(matches!(result, Err(Error::ProducerFenced { .. })));

        // New epoch should work
        let result = coordinator.validate_epoch(pid, new_epoch);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transaction_coordinator_begin_commit() {
        let coordinator = TransactionCoordinator::new(0, Duration::from_secs(60));

        let (pid, epoch) = coordinator
            .init_producer_id(Some("txn-1"), None)
            .unwrap();

        coordinator.begin_transaction(pid, epoch).unwrap();

        coordinator
            .add_partition_to_transaction(pid, epoch, "test", 0)
            .unwrap();

        coordinator.commit_transaction(pid, epoch).unwrap();

        let state = coordinator.get_producer_state(pid).unwrap();
        assert_eq!(state.transaction_state(), TransactionState::CompleteCommit);
    }

    #[test]
    fn test_transaction_coordinator_begin_abort() {
        let coordinator = TransactionCoordinator::new(0, Duration::from_secs(60));

        let (pid, epoch) = coordinator
            .init_producer_id(Some("txn-1"), None)
            .unwrap();

        coordinator.begin_transaction(pid, epoch).unwrap();
        coordinator.abort_transaction(pid, epoch).unwrap();

        let state = coordinator.get_producer_state(pid).unwrap();
        assert_eq!(state.transaction_state(), TransactionState::CompleteAbort);
    }

    #[test]
    fn test_transaction_coordinator_dedup() {
        let coordinator = TransactionCoordinator::new(0, Duration::from_secs(60));

        let (pid, epoch) = coordinator.init_producer_id(None, None).unwrap();

        // First batch
        coordinator
            .check_sequence(pid, epoch, "test", 0, 0, 5)
            .unwrap();

        // Duplicate
        let result = coordinator.check_sequence(pid, epoch, "test", 0, 0, 5);
        assert!(matches!(result, Err(Error::DuplicateSequence { .. })));
    }

    #[test]
    fn test_transaction_coordinator_stats() {
        let coordinator = TransactionCoordinator::new(0, Duration::from_secs(60));

        coordinator.init_producer_id(None, None).unwrap();
        let (pid, epoch) = coordinator
            .init_producer_id(Some("txn-1"), None)
            .unwrap();

        coordinator.begin_transaction(pid, epoch).unwrap();

        let stats = coordinator.stats();
        assert_eq!(stats.total_producers, 2);
        assert_eq!(stats.active_transactions, 1);
        assert_eq!(stats.transactional_ids, 1);
    }

    #[test]
    fn test_transaction_state_is_active() {
        assert!(TransactionState::Empty.is_active());
        assert!(TransactionState::Ongoing.is_active());
        assert!(!TransactionState::PrepareCommit.is_active());
        assert!(!TransactionState::CompleteCommit.is_active());
    }

    #[test]
    fn test_transaction_state_is_terminal() {
        assert!(!TransactionState::Empty.is_terminal());
        assert!(!TransactionState::Ongoing.is_terminal());
        assert!(TransactionState::CompleteCommit.is_terminal());
        assert!(TransactionState::CompleteAbort.is_terminal());
        assert!(TransactionState::Dead.is_terminal());
    }

    #[test]
    fn test_transaction_metadata() {
        let meta = TransactionMetadata::new(
            "txn-1".to_string(),
            1,
            0,
            Duration::from_secs(60),
        );

        assert_eq!(meta.transactional_id, "txn-1");
        assert_eq!(meta.producer_id, 1);
        assert_eq!(meta.producer_epoch, 0);
        assert!(!meta.is_expired());
    }
}
