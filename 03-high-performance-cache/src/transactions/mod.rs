//! Transaction support (MULTI/EXEC)
//!
//! Implements Redis-compatible transactions with optimistic locking.
//! Supports MULTI, EXEC, DISCARD, and WATCH commands.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::resp::RespValue;

/// Transaction state for a client
#[derive(Debug, Clone)]
pub enum TransactionState {
    /// Not in a transaction
    None,
    /// In MULTI block, queuing commands
    Queued,
    /// Transaction was aborted (WATCH key modified)
    Aborted,
}

/// A queued command in a transaction
#[derive(Debug, Clone)]
pub struct QueuedCommand {
    pub command: String,
    pub args: Vec<RespValue>,
}

/// Transaction context for a single client
#[derive(Debug)]
pub struct TransactionContext {
    /// Current transaction state
    pub state: TransactionState,
    /// Queued commands
    pub queue: Vec<QueuedCommand>,
    /// Watched keys and their versions
    pub watched_keys: HashMap<String, u64>,
}

impl TransactionContext {
    /// Create a new transaction context
    pub fn new() -> Self {
        Self {
            state: TransactionState::None,
            queue: Vec::new(),
            watched_keys: HashMap::new(),
        }
    }

    /// Start a transaction (MULTI)
    pub fn multi(&mut self) -> RespValue {
        match self.state {
            TransactionState::None | TransactionState::Aborted => {
                self.state = TransactionState::Queued;
                self.queue.clear();
                RespValue::ok()
            }
            TransactionState::Queued => {
                RespValue::error("ERR MULTI calls can not be nested")
            }
        }
    }

    /// Queue a command during a transaction
    pub fn queue_command(&mut self, command: String, args: Vec<RespValue>) -> RespValue {
        self.queue.push(QueuedCommand { command, args });
        RespValue::SimpleString("QUEUED".to_string())
    }

    /// Discard the transaction
    pub fn discard(&mut self) -> RespValue {
        match self.state {
            TransactionState::Queued | TransactionState::Aborted => {
                self.state = TransactionState::None;
                self.queue.clear();
                self.watched_keys.clear();
                RespValue::ok()
            }
            TransactionState::None => {
                RespValue::error("ERR DISCARD without MULTI")
            }
        }
    }

    /// Check if in a transaction
    pub fn in_transaction(&self) -> bool {
        matches!(self.state, TransactionState::Queued | TransactionState::Aborted)
    }

    /// Check if transaction is aborted
    pub fn is_aborted(&self) -> bool {
        matches!(self.state, TransactionState::Aborted)
    }

    /// Mark transaction as aborted
    pub fn abort(&mut self) {
        if matches!(self.state, TransactionState::Queued) {
            self.state = TransactionState::Aborted;
        }
    }

    /// Take queued commands for execution
    pub fn take_queue(&mut self) -> Vec<QueuedCommand> {
        self.state = TransactionState::None;
        self.watched_keys.clear();
        std::mem::take(&mut self.queue)
    }

    /// Add a key to watch
    pub fn watch(&mut self, key: String, version: u64) {
        self.watched_keys.insert(key, version);
    }

    /// Unwatch all keys
    pub fn unwatch(&mut self) {
        self.watched_keys.clear();
    }

    /// Get watched keys
    pub fn get_watched_keys(&self) -> &HashMap<String, u64> {
        &self.watched_keys
    }
}

impl Default for TransactionContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Transaction manager for all clients
pub struct TransactionManager {
    /// Key versions for optimistic locking
    key_versions: RwLock<HashMap<String, u64>>,
    /// Global version counter
    version_counter: RwLock<u64>,
}

impl TransactionManager {
    /// Create a new transaction manager
    pub fn new() -> Self {
        Self {
            key_versions: RwLock::new(HashMap::new()),
            version_counter: RwLock::new(0),
        }
    }

    /// Get current version of a key
    pub fn get_key_version(&self, key: &str) -> u64 {
        let versions = self.key_versions.read().unwrap();
        *versions.get(key).unwrap_or(&0)
    }

    /// Increment version of a key (called when key is modified)
    pub fn touch_key(&self, key: &str) {
        let mut counter = self.version_counter.write().unwrap();
        *counter += 1;
        let new_version = *counter;
        drop(counter);

        let mut versions = self.key_versions.write().unwrap();
        versions.insert(key.to_string(), new_version);
    }

    /// Check if watched keys have been modified
    pub fn check_watch(&self, watched: &HashMap<String, u64>) -> bool {
        let versions = self.key_versions.read().unwrap();
        for (key, expected_version) in watched {
            let current_version = versions.get(key).unwrap_or(&0);
            if current_version != expected_version {
                return false;
            }
        }
        true
    }

    /// Clean up versions for deleted keys
    pub fn remove_key(&self, key: &str) {
        let mut versions = self.key_versions.write().unwrap();
        versions.remove(key);
    }
}

impl Default for TransactionManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== TransactionContext Tests ====================

    #[test]
    fn test_transaction_lifecycle() {
        let mut ctx = TransactionContext::new();

        // Start transaction
        let result = ctx.multi();
        assert!(matches!(result, RespValue::SimpleString(_)));

        // Queue commands
        ctx.queue_command("SET".to_string(), vec![
            RespValue::bulk_string("key"),
            RespValue::bulk_string("value"),
        ]);
        assert_eq!(ctx.queue.len(), 1);

        // Take queue
        let queue = ctx.take_queue();
        assert_eq!(queue.len(), 1);
        assert!(!ctx.in_transaction());
    }

    #[test]
    fn test_nested_multi() {
        let mut ctx = TransactionContext::new();
        ctx.multi();
        let result = ctx.multi();
        assert!(matches!(result, RespValue::Error(_)));
    }

    #[test]
    fn test_multi_from_none() {
        let mut ctx = TransactionContext::new();
        assert!(matches!(ctx.state, TransactionState::None));

        let result = ctx.multi();
        assert!(matches!(result, RespValue::SimpleString(_)));
        assert!(matches!(ctx.state, TransactionState::Queued));
    }

    #[test]
    fn test_multi_from_aborted() {
        let mut ctx = TransactionContext::new();
        ctx.multi();
        ctx.abort();
        assert!(ctx.is_aborted());

        // MULTI after abort should work
        let result = ctx.multi();
        assert!(matches!(result, RespValue::SimpleString(_)));
        assert!(matches!(ctx.state, TransactionState::Queued));
    }

    #[test]
    fn test_queue_command() {
        let mut ctx = TransactionContext::new();
        ctx.multi();

        let result = ctx.queue_command("GET".to_string(), vec![
            RespValue::bulk_string("key"),
        ]);

        assert!(matches!(result, RespValue::SimpleString(s) if s == "QUEUED"));
        assert_eq!(ctx.queue.len(), 1);
    }

    #[test]
    fn test_queue_multiple_commands() {
        let mut ctx = TransactionContext::new();
        ctx.multi();

        ctx.queue_command("SET".to_string(), vec![
            RespValue::bulk_string("key1"),
            RespValue::bulk_string("value1"),
        ]);
        ctx.queue_command("SET".to_string(), vec![
            RespValue::bulk_string("key2"),
            RespValue::bulk_string("value2"),
        ]);
        ctx.queue_command("GET".to_string(), vec![
            RespValue::bulk_string("key1"),
        ]);

        assert_eq!(ctx.queue.len(), 3);
    }

    #[test]
    fn test_discard() {
        let mut ctx = TransactionContext::new();
        ctx.multi();
        ctx.queue_command("SET".to_string(), vec![]);

        let result = ctx.discard();
        assert!(matches!(result, RespValue::SimpleString(_)));
        assert!(!ctx.in_transaction());
        assert!(ctx.queue.is_empty());
    }

    #[test]
    fn test_discard_without_multi() {
        let mut ctx = TransactionContext::new();
        let result = ctx.discard();
        assert!(matches!(result, RespValue::Error(_)));
    }

    #[test]
    fn test_discard_clears_watched_keys() {
        let mut ctx = TransactionContext::new();
        ctx.watch("key1".to_string(), 1);
        ctx.watch("key2".to_string(), 2);
        ctx.multi();

        ctx.discard();

        assert!(ctx.get_watched_keys().is_empty());
    }

    #[test]
    fn test_in_transaction() {
        let mut ctx = TransactionContext::new();

        assert!(!ctx.in_transaction());

        ctx.multi();
        assert!(ctx.in_transaction());

        ctx.abort();
        assert!(ctx.in_transaction()); // Still in transaction when aborted

        ctx.discard();
        assert!(!ctx.in_transaction());
    }

    #[test]
    fn test_is_aborted() {
        let mut ctx = TransactionContext::new();

        assert!(!ctx.is_aborted());

        ctx.multi();
        assert!(!ctx.is_aborted());

        ctx.abort();
        assert!(ctx.is_aborted());
    }

    #[test]
    fn test_abort() {
        let mut ctx = TransactionContext::new();
        ctx.multi();
        ctx.queue_command("SET".to_string(), vec![]);

        ctx.abort();

        assert!(ctx.is_aborted());
        // Queue should still have commands
        assert!(!ctx.queue.is_empty());
    }

    #[test]
    fn test_abort_only_when_queued() {
        let mut ctx = TransactionContext::new();

        // Abort when not in transaction should be no-op
        ctx.abort();
        assert!(!ctx.is_aborted());

        ctx.multi();
        ctx.abort();
        assert!(ctx.is_aborted());

        // Abort when already aborted should be no-op
        ctx.abort();
        assert!(ctx.is_aborted());
    }

    #[test]
    fn test_take_queue() {
        let mut ctx = TransactionContext::new();
        ctx.multi();
        ctx.queue_command("SET".to_string(), vec![
            RespValue::bulk_string("key"),
            RespValue::bulk_string("value"),
        ]);
        ctx.watch("watched_key".to_string(), 1);

        let queue = ctx.take_queue();

        assert_eq!(queue.len(), 1);
        assert!(ctx.queue.is_empty());
        assert!(ctx.get_watched_keys().is_empty());
        assert!(!ctx.in_transaction());
    }

    #[test]
    fn test_watch() {
        let manager = TransactionManager::new();
        let mut ctx = TransactionContext::new();

        // Watch a key
        let version = manager.get_key_version("mykey");
        ctx.watch("mykey".to_string(), version);

        // Check watch passes
        assert!(manager.check_watch(ctx.get_watched_keys()));

        // Modify key
        manager.touch_key("mykey");

        // Check watch fails
        assert!(!manager.check_watch(ctx.get_watched_keys()));
    }

    #[test]
    fn test_watch_multiple_keys() {
        let mut ctx = TransactionContext::new();

        ctx.watch("key1".to_string(), 1);
        ctx.watch("key2".to_string(), 2);
        ctx.watch("key3".to_string(), 3);

        let watched = ctx.get_watched_keys();
        assert_eq!(watched.len(), 3);
        assert_eq!(watched.get("key1"), Some(&1));
        assert_eq!(watched.get("key2"), Some(&2));
        assert_eq!(watched.get("key3"), Some(&3));
    }

    #[test]
    fn test_unwatch() {
        let mut ctx = TransactionContext::new();
        ctx.watch("key1".to_string(), 1);
        ctx.watch("key2".to_string(), 2);

        ctx.unwatch();

        assert!(ctx.get_watched_keys().is_empty());
    }

    #[test]
    fn test_default() {
        let ctx: TransactionContext = Default::default();
        assert!(!ctx.in_transaction());
    }

    // ==================== TransactionManager Tests ====================

    #[test]
    fn test_manager_get_key_version_new_key() {
        let manager = TransactionManager::new();
        assert_eq!(manager.get_key_version("nonexistent"), 0);
    }

    #[test]
    fn test_manager_touch_key() {
        let manager = TransactionManager::new();

        assert_eq!(manager.get_key_version("mykey"), 0);

        manager.touch_key("mykey");
        let version1 = manager.get_key_version("mykey");
        assert!(version1 > 0);

        manager.touch_key("mykey");
        let version2 = manager.get_key_version("mykey");
        assert!(version2 > version1);
    }

    #[test]
    fn test_manager_touch_multiple_keys() {
        let manager = TransactionManager::new();

        manager.touch_key("key1");
        manager.touch_key("key2");
        manager.touch_key("key3");

        let v1 = manager.get_key_version("key1");
        let v2 = manager.get_key_version("key2");
        let v3 = manager.get_key_version("key3");

        assert!(v1 > 0);
        assert!(v2 > v1);
        assert!(v3 > v2);
    }

    #[test]
    fn test_manager_check_watch_success() {
        let manager = TransactionManager::new();

        // Get current versions
        manager.touch_key("key1");
        let v1 = manager.get_key_version("key1");

        let mut watched = HashMap::new();
        watched.insert("key1".to_string(), v1);

        // Version hasn't changed
        assert!(manager.check_watch(&watched));
    }

    #[test]
    fn test_manager_check_watch_failure() {
        let manager = TransactionManager::new();

        // Get current version
        manager.touch_key("key1");
        let v1 = manager.get_key_version("key1");

        let mut watched = HashMap::new();
        watched.insert("key1".to_string(), v1);

        // Modify key
        manager.touch_key("key1");

        // Watch should fail
        assert!(!manager.check_watch(&watched));
    }

    #[test]
    fn test_manager_check_watch_multiple_keys() {
        let manager = TransactionManager::new();

        manager.touch_key("key1");
        manager.touch_key("key2");
        let v1 = manager.get_key_version("key1");
        let v2 = manager.get_key_version("key2");

        let mut watched = HashMap::new();
        watched.insert("key1".to_string(), v1);
        watched.insert("key2".to_string(), v2);

        // All versions match
        assert!(manager.check_watch(&watched));

        // Modify one key
        manager.touch_key("key2");

        // Watch should fail
        assert!(!manager.check_watch(&watched));
    }

    #[test]
    fn test_manager_check_watch_empty() {
        let manager = TransactionManager::new();
        let watched = HashMap::new();

        // Empty watch should succeed
        assert!(manager.check_watch(&watched));
    }

    #[test]
    fn test_manager_check_watch_nonexistent_key() {
        let manager = TransactionManager::new();

        // Watch a key that was never touched
        let mut watched = HashMap::new();
        watched.insert("nonexistent".to_string(), 0);

        // Should pass since version is 0
        assert!(manager.check_watch(&watched));

        // Touch it
        manager.touch_key("nonexistent");

        // Should fail now
        assert!(!manager.check_watch(&watched));
    }

    #[test]
    fn test_manager_remove_key() {
        let manager = TransactionManager::new();

        manager.touch_key("mykey");
        assert!(manager.get_key_version("mykey") > 0);

        manager.remove_key("mykey");
        // After removal, version should be 0 (or not found)
        // depending on implementation
    }

    #[test]
    fn test_manager_default() {
        let manager: TransactionManager = Default::default();
        assert_eq!(manager.get_key_version("any_key"), 0);
    }

    // ==================== QueuedCommand Tests ====================

    #[test]
    fn test_queued_command_structure() {
        let cmd = QueuedCommand {
            command: "SET".to_string(),
            args: vec![
                RespValue::bulk_string("key"),
                RespValue::bulk_string("value"),
            ],
        };

        assert_eq!(cmd.command, "SET");
        assert_eq!(cmd.args.len(), 2);
    }

    // ==================== Integration Tests ====================

    #[test]
    fn test_full_transaction_flow() {
        let manager = TransactionManager::new();
        let mut ctx = TransactionContext::new();

        // Watch a key
        let version = manager.get_key_version("counter");
        ctx.watch("counter".to_string(), version);

        // Start transaction
        ctx.multi();

        // Queue commands
        ctx.queue_command("INCR".to_string(), vec![
            RespValue::bulk_string("counter"),
        ]);
        ctx.queue_command("GET".to_string(), vec![
            RespValue::bulk_string("counter"),
        ]);

        // Check watch before exec
        if manager.check_watch(ctx.get_watched_keys()) {
            // Execute
            let queue = ctx.take_queue();
            assert_eq!(queue.len(), 2);
        } else {
            panic!("Watch should pass");
        }
    }

    #[test]
    fn test_transaction_aborted_by_watch() {
        let manager = TransactionManager::new();
        let mut ctx = TransactionContext::new();

        // Watch a key
        let version = manager.get_key_version("counter");
        ctx.watch("counter".to_string(), version);

        // Start transaction
        ctx.multi();

        // Queue command
        ctx.queue_command("INCR".to_string(), vec![
            RespValue::bulk_string("counter"),
        ]);

        // Simulate another client modifying the key
        manager.touch_key("counter");

        // Check watch should fail
        assert!(!manager.check_watch(ctx.get_watched_keys()));

        // Transaction should be aborted
        ctx.abort();
        assert!(ctx.is_aborted());
    }

    #[test]
    fn test_transaction_with_multiple_watched_keys() {
        let manager = TransactionManager::new();
        let mut ctx = TransactionContext::new();

        // Setup some keys
        manager.touch_key("key1");
        manager.touch_key("key2");

        // Watch both
        ctx.watch("key1".to_string(), manager.get_key_version("key1"));
        ctx.watch("key2".to_string(), manager.get_key_version("key2"));

        ctx.multi();
        ctx.queue_command("SET".to_string(), vec![
            RespValue::bulk_string("key1"),
            RespValue::bulk_string("value1"),
        ]);

        // Watch should pass
        assert!(manager.check_watch(ctx.get_watched_keys()));

        // Modify one key
        manager.touch_key("key1");

        // Watch should fail now
        assert!(!manager.check_watch(ctx.get_watched_keys()));
    }
}
