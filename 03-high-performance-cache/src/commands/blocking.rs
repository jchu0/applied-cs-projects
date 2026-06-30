//! Blocking list commands (BLPOP, BRPOP, BRPOPLPUSH, BLMOVE)
//!
//! These commands block the client until data is available or timeout occurs.
//! In this implementation, we provide infrastructure for blocking operations
//! that can be integrated with an async event loop.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use crate::resp::RespValue;
use crate::storage::{Database, RedisObject};

/// A blocked client waiting for list data
#[derive(Debug, Clone)]
pub struct BlockedClient {
    /// Client identifier
    pub client_id: u64,
    /// Keys the client is waiting on
    pub keys: Vec<String>,
    /// Direction to pop from (left or right)
    pub direction: PopDirection,
    /// Timeout deadline
    pub deadline: Option<Instant>,
    /// Target key for move operations (BRPOPLPUSH)
    pub target_key: Option<String>,
    /// Target direction for move operations
    pub target_direction: Option<PopDirection>,
}

/// Direction for pop operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PopDirection {
    Left,
    Right,
}

/// Manager for blocked clients
#[derive(Default)]
pub struct BlockingManager {
    /// Clients blocked on keys (key -> list of client_ids)
    blocked_on_key: HashMap<String, VecDeque<u64>>,
    /// Client details (client_id -> BlockedClient)
    blocked_clients: HashMap<u64, BlockedClient>,
    /// Next client ID
    next_client_id: u64,
}

impl BlockingManager {
    /// Create a new blocking manager
    pub fn new() -> Self {
        Self {
            blocked_on_key: HashMap::new(),
            blocked_clients: HashMap::new(),
            next_client_id: 1,
        }
    }

    /// Block a client waiting for data on specified keys
    pub fn block_client(
        &mut self,
        keys: Vec<String>,
        direction: PopDirection,
        timeout_secs: f64,
        target_key: Option<String>,
        target_direction: Option<PopDirection>,
    ) -> u64 {
        let client_id = self.next_client_id;
        self.next_client_id += 1;

        let deadline = if timeout_secs > 0.0 {
            Some(Instant::now() + Duration::from_secs_f64(timeout_secs))
        } else {
            None // Block indefinitely
        };

        let blocked = BlockedClient {
            client_id,
            keys: keys.clone(),
            direction,
            deadline,
            target_key,
            target_direction,
        };

        // Register client for each key
        for key in &keys {
            self.blocked_on_key
                .entry(key.clone())
                .or_default()
                .push_back(client_id);
        }

        self.blocked_clients.insert(client_id, blocked);
        client_id
    }

    /// Unblock a client and remove from tracking
    pub fn unblock_client(&mut self, client_id: u64) -> Option<BlockedClient> {
        if let Some(client) = self.blocked_clients.remove(&client_id) {
            // Remove from key mappings
            for key in &client.keys {
                if let Some(clients) = self.blocked_on_key.get_mut(key) {
                    clients.retain(|&id| id != client_id);
                    if clients.is_empty() {
                        self.blocked_on_key.remove(key);
                    }
                }
            }
            Some(client)
        } else {
            None
        }
    }

    /// Check if any client is blocked on a key
    pub fn has_blocked_clients(&self, key: &str) -> bool {
        self.blocked_on_key
            .get(key)
            .map_or(false, |clients| !clients.is_empty())
    }

    /// Get the first blocked client waiting on a key
    pub fn get_first_blocked(&self, key: &str) -> Option<&BlockedClient> {
        self.blocked_on_key
            .get(key)
            .and_then(|clients| clients.front())
            .and_then(|id| self.blocked_clients.get(id))
    }

    /// Get expired blocked clients
    pub fn get_expired_clients(&self) -> Vec<u64> {
        let now = Instant::now();
        self.blocked_clients
            .iter()
            .filter_map(|(id, client)| {
                client.deadline.filter(|&d| d <= now).map(|_| *id)
            })
            .collect()
    }

    /// Process push event - returns client to wake and the data to send
    pub fn process_push(
        &mut self,
        key: &str,
        db: &mut Database,
    ) -> Option<(u64, String, Vec<u8>)> {
        // Get first blocked client
        let client_id = self.blocked_on_key.get(key)?.front().copied()?;
        let client = self.blocked_clients.get(&client_id)?;

        // Try to pop from the list
        let value = match db.get_mut(key) {
            Some(RedisObject::List(list)) if !list.is_empty() => {
                match client.direction {
                    PopDirection::Left => list.remove(0),
                    PopDirection::Right => list.pop()?,
                }
            }
            _ => return None,
        };

        // Handle BRPOPLPUSH/BLMOVE
        if let (Some(target), Some(target_dir)) = (&client.target_key, client.target_direction) {
            // Push to target list
            let target_key = target.clone();
            if !db.exists(&target_key) {
                db.set(target_key.clone(), RedisObject::List(Vec::new()));
            }
            if let Some(RedisObject::List(target_list)) = db.get_mut(&target_key) {
                match target_dir {
                    PopDirection::Left => target_list.insert(0, value.clone()),
                    PopDirection::Right => target_list.push(value.clone()),
                }
            }
        }

        // Unblock the client
        let _ = self.unblock_client(client_id);

        Some((client_id, key.to_string(), value))
    }

    /// Get number of blocked clients
    pub fn blocked_count(&self) -> usize {
        self.blocked_clients.len()
    }
}

/// Result of a blocking operation
#[derive(Debug)]
pub enum BlockingResult {
    /// Data was immediately available
    Ready(RespValue),
    /// Client should be blocked
    Block(u64),
    /// Error occurred
    Error(RespValue),
}

/// BLPOP key [key ...] timeout
pub fn blpop(args: &[RespValue], db: &mut Database, manager: &mut BlockingManager) -> BlockingResult {
    blocking_pop(args, db, manager, PopDirection::Left)
}

/// BRPOP key [key ...] timeout
pub fn brpop(args: &[RespValue], db: &mut Database, manager: &mut BlockingManager) -> BlockingResult {
    blocking_pop(args, db, manager, PopDirection::Right)
}

/// Generic blocking pop implementation
fn blocking_pop(
    args: &[RespValue],
    db: &mut Database,
    manager: &mut BlockingManager,
    direction: PopDirection,
) -> BlockingResult {
    if args.len() < 2 {
        return BlockingResult::Error(RespValue::error(
            "ERR wrong number of arguments for blocking pop command"
        ));
    }

    // Last argument is timeout
    let timeout = match args.last().and_then(|v| v.as_str()) {
        Some(t) => match t.parse::<f64>() {
            Ok(t) if t >= 0.0 => t,
            _ => return BlockingResult::Error(RespValue::error("ERR timeout is not a float or out of range")),
        },
        None => return BlockingResult::Error(RespValue::error("ERR timeout is not a float or out of range")),
    };

    // Parse keys (all args except last)
    let keys: Vec<String> = args[..args.len() - 1]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    if keys.is_empty() {
        return BlockingResult::Error(RespValue::error("ERR wrong number of arguments"));
    }

    // Check if any key has data immediately available
    for key in &keys {
        if let Some(RedisObject::List(list)) = db.get_mut(key) {
            if !list.is_empty() {
                let value = match direction {
                    PopDirection::Left => list.remove(0),
                    PopDirection::Right => list.pop().unwrap(),
                };
                return BlockingResult::Ready(RespValue::array(vec![
                    RespValue::bulk(key.as_bytes().to_vec()),
                    RespValue::bulk(value),
                ]));
            }
        }
    }

    // No data available - check if timeout is 0 (non-blocking)
    if timeout == 0.0 {
        // Block indefinitely - register the client
        let client_id = manager.block_client(keys, direction, 0.0, None, None);
        return BlockingResult::Block(client_id);
    }

    // Register blocking with timeout
    let client_id = manager.block_client(keys, direction, timeout, None, None);
    BlockingResult::Block(client_id)
}

/// BRPOPLPUSH source destination timeout
pub fn brpoplpush(
    args: &[RespValue],
    db: &mut Database,
    manager: &mut BlockingManager,
) -> BlockingResult {
    if args.len() != 3 {
        return BlockingResult::Error(RespValue::error(
            "ERR wrong number of arguments for 'brpoplpush' command"
        ));
    }

    let source = match args[0].as_str() {
        Some(s) => s.to_string(),
        None => return BlockingResult::Error(RespValue::error("ERR invalid source key")),
    };

    let dest = match args[1].as_str() {
        Some(d) => d.to_string(),
        None => return BlockingResult::Error(RespValue::error("ERR invalid destination key")),
    };

    let timeout = match args[2].as_str() {
        Some(t) => match t.parse::<f64>() {
            Ok(t) if t >= 0.0 => t,
            _ => return BlockingResult::Error(RespValue::error("ERR timeout is not a float or out of range")),
        },
        None => return BlockingResult::Error(RespValue::error("ERR timeout is not a float or out of range")),
    };

    // Check if source has data immediately available
    if let Some(RedisObject::List(list)) = db.get_mut(&source) {
        if !list.is_empty() {
            let value = list.pop().unwrap();

            // Push to destination
            if !db.exists(&dest) {
                db.set(dest.clone(), RedisObject::List(Vec::new()));
            }
            if let Some(RedisObject::List(dest_list)) = db.get_mut(&dest) {
                dest_list.insert(0, value.clone());
            }

            return BlockingResult::Ready(RespValue::bulk(value));
        }
    }

    // Register blocking
    let client_id = manager.block_client(
        vec![source],
        PopDirection::Right,
        timeout,
        Some(dest),
        Some(PopDirection::Left),
    );
    BlockingResult::Block(client_id)
}

/// BLMOVE source destination LEFT|RIGHT LEFT|RIGHT timeout
pub fn blmove(
    args: &[RespValue],
    db: &mut Database,
    manager: &mut BlockingManager,
) -> BlockingResult {
    if args.len() != 5 {
        return BlockingResult::Error(RespValue::error(
            "ERR wrong number of arguments for 'blmove' command"
        ));
    }

    let source = match args[0].as_str() {
        Some(s) => s.to_string(),
        None => return BlockingResult::Error(RespValue::error("ERR invalid source key")),
    };

    let dest = match args[1].as_str() {
        Some(d) => d.to_string(),
        None => return BlockingResult::Error(RespValue::error("ERR invalid destination key")),
    };

    let src_dir = match args[2].as_str().map(|s| s.to_uppercase()).as_deref() {
        Some("LEFT") => PopDirection::Left,
        Some("RIGHT") => PopDirection::Right,
        _ => return BlockingResult::Error(RespValue::error("ERR invalid source direction")),
    };

    let dest_dir = match args[3].as_str().map(|s| s.to_uppercase()).as_deref() {
        Some("LEFT") => PopDirection::Left,
        Some("RIGHT") => PopDirection::Right,
        _ => return BlockingResult::Error(RespValue::error("ERR invalid destination direction")),
    };

    let timeout = match args[4].as_str() {
        Some(t) => match t.parse::<f64>() {
            Ok(t) if t >= 0.0 => t,
            _ => return BlockingResult::Error(RespValue::error("ERR timeout is not a float or out of range")),
        },
        None => return BlockingResult::Error(RespValue::error("ERR timeout is not a float or out of range")),
    };

    // Check if source has data immediately available
    if let Some(RedisObject::List(list)) = db.get_mut(&source) {
        if !list.is_empty() {
            let value = match src_dir {
                PopDirection::Left => list.remove(0),
                PopDirection::Right => list.pop().unwrap(),
            };

            // Push to destination
            if !db.exists(&dest) {
                db.set(dest.clone(), RedisObject::List(Vec::new()));
            }
            if let Some(RedisObject::List(dest_list)) = db.get_mut(&dest) {
                match dest_dir {
                    PopDirection::Left => dest_list.insert(0, value.clone()),
                    PopDirection::Right => dest_list.push(value.clone()),
                }
            }

            return BlockingResult::Ready(RespValue::bulk(value));
        }
    }

    // Register blocking
    let client_id = manager.block_client(
        vec![source],
        src_dir,
        timeout,
        Some(dest),
        Some(dest_dir),
    );
    BlockingResult::Block(client_id)
}

/// LMPOP numkeys key [key ...] LEFT|RIGHT [COUNT count]
pub fn lmpop(args: &[RespValue], db: &mut Database) -> RespValue {
    if args.len() < 3 {
        return RespValue::error("ERR wrong number of arguments for 'lmpop' command");
    }

    let numkeys = match args[0].as_int() {
        Some(n) if n > 0 => n as usize,
        _ => return RespValue::error("ERR numkeys should be greater than 0"),
    };

    if args.len() < numkeys + 2 {
        return RespValue::error("ERR wrong number of arguments for 'lmpop' command");
    }

    // Parse keys
    let keys: Vec<String> = args[1..=numkeys]
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    // Parse direction
    let direction = match args[numkeys + 1].as_str().map(|s| s.to_uppercase()).as_deref() {
        Some("LEFT") => PopDirection::Left,
        Some("RIGHT") => PopDirection::Right,
        _ => return RespValue::error("ERR invalid direction"),
    };

    // Parse optional COUNT
    let count = if args.len() > numkeys + 2 {
        match args[numkeys + 2].as_str().map(|s| s.to_uppercase()).as_deref() {
            Some("COUNT") => {
                if args.len() > numkeys + 3 {
                    match args[numkeys + 3].as_int() {
                        Some(c) if c > 0 => c as usize,
                        _ => return RespValue::error("ERR count should be greater than 0"),
                    }
                } else {
                    1
                }
            }
            _ => 1,
        }
    } else {
        1
    };

    // Find first key with data
    for key in &keys {
        if let Some(RedisObject::List(list)) = db.get_mut(key) {
            if !list.is_empty() {
                let mut popped = Vec::new();
                for _ in 0..count {
                    if list.is_empty() {
                        break;
                    }
                    let value = match direction {
                        PopDirection::Left => list.remove(0),
                        PopDirection::Right => list.pop().unwrap(),
                    };
                    popped.push(RespValue::bulk(value));
                }

                return RespValue::array(vec![
                    RespValue::bulk(key.as_bytes().to_vec()),
                    RespValue::array(popped),
                ]);
            }
        }
    }

    RespValue::null()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocking_manager_new() {
        let manager = BlockingManager::new();
        assert_eq!(manager.blocked_count(), 0);
    }

    #[test]
    fn test_block_client() {
        let mut manager = BlockingManager::new();
        let keys = vec!["key1".to_string(), "key2".to_string()];
        let client_id = manager.block_client(keys, PopDirection::Left, 5.0, None, None);

        assert_eq!(manager.blocked_count(), 1);
        assert!(manager.has_blocked_clients("key1"));
        assert!(manager.has_blocked_clients("key2"));
        assert!(!manager.has_blocked_clients("key3"));
        assert!(client_id > 0);
    }

    #[test]
    fn test_unblock_client() {
        let mut manager = BlockingManager::new();
        let keys = vec!["key1".to_string()];
        let client_id = manager.block_client(keys, PopDirection::Left, 5.0, None, None);

        assert_eq!(manager.blocked_count(), 1);

        let client = manager.unblock_client(client_id);
        assert!(client.is_some());
        assert_eq!(manager.blocked_count(), 0);
        assert!(!manager.has_blocked_clients("key1"));
    }

    #[test]
    fn test_get_expired_clients() {
        let mut manager = BlockingManager::new();

        // Block with very short timeout
        let keys = vec!["key1".to_string()];
        manager.block_client(keys, PopDirection::Left, 0.001, None, None);

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(5));

        let expired = manager.get_expired_clients();
        assert_eq!(expired.len(), 1);
    }

    #[test]
    fn test_blpop_immediate() {
        let mut db = Database::new();
        let mut manager = BlockingManager::new();

        // Add data to list
        db.set("mylist".to_string(), RedisObject::List(vec![b"value1".to_vec(), b"value2".to_vec()]));

        let args = vec![
            RespValue::bulk_string("mylist"),
            RespValue::bulk_string("0"),
        ];

        let result = blpop(&args, &mut db, &mut manager);
        match result {
            BlockingResult::Ready(resp) => {
                assert!(matches!(resp, RespValue::Array(_)));
            }
            _ => panic!("Expected Ready result"),
        }
    }

    #[test]
    fn test_blpop_block() {
        let mut db = Database::new();
        let mut manager = BlockingManager::new();

        let args = vec![
            RespValue::bulk_string("empty_list"),
            RespValue::bulk_string("1"),
        ];

        let result = blpop(&args, &mut db, &mut manager);
        match result {
            BlockingResult::Block(client_id) => {
                assert!(client_id > 0);
                assert!(manager.has_blocked_clients("empty_list"));
            }
            _ => panic!("Expected Block result"),
        }
    }

    #[test]
    fn test_lmpop() {
        let mut db = Database::new();

        // Add data to list
        db.set("list1".to_string(), RedisObject::List(vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]));

        let args = vec![
            RespValue::bulk_string("1"),
            RespValue::bulk_string("list1"),
            RespValue::bulk_string("LEFT"),
            RespValue::bulk_string("COUNT"),
            RespValue::bulk_string("2"),
        ];

        let result = lmpop(&args, &mut db);
        match result {
            RespValue::Array(Some(arr)) => {
                assert_eq!(arr.len(), 2);
            }
            _ => panic!("Expected array result"),
        }
    }

    #[test]
    fn test_process_push() {
        let mut db = Database::new();
        let mut manager = BlockingManager::new();

        // Block a client on a key
        let keys = vec!["mykey".to_string()];
        let client_id = manager.block_client(keys, PopDirection::Left, 5.0, None, None);

        // Add data to the list
        db.set("mykey".to_string(), RedisObject::List(vec![b"pushed_value".to_vec()]));

        // Process push should wake the client
        let result = manager.process_push("mykey", &mut db);
        assert!(result.is_some());

        let (woken_id, key, value) = result.unwrap();
        assert_eq!(woken_id, client_id);
        assert_eq!(key, "mykey");
        assert_eq!(value, b"pushed_value".to_vec());

        // Client should be unblocked
        assert_eq!(manager.blocked_count(), 0);
    }

    #[test]
    fn test_brpoplpush_immediate() {
        let mut db = Database::new();
        let mut manager = BlockingManager::new();

        // Add data to source list
        db.set("source".to_string(), RedisObject::List(vec![b"v1".to_vec(), b"v2".to_vec()]));

        let args = vec![
            RespValue::bulk_string("source"),
            RespValue::bulk_string("dest"),
            RespValue::bulk_string("0"),
        ];

        let result = brpoplpush(&args, &mut db, &mut manager);
        match result {
            BlockingResult::Ready(resp) => {
                assert_eq!(resp, RespValue::BulkString(Some(b"v2".to_vec())));

                // Check dest has the value
                match db.get("dest") {
                    Some(RedisObject::List(list)) => {
                        assert_eq!(list.len(), 1);
                        assert_eq!(list[0], b"v2".to_vec());
                    }
                    _ => panic!("Expected list in dest"),
                }
            }
            _ => panic!("Expected Ready result"),
        }
    }

    #[test]
    fn test_blmove() {
        let mut db = Database::new();
        let mut manager = BlockingManager::new();

        // Add data to source list
        db.set("src".to_string(), RedisObject::List(vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]));

        let args = vec![
            RespValue::bulk_string("src"),
            RespValue::bulk_string("dst"),
            RespValue::bulk_string("RIGHT"),
            RespValue::bulk_string("LEFT"),
            RespValue::bulk_string("0"),
        ];

        let result = blmove(&args, &mut db, &mut manager);
        match result {
            BlockingResult::Ready(resp) => {
                assert_eq!(resp, RespValue::BulkString(Some(b"c".to_vec())));
            }
            _ => panic!("Expected Ready result"),
        }
    }
}
