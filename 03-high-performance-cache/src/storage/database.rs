use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::{Dict, RedisObject, StringObject};

/// Redis database
pub struct Database {
    /// Main key-value store
    data: Dict<String, RedisObject>,
    /// Keys with expiration times
    expires: HashMap<String, Instant>,
}

impl Database {
    /// Create a new database
    pub fn new() -> Self {
        Self {
            data: Dict::new(),
            expires: HashMap::new(),
        }
    }

    /// Get a value by key
    pub fn get(&mut self, key: &str) -> Option<&RedisObject> {
        // Check expiration
        if self.is_expired(key) {
            self.delete(key);
            return None;
        }

        self.data.get(&key.to_string())
    }

    /// Get a mutable reference to a value
    pub fn get_mut(&mut self, key: &str) -> Option<&mut RedisObject> {
        // Check expiration
        if self.is_expired(key) {
            self.delete(key);
            return None;
        }

        self.data.get_mut(&key.to_string())
    }

    /// Set a value
    pub fn set(&mut self, key: String, value: RedisObject) -> Option<RedisObject> {
        // Remove any existing expiration
        self.expires.remove(&key);
        self.data.insert(key, value)
    }

    /// Set a string value
    pub fn set_string(&mut self, key: String, value: Vec<u8>) -> Option<RedisObject> {
        let obj = RedisObject::String(StringObject::from_bytes(value));
        self.set(key, obj)
    }

    /// Get a string value
    pub fn get_string(&mut self, key: &str) -> Option<Vec<u8>> {
        match self.get(key)? {
            RedisObject::String(s) => Some(s.as_bytes()),
            _ => None,
        }
    }

    /// Delete a key
    pub fn delete(&mut self, key: &str) -> bool {
        self.expires.remove(key);
        self.data.remove(&key.to_string()).is_some()
    }

    /// Check if key exists
    pub fn exists(&mut self, key: &str) -> bool {
        if self.is_expired(key) {
            self.delete(key);
            return false;
        }
        self.data.contains_key(&key.to_string())
    }

    /// Set expiration time
    pub fn expire(&mut self, key: &str, duration: Duration) -> bool {
        if !self.data.contains_key(&key.to_string()) {
            return false;
        }
        self.expires.insert(key.to_string(), Instant::now() + duration);
        true
    }

    /// Set expiration at specific time
    pub fn expire_at(&mut self, key: &str, time: Instant) -> bool {
        if !self.data.contains_key(&key.to_string()) {
            return false;
        }
        self.expires.insert(key.to_string(), time);
        true
    }

    /// Get TTL (time to live) in seconds
    pub fn ttl(&self, key: &str) -> Option<i64> {
        if !self.data.contains_key(&key.to_string()) {
            return Some(-2); // Key doesn't exist
        }

        match self.expires.get(key) {
            Some(expire_time) => {
                let now = Instant::now();
                if *expire_time <= now {
                    Some(-2) // Key expired
                } else {
                    Some((*expire_time - now).as_secs() as i64)
                }
            }
            None => Some(-1), // Key exists but has no TTL
        }
    }

    /// Get TTL in milliseconds
    pub fn pttl(&self, key: &str) -> Option<i64> {
        if !self.data.contains_key(&key.to_string()) {
            return Some(-2);
        }

        match self.expires.get(key) {
            Some(expire_time) => {
                let now = Instant::now();
                if *expire_time <= now {
                    Some(-2)
                } else {
                    Some((*expire_time - now).as_millis() as i64)
                }
            }
            None => Some(-1),
        }
    }

    /// Remove expiration from key
    pub fn persist(&mut self, key: &str) -> bool {
        self.expires.remove(key).is_some()
    }

    /// Check if key is expired
    fn is_expired(&self, key: &str) -> bool {
        if let Some(expire_time) = self.expires.get(key) {
            *expire_time <= Instant::now()
        } else {
            false
        }
    }

    /// Get type of a key
    pub fn key_type(&mut self, key: &str) -> Option<&'static str> {
        match self.get(key)? {
            RedisObject::String(_) => Some("string"),
            RedisObject::List(_) => Some("list"),
            RedisObject::Set(_) => Some("set"),
            RedisObject::Hash(_) => Some("hash"),
            RedisObject::ZSet(_) => Some("zset"),
        }
    }

    /// Get number of keys
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if database is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Clear all keys
    pub fn flush(&mut self) {
        self.data = Dict::new();
        self.expires.clear();
    }

    /// Get all keys matching a pattern (simple glob)
    pub fn keys(&mut self, pattern: &str) -> Vec<String> {
        // For now, just return all keys if pattern is "*"
        // A full implementation would support glob patterns
        if pattern == "*" {
            let keys = self.data.random_keys(self.data.len() * 2);
            // Deduplicate
            let mut seen = std::collections::HashSet::new();
            keys.into_iter()
                .filter(|k| {
                    if !self.is_expired(k) && seen.insert(k.clone()) {
                        true
                    } else {
                        false
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Increment a string value
    pub fn incr(&mut self, key: &str, amount: i64) -> Result<i64, &'static str> {
        let key_string = key.to_string();

        // Check expiration
        if self.is_expired(key) {
            self.delete(key);
        }

        if let Some(obj) = self.data.get_mut(&key_string) {
            match obj {
                RedisObject::String(s) => s.incr(amount),
                _ => Err("WRONGTYPE Operation against a key holding the wrong kind of value"),
            }
        } else {
            // Create new key with value
            self.set(key_string, RedisObject::String(StringObject::Int(amount)));
            Ok(amount)
        }
    }

    /// Append to a string value
    pub fn append(&mut self, key: &str, value: &[u8]) -> Result<usize, &'static str> {
        let key_string = key.to_string();

        // Check expiration
        if self.is_expired(key) {
            self.delete(key);
        }

        if let Some(obj) = self.data.get_mut(&key_string) {
            match obj {
                RedisObject::String(s) => Ok(s.append(value)),
                _ => Err("WRONGTYPE Operation against a key holding the wrong kind of value"),
            }
        } else {
            // Create new key with value
            let len = value.len();
            self.set(key_string, RedisObject::String(StringObject::Raw(value.to_vec())));
            Ok(len)
        }
    }

    /// Get string length
    pub fn strlen(&mut self, key: &str) -> Result<usize, &'static str> {
        match self.get(key) {
            Some(RedisObject::String(s)) => Ok(s.len()),
            Some(_) => Err("WRONGTYPE Operation against a key holding the wrong kind of value"),
            None => Ok(0),
        }
    }

    /// Set multiple keys
    pub fn mset(&mut self, pairs: Vec<(String, Vec<u8>)>) {
        for (key, value) in pairs {
            self.set_string(key, value);
        }
    }

    /// Get multiple keys
    pub fn mget(&mut self, keys: &[String]) -> Vec<Option<Vec<u8>>> {
        keys.iter()
            .map(|key| self.get_string(key))
            .collect()
    }

    /// Set if not exists
    pub fn setnx(&mut self, key: String, value: Vec<u8>) -> bool {
        if self.exists(&key) {
            false
        } else {
            self.set_string(key, value);
            true
        }
    }

    /// Set with expiration
    pub fn setex(&mut self, key: String, seconds: u64, value: Vec<u8>) {
        self.set_string(key.clone(), value);
        self.expire(&key, Duration::from_secs(seconds));
    }

    /// Get random keys for eviction sampling
    pub fn random_keys(&self, count: usize) -> Vec<String> {
        self.data.random_keys(count)
    }

    /// Get all keys (for RDB persistence)
    pub fn all_keys(&self) -> Vec<String> {
        self.data.iter_keys()
    }

    /// Get number of keys with expiration
    pub fn expires_count(&self) -> usize {
        self.expires.len()
    }
}

impl Default for Database {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_get() {
        let mut db = Database::new();

        db.set_string("foo".to_string(), b"bar".to_vec());
        assert_eq!(db.get_string("foo"), Some(b"bar".to_vec()));
        assert_eq!(db.get_string("nonexistent"), None);
    }

    #[test]
    fn test_delete() {
        let mut db = Database::new();

        db.set_string("foo".to_string(), b"bar".to_vec());
        assert!(db.delete("foo"));
        assert!(!db.exists("foo"));
        assert!(!db.delete("foo"));
    }

    #[test]
    fn test_expiration() {
        let mut db = Database::new();

        db.set_string("foo".to_string(), b"bar".to_vec());
        db.expire("foo", Duration::from_millis(1));

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        assert!(!db.exists("foo"));
        assert_eq!(db.get_string("foo"), None);
    }

    #[test]
    fn test_incr() {
        let mut db = Database::new();

        // INCR on new key
        assert_eq!(db.incr("counter", 1), Ok(1));
        assert_eq!(db.incr("counter", 1), Ok(2));
        assert_eq!(db.incr("counter", 10), Ok(12));
        assert_eq!(db.incr("counter", -5), Ok(7));
    }

    #[test]
    fn test_ttl() {
        let mut db = Database::new();

        // Key doesn't exist
        assert_eq!(db.ttl("nonexistent"), Some(-2));

        // Key exists but no TTL
        db.set_string("foo".to_string(), b"bar".to_vec());
        assert_eq!(db.ttl("foo"), Some(-1));

        // Key with TTL
        db.expire("foo", Duration::from_secs(100));
        let ttl = db.ttl("foo").unwrap();
        assert!(ttl > 0 && ttl <= 100);
    }
}
