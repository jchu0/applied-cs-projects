//! Eviction manager for memory management
//!
//! Uses approximate algorithms with random sampling for O(1) eviction
//! decisions, similar to Redis's approach.

use std::time::Instant;

use super::EvictionPolicy;
use crate::storage::Database;

/// Sample size for eviction algorithms
const DEFAULT_SAMPLE_SIZE: usize = 5;

/// Eviction manager configuration
pub struct EvictionConfig {
    /// Maximum memory in bytes (0 = unlimited)
    pub max_memory: usize,
    /// Eviction policy
    pub policy: EvictionPolicy,
    /// Number of keys to sample
    pub sample_size: usize,
}

impl Default for EvictionConfig {
    fn default() -> Self {
        Self {
            max_memory: 0,
            policy: EvictionPolicy::NoEviction,
            sample_size: DEFAULT_SAMPLE_SIZE,
        }
    }
}

/// Eviction manager
pub struct EvictionManager {
    config: EvictionConfig,
    /// Current memory usage estimate
    current_memory: usize,
    /// Eviction statistics
    evicted_keys: u64,
}

impl EvictionManager {
    /// Create a new eviction manager
    pub fn new(config: EvictionConfig) -> Self {
        Self {
            config,
            current_memory: 0,
            evicted_keys: 0,
        }
    }

    /// Update memory usage
    pub fn update_memory(&mut self, delta: isize) {
        if delta > 0 {
            self.current_memory = self.current_memory.saturating_add(delta as usize);
        } else {
            self.current_memory = self.current_memory.saturating_sub((-delta) as usize);
        }
    }

    /// Set current memory usage
    pub fn set_memory(&mut self, bytes: usize) {
        self.current_memory = bytes;
    }

    /// Check if eviction is needed
    pub fn needs_eviction(&self) -> bool {
        self.config.max_memory > 0 && self.current_memory > self.config.max_memory
    }

    /// Perform eviction if needed
    /// Returns number of keys evicted
    pub fn evict_if_needed(&mut self, db: &mut Database) -> Result<usize, &'static str> {
        if self.config.max_memory == 0 {
            return Ok(0);
        }

        if self.current_memory <= self.config.max_memory {
            return Ok(0);
        }

        if self.config.policy == EvictionPolicy::NoEviction {
            return Err("OOM command not allowed when used memory > 'maxmemory'");
        }

        let mut evicted = 0;
        let target_memory = (self.config.max_memory * 95) / 100; // Target 95% of max

        while self.current_memory > target_memory {
            let key = match self.select_key_to_evict(db) {
                Some(k) => k,
                None => break,
            };

            // Estimate memory freed (simplified)
            let freed = self.estimate_key_memory(db, &key);

            if db.delete(&key) {
                self.current_memory = self.current_memory.saturating_sub(freed);
                self.evicted_keys += 1;
                evicted += 1;
            } else {
                break;
            }
        }

        Ok(evicted)
    }

    /// Select a key to evict based on policy
    fn select_key_to_evict(&self, db: &mut Database) -> Option<String> {
        match self.config.policy {
            EvictionPolicy::NoEviction => None,
            EvictionPolicy::AllKeysLRU => self.select_lru(db, false),
            EvictionPolicy::VolatileLRU => self.select_lru(db, true),
            EvictionPolicy::AllKeysLFU => self.select_lfu(db, false),
            EvictionPolicy::VolatileLFU => self.select_lfu(db, true),
            EvictionPolicy::AllKeysRandom => self.select_random(db, false),
            EvictionPolicy::VolatileRandom => self.select_random(db, true),
            EvictionPolicy::VolatileTTL => self.select_ttl(db),
        }
    }

    /// Select key using LRU (Least Recently Used) with sampling
    fn select_lru(&self, db: &mut Database, volatile_only: bool) -> Option<String> {
        let samples = db.random_keys(self.config.sample_size);
        if samples.is_empty() {
            return None;
        }

        let mut best_key: Option<String> = None;
        let mut oldest_time = Instant::now();

        for key in samples {
            // Check if volatile_only
            if volatile_only && db.ttl(&key) == Some(-1) {
                continue;
            }

            // For LRU, we'd need access time tracking
            // Using a simplified approach here
            if best_key.is_none() {
                best_key = Some(key);
                oldest_time = Instant::now();
            }
        }

        best_key
    }

    /// Select key using LFU (Least Frequently Used) with sampling
    fn select_lfu(&self, db: &mut Database, volatile_only: bool) -> Option<String> {
        let samples = db.random_keys(self.config.sample_size);
        if samples.is_empty() {
            return None;
        }

        let mut best_key: Option<String> = None;

        for key in samples {
            if volatile_only && db.ttl(&key) == Some(-1) {
                continue;
            }

            // For LFU, we'd need frequency tracking
            // Using first valid key as simplified approach
            if best_key.is_none() {
                best_key = Some(key);
            }
        }

        best_key
    }

    /// Select random key
    fn select_random(&self, db: &mut Database, volatile_only: bool) -> Option<String> {
        let samples = db.random_keys(1);
        if samples.is_empty() {
            return None;
        }

        let key = &samples[0];
        if volatile_only && db.ttl(key) == Some(-1) {
            // Try again with more samples
            let more_samples = db.random_keys(self.config.sample_size);
            for k in more_samples {
                if db.ttl(&k) != Some(-1) {
                    return Some(k);
                }
            }
            return None;
        }

        Some(key.clone())
    }

    /// Select key with nearest TTL
    fn select_ttl(&self, db: &mut Database) -> Option<String> {
        let samples = db.random_keys(self.config.sample_size);
        if samples.is_empty() {
            return None;
        }

        let mut best_key: Option<String> = None;
        let mut min_ttl = i64::MAX;

        for key in samples {
            if let Some(ttl) = db.pttl(&key) {
                if ttl >= 0 && ttl < min_ttl {
                    min_ttl = ttl;
                    best_key = Some(key);
                }
            }
        }

        best_key
    }

    /// Estimate memory used by a key (simplified)
    fn estimate_key_memory(&self, db: &mut Database, key: &str) -> usize {
        // Base overhead for key
        let mut size = key.len() + 64;

        // Add value size estimate
        if let Some(obj) = db.get(key) {
            size += match obj {
                crate::storage::RedisObject::String(s) => s.len() + 32,
                crate::storage::RedisObject::List(l) => l.iter().map(|v| v.len() + 16).sum::<usize>() + 32,
                crate::storage::RedisObject::Set(s) => s.iter().map(|v| v.len() + 16).sum::<usize>() + 32,
                crate::storage::RedisObject::Hash(h) => {
                    h.iter().map(|(k, v)| k.len() + v.len() + 32).sum::<usize>() + 32
                }
                crate::storage::RedisObject::ZSet(z) => {
                    z.dict.iter().map(|(k, _)| k.len() + 24).sum::<usize>() + 64
                }
            };
        }

        size
    }

    /// Get eviction statistics
    pub fn stats(&self) -> EvictionStats {
        EvictionStats {
            policy: self.config.policy,
            max_memory: self.config.max_memory,
            current_memory: self.current_memory,
            evicted_keys: self.evicted_keys,
        }
    }

    /// Get current memory usage
    pub fn current_memory(&self) -> usize {
        self.current_memory
    }

    /// Get max memory limit
    pub fn max_memory(&self) -> usize {
        self.config.max_memory
    }
}

/// Eviction statistics
#[derive(Debug, Clone)]
pub struct EvictionStats {
    pub policy: EvictionPolicy,
    pub max_memory: usize,
    pub current_memory: usize,
    pub evicted_keys: u64,
}
