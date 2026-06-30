//! Eviction policy definitions

/// Eviction policy types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Don't evict, return error on memory limit
    NoEviction,
    /// Evict least recently used keys from all keys
    AllKeysLRU,
    /// Evict least recently used keys with TTL
    VolatileLRU,
    /// Evict least frequently used keys from all keys
    AllKeysLFU,
    /// Evict least frequently used keys with TTL
    VolatileLFU,
    /// Evict random keys from all keys
    AllKeysRandom,
    /// Evict random keys with TTL
    VolatileRandom,
    /// Evict keys with nearest TTL
    VolatileTTL,
}

impl EvictionPolicy {
    /// Check if policy targets only volatile (keys with TTL) keys
    pub fn is_volatile(&self) -> bool {
        matches!(
            self,
            EvictionPolicy::VolatileLRU
                | EvictionPolicy::VolatileLFU
                | EvictionPolicy::VolatileRandom
                | EvictionPolicy::VolatileTTL
        )
    }

    /// Check if policy uses LRU algorithm
    pub fn is_lru(&self) -> bool {
        matches!(self, EvictionPolicy::AllKeysLRU | EvictionPolicy::VolatileLRU)
    }

    /// Check if policy uses LFU algorithm
    pub fn is_lfu(&self) -> bool {
        matches!(self, EvictionPolicy::AllKeysLFU | EvictionPolicy::VolatileLFU)
    }
}

impl From<&str> for EvictionPolicy {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "allkeys-lru" => EvictionPolicy::AllKeysLRU,
            "volatile-lru" => EvictionPolicy::VolatileLRU,
            "allkeys-lfu" => EvictionPolicy::AllKeysLFU,
            "volatile-lfu" => EvictionPolicy::VolatileLFU,
            "allkeys-random" => EvictionPolicy::AllKeysRandom,
            "volatile-random" => EvictionPolicy::VolatileRandom,
            "volatile-ttl" => EvictionPolicy::VolatileTTL,
            _ => EvictionPolicy::NoEviction,
        }
    }
}

impl std::fmt::Display for EvictionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            EvictionPolicy::NoEviction => "noeviction",
            EvictionPolicy::AllKeysLRU => "allkeys-lru",
            EvictionPolicy::VolatileLRU => "volatile-lru",
            EvictionPolicy::AllKeysLFU => "allkeys-lfu",
            EvictionPolicy::VolatileLFU => "volatile-lfu",
            EvictionPolicy::AllKeysRandom => "allkeys-random",
            EvictionPolicy::VolatileRandom => "volatile-random",
            EvictionPolicy::VolatileTTL => "volatile-ttl",
        };
        write!(f, "{}", s)
    }
}
