use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub bind: String,
    pub port: u16,
    pub maxmemory: usize,
    pub maxmemory_policy: String,
    pub databases: usize,
    pub dbfilename: String,
    pub dir: String,
    pub appendonly: bool,
    pub appendfilename: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1".to_string(),
            port: 6379,
            maxmemory: 0,
            maxmemory_policy: "noeviction".to_string(),
            databases: 16,
            dbfilename: "dump.rdb".to_string(),
            dir: ".".to_string(),
            appendonly: false,
            appendfilename: "appendonly.aof".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    NoEviction,
    AllKeysLRU,
    VolatileLRU,
    AllKeysLFU,
    VolatileLFU,
    AllKeysRandom,
    VolatileRandom,
    VolatileTTL,
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
