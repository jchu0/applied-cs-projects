//! Memory eviction module
//!
//! Implements LRU (Least Recently Used) and LFU (Least Frequently Used)
//! eviction algorithms using approximate sampling, similar to Redis.

mod manager;
mod policy;

pub use manager::EvictionManager;
pub use policy::EvictionPolicy;
