//! Memory eviction module
//!
//! Implements LRU (Least Recently Used) and LFU (Least Frequently Used)
//! eviction algorithms using approximate sampling, similar to Redis.

pub mod manager;
mod policy;

pub use manager::{EvictionConfig, EvictionManager};
pub use policy::EvictionPolicy;
