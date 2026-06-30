//! Storage engine for the time-series database
//!
//! This module provides the core storage functionality including:
//! - In-memory buffer for recent writes
//! - SSTable format for persistent storage
//! - Memory-mapped file access
//! - Time-based partitioning

pub mod memtable;
pub mod sstable;
pub mod shard;
pub mod engine;

pub use memtable::MemTable;
pub use sstable::{SSTable, SSTableBuilder, SSTableReader};
pub use shard::TimeShard;
pub use engine::{StorageEngine, StorageConfig};
