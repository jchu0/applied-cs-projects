//! Storage engine that coordinates all storage components
//!
//! The storage engine provides a unified interface for:
//! - Writing data points
//! - Querying data
//! - Managing shards and compaction
//! - Handling retention policies

use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;

use crate::error::{Result, TsdbError};
use crate::types::{DataPoint, SeriesKey, Metric, Tags};
use super::shard::{ShardManager, DEFAULT_SHARD_DURATION};
use super::memtable::MemTable;

/// Storage engine configuration
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Base directory for data files
    pub data_dir: PathBuf,
    /// Shard duration in nanoseconds
    pub shard_duration: i64,
    /// Maximum memtable size before flush
    pub max_memtable_size: usize,
    /// Enable WAL
    pub enable_wal: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("data"),
            shard_duration: DEFAULT_SHARD_DURATION,
            max_memtable_size: 64 * 1024 * 1024, // 64MB
            enable_wal: true,
        }
    }
}

impl StorageConfig {
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    pub fn with_shard_duration(mut self, duration: i64) -> Self {
        self.shard_duration = duration;
        self
    }

    pub fn with_max_memtable_size(mut self, size: usize) -> Self {
        self.max_memtable_size = size;
        self
    }

    pub fn with_wal(mut self, enable: bool) -> Self {
        self.enable_wal = enable;
        self
    }
}

/// Storage engine statistics
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    /// Total number of series
    pub series_count: usize,
    /// Total number of data points
    pub point_count: usize,
    /// Number of shards
    pub shard_count: usize,
    /// Total size in bytes
    pub total_size: u64,
    /// Number of SSTables
    pub sstable_count: usize,
}

/// The main storage engine
#[derive(Debug)]
pub struct StorageEngine {
    /// Configuration
    config: StorageConfig,
    /// Shard manager
    shards: ShardManager,
    /// Series metadata (series key -> metric)
    series_index: RwLock<hashbrown::HashMap<SeriesKey, Metric>>,
    /// Closed flag
    closed: std::sync::atomic::AtomicBool,
}

impl StorageEngine {
    /// Create a new storage engine
    pub fn new(config: StorageConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.data_dir)?;

        let shards_dir = config.data_dir.join("shards");
        let shards = ShardManager::new(&shards_dir, config.shard_duration)?;

        Ok(Self {
            config,
            shards,
            series_index: RwLock::new(hashbrown::HashMap::new()),
            closed: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Open an existing storage engine or create a new one
    pub fn open(config: StorageConfig) -> Result<Self> {
        let engine = Self::new(config)?;
        engine.shards.load_existing()?;
        Ok(engine)
    }

    /// Check if the engine is closed
    fn check_open(&self) -> Result<()> {
        if self.closed.load(std::sync::atomic::Ordering::Relaxed) {
            Err(TsdbError::DatabaseClosed)
        } else {
            Ok(())
        }
    }

    /// Write a single data point
    pub fn write(&self, metric: &str, tags: &Tags, timestamp: i64, value: f64) -> Result<()> {
        self.check_open()?;

        let metric = Metric::with_tags(metric.to_string(), tags.clone());
        let point = DataPoint::new(timestamp, value);

        // Update series index
        {
            let mut index = self.series_index.write();
            let key = metric.series_key();
            index.entry(key).or_insert_with(|| metric.clone());
        }

        self.shards.insert(&metric, point)
    }

    /// Write multiple data points
    pub fn write_batch(&self, points: &[(String, Tags, i64, f64)]) -> Result<()> {
        self.check_open()?;

        let batch: Vec<_> = points
            .iter()
            .map(|(name, tags, ts, val)| {
                let metric = Metric::with_tags(name.clone(), tags.clone());
                (metric, DataPoint::new(*ts, *val))
            })
            .collect();

        // Update series index
        {
            let mut index = self.series_index.write();
            for (metric, _) in &batch {
                let key = metric.series_key();
                index.entry(key).or_insert_with(|| metric.clone());
            }
        }

        self.shards.insert_batch(&batch)
    }

    /// Write data points with pre-built metrics
    pub fn write_points(&self, points: &[(Metric, DataPoint)]) -> Result<()> {
        self.check_open()?;

        // Update series index
        {
            let mut index = self.series_index.write();
            for (metric, _) in points {
                let key = metric.series_key();
                index.entry(key).or_insert_with(|| metric.clone());
            }
        }

        self.shards.insert_batch(points)
    }

    /// Query points in a time range for a specific series
    pub fn query_range(&self, series_key: SeriesKey, start: i64, end: i64) -> Result<Vec<DataPoint>> {
        self.check_open()?;
        self.shards.query_range(series_key, start, end)
    }

    /// Query by metric name and tags
    pub fn query_metric(&self, name: &str, tags: &Tags, start: i64, end: i64) -> Result<Vec<DataPoint>> {
        let metric = Metric::with_tags(name.to_string(), tags.clone());
        self.query_range(metric.series_key(), start, end)
    }

    /// Get all series keys
    pub fn series_keys(&self) -> Vec<SeriesKey> {
        self.series_index.read().keys().copied().collect()
    }

    /// Get metric for a series key
    pub fn get_metric(&self, series_key: SeriesKey) -> Option<Metric> {
        self.series_index.read().get(&series_key).cloned()
    }

    /// Find series matching a metric name pattern
    pub fn find_series(&self, name_prefix: &str) -> Vec<(SeriesKey, Metric)> {
        self.series_index
            .read()
            .iter()
            .filter(|(_, m)| m.name.starts_with(name_prefix))
            .map(|(k, m)| (*k, m.clone()))
            .collect()
    }

    /// Flush all in-memory data to disk
    pub fn flush(&self) -> Result<()> {
        self.check_open()?;
        self.shards.flush_all()?;
        Ok(())
    }

    /// Compact storage
    pub fn compact(&self) -> Result<()> {
        self.check_open()?;
        self.shards.compact_all()?;
        Ok(())
    }

    /// Drop data older than a timestamp
    pub fn drop_before(&self, timestamp: i64) -> Result<usize> {
        self.check_open()?;
        self.shards.drop_before(timestamp)
    }

    /// Get storage statistics
    pub fn stats(&self) -> StorageStats {
        let shard_stats = self.shards.all_stats();

        StorageStats {
            series_count: self.series_index.read().len(),
            point_count: shard_stats
                .iter()
                .map(|s| s.memtable_points)
                .sum(),
            shard_count: shard_stats.len(),
            total_size: self.shards.total_size(),
            sstable_count: shard_stats
                .iter()
                .map(|s| s.sstable_count)
                .sum(),
        }
    }

    /// Close the storage engine
    pub fn close(&self) -> Result<()> {
        if self
            .closed
            .compare_exchange(
                false,
                true,
                std::sync::atomic::Ordering::SeqCst,
                std::sync::atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            // Flush directly without check_open since we just set closed = true
            self.shards.flush_all()?;
        }
        Ok(())
    }

    /// Get the configuration
    pub fn config(&self) -> &StorageConfig {
        &self.config
    }
}

impl Drop for StorageEngine {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::types::duration;

    #[test]
    fn test_storage_engine_write_and_query() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let mut tags = Tags::new();
        tags.insert("host".into(), "server1".into());

        // Write some data
        for i in 0..100 {
            engine
                .write("cpu.usage", &tags, i * 60, i as f64)
                .unwrap();
        }

        // Query
        let metric = Metric::with_tags("cpu.usage", tags.clone());
        let points = engine.query_range(metric.series_key(), 0, 6000).unwrap();

        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_storage_engine_batch_write() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let mut tags = Tags::new();
        tags.insert("host".into(), "server1".into());

        let batch: Vec<_> = (0..100)
            .map(|i| ("cpu.usage".to_string(), tags.clone(), i * 60, i as f64))
            .collect();

        engine.write_batch(&batch).unwrap();

        let stats = engine.stats();
        assert!(stats.series_count >= 1);
    }

    #[test]
    fn test_storage_engine_flush() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let mut tags = Tags::new();
        tags.insert("host".into(), "server1".into());

        for i in 0..100 {
            engine
                .write("cpu.usage", &tags, i * 60, i as f64)
                .unwrap();
        }

        engine.flush().unwrap();

        // Should still be queryable
        let metric = Metric::with_tags("cpu.usage", tags);
        let points = engine.query_range(metric.series_key(), 0, 6000).unwrap();
        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_storage_engine_find_series() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let tags = Tags::new();

        engine.write("cpu.usage", &tags, 100, 1.0).unwrap();
        engine.write("cpu.system", &tags, 100, 2.0).unwrap();
        engine.write("memory.used", &tags, 100, 3.0).unwrap();

        let cpu_series = engine.find_series("cpu.");
        assert_eq!(cpu_series.len(), 2);

        let memory_series = engine.find_series("memory.");
        assert_eq!(memory_series.len(), 1);
    }

    #[test]
    fn test_storage_engine_drop_before() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path())
            .with_shard_duration(duration::HOUR);
        let engine = StorageEngine::new(config).unwrap();

        let tags = Tags::new();

        // Write data across multiple hours
        for hour in 0..3 {
            let base = hour * duration::HOUR;
            for i in 0..10 {
                engine
                    .write("test", &tags, base + i * 60, i as f64)
                    .unwrap();
            }
        }

        assert_eq!(engine.stats().shard_count, 3);

        // Drop first hour
        let dropped = engine.drop_before(duration::HOUR).unwrap();
        assert_eq!(dropped, 1);
        assert_eq!(engine.stats().shard_count, 2);
    }

    #[test]
    fn test_storage_engine_close() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        engine.close().unwrap();

        // Operations should fail after close
        let result = engine.write("test", &Tags::new(), 100, 1.0);
        assert!(matches!(result, Err(TsdbError::DatabaseClosed)));
    }

    #[test]
    fn test_storage_engine_query_metric() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let mut tags = Tags::new();
        tags.insert("env".into(), "prod".into());

        for i in 0..50 {
            engine.write("requests", &tags, i * 60, i as f64).unwrap();
        }

        let points = engine.query_metric("requests", &tags, 0, 3000).unwrap();
        assert_eq!(points.len(), 50);
    }

    #[test]
    fn test_storage_engine_stats() {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();

        let stats = engine.stats();
        assert_eq!(stats.series_count, 0);
        assert_eq!(stats.point_count, 0);

        let tags = Tags::new();
        for i in 0..100 {
            engine.write("test", &tags, i * 60, i as f64).unwrap();
        }

        let stats = engine.stats();
        assert_eq!(stats.series_count, 1);
    }

    #[test]
    fn test_storage_config_builder() {
        let config = StorageConfig::new("/tmp/test")
            .with_shard_duration(duration::DAY)
            .with_max_memtable_size(128 * 1024 * 1024)
            .with_wal(false);

        assert_eq!(config.shard_duration, duration::DAY);
        assert_eq!(config.max_memtable_size, 128 * 1024 * 1024);
        assert!(!config.enable_wal);
    }
}
