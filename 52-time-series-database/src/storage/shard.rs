//! Time-based sharding for time-series data
//!
//! Data is partitioned by time ranges (shards) to:
//! - Enable efficient retention (drop old shards)
//! - Parallelize queries across time ranges
//! - Limit the size of individual SSTables

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use parking_lot::RwLock;

use crate::error::{Result, TsdbError};
use crate::types::{DataPoint, SeriesKey, Metric, Series, duration};
use super::memtable::MemTable;
use super::sstable::{SSTable, SSTableMeta};

/// Default shard duration: 1 hour
pub const DEFAULT_SHARD_DURATION: i64 = duration::HOUR;

/// Time shard containing data for a specific time range
#[derive(Debug)]
pub struct TimeShard {
    /// Start timestamp (inclusive)
    pub start_time: i64,
    /// End timestamp (exclusive)
    pub end_time: i64,
    /// In-memory data for this shard
    memtable: MemTable,
    /// Flushed SSTables for this shard
    sstables: RwLock<Vec<SSTable>>,
    /// Directory for SSTable files
    data_dir: PathBuf,
    /// SSTable counter for unique filenames
    sstable_counter: std::sync::atomic::AtomicU64,
}

impl TimeShard {
    /// Create a new time shard
    pub fn new<P: AsRef<Path>>(start_time: i64, duration: i64, data_dir: P) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        Ok(Self {
            start_time,
            end_time: start_time + duration,
            memtable: MemTable::new(),
            sstables: RwLock::new(Vec::new()),
            data_dir,
            sstable_counter: std::sync::atomic::AtomicU64::new(0),
        })
    }

    /// Check if a timestamp belongs to this shard
    pub fn contains_timestamp(&self, timestamp: i64) -> bool {
        timestamp >= self.start_time && timestamp < self.end_time
    }

    /// Insert a data point
    pub fn insert(&self, metric: &Metric, point: DataPoint) -> Result<()> {
        if !self.contains_timestamp(point.timestamp) {
            return Err(TsdbError::storage(format!(
                "Timestamp {} is outside shard range [{}, {})",
                point.timestamp, self.start_time, self.end_time
            )));
        }
        self.memtable.insert(metric, point)
    }

    /// Insert multiple points
    pub fn insert_batch(&self, points: &[(Metric, DataPoint)]) -> Result<()> {
        // Filter points that belong to this shard
        let valid_points: Vec<_> = points
            .iter()
            .filter(|(_, p)| self.contains_timestamp(p.timestamp))
            .cloned()
            .collect();

        if valid_points.is_empty() {
            return Ok(());
        }

        self.memtable.insert_batch(&valid_points)
    }

    /// Query points in a time range for a specific series
    pub fn query_range(&self, series_key: SeriesKey, start: i64, end: i64) -> Result<Vec<DataPoint>> {
        let mut result = Vec::new();

        // Query memtable
        result.extend(self.memtable.query_range(series_key, start, end));

        // Query SSTables
        let sstables = self.sstables.read();
        for sstable in sstables.iter() {
            if sstable.overlaps(start, end) {
                result.extend(sstable.query_range(series_key, start, end)?);
            }
        }

        // Sort and deduplicate
        result.sort_by_key(|p| p.timestamp);
        result.dedup_by_key(|p| p.timestamp);

        Ok(result)
    }

    /// Get all points for a series
    pub fn query_series(&self, series_key: SeriesKey) -> Result<Vec<DataPoint>> {
        self.query_range(series_key, self.start_time, self.end_time)
    }

    /// Flush memtable to SSTable
    pub fn flush(&self) -> Result<Option<SSTableMeta>> {
        let data = self.memtable.take();

        if data.is_empty() {
            return Ok(None);
        }

        let counter = self
            .sstable_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let filename = format!(
            "shard_{}_{}_{}.sst",
            self.start_time, self.end_time, counter
        );
        let path = self.data_dir.join(filename);

        let meta = SSTable::create_from_map(&path, &data)?;

        // Load the SSTable for querying
        let sstable = SSTable::open(&path)?;
        self.sstables.write().push(sstable);

        Ok(Some(meta))
    }

    /// Check if memtable should be flushed
    pub fn should_flush(&self) -> bool {
        self.memtable.should_flush()
    }

    /// Get shard statistics
    pub fn stats(&self) -> ShardStats {
        let sstables = self.sstables.read();
        ShardStats {
            start_time: self.start_time,
            end_time: self.end_time,
            memtable_size: self.memtable.size(),
            memtable_series: self.memtable.series_count(),
            memtable_points: self.memtable.point_count(),
            sstable_count: sstables.len(),
            sstable_size: sstables.iter().map(|s| s.meta().file_size).sum(),
        }
    }

    /// Get all series keys in this shard
    pub fn series_keys(&self) -> Vec<SeriesKey> {
        let mut keys: Vec<_> = self.memtable.series_keys();

        let sstables = self.sstables.read();
        for sstable in sstables.iter() {
            for key in sstable.series_keys() {
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
        }

        keys
    }

    /// Compact all SSTables into one
    pub fn compact(&self) -> Result<Option<SSTableMeta>> {
        // Flush memtable first
        self.flush()?;

        let mut sstables = self.sstables.write();

        if sstables.len() <= 1 {
            return Ok(None);
        }

        // Merge all series
        let mut merged: BTreeMap<SeriesKey, Series> = BTreeMap::new();

        for sstable in sstables.iter() {
            for key in sstable.series_keys() {
                if let Some(series) = sstable.read_series(key)? {
                    let entry = merged.entry(key).or_insert_with(|| {
                        Series::new(series.metric.clone())
                    });
                    for point in series.points {
                        entry.insert_sorted(point);
                    }
                }
            }
        }

        if merged.is_empty() {
            return Ok(None);
        }

        // Create new compacted SSTable
        let counter = self
            .sstable_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let filename = format!(
            "shard_{}_{}_{}_compacted.sst",
            self.start_time, self.end_time, counter
        );
        let path = self.data_dir.join(&filename);

        let meta = SSTable::create_from_map(&path, &merged)?;

        // Remove old SSTables
        let old_paths: Vec<_> = sstables.iter().map(|s| s.meta().path.clone()).collect();
        sstables.clear();

        // Load new SSTable
        let new_sstable = SSTable::open(&path)?;
        sstables.push(new_sstable);

        // Delete old files
        for old_path in old_paths {
            let _ = std::fs::remove_file(old_path);
        }

        Ok(Some(meta))
    }

    /// Get the total size of this shard
    pub fn total_size(&self) -> u64 {
        let sstables = self.sstables.read();
        let sstable_size: u64 = sstables.iter().map(|s| s.meta().file_size).sum();
        sstable_size + self.memtable.size() as u64
    }

    /// Check if this shard is empty
    pub fn is_empty(&self) -> bool {
        self.memtable.is_empty() && self.sstables.read().is_empty()
    }

    /// Load existing SSTables from disk
    pub fn load_sstables(&self) -> Result<usize> {
        let mut loaded = 0;
        let mut sstables = self.sstables.write();

        for entry in std::fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |ext| ext == "sst") {
                match SSTable::open(&path) {
                    Ok(sstable) => {
                        sstables.push(sstable);
                        loaded += 1;
                    }
                    Err(e) => {
                        eprintln!("Failed to load SSTable {:?}: {}", path, e);
                    }
                }
            }
        }

        Ok(loaded)
    }
}

/// Statistics for a time shard
#[derive(Debug, Clone)]
pub struct ShardStats {
    pub start_time: i64,
    pub end_time: i64,
    pub memtable_size: usize,
    pub memtable_series: usize,
    pub memtable_points: usize,
    pub sstable_count: usize,
    pub sstable_size: u64,
}

/// Shard manager handles multiple time shards
#[derive(Debug)]
pub struct ShardManager {
    /// Base directory for all shards
    data_dir: PathBuf,
    /// Shard duration in nanoseconds
    shard_duration: i64,
    /// Active shards
    shards: RwLock<BTreeMap<i64, TimeShard>>,
}

impl ShardManager {
    /// Create a new shard manager
    pub fn new<P: AsRef<Path>>(data_dir: P, shard_duration: i64) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&data_dir)?;

        Ok(Self {
            data_dir,
            shard_duration,
            shards: RwLock::new(BTreeMap::new()),
        })
    }

    /// Get the shard start time for a timestamp
    fn shard_start_time(&self, timestamp: i64) -> i64 {
        (timestamp / self.shard_duration) * self.shard_duration
    }

    /// Get or create a shard for a timestamp
    pub fn get_or_create_shard(&self, timestamp: i64) -> Result<()> {
        let start_time = self.shard_start_time(timestamp);

        {
            let shards = self.shards.read();
            if shards.contains_key(&start_time) {
                return Ok(());
            }
        }

        // Need to create new shard
        let mut shards = self.shards.write();
        if !shards.contains_key(&start_time) {
            let shard_dir = self.data_dir.join(format!("shard_{}", start_time));
            let shard = TimeShard::new(start_time, self.shard_duration, shard_dir)?;
            shards.insert(start_time, shard);
        }

        Ok(())
    }

    /// Insert a data point
    pub fn insert(&self, metric: &Metric, point: DataPoint) -> Result<()> {
        let start_time = self.shard_start_time(point.timestamp);
        self.get_or_create_shard(point.timestamp)?;

        let shards = self.shards.read();
        if let Some(shard) = shards.get(&start_time) {
            shard.insert(metric, point)?;
        }

        Ok(())
    }

    /// Insert a batch of points
    pub fn insert_batch(&self, points: &[(Metric, DataPoint)]) -> Result<()> {
        // Group points by shard
        let mut by_shard: BTreeMap<i64, Vec<(Metric, DataPoint)>> = BTreeMap::new();

        for (metric, point) in points {
            let start_time = self.shard_start_time(point.timestamp);
            by_shard
                .entry(start_time)
                .or_default()
                .push((metric.clone(), *point));
        }

        // Insert into each shard
        for (start_time, shard_points) in by_shard {
            self.get_or_create_shard(start_time)?;
            let shards = self.shards.read();
            if let Some(shard) = shards.get(&start_time) {
                shard.insert_batch(&shard_points)?;
            }
        }

        Ok(())
    }

    /// Query points in a time range
    pub fn query_range(&self, series_key: SeriesKey, start: i64, end: i64) -> Result<Vec<DataPoint>> {
        let mut result = Vec::new();
        let shards = self.shards.read();

        for shard in shards.values() {
            if shard.end_time >= start && shard.start_time <= end {
                result.extend(shard.query_range(series_key, start, end)?);
            }
        }

        result.sort_by_key(|p| p.timestamp);
        result.dedup_by_key(|p| p.timestamp);

        Ok(result)
    }

    /// Flush all shards
    pub fn flush_all(&self) -> Result<Vec<SSTableMeta>> {
        let mut metas = Vec::new();
        let shards = self.shards.read();

        for shard in shards.values() {
            if let Some(meta) = shard.flush()? {
                metas.push(meta);
            }
        }

        Ok(metas)
    }

    /// Compact all shards
    pub fn compact_all(&self) -> Result<Vec<SSTableMeta>> {
        let mut metas = Vec::new();
        let shards = self.shards.read();

        for shard in shards.values() {
            if let Some(meta) = shard.compact()? {
                metas.push(meta);
            }
        }

        Ok(metas)
    }

    /// Drop shards older than a timestamp
    pub fn drop_before(&self, timestamp: i64) -> Result<usize> {
        let mut shards = self.shards.write();
        let old_keys: Vec<_> = shards
            .keys()
            .filter(|&&start| start + self.shard_duration <= timestamp)
            .copied()
            .collect();

        for key in &old_keys {
            if let Some(shard) = shards.remove(key) {
                // Delete shard directory
                let shard_dir = self.data_dir.join(format!("shard_{}", key));
                let _ = std::fs::remove_dir_all(shard_dir);
            }
        }

        Ok(old_keys.len())
    }

    /// Get all shard statistics
    pub fn all_stats(&self) -> Vec<ShardStats> {
        let shards = self.shards.read();
        shards.values().map(|s| s.stats()).collect()
    }

    /// Get total size across all shards
    pub fn total_size(&self) -> u64 {
        let shards = self.shards.read();
        shards.values().map(|s| s.total_size()).sum()
    }

    /// Get shard count
    pub fn shard_count(&self) -> usize {
        self.shards.read().len()
    }

    /// Load existing shards from disk
    pub fn load_existing(&self) -> Result<usize> {
        let mut loaded = 0;

        for entry in std::fs::read_dir(&self.data_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("shard_") {
                        if let Ok(start_time) = name[6..].parse::<i64>() {
                            let shard = TimeShard::new(start_time, self.shard_duration, &path)?;
                            shard.load_sstables()?;
                            self.shards.write().insert(start_time, shard);
                            loaded += 1;
                        }
                    }
                }
            }
        }

        Ok(loaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_time_shard_insert_and_query() {
        let dir = tempdir().unwrap();
        let shard = TimeShard::new(0, duration::HOUR, dir.path()).unwrap();

        let metric = Metric::new("test").tag("host", "server1");

        // Insert points within shard range
        for i in 0..100 {
            shard.insert(&metric, DataPoint::new(i * 60, i as f64)).unwrap();
        }

        let points = shard.query_range(metric.series_key(), 0, 3600).unwrap();
        assert_eq!(points.len(), 61); // Points 0-60 minutes (inclusive range)
    }

    #[test]
    fn test_time_shard_out_of_range() {
        let dir = tempdir().unwrap();
        let shard = TimeShard::new(1000, 1000, dir.path()).unwrap();

        let metric = Metric::new("test");

        // Try to insert point outside shard range
        let result = shard.insert(&metric, DataPoint::new(500, 1.0));
        assert!(result.is_err());
    }

    #[test]
    fn test_time_shard_flush() {
        let dir = tempdir().unwrap();
        let shard = TimeShard::new(0, duration::HOUR, dir.path()).unwrap();

        let metric = Metric::new("test");
        for i in 0..100 {
            shard.insert(&metric, DataPoint::new(i * 60, i as f64)).unwrap();
        }

        let meta = shard.flush().unwrap();
        assert!(meta.is_some());

        // Should still be queryable after flush
        let points = shard.query_range(metric.series_key(), 0, 6000).unwrap();
        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_shard_manager_auto_shard() {
        let dir = tempdir().unwrap();
        let manager = ShardManager::new(dir.path(), duration::HOUR).unwrap();

        let metric = Metric::new("test");

        // Insert points spanning multiple hours (120 minutes = 2 hours)
        for i in 0..120 {
            manager
                .insert(&metric, DataPoint::new(i * duration::MINUTE, i as f64))
                .unwrap();
        }

        assert_eq!(manager.shard_count(), 2); // 2 hours worth of data
    }

    #[test]
    fn test_shard_manager_query_range() {
        let dir = tempdir().unwrap();
        let manager = ShardManager::new(dir.path(), duration::HOUR).unwrap();

        let metric = Metric::new("test");

        for i in 0..120 {
            manager
                .insert(&metric, DataPoint::new(i * duration::MINUTE, i as f64))
                .unwrap();
        }

        // Query across shard boundary
        let points = manager
            .query_range(metric.series_key(), 50 * duration::MINUTE, 70 * duration::MINUTE)
            .unwrap();

        assert_eq!(points.len(), 21); // 50-70 inclusive
    }

    #[test]
    fn test_shard_manager_drop_before() {
        let dir = tempdir().unwrap();
        let manager = ShardManager::new(dir.path(), duration::HOUR).unwrap();

        let metric = Metric::new("test");

        // Insert into 3 different hours
        for hour in 0..3 {
            let base = hour * duration::HOUR;
            for i in 0..10 {
                manager
                    .insert(&metric, DataPoint::new(base + i * 60, i as f64))
                    .unwrap();
            }
        }

        assert_eq!(manager.shard_count(), 3);

        // Drop first hour
        let dropped = manager.drop_before(duration::HOUR).unwrap();
        assert_eq!(dropped, 1);
        assert_eq!(manager.shard_count(), 2);
    }

    #[test]
    fn test_shard_compact() {
        let dir = tempdir().unwrap();
        let shard = TimeShard::new(0, duration::HOUR, dir.path()).unwrap();

        let metric = Metric::new("test");

        // Create multiple SSTables
        for batch in 0..3 {
            for i in 0..10 {
                shard
                    .insert(&metric, DataPoint::new((batch * 100 + i) * 60, i as f64))
                    .unwrap();
            }
            shard.flush().unwrap();
        }

        let stats_before = shard.stats();
        assert_eq!(stats_before.sstable_count, 3);

        // Compact
        shard.compact().unwrap();

        let stats_after = shard.stats();
        assert_eq!(stats_after.sstable_count, 1);
    }

    #[test]
    fn test_shard_batch_insert() {
        let dir = tempdir().unwrap();
        let shard = TimeShard::new(0, duration::HOUR, dir.path()).unwrap();

        let metric = Metric::new("test");
        let points: Vec<_> = (0..100)
            .map(|i| (metric.clone(), DataPoint::new(i * 60, i as f64)))
            .collect();

        shard.insert_batch(&points).unwrap();

        assert_eq!(shard.stats().memtable_points, 100);
    }

    #[test]
    fn test_shard_series_keys() {
        let dir = tempdir().unwrap();
        let shard = TimeShard::new(0, duration::HOUR, dir.path()).unwrap();

        let metric1 = Metric::new("metric1");
        let metric2 = Metric::new("metric2");

        shard.insert(&metric1, DataPoint::new(100, 1.0)).unwrap();
        shard.insert(&metric2, DataPoint::new(200, 2.0)).unwrap();

        let keys = shard.series_keys();
        assert_eq!(keys.len(), 2);
    }
}
