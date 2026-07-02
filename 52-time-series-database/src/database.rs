//! Main database interface
//!
//! The TimeSeriesDB provides a high-level interface for:
//! - Writing time-series data
//! - Querying with aggregations and filtering
//! - Managing retention policies
//! - Background maintenance tasks

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use parking_lot::RwLock;
use crossbeam_channel::{bounded, Sender, Receiver};

use crate::error::{Result, TsdbError};
use crate::types::{DataPoint, SeriesKey, Metric, Tags, duration};
use crate::storage::{StorageEngine, StorageConfig};
use crate::query::{Query, QueryResult, QueryExecutor, Aggregation};
use crate::retention::{RetentionPolicy, RetentionManager};
use crate::wal::{WriteAheadLog, WalEntry};

/// Database configuration
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Data directory
    pub data_dir: PathBuf,
    /// Enable WAL for durability
    pub enable_wal: bool,
    /// Shard duration
    pub shard_duration: i64,
    /// Maximum memtable size before flush
    pub max_memtable_size: usize,
    /// Background flush interval
    pub flush_interval: Duration,
    /// Background compaction interval
    pub compaction_interval: Duration,
    /// Background retention check interval
    pub retention_check_interval: Duration,
    /// Enable background tasks
    pub enable_background_tasks: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("tsdb_data"),
            enable_wal: true,
            shard_duration: duration::HOUR,
            max_memtable_size: 64 * 1024 * 1024,
            flush_interval: Duration::from_secs(60),
            compaction_interval: Duration::from_secs(3600),
            retention_check_interval: Duration::from_secs(3600),
            enable_background_tasks: true,
        }
    }
}

impl DatabaseConfig {
    /// Create a config with just a data directory
    pub fn new<P: AsRef<Path>>(data_dir: P) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
            ..Default::default()
        }
    }

    /// Disable WAL (for testing)
    pub fn without_wal(mut self) -> Self {
        self.enable_wal = false;
        self
    }

    /// Disable background tasks (for testing)
    pub fn without_background_tasks(mut self) -> Self {
        self.enable_background_tasks = false;
        self
    }

    /// Set shard duration
    pub fn with_shard_duration(mut self, duration: i64) -> Self {
        self.shard_duration = duration;
        self
    }
}

/// Background task message
enum BackgroundMessage {
    Flush,
    Compact,
    CheckRetention,
    Shutdown,
}

/// Time-series database
pub struct TimeSeriesDB {
    /// Storage engine
    storage: Arc<StorageEngine>,
    /// Write-ahead log
    wal: Option<Arc<WriteAheadLog>>,
    /// Retention manager
    retention: Arc<RetentionManager>,
    /// Configuration
    config: DatabaseConfig,
    /// Background task sender
    bg_sender: Option<Sender<BackgroundMessage>>,
    /// Shutdown flag
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    /// Count of errors encountered by background flush/compact/retention tasks
    bg_error_count: Arc<std::sync::atomic::AtomicU64>,
}

impl TimeSeriesDB {
    /// Open or create a new database
    pub fn open(config: DatabaseConfig) -> Result<Self> {
        std::fs::create_dir_all(&config.data_dir)?;

        // Create storage engine
        let storage_config = StorageConfig::new(&config.data_dir)
            .with_shard_duration(config.shard_duration)
            .with_max_memtable_size(config.max_memtable_size)
            .with_wal(config.enable_wal);

        let storage = Arc::new(StorageEngine::open(storage_config)?);

        // Create WAL
        let wal = if config.enable_wal {
            let wal_dir = config.data_dir.join("wal");
            Some(Arc::new(WriteAheadLog::new(&wal_dir)?))
        } else {
            None
        };

        // Create retention manager
        let retention = Arc::new(RetentionManager::new());

        // Setup background tasks
        let shutdown = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let bg_error_count = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let bg_sender = if config.enable_background_tasks {
            let (sender, receiver) = bounded::<BackgroundMessage>(100);
            let storage_clone = storage.clone();
            let retention_clone = retention.clone();
            let shutdown_clone = shutdown.clone();
            let error_count_clone = bg_error_count.clone();
            let flush_interval = config.flush_interval;
            let compaction_interval = config.compaction_interval;
            let retention_interval = config.retention_check_interval;

            thread::spawn(move || {
                background_worker(
                    storage_clone,
                    retention_clone,
                    receiver,
                    shutdown_clone,
                    flush_interval,
                    compaction_interval,
                    retention_interval,
                    error_count_clone,
                );
            });

            Some(sender)
        } else {
            None
        };

        let db = Self {
            storage,
            wal,
            retention,
            config,
            bg_sender,
            shutdown,
            bg_error_count,
        };

        // Replay WAL if enabled
        if let Some(ref wal) = db.wal {
            db.replay_wal(wal)?;
        }

        Ok(db)
    }

    /// Replay WAL entries
    fn replay_wal(&self, wal: &WriteAheadLog) -> Result<u64> {
        let storage = self.storage.clone();
        wal.replay(|entry| {
            match entry {
                WalEntry::Write {
                    metric_name,
                    tags,
                    timestamp,
                    value,
                } => {
                    storage.write(&metric_name, &tags, timestamp, value)?;
                }
                WalEntry::WriteBatch { points } => {
                    storage.write_batch(&points)?;
                }
                WalEntry::Checkpoint { .. } => {
                    // Checkpoints are just markers
                }
            }
            Ok(())
        })
    }

    /// Write a single data point
    pub fn write(&self, metric: &str, tags: &Tags, timestamp: i64, value: f64) -> Result<()> {
        // Write to WAL first
        if let Some(ref wal) = self.wal {
            wal.append(&WalEntry::Write {
                metric_name: metric.to_string(),
                tags: tags.clone(),
                timestamp,
                value,
            })?;
        }

        // Write to storage
        self.storage.write(metric, tags, timestamp, value)
    }

    /// Write multiple data points in a batch
    pub fn write_batch(&self, points: &[(String, Tags, i64, f64)]) -> Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        // Write to WAL first
        if let Some(ref wal) = self.wal {
            wal.append(&WalEntry::WriteBatch {
                points: points.to_vec(),
            })?;
        }

        // Write to storage
        self.storage.write_batch(points)
    }

    /// Query data points in a time range
    pub fn query_range(&self, series_key: SeriesKey, start: i64, end: i64) -> Result<Vec<DataPoint>> {
        self.storage.query_range(series_key, start, end)
    }

    /// Query by metric name and tags
    pub fn query_metric(&self, name: &str, tags: &Tags, start: i64, end: i64) -> Result<Vec<DataPoint>> {
        self.storage.query_metric(name, tags, start, end)
    }

    /// Execute a query
    pub fn execute(&self, query: &Query) -> Result<QueryResult> {
        let executor = QueryExecutor::new(&self.storage);
        executor.execute(query)
    }

    /// Get a query executor
    pub fn executor(&self) -> QueryExecutor {
        QueryExecutor::new(&self.storage)
    }

    /// Aggregate data in a time range
    pub fn aggregate(
        &self,
        series_key: SeriesKey,
        start: i64,
        end: i64,
        aggregation: Aggregation,
    ) -> Result<f64> {
        let executor = QueryExecutor::new(&self.storage);
        executor.aggregate(series_key, start, end, aggregation)
    }

    /// Downsample data in a time range
    pub fn downsample(
        &self,
        series_key: SeriesKey,
        start: i64,
        end: i64,
        interval: i64,
        aggregation: Aggregation,
    ) -> Result<Vec<(i64, f64)>> {
        let executor = QueryExecutor::new(&self.storage);
        executor.downsample(series_key, start, end, interval, aggregation)
    }

    /// Get series key for a metric
    pub fn series_key(&self, name: &str, tags: &Tags) -> SeriesKey {
        Metric::with_tags(name.to_string(), tags.clone()).series_key()
    }

    /// Find series by name prefix
    pub fn find_series(&self, name_prefix: &str) -> Vec<(SeriesKey, Metric)> {
        self.storage.find_series(name_prefix)
    }

    /// Get all series keys
    pub fn series_keys(&self) -> Vec<SeriesKey> {
        self.storage.series_keys()
    }

    /// Set retention policy
    pub fn set_retention_policy(&self, policy: RetentionPolicy) {
        self.retention.add_policy(policy);
    }

    /// Set default retention policy
    pub fn set_default_retention(&self, policy: RetentionPolicy) {
        self.retention.set_default(policy);
    }

    /// Flush all data to disk
    pub fn flush(&self) -> Result<()> {
        if let Some(ref wal) = self.wal {
            wal.sync()?;
        }
        self.storage.flush()
    }

    /// Trigger compaction
    pub fn compact(&self) -> Result<()> {
        self.storage.compact()
    }

    /// Apply retention policies
    pub fn apply_retention(&self) -> Result<usize> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;

        let drop_before = self.retention.calculate_drop_before(None, now);
        self.storage.drop_before(drop_before)
    }

    /// Get database statistics
    pub fn stats(&self) -> DatabaseStats {
        let storage_stats = self.storage.stats();
        DatabaseStats {
            series_count: storage_stats.series_count,
            point_count: storage_stats.point_count,
            shard_count: storage_stats.shard_count,
            sstable_count: storage_stats.sstable_count,
            total_size_bytes: storage_stats.total_size,
            wal_sequence: self.wal.as_ref().map(|w| w.sequence()).unwrap_or(0),
            background_error_count: self
                .bg_error_count
                .load(std::sync::atomic::Ordering::Relaxed),
        }
    }

    /// Close the database
    pub fn close(&self) -> Result<()> {
        // Signal shutdown
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);

        // Stop background tasks
        if let Some(ref sender) = self.bg_sender {
            let _ = sender.send(BackgroundMessage::Shutdown);
        }

        // Flush and sync
        self.flush()?;

        // Close WAL
        if let Some(ref wal) = self.wal {
            wal.close()?;
        }

        // Close storage
        self.storage.close()
    }
}

impl Drop for TimeSeriesDB {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Background worker function
fn background_worker(
    storage: Arc<StorageEngine>,
    retention: Arc<RetentionManager>,
    receiver: Receiver<BackgroundMessage>,
    shutdown: Arc<std::sync::atomic::AtomicBool>,
    flush_interval: Duration,
    compaction_interval: Duration,
    retention_interval: Duration,
    error_count: Arc<std::sync::atomic::AtomicU64>,
) {
    let mut last_flush = std::time::Instant::now();
    let mut last_compact = std::time::Instant::now();
    let mut last_retention = std::time::Instant::now();

    // Log a background task error and record it in the error counter.
    // The worker keeps running: a failed flush/compact/retention pass will
    // be retried on the next interval, but the failure must be visible.
    let report_error = |task: &str, err: &dyn std::fmt::Display| {
        error_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        eprintln!("[tsdb::background] {} failed: {}", task, err);
    };

    loop {
        if shutdown.load(std::sync::atomic::Ordering::Relaxed) {
            break;
        }

        // Check for messages with timeout
        match receiver.recv_timeout(Duration::from_secs(1)) {
            Ok(BackgroundMessage::Shutdown) => break,
            Ok(BackgroundMessage::Flush) => {
                if let Err(e) = storage.flush() {
                    report_error("requested flush", &e);
                }
                last_flush = std::time::Instant::now();
            }
            Ok(BackgroundMessage::Compact) => {
                if let Err(e) = storage.compact() {
                    report_error("requested compaction", &e);
                }
                last_compact = std::time::Instant::now();
            }
            Ok(BackgroundMessage::CheckRetention) => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as i64;
                let drop_before = retention.calculate_drop_before(None, now);
                if let Err(e) = storage.drop_before(drop_before) {
                    report_error("requested retention check", &e);
                }
                last_retention = std::time::Instant::now();
            }
            Err(_) => {
                // Timeout - check if we need to run periodic tasks
            }
        }

        // Periodic flush
        if last_flush.elapsed() >= flush_interval {
            if let Err(e) = storage.flush() {
                report_error("periodic flush", &e);
            }
            last_flush = std::time::Instant::now();
        }

        // Periodic compaction
        if last_compact.elapsed() >= compaction_interval {
            if let Err(e) = storage.compact() {
                report_error("periodic compaction", &e);
            }
            last_compact = std::time::Instant::now();
        }

        // Periodic retention check
        if last_retention.elapsed() >= retention_interval {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64;
            let drop_before = retention.calculate_drop_before(None, now);
            if let Err(e) = storage.drop_before(drop_before) {
                report_error("periodic retention check", &e);
            }
            last_retention = std::time::Instant::now();
        }
    }
}

/// Database statistics
#[derive(Debug, Clone, Default)]
pub struct DatabaseStats {
    /// Number of unique time series
    pub series_count: usize,
    /// Total number of data points
    pub point_count: usize,
    /// Number of time shards
    pub shard_count: usize,
    /// Number of SSTables
    pub sstable_count: usize,
    /// Total size in bytes
    pub total_size_bytes: u64,
    /// Current WAL sequence number
    pub wal_sequence: u64,
    /// Number of errors encountered by background flush/compact/retention tasks
    pub background_error_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_config(dir: &Path) -> DatabaseConfig {
        DatabaseConfig::new(dir)
            .without_wal()
            .without_background_tasks()
    }

    #[test]
    fn test_database_write_and_query() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();

        for i in 0..100 {
            db.write("cpu.usage", &tags, i * 60, i as f64).unwrap();
        }

        let series_key = db.series_key("cpu.usage", &tags);
        let points = db.query_range(series_key, 0, 6000).unwrap();

        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_database_batch_write() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        let batch: Vec<_> = (0..100)
            .map(|i| ("cpu.usage".to_string(), tags.clone(), i * 60, i as f64))
            .collect();

        db.write_batch(&batch).unwrap();

        let series_key = db.series_key("cpu.usage", &tags);
        let points = db.query_range(series_key, 0, 6000).unwrap();

        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_database_aggregation() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        for i in 0..100 {
            db.write("test", &tags, i * 60, i as f64).unwrap();
        }

        let series_key = db.series_key("test", &tags);
        let sum = db.aggregate(series_key, 0, 6000, Aggregation::Sum).unwrap();

        assert_eq!(sum, (0..100).sum::<i32>() as f64);
    }

    #[test]
    fn test_database_downsample() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        for i in 0..100 {
            db.write("test", &tags, i * 60, i as f64).unwrap();
        }

        let series_key = db.series_key("test", &tags);
        let result = db.downsample(series_key, 0, 6000, 600, Aggregation::Avg).unwrap();

        assert_eq!(result.len(), 10); // 6000 / 600 = 10 buckets
    }

    #[test]
    fn test_database_find_series() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        db.write("cpu.usage", &tags, 100, 1.0).unwrap();
        db.write("cpu.system", &tags, 100, 2.0).unwrap();
        db.write("memory.used", &tags, 100, 3.0).unwrap();

        let cpu_series = db.find_series("cpu.");
        assert_eq!(cpu_series.len(), 2);
    }

    #[test]
    fn test_database_retention() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        // Set short retention
        db.set_default_retention(RetentionPolicy::new("test", duration::DAY));

        let tags = Tags::new();
        db.write("test", &tags, 100, 1.0).unwrap();

        db.apply_retention().unwrap();
    }

    #[test]
    fn test_database_flush_compact() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        for i in 0..100 {
            db.write("test", &tags, i * 60, i as f64).unwrap();
        }

        db.flush().unwrap();
        db.compact().unwrap();

        // Data should still be accessible
        let series_key = db.series_key("test", &tags);
        let points = db.query_range(series_key, 0, 6000).unwrap();
        assert_eq!(points.len(), 100);
    }

    #[test]
    fn test_database_stats() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let stats = db.stats();
        assert_eq!(stats.series_count, 0);

        let tags = Tags::new();
        db.write("test", &tags, 100, 1.0).unwrap();

        let stats = db.stats();
        assert_eq!(stats.series_count, 1);
    }

    #[test]
    fn test_database_with_wal() {
        let dir = tempdir().unwrap();

        // Write some data
        {
            let config = DatabaseConfig::new(dir.path()).without_background_tasks();
            let db = TimeSeriesDB::open(config).unwrap();

            let tags = Tags::new();
            for i in 0..10 {
                db.write("test", &tags, i * 60, i as f64).unwrap();
            }

            db.close().unwrap();
        }

        // Reopen and verify
        {
            let config = DatabaseConfig::new(dir.path()).without_background_tasks();
            let db = TimeSeriesDB::open(config).unwrap();

            let tags = Tags::new();
            let series_key = db.series_key("test", &tags);
            let points = db.query_range(series_key, 0, 1000).unwrap();

            assert_eq!(points.len(), 10);
        }
    }

    #[test]
    fn test_database_close() {
        let dir = tempdir().unwrap();
        let db = TimeSeriesDB::open(test_config(dir.path())).unwrap();

        let tags = Tags::new();
        db.write("test", &tags, 100, 1.0).unwrap();

        db.close().unwrap();
    }

    #[test]
    fn test_database_config_builder() {
        let config = DatabaseConfig::new("/tmp/test")
            .without_wal()
            .without_background_tasks()
            .with_shard_duration(duration::DAY);

        assert!(!config.enable_wal);
        assert!(!config.enable_background_tasks);
        assert_eq!(config.shard_duration, duration::DAY);
    }
}
