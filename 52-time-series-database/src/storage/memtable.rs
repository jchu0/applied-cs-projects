//! In-memory table for buffering recent writes
//!
//! The MemTable provides fast writes and reads for recent data.
//! When it reaches a size threshold, it's flushed to an SSTable.

use std::collections::BTreeMap;
use parking_lot::RwLock;
use crate::error::Result;
use crate::types::{DataPoint, SeriesKey, Metric, Series};

/// In-memory table for buffering writes
#[derive(Debug)]
pub struct MemTable {
    /// Data organized by series key, then by timestamp
    data: RwLock<BTreeMap<SeriesKey, Series>>,
    /// Approximate size in bytes
    size: std::sync::atomic::AtomicUsize,
    /// Maximum size before flush
    max_size: usize,
}

impl MemTable {
    /// Create a new memtable with default max size (64MB)
    pub fn new() -> Self {
        Self::with_max_size(64 * 1024 * 1024)
    }

    /// Create a memtable with custom max size
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            data: RwLock::new(BTreeMap::new()),
            size: std::sync::atomic::AtomicUsize::new(0),
            max_size,
        }
    }

    /// Insert a data point
    pub fn insert(&self, metric: &Metric, point: DataPoint) -> Result<()> {
        let key = metric.series_key();
        let mut data = self.data.write();

        let series = data.entry(key).or_insert_with(|| Series::new(metric.clone()));
        series.insert_sorted(point);

        // Approximate size increase
        let point_size = std::mem::size_of::<DataPoint>();
        self.size.fetch_add(point_size, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Insert multiple data points
    pub fn insert_batch(&self, points: &[(Metric, DataPoint)]) -> Result<()> {
        let mut data = self.data.write();

        for (metric, point) in points {
            let key = metric.series_key();
            let series = data.entry(key).or_insert_with(|| Series::new(metric.clone()));
            series.insert_sorted(*point);
        }

        let added_size = points.len() * std::mem::size_of::<DataPoint>();
        self.size.fetch_add(added_size, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }

    /// Query points in a time range for a specific series
    pub fn query_range(&self, series_key: SeriesKey, start: i64, end: i64) -> Vec<DataPoint> {
        let data = self.data.read();

        if let Some(series) = data.get(&series_key) {
            series.range(start, end).to_vec()
        } else {
            Vec::new()
        }
    }

    /// Query all points for a series
    pub fn query_series(&self, series_key: SeriesKey) -> Option<Vec<DataPoint>> {
        let data = self.data.read();
        data.get(&series_key).map(|s| s.points.clone())
    }

    /// Get all series in the memtable
    pub fn all_series(&self) -> Vec<Series> {
        let data = self.data.read();
        data.values().cloned().collect()
    }

    /// Get all series keys
    pub fn series_keys(&self) -> Vec<SeriesKey> {
        let data = self.data.read();
        data.keys().copied().collect()
    }

    /// Check if the memtable should be flushed
    pub fn should_flush(&self) -> bool {
        self.size.load(std::sync::atomic::Ordering::Relaxed) >= self.max_size
    }

    /// Get approximate size in bytes
    pub fn size(&self) -> usize {
        self.size.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get number of series
    pub fn series_count(&self) -> usize {
        self.data.read().len()
    }

    /// Get total number of points
    pub fn point_count(&self) -> usize {
        self.data.read().values().map(|s| s.len()).sum()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.read().is_empty()
    }

    /// Clear all data
    pub fn clear(&self) {
        let mut data = self.data.write();
        data.clear();
        self.size.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Take all data, leaving the memtable empty
    pub fn take(&self) -> BTreeMap<SeriesKey, Series> {
        let mut data = self.data.write();
        self.size.store(0, std::sync::atomic::Ordering::Relaxed);
        std::mem::take(&mut *data)
    }

    /// Get the minimum timestamp across all series
    pub fn min_timestamp(&self) -> Option<i64> {
        let data = self.data.read();
        data.values().filter_map(|s| s.first_timestamp()).min()
    }

    /// Get the maximum timestamp across all series
    pub fn max_timestamp(&self) -> Option<i64> {
        let data = self.data.read();
        data.values().filter_map(|s| s.last_timestamp()).max()
    }

    /// Check if a series exists
    pub fn contains_series(&self, series_key: SeriesKey) -> bool {
        self.data.read().contains_key(&series_key)
    }

    /// Get metric info for a series
    pub fn get_metric(&self, series_key: SeriesKey) -> Option<Metric> {
        self.data.read().get(&series_key).map(|s| s.metric.clone())
    }
}

impl Default for MemTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metric() -> Metric {
        Metric::new("test.metric").tag("host", "server1")
    }

    #[test]
    fn test_insert_and_query() {
        let memtable = MemTable::new();
        let metric = test_metric();

        memtable.insert(&metric, DataPoint::new(1000, 1.0)).unwrap();
        memtable.insert(&metric, DataPoint::new(2000, 2.0)).unwrap();
        memtable.insert(&metric, DataPoint::new(3000, 3.0)).unwrap();

        let points = memtable.query_range(metric.series_key(), 0, 5000);
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].value, 1.0);
        assert_eq!(points[1].value, 2.0);
        assert_eq!(points[2].value, 3.0);
    }

    #[test]
    fn test_query_range() {
        let memtable = MemTable::new();
        let metric = test_metric();

        for i in 0..100 {
            memtable.insert(&metric, DataPoint::new(i * 100, i as f64)).unwrap();
        }

        let points = memtable.query_range(metric.series_key(), 2000, 5000);
        assert_eq!(points.len(), 31); // 20, 21, ..., 50

        let first = points.first().unwrap();
        let last = points.last().unwrap();
        assert_eq!(first.timestamp, 2000);
        assert_eq!(last.timestamp, 5000);
    }

    #[test]
    fn test_insert_out_of_order() {
        let memtable = MemTable::new();
        let metric = test_metric();

        memtable.insert(&metric, DataPoint::new(3000, 3.0)).unwrap();
        memtable.insert(&metric, DataPoint::new(1000, 1.0)).unwrap();
        memtable.insert(&metric, DataPoint::new(2000, 2.0)).unwrap();

        let points = memtable.query_series(metric.series_key()).unwrap();
        assert_eq!(points[0].timestamp, 1000);
        assert_eq!(points[1].timestamp, 2000);
        assert_eq!(points[2].timestamp, 3000);
    }

    #[test]
    fn test_batch_insert() {
        let memtable = MemTable::new();
        let metric = test_metric();

        let points: Vec<_> = (0..100)
            .map(|i| (metric.clone(), DataPoint::new(i * 60, i as f64)))
            .collect();

        memtable.insert_batch(&points).unwrap();

        assert_eq!(memtable.point_count(), 100);
        assert_eq!(memtable.series_count(), 1);
    }

    #[test]
    fn test_multiple_series() {
        let memtable = MemTable::new();

        let metric1 = Metric::new("cpu.usage").tag("host", "server1");
        let metric2 = Metric::new("cpu.usage").tag("host", "server2");

        memtable.insert(&metric1, DataPoint::new(1000, 50.0)).unwrap();
        memtable.insert(&metric2, DataPoint::new(1000, 60.0)).unwrap();

        assert_eq!(memtable.series_count(), 2);

        let points1 = memtable.query_series(metric1.series_key()).unwrap();
        let points2 = memtable.query_series(metric2.series_key()).unwrap();

        assert_eq!(points1[0].value, 50.0);
        assert_eq!(points2[0].value, 60.0);
    }

    #[test]
    fn test_should_flush() {
        let memtable = MemTable::with_max_size(1000);

        assert!(!memtable.should_flush());

        // Insert enough points to exceed size
        let metric = test_metric();
        for i in 0..100 {
            memtable.insert(&metric, DataPoint::new(i, i as f64)).unwrap();
        }

        assert!(memtable.should_flush());
    }

    #[test]
    fn test_clear() {
        let memtable = MemTable::new();
        let metric = test_metric();

        memtable.insert(&metric, DataPoint::new(1000, 1.0)).unwrap();
        assert!(!memtable.is_empty());

        memtable.clear();
        assert!(memtable.is_empty());
        assert_eq!(memtable.size(), 0);
    }

    #[test]
    fn test_take() {
        let memtable = MemTable::new();
        let metric = test_metric();

        memtable.insert(&metric, DataPoint::new(1000, 1.0)).unwrap();

        let data = memtable.take();
        assert_eq!(data.len(), 1);
        assert!(memtable.is_empty());
    }

    #[test]
    fn test_min_max_timestamp() {
        let memtable = MemTable::new();
        let metric = test_metric();

        memtable.insert(&metric, DataPoint::new(1000, 1.0)).unwrap();
        memtable.insert(&metric, DataPoint::new(5000, 5.0)).unwrap();
        memtable.insert(&metric, DataPoint::new(3000, 3.0)).unwrap();

        assert_eq!(memtable.min_timestamp(), Some(1000));
        assert_eq!(memtable.max_timestamp(), Some(5000));
    }

    #[test]
    fn test_empty_memtable() {
        let memtable = MemTable::new();

        assert!(memtable.is_empty());
        assert_eq!(memtable.min_timestamp(), None);
        assert_eq!(memtable.max_timestamp(), None);
        assert!(memtable.query_series(123).is_none());
    }

    #[test]
    fn test_all_series() {
        let memtable = MemTable::new();

        let metric1 = Metric::new("metric1");
        let metric2 = Metric::new("metric2");

        memtable.insert(&metric1, DataPoint::new(1000, 1.0)).unwrap();
        memtable.insert(&metric2, DataPoint::new(1000, 2.0)).unwrap();

        let series = memtable.all_series();
        assert_eq!(series.len(), 2);
    }

    #[test]
    fn test_get_metric() {
        let memtable = MemTable::new();
        let metric = Metric::new("test").tag("env", "prod");

        memtable.insert(&metric, DataPoint::new(1000, 1.0)).unwrap();

        let retrieved = memtable.get_metric(metric.series_key()).unwrap();
        assert_eq!(retrieved.name, "test");
        assert_eq!(retrieved.tags.get("env"), Some(&"prod".to_string()));
    }

    #[test]
    fn test_contains_series() {
        let memtable = MemTable::new();
        let metric = test_metric();

        assert!(!memtable.contains_series(metric.series_key()));

        memtable.insert(&metric, DataPoint::new(1000, 1.0)).unwrap();

        assert!(memtable.contains_series(metric.series_key()));
    }
}
