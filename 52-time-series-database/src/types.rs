//! Core types for the time-series database

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use fnv::FnvHasher;

/// Tags are key-value pairs associated with a metric
pub type Tags = BTreeMap<String, String>;

/// Series key is a unique identifier for a time series
pub type SeriesKey = u64;

/// A single data point in a time series
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DataPoint {
    /// Unix timestamp in nanoseconds
    pub timestamp: i64,
    /// The metric value
    pub value: f64,
}

impl DataPoint {
    /// Create a new data point
    pub fn new(timestamp: i64, value: f64) -> Self {
        Self { timestamp, value }
    }

    /// Create a data point at the current time
    pub fn now(value: f64) -> Self {
        Self {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as i64,
            value,
        }
    }
}

impl Default for DataPoint {
    fn default() -> Self {
        Self {
            timestamp: 0,
            value: 0.0,
        }
    }
}

/// A metric represents a named measurement with optional tags
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Metric {
    /// The metric name (e.g., "cpu.usage", "memory.used")
    pub name: String,
    /// Tags associated with this metric
    pub tags: Tags,
}

impl Metric {
    /// Create a new metric with the given name
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            tags: Tags::new(),
        }
    }

    /// Create a metric with name and tags
    pub fn with_tags<S: Into<String>>(name: S, tags: Tags) -> Self {
        Self {
            name: name.into(),
            tags,
        }
    }

    /// Add a tag to this metric
    pub fn tag<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Compute the series key for this metric
    pub fn series_key(&self) -> SeriesKey {
        compute_series_key(&self.name, &self.tags)
    }
}

impl Hash for Metric {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        for (k, v) in &self.tags {
            k.hash(state);
            v.hash(state);
        }
    }
}

/// Compute a series key from metric name and tags
pub fn compute_series_key(name: &str, tags: &Tags) -> SeriesKey {
    let mut hasher = FnvHasher::default();
    name.hash(&mut hasher);
    for (k, v) in tags {
        k.hash(&mut hasher);
        v.hash(&mut hasher);
    }
    hasher.finish()
}

/// A time series is a collection of data points for a single metric
#[derive(Debug, Clone)]
pub struct Series {
    /// The unique series key
    pub key: SeriesKey,
    /// The metric this series represents
    pub metric: Metric,
    /// Data points in this series (sorted by timestamp)
    pub points: Vec<DataPoint>,
}

impl Series {
    /// Create a new empty series
    pub fn new(metric: Metric) -> Self {
        let key = metric.series_key();
        Self {
            key,
            metric,
            points: Vec::new(),
        }
    }

    /// Create a series with pre-allocated capacity
    pub fn with_capacity(metric: Metric, capacity: usize) -> Self {
        let key = metric.series_key();
        Self {
            key,
            metric,
            points: Vec::with_capacity(capacity),
        }
    }

    /// Add a data point to the series
    pub fn push(&mut self, point: DataPoint) {
        self.points.push(point);
    }

    /// Add a data point, maintaining sorted order
    pub fn insert_sorted(&mut self, point: DataPoint) {
        match self.points.binary_search_by_key(&point.timestamp, |p| p.timestamp) {
            Ok(idx) => self.points[idx] = point, // Replace existing
            Err(idx) => self.points.insert(idx, point),
        }
    }

    /// Get points in a time range
    pub fn range(&self, start: i64, end: i64) -> &[DataPoint] {
        let start_idx = self.points.partition_point(|p| p.timestamp < start);
        let end_idx = self.points.partition_point(|p| p.timestamp <= end);
        &self.points[start_idx..end_idx]
    }

    /// Get the number of points in this series
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if this series is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Get the first timestamp in this series
    pub fn first_timestamp(&self) -> Option<i64> {
        self.points.first().map(|p| p.timestamp)
    }

    /// Get the last timestamp in this series
    pub fn last_timestamp(&self) -> Option<i64> {
        self.points.last().map(|p| p.timestamp)
    }

    /// Sort points by timestamp
    pub fn sort(&mut self) {
        self.points.sort_by_key(|p| p.timestamp);
    }

    /// Remove duplicate timestamps, keeping the last value
    pub fn deduplicate(&mut self) {
        self.points.dedup_by_key(|p| p.timestamp);
    }
}

/// A batch of data points for efficient writing
#[derive(Debug, Clone)]
pub struct WriteBatch {
    /// Points grouped by series key
    pub points: Vec<(Metric, DataPoint)>,
}

impl WriteBatch {
    /// Create a new empty write batch
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Create a write batch with capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    /// Add a point to the batch
    pub fn push(&mut self, metric: Metric, point: DataPoint) {
        self.points.push((metric, point));
    }

    /// Get the number of points in this batch
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if this batch is empty
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Clear the batch
    pub fn clear(&mut self) {
        self.points.clear();
    }
}

impl Default for WriteBatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Time duration constants (in nanoseconds)
pub mod duration {
    pub const NANOSECOND: i64 = 1;
    pub const MICROSECOND: i64 = 1_000;
    pub const MILLISECOND: i64 = 1_000_000;
    pub const SECOND: i64 = 1_000_000_000;
    pub const MINUTE: i64 = 60 * SECOND;
    pub const HOUR: i64 = 60 * MINUTE;
    pub const DAY: i64 = 24 * HOUR;
    pub const WEEK: i64 = 7 * DAY;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_point_creation() {
        let dp = DataPoint::new(1000, 42.5);
        assert_eq!(dp.timestamp, 1000);
        assert_eq!(dp.value, 42.5);
    }

    #[test]
    fn test_metric_creation() {
        let metric = Metric::new("cpu.usage")
            .tag("host", "server1")
            .tag("region", "us-east");

        assert_eq!(metric.name, "cpu.usage");
        assert_eq!(metric.tags.get("host"), Some(&"server1".to_string()));
        assert_eq!(metric.tags.get("region"), Some(&"us-east".to_string()));
    }

    #[test]
    fn test_series_key_consistency() {
        let metric1 = Metric::new("test").tag("a", "1").tag("b", "2");
        let metric2 = Metric::new("test").tag("a", "1").tag("b", "2");
        let metric3 = Metric::new("test").tag("b", "2").tag("a", "1");

        assert_eq!(metric1.series_key(), metric2.series_key());
        // BTreeMap maintains order, so same tags in different order = same key
        assert_eq!(metric1.series_key(), metric3.series_key());
    }

    #[test]
    fn test_series_range() {
        let mut series = Series::new(Metric::new("test"));
        for i in 0..100 {
            series.push(DataPoint::new(i * 10, i as f64));
        }

        let range = series.range(250, 500);
        assert_eq!(range.len(), 26); // 25, 26, ..., 50
        assert_eq!(range[0].timestamp, 250);
        assert_eq!(range[range.len() - 1].timestamp, 500);
    }

    #[test]
    fn test_series_insert_sorted() {
        let mut series = Series::new(Metric::new("test"));
        series.insert_sorted(DataPoint::new(100, 1.0));
        series.insert_sorted(DataPoint::new(50, 0.5));
        series.insert_sorted(DataPoint::new(150, 1.5));
        series.insert_sorted(DataPoint::new(75, 0.75));

        assert_eq!(series.points[0].timestamp, 50);
        assert_eq!(series.points[1].timestamp, 75);
        assert_eq!(series.points[2].timestamp, 100);
        assert_eq!(series.points[3].timestamp, 150);
    }

    #[test]
    fn test_write_batch() {
        let mut batch = WriteBatch::new();
        batch.push(Metric::new("test"), DataPoint::new(100, 1.0));
        batch.push(Metric::new("test2"), DataPoint::new(200, 2.0));

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }
}
