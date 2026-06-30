//! Query executor for running queries against the storage engine

use std::collections::HashMap;

use crate::error::{Result, TsdbError};
use crate::types::{DataPoint, SeriesKey, Metric, Tags};
use crate::storage::StorageEngine;
use super::aggregation::{Aggregation, Aggregator, aggregate_with_interval, AggregateResult};
use super::predicate::Predicate;

/// Query definition
#[derive(Debug, Clone)]
pub struct Query {
    /// Series key(s) to query
    pub series_keys: Vec<SeriesKey>,
    /// Start timestamp
    pub start: i64,
    /// End timestamp
    pub end: i64,
    /// Optional predicate for filtering
    pub predicate: Option<Predicate>,
    /// Optional aggregation
    pub aggregation: Option<Aggregation>,
    /// Optional downsampling interval
    pub interval: Option<i64>,
    /// Limit number of results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

impl Query {
    /// Create a new query
    pub fn new(series_key: SeriesKey, start: i64, end: i64) -> Self {
        Self {
            series_keys: vec![series_key],
            start,
            end,
            predicate: None,
            aggregation: None,
            interval: None,
            limit: None,
            offset: None,
        }
    }

    /// Create a query for multiple series
    pub fn multi(series_keys: Vec<SeriesKey>, start: i64, end: i64) -> Self {
        Self {
            series_keys,
            start,
            end,
            predicate: None,
            aggregation: None,
            interval: None,
            limit: None,
            offset: None,
        }
    }

    /// Add a predicate filter
    pub fn with_predicate(mut self, predicate: Predicate) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Add an aggregation
    pub fn with_aggregation(mut self, aggregation: Aggregation) -> Self {
        self.aggregation = Some(aggregation);
        self
    }

    /// Add downsampling interval
    pub fn with_interval(mut self, interval: i64) -> Self {
        self.interval = Some(interval);
        self
    }

    /// Add limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Add offset
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }
}

/// Query result
#[derive(Debug, Clone)]
pub struct QueryResult {
    /// Results per series
    pub series: HashMap<SeriesKey, SeriesResult>,
    /// Total number of points across all series
    pub total_points: usize,
    /// Query execution time in nanoseconds
    pub execution_time_ns: u64,
}

/// Result for a single series
#[derive(Debug, Clone)]
pub struct SeriesResult {
    /// Series key
    pub series_key: SeriesKey,
    /// Data points (if not aggregated)
    pub points: Vec<DataPoint>,
    /// Aggregated values (if aggregated with interval)
    pub aggregated: Option<Vec<(i64, f64)>>,
    /// Single aggregated value (if aggregated without interval)
    pub aggregate_value: Option<f64>,
}

impl SeriesResult {
    fn new_points(series_key: SeriesKey, points: Vec<DataPoint>) -> Self {
        Self {
            series_key,
            points,
            aggregated: None,
            aggregate_value: None,
        }
    }

    fn new_aggregated(series_key: SeriesKey, aggregated: Vec<(i64, f64)>) -> Self {
        Self {
            series_key,
            points: Vec::new(),
            aggregated: Some(aggregated),
            aggregate_value: None,
        }
    }

    fn new_aggregate_value(series_key: SeriesKey, value: f64) -> Self {
        Self {
            series_key,
            points: Vec::new(),
            aggregated: None,
            aggregate_value: Some(value),
        }
    }
}

impl QueryResult {
    /// Get result for a specific series
    pub fn get(&self, series_key: SeriesKey) -> Option<&SeriesResult> {
        self.series.get(&series_key)
    }

    /// Check if result is empty
    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }

    /// Get number of series in result
    pub fn series_count(&self) -> usize {
        self.series.len()
    }
}

/// Query executor
#[derive(Debug)]
pub struct QueryExecutor<'a> {
    storage: &'a StorageEngine,
}

impl<'a> QueryExecutor<'a> {
    /// Create a new query executor
    pub fn new(storage: &'a StorageEngine) -> Self {
        Self { storage }
    }

    /// Execute a query
    pub fn execute(&self, query: &Query) -> Result<QueryResult> {
        let start_time = std::time::Instant::now();
        let mut series_results = HashMap::new();
        let mut total_points = 0;

        for &series_key in &query.series_keys {
            // Get raw points from storage
            let mut points = self.storage.query_range(series_key, query.start, query.end)?;

            // Apply predicate filter
            if let Some(ref predicate) = query.predicate {
                points = predicate.filter_owned(&points);
            }

            total_points += points.len();

            // Apply aggregation
            let result = if let Some(aggregation) = query.aggregation {
                if let Some(interval) = query.interval {
                    // Downsampled aggregation
                    let aggregated = aggregate_with_interval(
                        &points,
                        aggregation,
                        interval,
                        query.start,
                        query.end,
                    );
                    SeriesResult::new_aggregated(series_key, aggregated)
                } else {
                    // Single aggregation
                    let mut aggregator = Aggregator::new(aggregation);
                    aggregator.add_all(&points);
                    let value = aggregator.result();
                    SeriesResult::new_aggregate_value(series_key, value)
                }
            } else {
                // Apply limit and offset
                let points = apply_limit_offset(points, query.offset, query.limit);
                SeriesResult::new_points(series_key, points)
            };

            series_results.insert(series_key, result);
        }

        let execution_time_ns = start_time.elapsed().as_nanos() as u64;

        Ok(QueryResult {
            series: series_results,
            total_points,
            execution_time_ns,
        })
    }

    /// Execute a simple range query
    pub fn range(&self, series_key: SeriesKey, start: i64, end: i64) -> Result<Vec<DataPoint>> {
        self.storage.query_range(series_key, start, end)
    }

    /// Execute an aggregation query
    pub fn aggregate(
        &self,
        series_key: SeriesKey,
        start: i64,
        end: i64,
        aggregation: Aggregation,
    ) -> Result<f64> {
        let points = self.storage.query_range(series_key, start, end)?;
        let mut aggregator = Aggregator::new(aggregation);
        aggregator.add_all(&points);
        Ok(aggregator.result())
    }

    /// Execute a downsampling query
    pub fn downsample(
        &self,
        series_key: SeriesKey,
        start: i64,
        end: i64,
        interval: i64,
        aggregation: Aggregation,
    ) -> Result<Vec<(i64, f64)>> {
        let points = self.storage.query_range(series_key, start, end)?;
        Ok(aggregate_with_interval(&points, aggregation, interval, start, end))
    }

    /// Find series matching criteria
    pub fn find_series(&self, name_prefix: &str) -> Vec<(SeriesKey, Metric)> {
        self.storage.find_series(name_prefix)
    }
}

fn apply_limit_offset(
    points: Vec<DataPoint>,
    offset: Option<usize>,
    limit: Option<usize>,
) -> Vec<DataPoint> {
    let offset = offset.unwrap_or(0);
    let points: Vec<_> = points.into_iter().skip(offset).collect();

    match limit {
        Some(limit) => points.into_iter().take(limit).collect(),
        None => points,
    }
}

/// Query builder for fluent query construction
#[derive(Debug)]
pub struct QueryBuilder {
    series_keys: Vec<SeriesKey>,
    start: i64,
    end: i64,
    predicate: Option<Predicate>,
    aggregation: Option<Aggregation>,
    interval: Option<i64>,
    limit: Option<usize>,
    offset: Option<usize>,
}

impl QueryBuilder {
    /// Create a new query builder
    pub fn new() -> Self {
        Self {
            series_keys: Vec::new(),
            start: 0,
            end: i64::MAX,
            predicate: None,
            aggregation: None,
            interval: None,
            limit: None,
            offset: None,
        }
    }

    /// Add a series key
    pub fn series(mut self, series_key: SeriesKey) -> Self {
        self.series_keys.push(series_key);
        self
    }

    /// Add multiple series keys
    pub fn series_multi(mut self, series_keys: Vec<SeriesKey>) -> Self {
        self.series_keys.extend(series_keys);
        self
    }

    /// Set time range
    pub fn time_range(mut self, start: i64, end: i64) -> Self {
        self.start = start;
        self.end = end;
        self
    }

    /// Set start time
    pub fn from(mut self, start: i64) -> Self {
        self.start = start;
        self
    }

    /// Set end time
    pub fn to(mut self, end: i64) -> Self {
        self.end = end;
        self
    }

    /// Add predicate
    pub fn filter(mut self, predicate: Predicate) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Set aggregation
    pub fn aggregate(mut self, aggregation: Aggregation) -> Self {
        self.aggregation = Some(aggregation);
        self
    }

    /// Set downsampling interval
    pub fn downsample(mut self, interval: i64) -> Self {
        self.interval = Some(interval);
        self
    }

    /// Set limit
    pub fn limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set offset
    pub fn offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Build the query
    pub fn build(self) -> Result<Query> {
        if self.series_keys.is_empty() {
            return Err(TsdbError::invalid_query("No series keys specified"));
        }

        Ok(Query {
            series_keys: self.series_keys,
            start: self.start,
            end: self.end,
            predicate: self.predicate,
            aggregation: self.aggregation,
            interval: self.interval,
            limit: self.limit,
            offset: self.offset,
        })
    }
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use crate::storage::StorageConfig;
    use crate::types::Metric;

    fn setup_storage() -> (tempfile::TempDir, StorageEngine) {
        let dir = tempdir().unwrap();
        let config = StorageConfig::new(dir.path());
        let engine = StorageEngine::new(config).unwrap();
        (dir, engine)
    }

    fn insert_test_data(engine: &StorageEngine, metric: &Metric, count: usize) {
        for i in 0..count {
            let point = DataPoint::new(i as i64 * 60, i as f64);
            engine.write_points(&[(metric.clone(), point)]).unwrap();
        }
    }

    #[test]
    fn test_basic_query() {
        let (_dir, engine) = setup_storage();
        let metric = Metric::new("test");
        insert_test_data(&engine, &metric, 100);

        let executor = QueryExecutor::new(&engine);
        let points = executor.range(metric.series_key(), 0, 3000).unwrap();

        assert_eq!(points.len(), 51); // 0-50 minutes (inclusive range)
    }

    #[test]
    fn test_query_with_predicate() {
        let (_dir, engine) = setup_storage();
        let metric = Metric::new("test");
        insert_test_data(&engine, &metric, 100);

        let query = Query::new(metric.series_key(), 0, 6000)
            .with_predicate(Predicate::value_gt(25.0));

        let executor = QueryExecutor::new(&engine);
        let result = executor.execute(&query).unwrap();

        let series_result = result.get(metric.series_key()).unwrap();
        assert!(series_result.points.iter().all(|p| p.value > 25.0));
    }

    #[test]
    fn test_query_with_aggregation() {
        let (_dir, engine) = setup_storage();
        let metric = Metric::new("test");
        insert_test_data(&engine, &metric, 100);

        let executor = QueryExecutor::new(&engine);
        let sum = executor
            .aggregate(metric.series_key(), 0, 6000, Aggregation::Sum)
            .unwrap();

        // Sum of 0..100 = 4950
        assert_eq!(sum, (0..100).sum::<i32>() as f64);
    }

    #[test]
    fn test_query_with_downsampling() {
        let (_dir, engine) = setup_storage();
        let metric = Metric::new("test");
        insert_test_data(&engine, &metric, 100);

        let executor = QueryExecutor::new(&engine);
        let result = executor
            .downsample(metric.series_key(), 0, 6000, 600, Aggregation::Avg)
            .unwrap();

        // 6000 / 600 = 10 buckets
        assert_eq!(result.len(), 10);
    }

    #[test]
    fn test_query_with_limit_offset() {
        let (_dir, engine) = setup_storage();
        let metric = Metric::new("test");
        insert_test_data(&engine, &metric, 100);

        let query = Query::new(metric.series_key(), 0, 6000)
            .with_limit(10)
            .with_offset(5);

        let executor = QueryExecutor::new(&engine);
        let result = executor.execute(&query).unwrap();

        let series_result = result.get(metric.series_key()).unwrap();
        assert_eq!(series_result.points.len(), 10);
        assert_eq!(series_result.points[0].value, 5.0); // Offset by 5
    }

    #[test]
    fn test_multi_series_query() {
        let (_dir, engine) = setup_storage();
        let metric1 = Metric::new("metric1");
        let metric2 = Metric::new("metric2");

        insert_test_data(&engine, &metric1, 50);
        insert_test_data(&engine, &metric2, 50);

        let query = Query::multi(
            vec![metric1.series_key(), metric2.series_key()],
            0,
            3000,
        );

        let executor = QueryExecutor::new(&engine);
        let result = executor.execute(&query).unwrap();

        assert_eq!(result.series_count(), 2);
    }

    #[test]
    fn test_query_builder() {
        let (_dir, engine) = setup_storage();
        let metric = Metric::new("test");
        insert_test_data(&engine, &metric, 100);

        let query = QueryBuilder::new()
            .series(metric.series_key())
            .time_range(0, 3000)
            .aggregate(Aggregation::Avg)
            .build()
            .unwrap();

        let executor = QueryExecutor::new(&engine);
        let result = executor.execute(&query).unwrap();

        let series_result = result.get(metric.series_key()).unwrap();
        assert!(series_result.aggregate_value.is_some());
    }

    #[test]
    fn test_query_builder_validation() {
        let result = QueryBuilder::new()
            .time_range(0, 1000)
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_find_series() {
        let (_dir, engine) = setup_storage();

        let metric1 = Metric::new("cpu.usage");
        let metric2 = Metric::new("cpu.system");
        let metric3 = Metric::new("memory.used");

        engine.write_points(&[(metric1.clone(), DataPoint::new(100, 1.0))]).unwrap();
        engine.write_points(&[(metric2.clone(), DataPoint::new(100, 2.0))]).unwrap();
        engine.write_points(&[(metric3.clone(), DataPoint::new(100, 3.0))]).unwrap();

        let executor = QueryExecutor::new(&engine);
        let cpu_series = executor.find_series("cpu.");

        assert_eq!(cpu_series.len(), 2);
    }

    #[test]
    fn test_query_result_methods() {
        let result = QueryResult {
            series: HashMap::new(),
            total_points: 0,
            execution_time_ns: 100,
        };

        assert!(result.is_empty());
        assert_eq!(result.series_count(), 0);
        assert!(result.get(12345).is_none());
    }
}
