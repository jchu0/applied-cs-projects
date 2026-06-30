//! Aggregation functions for time-series data

use crate::types::DataPoint;

/// Aggregation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aggregation {
    /// Sum of all values
    Sum,
    /// Average of all values
    Avg,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
    /// Count of values
    Count,
    /// First value
    First,
    /// Last value
    Last,
    /// Standard deviation
    StdDev,
    /// Variance
    Variance,
    /// Percentile (0-100)
    Percentile(u8),
    /// Rate of change per second
    Rate,
    /// Difference between first and last
    Delta,
}

/// Aggregation query
#[derive(Debug, Clone)]
pub struct AggregateQuery {
    /// Start timestamp
    pub start: i64,
    /// End timestamp
    pub end: i64,
    /// Aggregation function
    pub aggregation: Aggregation,
    /// Downsampling interval (optional)
    pub interval: Option<i64>,
}

impl AggregateQuery {
    pub fn new(start: i64, end: i64, aggregation: Aggregation) -> Self {
        Self {
            start,
            end,
            aggregation,
            interval: None,
        }
    }

    pub fn with_interval(mut self, interval: i64) -> Self {
        self.interval = Some(interval);
        self
    }
}

/// Result of an aggregation query
#[derive(Debug, Clone)]
pub struct AggregateResult {
    /// Aggregated values (one per interval, or single value if no interval)
    pub values: Vec<(i64, f64)>,
    /// The aggregation function used
    pub aggregation: Aggregation,
    /// Number of points that were aggregated
    pub point_count: usize,
}

impl AggregateResult {
    /// Get the single aggregated value (if no interval was used)
    pub fn value(&self) -> Option<f64> {
        self.values.first().map(|(_, v)| *v)
    }
}

/// Stateful aggregator for computing aggregations incrementally
#[derive(Debug)]
pub struct Aggregator {
    aggregation: Aggregation,
    sum: f64,
    count: usize,
    min: f64,
    max: f64,
    first: Option<(i64, f64)>,
    last: Option<(i64, f64)>,
    values: Vec<f64>, // For percentile and stddev
    sum_squared: f64, // For variance
}

impl Aggregator {
    /// Create a new aggregator
    pub fn new(aggregation: Aggregation) -> Self {
        Self {
            aggregation,
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            first: None,
            last: None,
            values: Vec::new(),
            sum_squared: 0.0,
        }
    }

    /// Add a data point to the aggregation
    pub fn add(&mut self, point: &DataPoint) {
        self.sum += point.value;
        self.sum_squared += point.value * point.value;
        self.count += 1;

        if point.value < self.min {
            self.min = point.value;
        }
        if point.value > self.max {
            self.max = point.value;
        }

        if self.first.is_none() || point.timestamp < self.first.unwrap().0 {
            self.first = Some((point.timestamp, point.value));
        }
        if self.last.is_none() || point.timestamp > self.last.unwrap().0 {
            self.last = Some((point.timestamp, point.value));
        }

        // Store values for percentile and stddev
        match self.aggregation {
            Aggregation::Percentile(_) | Aggregation::StdDev | Aggregation::Variance => {
                self.values.push(point.value);
            }
            _ => {}
        }
    }

    /// Add multiple data points
    pub fn add_all(&mut self, points: &[DataPoint]) {
        for point in points {
            self.add(point);
        }
    }

    /// Get the aggregated result
    pub fn result(&mut self) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }

        match self.aggregation {
            Aggregation::Sum => self.sum,
            Aggregation::Avg => self.sum / self.count as f64,
            Aggregation::Min => self.min,
            Aggregation::Max => self.max,
            Aggregation::Count => self.count as f64,
            Aggregation::First => self.first.map(|(_, v)| v).unwrap_or(f64::NAN),
            Aggregation::Last => self.last.map(|(_, v)| v).unwrap_or(f64::NAN),
            Aggregation::StdDev => {
                let mean = self.sum / self.count as f64;
                let variance: f64 = self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / self.count as f64;
                variance.sqrt()
            }
            Aggregation::Variance => {
                let mean = self.sum / self.count as f64;
                self.values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / self.count as f64
            }
            Aggregation::Percentile(p) => {
                self.values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((p as f64 / 100.0) * (self.values.len() - 1) as f64).round() as usize;
                self.values.get(idx).copied().unwrap_or(f64::NAN)
            }
            Aggregation::Rate => {
                if let (Some((first_ts, first_val)), Some((last_ts, last_val))) =
                    (self.first, self.last)
                {
                    if last_ts > first_ts {
                        (last_val - first_val) / ((last_ts - first_ts) as f64 / 1_000_000_000.0)
                    } else {
                        0.0
                    }
                } else {
                    f64::NAN
                }
            }
            Aggregation::Delta => {
                if let (Some((_, first_val)), Some((_, last_val))) = (self.first, self.last) {
                    last_val - first_val
                } else {
                    f64::NAN
                }
            }
        }
    }

    /// Reset the aggregator for reuse
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.first = None;
        self.last = None;
        self.values.clear();
        self.sum_squared = 0.0;
    }

    /// Get the point count
    pub fn count(&self) -> usize {
        self.count
    }
}

/// Aggregate points with downsampling
pub fn aggregate_with_interval(
    points: &[DataPoint],
    aggregation: Aggregation,
    interval: i64,
    start: i64,
    end: i64,
) -> Vec<(i64, f64)> {
    if points.is_empty() || interval <= 0 {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut bucket_start = start;

    while bucket_start < end {
        let bucket_end = bucket_start + interval;
        let mut aggregator = Aggregator::new(aggregation);

        for point in points {
            if point.timestamp >= bucket_start && point.timestamp < bucket_end {
                aggregator.add(point);
            }
        }

        if aggregator.count() > 0 {
            result.push((bucket_start, aggregator.result()));
        }

        bucket_start = bucket_end;
    }

    result
}

/// Compute a simple aggregation over points
pub fn aggregate(points: &[DataPoint], aggregation: Aggregation) -> f64 {
    let mut aggregator = Aggregator::new(aggregation);
    aggregator.add_all(points);
    aggregator.result()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_points() -> Vec<DataPoint> {
        vec![
            DataPoint::new(1000, 10.0),
            DataPoint::new(2000, 20.0),
            DataPoint::new(3000, 30.0),
            DataPoint::new(4000, 40.0),
            DataPoint::new(5000, 50.0),
        ]
    }

    #[test]
    fn test_sum() {
        let points = test_points();
        let result = aggregate(&points, Aggregation::Sum);
        assert_eq!(result, 150.0);
    }

    #[test]
    fn test_avg() {
        let points = test_points();
        let result = aggregate(&points, Aggregation::Avg);
        assert_eq!(result, 30.0);
    }

    #[test]
    fn test_min() {
        let points = test_points();
        let result = aggregate(&points, Aggregation::Min);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_max() {
        let points = test_points();
        let result = aggregate(&points, Aggregation::Max);
        assert_eq!(result, 50.0);
    }

    #[test]
    fn test_count() {
        let points = test_points();
        let result = aggregate(&points, Aggregation::Count);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_first() {
        let points = test_points();
        let result = aggregate(&points, Aggregation::First);
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_last() {
        let points = test_points();
        let result = aggregate(&points, Aggregation::Last);
        assert_eq!(result, 50.0);
    }

    #[test]
    fn test_delta() {
        let points = test_points();
        let result = aggregate(&points, Aggregation::Delta);
        assert_eq!(result, 40.0); // 50 - 10
    }

    #[test]
    fn test_rate() {
        let points = vec![
            DataPoint::new(0, 0.0),
            DataPoint::new(1_000_000_000, 10.0), // 1 second later
        ];
        let result = aggregate(&points, Aggregation::Rate);
        assert_eq!(result, 10.0); // 10 per second
    }

    #[test]
    fn test_percentile() {
        let points: Vec<DataPoint> = (1..=100)
            .map(|i| DataPoint::new(i as i64, i as f64))
            .collect();

        let p50 = aggregate(&points, Aggregation::Percentile(50));
        assert!((p50 - 50.0).abs() <= 2.0, "p50={p50}");

        let p90 = aggregate(&points, Aggregation::Percentile(90));
        assert!((p90 - 90.0).abs() <= 2.0, "p90={p90}");
    }

    #[test]
    fn test_stddev() {
        let points = vec![
            DataPoint::new(1, 2.0),
            DataPoint::new(2, 4.0),
            DataPoint::new(3, 4.0),
            DataPoint::new(4, 4.0),
            DataPoint::new(5, 5.0),
            DataPoint::new(6, 5.0),
            DataPoint::new(7, 7.0),
            DataPoint::new(8, 9.0),
        ];

        let result = aggregate(&points, Aggregation::StdDev);
        assert!((result - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_variance() {
        let points = vec![
            DataPoint::new(1, 2.0),
            DataPoint::new(2, 4.0),
            DataPoint::new(3, 4.0),
            DataPoint::new(4, 4.0),
            DataPoint::new(5, 5.0),
            DataPoint::new(6, 5.0),
            DataPoint::new(7, 7.0),
            DataPoint::new(8, 9.0),
        ];

        let result = aggregate(&points, Aggregation::Variance);
        assert!((result - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_aggregate_with_interval() {
        let points: Vec<DataPoint> = (0..100)
            .map(|i| DataPoint::new(i * 10, i as f64))
            .collect();

        let result = aggregate_with_interval(&points, Aggregation::Sum, 100, 0, 1000);

        assert_eq!(result.len(), 10);
        // First bucket: 0, 1, 2, ..., 9 = sum of 0+1+2+...+9 = 45
        assert_eq!(result[0].1, 45.0);
    }

    #[test]
    fn test_empty_aggregation() {
        let points: Vec<DataPoint> = vec![];
        let result = aggregate(&points, Aggregation::Sum);
        assert!(result.is_nan());
    }

    #[test]
    fn test_aggregator_reset() {
        let mut aggregator = Aggregator::new(Aggregation::Sum);

        aggregator.add(&DataPoint::new(100, 10.0));
        assert_eq!(aggregator.result(), 10.0);

        aggregator.reset();
        aggregator.add(&DataPoint::new(200, 20.0));
        assert_eq!(aggregator.result(), 20.0);
    }

    #[test]
    fn test_aggregate_query() {
        let query = AggregateQuery::new(0, 1000, Aggregation::Avg).with_interval(100);

        assert_eq!(query.start, 0);
        assert_eq!(query.end, 1000);
        assert_eq!(query.aggregation, Aggregation::Avg);
        assert_eq!(query.interval, Some(100));
    }

    #[test]
    fn test_aggregate_result_value() {
        let result = AggregateResult {
            values: vec![(0, 42.0)],
            aggregation: Aggregation::Sum,
            point_count: 5,
        };

        assert_eq!(result.value(), Some(42.0));
    }
}
