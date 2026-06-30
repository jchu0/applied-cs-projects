//! Time-series specific functions
//!
//! Provides Prometheus-compatible functions for counter/gauge metrics:
//! - rate(): per-second rate of increase
//! - increase(): total increase over range
//! - irate(): instant rate using last two points
//! - delta(): difference between first and last
//! - idelta(): instant delta using last two points
//! - deriv(): linear regression derivative
//! - predict_linear(): linear prediction

use crate::types::DataPoint;

/// Calculate the per-second average rate of increase over a time range.
/// For counter metrics that can reset to 0.
///
/// This function handles counter resets by detecting when the value decreases
/// and treating it as a reset (the counter wrapped or was restarted).
pub fn rate(points: &[DataPoint]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let first = &points[0];
    let last = &points[points.len() - 1];

    // Calculate duration in seconds
    let duration_secs = (last.timestamp - first.timestamp) as f64 / 1_000_000_000.0;
    if duration_secs <= 0.0 {
        return None;
    }

    // Calculate increase handling resets
    let increase = counter_increase(points);

    Some(increase / duration_secs)
}

/// Calculate the total increase over a time range.
/// For counter metrics that can reset to 0.
pub fn increase(points: &[DataPoint]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    Some(counter_increase(points))
}

/// Calculate the instant rate using the last two data points.
/// More responsive to recent changes than rate().
pub fn irate(points: &[DataPoint]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let len = points.len();
    let prev = &points[len - 2];
    let curr = &points[len - 1];

    let duration_secs = (curr.timestamp - prev.timestamp) as f64 / 1_000_000_000.0;
    if duration_secs <= 0.0 {
        return None;
    }

    // Handle potential counter reset
    let increase = if curr.value < prev.value {
        curr.value // Reset occurred, increase is just the current value
    } else {
        curr.value - prev.value
    };

    Some(increase / duration_secs)
}

/// Calculate the difference between the first and last value.
/// For gauge metrics (not counters).
pub fn delta(points: &[DataPoint]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let first = &points[0];
    let last = &points[points.len() - 1];

    Some(last.value - first.value)
}

/// Calculate the instant delta using the last two data points.
/// For gauge metrics.
pub fn idelta(points: &[DataPoint]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let len = points.len();
    let prev = &points[len - 2];
    let curr = &points[len - 1];

    Some(curr.value - prev.value)
}

/// Calculate the per-second derivative using linear regression.
/// For gauge metrics.
pub fn deriv(points: &[DataPoint]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let (slope, _) = linear_regression(points)?;

    // Slope is per-nanosecond, convert to per-second
    Some(slope * 1_000_000_000.0)
}

/// Predict the value at a future time using linear regression.
///
/// # Arguments
/// * `points` - Historical data points
/// * `seconds_ahead` - How many seconds in the future to predict
pub fn predict_linear(points: &[DataPoint], seconds_ahead: f64) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }

    let (slope, intercept) = linear_regression(points)?;
    let last_timestamp = points[points.len() - 1].timestamp;

    // Predict at future timestamp
    let future_timestamp = last_timestamp + (seconds_ahead * 1_000_000_000.0) as i64;
    Some(slope * future_timestamp as f64 + intercept)
}

/// Calculate the number of times the value has changed.
pub fn changes(points: &[DataPoint]) -> usize {
    if points.len() < 2 {
        return 0;
    }

    let mut count = 0;
    for i in 1..points.len() {
        if (points[i].value - points[i - 1].value).abs() > f64::EPSILON {
            count += 1;
        }
    }
    count
}

/// Count the number of times a counter has been reset.
pub fn resets(points: &[DataPoint]) -> usize {
    if points.len() < 2 {
        return 0;
    }

    let mut count = 0;
    for i in 1..points.len() {
        if points[i].value < points[i - 1].value {
            count += 1;
        }
    }
    count
}

/// Calculate the time since the metric had a certain value.
/// Returns None if the value is never seen.
pub fn time_since_value(points: &[DataPoint], target_value: f64, current_time: i64) -> Option<i64> {
    for point in points.iter().rev() {
        if (point.value - target_value).abs() < f64::EPSILON {
            return Some(current_time - point.timestamp);
        }
    }
    None
}

/// Calculate the total counter increase handling resets.
fn counter_increase(points: &[DataPoint]) -> f64 {
    let mut total_increase = 0.0;
    let mut prev_value = points[0].value;

    for point in points.iter().skip(1) {
        if point.value < prev_value {
            // Counter reset detected
            total_increase += prev_value; // Add value before reset
            total_increase += point.value; // Add value after reset
        } else {
            total_increase += point.value - prev_value;
        }
        prev_value = point.value;
    }

    total_increase
}

/// Calculate linear regression (slope and intercept).
fn linear_regression(points: &[DataPoint]) -> Option<(f64, f64)> {
    let n = points.len() as f64;
    if n < 2.0 {
        return None;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;

    for point in points {
        let x = point.timestamp as f64;
        let y = point.value;
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_xx += x * x;
    }

    let mean_x = sum_x / n;
    let mean_y = sum_y / n;

    let denominator = sum_xx - n * mean_x * mean_x;
    if denominator.abs() < f64::EPSILON {
        return None;
    }

    let slope = (sum_xy - n * mean_x * mean_y) / denominator;
    let intercept = mean_y - slope * mean_x;

    Some((slope, intercept))
}

/// Histogram bucket representation
#[derive(Debug, Clone)]
pub struct HistogramBucket {
    /// Upper bound of the bucket
    pub le: f64,
    /// Cumulative count
    pub count: f64,
}

/// Calculate histogram quantile from bucket counts.
///
/// # Arguments
/// * `quantile` - Quantile to calculate (0.0 to 1.0)
/// * `buckets` - Histogram buckets sorted by le (upper bound)
pub fn histogram_quantile(quantile: f64, buckets: &[HistogramBucket]) -> Option<f64> {
    if buckets.is_empty() || quantile < 0.0 || quantile > 1.0 {
        return None;
    }

    // Find total count (last bucket should have le=+Inf)
    let total = buckets.last()?.count;
    if total == 0.0 {
        return None;
    }

    let target = quantile * total;

    // Find the bucket that contains the target count
    let mut prev_count = 0.0;
    let mut prev_le = 0.0;

    for bucket in buckets {
        if bucket.count >= target {
            // Linear interpolation within the bucket
            if bucket.count == prev_count {
                return Some(bucket.le);
            }
            let fraction = (target - prev_count) / (bucket.count - prev_count);
            return Some(prev_le + fraction * (bucket.le - prev_le));
        }
        prev_count = bucket.count;
        prev_le = bucket.le;
    }

    // Target is beyond last bucket
    Some(buckets.last()?.le)
}

/// Calculate the absent value - returns 1 if no samples exist, empty otherwise.
/// Useful for alerting on missing data.
pub fn absent(points: &[DataPoint]) -> Option<f64> {
    if points.is_empty() {
        Some(1.0)
    } else {
        None
    }
}

/// Clamp values between min and max.
pub fn clamp(points: &[DataPoint], min: f64, max: f64) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.clamp(min, max)))
        .collect()
}

/// Clamp values to a minimum.
pub fn clamp_min(points: &[DataPoint], min: f64) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.max(min)))
        .collect()
}

/// Clamp values to a maximum.
pub fn clamp_max(points: &[DataPoint], max: f64) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.min(max)))
        .collect()
}

/// Apply absolute value to all points.
pub fn abs(points: &[DataPoint]) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.abs()))
        .collect()
}

/// Apply ceiling to all points.
pub fn ceil(points: &[DataPoint]) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.ceil()))
        .collect()
}

/// Apply floor to all points.
pub fn floor(points: &[DataPoint]) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.floor()))
        .collect()
}

/// Apply round to all points.
pub fn round(points: &[DataPoint], to_nearest: f64) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| {
            let rounded = if to_nearest == 0.0 {
                p.value.round()
            } else {
                (p.value / to_nearest).round() * to_nearest
            };
            DataPoint::new(p.timestamp, rounded)
        })
        .collect()
}

/// Apply exponential function to all points.
pub fn exp(points: &[DataPoint]) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.exp()))
        .collect()
}

/// Apply natural logarithm to all points.
pub fn ln(points: &[DataPoint]) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.ln()))
        .collect()
}

/// Apply log2 to all points.
pub fn log2(points: &[DataPoint]) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.log2()))
        .collect()
}

/// Apply log10 to all points.
pub fn log10(points: &[DataPoint]) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.log10()))
        .collect()
}

/// Apply square root to all points.
pub fn sqrt(points: &[DataPoint]) -> Vec<DataPoint> {
    points
        .iter()
        .map(|p| DataPoint::new(p.timestamp, p.value.sqrt()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_points(values: &[(i64, f64)]) -> Vec<DataPoint> {
        values.iter().map(|(t, v)| DataPoint::new(*t, *v)).collect()
    }

    #[test]
    fn test_rate_constant() {
        // 100 units over 10 seconds = 10 per second
        let points = make_points(&[
            (0, 0.0),
            (5_000_000_000, 50.0),  // 5 seconds
            (10_000_000_000, 100.0), // 10 seconds
        ]);

        let r = rate(&points).unwrap();
        assert!((r - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_rate_with_reset() {
        // Counter resets in the middle
        let points = make_points(&[
            (0, 50.0),
            (1_000_000_000, 100.0), // +50
            (2_000_000_000, 20.0),  // Reset! Value was 100, now 20
            (3_000_000_000, 70.0),  // +50
        ]);

        let r = rate(&points).unwrap();
        // Total increase: 50 + 100 (before reset) + 20 + 50 = 220
        // Duration: 3 seconds
        // Rate: ~73.33
        assert!((r - 73.33).abs() < 0.1);
    }

    #[test]
    fn test_increase() {
        let points = make_points(&[
            (0, 0.0),
            (5_000_000_000, 50.0),
            (10_000_000_000, 100.0),
        ]);

        let inc = increase(&points).unwrap();
        assert!((inc - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_irate() {
        // Instant rate from last two points
        let points = make_points(&[
            (0, 0.0),
            (5_000_000_000, 50.0),
            (6_000_000_000, 70.0), // +20 in 1 second = 20/sec
        ]);

        let ir = irate(&points).unwrap();
        assert!((ir - 20.0).abs() < 0.001);
    }

    #[test]
    fn test_delta() {
        let points = make_points(&[
            (0, 10.0),
            (1_000_000_000, 15.0),
            (2_000_000_000, 8.0), // Gauge can go down
        ]);

        let d = delta(&points).unwrap();
        assert!((d - (-2.0)).abs() < 0.001); // 8 - 10 = -2
    }

    #[test]
    fn test_idelta() {
        let points = make_points(&[
            (0, 10.0),
            (1_000_000_000, 15.0),
            (2_000_000_000, 8.0),
        ]);

        let id = idelta(&points).unwrap();
        assert!((id - (-7.0)).abs() < 0.001); // 8 - 15 = -7
    }

    #[test]
    fn test_deriv() {
        // Linear increase: 0 -> 100 over 10 seconds = 10/sec
        let points = make_points(&[
            (0, 0.0),
            (5_000_000_000, 50.0),
            (10_000_000_000, 100.0),
        ]);

        let d = deriv(&points).unwrap();
        assert!((d - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_predict_linear() {
        // Linear trend: 0 -> 100 over 10 seconds
        let points = make_points(&[
            (0, 0.0),
            (5_000_000_000, 50.0),
            (10_000_000_000, 100.0),
        ]);

        // Predict 5 seconds ahead
        let pred = predict_linear(&points, 5.0).unwrap();
        assert!((pred - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_changes() {
        let points = make_points(&[
            (0, 1.0),
            (1, 1.0), // No change
            (2, 2.0), // Change
            (3, 2.0), // No change
            (4, 3.0), // Change
        ]);

        assert_eq!(changes(&points), 2);
    }

    #[test]
    fn test_resets() {
        let points = make_points(&[
            (0, 100.0),
            (1, 150.0),
            (2, 50.0),  // Reset
            (3, 100.0),
            (4, 30.0),  // Reset
        ]);

        assert_eq!(resets(&points), 2);
    }

    #[test]
    fn test_histogram_quantile() {
        let buckets = vec![
            HistogramBucket { le: 1.0, count: 10.0 },
            HistogramBucket { le: 5.0, count: 50.0 },
            HistogramBucket { le: 10.0, count: 100.0 },
            HistogramBucket { le: f64::INFINITY, count: 100.0 },
        ];

        // 50th percentile should be around 5 (50% of samples)
        let p50 = histogram_quantile(0.5, &buckets).unwrap();
        assert!((p50 - 5.0).abs() < 0.1);

        // 90th percentile
        let p90 = histogram_quantile(0.9, &buckets).unwrap();
        assert!(p90 >= 5.0 && p90 <= 10.0);
    }

    #[test]
    fn test_absent() {
        assert_eq!(absent(&[]), Some(1.0));
        assert_eq!(absent(&make_points(&[(0, 1.0)])), None);
    }

    #[test]
    fn test_clamp() {
        let points = make_points(&[(0, -5.0), (1, 5.0), (2, 15.0)]);
        let clamped = clamp(&points, 0.0, 10.0);

        assert_eq!(clamped[0].value, 0.0);
        assert_eq!(clamped[1].value, 5.0);
        assert_eq!(clamped[2].value, 10.0);
    }

    #[test]
    fn test_math_functions() {
        let points = make_points(&[(0, 4.0), (1, 9.0)]);

        let sqrts = sqrt(&points);
        assert_eq!(sqrts[0].value, 2.0);
        assert_eq!(sqrts[1].value, 3.0);

        let abs_points = make_points(&[(0, -5.0), (1, 5.0)]);
        let absolutes = abs(&abs_points);
        assert_eq!(absolutes[0].value, 5.0);
        assert_eq!(absolutes[1].value, 5.0);
    }

    #[test]
    fn test_round() {
        let points = make_points(&[(0, 1.4), (1, 1.5), (2, 1.6)]);
        let rounded = round(&points, 0.0);

        assert_eq!(rounded[0].value, 1.0);
        assert_eq!(rounded[1].value, 2.0);
        assert_eq!(rounded[2].value, 2.0);
    }

    #[test]
    fn test_insufficient_points() {
        let single = make_points(&[(0, 1.0)]);
        let empty: Vec<DataPoint> = vec![];

        assert!(rate(&single).is_none());
        assert!(rate(&empty).is_none());
        assert!(increase(&single).is_none());
        assert!(delta(&empty).is_none());
        assert!(deriv(&single).is_none());
    }
}
