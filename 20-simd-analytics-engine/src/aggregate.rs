//! Vectorized aggregation operations.

use crate::column::Column;
use crate::filter::SelectionVector;
use crate::simd::SimdOps;
use crate::{Error, Result};
use std::collections::HashMap;

/// Aggregate operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOp {
    Sum,
    Count,
    Min,
    Max,
    Avg,
}

/// Aggregation result value.
#[derive(Debug, Clone)]
pub enum AggregateValue {
    Int64(i64),
    Float64(f64),
    Count(usize),
    Null,
}

impl AggregateValue {
    /// Get as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            AggregateValue::Int64(v) => Some(*v as f64),
            AggregateValue::Float64(v) => Some(*v),
            AggregateValue::Count(v) => Some(*v as f64),
            AggregateValue::Null => None,
        }
    }
}

/// Aggregator for computing aggregate functions.
pub struct Aggregator;

impl Aggregator {
    /// Compute aggregate on f32 column.
    pub fn aggregate_f32(data: &[f32], op: AggregateOp) -> AggregateValue {
        if data.is_empty() {
            return AggregateValue::Null;
        }

        match op {
            AggregateOp::Sum => {
                let sum = SimdOps::sum_f32(data);
                AggregateValue::Float64(sum as f64)
            }
            AggregateOp::Count => AggregateValue::Count(data.len()),
            AggregateOp::Min => {
                let min = SimdOps::min_f32(data).unwrap();
                AggregateValue::Float64(min as f64)
            }
            AggregateOp::Max => {
                let max = SimdOps::max_f32(data).unwrap();
                AggregateValue::Float64(max as f64)
            }
            AggregateOp::Avg => {
                let sum = SimdOps::sum_f32(data);
                AggregateValue::Float64(sum as f64 / data.len() as f64)
            }
        }
    }

    /// Compute aggregate on f64 column.
    pub fn aggregate_f64(data: &[f64], op: AggregateOp) -> AggregateValue {
        if data.is_empty() {
            return AggregateValue::Null;
        }

        match op {
            AggregateOp::Sum => {
                let sum = SimdOps::sum_f64(data);
                AggregateValue::Float64(sum)
            }
            AggregateOp::Count => AggregateValue::Count(data.len()),
            AggregateOp::Min => {
                let min = SimdOps::min_f64(data).unwrap();
                AggregateValue::Float64(min)
            }
            AggregateOp::Max => {
                let max = SimdOps::max_f64(data).unwrap();
                AggregateValue::Float64(max)
            }
            AggregateOp::Avg => {
                let sum = SimdOps::sum_f64(data);
                AggregateValue::Float64(sum / data.len() as f64)
            }
        }
    }

    /// Compute aggregate on i64 column.
    pub fn aggregate_i64(data: &[i64], op: AggregateOp) -> AggregateValue {
        if data.is_empty() {
            return AggregateValue::Null;
        }

        match op {
            AggregateOp::Sum => {
                let sum = SimdOps::sum_i64(data);
                AggregateValue::Int64(sum)
            }
            AggregateOp::Count => AggregateValue::Count(data.len()),
            AggregateOp::Min => {
                let min = data.iter().copied().min().unwrap();
                AggregateValue::Int64(min)
            }
            AggregateOp::Max => {
                let max = data.iter().copied().max().unwrap();
                AggregateValue::Int64(max)
            }
            AggregateOp::Avg => {
                let sum = SimdOps::sum_i64(data);
                AggregateValue::Float64(sum as f64 / data.len() as f64)
            }
        }
    }

    /// Compute aggregate on column.
    pub fn aggregate(column: &Column, op: AggregateOp) -> Result<AggregateValue> {
        match column {
            Column::Int32(v) => {
                let data: Vec<i64> = v.as_slice().iter().map(|&x| x as i64).collect();
                Ok(Self::aggregate_i64(&data, op))
            }
            Column::Int64(v) => Ok(Self::aggregate_i64(v.as_slice(), op)),
            Column::Float32(v) => Ok(Self::aggregate_f32(v.as_slice(), op)),
            Column::Float64(v) => Ok(Self::aggregate_f64(v.as_slice(), op)),
            Column::Bool(v) => {
                // Count true values
                let count = v.as_slice().iter().filter(|&&b| b != 0).count();
                match op {
                    AggregateOp::Sum | AggregateOp::Count => Ok(AggregateValue::Count(count)),
                    _ => Err(Error::InvalidOperation(
                        "Unsupported aggregate on bool column".into(),
                    )),
                }
            }
        }
    }

    /// Compute filtered aggregate.
    pub fn aggregate_filtered_f64(
        data: &[f64],
        selection: &SelectionVector,
        op: AggregateOp,
    ) -> AggregateValue {
        if selection.count() == 0 {
            return AggregateValue::Null;
        }

        match op {
            AggregateOp::Sum => {
                let mut sum = 0.0f64;
                for i in 0..data.len() {
                    if selection.is_selected(i) {
                        sum += data[i];
                    }
                }
                AggregateValue::Float64(sum)
            }
            AggregateOp::Count => AggregateValue::Count(selection.count()),
            AggregateOp::Min => {
                let mut min = f64::INFINITY;
                for i in 0..data.len() {
                    if selection.is_selected(i) && data[i] < min {
                        min = data[i];
                    }
                }
                AggregateValue::Float64(min)
            }
            AggregateOp::Max => {
                let mut max = f64::NEG_INFINITY;
                for i in 0..data.len() {
                    if selection.is_selected(i) && data[i] > max {
                        max = data[i];
                    }
                }
                AggregateValue::Float64(max)
            }
            AggregateOp::Avg => {
                let mut sum = 0.0f64;
                for i in 0..data.len() {
                    if selection.is_selected(i) {
                        sum += data[i];
                    }
                }
                AggregateValue::Float64(sum / selection.count() as f64)
            }
        }
    }
}

/// Hash aggregation state.
#[derive(Debug)]
struct AggregateState {
    sum: f64,
    count: usize,
    min: f64,
    max: f64,
}

impl Default for AggregateState {
    fn default() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
}

/// Hash aggregator for GROUP BY.
pub struct HashAggregator {
    /// Hash table: key -> state.
    table: HashMap<i64, AggregateState>,
}

impl HashAggregator {
    /// Create new hash aggregator.
    pub fn new() -> Self {
        Self {
            table: HashMap::new(),
        }
    }

    /// Aggregate data with grouping.
    pub fn aggregate_by_key(
        &mut self,
        keys: &[i64],
        values: &[f64],
    ) -> Result<()> {
        if keys.len() != values.len() {
            return Err(Error::DimensionMismatch(format!(
                "Keys ({}) and values ({}) length mismatch",
                keys.len(), values.len()
            )));
        }

        for i in 0..keys.len() {
            let key = keys[i];
            let value = values[i];

            let state = self.table.entry(key).or_default();
            state.sum += value;
            state.count += 1;
            state.min = state.min.min(value);
            state.max = state.max.max(value);
        }

        Ok(())
    }

    /// Get result for specific aggregate.
    pub fn get_results(&self, op: AggregateOp) -> Vec<(i64, AggregateValue)> {
        self.table
            .iter()
            .map(|(&key, state)| {
                let value = match op {
                    AggregateOp::Sum => AggregateValue::Float64(state.sum),
                    AggregateOp::Count => AggregateValue::Count(state.count),
                    AggregateOp::Min => AggregateValue::Float64(state.min),
                    AggregateOp::Max => AggregateValue::Float64(state.max),
                    AggregateOp::Avg => {
                        AggregateValue::Float64(state.sum / state.count as f64)
                    }
                };
                (key, value)
            })
            .collect()
    }

    /// Get number of groups.
    pub fn num_groups(&self) -> usize {
        self.table.len()
    }

    /// Clear the aggregator.
    pub fn clear(&mut self) {
        self.table.clear();
    }
}

impl Default for HashAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Parallel aggregator state for merging.
#[derive(Debug, Clone)]
pub struct PartialAggregate {
    pub sum: f64,
    pub count: usize,
    pub min: f64,
    pub max: f64,
}

impl Default for PartialAggregate {
    fn default() -> Self {
        Self {
            sum: 0.0,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }
}

impl PartialAggregate {
    /// Merge another partial aggregate.
    pub fn merge(&mut self, other: &PartialAggregate) {
        self.sum += other.sum;
        self.count += other.count;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }

    /// Get final value for operation.
    pub fn finalize(&self, op: AggregateOp) -> AggregateValue {
        if self.count == 0 {
            return AggregateValue::Null;
        }

        match op {
            AggregateOp::Sum => AggregateValue::Float64(self.sum),
            AggregateOp::Count => AggregateValue::Count(self.count),
            AggregateOp::Min => AggregateValue::Float64(self.min),
            AggregateOp::Max => AggregateValue::Float64(self.max),
            AggregateOp::Avg => AggregateValue::Float64(self.sum / self.count as f64),
        }
    }

    /// Add value to partial aggregate.
    pub fn add(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Compute partial aggregate from slice.
    pub fn from_slice(data: &[f64]) -> Self {
        let mut partial = Self::default();
        for &value in data {
            partial.add(value);
        }
        partial
    }
}

/// Running statistics with numerically stable algorithms.
#[derive(Debug, Clone)]
pub struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64, // For Welford's algorithm
    min: f64,
    max: f64,
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningStats {
    /// Create new running stats.
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Add a value (Welford's online algorithm).
    pub fn add(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    /// Get count.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Get sum.
    pub fn sum(&self) -> f64 {
        self.mean * self.count as f64
    }

    /// Get variance.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count - 1) as f64
    }

    /// Get standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get min.
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Get max.
    pub fn max(&self) -> f64 {
        self.max
    }

    /// Merge with another RunningStats (parallel combine).
    pub fn merge(&mut self, other: &RunningStats) {
        if other.count == 0 {
            return;
        }
        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let total = self.count + other.count;
        let delta = other.mean - self.mean;

        self.mean = (self.count as f64 * self.mean + other.count as f64 * other.mean) / total as f64;
        self.m2 = self.m2 + other.m2 + delta * delta * (self.count * other.count) as f64 / total as f64;
        self.count = total;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_f64() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let sum = Aggregator::aggregate_f64(&data, AggregateOp::Sum);
        assert_eq!(sum.as_f64(), Some(15.0));

        let count = Aggregator::aggregate_f64(&data, AggregateOp::Count);
        assert_eq!(count.as_f64(), Some(5.0));

        let min = Aggregator::aggregate_f64(&data, AggregateOp::Min);
        assert_eq!(min.as_f64(), Some(1.0));

        let max = Aggregator::aggregate_f64(&data, AggregateOp::Max);
        assert_eq!(max.as_f64(), Some(5.0));

        let avg = Aggregator::aggregate_f64(&data, AggregateOp::Avg);
        assert_eq!(avg.as_f64(), Some(3.0));
    }

    #[test]
    fn test_hash_aggregator() {
        let mut agg = HashAggregator::new();

        let keys = vec![1, 1, 2, 2, 2];
        let values = vec![10.0, 20.0, 5.0, 10.0, 15.0];

        agg.aggregate_by_key(&keys, &values).unwrap();

        let results = agg.get_results(AggregateOp::Sum);
        assert_eq!(agg.num_groups(), 2);

        let group1: Vec<_> = results.iter().filter(|(k, _)| *k == 1).collect();
        let group2: Vec<_> = results.iter().filter(|(k, _)| *k == 2).collect();

        assert_eq!(group1[0].1.as_f64(), Some(30.0));
        assert_eq!(group2[0].1.as_f64(), Some(30.0));
    }

    #[test]
    fn test_partial_aggregate() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let partial = PartialAggregate::from_slice(&data);

        assert_eq!(partial.sum, 15.0);
        assert_eq!(partial.count, 5);
        assert_eq!(partial.min, 1.0);
        assert_eq!(partial.max, 5.0);
    }

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::new();
        for i in 1..=100 {
            stats.add(i as f64);
        }

        assert_eq!(stats.count(), 100);
        assert!((stats.mean() - 50.5).abs() < 0.001);
        assert_eq!(stats.min(), 1.0);
        assert_eq!(stats.max(), 100.0);
    }

    #[test]
    fn test_running_stats_merge() {
        let mut stats1 = RunningStats::new();
        let mut stats2 = RunningStats::new();

        for i in 1..=50 {
            stats1.add(i as f64);
        }
        for i in 51..=100 {
            stats2.add(i as f64);
        }

        stats1.merge(&stats2);

        assert_eq!(stats1.count(), 100);
        assert!((stats1.mean() - 50.5).abs() < 0.001);
    }
}
