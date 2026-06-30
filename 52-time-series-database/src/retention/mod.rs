//! Retention policies and downsampling for time-series data
//!
//! This module provides:
//! - Configurable retention policies
//! - Automatic data expiration
//! - Downsampling for long-term storage
//! - Compaction strategies

use std::sync::Arc;
use parking_lot::RwLock;

use crate::error::{Result, TsdbError};
use crate::types::{DataPoint, duration};
use crate::query::Aggregation;

/// Retention policy configuration
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    /// Name of the policy
    pub name: String,
    /// Duration to keep raw data (in nanoseconds)
    pub raw_duration: i64,
    /// Downsampling rules
    pub downsample_rules: Vec<DownsampleRule>,
    /// Whether to delete data after retention period
    pub drop_after_retention: bool,
}

impl RetentionPolicy {
    /// Create a new retention policy
    pub fn new<S: Into<String>>(name: S, raw_duration: i64) -> Self {
        Self {
            name: name.into(),
            raw_duration,
            downsample_rules: Vec::new(),
            drop_after_retention: true,
        }
    }

    /// Add a downsampling rule
    pub fn with_downsample(mut self, rule: DownsampleRule) -> Self {
        self.downsample_rules.push(rule);
        self.downsample_rules.sort_by_key(|r| r.after);
        self
    }

    /// Set whether to drop data after retention
    pub fn with_drop_after(mut self, drop: bool) -> Self {
        self.drop_after_retention = drop;
        self
    }

    /// Check if data should be dropped at a given age
    pub fn should_drop(&self, age: i64) -> bool {
        if !self.drop_after_retention {
            return false;
        }

        // Check if past all retention periods
        let max_retention = self.max_retention();
        age > max_retention
    }

    /// Get the maximum retention period
    pub fn max_retention(&self) -> i64 {
        let mut max = self.raw_duration;
        for rule in &self.downsample_rules {
            if rule.keep_for > max {
                max = rule.keep_for;
            }
        }
        max
    }

    /// Get the appropriate resolution for data at a given age
    pub fn resolution_for_age(&self, age: i64) -> Option<i64> {
        // Find the most specific applicable rule (largest `after` that still applies)
        let mut best: Option<&DownsampleRule> = None;
        for rule in &self.downsample_rules {
            if age >= rule.after {
                match best {
                    Some(b) if rule.after > b.after => best = Some(rule),
                    None => best = Some(rule),
                    _ => {}
                }
            }
        }
        if let Some(rule) = best {
            return Some(rule.interval);
        }

        // Return None for raw data
        None
    }

    /// Check if data at a given age should be downsampled
    pub fn needs_downsampling(&self, age: i64) -> Option<&DownsampleRule> {
        for rule in &self.downsample_rules {
            if age >= rule.after && age < rule.after + rule.keep_for {
                return Some(rule);
            }
        }
        None
    }
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self::new("default", 7 * duration::DAY)
    }
}

/// Downsampling rule
#[derive(Debug, Clone)]
pub struct DownsampleRule {
    /// Age after which to apply this rule (in nanoseconds)
    pub after: i64,
    /// Aggregation interval (in nanoseconds)
    pub interval: i64,
    /// How long to keep downsampled data
    pub keep_for: i64,
    /// Aggregation function to use
    pub aggregation: Aggregation,
}

impl DownsampleRule {
    /// Create a new downsampling rule
    pub fn new(after: i64, interval: i64, keep_for: i64, aggregation: Aggregation) -> Self {
        Self {
            after,
            interval,
            keep_for,
            aggregation,
        }
    }

    /// Apply the downsampling rule to a set of points
    pub fn apply(&self, points: &[DataPoint], start: i64, end: i64) -> Vec<DataPoint> {
        if points.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        let mut bucket_start = (start / self.interval) * self.interval;

        while bucket_start < end {
            let bucket_end = bucket_start + self.interval;

            // Collect points in this bucket
            let bucket_points: Vec<_> = points
                .iter()
                .filter(|p| p.timestamp >= bucket_start && p.timestamp < bucket_end)
                .copied()
                .collect();

            if !bucket_points.is_empty() {
                let value = aggregate_points(&bucket_points, self.aggregation);
                result.push(DataPoint::new(bucket_start, value));
            }

            bucket_start = bucket_end;
        }

        result
    }
}

/// Aggregate points using the specified aggregation function
fn aggregate_points(points: &[DataPoint], aggregation: Aggregation) -> f64 {
    if points.is_empty() {
        return f64::NAN;
    }

    match aggregation {
        Aggregation::Sum => points.iter().map(|p| p.value).sum(),
        Aggregation::Avg => {
            let sum: f64 = points.iter().map(|p| p.value).sum();
            sum / points.len() as f64
        }
        Aggregation::Min => points
            .iter()
            .map(|p| p.value)
            .fold(f64::INFINITY, f64::min),
        Aggregation::Max => points
            .iter()
            .map(|p| p.value)
            .fold(f64::NEG_INFINITY, f64::max),
        Aggregation::Count => points.len() as f64,
        Aggregation::First => points.first().map(|p| p.value).unwrap_or(f64::NAN),
        Aggregation::Last => points.last().map(|p| p.value).unwrap_or(f64::NAN),
        _ => {
            // For other aggregations, fall back to average
            let sum: f64 = points.iter().map(|p| p.value).sum();
            sum / points.len() as f64
        }
    }
}

/// Retention policy manager
#[derive(Debug)]
pub struct RetentionManager {
    /// Policies by name
    policies: RwLock<std::collections::HashMap<String, RetentionPolicy>>,
    /// Default policy
    default_policy: RwLock<RetentionPolicy>,
}

impl RetentionManager {
    /// Create a new retention manager
    pub fn new() -> Self {
        Self {
            policies: RwLock::new(std::collections::HashMap::new()),
            default_policy: RwLock::new(RetentionPolicy::default()),
        }
    }

    /// Set the default retention policy
    pub fn set_default(&self, policy: RetentionPolicy) {
        *self.default_policy.write() = policy;
    }

    /// Add a named retention policy
    pub fn add_policy(&self, policy: RetentionPolicy) {
        self.policies.write().insert(policy.name.clone(), policy);
    }

    /// Get a retention policy by name
    pub fn get_policy(&self, name: &str) -> Option<RetentionPolicy> {
        self.policies.read().get(name).cloned()
    }

    /// Get the default policy
    pub fn default_policy(&self) -> RetentionPolicy {
        self.default_policy.read().clone()
    }

    /// Remove a policy by name
    pub fn remove_policy(&self, name: &str) -> Option<RetentionPolicy> {
        self.policies.write().remove(name)
    }

    /// List all policy names
    pub fn policy_names(&self) -> Vec<String> {
        self.policies.read().keys().cloned().collect()
    }

    /// Calculate what data should be dropped based on current time
    pub fn calculate_drop_before(&self, policy_name: Option<&str>, now: i64) -> i64 {
        let policy = policy_name
            .and_then(|n| self.get_policy(n))
            .unwrap_or_else(|| self.default_policy());

        now - policy.max_retention()
    }
}

impl Default for RetentionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-defined retention policy templates
pub mod templates {
    use super::*;

    /// Short-term monitoring (7 days raw, then drop)
    pub fn short_term() -> RetentionPolicy {
        RetentionPolicy::new("short_term", 7 * duration::DAY)
    }

    /// Standard monitoring (7 days raw, 30 days 1-minute, 1 year 1-hour)
    pub fn standard() -> RetentionPolicy {
        RetentionPolicy::new("standard", 7 * duration::DAY)
            .with_downsample(DownsampleRule::new(
                7 * duration::DAY,
                duration::MINUTE,
                30 * duration::DAY,
                Aggregation::Avg,
            ))
            .with_downsample(DownsampleRule::new(
                30 * duration::DAY,
                duration::HOUR,
                365 * duration::DAY,
                Aggregation::Avg,
            ))
    }

    /// Long-term archival (30 days raw, 1 year 5-minute, 5 years 1-hour)
    pub fn long_term() -> RetentionPolicy {
        RetentionPolicy::new("long_term", 30 * duration::DAY)
            .with_downsample(DownsampleRule::new(
                30 * duration::DAY,
                5 * duration::MINUTE,
                365 * duration::DAY,
                Aggregation::Avg,
            ))
            .with_downsample(DownsampleRule::new(
                365 * duration::DAY,
                duration::HOUR,
                5 * 365 * duration::DAY,
                Aggregation::Avg,
            ))
    }

    /// High-resolution (24 hours raw only, no downsampling)
    pub fn high_resolution() -> RetentionPolicy {
        RetentionPolicy::new("high_resolution", duration::DAY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retention_policy_basic() {
        let policy = RetentionPolicy::new("test", 7 * duration::DAY);

        assert_eq!(policy.max_retention(), 7 * duration::DAY);
        assert!(policy.should_drop(8 * duration::DAY));
        assert!(!policy.should_drop(6 * duration::DAY));
    }

    #[test]
    fn test_retention_policy_with_downsample() {
        let policy = templates::standard();

        // Raw data for first 7 days
        assert_eq!(policy.resolution_for_age(duration::DAY), None);

        // 1-minute data from 7-30 days
        assert_eq!(
            policy.resolution_for_age(10 * duration::DAY),
            Some(duration::MINUTE)
        );

        // 1-hour data after 30 days
        assert_eq!(
            policy.resolution_for_age(60 * duration::DAY),
            Some(duration::HOUR)
        );
    }

    #[test]
    fn test_retention_policy_max_retention() {
        let policy = templates::standard();

        // Should keep data for up to 1 year
        assert_eq!(policy.max_retention(), 365 * duration::DAY);
    }

    #[test]
    fn test_downsample_rule_apply() {
        let rule = DownsampleRule::new(
            0,
            100, // 100ns interval
            1000,
            Aggregation::Avg,
        );

        let points = vec![
            DataPoint::new(10, 1.0),
            DataPoint::new(20, 2.0),
            DataPoint::new(30, 3.0),
            DataPoint::new(110, 4.0),
            DataPoint::new(120, 5.0),
        ];

        let result = rule.apply(&points, 0, 200);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].timestamp, 0);
        assert_eq!(result[0].value, 2.0); // avg(1, 2, 3)
        assert_eq!(result[1].timestamp, 100);
        assert_eq!(result[1].value, 4.5); // avg(4, 5)
    }

    #[test]
    fn test_downsample_rule_sum() {
        let rule = DownsampleRule::new(0, 100, 1000, Aggregation::Sum);

        let points = vec![
            DataPoint::new(10, 1.0),
            DataPoint::new(20, 2.0),
            DataPoint::new(30, 3.0),
        ];

        let result = rule.apply(&points, 0, 100);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].value, 6.0); // sum(1, 2, 3)
    }

    #[test]
    fn test_retention_manager() {
        let manager = RetentionManager::new();

        // Add custom policy
        manager.add_policy(templates::standard());

        // Get policy
        let policy = manager.get_policy("standard").unwrap();
        assert_eq!(policy.name, "standard");

        // List policies
        let names = manager.policy_names();
        assert!(names.contains(&"standard".to_string()));

        // Remove policy
        manager.remove_policy("standard");
        assert!(manager.get_policy("standard").is_none());
    }

    #[test]
    fn test_retention_manager_default() {
        let manager = RetentionManager::new();

        let default = manager.default_policy();
        assert_eq!(default.name, "default");

        manager.set_default(templates::high_resolution());
        let default = manager.default_policy();
        assert_eq!(default.name, "high_resolution");
    }

    #[test]
    fn test_calculate_drop_before() {
        let manager = RetentionManager::new();
        manager.add_policy(templates::short_term());

        let now = 100 * duration::DAY;
        let drop_before = manager.calculate_drop_before(Some("short_term"), now);

        // Should drop data older than 7 days
        assert_eq!(drop_before, 93 * duration::DAY);
    }

    #[test]
    fn test_template_policies() {
        // Short term
        let short = templates::short_term();
        assert_eq!(short.max_retention(), 7 * duration::DAY);

        // Standard
        let standard = templates::standard();
        assert_eq!(standard.downsample_rules.len(), 2);

        // Long term
        let long = templates::long_term();
        assert!(long.max_retention() > 365 * duration::DAY);

        // High resolution
        let high = templates::high_resolution();
        assert!(high.downsample_rules.is_empty());
    }

    #[test]
    fn test_policy_drop_after_retention() {
        let policy = RetentionPolicy::new("test", duration::DAY)
            .with_drop_after(false);

        // Even old data shouldn't be dropped
        assert!(!policy.should_drop(100 * duration::DAY));
    }

    #[test]
    fn test_needs_downsampling() {
        let policy = templates::standard();

        // Raw data doesn't need downsampling
        assert!(policy.needs_downsampling(duration::DAY).is_none());

        // 10 days old needs 1-minute downsampling
        let rule = policy.needs_downsampling(10 * duration::DAY);
        assert!(rule.is_some());
        assert_eq!(rule.unwrap().interval, duration::MINUTE);

        // 60 days old needs 1-hour downsampling
        let rule = policy.needs_downsampling(60 * duration::DAY);
        assert!(rule.is_some());
        assert_eq!(rule.unwrap().interval, duration::HOUR);
    }

    #[test]
    fn test_aggregate_points() {
        let points = vec![
            DataPoint::new(100, 10.0),
            DataPoint::new(200, 20.0),
            DataPoint::new(300, 30.0),
        ];

        assert_eq!(aggregate_points(&points, Aggregation::Sum), 60.0);
        assert_eq!(aggregate_points(&points, Aggregation::Avg), 20.0);
        assert_eq!(aggregate_points(&points, Aggregation::Min), 10.0);
        assert_eq!(aggregate_points(&points, Aggregation::Max), 30.0);
        assert_eq!(aggregate_points(&points, Aggregation::Count), 3.0);
        assert_eq!(aggregate_points(&points, Aggregation::First), 10.0);
        assert_eq!(aggregate_points(&points, Aggregation::Last), 30.0);
    }

    #[test]
    fn test_empty_aggregate() {
        let points: Vec<DataPoint> = vec![];
        assert!(aggregate_points(&points, Aggregation::Sum).is_nan());
    }
}
