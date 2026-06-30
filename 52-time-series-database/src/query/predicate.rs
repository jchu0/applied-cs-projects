//! Predicate evaluation for filtering time-series data

use crate::types::DataPoint;

/// Predicate operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PredicateOp {
    /// Equal to
    Eq,
    /// Not equal to
    Ne,
    /// Less than
    Lt,
    /// Less than or equal to
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal to
    Ge,
}

/// A predicate for filtering data points
#[derive(Debug, Clone)]
pub enum Predicate {
    /// Compare value with a constant
    Value(PredicateOp, f64),
    /// Compare timestamp with a constant
    Timestamp(PredicateOp, i64),
    /// Check if value is within a range
    ValueRange(f64, f64),
    /// Check if timestamp is within a range
    TimestampRange(i64, i64),
    /// Check if value is NaN
    IsNan,
    /// Check if value is not NaN
    IsNotNan,
    /// Check if value is finite
    IsFinite,
    /// Check if value is infinite
    IsInfinite,
    /// Logical AND of two predicates
    And(Box<Predicate>, Box<Predicate>),
    /// Logical OR of two predicates
    Or(Box<Predicate>, Box<Predicate>),
    /// Logical NOT of a predicate
    Not(Box<Predicate>),
    /// Always true
    True,
    /// Always false
    False,
}

impl Predicate {
    /// Create a value equals predicate
    pub fn value_eq(value: f64) -> Self {
        Predicate::Value(PredicateOp::Eq, value)
    }

    /// Create a value greater than predicate
    pub fn value_gt(value: f64) -> Self {
        Predicate::Value(PredicateOp::Gt, value)
    }

    /// Create a value less than predicate
    pub fn value_lt(value: f64) -> Self {
        Predicate::Value(PredicateOp::Lt, value)
    }

    /// Create a value range predicate
    pub fn value_between(min: f64, max: f64) -> Self {
        Predicate::ValueRange(min, max)
    }

    /// Create a timestamp range predicate
    pub fn timestamp_between(start: i64, end: i64) -> Self {
        Predicate::TimestampRange(start, end)
    }

    /// Combine with AND
    pub fn and(self, other: Predicate) -> Self {
        Predicate::And(Box::new(self), Box::new(other))
    }

    /// Combine with OR
    pub fn or(self, other: Predicate) -> Self {
        Predicate::Or(Box::new(self), Box::new(other))
    }

    /// Negate the predicate
    pub fn not(self) -> Self {
        Predicate::Not(Box::new(self))
    }

    /// Evaluate the predicate against a data point
    pub fn evaluate(&self, point: &DataPoint) -> bool {
        match self {
            Predicate::Value(op, value) => compare_f64(*op, point.value, *value),
            Predicate::Timestamp(op, ts) => compare_i64(*op, point.timestamp, *ts),
            Predicate::ValueRange(min, max) => point.value >= *min && point.value <= *max,
            Predicate::TimestampRange(start, end) => {
                point.timestamp >= *start && point.timestamp <= *end
            }
            Predicate::IsNan => point.value.is_nan(),
            Predicate::IsNotNan => !point.value.is_nan(),
            Predicate::IsFinite => point.value.is_finite(),
            Predicate::IsInfinite => point.value.is_infinite(),
            Predicate::And(left, right) => left.evaluate(point) && right.evaluate(point),
            Predicate::Or(left, right) => left.evaluate(point) || right.evaluate(point),
            Predicate::Not(pred) => !pred.evaluate(point),
            Predicate::True => true,
            Predicate::False => false,
        }
    }

    /// Filter a slice of data points using this predicate
    pub fn filter<'a>(&self, points: &'a [DataPoint]) -> Vec<&'a DataPoint> {
        points.iter().filter(|p| self.evaluate(p)).collect()
    }

    /// Filter and collect data points
    pub fn filter_owned(&self, points: &[DataPoint]) -> Vec<DataPoint> {
        points.iter().filter(|p| self.evaluate(p)).copied().collect()
    }

    /// Check if the predicate can be evaluated without looking at the data
    pub fn is_constant(&self) -> Option<bool> {
        match self {
            Predicate::True => Some(true),
            Predicate::False => Some(false),
            _ => None,
        }
    }

    /// Simplify the predicate
    pub fn simplify(self) -> Self {
        match self {
            Predicate::And(left, right) => {
                let left = left.simplify();
                let right = right.simplify();

                match (left.is_constant(), right.is_constant()) {
                    (Some(false), _) | (_, Some(false)) => Predicate::False,
                    (Some(true), _) => right,
                    (_, Some(true)) => left,
                    _ => Predicate::And(Box::new(left), Box::new(right)),
                }
            }
            Predicate::Or(left, right) => {
                let left = left.simplify();
                let right = right.simplify();

                match (left.is_constant(), right.is_constant()) {
                    (Some(true), _) | (_, Some(true)) => Predicate::True,
                    (Some(false), _) => right,
                    (_, Some(false)) => left,
                    _ => Predicate::Or(Box::new(left), Box::new(right)),
                }
            }
            Predicate::Not(pred) => {
                let pred = pred.simplify();
                match pred.is_constant() {
                    Some(true) => Predicate::False,
                    Some(false) => Predicate::True,
                    None => Predicate::Not(Box::new(pred)),
                }
            }
            other => other,
        }
    }
}

fn compare_f64(op: PredicateOp, a: f64, b: f64) -> bool {
    match op {
        PredicateOp::Eq => (a - b).abs() < f64::EPSILON,
        PredicateOp::Ne => (a - b).abs() >= f64::EPSILON,
        PredicateOp::Lt => a < b,
        PredicateOp::Le => a <= b,
        PredicateOp::Gt => a > b,
        PredicateOp::Ge => a >= b,
    }
}

fn compare_i64(op: PredicateOp, a: i64, b: i64) -> bool {
    match op {
        PredicateOp::Eq => a == b,
        PredicateOp::Ne => a != b,
        PredicateOp::Lt => a < b,
        PredicateOp::Le => a <= b,
        PredicateOp::Gt => a > b,
        PredicateOp::Ge => a >= b,
    }
}

/// Builder for creating complex predicates
#[derive(Debug, Default)]
pub struct PredicateBuilder {
    predicates: Vec<Predicate>,
}

impl PredicateBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn value_gt(mut self, value: f64) -> Self {
        self.predicates.push(Predicate::value_gt(value));
        self
    }

    pub fn value_lt(mut self, value: f64) -> Self {
        self.predicates.push(Predicate::value_lt(value));
        self
    }

    pub fn value_between(mut self, min: f64, max: f64) -> Self {
        self.predicates.push(Predicate::value_between(min, max));
        self
    }

    pub fn timestamp_between(mut self, start: i64, end: i64) -> Self {
        self.predicates.push(Predicate::timestamp_between(start, end));
        self
    }

    pub fn is_finite(mut self) -> Self {
        self.predicates.push(Predicate::IsFinite);
        self
    }

    pub fn is_not_nan(mut self) -> Self {
        self.predicates.push(Predicate::IsNotNan);
        self
    }

    /// Build with AND logic (all predicates must match)
    pub fn build_and(self) -> Predicate {
        if self.predicates.is_empty() {
            return Predicate::True;
        }

        let mut result = self.predicates.into_iter();
        let first = result.next().unwrap();
        result.fold(first, |acc, pred| acc.and(pred))
    }

    /// Build with OR logic (any predicate must match)
    pub fn build_or(self) -> Predicate {
        if self.predicates.is_empty() {
            return Predicate::False;
        }

        let mut result = self.predicates.into_iter();
        let first = result.next().unwrap();
        result.fold(first, |acc, pred| acc.or(pred))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_predicates() {
        let point = DataPoint::new(100, 50.0);

        assert!(Predicate::value_eq(50.0).evaluate(&point));
        assert!(!Predicate::value_eq(51.0).evaluate(&point));

        assert!(Predicate::value_gt(40.0).evaluate(&point));
        assert!(!Predicate::value_gt(60.0).evaluate(&point));

        assert!(Predicate::value_lt(60.0).evaluate(&point));
        assert!(!Predicate::value_lt(40.0).evaluate(&point));
    }

    #[test]
    fn test_range_predicates() {
        let point = DataPoint::new(100, 50.0);

        assert!(Predicate::value_between(40.0, 60.0).evaluate(&point));
        assert!(!Predicate::value_between(60.0, 70.0).evaluate(&point));

        assert!(Predicate::timestamp_between(50, 150).evaluate(&point));
        assert!(!Predicate::timestamp_between(200, 300).evaluate(&point));
    }

    #[test]
    fn test_special_value_predicates() {
        let nan_point = DataPoint::new(100, f64::NAN);
        let inf_point = DataPoint::new(100, f64::INFINITY);
        let normal_point = DataPoint::new(100, 50.0);

        assert!(Predicate::IsNan.evaluate(&nan_point));
        assert!(!Predicate::IsNan.evaluate(&normal_point));

        assert!(Predicate::IsNotNan.evaluate(&normal_point));
        assert!(!Predicate::IsNotNan.evaluate(&nan_point));

        assert!(Predicate::IsInfinite.evaluate(&inf_point));
        assert!(!Predicate::IsInfinite.evaluate(&normal_point));

        assert!(Predicate::IsFinite.evaluate(&normal_point));
        assert!(!Predicate::IsFinite.evaluate(&inf_point));
    }

    #[test]
    fn test_logical_operators() {
        let point = DataPoint::new(100, 50.0);

        // AND
        let pred = Predicate::value_gt(40.0).and(Predicate::value_lt(60.0));
        assert!(pred.evaluate(&point));

        let pred = Predicate::value_gt(60.0).and(Predicate::value_lt(70.0));
        assert!(!pred.evaluate(&point));

        // OR
        let pred = Predicate::value_lt(40.0).or(Predicate::value_gt(40.0));
        assert!(pred.evaluate(&point));

        // NOT
        let pred = Predicate::value_lt(40.0).not();
        assert!(pred.evaluate(&point));
    }

    #[test]
    fn test_filter() {
        let points = vec![
            DataPoint::new(100, 10.0),
            DataPoint::new(200, 50.0),
            DataPoint::new(300, 90.0),
        ];

        let pred = Predicate::value_gt(30.0);
        let filtered = pred.filter_owned(&points);

        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].value, 50.0);
        assert_eq!(filtered[1].value, 90.0);
    }

    #[test]
    fn test_simplify() {
        // AND with False
        let pred = Predicate::True.and(Predicate::False).simplify();
        assert_eq!(pred.is_constant(), Some(false));

        // AND with True
        let pred = Predicate::value_gt(10.0).and(Predicate::True).simplify();
        assert!(matches!(pred, Predicate::Value(PredicateOp::Gt, _)));

        // OR with True
        let pred = Predicate::value_gt(10.0).or(Predicate::True).simplify();
        assert_eq!(pred.is_constant(), Some(true));

        // NOT True
        let pred = Predicate::True.not().simplify();
        assert_eq!(pred.is_constant(), Some(false));
    }

    #[test]
    fn test_predicate_builder() {
        let points = vec![
            DataPoint::new(100, 10.0),
            DataPoint::new(200, 50.0),
            DataPoint::new(300, 90.0),
        ];

        let pred = PredicateBuilder::new()
            .value_gt(20.0)
            .value_lt(80.0)
            .timestamp_between(0, 500)
            .build_and();

        let filtered = pred.filter_owned(&points);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].value, 50.0);
    }

    #[test]
    fn test_predicate_builder_or() {
        let points = vec![
            DataPoint::new(100, 10.0),
            DataPoint::new(200, 50.0),
            DataPoint::new(300, 90.0),
        ];

        let pred = PredicateBuilder::new()
            .value_lt(20.0)
            .value_gt(80.0)
            .build_or();

        let filtered = pred.filter_owned(&points);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_empty_builder() {
        let pred_and = PredicateBuilder::new().build_and();
        assert!(matches!(pred_and, Predicate::True));

        let pred_or = PredicateBuilder::new().build_or();
        assert!(matches!(pred_or, Predicate::False));
    }

    #[test]
    fn test_timestamp_predicates() {
        let point = DataPoint::new(100, 50.0);

        let pred = Predicate::Timestamp(PredicateOp::Eq, 100);
        assert!(pred.evaluate(&point));

        let pred = Predicate::Timestamp(PredicateOp::Lt, 200);
        assert!(pred.evaluate(&point));

        let pred = Predicate::Timestamp(PredicateOp::Gt, 50);
        assert!(pred.evaluate(&point));
    }
}
