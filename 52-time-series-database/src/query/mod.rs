//! Query engine for time-series data
//!
//! Provides:
//! - Time-range queries
//! - Aggregation functions (sum, avg, min, max, count, percentiles)
//! - Group-by operations
//! - Downsampling queries
//! - Predicate evaluation
//! - Label matchers for series selection
//! - Time-series functions (rate, increase, irate, delta)

pub mod aggregation;
pub mod predicate;
pub mod executor;
pub mod label_matcher;
pub mod functions;

pub use aggregation::{Aggregation, AggregateQuery, AggregateResult, Aggregator};
pub use predicate::{Predicate, PredicateOp};
pub use executor::{QueryExecutor, Query, QueryResult, QueryBuilder};
pub use label_matcher::{LabelMatcher, LabelMatchers, LabelMatcherBuilder, MatchOp, SeriesSelector};
pub use functions::{rate, increase, irate, delta, idelta, deriv, predict_linear, changes, resets, histogram_quantile, HistogramBucket};
