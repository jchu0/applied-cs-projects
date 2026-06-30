//! Time-Series Database
//!
//! A high-performance time-series database with efficient compression,
//! fast ingestion, and flexible query capabilities.

pub mod storage;
pub mod compression;
pub mod query;
pub mod wal;
pub mod retention;
pub mod types;
pub mod error;
pub mod database;

pub use database::TimeSeriesDB;
pub use types::{DataPoint, Metric, Series, Tags, SeriesKey};
pub use query::{
    Query, QueryResult, Aggregation, AggregateQuery, QueryBuilder,
    LabelMatcher, LabelMatchers, LabelMatcherBuilder, MatchOp, SeriesSelector,
    rate, increase, irate, delta, idelta, deriv, predict_linear, changes, resets,
    histogram_quantile, HistogramBucket,
};
pub use retention::RetentionPolicy;
pub use error::{TsdbError, Result};
