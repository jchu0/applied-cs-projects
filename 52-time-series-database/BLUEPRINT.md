# Time-Series Database Blueprint

## Overview

A high-performance time-series database (TSDB) designed for storing and querying time-stamped data with efficient compression, fast ingestion, and flexible query capabilities.

> **Concepts covered:** [§01 Rust async](../../01-software-engineering/rust/05-async-rust/rust-async.md) · [§02 Data warehousing](../../02-data-engineering/03-data-warehousing/) · [§02 Real-time analytics](../../02-data-engineering/04-streaming/real-time-analytics/). Compare to [Project 17 (columnar query engine)](../17-columnar-query-engine/) for the OLAP angle. Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture

### Core Components

1. **Storage Engine**
   - Columnar storage for efficient compression and vectorized operations
   - LSM-tree based write path for high ingestion rates
   - Time-based partitioning (shards by time range)
   - Memory-mapped files for fast access

2. **Compression Layer**
   - Delta encoding for timestamps
   - Gorilla/XOR compression for floating-point values
   - Run-length encoding for repeated values
   - Dictionary encoding for string tags
   - Variable-length integer encoding (varint)

3. **Query Engine**
   - Time-range queries with O(log n) lookup
   - Aggregation functions (sum, avg, min, max, count, percentiles)
   - Group-by operations on tags/labels
   - Downsampling/rollup queries
   - Predicate pushdown for efficient filtering

4. **Write Path**
   - Write-ahead log (WAL) for durability
   - In-memory buffer with size-based flushing
   - Batch writes for throughput optimization
   - Concurrent write support with sharding

5. **Retention & Lifecycle**
   - Configurable retention policies
   - Automatic data expiration
   - Downsampling for long-term storage
   - Compaction for space optimization

## Data Model

```
Metric {
    name: String,           // e.g., "cpu.usage"
    tags: Map<String, String>,  // e.g., {"host": "server1", "region": "us-east"}
    timestamp: i64,         // Unix timestamp in nanoseconds
    value: f64,             // Metric value
}

Series {
    series_key: u64,        // Hash of metric name + sorted tags
    points: Vec<(i64, f64)>,
}
```

## Implementation Phases

### Phase 1: Core Storage (30%)
- [x] Data point representation
- [x] Series and metric structures
- [x] In-memory storage
- [x] Basic time-range queries

### Phase 2: Compression (25%)
- [x] Delta encoding for timestamps
- [x] Gorilla compression for values
- [x] Run-length encoding
- [x] Varint encoding
- [x] Dictionary encoding

### Phase 3: Persistence (20%)
- [x] Write-ahead log
- [x] SSTable format
- [x] Block-based storage
- [x] Memory-mapped file access

### Phase 4: Query Engine (15%)
- [x] Aggregation functions
- [x] Group-by operations
- [x] Downsampling queries
- [x] Predicate evaluation

### Phase 5: Retention & Optimization (10%)
- [x] Retention policies
- [x] Automatic expiration
- [x] Compaction
- [x] Background maintenance

## API Design

```rust
// Write API
fn write(&mut self, metric: &str, tags: &Tags, timestamp: i64, value: f64) -> Result<()>;
fn write_batch(&mut self, points: &[DataPoint]) -> Result<()>;

// Query API
fn query(&self, query: &Query) -> Result<QueryResult>;
fn query_range(&self, series_key: u64, start: i64, end: i64) -> Result<Vec<DataPoint>>;

// Aggregation API
fn aggregate(&self, query: &AggregateQuery) -> Result<AggregateResult>;

// Management API
fn set_retention_policy(&mut self, policy: RetentionPolicy) -> Result<()>;
fn compact(&mut self) -> Result<()>;
fn flush(&mut self) -> Result<()>;
```

## Performance Targets

- Write throughput: 1M+ points/second
- Query latency: <10ms for simple time-range queries
- Compression ratio: 10-20x for typical time-series data
- Memory efficiency: <100 bytes overhead per series

## Key Technologies

- Rust for memory safety and performance
- Memory-mapped files (mmap)
- Lock-free data structures where possible
- SIMD for vectorized operations (future)
