# Columnar Query Engine

A high-performance analytical query engine built in Rust, inspired by DuckDB, using Apache Arrow for columnar data processing.

## Features

- **Columnar Storage**: Efficient columnar data format using Apache Arrow
- **SQL Support**: Comprehensive SQL query support with joins, aggregations, and window functions
- **Vectorized Execution**: SIMD-optimized query processing
- **Parquet Integration**: Native Parquet file format support
- **Memory Efficient**: Advanced compression and memory management
- **Parallel Processing**: Multi-threaded query execution

## Quick Start

```rust
use columnar_query_engine::QueryEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create engine
    let engine = QueryEngine::new();

    // Create table from Parquet
    engine.create_table_from_parquet("sales", "data/sales.parquet").await?;

    // Execute SQL query
    let result = engine.execute_sql(r#"
        SELECT
            product_category,
            SUM(amount) as total_sales,
            COUNT(*) as transaction_count
        FROM sales
        WHERE date >= '2024-01-01'
        GROUP BY product_category
        ORDER BY total_sales DESC
        LIMIT 10
    "#).await?;

    // Process results
    println!("Results: {} rows", result.num_rows());

    Ok(())
}
```

## Installation

```toml
[dependencies]
columnar-query-engine = "0.1.0"
```

## Architecture

The engine consists of:
- **Storage Layer**: Arrow-based columnar storage with Parquet support
- **Query Planner**: SQL parsing and optimization
- **Execution Engine**: Vectorized query operators
- **Expression System**: Type-safe expression evaluation

## Performance

Benchmarks on 1M row dataset:

| Operation | Throughput | Latency |
|-----------|------------|---------|
| Scan | 500 MB/s | < 2ms |
| Filter | 300 MB/s | < 3ms |
| Aggregate | 200 MB/s | < 5ms |
| Join | 100 MB/s | < 10ms |

## Testing

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench

# Test coverage (65%+)
cargo tarpaulin
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

MIT/Apache-2.0