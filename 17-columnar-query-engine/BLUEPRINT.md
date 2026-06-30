# Columnar Query Engine (DuckDB-lite)

## Project Overview

A high-performance in-memory analytical query engine inspired by DuckDB. This project implements columnar storage, vectorized execution, and query optimization to efficiently process analytical SQL queries. The engine focuses on OLAP workloads with support for complex aggregations, joins, and window functions.

> **Concepts covered:** [§02 SQL optimization](../../02-data-engineering/03-data-warehousing/sql-optimization/sql-optimization.md) · [§02 DuckDB / data processing](../../02-data-engineering/02-data-processing/duckdb/) · [§02 Dimensional modeling](../../02-data-engineering/03-data-warehousing/dimensional-modeling/dimensional-modeling.md). Pairs with [Project 07 (lakehouse — input data)](../07-data-lakehouse/), [Project 10 (semantic layer — query consumer)](../10-warehouse-semantic-layer/), [Project 52 (time-series DB — sibling columnar store)](../52-time-series-database/), [Project 20 (SIMD analytics — vectorized execution)](../20-simd-analytics-engine/), [Project 36 (streaming analytics SQL planner)](../36-distributed-streaming-analytics/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        SQL Query                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │      SQL Parser       │
              │   (pest/nom/LALR)     │
              └───────────┬───────────┘
                          │ AST
              ┌───────────▼───────────┐
              │    Query Analyzer     │
              │  (binding/type check) │
              └───────────┬───────────┘
                          │ Bound AST
              ┌───────────▼───────────┐
              │   Logical Planner     │
              │  (relational algebra) │
              └───────────┬───────────┘
                          │ Logical Plan
              ┌───────────▼───────────┐
              │      Optimizer        │
              │ (rule + cost-based)   │
              └───────────┬───────────┘
                          │ Optimized Plan
              ┌───────────▼───────────┐
              │   Physical Planner    │
              │  (execution strategy) │
              └───────────┬───────────┘
                          │ Physical Plan
              ┌───────────▼───────────┐
              │   Execution Engine    │
              │    (vectorized)       │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │    Storage Engine     │
              │     (columnar)        │
              └───────────────────────┘
```

### Core Components

#### 1. Parser/Frontend
- **Lexer**: Tokenize SQL input
- **Parser**: Build abstract syntax tree
- **Binder**: Resolve names, check types
- **Output**: Bound query tree with type information

#### 2. Planner
- **Logical Planner**: Convert to relational algebra
- **Optimizer**: Apply transformation rules
- **Physical Planner**: Select execution algorithms
- **Output**: Executable physical plan

#### 3. Execution Engine
- **Vectorized Operators**: Process data in chunks
- **Pipeline Execution**: Stream data through operators
- **Memory Management**: Buffer pool, spilling

#### 4. Storage Engine
- **Columnar Layout**: Per-column data files
- **Compression**: Dictionary, RLE, bit-packing
- **Indexing**: Zone maps, bloom filters

## Storage Engine Internals

### Columnar Layout

```rust
// Table storage structure
struct Table {
    schema: Schema,
    row_groups: Vec<RowGroup>,
    statistics: TableStatistics,
}

struct RowGroup {
    num_rows: usize,
    columns: Vec<ColumnChunk>,
}

struct ColumnChunk {
    data_type: DataType,
    encoding: Encoding,
    compression: Compression,
    data: Vec<u8>,
    null_bitmap: BitVec,
    statistics: ColumnStatistics,
}

struct ColumnStatistics {
    null_count: usize,
    distinct_count: Option<usize>,
    min_value: Option<Value>,
    max_value: Option<Value>,
}
```

### Data Types

```rust
enum DataType {
    Boolean,
    Int8, Int16, Int32, Int64,
    UInt8, UInt16, UInt32, UInt64,
    Float32, Float64,
    Decimal(precision, scale),
    Date,
    Timestamp,
    Interval,
    String,
    Binary,
    List(Box<DataType>),
    Struct(Vec<(String, DataType)>),
}
```

### Encoding Schemes

```rust
enum Encoding {
    Plain,                    // Raw values
    Dictionary,               // Dictionary + indices
    RunLengthEncoding,        // For repeated values
    BitPacking,               // For small integers
    DeltaEncoding,            // For sorted/sequential
    DeltaLengthByteArray,     // For strings
}

// Dictionary encoding example
struct DictionaryEncoded {
    dictionary: Vec<Value>,           // Unique values
    indices: Vec<u16>,                // Index per value
    bit_width: u8,                    // Bits per index
}

// RLE encoding example
struct RLEEncoded {
    runs: Vec<(Value, u32)>,          // (value, count)
}
```

### Zone Maps

```rust
// Min/max indexes per chunk for predicate pushdown
struct ZoneMap {
    chunk_id: usize,
    min_value: Value,
    max_value: Value,
    null_count: usize,
}

impl ZoneMap {
    fn can_contain(&self, predicate: &Predicate) -> bool {
        match predicate {
            Predicate::Eq(val) => self.min_value <= val && val <= self.max_value,
            Predicate::Lt(val) => self.min_value < val,
            Predicate::Gt(val) => self.max_value > val,
            Predicate::Between(lo, hi) => self.max_value >= lo && self.min_value <= hi,
            _ => true,
        }
    }
}
```

## Vectorized Execution

### Vector Structure

```rust
const VECTOR_SIZE: usize = 2048;  // Tuples per vector

struct Vector {
    data_type: DataType,
    data: VectorData,
    validity: BitVec,              // Null bitmap
    selection: Option<SelectionVector>,
}

enum VectorData {
    Flat(FlatVector),              // Contiguous array
    Dictionary(DictVector),         // Dictionary encoded
    Constant(Value),               // Single value repeated
    Sequence(i64, i64),            // Start, increment
}

struct FlatVector {
    buffer: AlignedBuffer,         // Cache-aligned data
}

struct SelectionVector {
    indices: Vec<u16>,             // Selected positions
    count: usize,
}
```

### Chunk Processing

```rust
struct DataChunk {
    columns: Vec<Vector>,
    count: usize,                  // Active rows
}

impl DataChunk {
    fn slice(&self, offset: usize, length: usize) -> DataChunk;
    fn append(&mut self, other: &DataChunk);
    fn flatten(&mut self);         // Materialize selections
}
```

### Vectorized Operators

```rust
// Filter operator - processes entire vectors
fn vector_filter(input: &Vector, predicate: &CompiledExpr) -> SelectionVector {
    let mut selection = SelectionVector::new();

    match input.data_type {
        DataType::Int64 => {
            let data = input.as_slice::<i64>();
            for i in 0..input.count {
                if predicate.eval_int64(data[i]) {
                    selection.push(i);
                }
            }
        }
        // ... other types
    }

    selection
}

// Aggregation with vectorized sum
fn vector_sum_int64(input: &Vector, selection: &SelectionVector) -> i64 {
    let data = input.as_slice::<i64>();
    let mut sum: i64 = 0;

    // Tight loop for CPU cache efficiency
    for &idx in &selection.indices {
        sum += data[idx as usize];
    }

    sum
}
```

## Query Planning

### Logical Plan Nodes

```rust
enum LogicalPlan {
    Scan {
        table: TableRef,
        projection: Vec<usize>,
        filter: Option<Expr>,
    },
    Filter {
        input: Box<LogicalPlan>,
        predicate: Expr,
    },
    Project {
        input: Box<LogicalPlan>,
        expressions: Vec<Expr>,
    },
    Aggregate {
        input: Box<LogicalPlan>,
        group_by: Vec<Expr>,
        aggregates: Vec<AggregateExpr>,
    },
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        join_type: JoinType,
        condition: Expr,
    },
    Sort {
        input: Box<LogicalPlan>,
        order_by: Vec<(Expr, SortOrder)>,
    },
    Limit {
        input: Box<LogicalPlan>,
        offset: usize,
        limit: usize,
    },
}
```

### Physical Plan Nodes

```rust
enum PhysicalPlan {
    TableScan {
        table: TableRef,
        projection: Vec<usize>,
        filters: Vec<Expr>,
    },
    Filter {
        input: Box<PhysicalPlan>,
        predicate: CompiledExpr,
    },
    Project {
        input: Box<PhysicalPlan>,
        expressions: Vec<CompiledExpr>,
    },
    HashAggregate {
        input: Box<PhysicalPlan>,
        group_by: Vec<CompiledExpr>,
        aggregates: Vec<CompiledAggr>,
    },
    HashJoin {
        build: Box<PhysicalPlan>,
        probe: Box<PhysicalPlan>,
        build_keys: Vec<CompiledExpr>,
        probe_keys: Vec<CompiledExpr>,
        join_type: JoinType,
    },
    MergeSort {
        input: Box<PhysicalPlan>,
        order_by: Vec<(CompiledExpr, SortOrder)>,
    },
    TopN {
        input: Box<PhysicalPlan>,
        order_by: Vec<(CompiledExpr, SortOrder)>,
        limit: usize,
    },
}
```

### Optimization Rules

```rust
// Rule-based optimizer
trait OptimizationRule {
    fn apply(&self, plan: &LogicalPlan) -> Option<LogicalPlan>;
}

// Predicate pushdown
struct PredicatePushdown;
impl OptimizationRule for PredicatePushdown {
    fn apply(&self, plan: &LogicalPlan) -> Option<LogicalPlan> {
        // Push filters below projections and into scans
    }
}

// Projection pushdown
struct ProjectionPushdown;
impl OptimizationRule for ProjectionPushdown {
    fn apply(&self, plan: &LogicalPlan) -> Option<LogicalPlan> {
        // Only read required columns
    }
}

// Join reordering
struct JoinReorder;
impl OptimizationRule for JoinReorder {
    fn apply(&self, plan: &LogicalPlan) -> Option<LogicalPlan> {
        // Reorder joins based on cardinality estimates
    }
}
```

## API Design

### SQL Interface

```sql
-- DDL
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR,
    email VARCHAR,
    created_at TIMESTAMP
);

-- DML
INSERT INTO users VALUES (1, 'Alice', 'alice@example.com', NOW());
COPY users FROM 'data.csv' (FORMAT CSV, HEADER);

-- Queries
SELECT
    date_trunc('month', created_at) as month,
    COUNT(*) as user_count
FROM users
WHERE created_at >= '2024-01-01'
GROUP BY 1
ORDER BY 1;
```

### Programmatic API

```rust
// Database connection
let db = Database::open("analytics.db")?;
let conn = db.connect()?;

// Execute query
let result = conn.execute("SELECT * FROM users WHERE id = ?", &[&1])?;

// Stream results
let mut stmt = conn.prepare("SELECT * FROM large_table")?;
for chunk in stmt.query_chunked(1024)? {
    process_chunk(chunk)?;
}

// Bulk insert
let appender = conn.appender("users")?;
for user in users {
    appender.append_row(&[&user.id, &user.name, &user.email])?;
}
appender.flush()?;

// Catalog operations
let tables = conn.catalog().tables()?;
let schema = conn.catalog().table_schema("users")?;
```

### DataFrame API (Optional)

```rust
let df = conn.table("orders")?
    .filter(col("status").eq("completed"))?
    .select(&[col("customer_id"), col("total")])?
    .group_by(&[col("customer_id")])?
    .agg(&[sum(col("total")).alias("total_spent")])?
    .sort(&[col("total_spent").desc()])?
    .limit(10)?
    .collect()?;
```

## Enterprise Features

### Multi-threaded Execution

```rust
struct ParallelExecutor {
    thread_pool: ThreadPool,
    num_threads: usize,
}

impl ParallelExecutor {
    fn execute(&self, plan: PhysicalPlan) -> Result<DataChunk> {
        // Partition scan across threads
        let partitions = self.partition_scan(&plan);

        // Execute partitions in parallel
        let handles: Vec<_> = partitions
            .into_iter()
            .map(|p| self.thread_pool.spawn(move || execute_partition(p)))
            .collect();

        // Merge results
        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join())
            .collect::<Result<_>>()?;

        merge_results(results)
    }
}

// Parallel hash aggregate
struct ParallelHashAggregate {
    // Per-thread hash tables
    local_tables: Vec<Mutex<HashMap<GroupKey, AggState>>>,
    // Global hash table for merge
    global_table: RwLock<HashMap<GroupKey, AggState>>,
}
```

### Cost-Based Optimizer

```rust
struct CostModel {
    // Cardinality estimation
    fn estimate_cardinality(&self, plan: &LogicalPlan) -> f64;

    // Cost estimation
    fn estimate_cost(&self, plan: &PhysicalPlan) -> Cost;
}

struct Cost {
    cpu_cost: f64,     // CPU cycles
    io_cost: f64,      // Disk I/O
    memory: usize,     // Memory usage
}

// Statistics for estimation
struct TableStatistics {
    row_count: usize,
    column_stats: HashMap<String, ColumnStats>,
}

struct ColumnStats {
    distinct_count: usize,
    null_fraction: f64,
    histogram: Histogram,
    most_common_values: Vec<(Value, f64)>,
}
```

### Query Result Caching

```rust
struct QueryCache {
    cache: LruCache<QueryHash, CachedResult>,
    max_size: usize,
}

struct CachedResult {
    data: DataChunk,
    created_at: Instant,
    dependencies: Vec<TableId>,  // For invalidation
}

impl QueryCache {
    fn get(&self, query: &str) -> Option<&CachedResult>;
    fn put(&mut self, query: &str, result: DataChunk);
    fn invalidate(&mut self, table: TableId);
}
```

## Performance Considerations

### Memory Management

```rust
// Buffer pool for columnar data
struct BufferPool {
    buffers: HashMap<BufferId, BufferEntry>,
    total_size: usize,
    max_size: usize,
    eviction_policy: EvictionPolicy,
}

// Memory-mapped files for large datasets
struct MappedColumn {
    mmap: Mmap,
    offset: usize,
    length: usize,
}

// Spill to disk for large aggregations
struct SpillableHashTable {
    memory_table: HashMap<Key, Value>,
    spill_partitions: Vec<SpillPartition>,
    memory_limit: usize,
}
```

### Cache Optimization

```rust
// Cache-aligned vectors
#[repr(align(64))]
struct AlignedBuffer {
    data: Vec<u8>,
}

// Prefetching in scan
fn prefetch_next_chunk(chunk: &ColumnChunk) {
    unsafe {
        let ptr = chunk.data.as_ptr().add(PREFETCH_DISTANCE);
        std::arch::x86_64::_mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
}

// Batch processing for cache efficiency
const BATCH_SIZE: usize = 2048;  // Fits in L1 cache
```

### Benchmarks

| Query Type | Target (1M rows) | Target (100M rows) |
|------------|------------------|---------------------|
| Simple scan | < 10ms | < 500ms |
| Filter + Project | < 20ms | < 800ms |
| Hash aggregate | < 50ms | < 2s |
| Hash join | < 100ms | < 5s |
| Sort | < 100ms | < 3s |

## Implementation Phases

### Phase 1: Storage Engine (Weeks 1-3)
- [ ] Column chunk data structure
- [ ] Basic data types (integers, strings)
- [ ] Plain encoding
- [ ] Table/RowGroup organization
- [ ] CSV reader for data loading
- [ ] Memory buffer management

### Phase 2: Execution Engine (Weeks 4-6)
- [ ] Vector data structure
- [ ] DataChunk operations
- [ ] Scan operator
- [ ] Filter operator (vectorized)
- [ ] Project operator
- [ ] Basic expressions (arithmetic, comparison)

### Phase 3: SQL Parser (Weeks 7-8)
- [ ] Lexer implementation
- [ ] Parser (SELECT, FROM, WHERE, GROUP BY)
- [ ] AST data structures
- [ ] Name resolution/binding
- [ ] Type checking

### Phase 4: Query Planner (Weeks 9-10)
- [ ] Logical plan generation
- [ ] Physical plan generation
- [ ] Basic optimization rules
- [ ] Predicate pushdown
- [ ] Projection pushdown

### Phase 5: Aggregations (Weeks 11-12)
- [ ] Hash aggregate operator
- [ ] Aggregate functions (SUM, COUNT, AVG, MIN, MAX)
- [ ] GROUP BY execution
- [ ] HAVING clause

### Phase 6: Joins (Weeks 13-14)
- [ ] Hash join operator
- [ ] Build/probe phases
- [ ] Join types (INNER, LEFT, RIGHT)
- [ ] Multiple join conditions

### Phase 7: Advanced Features (Weeks 15-16)
- [ ] ORDER BY with sort operator
- [ ] LIMIT/OFFSET
- [ ] Subqueries
- [ ] Window functions (basic)

### Phase 8: Enterprise Features (Weeks 17-18)
- [ ] Multi-threaded scan
- [ ] Parallel aggregation
- [ ] Cost-based optimizer basics
- [ ] Dictionary encoding
- [ ] Column statistics

### Phase 9: Polish & Performance (Weeks 19-20)
- [ ] Query result caching
- [ ] Memory spilling
- [ ] Comprehensive benchmarks
- [ ] Performance tuning
- [ ] Documentation

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_vector_filter() {
    let vector = Vector::from_iter(0..100i64);
    let selection = vector_filter(&vector, &|x| x > 50);
    assert_eq!(selection.count(), 49);
}

#[test]
fn test_hash_aggregate() {
    let chunk = create_test_chunk();
    let agg = HashAggregate::new(
        vec![col("category")],
        vec![sum(col("amount"))],
    );
    let result = agg.execute(chunk)?;
    // Verify aggregation results
}
```

### SQL Tests

```rust
#[test]
fn test_simple_select() {
    let db = Database::in_memory()?;
    db.execute("CREATE TABLE t (a INT, b VARCHAR)")?;
    db.execute("INSERT INTO t VALUES (1, 'x'), (2, 'y')")?;

    let result = db.execute("SELECT a, b FROM t WHERE a > 1")?;
    assert_eq!(result.row_count(), 1);
    assert_eq!(result.column("a").get(0), Value::Int(2));
}

#[test]
fn test_aggregation() {
    let db = create_test_db()?;
    let result = db.execute("
        SELECT category, SUM(amount) as total
        FROM orders
        GROUP BY category
        ORDER BY total DESC
    ")?;
    // Verify results
}
```

### Property-Based Tests

```rust
proptest! {
    #[test]
    fn test_join_correctness(
        left in vec(row_strategy(), 0..1000),
        right in vec(row_strategy(), 0..1000)
    ) {
        let hash_result = hash_join(&left, &right);
        let nested_result = nested_loop_join(&left, &right);
        assert_eq!(hash_result, nested_result);
    }
}
```

### Benchmark Tests

```rust
#[bench]
fn bench_scan_1m_rows(b: &mut Bencher) {
    let table = create_table_with_rows(1_000_000);
    b.iter(|| {
        let scan = TableScan::new(&table, vec![0, 1, 2]);
        while let Some(chunk) = scan.next()? {
            black_box(chunk);
        }
    });
}
```

## Stretch Goals

### Parquet Reader
- Read Parquet files directly
- Leverage existing encoding
- Predicate pushdown to row groups
- Column projection

### JIT Compilation
- Compile expressions to native code
- Use LLVM or Cranelift
- Specialize for data types
- Inline predicates

### Advanced Joins
- Sort-merge join
- Index nested loop join
- Bloom filter for semi-joins
- Partitioned hash join

### Window Functions
- OVER clause parsing
- Window frame management
- ROW_NUMBER, RANK, DENSE_RANK
- Aggregate window functions

## Technology Stack

- **Language**: Rust
- **Parser**: pest or nom
- **Serialization**: Apache Arrow format compatibility
- **Testing**: proptest for property-based testing
- **Benchmarking**: criterion

## References

- [DuckDB Paper](https://duckdb.org/pdf/SIGMOD2019-demo-duckdb.pdf)
- [MonetDB/X100](https://www.cidrdb.org/cidr2005/papers/P19.pdf) - Vectorized execution
- [Volcano Model](https://paperhub.s3.amazonaws.com/dace52a42c07f7f8348b08dc2b186061.pdf)
- [How Query Engines Work](https://howqueryengineswork.com/)
- [CMU Database Course](https://15445.courses.cs.cmu.edu/)
