# Project 36: Distributed Streaming Analytics - Technical Blueprint

## Executive Summary

> **Concepts covered:** [§02 Flink streaming](../../02-data-engineering/04-streaming/flink/flink-streaming.md) (this project *implements* the Flink-style execution model: DataStream API, windowing, keyed state, checkpoints) · [§02 Real-time analytics](../../02-data-engineering/04-streaming/real-time-analytics/real-time-analytics.md) · [§02 Data warehousing](../../02-data-engineering/03-data-warehousing/) (the SQL planner side). Compare to [Project 17 (columnar query engine)](../17-columnar-query-engine/) for the OLAP angle; ingest typically comes from [Project 12](../12-distributed-log-system/) or [Project 51](../51-message-queue/). Map: [`CONCEPT_TO_PROJECT_MAP.md`](../CONCEPT_TO_PROJECT_MAP.md).

A Flink-inspired stream processing framework written in Python, paired with a
self-contained SQL planner and cost-based optimizer. The streaming side
provides a `DataStream` fluent API, time-based windowing (tumbling, sliding,
session, global), keyed state primitives (value/list/map/reducing/aggregating),
and a checkpoint subsystem with pluggable storage. The SQL side provides a
hand-written lexer/parser for the analytical subset of SQL, an AST, a logical
plan, a rule-based logical optimizer, a physical planner with hash/broadcast/
merge join strategies, and a cost-based optimizer with dynamic-programming
join enumeration and statistics-driven cardinality estimation.

The two subsystems share a package (`streamanalytics`) but are otherwise
decoupled: the SQL planner produces physical plan trees as data, and the
streaming engine executes `DataStream` pipelines via asyncio. There is no
wire-level distributed runtime — the focus of this project is on the design
and correctness of the building blocks, not on cluster-level execution.

---

## Table of Contents

1. [Scope and Non-Goals](#scope-and-non-goals)
2. [Package Layout](#package-layout)
3. [Streaming Subsystem](#streaming-subsystem)
4. [Windowing](#windowing)
5. [State Management](#state-management)
6. [Checkpointing](#checkpointing)
7. [SQL Subsystem](#sql-subsystem)
8. [Logical Plan and Rule-Based Optimizer](#logical-plan-and-rule-based-optimizer)
9. [Physical Planner](#physical-planner)
10. [Cost-Based Optimization](#cost-based-optimization)
11. [Data Flow](#data-flow)
12. [Public API](#public-api)
13. [Testing Strategy](#testing-strategy)
14. [Implementation Phases (As Built)](#implementation-phases-as-built)
15. [Known Gaps and Stretch Goals](#known-gaps-and-stretch-goals)

---

## Scope and Non-Goals

### In scope (built and tested)

- DataStream API with map / filter / flat_map / key_by / union / sink and an
  asyncio-based execution loop.
- Keyed streams with `reduce`, `sum`, `count`, `process`, and `window(...)`.
- Window assigners (tumbling, sliding, session, global) and triggers
  (event-time, processing-time, count) with `reduce`, `aggregate`, `process`
  application functions.
- A keyed-state hierarchy (`ValueState`, `ListState`, `MapState`,
  `ReducingState`, `AggregatingState`) with key-context switching.
- A `MemoryStateBackend` and a file-backed state backend (named
  `RocksDBStateBackend` for parity with Flink; in practice it serializes via
  pickle to a directory — see "Known Gaps").
- Checkpoint dataclass, file-based checkpoint storage, and a
  `CheckpointCoordinator` that snapshots a list of state backends and prunes
  history.
- A complete hand-written SQL lexer and recursive-descent parser covering
  SELECT/FROM/JOIN/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT/UNION, plus IN,
  BETWEEN, LIKE, IS NULL, CASE, CAST, EXISTS, scalar subqueries, and the
  five core aggregate functions.
- Logical plan nodes (Scan, Filter, Project, Join, Aggregate, Sort, Limit,
  Union, Distinct) and a builder that turns a `SelectStatement` into a
  logical tree.
- Rule-based optimizer: PredicatePushdown (through Project and Join with
  conjunctive splitting), ProjectionPruning, JoinReordering (heuristic),
  ConstantFolding, CommonSubexpressionElimination (detection only).
- Physical plan nodes including `PhysicalHashJoin`, `PhysicalBroadcastJoin`,
  `PhysicalMergeJoin`, two-phase `PhysicalAggregate`, `PhysicalSort`,
  `PhysicalLimit`, and `PhysicalExchange` (hash / broadcast / gather /
  round-robin); a `PhysicalPlanner` that selects strategies, inserts
  shuffles, and ensures a singleton root distribution.
- Cost model with separate CPU / IO / network / memory costs, column-level
  statistics, selectivity estimation (equality, range, LIKE, IN, BETWEEN,
  AND/OR with independence assumption), and a `DPJoinOptimizer` implementing
  classical bottom-up DP join enumeration with memoization on subsets of
  the relation set.
- A `CostBasedOptimizer` that runs rule-based passes to fixpoint, then the
  DP join re-orderer, then a statistics-propagation pass.

### Out of scope (not built)

- A distributed runtime. Streams execute in a single Python process under
  asyncio. There is no worker pool, no shuffle network, no task scheduler.
- Physical-plan execution. `PhysicalScan`, `PhysicalExchange`, etc. are pure
  data structures that the planner emits and tests inspect. There is no
  executor that consumes a `PhysicalPlan` and produces rows.
- A SQL-on-streams bridge. The SQL planner has no `DataStream` source or
  catalog adapter; the two subsystems are independent.
- Real connectors (Kafka, files, sockets). Sources are
  `from_collection` / `from_elements` / `add_source(iterator)`.
- Real RocksDB. The "RocksDB" backend writes pickle files per state
  descriptor to a directory.
- Watermark generation/propagation. The `Event` dataclass has a `watermark`
  field but the runtime never advances it; window triggers are checked
  on-element rather than on-watermark.

---

## Package Layout

```
36-distributed-streaming-analytics/
├── BLUEPRINT.md                     (this file)
├── PROGRESS.md
├── README.md
├── pyproject.toml
├── src/streamanalytics/
│   ├── __init__.py                  Re-exports for top-level API
│   ├── core/
│   │   └── stream.py                Event, DataStream, KeyedStream,
│   │                                StreamExecutionEnvironment
│   ├── operators/
│   │   └── operators.py             Operator base class + Map/Filter/
│   │                                FlatMap/Reduce/Aggregate/KeyBy/Union/
│   │                                Process/CoMap/AsyncMap/Sink
│   ├── windowing/
│   │   └── windows.py               Window, assigners, triggers,
│   │                                WindowFunction, WindowedStream
│   ├── state/
│   │   └── backend.py               Keyed state classes, MemoryStateBackend,
│   │                                RocksDBStateBackend (file-backed),
│   │                                Checkpoint, FileCheckpointStorage,
│   │                                CheckpointCoordinator
│   └── sql/
│       ├── parser.py                Lexer + recursive-descent parser
│       ├── ast.py                   Expression and statement AST nodes
│       ├── logical.py               Logical plan nodes + LogicalPlanBuilder
│       ├── optimizer.py             Rule-based optimizer
│       ├── planner.py               PhysicalPlanner + physical operator nodes
│       └── cost_model.py            Cost constants, statistics estimator,
│                                    cost model, DP join optimizer,
│                                    CostBasedOptimizer
└── tests/                           302 tests, six modules
    ├── test_datastream.py
    ├── test_windowing.py
    ├── test_state.py
    ├── test_checkpointing.py
    ├── test_sql.py
    └── test_cost_model.py
```

Source size (excluding `__init__.py`s):
- streaming core: 305 LOC
- operators: 282 LOC
- windowing: 433 LOC
- state and checkpointing: 592 LOC
- SQL ast: 709 LOC
- SQL parser: 1064 LOC
- SQL logical plan: 590 LOC
- SQL rule-based optimizer: 670 LOC
- SQL physical planner: 876 LOC
- SQL cost model and DP optimizer: 1207 LOC

Total: ~6,700 LOC source, ~3,400 LOC tests.

---

## Streaming Subsystem

### High-level architecture

```
                  +-----------------------+
                  |  StreamExecution-     |
                  |  Environment          |
                  |  (env)                |
                  +-----------+-----------+
                              |
                              | from_collection / from_elements / add_source
                              v
+-----------+    map      +----------+    key_by    +-------------+
|  Source   +-----.-----> |DataStream+------------> | KeyedStream |
|  (async   |  filter     |          |              |             |
|  iterator)|  flat_map   +----------+              +------+------+
+-----------+   union           |                          |
                                | sink                     | window(...)
                                v                          v
                          +-----+-----+              +-----+-------+
                          |  Results  |              |WindowedStream|
                          +-----------+              +-----+-------+
                                                           |
                              reduce / aggregate / process |
                                                           v
                                                     +----------+
                                                     |DataStream|
                                                     +----------+
```

### Event model

```python
@dataclass
class Event(Generic[T]):
    value: T
    timestamp: float = field(default_factory=time.time)
    key: Optional[Any] = None
    watermark: Optional[float] = None
```

The `watermark` field exists for API compatibility but is not currently
populated or consumed by the runtime — see "Known Gaps". Event timestamps
default to wall-clock time when not specified, so user code that wants event
time must set `Event.timestamp` explicitly (or feed timestamped tuples and let
operators construct `Event`s).

### Execution model

A `DataStream` is a small data structure carrying:

- `env`: back-reference to the `StreamExecutionEnvironment`.
- `_source`: an async generator yielding `Event` objects, or `None` for
  derived streams.
- `_operators`: a list of synchronous callables `Event -> Event | None | list[Event]`.
- `_sink`: optional sink callback.
- `_parallelism`: configured parallelism (recorded; not enforced — the
  runtime is single-process single-threaded).

Each transformation (`map`, `filter`, `flat_map`, `key_by`) returns a *new*
`DataStream` whose `_operators` list extends the parent's. This keeps the API
immutable while producing flat operator chains rather than nested wrappers.

The execution loop (`DataStream.execute`) iterates the async source and
threads each event through the operator list. Three sentinel values matter:

- An operator returning `None` means "drop this event" (filter semantics).
- An operator returning a `list[Event]` means "fan out" (flat-map semantics);
  the loop emits each element to the sink/results and stops further operators
  on the same input.
- An operator returning an `Event` continues to the next operator.

`StreamExecutionEnvironment.execute_sync` wraps `asyncio.run(execute())`.

### Operators

The functional operators on `DataStream` are inlined as closures, but
`streamanalytics/operators/operators.py` also exposes an `Operator` base class
hierarchy that mirrors the Flink-style API:

| Class               | Role                                                    |
|---------------------|---------------------------------------------------------|
| `Operator`          | Abstract base with `open` / `process` / `close`         |
| `MapOperator`       | 1-to-1                                                   |
| `FilterOperator`    | 1-to-{0,1}                                               |
| `FlatMapOperator`   | 1-to-N                                                   |
| `ReduceOperator`    | Keyed reduce holding `Dict[K, T]` state                  |
| `AggregateOperator` | Keyed aggregate with `create / add / get_result / merge` |
| `KeyByOperator`     | Partitioning marker; key extraction only                 |
| `UnionOperator`     | Pass-through                                             |
| `ProcessOperator`   | Keyed processing with a `ProcessContext` for state/timers|
| `CoMapOperator`     | Connected-stream pair (`process_first` / `process_second`)|
| `AsyncMapOperator`  | I/O-bound map (currently degraded to sync — see gaps)    |
| `SinkOperator`      | Terminal                                                 |

These are usable standalone (tests construct them directly) and parallel the
closures embedded in `DataStream`. They share an `Operator.process(element)
-> Iterator[R]` contract so they can be chained imperatively.

### Keyed streams

```python
class KeyedStream(DataStream[T], Generic[T, K]):
    def __init__(self, env, name, key_selector: Callable[[T], K]):
        ...
        self.key_selector = key_selector
        self._state_backend = None
```

`KeyedStream` exposes `reduce(func)`, `sum(field=None)`, `count()`,
`window(assigner) -> WindowedStream`, and `process(func)`. Each method
appends a stateful closure to `_operators`; per-key state lives in the
closure's enclosing dict.

This is intentionally simple — for production parallel execution you would
need a key-group mapping and a per-key shard, but as a single-process
implementation the closure approach is sufficient and matches the test
expectations.

---

## Windowing

### Window types

```python
@dataclass
class Window:
    start: float       # seconds since epoch
    end: float         # exclusive

    def max_timestamp(self) -> float:
        return self.end - 1
```

Subclasses (`TumblingWindow`, `SlidingWindow`, `SessionWindow`, `GlobalWindow`)
all use `@dataclass(eq=False)` so they remain hashable via the base class's
explicit `__hash__` / `__eq__` on `(start, end)` — necessary because windows
are used as dict keys in trigger/content storage.

### Window assigners

| Assigner                  | Strategy                                                       |
|---------------------------|----------------------------------------------------------------|
| `TumblingWindowAssigner`  | `start = ts - (ts % size)`; one window per event              |
| `SlidingWindowAssigner`   | All sliding windows covering `ts`; emits N = size/slide windows|
| `SessionWindowAssigner`   | `[ts, ts + gap)` per event (merge is left to downstream)       |
| `GlobalWindowAssigner`    | Single `(-inf, +inf)` window                                   |

Window math is done in millisecond integers internally and converted to
seconds for the `Window` object.

### Triggers

Triggers decide when a window's contents are emitted:

- `EventTimeTrigger.on_event_time(time, window) -> time >= window.end`
- `ProcessingTimeTrigger.on_processing_time(time, window) -> time >= window.end`
- `CountTrigger.on_element(...)`: counts per `(key, window)` and fires
  when the count crosses the threshold; resets the count after firing so
  successive batches of N elements each emit one window result.

The trigger protocol has three hooks (`on_element`, `on_event_time`,
`on_processing_time`), but as built, only `on_element` is invoked from
`WindowedStream`. The on-time hooks are wired up for future use but the
runtime does not advance watermark or processing-time clocks — see gaps.

### WindowedStream and window functions

```python
class WindowedStream(Generic[T, K]):
    def __init__(self, keyed_stream, assigner):
        self.keyed_stream = keyed_stream
        self.assigner = assigner
        self.trigger = assigner.get_default_trigger()
        self._window_contents: Dict[Tuple[K, Window], List[T]] = defaultdict(list)
```

`WindowedStream.reduce(func)`, `aggregate(create, add, get_result)`, and
`process(func)` each install an operator that:

1. Computes `windows = assigner.assign_windows(value, timestamp)`.
2. Appends the value to the per-`(key, window)` buffer.
3. Calls `trigger.on_element(value, ts, window, key)` and, if it returns
   `True`, applies the reduce/aggregate/process function to the buffer
   and emits one `Event` per fired window.
4. Clears the fired window's buffer.

Convenience helpers `sum(field=None)`, `count()`, `min(field=None)`, and
`max(field=None)` are sugar over `reduce` / `aggregate`.

The window function hierarchy (`ReduceFunction`, `AggregateFunction`,
`ProcessWindowFunction`) lives in `windows.py` alongside the runtime so users
can subclass instead of passing callables. All three inherit from
`WindowFunction[T, R]` whose contract is
`apply(key, window, elements) -> Iterator[R]`.

---

## State Management

### State hierarchy

```
KeyedState[K, T]   (abstract, holds _current_key)
 ├── ValueState[K, T]
 ├── ListState[K, T]
 ├── MapState[K, V]          (per-key dict[str, V])
 ├── ReducingState[K, T]     (reduce_func: (T, T) -> T)
 └── AggregatingState[K, T, ACC, V]
```

Each state object is keyed: `set_current_key(k)` switches the active key,
and all subsequent operations (`value()`, `update(...)`, `add(...)`, etc.)
read or write the slot for that key. This mirrors Flink's keyed-state
model, where the operator runtime is responsible for setting the key
context before invoking user code.

`MapState` is a per-key dictionary (i.e. nested map: outer key is the stream
key, inner key is a user-supplied string), not a single global map.
`ReducingState` reduces on `add`. `AggregatingState` accumulates incrementally
and returns the result of `get_result(acc)`.

### State backends

```python
class StateBackend(ABC):
    def create_value_state(self, descriptor) -> ValueState: ...
    def create_list_state(self, descriptor) -> ListState: ...
    def create_map_state(self, descriptor) -> MapState: ...
    def checkpoint(self, checkpoint_id) -> Checkpoint: ...
    def restore(self, checkpoint) -> None: ...
```

Two concrete backends:

- **MemoryStateBackend**: holds the state object dict in-process behind a
  `threading.Lock`. `checkpoint` walks each registered state and pickles
  its per-key map into the `state_handles` dict. `restore` does the
  reverse.
- **RocksDBStateBackend**: presents the same interface but writes/reads
  pickled state to per-descriptor files (e.g.
  `<db_path>/<name>.state`) under a base directory. It is *not* backed by
  RocksDB. `flush()` writes through to disk; `checkpoint()` first flushes,
  then delegates to an internal `MemoryStateBackend`. This is documented
  honestly in a module docstring.

Both backends expose the same checkpoint/restore protocol, so callers can
swap them freely.

---

## Checkpointing

### Checkpoint structure

```python
@dataclass
class Checkpoint:
    checkpoint_id: int
    timestamp: float
    state_handles: Dict[str, bytes]   # name -> pickled bytes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> bytes:     # pickle of the whole dataclass dict
    @staticmethod
    def deserialize(data: bytes) -> 'Checkpoint': ...
```

### Storage

`CheckpointStorage` is abstract; `FileCheckpointStorage(base_path)` writes
`checkpoint_<id>.bin` files and returns the path as the storage handle.
`load(handle)` reads the file and reconstructs the dataclass. `delete(handle)`
unlinks the file.

### Coordinator

```python
class CheckpointCoordinator:
    def __init__(self, storage, interval_ms=60000,
                 min_pause_ms=0, timeout_ms=600000):
        ...
        self._max_retained = 3

    def trigger_checkpoint(self, state_backends: List[StateBackend])
        -> Optional[Checkpoint]:
        ...

    def restore_latest(self, state_backends) -> bool: ...
```

`trigger_checkpoint` walks the supplied state backends, asks each to produce
its own checkpoint, namespaces the handles with the backend index
(`"<i>_<name>"`), writes a combined `Checkpoint` to storage, and prunes
anything beyond `_max_retained`. `restore_latest` reverses the namespacing
and dispatches each per-backend slice back to its origin.

The coordinator is *not* driven by the running stream — there is no
`EnvironmentImpl` thread that calls `trigger_checkpoint` on a clock. Callers
invoke it manually. The tests exercise the snapshot/restore cycle but not
in-flight checkpointing.

---

## SQL Subsystem

The SQL subsystem is a self-contained query compiler that turns SQL text into
a physical plan tree. It is the larger half of the project by line count and
the more architecturally interesting half.

### Pipeline

```
SQL text
   |
   v
+----------+
| SQLLexer |  produces stream of Token(type, value, line, column)
+----+-----+
     v
+----------+
| SQLParser|  recursive descent; produces SelectStatement (AST)
+----+-----+
     v
+----------------------+
|  LogicalPlanBuilder  |  AST -> LogicalPlan tree
+----+-----------------+
     v
+---------------------+    +----------------------+
|   LogicalOptimizer  | OR |  CostBasedOptimizer  |  fixpoint over rules
+----+----------------+    +----+-----------------+
     v                          v
            +----------+
            |Physical- |  selects join/agg strategies, inserts Exchanges
            | Planner  |
            +----+-----+
                 v
            PhysicalPlan  (data structure only — not executed in this project)
```

### Lexer

`SQLLexer` is a single-pass tokenizer. It recognizes:

- numeric literals (with `.` and `eE` for floats),
- single- and double-quoted strings with `''` escape,
- back-tick and double-quote quoted identifiers,
- 50+ keywords including all join/set/null/case/order keywords,
- two-character operators (`!=`, `<>`, `<=`, `>=`, `||`, `::`),
- single-character operators (`+ - * / % = < > ( ) , . ; :`),
- `--` line comments and `/* ... */` block comments.

Position tracking (`line`, `column`) is maintained for error messages, which
the parser surfaces through `ParseError`.

### Parser

`SQLParser` is a hand-written recursive-descent parser with classical
precedence climbing for expressions. Grammar coverage:

- `SELECT [DISTINCT|ALL] <items> FROM <table-or-join> [WHERE ...]
  [GROUP BY ... [WITH ROLLUP|CUBE]] [HAVING ...] [ORDER BY ...]
  [LIMIT n [OFFSET m]] [UNION|INTERSECT|EXCEPT [ALL] <select>]`
- Join types: `INNER`, `LEFT [OUTER]`, `RIGHT [OUTER]`, `FULL [OUTER]`,
  `CROSS`, plus `USING (cols)` and `ON <expr>`.
- Expressions: logical (`AND`/`OR`/`NOT`), comparison (`= != <> < <= > >=`),
  arithmetic (`+ - * / %`), concatenation (`||`), `IS [NOT] NULL`,
  `[NOT] IN (list | subquery)`, `[NOT] BETWEEN ... AND ...`,
  `[NOT] LIKE ... [ESCAPE 'c']`, `CASE [operand] WHEN ... THEN ... [ELSE]
  END`, `CAST(expr AS type)`, `EXISTS (subquery)`, scalar subqueries,
  parenthesized subexpressions.
- Aggregates: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX` with optional `DISTINCT`
  and `FILTER (WHERE ...)`.
- Function calls: arbitrary `<ident>(args)` for non-aggregate functions.

### AST

The AST is dataclass-based with `Expression` as the abstract root. Every node
has:

- an `alias` and a `data_type` (inferred lazily from context),
- a `children()` method returning child expressions,
- a `to_sql()` method that round-trips to a SQL fragment (used by
  `to_string()` plan dumps and by `CommonSubexpressionElimination` as a
  cheap structural hash),
- a `transform(func)` method that maps a function over the tree bottom-up,
  with a `_with_children(new_children)` slot for cloning.

Statement nodes (`SelectStatement`, `CreateTableStatement`,
`InsertStatement`) compose the clause nodes (`FromClause`, `JoinClause`,
`WhereClause`, `GroupByClause`, `HavingClause`, `OrderByClause`,
`LimitClause`) and round-trip via `to_sql()`.

---

## Logical Plan and Rule-Based Optimizer

### Logical plan nodes

All logical plan nodes inherit from a `LogicalPlan` dataclass that carries a
`Schema` and optional `Statistics`. Each node exposes `children()`,
`node_type()`, `to_string(indent=0)`, `_with_children(...)`, and
`collect_expressions()`.

| Node               | Children   | Carries                                  |
|--------------------|------------|------------------------------------------|
| `LogicalScan`      | (leaf)     | table_name, schema, alias, projection   |
| `LogicalFilter`    | [input]    | predicate                                |
| `LogicalProject`   | [input]    | expressions, aliases                     |
| `LogicalJoin`      | [left,right]| join_type, condition, using_columns     |
| `LogicalAggregate` | [input]    | group_by, aggregates                     |
| `LogicalSort`      | [input]    | sort_expressions, ascending, nulls_first |
| `LogicalLimit`     | [input]    | limit, offset                            |
| `LogicalUnion`     | [left,right]| all (bool)                              |
| `LogicalDistinct`  | [input]    |                                          |

`LogicalPlanBuilder.build(stmt)` constructs the tree bottom-up:

```
build_from_clause -> build_where -> build_aggregation -> build_having
  -> build_projection -> build_distinct -> build_order_by -> build_limit
  -> build_set_operation
```

This order is significant: WHERE is a filter on raw inputs, while HAVING
is a filter on aggregate output (both implemented as `LogicalFilter`, but
placed at the appropriate point in the tree).

### Rule-based optimizer

The optimizer (`LogicalOptimizer`) applies a list of rules to fixpoint
(default max 10 iterations). Each rule returns `(plan, changed)`:

#### PredicatePushdown

- Splits the predicate on `AND` into conjuncts.
- For each conjunct, computes the set of referenced tables.
- Partitions conjuncts into (left-only, right-only, cross-side, remaining)
  when pushing through a join.
- Wraps the left and right inputs in `LogicalFilter` for their respective
  conjuncts; combines cross-side conjuncts into the join condition;
  leaves the remainder above the join.
- Pushes through `LogicalProject` by rewriting filter-over-project as
  project-over-filter when the predicate references only input columns.

#### ProjectionPruning

- Walks the plan top-down, collecting the set of column names actually
  referenced in any node's expressions.
- For each `LogicalProject` discards entries whose alias/expression is
  unreferenced.
- For each `LogicalScan` whose `Schema` is known, computes a projection
  index list and embeds it in the scan (so a downstream physical scan
  can read only the necessary columns).
- Always preserves at least one expression per node.

#### JoinReordering (heuristic)

Flattens nested INNER joins into a list of inputs and a list of
conditions, orders inputs by estimated row count (`stats.row_count` if
present, else heuristic constants 1000 / 100 / 10000), and rebuilds a
left-deep tree, threading the right condition between each pair.

The cost-based DP version is a separate rule (`CostBasedJoinReordering`)
in `cost_model.py`.

#### ConstantFolding

Bottom-up over expressions:

- Folds children first.
- If all children of a `BinaryOp` or `UnaryOp` are `Literal`, evaluates
  the operator (with a Python dispatch table for `+ - * / % = <> < <= > >=
  AND OR ||` and `- NOT`) and substitutes the result.
- Updates the enclosing plan node's expressions (currently for
  `LogicalFilter` and `LogicalProject`; other nodes are passed through
  unchanged).

#### CommonSubexpressionElimination

Counts expression occurrences across the plan using `Expression.to_sql()`
as a structural hash. Marks duplicates. Note that the actual *elimination*
(extraction into a shared projection) is not implemented; this rule
currently only detects duplication. See "Known Gaps".

### Optimizer driver

```python
class LogicalOptimizer:
    def optimize(self, plan):
        current = plan
        for _ in range(self.max_iterations):
            changed = False
            for rule in self.rules:
                new_plan, rule_changed = rule.apply(current)
                if rule_changed:
                    changed = True
                    current = new_plan
            if not changed:
                break
        return current

    def explain(self, plan) -> str: ...
```

`explain` produces a step-by-step dump showing the plan after each rule.

---

## Physical Planner

### Physical plan nodes

The physical plan is a separate node hierarchy with its own `PhysicalPlan`
base. In addition to `schema` and `stats`, each physical node carries a
`distribution: DataDistribution` and a numeric `cost`.

| Node                       | Notes                                              |
|----------------------------|----------------------------------------------------|
| `PhysicalScan`             | filters, partition_filters                         |
| `PhysicalFilter`           | predicate, input                                   |
| `PhysicalProject`          | expressions, input                                 |
| `PhysicalHashJoin`         | left_keys, right_keys, build_side                  |
| `PhysicalBroadcastJoin`    | broadcast_side                                     |
| `PhysicalMergeJoin`        | requires sorted inputs                             |
| `PhysicalAggregate`        | strategy (SINGLE/TWO_PHASE/STREAMING), is_partial |
| `PhysicalSort`             | global_sort flag                                   |
| `PhysicalLimit`            | is_local flag                                      |
| `PhysicalExchange`         | exchange_type: hash / broadcast / gather / round_robin |
| `PhysicalUnion`            | inputs list, all flag                              |

### Data distribution

```python
class DistributionType(Enum):
    SINGLETON, HASH, BROADCAST, RANGE, ROUND_ROBIN, RANDOM

@dataclass
class DataDistribution:
    dist_type: DistributionType
    partition_columns: List[str]
    partition_count: int

    def requires_exchange(self, required: DataDistribution) -> bool: ...
```

`requires_exchange` is the planner's main correctness check: it answers
"given my current distribution, do I need to insert a shuffle to satisfy a
downstream operator that requires `required`?". The planner uses it
implicitly when inserting exchanges before joins, aggregates, sorts, and
limits.

### Planning rules

`PhysicalPlanner` is configured via `PlannerConfig`:

```python
@dataclass
class PlannerConfig:
    broadcast_threshold_bytes: int = 10 * 1024 * 1024  # 10 MB
    broadcast_threshold_rows: int = 100_000
    target_partition_count: int = 200
    prefer_merge_join: bool = False
    enable_adaptive: bool = True
```

Per-node rules:

- **Scan**: emits a `PhysicalScan` with `DistributionType.HASH` and the
  configured target partition count.
- **Join**: extracts equi-join keys (`a.x = b.y` conjuncts within a top-
  level AND), estimates each side's size from `stats`, and picks:
  - `BROADCAST` if either side fits within the broadcast threshold,
  - `MERGE` if `prefer_merge_join` is true,
  - `HASH` otherwise.
  Hash join inserts `PhysicalExchange(hash)` on both sides keyed by the
  respective join keys; broadcast join inserts a `broadcast` exchange on
  the smaller side; merge join inserts `PhysicalSort` on each side.
- **Aggregate**: two-phase by default (partial + hash-exchange-on-group-
  keys + final). Single-phase only when the input is already hash-
  partitioned on the group keys (detected via `output_partitioning`).
  Global (no GROUP BY) aggregation gathers to a singleton before the
  final phase.
- **Sort**: local sort + gather + global sort.
- **Limit**: local limit (with `limit + offset` per partition) + gather +
  global limit.
- **Distinct**: lowered to a `PhysicalAggregate` with all columns as
  group-by and no aggregates.

`_ensure_root_distribution` wraps the root in a gather exchange so the
final output is a singleton.

`explain(plan)` formats the tree with one indent level per child, including
join keys and exchange types.

---

## Cost-Based Optimization

The cost-based subsystem lives in `sql/cost_model.py`. It is built on three
layers: cost constants, a statistics estimator, and a cost model that walks
plans recursively.

### Cost constants

```python
@dataclass
class CostConstants:
    # CPU
    cpu_tuple_cost: float = 0.01
    cpu_index_cost: float = 0.005
    cpu_comparison_cost: float = 0.0025
    cpu_hash_cost: float = 0.02
    cpu_sort_comparison: float = 0.03
    # I/O
    seq_page_cost: float = 1.0
    random_page_cost: float = 4.0
    # Network
    network_transfer_cost: float = 0.1       # per KB
    network_latency_cost: float = 10.0       # per message
    # Memory
    memory_tuple_cost: float = 0.001
    # Hash join
    hash_build_factor: float = 2.0
    hash_probe_factor: float = 1.0
    # Sort/merge
    sort_factor: float = 1.5
    merge_factor: float = 1.0
    # Defaults
    default_row_count: int = 1000
    default_row_width: int = 100
    page_size: int = 8192
```

These are deliberately PostgreSQL-flavored (the relative ratios match its
default cost coefficients) so the planner's choices are intuitive.

### StatisticsEstimator

```python
@dataclass
class ColumnStatistics:
    distinct_count: int
    null_fraction: float
    min_value: Optional[Any]
    max_value: Optional[Any]
    histogram: Optional[List[Tuple[Any, float]]]
    most_common_values: Optional[List[Tuple[Any, float]]]

@dataclass
class TableStatistics:
    row_count: int
    row_width_bytes: int = 100
    column_stats: Dict[str, ColumnStatistics] = {}
```

Selectivity estimation:

| Predicate                  | Selectivity                                          |
|----------------------------|------------------------------------------------------|
| `a = b`                    | `1 / distinct(a)` if known, else `0.1`                |
| `a <> b`                   | `1 - eq_selectivity`                                  |
| `a < b` / `>` / `<=` / `>=`| linear interpolation in `[min, max]` if known else `0.33` |
| `LIKE 'foo'`               | `0.05`                                                |
| `LIKE 'foo%'`              | `0.1`                                                 |
| `LIKE '%foo%'`             | `0.5`                                                 |
| `IN (...)`                 | `0.25`                                                |
| `BETWEEN ...`              | `0.25`                                                |
| `AND`                      | `s(L) * s(R)` (independence assumption)               |
| `OR`                       | `s(L) + s(R) - s(L)*s(R)`                            |
| `NOT x`                    | `1 - s(x)`                                            |

Cardinality propagation: each `estimate_*` method takes child stats and
returns output stats. Join cardinality uses the standard
`max(L, R) * sel` formula for equi-joins, falls back to `L * R * sel` for
non-equi/cross.

### Cost model

`CostModel.estimate_plan_cost(plan) -> (cost: float, output_stats:
Statistics)`. Per-node cost model:

| Operator   | CPU                                       | I/O                | Memory                    | Network |
|------------|-------------------------------------------|--------------------|---------------------------|---------|
| Scan       | `rows * cpu_tuple_cost`                   | `pages * seq_page_cost` |                          |         |
| Filter     | `rows * cpu_comparison_cost`              |                    |                           |         |
| Project    | `rows * nexprs * cpu_tuple_cost`          |                    |                           |         |
| Hash Join  | `build*hash*2.0 + probe*hash*1.0`         |                    | `build * memory_tuple`    |         |
| Aggregate  | `rows*hash + rows*nagg*tuple`             |                    | `groups * memory_tuple`   |         |
| Sort       | `n log2(n) * sort_comp * sort_factor`     |                    | `n * memory_tuple`        |         |
| Limit      | `limit * cpu_tuple * 0.1`                 |                    |                           |         |
| Union all  | (children only)                           |                    |                           |         |
| Distinct   | `rows * cpu_hash`                         |                    | `out * memory_tuple`      |         |

The planner uses these models to pick between hash, broadcast, and merge
join strategies in `CostBasedPhysicalPlanner.select_join_strategy`, which
also folds in `network_transfer_cost` per KB shuffled and a per-partition
`network_latency_cost`.

### Dynamic-programming join enumeration

`DPJoinOptimizer.optimize_join_order(relations, conditions)` implements the
classic bottom-up DP algorithm:

```
1. Build join graph from conditions; each equi-join condition yields an
   edge (left_table, right_table, condition, selectivity).
2. dp: dict[frozenset[str], JoinNode]
3. Base case: for each input relation, dp[{table}] = JoinNode(plan, ...)
4. For size s = 2 .. N:
     For each subset S of size s:
       For each non-empty proper sub-subset L of S:
         R = S \ L
         if L in dp and R in dp:
           if condition exists between L and R:
             cost = dp[L].cost + dp[R].cost + estimate_join_cost(L, R)
             if cost < best:
               best_node = JoinNode(LogicalJoin(left=dp[L].plan, right=dp[R].plan, cond), ...)
       dp[S] = best_node
5. Return dp[all_tables].plan; fall back to a left-deep tree if no plan was found.
```

Subset enumeration is done by `_subsets_of_size`, which recursively picks
positions from the element list. The state space is O(3^N) over the number
of relations N, which the project caps in practice by only running DP on
flattened INNER-join groups.

`_find_join_condition(L, R, edges)` returns the conjunction of all edges
that cross the cut. Cross joins (no edge) are not pruned at construction
time; they appear in the search space but generally lose on cost.

### Integrated optimizer

```python
class CostBasedOptimizer:
    def optimize(self, plan):
        # Phase 1: rule-based to fixpoint
        for _ in range(self.max_iterations):
            changed = False
            for rule in self.rule_optimizer_rules:  # constant folding, predicate pushdown,
                                                    # projection pruning, CSE
                new_plan, rc = rule.apply(current)
                if rc: changed = True; current = new_plan
            if not changed: break

        # Phase 2: cost-based
        for rule in self.cost_based_rules:           # [CostBasedJoinReordering]
            current, _ = rule.apply(current)

        # Phase 3: statistics propagation
        return self._propagate_statistics(current)
```

`register_table_stats(name, stats)` feeds the estimator. `explain(plan,
verbose=False)` produces a before/after dump with cost numbers and savings
percentage.

---

## Data Flow

### Streaming side (executed)

```
+-------------------+
| async source      |
| yields Event[T]   |
+---------+---------+
          |
          v
   +------+-------------+
   |  closure 1 (map)   |  Event -> Event
   +------+-------------+
          v
   +------+-------------+
   |  closure 2 (filter)|  Event -> Event | None
   +------+-------------+
          v
   +------+-------------+
   |  closure 3 (keyed  |  Event -> Event with key
   |     reduce)        |  (state in closure dict)
   +------+-------------+
          v
   +------+-------------+
   |  closure 4 (window |  Event -> Event | list[Event] | None
   |     reduce)        |  (state in WindowedStream)
   +------+-------------+
          v
   +------+-------------+
   |       sink         |
   +--------------------+
```

### SQL side (planned, not executed)

```
"SELECT u.name, SUM(o.total)
   FROM users u JOIN orders o ON u.id = o.user_id
  WHERE o.total > 100
  GROUP BY u.name"

  ── lexer ──>   [SELECT][IDENT u][DOT][IDENT name][COMMA][SUM][...]
  ── parser ──>  SelectStatement(select_list=[u.name, SUM(o.total)],
                                 from=FromClause(users u, [INNER JOIN orders o ON ...]),
                                 where=WhereClause(o.total > 100),
                                 group_by=GroupByClause([u.name]))
  ── builder ──>  LogicalAggregate(group_by=[u.name], aggregates=[SUM(o.total)])
                  └── LogicalProject([u.name, SUM(o.total)])
                      └── LogicalFilter(o.total > 100)
                          └── LogicalJoin(INNER, users u, orders o, u.id = o.user_id)
                              ├── LogicalScan(users u)
                              └── LogicalScan(orders o)

  ── PredicatePushdown ──> filter `o.total > 100` pushed below the join,
                            onto the orders scan branch.

  ── ProjectionPruning ──>  both scans gain a projection over only the
                            columns referenced (id, name for users;
                            user_id, total for orders).

  ── CostBasedJoinReordering ──> trivial here (two relations), no change.

  ── PhysicalPlanner ──>     PhysicalAggregate(TWO_PHASE, is_partial=False)
                              └── PhysicalExchange(hash by [u.name])
                                  └── PhysicalAggregate(TWO_PHASE, is_partial=True)
                                      └── PhysicalHashJoin (or BroadcastJoin if orders is small)
                                          ├── PhysicalExchange(hash by [u.id])
                                          │   └── PhysicalScan(users, cols [id, name])
                                          └── PhysicalExchange(hash by [o.user_id])
                                              └── PhysicalFilter(o.total > 100)
                                                  └── PhysicalScan(orders, cols [user_id, total])
```

The physical tree above is what the planner emits and the tests inspect.
No executor consumes it.

---

## Public API

The package re-exports a curated subset from `streamanalytics/__init__.py`:

```python
from streamanalytics import (
    # Core
    Event, DataStream, StreamExecutionEnvironment, KeyedStream,
    # Operators
    MapOperator, FilterOperator, FlatMapOperator,
    ReduceOperator, AggregateOperator,
    # Windowing
    Window, TumblingWindow, SlidingWindow, SessionWindow,
    WindowAssigner, WindowFunction,
    # State
    StateBackend, MemoryStateBackend, RocksDBStateBackend,
    KeyedState, ValueState, ListState, MapState, Checkpoint,
    # SQL
    SQLParser, SQLLexer, LogicalPlanBuilder,
    LogicalOptimizer, PhysicalPlanner,
)
```

`streamanalytics.sql` additionally exposes the cost-based subsystem
(`CostConstants`, `TableStatistics`, `StatisticsEstimator`, `CostModel`,
`DPJoinOptimizer`, `CostBasedJoinReordering`, `CostBasedPhysicalPlanner`,
`CostBasedOptimizer`).

### Typical usage

```python
# Streaming
from streamanalytics import StreamExecutionEnvironment
from streamanalytics.windowing import TumblingWindowAssigner

env = StreamExecutionEnvironment.get_execution_environment()
results = (
    env.from_collection([{"user": "a", "amt": 1}, {"user": "b", "amt": 2}])
       .key_by(lambda e: e["user"])
       .window(TumblingWindowAssigner(size_ms=60_000))
       .reduce(lambda x, y: {"user": x["user"], "amt": x["amt"] + y["amt"]})
       .execute_sync()
)
```

```python
# SQL planning
from streamanalytics.sql import (
    SQLParser, LogicalPlanBuilder, CostBasedOptimizer,
    PhysicalPlanner, TableStatistics,
)

stmt = SQLParser("SELECT u, COUNT(*) FROM orders WHERE total > 0 GROUP BY u").parse()
logical = LogicalPlanBuilder().build(stmt)

cbo = CostBasedOptimizer()
cbo.register_table_stats("orders", TableStatistics(row_count=1_000_000))
optimized = cbo.optimize(logical)

physical = PhysicalPlanner().plan(optimized)
print(physical.to_string())
```

---

## Testing Strategy

The project ships with 302 tests across six modules:

| Module                | Focus                                                  |
|-----------------------|--------------------------------------------------------|
| `test_datastream.py`  | DataStream transformations, KeyedStream, parallelism, sinks, Event semantics, complex pipelines |
| `test_windowing.py`   | Window equality and hashing, all four assigners, all three triggers (including per-key count behavior), window functions, end-to-end windowed reduce/aggregate/process |
| `test_state.py`       | Each state type's behavior under key switching, MemoryStateBackend, file-backed backend, integration scenarios |
| `test_checkpointing.py`| Checkpoint serialization round-trip, file storage, coordinator behavior including retention pruning, restore-latest, recovery scenarios, edge cases (empty backends, missing files) |
| `test_sql.py`         | Lexer (54 token types), parser (every clause and expression form), logical plan builder, each optimizer rule, physical planner including join strategy selection, full end-to-end SQL → physical plan, AST `to_sql()` round-trip |
| `test_cost_model.py`  | Cost constants, statistics estimator with histograms and ranges, cost model per-operator, DP join optimizer correctness on small graphs, cost-based join reordering rule, physical planner integration |

All tests pass under `pytest tests/ -v`. There is no benchmark suite — the
project focuses on correctness of building blocks rather than throughput.

---

## Implementation Phases (As Built)

### Phase 1 — Core streaming primitives

- `Event` dataclass with timestamp / key / watermark fields.
- `DataStream` and `KeyedStream` with the closure-based operator chain.
- `StreamExecutionEnvironment` and `execute_sync` via `asyncio.run`.
- Sources: `from_collection`, `from_elements`, `add_source`.

### Phase 2 — Operators module

- `Operator` base class and concrete subclasses for each transformation.
- `ProcessContext` for keyed processing with named state slots and timers.

### Phase 3 — Windowing

- Window types and assigners (tumbling / sliding / session / global).
- Triggers (event-time / processing-time / count) keyed per `(key, window)`
  for the count trigger.
- `WindowedStream` with `reduce / aggregate / process / sum / count / min / max`.

### Phase 4 — State and checkpointing

- Keyed state hierarchy with key-context switching.
- `MemoryStateBackend` and file-backed (`RocksDBStateBackend`) backends.
- `Checkpoint` dataclass and `FileCheckpointStorage`.
- `CheckpointCoordinator` with retention and multi-backend snapshotting.

### Phase 5 — SQL frontend

- `SQLLexer` with full keyword set and position tracking.
- `SQLParser` covering the analytical SQL subset (SELECT / JOIN / GROUP BY /
  HAVING / ORDER BY / LIMIT / set ops / subqueries).
- AST nodes with `to_sql()` round-trip and `transform()` traversal.

### Phase 6 — Logical plan and rule-based optimization

- Logical plan node hierarchy with `transform` and `collect_expressions`.
- `LogicalPlanBuilder.build(stmt)`.
- Five optimization rules driven to fixpoint by `LogicalOptimizer`.

### Phase 7 — Physical planner

- Physical plan nodes including the three join strategies and the
  `PhysicalExchange` operator with five distribution types.
- Heuristic join-strategy selection on size thresholds.
- Two-phase aggregation and global sort / limit lowerings.

### Phase 8 — Cost-based optimization

- `CostConstants`, `ColumnStatistics`, `TableStatistics`.
- `StatisticsEstimator` with per-predicate selectivity rules and range
  interpolation.
- `CostModel` with separate CPU / I/O / network / memory accounting.
- `DPJoinOptimizer` with classical bottom-up DP enumeration.
- `CostBasedOptimizer` orchestrating rule-based passes, DP join ordering,
  and statistics propagation.

---

## Known Gaps and Stretch Goals

The project is honest about what was *not* built. The list below is the
gap analysis that would precede any future work on this project.

### Genuine gaps in the current code

1. **No physical-plan executor.** `PhysicalScan`, `PhysicalHashJoin`,
   `PhysicalExchange`, etc. are pure data structures. There is no
   component that walks the tree, reads rows, executes a shuffle, and
   returns a result set. The SQL subsystem is a compiler, not an engine.
2. **No distributed runtime.** `set_parallelism(...)` is recorded on the
   stream but the runtime ignores it; everything runs single-threaded
   under `asyncio.run`. There is no worker process, no network shuffle,
   no scheduler.
3. **Watermarks are non-functional.** `Event.watermark` exists on the
   dataclass but the runtime never populates or advances it.
   `WindowedStream._watermark` is initialized to 0.0 and never updated,
   so the `EventTimeTrigger.on_event_time` hook is wired up but never
   fired by the engine. Window emission today is driven entirely by
   `on_element`, which means event-time tumbling windows in this code
   emit on every element rather than on watermark advancement.
4. **`AsyncMapOperator` is sync.** Its body is documented as a "simplified
   sync version" — there is no async I/O batching or unordered/ordered
   result handling.
5. **`RocksDBStateBackend` is misnamed.** It writes pickle files per
   state descriptor to a directory. The module docstring documents this
   honestly, but the public class name suggests RocksDB-backed storage.
6. **`CommonSubexpressionElimination` detects but does not eliminate.**
   The rule counts duplicate expressions across the plan and identifies
   candidates, but the actual extraction into a shared projection is
   marked as future work in a comment.
7. **`CheckpointCoordinator` is not driven by the runtime.** Tests
   exercise the snapshot/restore cycle directly. There is no
   `enable_checkpointing(interval)` plumbing that periodically calls
   `trigger_checkpoint` on the registered backends.
8. **No source/sink connectors.** Sources are in-process iterators
   (`from_collection`, `from_elements`, `add_source`). There is no
   Kafka, file, socket, or HTTP source.
9. **SQL and streaming are independent.** There is no `tableEnv`-style
   SQL-on-streams adapter; the SQL planner has no catalog wired to the
   `DataStream` API.
10. **No `INTERSECT` / `EXCEPT` lowering.** The parser accepts both, but
    `LogicalPlanBuilder._build_set_operation` lowers them to
    `LogicalUnion`. Real INTERSECT/EXCEPT plan nodes would be needed.
11. **`LogicalDistinct` is lowered correctly at the physical level (as
    an all-columns hash aggregate) but lacks its own rule-based
    optimizations (e.g. DISTINCT-of-DISTINCT collapse).

### Reasonable next steps

Each item below is roughly one focused work session.

1. **Implement a watermark-driven event-time runtime.** Have sources
   emit watermark markers; have `WindowedStream` advance an internal
   watermark and call `trigger.on_event_time(watermark, window)` for
   each open window when the watermark crosses `window.end`. Add
   "allowed lateness" handling for late events.
2. **Wire `CheckpointCoordinator` to the execution loop.** A
   coordinator task scheduled at `_checkpoint_interval` that walks the
   registered backends and writes a checkpoint without blocking the
   stream.
3. **Build a minimal physical-plan executor.** Even a single-process
   executor that turns `PhysicalScan` into a list iterator and
   `PhysicalHashJoin` into a build/probe loop would close the loop and
   let the cost-based planner be measured end-to-end.
4. **Replace `RocksDBStateBackend` with a real RocksDB binding** (the
   `python-rocksdb` package), with a per-key serialization scheme and a
   write-back cache.
5. **Add a file source connector** (JSON-lines / CSV) and a
   corresponding sink, both with checkpointed offsets for replayable
   sources.
6. **Promote `CommonSubexpressionElimination` from detection to
   elimination** by extracting a shared `LogicalProject` above the
   common ancestor of the duplicate occurrences and rewriting both
   sites to reference the projected alias.
7. **SQL-on-streams.** Provide a `StreamCatalog` that registers
   `DataStream` instances as tables (with declared schemas) and a
   physical-plan-to-DataStream lowering for streaming-safe operators
   (projection, filter, keyed aggregate over event-time windows).

### Stretch goals (substantial work)

- True parallel execution: per-key shards and a local worker pool that
  fans out operators across threads or processes.
- A wire-level shuffle protocol so the planner's `PhysicalExchange`
  nodes correspond to actual cross-node data movement.
- A SQL-level windowing extension (`GROUP BY TUMBLE(t, INTERVAL '1' MINUTE)`)
  and a planner pass that lowers it onto the streaming `WindowedStream`
  primitives.

---

## References

- Apache Flink documentation, especially the DataStream API, windowing
  semantics, and the Asynchronous Barrier Snapshotting paper
  (Carbone et al., 2015).
- "Access Path Selection in a Relational Database Management System"
  (Selinger et al., 1979) — the classical DP join enumeration this
  project follows.
- PostgreSQL `costsize.c` — the cost coefficients here are deliberately
  reminiscent.
- "DBMSs On A Modern Processor: Where Does Time Go?" (Ailamaki et al.,
  1999) — context for the cost-model split into CPU / IO / memory / network.
