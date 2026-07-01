# Distributed Streaming Analytics

## Overview

This project is a Flink-inspired stream processing framework written from scratch in Python,
paired with a self-contained SQL query compiler and cost-based optimizer. It is built to teach
the design and correctness of the building blocks that real streaming and analytical engines are
made of — not to run a cluster.

The streaming side provides a `DataStream` fluent API, time-based windowing (tumbling, sliding,
session, global), keyed state primitives (value / list / map / reducing / aggregating), and a
checkpoint subsystem with pluggable storage. The SQL side provides a hand-written lexer and
recursive-descent parser for the analytical subset of SQL, an AST, a logical plan, a rule-based
logical optimizer, a physical planner with hash / broadcast / merge join strategies, and a
cost-based optimizer with dynamic-programming join enumeration and statistics-driven cardinality
estimation.

The two subsystems share a package (`streamanalytics`) but are otherwise decoupled: the SQL
planner produces physical plan trees as data, and the streaming engine executes `DataStream`
pipelines via asyncio. There is no wire-level distributed runtime, no physical-plan executor, and
no SQL-on-streams bridge. The concepts taught are: immutable operator-chain construction, keyed
windowing and triggers, keyed-state context switching, checkpoint snapshot/restore, SQL parsing
with precedence climbing, relational algebra plan trees, rule-based and cost-based query
optimization, and the classical Selinger DP join-enumeration algorithm.

### Scope and non-goals

In scope (built and tested):

- DataStream API with `map` / `filter` / `flat_map` / `key_by` / `union` / `sink` and an
  asyncio-based execution loop.
- Keyed streams with `reduce`, `sum`, `count`, `process`, and `window(...)`.
- Window assigners (tumbling, sliding, session, global) and triggers (event-time, processing-time,
  count) with `reduce`, `aggregate`, `process` application functions.
- A keyed-state hierarchy (`ValueState`, `ListState`, `MapState`, `ReducingState`,
  `AggregatingState`) with key-context switching.
- A `MemoryStateBackend` and a file-backed state backend (named `RocksDBStateBackend` for parity
  with Flink; in practice it serializes via pickle to a directory — see "Known Gaps").
- Checkpoint dataclass, file-based checkpoint storage, and a `CheckpointCoordinator` that snapshots
  a list of state backends and prunes history.
- A complete hand-written SQL lexer and recursive-descent parser covering
  SELECT/FROM/JOIN/WHERE/GROUP BY/HAVING/ORDER BY/LIMIT/UNION, plus IN, BETWEEN, LIKE, IS NULL,
  CASE, CAST, EXISTS, scalar subqueries, and the five core aggregate functions.
- Logical plan nodes (Scan, Filter, Project, Join, Aggregate, Sort, Limit, Union, Distinct) and a
  builder that turns a `SelectStatement` into a logical tree.
- Rule-based optimizer: PredicatePushdown (through Project and Join with conjunctive splitting),
  ProjectionPruning, JoinReordering (heuristic), ConstantFolding, CommonSubexpressionElimination
  (detection only).
- Physical plan nodes including `PhysicalHashJoin`, `PhysicalBroadcastJoin`, `PhysicalMergeJoin`,
  two-phase `PhysicalAggregate`, `PhysicalSort`, `PhysicalLimit`, and `PhysicalExchange` (hash /
  broadcast / gather / round-robin); a `PhysicalPlanner` that selects strategies, inserts shuffles,
  and ensures a singleton root distribution.
- Cost model with separate CPU / IO / network / memory costs, column-level statistics, selectivity
  estimation, and a `DPJoinOptimizer` implementing classical bottom-up DP join enumeration with
  memoization on subsets of the relation set.
- A `CostBasedOptimizer` that runs rule-based passes to fixpoint, then the DP join re-orderer, then
  a statistics-propagation pass.

Out of scope (not built): a distributed runtime, physical-plan execution, a SQL-on-streams bridge,
real connectors (Kafka, files, sockets), real RocksDB, and watermark generation/propagation. These
are detailed in "Known Gaps".

## Architecture

The two subsystems are independent. The streaming side executes; the SQL side compiles.

```mermaid
flowchart TD
    subgraph Streaming (executed)
        Env["StreamExecutionEnvironment"]
        Env -->|from_collection / from_elements / add_source| Source["async source (yields Event)"]
        Source -->|map / filter / flat_map / union| DataStream
        DataStream -->|key_by| KeyedStream
        KeyedStream -->|window| WindowedStream
        WindowedStream -->|reduce / aggregate / process| DataStream2["DataStream"]
        DataStream -->|sink| Results
        KeyedStream -.->|snapshot / restore| StateBackends["State backends + CheckpointCoordinator"]
    end
    subgraph SQL (planned, not executed)
        SQL["SQL text"] --> Lexer["SQLLexer"]
        Lexer --> Parser["SQLParser"]
        Parser --> Builder["LogicalPlanBuilder"]
        Builder --> Opt["LogicalOptimizer / CostBasedOptimizer"]
        Opt --> Planner["PhysicalPlanner"]
        Planner --> Plan["PhysicalPlan (data structure)"]
    end
```

The streaming pipeline is an immutable operator chain. Each transformation returns a new
`DataStream` whose operator list extends the parent's, producing flat chains rather than nested
wrappers. The asyncio execution loop iterates the async source and threads each event through the
operator list. Keyed streams attach per-key stateful closures; windowed streams buffer values per
`(key, window)` and apply a window function when a trigger fires. State backends and the checkpoint
coordinator sit beside the runtime; they are invoked explicitly by callers, not driven by a clock.

The SQL pipeline is a query compiler: text becomes a token stream, then an AST, then a logical
plan tree, which is rewritten by rule-based (and optionally cost-based) optimization, and finally
lowered to a physical plan that records join strategies and shuffle exchanges. The physical plan is
the terminal artifact — it is a data structure the tests inspect, with no executor consuming it.

### Package layout

```
36-distributed-streaming-analytics/
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

Approximate source size (excluding `__init__.py`s): streaming core 305 LOC, operators 282 LOC,
windowing 433 LOC, state and checkpointing 592 LOC, SQL AST 709 LOC, SQL parser 1064 LOC, SQL
logical plan 590 LOC, SQL rule-based optimizer 670 LOC, SQL physical planner 876 LOC, SQL cost
model and DP optimizer 1207 LOC. Total roughly 6,700 LOC of source and 3,400 LOC of tests.

## Core Components

### Streaming execution model

A `DataStream` is a small data structure carrying: `env` (back-reference to the environment),
`_source` (an async generator yielding `Event` objects, or `None` for derived streams),
`_operators` (a list of synchronous callables `Event -> Event | None | list[Event]`), `_sink`
(optional sink callback), and `_parallelism` (recorded but not enforced).

Each transformation (`map`, `filter`, `flat_map`, `key_by`) returns a *new* `DataStream` whose
`_operators` list extends the parent's. This keeps the API immutable while producing flat operator
chains. The execution loop (`DataStream.execute`) iterates the async source and threads each event
through the operator list, with three sentinel conventions:

- An operator returning `None` means "drop this event" (filter semantics).
- An operator returning a `list[Event]` means "fan out" (flat-map semantics); the loop emits each
  element to the sink/results and stops further operators on the same input.
- An operator returning an `Event` continues to the next operator.

`StreamExecutionEnvironment.execute_sync` wraps `asyncio.run(execute())`. Sources are constructed
via `from_collection(list)`, `from_elements(*items)` (sugar over `from_collection`), and
`add_source(iterator_factory)`. `union(*streams)` builds a fresh async source that interleaves the
parent and each argument stream's events into one chain, so a fan-in topology is expressed without
a dedicated merge operator.

The design consequence of the immutable-chain model is that there is no separate "compile" step:
the operator list *is* the physical representation of the streaming job. `execute` is the only place
that touches the source; everything upstream of it is pure structure-building. This makes a pipeline
trivially testable — a test can construct a `DataStream`, call `execute_sync`, and assert on the
returned list of results without any runtime or scheduler. The trade-off is that operator fusion,
reordering, and parallel partitioning (all of which a real engine performs between construction and
execution) are absent; the chain runs exactly as written.

### Operators

The functional operators on `DataStream` are inlined as closures, but `operators/operators.py` also
exposes an `Operator` base class hierarchy that mirrors the Flink-style API and is usable
standalone (tests construct these directly). All share an `Operator.process(element) ->
Iterator[R]` contract so they can be chained imperatively.

| Class               | Role                                                     |
|---------------------|----------------------------------------------------------|
| `Operator`          | Abstract base with `open` / `process` / `close`          |
| `MapOperator`       | 1-to-1                                                    |
| `FilterOperator`    | 1-to-(0,1)                                                |
| `FlatMapOperator`   | 1-to-N                                                    |
| `ReduceOperator`    | Keyed reduce holding `Dict(K, T)` state                  |
| `AggregateOperator` | Keyed aggregate with `create / add / get_result / merge` |
| `KeyByOperator`     | Partitioning marker; key extraction only                 |
| `UnionOperator`     | Pass-through                                              |
| `ProcessOperator`   | Keyed processing with a `ProcessContext` for state/timers|
| `CoMapOperator`     | Connected-stream pair (`process_first` / `process_second`)|
| `AsyncMapOperator`  | I/O-bound map (currently degraded to sync — see gaps)    |
| `SinkOperator`      | Terminal                                                 |

### Keyed streams

`KeyedStream` exposes `reduce(func)`, `sum(field=None)`, `count()`, `window(assigner) ->
WindowedStream`, and `process(func)`. Each method appends a stateful closure to `_operators`;
per-key state lives in the closure's enclosing dict. This is intentionally simple — production
parallel execution would need a key-group mapping and a per-key shard, but for a single-process
implementation the closure approach is sufficient and matches the test expectations.

### Windowing

Window assigners map an event to one or more windows. `Window` subclasses (`TumblingWindow`,
`SlidingWindow`, `SessionWindow`, `GlobalWindow`) use `@dataclass(eq=False)` so they remain
hashable via the base class's explicit `__hash__` / `__eq__` on `(start, end)` — necessary because
windows are used as dict keys.

| Assigner                  | Strategy                                                        |
|---------------------------|----------------------------------------------------------------|
| `TumblingWindowAssigner`  | `start = ts - (ts % size)`; one window per event               |
| `SlidingWindowAssigner`   | All sliding windows covering `ts`; emits size/slide windows    |
| `SessionWindowAssigner`   | `(ts, ts + gap)` per event (merge is left to downstream)       |
| `GlobalWindowAssigner`    | Single `(-inf, +inf)` window                                   |

Window math is done in millisecond integers internally and converted to seconds for the `Window`
object. Triggers decide when a window's contents are emitted:

- `EventTimeTrigger.on_event_time(time, window) -> time >= window.end`
- `ProcessingTimeTrigger.on_processing_time(time, window) -> time >= window.end`
- `CountTrigger.on_element(...)`: counts per `(key, window)` and fires when the count crosses the
  threshold, then resets the count so successive batches of N elements each emit one result.

The trigger protocol has three hooks (`on_element`, `on_event_time`, `on_processing_time`), but as
built only `on_element` is invoked from `WindowedStream`; the on-time hooks are wired up but never
fired because the runtime does not advance watermark or processing-time clocks (see "Known Gaps").

`WindowedStream.reduce(func)`, `aggregate(create, add, get_result)`, and `process(func)` each
install an operator that (1) computes `windows = assigner.assign_windows(value, timestamp)`, (2)
appends the value to the per-`(key, window)` buffer, (3) calls `trigger.on_element(...)` and, if it
returns `True`, applies the function to the buffer and emits one `Event` per fired window, and (4)
clears the fired window's buffer. Convenience helpers `sum(field=None)`, `count()`, `min(field=None)`,
and `max(field=None)` are sugar over `reduce` / `aggregate`. The window function hierarchy
(`ReduceFunction`, `AggregateFunction`, `ProcessWindowFunction`) all inherit from
`WindowFunction[T, R]` whose contract is `apply(key, window, elements) -> Iterator[R]`.

### State management

Each state object is keyed: `set_current_key(k)` switches the active key, and all subsequent
operations (`value()`, `update(...)`, `add(...)`, etc.) read or write the slot for that key. This
mirrors Flink's keyed-state model, where the operator runtime sets the key context before invoking
user code.

- `ValueState` — single value per key.
- `ListState` — list per key.
- `MapState` — a per-key dictionary (nested map: outer key is the stream key, inner key is a
  user-supplied string), not a single global map.
- `ReducingState` — reduces on `add` via `reduce_func: (T, T) -> T`.
- `AggregatingState` — accumulates incrementally and returns `get_result(acc)`.

Two concrete backends present the same checkpoint/restore protocol, so callers swap them freely:

- **MemoryStateBackend** holds the state-object dict in-process behind a `threading.Lock`.
  `checkpoint` walks each registered state and pickles its per-key map into the `state_handles`
  dict; `restore` reverses it.
- **RocksDBStateBackend** presents the same interface but writes/reads pickled state to
  per-descriptor files (e.g. `<db_path>/<name>.state`). It is *not* backed by RocksDB. `flush()`
  writes through to disk; `checkpoint()` first flushes, then delegates to an internal
  `MemoryStateBackend`. The misnomer is documented honestly in a module docstring.

### Checkpointing

`CheckpointStorage` is abstract; `FileCheckpointStorage(base_path)` writes `checkpoint_<id>.bin`
files and returns the path as the storage handle. `load(handle)` reads the file and reconstructs
the dataclass; `delete(handle)` unlinks it.

`CheckpointCoordinator.trigger_checkpoint(state_backends)` walks the supplied backends, asks each
to produce its own checkpoint, namespaces the handles with the backend index (`"<i>_<name>"`),
writes a combined `Checkpoint` to storage, and prunes anything beyond `_max_retained` (3).
`restore_latest(state_backends)` reverses the namespacing and dispatches each per-backend slice
back to its origin. The coordinator is *not* driven by the running stream — there is no thread
calling `trigger_checkpoint` on a clock; callers invoke it manually. The tests exercise the
snapshot/restore cycle but not in-flight checkpointing.

### SQL frontend: lexer and parser

`SQLLexer` is a single-pass tokenizer recognizing numeric literals (with `.` and `eE` for floats),
single- and double-quoted strings with `''` escape, back-tick and double-quote quoted identifiers,
50+ keywords (all join/set/null/case/order keywords), two-character operators (`!=`, `<>`, `<=`,
`>=`, `||`, `::`), single-character operators (`+ - * / % = < > ( ) , . ; :`), and `--` line and
`/* ... */` block comments. Position tracking (`line`, `column`) is maintained for error messages,
surfaced through `ParseError`.

`SQLParser` is a hand-written recursive-descent parser with classical precedence climbing for
expressions. Grammar coverage:

- `SELECT (DISTINCT|ALL) <items> FROM <table-or-join> (WHERE ...) (GROUP BY ... (WITH ROLLUP|CUBE))
  (HAVING ...) (ORDER BY ...) (LIMIT n (OFFSET m)) (UNION|INTERSECT|EXCEPT (ALL) <select>)`
- Join types: `INNER`, `LEFT (OUTER)`, `RIGHT (OUTER)`, `FULL (OUTER)`, `CROSS`, plus `USING (cols)`
  and `ON <expr>`.
- Expressions: logical (`AND`/`OR`/`NOT`), comparison (`= != <> < <= > >=`), arithmetic
  (`+ - * / %`), concatenation (`||`), `IS (NOT) NULL`, `(NOT) IN (list | subquery)`,
  `(NOT) BETWEEN ... AND ...`, `(NOT) LIKE ... (ESCAPE 'c')`,
  `CASE (operand) WHEN ... THEN ... (ELSE) END`, `CAST(expr AS type)`, `EXISTS (subquery)`, scalar
  subqueries, and parenthesized subexpressions.
- Aggregates: `COUNT`, `SUM`, `AVG`, `MIN`, `MAX` with optional `DISTINCT` and `FILTER (WHERE ...)`.
- Function calls: arbitrary `<ident>(args)` for non-aggregate functions.

The AST is dataclass-based with `Expression` as the abstract root. Every node has an `alias` and a
`data_type` (inferred lazily), a `children()` method, a `to_sql()` method that round-trips to a SQL
fragment (used by plan dumps and by `CommonSubexpressionElimination` as a structural hash), and a
`transform(func)` method that maps a function over the tree bottom-up with a
`_with_children(new_children)` slot for cloning. Statement nodes (`SelectStatement`,
`CreateTableStatement`, `InsertStatement`) compose the clause nodes and round-trip via `to_sql()`.

### Logical plan and rule-based optimizer

All logical plan nodes inherit from a `LogicalPlan` dataclass carrying a `Schema` and optional
`Statistics`. Each node exposes `children()`, `node_type()`, `to_string(indent=0)`,
`_with_children(...)`, and `collect_expressions()`.

| Node               | Children     | Carries                                  |
|--------------------|--------------|------------------------------------------|
| `LogicalScan`      | (leaf)       | table_name, schema, alias, projection    |
| `LogicalFilter`    | (input)      | predicate                                |
| `LogicalProject`   | (input)      | expressions, aliases                     |
| `LogicalJoin`      | (left,right) | join_type, condition, using_columns      |
| `LogicalAggregate` | (input)      | group_by, aggregates                     |
| `LogicalSort`      | (input)      | sort_expressions, ascending, nulls_first |
| `LogicalLimit`     | (input)      | limit, offset                            |
| `LogicalUnion`     | (left,right) | all (bool)                               |
| `LogicalDistinct`  | (input)      |                                          |

`LogicalPlanBuilder.build(stmt)` constructs the tree bottom-up in the order: `build_from_clause ->
build_where -> build_aggregation -> build_having -> build_projection -> build_distinct ->
build_order_by -> build_limit -> build_set_operation`. This order is significant: WHERE filters raw
inputs while HAVING filters aggregate output (both are `LogicalFilter`, placed at the appropriate
point in the tree).

The `LogicalOptimizer` applies a list of rules to fixpoint (default max 10 iterations); each rule
returns `(plan, changed)`:

- **PredicatePushdown** splits the predicate on `AND` into conjuncts, computes the set of tables
  each conjunct references, partitions them into (left-only, right-only, cross-side, remaining) when
  pushing through a join, wraps the left/right inputs in `LogicalFilter` for their conjuncts,
  combines cross-side conjuncts into the join condition, and leaves the remainder above the join.
  It also pushes through `LogicalProject` by rewriting filter-over-project as project-over-filter
  when the predicate references only input columns.
- **ProjectionPruning** walks top-down collecting referenced column names, discards unreferenced
  `LogicalProject` entries, and for each `LogicalScan` with a known `Schema` computes a projection
  index list embedded in the scan. It always preserves at least one expression per node.
- **JoinReordering (heuristic)** flattens nested INNER joins into a list of inputs and conditions,
  orders inputs by estimated row count (`stats.row_count` if present, else heuristic constants
  1000 / 100 / 10000), and rebuilds a left-deep tree threading the right condition between each pair.
- **ConstantFolding** works bottom-up over expressions: folds children first, and if all children
  of a `BinaryOp`/`UnaryOp` are `Literal`, evaluates the operator (via a Python dispatch table for
  `+ - * / % = <> < <= > >= AND OR ||` and `- NOT`) and substitutes the result. It updates the
  enclosing node's expressions for `LogicalFilter` and `LogicalProject`.
- **CommonSubexpressionElimination** counts expression occurrences across the plan using
  `Expression.to_sql()` as a structural hash and marks duplicates. The actual extraction into a
  shared projection is not implemented; this rule currently only detects duplication.

The driver iterates rules until no rule reports a change; `explain(plan)` produces a step-by-step
dump showing the plan after each rule.

The fixpoint loop matters for correctness as well as completeness: predicate pushdown can expose new
projection-pruning opportunities (a predicate pushed onto a scan narrows the columns that scan must
produce), and projection pruning can in turn expose constant-folding opportunities. Running each
rule once would miss these cascades, so the optimizer re-runs the full rule list until a complete
pass changes nothing, capped at ten iterations to guarantee termination even if a rule oscillates.
Each rule is written to be idempotent on an already-optimal plan — it returns `changed=False` rather
than rewriting to an equivalent form — which is what makes the fixpoint converge rather than loop.

### Physical planner

The physical plan is a separate node hierarchy with its own `PhysicalPlan` base. In addition to
`schema` and `stats`, each physical node carries a `distribution: DataDistribution` and a numeric
`cost`.

| Node                    | Notes                                                  |
|-------------------------|--------------------------------------------------------|
| `PhysicalScan`          | filters, partition_filters                             |
| `PhysicalFilter`        | predicate, input                                       |
| `PhysicalProject`       | expressions, input                                     |
| `PhysicalHashJoin`      | left_keys, right_keys, build_side                      |
| `PhysicalBroadcastJoin` | broadcast_side                                         |
| `PhysicalMergeJoin`     | requires sorted inputs                                 |
| `PhysicalAggregate`     | strategy (SINGLE/TWO_PHASE/STREAMING), is_partial      |
| `PhysicalSort`          | global_sort flag                                       |
| `PhysicalLimit`         | is_local flag                                          |
| `PhysicalExchange`      | exchange_type: hash / broadcast / gather / round_robin |
| `PhysicalUnion`         | inputs list, all flag                                  |

`DataDistribution.requires_exchange(required)` is the planner's main correctness check: given the
current distribution, does it need to insert a shuffle to satisfy a downstream operator's required
distribution? The planner uses it implicitly when inserting exchanges before joins, aggregates,
sorts, and limits. `PhysicalPlanner` is configured via `PlannerConfig` (broadcast thresholds,
target partition count, `prefer_merge_join`, `enable_adaptive`). Per-node rules:

- **Scan** emits a `PhysicalScan` with `DistributionType.HASH` and the configured target partition
  count.
- **Join** extracts equi-join keys (`a.x = b.y` conjuncts within a top-level AND), estimates each
  side's size from `stats`, and picks BROADCAST if either side fits the broadcast threshold, MERGE
  if `prefer_merge_join` is true, else HASH. Hash join inserts a hash exchange on both sides keyed
  by the respective join keys; broadcast join inserts a broadcast exchange on the smaller side;
  merge join inserts `PhysicalSort` on each side.
- **Aggregate** is two-phase by default (partial + hash-exchange-on-group-keys + final);
  single-phase only when the input is already hash-partitioned on the group keys. Global (no GROUP
  BY) aggregation gathers to a singleton before the final phase.
- **Sort** is local sort + gather + global sort. **Limit** is local limit (with `limit + offset`
  per partition) + gather + global limit. **Distinct** is lowered to a `PhysicalAggregate` with all
  columns as group-by and no aggregates.

`_ensure_root_distribution` wraps the root in a gather exchange so the final output is a singleton.
`explain(plan)` formats the tree with one indent level per child, including join keys and exchange
types.

The planner's central invariant is that every operator's input distribution satisfies that
operator's requirement. A hash join requires both inputs hash-partitioned on the join keys; an
aggregate requires its input hash-partitioned on the group keys; a global sort or limit requires a
singleton. The planner enforces this by asking each child for its output distribution and, where it
does not match the requirement, splicing in the appropriate `PhysicalExchange`. This is the same
"interesting orders / required properties" reasoning that production planners use, reduced to the
distribution dimension: the planner never silently assumes co-location, so the emitted tree is a
faithful (if unexecuted) description of where every shuffle would occur. Two-phase aggregation is
the clearest payoff — by computing partial aggregates before the shuffle, the exchange moves one row
per group per partition rather than every input row, which the cost model rewards directly.

### Cost-based optimization

The cost-based subsystem (`sql/cost_model.py`) is built on three layers: cost constants, a
statistics estimator, and a cost model that walks plans recursively. The cost constants are
deliberately PostgreSQL-flavored (relative ratios match its default coefficients) so the planner's
choices are intuitive; they separate CPU, I/O, network, and memory costs.

The `StatisticsEstimator` derives selectivity per predicate and propagates cardinality:

| Predicate                  | Selectivity                                               |
|----------------------------|-----------------------------------------------------------|
| `a = b`                    | `1 / distinct(a)` if known, else `0.1`                    |
| `a <> b`                   | `1 - eq_selectivity`                                      |
| `a < b` / `>` / `<=` / `>=`| linear interpolation in `(min, max)` if known, else `0.33`|
| `LIKE 'foo'`               | `0.05`                                                    |
| `LIKE 'foo%'`              | `0.1`                                                     |
| `LIKE '%foo%'`             | `0.5`                                                     |
| `IN (...)`                 | `0.25`                                                    |
| `BETWEEN ...`              | `0.25`                                                    |
| `AND`                      | `s(L) * s(R)` (independence assumption)                  |
| `OR`                       | `s(L) + s(R) - s(L)*s(R)`                                |
| `NOT x`                    | `1 - s(x)`                                                |

Each `estimate_*` method takes child stats and returns output stats. Join cardinality uses the
standard `max(L, R) * sel` formula for equi-joins, falling back to `L * R * sel` for
non-equi/cross joins. `CostModel.estimate_plan_cost(plan)` returns `(cost, output_stats)`, with
per-operator models: scan costs tuples plus pages; filter costs comparisons; project costs tuples
times expression count; hash join costs `build*hash*2.0 + probe*hash*1.0` plus build-side memory;
aggregate costs hashing plus per-aggregate tuples plus group memory; sort costs `n log2(n)` times a
sort factor plus memory; limit and distinct cost proportionally.

`DPJoinOptimizer.optimize_join_order(relations, conditions)` implements the classic bottom-up DP
algorithm: build a join graph from conditions; seed `dp[{table}]` per input; for each subset size
`s` from 2 to N, for each subset of that size, split into every non-empty proper sub-subset `L` and
complement `R`, and if both are in `dp` and a join condition crosses the cut, cost
`dp[L].cost + dp[R].cost + estimate_join_cost(L, R)` and keep the cheapest; return
`dp[all_tables].plan` (falling back to a left-deep tree if none found). Subset enumeration is done
by `_subsets_of_size`; the state space is `O(3^N)` over the number of relations, capped in practice
by only running DP on flattened INNER-join groups. `_find_join_condition(L, R, edges)` returns the
conjunction of all edges crossing the cut; cross joins appear in the search space but generally
lose on cost.

The DP optimizer's correctness rests on the principle of optimality: the cheapest plan for a set of
relations is built from the cheapest plans for two complementary subsets. Memoizing
`dp[frozenset]` ensures each subset's best plan is computed once and reused across every superset
that contains it, which is what collapses the naive factorial search into the `O(3^N)` bound — there
are `3^N` (subset, complement) splits to consider across all subset sizes. Because the DP runs only
on flattened groups of INNER joins (outer joins are not commutative/associative and are left in
place by `JoinReordering`), N stays small in practice, so the exponential factor is paid only on the
join-heavy fragments where reordering actually changes the cost.

`CostBasedOptimizer.optimize(plan)` runs three phases: (1) rule-based passes (constant folding,
predicate pushdown, projection pruning, CSE) to fixpoint; (2) cost-based join reordering
(`CostBasedJoinReordering`); (3) a statistics-propagation pass that recomputes each node's output
`Statistics` from its children so downstream estimates reflect the reordered tree.
`register_table_stats(name, stats)` feeds the estimator, and `explain(plan, verbose=False)` produces
a before/after dump with cost numbers and a savings percentage, which is the primary way the tests
assert that reordering actually lowered the estimated cost.

## Data Structures

### Event model

```python
@dataclass
class Event(Generic[T]):
    value: T
    timestamp: float = field(default_factory=time.time)
    key: Optional[Any] = None
    watermark: Optional[float] = None
```

The `watermark` field exists for API compatibility but is not populated or consumed by the runtime.
Event timestamps default to wall-clock time when unspecified, so user code that wants event time
must set `Event.timestamp` explicitly.

### Window

```python
@dataclass
class Window:
    start: float       # seconds since epoch
    end: float         # exclusive

    def max_timestamp(self) -> float:
        return self.end - 1
```

`TumblingWindow`, `SlidingWindow`, `SessionWindow`, and `GlobalWindow` are `@dataclass(eq=False)`
subclasses, hashable via the base class's `__hash__` / `__eq__` over `(start, end)`.

### Keyed state hierarchy

```python
class KeyedState(ABC, Generic[K, T]):
    # holds _current_key; set_current_key(k) switches the active slot
    ...

# ValueState[K, T]            single value per key
# ListState[K, T]             list per key
# MapState[K, V]              per-key dict[str, V]
# ReducingState[K, T]         reduce_func: (T, T) -> T, reduces on add
# AggregatingState[K, T, ACC, V]  incremental accumulate; get_result(acc) -> V
```

```python
class StateBackend(ABC):
    def create_value_state(self, descriptor) -> ValueState: ...
    def create_list_state(self, descriptor) -> ListState: ...
    def create_map_state(self, descriptor) -> MapState: ...
    def checkpoint(self, checkpoint_id) -> Checkpoint: ...
    def restore(self, checkpoint) -> None: ...
```

### Checkpoint

```python
@dataclass
class Checkpoint:
    checkpoint_id: int
    timestamp: float
    state_handles: Dict[str, bytes]   # name -> pickled bytes
    metadata: Dict[str, Any] = field(default_factory=dict)

    def serialize(self) -> bytes: ...        # pickle of the whole dataclass dict
    @staticmethod
    def deserialize(data: bytes) -> 'Checkpoint': ...
```

### Cost model and statistics

```python
@dataclass
class CostConstants:
    cpu_tuple_cost: float = 0.01
    cpu_index_cost: float = 0.005
    cpu_comparison_cost: float = 0.0025
    cpu_hash_cost: float = 0.02
    cpu_sort_comparison: float = 0.03
    seq_page_cost: float = 1.0
    random_page_cost: float = 4.0
    network_transfer_cost: float = 0.1       # per KB
    network_latency_cost: float = 10.0       # per message
    memory_tuple_cost: float = 0.001
    hash_build_factor: float = 2.0
    hash_probe_factor: float = 1.0
    sort_factor: float = 1.5
    merge_factor: float = 1.0
    default_row_count: int = 1000
    default_row_width: int = 100
    page_size: int = 8192

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

### Planner config

```python
@dataclass
class PlannerConfig:
    broadcast_threshold_bytes: int = 10 * 1024 * 1024  # 10 MB
    broadcast_threshold_rows: int = 100_000
    target_partition_count: int = 200
    prefer_merge_join: bool = False
    enable_adaptive: bool = True
```

## API Design

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

`streamanalytics.sql` additionally exposes the cost-based subsystem (`CostConstants`,
`TableStatistics`, `StatisticsEstimator`, `CostModel`, `DPJoinOptimizer`,
`CostBasedJoinReordering`, `CostBasedPhysicalPlanner`, `CostBasedOptimizer`).

### Streaming usage

```python
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

### SQL planning usage

```python
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

### Worked SQL example

For the query
`SELECT u.name, SUM(o.total) FROM users u JOIN orders o ON u.id = o.user_id WHERE o.total > 100 GROUP BY u.name`,
the pipeline produces:

```text
Token stream    [SELECT][IDENT u][DOT][IDENT name][COMMA][SUM]...
AST             SelectStatement(
                  select_list=[u.name, SUM(o.total)],
                  from=FromClause(users u, [INNER JOIN orders o ON ...]),
                  where=WhereClause(o.total > 100),
                  group_by=GroupByClause([u.name]))
Logical plan    LogicalAggregate -> LogicalProject -> LogicalFilter(o.total > 100)
                  -> LogicalJoin(INNER, users u, orders o, u.id = o.user_id)
                  -> LogicalScan(users), LogicalScan(orders)
Predicate       filter `o.total > 100` pushed below the join, onto the orders scan branch
  pushdown
Projection      both scans gain a projection over only referenced columns
  pruning         (id, name for users; user_id, total for orders)
Join            trivial here (two relations), no change
  reordering
Physical plan   PhysicalAggregate(TWO_PHASE, final)
                  -> Exchange(hash by [u.name])
                  -> PhysicalAggregate(TWO_PHASE, partial)
                  -> PhysicalHashJoin (or BroadcastJoin if orders is small)
                       over Exchange(hash by [u.id]) -> Scan(users, [id, name])
                       and  Exchange(hash by [o.user_id])
                              -> Filter(o.total > 100) -> Scan(orders, [user_id, total])
```

The physical tree above is what the planner emits and the tests inspect. No executor consumes it.

## Performance

This project targets correctness of building blocks, not throughput, and ships no benchmark suite —
so there are no measured latency or throughput numbers to report. The design choices that bound
cost are nonetheless explicit:

- **Operator chaining is flat, not nested.** Each transformation appends to a list of closures
  rather than wrapping the parent, so a pipeline of depth N costs N synchronous calls per event
  with no recursion overhead.
- **Window buffering is keyed.** Contents live in a `Dict[(key, window), list]`, so per-window
  emission touches only the fired window's buffer, which is cleared on fire to bound memory.
- **State is per-key with explicit key context**, avoiding global locks on the hot path beyond the
  single `threading.Lock` in `MemoryStateBackend`.
- **Checkpoint retention is bounded** to `_max_retained = 3`, so storage does not grow without
  limit across repeated snapshots.
- **The cost model is the performance story for the SQL side.** It uses PostgreSQL-flavored
  coefficients across CPU / I/O / network / memory to rank physical alternatives, and the DP join
  optimizer trades an `O(3^N)` search (bounded by running only on flattened INNER-join groups) for
  provably-optimal left-deep-or-bushy join orders under the cost model.

Because there is no distributed runtime and no physical-plan executor, end-to-end query latency is
not a meaningful metric here — the SQL subsystem stops at plan emission.

## Testing Strategy

The project ships with 302 tests across six modules, all passing under `pytest tests/ -v`. No
external services are required.

| Module                 | Focus                                                                 |
|------------------------|-----------------------------------------------------------------------|
| `test_datastream.py`   | DataStream transformations, KeyedStream, parallelism, sinks, Event semantics, complex pipelines |
| `test_windowing.py`    | Window equality and hashing, all four assigners, all three triggers (including per-key count behavior), window functions, end-to-end windowed reduce/aggregate/process |
| `test_state.py`        | Each state type under key switching, MemoryStateBackend, file-backed backend, integration scenarios |
| `test_checkpointing.py`| Checkpoint serialization round-trip, file storage, coordinator behavior including retention pruning, restore-latest, recovery, edge cases (empty backends, missing files) |
| `test_sql.py`          | Lexer token types, parser (every clause and expression form), logical plan builder, each optimizer rule, physical planner including join strategy selection, full end-to-end SQL to physical plan, AST `to_sql()` round-trip |
| `test_cost_model.py`   | Cost constants, statistics estimator with histograms and ranges, cost model per operator, DP join optimizer on small graphs, cost-based join reordering rule, physical planner integration |

The unit level exercises each component in isolation (a single assigner, one optimizer rule, one
state type under key switching). The integration level threads whole pipelines: end-to-end windowed
aggregations on the streaming side, and full SQL-text-to-physical-plan on the SQL side. Edge cases
are explicit — empty state backends, missing checkpoint files, per-key count-trigger resets, and
AST round-trips that re-parse `to_sql()` output. There is no benchmark suite; the project asserts
behavioral correctness rather than performance.

## Known Gaps

The project is honest about what was not built.

1. **No physical-plan executor.** `PhysicalScan`, `PhysicalHashJoin`, `PhysicalExchange`, etc. are
   pure data structures. Nothing walks the tree, reads rows, executes a shuffle, and returns a
   result set. The SQL subsystem is a compiler, not an engine.
2. **No distributed runtime.** `set_parallelism(...)` is recorded on the stream but ignored;
   everything runs single-threaded under `asyncio.run`. There is no worker process, network
   shuffle, or scheduler.
3. **Watermarks are non-functional.** `Event.watermark` exists but the runtime never populates or
   advances it. `WindowedStream._watermark` is initialized to 0.0 and never updated, so
   `EventTimeTrigger.on_event_time` is wired up but never fired. Window emission is driven entirely
   by `on_element`, so event-time tumbling windows emit on every element rather than on watermark
   advancement.
4. **`AsyncMapOperator` is sync.** Its body is documented as a "simplified sync version" — no async
   I/O batching or ordered/unordered result handling.
5. **`RocksDBStateBackend` is misnamed.** It writes pickle files per state descriptor to a
   directory; the module docstring documents this honestly.
6. **`CommonSubexpressionElimination` detects but does not eliminate.** It counts duplicate
   expressions and identifies candidates, but the extraction into a shared projection is marked as
   future work in a comment.
7. **`CheckpointCoordinator` is not driven by the runtime.** Tests exercise the snapshot/restore
   cycle directly; there is no `enable_checkpointing(interval)` plumbing that periodically calls
   `trigger_checkpoint`.
8. **No source/sink connectors.** Sources are in-process iterators (`from_collection`,
   `from_elements`, `add_source`); there is no Kafka, file, socket, or HTTP source.
9. **SQL and streaming are independent.** There is no `tableEnv`-style SQL-on-streams adapter; the
   SQL planner has no catalog wired to the `DataStream` API.
10. **No INTERSECT / EXCEPT lowering.** The parser accepts both, but
    `LogicalPlanBuilder._build_set_operation` lowers them to `LogicalUnion`.
11. **`LogicalDistinct`** is lowered correctly at the physical level (an all-columns hash aggregate)
    but lacks its own rule-based optimizations (e.g. DISTINCT-of-DISTINCT collapse).

## References

- Apache Flink documentation — the DataStream API, windowing semantics, and the Asynchronous
  Barrier Snapshotting paper (Carbone et al., 2015).
- "Access Path Selection in a Relational Database Management System" (Selinger et al., 1979) — the
  classical DP join enumeration this project follows.
- PostgreSQL `costsize.c` — the cost coefficients here are deliberately reminiscent.
- "DBMSs On A Modern Processor: Where Does Time Go?" (Ailamaki et al., 1999) — context for the
  cost-model split into CPU / I/O / memory / network.
