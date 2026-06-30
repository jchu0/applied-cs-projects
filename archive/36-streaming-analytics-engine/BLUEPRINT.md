# Distributed Streaming Analytics Engine - Technical Blueprint

## Executive Summary

A massively parallel processing (MPP) execution engine inspired by Snowflake's architecture, designed for high-throughput streaming analytics with sophisticated query planning, distributed shuffle operations, and adaptive resource scheduling. This system enables real-time analytical queries over continuous data streams while maintaining SQL semantics and enterprise-grade reliability.

## System Architecture

### High-Level Architecture

```
+------------------+     +-------------------+     +------------------+
|   SQL Parser     |     |  Query Planner    |     | Cost Optimizer   |
|   (ANTLR/SQL)    | --> | (Logical Plan)    | --> | (Statistics)     |
+------------------+     +-------------------+     +------------------+
                                                           |
                              +----------------------------+
                              v
+------------------+     +-------------------+     +------------------+
| Physical Planner |     | Stage Builder     |     | Exchange Planner |
| (Operators)      | --> | (DAG)             | --> | (Shuffle/Sort)   |
+------------------+     +-------------------+     +------------------+
                                                           |
                              +----------------------------+
                              v
+----------------------------------------------------------+
|                    Coordinator Node                       |
|  +-------------+  +-------------+  +------------------+  |
|  | Stage       |  | Resource    |  | Failure          |  |
|  | Scheduler   |  | Allocator   |  | Recovery         |  |
|  +-------------+  +-------------+  +------------------+  |
+----------------------------------------------------------+
           |                    |                    |
           v                    v                    v
    +----------+         +----------+         +----------+
    | Worker 1 |         | Worker 2 |         | Worker N |
    | +------+ |         | +------+ |         | +------+ |
    | |Batch | |         | |Batch | |         | |Batch | |
    | |Exec  | |         | |Exec  | |         | |Exec  | |
    | +------+ |         | +------+ |         | +------+ |
    | +------+ |         | +------+ |         | +------+ |
    | |Memory| |         | |Memory| |         | |Memory| |
    | |Mgmt  | |         | |Mgmt  | |         | |Mgmt  | |
    | +------+ |         | +------+ |         | +------+ |
    +----------+         +----------+         +----------+
```

### Core Design Principles

1. **Decoupled Storage and Compute**: Stateless workers with external state stores
2. **Push-Based Execution**: Producers push batches to consumers for better pipelining
3. **Stage-Based Parallelism**: Queries split at shuffle boundaries into parallel stages
4. **Adaptive Execution**: Runtime optimization based on actual data characteristics
5. **Fault Tolerance**: At-least-once semantics with exactly-once achievable via idempotency

## Component Design

### 1. Query Planner (SQL to Logical Plan)

#### SQL Parser

```python
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum
import re

class ASTNodeType(Enum):
    SELECT = "select"
    PROJECT = "project"
    FILTER = "filter"
    JOIN = "join"
    AGGREGATE = "aggregate"
    SORT = "sort"
    LIMIT = "limit"
    SCAN = "scan"
    SUBQUERY = "subquery"

@dataclass
class Expression:
    """Base expression node"""
    pass

@dataclass
class ColumnRef(Expression):
    table: Optional[str]
    column: str
    alias: Optional[str] = None

@dataclass
class FunctionCall(Expression):
    name: str
    args: List[Expression]
    distinct: bool = False

@dataclass
class BinaryOp(Expression):
    op: str  # +, -, *, /, =, <>, <, >, <=, >=, AND, OR
    left: Expression
    right: Expression

@dataclass
class Literal(Expression):
    value: Union[int, float, str, bool, None]
    dtype: str

@dataclass
class LogicalPlan:
    """Logical plan node"""
    node_type: ASTNodeType
    children: List['LogicalPlan']
    expressions: List[Expression]
    schema: List[tuple]  # [(name, type), ...]
    properties: dict

class SQLParser:
    """
    Recursive descent SQL parser for analytical queries.
    Supports SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER BY, LIMIT, JOIN.
    """

    def __init__(self):
        self.tokens = []
        self.pos = 0

    def parse(self, sql: str) -> LogicalPlan:
        """Parse SQL string into logical plan tree"""
        self.tokens = self._tokenize(sql)
        self.pos = 0
        return self._parse_select()

    def _tokenize(self, sql: str) -> List[str]:
        """Tokenize SQL into words and operators"""
        pattern = r"(\b\w+\b|[<>=!]+|[(),;*+\-/])"
        tokens = re.findall(pattern, sql)
        return [t.upper() if t.isalpha() else t for t in tokens]

    def _parse_select(self) -> LogicalPlan:
        """Parse SELECT statement"""
        self._expect('SELECT')

        # Parse projection list
        distinct = self._match('DISTINCT')
        projections = self._parse_expression_list()

        # Parse FROM clause
        self._expect('FROM')
        from_plan = self._parse_from()

        # Parse WHERE clause
        if self._match('WHERE'):
            predicate = self._parse_expression()
            from_plan = LogicalPlan(
                node_type=ASTNodeType.FILTER,
                children=[from_plan],
                expressions=[predicate],
                schema=from_plan.schema,
                properties={'selectivity_estimate': 0.1}
            )

        # Parse GROUP BY clause
        if self._match('GROUP'):
            self._expect('BY')
            group_keys = self._parse_expression_list()

            # Parse HAVING clause
            having = None
            if self._match('HAVING'):
                having = self._parse_expression()

            from_plan = LogicalPlan(
                node_type=ASTNodeType.AGGREGATE,
                children=[from_plan],
                expressions=group_keys + ([having] if having else []),
                schema=self._infer_agg_schema(projections, group_keys),
                properties={'aggregates': self._extract_aggregates(projections)}
            )

        # Parse ORDER BY clause
        if self._match('ORDER'):
            self._expect('BY')
            sort_keys = self._parse_sort_keys()
            from_plan = LogicalPlan(
                node_type=ASTNodeType.SORT,
                children=[from_plan],
                expressions=sort_keys,
                schema=from_plan.schema,
                properties={'sort_directions': self._extract_sort_dirs(sort_keys)}
            )

        # Parse LIMIT clause
        if self._match('LIMIT'):
            limit_val = int(self._current())
            self._advance()
            from_plan = LogicalPlan(
                node_type=ASTNodeType.LIMIT,
                children=[from_plan],
                expressions=[Literal(limit_val, 'int')],
                schema=from_plan.schema,
                properties={'limit': limit_val}
            )

        # Add projection
        return LogicalPlan(
            node_type=ASTNodeType.PROJECT,
            children=[from_plan],
            expressions=projections,
            schema=self._infer_project_schema(projections),
            properties={'distinct': distinct}
        )

    def _parse_from(self) -> LogicalPlan:
        """Parse FROM clause with JOIN support"""
        left = self._parse_table_ref()

        while self._peek() in ('JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'CROSS'):
            join_type = 'INNER'
            if self._match('LEFT'):
                join_type = 'LEFT'
                self._match('OUTER')
            elif self._match('RIGHT'):
                join_type = 'RIGHT'
                self._match('OUTER')
            elif self._match('FULL'):
                join_type = 'FULL'
                self._match('OUTER')
            elif self._match('CROSS'):
                join_type = 'CROSS'

            self._expect('JOIN')
            right = self._parse_table_ref()

            condition = None
            if self._match('ON'):
                condition = self._parse_expression()

            left = LogicalPlan(
                node_type=ASTNodeType.JOIN,
                children=[left, right],
                expressions=[condition] if condition else [],
                schema=left.schema + right.schema,
                properties={'join_type': join_type}
            )

        return left

    def _parse_table_ref(self) -> LogicalPlan:
        """Parse table reference or subquery"""
        if self._match('('):
            subquery = self._parse_select()
            self._expect(')')
            alias = self._parse_alias()
            return LogicalPlan(
                node_type=ASTNodeType.SUBQUERY,
                children=[subquery],
                expressions=[],
                schema=subquery.schema,
                properties={'alias': alias}
            )

        table_name = self._current()
        self._advance()
        alias = self._parse_alias()

        return LogicalPlan(
            node_type=ASTNodeType.SCAN,
            children=[],
            expressions=[],
            schema=self._get_table_schema(table_name),
            properties={'table': table_name, 'alias': alias or table_name}
        )
```

#### Logical Plan Optimizer

```python
class LogicalOptimizer:
    """
    Rule-based logical plan optimizer.
    Applies transformations to improve query efficiency.
    """

    def __init__(self):
        self.rules = [
            self._predicate_pushdown,
            self._projection_pruning,
            self._join_reordering,
            self._subquery_decorrelation,
            self._common_subexpression_elimination,
        ]

    def optimize(self, plan: LogicalPlan) -> LogicalPlan:
        """Apply optimization rules until fixed point"""
        changed = True
        iterations = 0
        max_iterations = 10

        while changed and iterations < max_iterations:
            changed = False
            for rule in self.rules:
                new_plan, applied = rule(plan)
                if applied:
                    plan = new_plan
                    changed = True
            iterations += 1

        return plan

    def _predicate_pushdown(self, plan: LogicalPlan) -> tuple[LogicalPlan, bool]:
        """Push filters below joins and projections"""
        if plan.node_type != ASTNodeType.FILTER:
            # Recursively process children
            new_children = []
            any_changed = False
            for child in plan.children:
                new_child, changed = self._predicate_pushdown(child)
                new_children.append(new_child)
                any_changed = any_changed or changed

            if any_changed:
                return self._copy_with_children(plan, new_children), True
            return plan, False

        predicate = plan.expressions[0]
        child = plan.children[0]

        if child.node_type == ASTNodeType.JOIN:
            # Decompose predicate and push to appropriate side
            left_preds, right_preds, remaining = self._split_predicate(
                predicate,
                child.children[0].schema,
                child.children[1].schema
            )

            new_left = child.children[0]
            new_right = child.children[1]

            if left_preds:
                new_left = LogicalPlan(
                    node_type=ASTNodeType.FILTER,
                    children=[new_left],
                    expressions=[self._combine_predicates(left_preds)],
                    schema=new_left.schema,
                    properties={}
                )

            if right_preds:
                new_right = LogicalPlan(
                    node_type=ASTNodeType.FILTER,
                    children=[new_right],
                    expressions=[self._combine_predicates(right_preds)],
                    schema=new_right.schema,
                    properties={}
                )

            new_join = self._copy_with_children(child, [new_left, new_right])

            if remaining:
                return LogicalPlan(
                    node_type=ASTNodeType.FILTER,
                    children=[new_join],
                    expressions=[self._combine_predicates(remaining)],
                    schema=new_join.schema,
                    properties={}
                ), True

            return new_join, True

        return plan, False

    def _join_reordering(self, plan: LogicalPlan) -> tuple[LogicalPlan, bool]:
        """Reorder joins based on estimated cardinalities using dynamic programming"""
        if plan.node_type != ASTNodeType.JOIN:
            new_children = []
            any_changed = False
            for child in plan.children:
                new_child, changed = self._join_reordering(child)
                new_children.append(new_child)
                any_changed = any_changed or changed

            if any_changed:
                return self._copy_with_children(plan, new_children), True
            return plan, False

        # Collect all tables in join tree
        tables = self._collect_join_tables(plan)
        predicates = self._collect_join_predicates(plan)

        if len(tables) <= 2:
            return plan, False

        # Dynamic programming for optimal join order
        best_plan = self._dp_join_ordering(tables, predicates)
        return best_plan, True

    def _dp_join_ordering(self, tables: List[LogicalPlan],
                          predicates: List[Expression]) -> LogicalPlan:
        """
        Find optimal join order using dynamic programming.
        Considers join selectivity and intermediate result sizes.
        """
        n = len(tables)

        # dp[mask] = (cost, plan) for joining tables in bitmask
        dp = {}

        # Base case: single tables
        for i in range(n):
            mask = 1 << i
            dp[mask] = (self._estimate_cardinality(tables[i]), tables[i])

        # Fill DP table
        for size in range(2, n + 1):
            for mask in range(1, 1 << n):
                if bin(mask).count('1') != size:
                    continue

                best_cost = float('inf')
                best_plan = None

                # Try all ways to split this set
                submask = mask
                while submask:
                    complement = mask ^ submask
                    if complement and submask < complement:  # Avoid duplicates
                        if submask in dp and complement in dp:
                            # Find applicable predicates
                            join_preds = self._find_applicable_predicates(
                                tables, predicates, submask, complement
                            )

                            # Estimate join cost
                            left_cost, left_plan = dp[submask]
                            right_cost, right_plan = dp[complement]
                            join_cost = self._estimate_join_cost(
                                left_plan, right_plan, join_preds
                            )

                            total_cost = left_cost + right_cost + join_cost

                            if total_cost < best_cost:
                                best_cost = total_cost
                                best_plan = LogicalPlan(
                                    node_type=ASTNodeType.JOIN,
                                    children=[left_plan, right_plan],
                                    expressions=join_preds,
                                    schema=left_plan.schema + right_plan.schema,
                                    properties={'join_type': 'INNER'}
                                )

                    submask = (submask - 1) & mask

                if best_plan:
                    dp[mask] = (best_cost, best_plan)

        full_mask = (1 << n) - 1
        return dp[full_mask][1]
```

### 2. Physical Plan Builder

```python
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import uuid

class PhysicalOperator(Enum):
    TABLE_SCAN = "table_scan"
    INDEX_SCAN = "index_scan"
    FILTER = "filter"
    PROJECT = "project"
    HASH_JOIN = "hash_join"
    MERGE_JOIN = "merge_join"
    BROADCAST_JOIN = "broadcast_join"
    HASH_AGGREGATE = "hash_aggregate"
    SORT_AGGREGATE = "sort_aggregate"
    SORT = "sort"
    LIMIT = "limit"
    EXCHANGE = "exchange"
    SHUFFLE = "shuffle"
    BROADCAST = "broadcast"
    GATHER = "gather"
    UNION = "union"

class PartitioningScheme(Enum):
    SINGLE = "single"           # All data on one node
    ROUND_ROBIN = "round_robin" # Distribute evenly
    HASH = "hash"               # Partition by hash of key
    RANGE = "range"             # Partition by range of key
    BROADCAST = "broadcast"     # Copy to all nodes

@dataclass
class DataDistribution:
    scheme: PartitioningScheme
    partition_keys: List[str] = field(default_factory=list)
    num_partitions: int = 1
    sorted_by: List[str] = field(default_factory=list)

@dataclass
class PhysicalPlan:
    """Physical execution plan with distribution properties"""
    operator: PhysicalOperator
    children: List['PhysicalPlan']
    properties: Dict[str, Any]
    distribution: DataDistribution
    estimated_rows: int
    estimated_cost: float
    stage_id: Optional[str] = None

class PhysicalPlanner:
    """
    Converts logical plan to physical plan with exchange operators.
    Considers data distribution and parallelism.
    """

    def __init__(self, cluster_info: dict):
        self.num_workers = cluster_info['num_workers']
        self.worker_memory = cluster_info['worker_memory_mb']
        self.statistics = {}  # table -> column statistics

    def plan(self, logical: LogicalPlan) -> PhysicalPlan:
        """Convert logical plan to physical plan"""
        physical = self._to_physical(logical)
        physical = self._add_exchanges(physical)
        physical = self._assign_stages(physical)
        return physical

    def _to_physical(self, logical: LogicalPlan) -> PhysicalPlan:
        """Convert single logical node to physical"""

        if logical.node_type == ASTNodeType.SCAN:
            return self._plan_scan(logical)

        elif logical.node_type == ASTNodeType.FILTER:
            child = self._to_physical(logical.children[0])
            return PhysicalPlan(
                operator=PhysicalOperator.FILTER,
                children=[child],
                properties={'predicate': logical.expressions[0]},
                distribution=child.distribution,
                estimated_rows=int(child.estimated_rows * 0.1),  # Estimate 10% selectivity
                estimated_cost=child.estimated_cost + child.estimated_rows * 0.01
            )

        elif logical.node_type == ASTNodeType.PROJECT:
            child = self._to_physical(logical.children[0])
            return PhysicalPlan(
                operator=PhysicalOperator.PROJECT,
                children=[child],
                properties={'expressions': logical.expressions},
                distribution=child.distribution,
                estimated_rows=child.estimated_rows,
                estimated_cost=child.estimated_cost + child.estimated_rows * 0.001
            )

        elif logical.node_type == ASTNodeType.JOIN:
            return self._plan_join(logical)

        elif logical.node_type == ASTNodeType.AGGREGATE:
            return self._plan_aggregate(logical)

        elif logical.node_type == ASTNodeType.SORT:
            return self._plan_sort(logical)

        elif logical.node_type == ASTNodeType.LIMIT:
            child = self._to_physical(logical.children[0])
            return PhysicalPlan(
                operator=PhysicalOperator.LIMIT,
                children=[child],
                properties={'limit': logical.properties['limit']},
                distribution=child.distribution,
                estimated_rows=min(child.estimated_rows, logical.properties['limit']),
                estimated_cost=child.estimated_cost
            )

    def _plan_join(self, logical: LogicalPlan) -> PhysicalPlan:
        """Choose join strategy based on data sizes and distribution"""
        left = self._to_physical(logical.children[0])
        right = self._to_physical(logical.children[1])

        join_type = logical.properties['join_type']
        join_keys = self._extract_join_keys(logical.expressions[0])

        # Choose join strategy
        left_rows = left.estimated_rows
        right_rows = right.estimated_rows

        # Broadcast smaller side if it fits in memory
        broadcast_threshold = self.worker_memory * 0.3 * 1024 * 1024 / 100  # ~30% of memory, 100 bytes/row

        if right_rows < broadcast_threshold and right_rows < left_rows * 0.1:
            return PhysicalPlan(
                operator=PhysicalOperator.BROADCAST_JOIN,
                children=[left, right],
                properties={
                    'join_type': join_type,
                    'join_keys': join_keys,
                    'broadcast_side': 'right'
                },
                distribution=left.distribution,
                estimated_rows=self._estimate_join_rows(left_rows, right_rows),
                estimated_cost=left.estimated_cost + right.estimated_cost + left_rows
            )

        if left_rows < broadcast_threshold and left_rows < right_rows * 0.1:
            return PhysicalPlan(
                operator=PhysicalOperator.BROADCAST_JOIN,
                children=[left, right],
                properties={
                    'join_type': join_type,
                    'join_keys': join_keys,
                    'broadcast_side': 'left'
                },
                distribution=right.distribution,
                estimated_rows=self._estimate_join_rows(left_rows, right_rows),
                estimated_cost=left.estimated_cost + right.estimated_cost + right_rows
            )

        # Check if data is already partitioned on join keys
        if self._compatible_partitioning(left, right, join_keys):
            return PhysicalPlan(
                operator=PhysicalOperator.HASH_JOIN,
                children=[left, right],
                properties={
                    'join_type': join_type,
                    'join_keys': join_keys,
                    'needs_repartition': False
                },
                distribution=left.distribution,
                estimated_rows=self._estimate_join_rows(left_rows, right_rows),
                estimated_cost=left.estimated_cost + right.estimated_cost + left_rows + right_rows
            )

        # Need shuffle join - will add exchanges in _add_exchanges
        return PhysicalPlan(
            operator=PhysicalOperator.HASH_JOIN,
            children=[left, right],
            properties={
                'join_type': join_type,
                'join_keys': join_keys,
                'needs_repartition': True
            },
            distribution=DataDistribution(
                scheme=PartitioningScheme.HASH,
                partition_keys=join_keys['left'],
                num_partitions=self.num_workers
            ),
            estimated_rows=self._estimate_join_rows(left_rows, right_rows),
            estimated_cost=left.estimated_cost + right.estimated_cost +
                          (left_rows + right_rows) * 10  # Shuffle cost
        )

    def _plan_aggregate(self, logical: LogicalPlan) -> PhysicalPlan:
        """Plan aggregation with optional two-phase execution"""
        child = self._to_physical(logical.children[0])
        group_keys = [self._expr_to_string(e) for e in logical.expressions]
        aggregates = logical.properties['aggregates']

        # Single-phase aggregation if few groups expected or already partitioned
        estimated_groups = self._estimate_distinct_values(child, group_keys)

        if estimated_groups < 1000 or len(group_keys) == 0:
            # Single-phase hash aggregate
            return PhysicalPlan(
                operator=PhysicalOperator.HASH_AGGREGATE,
                children=[child],
                properties={
                    'group_keys': group_keys,
                    'aggregates': aggregates,
                    'phase': 'single'
                },
                distribution=DataDistribution(
                    scheme=PartitioningScheme.SINGLE,
                    num_partitions=1
                ),
                estimated_rows=estimated_groups,
                estimated_cost=child.estimated_cost + child.estimated_rows
            )

        # Two-phase aggregation: partial + final
        # Phase 1: Partial aggregation on each partition
        partial_agg = PhysicalPlan(
            operator=PhysicalOperator.HASH_AGGREGATE,
            children=[child],
            properties={
                'group_keys': group_keys,
                'aggregates': self._to_partial_aggregates(aggregates),
                'phase': 'partial'
            },
            distribution=child.distribution,
            estimated_rows=min(child.estimated_rows, estimated_groups * self.num_workers),
            estimated_cost=child.estimated_cost + child.estimated_rows
        )

        # Phase 2: Final aggregation (will add exchange in _add_exchanges)
        return PhysicalPlan(
            operator=PhysicalOperator.HASH_AGGREGATE,
            children=[partial_agg],
            properties={
                'group_keys': group_keys,
                'aggregates': self._to_final_aggregates(aggregates),
                'phase': 'final',
                'needs_repartition': True
            },
            distribution=DataDistribution(
                scheme=PartitioningScheme.HASH,
                partition_keys=group_keys,
                num_partitions=self.num_workers
            ),
            estimated_rows=estimated_groups,
            estimated_cost=partial_agg.estimated_cost + partial_agg.estimated_rows * 5
        )

    def _add_exchanges(self, plan: PhysicalPlan) -> PhysicalPlan:
        """Insert exchange operators where data distribution changes"""
        new_children = []

        for child in plan.children:
            child = self._add_exchanges(child)

            # Check if exchange needed
            required_dist = self._get_required_distribution(plan, child)
            if not self._satisfies_distribution(child.distribution, required_dist):
                exchange_type = self._choose_exchange_type(required_dist)
                child = PhysicalPlan(
                    operator=exchange_type,
                    children=[child],
                    properties={
                        'target_distribution': required_dist,
                        'partition_keys': required_dist.partition_keys
                    },
                    distribution=required_dist,
                    estimated_rows=child.estimated_rows,
                    estimated_cost=child.estimated_cost + child.estimated_rows * 5
                )

            new_children.append(child)

        return PhysicalPlan(
            operator=plan.operator,
            children=new_children,
            properties=plan.properties,
            distribution=plan.distribution,
            estimated_rows=plan.estimated_rows,
            estimated_cost=plan.estimated_cost
        )

    def _assign_stages(self, plan: PhysicalPlan, stage_id: str = None) -> PhysicalPlan:
        """Assign stage IDs - new stage at each exchange boundary"""
        if stage_id is None:
            stage_id = str(uuid.uuid4())[:8]

        plan.stage_id = stage_id

        new_children = []
        for child in plan.children:
            if child.operator in (PhysicalOperator.SHUFFLE,
                                  PhysicalOperator.BROADCAST,
                                  PhysicalOperator.GATHER):
                # Exchange creates new stage
                new_stage_id = str(uuid.uuid4())[:8]
                child = self._assign_stages(child.children[0], new_stage_id)
                # Wrap with exchange
                new_children.append(PhysicalPlan(
                    operator=child.operator if hasattr(child, 'original_op') else plan.children[plan.children.index(child)].operator,
                    children=[child],
                    properties=plan.children[new_children.__len__()].properties if len(plan.children) > len(new_children) else {},
                    distribution=child.distribution,
                    estimated_rows=child.estimated_rows,
                    estimated_cost=child.estimated_cost,
                    stage_id=stage_id
                ))
            else:
                child = self._assign_stages(child, stage_id)
                new_children.append(child)

        return PhysicalPlan(
            operator=plan.operator,
            children=new_children,
            properties=plan.properties,
            distribution=plan.distribution,
            estimated_rows=plan.estimated_rows,
            estimated_cost=plan.estimated_cost,
            stage_id=stage_id
        )
```

### 3. Cluster Execution Engine

```python
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
from enum import Enum
import time
import logging

class StageState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

@dataclass
class Task:
    """Unit of work assigned to a worker"""
    task_id: str
    stage_id: str
    partition_id: int
    operator_chain: List[PhysicalPlan]
    input_partitions: List[tuple]  # (stage_id, partition_id) of inputs
    state: TaskState = TaskState.PENDING
    worker_id: Optional[str] = None
    attempt: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metrics: Dict = None

@dataclass
class Stage:
    """Collection of parallel tasks with same operator chain"""
    stage_id: str
    tasks: List[Task]
    dependencies: List[str]  # stage_ids this stage depends on
    state: StageState = StageState.PENDING
    num_finished_tasks: int = 0

class Coordinator:
    """
    Central coordinator for distributed query execution.
    Manages stage scheduling, task assignment, and failure recovery.
    """

    def __init__(self, workers: List['WorkerClient']):
        self.workers = {w.worker_id: w for w in workers}
        self.stages: Dict[str, Stage] = {}
        self.running_tasks: Dict[str, Task] = {}
        self.pending_tasks: List[Task] = []
        self.logger = logging.getLogger('coordinator')

        # Configuration
        self.max_task_attempts = 3
        self.task_timeout_seconds = 300
        self.speculation_enabled = True
        self.speculation_threshold = 0.75

    async def execute_query(self, physical_plan: PhysicalPlan) -> 'QueryResult':
        """Execute a query plan across the cluster"""
        query_id = str(uuid.uuid4())[:8]
        self.logger.info(f"Starting query {query_id}")

        try:
            # Build stages from physical plan
            self._build_stages(physical_plan)

            # Execute stages in dependency order
            result = await self._execute_stages(query_id)

            return result

        except Exception as e:
            self.logger.error(f"Query {query_id} failed: {e}")
            await self._cleanup_query(query_id)
            raise

    def _build_stages(self, plan: PhysicalPlan):
        """Convert physical plan into execution stages"""
        # Group operators by stage_id
        stage_operators: Dict[str, List[PhysicalPlan]] = {}
        stage_deps: Dict[str, Set[str]] = {}

        def collect_stages(node: PhysicalPlan):
            stage_id = node.stage_id
            if stage_id not in stage_operators:
                stage_operators[stage_id] = []
                stage_deps[stage_id] = set()

            stage_operators[stage_id].append(node)

            for child in node.children:
                if child.stage_id != stage_id:
                    # Exchange boundary - dependency
                    stage_deps[stage_id].add(child.stage_id)
                collect_stages(child)

        collect_stages(plan)

        # Create stages with tasks
        for stage_id, operators in stage_operators.items():
            num_partitions = operators[0].distribution.num_partitions

            tasks = []
            for partition_id in range(num_partitions):
                task = Task(
                    task_id=f"{stage_id}-{partition_id}",
                    stage_id=stage_id,
                    partition_id=partition_id,
                    operator_chain=operators,
                    input_partitions=self._get_input_partitions(
                        operators, stage_deps[stage_id], partition_id
                    )
                )
                tasks.append(task)

            self.stages[stage_id] = Stage(
                stage_id=stage_id,
                tasks=tasks,
                dependencies=list(stage_deps[stage_id])
            )

    async def _execute_stages(self, query_id: str) -> 'QueryResult':
        """Execute stages respecting dependencies"""
        completed_stages: Set[str] = set()
        final_stage_id = None

        # Find root stage (no other stages depend on it)
        all_deps = set()
        for stage in self.stages.values():
            all_deps.update(stage.dependencies)

        for stage_id in self.stages:
            if stage_id not in all_deps:
                final_stage_id = stage_id
                break

        while len(completed_stages) < len(self.stages):
            # Find runnable stages
            runnable = []
            for stage_id, stage in self.stages.items():
                if stage.state == StageState.PENDING:
                    deps_met = all(d in completed_stages for d in stage.dependencies)
                    if deps_met:
                        runnable.append(stage_id)

            if not runnable:
                # Wait for running stages
                await asyncio.sleep(0.1)
                continue

            # Start runnable stages
            for stage_id in runnable:
                stage = self.stages[stage_id]
                stage.state = StageState.RUNNING

                # Add tasks to pending queue
                for task in stage.tasks:
                    self.pending_tasks.append(task)

                self.logger.info(f"Started stage {stage_id} with {len(stage.tasks)} tasks")

            # Schedule pending tasks
            await self._schedule_tasks()

            # Check for completed stages
            for stage_id, stage in self.stages.items():
                if stage.state == StageState.RUNNING:
                    if stage.num_finished_tasks == len(stage.tasks):
                        stage.state = StageState.FINISHED
                        completed_stages.add(stage_id)
                        self.logger.info(f"Completed stage {stage_id}")

        # Collect results from final stage
        return await self._collect_results(final_stage_id)

    async def _schedule_tasks(self):
        """Assign pending tasks to available workers"""
        while self.pending_tasks:
            # Find available worker
            worker = self._find_available_worker()
            if not worker:
                break

            # Get next task (could implement better scheduling here)
            task = self.pending_tasks.pop(0)

            # Assign to worker
            task.state = TaskState.RUNNING
            task.worker_id = worker.worker_id
            task.start_time = time.time()
            task.attempt += 1

            self.running_tasks[task.task_id] = task

            # Send to worker (async)
            asyncio.create_task(self._run_task_on_worker(task, worker))

    async def _run_task_on_worker(self, task: Task, worker: 'WorkerClient'):
        """Execute task on worker and handle result"""
        try:
            result = await asyncio.wait_for(
                worker.execute_task(task),
                timeout=self.task_timeout_seconds
            )

            task.state = TaskState.FINISHED
            task.end_time = time.time()
            task.metrics = result.metrics

            # Update stage
            stage = self.stages[task.stage_id]
            stage.num_finished_tasks += 1

            self.logger.debug(
                f"Task {task.task_id} completed in "
                f"{task.end_time - task.start_time:.2f}s"
            )

        except asyncio.TimeoutError:
            self.logger.warning(f"Task {task.task_id} timed out on worker {worker.worker_id}")
            await self._handle_task_failure(task, "timeout")

        except Exception as e:
            self.logger.warning(f"Task {task.task_id} failed: {e}")
            await self._handle_task_failure(task, str(e))

        finally:
            del self.running_tasks[task.task_id]

    async def _handle_task_failure(self, task: Task, reason: str):
        """Handle failed task with retry logic"""
        task.state = TaskState.FAILED
        task.end_time = time.time()

        if task.attempt < self.max_task_attempts:
            # Retry on different worker
            task.state = TaskState.PENDING
            task.worker_id = None
            self.pending_tasks.append(task)
            self.logger.info(f"Retrying task {task.task_id} (attempt {task.attempt + 1})")
        else:
            # Mark stage as failed
            stage = self.stages[task.stage_id]
            stage.state = StageState.FAILED
            raise RuntimeError(f"Task {task.task_id} failed after {task.attempt} attempts: {reason}")

    def _find_available_worker(self) -> Optional['WorkerClient']:
        """Find worker with available slots using locality-aware scheduling"""
        best_worker = None
        min_load = float('inf')

        for worker in self.workers.values():
            if worker.is_alive and worker.available_slots > 0:
                if worker.current_load < min_load:
                    min_load = worker.current_load
                    best_worker = worker

        return best_worker


class Worker:
    """
    Execution worker that processes tasks.
    Implements push-based batch processing with memory management.
    """

    def __init__(self, worker_id: str, memory_limit_mb: int):
        self.worker_id = worker_id
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.memory_used = 0
        self.batch_size = 10000  # rows per batch
        self.spill_threshold = 0.8  # spill when memory usage exceeds this

        # Operator implementations
        self.operators = {
            PhysicalOperator.FILTER: self._execute_filter,
            PhysicalOperator.PROJECT: self._execute_project,
            PhysicalOperator.HASH_JOIN: self._execute_hash_join,
            PhysicalOperator.HASH_AGGREGATE: self._execute_hash_aggregate,
            PhysicalOperator.SORT: self._execute_sort,
        }

    async def execute_task(self, task: Task) -> 'TaskResult':
        """Execute a task and return results"""
        metrics = {'rows_processed': 0, 'memory_peak': 0, 'spill_count': 0}

        try:
            # Build operator pipeline
            pipeline = self._build_pipeline(task.operator_chain)

            # Fetch input batches
            input_stream = self._fetch_inputs(task.input_partitions)

            # Process batches through pipeline
            output_batches = []
            async for batch in input_stream:
                metrics['rows_processed'] += len(batch)

                # Execute pipeline
                result = await self._execute_pipeline(pipeline, batch)

                if result:
                    output_batches.append(result)

                # Check memory
                if self.memory_used > self.memory_limit * self.spill_threshold:
                    await self._spill_to_disk()
                    metrics['spill_count'] += 1

                metrics['memory_peak'] = max(metrics['memory_peak'], self.memory_used)

            # Finalize stateful operators
            final_results = await self._finalize_pipeline(pipeline)
            output_batches.extend(final_results)

            # Write output
            await self._write_output(task.task_id, output_batches)

            return TaskResult(success=True, metrics=metrics)

        except Exception as e:
            return TaskResult(success=False, error=str(e), metrics=metrics)

    async def _execute_hash_join(self, left_batch: 'Batch', right_batch: 'Batch',
                                  properties: dict) -> 'Batch':
        """Execute hash join with build/probe phases"""
        join_keys = properties['join_keys']
        join_type = properties['join_type']

        # Build phase: hash the smaller side
        hash_table = {}
        for row in right_batch:
            key = tuple(row[k] for k in join_keys['right'])
            if key not in hash_table:
                hash_table[key] = []
            hash_table[key].append(row)

        # Probe phase: scan the larger side
        results = []
        for row in left_batch:
            key = tuple(row[k] for k in join_keys['left'])
            matches = hash_table.get(key, [])

            if matches:
                for match in matches:
                    results.append({**row, **match})
            elif join_type == 'LEFT':
                # Left outer join - emit with nulls
                null_right = {k: None for k in right_batch.schema}
                results.append({**row, **null_right})

        return Batch(results, left_batch.schema + right_batch.schema)

    async def _execute_hash_aggregate(self, batch: 'Batch', properties: dict) -> 'Batch':
        """Execute hash-based aggregation"""
        group_keys = properties['group_keys']
        aggregates = properties['aggregates']

        # Hash table for groups
        groups = {}

        for row in batch:
            key = tuple(row[k] for k in group_keys)

            if key not in groups:
                groups[key] = {agg['name']: agg['init']() for agg in aggregates}

            # Update aggregates
            for agg in aggregates:
                value = row[agg['input']]
                groups[key][agg['name']] = agg['update'](groups[key][agg['name']], value)

        # Finalize and emit results
        results = []
        for key, agg_state in groups.items():
            row = dict(zip(group_keys, key))
            for agg in aggregates:
                row[agg['output']] = agg['finalize'](agg_state[agg['name']])
            results.append(row)

        return Batch(results, group_keys + [a['output'] for a in aggregates])

    async def _spill_to_disk(self):
        """Spill data to disk when memory pressure is high"""
        # Implementation would serialize batches to temporary files
        self.logger.info(f"Spilling to disk, memory used: {self.memory_used / 1024 / 1024:.1f}MB")
        # ... spill logic ...
        self.memory_used = 0
```

### 4. Exchange Operators

```python
import asyncio
from typing import AsyncIterator, Dict, List
import hashlib
import struct

class ShuffleWriter:
    """
    Partitions output data and writes to shuffle files.
    Each shuffle block is written to a separate file per reducer.
    """

    def __init__(self, num_partitions: int, partition_keys: List[str]):
        self.num_partitions = num_partitions
        self.partition_keys = partition_keys
        self.buffers: List[List] = [[] for _ in range(num_partitions)]
        self.buffer_limit = 10000  # rows before flush
        self.files = []

    async def write(self, batch: 'Batch'):
        """Partition batch and buffer for output"""
        for row in batch:
            partition_id = self._partition(row)
            self.buffers[partition_id].append(row)

            if len(self.buffers[partition_id]) >= self.buffer_limit:
                await self._flush_buffer(partition_id)

    def _partition(self, row: dict) -> int:
        """Compute partition ID using hash partitioning"""
        key_values = tuple(str(row[k]) for k in self.partition_keys)
        key_bytes = '|'.join(key_values).encode()
        hash_value = int(hashlib.md5(key_bytes).hexdigest(), 16)
        return hash_value % self.num_partitions

    async def _flush_buffer(self, partition_id: int):
        """Flush buffer to shuffle file"""
        if not self.buffers[partition_id]:
            return

        # Write to partition file
        filename = f"/tmp/shuffle/{self.shuffle_id}/part-{partition_id}"
        async with aiofiles.open(filename, 'ab') as f:
            data = self._serialize_rows(self.buffers[partition_id])
            await f.write(data)

        self.buffers[partition_id] = []

    async def finalize(self) -> Dict[int, str]:
        """Flush all buffers and return file locations"""
        for partition_id in range(self.num_partitions):
            await self._flush_buffer(partition_id)

        return {
            i: f"/tmp/shuffle/{self.shuffle_id}/part-{i}"
            for i in range(self.num_partitions)
        }


class ShuffleReader:
    """
    Reads shuffle data from remote workers.
    Implements fetch with retry and speculative execution.
    """

    def __init__(self, shuffle_locations: Dict[str, Dict[int, str]]):
        self.shuffle_locations = shuffle_locations  # worker_id -> {partition -> file}
        self.fetch_timeout = 30
        self.max_retries = 3

    async def read(self, partition_id: int) -> AsyncIterator['Batch']:
        """Fetch all shuffle blocks for a partition"""
        fetch_tasks = []

        for worker_id, partitions in self.shuffle_locations.items():
            if partition_id in partitions:
                task = self._fetch_from_worker(
                    worker_id,
                    partitions[partition_id]
                )
                fetch_tasks.append(task)

        # Fetch in parallel with error handling
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                raise result
            async for batch in result:
                yield batch

    async def _fetch_from_worker(self, worker_id: str,
                                  file_path: str) -> AsyncIterator['Batch']:
        """Fetch shuffle file from remote worker"""
        for attempt in range(self.max_retries):
            try:
                async with self._connect(worker_id) as conn:
                    response = await asyncio.wait_for(
                        conn.fetch_shuffle_file(file_path),
                        timeout=self.fetch_timeout
                    )

                    async for chunk in response:
                        yield self._deserialize_batch(chunk)
                    return

            except (asyncio.TimeoutError, ConnectionError) as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff


class BroadcastExchange:
    """
    Broadcasts small table to all workers.
    Uses distributed cache for efficiency.
    """

    def __init__(self, coordinator: 'Coordinator'):
        self.coordinator = coordinator
        self.cache = {}  # broadcast_id -> data

    async def broadcast(self, data: 'Batch', broadcast_id: str):
        """Broadcast data to all workers"""
        # Serialize data
        serialized = self._serialize(data)

        # Upload to distributed storage
        location = await self._upload_to_storage(broadcast_id, serialized)

        # Notify all workers
        tasks = []
        for worker in self.coordinator.workers.values():
            task = worker.register_broadcast(broadcast_id, location)
            tasks.append(task)

        await asyncio.gather(*tasks)

        self.cache[broadcast_id] = location

    async def fetch(self, broadcast_id: str) -> 'Batch':
        """Fetch broadcast data (cached locally after first fetch)"""
        if broadcast_id in self.local_cache:
            return self.local_cache[broadcast_id]

        location = self.cache[broadcast_id]
        data = await self._download_from_storage(location)

        self.local_cache[broadcast_id] = data
        return data
```

## Adaptive Query Execution

```python
class AdaptiveExecutor:
    """
    Runtime query optimization based on actual data statistics.
    Implements Adaptive Query Execution (AQE) similar to Spark 3.0+.
    """

    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.runtime_stats = {}

    async def execute_with_adaptation(self, plan: PhysicalPlan) -> 'QueryResult':
        """Execute with runtime optimization"""
        # Execute each stage and potentially re-optimize
        stages = self._topological_sort(plan)

        for stage_id in stages:
            stage = self.coordinator.stages[stage_id]

            # Execute stage
            await self._execute_stage(stage)

            # Collect runtime statistics
            stats = await self._collect_stage_stats(stage)
            self.runtime_stats[stage_id] = stats

            # Check for optimization opportunities
            await self._adapt_downstream_stages(stage_id, stats)

        return await self.coordinator._collect_results(stages[-1])

    async def _adapt_downstream_stages(self, completed_stage: str, stats: dict):
        """Re-optimize stages based on runtime statistics"""

        # 1. Coalesce small shuffle partitions
        if stats['shuffle_partition_sizes']:
            small_partitions = [
                p for p, size in enumerate(stats['shuffle_partition_sizes'])
                if size < 64 * 1024 * 1024  # Less than 64MB
            ]
            if len(small_partitions) > len(stats['shuffle_partition_sizes']) * 0.3:
                await self._coalesce_partitions(completed_stage, small_partitions)

        # 2. Handle skewed partitions
        if stats['max_partition_size'] > stats['avg_partition_size'] * 10:
            await self._handle_skew(completed_stage, stats)

        # 3. Switch join strategy if build side too large
        downstream_joins = self._find_downstream_joins(completed_stage)
        for join_stage in downstream_joins:
            actual_size = stats.get('output_size_bytes', 0)
            if actual_size > self.coordinator.worker_memory * 0.3:
                await self._switch_to_sort_merge_join(join_stage)

    async def _handle_skew(self, stage_id: str, stats: dict):
        """Handle data skew by splitting hot partitions"""
        # Find skewed partitions
        avg_size = stats['avg_partition_size']
        skewed = [
            (p, size) for p, size in enumerate(stats['shuffle_partition_sizes'])
            if size > avg_size * 5
        ]

        for partition_id, size in skewed:
            # Split skewed partition
            num_splits = int(size / avg_size) + 1

            # Update downstream tasks to read from split partitions
            for downstream_stage in self._get_downstream_stages(stage_id):
                stage = self.coordinator.stages[downstream_stage]

                # Create additional tasks for split partitions
                for split_id in range(num_splits):
                    new_task = Task(
                        task_id=f"{downstream_stage}-{partition_id}-{split_id}",
                        stage_id=downstream_stage,
                        partition_id=partition_id,
                        operator_chain=stage.tasks[0].operator_chain,
                        input_partitions=[(stage_id, partition_id, split_id)]
                    )
                    stage.tasks.append(new_task)
                    self.coordinator.pending_tasks.append(new_task)

    async def _coalesce_partitions(self, stage_id: str, small_partitions: List[int]):
        """Combine small partitions to reduce task overhead"""
        # Group small partitions together
        groups = []
        current_group = []
        current_size = 0
        target_size = 128 * 1024 * 1024  # 128MB target

        for p in small_partitions:
            size = self.runtime_stats[stage_id]['shuffle_partition_sizes'][p]
            if current_size + size > target_size and current_group:
                groups.append(current_group)
                current_group = [p]
                current_size = size
            else:
                current_group.append(p)
                current_size += size

        if current_group:
            groups.append(current_group)

        # Update downstream tasks to read coalesced partitions
        for downstream_stage in self._get_downstream_stages(stage_id):
            stage = self.coordinator.stages[downstream_stage]

            # Replace tasks with coalesced versions
            new_tasks = []
            for i, group in enumerate(groups):
                task = Task(
                    task_id=f"{downstream_stage}-coalesced-{i}",
                    stage_id=downstream_stage,
                    partition_id=i,
                    operator_chain=stage.tasks[0].operator_chain,
                    input_partitions=[(stage_id, p) for p in group]
                )
                new_tasks.append(task)

            stage.tasks = new_tasks
```

## Enterprise Features

### Query Profile and Monitoring

```python
@dataclass
class QueryProfile:
    """Detailed query execution profile for debugging and optimization"""
    query_id: str
    sql: str
    logical_plan: str
    physical_plan: str
    stages: List['StageProfile']
    total_time_ms: int
    rows_processed: int
    bytes_processed: int
    peak_memory_bytes: int
    spill_bytes: int

@dataclass
class StageProfile:
    stage_id: str
    operators: List[str]
    tasks: List['TaskProfile']
    input_rows: int
    output_rows: int
    shuffle_read_bytes: int
    shuffle_write_bytes: int
    time_ms: int

@dataclass
class TaskProfile:
    task_id: str
    worker_id: str
    attempt: int
    time_ms: int
    input_rows: int
    output_rows: int
    memory_bytes: int
    spill_bytes: int
    gc_time_ms: int

class QueryProfiler:
    """Collects and analyzes query execution profiles"""

    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.profiles: Dict[str, QueryProfile] = {}

    async def profile_query(self, query_id: str) -> QueryProfile:
        """Collect profile after query execution"""
        stages = []
        total_rows = 0
        total_bytes = 0
        peak_memory = 0

        for stage_id, stage in self.coordinator.stages.items():
            task_profiles = []

            for task in stage.tasks:
                if task.metrics:
                    task_profile = TaskProfile(
                        task_id=task.task_id,
                        worker_id=task.worker_id,
                        attempt=task.attempt,
                        time_ms=int((task.end_time - task.start_time) * 1000),
                        input_rows=task.metrics.get('input_rows', 0),
                        output_rows=task.metrics.get('output_rows', 0),
                        memory_bytes=task.metrics.get('memory_peak', 0),
                        spill_bytes=task.metrics.get('spill_bytes', 0),
                        gc_time_ms=task.metrics.get('gc_time_ms', 0)
                    )
                    task_profiles.append(task_profile)

                    total_rows += task_profile.output_rows
                    peak_memory = max(peak_memory, task_profile.memory_bytes)

            stage_profile = StageProfile(
                stage_id=stage_id,
                operators=[op.operator.value for op in stage.tasks[0].operator_chain],
                tasks=task_profiles,
                input_rows=sum(t.input_rows for t in task_profiles),
                output_rows=sum(t.output_rows for t in task_profiles),
                shuffle_read_bytes=sum(t.metrics.get('shuffle_read', 0) for t in stage.tasks if t.metrics),
                shuffle_write_bytes=sum(t.metrics.get('shuffle_write', 0) for t in stage.tasks if t.metrics),
                time_ms=max(t.time_ms for t in task_profiles) if task_profiles else 0
            )
            stages.append(stage_profile)

        return QueryProfile(
            query_id=query_id,
            sql=self.coordinator.query_sql.get(query_id, ''),
            logical_plan=str(self.coordinator.logical_plans.get(query_id)),
            physical_plan=str(self.coordinator.physical_plans.get(query_id)),
            stages=stages,
            total_time_ms=sum(s.time_ms for s in stages),
            rows_processed=total_rows,
            bytes_processed=total_bytes,
            peak_memory_bytes=peak_memory,
            spill_bytes=sum(t.spill_bytes for s in stages for t in s.tasks)
        )

    def generate_flamegraph(self, profile: QueryProfile) -> str:
        """Generate flamegraph data for visualization"""
        lines = []

        for stage in profile.stages:
            for task in stage.tasks:
                # Stack: query > stage > task > operators
                stack = f"{profile.query_id};{stage.stage_id};{task.task_id}"
                for op in stage.operators:
                    stack += f";{op}"
                lines.append(f"{stack} {task.time_ms}")

        return '\n'.join(lines)
```

## Development Phases

### Phase 1: Query Planning Foundation (Weeks 1-3)
- SQL parser with SELECT, FROM, WHERE, GROUP BY, ORDER BY, LIMIT
- Logical plan representation
- Basic optimizations (predicate pushdown, projection pruning)
- Physical plan generation with hash join and hash aggregate

### Phase 2: Distributed Execution (Weeks 4-6)
- Coordinator and worker architecture
- Stage-based execution with task scheduling
- Shuffle operators (hash partition, sort merge)
- Basic fault tolerance with task retries

### Phase 3: Exchange and Shuffles (Weeks 7-8)
- Efficient shuffle file format
- Network transfer with compression
- Broadcast exchange for small tables
- Gather exchange for final results

### Phase 4: Performance Optimization (Weeks 9-10)
- Vectorized batch processing
- Memory-efficient operators
- Spill to disk for large aggregations
- Pipeline parallelism

### Phase 5: Enterprise Features (Weeks 11-12)
- Adaptive query execution
- Skew handling
- Query profiling and UI
- Resource management

### Phase 6: Stretch Goals (Weeks 13+)
- Vectorized operators with SIMD
- Columnar compression codecs (RLE, delta, dictionary)
- Cost-based optimizer with statistics
- Multi-tenant resource isolation

## Testing Strategy

### Unit Tests
- SQL parser correctness for all SQL constructs
- Logical plan optimization rules
- Physical operator implementations
- Shuffle serialization/deserialization

### Integration Tests
- End-to-end query execution
- Multi-stage queries with shuffles
- Join strategies (hash, broadcast, merge)
- Aggregation with spilling

### Performance Tests
- TPC-H benchmark queries
- Shuffle performance at scale
- Memory usage under pressure
- Adaptive execution effectiveness

### Chaos Tests
- Worker failures during execution
- Network partitions
- Coordinator failover
- Task timeout handling

## Performance Targets

| Metric | Target |
|--------|--------|
| Query latency (1GB, simple) | < 5 seconds |
| Throughput | > 100 MB/s per worker |
| Shuffle efficiency | > 80% network utilization |
| Memory efficiency | < 10% overhead |
| Skew handling | < 2x slowdown for 10x skew |

## Dependencies

- **ANTLR or pest**: SQL parsing
- **Apache Arrow**: Columnar batch format
- **tokio**: Async runtime
- **tonic/gRPC**: Worker communication
- **serde**: Serialization
- **rocksdb**: Spill storage

## References

- Snowflake: A New Architecture for a Cloud Data Warehouse
- Spark SQL: Relational Data Processing in Spark
- Adaptive Query Execution in Spark 3.0
- Volcano: An Extensible and Parallel Query Evaluation System
