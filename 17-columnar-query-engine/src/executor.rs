//! Vectorized query executor.

use crate::expression::Expression;
use crate::parallel::{ParallelConfig, ParallelHashAggregateOperator, ParallelScanOperator, ThreadPool};
use crate::plan::PhysicalPlan;
use crate::storage::{Catalog, TableScanner};
use crate::types::{AggregateFunction, JoinType, SortOrder, Value};
use crate::vector::{DataChunk, Vector};
use crate::{Error, Result, VECTOR_SIZE};
use std::collections::HashMap;
use std::sync::Arc;

/// Query executor.
pub struct Executor {
    /// Catalog for table lookups.
    catalog: Arc<Catalog>,
    /// Thread pool for parallel execution.
    thread_pool: Option<Arc<ThreadPool>>,
    /// Parallel execution configuration.
    parallel_config: ParallelConfig,
}

impl Executor {
    /// Create new executor.
    pub fn new(catalog: Arc<Catalog>) -> Self {
        Self {
            catalog,
            thread_pool: None,
            parallel_config: ParallelConfig::default(),
        }
    }

    /// Create executor with parallel execution enabled.
    pub fn with_parallelism(catalog: Arc<Catalog>, config: ParallelConfig) -> Self {
        let thread_pool = Arc::new(ThreadPool::new(config.num_threads));
        Self {
            catalog,
            thread_pool: Some(thread_pool),
            parallel_config: config,
        }
    }

    /// Get reference to thread pool.
    pub fn thread_pool(&self) -> Option<&Arc<ThreadPool>> {
        self.thread_pool.as_ref()
    }

    /// Execute a physical plan.
    pub fn execute(&self, plan: &PhysicalPlan) -> Result<Box<dyn Operator>> {
        match plan {
            PhysicalPlan::SeqScan {
                table_name,
                projection,
                filter,
            } => {
                let table = self
                    .catalog
                    .get_table(table_name)
                    .ok_or_else(|| Error::TableNotFound(table_name.clone()))?;
                let scanner = table.scan(projection);
                Ok(Box::new(ScanOperator::new(scanner, filter.clone())))
            }

            PhysicalPlan::ParallelScan {
                table_name,
                projection,
                filter,
                num_partitions,
            } => {
                let mut parallel_scan = ParallelScanOperator::new(
                    self.catalog.clone(),
                    table_name,
                    projection,
                    filter.clone(),
                    *num_partitions,
                )?;

                // Execute scan in parallel if thread pool available
                if let Some(pool) = &self.thread_pool {
                    parallel_scan.execute_parallel(pool)?;
                }

                Ok(Box::new(parallel_scan))
            }

            PhysicalPlan::Filter { input, predicate } => {
                let child = self.execute(input)?;
                Ok(Box::new(FilterOperator::new(child, predicate.clone())))
            }

            PhysicalPlan::Project { input, expressions } => {
                let child = self.execute(input)?;
                Ok(Box::new(ProjectOperator::new(child, expressions.clone())))
            }

            PhysicalPlan::HashAggregate {
                input,
                group_by,
                aggregates,
            } => {
                let child = self.execute(input)?;
                Ok(Box::new(HashAggregateOperator::new(
                    child,
                    group_by.clone(),
                    aggregates.clone(),
                )))
            }

            PhysicalPlan::ParallelHashAggregate {
                input,
                group_by,
                aggregates,
                num_partitions,
            } => {
                let child = self.execute(input)?;
                let mut parallel_agg = ParallelHashAggregateOperator::new(
                    child,
                    group_by.clone(),
                    aggregates.clone(),
                    *num_partitions,
                );

                // Execute aggregation in parallel if thread pool available
                if let Some(pool) = &self.thread_pool {
                    parallel_agg.execute_parallel(pool)?;
                }

                Ok(Box::new(parallel_agg))
            }

            PhysicalPlan::Sort { input, order_by } => {
                let child = self.execute(input)?;
                Ok(Box::new(SortOperator::new(child, order_by.clone())))
            }

            PhysicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                let child = self.execute(input)?;
                Ok(Box::new(LimitOperator::new(child, *limit, *offset)))
            }

            PhysicalPlan::HashJoin {
                left,
                right,
                join_type,
                left_keys,
                right_keys,
                condition,
            } => {
                let left_op = self.execute(left)?;
                let right_op = self.execute(right)?;
                Ok(Box::new(HashJoinOperator::new(
                    left_op,
                    right_op,
                    *join_type,
                    left_keys.clone(),
                    right_keys.clone(),
                    condition.clone(),
                )))
            }

            PhysicalPlan::UnionAll { inputs } => {
                let operators: Vec<Box<dyn Operator>> = inputs
                    .iter()
                    .map(|p| self.execute(p))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Box::new(UnionOperator::new(operators)))
            }

            PhysicalPlan::HashDistinct { input } => {
                let child = self.execute(input)?;
                Ok(Box::new(DistinctOperator::new(child)))
            }

            PhysicalPlan::Values { values } => {
                Ok(Box::new(ValuesOperator::new(values.clone())))
            }

            PhysicalPlan::Empty => Ok(Box::new(EmptyOperator)),

            _ => Err(Error::Execution(format!(
                "Unsupported physical plan: {:?}",
                plan
            ))),
        }
    }

    /// Execute and collect all results.
    pub fn collect(&self, plan: &PhysicalPlan) -> Result<Vec<DataChunk>> {
        let mut operator = self.execute(plan)?;
        let mut results = Vec::new();

        while let Some(chunk) = operator.next()? {
            results.push(chunk);
        }

        Ok(results)
    }
}

/// Operator trait for vectorized execution.
pub trait Operator: Send {
    /// Get next batch of data.
    fn next(&mut self) -> Result<Option<DataChunk>>;

    /// Reset operator for re-execution.
    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Table scan operator.
pub struct ScanOperator {
    scanner: TableScanner,
    filter: Option<Expression>,
}

impl ScanOperator {
    fn new(scanner: TableScanner, filter: Option<Expression>) -> Self {
        Self { scanner, filter }
    }
}

impl Operator for ScanOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        match self.scanner.next() {
            Some(Ok(chunk)) => {
                // Apply filter if present
                if let Some(ref filter) = self.filter {
                    let result = filter.evaluate(&chunk)?;
                    let selection: Vec<usize> = (0..chunk.len())
                        .filter(|&i| {
                            matches!(result.get(i), Ok(Value::Boolean(true)))
                        })
                        .collect();

                    if selection.is_empty() {
                        return self.next(); // Get next chunk
                    }

                    return Ok(Some(chunk.filter(&selection)?));
                }
                Ok(Some(chunk))
            }
            Some(Err(e)) => Err(e),
            None => Ok(None),
        }
    }
}

/// Filter operator.
pub struct FilterOperator {
    child: Box<dyn Operator>,
    predicate: Expression,
}

impl FilterOperator {
    fn new(child: Box<dyn Operator>, predicate: Expression) -> Self {
        Self { child, predicate }
    }
}

impl Operator for FilterOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        while let Some(chunk) = self.child.next()? {
            let result = self.predicate.evaluate(&chunk)?;
            let selection: Vec<usize> = (0..chunk.len())
                .filter(|&i| matches!(result.get(i), Ok(Value::Boolean(true))))
                .collect();

            if !selection.is_empty() {
                return Ok(Some(chunk.filter(&selection)?));
            }
        }
        Ok(None)
    }
}

/// Project operator.
pub struct ProjectOperator {
    child: Box<dyn Operator>,
    expressions: Vec<Expression>,
}

impl ProjectOperator {
    fn new(child: Box<dyn Operator>, expressions: Vec<Expression>) -> Self {
        Self { child, expressions }
    }
}

impl Operator for ProjectOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        if let Some(chunk) = self.child.next()? {
            let vectors: Vec<Vector> = self
                .expressions
                .iter()
                .map(|expr| expr.evaluate(&chunk))
                .collect::<Result<Vec<_>>>()?;

            Ok(Some(DataChunk::new(vectors)))
        } else {
            Ok(None)
        }
    }
}

/// Hash aggregate operator.
pub struct HashAggregateOperator {
    child: Box<dyn Operator>,
    group_by: Vec<Expression>,
    aggregates: Vec<(AggregateFunction, Expression)>,
    state: HashMap<Vec<u8>, AggregateState>,
    done: bool,
    emitted: bool,
}

/// Aggregate state for a group.
#[derive(Debug, Clone)]
struct AggregateState {
    group_values: Vec<Value>,
    accumulators: Vec<Accumulator>,
}

/// Accumulator for aggregate functions.
#[derive(Debug, Clone)]
pub enum Accumulator {
    Count(i64),
    Sum(f64),
    Avg { sum: f64, count: i64 },
    Min(Option<Value>),
    Max(Option<Value>),
    First(Option<Value>),
    Last(Option<Value>),
}

impl Accumulator {
    /// Create a new accumulator for the given aggregate function.
    pub fn new(func: AggregateFunction) -> Self {
        match func {
            AggregateFunction::Count => Accumulator::Count(0),
            AggregateFunction::Sum => Accumulator::Sum(0.0),
            AggregateFunction::Avg => Accumulator::Avg { sum: 0.0, count: 0 },
            AggregateFunction::Min => Accumulator::Min(None),
            AggregateFunction::Max => Accumulator::Max(None),
            AggregateFunction::First => Accumulator::First(None),
            AggregateFunction::Last => Accumulator::Last(None),
        }
    }

    /// Update the accumulator with a new value.
    pub fn update(&mut self, value: &Value) {
        if value.is_null() {
            return;
        }

        match self {
            Accumulator::Count(c) => *c += 1,
            Accumulator::Sum(s) => {
                if let Some(f) = value.as_f64() {
                    *s += f;
                }
            }
            Accumulator::Avg { sum, count } => {
                if let Some(f) = value.as_f64() {
                    *sum += f;
                    *count += 1;
                }
            }
            Accumulator::Min(min) => match min {
                None => *min = Some(value.clone()),
                Some(m) if value < m => *min = Some(value.clone()),
                _ => {}
            },
            Accumulator::Max(max) => match max {
                None => *max = Some(value.clone()),
                Some(m) if value > m => *max = Some(value.clone()),
                _ => {}
            },
            Accumulator::First(first) => {
                if first.is_none() {
                    *first = Some(value.clone());
                }
            }
            Accumulator::Last(last) => {
                *last = Some(value.clone());
            }
        }
    }

    /// Finalize the accumulator and return the result.
    pub fn finalize(&self) -> Value {
        match self {
            Accumulator::Count(c) => Value::Int64(*c),
            Accumulator::Sum(s) => Value::Float64(*s),
            Accumulator::Avg { sum, count } => {
                if *count == 0 {
                    Value::Null
                } else {
                    Value::Float64(*sum / *count as f64)
                }
            }
            Accumulator::Min(min) => min.clone().unwrap_or(Value::Null),
            Accumulator::Max(max) => max.clone().unwrap_or(Value::Null),
            Accumulator::First(first) => first.clone().unwrap_or(Value::Null),
            Accumulator::Last(last) => last.clone().unwrap_or(Value::Null),
        }
    }
}

impl HashAggregateOperator {
    fn new(
        child: Box<dyn Operator>,
        group_by: Vec<Expression>,
        aggregates: Vec<(AggregateFunction, Expression)>,
    ) -> Self {
        Self {
            child,
            group_by,
            aggregates,
            state: HashMap::new(),
            done: false,
            emitted: false,
        }
    }

    fn hash_group(&self, values: &[Value]) -> Vec<u8> {
        // Simple hash key from group values
        let json = serde_json::to_vec(values).unwrap_or_default();
        json
    }
}

impl Operator for HashAggregateOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        // First, consume all input
        if !self.done {
            while let Some(chunk) = self.child.next()? {
                // Evaluate group by expressions
                let group_vectors: Vec<Vector> = self
                    .group_by
                    .iter()
                    .map(|expr| expr.evaluate(&chunk))
                    .collect::<Result<Vec<_>>>()?;

                // Evaluate aggregate expressions
                let agg_vectors: Vec<Vector> = self
                    .aggregates
                    .iter()
                    .map(|(_, expr)| expr.evaluate(&chunk))
                    .collect::<Result<Vec<_>>>()?;

                // Process each row
                for i in 0..chunk.len() {
                    // Get group key
                    let group_values: Vec<Value> = group_vectors
                        .iter()
                        .map(|v| v.get(i))
                        .collect::<Result<Vec<_>>>()?;

                    let key = self.hash_group(&group_values);

                    // Get or create state
                    let state = self.state.entry(key).or_insert_with(|| AggregateState {
                        group_values: group_values.clone(),
                        accumulators: self
                            .aggregates
                            .iter()
                            .map(|(func, _)| Accumulator::new(*func))
                            .collect(),
                    });

                    // Update accumulators
                    for (j, vec) in agg_vectors.iter().enumerate() {
                        let value = vec.get(i)?;
                        state.accumulators[j].update(&value);
                    }
                }
            }
            self.done = true;
        }

        // Emit results
        if self.emitted {
            return Ok(None);
        }
        self.emitted = true;

        if self.state.is_empty() {
            // No groups - emit single row with initial values for non-grouped aggregates
            if self.group_by.is_empty() {
                let mut vectors = Vec::new();
                for (func, _) in &self.aggregates {
                    let mut vec = Vector::new(crate::types::DataType::Float64);
                    let acc = Accumulator::new(*func);
                    vec.push(acc.finalize())?;
                    vectors.push(vec);
                }
                return Ok(Some(DataChunk::new(vectors)));
            }
            return Ok(None);
        }

        // Build result chunk
        let num_groups = self.state.len();
        let num_group_cols = self.group_by.len();
        let num_agg_cols = self.aggregates.len();

        // Create vectors for group columns
        let mut group_vectors: Vec<Vector> = (0..num_group_cols)
            .map(|_| Vector::new(crate::types::DataType::String)) // Simplified
            .collect();

        // Create vectors for aggregate results
        let mut agg_vectors: Vec<Vector> = (0..num_agg_cols)
            .map(|_| Vector::new(crate::types::DataType::Float64))
            .collect();

        // Fill vectors
        for state in self.state.values() {
            // Group values
            for (i, value) in state.group_values.iter().enumerate() {
                group_vectors[i].push(value.clone())?;
            }

            // Aggregate values
            for (i, acc) in state.accumulators.iter().enumerate() {
                agg_vectors[i].push(acc.finalize())?;
            }
        }

        // Combine vectors
        let mut vectors = group_vectors;
        vectors.extend(agg_vectors);

        Ok(Some(DataChunk {
            vectors,
            len: num_groups,
        }))
    }
}

/// Sort operator.
pub struct SortOperator {
    child: Box<dyn Operator>,
    order_by: Vec<(usize, SortOrder, bool)>,
    sorted_chunks: Vec<DataChunk>,
    current_idx: usize,
    collected: bool,
}

impl SortOperator {
    fn new(child: Box<dyn Operator>, order_by: Vec<(usize, SortOrder, bool)>) -> Self {
        Self {
            child,
            order_by,
            sorted_chunks: Vec::new(),
            current_idx: 0,
            collected: false,
        }
    }
}

impl Operator for SortOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        // Collect all input first
        if !self.collected {
            let mut all_rows: Vec<(Vec<Value>, usize, usize)> = Vec::new(); // (sort_keys, chunk_idx, row_idx)
            let mut chunks = Vec::new();

            while let Some(chunk) = self.child.next()? {
                let chunk_idx = chunks.len();
                for row_idx in 0..chunk.len() {
                    let mut sort_keys = Vec::new();
                    for (col_idx, _, _) in &self.order_by {
                        if let Some(vec) = chunk.vector(*col_idx) {
                            sort_keys.push(vec.get(row_idx)?);
                        }
                    }
                    all_rows.push((sort_keys, chunk_idx, row_idx));
                }
                chunks.push(chunk);
            }

            // Sort rows
            let order_by = self.order_by.clone();
            all_rows.sort_by(|a, b| {
                for (i, (_, order, nulls_first)) in order_by.iter().enumerate() {
                    let cmp = match (&a.0[i], &b.0[i]) {
                        (Value::Null, Value::Null) => std::cmp::Ordering::Equal,
                        (Value::Null, _) => {
                            if *nulls_first {
                                std::cmp::Ordering::Less
                            } else {
                                std::cmp::Ordering::Greater
                            }
                        }
                        (_, Value::Null) => {
                            if *nulls_first {
                                std::cmp::Ordering::Greater
                            } else {
                                std::cmp::Ordering::Less
                            }
                        }
                        (va, vb) => va.partial_cmp(vb).unwrap_or(std::cmp::Ordering::Equal),
                    };

                    let cmp = match order {
                        SortOrder::Ascending => cmp,
                        SortOrder::Descending => cmp.reverse(),
                    };

                    if cmp != std::cmp::Ordering::Equal {
                        return cmp;
                    }
                }
                std::cmp::Ordering::Equal
            });

            // Rebuild chunks in sorted order
            if !chunks.is_empty() && !all_rows.is_empty() {
                let num_cols = chunks[0].num_columns();
                let mut result_vectors: Vec<Vector> = (0..num_cols)
                    .map(|i| Vector::new(chunks[0].vector(i).unwrap().data_type.clone()))
                    .collect();

                for (_, chunk_idx, row_idx) in all_rows {
                    let chunk = &chunks[chunk_idx];
                    for (col_idx, vec) in result_vectors.iter_mut().enumerate() {
                        let value = chunk.vector(col_idx).unwrap().get(row_idx)?;
                        vec.push(value)?;
                    }
                }

                let len = result_vectors.first().map(|v| v.len()).unwrap_or(0);
                self.sorted_chunks.push(DataChunk {
                    vectors: result_vectors,
                    len,
                });
            }

            self.collected = true;
        }

        // Emit sorted chunks
        if self.current_idx < self.sorted_chunks.len() {
            let chunk = self.sorted_chunks[self.current_idx].clone();
            self.current_idx += 1;
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }
}

/// Limit operator.
pub struct LimitOperator {
    child: Box<dyn Operator>,
    limit: usize,
    offset: usize,
    emitted: usize,
    skipped: usize,
}

impl LimitOperator {
    fn new(child: Box<dyn Operator>, limit: usize, offset: usize) -> Self {
        Self {
            child,
            limit,
            offset,
            emitted: 0,
            skipped: 0,
        }
    }
}

impl Operator for LimitOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        if self.emitted >= self.limit {
            return Ok(None);
        }

        while let Some(chunk) = self.child.next()? {
            // Handle offset
            if self.skipped < self.offset {
                let to_skip = (self.offset - self.skipped).min(chunk.len());
                self.skipped += to_skip;

                if to_skip < chunk.len() {
                    // Partial skip
                    let remaining = chunk.len() - to_skip;
                    let to_take = remaining.min(self.limit - self.emitted);
                    let result = chunk.slice(to_skip, to_take)?;
                    self.emitted += to_take;
                    return Ok(Some(result));
                }
                continue;
            }

            // Handle limit
            let to_take = chunk.len().min(self.limit - self.emitted);
            if to_take == 0 {
                return Ok(None);
            }

            let result = if to_take < chunk.len() {
                chunk.slice(0, to_take)?
            } else {
                chunk
            };

            self.emitted += to_take;
            return Ok(Some(result));
        }

        Ok(None)
    }
}

/// Hash join operator.
pub struct HashJoinOperator {
    left: Box<dyn Operator>,
    right: Box<dyn Operator>,
    join_type: JoinType,
    left_keys: Vec<usize>,
    right_keys: Vec<usize>,
    condition: Option<Expression>,
    hash_table: HashMap<Vec<u8>, Vec<DataChunk>>,
    built: bool,
    current_right: Option<DataChunk>,
    current_right_idx: usize,
}

impl HashJoinOperator {
    fn new(
        left: Box<dyn Operator>,
        right: Box<dyn Operator>,
        join_type: JoinType,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        condition: Option<Expression>,
    ) -> Self {
        Self {
            left,
            right,
            join_type,
            left_keys,
            right_keys,
            condition,
            hash_table: HashMap::new(),
            built: false,
            current_right: None,
            current_right_idx: 0,
        }
    }

    fn hash_keys(&self, chunk: &DataChunk, keys: &[usize], row: usize) -> Result<Vec<u8>> {
        let values: Vec<Value> = keys
            .iter()
            .map(|&k| chunk.vector(k).unwrap().get(row))
            .collect::<Result<Vec<_>>>()?;
        Ok(serde_json::to_vec(&values).unwrap_or_default())
    }
}

impl Operator for HashJoinOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        // Build phase - hash the left side
        if !self.built {
            while let Some(chunk) = self.left.next()? {
                for i in 0..chunk.len() {
                    let key = self.hash_keys(&chunk, &self.left_keys, i)?;
                    // Store single-row chunks for simplicity
                    let row_chunk = chunk.slice(i, 1)?;
                    self.hash_table.entry(key).or_default().push(row_chunk);
                }
            }
            self.built = true;
        }

        // Probe phase - scan right side
        loop {
            // Get next right chunk if needed
            if self.current_right.is_none() {
                match self.right.next()? {
                    Some(chunk) => {
                        self.current_right = Some(chunk);
                        self.current_right_idx = 0;
                    }
                    None => return Ok(None),
                }
            }

            let right_chunk = self.current_right.as_ref().unwrap();

            // Process rows from right
            while self.current_right_idx < right_chunk.len() {
                let key = self.hash_keys(right_chunk, &self.right_keys, self.current_right_idx)?;
                self.current_right_idx += 1;

                if let Some(left_chunks) = self.hash_table.get(&key) {
                    // Found matches - combine first match (simplified)
                    if let Some(left_chunk) = left_chunks.first() {
                        let right_row = right_chunk.slice(self.current_right_idx - 1, 1)?;

                        // Combine left and right
                        let mut vectors = left_chunk.vectors.clone();
                        vectors.extend(right_row.vectors);

                        let result = DataChunk::new(vectors);

                        // Apply additional condition if present
                        if let Some(ref cond) = self.condition {
                            let filter_result = cond.evaluate(&result)?;
                            if !matches!(filter_result.get(0), Ok(Value::Boolean(true))) {
                                continue;
                            }
                        }

                        return Ok(Some(result));
                    }
                } else if self.join_type == JoinType::Left || self.join_type == JoinType::Full {
                    // No match - emit right with nulls for left (for outer joins)
                    // Simplified - just skip for now
                }
            }

            // Move to next right chunk
            self.current_right = None;
        }
    }
}

/// Union operator.
pub struct UnionOperator {
    inputs: Vec<Box<dyn Operator>>,
    current_idx: usize,
}

impl UnionOperator {
    fn new(inputs: Vec<Box<dyn Operator>>) -> Self {
        Self {
            inputs,
            current_idx: 0,
        }
    }
}

impl Operator for UnionOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        while self.current_idx < self.inputs.len() {
            if let Some(chunk) = self.inputs[self.current_idx].next()? {
                return Ok(Some(chunk));
            }
            self.current_idx += 1;
        }
        Ok(None)
    }
}

/// Distinct operator.
pub struct DistinctOperator {
    child: Box<dyn Operator>,
    seen: std::collections::HashSet<Vec<u8>>,
}

impl DistinctOperator {
    fn new(child: Box<dyn Operator>) -> Self {
        Self {
            child,
            seen: std::collections::HashSet::new(),
        }
    }
}

impl Operator for DistinctOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        while let Some(chunk) = self.child.next()? {
            let mut selection = Vec::new();

            for i in 0..chunk.len() {
                // Hash row
                let values: Vec<Value> = (0..chunk.num_columns())
                    .map(|j| chunk.vector(j).unwrap().get(i))
                    .collect::<Result<Vec<_>>>()?;
                let key = serde_json::to_vec(&values).unwrap_or_default();

                if self.seen.insert(key) {
                    selection.push(i);
                }
            }

            if !selection.is_empty() {
                return Ok(Some(chunk.filter(&selection)?));
            }
        }
        Ok(None)
    }
}

/// Values operator (inline data).
pub struct ValuesOperator {
    values: Vec<Vec<Expression>>,
    emitted: bool,
}

impl ValuesOperator {
    fn new(values: Vec<Vec<Expression>>) -> Self {
        Self {
            values,
            emitted: false,
        }
    }
}

impl Operator for ValuesOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        if self.emitted || self.values.is_empty() {
            return Ok(None);
        }
        self.emitted = true;

        let num_cols = self.values[0].len();
        let mut vectors: Vec<Vector> = (0..num_cols)
            .map(|_| Vector::new(crate::types::DataType::String)) // Determine type from values
            .collect();

        // Create a dummy chunk for evaluation
        let dummy = DataChunk::new(vec![]);

        for row in &self.values {
            for (i, expr) in row.iter().enumerate() {
                if let Expression::Literal(value) = expr {
                    vectors[i].push(value.clone())?;
                } else {
                    // Evaluate expression
                    let result = expr.evaluate(&dummy)?;
                    if result.len() > 0 {
                        vectors[i].push(result.get(0)?)?;
                    }
                }
            }
        }

        let len = self.values.len();
        Ok(Some(DataChunk { vectors, len }))
    }
}

/// Empty operator.
pub struct EmptyOperator;

impl Operator for EmptyOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        Ok(None)
    }
}
