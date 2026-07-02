//! Parallel query execution.
//!
//! This module provides multi-threaded execution of query operators
//! using a thread pool and work stealing scheduler.

use crate::expression::Expression;
use crate::executor::{Operator, Accumulator};
use crate::plan::PhysicalPlan;
use crate::storage::{Catalog, TableScanner, ColumnStats};
use crate::types::{AggregateFunction, DataType, JoinType, SortOrder, Value};
use crate::vector::{DataChunk, Vector};
use crate::{Error, Result, VECTOR_SIZE};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

/// Configuration for parallel execution.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads.
    pub num_threads: usize,
    /// Minimum rows to enable parallelism.
    pub parallel_threshold: usize,
    /// Batch size for work distribution.
    pub batch_size: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get().max(1),
            parallel_threshold: 10_000,
            batch_size: VECTOR_SIZE,
        }
    }
}

impl ParallelConfig {
    /// Create config with specific thread count.
    pub fn with_threads(num_threads: usize) -> Self {
        Self {
            num_threads: num_threads.max(1),
            ..Default::default()
        }
    }
}

/// Thread pool for parallel execution.
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: crossbeam_channel::Sender<Task>,
    shutdown: Arc<AtomicBool>,
}

struct Worker {
    handle: Option<thread::JoinHandle<()>>,
}

type Task = Box<dyn FnOnce() + Send + 'static>;

impl ThreadPool {
    /// Create a new thread pool.
    pub fn new(num_threads: usize) -> Self {
        let (sender, receiver) = crossbeam_channel::unbounded::<Task>();
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut workers = Vec::with_capacity(num_threads);

        for _ in 0..num_threads {
            let receiver = receiver.clone();
            let shutdown = shutdown.clone();

            let handle = thread::spawn(move || {
                while !shutdown.load(Ordering::Relaxed) {
                    match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(task) => task(),
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                }
            });

            workers.push(Worker {
                handle: Some(handle),
            });
        }

        Self {
            workers,
            sender,
            shutdown,
        }
    }

    /// Submit a task to the thread pool.
    pub fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        self.sender.send(Box::new(f)).expect("Thread pool channel closed");
    }

    /// Get number of worker threads.
    pub fn num_threads(&self) -> usize {
        self.workers.len()
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        for worker in &mut self.workers {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

/// Parallel table scan operator.
///
/// Partitions the table into chunks and scans them in parallel.
pub struct ParallelScanOperator {
    scanners: Vec<TableScanner>,
    filter: Option<Expression>,
    results: Arc<Mutex<Vec<DataChunk>>>,
    current_idx: usize,
    done: bool,
}

impl ParallelScanOperator {
    /// Create a new parallel scan operator.
    pub fn new(
        catalog: Arc<Catalog>,
        table_name: &str,
        projection: &[usize],
        filter: Option<Expression>,
        num_partitions: usize,
    ) -> Result<Self> {
        let table = catalog
            .get_table(table_name)
            .ok_or_else(|| Error::TableNotFound(table_name.to_string()))?;

        // Create one scanner per partition, each covering a disjoint subset
        // of the table's row groups so rows are scanned exactly once.
        let num_partitions = num_partitions.max(1);
        let scanners: Vec<TableScanner> = (0..num_partitions)
            .map(|i| table.scan_partition(projection, i, num_partitions))
            .collect();

        Ok(Self {
            scanners,
            filter,
            results: Arc::new(Mutex::new(Vec::new())),
            current_idx: 0,
            done: false,
        })
    }

    /// Execute scan in parallel using thread pool.
    pub fn execute_parallel(&mut self, pool: &ThreadPool) -> Result<()> {
        if self.done {
            return Ok(());
        }

        let results = self.results.clone();
        let filter = self.filter.clone();
        let num_scanners = self.scanners.len();
        let (done_tx, done_rx) = crossbeam_channel::bounded::<()>(num_scanners);

        // Process each scanner in parallel; each scanner owns a disjoint
        // partition of the table's row groups.
        for scanner in self.scanners.drain(..) {
            let results = results.clone();
            let filter = filter.clone();
            let done_tx = done_tx.clone();

            pool.execute(move || {
                let mut scanner = scanner;
                while let Some(Ok(chunk)) = scanner.next() {
                    let filtered = if let Some(ref f) = filter {
                        // Apply filter
                        if let Ok(result) = f.evaluate(&chunk) {
                            let selection: Vec<usize> = (0..chunk.len())
                                .filter(|&i| matches!(result.get(i), Ok(Value::Boolean(true))))
                                .collect();
                            if selection.is_empty() {
                                continue;
                            }
                            chunk.filter(&selection).ok()
                        } else {
                            Some(chunk)
                        }
                    } else {
                        Some(chunk)
                    };

                    if let Some(c) = filtered {
                        results.lock().unwrap().push(c);
                    }
                }
                let _ = done_tx.send(());
            });
        }
        drop(done_tx);

        // Wait for all scanners to complete. If a worker panics, its sender
        // is dropped and recv returns an error instead of hanging forever.
        for _ in 0..num_scanners {
            done_rx.recv().map_err(|_| {
                Error::Execution("parallel scan worker terminated unexpectedly".to_string())
            })?;
        }

        self.done = true;
        Ok(())
    }
}

impl Operator for ParallelScanOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        let results = self.results.lock().unwrap();
        if self.current_idx < results.len() {
            let chunk = results[self.current_idx].clone();
            self.current_idx += 1;
            Ok(Some(chunk))
        } else {
            Ok(None)
        }
    }
}

/// Parallel hash aggregate operator.
///
/// Performs aggregation in parallel using a partitioned hash table.
pub struct ParallelHashAggregateOperator {
    child: Box<dyn Operator>,
    group_by: Vec<Expression>,
    aggregates: Vec<(AggregateFunction, Expression)>,
    partial_states: Vec<Arc<RwLock<HashMap<Vec<u8>, PartialAggregateState>>>>,
    final_results: Option<DataChunk>,
    emitted: bool,
    num_partitions: usize,
}

/// Partial aggregate state for parallel aggregation.
#[derive(Debug, Clone)]
struct PartialAggregateState {
    group_values: Vec<Value>,
    accumulators: Vec<Accumulator>,
}

impl ParallelHashAggregateOperator {
    /// Create a new parallel hash aggregate operator.
    pub fn new(
        child: Box<dyn Operator>,
        group_by: Vec<Expression>,
        aggregates: Vec<(AggregateFunction, Expression)>,
        num_partitions: usize,
    ) -> Self {
        let partial_states: Vec<_> = (0..num_partitions)
            .map(|_| Arc::new(RwLock::new(HashMap::new())))
            .collect();

        Self {
            child,
            group_by,
            aggregates,
            partial_states,
            final_results: None,
            emitted: false,
            num_partitions,
        }
    }

    /// Hash key to determine partition.
    fn partition_key(&self, key: &[u8]) -> usize {
        // Simple hash-based partitioning
        let hash: usize = key.iter().fold(0, |acc, &b| acc.wrapping_mul(31).wrapping_add(b as usize));
        hash % self.num_partitions
    }

    /// Execute aggregation with parallel partial aggregates.
    pub fn execute_parallel(&mut self, pool: &ThreadPool) -> Result<()> {
        // Phase 1: Consume input and build partial aggregates
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

                let key = serde_json::to_vec(&group_values).unwrap_or_default();
                let partition = self.partition_key(&key);

                // Get or create state in partition
                let mut states = self.partial_states[partition].write().unwrap();
                let state = states.entry(key).or_insert_with(|| PartialAggregateState {
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

        // Phase 2: Merge partial results
        let mut final_states: HashMap<Vec<u8>, PartialAggregateState> = HashMap::new();

        for partial in &self.partial_states {
            let states = partial.read().unwrap();
            for (key, state) in states.iter() {
                final_states
                    .entry(key.clone())
                    .and_modify(|existing| {
                        for (i, acc) in state.accumulators.iter().enumerate() {
                            existing.accumulators[i].merge(acc);
                        }
                    })
                    .or_insert_with(|| state.clone());
            }
        }

        // Phase 3: Build final result chunk
        if final_states.is_empty() && !self.group_by.is_empty() {
            return Ok(());
        }

        let num_groups = final_states.len().max(1);
        let num_group_cols = self.group_by.len();
        let num_agg_cols = self.aggregates.len();

        // Create vectors for group columns
        let mut group_vectors: Vec<Vector> = (0..num_group_cols)
            .map(|_| Vector::new(DataType::String))
            .collect();

        // Create vectors for aggregate results
        let mut agg_vectors: Vec<Vector> = (0..num_agg_cols)
            .map(|_| Vector::new(DataType::Float64))
            .collect();

        if final_states.is_empty() {
            // No groups - emit single row with initial values for non-grouped aggregates
            for (func, _) in &self.aggregates {
                let acc = Accumulator::new(*func);
                agg_vectors[0].push(acc.finalize())?;
            }
        } else {
            // Fill vectors
            for state in final_states.values() {
                for (i, value) in state.group_values.iter().enumerate() {
                    group_vectors[i].push(value.clone())?;
                }
                for (i, acc) in state.accumulators.iter().enumerate() {
                    agg_vectors[i].push(acc.finalize())?;
                }
            }
        }

        // Combine vectors
        let mut vectors = group_vectors;
        vectors.extend(agg_vectors);

        self.final_results = Some(DataChunk {
            vectors,
            len: num_groups,
        });

        Ok(())
    }
}

impl Operator for ParallelHashAggregateOperator {
    fn next(&mut self) -> Result<Option<DataChunk>> {
        if self.emitted {
            return Ok(None);
        }
        self.emitted = true;
        Ok(self.final_results.take())
    }
}

/// Trait extension for merging accumulators.
trait AccumulatorMerge {
    fn merge(&mut self, other: &Self);
}

impl AccumulatorMerge for Accumulator {
    fn merge(&mut self, other: &Self) {
        match (self, other) {
            (Accumulator::Count(c1), Accumulator::Count(c2)) => *c1 += c2,
            (Accumulator::Sum(s1), Accumulator::Sum(s2)) => *s1 += s2,
            (Accumulator::Avg { sum: s1, count: c1 }, Accumulator::Avg { sum: s2, count: c2 }) => {
                *s1 += s2;
                *c1 += c2;
            }
            (Accumulator::Min(m1), Accumulator::Min(m2)) => {
                if let (Some(v1), Some(v2)) = (m1.as_ref(), m2.as_ref()) {
                    if v2 < v1 {
                        *m1 = Some(v2.clone());
                    }
                } else if m1.is_none() && m2.is_some() {
                    *m1 = m2.clone();
                }
            }
            (Accumulator::Max(m1), Accumulator::Max(m2)) => {
                if let (Some(v1), Some(v2)) = (m1.as_ref(), m2.as_ref()) {
                    if v2 > v1 {
                        *m1 = Some(v2.clone());
                    }
                } else if m1.is_none() && m2.is_some() {
                    *m1 = m2.clone();
                }
            }
            _ => {}
        }
    }
}

/// Cost-based optimizer with statistics.
pub struct CostBasedOptimizer {
    /// Table statistics.
    table_stats: HashMap<String, TableStats>,
    /// Cost model configuration.
    config: CostConfig,
}

/// Table statistics for cost estimation.
#[derive(Debug, Clone)]
pub struct TableStats {
    /// Number of rows.
    pub row_count: usize,
    /// Number of distinct values per column.
    pub column_ndv: HashMap<String, usize>,
    /// Column statistics (min, max, null count).
    pub column_stats: HashMap<String, ColumnStats>,
    /// Average row size in bytes.
    pub avg_row_size: usize,
}

impl TableStats {
    /// Create empty stats.
    pub fn empty() -> Self {
        Self {
            row_count: 0,
            column_ndv: HashMap::new(),
            column_stats: HashMap::new(),
            avg_row_size: 0,
        }
    }

    /// Estimate selectivity of a predicate.
    pub fn estimate_selectivity(&self, _predicate: &Expression) -> f64 {
        // Simplified - return default selectivity
        // Real implementation would analyze predicate and use statistics
        0.1
    }
}

/// Cost model configuration.
#[derive(Debug, Clone)]
pub struct CostConfig {
    /// Cost per sequential I/O page.
    pub seq_page_cost: f64,
    /// Cost per random I/O page.
    pub random_page_cost: f64,
    /// Cost per CPU tuple operation.
    pub cpu_tuple_cost: f64,
    /// Cost per hash comparison.
    pub hash_cost: f64,
    /// Cost per sort comparison.
    pub sort_cost: f64,
    /// Page size in bytes.
    pub page_size: usize,
}

impl Default for CostConfig {
    fn default() -> Self {
        Self {
            seq_page_cost: 1.0,
            random_page_cost: 4.0,
            cpu_tuple_cost: 0.01,
            hash_cost: 0.02,
            sort_cost: 0.05,
            page_size: 8192,
        }
    }
}

impl CostBasedOptimizer {
    /// Create a new cost-based optimizer.
    pub fn new() -> Self {
        Self {
            table_stats: HashMap::new(),
            config: CostConfig::default(),
        }
    }

    /// Create optimizer with custom config.
    pub fn with_config(config: CostConfig) -> Self {
        Self {
            table_stats: HashMap::new(),
            config,
        }
    }

    /// Add table statistics.
    pub fn add_table_stats(&mut self, table_name: String, stats: TableStats) {
        self.table_stats.insert(table_name, stats);
    }

    /// Get table statistics.
    pub fn get_table_stats(&self, table_name: &str) -> Option<&TableStats> {
        self.table_stats.get(table_name)
    }

    /// Estimate cost of a physical plan.
    pub fn estimate_cost(&self, plan: &PhysicalPlan) -> PlanCost {
        match plan {
            PhysicalPlan::SeqScan { table_name, filter, .. } => {
                let stats = self.table_stats.get(table_name);
                let row_count = stats.map(|s| s.row_count).unwrap_or(1000) as f64;
                let selectivity = filter
                    .as_ref()
                    .map(|f| stats.map(|s| s.estimate_selectivity(f)).unwrap_or(0.1))
                    .unwrap_or(1.0);

                let pages = (row_count * stats.map(|s| s.avg_row_size).unwrap_or(100) as f64
                    / self.config.page_size as f64)
                    .ceil();

                PlanCost {
                    startup_cost: 0.0,
                    total_cost: pages * self.config.seq_page_cost
                        + row_count * self.config.cpu_tuple_cost,
                    output_rows: (row_count * selectivity).max(1.0),
                }
            }

            PhysicalPlan::IndexScan { table_name, filter, .. } => {
                let stats = self.table_stats.get(table_name);
                let row_count = stats.map(|s| s.row_count).unwrap_or(1000) as f64;
                let selectivity = filter
                    .as_ref()
                    .map(|f| stats.map(|s| s.estimate_selectivity(f)).unwrap_or(0.01))
                    .unwrap_or(0.01);

                let expected_rows = (row_count * selectivity).max(1.0);
                let pages = expected_rows; // Assume one page per row for index scan

                PlanCost {
                    startup_cost: 0.0,
                    total_cost: pages * self.config.random_page_cost
                        + expected_rows * self.config.cpu_tuple_cost,
                    output_rows: expected_rows,
                }
            }

            PhysicalPlan::Filter { input, predicate } => {
                let child_cost = self.estimate_cost(input);
                let selectivity = 0.1; // Default filter selectivity

                PlanCost {
                    startup_cost: child_cost.startup_cost,
                    total_cost: child_cost.total_cost
                        + child_cost.output_rows * self.config.cpu_tuple_cost,
                    output_rows: (child_cost.output_rows * selectivity).max(1.0),
                }
            }

            PhysicalPlan::Project { input, .. } => {
                let child_cost = self.estimate_cost(input);
                PlanCost {
                    startup_cost: child_cost.startup_cost,
                    total_cost: child_cost.total_cost
                        + child_cost.output_rows * self.config.cpu_tuple_cost,
                    output_rows: child_cost.output_rows,
                }
            }

            PhysicalPlan::HashAggregate { input, group_by, .. } => {
                let child_cost = self.estimate_cost(input);
                let num_groups = if group_by.is_empty() {
                    1.0
                } else {
                    (child_cost.output_rows / 10.0).max(1.0)
                };

                PlanCost {
                    startup_cost: child_cost.total_cost
                        + child_cost.output_rows * self.config.hash_cost,
                    total_cost: child_cost.total_cost
                        + child_cost.output_rows * self.config.hash_cost
                        + num_groups * self.config.cpu_tuple_cost,
                    output_rows: num_groups,
                }
            }

            PhysicalPlan::Sort { input, .. } => {
                let child_cost = self.estimate_cost(input);
                let n = child_cost.output_rows;
                let sort_cost = if n > 1.0 {
                    n * n.log2() * self.config.sort_cost
                } else {
                    0.0
                };

                PlanCost {
                    startup_cost: child_cost.total_cost + sort_cost,
                    total_cost: child_cost.total_cost + sort_cost + n * self.config.cpu_tuple_cost,
                    output_rows: n,
                }
            }

            PhysicalPlan::HashJoin { left, right, .. } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);

                // Build cost
                let build_cost = left_cost.total_cost + left_cost.output_rows * self.config.hash_cost;

                // Probe cost
                let probe_cost =
                    right_cost.total_cost + right_cost.output_rows * self.config.hash_cost;

                // Estimated output (simplified)
                let output_rows = (left_cost.output_rows * right_cost.output_rows * 0.1).max(1.0);

                PlanCost {
                    startup_cost: build_cost,
                    total_cost: build_cost + probe_cost + output_rows * self.config.cpu_tuple_cost,
                    output_rows,
                }
            }

            PhysicalPlan::NestedLoopJoin { left, right, .. } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);

                let total_cost = left_cost.total_cost
                    + left_cost.output_rows * right_cost.total_cost
                    + left_cost.output_rows * right_cost.output_rows * self.config.cpu_tuple_cost;

                PlanCost {
                    startup_cost: left_cost.startup_cost,
                    total_cost,
                    output_rows: (left_cost.output_rows * right_cost.output_rows * 0.1).max(1.0),
                }
            }

            PhysicalPlan::MergeJoin { left, right, .. } => {
                let left_cost = self.estimate_cost(left);
                let right_cost = self.estimate_cost(right);

                // Merge joins require sorted input
                let sort_cost_left = if matches!(left.as_ref(), PhysicalPlan::Sort { .. }) {
                    0.0
                } else {
                    left_cost.output_rows * left_cost.output_rows.log2() * self.config.sort_cost
                };
                let sort_cost_right = if matches!(right.as_ref(), PhysicalPlan::Sort { .. }) {
                    0.0
                } else {
                    right_cost.output_rows * right_cost.output_rows.log2() * self.config.sort_cost
                };

                let merge_cost =
                    (left_cost.output_rows + right_cost.output_rows) * self.config.cpu_tuple_cost;

                PlanCost {
                    startup_cost: left_cost.total_cost + right_cost.total_cost + sort_cost_left
                        + sort_cost_right,
                    total_cost: left_cost.total_cost + right_cost.total_cost + sort_cost_left
                        + sort_cost_right + merge_cost,
                    output_rows: (left_cost.output_rows * right_cost.output_rows * 0.1).max(1.0),
                }
            }

            PhysicalPlan::Limit { input, limit, .. } => {
                let child_cost = self.estimate_cost(input);
                let output_rows = (*limit as f64).min(child_cost.output_rows);

                PlanCost {
                    startup_cost: child_cost.startup_cost,
                    total_cost: child_cost.startup_cost
                        + output_rows / child_cost.output_rows * child_cost.total_cost,
                    output_rows,
                }
            }

            PhysicalPlan::UnionAll { inputs } => {
                let costs: Vec<_> = inputs.iter().map(|i| self.estimate_cost(i)).collect();
                let total_cost: f64 = costs.iter().map(|c| c.total_cost).sum();
                let output_rows: f64 = costs.iter().map(|c| c.output_rows).sum();

                PlanCost {
                    startup_cost: costs.first().map(|c| c.startup_cost).unwrap_or(0.0),
                    total_cost,
                    output_rows,
                }
            }

            PhysicalPlan::Values { values } => PlanCost {
                startup_cost: 0.0,
                total_cost: values.len() as f64 * self.config.cpu_tuple_cost,
                output_rows: values.len() as f64,
            },

            PhysicalPlan::Empty => PlanCost {
                startup_cost: 0.0,
                total_cost: 0.0,
                output_rows: 0.0,
            },

            _ => PlanCost {
                startup_cost: 100.0,
                total_cost: 1000.0,
                output_rows: 1000.0,
            },
        }
    }

    /// Choose best join algorithm based on cost.
    pub fn choose_join_algorithm(
        &self,
        left: &PhysicalPlan,
        right: &PhysicalPlan,
        join_type: JoinType,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        condition: Option<Expression>,
    ) -> PhysicalPlan {
        let hash_join = PhysicalPlan::HashJoin {
            left: Arc::new(left.clone()),
            right: Arc::new(right.clone()),
            join_type,
            left_keys: left_keys.clone(),
            right_keys: right_keys.clone(),
            condition: condition.clone(),
        };

        let merge_join = PhysicalPlan::MergeJoin {
            left: Arc::new(left.clone()),
            right: Arc::new(right.clone()),
            join_type,
            left_keys: left_keys.clone(),
            right_keys: right_keys.clone(),
        };

        let nested_loop = PhysicalPlan::NestedLoopJoin {
            left: Arc::new(left.clone()),
            right: Arc::new(right.clone()),
            join_type,
            condition,
        };

        // Compare costs
        let hash_cost = self.estimate_cost(&hash_join);
        let merge_cost = self.estimate_cost(&merge_join);
        let nested_cost = self.estimate_cost(&nested_loop);

        let min_cost = hash_cost.total_cost.min(merge_cost.total_cost).min(nested_cost.total_cost);

        if (hash_cost.total_cost - min_cost).abs() < f64::EPSILON {
            hash_join
        } else if (merge_cost.total_cost - min_cost).abs() < f64::EPSILON {
            merge_join
        } else {
            nested_loop
        }
    }

    /// Reorder joins for optimal execution.
    pub fn reorder_joins(&self, tables: Vec<&str>, join_predicates: &[(usize, usize, Expression)]) -> Vec<(usize, usize)> {
        // Simple greedy algorithm: always join smallest tables first
        if tables.len() <= 2 {
            return if tables.len() == 2 {
                vec![(0, 1)]
            } else {
                vec![]
            };
        }

        let mut order = Vec::new();
        let mut joined: Vec<bool> = vec![false; tables.len()];

        // Get table sizes
        let sizes: Vec<usize> = tables
            .iter()
            .map(|t| {
                self.table_stats
                    .get(*t)
                    .map(|s| s.row_count)
                    .unwrap_or(1000)
            })
            .collect();

        // Find two smallest tables to start
        let mut sorted_indices: Vec<usize> = (0..tables.len()).collect();
        sorted_indices.sort_by_key(|&i| sizes[i]);

        let first = sorted_indices[0];
        let second = sorted_indices[1];
        order.push((first.min(second), first.max(second)));
        joined[first] = true;
        joined[second] = true;

        // Greedily add remaining tables
        for &i in &sorted_indices[2..] {
            if !joined[i] {
                // Find best table to join with
                let best = (0..tables.len())
                    .filter(|&j| joined[j])
                    .min_by_key(|&j| sizes[j])
                    .unwrap_or(0);
                order.push((best.min(i), best.max(i)));
                joined[i] = true;
            }
        }

        order
    }
}

/// Cost estimate for a plan.
#[derive(Debug, Clone, Copy)]
pub struct PlanCost {
    /// Cost before first row is produced.
    pub startup_cost: f64,
    /// Total cost to produce all rows.
    pub total_cost: f64,
    /// Estimated number of output rows.
    pub output_rows: f64,
}

impl PlanCost {
    /// Compare costs.
    pub fn is_better_than(&self, other: &PlanCost) -> bool {
        self.total_cost < other.total_cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.num_threads >= 1);
        assert!(config.parallel_threshold > 0);
    }

    #[test]
    fn test_parallel_config_with_threads() {
        let config = ParallelConfig::with_threads(4);
        assert_eq!(config.num_threads, 4);
    }

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new(2);
        assert_eq!(pool.num_threads(), 2);
    }

    #[test]
    fn test_cost_based_optimizer_creation() {
        let optimizer = CostBasedOptimizer::new();
        assert!(optimizer.table_stats.is_empty());
    }

    #[test]
    fn test_table_stats() {
        let mut optimizer = CostBasedOptimizer::new();
        let stats = TableStats {
            row_count: 1000,
            column_ndv: HashMap::new(),
            column_stats: HashMap::new(),
            avg_row_size: 100,
        };
        optimizer.add_table_stats("test".to_string(), stats);

        assert!(optimizer.get_table_stats("test").is_some());
        assert_eq!(optimizer.get_table_stats("test").unwrap().row_count, 1000);
    }

    #[test]
    fn test_cost_estimation_seq_scan() {
        let mut optimizer = CostBasedOptimizer::new();
        optimizer.add_table_stats(
            "test".to_string(),
            TableStats {
                row_count: 1000,
                column_ndv: HashMap::new(),
                column_stats: HashMap::new(),
                avg_row_size: 100,
            },
        );

        let plan = PhysicalPlan::SeqScan {
            table_name: "test".to_string(),
            projection: vec![0, 1],
            filter: None,
        };

        let cost = optimizer.estimate_cost(&plan);
        assert!(cost.total_cost > 0.0);
        assert_eq!(cost.output_rows, 1000.0);
    }

    #[test]
    fn test_cost_comparison() {
        let cost1 = PlanCost {
            startup_cost: 0.0,
            total_cost: 100.0,
            output_rows: 1000.0,
        };
        let cost2 = PlanCost {
            startup_cost: 0.0,
            total_cost: 200.0,
            output_rows: 1000.0,
        };

        assert!(cost1.is_better_than(&cost2));
        assert!(!cost2.is_better_than(&cost1));
    }

    #[test]
    fn test_join_reordering() {
        let mut optimizer = CostBasedOptimizer::new();

        // Add stats for three tables with different sizes
        optimizer.add_table_stats(
            "large".to_string(),
            TableStats {
                row_count: 10000,
                ..TableStats::empty()
            },
        );
        optimizer.add_table_stats(
            "medium".to_string(),
            TableStats {
                row_count: 1000,
                ..TableStats::empty()
            },
        );
        optimizer.add_table_stats(
            "small".to_string(),
            TableStats {
                row_count: 100,
                ..TableStats::empty()
            },
        );

        let tables = vec!["large", "medium", "small"];
        let order = optimizer.reorder_joins(tables, &[]);

        // Should join smallest tables first
        assert_eq!(order.len(), 2);
    }
}
