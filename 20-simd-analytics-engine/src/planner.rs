//! Cost model and query planner for analytics operations.
//!
//! This module provides cost estimation and query optimization.

use crate::{Error, Result, BLOCK_SIZE, CACHE_LINE_SIZE, VECTOR_WIDTH};
use std::time::Duration;

/// Hardware parameters for cost estimation.
#[derive(Debug, Clone)]
pub struct HardwareParams {
    /// CPU frequency in GHz.
    pub cpu_freq_ghz: f64,
    /// Number of cores.
    pub num_cores: usize,
    /// SIMD width (elements per operation).
    pub simd_width: usize,
    /// L1 cache size in bytes.
    pub l1_cache_size: usize,
    /// L2 cache size in bytes.
    pub l2_cache_size: usize,
    /// L3 cache size in bytes.
    pub l3_cache_size: usize,
    /// L1 latency in cycles.
    pub l1_latency_cycles: f64,
    /// L2 latency in cycles.
    pub l2_latency_cycles: f64,
    /// L3 latency in cycles.
    pub l3_latency_cycles: f64,
    /// Memory latency in cycles.
    pub mem_latency_cycles: f64,
    /// Memory bandwidth in GB/s.
    pub mem_bandwidth_gb_s: f64,
}

impl Default for HardwareParams {
    fn default() -> Self {
        Self {
            cpu_freq_ghz: 3.0,
            num_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            simd_width: VECTOR_WIDTH,
            l1_cache_size: 32 * 1024,       // 32KB
            l2_cache_size: 256 * 1024,      // 256KB
            l3_cache_size: 8 * 1024 * 1024, // 8MB
            l1_latency_cycles: 4.0,
            l2_latency_cycles: 12.0,
            l3_latency_cycles: 40.0,
            mem_latency_cycles: 100.0,
            mem_bandwidth_gb_s: 50.0,
        }
    }
}

impl HardwareParams {
    /// Create with detected/specified values.
    pub fn new(
        cpu_freq_ghz: f64,
        num_cores: usize,
        mem_bandwidth_gb_s: f64,
    ) -> Self {
        Self {
            cpu_freq_ghz,
            num_cores,
            mem_bandwidth_gb_s,
            ..Default::default()
        }
    }

    /// Cycles per second.
    pub fn cycles_per_second(&self) -> f64 {
        self.cpu_freq_ghz * 1e9
    }

    /// Cycles to seconds.
    pub fn cycles_to_seconds(&self, cycles: f64) -> f64 {
        cycles / self.cycles_per_second()
    }
}

/// Cost estimate for an operation.
#[derive(Debug, Clone)]
pub struct CostEstimate {
    /// Estimated CPU cycles.
    pub cpu_cycles: f64,
    /// Estimated memory accesses.
    pub memory_accesses: usize,
    /// Estimated cache misses.
    pub cache_misses: usize,
    /// Estimated time in seconds.
    pub time_seconds: f64,
    /// Whether operation is memory-bound.
    pub memory_bound: bool,
}

impl CostEstimate {
    /// Create from components.
    pub fn new(
        cpu_cycles: f64,
        memory_accesses: usize,
        cache_misses: usize,
        time_seconds: f64,
        memory_bound: bool,
    ) -> Self {
        Self {
            cpu_cycles,
            memory_accesses,
            cache_misses,
            time_seconds,
            memory_bound,
        }
    }

    /// Get estimated duration.
    pub fn duration(&self) -> Duration {
        Duration::from_secs_f64(self.time_seconds)
    }
}

/// Cost model for analytics operations.
#[derive(Debug)]
pub struct CostModel {
    /// Hardware parameters.
    params: HardwareParams,
}

impl CostModel {
    /// Create with default parameters.
    pub fn new() -> Self {
        Self {
            params: HardwareParams::default(),
        }
    }

    /// Create with specific parameters.
    pub fn with_params(params: HardwareParams) -> Self {
        Self { params }
    }

    /// Estimate cost of sequential scan.
    pub fn estimate_scan(&self, num_rows: usize, row_size_bytes: usize) -> CostEstimate {
        let total_bytes = num_rows * row_size_bytes;
        let cache_line_accesses = (total_bytes + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;

        // Estimate cache misses based on data size
        let cache_misses = if total_bytes <= self.params.l1_cache_size {
            cache_line_accesses // Compulsory misses only
        } else if total_bytes <= self.params.l3_cache_size {
            cache_line_accesses // Most will miss L1/L2
        } else {
            cache_line_accesses // All lines will be fetched from memory
        };

        // CPU cycles for scan (1 cycle per element with SIMD)
        let cpu_cycles = num_rows as f64 / self.params.simd_width as f64;

        // Memory access time
        let mem_time = (total_bytes as f64 / 1e9) / self.params.mem_bandwidth_gb_s;

        // CPU time
        let cpu_time = self.params.cycles_to_seconds(cpu_cycles);

        // Total time (memory-bound typically)
        let time = mem_time.max(cpu_time);

        CostEstimate::new(cpu_cycles, cache_line_accesses, cache_misses, time, mem_time > cpu_time)
    }

    /// Estimate cost of filter operation.
    pub fn estimate_filter(&self, num_rows: usize, selectivity: f64) -> CostEstimate {
        // Filter needs to scan all rows but only outputs selectivity fraction
        let scan_cost = self.estimate_scan(num_rows, std::mem::size_of::<f64>());

        // Additional cost for predicate evaluation
        let predicate_cycles = num_rows as f64 / self.params.simd_width as f64;

        // Output rows
        let output_rows = (num_rows as f64 * selectivity) as usize;
        let output_bytes = output_rows * std::mem::size_of::<f64>();

        let total_cycles = scan_cost.cpu_cycles + predicate_cycles;
        let time = scan_cost.time_seconds * 1.1; // 10% overhead for predicate

        CostEstimate::new(
            total_cycles,
            scan_cost.memory_accesses,
            scan_cost.cache_misses,
            time,
            scan_cost.memory_bound,
        )
    }

    /// Estimate cost of aggregation.
    pub fn estimate_aggregate(&self, num_rows: usize) -> CostEstimate {
        let scan_cost = self.estimate_scan(num_rows, std::mem::size_of::<f64>());

        // Aggregation adds one operation per row
        let agg_cycles = num_rows as f64 / self.params.simd_width as f64;

        let total_cycles = scan_cost.cpu_cycles + agg_cycles;

        CostEstimate::new(
            total_cycles,
            scan_cost.memory_accesses,
            scan_cost.cache_misses,
            scan_cost.time_seconds * 1.05, // Slight overhead
            scan_cost.memory_bound,
        )
    }

    /// Estimate cost of hash aggregation (GROUP BY).
    pub fn estimate_hash_aggregate(&self, num_rows: usize, num_groups: usize) -> CostEstimate {
        let scan_cost = self.estimate_scan(num_rows, std::mem::size_of::<f64>());

        // Hash cost per row
        let hash_cycles = num_rows as f64 * 10.0; // ~10 cycles per hash

        // Hash table lookups (random access)
        let lookup_cycles = num_rows as f64 * self.params.l2_latency_cycles;

        // If groups fit in cache, faster
        let group_size = num_groups * 64; // ~64 bytes per group entry
        let additional_cache_misses = if group_size <= self.params.l2_cache_size {
            0
        } else {
            num_groups
        };

        let total_cycles = scan_cost.cpu_cycles + hash_cycles + lookup_cycles;
        let time = self.params.cycles_to_seconds(total_cycles);

        CostEstimate::new(
            total_cycles,
            scan_cost.memory_accesses + num_groups,
            scan_cost.cache_misses + additional_cache_misses,
            time.max(scan_cost.time_seconds),
            false, // Usually compute-bound
        )
    }

    /// Estimate cost of hash join.
    pub fn estimate_hash_join(
        &self,
        build_rows: usize,
        probe_rows: usize,
        output_rows: usize,
    ) -> CostEstimate {
        // Build phase
        let build_cost = self.estimate_hash_aggregate(build_rows, build_rows);

        // Probe phase
        let probe_scan = self.estimate_scan(probe_rows, std::mem::size_of::<f64>());
        let probe_hash_cycles = probe_rows as f64 * 10.0;
        let probe_lookup_cycles = probe_rows as f64 * self.params.l2_latency_cycles;

        // Output
        let output_cost = self.estimate_scan(output_rows, std::mem::size_of::<f64>() * 2);

        let total_cycles = build_cost.cpu_cycles + probe_scan.cpu_cycles
            + probe_hash_cycles + probe_lookup_cycles + output_cost.cpu_cycles;

        let time = self.params.cycles_to_seconds(total_cycles);

        CostEstimate::new(
            total_cycles,
            build_cost.memory_accesses + probe_scan.memory_accesses + output_cost.memory_accesses,
            build_cost.cache_misses + probe_scan.cache_misses,
            time,
            false,
        )
    }

    /// Estimate cost of sort.
    pub fn estimate_sort(&self, num_rows: usize) -> CostEstimate {
        let scan_cost = self.estimate_scan(num_rows, std::mem::size_of::<f64>());

        // O(n log n) comparisons
        let comparisons = num_rows as f64 * (num_rows as f64).log2();
        let sort_cycles = comparisons * 5.0; // ~5 cycles per comparison

        // Random access pattern increases cache misses
        let cache_misses = if num_rows * std::mem::size_of::<f64>() <= self.params.l3_cache_size {
            scan_cost.cache_misses * 2
        } else {
            num_rows // Worst case: every access is a miss
        };

        let total_cycles = scan_cost.cpu_cycles + sort_cycles;
        let time = self.params.cycles_to_seconds(total_cycles);

        CostEstimate::new(
            total_cycles,
            scan_cost.memory_accesses * 2,
            cache_misses,
            time,
            num_rows * std::mem::size_of::<f64>() > self.params.l3_cache_size,
        )
    }

    /// Estimate speedup from parallelization.
    pub fn estimate_parallel_speedup(&self, sequential_cost: &CostEstimate, num_threads: usize) -> CostEstimate {
        let effective_threads = num_threads.min(self.params.num_cores);

        // Memory-bound operations have limited parallel speedup due to bandwidth
        let speedup = if sequential_cost.memory_bound {
            (effective_threads as f64).min(2.0) // Limited by memory bandwidth
        } else {
            effective_threads as f64 * 0.8 // 80% parallel efficiency
        };

        CostEstimate::new(
            sequential_cost.cpu_cycles / speedup,
            sequential_cost.memory_accesses,
            sequential_cost.cache_misses,
            sequential_cost.time_seconds / speedup,
            sequential_cost.memory_bound,
        )
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Query operation types.
#[derive(Debug, Clone)]
pub enum QueryOp {
    /// Scan data.
    Scan { rows: usize, row_size: usize },
    /// Filter with selectivity.
    Filter { input_rows: usize, selectivity: f64 },
    /// Aggregate.
    Aggregate { rows: usize },
    /// Hash aggregate with groups.
    HashAggregate { rows: usize, groups: usize },
    /// Hash join.
    HashJoin { build_rows: usize, probe_rows: usize, output_rows: usize },
    /// Sort.
    Sort { rows: usize },
}

/// Query plan node.
#[derive(Debug)]
pub struct PlanNode {
    /// Operation.
    pub op: QueryOp,
    /// Input rows (from previous operation).
    pub input_rows: usize,
    /// Output rows.
    pub output_rows: usize,
    /// Estimated cost.
    pub cost: CostEstimate,
    /// Children nodes.
    pub children: Vec<PlanNode>,
}

impl PlanNode {
    /// Create a new plan node.
    pub fn new(op: QueryOp, cost: CostEstimate, input_rows: usize, output_rows: usize) -> Self {
        Self {
            op,
            input_rows,
            output_rows,
            cost,
            children: Vec::new(),
        }
    }

    /// Add child node.
    pub fn add_child(&mut self, child: PlanNode) {
        self.children.push(child);
    }

    /// Total cost including children.
    pub fn total_cost(&self) -> f64 {
        let child_cost: f64 = self.children.iter().map(|c| c.total_cost()).sum();
        self.cost.time_seconds + child_cost
    }
}

/// Query planner.
#[derive(Debug)]
pub struct QueryPlanner {
    /// Cost model.
    cost_model: CostModel,
}

impl QueryPlanner {
    /// Create new planner.
    pub fn new() -> Self {
        Self {
            cost_model: CostModel::new(),
        }
    }

    /// Create with specific cost model.
    pub fn with_cost_model(cost_model: CostModel) -> Self {
        Self { cost_model }
    }

    /// Plan a sequence of operations.
    pub fn plan(&self, operations: Vec<QueryOp>) -> Vec<PlanNode> {
        let mut nodes = Vec::new();
        let mut current_rows = 0usize;

        for op in operations {
            let (cost, output_rows) = match &op {
                QueryOp::Scan { rows, row_size } => {
                    current_rows = *rows;
                    let cost = self.cost_model.estimate_scan(*rows, *row_size);
                    (cost, *rows)
                }
                QueryOp::Filter { input_rows, selectivity } => {
                    let rows = if current_rows > 0 { current_rows } else { *input_rows };
                    let cost = self.cost_model.estimate_filter(rows, *selectivity);
                    let output = (rows as f64 * selectivity) as usize;
                    current_rows = output;
                    (cost, output)
                }
                QueryOp::Aggregate { rows } => {
                    let input = if current_rows > 0 { current_rows } else { *rows };
                    let cost = self.cost_model.estimate_aggregate(input);
                    current_rows = 1;
                    (cost, 1)
                }
                QueryOp::HashAggregate { rows, groups } => {
                    let input = if current_rows > 0 { current_rows } else { *rows };
                    let cost = self.cost_model.estimate_hash_aggregate(input, *groups);
                    current_rows = *groups;
                    (cost, *groups)
                }
                QueryOp::HashJoin { build_rows, probe_rows, output_rows } => {
                    let cost = self.cost_model.estimate_hash_join(*build_rows, *probe_rows, *output_rows);
                    current_rows = *output_rows;
                    (cost, *output_rows)
                }
                QueryOp::Sort { rows } => {
                    let input = if current_rows > 0 { current_rows } else { *rows };
                    let cost = self.cost_model.estimate_sort(input);
                    (cost, input)
                }
            };

            nodes.push(PlanNode::new(op, cost, current_rows, output_rows));
        }

        nodes
    }

    /// Estimate total time for plan.
    pub fn estimate_total_time(&self, nodes: &[PlanNode]) -> f64 {
        nodes.iter().map(|n| n.cost.time_seconds).sum()
    }

    /// Check if plan would benefit from parallelization.
    pub fn should_parallelize(&self, nodes: &[PlanNode]) -> bool {
        let total_rows: usize = nodes.iter().map(|n| n.input_rows).max().unwrap_or(0);
        total_rows > 10_000 // Parallelize for larger datasets
    }

    /// Optimize plan by reordering operations.
    pub fn optimize(&self, nodes: Vec<PlanNode>) -> Vec<PlanNode> {
        // Simple optimization: push filters before other operations when possible
        let mut optimized = nodes;

        // For a real optimizer, we would consider:
        // 1. Filter pushdown
        // 2. Join reordering
        // 3. Aggregation pushdown
        // 4. Parallel execution opportunities

        optimized
    }
}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for cardinality estimation.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Number of rows.
    pub row_count: usize,
    /// Number of distinct values.
    pub distinct_count: usize,
    /// Null count.
    pub null_count: usize,
    /// Minimum value (if applicable).
    pub min_value: Option<f64>,
    /// Maximum value (if applicable).
    pub max_value: Option<f64>,
    /// Average value (if applicable).
    pub avg_value: Option<f64>,
}

impl ColumnStats {
    /// Create new column stats.
    pub fn new(row_count: usize) -> Self {
        Self {
            row_count,
            distinct_count: row_count,
            null_count: 0,
            min_value: None,
            max_value: None,
            avg_value: None,
        }
    }

    /// Estimate selectivity for equality predicate.
    pub fn estimate_equality_selectivity(&self) -> f64 {
        if self.distinct_count > 0 {
            1.0 / self.distinct_count as f64
        } else {
            1.0
        }
    }

    /// Estimate selectivity for range predicate.
    pub fn estimate_range_selectivity(&self, low: f64, high: f64) -> f64 {
        match (self.min_value, self.max_value) {
            (Some(min), Some(max)) if max > min => {
                let range = max - min;
                let selected = (high.min(max) - low.max(min)).max(0.0);
                (selected / range).clamp(0.0, 1.0)
            }
            _ => 0.5, // Default estimate
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_params() {
        let params = HardwareParams::default();
        assert!(params.cpu_freq_ghz > 0.0);
        assert!(params.num_cores > 0);
        assert!(params.mem_bandwidth_gb_s > 0.0);
    }

    #[test]
    fn test_cost_model_scan() {
        let model = CostModel::new();
        let cost = model.estimate_scan(1_000_000, 8);

        assert!(cost.cpu_cycles > 0.0);
        assert!(cost.memory_accesses > 0);
        assert!(cost.time_seconds > 0.0);
    }

    #[test]
    fn test_cost_model_filter() {
        let model = CostModel::new();
        let cost = model.estimate_filter(1_000_000, 0.1);

        assert!(cost.time_seconds > 0.0);
    }

    #[test]
    fn test_cost_model_aggregate() {
        let model = CostModel::new();
        let cost = model.estimate_aggregate(1_000_000);

        assert!(cost.time_seconds > 0.0);
    }

    #[test]
    fn test_cost_model_hash_aggregate() {
        let model = CostModel::new();
        let cost = model.estimate_hash_aggregate(1_000_000, 1000);

        assert!(cost.time_seconds > 0.0);
    }

    #[test]
    fn test_cost_model_sort() {
        let model = CostModel::new();
        let cost = model.estimate_sort(1_000_000);

        assert!(cost.time_seconds > 0.0);
        assert!(cost.cpu_cycles > 0.0);
    }

    #[test]
    fn test_cost_model_parallel_speedup() {
        let model = CostModel::new();
        let sequential = model.estimate_aggregate(1_000_000);
        let parallel = model.estimate_parallel_speedup(&sequential, 4);

        assert!(parallel.time_seconds < sequential.time_seconds);
    }

    #[test]
    fn test_query_planner() {
        let planner = QueryPlanner::new();

        let ops = vec![
            QueryOp::Scan { rows: 1_000_000, row_size: 8 },
            QueryOp::Filter { input_rows: 1_000_000, selectivity: 0.1 },
            QueryOp::Aggregate { rows: 100_000 },
        ];

        let nodes = planner.plan(ops);
        assert_eq!(nodes.len(), 3);

        let total_time = planner.estimate_total_time(&nodes);
        assert!(total_time > 0.0);
    }

    #[test]
    fn test_column_stats() {
        let mut stats = ColumnStats::new(1000);
        stats.distinct_count = 100;
        stats.min_value = Some(0.0);
        stats.max_value = Some(100.0);

        let eq_sel = stats.estimate_equality_selectivity();
        assert!((eq_sel - 0.01).abs() < 0.001); // 1/100

        let range_sel = stats.estimate_range_selectivity(25.0, 75.0);
        assert!((range_sel - 0.5).abs() < 0.01); // 50% of range
    }
}
