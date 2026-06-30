//! Query plan nodes.

use crate::expression::Expression;
use crate::types::{AggregateFunction, JoinType, Schema, SortOrder};
use std::sync::Arc;

/// Logical plan node.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan a table.
    Scan {
        table_name: String,
        projection: Option<Vec<usize>>,
        filter: Option<Expression>,
        schema: Schema,
    },

    /// Filter rows.
    Filter {
        input: Arc<LogicalPlan>,
        predicate: Expression,
    },

    /// Project columns.
    Projection {
        input: Arc<LogicalPlan>,
        expressions: Vec<Expression>,
        schema: Schema,
    },

    /// Aggregate with grouping.
    Aggregate {
        input: Arc<LogicalPlan>,
        group_by: Vec<Expression>,
        aggregates: Vec<(AggregateFunction, Expression, String)>,
        schema: Schema,
    },

    /// Sort rows.
    Sort {
        input: Arc<LogicalPlan>,
        order_by: Vec<(Expression, SortOrder, bool)>, // expr, order, nulls_first
    },

    /// Limit rows.
    Limit {
        input: Arc<LogicalPlan>,
        limit: usize,
        offset: usize,
    },

    /// Join two tables.
    Join {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        join_type: JoinType,
        condition: Option<Expression>,
        schema: Schema,
    },

    /// Union of two queries.
    Union {
        left: Arc<LogicalPlan>,
        right: Arc<LogicalPlan>,
        all: bool,
    },

    /// Distinct rows.
    Distinct {
        input: Arc<LogicalPlan>,
    },

    /// Values (inline data).
    Values {
        values: Vec<Vec<Expression>>,
        schema: Schema,
    },

    /// Subquery.
    Subquery {
        input: Arc<LogicalPlan>,
        alias: String,
    },

    /// Empty result.
    Empty {
        schema: Schema,
    },
}

impl LogicalPlan {
    /// Get output schema.
    pub fn schema(&self) -> &Schema {
        match self {
            LogicalPlan::Scan { schema, .. } => schema,
            LogicalPlan::Filter { input, .. } => input.schema(),
            LogicalPlan::Projection { schema, .. } => schema,
            LogicalPlan::Aggregate { schema, .. } => schema,
            LogicalPlan::Sort { input, .. } => input.schema(),
            LogicalPlan::Limit { input, .. } => input.schema(),
            LogicalPlan::Join { schema, .. } => schema,
            LogicalPlan::Union { left, .. } => left.schema(),
            LogicalPlan::Distinct { input } => input.schema(),
            LogicalPlan::Values { schema, .. } => schema,
            LogicalPlan::Subquery { input, .. } => input.schema(),
            LogicalPlan::Empty { schema } => schema,
        }
    }

    /// Get child plans.
    pub fn children(&self) -> Vec<&LogicalPlan> {
        match self {
            LogicalPlan::Scan { .. } | LogicalPlan::Values { .. } | LogicalPlan::Empty { .. } => {
                vec![]
            }
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Projection { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Subquery { input, .. } => vec![input],
            LogicalPlan::Join { left, right, .. } | LogicalPlan::Union { left, right, .. } => {
                vec![left, right]
            }
        }
    }

    /// Apply a transformation to all nodes.
    pub fn transform<F>(self, f: &F) -> Self
    where
        F: Fn(LogicalPlan) -> LogicalPlan,
    {
        let transformed = match self {
            LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
                input: Arc::new((*input).clone().transform(f)),
                predicate,
            },
            LogicalPlan::Projection {
                input,
                expressions,
                schema,
            } => LogicalPlan::Projection {
                input: Arc::new((*input).clone().transform(f)),
                expressions,
                schema,
            },
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
                schema,
            } => LogicalPlan::Aggregate {
                input: Arc::new((*input).clone().transform(f)),
                group_by,
                aggregates,
                schema,
            },
            LogicalPlan::Sort { input, order_by } => LogicalPlan::Sort {
                input: Arc::new((*input).clone().transform(f)),
                order_by,
            },
            LogicalPlan::Limit {
                input,
                limit,
                offset,
            } => LogicalPlan::Limit {
                input: Arc::new((*input).clone().transform(f)),
                limit,
                offset,
            },
            LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
                input: Arc::new((*input).clone().transform(f)),
            },
            LogicalPlan::Join {
                left,
                right,
                join_type,
                condition,
                schema,
            } => LogicalPlan::Join {
                left: Arc::new((*left).clone().transform(f)),
                right: Arc::new((*right).clone().transform(f)),
                join_type,
                condition,
                schema,
            },
            LogicalPlan::Union { left, right, all } => LogicalPlan::Union {
                left: Arc::new((*left).clone().transform(f)),
                right: Arc::new((*right).clone().transform(f)),
                all,
            },
            other => other,
        };

        f(transformed)
    }
}

/// Physical plan node.
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Sequential scan.
    SeqScan {
        table_name: String,
        projection: Vec<usize>,
        filter: Option<Expression>,
    },

    /// Parallel scan (partitioned across threads).
    ParallelScan {
        table_name: String,
        projection: Vec<usize>,
        filter: Option<Expression>,
        num_partitions: usize,
    },

    /// Index scan.
    IndexScan {
        table_name: String,
        index_name: String,
        projection: Vec<usize>,
        filter: Option<Expression>,
    },

    /// Filter operator.
    Filter {
        input: Arc<PhysicalPlan>,
        predicate: Expression,
    },

    /// Project operator.
    Project {
        input: Arc<PhysicalPlan>,
        expressions: Vec<Expression>,
    },

    /// Hash aggregate.
    HashAggregate {
        input: Arc<PhysicalPlan>,
        group_by: Vec<Expression>,
        aggregates: Vec<(AggregateFunction, Expression)>,
    },

    /// Parallel hash aggregate (partitioned across threads).
    ParallelHashAggregate {
        input: Arc<PhysicalPlan>,
        group_by: Vec<Expression>,
        aggregates: Vec<(AggregateFunction, Expression)>,
        num_partitions: usize,
    },

    /// Sort-based aggregate.
    SortAggregate {
        input: Arc<PhysicalPlan>,
        group_by: Vec<Expression>,
        aggregates: Vec<(AggregateFunction, Expression)>,
    },

    /// External sort.
    Sort {
        input: Arc<PhysicalPlan>,
        order_by: Vec<(usize, SortOrder, bool)>,
    },

    /// Top-N sort (limit with sort).
    TopN {
        input: Arc<PhysicalPlan>,
        order_by: Vec<(usize, SortOrder, bool)>,
        limit: usize,
    },

    /// Limit operator.
    Limit {
        input: Arc<PhysicalPlan>,
        limit: usize,
        offset: usize,
    },

    /// Hash join.
    HashJoin {
        left: Arc<PhysicalPlan>,
        right: Arc<PhysicalPlan>,
        join_type: JoinType,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
        condition: Option<Expression>,
    },

    /// Nested loop join.
    NestedLoopJoin {
        left: Arc<PhysicalPlan>,
        right: Arc<PhysicalPlan>,
        join_type: JoinType,
        condition: Option<Expression>,
    },

    /// Sort-merge join.
    MergeJoin {
        left: Arc<PhysicalPlan>,
        right: Arc<PhysicalPlan>,
        join_type: JoinType,
        left_keys: Vec<usize>,
        right_keys: Vec<usize>,
    },

    /// Union all.
    UnionAll {
        inputs: Vec<Arc<PhysicalPlan>>,
    },

    /// Hash distinct.
    HashDistinct {
        input: Arc<PhysicalPlan>,
    },

    /// Values (inline data).
    Values {
        values: Vec<Vec<Expression>>,
    },

    /// Empty result.
    Empty,
}

impl PhysicalPlan {
    /// Get child plans.
    pub fn children(&self) -> Vec<&PhysicalPlan> {
        match self {
            PhysicalPlan::SeqScan { .. }
            | PhysicalPlan::ParallelScan { .. }
            | PhysicalPlan::IndexScan { .. }
            | PhysicalPlan::Values { .. }
            | PhysicalPlan::Empty => vec![],

            PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Project { input, .. }
            | PhysicalPlan::HashAggregate { input, .. }
            | PhysicalPlan::ParallelHashAggregate { input, .. }
            | PhysicalPlan::SortAggregate { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::TopN { input, .. }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::HashDistinct { input } => vec![input],

            PhysicalPlan::HashJoin { left, right, .. }
            | PhysicalPlan::NestedLoopJoin { left, right, .. }
            | PhysicalPlan::MergeJoin { left, right, .. } => vec![left, right],

            PhysicalPlan::UnionAll { inputs } => inputs.iter().map(|i| i.as_ref()).collect(),
        }
    }

    /// Estimate cost for query optimization.
    pub fn estimate_cost(&self) -> f64 {
        match self {
            PhysicalPlan::SeqScan { .. } => 1000.0,
            PhysicalPlan::ParallelScan { num_partitions, .. } => {
                // Parallel scan divides work across partitions
                1000.0 / (*num_partitions as f64).max(1.0)
            }
            PhysicalPlan::IndexScan { .. } => 100.0,
            PhysicalPlan::Filter { input, .. } => input.estimate_cost() * 0.5,
            PhysicalPlan::Project { input, .. } => input.estimate_cost() * 1.1,
            PhysicalPlan::HashAggregate { input, .. } => input.estimate_cost() * 1.5,
            PhysicalPlan::ParallelHashAggregate { input, num_partitions, .. } => {
                // Parallel aggregate divides work across partitions
                input.estimate_cost() * 1.5 / (*num_partitions as f64).max(1.0)
            }
            PhysicalPlan::SortAggregate { input, .. } => input.estimate_cost() * 2.0,
            PhysicalPlan::Sort { input, .. } => input.estimate_cost() * 2.0,
            PhysicalPlan::TopN { input, limit, .. } => {
                input.estimate_cost() * (*limit as f64 / 1000.0).min(1.0)
            }
            PhysicalPlan::Limit { input, .. } => input.estimate_cost(),
            PhysicalPlan::HashJoin { left, right, .. } => {
                left.estimate_cost() + right.estimate_cost() * 1.2
            }
            PhysicalPlan::NestedLoopJoin { left, right, .. } => {
                left.estimate_cost() * right.estimate_cost()
            }
            PhysicalPlan::MergeJoin { left, right, .. } => {
                left.estimate_cost() + right.estimate_cost()
            }
            PhysicalPlan::UnionAll { inputs } => {
                inputs.iter().map(|i| i.estimate_cost()).sum()
            }
            PhysicalPlan::HashDistinct { input } => input.estimate_cost() * 1.3,
            PhysicalPlan::Values { values } => values.len() as f64,
            PhysicalPlan::Empty => 0.0,
        }
    }
}

/// Query optimizer.
pub struct Optimizer {
    /// Enable predicate pushdown.
    pub predicate_pushdown: bool,
    /// Enable projection pushdown.
    pub projection_pushdown: bool,
    /// Enable join reordering.
    pub join_reorder: bool,
    /// Enable parallel execution.
    pub enable_parallel: bool,
    /// Row threshold for enabling parallel execution.
    pub parallel_threshold: usize,
    /// Number of partitions for parallel operators.
    pub num_partitions: usize,
    /// Table row counts for cost estimation.
    table_row_counts: std::collections::HashMap<String, usize>,
}

impl Default for Optimizer {
    fn default() -> Self {
        Self {
            predicate_pushdown: true,
            projection_pushdown: true,
            join_reorder: true,
            enable_parallel: false,
            parallel_threshold: 10_000,
            num_partitions: num_cpus::get().max(1),
            table_row_counts: std::collections::HashMap::new(),
        }
    }
}

impl Optimizer {
    /// Create optimizer with parallel execution enabled.
    pub fn with_parallelism(num_partitions: usize, parallel_threshold: usize) -> Self {
        Self {
            enable_parallel: true,
            parallel_threshold,
            num_partitions,
            ..Default::default()
        }
    }

    /// Set row count for a table (used for cost-based decisions).
    pub fn set_table_row_count(&mut self, table_name: &str, row_count: usize) {
        self.table_row_counts.insert(table_name.to_string(), row_count);
    }

    /// Get estimated row count for a table.
    pub fn get_table_row_count(&self, table_name: &str) -> Option<usize> {
        self.table_row_counts.get(table_name).copied()
    }

    /// Check if parallelism should be used for given row count.
    fn should_parallelize(&self, row_count: usize) -> bool {
        self.enable_parallel && row_count >= self.parallel_threshold
    }

    /// Estimate number of rows from a logical plan.
    fn estimate_input_rows(&self, plan: &LogicalPlan) -> usize {
        match plan {
            LogicalPlan::Scan { table_name, filter, .. } => {
                let base_rows = self.get_table_row_count(table_name).unwrap_or(1000);
                // Apply selectivity estimate for filters
                if filter.is_some() {
                    base_rows / 10 // Assume 10% selectivity
                } else {
                    base_rows
                }
            }
            LogicalPlan::Filter { input, .. } => {
                self.estimate_input_rows(input) / 10 // Assume 10% selectivity
            }
            LogicalPlan::Projection { input, .. } => self.estimate_input_rows(input),
            LogicalPlan::Aggregate { input, group_by, .. } => {
                let input_rows = self.estimate_input_rows(input);
                if group_by.is_empty() {
                    1
                } else {
                    input_rows / 10 // Assume ~10% unique groups
                }
            }
            LogicalPlan::Sort { input, .. } => self.estimate_input_rows(input),
            LogicalPlan::Limit { limit, .. } => *limit,
            LogicalPlan::Join { left, right, .. } => {
                let left_rows = self.estimate_input_rows(left);
                let right_rows = self.estimate_input_rows(right);
                (left_rows * right_rows) / 10 // Assume 10% selectivity
            }
            LogicalPlan::Union { left, right, all } => {
                let combined = self.estimate_input_rows(left) + self.estimate_input_rows(right);
                if *all { combined } else { combined / 2 }
            }
            LogicalPlan::Distinct { input } => self.estimate_input_rows(input) / 2,
            LogicalPlan::Values { values, .. } => values.len(),
            LogicalPlan::Subquery { input, .. } => self.estimate_input_rows(input),
            LogicalPlan::Empty { .. } => 0,
        }
    }

    /// Optimize a logical plan.
    pub fn optimize(&self, plan: LogicalPlan) -> LogicalPlan {
        let mut optimized = plan;

        if self.predicate_pushdown {
            optimized = self.push_down_predicates(optimized);
        }

        if self.projection_pushdown {
            optimized = self.push_down_projections(optimized);
        }

        optimized
    }

    /// Push predicates down closer to scans.
    fn push_down_predicates(&self, plan: LogicalPlan) -> LogicalPlan {
        plan.transform(&|node| match node {
            LogicalPlan::Filter {
                input,
                predicate,
            } => {
                if let LogicalPlan::Scan {
                    table_name,
                    projection,
                    filter,
                    schema,
                } = (*input).clone()
                {
                    // Merge filter into scan
                    let new_filter = match filter {
                        Some(existing) => Some(Expression::BinaryOp {
                            left: Box::new(existing),
                            op: crate::expression::BinaryOperator::And,
                            right: Box::new(predicate),
                        }),
                        None => Some(predicate),
                    };

                    LogicalPlan::Scan {
                        table_name,
                        projection,
                        filter: new_filter,
                        schema,
                    }
                } else {
                    LogicalPlan::Filter { input, predicate }
                }
            }
            other => other,
        })
    }

    /// Push projections down to reduce data volume.
    fn push_down_projections(&self, plan: LogicalPlan) -> LogicalPlan {
        // Simplified - just return the plan
        plan
    }

    /// Convert logical plan to physical plan.
    pub fn to_physical(&self, plan: &LogicalPlan) -> PhysicalPlan {
        match plan {
            LogicalPlan::Scan {
                table_name,
                projection,
                filter,
                schema,
            } => {
                let proj = projection
                    .clone()
                    .unwrap_or_else(|| (0..schema.len()).collect());

                // Check if we should use parallel scan based on table statistics
                let row_count = self.get_table_row_count(table_name).unwrap_or(0);
                if self.should_parallelize(row_count) {
                    PhysicalPlan::ParallelScan {
                        table_name: table_name.clone(),
                        projection: proj,
                        filter: filter.clone(),
                        num_partitions: self.num_partitions,
                    }
                } else {
                    PhysicalPlan::SeqScan {
                        table_name: table_name.clone(),
                        projection: proj,
                        filter: filter.clone(),
                    }
                }
            }

            LogicalPlan::Filter { input, predicate } => PhysicalPlan::Filter {
                input: Arc::new(self.to_physical(input)),
                predicate: predicate.clone(),
            },

            LogicalPlan::Projection { input, expressions, .. } => PhysicalPlan::Project {
                input: Arc::new(self.to_physical(input)),
                expressions: expressions.clone(),
            },

            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
                ..
            } => {
                let child = Arc::new(self.to_physical(input));
                let agg_exprs: Vec<(AggregateFunction, Expression)> = aggregates
                    .iter()
                    .map(|(func, expr, _)| (*func, expr.clone()))
                    .collect();

                // Estimate input rows to decide on parallel aggregate
                let estimated_rows = self.estimate_input_rows(input);
                if self.should_parallelize(estimated_rows) {
                    PhysicalPlan::ParallelHashAggregate {
                        input: child,
                        group_by: group_by.clone(),
                        aggregates: agg_exprs,
                        num_partitions: self.num_partitions,
                    }
                } else {
                    PhysicalPlan::HashAggregate {
                        input: child,
                        group_by: group_by.clone(),
                        aggregates: agg_exprs,
                    }
                }
            },

            LogicalPlan::Sort { input, order_by } => {
                // Convert expressions to column indices (simplified)
                let order: Vec<(usize, SortOrder, bool)> = order_by
                    .iter()
                    .enumerate()
                    .map(|(i, (_, order, nulls_first))| (i, *order, *nulls_first))
                    .collect();

                PhysicalPlan::Sort {
                    input: Arc::new(self.to_physical(input)),
                    order_by: order,
                }
            }

            LogicalPlan::Limit { input, limit, offset } => PhysicalPlan::Limit {
                input: Arc::new(self.to_physical(input)),
                limit: *limit,
                offset: *offset,
            },

            LogicalPlan::Join {
                left,
                right,
                join_type,
                condition,
                ..
            } => {
                // Use hash join by default
                PhysicalPlan::HashJoin {
                    left: Arc::new(self.to_physical(left)),
                    right: Arc::new(self.to_physical(right)),
                    join_type: *join_type,
                    left_keys: vec![0], // Simplified
                    right_keys: vec![0],
                    condition: condition.clone(),
                }
            }

            LogicalPlan::Union { left, right, all } => {
                if *all {
                    PhysicalPlan::UnionAll {
                        inputs: vec![
                            Arc::new(self.to_physical(left)),
                            Arc::new(self.to_physical(right)),
                        ],
                    }
                } else {
                    PhysicalPlan::HashDistinct {
                        input: Arc::new(PhysicalPlan::UnionAll {
                            inputs: vec![
                                Arc::new(self.to_physical(left)),
                                Arc::new(self.to_physical(right)),
                            ],
                        }),
                    }
                }
            }

            LogicalPlan::Distinct { input } => PhysicalPlan::HashDistinct {
                input: Arc::new(self.to_physical(input)),
            },

            LogicalPlan::Values { values, .. } => PhysicalPlan::Values {
                values: values.clone(),
            },

            LogicalPlan::Subquery { input, .. } => self.to_physical(input),

            LogicalPlan::Empty { .. } => PhysicalPlan::Empty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DataType;

    fn create_test_schema() -> Schema {
        Schema::from_pairs(vec![
            ("id".to_string(), DataType::Int64),
            ("name".to_string(), DataType::String),
        ])
    }

    #[test]
    fn test_parallel_scan_decision() {
        let mut optimizer = Optimizer::with_parallelism(4, 10_000);
        optimizer.set_table_row_count("large_table", 100_000);
        optimizer.set_table_row_count("small_table", 1_000);

        let schema = create_test_schema();

        // Large table should use parallel scan
        let large_scan = LogicalPlan::Scan {
            table_name: "large_table".to_string(),
            projection: None,
            filter: None,
            schema: schema.clone(),
        };
        let physical = optimizer.to_physical(&large_scan);
        assert!(matches!(physical, PhysicalPlan::ParallelScan { num_partitions: 4, .. }));

        // Small table should use sequential scan
        let small_scan = LogicalPlan::Scan {
            table_name: "small_table".to_string(),
            projection: None,
            filter: None,
            schema,
        };
        let physical = optimizer.to_physical(&small_scan);
        assert!(matches!(physical, PhysicalPlan::SeqScan { .. }));
    }

    #[test]
    fn test_parallel_aggregate_decision() {
        let mut optimizer = Optimizer::with_parallelism(4, 10_000);
        optimizer.set_table_row_count("large_table", 100_000);

        let schema = create_test_schema();

        // Large aggregate should use parallel hash aggregate
        let aggregate = LogicalPlan::Aggregate {
            input: Arc::new(LogicalPlan::Scan {
                table_name: "large_table".to_string(),
                projection: None,
                filter: None,
                schema,
            }),
            group_by: vec![Expression::ColumnRef(0)],
            aggregates: vec![(AggregateFunction::Count, Expression::ColumnRef(0), "count".to_string())],
            schema: Schema::from_pairs(vec![
                ("id".to_string(), DataType::Int64),
                ("count".to_string(), DataType::Int64),
            ]),
        };

        let physical = optimizer.to_physical(&aggregate);
        assert!(matches!(physical, PhysicalPlan::ParallelHashAggregate { num_partitions: 4, .. }));
    }

    #[test]
    fn test_non_parallel_mode() {
        let optimizer = Optimizer::default();

        let schema = create_test_schema();
        let scan = LogicalPlan::Scan {
            table_name: "any_table".to_string(),
            projection: None,
            filter: None,
            schema,
        };

        // Without parallel mode, should always use sequential scan
        let physical = optimizer.to_physical(&scan);
        assert!(matches!(physical, PhysicalPlan::SeqScan { .. }));
    }

    #[test]
    fn test_estimate_input_rows() {
        let mut optimizer = Optimizer::default();
        optimizer.set_table_row_count("test_table", 10_000);

        let schema = create_test_schema();

        // Scan returns base row count
        let scan = LogicalPlan::Scan {
            table_name: "test_table".to_string(),
            projection: None,
            filter: None,
            schema,
        };
        assert_eq!(optimizer.estimate_input_rows(&scan), 10_000);

        // Limit returns limit value
        let limit = LogicalPlan::Limit {
            input: Arc::new(scan.clone()),
            limit: 100,
            offset: 0,
        };
        assert_eq!(optimizer.estimate_input_rows(&limit), 100);
    }

    #[test]
    fn test_parallel_scan_cost_estimate() {
        let seq_scan = PhysicalPlan::SeqScan {
            table_name: "test".to_string(),
            projection: vec![0, 1],
            filter: None,
        };

        let parallel_scan = PhysicalPlan::ParallelScan {
            table_name: "test".to_string(),
            projection: vec![0, 1],
            filter: None,
            num_partitions: 4,
        };

        // Parallel scan should have lower cost estimate
        assert!(parallel_scan.estimate_cost() < seq_scan.estimate_cost());
    }

    #[test]
    fn test_parallel_aggregate_cost_estimate() {
        let seq_agg = PhysicalPlan::HashAggregate {
            input: Arc::new(PhysicalPlan::SeqScan {
                table_name: "test".to_string(),
                projection: vec![0],
                filter: None,
            }),
            group_by: vec![],
            aggregates: vec![],
        };

        let parallel_agg = PhysicalPlan::ParallelHashAggregate {
            input: Arc::new(PhysicalPlan::SeqScan {
                table_name: "test".to_string(),
                projection: vec![0],
                filter: None,
            }),
            group_by: vec![],
            aggregates: vec![],
            num_partitions: 4,
        };

        // Parallel aggregate should have lower cost estimate
        assert!(parallel_agg.estimate_cost() < seq_agg.estimate_cost());
    }
}
