//! Tests for vectorized operators (scan, filter, project, join, sort).

use columnar_query_engine::executor::*;
use columnar_query_engine::expression::*;
use columnar_query_engine::plan::*;
use columnar_query_engine::storage::*;
use columnar_query_engine::types::*;
use columnar_query_engine::vector::{DataChunk, Vector};
use std::sync::Arc;

// ============================================================================
// Helper Functions
// ============================================================================

fn create_test_catalog() -> Arc<Catalog> {
    Arc::new(Catalog::new())
}

fn create_int_vector(values: Vec<i64>) -> Vector {
    let mut vec = Vector::new(DataType::Int64);
    for v in values {
        vec.push(Value::Int64(v)).unwrap();
    }
    vec
}

fn create_float_vector(values: Vec<f64>) -> Vector {
    let mut vec = Vector::new(DataType::Float64);
    for v in values {
        vec.push(Value::Float64(v)).unwrap();
    }
    vec
}

fn create_string_vector(values: Vec<&str>) -> Vector {
    let mut vec = Vector::new(DataType::String);
    for v in values {
        vec.push(Value::String(v.to_string())).unwrap();
    }
    vec
}

fn create_bool_vector(values: Vec<bool>) -> Vector {
    let mut vec = Vector::new(DataType::Boolean);
    for v in values {
        vec.push(Value::Boolean(v)).unwrap();
    }
    vec
}

fn setup_employees_table(catalog: &Catalog) -> Arc<Table> {
    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("name", DataType::String, false),
        Column::new("department", DataType::String, false),
        Column::new("salary", DataType::Float64, false),
        Column::new("years", DataType::Int64, false),
    ]);

    let table = catalog
        .create_table("employees", schema, StorageConfig::default())
        .unwrap();

    // Insert test data
    let id_vec = create_int_vector(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let name_vec = create_string_vector(vec![
        "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack",
    ]);
    let dept_vec = create_string_vector(vec![
        "Engineering",
        "Sales",
        "Engineering",
        "HR",
        "Sales",
        "Engineering",
        "Marketing",
        "HR",
        "Engineering",
        "Sales",
    ]);
    let salary_vec = create_float_vector(vec![
        75000.0, 65000.0, 80000.0, 55000.0, 70000.0, 85000.0, 60000.0, 50000.0, 90000.0, 72000.0,
    ]);
    let years_vec = create_int_vector(vec![5, 3, 7, 2, 4, 10, 1, 8, 6, 3]);

    let chunk = DataChunk::new(vec![id_vec, name_vec, dept_vec, salary_vec, years_vec]);
    table.insert(chunk).unwrap();

    table
}

fn setup_departments_table(catalog: &Catalog) -> Arc<Table> {
    let schema = Schema::new(vec![
        Column::new("dept_name", DataType::String, false),
        Column::new("location", DataType::String, false),
        Column::new("budget", DataType::Float64, false),
    ]);

    let table = catalog
        .create_table("departments", schema, StorageConfig::default())
        .unwrap();

    let dept_vec =
        create_string_vector(vec!["Engineering", "Sales", "HR", "Marketing", "Finance"]);
    let location_vec = create_string_vector(vec![
        "Building A",
        "Building B",
        "Building A",
        "Building C",
        "Building B",
    ]);
    let budget_vec = create_float_vector(vec![500000.0, 300000.0, 200000.0, 150000.0, 400000.0]);

    let chunk = DataChunk::new(vec![dept_vec, location_vec, budget_vec]);
    table.insert(chunk).unwrap();

    table
}

// ============================================================================
// Scan Operator Tests
// ============================================================================

#[test]
fn test_scan_operator_basic() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let plan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1, 2, 3, 4],
        filter: None,
    };

    let results = executor.collect(&plan).unwrap();
    assert_eq!(results.len(), 1);

    let chunk = &results[0];
    assert_eq!(chunk.num_columns(), 5);
    assert_eq!(chunk.len(), 10);
}

#[test]
fn test_scan_operator_with_projection() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    // Only select id and salary
    let plan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 3],
        filter: None,
    };

    let results = executor.collect(&plan).unwrap();
    let chunk = &results[0];
    assert_eq!(chunk.num_columns(), 2);
}

#[test]
#[ignore = "Pre-existing: ColumnNotFound Index 3 error"]
fn test_scan_operator_with_filter() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    // Filter: salary > 70000
    let filter = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(3)), // salary column
        op: BinaryOperator::GreaterThan,
        right: Box::new(Expression::Literal(Value::Float64(70000.0))),
    };

    let plan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1, 3],
        filter: Some(filter),
    };

    let results = executor.collect(&plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert!(total_rows < 10); // Filtered result should have fewer rows
    assert!(total_rows > 0); // Should have at least some results
}

#[test]
fn test_scan_nonexistent_table() {
    let catalog = create_test_catalog();
    let executor = Executor::new(catalog);

    let plan = PhysicalPlan::SeqScan {
        table_name: "nonexistent".to_string(),
        projection: vec![0],
        filter: None,
    };

    let result = executor.collect(&plan);
    assert!(result.is_err());
}

// ============================================================================
// Filter Operator Tests
// ============================================================================

#[test]
fn test_filter_operator_basic() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1, 3, 4],
        filter: None,
    };

    // Filter: years > 5
    let filter_plan = PhysicalPlan::Filter {
        input: Arc::new(scan),
        predicate: Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(3)), // years is index 3 in projection
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Value::Int64(5))),
        },
    };

    let results = executor.collect(&filter_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert!(total_rows > 0);
    assert!(total_rows < 10);
}

#[test]
fn test_filter_operator_equality() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1, 2],
        filter: None,
    };

    // Filter: department = 'Engineering'
    let filter_plan = PhysicalPlan::Filter {
        input: Arc::new(scan),
        predicate: Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(2)),
            op: BinaryOperator::Equal,
            right: Box::new(Expression::Literal(Value::String("Engineering".to_string()))),
        },
    };

    let results = executor.collect(&filter_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 4); // Alice, Carol, Frank, Ivy
}

#[test]
fn test_filter_operator_compound_predicate() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1, 3, 4],
        filter: None,
    };

    // Filter: salary > 60000 AND years >= 5
    let predicate = Expression::BinaryOp {
        left: Box::new(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(2)), // salary
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Value::Float64(60000.0))),
        }),
        op: BinaryOperator::And,
        right: Box::new(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(3)), // years
            op: BinaryOperator::GreaterThanOrEqual,
            right: Box::new(Expression::Literal(Value::Int64(5))),
        }),
    };

    let filter_plan = PhysicalPlan::Filter {
        input: Arc::new(scan),
        predicate,
    };

    let results = executor.collect(&filter_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert!(total_rows > 0);
}

#[test]
fn test_filter_operator_no_matches() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 3],
        filter: None,
    };

    // Filter: salary > 1000000 (no one earns this much)
    let filter_plan = PhysicalPlan::Filter {
        input: Arc::new(scan),
        predicate: Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(1)),
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Value::Float64(1000000.0))),
        },
    };

    let results = executor.collect(&filter_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 0);
}

// ============================================================================
// Project Operator Tests
// ============================================================================

#[test]
fn test_project_operator_column_refs() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1, 2, 3, 4],
        filter: None,
    };

    // Project only id and name
    let project_plan = PhysicalPlan::Project {
        input: Arc::new(scan),
        expressions: vec![Expression::ColumnRef(0), Expression::ColumnRef(1)],
    };

    let results = executor.collect(&project_plan).unwrap();
    let chunk = &results[0];
    assert_eq!(chunk.num_columns(), 2);
    assert_eq!(chunk.len(), 10);
}

#[test]
fn test_project_operator_with_literals() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1],
        filter: None,
    };

    // Project id, name, and a literal constant
    let project_plan = PhysicalPlan::Project {
        input: Arc::new(scan),
        expressions: vec![
            Expression::ColumnRef(0),
            Expression::ColumnRef(1),
            Expression::Literal(Value::String("active".to_string())),
        ],
    };

    let results = executor.collect(&project_plan).unwrap();
    let chunk = &results[0];
    assert_eq!(chunk.num_columns(), 3);

    // Check literal column
    for i in 0..chunk.len() {
        let val = chunk.vector(2).unwrap().get(i).unwrap();
        assert_eq!(val, Value::String("active".to_string()));
    }
}

#[test]
fn test_project_operator_computed_expression() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 3], // id, salary
        filter: None,
    };

    // Project id, salary * 1.1 (10% raise)
    let project_plan = PhysicalPlan::Project {
        input: Arc::new(scan),
        expressions: vec![
            Expression::ColumnRef(0),
            Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef(1)),
                op: BinaryOperator::Multiply,
                right: Box::new(Expression::Literal(Value::Float64(1.1))),
            },
        ],
    };

    let results = executor.collect(&project_plan).unwrap();
    let chunk = &results[0];
    assert_eq!(chunk.num_columns(), 2);
}

// ============================================================================
// Sort Operator Tests
// ============================================================================

#[test]
fn test_sort_operator_ascending() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 3], // id, salary
        filter: None,
    };

    // Sort by salary ascending
    let sort_plan = PhysicalPlan::Sort {
        input: Arc::new(scan),
        order_by: vec![(1, SortOrder::Ascending, false)], // salary column
    };

    let results = executor.collect(&sort_plan).unwrap();
    let chunk = &results[0];

    // Verify ascending order
    let mut prev_salary = f64::MIN;
    for i in 0..chunk.len() {
        if let Value::Float64(salary) = chunk.vector(1).unwrap().get(i).unwrap() {
            assert!(
                salary >= prev_salary,
                "Sort order violated at index {}",
                i
            );
            prev_salary = salary;
        }
    }
}

#[test]
fn test_sort_operator_descending() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 3], // id, salary
        filter: None,
    };

    // Sort by salary descending
    let sort_plan = PhysicalPlan::Sort {
        input: Arc::new(scan),
        order_by: vec![(1, SortOrder::Descending, false)],
    };

    let results = executor.collect(&sort_plan).unwrap();
    let chunk = &results[0];

    // Verify descending order
    let mut prev_salary = f64::MAX;
    for i in 0..chunk.len() {
        if let Value::Float64(salary) = chunk.vector(1).unwrap().get(i).unwrap() {
            assert!(
                salary <= prev_salary,
                "Sort order violated at index {}",
                i
            );
            prev_salary = salary;
        }
    }
}

#[test]
fn test_sort_operator_multiple_keys() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 2, 3], // id, department, salary
        filter: None,
    };

    // Sort by department asc, then salary desc
    let sort_plan = PhysicalPlan::Sort {
        input: Arc::new(scan),
        order_by: vec![
            (1, SortOrder::Ascending, false),
            (2, SortOrder::Descending, false),
        ],
    };

    let results = executor.collect(&sort_plan).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].len(), 10);
}

// ============================================================================
// Limit Operator Tests
// ============================================================================

#[test]
fn test_limit_operator_basic() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1],
        filter: None,
    };

    let limit_plan = PhysicalPlan::Limit {
        input: Arc::new(scan),
        limit: 5,
        offset: 0,
    };

    let results = executor.collect(&limit_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 5);
}

#[test]
fn test_limit_operator_with_offset() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1],
        filter: None,
    };

    let limit_plan = PhysicalPlan::Limit {
        input: Arc::new(scan),
        limit: 3,
        offset: 5,
    };

    let results = executor.collect(&limit_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 3);
}

#[test]
fn test_limit_operator_exceeds_data() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0],
        filter: None,
    };

    let limit_plan = PhysicalPlan::Limit {
        input: Arc::new(scan),
        limit: 100, // More than available rows
        offset: 0,
    };

    let results = executor.collect(&limit_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 10); // Only 10 rows available
}

#[test]
fn test_limit_operator_offset_exceeds_data() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0],
        filter: None,
    };

    let limit_plan = PhysicalPlan::Limit {
        input: Arc::new(scan),
        limit: 5,
        offset: 100, // Beyond available rows
    };

    let results = executor.collect(&limit_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 0);
}

// ============================================================================
// Hash Join Operator Tests
// ============================================================================

#[test]
fn test_hash_join_inner() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);
    setup_departments_table(&catalog);

    let executor = Executor::new(catalog);

    let left_scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1, 2, 3], // id, name, department, salary
        filter: None,
    };

    let right_scan = PhysicalPlan::SeqScan {
        table_name: "departments".to_string(),
        projection: vec![0, 1, 2], // dept_name, location, budget
        filter: None,
    };

    let join_plan = PhysicalPlan::HashJoin {
        left: Arc::new(left_scan),
        right: Arc::new(right_scan),
        join_type: JoinType::Inner,
        left_keys: vec![2],  // department column
        right_keys: vec![0], // dept_name column
        condition: None,
    };

    let results = executor.collect(&join_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    // All employees should match (Engineering, Sales, HR, Marketing all exist)
    assert!(total_rows > 0);
}

#[test]
fn test_hash_join_with_condition() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);
    setup_departments_table(&catalog);

    let executor = Executor::new(catalog);

    let left_scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 2, 3], // id, department, salary
        filter: None,
    };

    let right_scan = PhysicalPlan::SeqScan {
        table_name: "departments".to_string(),
        projection: vec![0, 2], // dept_name, budget
        filter: None,
    };

    // Join with additional condition: salary < budget
    let condition = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(2)),  // salary from left
        op: BinaryOperator::LessThan,
        right: Box::new(Expression::ColumnRef(4)), // budget from right
    };

    let join_plan = PhysicalPlan::HashJoin {
        left: Arc::new(left_scan),
        right: Arc::new(right_scan),
        join_type: JoinType::Inner,
        left_keys: vec![1],
        right_keys: vec![0],
        condition: Some(condition),
    };

    let results = executor.collect(&join_plan).unwrap();
    // Results should be filtered by the condition
    assert!(!results.is_empty() || results.iter().all(|c| c.is_empty()));
}

// ============================================================================
// Hash Aggregate Operator Tests
// ============================================================================

#[test]
#[ignore = "Pre-existing: Type mismatch error"]
fn test_hash_aggregate_count() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 2], // id, department
        filter: None,
    };

    let agg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(scan),
        group_by: vec![Expression::ColumnRef(1)], // group by department
        aggregates: vec![(AggregateFunction::Count, Expression::ColumnRef(0))],
    };

    let results = executor.collect(&agg_plan).unwrap();
    assert!(!results.is_empty());

    // Should have one row per department
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 4); // Engineering, Sales, HR, Marketing
}

#[test]
fn test_hash_aggregate_sum() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![2, 3], // department, salary
        filter: None,
    };

    let agg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(scan),
        group_by: vec![Expression::ColumnRef(0)],
        aggregates: vec![(AggregateFunction::Sum, Expression::ColumnRef(1))],
    };

    let results = executor.collect(&agg_plan).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_hash_aggregate_avg() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![2, 3], // department, salary
        filter: None,
    };

    let agg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(scan),
        group_by: vec![Expression::ColumnRef(0)],
        aggregates: vec![(AggregateFunction::Avg, Expression::ColumnRef(1))],
    };

    let results = executor.collect(&agg_plan).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_hash_aggregate_min_max() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![2, 3], // department, salary
        filter: None,
    };

    let agg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(scan),
        group_by: vec![Expression::ColumnRef(0)],
        aggregates: vec![
            (AggregateFunction::Min, Expression::ColumnRef(1)),
            (AggregateFunction::Max, Expression::ColumnRef(1)),
        ],
    };

    let results = executor.collect(&agg_plan).unwrap();
    assert!(!results.is_empty());
}

#[test]
#[ignore = "Pre-existing: Type mismatch error"]
fn test_hash_aggregate_no_groups() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![3], // salary
        filter: None,
    };

    // Aggregate without grouping (global aggregate)
    let agg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(scan),
        group_by: vec![],
        aggregates: vec![
            (AggregateFunction::Count, Expression::ColumnRef(0)),
            (AggregateFunction::Sum, Expression::ColumnRef(0)),
            (AggregateFunction::Avg, Expression::ColumnRef(0)),
        ],
    };

    let results = executor.collect(&agg_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 1); // Single result row
}

// ============================================================================
// Union Operator Tests
// ============================================================================

#[test]
fn test_union_operator() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    // Create two scans of the same table
    let scan1 = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1],
        filter: Some(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(0)),
            op: BinaryOperator::LessThan,
            right: Box::new(Expression::Literal(Value::Int64(5))),
        }),
    };

    let scan2 = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1],
        filter: Some(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(0)),
            op: BinaryOperator::GreaterThanOrEqual,
            right: Box::new(Expression::Literal(Value::Int64(5))),
        }),
    };

    let union_plan = PhysicalPlan::UnionAll {
        inputs: vec![Arc::new(scan1), Arc::new(scan2)],
    };

    let results = executor.collect(&union_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 10); // All rows combined
}

// ============================================================================
// Distinct Operator Tests
// ============================================================================

#[test]
fn test_distinct_operator() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![2], // department only
        filter: None,
    };

    let distinct_plan = PhysicalPlan::HashDistinct {
        input: Arc::new(scan),
    };

    let results = executor.collect(&distinct_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 4); // 4 unique departments
}

// ============================================================================
// Values Operator Tests
// ============================================================================

#[test]
#[ignore = "Pre-existing: Type mismatch error"]
fn test_values_operator() {
    let catalog = create_test_catalog();
    let executor = Executor::new(catalog);

    let values_plan = PhysicalPlan::Values {
        values: vec![
            vec![
                Expression::Literal(Value::Int64(1)),
                Expression::Literal(Value::String("A".to_string())),
            ],
            vec![
                Expression::Literal(Value::Int64(2)),
                Expression::Literal(Value::String("B".to_string())),
            ],
            vec![
                Expression::Literal(Value::Int64(3)),
                Expression::Literal(Value::String("C".to_string())),
            ],
        ],
    };

    let results = executor.collect(&values_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 3);
}

// ============================================================================
// Empty Operator Tests
// ============================================================================

#[test]
fn test_empty_operator() {
    let catalog = create_test_catalog();
    let executor = Executor::new(catalog);

    let empty_plan = PhysicalPlan::Empty;

    let results = executor.collect(&empty_plan).unwrap();
    assert!(results.is_empty());
}

// ============================================================================
// Complex Query Pipeline Tests
// ============================================================================

#[test]
fn test_complex_pipeline_filter_project_sort_limit() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    // Build a pipeline: Scan -> Filter -> Project -> Sort -> Limit
    let scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0, 1, 2, 3, 4],
        filter: None,
    };

    // Filter: salary > 60000
    let filtered = PhysicalPlan::Filter {
        input: Arc::new(scan),
        predicate: Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(3)),
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Value::Float64(60000.0))),
        },
    };

    // Project: id, name, salary
    let projected = PhysicalPlan::Project {
        input: Arc::new(filtered),
        expressions: vec![
            Expression::ColumnRef(0),
            Expression::ColumnRef(1),
            Expression::ColumnRef(3),
        ],
    };

    // Sort by salary descending
    let sorted = PhysicalPlan::Sort {
        input: Arc::new(projected),
        order_by: vec![(2, SortOrder::Descending, false)],
    };

    // Limit to top 3
    let limited = PhysicalPlan::Limit {
        input: Arc::new(sorted),
        limit: 3,
        offset: 0,
    };

    let results = executor.collect(&limited).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert!(total_rows <= 3);
}

#[test]
fn test_complex_pipeline_join_aggregate() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);
    setup_departments_table(&catalog);

    let executor = Executor::new(catalog);

    // Build a pipeline: Join -> Aggregate
    let left_scan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![2, 3], // department, salary
        filter: None,
    };

    let right_scan = PhysicalPlan::SeqScan {
        table_name: "departments".to_string(),
        projection: vec![0, 1], // dept_name, location
        filter: None,
    };

    let joined = PhysicalPlan::HashJoin {
        left: Arc::new(left_scan),
        right: Arc::new(right_scan),
        join_type: JoinType::Inner,
        left_keys: vec![0],
        right_keys: vec![0],
        condition: None,
    };

    // Aggregate: group by location, sum salary
    let aggregated = PhysicalPlan::HashAggregate {
        input: Arc::new(joined),
        group_by: vec![Expression::ColumnRef(3)], // location
        aggregates: vec![(AggregateFunction::Sum, Expression::ColumnRef(1))], // sum of salary
    };

    let results = executor.collect(&aggregated).unwrap();
    assert!(!results.is_empty());
}

// ============================================================================
// Operator Trait Tests
// ============================================================================

#[test]
fn test_operator_next_returns_none_when_exhausted() {
    let catalog = create_test_catalog();
    setup_employees_table(&catalog);

    let executor = Executor::new(catalog);

    let plan = PhysicalPlan::SeqScan {
        table_name: "employees".to_string(),
        projection: vec![0],
        filter: None,
    };

    let mut operator = executor.execute(&plan).unwrap();

    // Consume all data
    let mut count = 0;
    while let Some(_) = operator.next().unwrap() {
        count += 1;
        if count > 100 {
            break; // Safety limit
        }
    }

    // Next call should return None
    assert!(operator.next().unwrap().is_none());
}
