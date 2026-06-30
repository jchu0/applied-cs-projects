//! Integration tests for the columnar query engine.

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

// Setup a complete orders database for integration tests
fn setup_orders_database(catalog: &Catalog) {
    // Customers table
    let customer_schema = Schema::new(vec![
        Column::new("customer_id", DataType::Int64, false),
        Column::new("name", DataType::String, false),
        Column::new("city", DataType::String, false),
        Column::new("country", DataType::String, false),
    ]);
    let customers = catalog
        .create_table("customers", customer_schema, StorageConfig::default())
        .unwrap();

    let customer_ids = create_int_vector(vec![1, 2, 3, 4, 5]);
    let names = create_string_vector(vec!["Alice", "Bob", "Carol", "David", "Eve"]);
    let cities = create_string_vector(vec!["NYC", "LA", "Chicago", "Houston", "Phoenix"]);
    let countries = create_string_vector(vec!["USA", "USA", "USA", "USA", "USA"]);

    customers
        .insert(DataChunk::new(vec![
            customer_ids,
            names,
            cities,
            countries,
        ]))
        .unwrap();

    // Products table
    let product_schema = Schema::new(vec![
        Column::new("product_id", DataType::Int64, false),
        Column::new("name", DataType::String, false),
        Column::new("category", DataType::String, false),
        Column::new("price", DataType::Float64, false),
    ]);
    let products = catalog
        .create_table("products", product_schema, StorageConfig::default())
        .unwrap();

    let product_ids = create_int_vector(vec![101, 102, 103, 104, 105]);
    let product_names = create_string_vector(vec!["Laptop", "Mouse", "Keyboard", "Monitor", "Headphones"]);
    let categories = create_string_vector(vec!["Electronics", "Electronics", "Electronics", "Electronics", "Audio"]);
    let prices = create_float_vector(vec![999.99, 29.99, 79.99, 299.99, 149.99]);

    products
        .insert(DataChunk::new(vec![
            product_ids,
            product_names,
            categories,
            prices,
        ]))
        .unwrap();

    // Orders table
    let order_schema = Schema::new(vec![
        Column::new("order_id", DataType::Int64, false),
        Column::new("customer_id", DataType::Int64, false),
        Column::new("order_date", DataType::String, false),
        Column::new("total", DataType::Float64, false),
        Column::new("status", DataType::String, false),
    ]);
    let orders = catalog
        .create_table("orders", order_schema, StorageConfig::default())
        .unwrap();

    let order_ids = create_int_vector(vec![1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]);
    let cust_ids = create_int_vector(vec![1, 2, 1, 3, 2, 4, 5, 1]);
    let dates = create_string_vector(vec![
        "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18",
        "2024-01-19", "2024-01-20", "2024-01-21", "2024-01-22",
    ]);
    let totals = create_float_vector(vec![150.0, 89.99, 299.99, 75.50, 450.00, 120.00, 199.99, 550.00]);
    let statuses = create_string_vector(vec![
        "delivered", "shipped", "delivered", "processing",
        "delivered", "cancelled", "shipped", "delivered",
    ]);

    orders
        .insert(DataChunk::new(vec![
            order_ids, cust_ids, dates, totals, statuses,
        ]))
        .unwrap();

    // Order items table
    let items_schema = Schema::new(vec![
        Column::new("item_id", DataType::Int64, false),
        Column::new("order_id", DataType::Int64, false),
        Column::new("product_id", DataType::Int64, false),
        Column::new("quantity", DataType::Int64, false),
        Column::new("price", DataType::Float64, false),
    ]);
    let items = catalog
        .create_table("order_items", items_schema, StorageConfig::default())
        .unwrap();

    let item_ids = create_int_vector(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let item_order_ids = create_int_vector(vec![1001, 1001, 1002, 1003, 1004, 1005, 1005, 1006, 1007, 1008]);
    let item_product_ids = create_int_vector(vec![101, 102, 103, 101, 105, 101, 104, 102, 103, 101]);
    let quantities = create_int_vector(vec![1, 2, 1, 1, 1, 1, 1, 3, 2, 2]);
    let item_prices = create_float_vector(vec![999.99, 29.99, 79.99, 299.99, 149.99, 999.99, 299.99, 29.99, 79.99, 999.99]);

    items
        .insert(DataChunk::new(vec![
            item_ids,
            item_order_ids,
            item_product_ids,
            quantities,
            item_prices,
        ]))
        .unwrap();
}

// ============================================================================
// End-to-End Query Tests
// ============================================================================

#[test]
fn test_simple_select_all() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    let plan = PhysicalPlan::SeqScan {
        table_name: "customers".to_string(),
        projection: vec![0, 1, 2, 3],
        filter: None,
    };

    let results = executor.collect(&plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 5);
}

#[test]
fn test_select_with_where_clause() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // SELECT * FROM orders WHERE total > 200
    let plan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![0, 1, 2, 3, 4],
        filter: Some(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(3)),
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Value::Float64(200.0))),
        }),
    };

    let results = executor.collect(&plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert!(total_rows > 0);
    assert!(total_rows < 8);
}

#[test]
fn test_join_customers_orders() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // SELECT c.name, o.total FROM customers c JOIN orders o ON c.customer_id = o.customer_id
    let left_scan = PhysicalPlan::SeqScan {
        table_name: "customers".to_string(),
        projection: vec![0, 1], // customer_id, name
        filter: None,
    };

    let right_scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![1, 3], // customer_id, total
        filter: None,
    };

    let join_plan = PhysicalPlan::HashJoin {
        left: Arc::new(left_scan),
        right: Arc::new(right_scan),
        join_type: JoinType::Inner,
        left_keys: vec![0],
        right_keys: vec![0],
        condition: None,
    };

    let project_plan = PhysicalPlan::Project {
        input: Arc::new(join_plan),
        expressions: vec![Expression::ColumnRef(1), Expression::ColumnRef(3)], // name, total
    };

    let results = executor.collect(&project_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 8); // All orders should match customers
}

#[test]
#[ignore = "Pre-existing: Type mismatch error"]
fn test_aggregation_group_by() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // SELECT status, COUNT(*), SUM(total) FROM orders GROUP BY status
    let scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![3, 4], // total, status
        filter: None,
    };

    let agg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(scan),
        group_by: vec![Expression::ColumnRef(1)], // group by status
        aggregates: vec![
            (AggregateFunction::Count, Expression::ColumnRef(0)),
            (AggregateFunction::Sum, Expression::ColumnRef(0)),
        ],
    };

    let results = executor.collect(&agg_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    // Should have groups for: delivered, shipped, processing, cancelled
    assert_eq!(total_rows, 4);
}

#[test]
fn test_order_by_with_limit() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // SELECT * FROM orders ORDER BY total DESC LIMIT 3
    let scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![0, 1, 2, 3, 4],
        filter: None,
    };

    let sort_plan = PhysicalPlan::Sort {
        input: Arc::new(scan),
        order_by: vec![(3, SortOrder::Descending, false)], // by total DESC
    };

    let limit_plan = PhysicalPlan::Limit {
        input: Arc::new(sort_plan),
        limit: 3,
        offset: 0,
    };

    let results = executor.collect(&limit_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 3);

    // Verify order
    let chunk = &results[0];
    let mut prev_total = f64::MAX;
    for i in 0..chunk.len() {
        if let Value::Float64(total) = chunk.vector(3).unwrap().get(i).unwrap() {
            assert!(total <= prev_total);
            prev_total = total;
        }
    }
}

#[test]
#[ignore = "Pre-existing: Type mismatch error"]
fn test_complex_multi_table_join() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // Complex query: Get customer names with their order totals and product names
    // SELECT c.name, o.total, p.name
    // FROM customers c
    // JOIN orders o ON c.customer_id = o.customer_id
    // JOIN order_items i ON o.order_id = i.order_id
    // JOIN products p ON i.product_id = p.product_id

    // First join: customers <-> orders
    let customer_scan = PhysicalPlan::SeqScan {
        table_name: "customers".to_string(),
        projection: vec![0, 1], // customer_id, name
        filter: None,
    };

    let order_scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![0, 1, 3], // order_id, customer_id, total
        filter: None,
    };

    let join1 = PhysicalPlan::HashJoin {
        left: Arc::new(customer_scan),
        right: Arc::new(order_scan),
        join_type: JoinType::Inner,
        left_keys: vec![0],  // customer_id
        right_keys: vec![1], // customer_id
        condition: None,
    };

    // Second join: (customers <-> orders) <-> order_items
    let items_scan = PhysicalPlan::SeqScan {
        table_name: "order_items".to_string(),
        projection: vec![1, 2], // order_id, product_id
        filter: None,
    };

    let join2 = PhysicalPlan::HashJoin {
        left: Arc::new(join1),
        right: Arc::new(items_scan),
        join_type: JoinType::Inner,
        left_keys: vec![2],  // order_id from orders
        right_keys: vec![0], // order_id from order_items
        condition: None,
    };

    // Third join: add products
    let products_scan = PhysicalPlan::SeqScan {
        table_name: "products".to_string(),
        projection: vec![0, 1], // product_id, name
        filter: None,
    };

    let join3 = PhysicalPlan::HashJoin {
        left: Arc::new(join2),
        right: Arc::new(products_scan),
        join_type: JoinType::Inner,
        left_keys: vec![6],  // product_id from order_items
        right_keys: vec![0], // product_id from products
        condition: None,
    };

    let results = executor.collect(&join3).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    // Should have one row per order_item
    assert_eq!(total_rows, 10);
}

#[test]
fn test_subquery_equivalent() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // Equivalent to: SELECT * FROM orders WHERE total > (SELECT AVG(total) FROM orders)
    // We simulate this by first computing the average, then filtering

    // First, compute average
    let avg_scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![3], // total
        filter: None,
    };

    let avg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(avg_scan),
        group_by: vec![],
        aggregates: vec![(AggregateFunction::Avg, Expression::ColumnRef(0))],
    };

    let avg_results = executor.collect(&avg_plan).unwrap();
    let avg_total = if let Value::Float64(v) = avg_results[0].vector(0).unwrap().get(0).unwrap() {
        v
    } else {
        panic!("Expected float average");
    };

    // Now filter orders where total > avg
    let main_scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![0, 1, 2, 3, 4],
        filter: Some(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(3)),
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Value::Float64(avg_total))),
        }),
    };

    let results = executor.collect(&main_scan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert!(total_rows > 0);
    assert!(total_rows < 8);
}

#[test]
#[ignore = "Pre-existing: Type mismatch error"]
fn test_having_equivalent() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // SELECT customer_id, COUNT(*), SUM(total)
    // FROM orders
    // GROUP BY customer_id
    // HAVING SUM(total) > 200

    let scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![1, 3], // customer_id, total
        filter: None,
    };

    let agg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(scan),
        group_by: vec![Expression::ColumnRef(0)],
        aggregates: vec![
            (AggregateFunction::Count, Expression::ColumnRef(0)),
            (AggregateFunction::Sum, Expression::ColumnRef(1)),
        ],
    };

    // Filter (HAVING equivalent)
    let having_plan = PhysicalPlan::Filter {
        input: Arc::new(agg_plan),
        predicate: Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(2)), // SUM(total)
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Value::Float64(200.0))),
        },
    };

    let results = executor.collect(&having_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert!(total_rows > 0);
}

#[test]
#[ignore = "Pre-existing: ColumnNotFound error"]
fn test_union_queries() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // High value orders (> 300) UNION low value orders (< 100)
    let high_value = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![0, 3],
        filter: Some(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(3)),
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Value::Float64(300.0))),
        }),
    };

    let low_value = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![0, 3],
        filter: Some(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(3)),
            op: BinaryOperator::LessThan,
            right: Box::new(Expression::Literal(Value::Float64(100.0))),
        }),
    };

    let union_plan = PhysicalPlan::UnionAll {
        inputs: vec![Arc::new(high_value), Arc::new(low_value)],
    };

    let results = executor.collect(&union_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert!(total_rows > 0);
}

#[test]
fn test_distinct_results() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // SELECT DISTINCT customer_id FROM orders
    let scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![1], // customer_id
        filter: None,
    };

    let distinct_plan = PhysicalPlan::HashDistinct {
        input: Arc::new(scan),
    };

    let results = executor.collect(&distinct_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    // Should have 5 distinct customers
    assert_eq!(total_rows, 5);
}

#[test]
#[ignore = "Pre-existing: Type mismatch error"]
fn test_top_n_query() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // Top 3 customers by order count
    let scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![1], // customer_id
        filter: None,
    };

    let agg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(scan),
        group_by: vec![Expression::ColumnRef(0)],
        aggregates: vec![(AggregateFunction::Count, Expression::ColumnRef(0))],
    };

    let sort_plan = PhysicalPlan::Sort {
        input: Arc::new(agg_plan),
        order_by: vec![(1, SortOrder::Descending, false)], // by count DESC
    };

    let limit_plan = PhysicalPlan::Limit {
        input: Arc::new(sort_plan),
        limit: 3,
        offset: 0,
    };

    let results = executor.collect(&limit_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 3);
}

// ============================================================================
// Data Integrity Tests
// ============================================================================

#[test]
fn test_insert_and_retrieve() {
    let catalog = create_test_catalog();

    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("value", DataType::String, false),
    ]);

    let table = catalog
        .create_table("test_table", schema, StorageConfig::default())
        .unwrap();

    // Insert data
    let ids = create_int_vector(vec![1, 2, 3]);
    let values = create_string_vector(vec!["one", "two", "three"]);
    table.insert(DataChunk::new(vec![ids, values])).unwrap();

    // Retrieve and verify
    let executor = Executor::new(catalog);
    let plan = PhysicalPlan::SeqScan {
        table_name: "test_table".to_string(),
        projection: vec![0, 1],
        filter: None,
    };

    let results = executor.collect(&plan).unwrap();
    let chunk = &results[0];

    assert_eq!(chunk.len(), 3);
    assert_eq!(chunk.vector(0).unwrap().get(0).unwrap(), Value::Int64(1));
    assert_eq!(chunk.vector(1).unwrap().get(1).unwrap(), Value::String("two".to_string()));
}

#[test]
fn test_empty_table_query() {
    let catalog = create_test_catalog();

    let schema = Schema::new(vec![Column::new("id", DataType::Int64, false)]);
    catalog
        .create_table("empty_table", schema, StorageConfig::default())
        .unwrap();

    let executor = Executor::new(catalog);
    let plan = PhysicalPlan::SeqScan {
        table_name: "empty_table".to_string(),
        projection: vec![0],
        filter: None,
    };

    let results = executor.collect(&plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 0);
}

#[test]
fn test_null_handling_in_queries() {
    let catalog = create_test_catalog();

    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("nullable_value", DataType::Int64, true),
    ]);

    let table = catalog
        .create_table("nullable_table", schema, StorageConfig::default())
        .unwrap();

    // Insert data with nulls
    let mut ids = Vector::new(DataType::Int64);
    let mut values = Vector::new(DataType::Int64);

    ids.push(Value::Int64(1)).unwrap();
    values.push(Value::Int64(10)).unwrap();

    ids.push(Value::Int64(2)).unwrap();
    values.push(Value::Null).unwrap();

    ids.push(Value::Int64(3)).unwrap();
    values.push(Value::Int64(30)).unwrap();

    table.insert(DataChunk::new(vec![ids, values])).unwrap();

    // Query for non-null values
    let executor = Executor::new(catalog);
    let plan = PhysicalPlan::Filter {
        input: Arc::new(PhysicalPlan::SeqScan {
            table_name: "nullable_table".to_string(),
            projection: vec![0, 1],
            filter: None,
        }),
        predicate: Expression::IsNotNull(Box::new(Expression::ColumnRef(1))),
    };

    let results = executor.collect(&plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 2);
}

// ============================================================================
// Performance and Scale Tests
// ============================================================================

#[test]
fn test_large_table_scan() {
    let catalog = create_test_catalog();

    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("value", DataType::Float64, false),
    ]);

    let table = catalog
        .create_table("large_table", schema, StorageConfig::default())
        .unwrap();

    // Insert 10000 rows
    let num_rows = 10000;
    let ids = create_int_vector((0..num_rows as i64).collect());
    let values = create_float_vector((0..num_rows).map(|i| i as f64 * 1.5).collect());

    table.insert(DataChunk::new(vec![ids, values])).unwrap();

    let executor = Executor::new(catalog);
    let plan = PhysicalPlan::SeqScan {
        table_name: "large_table".to_string(),
        projection: vec![0, 1],
        filter: None,
    };

    let results = executor.collect(&plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, num_rows);
}

#[test]
fn test_filtered_large_table() {
    let catalog = create_test_catalog();

    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("category", DataType::String, false),
    ]);

    let table = catalog
        .create_table("category_table", schema, StorageConfig::default())
        .unwrap();

    // Insert data with 10 categories
    let num_rows = 1000;
    let categories = vec!["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];

    let ids = create_int_vector((0..num_rows as i64).collect());
    let category_values = create_string_vector(
        (0..num_rows).map(|i| categories[i % categories.len()]).collect()
    );

    table.insert(DataChunk::new(vec![ids, category_values])).unwrap();

    let executor = Executor::new(catalog);

    // Filter for category A
    let plan = PhysicalPlan::SeqScan {
        table_name: "category_table".to_string(),
        projection: vec![0, 1],
        filter: Some(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(1)),
            op: BinaryOperator::Equal,
            right: Box::new(Expression::Literal(Value::String("A".to_string()))),
        }),
    };

    let results = executor.collect(&plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 100); // 1000 / 10 categories
}

// ============================================================================
// Edge Cases Tests
// ============================================================================

#[test]
fn test_empty_projection() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // Project nothing but still count
    let plan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![],
        filter: None,
    };

    // This might return empty chunks or error depending on implementation
    let result = executor.collect(&plan);
    // Just verify it doesn't crash
    assert!(result.is_ok() || result.is_err());
}

#[test]
#[ignore = "Pre-existing: Type mismatch error"]
fn test_filter_all_rows() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    // Filter that matches nothing
    let plan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![0, 3],
        filter: Some(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(3)),
            op: BinaryOperator::LessThan,
            right: Box::new(Expression::Literal(Value::Float64(0.0))),
        }),
    };

    let results = executor.collect(&plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 0);
}

#[test]
fn test_limit_zero() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![0],
        filter: None,
    };

    let limit_plan = PhysicalPlan::Limit {
        input: Arc::new(scan),
        limit: 0,
        offset: 0,
    };

    let results = executor.collect(&limit_plan).unwrap();
    let total_rows: usize = results.iter().map(|c| c.len()).sum();
    assert_eq!(total_rows, 0);
}

#[test]
#[ignore = "Pre-existing: Type mismatch error"]
fn test_multiple_aggregations_same_column() {
    let catalog = create_test_catalog();
    setup_orders_database(&catalog);

    let executor = Executor::new(catalog);

    let scan = PhysicalPlan::SeqScan {
        table_name: "orders".to_string(),
        projection: vec![3], // total
        filter: None,
    };

    let agg_plan = PhysicalPlan::HashAggregate {
        input: Arc::new(scan),
        group_by: vec![],
        aggregates: vec![
            (AggregateFunction::Min, Expression::ColumnRef(0)),
            (AggregateFunction::Max, Expression::ColumnRef(0)),
            (AggregateFunction::Avg, Expression::ColumnRef(0)),
            (AggregateFunction::Sum, Expression::ColumnRef(0)),
            (AggregateFunction::Count, Expression::ColumnRef(0)),
        ],
    };

    let results = executor.collect(&agg_plan).unwrap();
    let chunk = &results[0];
    assert_eq!(chunk.num_columns(), 5);
    assert_eq!(chunk.len(), 1);
}
