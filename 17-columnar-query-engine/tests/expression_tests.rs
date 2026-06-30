//! Tests for expression types and evaluation.

use columnar_query_engine::expression::*;
use columnar_query_engine::types::*;
use columnar_query_engine::vector::{DataChunk, Vector};

// ============================================================================
// Helper Functions
// ============================================================================

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

fn create_test_chunk() -> DataChunk {
    // Create a test chunk with 5 columns and 5 rows
    let col_a = create_int_vector(vec![1, 2, 3, 4, 5]);
    let col_b = create_int_vector(vec![10, 20, 30, 40, 50]);
    let col_c = create_float_vector(vec![1.5, 2.5, 3.5, 4.5, 5.5]);
    let col_d = create_string_vector(vec!["apple", "banana", "cherry", "date", "elderberry"]);
    let col_e = create_bool_vector(vec![true, false, true, false, true]);

    DataChunk::new(vec![col_a, col_b, col_c, col_d, col_e])
}

// ============================================================================
// Column Reference Tests
// ============================================================================

#[test]
fn test_column_ref_evaluation() {
    let chunk = create_test_chunk();

    let expr = Expression::ColumnRef(0);
    let result = expr.evaluate(&chunk).unwrap();

    assert_eq!(result.len(), 5);
    assert_eq!(result.get(0).unwrap(), Value::Int64(1));
    assert_eq!(result.get(4).unwrap(), Value::Int64(5));
}

#[test]
fn test_column_ref_string() {
    let chunk = create_test_chunk();

    let expr = Expression::ColumnRef(3);
    let result = expr.evaluate(&chunk).unwrap();

    assert_eq!(result.get(0).unwrap(), Value::String("apple".to_string()));
    assert_eq!(
        result.get(2).unwrap(),
        Value::String("cherry".to_string())
    );
}

#[test]
fn test_column_ref_out_of_bounds() {
    let chunk = create_test_chunk();

    let expr = Expression::ColumnRef(100);
    let result = expr.evaluate(&chunk);

    assert!(result.is_err());
}

// ============================================================================
// Literal Expression Tests
// ============================================================================

#[test]
fn test_literal_int() {
    let chunk = create_test_chunk();

    let expr = Expression::Literal(Value::Int64(42));
    let result = expr.evaluate(&chunk).unwrap();

    // Literal should be replicated for all rows
    assert_eq!(result.len(), 5);
    for i in 0..5 {
        assert_eq!(result.get(i).unwrap(), Value::Int64(42));
    }
}

#[test]
fn test_literal_float() {
    let chunk = create_test_chunk();

    let expr = Expression::Literal(Value::Float64(3.14));
    let result = expr.evaluate(&chunk).unwrap();

    assert_eq!(result.len(), 5);
    for i in 0..5 {
        assert_eq!(result.get(i).unwrap(), Value::Float64(3.14));
    }
}

#[test]
fn test_literal_string() {
    let chunk = create_test_chunk();

    let expr = Expression::Literal(Value::String("hello".to_string()));
    let result = expr.evaluate(&chunk).unwrap();

    for i in 0..5 {
        assert_eq!(
            result.get(i).unwrap(),
            Value::String("hello".to_string())
        );
    }
}

#[test]
fn test_literal_bool() {
    let chunk = create_test_chunk();

    let expr = Expression::Literal(Value::Boolean(true));
    let result = expr.evaluate(&chunk).unwrap();

    for i in 0..5 {
        assert_eq!(result.get(i).unwrap(), Value::Boolean(true));
    }
}

#[test]
fn test_literal_null() {
    let chunk = create_test_chunk();

    let expr = Expression::Literal(Value::Null);
    let result = expr.evaluate(&chunk).unwrap();

    for i in 0..5 {
        assert!(result.is_null(i));
    }
}

// ============================================================================
// Arithmetic Binary Operations Tests
// ============================================================================

#[test]
fn test_binary_add() {
    let chunk = create_test_chunk();

    // col_a + col_b (1+10, 2+20, ...)
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::Add,
        right: Box::new(Expression::ColumnRef(1)),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Float64(11.0));
    assert_eq!(result.get(1).unwrap(), Value::Float64(22.0));
    assert_eq!(result.get(4).unwrap(), Value::Float64(55.0));
}

#[test]
fn test_binary_subtract() {
    let chunk = create_test_chunk();

    // col_b - col_a (10-1, 20-2, ...)
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(1)),
        op: BinaryOperator::Subtract,
        right: Box::new(Expression::ColumnRef(0)),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Float64(9.0));
    assert_eq!(result.get(1).unwrap(), Value::Float64(18.0));
}

#[test]
fn test_binary_multiply() {
    let chunk = create_test_chunk();

    // col_a * 2
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::Multiply,
        right: Box::new(Expression::Literal(Value::Int64(2))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Float64(2.0));
    assert_eq!(result.get(2).unwrap(), Value::Float64(6.0));
}

#[test]
fn test_binary_divide() {
    let chunk = create_test_chunk();

    // col_b / col_a (10/1, 20/2, ...)
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(1)),
        op: BinaryOperator::Divide,
        right: Box::new(Expression::ColumnRef(0)),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Float64(10.0));
    assert_eq!(result.get(1).unwrap(), Value::Float64(10.0));
}

#[test]
fn test_binary_divide_by_zero() {
    let chunk = create_test_chunk();

    // col_a / 0
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::Divide,
        right: Box::new(Expression::Literal(Value::Int64(0))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    // Division by zero should produce NaN
    if let Value::Float64(v) = result.get(0).unwrap() {
        assert!(v.is_nan() || v.is_infinite());
    }
}

// ============================================================================
// Comparison Binary Operations Tests
// ============================================================================

#[test]
fn test_binary_equal() {
    let chunk = create_test_chunk();

    // col_a = 3
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::Equal,
        right: Box::new(Expression::Literal(Value::Int64(3))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false)); // 1 = 3
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true)); // 3 = 3
}

#[test]
fn test_binary_not_equal() {
    let chunk = create_test_chunk();

    // col_a != 3
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::NotEqual,
        right: Box::new(Expression::Literal(Value::Int64(3))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(true)); // 1 != 3
    assert_eq!(result.get(2).unwrap(), Value::Boolean(false)); // 3 != 3
}

#[test]
fn test_binary_less_than() {
    let chunk = create_test_chunk();

    // col_a < 3
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::LessThan,
        right: Box::new(Expression::Literal(Value::Int64(3))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(true)); // 1 < 3
    assert_eq!(result.get(1).unwrap(), Value::Boolean(true)); // 2 < 3
    assert_eq!(result.get(2).unwrap(), Value::Boolean(false)); // 3 < 3
    assert_eq!(result.get(3).unwrap(), Value::Boolean(false)); // 4 < 3
}

#[test]
fn test_binary_less_than_or_equal() {
    let chunk = create_test_chunk();

    // col_a <= 3
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::LessThanOrEqual,
        right: Box::new(Expression::Literal(Value::Int64(3))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(true)); // 1 <= 3
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true)); // 3 <= 3
    assert_eq!(result.get(3).unwrap(), Value::Boolean(false)); // 4 <= 3
}

#[test]
fn test_binary_greater_than() {
    let chunk = create_test_chunk();

    // col_a > 3
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::GreaterThan,
        right: Box::new(Expression::Literal(Value::Int64(3))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(2).unwrap(), Value::Boolean(false)); // 3 > 3
    assert_eq!(result.get(3).unwrap(), Value::Boolean(true)); // 4 > 3
    assert_eq!(result.get(4).unwrap(), Value::Boolean(true)); // 5 > 3
}

#[test]
fn test_binary_greater_than_or_equal() {
    let chunk = create_test_chunk();

    // col_a >= 3
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::GreaterThanOrEqual,
        right: Box::new(Expression::Literal(Value::Int64(3))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(1).unwrap(), Value::Boolean(false)); // 2 >= 3
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true)); // 3 >= 3
    assert_eq!(result.get(3).unwrap(), Value::Boolean(true)); // 4 >= 3
}

#[test]
fn test_string_comparison() {
    let chunk = create_test_chunk();

    // col_d = "cherry"
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(3)),
        op: BinaryOperator::Equal,
        right: Box::new(Expression::Literal(Value::String("cherry".to_string()))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false));
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true));
}

// ============================================================================
// Logical Binary Operations Tests
// ============================================================================

#[test]
fn test_binary_and() {
    let chunk = create_test_chunk();

    // (col_a > 2) AND (col_a < 5)
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(0)),
            op: BinaryOperator::GreaterThan,
            right: Box::new(Expression::Literal(Value::Int64(2))),
        }),
        op: BinaryOperator::And,
        right: Box::new(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(0)),
            op: BinaryOperator::LessThan,
            right: Box::new(Expression::Literal(Value::Int64(5))),
        }),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false)); // 1
    assert_eq!(result.get(1).unwrap(), Value::Boolean(false)); // 2
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true)); // 3
    assert_eq!(result.get(3).unwrap(), Value::Boolean(true)); // 4
    assert_eq!(result.get(4).unwrap(), Value::Boolean(false)); // 5
}

#[test]
fn test_binary_or() {
    let chunk = create_test_chunk();

    // (col_a = 1) OR (col_a = 5)
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(0)),
            op: BinaryOperator::Equal,
            right: Box::new(Expression::Literal(Value::Int64(1))),
        }),
        op: BinaryOperator::Or,
        right: Box::new(Expression::BinaryOp {
            left: Box::new(Expression::ColumnRef(0)),
            op: BinaryOperator::Equal,
            right: Box::new(Expression::Literal(Value::Int64(5))),
        }),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(true)); // 1
    assert_eq!(result.get(1).unwrap(), Value::Boolean(false)); // 2
    assert_eq!(result.get(4).unwrap(), Value::Boolean(true)); // 5
}

#[test]
fn test_binary_concat() {
    let mut col_a = Vector::new(DataType::String);
    col_a.push(Value::String("Hello".to_string())).unwrap();
    col_a.push(Value::String("Good".to_string())).unwrap();

    let mut col_b = Vector::new(DataType::String);
    col_b.push(Value::String(" World".to_string())).unwrap();
    col_b.push(Value::String("bye".to_string())).unwrap();

    let chunk = DataChunk::new(vec![col_a, col_b]);

    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::Concat,
        right: Box::new(Expression::ColumnRef(1)),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(
        result.get(0).unwrap(),
        Value::String("Hello World".to_string())
    );
    assert_eq!(
        result.get(1).unwrap(),
        Value::String("Goodbye".to_string())
    );
}

// ============================================================================
// Unary Operations Tests
// ============================================================================

#[test]
fn test_unary_not() {
    let chunk = create_test_chunk();

    // NOT col_e
    let expr = Expression::UnaryOp {
        op: UnaryOperator::Not,
        expr: Box::new(Expression::ColumnRef(4)),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false)); // NOT true
    assert_eq!(result.get(1).unwrap(), Value::Boolean(true)); // NOT false
    assert_eq!(result.get(2).unwrap(), Value::Boolean(false)); // NOT true
}

#[test]
fn test_unary_negate() {
    let chunk = create_test_chunk();

    // -col_a
    let expr = Expression::UnaryOp {
        op: UnaryOperator::Negate,
        expr: Box::new(Expression::ColumnRef(0)),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Float64(-1.0));
    assert_eq!(result.get(2).unwrap(), Value::Float64(-3.0));
}

#[test]
fn test_unary_bitwise_not() {
    let chunk = create_test_chunk();

    // ~col_a
    let expr = Expression::UnaryOp {
        op: UnaryOperator::BitwiseNot,
        expr: Box::new(Expression::ColumnRef(0)),
    };

    let result = expr.evaluate(&chunk).unwrap();
    // ~1 = -2, ~2 = -3, etc.
    assert_eq!(result.get(0).unwrap(), Value::Int64(-2));
    assert_eq!(result.get(1).unwrap(), Value::Int64(-3));
}

// ============================================================================
// IS NULL / IS NOT NULL Tests
// ============================================================================

#[test]
fn test_is_null() {
    let mut col = Vector::new(DataType::Int64);
    col.push(Value::Int64(1)).unwrap();
    col.push(Value::Null).unwrap();
    col.push(Value::Int64(3)).unwrap();
    col.push(Value::Null).unwrap();

    let chunk = DataChunk::new(vec![col]);

    let expr = Expression::IsNull(Box::new(Expression::ColumnRef(0)));
    let result = expr.evaluate(&chunk).unwrap();

    assert_eq!(result.get(0).unwrap(), Value::Boolean(false));
    assert_eq!(result.get(1).unwrap(), Value::Boolean(true));
    assert_eq!(result.get(2).unwrap(), Value::Boolean(false));
    assert_eq!(result.get(3).unwrap(), Value::Boolean(true));
}

#[test]
fn test_is_not_null() {
    let mut col = Vector::new(DataType::Int64);
    col.push(Value::Int64(1)).unwrap();
    col.push(Value::Null).unwrap();
    col.push(Value::Int64(3)).unwrap();

    let chunk = DataChunk::new(vec![col]);

    let expr = Expression::IsNotNull(Box::new(Expression::ColumnRef(0)));
    let result = expr.evaluate(&chunk).unwrap();

    assert_eq!(result.get(0).unwrap(), Value::Boolean(true));
    assert_eq!(result.get(1).unwrap(), Value::Boolean(false));
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true));
}

// ============================================================================
// CAST Tests
// ============================================================================

#[test]
fn test_cast_int_to_float() {
    let chunk = create_test_chunk();

    let expr = Expression::Cast {
        expr: Box::new(Expression::ColumnRef(0)),
        target_type: DataType::Float64,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Float64(1.0));
    assert_eq!(result.get(2).unwrap(), Value::Float64(3.0));
}

#[test]
fn test_cast_int_to_string() {
    let chunk = create_test_chunk();

    let expr = Expression::Cast {
        expr: Box::new(Expression::ColumnRef(0)),
        target_type: DataType::String,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::String("1".to_string()));
    assert_eq!(result.get(2).unwrap(), Value::String("3".to_string()));
}

#[test]
fn test_cast_bool_to_int() {
    let chunk = create_test_chunk();

    let expr = Expression::Cast {
        expr: Box::new(Expression::ColumnRef(4)),
        target_type: DataType::Int64,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Int64(1)); // true -> 1
    assert_eq!(result.get(1).unwrap(), Value::Int64(0)); // false -> 0
}

// ============================================================================
// IN List Tests
// ============================================================================

#[test]
fn test_in_list() {
    let chunk = create_test_chunk();

    // col_a IN (2, 4)
    let expr = Expression::InList {
        expr: Box::new(Expression::ColumnRef(0)),
        list: vec![
            Expression::Literal(Value::Int64(2)),
            Expression::Literal(Value::Int64(4)),
        ],
        negated: false,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false)); // 1
    assert_eq!(result.get(1).unwrap(), Value::Boolean(true)); // 2
    assert_eq!(result.get(2).unwrap(), Value::Boolean(false)); // 3
    assert_eq!(result.get(3).unwrap(), Value::Boolean(true)); // 4
    assert_eq!(result.get(4).unwrap(), Value::Boolean(false)); // 5
}

#[test]
fn test_not_in_list() {
    let chunk = create_test_chunk();

    // col_a NOT IN (2, 4)
    let expr = Expression::InList {
        expr: Box::new(Expression::ColumnRef(0)),
        list: vec![
            Expression::Literal(Value::Int64(2)),
            Expression::Literal(Value::Int64(4)),
        ],
        negated: true,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(true)); // 1
    assert_eq!(result.get(1).unwrap(), Value::Boolean(false)); // 2
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true)); // 3
    assert_eq!(result.get(3).unwrap(), Value::Boolean(false)); // 4
}

// ============================================================================
// BETWEEN Tests
// ============================================================================

#[test]
fn test_between() {
    let chunk = create_test_chunk();

    // col_a BETWEEN 2 AND 4
    let expr = Expression::Between {
        expr: Box::new(Expression::ColumnRef(0)),
        low: Box::new(Expression::Literal(Value::Int64(2))),
        high: Box::new(Expression::Literal(Value::Int64(4))),
        negated: false,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false)); // 1
    assert_eq!(result.get(1).unwrap(), Value::Boolean(true)); // 2
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true)); // 3
    assert_eq!(result.get(3).unwrap(), Value::Boolean(true)); // 4
    assert_eq!(result.get(4).unwrap(), Value::Boolean(false)); // 5
}

#[test]
fn test_not_between() {
    let chunk = create_test_chunk();

    // col_a NOT BETWEEN 2 AND 4
    let expr = Expression::Between {
        expr: Box::new(Expression::ColumnRef(0)),
        low: Box::new(Expression::Literal(Value::Int64(2))),
        high: Box::new(Expression::Literal(Value::Int64(4))),
        negated: true,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(true)); // 1
    assert_eq!(result.get(1).unwrap(), Value::Boolean(false)); // 2
    assert_eq!(result.get(4).unwrap(), Value::Boolean(true)); // 5
}

// ============================================================================
// LIKE Tests
// ============================================================================

#[test]
fn test_like_prefix() {
    let chunk = create_test_chunk();

    // col_d LIKE 'a%'
    let expr = Expression::Like {
        expr: Box::new(Expression::ColumnRef(3)),
        pattern: "a%".to_string(),
        negated: false,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(true)); // apple
    assert_eq!(result.get(1).unwrap(), Value::Boolean(false)); // banana
}

#[test]
fn test_like_suffix() {
    let chunk = create_test_chunk();

    // col_d LIKE '%ry'
    let expr = Expression::Like {
        expr: Box::new(Expression::ColumnRef(3)),
        pattern: "%ry".to_string(),
        negated: false,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true)); // cherry
    assert_eq!(result.get(4).unwrap(), Value::Boolean(true)); // elderberry
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false)); // apple
}

#[test]
fn test_like_contains() {
    let chunk = create_test_chunk();

    // col_d LIKE '%an%'
    let expr = Expression::Like {
        expr: Box::new(Expression::ColumnRef(3)),
        pattern: "%an%".to_string(),
        negated: false,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(1).unwrap(), Value::Boolean(true)); // banana
}

#[test]
fn test_like_single_char() {
    let mut col = Vector::new(DataType::String);
    col.push(Value::String("cat".to_string())).unwrap();
    col.push(Value::String("cut".to_string())).unwrap();
    col.push(Value::String("coat".to_string())).unwrap();

    let chunk = DataChunk::new(vec![col]);

    // col LIKE 'c_t'
    let expr = Expression::Like {
        expr: Box::new(Expression::ColumnRef(0)),
        pattern: "c_t".to_string(),
        negated: false,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(true)); // cat
    assert_eq!(result.get(1).unwrap(), Value::Boolean(true)); // cut
    assert_eq!(result.get(2).unwrap(), Value::Boolean(false)); // coat
}

#[test]
fn test_not_like() {
    let chunk = create_test_chunk();

    // col_d NOT LIKE 'a%'
    let expr = Expression::Like {
        expr: Box::new(Expression::ColumnRef(3)),
        pattern: "a%".to_string(),
        negated: true,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false)); // apple
    assert_eq!(result.get(1).unwrap(), Value::Boolean(true)); // banana
}

// ============================================================================
// CASE Expression Tests
// ============================================================================

#[test]
fn test_case_simple() {
    let chunk = create_test_chunk();

    // CASE WHEN col_a > 3 THEN 'high' ELSE 'low' END
    let expr = Expression::Case {
        operand: None,
        when_clauses: vec![(
            Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef(0)),
                op: BinaryOperator::GreaterThan,
                right: Box::new(Expression::Literal(Value::Int64(3))),
            },
            Expression::Literal(Value::String("high".to_string())),
        )],
        else_result: Some(Box::new(Expression::Literal(Value::String(
            "low".to_string(),
        )))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::String("low".to_string())); // 1
    assert_eq!(result.get(3).unwrap(), Value::String("high".to_string())); // 4
    assert_eq!(result.get(4).unwrap(), Value::String("high".to_string())); // 5
}

#[test]
fn test_case_multiple_when() {
    let chunk = create_test_chunk();

    // CASE
    //   WHEN col_a > 4 THEN 'very high'
    //   WHEN col_a > 2 THEN 'high'
    //   ELSE 'low'
    // END
    let expr = Expression::Case {
        operand: None,
        when_clauses: vec![
            (
                Expression::BinaryOp {
                    left: Box::new(Expression::ColumnRef(0)),
                    op: BinaryOperator::GreaterThan,
                    right: Box::new(Expression::Literal(Value::Int64(4))),
                },
                Expression::Literal(Value::String("very high".to_string())),
            ),
            (
                Expression::BinaryOp {
                    left: Box::new(Expression::ColumnRef(0)),
                    op: BinaryOperator::GreaterThan,
                    right: Box::new(Expression::Literal(Value::Int64(2))),
                },
                Expression::Literal(Value::String("high".to_string())),
            ),
        ],
        else_result: Some(Box::new(Expression::Literal(Value::String(
            "low".to_string(),
        )))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::String("low".to_string())); // 1
    assert_eq!(result.get(2).unwrap(), Value::String("high".to_string())); // 3
    assert_eq!(
        result.get(4).unwrap(),
        Value::String("very high".to_string())
    ); // 5
}

#[test]
fn test_case_no_else() {
    let chunk = create_test_chunk();

    // CASE WHEN col_a = 3 THEN 'three' END
    let expr = Expression::Case {
        operand: None,
        when_clauses: vec![(
            Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef(0)),
                op: BinaryOperator::Equal,
                right: Box::new(Expression::Literal(Value::Int64(3))),
            },
            Expression::Literal(Value::String("three".to_string())),
        )],
        else_result: None,
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert!(result.is_null(0)); // 1 -> NULL
    assert_eq!(
        result.get(2).unwrap(),
        Value::String("three".to_string())
    ); // 3
    assert!(result.is_null(4)); // 5 -> NULL
}

// ============================================================================
// Function Tests
// ============================================================================

#[test]
fn test_function_abs() {
    let mut col = Vector::new(DataType::Float64);
    col.push(Value::Float64(-5.0)).unwrap();
    col.push(Value::Float64(3.0)).unwrap();
    col.push(Value::Float64(-2.5)).unwrap();

    let chunk = DataChunk::new(vec![col]);

    let expr = Expression::Function {
        name: "ABS".to_string(),
        args: vec![Expression::ColumnRef(0)],
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Float64(5.0));
    assert_eq!(result.get(1).unwrap(), Value::Float64(3.0));
    assert_eq!(result.get(2).unwrap(), Value::Float64(2.5));
}

#[test]
fn test_function_upper() {
    let chunk = create_test_chunk();

    let expr = Expression::Function {
        name: "UPPER".to_string(),
        args: vec![Expression::ColumnRef(3)],
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::String("APPLE".to_string()));
    assert_eq!(result.get(1).unwrap(), Value::String("BANANA".to_string()));
}

#[test]
fn test_function_lower() {
    let mut col = Vector::new(DataType::String);
    col.push(Value::String("HELLO".to_string())).unwrap();
    col.push(Value::String("World".to_string())).unwrap();

    let chunk = DataChunk::new(vec![col]);

    let expr = Expression::Function {
        name: "LOWER".to_string(),
        args: vec![Expression::ColumnRef(0)],
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::String("hello".to_string()));
    assert_eq!(result.get(1).unwrap(), Value::String("world".to_string()));
}

#[test]
fn test_function_length() {
    let chunk = create_test_chunk();

    let expr = Expression::Function {
        name: "LENGTH".to_string(),
        args: vec![Expression::ColumnRef(3)],
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Int64(5)); // apple
    assert_eq!(result.get(1).unwrap(), Value::Int64(6)); // banana
    assert_eq!(result.get(4).unwrap(), Value::Int64(10)); // elderberry
}

#[test]
fn test_function_coalesce() {
    let mut col1 = Vector::new(DataType::Int64);
    col1.push(Value::Null).unwrap();
    col1.push(Value::Int64(2)).unwrap();
    col1.push(Value::Null).unwrap();

    let mut col2 = Vector::new(DataType::Int64);
    col2.push(Value::Int64(10)).unwrap();
    col2.push(Value::Null).unwrap();
    col2.push(Value::Null).unwrap();

    let mut col3 = Vector::new(DataType::Int64);
    col3.push(Value::Int64(100)).unwrap();
    col3.push(Value::Int64(100)).unwrap();
    col3.push(Value::Int64(100)).unwrap();

    let chunk = DataChunk::new(vec![col1, col2, col3]);

    let expr = Expression::Function {
        name: "COALESCE".to_string(),
        args: vec![
            Expression::ColumnRef(0),
            Expression::ColumnRef(1),
            Expression::ColumnRef(2),
        ],
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Int64(10)); // NULL, 10, 100 -> 10
    assert_eq!(result.get(1).unwrap(), Value::Int64(2)); // 2, NULL, 100 -> 2
    assert_eq!(result.get(2).unwrap(), Value::Int64(100)); // NULL, NULL, 100 -> 100
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

#[test]
fn test_nested_expression() {
    let chunk = create_test_chunk();

    // ((col_a + 1) * 2) > 5
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::BinaryOp {
            left: Box::new(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef(0)),
                op: BinaryOperator::Add,
                right: Box::new(Expression::Literal(Value::Int64(1))),
            }),
            op: BinaryOperator::Multiply,
            right: Box::new(Expression::Literal(Value::Int64(2))),
        }),
        op: BinaryOperator::GreaterThan,
        right: Box::new(Expression::Literal(Value::Float64(5.0))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    // (1+1)*2=4 > 5 = false
    // (2+1)*2=6 > 5 = true
    // (3+1)*2=8 > 5 = true
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false));
    assert_eq!(result.get(1).unwrap(), Value::Boolean(true));
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true));
}

#[test]
fn test_complex_boolean_expression() {
    let chunk = create_test_chunk();

    // (col_a > 2 AND col_a < 4) OR col_e
    let expr = Expression::BinaryOp {
        left: Box::new(Expression::BinaryOp {
            left: Box::new(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef(0)),
                op: BinaryOperator::GreaterThan,
                right: Box::new(Expression::Literal(Value::Int64(2))),
            }),
            op: BinaryOperator::And,
            right: Box::new(Expression::BinaryOp {
                left: Box::new(Expression::ColumnRef(0)),
                op: BinaryOperator::LessThan,
                right: Box::new(Expression::Literal(Value::Int64(4))),
            }),
        }),
        op: BinaryOperator::Or,
        right: Box::new(Expression::ColumnRef(4)),
    };

    let result = expr.evaluate(&chunk).unwrap();
    // Row 0: (1>2 AND 1<4) OR true = false OR true = true
    // Row 1: (2>2 AND 2<4) OR false = false OR false = false
    // Row 2: (3>2 AND 3<4) OR true = true OR true = true
    // Row 3: (4>2 AND 4<4) OR false = false OR false = false
    // Row 4: (5>2 AND 5<4) OR true = false OR true = true
    assert_eq!(result.get(0).unwrap(), Value::Boolean(true));
    assert_eq!(result.get(1).unwrap(), Value::Boolean(false));
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true));
    assert_eq!(result.get(3).unwrap(), Value::Boolean(false));
    assert_eq!(result.get(4).unwrap(), Value::Boolean(true));
}

// ============================================================================
// Null Handling Tests
// ============================================================================

#[test]
fn test_arithmetic_with_nulls() {
    let mut col = Vector::new(DataType::Int64);
    col.push(Value::Int64(1)).unwrap();
    col.push(Value::Null).unwrap();
    col.push(Value::Int64(3)).unwrap();

    let chunk = DataChunk::new(vec![col]);

    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::Add,
        right: Box::new(Expression::Literal(Value::Int64(10))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Float64(11.0));
    assert!(result.is_null(1)); // NULL + 10 = NULL
    assert_eq!(result.get(2).unwrap(), Value::Float64(13.0));
}

#[test]
fn test_comparison_with_nulls() {
    let mut col = Vector::new(DataType::Int64);
    col.push(Value::Int64(1)).unwrap();
    col.push(Value::Null).unwrap();
    col.push(Value::Int64(3)).unwrap();

    let chunk = DataChunk::new(vec![col]);

    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::GreaterThan,
        right: Box::new(Expression::Literal(Value::Int64(2))),
    };

    let result = expr.evaluate(&chunk).unwrap();
    assert_eq!(result.get(0).unwrap(), Value::Boolean(false));
    assert!(result.is_null(1)); // NULL > 2 = NULL
    assert_eq!(result.get(2).unwrap(), Value::Boolean(true));
}

// ============================================================================
// Data Type Inference Tests
// ============================================================================

#[test]
fn test_data_type_column_ref() {
    let input_types = vec![DataType::Int64, DataType::String, DataType::Float64];

    let expr = Expression::ColumnRef(1);
    let result_type = expr.data_type(&input_types).unwrap();

    assert_eq!(result_type, DataType::String);
}

#[test]
fn test_data_type_literal() {
    let expr = Expression::Literal(Value::Float64(3.14));
    let result_type = expr.data_type(&[]).unwrap();

    assert_eq!(result_type, DataType::Float64);
}

#[test]
fn test_data_type_comparison() {
    let input_types = vec![DataType::Int64];

    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::GreaterThan,
        right: Box::new(Expression::Literal(Value::Int64(5))),
    };

    let result_type = expr.data_type(&input_types).unwrap();
    assert_eq!(result_type, DataType::Boolean);
}

#[test]
fn test_data_type_arithmetic() {
    let input_types = vec![DataType::Int64, DataType::Float64];

    let expr = Expression::BinaryOp {
        left: Box::new(Expression::ColumnRef(0)),
        op: BinaryOperator::Add,
        right: Box::new(Expression::ColumnRef(1)),
    };

    let result_type = expr.data_type(&input_types).unwrap();
    assert_eq!(result_type, DataType::Float64);
}

#[test]
fn test_data_type_cast() {
    let input_types = vec![DataType::Int64];

    let expr = Expression::Cast {
        expr: Box::new(Expression::ColumnRef(0)),
        target_type: DataType::String,
    };

    let result_type = expr.data_type(&input_types).unwrap();
    assert_eq!(result_type, DataType::String);
}

#[test]
fn test_data_type_is_null() {
    let input_types = vec![DataType::Int64];

    let expr = Expression::IsNull(Box::new(Expression::ColumnRef(0)));
    let result_type = expr.data_type(&input_types).unwrap();

    assert_eq!(result_type, DataType::Boolean);
}
