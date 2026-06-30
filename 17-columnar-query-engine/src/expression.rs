//! Expression types and evaluation.

use crate::types::{AggregateFunction, DataType, SortOrder, Value};
use crate::vector::{DataChunk, Vector};
use crate::{Error, Result};
use serde::{Deserialize, Serialize};

/// Expression node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Expression {
    /// Column reference by index.
    ColumnRef(usize),
    /// Column reference by name.
    ColumnName(String),
    /// Literal value.
    Literal(Value),
    /// Binary operation.
    BinaryOp {
        left: Box<Expression>,
        op: BinaryOperator,
        right: Box<Expression>,
    },
    /// Unary operation.
    UnaryOp {
        op: UnaryOperator,
        expr: Box<Expression>,
    },
    /// Function call.
    Function {
        name: String,
        args: Vec<Expression>,
    },
    /// Aggregate function.
    Aggregate {
        func: AggregateFunction,
        expr: Box<Expression>,
        distinct: bool,
    },
    /// CASE expression.
    Case {
        operand: Option<Box<Expression>>,
        when_clauses: Vec<(Expression, Expression)>,
        else_result: Option<Box<Expression>>,
    },
    /// CAST expression.
    Cast {
        expr: Box<Expression>,
        target_type: DataType,
    },
    /// IS NULL.
    IsNull(Box<Expression>),
    /// IS NOT NULL.
    IsNotNull(Box<Expression>),
    /// IN list.
    InList {
        expr: Box<Expression>,
        list: Vec<Expression>,
        negated: bool,
    },
    /// BETWEEN.
    Between {
        expr: Box<Expression>,
        low: Box<Expression>,
        high: Box<Expression>,
        negated: bool,
    },
    /// LIKE pattern.
    Like {
        expr: Box<Expression>,
        pattern: String,
        negated: bool,
    },
    /// Sort specification.
    Sort {
        expr: Box<Expression>,
        order: SortOrder,
        nulls_first: bool,
    },
    /// Wildcard (*).
    Wildcard,
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOperator {
    // Arithmetic
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    // Comparison
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    // Logical
    And,
    Or,
    // String
    Concat,
    // Bitwise
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Negate,
    BitwiseNot,
    Plus,
}

impl Expression {
    /// Evaluate expression on a data chunk.
    pub fn evaluate(&self, chunk: &DataChunk) -> Result<Vector> {
        match self {
            Expression::ColumnRef(idx) => {
                chunk
                    .vector(*idx)
                    .cloned()
                    .ok_or_else(|| Error::ColumnNotFound(format!("Index {}", idx)))
            }

            Expression::Literal(value) => {
                Ok(Vector::constant(value.clone(), chunk.len()))
            }

            Expression::BinaryOp { left, op, right } => {
                let left_vec = left.evaluate(chunk)?;
                let right_vec = right.evaluate(chunk)?;
                evaluate_binary_op(&left_vec, *op, &right_vec)
            }

            Expression::UnaryOp { op, expr } => {
                let vec = expr.evaluate(chunk)?;
                evaluate_unary_op(*op, &vec)
            }

            Expression::IsNull(expr) => {
                let vec = expr.evaluate(chunk)?;
                let mut result = Vector::new(DataType::Boolean);
                for i in 0..vec.len() {
                    result.push(Value::Boolean(vec.is_null(i)))?;
                }
                Ok(result)
            }

            Expression::IsNotNull(expr) => {
                let vec = expr.evaluate(chunk)?;
                let mut result = Vector::new(DataType::Boolean);
                for i in 0..vec.len() {
                    result.push(Value::Boolean(!vec.is_null(i)))?;
                }
                Ok(result)
            }

            Expression::Cast { expr, target_type } => {
                let vec = expr.evaluate(chunk)?;
                cast_vector(&vec, target_type)
            }

            Expression::Case {
                operand,
                when_clauses,
                else_result,
            } => {
                evaluate_case(chunk, operand, when_clauses, else_result)
            }

            Expression::InList { expr, list, negated } => {
                let vec = expr.evaluate(chunk)?;
                let list_values: Vec<Value> = list
                    .iter()
                    .map(|e| {
                        if let Expression::Literal(v) = e {
                            Ok(v.clone())
                        } else {
                            Err(Error::InvalidOperation("IN list must contain literals".into()))
                        }
                    })
                    .collect::<Result<Vec<_>>>()?;

                let mut result = Vector::new(DataType::Boolean);
                for i in 0..vec.len() {
                    if vec.is_null(i) {
                        result.push(Value::Null)?;
                    } else {
                        let value = vec.get(i)?;
                        let found = list_values.contains(&value);
                        result.push(Value::Boolean(if *negated { !found } else { found }))?;
                    }
                }
                Ok(result)
            }

            Expression::Between {
                expr,
                low,
                high,
                negated,
            } => {
                let vec = expr.evaluate(chunk)?;
                let low_vec = low.evaluate(chunk)?;
                let high_vec = high.evaluate(chunk)?;

                let mut result = Vector::new(DataType::Boolean);
                for i in 0..vec.len() {
                    if vec.is_null(i) || low_vec.is_null(i) || high_vec.is_null(i) {
                        result.push(Value::Null)?;
                    } else {
                        let v = vec.get(i)?;
                        let l = low_vec.get(i)?;
                        let h = high_vec.get(i)?;
                        let between = v >= l && v <= h;
                        result.push(Value::Boolean(if *negated { !between } else { between }))?;
                    }
                }
                Ok(result)
            }

            Expression::Like { expr, pattern, negated } => {
                let vec = expr.evaluate(chunk)?;
                let regex = like_to_regex(pattern);
                let re = regex::Regex::new(&regex)
                    .map_err(|e| Error::InvalidOperation(e.to_string()))?;

                let mut result = Vector::new(DataType::Boolean);
                for i in 0..vec.len() {
                    if vec.is_null(i) {
                        result.push(Value::Null)?;
                    } else if let Value::String(s) = vec.get(i)? {
                        let matches = re.is_match(&s);
                        result.push(Value::Boolean(if *negated { !matches } else { matches }))?;
                    } else {
                        result.push(Value::Null)?;
                    }
                }
                Ok(result)
            }

            Expression::Function { name, args } => {
                let arg_vecs: Vec<Vector> = args
                    .iter()
                    .map(|a| a.evaluate(chunk))
                    .collect::<Result<Vec<_>>>()?;
                evaluate_function(name, &arg_vecs, chunk.len())
            }

            _ => Err(Error::InvalidOperation(format!(
                "Expression type {:?} cannot be evaluated directly",
                self
            ))),
        }
    }

    /// Get the output data type.
    pub fn data_type(&self, input_types: &[DataType]) -> Result<DataType> {
        match self {
            Expression::ColumnRef(idx) => input_types
                .get(*idx)
                .cloned()
                .ok_or_else(|| Error::ColumnNotFound(format!("Index {}", idx))),

            Expression::Literal(v) => Ok(v.data_type()),

            Expression::BinaryOp { left, op, right } => {
                let left_type = left.data_type(input_types)?;
                let right_type = right.data_type(input_types)?;
                binary_op_result_type(&left_type, *op, &right_type)
            }

            Expression::UnaryOp { op, expr } => {
                let expr_type = expr.data_type(input_types)?;
                unary_op_result_type(*op, &expr_type)
            }

            Expression::IsNull(_) | Expression::IsNotNull(_) => Ok(DataType::Boolean),

            Expression::Cast { target_type, .. } => Ok(target_type.clone()),

            Expression::Aggregate { func, .. } => match func {
                AggregateFunction::Count => Ok(DataType::Int64),
                AggregateFunction::Sum | AggregateFunction::Avg => Ok(DataType::Float64),
                _ => Ok(DataType::Float64),
            },

            _ => Ok(DataType::String),
        }
    }
}

/// Evaluate binary operation.
fn evaluate_binary_op(left: &Vector, op: BinaryOperator, right: &Vector) -> Result<Vector> {
    let len = left.len().max(right.len());

    match op {
        // Arithmetic operations
        BinaryOperator::Add | BinaryOperator::Subtract | BinaryOperator::Multiply | BinaryOperator::Divide => {
            let mut result = Vector::new(DataType::Float64);
            for i in 0..len {
                let l = left.get(i.min(left.len() - 1))?;
                let r = right.get(i.min(right.len() - 1))?;

                if l.is_null() || r.is_null() {
                    result.push(Value::Null)?;
                } else {
                    let lv = l.as_f64().unwrap_or(0.0);
                    let rv = r.as_f64().unwrap_or(0.0);
                    let res = match op {
                        BinaryOperator::Add => lv + rv,
                        BinaryOperator::Subtract => lv - rv,
                        BinaryOperator::Multiply => lv * rv,
                        BinaryOperator::Divide => {
                            if rv == 0.0 {
                                f64::NAN
                            } else {
                                lv / rv
                            }
                        }
                        _ => unreachable!(),
                    };
                    result.push(Value::Float64(res))?;
                }
            }
            Ok(result)
        }

        // Comparison operations
        BinaryOperator::Equal | BinaryOperator::NotEqual |
        BinaryOperator::LessThan | BinaryOperator::LessThanOrEqual |
        BinaryOperator::GreaterThan | BinaryOperator::GreaterThanOrEqual => {
            let mut result = Vector::new(DataType::Boolean);
            for i in 0..len {
                let l = left.get(i.min(left.len() - 1))?;
                let r = right.get(i.min(right.len() - 1))?;

                if l.is_null() || r.is_null() {
                    result.push(Value::Null)?;
                } else {
                    let res = match op {
                        BinaryOperator::Equal => l == r,
                        BinaryOperator::NotEqual => l != r,
                        BinaryOperator::LessThan => l < r,
                        BinaryOperator::LessThanOrEqual => l <= r,
                        BinaryOperator::GreaterThan => l > r,
                        BinaryOperator::GreaterThanOrEqual => l >= r,
                        _ => unreachable!(),
                    };
                    result.push(Value::Boolean(res))?;
                }
            }
            Ok(result)
        }

        // Logical operations
        BinaryOperator::And => {
            let mut result = Vector::new(DataType::Boolean);
            for i in 0..len {
                let l = left.get(i.min(left.len() - 1))?;
                let r = right.get(i.min(right.len() - 1))?;

                match (l, r) {
                    (Value::Boolean(false), _) | (_, Value::Boolean(false)) => {
                        result.push(Value::Boolean(false))?;
                    }
                    (Value::Boolean(true), Value::Boolean(true)) => {
                        result.push(Value::Boolean(true))?;
                    }
                    _ => result.push(Value::Null)?,
                }
            }
            Ok(result)
        }

        BinaryOperator::Or => {
            let mut result = Vector::new(DataType::Boolean);
            for i in 0..len {
                let l = left.get(i.min(left.len() - 1))?;
                let r = right.get(i.min(right.len() - 1))?;

                match (l, r) {
                    (Value::Boolean(true), _) | (_, Value::Boolean(true)) => {
                        result.push(Value::Boolean(true))?;
                    }
                    (Value::Boolean(false), Value::Boolean(false)) => {
                        result.push(Value::Boolean(false))?;
                    }
                    _ => result.push(Value::Null)?,
                }
            }
            Ok(result)
        }

        BinaryOperator::Concat => {
            let mut result = Vector::new(DataType::String);
            for i in 0..len {
                let l = left.get(i.min(left.len() - 1))?;
                let r = right.get(i.min(right.len() - 1))?;

                if l.is_null() || r.is_null() {
                    result.push(Value::Null)?;
                } else {
                    let ls = match l {
                        Value::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    let rs = match r {
                        Value::String(s) => s,
                        v => format!("{:?}", v),
                    };
                    result.push(Value::String(format!("{}{}", ls, rs)))?;
                }
            }
            Ok(result)
        }

        _ => Err(Error::InvalidOperation(format!("Unsupported binary operator: {:?}", op))),
    }
}

/// Evaluate unary operation.
fn evaluate_unary_op(op: UnaryOperator, vec: &Vector) -> Result<Vector> {
    match op {
        UnaryOperator::Not => {
            let mut result = Vector::new(DataType::Boolean);
            for i in 0..vec.len() {
                let v = vec.get(i)?;
                match v {
                    Value::Boolean(b) => result.push(Value::Boolean(!b))?,
                    Value::Null => result.push(Value::Null)?,
                    _ => return Err(Error::Type("NOT requires boolean".into())),
                }
            }
            Ok(result)
        }

        UnaryOperator::Negate => {
            let mut result = Vector::new(DataType::Float64);
            for i in 0..vec.len() {
                let v = vec.get(i)?;
                if v.is_null() {
                    result.push(Value::Null)?;
                } else if let Some(f) = v.as_f64() {
                    result.push(Value::Float64(-f))?;
                } else {
                    return Err(Error::Type("Negate requires numeric".into()));
                }
            }
            Ok(result)
        }

        UnaryOperator::BitwiseNot => {
            let mut result = Vector::new(DataType::Int64);
            for i in 0..vec.len() {
                let v = vec.get(i)?;
                if v.is_null() {
                    result.push(Value::Null)?;
                } else if let Some(i) = v.as_i64() {
                    result.push(Value::Int64(!i))?;
                } else {
                    return Err(Error::Type("Bitwise NOT requires integer".into()));
                }
            }
            Ok(result)
        }

        UnaryOperator::Plus => {
            // Plus is a no-op, just returns the value as-is
            let mut result = Vector::new(DataType::Float64);
            for i in 0..vec.len() {
                let v = vec.get(i)?;
                if v.is_null() {
                    result.push(Value::Null)?;
                } else if let Some(f) = v.as_f64() {
                    result.push(Value::Float64(f))?;
                } else {
                    return Err(Error::Type("Plus requires numeric".into()));
                }
            }
            Ok(result)
        }
    }
}

/// Get result type for binary operation.
fn binary_op_result_type(left: &DataType, op: BinaryOperator, right: &DataType) -> Result<DataType> {
    match op {
        BinaryOperator::Add | BinaryOperator::Subtract | BinaryOperator::Multiply | BinaryOperator::Divide => {
            if left.is_numeric() && right.is_numeric() {
                Ok(DataType::Float64)
            } else {
                Err(Error::Type("Arithmetic requires numeric types".into()))
            }
        }

        BinaryOperator::Equal | BinaryOperator::NotEqual |
        BinaryOperator::LessThan | BinaryOperator::LessThanOrEqual |
        BinaryOperator::GreaterThan | BinaryOperator::GreaterThanOrEqual |
        BinaryOperator::And | BinaryOperator::Or => Ok(DataType::Boolean),

        BinaryOperator::Concat => Ok(DataType::String),

        _ => Ok(DataType::Int64),
    }
}

/// Get result type for unary operation.
fn unary_op_result_type(op: UnaryOperator, _expr_type: &DataType) -> Result<DataType> {
    match op {
        UnaryOperator::Not => Ok(DataType::Boolean),
        UnaryOperator::Negate => Ok(DataType::Float64),
        UnaryOperator::BitwiseNot => Ok(DataType::Int64),
        UnaryOperator::Plus => Ok(DataType::Float64),
    }
}

/// Cast vector to target type.
fn cast_vector(vec: &Vector, target_type: &DataType) -> Result<Vector> {
    let mut result = Vector::new(target_type.clone());

    for i in 0..vec.len() {
        let value = vec.get(i)?;
        let casted = cast_value(value, target_type)?;
        result.push(casted)?;
    }

    Ok(result)
}

/// Cast single value.
fn cast_value(value: Value, target_type: &DataType) -> Result<Value> {
    if value.is_null() {
        return Ok(Value::Null);
    }

    match target_type {
        DataType::String => {
            let s = match value {
                Value::String(s) => s,
                Value::Int64(i) => i.to_string(),
                Value::Float64(f) => f.to_string(),
                Value::Boolean(b) => b.to_string(),
                _ => format!("{:?}", value),
            };
            Ok(Value::String(s))
        }

        DataType::Int64 => {
            let i = match value {
                Value::Int64(i) => i,
                Value::Float64(f) => f as i64,
                Value::String(s) => s.parse().map_err(|_| Error::Type("Invalid integer".into()))?,
                Value::Boolean(b) => if b { 1 } else { 0 },
                _ => return Err(Error::Type("Cannot cast to Int64".into())),
            };
            Ok(Value::Int64(i))
        }

        DataType::Float64 => {
            let f = match value {
                Value::Float64(f) => f,
                Value::Int64(i) => i as f64,
                Value::String(s) => s.parse().map_err(|_| Error::Type("Invalid float".into()))?,
                _ => return Err(Error::Type("Cannot cast to Float64".into())),
            };
            Ok(Value::Float64(f))
        }

        DataType::Boolean => {
            let b = match value {
                Value::Boolean(b) => b,
                Value::Int64(i) => i != 0,
                Value::String(s) => !s.is_empty() && s != "false" && s != "0",
                _ => return Err(Error::Type("Cannot cast to Boolean".into())),
            };
            Ok(Value::Boolean(b))
        }

        _ => Err(Error::Type(format!("Unsupported cast to {:?}", target_type))),
    }
}

/// Evaluate CASE expression.
fn evaluate_case(
    chunk: &DataChunk,
    operand: &Option<Box<Expression>>,
    when_clauses: &[(Expression, Expression)],
    else_result: &Option<Box<Expression>>,
) -> Result<Vector> {
    let len = chunk.len();
    let operand_vec = operand.as_ref().map(|e| e.evaluate(chunk)).transpose()?;

    // Determine result type from first THEN clause
    let result_type = if let Some((_, then_expr)) = when_clauses.first() {
        then_expr.data_type(&[])?
    } else {
        DataType::String
    };

    let mut result = Vector::new(result_type);

    for i in 0..len {
        let mut matched = false;

        for (when_expr, then_expr) in when_clauses {
            let condition = if let Some(ref op_vec) = operand_vec {
                // Simple CASE: compare operand with WHEN value
                let when_vec = when_expr.evaluate(chunk)?;
                op_vec.get(i)? == when_vec.get(i)?
            } else {
                // Searched CASE: evaluate WHEN as boolean
                let when_vec = when_expr.evaluate(chunk)?;
                matches!(when_vec.get(i)?, Value::Boolean(true))
            };

            if condition {
                let then_vec = then_expr.evaluate(chunk)?;
                result.push(then_vec.get(i)?)?;
                matched = true;
                break;
            }
        }

        if !matched {
            if let Some(else_expr) = else_result {
                let else_vec = else_expr.evaluate(chunk)?;
                result.push(else_vec.get(i)?)?;
            } else {
                result.push(Value::Null)?;
            }
        }
    }

    Ok(result)
}

/// Evaluate scalar function.
fn evaluate_function(name: &str, args: &[Vector], len: usize) -> Result<Vector> {
    match name.to_uppercase().as_str() {
        "ABS" => {
            let mut result = Vector::new(DataType::Float64);
            for i in 0..len {
                let v = args[0].get(i)?;
                if v.is_null() {
                    result.push(Value::Null)?;
                } else if let Some(f) = v.as_f64() {
                    result.push(Value::Float64(f.abs()))?;
                } else {
                    result.push(Value::Null)?;
                }
            }
            Ok(result)
        }

        "UPPER" => {
            let mut result = Vector::new(DataType::String);
            for i in 0..len {
                let v = args[0].get(i)?;
                match v {
                    Value::String(s) => result.push(Value::String(s.to_uppercase()))?,
                    Value::Null => result.push(Value::Null)?,
                    _ => result.push(Value::Null)?,
                }
            }
            Ok(result)
        }

        "LOWER" => {
            let mut result = Vector::new(DataType::String);
            for i in 0..len {
                let v = args[0].get(i)?;
                match v {
                    Value::String(s) => result.push(Value::String(s.to_lowercase()))?,
                    Value::Null => result.push(Value::Null)?,
                    _ => result.push(Value::Null)?,
                }
            }
            Ok(result)
        }

        "LENGTH" => {
            let mut result = Vector::new(DataType::Int64);
            for i in 0..len {
                let v = args[0].get(i)?;
                match v {
                    Value::String(s) => result.push(Value::Int64(s.len() as i64))?,
                    Value::Null => result.push(Value::Null)?,
                    _ => result.push(Value::Null)?,
                }
            }
            Ok(result)
        }

        "COALESCE" => {
            let result_type = args.first().map(|v| v.data_type.clone()).unwrap_or(DataType::String);
            let mut result = Vector::new(result_type);
            for i in 0..len {
                let mut found = false;
                for arg in args {
                    let v = arg.get(i)?;
                    if !v.is_null() {
                        result.push(v)?;
                        found = true;
                        break;
                    }
                }
                if !found {
                    result.push(Value::Null)?;
                }
            }
            Ok(result)
        }

        _ => Err(Error::InvalidOperation(format!("Unknown function: {}", name))),
    }
}

/// Convert LIKE pattern to regex.
fn like_to_regex(pattern: &str) -> String {
    let mut regex = String::from("^");
    for ch in pattern.chars() {
        match ch {
            '%' => regex.push_str(".*"),
            '_' => regex.push('.'),
            '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | '\\' => {
                regex.push('\\');
                regex.push(ch);
            }
            _ => regex.push(ch),
        }
    }
    regex.push('$');
    regex
}
