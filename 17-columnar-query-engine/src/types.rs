//! Data types for the query engine.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

/// Data types supported by the engine.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    Boolean,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    String,
    Binary,
    Date,
    Timestamp,
    Decimal { precision: u8, scale: u8 },
    List(Box<DataType>),
    Struct(Vec<(String, DataType)>),
}

impl DataType {
    /// Get the size in bytes of a single value.
    pub fn size(&self) -> usize {
        match self {
            DataType::Boolean => 1,
            DataType::Int8 | DataType::UInt8 => 1,
            DataType::Int16 | DataType::UInt16 => 2,
            DataType::Int32 | DataType::UInt32 | DataType::Float32 | DataType::Date => 4,
            DataType::Int64 | DataType::UInt64 | DataType::Float64 | DataType::Timestamp => 8,
            DataType::Decimal { .. } => 16,
            DataType::String | DataType::Binary => 16, // Pointer + length
            DataType::List(_) => 16,
            DataType::Struct(_) => 0, // Variable
        }
    }

    /// Check if type is numeric.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
                | DataType::Decimal { .. }
        )
    }

    /// Check if type is integer.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
        )
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Boolean => write!(f, "BOOLEAN"),
            DataType::Int8 => write!(f, "INT8"),
            DataType::Int16 => write!(f, "INT16"),
            DataType::Int32 => write!(f, "INT32"),
            DataType::Int64 => write!(f, "INT64"),
            DataType::UInt8 => write!(f, "UINT8"),
            DataType::UInt16 => write!(f, "UINT16"),
            DataType::UInt32 => write!(f, "UINT32"),
            DataType::UInt64 => write!(f, "UINT64"),
            DataType::Float32 => write!(f, "FLOAT32"),
            DataType::Float64 => write!(f, "FLOAT64"),
            DataType::String => write!(f, "VARCHAR"),
            DataType::Binary => write!(f, "BLOB"),
            DataType::Date => write!(f, "DATE"),
            DataType::Timestamp => write!(f, "TIMESTAMP"),
            DataType::Decimal { precision, scale } => write!(f, "DECIMAL({},{})", precision, scale),
            DataType::List(inner) => write!(f, "{}[]", inner),
            DataType::Struct(fields) => {
                write!(f, "STRUCT(")?;
                for (i, (name, dt)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{} {}", name, dt)?;
                }
                write!(f, ")")
            }
        }
    }
}

/// Scalar value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Null,
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Int64(i64),
    UInt8(u8),
    UInt16(u16),
    UInt32(u32),
    UInt64(u64),
    Float32(f32),
    Float64(f64),
    String(String),
    Binary(Vec<u8>),
    Date(i32),      // Days since epoch
    Timestamp(i64), // Microseconds since epoch
    Decimal(i128),
    List(Vec<Value>),
    Struct(Vec<(String, Value)>),
}

impl Value {
    /// Get the data type of this value.
    pub fn data_type(&self) -> DataType {
        match self {
            Value::Null => DataType::Boolean, // Null can be any type
            Value::Boolean(_) => DataType::Boolean,
            Value::Int8(_) => DataType::Int8,
            Value::Int16(_) => DataType::Int16,
            Value::Int32(_) => DataType::Int32,
            Value::Int64(_) => DataType::Int64,
            Value::UInt8(_) => DataType::UInt8,
            Value::UInt16(_) => DataType::UInt16,
            Value::UInt32(_) => DataType::UInt32,
            Value::UInt64(_) => DataType::UInt64,
            Value::Float32(_) => DataType::Float32,
            Value::Float64(_) => DataType::Float64,
            Value::String(_) => DataType::String,
            Value::Binary(_) => DataType::Binary,
            Value::Date(_) => DataType::Date,
            Value::Timestamp(_) => DataType::Timestamp,
            Value::Decimal(_) => DataType::Decimal {
                precision: 38,
                scale: 0,
            },
            Value::List(items) => {
                let inner = items.first().map(|v| v.data_type()).unwrap_or(DataType::Int64);
                DataType::List(Box::new(inner))
            }
            Value::Struct(fields) => DataType::Struct(
                fields
                    .iter()
                    .map(|(name, val)| (name.clone(), val.data_type()))
                    .collect(),
            ),
        }
    }

    /// Check if value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Try to get as i64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Int8(v) => Some(*v as i64),
            Value::Int16(v) => Some(*v as i64),
            Value::Int32(v) => Some(*v as i64),
            Value::Int64(v) => Some(*v),
            Value::UInt8(v) => Some(*v as i64),
            Value::UInt16(v) => Some(*v as i64),
            Value::UInt32(v) => Some(*v as i64),
            _ => None,
        }
    }

    /// Try to get as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float32(v) => Some(*v as f64),
            Value::Float64(v) => Some(*v),
            Value::Int64(v) => Some(*v as f64),
            _ => None,
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Null, Value::Null) => true,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Int64(a), Value::Int64(b)) => a == b,
            (Value::Float64(a), Value::Float64(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            _ => false,
        }
    }
}

impl PartialOrd for Value {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Value::Int64(a), Value::Int64(b)) => a.partial_cmp(b),
            (Value::Float64(a), Value::Float64(b)) => a.partial_cmp(b),
            (Value::String(a), Value::String(b)) => a.partial_cmp(b),
            (Value::Boolean(a), Value::Boolean(b)) => a.partial_cmp(b),
            _ => None,
        }
    }
}

/// Table schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Column definitions.
    pub columns: Vec<Column>,
}

impl Schema {
    /// Create a new schema from Column definitions.
    pub fn new(columns: Vec<Column>) -> Self {
        Self { columns }
    }

    /// Create a schema from (name, datatype) pairs.
    pub fn from_pairs(columns: Vec<(String, DataType)>) -> Self {
        Self {
            columns: columns
                .into_iter()
                .map(|(name, dt)| Column::new(name, dt, true))
                .collect(),
        }
    }

    /// Create an empty schema.
    pub fn empty() -> Self {
        Self { columns: vec![] }
    }

    /// Get column by name.
    pub fn column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }

    /// Get column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Get column type by name.
    pub fn get_column_type(&self, name: &str) -> Option<DataType> {
        self.columns
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.data_type.clone())
    }

    /// Get number of columns.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// Check if schema is empty.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// Get columns as (name, datatype) pairs.
    pub fn columns(&self) -> Vec<(String, DataType)> {
        self.columns
            .iter()
            .map(|c| (c.name.clone(), c.data_type.clone()))
            .collect()
    }

    /// Merge with another schema.
    pub fn merge(&self, other: &Schema) -> Schema {
        let mut columns = self.columns.clone();
        columns.extend(other.columns.clone());
        Schema { columns }
    }
}

/// Column definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Column {
    /// Column name.
    pub name: String,
    /// Data type.
    pub data_type: DataType,
    /// Whether column is nullable.
    pub nullable: bool,
}

impl Column {
    /// Create a new column.
    pub fn new(name: impl Into<String>, data_type: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

/// Sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SortOrder {
    Ascending,
    Descending,
}

/// Join type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
    Semi,
    Anti,
}

/// Aggregate function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AggregateFunction {
    Count,
    Sum,
    Avg,
    Min,
    Max,
    First,
    Last,
}
