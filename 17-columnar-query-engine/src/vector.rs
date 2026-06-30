//! Vector types for vectorized execution.

use crate::types::{DataType, Value};
use crate::{Error, Result, VECTOR_SIZE};
use bitvec::prelude::*;
use std::sync::Arc;

/// A vector of values for vectorized execution.
#[derive(Debug, Clone)]
pub struct Vector {
    /// Data type.
    pub data_type: DataType,
    /// Number of values.
    pub len: usize,
    /// Validity bitmap (null mask).
    pub validity: BitVec<u64, Lsb0>,
    /// Data buffer.
    pub data: VectorData,
}

/// Vector data storage.
#[derive(Debug, Clone)]
pub enum VectorData {
    Boolean(Vec<bool>),
    Int8(Vec<i8>),
    Int16(Vec<i16>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    UInt8(Vec<u8>),
    UInt16(Vec<u16>),
    UInt32(Vec<u32>),
    UInt64(Vec<u64>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    String(Vec<String>),
    Binary(Vec<Vec<u8>>),
    /// Constant value replicated.
    Constant(Box<Value>),
    /// Dictionary encoded.
    Dictionary {
        indices: Vec<u32>,
        dictionary: Arc<Vector>,
    },
}

impl Vector {
    /// Create a new vector with the given type.
    pub fn new(data_type: DataType) -> Self {
        let data = match &data_type {
            DataType::Boolean => VectorData::Boolean(Vec::with_capacity(VECTOR_SIZE)),
            DataType::Int8 => VectorData::Int8(Vec::with_capacity(VECTOR_SIZE)),
            DataType::Int16 => VectorData::Int16(Vec::with_capacity(VECTOR_SIZE)),
            DataType::Int32 | DataType::Date => VectorData::Int32(Vec::with_capacity(VECTOR_SIZE)),
            DataType::Int64 | DataType::Timestamp => {
                VectorData::Int64(Vec::with_capacity(VECTOR_SIZE))
            }
            DataType::UInt8 => VectorData::UInt8(Vec::with_capacity(VECTOR_SIZE)),
            DataType::UInt16 => VectorData::UInt16(Vec::with_capacity(VECTOR_SIZE)),
            DataType::UInt32 => VectorData::UInt32(Vec::with_capacity(VECTOR_SIZE)),
            DataType::UInt64 => VectorData::UInt64(Vec::with_capacity(VECTOR_SIZE)),
            DataType::Float32 => VectorData::Float32(Vec::with_capacity(VECTOR_SIZE)),
            DataType::Float64 | DataType::Decimal { .. } => {
                VectorData::Float64(Vec::with_capacity(VECTOR_SIZE))
            }
            DataType::String => VectorData::String(Vec::with_capacity(VECTOR_SIZE)),
            DataType::Binary => VectorData::Binary(Vec::with_capacity(VECTOR_SIZE)),
            DataType::List(_) | DataType::Struct(_) => {
                VectorData::String(Vec::with_capacity(VECTOR_SIZE))
            }
        };

        Self {
            data_type,
            len: 0,
            validity: BitVec::with_capacity(VECTOR_SIZE),
            data,
        }
    }

    /// Create a constant vector.
    pub fn constant(value: Value, len: usize) -> Self {
        let data_type = value.data_type();
        let mut validity = BitVec::with_capacity(len);
        validity.resize(len, !value.is_null());

        Self {
            data_type,
            len,
            validity,
            data: VectorData::Constant(Box::new(value)),
        }
    }

    /// Create from values.
    pub fn from_values(values: Vec<Value>) -> Result<Self> {
        if values.is_empty() {
            return Ok(Self::new(DataType::Int64));
        }

        let data_type = values[0].data_type();
        let mut vector = Self::new(data_type.clone());

        for value in values {
            vector.push(value)?;
        }

        Ok(vector)
    }

    /// Push a value.
    pub fn push(&mut self, value: Value) -> Result<()> {
        let is_null = value.is_null();
        self.validity.push(!is_null);

        match (&mut self.data, value) {
            (VectorData::Boolean(v), Value::Boolean(b)) => v.push(b),
            (VectorData::Boolean(v), Value::Null) => v.push(false),
            (VectorData::Int8(v), Value::Int8(i)) => v.push(i),
            (VectorData::Int8(v), Value::Null) => v.push(0),
            (VectorData::Int16(v), Value::Int16(i)) => v.push(i),
            (VectorData::Int16(v), Value::Null) => v.push(0),
            (VectorData::Int32(v), Value::Int32(i)) => v.push(i),
            (VectorData::Int32(v), Value::Date(d)) => v.push(d),
            (VectorData::Int32(v), Value::Null) => v.push(0),
            (VectorData::Int64(v), Value::Int64(i)) => v.push(i),
            (VectorData::Int64(v), Value::Timestamp(t)) => v.push(t),
            (VectorData::Int64(v), Value::Null) => v.push(0),
            (VectorData::UInt8(v), Value::UInt8(i)) => v.push(i),
            (VectorData::UInt8(v), Value::Null) => v.push(0),
            (VectorData::UInt16(v), Value::UInt16(i)) => v.push(i),
            (VectorData::UInt16(v), Value::Null) => v.push(0),
            (VectorData::UInt32(v), Value::UInt32(i)) => v.push(i),
            (VectorData::UInt32(v), Value::Null) => v.push(0),
            (VectorData::UInt64(v), Value::UInt64(i)) => v.push(i),
            (VectorData::UInt64(v), Value::Null) => v.push(0),
            (VectorData::Float32(v), Value::Float32(f)) => v.push(f),
            (VectorData::Float32(v), Value::Null) => v.push(0.0),
            (VectorData::Float64(v), Value::Float64(f)) => v.push(f),
            (VectorData::Float64(v), Value::Decimal(d)) => v.push(d as f64),
            (VectorData::Float64(v), Value::Null) => v.push(0.0),
            (VectorData::String(v), Value::String(s)) => v.push(s),
            (VectorData::String(v), Value::Null) => v.push(String::new()),
            (VectorData::Binary(v), Value::Binary(b)) => v.push(b),
            (VectorData::Binary(v), Value::Null) => v.push(Vec::new()),
            _ => return Err(Error::Type("Type mismatch".into())),
        }

        self.len += 1;
        Ok(())
    }

    /// Get value at index.
    pub fn get(&self, index: usize) -> Result<Value> {
        if index >= self.len {
            return Err(Error::InvalidOperation("Index out of bounds".into()));
        }

        if !self.validity[index] {
            return Ok(Value::Null);
        }

        match &self.data {
            VectorData::Boolean(v) => Ok(Value::Boolean(v[index])),
            VectorData::Int8(v) => Ok(Value::Int8(v[index])),
            VectorData::Int16(v) => Ok(Value::Int16(v[index])),
            VectorData::Int32(v) => {
                if matches!(self.data_type, DataType::Date) {
                    Ok(Value::Date(v[index]))
                } else {
                    Ok(Value::Int32(v[index]))
                }
            }
            VectorData::Int64(v) => {
                if matches!(self.data_type, DataType::Timestamp) {
                    Ok(Value::Timestamp(v[index]))
                } else {
                    Ok(Value::Int64(v[index]))
                }
            }
            VectorData::UInt8(v) => Ok(Value::UInt8(v[index])),
            VectorData::UInt16(v) => Ok(Value::UInt16(v[index])),
            VectorData::UInt32(v) => Ok(Value::UInt32(v[index])),
            VectorData::UInt64(v) => Ok(Value::UInt64(v[index])),
            VectorData::Float32(v) => Ok(Value::Float32(v[index])),
            VectorData::Float64(v) => {
                if matches!(self.data_type, DataType::Decimal { .. }) {
                    Ok(Value::Decimal(v[index] as i128))
                } else {
                    Ok(Value::Float64(v[index]))
                }
            }
            VectorData::String(v) => Ok(Value::String(v[index].clone())),
            VectorData::Binary(v) => Ok(Value::Binary(v[index].clone())),
            VectorData::Constant(v) => Ok((**v).clone()),
            VectorData::Dictionary {
                indices,
                dictionary,
            } => dictionary.get(indices[index] as usize),
        }
    }

    /// Check if value at index is null.
    pub fn is_null(&self, index: usize) -> bool {
        !self.validity[index]
    }

    /// Get length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Slice the vector.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        if offset + len > self.len {
            return Err(Error::InvalidOperation("Slice out of bounds".into()));
        }

        let validity = self.validity[offset..offset + len].to_bitvec();

        let data = match &self.data {
            VectorData::Boolean(v) => VectorData::Boolean(v[offset..offset + len].to_vec()),
            VectorData::Int8(v) => VectorData::Int8(v[offset..offset + len].to_vec()),
            VectorData::Int16(v) => VectorData::Int16(v[offset..offset + len].to_vec()),
            VectorData::Int32(v) => VectorData::Int32(v[offset..offset + len].to_vec()),
            VectorData::Int64(v) => VectorData::Int64(v[offset..offset + len].to_vec()),
            VectorData::UInt8(v) => VectorData::UInt8(v[offset..offset + len].to_vec()),
            VectorData::UInt16(v) => VectorData::UInt16(v[offset..offset + len].to_vec()),
            VectorData::UInt32(v) => VectorData::UInt32(v[offset..offset + len].to_vec()),
            VectorData::UInt64(v) => VectorData::UInt64(v[offset..offset + len].to_vec()),
            VectorData::Float32(v) => VectorData::Float32(v[offset..offset + len].to_vec()),
            VectorData::Float64(v) => VectorData::Float64(v[offset..offset + len].to_vec()),
            VectorData::String(v) => VectorData::String(v[offset..offset + len].to_vec()),
            VectorData::Binary(v) => VectorData::Binary(v[offset..offset + len].to_vec()),
            VectorData::Constant(v) => VectorData::Constant(v.clone()),
            VectorData::Dictionary {
                indices,
                dictionary,
            } => VectorData::Dictionary {
                indices: indices[offset..offset + len].to_vec(),
                dictionary: dictionary.clone(),
            },
        };

        Ok(Self {
            data_type: self.data_type.clone(),
            len,
            validity,
            data,
        })
    }

    /// Filter by selection vector.
    pub fn filter(&self, selection: &[usize]) -> Result<Self> {
        let mut result = Self::new(self.data_type.clone());

        for &idx in selection {
            let value = self.get(idx)?;
            result.push(value)?;
        }

        Ok(result)
    }

    /// Get as i64 slice (for numeric operations).
    pub fn as_i64_slice(&self) -> Option<&[i64]> {
        match &self.data {
            VectorData::Int64(v) => Some(v),
            _ => None,
        }
    }

    /// Get as f64 slice (for numeric operations).
    pub fn as_f64_slice(&self) -> Option<&[f64]> {
        match &self.data {
            VectorData::Float64(v) => Some(v),
            _ => None,
        }
    }

    /// Get as string slice.
    pub fn as_string_slice(&self) -> Option<&[String]> {
        match &self.data {
            VectorData::String(v) => Some(v),
            _ => None,
        }
    }
}

/// A chunk of vectors (columnar batch).
#[derive(Debug, Clone)]
pub struct DataChunk {
    /// Vectors (one per column).
    pub vectors: Vec<Vector>,
    /// Number of rows.
    pub len: usize,
}

impl DataChunk {
    /// Create a new data chunk.
    pub fn new(vectors: Vec<Vector>) -> Self {
        let len = vectors.first().map(|v| v.len()).unwrap_or(0);
        Self { vectors, len }
    }

    /// Create empty chunk with schema.
    pub fn empty(data_types: &[DataType]) -> Self {
        let vectors = data_types.iter().map(|dt| Vector::new(dt.clone())).collect();
        Self { vectors, len: 0 }
    }

    /// Get number of columns.
    pub fn num_columns(&self) -> usize {
        self.vectors.len()
    }

    /// Get number of rows.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get vector by index.
    pub fn vector(&self, index: usize) -> Option<&Vector> {
        self.vectors.get(index)
    }

    /// Get mutable vector by index.
    pub fn vector_mut(&mut self, index: usize) -> Option<&mut Vector> {
        self.vectors.get_mut(index)
    }

    /// Slice the chunk.
    pub fn slice(&self, offset: usize, len: usize) -> Result<Self> {
        let vectors = self
            .vectors
            .iter()
            .map(|v| v.slice(offset, len))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { vectors, len })
    }

    /// Filter by selection vector.
    pub fn filter(&self, selection: &[usize]) -> Result<Self> {
        let vectors = self
            .vectors
            .iter()
            .map(|v| v.filter(selection))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            vectors,
            len: selection.len(),
        })
    }

    /// Append another chunk.
    pub fn append(&mut self, other: &DataChunk) -> Result<()> {
        if self.vectors.len() != other.vectors.len() {
            return Err(Error::InvalidOperation("Column count mismatch".into()));
        }

        for (i, other_vec) in other.vectors.iter().enumerate() {
            for j in 0..other_vec.len() {
                let value = other_vec.get(j)?;
                self.vectors[i].push(value)?;
            }
        }

        self.len += other.len;
        Ok(())
    }
}

/// Selection vector for filtering.
#[derive(Debug, Clone)]
pub struct SelectionVector {
    /// Selected indices.
    pub indices: Vec<usize>,
}

impl SelectionVector {
    /// Create from indices.
    pub fn new(indices: Vec<usize>) -> Self {
        Self { indices }
    }

    /// Create sequential selection.
    pub fn sequential(len: usize) -> Self {
        Self {
            indices: (0..len).collect(),
        }
    }

    /// Get length.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}
