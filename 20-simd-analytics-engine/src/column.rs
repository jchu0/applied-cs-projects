//! Columnar storage with aligned memory allocations.

use crate::{Error, Result, CACHE_LINE_SIZE};
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Aligned vector for SIMD operations.
#[derive(Debug)]
pub struct AlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    alignment: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy + Default> AlignedVec<T> {
    /// Create a new aligned vector with specified capacity and alignment.
    pub fn with_capacity_aligned(capacity: usize, alignment: usize) -> Result<Self> {
        if capacity == 0 {
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
                capacity: 0,
                alignment,
                _marker: PhantomData,
            });
        }

        let layout = Layout::from_size_align(
            capacity * std::mem::size_of::<T>(),
            alignment,
        ).map_err(|_| Error::AlignmentError(format!(
            "Invalid alignment {} for size {}",
            alignment,
            capacity * std::mem::size_of::<T>()
        )))?;

        let ptr = unsafe {
            let ptr = alloc(layout) as *mut T;
            if ptr.is_null() {
                return Err(Error::AlignmentError("Allocation failed".into()));
            }
            NonNull::new_unchecked(ptr)
        };

        Ok(Self {
            ptr,
            len: 0,
            capacity,
            alignment,
            _marker: PhantomData,
        })
    }

    /// Create with default cache-line alignment.
    pub fn with_capacity(capacity: usize) -> Result<Self> {
        Self::with_capacity_aligned(capacity, CACHE_LINE_SIZE)
    }

    /// Create from existing data with alignment.
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let mut vec = Self::with_capacity(data.len())?;
        for &item in data {
            vec.push(item)?;
        }
        Ok(vec)
    }

    /// Create with specified length and default values.
    pub fn with_len(len: usize) -> Result<Self> {
        let mut vec = Self::with_capacity(len)?;
        vec.len = len;
        // Initialize with defaults
        for i in 0..len {
            unsafe {
                vec.ptr.as_ptr().add(i).write(T::default());
            }
        }
        Ok(vec)
    }

    /// Push an element.
    pub fn push(&mut self, value: T) -> Result<()> {
        if self.len >= self.capacity {
            return Err(Error::InvalidOperation("Vector is full".into()));
        }
        unsafe {
            self.ptr.as_ptr().add(self.len).write(value);
        }
        self.len += 1;
        Ok(())
    }

    /// Get element at index.
    pub fn get(&self, index: usize) -> Result<T> {
        if index >= self.len {
            return Err(Error::IndexOutOfBounds { index, len: self.len });
        }
        unsafe { Ok(*self.ptr.as_ptr().add(index)) }
    }

    /// Set element at index.
    pub fn set(&mut self, index: usize, value: T) -> Result<()> {
        if index >= self.len {
            return Err(Error::IndexOutOfBounds { index, len: self.len });
        }
        unsafe {
            self.ptr.as_ptr().add(index).write(value);
        }
        Ok(())
    }

    /// Get length.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get raw pointer (aligned).
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer (aligned).
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get as slice.
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get as mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Check if pointer is aligned.
    pub fn is_aligned(&self, alignment: usize) -> bool {
        (self.ptr.as_ptr() as usize) % alignment == 0
    }
}

impl<T> Drop for AlignedVec<T> {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let layout = Layout::from_size_align(
                self.capacity * std::mem::size_of::<T>(),
                self.alignment,
            ).unwrap();
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

// Safety: AlignedVec can be sent between threads if T is Send
unsafe impl<T: Copy + Default + Send> Send for AlignedVec<T> {}
unsafe impl<T: Copy + Default + Sync> Sync for AlignedVec<T> {}

/// Column data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnType {
    Int32,
    Int64,
    Float32,
    Float64,
    Bool,
}

impl ColumnType {
    /// Get size in bytes.
    pub fn size(&self) -> usize {
        match self {
            ColumnType::Int32 => 4,
            ColumnType::Int64 => 8,
            ColumnType::Float32 => 4,
            ColumnType::Float64 => 8,
            ColumnType::Bool => 1,
        }
    }
}

/// Column storage with type information.
#[derive(Debug)]
pub enum Column {
    Int32(AlignedVec<i32>),
    Int64(AlignedVec<i64>),
    Float32(AlignedVec<f32>),
    Float64(AlignedVec<f64>),
    Bool(AlignedVec<u8>),
}

impl Column {
    /// Create Int32 column from data.
    pub fn from_i32(data: &[i32]) -> Result<Self> {
        Ok(Column::Int32(AlignedVec::from_slice(data)?))
    }

    /// Create Int64 column from data.
    pub fn from_i64(data: &[i64]) -> Result<Self> {
        Ok(Column::Int64(AlignedVec::from_slice(data)?))
    }

    /// Create Float32 column from data.
    pub fn from_f32(data: &[f32]) -> Result<Self> {
        Ok(Column::Float32(AlignedVec::from_slice(data)?))
    }

    /// Create Float64 column from data.
    pub fn from_f64(data: &[f64]) -> Result<Self> {
        Ok(Column::Float64(AlignedVec::from_slice(data)?))
    }

    /// Create Bool column from data.
    pub fn from_bool(data: &[bool]) -> Result<Self> {
        let bytes: Vec<u8> = data.iter().map(|&b| if b { 1 } else { 0 }).collect();
        Ok(Column::Bool(AlignedVec::from_slice(&bytes)?))
    }

    /// Get column type.
    pub fn column_type(&self) -> ColumnType {
        match self {
            Column::Int32(_) => ColumnType::Int32,
            Column::Int64(_) => ColumnType::Int64,
            Column::Float32(_) => ColumnType::Float32,
            Column::Float64(_) => ColumnType::Float64,
            Column::Bool(_) => ColumnType::Bool,
        }
    }

    /// Get length.
    pub fn len(&self) -> usize {
        match self {
            Column::Int32(v) => v.len(),
            Column::Int64(v) => v.len(),
            Column::Float32(v) => v.len(),
            Column::Float64(v) => v.len(),
            Column::Bool(v) => v.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get Int32 data.
    pub fn as_i32(&self) -> Result<&[i32]> {
        match self {
            Column::Int32(v) => Ok(v.as_slice()),
            _ => Err(Error::TypeMismatch {
                expected: "Int32".into(),
                got: format!("{:?}", self.column_type()),
            }),
        }
    }

    /// Get Int64 data.
    pub fn as_i64(&self) -> Result<&[i64]> {
        match self {
            Column::Int64(v) => Ok(v.as_slice()),
            _ => Err(Error::TypeMismatch {
                expected: "Int64".into(),
                got: format!("{:?}", self.column_type()),
            }),
        }
    }

    /// Get Float32 data.
    pub fn as_f32(&self) -> Result<&[f32]> {
        match self {
            Column::Float32(v) => Ok(v.as_slice()),
            _ => Err(Error::TypeMismatch {
                expected: "Float32".into(),
                got: format!("{:?}", self.column_type()),
            }),
        }
    }

    /// Get Float64 data.
    pub fn as_f64(&self) -> Result<&[f64]> {
        match self {
            Column::Float64(v) => Ok(v.as_slice()),
            _ => Err(Error::TypeMismatch {
                expected: "Float64".into(),
                got: format!("{:?}", self.column_type()),
            }),
        }
    }

    /// Get mutable Float32 data.
    pub fn as_f32_mut(&mut self) -> Result<&mut [f32]> {
        match self {
            Column::Float32(v) => Ok(v.as_mut_slice()),
            _ => Err(Error::TypeMismatch {
                expected: "Float32".into(),
                got: format!("{:?}", self.column_type()),
            }),
        }
    }

    /// Get mutable Float64 data.
    pub fn as_f64_mut(&mut self) -> Result<&mut [f64]> {
        match self {
            Column::Float64(v) => Ok(v.as_mut_slice()),
            _ => Err(Error::TypeMismatch {
                expected: "Float64".into(),
                got: format!("{:?}", self.column_type()),
            }),
        }
    }

    /// Get size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.len() * self.column_type().size()
    }
}

/// Data chunk containing multiple columns.
#[derive(Debug)]
pub struct DataChunk {
    /// Columns in this chunk.
    pub columns: Vec<Column>,
    /// Number of rows.
    pub num_rows: usize,
}

impl DataChunk {
    /// Create new data chunk.
    pub fn new(columns: Vec<Column>) -> Result<Self> {
        if columns.is_empty() {
            return Err(Error::EmptyData);
        }

        let num_rows = columns[0].len();
        for col in &columns {
            if col.len() != num_rows {
                return Err(Error::DimensionMismatch(format!(
                    "Column lengths don't match: {} vs {}",
                    num_rows, col.len()
                )));
            }
        }

        Ok(Self { columns, num_rows })
    }

    /// Get number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// Get column by index.
    pub fn column(&self, index: usize) -> Result<&Column> {
        self.columns.get(index).ok_or(Error::IndexOutOfBounds {
            index,
            len: self.columns.len(),
        })
    }

    /// Get mutable column by index.
    pub fn column_mut(&mut self, index: usize) -> Result<&mut Column> {
        let len = self.columns.len();
        self.columns.get_mut(index).ok_or(Error::IndexOutOfBounds {
            index,
            len,
        })
    }

    /// Get total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.columns.iter().map(|c| c.size_bytes()).sum()
    }

    /// Slice chunk to create a view.
    pub fn slice(&self, offset: usize, length: usize) -> Result<DataChunk> {
        if offset + length > self.num_rows {
            return Err(Error::IndexOutOfBounds {
                index: offset + length,
                len: self.num_rows,
            });
        }

        let columns = self.columns.iter().map(|col| {
            match col {
                Column::Int32(v) => {
                    let slice = &v.as_slice()[offset..offset + length];
                    Column::from_i32(slice)
                }
                Column::Int64(v) => {
                    let slice = &v.as_slice()[offset..offset + length];
                    Column::from_i64(slice)
                }
                Column::Float32(v) => {
                    let slice = &v.as_slice()[offset..offset + length];
                    Column::from_f32(slice)
                }
                Column::Float64(v) => {
                    let slice = &v.as_slice()[offset..offset + length];
                    Column::from_f64(slice)
                }
                Column::Bool(v) => {
                    let slice: Vec<bool> = v.as_slice()[offset..offset + length]
                        .iter()
                        .map(|&b| b != 0)
                        .collect();
                    Column::from_bool(&slice)
                }
            }
        }).collect::<Result<Vec<_>>>()?;

        DataChunk::new(columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_vec() {
        let mut vec: AlignedVec<f32> = AlignedVec::with_capacity(100).unwrap();
        assert!(vec.is_aligned(CACHE_LINE_SIZE));

        for i in 0..100 {
            vec.push(i as f32).unwrap();
        }

        assert_eq!(vec.len(), 100);
        assert_eq!(vec.get(50).unwrap(), 50.0);
    }

    #[test]
    fn test_column() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let col = Column::from_f32(&data).unwrap();

        assert_eq!(col.len(), 1000);
        assert_eq!(col.column_type(), ColumnType::Float32);
    }

    #[test]
    fn test_data_chunk() {
        let col1 = Column::from_f32(&[1.0, 2.0, 3.0]).unwrap();
        let col2 = Column::from_i64(&[10, 20, 30]).unwrap();

        let chunk = DataChunk::new(vec![col1, col2]).unwrap();
        assert_eq!(chunk.num_rows, 3);
        assert_eq!(chunk.num_columns(), 2);
    }
}
