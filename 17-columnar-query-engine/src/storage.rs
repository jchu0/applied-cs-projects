//! Columnar storage engine.

use crate::types::{Column, DataType, Schema, Value};
use crate::vector::{DataChunk, Vector};
use crate::{Error, Result, VECTOR_SIZE};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Storage configuration.
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Row group size.
    pub row_group_size: usize,
    /// Enable compression.
    pub compression: bool,
    /// Enable statistics.
    pub statistics: bool,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            row_group_size: 100_000,
            compression: true,
            statistics: true,
        }
    }
}

/// Column chunk within a row group.
#[derive(Debug, Clone)]
pub struct ColumnChunk {
    /// Column index.
    pub column_idx: usize,
    /// Data type.
    pub data_type: DataType,
    /// Compressed data.
    pub data: Vec<u8>,
    /// Number of values.
    pub num_values: usize,
    /// Null count.
    pub null_count: usize,
    /// Statistics.
    pub stats: Option<ColumnStats>,
    /// Encoding used.
    pub encoding: Encoding,
}

/// Column statistics.
#[derive(Debug, Clone)]
pub struct ColumnStats {
    /// Minimum value.
    pub min: Value,
    /// Maximum value.
    pub max: Value,
    /// Null count.
    pub null_count: usize,
    /// Distinct count (approximate).
    pub distinct_count: Option<usize>,
}

/// Data encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// Plain encoding.
    Plain,
    /// Run-length encoding.
    Rle,
    /// Dictionary encoding.
    Dictionary,
    /// Delta encoding.
    Delta,
    /// Bit-packed.
    BitPacked,
}

/// Row group containing column chunks.
#[derive(Debug)]
pub struct RowGroup {
    /// Row group index.
    pub index: usize,
    /// Number of rows.
    pub num_rows: usize,
    /// Column chunks.
    pub columns: Vec<ColumnChunk>,
}

impl RowGroup {
    /// Create new row group.
    pub fn new(index: usize, num_rows: usize, columns: Vec<ColumnChunk>) -> Self {
        Self {
            index,
            num_rows,
            columns,
        }
    }

    /// Get column chunk.
    pub fn column(&self, idx: usize) -> Option<&ColumnChunk> {
        self.columns.get(idx)
    }

    /// Read column as vector.
    pub fn read_column(&self, idx: usize) -> Result<Vector> {
        let chunk = self
            .column(idx)
            .ok_or_else(|| Error::ColumnNotFound(format!("Column index {}", idx)))?;

        decode_column(chunk)
    }

    /// Read as data chunk.
    pub fn read(&self, column_indices: &[usize]) -> Result<DataChunk> {
        let mut vectors = Vec::with_capacity(column_indices.len());

        for &idx in column_indices {
            vectors.push(self.read_column(idx)?);
        }

        Ok(DataChunk::new(vectors))
    }
}

/// Table storage.
pub struct Table {
    /// Table name.
    pub name: String,
    /// Schema.
    pub schema: Schema,
    /// Row groups.
    pub row_groups: RwLock<Vec<Arc<RowGroup>>>,
    /// Total row count.
    pub row_count: RwLock<usize>,
    /// Storage configuration.
    pub config: StorageConfig,
}

impl Table {
    /// Create new table.
    pub fn new(name: impl Into<String>, schema: Schema, config: StorageConfig) -> Self {
        Self {
            name: name.into(),
            schema,
            row_groups: RwLock::new(Vec::new()),
            row_count: RwLock::new(0),
            config,
        }
    }

    /// Get row count.
    pub fn row_count(&self) -> usize {
        *self.row_count.read()
    }

    /// Get row group count.
    pub fn row_group_count(&self) -> usize {
        self.row_groups.read().len()
    }

    /// Insert a data chunk.
    pub fn insert(&self, chunk: DataChunk) -> Result<()> {
        if chunk.num_columns() != self.schema.len() {
            return Err(Error::InvalidOperation("Column count mismatch".into()));
        }

        // Build row group from chunk
        let row_group = self.build_row_group(chunk)?;

        let mut row_groups = self.row_groups.write();
        let mut row_count = self.row_count.write();

        *row_count += row_group.num_rows;
        row_groups.push(Arc::new(row_group));

        Ok(())
    }

    /// Build row group from data chunk.
    fn build_row_group(&self, chunk: DataChunk) -> Result<RowGroup> {
        let index = self.row_groups.read().len();
        let num_rows = chunk.len();
        let mut columns = Vec::with_capacity(chunk.num_columns());

        for (i, vector) in chunk.vectors.iter().enumerate() {
            let column_chunk = encode_column(i, vector, &self.config)?;
            columns.push(column_chunk);
        }

        Ok(RowGroup::new(index, num_rows, columns))
    }

    /// Scan table with column projection.
    pub fn scan(&self, column_indices: &[usize]) -> TableScanner {
        TableScanner::new(self, column_indices.to_vec())
    }

    /// Scan all columns.
    pub fn scan_all(&self) -> TableScanner {
        let indices: Vec<usize> = (0..self.schema.len()).collect();
        self.scan(&indices)
    }

    /// Scan a disjoint partition of the table's row groups.
    ///
    /// Row groups are assigned to partitions round-robin, so the union of
    /// all `num_partitions` scanners covers every row exactly once.
    pub fn scan_partition(
        &self,
        column_indices: &[usize],
        partition: usize,
        num_partitions: usize,
    ) -> TableScanner {
        TableScanner::new_partition(self, column_indices.to_vec(), partition, num_partitions)
    }

    /// Get column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.schema.column_index(name)
    }
}

/// Table scanner for reading data.
pub struct TableScanner {
    /// Row groups to scan.
    row_groups: Vec<Arc<RowGroup>>,
    /// Column indices to read.
    column_indices: Vec<usize>,
    /// Current row group index.
    current_rg: usize,
    /// Current offset within row group.
    current_offset: usize,
}

impl TableScanner {
    /// Create new scanner.
    fn new(table: &Table, column_indices: Vec<usize>) -> Self {
        let row_groups = table.row_groups.read().clone();
        Self {
            row_groups,
            column_indices,
            current_rg: 0,
            current_offset: 0,
        }
    }

    /// Create a scanner over a disjoint subset of the table's row groups.
    fn new_partition(
        table: &Table,
        column_indices: Vec<usize>,
        partition: usize,
        num_partitions: usize,
    ) -> Self {
        let num_partitions = num_partitions.max(1);
        let row_groups = table
            .row_groups
            .read()
            .iter()
            .enumerate()
            .filter(|(i, _)| i % num_partitions == partition)
            .map(|(_, rg)| Arc::clone(rg))
            .collect();
        Self {
            row_groups,
            column_indices,
            current_rg: 0,
            current_offset: 0,
        }
    }

    /// Get next batch.
    pub fn next_batch(&mut self, batch_size: usize) -> Result<Option<DataChunk>> {
        if self.current_rg >= self.row_groups.len() {
            return Ok(None);
        }

        let rg = &self.row_groups[self.current_rg];
        let remaining = rg.num_rows - self.current_offset;

        if remaining == 0 {
            self.current_rg += 1;
            self.current_offset = 0;
            return self.next_batch(batch_size);
        }

        let len = remaining.min(batch_size);
        let chunk = rg.read(&self.column_indices)?;
        let result = chunk.slice(self.current_offset, len)?;

        self.current_offset += len;

        Ok(Some(result))
    }
}

impl Iterator for TableScanner {
    type Item = Result<DataChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_batch(VECTOR_SIZE) {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Database catalog.
pub struct Catalog {
    /// Tables.
    tables: RwLock<HashMap<String, Arc<Table>>>,
}

impl Catalog {
    /// Create new catalog.
    pub fn new() -> Self {
        Self {
            tables: RwLock::new(HashMap::new()),
        }
    }

    /// Create a table.
    pub fn create_table(
        &self,
        name: impl Into<String>,
        schema: Schema,
        config: StorageConfig,
    ) -> Result<Arc<Table>> {
        let name = name.into();
        let mut tables = self.tables.write();

        if tables.contains_key(&name) {
            return Err(Error::InvalidOperation(format!(
                "Table '{}' already exists",
                name
            )));
        }

        let table = Arc::new(Table::new(name.clone(), schema, config));
        tables.insert(name, table.clone());

        Ok(table)
    }

    /// Get a table.
    pub fn get_table(&self, name: &str) -> Option<Arc<Table>> {
        self.tables.read().get(name).cloned()
    }

    /// Drop a table.
    pub fn drop_table(&self, name: &str) -> Result<()> {
        let mut tables = self.tables.write();

        if tables.remove(name).is_none() {
            return Err(Error::TableNotFound(name.to_string()));
        }

        Ok(())
    }

    /// List tables.
    pub fn list_tables(&self) -> Vec<String> {
        self.tables.read().keys().cloned().collect()
    }

    /// Get table schema by name.
    pub fn get_table_schema(&self, name: &str) -> Result<Schema> {
        self.tables
            .read()
            .get(name)
            .map(|t| t.schema.clone())
            .ok_or_else(|| Error::TableNotFound(name.to_string()))
    }

    /// Register a schema for a table (creates empty table).
    pub fn register_schema(&mut self, name: &str, schema: Schema) {
        let table = Arc::new(Table::new(name.to_string(), schema, StorageConfig::default()));
        self.tables.write().insert(name.to_string(), table);
    }
}

impl Default for Catalog {
    fn default() -> Self {
        Self::new()
    }
}

/// Encode column to column chunk.
fn encode_column(column_idx: usize, vector: &Vector, config: &StorageConfig) -> Result<ColumnChunk> {
    let data_type = vector.data_type.clone();
    let num_values = vector.len();

    // Count nulls
    let null_count = (0..num_values).filter(|&i| vector.is_null(i)).count();

    // Compute statistics if enabled
    let stats = if config.statistics {
        compute_stats(vector)
    } else {
        None
    };

    // Encode data (simplified - just serialize)
    let data = encode_vector_data(vector)?;

    Ok(ColumnChunk {
        column_idx,
        data_type,
        data,
        num_values,
        null_count,
        stats,
        encoding: Encoding::Plain,
    })
}

/// Encode vector data to bytes.
fn encode_vector_data(vector: &Vector) -> Result<Vec<u8>> {
    // Simplified encoding - in production would use proper formats
    let mut data = Vec::new();

    // Encode validity bitmap
    let validity_bytes: Vec<u8> = vector
        .validity
        .iter()
        .map(|b| if *b { 1u8 } else { 0u8 })
        .collect();
    data.extend_from_slice(&(validity_bytes.len() as u32).to_le_bytes());
    data.extend(validity_bytes);

    // Encode values based on type
    match &vector.data {
        crate::vector::VectorData::Int64(values) => {
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        crate::vector::VectorData::Float64(values) => {
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        crate::vector::VectorData::String(values) => {
            for s in values {
                let bytes = s.as_bytes();
                data.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                data.extend(bytes);
            }
        }
        crate::vector::VectorData::Boolean(values) => {
            for &v in values {
                data.push(if v { 1 } else { 0 });
            }
        }
        crate::vector::VectorData::Int32(values) => {
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        _ => {
            // Fallback - encode as JSON
            for i in 0..vector.len() {
                let value = vector.get(i)?;
                let json = serde_json::to_vec(&value).map_err(|e| Error::Execution(e.to_string()))?;
                data.extend_from_slice(&(json.len() as u32).to_le_bytes());
                data.extend(json);
            }
        }
    }

    Ok(data)
}

/// Decode column chunk to vector.
fn decode_column(chunk: &ColumnChunk) -> Result<Vector> {
    let mut vector = Vector::new(chunk.data_type.clone());
    let data = &chunk.data;
    let mut offset = 0;

    // Decode validity bitmap
    let validity_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;
    let validity: Vec<bool> = data[offset..offset + validity_len]
        .iter()
        .map(|&b| b != 0)
        .collect();
    offset += validity_len;

    // Decode values
    match &chunk.data_type {
        DataType::Int64 | DataType::Timestamp => {
            for (i, &valid) in validity.iter().enumerate() {
                if valid {
                    let value =
                        i64::from_le_bytes(data[offset + i * 8..offset + (i + 1) * 8].try_into().unwrap());
                    if matches!(chunk.data_type, DataType::Timestamp) {
                        vector.push(Value::Timestamp(value))?;
                    } else {
                        vector.push(Value::Int64(value))?;
                    }
                } else {
                    vector.push(Value::Null)?;
                }
            }
        }
        DataType::Float64 => {
            for (i, &valid) in validity.iter().enumerate() {
                if valid {
                    let value =
                        f64::from_le_bytes(data[offset + i * 8..offset + (i + 1) * 8].try_into().unwrap());
                    vector.push(Value::Float64(value))?;
                } else {
                    vector.push(Value::Null)?;
                }
            }
        }
        DataType::Int32 | DataType::Date => {
            for (i, &valid) in validity.iter().enumerate() {
                if valid {
                    let value =
                        i32::from_le_bytes(data[offset + i * 4..offset + (i + 1) * 4].try_into().unwrap());
                    if matches!(chunk.data_type, DataType::Date) {
                        vector.push(Value::Date(value))?;
                    } else {
                        vector.push(Value::Int32(value))?;
                    }
                } else {
                    vector.push(Value::Null)?;
                }
            }
        }
        DataType::Boolean => {
            for (i, &valid) in validity.iter().enumerate() {
                if valid {
                    vector.push(Value::Boolean(data[offset + i] != 0))?;
                } else {
                    vector.push(Value::Null)?;
                }
            }
        }
        DataType::String => {
            for &valid in &validity {
                if valid {
                    let len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                    offset += 4;
                    let s = String::from_utf8_lossy(&data[offset..offset + len]).to_string();
                    offset += len;
                    vector.push(Value::String(s))?;
                } else {
                    vector.push(Value::Null)?;
                }
            }
        }
        _ => {
            return Err(Error::Type(format!(
                "Unsupported type for decoding: {:?}",
                chunk.data_type
            )));
        }
    }

    Ok(vector)
}

/// Compute statistics for a vector.
fn compute_stats(vector: &Vector) -> Option<ColumnStats> {
    if vector.is_empty() {
        return None;
    }

    let mut min: Option<Value> = None;
    let mut max: Option<Value> = None;
    let mut null_count = 0;

    for i in 0..vector.len() {
        if vector.is_null(i) {
            null_count += 1;
            continue;
        }

        if let Ok(value) = vector.get(i) {
            match (&min, &value) {
                (None, _) => min = Some(value.clone()),
                (Some(m), v) if v < m => min = Some(value.clone()),
                _ => {}
            }
            match (&max, &value) {
                (None, _) => max = Some(value.clone()),
                (Some(m), v) if v > m => max = Some(value.clone()),
                _ => {}
            }
        }
    }

    Some(ColumnStats {
        min: min.unwrap_or(Value::Null),
        max: max.unwrap_or(Value::Null),
        null_count,
        distinct_count: None,
    })
}
