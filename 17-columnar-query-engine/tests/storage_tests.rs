//! Tests for column storage and encodings.

use columnar_query_engine::storage::*;
use columnar_query_engine::types::*;
use columnar_query_engine::vector::{DataChunk, Vector};
use std::sync::Arc;

// ============================================================================
// Column Chunk and Row Group Tests
// ============================================================================

#[test]
fn test_row_group_creation() {
    let chunk = ColumnChunk {
        column_idx: 0,
        data_type: DataType::Int64,
        data: vec![1, 2, 3, 4],
        num_values: 4,
        null_count: 0,
        stats: None,
        encoding: Encoding::Plain,
    };

    let row_group = RowGroup::new(0, 4, vec![chunk]);
    assert_eq!(row_group.index, 0);
    assert_eq!(row_group.num_rows, 4);
    assert_eq!(row_group.columns.len(), 1);
}

#[test]
fn test_row_group_column_access() {
    let chunk1 = ColumnChunk {
        column_idx: 0,
        data_type: DataType::Int64,
        data: vec![],
        num_values: 10,
        null_count: 0,
        stats: None,
        encoding: Encoding::Plain,
    };
    let chunk2 = ColumnChunk {
        column_idx: 1,
        data_type: DataType::String,
        data: vec![],
        num_values: 10,
        null_count: 0,
        stats: None,
        encoding: Encoding::Plain,
    };

    let row_group = RowGroup::new(0, 10, vec![chunk1, chunk2]);

    assert!(row_group.column(0).is_some());
    assert!(row_group.column(1).is_some());
    assert!(row_group.column(2).is_none());
}

// ============================================================================
// Encoding Tests
// ============================================================================

#[test]
fn test_encoding_variants() {
    // Test all encoding variants exist and can be compared
    let encodings = vec![
        Encoding::Plain,
        Encoding::Rle,
        Encoding::Dictionary,
        Encoding::Delta,
        Encoding::BitPacked,
    ];

    for encoding in &encodings {
        assert_eq!(*encoding, *encoding);
    }

    assert_ne!(Encoding::Plain, Encoding::Rle);
    assert_ne!(Encoding::Dictionary, Encoding::Delta);
}

// ============================================================================
// Column Statistics Tests
// ============================================================================

#[test]
fn test_column_stats_creation() {
    let stats = ColumnStats {
        min: Value::Int64(0),
        max: Value::Int64(100),
        null_count: 5,
        distinct_count: Some(50),
    };

    assert_eq!(stats.null_count, 5);
    assert_eq!(stats.distinct_count, Some(50));
}

#[test]
fn test_column_stats_with_string_values() {
    let stats = ColumnStats {
        min: Value::String("apple".to_string()),
        max: Value::String("zebra".to_string()),
        null_count: 0,
        distinct_count: None,
    };

    if let Value::String(s) = &stats.min {
        assert_eq!(s, "apple");
    } else {
        panic!("Expected String value");
    }
}

// ============================================================================
// Storage Configuration Tests
// ============================================================================

#[test]
fn test_storage_config_default() {
    let config = StorageConfig::default();
    assert_eq!(config.row_group_size, 100_000);
    assert!(config.compression);
    assert!(config.statistics);
}

#[test]
fn test_storage_config_custom() {
    let config = StorageConfig {
        row_group_size: 50_000,
        compression: false,
        statistics: true,
    };

    assert_eq!(config.row_group_size, 50_000);
    assert!(!config.compression);
    assert!(config.statistics);
}

// ============================================================================
// Table Creation Tests
// ============================================================================

#[test]
fn test_table_creation() {
    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("name", DataType::String, true),
    ]);

    let table = Table::new("test_table", schema, StorageConfig::default());

    assert_eq!(table.name, "test_table");
    assert_eq!(table.schema.len(), 2);
    assert_eq!(table.row_count(), 0);
    assert_eq!(table.row_group_count(), 0);
}

#[test]
fn test_table_column_index_lookup() {
    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("name", DataType::String, true),
        Column::new("age", DataType::Int32, false),
    ]);

    let table = Table::new("test_table", schema, StorageConfig::default());

    assert_eq!(table.column_index("id"), Some(0));
    assert_eq!(table.column_index("name"), Some(1));
    assert_eq!(table.column_index("age"), Some(2));
    assert_eq!(table.column_index("nonexistent"), None);
}

// ============================================================================
// Table Insert and Scan Tests
// ============================================================================

#[test]
fn test_table_insert_and_row_count() {
    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("value", DataType::Float64, false),
    ]);

    let table = Table::new("test_table", schema, StorageConfig::default());

    // Create a data chunk with 10 rows
    let mut id_vec = Vector::new(DataType::Int64);
    let mut value_vec = Vector::new(DataType::Float64);

    for i in 0..10 {
        id_vec.push(Value::Int64(i as i64)).unwrap();
        value_vec.push(Value::Float64(i as f64 * 1.5)).unwrap();
    }

    let chunk = DataChunk::new(vec![id_vec, value_vec]);
    table.insert(chunk).unwrap();

    assert_eq!(table.row_count(), 10);
    assert_eq!(table.row_group_count(), 1);
}

#[test]
fn test_table_insert_multiple_chunks() {
    let schema = Schema::new(vec![Column::new("id", DataType::Int64, false)]);
    let table = Table::new("test_table", schema, StorageConfig::default());

    // Insert first chunk
    let mut vec1 = Vector::new(DataType::Int64);
    for i in 0..5 {
        vec1.push(Value::Int64(i)).unwrap();
    }
    table.insert(DataChunk::new(vec![vec1])).unwrap();

    // Insert second chunk
    let mut vec2 = Vector::new(DataType::Int64);
    for i in 5..15 {
        vec2.push(Value::Int64(i)).unwrap();
    }
    table.insert(DataChunk::new(vec![vec2])).unwrap();

    assert_eq!(table.row_count(), 15);
    assert_eq!(table.row_group_count(), 2);
}

#[test]
fn test_table_insert_column_mismatch() {
    let schema = Schema::new(vec![
        Column::new("a", DataType::Int64, false),
        Column::new("b", DataType::Int64, false),
    ]);

    let table = Table::new("test_table", schema, StorageConfig::default());

    // Try to insert with wrong number of columns
    let vec1 = Vector::new(DataType::Int64);
    let chunk = DataChunk::new(vec![vec1]); // Only 1 column, table expects 2

    let result = table.insert(chunk);
    assert!(result.is_err());
}

#[test]
fn test_table_scan_all() {
    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("name", DataType::String, true),
    ]);

    let table = Table::new("test_table", schema, StorageConfig::default());

    // Insert data
    let mut id_vec = Vector::new(DataType::Int64);
    let mut name_vec = Vector::new(DataType::String);

    for i in 0..5 {
        id_vec.push(Value::Int64(i as i64)).unwrap();
        name_vec.push(Value::String(format!("name_{}", i))).unwrap();
    }

    table.insert(DataChunk::new(vec![id_vec, name_vec])).unwrap();

    // Scan all columns
    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();

    assert_eq!(chunk.num_columns(), 2);
    assert_eq!(chunk.len(), 5);
}

#[test]
fn test_table_scan_with_projection() {
    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("name", DataType::String, true),
        Column::new("age", DataType::Int32, false),
    ]);

    let table = Table::new("test_table", schema, StorageConfig::default());

    // Insert data
    let mut id_vec = Vector::new(DataType::Int64);
    let mut name_vec = Vector::new(DataType::String);
    let mut age_vec = Vector::new(DataType::Int32);

    for i in 0..5 {
        id_vec.push(Value::Int64(i as i64)).unwrap();
        name_vec.push(Value::String(format!("name_{}", i))).unwrap();
        age_vec.push(Value::Int32(20 + i as i32)).unwrap();
    }

    table
        .insert(DataChunk::new(vec![id_vec, name_vec, age_vec]))
        .unwrap();

    // Scan only columns 0 and 2 (id and age)
    let mut scanner = table.scan(&[0, 2]);
    let chunk = scanner.next().unwrap().unwrap();

    assert_eq!(chunk.num_columns(), 2);
    assert_eq!(chunk.len(), 5);
}

// ============================================================================
// TableScanner Tests
// ============================================================================

#[test]
fn test_table_scanner_iterator() {
    let schema = Schema::new(vec![Column::new("id", DataType::Int64, false)]);
    let table = Table::new("test_table", schema, StorageConfig::default());

    // Insert multiple chunks
    for batch in 0..3 {
        let mut vec = Vector::new(DataType::Int64);
        for i in 0..10 {
            vec.push(Value::Int64((batch * 10 + i) as i64)).unwrap();
        }
        table.insert(DataChunk::new(vec![vec])).unwrap();
    }

    // Iterate through scanner
    let scanner = table.scan_all();
    let mut total_rows = 0;

    for result in scanner {
        let chunk = result.unwrap();
        total_rows += chunk.len();
    }

    assert_eq!(total_rows, 30);
}

#[test]
fn test_table_scanner_next_batch() {
    let schema = Schema::new(vec![Column::new("value", DataType::Int64, false)]);
    let table = Table::new("test_table", schema, StorageConfig::default());

    // Insert data
    let mut vec = Vector::new(DataType::Int64);
    for i in 0..100 {
        vec.push(Value::Int64(i)).unwrap();
    }
    table.insert(DataChunk::new(vec![vec])).unwrap();

    // Use custom batch size
    let mut scanner = table.scan_all();
    let batch = scanner.next_batch(25).unwrap().unwrap();

    assert_eq!(batch.len(), 25);
}

// ============================================================================
// Catalog Tests
// ============================================================================

#[test]
fn test_catalog_creation() {
    let catalog = Catalog::new();
    assert_eq!(catalog.list_tables().len(), 0);
}

#[test]
fn test_catalog_create_table() {
    let catalog = Catalog::new();
    let schema = Schema::new(vec![Column::new("id", DataType::Int64, false)]);

    let table = catalog
        .create_table("users", schema, StorageConfig::default())
        .unwrap();

    assert_eq!(table.name, "users");
    assert!(catalog.get_table("users").is_some());
}

#[test]
fn test_catalog_create_duplicate_table() {
    let catalog = Catalog::new();
    let schema = Schema::new(vec![Column::new("id", DataType::Int64, false)]);

    catalog
        .create_table("users", schema.clone(), StorageConfig::default())
        .unwrap();

    // Try to create table with same name
    let result = catalog.create_table("users", schema, StorageConfig::default());
    assert!(result.is_err());
}

#[test]
fn test_catalog_get_nonexistent_table() {
    let catalog = Catalog::new();
    assert!(catalog.get_table("nonexistent").is_none());
}

#[test]
fn test_catalog_drop_table() {
    let catalog = Catalog::new();
    let schema = Schema::new(vec![Column::new("id", DataType::Int64, false)]);

    catalog
        .create_table("users", schema, StorageConfig::default())
        .unwrap();
    assert!(catalog.get_table("users").is_some());

    catalog.drop_table("users").unwrap();
    assert!(catalog.get_table("users").is_none());
}

#[test]
fn test_catalog_drop_nonexistent_table() {
    let catalog = Catalog::new();
    let result = catalog.drop_table("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_catalog_list_tables() {
    let catalog = Catalog::new();

    catalog
        .create_table(
            "users",
            Schema::new(vec![Column::new("id", DataType::Int64, false)]),
            StorageConfig::default(),
        )
        .unwrap();

    catalog
        .create_table(
            "orders",
            Schema::new(vec![Column::new("id", DataType::Int64, false)]),
            StorageConfig::default(),
        )
        .unwrap();

    let tables = catalog.list_tables();
    assert_eq!(tables.len(), 2);
    assert!(tables.contains(&"users".to_string()));
    assert!(tables.contains(&"orders".to_string()));
}

// ============================================================================
// Data Type Encoding/Decoding Round-Trip Tests
// ============================================================================

#[test]
fn test_int64_encode_decode_roundtrip() {
    let schema = Schema::new(vec![Column::new("value", DataType::Int64, false)]);
    let table = Table::new("test", schema, StorageConfig::default());

    let mut vec = Vector::new(DataType::Int64);
    let test_values: Vec<i64> = vec![-100, -1, 0, 1, 100, i64::MAX, i64::MIN];

    for &v in &test_values {
        vec.push(Value::Int64(v)).unwrap();
    }

    table.insert(DataChunk::new(vec![vec])).unwrap();

    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();
    let result_vec = chunk.vector(0).unwrap();

    for (i, &expected) in test_values.iter().enumerate() {
        if let Value::Int64(v) = result_vec.get(i).unwrap() {
            assert_eq!(v, expected);
        } else {
            panic!("Expected Int64 value at index {}", i);
        }
    }
}

#[test]
fn test_float64_encode_decode_roundtrip() {
    let schema = Schema::new(vec![Column::new("value", DataType::Float64, false)]);
    let table = Table::new("test", schema, StorageConfig::default());

    let mut vec = Vector::new(DataType::Float64);
    let test_values: Vec<f64> = vec![-3.14, 0.0, 1.5, 100.001, f64::MAX, f64::MIN];

    for &v in &test_values {
        vec.push(Value::Float64(v)).unwrap();
    }

    table.insert(DataChunk::new(vec![vec])).unwrap();

    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();
    let result_vec = chunk.vector(0).unwrap();

    for (i, &expected) in test_values.iter().enumerate() {
        if let Value::Float64(v) = result_vec.get(i).unwrap() {
            assert!((v - expected).abs() < 1e-10 || (v.is_infinite() && expected.is_infinite()));
        } else {
            panic!("Expected Float64 value at index {}", i);
        }
    }
}

#[test]
fn test_string_encode_decode_roundtrip() {
    let schema = Schema::new(vec![Column::new("value", DataType::String, false)]);
    let table = Table::new("test", schema, StorageConfig::default());

    let mut vec = Vector::new(DataType::String);
    let test_values: Vec<&str> = vec!["", "hello", "world", "hello world", "with\nnewline"];

    for v in &test_values {
        vec.push(Value::String(v.to_string())).unwrap();
    }

    table.insert(DataChunk::new(vec![vec])).unwrap();

    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();
    let result_vec = chunk.vector(0).unwrap();

    for (i, &expected) in test_values.iter().enumerate() {
        if let Value::String(s) = result_vec.get(i).unwrap() {
            assert_eq!(s, expected);
        } else {
            panic!("Expected String value at index {}", i);
        }
    }
}

#[test]
fn test_boolean_encode_decode_roundtrip() {
    let schema = Schema::new(vec![Column::new("value", DataType::Boolean, false)]);
    let table = Table::new("test", schema, StorageConfig::default());

    let mut vec = Vector::new(DataType::Boolean);
    let test_values: Vec<bool> = vec![true, false, true, true, false];

    for v in &test_values {
        vec.push(Value::Boolean(*v)).unwrap();
    }

    table.insert(DataChunk::new(vec![vec])).unwrap();

    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();
    let result_vec = chunk.vector(0).unwrap();

    for (i, &expected) in test_values.iter().enumerate() {
        if let Value::Boolean(b) = result_vec.get(i).unwrap() {
            assert_eq!(b, expected);
        } else {
            panic!("Expected Boolean value at index {}", i);
        }
    }
}

#[test]
fn test_int32_encode_decode_roundtrip() {
    let schema = Schema::new(vec![Column::new("value", DataType::Int32, false)]);
    let table = Table::new("test", schema, StorageConfig::default());

    let mut vec = Vector::new(DataType::Int32);
    let test_values: Vec<i32> = vec![-1000, -1, 0, 1, 1000, i32::MAX, i32::MIN];

    for &v in &test_values {
        vec.push(Value::Int32(v)).unwrap();
    }

    table.insert(DataChunk::new(vec![vec])).unwrap();

    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();
    let result_vec = chunk.vector(0).unwrap();

    for (i, &expected) in test_values.iter().enumerate() {
        if let Value::Int32(v) = result_vec.get(i).unwrap() {
            assert_eq!(v, expected);
        } else {
            panic!("Expected Int32 value at index {}", i);
        }
    }
}

// ============================================================================
// Null Value Tests
// ============================================================================

#[test]
fn test_null_value_handling() {
    let schema = Schema::new(vec![Column::new("value", DataType::Int64, true)]);
    let table = Table::new("test", schema, StorageConfig::default());

    let mut vec = Vector::new(DataType::Int64);
    vec.push(Value::Int64(1)).unwrap();
    vec.push(Value::Null).unwrap();
    vec.push(Value::Int64(3)).unwrap();
    vec.push(Value::Null).unwrap();
    vec.push(Value::Int64(5)).unwrap();

    table.insert(DataChunk::new(vec![vec])).unwrap();

    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();
    let result_vec = chunk.vector(0).unwrap();

    assert!(!result_vec.is_null(0));
    assert!(result_vec.is_null(1));
    assert!(!result_vec.is_null(2));
    assert!(result_vec.is_null(3));
    assert!(!result_vec.is_null(4));
}

// ============================================================================
// Multi-Column Tests
// ============================================================================

#[test]
fn test_multi_column_table() {
    let schema = Schema::new(vec![
        Column::new("id", DataType::Int64, false),
        Column::new("name", DataType::String, false),
        Column::new("score", DataType::Float64, false),
        Column::new("active", DataType::Boolean, false),
    ]);

    let table = Table::new("test", schema, StorageConfig::default());

    let mut id_vec = Vector::new(DataType::Int64);
    let mut name_vec = Vector::new(DataType::String);
    let mut score_vec = Vector::new(DataType::Float64);
    let mut active_vec = Vector::new(DataType::Boolean);

    for i in 0..10 {
        id_vec.push(Value::Int64(i as i64)).unwrap();
        name_vec.push(Value::String(format!("user_{}", i))).unwrap();
        score_vec.push(Value::Float64(i as f64 * 10.5)).unwrap();
        active_vec.push(Value::Boolean(i % 2 == 0)).unwrap();
    }

    table
        .insert(DataChunk::new(vec![id_vec, name_vec, score_vec, active_vec]))
        .unwrap();

    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();

    assert_eq!(chunk.num_columns(), 4);
    assert_eq!(chunk.len(), 10);

    // Verify first row
    assert_eq!(chunk.vector(0).unwrap().get(0).unwrap(), Value::Int64(0));
    assert_eq!(
        chunk.vector(1).unwrap().get(0).unwrap(),
        Value::String("user_0".to_string())
    );
    assert_eq!(
        chunk.vector(2).unwrap().get(0).unwrap(),
        Value::Float64(0.0)
    );
    assert_eq!(
        chunk.vector(3).unwrap().get(0).unwrap(),
        Value::Boolean(true)
    );
}

// ============================================================================
// Date and Timestamp Tests
// ============================================================================

#[test]
fn test_date_encode_decode_roundtrip() {
    let schema = Schema::new(vec![Column::new("value", DataType::Date, false)]);
    let table = Table::new("test", schema, StorageConfig::default());

    let mut vec = Vector::new(DataType::Date);
    let test_values: Vec<i32> = vec![0, 1, 365, 18628, -1000]; // Days since epoch

    for &v in &test_values {
        vec.push(Value::Date(v)).unwrap();
    }

    table.insert(DataChunk::new(vec![vec])).unwrap();

    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();
    let result_vec = chunk.vector(0).unwrap();

    for (i, &expected) in test_values.iter().enumerate() {
        if let Value::Date(v) = result_vec.get(i).unwrap() {
            assert_eq!(v, expected);
        } else {
            panic!("Expected Date value at index {}", i);
        }
    }
}

#[test]
fn test_timestamp_encode_decode_roundtrip() {
    let schema = Schema::new(vec![Column::new("value", DataType::Timestamp, false)]);
    let table = Table::new("test", schema, StorageConfig::default());

    let mut vec = Vector::new(DataType::Timestamp);
    let test_values: Vec<i64> = vec![0, 1000000, 1609459200000000, -1000000]; // Microseconds since epoch

    for &v in &test_values {
        vec.push(Value::Timestamp(v)).unwrap();
    }

    table.insert(DataChunk::new(vec![vec])).unwrap();

    let mut scanner = table.scan_all();
    let chunk = scanner.next().unwrap().unwrap();
    let result_vec = chunk.vector(0).unwrap();

    for (i, &expected) in test_values.iter().enumerate() {
        if let Value::Timestamp(v) = result_vec.get(i).unwrap() {
            assert_eq!(v, expected);
        } else {
            panic!("Expected Timestamp value at index {}", i);
        }
    }
}
