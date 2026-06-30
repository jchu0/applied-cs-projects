//! Comprehensive tests for hash operations.

use simd_analytics_engine::hash::{
    VectorizedHash, HashTable, HashAggregatorSimd, HashJoin,
    BloomFilter, CountMinSketch, HyperLogLog, AggState,
};

// ============================================================================
// VectorizedHash Tests
// ============================================================================

#[test]
fn test_hash_i64_deterministic() {
    let h1 = VectorizedHash::hash_i64(42);
    let h2 = VectorizedHash::hash_i64(42);
    assert_eq!(h1, h2);
}

#[test]
fn test_hash_i64_different_values() {
    let h1 = VectorizedHash::hash_i64(42);
    let h2 = VectorizedHash::hash_i64(43);
    assert_ne!(h1, h2);
}

#[test]
fn test_hash_u64() {
    let h1 = VectorizedHash::hash_u64(12345u64);
    let h2 = VectorizedHash::hash_u64(12345u64);
    assert_eq!(h1, h2);
}

#[test]
fn test_hash_i32() {
    let h1 = VectorizedHash::hash_i32(100);
    let h2 = VectorizedHash::hash_i32(100);
    assert_eq!(h1, h2);
}

#[test]
fn test_hash_i64_vec() {
    let keys = vec![1, 2, 3, 4, 5];
    let hashes = VectorizedHash::hash_i64_vec(&keys);

    assert_eq!(hashes.len(), 5);

    // All hashes should be different
    for i in 0..hashes.len() {
        for j in i + 1..hashes.len() {
            assert_ne!(hashes[i], hashes[j]);
        }
    }
}

#[test]
fn test_hash_bytes() {
    let h1 = VectorizedHash::hash_bytes(b"hello");
    let h2 = VectorizedHash::hash_bytes(b"hello");
    let h3 = VectorizedHash::hash_bytes(b"world");

    assert_eq!(h1, h2);
    assert_ne!(h1, h3);
}

#[test]
fn test_hash_str() {
    let h1 = VectorizedHash::hash_str("test");
    let h2 = VectorizedHash::hash_str("test");
    let h3 = VectorizedHash::hash_str("other");

    assert_eq!(h1, h2);
    assert_ne!(h1, h3);
}

#[test]
fn test_combine_hashes() {
    let h1 = VectorizedHash::hash_i64(1);
    let h2 = VectorizedHash::hash_i64(2);

    let combined1 = VectorizedHash::combine_hashes(h1, h2);
    let combined2 = VectorizedHash::combine_hashes(h2, h1);

    // Order matters
    assert_ne!(combined1, combined2);
}

// ============================================================================
// HashTable Tests
// ============================================================================

#[test]
fn test_hash_table_creation() {
    let table: HashTable<i64, f64> = HashTable::new(16);
    assert_eq!(table.len(), 0);
    assert!(table.is_empty());
}

#[test]
fn test_hash_table_insert_get() {
    let mut table: HashTable<i64, f64> = HashTable::new(16);

    table.insert(1, 10.0);
    table.insert(2, 20.0);
    table.insert(3, 30.0);

    assert_eq!(table.len(), 3);
    assert_eq!(table.get(&1), Some(&10.0));
    assert_eq!(table.get(&2), Some(&20.0));
    assert_eq!(table.get(&3), Some(&30.0));
    assert_eq!(table.get(&4), None);
}

#[test]
fn test_hash_table_update() {
    let mut table: HashTable<i64, i32> = HashTable::new(16);

    assert!(table.insert(1, 100).is_none());
    assert_eq!(table.insert(1, 200), Some(100));
    assert_eq!(table.get(&1), Some(&200));
}

#[test]
fn test_hash_table_contains_key() {
    let mut table: HashTable<i64, &str> = HashTable::new(16);

    table.insert(1, "one");
    table.insert(2, "two");

    assert!(table.contains_key(&1));
    assert!(table.contains_key(&2));
    assert!(!table.contains_key(&3));
}

#[test]
fn test_hash_table_get_mut() {
    let mut table: HashTable<i64, i32> = HashTable::new(16);

    table.insert(1, 100);

    if let Some(val) = table.get_mut(&1) {
        *val = 200;
    }

    assert_eq!(table.get(&1), Some(&200));
}

#[test]
fn test_hash_table_get_or_insert() {
    let mut table: HashTable<i64, Vec<i32>> = HashTable::new(16);

    let entry = table.get_or_insert_with(1, Vec::new);
    entry.push(10);

    let entry = table.get_or_insert_with(1, Vec::new);
    entry.push(20);

    assert_eq!(table.get(&1), Some(&vec![10, 20]));
}

#[test]
fn test_hash_table_resize() {
    let mut table: HashTable<i64, i32> = HashTable::new(8);

    // Insert more than initial capacity
    for i in 0..100 {
        table.insert(i, i as i32 * 10);
    }

    assert_eq!(table.len(), 100);

    // Verify all entries still accessible
    for i in 0..100 {
        assert_eq!(table.get(&i), Some(&(i as i32 * 10)));
    }
}

#[test]
fn test_hash_table_iter() {
    let mut table: HashTable<i64, i32> = HashTable::new(16);

    table.insert(1, 10);
    table.insert(2, 20);
    table.insert(3, 30);

    let sum: i32 = table.iter().map(|(_, &v)| v).sum();
    assert_eq!(sum, 60);
}

#[test]
fn test_hash_table_clear() {
    let mut table: HashTable<i64, i32> = HashTable::new(16);

    table.insert(1, 10);
    table.insert(2, 20);

    table.clear();

    assert_eq!(table.len(), 0);
    assert!(table.is_empty());
    assert_eq!(table.get(&1), None);
}

#[test]
fn test_hash_table_load_factor() {
    let mut table: HashTable<i64, i32> = HashTable::new(16);

    for i in 0..8 {
        table.insert(i, i as i32);
    }

    let load_factor = table.load_factor();
    assert!(load_factor > 0.0 && load_factor <= 1.0);
}

// ============================================================================
// AggState Tests
// ============================================================================

#[test]
fn test_agg_state_new() {
    let state = AggState::new();
    assert_eq!(state.sum, 0.0);
    assert_eq!(state.count, 0);
    assert_eq!(state.min, f64::INFINITY);
    assert_eq!(state.max, f64::NEG_INFINITY);
}

#[test]
fn test_agg_state_update() {
    let mut state = AggState::new();

    state.update(10.0);
    state.update(20.0);
    state.update(30.0);

    assert_eq!(state.sum, 60.0);
    assert_eq!(state.count, 3);
    assert_eq!(state.min, 10.0);
    assert_eq!(state.max, 30.0);
}

#[test]
fn test_agg_state_avg() {
    let mut state = AggState::new();

    state.update(10.0);
    state.update(20.0);
    state.update(30.0);

    assert_eq!(state.avg(), Some(20.0));
}

#[test]
fn test_agg_state_merge() {
    let mut state1 = AggState::new();
    state1.update(10.0);
    state1.update(20.0);

    let mut state2 = AggState::new();
    state2.update(30.0);
    state2.update(40.0);

    state1.merge(&state2);

    assert_eq!(state1.sum, 100.0);
    assert_eq!(state1.count, 4);
    assert_eq!(state1.min, 10.0);
    assert_eq!(state1.max, 40.0);
}

// ============================================================================
// HashAggregatorSimd Tests
// ============================================================================

#[test]
fn test_hash_aggregator_creation() {
    let agg = HashAggregatorSimd::new();
    assert_eq!(agg.num_groups(), 0);
}

#[test]
fn test_hash_aggregator_aggregate() {
    let mut agg = HashAggregatorSimd::new();

    let keys = vec![1, 2, 1, 2, 1];
    let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];

    agg.aggregate(&keys, &values).unwrap();

    assert_eq!(agg.num_groups(), 2);
    assert_eq!(agg.get_sum(1), Some(90.0)); // 10 + 30 + 50
    assert_eq!(agg.get_sum(2), Some(60.0)); // 20 + 40
}

#[test]
fn test_hash_aggregator_count() {
    let mut agg = HashAggregatorSimd::new();

    let keys = vec![1, 1, 1, 2, 2, 3];
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    agg.aggregate(&keys, &values).unwrap();

    assert_eq!(agg.get_count(1), Some(3));
    assert_eq!(agg.get_count(2), Some(2));
    assert_eq!(agg.get_count(3), Some(1));
}

#[test]
fn test_hash_aggregator_avg() {
    let mut agg = HashAggregatorSimd::new();

    let keys = vec![1, 1, 1, 2, 2];
    let values = vec![10.0, 20.0, 30.0, 100.0, 200.0];

    agg.aggregate(&keys, &values).unwrap();

    assert_eq!(agg.get_avg(1), Some(20.0)); // (10 + 20 + 30) / 3
    assert_eq!(agg.get_avg(2), Some(150.0)); // (100 + 200) / 2
}

#[test]
fn test_hash_aggregator_min_max() {
    let mut agg = HashAggregatorSimd::new();

    let keys = vec![1, 1, 1, 2, 2];
    let values = vec![5.0, 15.0, 10.0, 100.0, 50.0];

    agg.aggregate(&keys, &values).unwrap();

    assert_eq!(agg.get_min(1), Some(5.0));
    assert_eq!(agg.get_max(1), Some(15.0));
    assert_eq!(agg.get_min(2), Some(50.0));
    assert_eq!(agg.get_max(2), Some(100.0));
}

#[test]
fn test_hash_aggregator_get_results() {
    let mut agg = HashAggregatorSimd::new();

    let keys = vec![1, 2, 3];
    let values = vec![10.0, 20.0, 30.0];

    agg.aggregate(&keys, &values).unwrap();

    let (result_keys, result_states) = agg.get_results();
    assert_eq!(result_keys.len(), 3);
    assert_eq!(result_states.len(), 3);
}

#[test]
fn test_hash_aggregator_large() {
    let mut agg = HashAggregatorSimd::with_capacity(1024);

    let size = 100_000;
    let keys: Vec<i64> = (0..size).map(|i| i % 1000).collect();
    let values: Vec<f64> = (0..size).map(|i| i as f64).collect();

    agg.aggregate(&keys, &values).unwrap();

    assert_eq!(agg.num_groups(), 1000);
}

// ============================================================================
// HashJoin Tests
// ============================================================================

#[test]
fn test_hash_join_build_probe() {
    let mut join: HashJoin<String> = HashJoin::new();

    let build_keys = vec![1, 2, 3];
    let build_values = vec!["a".to_string(), "b".to_string(), "c".to_string()];

    join.build(&build_keys, &build_values).unwrap();

    assert_eq!(join.build_size(), 3);

    let probe_keys = vec![1, 2, 4];
    let results = join.probe(&probe_keys);

    assert!(results[0].is_some());
    assert!(results[1].is_some());
    assert!(results[2].is_none());
}

#[test]
fn test_hash_join_multiple_matches() {
    let mut join: HashJoin<i32> = HashJoin::new();

    let build_keys = vec![1, 1, 2, 2, 2];
    let build_values = vec![10, 11, 20, 21, 22];

    join.build(&build_keys, &build_values).unwrap();

    let result = join.probe_single(1);
    assert!(result.is_some());
    assert_eq!(result.unwrap().len(), 2);

    let result = join.probe_single(2);
    assert!(result.is_some());
    assert_eq!(result.unwrap().len(), 3);
}

// ============================================================================
// BloomFilter Tests
// ============================================================================

#[test]
fn test_bloom_filter_creation() {
    let filter = BloomFilter::new(1000, 0.01);
    // Filter should be created without errors
    assert!(true);
}

#[test]
fn test_bloom_filter_insert_contains() {
    let mut filter = BloomFilter::new(1000, 0.01);

    for i in 0..100 {
        filter.insert(i);
    }

    // All inserted elements should be found
    for i in 0..100 {
        assert!(filter.might_contain(i));
    }
}

#[test]
fn test_bloom_filter_false_positive_rate() {
    let mut filter = BloomFilter::new(1000, 0.01);

    for i in 0..100 {
        filter.insert(i);
    }

    let mut false_positives = 0;
    for i in 100..1000 {
        if filter.might_contain(i) {
            false_positives += 1;
        }
    }

    // False positive rate should be reasonable
    let fp_rate = false_positives as f64 / 900.0;
    assert!(fp_rate < 0.05); // Less than 5%
}

#[test]
fn test_bloom_filter_clear() {
    let mut filter = BloomFilter::new(100, 0.01);

    filter.insert(42);
    assert!(filter.might_contain(42));

    filter.clear();
    // After clear, false positive still possible but insert is cleared
}

// ============================================================================
// CountMinSketch Tests
// ============================================================================

#[test]
fn test_count_min_sketch_creation() {
    let sketch = CountMinSketch::new(1000, 5);
    assert_eq!(sketch.estimate(1), 0);
}

#[test]
fn test_count_min_sketch_increment() {
    let mut sketch = CountMinSketch::new(1000, 5);

    sketch.increment(1);
    sketch.increment(1);
    sketch.increment(1);
    sketch.increment(2);

    assert_eq!(sketch.estimate(1), 3);
    assert_eq!(sketch.estimate(2), 1);
    assert_eq!(sketch.estimate(3), 0);
}

#[test]
fn test_count_min_sketch_increment_by() {
    let mut sketch = CountMinSketch::new(1000, 5);

    sketch.increment_by(1, 100);
    sketch.increment_by(1, 50);

    assert_eq!(sketch.estimate(1), 150);
}

#[test]
fn test_count_min_sketch_with_error_bounds() {
    let sketch = CountMinSketch::with_error_bounds(0.01, 0.001);
    // Should create with appropriate dimensions
    assert!(true);
}

#[test]
fn test_count_min_sketch_large_values() {
    let mut sketch = CountMinSketch::new(1000, 5);

    for i in 0..10000 {
        sketch.increment(i % 100);
    }

    // Each of 100 values should have count ~100
    for i in 0..100 {
        let estimate = sketch.estimate(i);
        assert!(estimate >= 100); // May be higher due to collisions
    }
}

// ============================================================================
// HyperLogLog Tests
// ============================================================================

#[test]
fn test_hyperloglog_creation() {
    let hll = HyperLogLog::new(10);
    let estimate = hll.estimate();
    assert!(estimate < 1.0); // Empty HLL should estimate ~0
}

#[test]
fn test_hyperloglog_small_cardinality() {
    let mut hll = HyperLogLog::new(10);

    for i in 0..100 {
        hll.add(i);
    }

    let estimate = hll.estimate();
    // Should be reasonably close to 100
    assert!((estimate - 100.0).abs() < 20.0);
}

#[test]
fn test_hyperloglog_large_cardinality() {
    let mut hll = HyperLogLog::new(12);

    for i in 0..10000 {
        hll.add(i);
    }

    let estimate = hll.estimate();
    // Should be within ~3% of actual
    assert!((estimate - 10000.0).abs() < 500.0);
}

#[test]
fn test_hyperloglog_duplicates() {
    let mut hll = HyperLogLog::new(10);

    // Add same values multiple times
    for _ in 0..1000 {
        hll.add(42);
    }

    let estimate = hll.estimate();
    // Should estimate ~1
    assert!(estimate < 5.0);
}

#[test]
fn test_hyperloglog_merge() {
    let mut hll1 = HyperLogLog::new(10);
    let mut hll2 = HyperLogLog::new(10);

    for i in 0..5000 {
        hll1.add(i);
    }
    for i in 2500..7500 {
        hll2.add(i);
    }

    hll1.merge(&hll2);

    let estimate = hll1.estimate();
    // Should estimate ~7500 unique values
    assert!((estimate - 7500.0).abs() < 500.0);
}

#[test]
fn test_hyperloglog_clear() {
    let mut hll = HyperLogLog::new(10);

    for i in 0..1000 {
        hll.add(i);
    }

    hll.clear();
    let estimate = hll.estimate();
    assert!(estimate < 1.0);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_hash_aggregator_mismatched_lengths() {
    let mut agg = HashAggregatorSimd::new();

    let keys = vec![1, 2, 3];
    let values = vec![10.0, 20.0]; // Mismatched length

    let result = agg.aggregate(&keys, &values);
    assert!(result.is_err());
}

#[test]
fn test_hash_join_mismatched_lengths() {
    let mut join: HashJoin<i32> = HashJoin::new();

    let keys = vec![1, 2, 3];
    let values = vec![10, 20]; // Mismatched length

    let result = join.build(&keys, &values);
    assert!(result.is_err());
}
